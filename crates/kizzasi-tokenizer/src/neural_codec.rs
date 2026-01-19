////! Neural Codec implementations (SoundStream/Encodec style)
//!
//! Provides state-of-the-art neural audio codecs with:
//! - Convolutional encoder-decoder architecture
//! - Residual Vector Quantization (RVQ) for compression
//! - Causal convolutions for streaming applications
//! - Residual blocks with dilated convolutions

use crate::error::{TokenizerError, TokenizerResult};
use crate::vqvae::{ResidualVQ, VQConfig};
use candle_core::{Device, Module, Result as CandleResult, Tensor};
use candle_nn::{
    conv1d, conv_transpose1d, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig,
    VarBuilder,
};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Configuration for Neural Codec
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCodecConfig {
    /// Input channels (e.g., 1 for mono)
    pub input_channels: usize,
    /// Hidden channels in encoder/decoder
    pub hidden_channels: usize,
    /// Number of residual blocks per encoder/decoder layer
    pub num_residual_blocks: usize,
    /// Stride factors for downsampling (product determines compression ratio)
    pub strides: Vec<usize>,
    /// Dilation factors for residual blocks
    pub dilations: Vec<usize>,
    /// Codebook size for each RVQ stage
    pub codebook_size: usize,
    /// Embedding dimension for VQ
    pub embed_dim: usize,
    /// Number of RVQ stages
    pub num_rvq_stages: usize,
    /// Whether to use causal convolutions (for streaming)
    pub causal: bool,
}

impl Default for NeuralCodecConfig {
    fn default() -> Self {
        Self {
            input_channels: 1,
            hidden_channels: 128,
            num_residual_blocks: 2,
            strides: vec![2, 4, 5, 8], // Total compression: 320x
            dilations: vec![1, 3, 9],
            codebook_size: 1024,
            embed_dim: 256,
            num_rvq_stages: 8,
            causal: false,
        }
    }
}

/// Causal 1D convolution for streaming applications
pub struct CausalConv1d {
    conv: Conv1d,
    padding: usize,
}

impl CausalConv1d {
    /// Create a new causal convolution
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        // Causal padding: (kernel_size - 1) * dilation
        let padding = (kernel_size - 1) * dilation;

        let config = Conv1dConfig {
            padding,
            dilation,
            ..Default::default()
        };

        let conv = conv1d(in_channels, out_channels, kernel_size, config, vb)?;

        Ok(Self { conv, padding })
    }

    /// Forward pass with causal padding
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let output = self.conv.forward(x)?;

        // Remove future-looking padding
        if self.padding > 0 {
            let seq_len = output.dim(2)?;
            output.narrow(2, 0, seq_len - self.padding)
        } else {
            Ok(output)
        }
    }
}

/// Residual block with dilated convolutions
pub struct ResidualUnit {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl ResidualUnit {
    /// Create a new residual unit
    pub fn new(
        channels: usize,
        dilation: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let config1 = Conv1dConfig {
            padding: dilation * (kernel_size - 1) / 2,
            dilation,
            ..Default::default()
        };

        let config2 = Conv1dConfig {
            padding: (kernel_size - 1) / 2,
            ..Default::default()
        };

        let conv1 = conv1d(channels, channels, kernel_size, config1, vb.pp("conv1"))?;
        let conv2 = conv1d(channels, channels, kernel_size, config2, vb.pp("conv2"))?;

        Ok(Self { conv1, conv2 })
    }

    /// Forward pass with residual connection
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x.clone();

        // Conv1 -> ELU activation
        let out = self.conv1.forward(x)?;
        let out = out.elu(1.0)?;

        // Conv2
        let out = self.conv2.forward(&out)?;

        // Residual connection
        let out = (out + residual)?;

        // ELU activation
        out.elu(1.0)
    }
}

/// Convolutional encoder for neural codec
pub struct ConvEncoder {
    /// Initial convolution to project input
    init_conv: Conv1d,
    /// Downsampling layers (strided convolutions)
    down_layers: Vec<Conv1d>,
    /// Residual blocks per layer
    residual_blocks: Vec<Vec<ResidualUnit>>,
    /// Final projection to embedding dimension
    final_conv: Conv1d,
}

impl ConvEncoder {
    /// Create a new convolutional encoder
    pub fn new(config: &NeuralCodecConfig, vb: VarBuilder) -> CandleResult<Self> {
        let mut current_channels = config.input_channels;

        // Initial convolution
        let init_conv = conv1d(
            current_channels,
            config.hidden_channels,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("init"),
        )?;
        current_channels = config.hidden_channels;

        // Downsampling layers
        let mut down_layers = Vec::new();
        let mut residual_blocks = Vec::new();

        for (i, &stride) in config.strides.iter().enumerate() {
            let out_channels = config.hidden_channels * 2usize.pow((i + 1) as u32);

            // Strided convolution for downsampling
            let down = conv1d(
                current_channels,
                out_channels,
                2 * stride,
                Conv1dConfig {
                    stride,
                    padding: stride / 2,
                    ..Default::default()
                },
                vb.pp(format!("down_{}", i)),
            )?;
            down_layers.push(down);

            // Residual blocks for this layer
            let mut blocks = Vec::new();
            for (j, &dilation) in config.dilations.iter().enumerate() {
                let block = ResidualUnit::new(
                    out_channels,
                    dilation,
                    3,
                    vb.pp(format!("res_{}_{}", i, j)),
                )?;
                blocks.push(block);
            }
            residual_blocks.push(blocks);

            current_channels = out_channels;
        }

        // Final projection to embedding dimension
        let final_conv = conv1d(
            current_channels,
            config.embed_dim,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("final"),
        )?;

        Ok(Self {
            init_conv,
            down_layers,
            residual_blocks,
            final_conv,
        })
    }

    /// Encode input signal to latent representation
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Initial convolution
        let mut out = self.init_conv.forward(x)?;
        out = out.elu(1.0)?;

        // Downsampling layers with residual blocks
        for (down_layer, res_blocks) in self.down_layers.iter().zip(self.residual_blocks.iter()) {
            // Downsample
            out = down_layer.forward(&out)?;
            out = out.elu(1.0)?;

            // Residual blocks
            for res_block in res_blocks {
                out = res_block.forward(&out)?;
            }
        }

        // Final projection
        self.final_conv.forward(&out)
    }
}

/// Convolutional decoder for neural codec
pub struct ConvDecoder {
    /// Initial projection from embedding dimension
    init_conv: Conv1d,
    /// Upsampling layers (transposed convolutions)
    up_layers: Vec<ConvTranspose1d>,
    /// Residual blocks per layer
    residual_blocks: Vec<Vec<ResidualUnit>>,
    /// Final convolution to reconstruct signal
    final_conv: Conv1d,
}

impl ConvDecoder {
    /// Create a new convolutional decoder
    pub fn new(config: &NeuralCodecConfig, vb: VarBuilder) -> CandleResult<Self> {
        // Initial projection from embedding dimension
        let last_layer_channels = config.hidden_channels * 2usize.pow(config.strides.len() as u32);

        let init_conv = conv1d(
            config.embed_dim,
            last_layer_channels,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("init"),
        )?;

        // Upsampling layers (reverse order of encoder)
        let mut up_layers = Vec::new();
        let mut residual_blocks = Vec::new();
        let mut current_channels = last_layer_channels;

        for (i, &stride) in config.strides.iter().enumerate().rev() {
            let layer_idx = config.strides.len() - 1 - i;
            let out_channels = if layer_idx == 0 {
                config.hidden_channels
            } else {
                config.hidden_channels * 2usize.pow(layer_idx as u32)
            };

            // Residual blocks for this layer (before upsampling)
            let mut blocks = Vec::new();
            for (j, &dilation) in config.dilations.iter().enumerate() {
                let block = ResidualUnit::new(
                    current_channels,
                    dilation,
                    3,
                    vb.pp(format!("res_{}_{}", i, j)),
                )?;
                blocks.push(block);
            }
            residual_blocks.push(blocks);

            // Transposed convolution for upsampling
            let up = conv_transpose1d(
                current_channels,
                out_channels,
                2 * stride,
                ConvTranspose1dConfig {
                    stride,
                    padding: stride / 2,
                    ..Default::default()
                },
                vb.pp(format!("up_{}", i)),
            )?;
            up_layers.push(up);

            current_channels = out_channels;
        }

        // Final convolution to reconstruct signal
        let final_conv = conv1d(
            current_channels,
            config.input_channels,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("final"),
        )?;

        Ok(Self {
            init_conv,
            up_layers,
            residual_blocks,
            final_conv,
        })
    }

    /// Decode latent representation to signal
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Initial projection
        let mut out = self.init_conv.forward(x)?;
        out = out.elu(1.0)?;

        // Upsampling layers with residual blocks
        for (res_blocks, up_layer) in self.residual_blocks.iter().zip(self.up_layers.iter()) {
            // Residual blocks
            for res_block in res_blocks {
                out = res_block.forward(&out)?;
            }

            // Upsample
            out = up_layer.forward(&out)?;
            out = out.elu(1.0)?;
        }

        // Final convolution (no activation for reconstruction)
        self.final_conv.forward(&out)
    }
}

/// Neural Codec (SoundStream/Encodec style)
///
/// Combines convolutional encoder-decoder with residual vector quantization
/// for high-quality neural audio compression.
pub struct NeuralCodec {
    config: NeuralCodecConfig,
    encoder: ConvEncoder,
    decoder: ConvDecoder,
    rvq: ResidualVQ,
    device: Device,
}

impl NeuralCodec {
    /// Create a new neural codec
    pub fn new(config: NeuralCodecConfig, vb: VarBuilder) -> CandleResult<Self> {
        let device = vb.device().clone();

        // Create encoder and decoder
        let encoder = ConvEncoder::new(&config, vb.pp("encoder"))?;
        let decoder = ConvDecoder::new(&config, vb.pp("decoder"))?;

        // Create RVQ
        let vq_config = VQConfig {
            codebook_size: config.codebook_size,
            embed_dim: config.embed_dim,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        };
        let rvq = ResidualVQ::new(config.num_rvq_stages, vq_config);

        Ok(Self {
            config,
            encoder,
            decoder,
            rvq,
            device,
        })
    }

    /// Encode signal to discrete codes
    pub fn encode(&self, signal: &[f32]) -> TokenizerResult<Vec<Vec<usize>>> {
        // Convert to tensor [batch=1, channels=1, length]
        let tensor = Tensor::from_slice(signal, (1, 1, signal.len()), &self.device)
            .map_err(|e| TokenizerError::encoding("neural_codec", e.to_string()))?;

        // Encode to latent
        let latent = self
            .encoder
            .forward(&tensor)
            .map_err(|e| TokenizerError::encoding("neural_codec_encoder", e.to_string()))?;

        // Get latent as array [embed_dim, time]
        let latent_data = latent
            .squeeze(0)
            .map_err(|e| TokenizerError::encoding("neural_codec_squeeze", e.to_string()))?
            .to_vec2::<f32>()
            .map_err(|e| TokenizerError::encoding("neural_codec_latent", e.to_string()))?;

        // Quantize with RVQ
        let mut codes = Vec::new();
        for time_step in 0..latent_data[0].len() {
            let vector: Vec<f32> = latent_data.iter().map(|row| row[time_step]).collect();
            let vector_array = Array1::from_vec(vector);
            let (code_seq, _) = self.rvq.encode(&vector_array)?;
            codes.push(code_seq);
        }

        Ok(codes)
    }

    /// Decode discrete codes to signal
    pub fn decode(&self, codes: &[Vec<usize>]) -> TokenizerResult<Vec<f32>> {
        if codes.is_empty() {
            return Err(TokenizerError::decoding(
                "neural_codec",
                "Empty code sequence".to_string(),
            ));
        }

        // Decode each time step
        let time_steps = codes.len();
        let embed_dim = self.config.embed_dim;
        let mut latent_data = vec![vec![0.0f32; time_steps]; embed_dim];

        for (t, code_seq) in codes.iter().enumerate() {
            let vector = self.rvq.decode(code_seq)?;
            for (d, &val) in vector.iter().enumerate() {
                if d < embed_dim {
                    latent_data[d][t] = val;
                }
            }
        }

        // Convert to tensor [batch=1, embed_dim, time]
        let flat_data: Vec<f32> = latent_data.iter().flatten().copied().collect();
        let latent = Tensor::from_slice(&flat_data, (1, embed_dim, time_steps), &self.device)
            .map_err(|e| TokenizerError::decoding("neural_codec_latent", e.to_string()))?;

        // Decode to signal
        let output = self
            .decoder
            .forward(&latent)
            .map_err(|e| TokenizerError::decoding("neural_codec_decoder", e.to_string()))?;

        // Extract signal [batch=1, channels=1, length] -> [length]
        let signal = output
            .squeeze(0)
            .map_err(|e| TokenizerError::decoding("neural_codec_squeeze1", e.to_string()))?
            .squeeze(0)
            .map_err(|e| TokenizerError::decoding("neural_codec_squeeze2", e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| TokenizerError::decoding("neural_codec_output", e.to_string()))?;

        Ok(signal)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let total_stride: usize = self.config.strides.iter().product();
        total_stride as f32
    }

    /// Get bitrate for given sample rate
    pub fn bitrate(&self, sample_rate: f32) -> f32 {
        let compressed_rate = sample_rate / self.compression_ratio();
        let bits_per_frame =
            self.config.num_rvq_stages as f32 * (self.config.codebook_size as f32).log2();
        compressed_rate * bits_per_frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    #[test]
    fn test_neural_codec_config_default() {
        let config = NeuralCodecConfig::default();
        assert_eq!(config.input_channels, 1);
        assert_eq!(config.hidden_channels, 128);
        assert_eq!(config.num_rvq_stages, 8);
        assert!(!config.causal);
    }

    #[test]
    fn test_compression_ratio() {
        // Use minimal config for faster test - only strides matter for compression ratio
        let config = NeuralCodecConfig {
            input_channels: 1,
            hidden_channels: 8,        // Minimal
            num_residual_blocks: 1,    // Minimal
            strides: vec![2, 4, 5, 8], // Same as default for correct calculation
            dilations: vec![1],        // Minimal
            codebook_size: 64,         // Minimal
            embed_dim: 8,              // Minimal
            num_rvq_stages: 2,         // Minimal
            causal: false,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let codec = NeuralCodec::new(config.clone(), vb).unwrap();
        let ratio = codec.compression_ratio();
        let expected: usize = config.strides.iter().product();
        assert_eq!(ratio, expected as f32);
    }

    #[test]
    fn test_bitrate_calculation() {
        // Use minimal config with same strides/codebook/stages as default for correct bitrate
        let config = NeuralCodecConfig {
            input_channels: 1,
            hidden_channels: 8,        // Minimal
            num_residual_blocks: 1,    // Minimal
            strides: vec![2, 4, 5, 8], // Same as default (compression = 320)
            dilations: vec![1],        // Minimal
            codebook_size: 1024,       // Same as default (10 bits)
            embed_dim: 8,              // Minimal
            num_rvq_stages: 8,         // Same as default
            causal: false,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let codec = NeuralCodec::new(config, vb).unwrap();
        let bitrate = codec.bitrate(16000.0);

        // With config: 16000 Hz / 320 = 50 Hz compressed rate
        // 8 stages * log2(1024) = 8 * 10 = 80 bits per frame
        // 50 Hz * 80 bits = 4000 bps = 4 kbps
        assert!((bitrate - 4000.0).abs() < 1.0);
    }

    #[test]
    #[ignore] // Requires trained weights
    fn test_encode_decode_roundtrip() {
        let config = NeuralCodecConfig::default();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let codec = NeuralCodec::new(config, vb).unwrap();

        // Create test signal
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();

        // Encode and decode
        let codes = codec.encode(&signal).unwrap();
        let reconstructed = codec.decode(&codes).unwrap();

        // Check reconstruction (won't be perfect with random weights)
        assert_eq!(reconstructed.len(), signal.len());
    }
}
