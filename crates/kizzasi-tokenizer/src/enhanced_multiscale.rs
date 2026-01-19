//! Enhanced multi-scale tokenization with advanced features
//!
//! This module extends the basic multi-scale tokenizer with:
//! - Wavelet decomposition for natural multi-scale analysis
//! - Learnable pooling/unpooling operations
//! - Attention-based scale fusion
//! - Cross-scale information flow
//!
//! These techniques enable better representation of signals at multiple scales
//! while allowing the model to learn optimal scale interactions.

use crate::error::{TokenizerError, TokenizerResult};
use crate::specialized::WaveletFamily;
use crate::SignalTokenizer;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::thread_rng;

/// Wavelet-based multi-scale tokenizer
///
/// Uses wavelet decomposition to naturally separate signal into multiple scales.
/// Each decomposition level produces approximation and detail coefficients.
pub struct WaveletMultiScaleTokenizer {
    /// Number of decomposition levels
    num_levels: usize,
    /// Wavelet family
    wavelet: WaveletFamily,
    /// Encoder for each level (approximation + details)
    encoders: Vec<Array2<f32>>,
    /// Decoder for each level
    decoders: Vec<Array2<f32>>,
    /// Embedding dimension per level
    embed_dim: usize,
    /// Original signal length
    signal_len: usize,
}

impl WaveletMultiScaleTokenizer {
    /// Create a new wavelet-based multi-scale tokenizer
    ///
    /// # Arguments
    ///
    /// * `signal_len` - Length of input signal (should be power of 2)
    /// * `num_levels` - Number of wavelet decomposition levels
    /// * `embed_dim` - Embedding dimension per level
    /// * `wavelet` - Wavelet family to use
    pub fn new(
        signal_len: usize,
        num_levels: usize,
        embed_dim: usize,
        wavelet: WaveletFamily,
    ) -> TokenizerResult<Self> {
        if !signal_len.is_power_of_two() {
            return Err(TokenizerError::InvalidConfig(
                "Signal length must be power of 2 for wavelet decomposition".into(),
            ));
        }

        let mut rng = thread_rng();
        let mut encoders = Vec::new();
        let mut decoders = Vec::new();

        // Create encoder/decoder for each level
        let mut level_len = signal_len;
        for _ in 0..num_levels {
            level_len /= 2;

            // Each level produces 2*level_len coefficients (approx + detail)
            let coeff_dim = 2 * level_len;

            // Xavier initialization
            let enc_scale = (2.0 / (coeff_dim + embed_dim) as f32).sqrt();
            let encoder = Array2::from_shape_fn((coeff_dim, embed_dim), |_| {
                (rng.random::<f32>() - 0.5) * 2.0 * enc_scale
            });

            let dec_scale = (2.0 / (embed_dim + coeff_dim) as f32).sqrt();
            let decoder = Array2::from_shape_fn((embed_dim, coeff_dim), |_| {
                (rng.random::<f32>() - 0.5) * 2.0 * dec_scale
            });

            encoders.push(encoder);
            decoders.push(decoder);
        }

        Ok(Self {
            num_levels,
            wavelet,
            encoders,
            decoders,
            embed_dim,
            signal_len,
        })
    }

    /// Perform wavelet decomposition
    fn decompose(&self, signal: &Array1<f32>) -> Vec<(Array1<f32>, Array1<f32>)> {
        let mut levels = Vec::new();
        let mut current = signal.clone();

        for _ in 0..self.num_levels {
            let (approx, detail) = self.wavelet_transform(&current);
            levels.push((approx.clone(), detail));
            current = approx;
        }

        levels
    }

    /// Simple wavelet transform (using Haar for efficiency)
    fn wavelet_transform(&self, signal: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let len = signal.len();
        let half_len = len / 2;

        let mut approx = Array1::zeros(half_len);
        let mut detail = Array1::zeros(half_len);

        match self.wavelet {
            WaveletFamily::Haar => {
                for i in 0..half_len {
                    let even = signal[2 * i];
                    let odd = signal[2 * i + 1];
                    approx[i] = (even + odd) / 2.0_f32.sqrt();
                    detail[i] = (even - odd) / 2.0_f32.sqrt();
                }
            }
            WaveletFamily::Daubechies4 => {
                // Daubechies-4 coefficients
                let h0 = 0.6830127;
                let h1 = 1.1830127;
                let h2 = 0.3169873;
                let h3 = -0.1830127;

                for i in 0..half_len {
                    let i0 = 2 * i;
                    let i1 = (2 * i + 1) % len;
                    let i2 = (2 * i + 2) % len;
                    let i3 = (2 * i + 3) % len;

                    approx[i] =
                        h0 * signal[i0] + h1 * signal[i1] + h2 * signal[i2] + h3 * signal[i3];
                    detail[i] =
                        h3 * signal[i0] - h2 * signal[i1] + h1 * signal[i2] - h0 * signal[i3];
                }
            }
        }

        (approx, detail)
    }

    /// Inverse wavelet transform
    fn wavelet_inverse(&self, approx: &Array1<f32>, detail: &Array1<f32>) -> Array1<f32> {
        let half_len = approx.len();
        let len = 2 * half_len;
        let mut signal = Array1::zeros(len);

        match self.wavelet {
            WaveletFamily::Haar => {
                for i in 0..half_len {
                    let a = approx[i];
                    let d = detail[i];
                    signal[2 * i] = (a + d) / 2.0_f32.sqrt();
                    signal[2 * i + 1] = (a - d) / 2.0_f32.sqrt();
                }
            }
            WaveletFamily::Daubechies4 => {
                // Inverse Daubechies-4
                let g0 = -0.1830127;
                let g1 = 0.3169873;
                let g2 = 1.1830127;
                let g3 = 0.6830127;

                for i in 0..half_len {
                    let a = approx[i];
                    let d = detail[i];

                    let i0 = 2 * i;
                    let i1 = (2 * i + 1) % len;

                    signal[i0] += g0 * d + g3 * a;
                    signal[i1] += g1 * d + g2 * a;
                }
            }
        }

        signal
    }
}

impl SignalTokenizer for WaveletMultiScaleTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if signal.len() != self.signal_len {
            return Err(TokenizerError::dim_mismatch(
                self.signal_len,
                signal.len(),
                "dimension validation",
            ));
        }

        // Decompose into wavelet levels
        let levels = self.decompose(signal);

        // Encode each level
        let mut embeddings = Vec::new();
        for (i, (approx, detail)) in levels.iter().enumerate() {
            // Concatenate approx and detail coefficients
            let mut coeffs = Array1::zeros(approx.len() + detail.len());
            for (j, &val) in approx.iter().enumerate() {
                coeffs[j] = val;
            }
            for (j, &val) in detail.iter().enumerate() {
                coeffs[approx.len() + j] = val;
            }

            // Project to embedding space
            let embedding = coeffs.dot(&self.encoders[i]);
            embeddings.push(embedding);
        }

        // Concatenate all level embeddings
        let total_dim = embeddings.len() * self.embed_dim;
        let mut result = Array1::zeros(total_dim);
        for (i, emb) in embeddings.iter().enumerate() {
            for (j, &val) in emb.iter().enumerate() {
                result[i * self.embed_dim + j] = val;
            }
        }

        Ok(result)
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let expected_dim = self.num_levels * self.embed_dim;
        if tokens.len() != expected_dim {
            return Err(TokenizerError::dim_mismatch(
                expected_dim,
                tokens.len(),
                "dimension validation",
            ));
        }

        // Split tokens into per-level embeddings
        let mut coeffs_levels = Vec::new();
        for i in 0..self.num_levels {
            let start = i * self.embed_dim;
            let end = start + self.embed_dim;
            let level_tokens = tokens.slice(s![start..end]).to_owned();

            // Decode to wavelet coefficients
            let coeffs = level_tokens.dot(&self.decoders[i]);
            coeffs_levels.push(coeffs);
        }

        // Reconstruct signal from wavelet coefficients
        let mut current = {
            let coeffs = &coeffs_levels[self.num_levels - 1];
            let half_len = coeffs.len() / 2;
            let approx = coeffs.slice(s![0..half_len]).to_owned();
            let detail = coeffs.slice(s![half_len..]).to_owned();
            self.wavelet_inverse(&approx, &detail)
        };

        // Inverse transform for each level
        for i in (0..self.num_levels - 1).rev() {
            let coeffs = &coeffs_levels[i];
            let half_len = coeffs.len() / 2;
            let detail = coeffs.slice(s![half_len..]).to_owned();

            current = self.wavelet_inverse(&current, &detail);
        }

        Ok(current)
    }

    fn embed_dim(&self) -> usize {
        self.num_levels * self.embed_dim
    }

    fn vocab_size(&self) -> usize {
        0 // Continuous
    }
}

/// Learnable pooling operation for downsampling
pub struct LearnablePooling {
    /// Pooling kernel weights [kernel_size]
    kernel: Array1<f32>,
    /// Stride
    stride: usize,
}

impl LearnablePooling {
    /// Create a new learnable pooling layer
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        let mut rng = thread_rng();

        // Initialize with normalized weights (sum to 1)
        let mut kernel = Array1::from_shape_fn(kernel_size, |_| rng.random::<f32>());
        let sum: f32 = kernel.iter().sum();
        kernel.mapv_inplace(|x| x / sum);

        Self { kernel, stride }
    }

    /// Apply pooling to signal
    pub fn pool(&self, signal: &Array1<f32>) -> Array1<f32> {
        let kernel_size = self.kernel.len();
        let output_len = (signal.len() - kernel_size) / self.stride + 1;
        let mut output = Array1::zeros(output_len);

        for i in 0..output_len {
            let start = i * self.stride;
            let mut sum = 0.0;
            for (k, &weight) in self.kernel.iter().enumerate() {
                if start + k < signal.len() {
                    sum += signal[start + k] * weight;
                }
            }
            output[i] = sum;
        }

        output
    }

    /// Get kernel weights (for gradient updates)
    pub fn kernel(&self) -> &Array1<f32> {
        &self.kernel
    }

    /// Update kernel weights
    pub fn update_kernel(&mut self, new_kernel: Array1<f32>) -> TokenizerResult<()> {
        if new_kernel.len() != self.kernel.len() {
            return Err(TokenizerError::InvalidConfig("Kernel size mismatch".into()));
        }

        // Normalize to sum to 1
        let sum: f32 = new_kernel.iter().sum();
        self.kernel = new_kernel.mapv(|x| x / sum);

        Ok(())
    }
}

/// Attention-based scale fusion
///
/// Learns to combine information from multiple scales using attention mechanism.
pub struct AttentionScaleFusion {
    /// Query projection per scale
    query_proj: Vec<Array2<f32>>,
    /// Key projection per scale
    key_proj: Vec<Array2<f32>>,
    /// Value projection per scale
    value_proj: Vec<Array2<f32>>,
    /// Number of scales
    num_scales: usize,
    /// Embedding dimension
    embed_dim: usize,
}

impl AttentionScaleFusion {
    /// Create a new attention-based scale fusion layer
    ///
    /// # Arguments
    ///
    /// * `num_scales` - Number of scales to fuse
    /// * `embed_dim` - Embedding dimension for each scale
    pub fn new(num_scales: usize, embed_dim: usize) -> Self {
        let mut rng = thread_rng();
        let scale = (1.0 / embed_dim as f32).sqrt();

        let mut query_proj = Vec::new();
        let mut key_proj = Vec::new();
        let mut value_proj = Vec::new();

        for _ in 0..num_scales {
            query_proj.push(Array2::from_shape_fn((embed_dim, embed_dim), |_| {
                (rng.random::<f32>() - 0.5) * 2.0 * scale
            }));
            key_proj.push(Array2::from_shape_fn((embed_dim, embed_dim), |_| {
                (rng.random::<f32>() - 0.5) * 2.0 * scale
            }));
            value_proj.push(Array2::from_shape_fn((embed_dim, embed_dim), |_| {
                (rng.random::<f32>() - 0.5) * 2.0 * scale
            }));
        }

        Self {
            query_proj,
            key_proj,
            value_proj,
            num_scales,
            embed_dim,
        }
    }

    /// Fuse embeddings from multiple scales using attention
    ///
    /// # Arguments
    ///
    /// * `scale_embeddings` - Embeddings from each scale [num_scales, embed_dim]
    ///
    /// # Returns
    ///
    /// Fused embedding [embed_dim]
    pub fn fuse(&self, scale_embeddings: &[Array1<f32>]) -> TokenizerResult<Array1<f32>> {
        if scale_embeddings.len() != self.num_scales {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected {} scales, got {}",
                self.num_scales,
                scale_embeddings.len()
            )));
        }

        // Compute queries, keys, values for each scale
        let mut queries = Vec::new();
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for (i, emb) in scale_embeddings.iter().enumerate() {
            queries.push(emb.dot(&self.query_proj[i]));
            keys.push(emb.dot(&self.key_proj[i]));
            values.push(emb.dot(&self.value_proj[i]));
        }

        // Compute attention scores: Q * K^T / sqrt(d)
        let scale_factor = (self.embed_dim as f32).sqrt();
        let mut attention_weights = Array2::zeros((self.num_scales, self.num_scales));

        for i in 0..self.num_scales {
            for j in 0..self.num_scales {
                let score = queries[i].dot(&keys[j]) / scale_factor;
                attention_weights[[i, j]] = score;
            }
        }

        // Softmax over attention weights
        for i in 0..self.num_scales {
            let row_max = attention_weights
                .row(i)
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let mut exp_sum = 0.0;
            for j in 0..self.num_scales {
                let exp_val = (attention_weights[[i, j]] - row_max).exp();
                attention_weights[[i, j]] = exp_val;
                exp_sum += exp_val;
            }

            for j in 0..self.num_scales {
                attention_weights[[i, j]] /= exp_sum;
            }
        }

        // Weighted sum of values
        let mut fused = Array1::zeros(self.embed_dim);
        for i in 0..self.num_scales {
            for j in 0..self.num_scales {
                let weight = attention_weights[[i, j]];
                for k in 0..self.embed_dim {
                    fused[k] += weight * values[j][k];
                }
            }
        }

        // Average over query scales
        fused.mapv_inplace(|x| x / self.num_scales as f32);

        Ok(fused)
    }
}

/// Cross-scale information flow using residual connections
///
/// Allows information to flow between different scales during encoding/decoding.
pub struct CrossScaleFlow {
    /// Skip connections from fine to coarse scales
    fine_to_coarse: Vec<Array2<f32>>,
    /// Skip connections from coarse to fine scales
    coarse_to_fine: Vec<Array2<f32>>,
    /// Number of scales
    num_scales: usize,
    /// Embedding dimension
    embed_dim: usize,
}

impl CrossScaleFlow {
    /// Create a new cross-scale flow module
    pub fn new(num_scales: usize, embed_dim: usize) -> Self {
        let mut rng = thread_rng();
        let scale = (2.0 / (2.0 * embed_dim as f32)).sqrt();

        let mut fine_to_coarse = Vec::new();
        let mut coarse_to_fine = Vec::new();

        for _ in 0..num_scales - 1 {
            fine_to_coarse.push(Array2::from_shape_fn((embed_dim, embed_dim), |_| {
                (rng.random::<f32>() - 0.5) * 2.0 * scale
            }));
            coarse_to_fine.push(Array2::from_shape_fn((embed_dim, embed_dim), |_| {
                (rng.random::<f32>() - 0.5) * 2.0 * scale
            }));
        }

        Self {
            fine_to_coarse,
            coarse_to_fine,
            num_scales,
            embed_dim,
        }
    }

    /// Apply cross-scale connections during encoding (fine to coarse)
    ///
    /// # Arguments
    ///
    /// * `scale_embeddings` - Embeddings from each scale (fine to coarse order)
    ///
    /// # Returns
    ///
    /// Updated embeddings with cross-scale information
    pub fn encode_flow(
        &self,
        scale_embeddings: &[Array1<f32>],
    ) -> TokenizerResult<Vec<Array1<f32>>> {
        if scale_embeddings.len() != self.num_scales {
            return Err(TokenizerError::InvalidConfig("Scale count mismatch".into()));
        }

        let mut result = Vec::new();
        result.push(scale_embeddings[0].clone());

        // Each coarser scale receives information from finer scale
        for i in 1..self.num_scales {
            let skip = scale_embeddings[i - 1].dot(&self.fine_to_coarse[i - 1]);
            let combined = &scale_embeddings[i] + &skip;
            result.push(combined);
        }

        Ok(result)
    }

    /// Apply cross-scale connections during decoding (coarse to fine)
    pub fn decode_flow(
        &self,
        scale_embeddings: &[Array1<f32>],
    ) -> TokenizerResult<Vec<Array1<f32>>> {
        if scale_embeddings.len() != self.num_scales {
            return Err(TokenizerError::InvalidConfig("Scale count mismatch".into()));
        }

        let mut result = vec![Array1::zeros(self.embed_dim); self.num_scales];
        result[self.num_scales - 1] = scale_embeddings[self.num_scales - 1].clone();

        // Each finer scale receives information from coarser scale
        for i in (0..self.num_scales - 1).rev() {
            let skip = result[i + 1].dot(&self.coarse_to_fine[i]);
            result[i] = &scale_embeddings[i] + &skip;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelet_multiscale_creation() {
        let tokenizer = WaveletMultiScaleTokenizer::new(64, 3, 16, WaveletFamily::Haar).unwrap();

        assert_eq!(tokenizer.num_levels, 3);
        assert_eq!(tokenizer.embed_dim(), 48); // 3 levels * 16 dim
    }

    #[test]
    fn test_wavelet_multiscale_encode_decode() {
        let tokenizer = WaveletMultiScaleTokenizer::new(64, 2, 8, WaveletFamily::Haar).unwrap();

        let signal = Array1::from_vec((0..64).map(|i| (i as f32 * 0.1).sin()).collect());

        let encoded = tokenizer.encode(&signal).unwrap();
        assert_eq!(encoded.len(), 16); // 2 levels * 8 dim

        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 64);
    }

    #[test]
    fn test_learnable_pooling() {
        let pooling = LearnablePooling::new(3, 2);

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let pooled = pooling.pool(&signal);

        assert_eq!(pooled.len(), 2);
    }

    #[test]
    fn test_attention_scale_fusion() {
        let fusion = AttentionScaleFusion::new(3, 8);

        let embeddings = vec![
            Array1::from_vec(vec![1.0; 8]),
            Array1::from_vec(vec![2.0; 8]),
            Array1::from_vec(vec![3.0; 8]),
        ];

        let fused = fusion.fuse(&embeddings).unwrap();
        assert_eq!(fused.len(), 8);
    }

    #[test]
    fn test_cross_scale_flow() {
        let flow = CrossScaleFlow::new(3, 8);

        let embeddings = vec![
            Array1::from_vec(vec![1.0; 8]),
            Array1::from_vec(vec![2.0; 8]),
            Array1::from_vec(vec![3.0; 8]),
        ];

        let encoded = flow.encode_flow(&embeddings).unwrap();
        assert_eq!(encoded.len(), 3);

        let decoded = flow.decode_flow(&embeddings).unwrap();
        assert_eq!(decoded.len(), 3);
    }
}
