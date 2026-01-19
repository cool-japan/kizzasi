//! Transformer-based signal tokenization using self-attention mechanisms.
//!
//! This module implements a modern transformer architecture for signal tokenization,
//! featuring multi-head self-attention, positional encoding, and encoder-decoder structure.
//! Inspired by "Attention Is All You Need" (Vaswani et al., 2017).
//!
//! # Architecture
//!
//! - Multi-head self-attention for capturing global dependencies
//! - Positional encoding for sequence order information
//! - Feed-forward networks for non-linear transformations
//! - Layer normalization for training stability
//! - Residual connections for gradient flow
//!
//! # Example
//!
//! ```
//! use kizzasi_tokenizer::{TransformerTokenizer, TransformerConfig, SignalTokenizer};
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = TransformerConfig {
//!     input_dim: 128,
//!     embed_dim: 256,
//!     num_heads: 8,
//!     num_encoder_layers: 4,
//!     num_decoder_layers: 4,
//!     feedforward_dim: 1024,
//!     dropout: 0.1,
//!     max_seq_len: 512,
//! };
//!
//! let tokenizer = TransformerTokenizer::new(config)?;
//! let signal = Array1::linspace(0.0, 1.0, 128);
//! let tokens = tokenizer.encode(&signal)?;
//! let reconstructed = tokenizer.decode(&tokens)?;
//! # Ok(())
//! # }
//! ```

use crate::error::{TokenizerError, TokenizerResult};
use crate::SignalTokenizer;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::{rngs::StdRng, Random};
use serde::{Deserialize, Serialize};

/// Configuration for the Transformer tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Input signal dimension
    pub input_dim: usize,
    /// Embedding dimension (must be divisible by num_heads)
    pub embed_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of encoder layers
    pub num_encoder_layers: usize,
    /// Number of decoder layers
    pub num_decoder_layers: usize,
    /// Feedforward network hidden dimension
    pub feedforward_dim: usize,
    /// Dropout probability (0.0 to 1.0)
    pub dropout: f32,
    /// Maximum sequence length for positional encoding
    pub max_seq_len: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 128,
            embed_dim: 256,
            num_heads: 8,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            feedforward_dim: 1024,
            dropout: 0.1,
            max_seq_len: 512,
        }
    }
}

impl TransformerConfig {
    /// Validate the configuration
    pub fn validate(&self) -> TokenizerResult<()> {
        if self.input_dim == 0 {
            return Err(TokenizerError::invalid_input(
                "input_dim must be positive",
                "TransformerConfig::validate",
            ));
        }
        if self.embed_dim == 0 {
            return Err(TokenizerError::invalid_input(
                "embed_dim must be positive",
                "TransformerConfig::validate",
            ));
        }
        if !self.embed_dim.is_multiple_of(self.num_heads) {
            return Err(TokenizerError::invalid_input(
                "embed_dim must be divisible by num_heads",
                "TransformerConfig::validate",
            ));
        }
        if self.num_heads == 0 {
            return Err(TokenizerError::invalid_input(
                "num_heads must be positive",
                "TransformerConfig::validate",
            ));
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(TokenizerError::invalid_input(
                "dropout must be in range [0.0, 1.0]",
                "TransformerConfig::validate",
            ));
        }
        if self.max_seq_len == 0 {
            return Err(TokenizerError::invalid_input(
                "max_seq_len must be positive",
                "TransformerConfig::validate",
            ));
        }
        Ok(())
    }
}

/// Multi-head self-attention mechanism
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Query projection weights [embed_dim, embed_dim]
    w_query: Array2<f32>,
    /// Key projection weights [embed_dim, embed_dim]
    w_key: Array2<f32>,
    /// Value projection weights [embed_dim, embed_dim]
    w_value: Array2<f32>,
    /// Output projection weights [embed_dim, embed_dim]
    w_out: Array2<f32>,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    pub fn new(embed_dim: usize, num_heads: usize) -> TokenizerResult<Self> {
        if !embed_dim.is_multiple_of(num_heads) {
            return Err(TokenizerError::invalid_input(
                "embed_dim must be divisible by num_heads",
                "MultiHeadAttention::new",
            ));
        }

        let head_dim = embed_dim / num_heads;
        let mut rng = Random::seed(42);

        // Xavier/Glorot initialization
        let scale = (2.0 / (embed_dim + embed_dim) as f32).sqrt();

        Ok(Self {
            num_heads,
            head_dim,
            w_query: Self::init_weights(embed_dim, embed_dim, scale, &mut rng),
            w_key: Self::init_weights(embed_dim, embed_dim, scale, &mut rng),
            w_value: Self::init_weights(embed_dim, embed_dim, scale, &mut rng),
            w_out: Self::init_weights(embed_dim, embed_dim, scale, &mut rng),
        })
    }

    /// Initialize weights with Xavier/Glorot uniform distribution
    fn init_weights(rows: usize, cols: usize, scale: f32, rng: &mut Random<StdRng>) -> Array2<f32> {
        let mut weights = Array2::zeros((rows, cols));
        for val in weights.iter_mut() {
            *val = (rng.gen_range(-1.0..1.0)) * scale;
        }
        weights
    }

    /// Forward pass through multi-head attention
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [seq_len, embed_dim]
    ///
    /// # Returns
    ///
    /// Output tensor [seq_len, embed_dim]
    pub fn forward(&self, x: &Array2<f32>) -> TokenizerResult<Array2<f32>> {
        let seq_len = x.nrows();
        let embed_dim = x.ncols();

        // Linear projections: Q, K, V = x @ W_q, x @ W_k, x @ W_v
        let query = x.dot(&self.w_query); // [seq_len, embed_dim]
        let key = x.dot(&self.w_key); // [seq_len, embed_dim]
        let value = x.dot(&self.w_value); // [seq_len, embed_dim]

        // Scaled dot-product attention for each head
        let scale = (self.head_dim as f32).sqrt();
        let mut attention_output = Array2::zeros((seq_len, embed_dim));

        for h in 0..self.num_heads {
            // Extract Q, K, V for this head
            let mut q_head = Array2::zeros((seq_len, self.head_dim));
            let mut k_head = Array2::zeros((seq_len, self.head_dim));
            let mut v_head = Array2::zeros((seq_len, self.head_dim));

            let start_idx = h * self.head_dim;
            for i in 0..seq_len {
                for j in 0..self.head_dim {
                    q_head[[i, j]] = query[[i, start_idx + j]];
                    k_head[[i, j]] = key[[i, start_idx + j]];
                    v_head[[i, j]] = value[[i, start_idx + j]];
                }
            }

            // Attention scores: Q @ K^T / sqrt(d_k)
            let scores = q_head.dot(&k_head.t()) / scale; // [seq_len, seq_len]

            // Softmax over the last dimension
            let attention_weights = Self::softmax(&scores)?;

            // Weighted sum: softmax(scores) @ V
            let head_output = attention_weights.dot(&v_head); // [seq_len, head_dim]

            // Copy to output tensor
            for i in 0..seq_len {
                for j in 0..self.head_dim {
                    attention_output[[i, start_idx + j]] = head_output[[i, j]];
                }
            }
        }

        // Final linear projection
        Ok(attention_output.dot(&self.w_out))
    }

    /// Apply softmax to each row
    fn softmax(x: &Array2<f32>) -> TokenizerResult<Array2<f32>> {
        let mut result = x.clone();
        for mut row in result.rows_mut() {
            // Subtract max for numerical stability
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
            }
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }
        Ok(result)
    }
}

/// Positional encoding using sinusoidal functions
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    /// Pre-computed positional encodings [max_seq_len, embed_dim]
    encodings: Array2<f32>,
}

impl PositionalEncoding {
    /// Create a new positional encoding
    pub fn new(max_seq_len: usize, embed_dim: usize) -> Self {
        let mut encodings = Array2::zeros((max_seq_len, embed_dim));

        for pos in 0..max_seq_len {
            for i in 0..embed_dim {
                let angle = pos as f32 / 10000.0_f32.powf(2.0 * (i / 2) as f32 / embed_dim as f32);
                if i % 2 == 0 {
                    encodings[[pos, i]] = angle.sin();
                } else {
                    encodings[[pos, i]] = angle.cos();
                }
            }
        }

        Self { encodings }
    }

    /// Add positional encoding to input tensor
    pub fn forward(&self, x: &Array2<f32>) -> TokenizerResult<Array2<f32>> {
        let seq_len = x.nrows();
        if seq_len > self.encodings.nrows() {
            return Err(TokenizerError::encoding(
                format!(
                    "Sequence length {} exceeds max_seq_len {}",
                    seq_len,
                    self.encodings.nrows()
                ),
                "PositionalEncoding::forward",
            ));
        }

        let pos_enc = self.encodings.slice(s![0..seq_len, ..]);
        Ok(x + &pos_enc)
    }
}

/// Layer normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Feature dimension
    dim: usize,
    /// Small constant for numerical stability
    eps: f32,
}

impl LayerNorm {
    /// Create a new layer normalization
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { dim, eps }
    }

    /// Apply layer normalization
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut result = x.clone();
        for mut row in result.rows_mut() {
            let mean = row.mean().unwrap_or(0.0);
            let variance = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / self.dim as f32;
            let std = (variance + self.eps).sqrt();

            for val in row.iter_mut() {
                *val = (*val - mean) / std;
            }
        }
        result
    }
}

/// Feedforward network with GELU activation
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// First linear layer weights
    w1: Array2<f32>,
    /// Second linear layer weights
    w2: Array2<f32>,
}

impl FeedForward {
    /// Create a new feedforward network
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = Random::seed(43);
        let scale1 = (2.0 / (embed_dim + hidden_dim) as f32).sqrt();
        let scale2 = (2.0 / (hidden_dim + embed_dim) as f32).sqrt();

        Self {
            w1: Self::init_weights(embed_dim, hidden_dim, scale1, &mut rng),
            w2: Self::init_weights(hidden_dim, embed_dim, scale2, &mut rng),
        }
    }

    /// Initialize weights
    fn init_weights(rows: usize, cols: usize, scale: f32, rng: &mut Random<StdRng>) -> Array2<f32> {
        let mut weights = Array2::zeros((rows, cols));
        for val in weights.iter_mut() {
            *val = (rng.gen_range(-1.0..1.0)) * scale;
        }
        weights
    }

    /// GELU activation function
    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    /// Forward pass
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let hidden = x.dot(&self.w1);
        let activated = hidden.mapv(Self::gelu);
        activated.dot(&self.w2)
    }
}

/// Transformer encoder layer
#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer {
    /// Multi-head attention
    attention: MultiHeadAttention,
    /// Feedforward network
    ffn: FeedForward,
    /// Layer normalization 1
    norm1: LayerNorm,
    /// Layer normalization 2
    norm2: LayerNorm,
}

impl TransformerEncoderLayer {
    /// Create a new encoder layer
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        feedforward_dim: usize,
    ) -> TokenizerResult<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new(embed_dim, num_heads)?,
            ffn: FeedForward::new(embed_dim, feedforward_dim),
            norm1: LayerNorm::new(embed_dim, 1e-5),
            norm2: LayerNorm::new(embed_dim, 1e-5),
        })
    }

    /// Forward pass with residual connections
    pub fn forward(&self, x: &Array2<f32>) -> TokenizerResult<Array2<f32>> {
        // Self-attention with residual
        let attn_out = self.attention.forward(x)?;
        let x = &(x + &attn_out);
        let x_norm = self.norm1.forward(x);

        // Feedforward with residual
        let ffn_out = self.ffn.forward(&x_norm);
        let out = &x_norm + &ffn_out;
        Ok(self.norm2.forward(&out))
    }
}

/// Transformer-based signal tokenizer
#[derive(Debug, Clone)]
pub struct TransformerTokenizer {
    /// Configuration
    config: TransformerConfig,
    /// Input projection
    input_proj: Array2<f32>,
    /// Output projection
    output_proj: Array2<f32>,
    /// Positional encoding
    pos_encoding: PositionalEncoding,
    /// Encoder layers
    encoder_layers: Vec<TransformerEncoderLayer>,
    /// Decoder layers
    decoder_layers: Vec<TransformerEncoderLayer>,
}

impl TransformerTokenizer {
    /// Create a new transformer tokenizer
    pub fn new(config: TransformerConfig) -> TokenizerResult<Self> {
        config.validate()?;

        let mut rng = Random::seed(44);
        let scale_in = (2.0 / (config.input_dim + config.embed_dim) as f32).sqrt();
        let scale_out = (2.0 / (config.embed_dim + config.input_dim) as f32).sqrt();

        // Initialize projection layers
        let mut input_proj = Array2::zeros((config.input_dim, config.embed_dim));
        let mut output_proj = Array2::zeros((config.embed_dim, config.input_dim));

        for val in input_proj.iter_mut() {
            *val = (rng.gen_range(-1.0..1.0)) * scale_in;
        }
        for val in output_proj.iter_mut() {
            *val = (rng.gen_range(-1.0..1.0)) * scale_out;
        }

        // Save these before moving config
        let max_seq_len = config.max_seq_len;
        let embed_dim = config.embed_dim;
        let num_encoder_layers = config.num_encoder_layers;
        let num_decoder_layers = config.num_decoder_layers;
        let num_heads = config.num_heads;
        let feedforward_dim = config.feedforward_dim;

        // Create encoder layers
        let mut encoder_layers = Vec::new();
        for _ in 0..num_encoder_layers {
            encoder_layers.push(TransformerEncoderLayer::new(
                embed_dim,
                num_heads,
                feedforward_dim,
            )?);
        }

        // Create decoder layers
        let mut decoder_layers = Vec::new();
        for _ in 0..num_decoder_layers {
            decoder_layers.push(TransformerEncoderLayer::new(
                embed_dim,
                num_heads,
                feedforward_dim,
            )?);
        }

        Ok(Self {
            config,
            input_proj,
            output_proj,
            pos_encoding: PositionalEncoding::new(max_seq_len, embed_dim),
            encoder_layers,
            decoder_layers,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

impl SignalTokenizer for TransformerTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let len = signal.len();
        if len > self.config.max_seq_len * self.config.input_dim {
            return Err(TokenizerError::encoding(
                format!(
                    "Signal too long: {} > {}",
                    len,
                    self.config.max_seq_len * self.config.input_dim
                ),
                "TransformerTokenizer::encode",
            ));
        }

        // Reshape to [seq_len, input_dim]
        let seq_len = len.div_ceil(self.config.input_dim);
        let mut padded = signal.to_vec();
        padded.resize(seq_len * self.config.input_dim, 0.0);

        let mut x = Array2::zeros((seq_len, self.config.input_dim));
        for i in 0..seq_len {
            for j in 0..self.config.input_dim {
                x[[i, j]] = padded[i * self.config.input_dim + j];
            }
        }

        // Project to embedding space
        let mut x = x.dot(&self.input_proj); // [seq_len, embed_dim]

        // Add positional encoding
        x = self.pos_encoding.forward(&x)?;

        // Encoder layers
        for layer in &self.encoder_layers {
            x = layer.forward(&x)?;
        }

        // Flatten to 1D
        let mut result = Vec::new();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                result.push(x[[i, j]]);
            }
        }

        Ok(Array1::from_vec(result))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let total_len = tokens.len();
        if !total_len.is_multiple_of(self.config.embed_dim) {
            return Err(TokenizerError::decoding(
                format!(
                    "Invalid token length: {} not divisible by {}",
                    total_len, self.config.embed_dim
                ),
                "TransformerTokenizer::decode",
            ));
        }

        let seq_len = total_len / self.config.embed_dim;

        // Reshape to [seq_len, embed_dim]
        let mut x = Array2::zeros((seq_len, self.config.embed_dim));
        for i in 0..seq_len {
            for j in 0..self.config.embed_dim {
                x[[i, j]] = tokens[i * self.config.embed_dim + j];
            }
        }

        // Decoder layers
        for layer in &self.decoder_layers {
            x = layer.forward(&x)?;
        }

        // Project back to input space
        x = x.dot(&self.output_proj); // [seq_len, input_dim]

        // Flatten to 1D
        let mut result = Vec::new();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                result.push(x[[i, j]]);
            }
        }

        Ok(Array1::from_vec(result))
    }

    fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    fn vocab_size(&self) -> usize {
        0 // Continuous tokenizer, no discrete vocabulary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_config_validation() {
        let config = TransformerConfig::default();
        assert!(config.validate().is_ok());

        let mut bad_config = config.clone();
        bad_config.embed_dim = 0;
        assert!(bad_config.validate().is_err());

        let mut bad_config = config.clone();
        bad_config.embed_dim = 100; // Not divisible by num_heads (8)
        assert!(bad_config.validate().is_err());

        let mut bad_config = config.clone();
        bad_config.dropout = 1.5;
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_multihead_attention_creation() {
        let mha = MultiHeadAttention::new(256, 8);
        assert!(mha.is_ok());

        let bad_mha = MultiHeadAttention::new(256, 7); // Not divisible
        assert!(bad_mha.is_err());
    }

    #[test]
    fn test_multihead_attention_forward() {
        let mha = MultiHeadAttention::new(64, 4).unwrap();
        let x = Array2::ones((10, 64)); // [seq_len=10, embed_dim=64]
        let out = mha.forward(&x);
        assert!(out.is_ok());
        let out = out.unwrap();
        assert_eq!(out.shape(), &[10, 64]);
    }

    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::new(100, 64);
        let x = Array2::zeros((50, 64));
        let out = pe.forward(&x);
        assert!(out.is_ok());
        let out = out.unwrap();
        assert_eq!(out.shape(), &[50, 64]);
    }

    #[test]
    fn test_positional_encoding_seq_too_long() {
        let pe = PositionalEncoding::new(10, 64);
        let x = Array2::zeros((20, 64)); // Longer than max_seq_len
        let out = pe.forward(&x);
        assert!(out.is_err());
    }

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(64, 1e-5);
        let x = Array2::from_shape_fn((10, 64), |(i, j)| (i + j) as f32);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[10, 64]);

        // Check that mean is approximately 0 and variance is approximately 1
        for row in out.rows() {
            let mean = row.mean().unwrap();
            let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 64.0;
            assert!((mean.abs()) < 1e-5);
            assert!((var - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_feedforward() {
        let ffn = FeedForward::new(64, 256);
        let x = Array2::ones((10, 64));
        let out = ffn.forward(&x);
        assert_eq!(out.shape(), &[10, 64]);
    }

    #[test]
    fn test_encoder_layer() {
        let layer = TransformerEncoderLayer::new(64, 4, 256).unwrap();
        let x = Array2::ones((10, 64));
        let out = layer.forward(&x);
        assert!(out.is_ok());
        let out = out.unwrap();
        assert_eq!(out.shape(), &[10, 64]);
    }

    #[test]
    fn test_transformer_tokenizer_creation() {
        let config = TransformerConfig {
            input_dim: 32,
            embed_dim: 64,
            num_heads: 4,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            feedforward_dim: 128,
            dropout: 0.1,
            max_seq_len: 100,
        };
        let tokenizer = TransformerTokenizer::new(config);
        assert!(tokenizer.is_ok());
    }

    #[test]
    fn test_transformer_encode_decode() {
        let config = TransformerConfig {
            input_dim: 16,
            embed_dim: 32,
            num_heads: 4,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            feedforward_dim: 64,
            dropout: 0.0,
            max_seq_len: 10,
        };
        let tokenizer = TransformerTokenizer::new(config).unwrap();

        let signal = Array1::linspace(0.0, 1.0, 64);
        let encoded = tokenizer.encode(&signal);
        assert!(encoded.is_ok());

        let encoded = encoded.unwrap();
        let decoded = tokenizer.decode(&encoded);
        assert!(decoded.is_ok());
        let decoded = decoded.unwrap();

        // Should preserve length (with padding)
        assert!(decoded.len() >= signal.len());
    }

    #[test]
    fn test_transformer_signal_too_long() {
        let config = TransformerConfig {
            input_dim: 16,
            embed_dim: 32,
            num_heads: 4,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            feedforward_dim: 64,
            dropout: 0.0,
            max_seq_len: 2, // Very small
        };
        let tokenizer = TransformerTokenizer::new(config).unwrap();

        let signal = Array1::linspace(0.0, 1.0, 1000); // Too long
        let encoded = tokenizer.encode(&signal);
        assert!(encoded.is_err());
    }

    #[test]
    fn test_softmax() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
        let result = MultiHeadAttention::softmax(&x).unwrap();

        // Check that each row sums to 1
        for row in result.rows() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }

        // Check that all values are positive
        for &val in result.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_gelu_activation() {
        // GELU(0) should be approximately 0
        assert!((FeedForward::gelu(0.0)).abs() < 1e-5);

        // GELU should be monotonic for positive values
        assert!(FeedForward::gelu(1.0) > FeedForward::gelu(0.5));
        assert!(FeedForward::gelu(2.0) > FeedForward::gelu(1.0));

        // GELU should preserve sign but be smooth
        assert!(FeedForward::gelu(-1.0) < 0.0);
        assert!(FeedForward::gelu(1.0) > 0.0);
    }
}
