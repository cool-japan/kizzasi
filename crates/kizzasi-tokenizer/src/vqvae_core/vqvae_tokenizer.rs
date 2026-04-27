//! VQVAETokenizer and SignalTokenizer implementation.
//!
//! Contains [`VQVAETokenizer`] with encoder/decoder projections
//! wrapping a [`VectorQuantizer`].

use super::vector_quantizer::{VQConfig, VectorQuantizer};
use crate::error::{TokenizerError, TokenizerResult};
use crate::SignalTokenizer;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;

/// VQ-VAE with encoder and decoder projections
#[derive(Debug, Clone)]
pub struct VQVAETokenizer {
    /// Encoder projection (input_dim -> embed_dim)
    encoder: Array2<f32>,
    /// Vector quantizer
    quantizer: VectorQuantizer,
    /// Decoder projection (embed_dim -> input_dim)
    decoder: Array2<f32>,
    /// Input dimension
    input_dim: usize,
}

impl VQVAETokenizer {
    /// Create a new VQ-VAE tokenizer
    pub fn new(input_dim: usize, config: VQConfig) -> Self {
        let mut rng = thread_rng();

        // Xavier initialization for encoder
        let enc_scale = (2.0 / (input_dim + config.embed_dim) as f32).sqrt();
        let encoder = Array2::from_shape_fn((input_dim, config.embed_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * enc_scale
        });

        // Xavier initialization for decoder
        let dec_scale = (2.0 / (config.embed_dim + input_dim) as f32).sqrt();
        let decoder = Array2::from_shape_fn((config.embed_dim, input_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * dec_scale
        });

        let quantizer = VectorQuantizer::new(config);

        Self {
            encoder,
            quantizer,
            decoder,
            input_dim,
        }
    }

    /// Encode and quantize
    pub fn encode_quantized(&self, signal: &Array1<f32>) -> TokenizerResult<(usize, Array1<f32>)> {
        if signal.len() != self.input_dim {
            return Err(TokenizerError::dim_mismatch(
                self.input_dim,
                signal.len(),
                "dimension validation",
            ));
        }

        // Encode to latent space
        let latent = signal.dot(&self.encoder);

        // Quantize
        self.quantizer.quantize(&latent)
    }

    /// Decode from quantized vector
    pub fn decode_quantized(&self, quantized: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if quantized.len() != self.quantizer.embed_dim() {
            return Err(TokenizerError::dim_mismatch(
                self.quantizer.embed_dim(),
                quantized.len(),
                "dimension validation",
            ));
        }

        Ok(quantized.dot(&self.decoder))
    }

    /// Decode from index
    pub fn decode_from_index(&self, idx: usize) -> TokenizerResult<Array1<f32>> {
        let quantized = self.quantizer.get_codebook_entry(idx)?;
        self.decode_quantized(&quantized)
    }

    /// Get reference to quantizer
    pub fn quantizer(&self) -> &VectorQuantizer {
        &self.quantizer
    }

    /// Get mutable reference to quantizer (for training)
    pub fn quantizer_mut(&mut self) -> &mut VectorQuantizer {
        &mut self.quantizer
    }

    /// Get encoder weights
    pub fn encoder(&self) -> &Array2<f32> {
        &self.encoder
    }

    /// Get decoder weights
    pub fn decoder(&self) -> &Array2<f32> {
        &self.decoder
    }

    /// Set encoder weights
    pub fn set_encoder(&mut self, weights: Array2<f32>) -> TokenizerResult<()> {
        if weights.shape() != [self.input_dim, self.quantizer.embed_dim()] {
            return Err(TokenizerError::dim_mismatch(
                self.input_dim * self.quantizer.embed_dim(),
                weights.len(),
                "dimension validation",
            ));
        }
        self.encoder = weights;
        Ok(())
    }

    /// Set decoder weights
    pub fn set_decoder(&mut self, weights: Array2<f32>) -> TokenizerResult<()> {
        if weights.shape() != [self.quantizer.embed_dim(), self.input_dim] {
            return Err(TokenizerError::dim_mismatch(
                self.quantizer.embed_dim() * self.input_dim,
                weights.len(),
                "dimension validation",
            ));
        }
        self.decoder = weights;
        Ok(())
    }
}

impl SignalTokenizer for VQVAETokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let (idx, _) = self.encode_quantized(signal)?;
        // Return index as float for embedding lookup
        Ok(Array1::from_elem(1, idx as f32))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if tokens.len() != 1 {
            return Err(TokenizerError::dim_mismatch(
                1,
                tokens.len(),
                "dimension validation",
            ));
        }
        let idx = tokens[0].round() as usize;
        self.decode_from_index(idx)
    }

    fn embed_dim(&self) -> usize {
        self.quantizer.embed_dim()
    }

    fn vocab_size(&self) -> usize {
        self.quantizer.codebook_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vqvae_tokenizer() {
        let config = VQConfig {
            codebook_size: 16,
            embed_dim: 8,
            ..Default::default()
        };
        let tokenizer = VQVAETokenizer::new(32, config);

        let signal = Array1::from_vec((0..32).map(|i| (i as f32 * 0.1).sin()).collect());

        let encoded = tokenizer.encode(&signal).unwrap();
        assert_eq!(encoded.len(), 1); // Returns single index

        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 32);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let config = VQConfig {
            codebook_size: 32,
            embed_dim: 16,
            ..Default::default()
        };
        let tokenizer = VQVAETokenizer::new(64, config);

        let signal = Array1::from_vec((0..64).map(|i| (i as f32 * 0.05).cos()).collect());

        let (idx, _quantized) = tokenizer.encode_quantized(&signal).unwrap();
        let decoded = tokenizer.decode_from_index(idx).unwrap();

        assert_eq!(decoded.len(), signal.len());
    }
}
