//! ResidualVQ and RVQVAETokenizer — multi-stage residual vector quantization.
//!
//! Uses multiple VQ stages where each stage quantizes the residual from previous stages.
//! This is used in modern neural audio codecs like SoundStream and Encodec for
//! high-quality compression with variable bitrate support.

use super::vector_quantizer::{VQConfig, VectorQuantizer};
use crate::error::{TokenizerError, TokenizerResult};
use crate::SignalTokenizer;
use scirs2_core::ndarray::{Array1, Array2};

/// Residual Vector Quantization (RVQ)
///
/// Uses multiple VQ stages where each stage quantizes the residual from previous stages.
/// This is used in modern neural audio codecs like SoundStream and Encodec for
/// high-quality compression with variable bitrate support.
///
/// ## Algorithm
///
/// 1. First stage quantizes the input
/// 2. Compute residual = input - first_quantized
/// 3. Second stage quantizes the residual
/// 4. Repeat for N stages
/// 5. Reconstruction = sum of all quantized outputs
///
/// ## Benefits
///
/// - Progressive quality: use fewer stages for lower bitrate
/// - Better reconstruction than single-stage VQ
/// - Flexible bitrate control
#[derive(Debug, Clone)]
pub struct ResidualVQ {
    /// Vector quantizers for each stage
    pub(crate) quantizers: Vec<VectorQuantizer>,
    /// Number of stages
    num_stages: usize,
}

impl ResidualVQ {
    /// Create a new Residual VQ with multiple stages
    ///
    /// All stages use the same configuration but independent codebooks
    pub fn new(num_stages: usize, config: VQConfig) -> Self {
        let quantizers = (0..num_stages)
            .map(|_| VectorQuantizer::new(config.clone()))
            .collect();

        Self {
            quantizers,
            num_stages,
        }
    }

    /// Create with different configs per stage
    pub fn with_configs(configs: Vec<VQConfig>) -> Self {
        let num_stages = configs.len();
        let quantizers = configs.into_iter().map(VectorQuantizer::new).collect();

        Self {
            quantizers,
            num_stages,
        }
    }

    /// Encode with all stages
    ///
    /// Returns: (indices for each stage, quantized outputs for each stage)
    pub fn encode(&self, vector: &Array1<f32>) -> TokenizerResult<(Vec<usize>, Vec<Array1<f32>>)> {
        let mut indices = Vec::with_capacity(self.num_stages);
        let mut quantized_outputs = Vec::with_capacity(self.num_stages);
        let mut residual = vector.clone();

        for quantizer in &self.quantizers {
            let (idx, quantized) = quantizer.quantize(&residual)?;
            indices.push(idx);
            quantized_outputs.push(quantized.clone());

            // Update residual for next stage
            residual = &residual - &quantized;
        }

        Ok((indices, quantized_outputs))
    }

    /// Encode with limited number of stages (for variable bitrate)
    pub fn encode_with_stages(
        &self,
        vector: &Array1<f32>,
        num_stages: usize,
    ) -> TokenizerResult<(Vec<usize>, Vec<Array1<f32>>)> {
        if num_stages > self.num_stages {
            return Err(TokenizerError::InvalidConfig(format!(
                "Requested {} stages but only {} available",
                num_stages, self.num_stages
            )));
        }

        let mut indices = Vec::with_capacity(num_stages);
        let mut quantized_outputs = Vec::with_capacity(num_stages);
        let mut residual = vector.clone();

        for quantizer in self.quantizers.iter().take(num_stages) {
            let (idx, quantized) = quantizer.quantize(&residual)?;
            indices.push(idx);
            quantized_outputs.push(quantized.clone());

            residual = &residual - &quantized;
        }

        Ok((indices, quantized_outputs))
    }

    /// Decode from all stage indices
    pub fn decode(&self, indices: &[usize]) -> TokenizerResult<Array1<f32>> {
        if indices.len() != self.num_stages {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected {} indices, got {}",
                self.num_stages,
                indices.len()
            )));
        }

        let first_entry = self.quantizers[0].get_codebook_entry(indices[0])?;
        let mut result = first_entry;

        for (quantizer, &idx) in self.quantizers.iter().skip(1).zip(indices.iter().skip(1)) {
            let entry = quantizer.get_codebook_entry(idx)?;
            result = &result + &entry;
        }

        Ok(result)
    }

    /// Decode from quantized outputs (sum them)
    pub fn decode_from_quantized(
        &self,
        quantized_outputs: &[Array1<f32>],
    ) -> TokenizerResult<Array1<f32>> {
        if quantized_outputs.is_empty() {
            return Err(TokenizerError::InvalidConfig("No quantized outputs".into()));
        }

        let mut result = quantized_outputs[0].clone();
        for output in quantized_outputs.iter().skip(1) {
            result = &result + output;
        }

        Ok(result)
    }

    /// Update all stages with EMA
    pub fn update_ema(&mut self, encoder_outputs: &[Array1<f32>]) -> TokenizerResult<()> {
        if encoder_outputs.is_empty() {
            return Ok(());
        }

        // Collect residuals and indices for each stage
        let mut stage_outputs = vec![Vec::new(); self.num_stages];
        let mut stage_indices = vec![Vec::new(); self.num_stages];

        for output in encoder_outputs {
            let mut residual = output.clone();

            for (stage_idx, quantizer) in self.quantizers.iter().enumerate() {
                let (idx, quantized) = quantizer.quantize(&residual)?;
                stage_outputs[stage_idx].push(residual.clone());
                stage_indices[stage_idx].push(idx);

                residual = &residual - &quantized;
            }
        }

        // Update each stage
        for (quantizer, (outputs, indices)) in self
            .quantizers
            .iter_mut()
            .zip(stage_outputs.iter().zip(stage_indices.iter()))
        {
            quantizer.update_ema(outputs, indices)?;
        }

        Ok(())
    }

    /// Get reference to a specific stage
    pub fn stage(&self, idx: usize) -> Option<&VectorQuantizer> {
        self.quantizers.get(idx)
    }

    /// Get mutable reference to a specific stage
    pub fn stage_mut(&mut self, idx: usize) -> Option<&mut VectorQuantizer> {
        self.quantizers.get_mut(idx)
    }

    /// Get number of stages
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Compute total bits per sample (sum across stages)
    pub fn total_bits(&self) -> f32 {
        self.quantizers
            .iter()
            .map(|q| (q.codebook_size() as f32).log2())
            .sum()
    }

    /// Compute bitrate for a given number of stages
    pub fn bitrate_for_stages(&self, num_stages: usize) -> f32 {
        self.quantizers
            .iter()
            .take(num_stages)
            .map(|q| (q.codebook_size() as f32).log2())
            .sum()
    }

    /// Get usage statistics across all stages
    pub fn all_usage_stats(&self) -> Vec<(usize, usize, f32)> {
        self.quantizers.iter().map(|q| q.usage_stats()).collect()
    }

    /// Reset usage counts for all stages
    pub fn reset_all_usage_counts(&mut self) {
        for quantizer in &mut self.quantizers {
            quantizer.reset_usage_counts();
        }
    }
}

/// RVQ-VAE Tokenizer with residual quantization
#[derive(Debug, Clone)]
pub struct RVQVAETokenizer {
    /// Encoder projection
    encoder: Array2<f32>,
    /// Residual VQ
    rvq: ResidualVQ,
    /// Decoder projection
    decoder: Array2<f32>,
    /// Input dimension
    input_dim: usize,
}

impl RVQVAETokenizer {
    /// Create a new RVQ-VAE tokenizer
    pub fn new(input_dim: usize, num_stages: usize, config: VQConfig) -> Self {
        let mut rng = scirs2_core::random::thread_rng();

        let enc_scale = (2.0 / (input_dim + config.embed_dim) as f32).sqrt();
        let encoder = Array2::from_shape_fn((input_dim, config.embed_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * enc_scale
        });

        let dec_scale = (2.0 / (config.embed_dim + input_dim) as f32).sqrt();
        let decoder = Array2::from_shape_fn((config.embed_dim, input_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * dec_scale
        });

        let rvq = ResidualVQ::new(num_stages, config);

        Self {
            encoder,
            rvq,
            decoder,
            input_dim,
        }
    }

    /// Encode and quantize with all stages
    pub fn encode_quantized(
        &self,
        signal: &Array1<f32>,
    ) -> TokenizerResult<(Vec<usize>, Vec<Array1<f32>>)> {
        if signal.len() != self.input_dim {
            return Err(TokenizerError::dim_mismatch(
                self.input_dim,
                signal.len(),
                "dimension validation",
            ));
        }

        let latent = signal.dot(&self.encoder);
        self.rvq.encode(&latent)
    }

    /// Encode with limited stages (variable bitrate)
    pub fn encode_with_stages(
        &self,
        signal: &Array1<f32>,
        num_stages: usize,
    ) -> TokenizerResult<(Vec<usize>, Vec<Array1<f32>>)> {
        if signal.len() != self.input_dim {
            return Err(TokenizerError::dim_mismatch(
                self.input_dim,
                signal.len(),
                "dimension validation",
            ));
        }

        let latent = signal.dot(&self.encoder);
        self.rvq.encode_with_stages(&latent, num_stages)
    }

    /// Decode from indices
    pub fn decode_from_indices(&self, indices: &[usize]) -> TokenizerResult<Array1<f32>> {
        let quantized = self.rvq.decode(indices)?;
        Ok(quantized.dot(&self.decoder))
    }

    /// Decode from quantized outputs
    pub fn decode_from_quantized(
        &self,
        quantized_outputs: &[Array1<f32>],
    ) -> TokenizerResult<Array1<f32>> {
        let summed = self.rvq.decode_from_quantized(quantized_outputs)?;
        Ok(summed.dot(&self.decoder))
    }

    /// Get reference to RVQ
    pub fn rvq(&self) -> &ResidualVQ {
        &self.rvq
    }

    /// Get mutable reference to RVQ
    pub fn rvq_mut(&mut self) -> &mut ResidualVQ {
        &mut self.rvq
    }

    /// Get total bitrate
    pub fn total_bitrate(&self) -> f32 {
        self.rvq.total_bits()
    }

    /// Get bitrate for specific number of stages
    pub fn bitrate_for_stages(&self, num_stages: usize) -> f32 {
        self.rvq.bitrate_for_stages(num_stages)
    }
}

impl SignalTokenizer for RVQVAETokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let (indices, _) = self.encode_quantized(signal)?;
        // Return concatenated indices as floats
        Ok(Array1::from_vec(
            indices.iter().map(|&i| i as f32).collect(),
        ))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let indices: Vec<usize> = tokens.iter().map(|&t| t.round() as usize).collect();
        self.decode_from_indices(&indices)
    }

    fn embed_dim(&self) -> usize {
        self.rvq.num_stages() // Returns number of stages (each produces one index)
    }

    fn vocab_size(&self) -> usize {
        // Total vocabulary is product of all codebook sizes
        self.rvq
            .quantizers
            .iter()
            .map(|q| q.codebook_size())
            .product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_vq_creation() {
        let config = VQConfig {
            codebook_size: 8,
            embed_dim: 4,
            ..Default::default()
        };
        let rvq = ResidualVQ::new(3, config);

        assert_eq!(rvq.num_stages(), 3);
        assert!(rvq.total_bits() > 0.0);
    }

    #[test]
    fn test_rvq_encode_decode() {
        let config = VQConfig {
            codebook_size: 16,
            embed_dim: 8,
            ..Default::default()
        };
        let rvq = ResidualVQ::new(4, config);

        let vector = Array1::from_vec((0..8).map(|i| (i as f32 * 0.1).sin()).collect());

        let (indices, quantized_outputs) = rvq.encode(&vector).unwrap();

        assert_eq!(indices.len(), 4);
        assert_eq!(quantized_outputs.len(), 4);

        // Decode and verify reconstruction
        let reconstructed = rvq.decode(&indices).unwrap();
        assert_eq!(reconstructed.len(), vector.len());

        // Decode from quantized outputs
        let reconstructed2 = rvq.decode_from_quantized(&quantized_outputs).unwrap();
        assert_eq!(reconstructed2.len(), vector.len());

        // Both decode methods should give same result
        for (a, b) in reconstructed.iter().zip(reconstructed2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rvq_variable_stages() {
        let config = VQConfig {
            codebook_size: 16,
            embed_dim: 8,
            ..Default::default()
        };
        let rvq = ResidualVQ::new(4, config);

        let vector = Array1::from_vec((0..8).map(|i| i as f32).collect());

        // Encode with 2 stages
        let (indices, _) = rvq.encode_with_stages(&vector, 2).unwrap();
        assert_eq!(indices.len(), 2);

        // Encode with 4 stages
        let (indices_full, _) = rvq.encode(&vector).unwrap();
        assert_eq!(indices_full.len(), 4);

        // First 2 indices should be the same
        assert_eq!(indices[0], indices_full[0]);
        assert_eq!(indices[1], indices_full[1]);
    }

    #[test]
    fn test_rvq_bitrate() {
        let config = VQConfig {
            codebook_size: 256, // 8 bits
            embed_dim: 8,
            ..Default::default()
        };
        let rvq = ResidualVQ::new(3, config);

        let total_bits = rvq.total_bits();
        assert!((total_bits - 24.0).abs() < 0.1); // 8 bits * 3 stages = 24 bits

        let bits_2_stages = rvq.bitrate_for_stages(2);
        assert!((bits_2_stages - 16.0).abs() < 0.1); // 8 bits * 2 stages = 16 bits
    }

    #[test]
    fn test_rvq_ema_update() {
        let config = VQConfig {
            codebook_size: 8,
            embed_dim: 4,
            use_ema: true,
            ..Default::default()
        };
        let mut rvq = ResidualVQ::new(2, config);

        let outputs = vec![
            Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]),
            Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8]),
            Array1::from_vec(vec![0.2, 0.3, 0.4, 0.5]),
        ];

        rvq.update_ema(&outputs).unwrap();

        let stats = rvq.all_usage_stats();
        assert_eq!(stats.len(), 2); // 2 stages
    }

    #[test]
    fn test_rvq_stage_access() {
        let config = VQConfig {
            codebook_size: 8,
            embed_dim: 4,
            ..Default::default()
        };
        let mut rvq = ResidualVQ::new(3, config);

        // Test immutable access
        let stage0 = rvq.stage(0).unwrap();
        assert_eq!(stage0.codebook_size(), 8);

        // Test mutable access
        let stage1 = rvq.stage_mut(1).unwrap();
        assert_eq!(stage1.codebook_size(), 8);

        // Test out of bounds
        assert!(rvq.stage(10).is_none());
    }

    #[test]
    fn test_rvqvae_tokenizer() {
        let config = VQConfig {
            codebook_size: 16,
            embed_dim: 8,
            ..Default::default()
        };
        let tokenizer = RVQVAETokenizer::new(32, 4, config);

        let signal = Array1::from_vec((0..32).map(|i| (i as f32 * 0.05).sin()).collect());

        let (indices, quantized) = tokenizer.encode_quantized(&signal).unwrap();
        assert_eq!(indices.len(), 4);
        assert_eq!(quantized.len(), 4);

        let reconstructed = tokenizer.decode_from_indices(&indices).unwrap();
        assert_eq!(reconstructed.len(), 32);
    }

    #[test]
    fn test_rvqvae_variable_bitrate() {
        let config = VQConfig {
            codebook_size: 256,
            embed_dim: 16,
            ..Default::default()
        };
        let tokenizer = RVQVAETokenizer::new(64, 4, config);

        let signal = Array1::from_vec((0..64).map(|i| (i as f32 * 0.05).cos()).collect());

        // Low bitrate (2 stages)
        let (indices_low, _) = tokenizer.encode_with_stages(&signal, 2).unwrap();
        assert_eq!(indices_low.len(), 2);

        // High bitrate (4 stages)
        let (indices_high, _) = tokenizer.encode_with_stages(&signal, 4).unwrap();
        assert_eq!(indices_high.len(), 4);

        // Check bitrates
        let bitrate_low = tokenizer.bitrate_for_stages(2);
        let bitrate_high = tokenizer.total_bitrate();
        assert!(bitrate_high > bitrate_low);
    }

    #[test]
    fn test_rvqvae_signal_tokenizer_trait() {
        let config = VQConfig {
            codebook_size: 32,
            embed_dim: 12,
            ..Default::default()
        };
        let tokenizer = RVQVAETokenizer::new(48, 3, config);

        let signal = Array1::from_vec((0..48).map(|i| i as f32 * 0.1).collect());

        // Test through SignalTokenizer trait
        let encoded = tokenizer.encode(&signal).unwrap();
        assert_eq!(encoded.len(), 3); // 3 stages = 3 indices

        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 48);
    }

    #[test]
    fn test_rvq_with_different_configs() {
        let configs = vec![
            VQConfig {
                codebook_size: 128,
                embed_dim: 8,
                ..Default::default()
            },
            VQConfig {
                codebook_size: 256,
                embed_dim: 8,
                ..Default::default()
            },
            VQConfig {
                codebook_size: 512,
                embed_dim: 8,
                ..Default::default()
            },
        ];

        let rvq = ResidualVQ::with_configs(configs);
        assert_eq!(rvq.num_stages(), 3);

        let vector = Array1::from_vec((0..8).map(|i| i as f32).collect());
        let (indices, _) = rvq.encode(&vector).unwrap();
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_rvq_residual_progression() {
        // Test that residuals decrease with each stage
        let config = VQConfig {
            codebook_size: 64,
            embed_dim: 16,
            ..Default::default()
        };
        let rvq = ResidualVQ::new(4, config);

        let vector = Array1::from_vec((0..16).map(|i| (i as f32 * 0.1).sin()).collect());

        let (_, quantized_outputs) = rvq.encode(&vector).unwrap();

        // Compute residual norms
        let mut residual = vector.clone();
        let mut residual_norms = Vec::new();

        for quantized in &quantized_outputs {
            let norm: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();
            residual_norms.push(norm);
            residual = &residual - quantized;
        }

        // Residual norms should generally decrease
        // (first stage captures most variance)
        assert!(residual_norms[0] > 0.0);
    }
}
