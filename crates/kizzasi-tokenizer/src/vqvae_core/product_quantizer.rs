//! Product Quantizer — sub-space factored vector quantization.
//!
//! Contains [`ProductQuantizerConfig`], [`BatchQuantizeResult`], and [`ProductQuantizer`].

use super::vector_quantizer::{VQConfig, VectorQuantizer};
use crate::error::{TokenizerError, TokenizerResult};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Product Quantization (PQ) Configuration
///
/// Product Quantization splits the embedding space into M subspaces and
/// quantizes each independently, enabling very large effective codebook sizes
/// with linear memory/compute scaling.
///
/// ## Example
///
/// With 4 subspaces and 256 codes per subspace:
/// - Memory: 4 * 256 * (embed_dim/4) parameters
/// - Effective codebook size: 256^4 = 4.3 billion codes
/// - vs. Standard VQ: needs 4.3B * embed_dim parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizerConfig {
    /// Number of subspaces (M)
    pub num_subspaces: usize,
    /// Codebook size per subspace (K)
    pub codebook_size_per_subspace: usize,
    /// Total embedding dimension (must be divisible by num_subspaces)
    pub embed_dim: usize,
    /// Beta parameter for commitment loss
    pub commitment_beta: f32,
    /// EMA decay rate
    pub ema_decay: f32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Use EMA updates
    pub use_ema: bool,
}

impl Default for ProductQuantizerConfig {
    fn default() -> Self {
        Self {
            num_subspaces: 4,
            codebook_size_per_subspace: 256,
            embed_dim: 64,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        }
    }
}

/// Type alias for batch quantization results
pub type BatchQuantizeResult = (Vec<Vec<usize>>, Vec<Array1<f32>>);

/// Product Quantizer
///
/// Implements Product Quantization (Jegou et al., 2011) for efficient
/// high-dimensional vector quantization with exponentially large effective
/// codebook sizes.
///
/// # Algorithm
///
/// 1. Split D-dimensional vector into M subspaces of D/M dimensions
/// 2. Quantize each subspace independently using its own codebook
/// 3. Concatenate quantized subspaces
/// 4. Effective codebook size: K^M (K = codes per subspace)
///
/// # Memory Complexity
///
/// - Standard VQ: O(K * D)
/// - Product VQ: O(M * K * D/M) = O(K * D)
/// - But effective codes: K^M vs K (exponential gain!)
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    /// Configuration
    config: ProductQuantizerConfig,
    /// Subspace quantizers (one per subspace)
    subspace_quantizers: Vec<VectorQuantizer>,
    /// Dimension per subspace
    subspace_dim: usize,
}

impl ProductQuantizer {
    /// Create a new product quantizer
    pub fn new(config: ProductQuantizerConfig) -> TokenizerResult<Self> {
        if !config.embed_dim.is_multiple_of(config.num_subspaces) {
            return Err(TokenizerError::InvalidConfig(format!(
                "embed_dim ({}) must be divisible by num_subspaces ({})",
                config.embed_dim, config.num_subspaces
            )));
        }

        let subspace_dim = config.embed_dim / config.num_subspaces;

        // Create a quantizer for each subspace
        let mut subspace_quantizers = Vec::with_capacity(config.num_subspaces);
        for _ in 0..config.num_subspaces {
            let subspace_config = VQConfig {
                codebook_size: config.codebook_size_per_subspace,
                embed_dim: subspace_dim,
                commitment_beta: config.commitment_beta,
                ema_decay: config.ema_decay,
                epsilon: config.epsilon,
                use_ema: config.use_ema,
            };
            subspace_quantizers.push(VectorQuantizer::new(subspace_config));
        }

        Ok(Self {
            config,
            subspace_quantizers,
            subspace_dim,
        })
    }

    /// Split a vector into subspaces
    pub fn split_into_subspaces(&self, vector: &Array1<f32>) -> TokenizerResult<Vec<Array1<f32>>> {
        if vector.len() != self.config.embed_dim {
            return Err(TokenizerError::dim_mismatch(
                self.config.embed_dim,
                vector.len(),
                "dimension validation",
            ));
        }

        let mut subspaces = Vec::with_capacity(self.config.num_subspaces);
        for i in 0..self.config.num_subspaces {
            let start = i * self.subspace_dim;
            let end = start + self.subspace_dim;
            let subspace =
                Array1::from_vec(vector.slice(scirs2_core::ndarray::s![start..end]).to_vec());
            subspaces.push(subspace);
        }

        Ok(subspaces)
    }

    /// Concatenate subspace vectors
    pub fn concatenate_subspaces(&self, subspaces: &[Array1<f32>]) -> TokenizerResult<Array1<f32>> {
        if subspaces.len() != self.config.num_subspaces {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected {} subspaces, got {}",
                self.config.num_subspaces,
                subspaces.len()
            )));
        }

        let mut result = Vec::with_capacity(self.config.embed_dim);
        for subspace in subspaces {
            let slice = subspace.as_slice().ok_or_else(|| {
                TokenizerError::InvalidConfig(
                    "Subspace array does not have contiguous layout".into(),
                )
            })?;
            result.extend_from_slice(slice);
        }

        Ok(Array1::from_vec(result))
    }

    /// Quantize a vector using product quantization
    pub fn quantize(&self, vector: &Array1<f32>) -> TokenizerResult<(Vec<usize>, Array1<f32>)> {
        let subspaces = self.split_into_subspaces(vector)?;
        let mut indices = Vec::with_capacity(self.config.num_subspaces);
        let mut quantized_subspaces = Vec::with_capacity(self.config.num_subspaces);

        for (subspace, quantizer) in subspaces.iter().zip(&self.subspace_quantizers) {
            let (idx, quantized) = quantizer.quantize(subspace)?;
            indices.push(idx);
            quantized_subspaces.push(quantized);
        }

        let quantized_vector = self.concatenate_subspaces(&quantized_subspaces)?;
        Ok((indices, quantized_vector))
    }

    /// Quantize multiple vectors
    pub fn quantize_batch(&self, vectors: &[Array1<f32>]) -> TokenizerResult<BatchQuantizeResult> {
        let mut all_indices = Vec::with_capacity(vectors.len());
        let mut all_quantized = Vec::with_capacity(vectors.len());

        for vector in vectors {
            let (indices, quantized) = self.quantize(vector)?;
            all_indices.push(indices);
            all_quantized.push(quantized);
        }

        Ok((all_indices, all_quantized))
    }

    /// Decode from subspace indices
    pub fn decode(&self, indices: &[usize]) -> TokenizerResult<Array1<f32>> {
        if indices.len() != self.config.num_subspaces {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected {} indices, got {}",
                self.config.num_subspaces,
                indices.len()
            )));
        }

        let mut subspaces = Vec::with_capacity(self.config.num_subspaces);
        for (idx, quantizer) in indices.iter().zip(&self.subspace_quantizers) {
            let entry = quantizer.get_codebook_entry(*idx)?;
            subspaces.push(entry);
        }

        self.concatenate_subspaces(&subspaces)
    }

    /// Initialize from data using k-means++ for each subspace
    pub fn initialize_from_data(&mut self, data: &[Array1<f32>]) -> TokenizerResult<()> {
        if data.is_empty() {
            return Err(TokenizerError::InvalidConfig(
                "Cannot initialize from empty data".into(),
            ));
        }

        // Split all data into subspaces
        let mut subspace_data: Vec<Vec<Array1<f32>>> =
            vec![Vec::with_capacity(data.len()); self.config.num_subspaces];

        for vector in data {
            let subspaces = self.split_into_subspaces(vector)?;
            for (i, subspace) in subspaces.into_iter().enumerate() {
                subspace_data[i].push(subspace);
            }
        }

        // Initialize each subspace quantizer
        for (quantizer, data) in self
            .subspace_quantizers
            .iter_mut()
            .zip(subspace_data.iter())
        {
            quantizer.initialize_from_data(data)?;
        }

        Ok(())
    }

    /// Update using EMA
    pub fn update_ema(
        &mut self,
        encoder_outputs: &[Array1<f32>],
        all_indices: &[Vec<usize>],
    ) -> TokenizerResult<()> {
        if encoder_outputs.len() != all_indices.len() {
            return Err(TokenizerError::InvalidConfig(
                "Mismatch between encoder_outputs and indices".into(),
            ));
        }

        // Split encoder outputs into subspaces
        let mut subspace_outputs: Vec<Vec<Array1<f32>>> =
            vec![Vec::with_capacity(encoder_outputs.len()); self.config.num_subspaces];
        let mut subspace_indices: Vec<Vec<usize>> =
            vec![Vec::with_capacity(encoder_outputs.len()); self.config.num_subspaces];

        for (output, indices) in encoder_outputs.iter().zip(all_indices.iter()) {
            if indices.len() != self.config.num_subspaces {
                return Err(TokenizerError::InvalidConfig(
                    "Invalid indices length".into(),
                ));
            }

            let subspaces = self.split_into_subspaces(output)?;
            for (i, (subspace, &idx)) in subspaces.into_iter().zip(indices.iter()).enumerate() {
                subspace_outputs[i].push(subspace);
                subspace_indices[i].push(idx);
            }
        }

        // Update each subspace quantizer
        for (quantizer, (outputs, indices)) in self
            .subspace_quantizers
            .iter_mut()
            .zip(subspace_outputs.iter().zip(subspace_indices.iter()))
        {
            quantizer.update_ema(outputs, indices)?;
        }

        Ok(())
    }

    /// Compute VQ losses
    pub fn compute_loss(
        &self,
        encoder_output: &Array1<f32>,
        quantized: &Array1<f32>,
    ) -> (f32, f32, f32) {
        // Codebook loss: ||sg[encoder_output] - quantized||^2
        let codebook_loss: f32 = encoder_output
            .iter()
            .zip(quantized.iter())
            .map(|(e, q)| (e - q).powi(2))
            .sum();

        // Commitment loss: ||encoder_output - sg[quantized]||^2
        let commitment_loss: f32 = encoder_output
            .iter()
            .zip(quantized.iter())
            .map(|(e, q)| (e - q).powi(2))
            .sum();

        let total_loss = codebook_loss + self.config.commitment_beta * commitment_loss;

        (total_loss, codebook_loss, commitment_loss)
    }

    /// Get effective codebook size (K^M)
    pub fn effective_codebook_size(&self) -> usize {
        self.config
            .codebook_size_per_subspace
            .pow(self.config.num_subspaces as u32)
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.config.num_subspaces * self.config.codebook_size_per_subspace * self.subspace_dim
    }

    /// Get embedding dimension
    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    /// Get number of subspaces
    pub fn num_subspaces(&self) -> usize {
        self.config.num_subspaces
    }

    /// Get codebook size per subspace
    pub fn codebook_size_per_subspace(&self) -> usize {
        self.config.codebook_size_per_subspace
    }

    /// Reset dead codes in all subspaces
    pub fn reset_unused_codes(
        &mut self,
        encoder_outputs: &[Array1<f32>],
        threshold: usize,
    ) -> TokenizerResult<usize> {
        if encoder_outputs.is_empty() {
            return Ok(0);
        }

        // Split encoder outputs into subspaces
        let mut subspace_outputs: Vec<Vec<Array1<f32>>> =
            vec![Vec::with_capacity(encoder_outputs.len()); self.config.num_subspaces];

        for output in encoder_outputs {
            let subspaces = self.split_into_subspaces(output)?;
            for (i, subspace) in subspaces.into_iter().enumerate() {
                subspace_outputs[i].push(subspace);
            }
        }

        // Reset unused codes in each subspace quantizer
        let mut total_reset = 0;
        for (quantizer, outputs) in self
            .subspace_quantizers
            .iter_mut()
            .zip(subspace_outputs.iter())
        {
            total_reset += quantizer.reset_unused_codes(outputs, threshold);
        }

        Ok(total_reset)
    }

    /// Get usage statistics for all subspaces
    pub fn usage_stats(&self) -> Vec<(usize, usize, f32)> {
        self.subspace_quantizers
            .iter()
            .map(|q| q.usage_stats())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_product_quantizer_creation() {
        let config = ProductQuantizerConfig {
            num_subspaces: 4,
            codebook_size_per_subspace: 16,
            embed_dim: 64,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config.clone()).unwrap();

        assert_eq!(pq.embed_dim(), 64);
        assert_eq!(pq.num_subspaces(), 4);
        assert_eq!(pq.codebook_size_per_subspace(), 16);
        assert_eq!(pq.effective_codebook_size(), 16_usize.pow(4)); // 65536
    }

    #[test]
    fn test_product_quantizer_invalid_config() {
        // embed_dim not divisible by num_subspaces
        let config = ProductQuantizerConfig {
            num_subspaces: 3,
            codebook_size_per_subspace: 16,
            embed_dim: 64, // 64 % 3 != 0
            ..Default::default()
        };

        assert!(ProductQuantizer::new(config).is_err());
    }

    #[test]
    fn test_product_quantize_decode() {
        let config = ProductQuantizerConfig {
            num_subspaces: 4,
            codebook_size_per_subspace: 8,
            embed_dim: 64,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config).unwrap();
        let vector = Array1::from_vec((0..64).map(|i| (i as f32) * 0.01).collect());

        let (indices, quantized) = pq.quantize(&vector).unwrap();

        assert_eq!(indices.len(), 4);
        assert_eq!(quantized.len(), 64);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 8);
        }

        // Decode should reconstruct the quantized vector
        let decoded = pq.decode(&indices).unwrap();
        assert_eq!(decoded.len(), 64);

        // Quantized and decoded should be identical
        for (q, d) in quantized.iter().zip(decoded.iter()) {
            assert!((q - d).abs() < 1e-6);
        }
    }

    #[test]
    fn test_product_quantizer_batch() {
        let config = ProductQuantizerConfig {
            num_subspaces: 2,
            codebook_size_per_subspace: 16,
            embed_dim: 32,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config).unwrap();

        let vectors = vec![
            Array1::from_vec((0..32).map(|i| i as f32 * 0.1).collect()),
            Array1::from_vec((0..32).map(|i| i as f32 * 0.2).collect()),
            Array1::from_vec((0..32).map(|i| i as f32 * 0.3).collect()),
        ];

        let (all_indices, all_quantized) = pq.quantize_batch(&vectors).unwrap();

        assert_eq!(all_indices.len(), 3);
        assert_eq!(all_quantized.len(), 3);

        for (i, (indices, quantized)) in all_indices.iter().zip(all_quantized.iter()).enumerate() {
            assert_eq!(indices.len(), 2);
            assert_eq!(quantized.len(), 32);

            // Verify decode
            let decoded = pq.decode(indices).unwrap();
            for (q, d) in quantized.iter().zip(decoded.iter()) {
                assert!((q - d).abs() < 1e-6, "Batch {}: quantized != decoded", i);
            }
        }
    }

    #[test]
    fn test_product_quantizer_split_concat() {
        let config = ProductQuantizerConfig {
            num_subspaces: 4,
            codebook_size_per_subspace: 8,
            embed_dim: 64,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config).unwrap();
        let vector = Array1::from_vec((0..64).map(|i| i as f32).collect());

        let subspaces = pq.split_into_subspaces(&vector).unwrap();

        assert_eq!(subspaces.len(), 4);
        for subspace in &subspaces {
            assert_eq!(subspace.len(), 16); // 64 / 4
        }

        // Concatenate back
        let reconstructed = pq.concatenate_subspaces(&subspaces).unwrap();
        assert_eq!(reconstructed.len(), 64);

        for (orig, recon) in vector.iter().zip(reconstructed.iter()) {
            assert_eq!(orig, recon);
        }
    }

    #[test]
    fn test_product_quantizer_ema_update() {
        let config = ProductQuantizerConfig {
            num_subspaces: 2,
            codebook_size_per_subspace: 8,
            embed_dim: 16,
            use_ema: true,
            ..Default::default()
        };

        let mut pq = ProductQuantizer::new(config).unwrap();

        let outputs = vec![
            Array1::from_vec((0..16).map(|i| i as f32 * 0.1).collect()),
            Array1::from_vec((0..16).map(|i| i as f32 * 0.2).collect()),
            Array1::from_vec((0..16).map(|i| i as f32 * 0.3).collect()),
        ];

        let (all_indices, _) = pq.quantize_batch(&outputs).unwrap();

        // Update should succeed
        pq.update_ema(&outputs, &all_indices).unwrap();

        // Check usage stats for each subspace
        let stats = pq.usage_stats();
        assert_eq!(stats.len(), 2); // 2 subspaces

        for (total, used, _util) in stats {
            assert_eq!(total, 3); // 3 vectors processed
            assert!(used > 0); // At least some codes used
            assert!(used <= 8); // At most all codes used
        }
    }

    #[test]
    fn test_product_quantizer_initialization() {
        let config = ProductQuantizerConfig {
            num_subspaces: 2,
            codebook_size_per_subspace: 4,
            embed_dim: 16,
            ..Default::default()
        };

        let mut pq = ProductQuantizer::new(config).unwrap();

        // Generate some sample data
        let data: Vec<Array1<f32>> = (0..20)
            .map(|i| Array1::from_vec((0..16).map(|j| ((i + j) as f32 * 0.1).sin()).collect()))
            .collect();

        // Initialize from data (k-means++)
        pq.initialize_from_data(&data).unwrap();

        // After initialization, quantization should work
        let (indices, _) = pq.quantize(&data[0]).unwrap();
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_product_quantizer_compute_loss() {
        let config = ProductQuantizerConfig {
            num_subspaces: 4,
            codebook_size_per_subspace: 8,
            embed_dim: 64,
            commitment_beta: 0.25,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config).unwrap();

        let encoder_output = Array1::from_vec((0..64).map(|i| i as f32 * 0.01).collect());
        let (_, quantized) = pq.quantize(&encoder_output).unwrap();

        let (total_loss, codebook_loss, commitment_loss) =
            pq.compute_loss(&encoder_output, &quantized);

        assert!(total_loss >= 0.0);
        assert!(codebook_loss >= 0.0);
        assert!(commitment_loss >= 0.0);

        // Total loss = codebook_loss + beta * commitment_loss
        let expected_total = codebook_loss + 0.25 * commitment_loss;
        assert!((total_loss - expected_total).abs() < 1e-6);
    }

    #[test]
    fn test_product_quantizer_effective_size() {
        let config = ProductQuantizerConfig {
            num_subspaces: 4,
            codebook_size_per_subspace: 256,
            embed_dim: 64,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config.clone()).unwrap();

        // Effective codebook size = 256^4
        assert_eq!(pq.effective_codebook_size(), 256_usize.pow(4));

        // Number of parameters = M * K * (D/M) = M * K * D/M = K * D
        let expected_params = config.num_subspaces
            * config.codebook_size_per_subspace
            * (config.embed_dim / config.num_subspaces);
        assert_eq!(pq.num_parameters(), expected_params);
    }

    #[test]
    fn test_product_quantizer_reset_unused_codes() {
        let config = ProductQuantizerConfig {
            num_subspaces: 2,
            codebook_size_per_subspace: 8,
            embed_dim: 16,
            ..Default::default()
        };

        let mut pq = ProductQuantizer::new(config).unwrap();

        // Generate some encoder outputs
        let outputs: Vec<Array1<f32>> = (0..5)
            .map(|i| Array1::from_vec((0..16).map(|j| (i + j) as f32 * 0.1).collect()))
            .collect();

        // Quantize to mark some codes as used
        let (all_indices, _) = pq.quantize_batch(&outputs).unwrap();

        // Update EMA to track usage
        pq.update_ema(&outputs, &all_indices).unwrap();

        // Reset codes that have been used less than 2 times
        let reset_count = pq.reset_unused_codes(&outputs, 2).unwrap();

        // Function should succeed and return a valid count
        assert!(reset_count <= pq.num_subspaces() * pq.codebook_size_per_subspace());
    }

    #[test]
    fn test_product_quantizer_memory_efficiency() {
        // Compare memory usage of PQ vs standard VQ

        // Standard VQ: 1M codes, 128-dim
        let standard_params = 1_000_000 * 128;

        // Product VQ: 4 subspaces, 100 codes each, 128-dim
        // Effective codes: 100^4 = 100M (100x more codes!)
        // Parameters: 4 * 100 * 32 = 12,800
        let pq_config = ProductQuantizerConfig {
            num_subspaces: 4,
            codebook_size_per_subspace: 100,
            embed_dim: 128,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(pq_config).unwrap();

        assert_eq!(pq.num_parameters(), 4 * 100 * 32);
        assert_eq!(pq.effective_codebook_size(), 100_usize.pow(4));

        // PQ uses ~10,000x fewer parameters for ~100x more codes
        let compression_ratio = standard_params as f32 / pq.num_parameters() as f32;
        assert!(compression_ratio > 9000.0);
    }
}
