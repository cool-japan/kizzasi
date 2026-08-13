//! VectorQuantizer — single-stage VQ with learned codebook and EMA updates.
//!
//! Contains [`VQConfig`] and [`VectorQuantizer`].

use crate::error::{TokenizerError, TokenizerResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::parallel_ops::parallel_map;
use scirs2_core::random::thread_rng;
use serde::{Deserialize, Serialize};

/// Vector quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VQConfig {
    /// Number of codebook entries
    pub codebook_size: usize,
    /// Dimension of each codebook entry
    pub embed_dim: usize,
    /// Beta parameter for commitment loss (typically 0.25)
    pub commitment_beta: f32,
    /// Decay rate for EMA updates (typically 0.99)
    pub ema_decay: f32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Whether to use EMA updates (vs gradient-based)
    pub use_ema: bool,
}

impl Default for VQConfig {
    fn default() -> Self {
        Self {
            codebook_size: 512,
            embed_dim: 64,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        }
    }
}

/// Squared Euclidean distance between two vectors.
///
/// Argmin/nearest-neighbour searches only need relative ordering, so the
/// `.sqrt()` a true Euclidean distance would need is dropped — squaring it
/// straight back afterward (as the D^2-proportional k-means++ sampling
/// below does) would otherwise compute a square root purely to discard it.
#[inline]
fn euclidean_distance_sq(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Vector Quantizer with learned codebook
#[derive(Debug, Clone)]
pub struct VectorQuantizer {
    /// Configuration
    config: VQConfig,
    /// Codebook embeddings: [codebook_size, embed_dim]
    pub(crate) codebook: Array2<f32>,
    /// EMA cluster sizes (for EMA updates)
    ema_cluster_size: Array1<f32>,
    /// EMA embedding sums (for EMA updates)
    ema_embed_sum: Array2<f32>,
    /// Number of times each code has been used
    pub(crate) usage_counts: Array1<usize>,
}

impl VectorQuantizer {
    /// Create a new vector quantizer with random initialization
    pub fn new(config: VQConfig) -> Self {
        let mut rng = thread_rng();

        // Initialize codebook with random values
        let scale = 1.0 / (config.embed_dim as f32).sqrt();
        let codebook = Array2::from_shape_fn((config.codebook_size, config.embed_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Seed the EMA accumulators so they represent "one pseudo-observation
        // already equal to the random init", not "zero observations of
        // zero". `update_ema` computes `codebook[i] = ema_embed_sum[i] /
        // (ema_cluster_size[i] + epsilon)`; starting both at zero means the
        // very first call divides ~0/~epsilon, driving every never-selected
        // code's row to ~0 regardless of its random init. Mirrors
        // `reset_unused_codes`'s seeding of the same pair of accumulators.
        let ema_cluster_size = Array1::ones(config.codebook_size);
        let ema_embed_sum = codebook.clone();
        let usage_counts = Array1::zeros(config.codebook_size);

        Self {
            config,
            codebook,
            ema_cluster_size,
            ema_embed_sum,
            usage_counts,
        }
    }

    /// Initialize codebook from data using k-means++
    ///
    /// # Errors
    /// Returns an error if `data` is empty, or if any point's length
    /// differs from `config.embed_dim` — a shorter or longer point would
    /// otherwise reach an unchecked `self.codebook[[i, j]]` write (panicking
    /// on a longer point) or silently leave part of a codebook row at its
    /// random-init value (for a shorter point).
    pub fn initialize_from_data(&mut self, data: &[Array1<f32>]) -> TokenizerResult<()> {
        if data.is_empty() {
            return Err(TokenizerError::InvalidConfig(
                "Cannot initialize from empty data".into(),
            ));
        }
        for (i, point) in data.iter().enumerate() {
            if point.len() != self.config.embed_dim {
                return Err(TokenizerError::dim_mismatch(
                    self.config.embed_dim,
                    point.len(),
                    format!("initialize_from_data: data[{i}]"),
                ));
            }
        }

        let mut rng = thread_rng();
        let mut centroids: Vec<&Array1<f32>> = Vec::with_capacity(self.config.codebook_size);

        // k-means++ initialization.
        // 1. Choose first centroid randomly.
        centroids.push(&data[rng.random_range(0..data.len())]);

        // 2. Choose remaining centroids with probability proportional to D^2.
        //
        // `min_dist` is maintained across iterations and updated with only
        // the most recently added centroid, giving O(codebook_size * data.len()
        // * embed_dim) total instead of recomputing every point's distance
        // to every previously-chosen centroid on each of the
        // `codebook_size` iterations (O(codebook_size^2 * data.len() * embed_dim)).
        let mut min_dist = vec![f32::INFINITY; data.len()];
        while centroids.len() < self.config.codebook_size {
            let last = centroids[centroids.len() - 1];
            for (i, point) in data.iter().enumerate() {
                let dist = euclidean_distance_sq(point, last);
                if dist < min_dist[i] {
                    min_dist[i] = dist;
                }
            }

            // Choose next centroid with probability proportional to distance^2
            // (min_dist is already squared).
            let total: f32 = min_dist.iter().sum();
            if total <= 0.0 {
                break;
            }

            let mut threshold = rng.random::<f32>() * total;
            let mut chosen = data.len() - 1;
            for (i, &dist) in min_dist.iter().enumerate() {
                threshold -= dist;
                if threshold <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            centroids.push(&data[chosen]);
        }

        // Update codebook. `data[i].len() == config.embed_dim` was validated
        // above, so every centroid row matches the codebook's column count.
        for (i, &centroid) in centroids.iter().enumerate().take(self.config.codebook_size) {
            self.codebook.row_mut(i).assign(centroid);
        }

        Ok(())
    }

    /// Find nearest codebook entry for a vector.
    ///
    /// Computes `argmin_i ||codebook[i] - vector||^2` via
    /// `||c||^2 - 2 * c.vector` (the `||vector||^2` term is constant across
    /// `i` and does not affect the argmin, so it is dropped): the cross
    /// term becomes a single matrix-vector product (`codebook.dot(vector)`)
    /// instead of `codebook_size` separate dot products, and `||c||^2` is a
    /// vectorized row-wise reduction instead of a manual nested loop.
    pub fn find_nearest(&self, vector: &Array1<f32>) -> TokenizerResult<usize> {
        if vector.len() != self.config.embed_dim {
            return Err(TokenizerError::dim_mismatch(
                self.config.embed_dim,
                vector.len(),
                "dimension validation",
            ));
        }

        let cross = self.codebook.dot(vector);
        let sq_norms = self.codebook.mapv(|v| v * v).sum_axis(Axis(1));

        let mut min_score = f32::INFINITY;
        let mut min_idx = 0;
        for (i, (&norm, &c)) in sq_norms.iter().zip(cross.iter()).enumerate() {
            let score = norm - 2.0 * c;
            if score < min_score {
                min_score = score;
                min_idx = i;
            }
        }

        Ok(min_idx)
    }

    /// Quantize a vector to its nearest codebook entry
    pub fn quantize(&self, vector: &Array1<f32>) -> TokenizerResult<(usize, Array1<f32>)> {
        let idx = self.find_nearest(vector)?;
        let quantized = self.codebook.row(idx).to_owned();
        Ok((idx, quantized))
    }

    /// Quantize multiple vectors in parallel (via `scirs2_core::parallel_ops`,
    /// falling back to sequential when the ecosystem's `parallel` feature is
    /// disabled). `quantize` takes `&self`, so calling it concurrently
    /// across vectors is safe.
    pub fn quantize_batch(
        &self,
        vectors: &[Array1<f32>],
    ) -> TokenizerResult<(Vec<usize>, Vec<Array1<f32>>)> {
        let results: Vec<TokenizerResult<(usize, Array1<f32>)>> =
            parallel_map(vectors, |vector| self.quantize(vector));

        let mut indices = Vec::with_capacity(vectors.len());
        let mut quantized = Vec::with_capacity(vectors.len());
        for result in results {
            let (idx, quant) = result?;
            indices.push(idx);
            quantized.push(quant);
        }

        Ok((indices, quantized))
    }

    /// Compute VQ losses: (total_loss, codebook_loss, commitment_loss)
    pub fn compute_loss(
        &self,
        encoder_output: &Array1<f32>,
        quantized: &Array1<f32>,
    ) -> (f32, f32, f32) {
        // Codebook loss: ||sg[encoder_output] - codebook||^2
        let codebook_loss: f32 = encoder_output
            .iter()
            .zip(quantized.iter())
            .map(|(e, q)| (e - q).powi(2))
            .sum();

        // Commitment loss: ||encoder_output - sg[codebook]||^2
        let commitment_loss: f32 = encoder_output
            .iter()
            .zip(quantized.iter())
            .map(|(e, q)| (e - q).powi(2))
            .sum();

        let total_loss = codebook_loss + self.config.commitment_beta * commitment_loss;

        (total_loss, codebook_loss, commitment_loss)
    }

    /// Update codebook using EMA.
    ///
    /// # Errors
    /// Returns [`TokenizerError::InvalidConfig`] if `config.use_ema` is
    /// `false`. This quantizer has no gradient-based codebook update path
    /// (the "vs gradient-based" alternative `VQConfig::use_ema` documents),
    /// so honoring `use_ema: false` by silently running the EMA update
    /// anyway would contradict the flag's documented meaning; erroring
    /// makes the gap visible instead. Set `use_ema: true`, or update the
    /// codebook through another path (e.g. [`VectorQuantizer::initialize_from_data`]),
    /// if `use_ema: false` is intentional.
    pub fn update_ema(
        &mut self,
        encoder_outputs: &[Array1<f32>],
        indices: &[usize],
    ) -> TokenizerResult<()> {
        if !self.config.use_ema {
            return Err(TokenizerError::InvalidConfig(
                "VectorQuantizer::update_ema called with config.use_ema == false: this quantizer \
                 has no gradient-based codebook update, so there is nothing else update_ema could \
                 honestly do; set use_ema: true to use EMA updates"
                    .into(),
            ));
        }

        if encoder_outputs.len() != indices.len() {
            return Err(TokenizerError::InvalidConfig(
                "Encoder outputs and indices length mismatch".into(),
            ));
        }

        // Reset temporary accumulators
        let mut cluster_sizes = Array1::<f32>::zeros(self.config.codebook_size);
        let mut embed_sums =
            Array2::<f32>::zeros((self.config.codebook_size, self.config.embed_dim));

        // Accumulate
        for (output, &idx) in encoder_outputs.iter().zip(indices.iter()) {
            cluster_sizes[idx] += 1.0;
            for (j, &val) in output.iter().enumerate() {
                embed_sums[[idx, j]] += val;
            }
            self.usage_counts[idx] += 1;
        }

        // EMA update
        let decay = self.config.ema_decay;
        let epsilon = self.config.epsilon;

        for i in 0..self.config.codebook_size {
            // Update cluster size EMA
            self.ema_cluster_size[i] =
                decay * self.ema_cluster_size[i] + (1.0 - decay) * cluster_sizes[i];

            // Laplace smoothing for numerical stability
            let n = self.ema_cluster_size[i] + epsilon;

            // Update embedding sum EMA and codebook
            for j in 0..self.config.embed_dim {
                self.ema_embed_sum[[i, j]] =
                    decay * self.ema_embed_sum[[i, j]] + (1.0 - decay) * embed_sums[[i, j]];

                // Update codebook entry
                self.codebook[[i, j]] = self.ema_embed_sum[[i, j]] / n;
            }
        }

        Ok(())
    }

    /// Reset unused codebook entries to random encoder outputs
    pub fn reset_unused_codes(
        &mut self,
        encoder_outputs: &[Array1<f32>],
        threshold: usize,
    ) -> usize {
        let mut rng = thread_rng();
        let mut reset_count = 0;

        for i in 0..self.config.codebook_size {
            if self.usage_counts[i] < threshold && !encoder_outputs.is_empty() {
                // Replace with random encoder output
                let random_idx = rng.random_range(0..encoder_outputs.len());
                let random_output = &encoder_outputs[random_idx];

                for (j, &val) in random_output.iter().enumerate() {
                    if j < self.config.embed_dim {
                        self.codebook[[i, j]] = val;
                    }
                }

                // Reset EMA statistics
                self.ema_cluster_size[i] = 1.0;
                for j in 0..self.config.embed_dim {
                    self.ema_embed_sum[[i, j]] = self.codebook[[i, j]];
                }
                self.usage_counts[i] = 0;

                reset_count += 1;
            }
        }

        reset_count
    }

    /// Get codebook entry by index
    pub fn get_codebook_entry(&self, idx: usize) -> TokenizerResult<Array1<f32>> {
        if idx >= self.config.codebook_size {
            return Err(TokenizerError::InvalidConfig(format!(
                "Index {} out of codebook range 0..{}",
                idx, self.config.codebook_size
            )));
        }
        Ok(self.codebook.row(idx).to_owned())
    }

    /// Get the full codebook
    pub fn codebook(&self) -> &Array2<f32> {
        &self.codebook
    }

    /// Get codebook size
    pub fn codebook_size(&self) -> usize {
        self.config.codebook_size
    }

    /// Get embedding dimension
    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    /// Get usage statistics
    pub fn usage_stats(&self) -> (usize, usize, f32) {
        let total_uses: usize = self.usage_counts.iter().sum();
        let used_codes = self.usage_counts.iter().filter(|&&c| c > 0).count();
        let utilization = used_codes as f32 / self.config.codebook_size as f32;
        (total_uses, used_codes, utilization)
    }

    /// Reset usage counters
    pub fn reset_usage_counts(&mut self) {
        self.usage_counts.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_quantizer_creation() {
        let config = VQConfig::default();
        let vq = VectorQuantizer::new(config.clone());

        assert_eq!(vq.codebook_size(), config.codebook_size);
        assert_eq!(vq.embed_dim(), config.embed_dim);
    }

    #[test]
    fn test_quantization() {
        let config = VQConfig {
            codebook_size: 8,
            embed_dim: 4,
            ..Default::default()
        };
        let vq = VectorQuantizer::new(config);

        let vector = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let (idx, quantized) = vq.quantize(&vector).unwrap();

        assert!(idx < 8);
        assert_eq!(quantized.len(), 4);
    }

    #[test]
    fn test_find_nearest() {
        let config = VQConfig {
            codebook_size: 4,
            embed_dim: 2,
            ..Default::default()
        };
        let mut vq = VectorQuantizer::new(config);

        // Manually set codebook entries
        vq.codebook[[0, 0]] = 0.0;
        vq.codebook[[0, 1]] = 0.0;
        vq.codebook[[1, 0]] = 1.0;
        vq.codebook[[1, 1]] = 0.0;
        vq.codebook[[2, 0]] = 0.0;
        vq.codebook[[2, 1]] = 1.0;
        vq.codebook[[3, 0]] = 1.0;
        vq.codebook[[3, 1]] = 1.0;

        let vector = Array1::from_vec(vec![0.9, 0.1]);
        let idx = vq.find_nearest(&vector).unwrap();
        assert_eq!(idx, 1); // Closest to [1.0, 0.0]
    }

    #[test]
    fn test_compute_loss() {
        let config = VQConfig::default();
        let vq = VectorQuantizer::new(config);

        let encoder_output = Array1::from_vec(vec![0.5; 64]);
        let quantized = Array1::from_vec(vec![0.4; 64]);

        let (total_loss, codebook_loss, commitment_loss) =
            vq.compute_loss(&encoder_output, &quantized);

        assert!(total_loss > 0.0);
        assert!(codebook_loss > 0.0);
        assert!(commitment_loss > 0.0);
    }

    #[test]
    fn test_ema_update() {
        let config = VQConfig {
            codebook_size: 4,
            embed_dim: 8,
            use_ema: true,
            ..Default::default()
        };
        let mut vq = VectorQuantizer::new(config);

        let outputs = vec![
            Array1::from_vec(vec![0.1; 8]),
            Array1::from_vec(vec![0.2; 8]),
            Array1::from_vec(vec![0.3; 8]),
        ];
        let indices = vec![0, 1, 0];

        vq.update_ema(&outputs, &indices).unwrap();

        // Check that codebook has been updated
        let (total, used, util) = vq.usage_stats();
        assert_eq!(total, 3);
        assert_eq!(used, 2); // Only indices 0 and 1 were used
        assert!(util > 0.0);
    }

    /// Regression: before the fix, `ema_cluster_size`/`ema_embed_sum`
    /// started at zero, so a single `update_ema` call zeroed out every
    /// codebook row that received no vectors in the batch (dividing
    /// `0.0 / (0.0 + epsilon)`), destroying the random initialization.
    /// Unused rows must instead stay close to their pre-update value.
    #[test]
    fn test_update_ema_preserves_unused_codebook_rows() {
        let config = VQConfig {
            codebook_size: 512,
            embed_dim: 8,
            use_ema: true,
            ..Default::default()
        };
        let mut vq = VectorQuantizer::new(config);
        let codebook_before = vq.codebook.clone();

        // Only 2 of 512 codes are used in this update.
        let outputs = vec![
            Array1::from_vec(vec![0.1; 8]),
            Array1::from_vec(vec![0.2; 8]),
            Array1::from_vec(vec![0.3; 8]),
        ];
        let indices = vec![0usize, 1, 0];
        vq.update_ema(&outputs, &indices).unwrap();

        for i in 2..512 {
            for j in 0..8 {
                let before = codebook_before[[i, j]];
                let after = vq.codebook[[i, j]];
                assert!(
                    (after - before).abs() < 1e-3,
                    "unused row {i} col {j} changed from {before} to {after} after a single update_ema call"
                );
                assert_ne!(
                    after, 0.0,
                    "unused row {i} col {j} collapsed to exactly 0.0"
                );
            }
        }
    }

    /// Regression: `update_ema` must not silently behave identically
    /// whether `use_ema` is true or false — it has no alternative
    /// (gradient-based) update path, so `use_ema: false` must be a real error.
    #[test]
    fn test_update_ema_errors_when_use_ema_false() {
        let config = VQConfig {
            codebook_size: 4,
            embed_dim: 8,
            use_ema: false,
            ..Default::default()
        };
        let mut vq = VectorQuantizer::new(config);
        let outputs = vec![Array1::from_vec(vec![0.1; 8])];
        let indices = vec![0usize];

        assert!(vq.update_ema(&outputs, &indices).is_err());
    }

    /// Regression: `initialize_from_data` used to index
    /// `self.codebook[[i, j]]` with no bound on `j`, panicking on any input
    /// vector longer than `embed_dim`.
    #[test]
    fn test_initialize_from_data_rejects_wrong_length_points() {
        let config = VQConfig {
            codebook_size: 2,
            embed_dim: 3,
            ..Default::default()
        };
        let mut vq = VectorQuantizer::new(config);

        let too_long = vec![
            Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]), // 4 != embed_dim 3
            Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]),
        ];
        assert!(vq.initialize_from_data(&too_long).is_err());

        let too_short = vec![
            Array1::from_vec(vec![0.0, 0.0]), // 2 != embed_dim 3
            Array1::from_vec(vec![1.0, 1.0]),
        ];
        assert!(vq.initialize_from_data(&too_short).is_err());
    }

    #[test]
    fn test_batch_quantization() {
        let config = VQConfig {
            codebook_size: 8,
            embed_dim: 4,
            ..Default::default()
        };
        let vq = VectorQuantizer::new(config);

        let vectors = vec![
            Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]),
            Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8]),
            Array1::from_vec(vec![0.9, 1.0, 1.1, 1.2]),
        ];

        let (indices, quantized) = vq.quantize_batch(&vectors).unwrap();

        assert_eq!(indices.len(), 3);
        assert_eq!(quantized.len(), 3);
        for q in &quantized {
            assert_eq!(q.len(), 4);
        }
    }

    #[test]
    fn test_reset_unused_codes() {
        let config = VQConfig {
            codebook_size: 8,
            embed_dim: 4,
            ..Default::default()
        };
        let mut vq = VectorQuantizer::new(config);

        // Simulate usage of only some codes
        vq.usage_counts[0] = 10;
        vq.usage_counts[1] = 5;
        // Rest are 0

        let encoder_outputs = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
            Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]),
        ];

        let reset_count = vq.reset_unused_codes(&encoder_outputs, 3);

        // Should reset codes with usage < 3 (that's codes 2-7, so 6 codes)
        assert_eq!(reset_count, 6);
    }

    #[test]
    fn test_initialization_from_data() {
        let config = VQConfig {
            codebook_size: 4,
            embed_dim: 3,
            ..Default::default()
        };
        let mut vq = VectorQuantizer::new(config);

        let data = vec![
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0, 0.0]),
            Array1::from_vec(vec![0.0, 0.0, 1.0]),
            Array1::from_vec(vec![1.0, 1.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0, 1.0]),
            Array1::from_vec(vec![0.0, 1.0, 1.0]),
            Array1::from_vec(vec![1.0, 1.0, 1.0]),
        ];

        vq.initialize_from_data(&data).unwrap();

        // Verify codebook was updated
        for i in 0..4 {
            let entry = vq.get_codebook_entry(i).unwrap();
            assert_eq!(entry.len(), 3);
        }
    }
}
