//! Model Compression Utilities
//!
//! Provides techniques for reducing model size and computational cost while
//! maintaining performance.
//!
//! # Techniques
//!
//! - **Pruning**: Remove less important weights
//! - **Knowledge Distillation**: Transfer knowledge from large to small models
//! - **Weight Sharing**: Share weights across layers
//! - **Low-Rank Factorization**: Decompose weight matrices
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::compression::{PruningConfig, prune_model};
//!
//! let config = PruningConfig::magnitude_based(0.3); // Prune 30% of weights
//! let compressed_model = prune_model(&model, &config)?;
//! ```

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pruning strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PruningStrategy {
    /// Magnitude-based pruning (remove smallest weights)
    Magnitude,
    /// Random pruning
    Random,
    /// Structured pruning (entire neurons/channels)
    Structured,
    /// Movement pruning (based on weight updates)
    Movement,
}

/// Pruning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Pruning strategy
    pub strategy: PruningStrategy,
    /// Sparsity ratio (0.0 - 1.0)
    pub sparsity: f32,
    /// Whether to use global or layer-wise threshold
    pub global_threshold: bool,
    /// Minimum sparsity per layer
    pub min_sparsity: f32,
    /// Maximum sparsity per layer
    pub max_sparsity: f32,
}

impl PruningConfig {
    /// Create magnitude-based pruning configuration
    pub fn magnitude_based(sparsity: f32) -> Self {
        Self {
            strategy: PruningStrategy::Magnitude,
            sparsity,
            global_threshold: true,
            min_sparsity: 0.0,
            max_sparsity: 0.95,
        }
    }

    /// Create structured pruning configuration
    pub fn structured(sparsity: f32) -> Self {
        Self {
            strategy: PruningStrategy::Structured,
            sparsity,
            global_threshold: false,
            min_sparsity: 0.0,
            max_sparsity: 0.9,
        }
    }

    /// Set global threshold flag
    pub fn global(mut self, global: bool) -> Self {
        self.global_threshold = global;
        self
    }

    /// Set sparsity bounds
    pub fn bounds(mut self, min: f32, max: f32) -> Self {
        self.min_sparsity = min;
        self.max_sparsity = max;
        self
    }
}

/// Pruning statistics
#[derive(Debug, Clone)]
pub struct PruningStats {
    /// Total number of parameters
    pub total_params: usize,
    /// Number of pruned parameters
    pub pruned_params: usize,
    /// Sparsity ratio achieved
    pub sparsity: f32,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Per-layer statistics
    pub layer_stats: HashMap<String, LayerPruningStats>,
}

/// Per-layer pruning statistics
#[derive(Debug, Clone)]
pub struct LayerPruningStats {
    /// Total parameters in layer
    pub total: usize,
    /// Pruned parameters in layer
    pub pruned: usize,
    /// Layer sparsity
    pub sparsity: f32,
}

impl PruningStats {
    /// Create new pruning statistics
    pub fn new() -> Self {
        Self {
            total_params: 0,
            pruned_params: 0,
            sparsity: 0.0,
            compression_ratio: 1.0,
            layer_stats: HashMap::new(),
        }
    }

    /// Calculate final statistics
    pub fn finalize(&mut self) {
        if self.total_params > 0 {
            self.sparsity = self.pruned_params as f32 / self.total_params as f32;
            self.compression_ratio = 1.0 / (1.0 - self.sparsity);
        }
    }

    /// Add layer statistics
    pub fn add_layer(&mut self, name: String, total: usize, pruned: usize) {
        self.total_params += total;
        self.pruned_params += pruned;

        let sparsity = if total > 0 {
            pruned as f32 / total as f32
        } else {
            0.0
        };

        self.layer_stats.insert(
            name,
            LayerPruningStats {
                total,
                pruned,
                sparsity,
            },
        );
    }

    /// Print summary
    pub fn print_summary(&self) {
        tracing::info!("=== Pruning Statistics ===");
        tracing::info!("Total parameters: {}", self.total_params);
        tracing::info!("Pruned parameters: {}", self.pruned_params);
        tracing::info!("Sparsity: {:.2}%", self.sparsity * 100.0);
        tracing::info!("Compression ratio: {:.2}x", self.compression_ratio);
        tracing::info!("\nPer-layer statistics:");
        for (name, stats) in &self.layer_stats {
            tracing::info!(
                "  {}: {}/{} ({:.2}%)",
                name,
                stats.pruned,
                stats.total,
                stats.sparsity * 100.0
            );
        }
    }
}

impl Default for PruningStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Prune a weight matrix using magnitude-based pruning
pub fn prune_magnitude(
    weights: &Array2<f32>,
    sparsity: f32,
) -> ModelResult<(Array2<f32>, Array2<bool>)> {
    if !(0.0..=1.0).contains(&sparsity) {
        return Err(ModelError::invalid_config(format!(
            "Pruning: Sparsity must be between 0 and 1, got {}",
            sparsity
        )));
    }

    let total_elements = weights.len();
    let num_to_prune = (total_elements as f32 * sparsity) as usize;

    // Get absolute values and sort
    let mut abs_weights: Vec<(f32, (usize, usize))> = weights
        .indexed_iter()
        .map(|(idx, &val)| (val.abs(), idx))
        .collect();

    abs_weights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Create pruning mask
    let mut mask = Array2::from_elem(weights.dim(), true);
    for i in 0..num_to_prune {
        if i < abs_weights.len() {
            let (_, idx) = abs_weights[i];
            mask[idx] = false;
        }
    }

    // Apply mask
    let pruned = weights * &mask.mapv(|x| if x { 1.0 } else { 0.0 });

    Ok((pruned, mask))
}

/// Prune weights based on a global threshold
pub fn prune_threshold(
    weights: &Array2<f32>,
    threshold: f32,
) -> ModelResult<(Array2<f32>, Array2<bool>)> {
    let mask = weights.mapv(|x| x.abs() >= threshold);
    let pruned = weights * &mask.mapv(|x| if x { 1.0 } else { 0.0 });

    Ok((pruned, mask))
}

/// Knowledge distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Temperature for softening probability distributions
    pub temperature: f32,
    /// Weight for distillation loss (0.0 - 1.0)
    pub alpha: f32,
    /// Weight for task loss (1-alpha typically)
    pub task_weight: f32,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 3.0,
            alpha: 0.7,
            task_weight: 0.3,
        }
    }
}

impl DistillationConfig {
    /// Create new distillation config
    pub fn new(temperature: f32, alpha: f32) -> Self {
        Self {
            temperature,
            alpha,
            task_weight: 1.0 - alpha,
        }
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set alpha (distillation weight)
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self.task_weight = 1.0 - alpha;
        self
    }
}

/// Compute distillation loss between teacher and student outputs
pub fn distillation_loss(
    student_logits: &Array1<f32>,
    teacher_logits: &Array1<f32>,
    temperature: f32,
) -> ModelResult<f32> {
    if student_logits.len() != teacher_logits.len() {
        return Err(ModelError::dimension_mismatch(
            "distillation loss",
            student_logits.len(),
            teacher_logits.len(),
        ));
    }

    // Apply temperature scaling
    let student_scaled = student_logits.mapv(|x| x / temperature);
    let teacher_scaled = teacher_logits.mapv(|x| x / temperature);

    // Compute softmax
    let student_max = student_scaled.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let teacher_max = teacher_scaled.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let student_exp = student_scaled.mapv(|x| (x - student_max).exp());
    let teacher_exp = teacher_scaled.mapv(|x| (x - teacher_max).exp());

    let student_sum = student_exp.sum();
    let teacher_sum = teacher_exp.sum();

    let student_probs = &student_exp / student_sum;
    let teacher_probs = &teacher_exp / teacher_sum;

    // KL divergence: sum(teacher * log(teacher / student))
    let mut kl_div = 0.0;
    for i in 0..student_probs.len() {
        if teacher_probs[i] > 1e-10 && student_probs[i] > 1e-10 {
            kl_div += teacher_probs[i] * (teacher_probs[i] / student_probs[i]).ln();
        }
    }

    // Scale by temperature squared (as per Hinton et al.)
    Ok(kl_div * temperature * temperature)
}

/// Low-rank factorization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowRankConfig {
    /// Rank for factorization
    pub rank: usize,
    /// Whether to use SVD or other methods
    pub use_svd: bool,
}

impl LowRankConfig {
    /// Create new low-rank config
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            use_svd: true,
        }
    }

    /// Set SVD flag
    pub fn svd(mut self, use_svd: bool) -> Self {
        self.use_svd = use_svd;
        self
    }
}

/// Compute compression ratio from original and compressed sizes
pub fn compression_ratio(original_size: usize, compressed_size: usize) -> f32 {
    if compressed_size == 0 {
        return f32::INFINITY;
    }
    original_size as f32 / compressed_size as f32
}

/// Weight sharing utilities
pub mod weight_sharing {
    use super::*;

    /// K-means clustering for weight sharing
    pub fn kmeans_cluster(weights: &Array2<f32>, num_clusters: usize) -> ModelResult<Array2<f32>> {
        if num_clusters == 0 || num_clusters > weights.len() {
            return Err(ModelError::invalid_config(format!(
                "K-means clustering: Invalid number of clusters: {}",
                num_clusters
            )));
        }

        // Simple k-means implementation
        // In production, use a proper clustering library
        let flat_weights: Vec<f32> = weights.iter().copied().collect();

        // Initialize centroids
        let mut centroids = Vec::new();
        let step = flat_weights.len() / num_clusters;
        for i in 0..num_clusters {
            if i * step < flat_weights.len() {
                centroids.push(flat_weights[i * step]);
            }
        }

        // Iterative refinement (simplified)
        for _ in 0..10 {
            let mut cluster_sums = vec![0.0; num_clusters];
            let mut cluster_counts = vec![0usize; num_clusters];

            for &weight in &flat_weights {
                let mut min_dist = f32::INFINITY;
                let mut cluster_id = 0;

                for (i, &centroid) in centroids.iter().enumerate() {
                    let dist = (weight - centroid).abs();
                    if dist < min_dist {
                        min_dist = dist;
                        cluster_id = i;
                    }
                }

                cluster_sums[cluster_id] += weight;
                cluster_counts[cluster_id] += 1;
            }

            // Update centroids
            for i in 0..num_clusters {
                if cluster_counts[i] > 0 {
                    centroids[i] = cluster_sums[i] / cluster_counts[i] as f32;
                }
            }
        }

        // Assign weights to nearest centroid
        let mut quantized = Array2::zeros(weights.dim());
        for (idx, &weight) in weights.indexed_iter() {
            let mut min_dist = f32::INFINITY;
            let mut best_centroid = centroids[0];

            for &centroid in &centroids {
                let dist = (weight - centroid).abs();
                if dist < min_dist {
                    min_dist = dist;
                    best_centroid = centroid;
                }
            }

            quantized[idx] = best_centroid;
        }

        Ok(quantized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prune_magnitude() {
        let weights = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0],
        )
        .expect("Failed to create test array");

        let (pruned, mask) = prune_magnitude(&weights, 0.5).expect("Failed to prune");

        // Should prune ~50% of smallest magnitude weights
        let num_zeros = pruned.iter().filter(|&&x| x == 0.0).count();
        assert!(num_zeros >= 4);
        assert_eq!(pruned.dim(), weights.dim());
        assert_eq!(mask.dim(), weights.dim());
    }

    #[test]
    fn test_prune_threshold() {
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.1, 2.0])
            .expect("Failed to create test array");

        let (pruned, mask) = prune_threshold(&weights, 0.6).expect("Failed to prune");

        assert_eq!(pruned[[0, 0]], 1.0);
        assert_eq!(pruned[[0, 1]], 0.0); // 0.5 < 0.6
        assert_eq!(pruned[[1, 0]], 0.0); // 0.1 < 0.6
        assert_eq!(pruned[[1, 1]], 2.0);

        assert!(mask[[0, 0]]);
        assert!(!mask[[0, 1]]);
        assert!(!mask[[1, 0]]);
        assert!(mask[[1, 1]]);
    }

    #[test]
    fn test_distillation_loss() {
        let student = Array1::from_vec(vec![2.0, 1.0, 0.1]);
        let teacher = Array1::from_vec(vec![2.5, 1.5, 0.5]);

        let loss = distillation_loss(&student, &teacher, 3.0).expect("Failed to compute loss");

        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_pruning_config() {
        let config = PruningConfig::magnitude_based(0.3)
            .global(false)
            .bounds(0.1, 0.8);

        assert_eq!(config.strategy, PruningStrategy::Magnitude);
        assert_eq!(config.sparsity, 0.3);
        assert!(!config.global_threshold);
        assert_eq!(config.min_sparsity, 0.1);
        assert_eq!(config.max_sparsity, 0.8);
    }

    #[test]
    fn test_distillation_config() {
        let config = DistillationConfig::new(5.0, 0.8);

        assert_eq!(config.temperature, 5.0);
        assert_eq!(config.alpha, 0.8);
        assert!((config.task_weight - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = compression_ratio(1000, 250);
        assert_eq!(ratio, 4.0);

        let ratio = compression_ratio(1000, 1000);
        assert_eq!(ratio, 1.0);
    }

    #[test]
    fn test_pruning_stats() {
        let mut stats = PruningStats::new();
        stats.add_layer("layer1".to_string(), 1000, 300);
        stats.add_layer("layer2".to_string(), 2000, 800);
        stats.finalize();

        assert_eq!(stats.total_params, 3000);
        assert_eq!(stats.pruned_params, 1100);
        assert!((stats.sparsity - 0.366667).abs() < 1e-5);
        assert!(stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_kmeans_weight_sharing() {
        let weights = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
            .expect("Failed to create test array");

        let quantized = weight_sharing::kmeans_cluster(&weights, 2).expect("Failed to cluster");

        assert_eq!(quantized.dim(), weights.dim());

        // Should have only 2 unique values
        let unique_vals: std::collections::HashSet<_> =
            quantized.iter().map(|&x| (x * 1000.0) as i32).collect();
        assert!(unique_vals.len() <= 2);
    }
}
