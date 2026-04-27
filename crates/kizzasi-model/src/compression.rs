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

// ---------------------------------------------------------------------------
// MagnitudePruner
// ---------------------------------------------------------------------------

/// Unstructured magnitude-based weight pruner.
///
/// Zeroes out all weight entries whose absolute value is strictly below
/// `threshold`. Tracks cumulative pruning statistics across calls.
#[derive(Debug, Clone)]
pub struct MagnitudePruner {
    /// Magnitude threshold: entries with |w| < threshold are zeroed
    pub threshold: f32,
    /// Total number of entries pruned so far
    pub pruned_count: usize,
    /// Total number of entries processed so far
    pub total_count: usize,
}

impl MagnitudePruner {
    /// Create a new `MagnitudePruner` with the given magnitude threshold.
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            pruned_count: 0,
            total_count: 0,
        }
    }

    /// Prune a 2D weight matrix in-place. Returns the sparsity fraction of
    /// this call (not cumulative).
    pub fn prune_matrix(&mut self, w: &mut Array2<f32>) -> f32 {
        let total = w.len();
        let mut pruned = 0usize;
        for v in w.iter_mut() {
            if v.abs() < self.threshold {
                *v = 0.0;
                pruned += 1;
            }
        }
        self.total_count += total;
        self.pruned_count += pruned;
        if total == 0 {
            0.0
        } else {
            pruned as f32 / total as f32
        }
    }

    /// Prune a 1D weight vector in-place. Returns the sparsity fraction of
    /// this call (not cumulative).
    pub fn prune_vector(&mut self, v: &mut Array1<f32>) -> f32 {
        let total = v.len();
        let mut pruned = 0usize;
        for x in v.iter_mut() {
            if x.abs() < self.threshold {
                *x = 0.0;
                pruned += 1;
            }
        }
        self.total_count += total;
        self.pruned_count += pruned;
        if total == 0 {
            0.0
        } else {
            pruned as f32 / total as f32
        }
    }

    /// Cumulative sparsity fraction across all processed entries.
    pub fn sparsity(&self) -> f32 {
        if self.total_count == 0 {
            0.0
        } else {
            self.pruned_count as f32 / self.total_count as f32
        }
    }

    /// Reset cumulative statistics (threshold is kept).
    pub fn reset_stats(&mut self) {
        self.pruned_count = 0;
        self.total_count = 0;
    }
}

// ---------------------------------------------------------------------------
// StructuredPruner
// ---------------------------------------------------------------------------

/// Structured pruner: removes entire rows (neurons/channels) with the
/// smallest L2 norms, keeping `keep_fraction` of rows.
#[derive(Debug, Clone)]
pub struct StructuredPruner {
    /// Fraction of rows to retain (0.0 – 1.0)
    pub keep_fraction: f32,
}

impl StructuredPruner {
    /// Create a new `StructuredPruner` that keeps `keep_fraction` of rows.
    pub fn new(keep_fraction: f32) -> Self {
        Self { keep_fraction }
    }

    /// Compute a boolean mask over rows (true = keep).
    ///
    /// The top `ceil(keep_fraction * nrows)` rows by L2 norm are kept.
    pub fn prune_rows(&self, w: &Array2<f32>) -> ModelResult<Vec<bool>> {
        let nrows = w.nrows();
        if nrows == 0 {
            return Err(ModelError::invalid_config(
                "StructuredPruner::prune_rows: empty matrix",
            ));
        }
        let keep = ((self.keep_fraction * nrows as f32).ceil() as usize).min(nrows);

        // Compute L2 norm per row
        let mut row_norms: Vec<(usize, f32)> = (0..nrows)
            .map(|i| {
                let norm = w.row(i).iter().map(|&x| x * x).sum::<f32>().sqrt();
                (i, norm)
            })
            .collect();

        // Sort descending by norm
        row_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut mask = vec![false; nrows];
        for (row_idx, _) in row_norms.iter().take(keep) {
            mask[*row_idx] = true;
        }
        Ok(mask)
    }

    /// Return a new matrix with pruned rows removed.
    pub fn compress_rows(&self, w: &Array2<f32>) -> ModelResult<Array2<f32>> {
        let mask = self.prune_rows(w)?;
        let kept_rows: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();

        if kept_rows.is_empty() {
            return Err(ModelError::invalid_config(
                "StructuredPruner::compress_rows: no rows kept",
            ));
        }

        let ncols = w.ncols();
        let mut out = Array2::<f32>::zeros((kept_rows.len(), ncols));
        for (new_i, &old_i) in kept_rows.iter().enumerate() {
            for j in 0..ncols {
                out[(new_i, j)] = w[(old_i, j)];
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// LowRankApprox
// ---------------------------------------------------------------------------

/// Low-rank approximation W ≈ U @ diag(S) @ V^T computed via pure-Rust
/// power iteration (no LAPACK / C dependencies).
#[derive(Debug, Clone)]
pub struct LowRankApprox {
    /// Target rank
    pub rank: usize,
    /// Left singular vectors — shape `(rows, rank)`
    pub u: Array2<f32>,
    /// Right singular vectors (transposed) — shape `(rank, cols)`
    pub vt: Array2<f32>,
    /// Singular values — shape `(rank,)`
    pub singular_values: Array1<f32>,
    /// Relative Frobenius reconstruction error `||W - approx||_F / ||W||_F`
    pub reconstruction_error: f32,
}

impl LowRankApprox {
    /// Compute a rank-`rank` approximation of `w` using power iteration.
    ///
    /// `num_iter` controls the number of power-iteration steps per component.
    /// Higher values give more accurate singular vectors.
    pub fn compute(w: &Array2<f32>, rank: usize, num_iter: usize) -> ModelResult<Self> {
        let rows = w.nrows();
        let cols = w.ncols();

        if rank == 0 {
            return Err(ModelError::invalid_config(
                "LowRankApprox: rank must be > 0",
            ));
        }
        let effective_rank = rank.min(rows.min(cols));

        let mut u_cols: Vec<Array1<f32>> = Vec::with_capacity(effective_rank);
        let mut vt_rows: Vec<Array1<f32>> = Vec::with_capacity(effective_rank);
        let mut sigmas: Vec<f32> = Vec::with_capacity(effective_rank);

        // Working copy for deflation
        let mut residual = w.clone();

        for k in 0..effective_rank {
            // Initialise right singular vector (deterministic)
            let mut v = Array1::<f32>::zeros(cols);
            v[k % cols] = 1.0;

            let iters = num_iter.max(1);
            for _ in 0..iters {
                // u = residual @ v  — shape (rows,)
                let mut u_vec = Array1::<f32>::zeros(rows);
                for i in 0..rows {
                    u_vec[i] = (0..cols).map(|j| residual[(i, j)] * v[j]).sum();
                }
                // sigma = ||u||
                let sigma = u_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if sigma < 1e-12 {
                    break;
                }
                // u = u / sigma
                let u_norm = u_vec.mapv(|x| x / sigma);

                // v_new = residual^T @ u_norm  — shape (cols,)
                let mut v_new = Array1::<f32>::zeros(cols);
                for j in 0..cols {
                    v_new[j] = (0..rows).map(|i| residual[(i, j)] * u_norm[i]).sum();
                }
                let v_norm_val = v_new.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if v_norm_val < 1e-12 {
                    break;
                }
                v = v_new.mapv(|x| x / v_norm_val);
            }

            // Final computation of u and sigma
            let mut u_vec = Array1::<f32>::zeros(rows);
            for i in 0..rows {
                u_vec[i] = (0..cols).map(|j| residual[(i, j)] * v[j]).sum();
            }
            let sigma = u_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if sigma < 1e-12 {
                // No more signal — fill remaining components with zeros
                u_cols.push(Array1::zeros(rows));
                vt_rows.push(Array1::zeros(cols));
                sigmas.push(0.0);
            } else {
                let u_final = u_vec.mapv(|x| x / sigma);

                // Deflate
                for i in 0..rows {
                    for j in 0..cols {
                        residual[(i, j)] -= sigma * u_final[i] * v[j];
                    }
                }

                u_cols.push(u_final);
                vt_rows.push(v);
                sigmas.push(sigma);
            }
        }

        // Assemble U (rows, rank) and Vt (rank, cols)
        let mut u_mat = Array2::<f32>::zeros((rows, effective_rank));
        let mut vt_mat = Array2::<f32>::zeros((effective_rank, cols));
        for k in 0..effective_rank {
            for i in 0..rows {
                u_mat[(i, k)] = u_cols[k][i];
            }
            for j in 0..cols {
                vt_mat[(k, j)] = vt_rows[k][j];
            }
        }
        let singular_values = Array1::from_vec(sigmas);

        // Reconstruction error
        let w_frob: f32 = w.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let rec_error = if w_frob < 1e-12 {
            0.0
        } else {
            // approx = U S Vt
            let mut err_sq = 0.0_f32;
            for i in 0..rows {
                for j in 0..cols {
                    let approx: f32 = (0..effective_rank)
                        .map(|k| u_mat[(i, k)] * singular_values[k] * vt_mat[(k, j)])
                        .sum();
                    err_sq += (w[(i, j)] - approx).powi(2);
                }
            }
            err_sq.sqrt() / w_frob
        };

        Ok(Self {
            rank: effective_rank,
            u: u_mat,
            vt: vt_mat,
            singular_values,
            reconstruction_error: rec_error,
        })
    }

    /// Reconstruct the full matrix: U @ diag(S) @ V^T.
    pub fn reconstruct(&self) -> ModelResult<Array2<f32>> {
        let rows = self.u.nrows();
        let cols = self.vt.ncols();
        let mut out = Array2::<f32>::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                out[(i, j)] = (0..self.rank)
                    .map(|k| self.u[(i, k)] * self.singular_values[k] * self.vt[(k, j)])
                    .sum();
            }
        }
        Ok(out)
    }

    /// Compression ratio: `(rows * cols) / (rows * rank + rank * cols)`.
    pub fn compression_ratio(&self) -> f32 {
        let rows = self.u.nrows();
        let cols = self.vt.ncols();
        let original = rows * cols;
        let compressed = rows * self.rank + self.rank * cols;
        if compressed == 0 {
            return f32::INFINITY;
        }
        original as f32 / compressed as f32
    }

    /// Fast forward pass using factored form: `(U @ diag(S)) @ (V^T @ x)`.
    ///
    /// `x` must have length equal to the number of columns (original input dim).
    pub fn forward(&self, x: &Array1<f32>) -> ModelResult<Array1<f32>> {
        let cols = self.vt.ncols();
        let rows = self.u.nrows();
        if x.len() != cols {
            return Err(ModelError::dimension_mismatch(
                "LowRankApprox::forward",
                cols,
                x.len(),
            ));
        }
        // intermediate = V^T @ x  — shape (rank,)
        let mut intermediate = Array1::<f32>::zeros(self.rank);
        for k in 0..self.rank {
            intermediate[k] = (0..cols).map(|j| self.vt[(k, j)] * x[j]).sum();
        }
        // scale by singular values
        for k in 0..self.rank {
            intermediate[k] *= self.singular_values[k];
        }
        // output = U @ intermediate  — shape (rows,)
        let mut out = Array1::<f32>::zeros(rows);
        for i in 0..rows {
            out[i] = (0..self.rank)
                .map(|k| self.u[(i, k)] * intermediate[k])
                .sum();
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// CompressionReport
// ---------------------------------------------------------------------------

/// Summary report of compression applied to a set of weight matrices.
#[derive(Debug, Clone)]
pub struct CompressionReport {
    /// Total number of parameters in original model
    pub original_params: usize,
    /// Total number of parameters after compression
    pub compressed_params: usize,
    /// Number of parameters set to zero (pruned)
    pub pruned_params: usize,
    /// `(layer_name, original_rank, compressed_rank)` per layer
    pub rank_reductions: Vec<(String, usize, usize)>,
    /// Overall compression ratio
    pub overall_compression_ratio: f32,
}

impl CompressionReport {
    /// Create a new, empty compression report.
    pub fn new() -> Self {
        Self {
            original_params: 0,
            compressed_params: 0,
            pruned_params: 0,
            rank_reductions: Vec::new(),
            overall_compression_ratio: 1.0,
        }
    }

    /// Register a layer's original and compressed weight matrices.
    ///
    /// Updates parameter counts and compression ratio automatically.
    pub fn add_layer(&mut self, name: &str, original: &Array2<f32>, compressed: &Array2<f32>) {
        let orig_params = original.nrows() * original.ncols();
        let comp_params = compressed.nrows() * compressed.ncols();

        // Count zero entries in original as "pruned"
        let pruned = original.iter().filter(|&&x| x == 0.0).count();

        self.original_params += orig_params;
        self.compressed_params += comp_params;
        self.pruned_params += pruned;

        let orig_rank = original.nrows().min(original.ncols());
        let comp_rank = compressed.nrows().min(compressed.ncols());
        self.rank_reductions
            .push((name.to_string(), orig_rank, comp_rank));

        self.overall_compression_ratio = if self.compressed_params == 0 {
            f32::INFINITY
        } else {
            self.original_params as f32 / self.compressed_params as f32
        };
    }

    /// Generate a human-readable summary string.
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "=== Compression Report ===".to_string(),
            format!("Original parameters : {}", self.original_params),
            format!("Compressed parameters: {}", self.compressed_params),
            format!("Pruned parameters   : {}", self.pruned_params),
            format!(
                "Overall compression ratio: {:.3}x",
                self.overall_compression_ratio
            ),
            String::new(),
            "Layer rank reductions:".to_string(),
        ];
        for (name, orig_rank, comp_rank) in &self.rank_reductions {
            lines.push(format!("  {}: rank {} -> {}", name, orig_rank, comp_rank));
        }
        lines.join("\n")
    }
}

impl Default for CompressionReport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // MagnitudePruner tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_magnitude_pruner_basic() {
        let mut pruner = MagnitudePruner::new(0.5);
        // Values: 0.1, 0.2, 0.3, 0.4 are below threshold; 0.6, 0.7, 0.8, 0.9 are above
        let mut w =
            Array2::from_shape_vec((2, 4), vec![0.1_f32, 0.6, 0.2, 0.7, 0.3, 0.8, 0.4, 0.9])
                .expect("shape");

        let sparsity = pruner.prune_matrix(&mut w);
        // 4 out of 8 entries are zeroed
        assert!(sparsity > 0.0, "sparsity should be > 0");
        let zero_count = w.iter().filter(|&&x| x == 0.0).count();
        assert_eq!(zero_count, 4);
        assert!(pruner.pruned_count > 0);
        assert!(pruner.total_count > 0);
    }

    #[test]
    fn test_magnitude_pruner_zero_threshold() {
        let mut pruner = MagnitudePruner::new(0.0);
        let mut w = Array2::from_shape_vec((2, 2), vec![0.5_f32, 1.0, -0.3, 2.0]).expect("shape");

        let sparsity = pruner.prune_matrix(&mut w);
        // threshold = 0 means |w| < 0, which is never true → nothing pruned
        assert_eq!(sparsity, 0.0, "zero threshold should prune nothing");
        assert_eq!(pruner.pruned_count, 0);
    }

    // -----------------------------------------------------------------------
    // StructuredPruner tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_structured_pruner_row_mask_count() {
        let w = Array2::from_shape_fn((10, 4), |(i, j)| (i * 4 + j) as f32);
        let pruner = StructuredPruner::new(0.6);
        let mask = pruner.prune_rows(&w).expect("prune_rows failed");

        let keep_count = mask.iter().filter(|&&k| k).count();
        // ceil(0.6 * 10) = 6
        assert_eq!(keep_count, 6, "expected 6 kept rows, got {keep_count}");
        assert_eq!(mask.len(), 10);
    }

    #[test]
    fn test_structured_pruner_compress_reduces_rows() {
        let w = Array2::from_shape_fn((8, 3), |(i, j)| (i + j) as f32);
        let pruner = StructuredPruner::new(0.5);
        let compressed = pruner.compress_rows(&w).expect("compress_rows failed");

        assert!(
            compressed.nrows() < w.nrows(),
            "compressed rows {} should be < original {}",
            compressed.nrows(),
            w.nrows()
        );
        assert_eq!(compressed.ncols(), w.ncols());
    }

    // -----------------------------------------------------------------------
    // LowRankApprox tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_low_rank_approx_shapes() {
        let w = Array2::from_shape_fn((8, 6), |(i, j)| (i * j) as f32 * 0.1);
        let lra = LowRankApprox::compute(&w, 3, 50).expect("compute failed");

        assert_eq!(lra.u.nrows(), 8);
        assert_eq!(lra.u.ncols(), 3);
        assert_eq!(lra.vt.nrows(), 3);
        assert_eq!(lra.vt.ncols(), 6);
        assert_eq!(lra.singular_values.len(), 3);
    }

    #[test]
    fn test_low_rank_approx_reconstruction_error() {
        // Identity matrix: rank-4 approx should reconstruct perfectly
        let mut data = vec![0.0_f32; 16];
        for i in 0..4 {
            data[i * 4 + i] = 1.0;
        }
        let w = Array2::from_shape_vec((4, 4), data).expect("shape");

        let lra = LowRankApprox::compute(&w, 4, 100).expect("compute failed");
        assert!(
            lra.reconstruction_error < 0.01,
            "reconstruction_error {} should be < 0.01",
            lra.reconstruction_error
        );
    }

    #[test]
    fn test_low_rank_approx_compression_ratio() {
        // 10x10, rank 2 → (10*10) / (10*2 + 2*10) = 100/40 = 2.5 > 1
        let w = Array2::from_shape_fn((10, 10), |(i, j)| (i as f32).sin() + (j as f32).cos());
        let lra = LowRankApprox::compute(&w, 2, 20).expect("compute failed");

        assert!(
            lra.compression_ratio() > 1.0,
            "compression_ratio {} should be > 1.0",
            lra.compression_ratio()
        );
    }

    #[test]
    fn test_low_rank_forward_shape() {
        // w: 8x6, rank=3 → forward(x: 6) → output shape (8,)
        let w = Array2::from_shape_fn((8, 6), |(i, j)| ((i + j) as f32) * 0.1);
        let lra = LowRankApprox::compute(&w, 3, 30).expect("compute failed");

        let x = Array1::from_vec(vec![1.0_f32; 6]);
        let out = lra.forward(&x).expect("forward failed");
        assert_eq!(out.len(), 8, "expected output len 8, got {}", out.len());
    }

    #[test]
    fn test_distillation_loss_same_logits() {
        let logits = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]);
        let loss = distillation_loss(&logits, &logits, 1.0).expect("distillation_loss failed");
        // KL(p || p) = 0
        assert!(loss < 1e-5, "same logits should give loss ≈ 0, got {loss}");
    }
}
