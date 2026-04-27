//! Model Weight Pruning for kizzasi-model
//!
//! Provides magnitude-based unstructured pruning, structured channel pruning,
//! and gradual pruning schedule support for weight maps of the form
//! `HashMap<String, Vec<f32>>`.
//!
//! # Design
//!
//! The module is weight-map oriented: the primary input is a
//! `HashMap<String, Vec<f32>>` (tensor name → flat f32 values).
//! Pruning produces a mask (`Vec<bool>`) and zeroes out selected weights.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use kizzasi_model::prune::{prune_magnitude, ModelPruner, PruneConfig};
//! use std::collections::HashMap;
//!
//! let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
//! weights.insert("proj.weight".to_string(), vec![0.1, -0.2, 0.05, 0.9]);
//!
//! // Convenience: magnitude pruning at 50% sparsity
//! let (pruned, result) = prune_magnitude(&weights, 0.5).unwrap();
//! println!("Overall sparsity: {:.2}", result.overall_sparsity);
//! println!("Compression ratio: {:.2}x", result.compression_ratio());
//! ```

use crate::error::{ModelError, ModelResult};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Pruning method to apply to weight tensors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruneMethod {
    /// Zero out the smallest-magnitude weights (unstructured sparsity).
    MagnitudeUnstructured,
    /// Remove entire output channels/rows with the lowest L2 norm (structured).
    StructuredMagnitude,
    /// Random unstructured pruning, useful for ablation studies.
    RandomUnstructured,
}

/// Configuration for the [`ModelPruner`].
#[derive(Debug, Clone)]
pub struct PruneConfig {
    /// Which pruning algorithm to apply.
    pub method: PruneMethod,
    /// Target sparsity ratio in [0.0, 1.0].
    /// 0.0 → fully dense, 1.0 → all weights zeroed.
    pub sparsity: f32,
    /// Only prune tensors whose names start with one of these prefixes.
    /// An empty list means "all tensors are candidates".
    pub include_prefixes: Vec<String>,
    /// Never prune tensors whose names start with one of these prefixes.
    /// Takes precedence over `include_prefixes`.
    pub exclude_prefixes: Vec<String>,
    /// Skip tensors with fewer than this many elements.
    pub min_tensor_size: usize,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            method: PruneMethod::MagnitudeUnstructured,
            sparsity: 0.5,
            include_prefixes: vec![],
            exclude_prefixes: vec!["bias".to_string(), "norm".to_string()],
            min_tensor_size: 16,
        }
    }
}

/// Statistics for a single tensor after pruning.
#[derive(Debug, Clone)]
pub struct TensorPruneResult {
    /// Tensor name (key in the weight map).
    pub name: String,
    /// Number of non-zero elements before pruning.
    pub original_nonzero: usize,
    /// Number of non-zero elements after pruning.
    pub pruned_nonzero: usize,
    /// Actual fraction of zeros introduced: `(original_nonzero - pruned_nonzero) / total`.
    pub actual_sparsity: f32,
}

/// Aggregate statistics for an entire weight-map pruning run.
#[derive(Debug, Clone)]
pub struct PruneResult {
    /// Per-tensor breakdown.
    pub tensor_results: Vec<TensorPruneResult>,
    /// Total number of parameters across all tensors (pruned or not).
    pub total_params: usize,
    /// Number of parameters zeroed by pruning.
    pub pruned_params: usize,
    /// `pruned_params / total_params`.
    pub overall_sparsity: f32,
}

impl PruneResult {
    /// Compression ratio relative to a dense model.
    ///
    /// Returns 1.0 when nothing was pruned; larger values indicate more compression.
    pub fn compression_ratio(&self) -> f32 {
        let remaining = self.total_params.saturating_sub(self.pruned_params);
        if remaining == 0 {
            return f32::INFINITY;
        }
        self.total_params as f32 / remaining as f32
    }
}

/// Boolean mask over a flat weight tensor.
/// `true` → keep the weight; `false` → zero it out.
pub type PruneMask = Vec<bool>;

// ---------------------------------------------------------------------------
// ModelPruner
// ---------------------------------------------------------------------------

/// Applies a [`PruneConfig`] to weight maps.
pub struct ModelPruner {
    config: PruneConfig,
}

impl ModelPruner {
    /// Create a new pruner with the given configuration.
    pub fn new(config: PruneConfig) -> Self {
        Self { config }
    }

    /// Compute a pruning mask for a single flat tensor.
    ///
    /// Elements with magnitude ≤ the computed threshold are marked `false`
    /// (to be zeroed). The exact number zeroed is `floor(n * sparsity)`.
    pub fn compute_mask(&self, values: &[f32]) -> PruneMask {
        let n = values.len();
        let n_prune = (n as f32 * self.config.sparsity) as usize;
        if n_prune == 0 {
            return vec![true; n];
        }

        match self.config.method {
            PruneMethod::MagnitudeUnstructured | PruneMethod::StructuredMagnitude => {
                magnitude_mask(values, n_prune)
            }
            PruneMethod::RandomUnstructured => random_mask(n, n_prune),
        }
    }

    /// Zero out every element where the corresponding mask entry is `false`.
    pub fn apply_mask(values: &[f32], mask: &PruneMask) -> Vec<f32> {
        values
            .iter()
            .zip(mask.iter())
            .map(|(&v, &keep)| if keep { v } else { 0.0 })
            .collect()
    }

    /// Prune every eligible tensor in `weights`.
    ///
    /// Returns the pruned weight map together with aggregate statistics.
    /// Tensors excluded by prefix rules or below `min_tensor_size` are copied
    /// verbatim into the output.
    pub fn prune_weights(
        &self,
        weights: &HashMap<String, Vec<f32>>,
    ) -> ModelResult<(HashMap<String, Vec<f32>>, PruneResult)> {
        let mut pruned_map: HashMap<String, Vec<f32>> = HashMap::with_capacity(weights.len());
        let mut tensor_results: Vec<TensorPruneResult> = Vec::new();
        let mut total_params: usize = 0;
        let mut pruned_params: usize = 0;

        for (name, values) in weights {
            total_params += values.len();

            if !self.should_prune(name) || values.len() < self.config.min_tensor_size {
                // Pass through unchanged.
                pruned_map.insert(name.clone(), values.clone());
                continue;
            }

            let original_nonzero = values.iter().filter(|&&v| v != 0.0).count();
            let mask = self.compute_mask(values);
            let pruned_values = Self::apply_mask(values, &mask);
            let pruned_nonzero = pruned_values.iter().filter(|&&v| v != 0.0).count();

            let zeroed = original_nonzero.saturating_sub(pruned_nonzero);
            pruned_params += zeroed;

            let actual_sparsity = if values.is_empty() {
                0.0
            } else {
                zeroed as f32 / values.len() as f32
            };

            tensor_results.push(TensorPruneResult {
                name: name.clone(),
                original_nonzero,
                pruned_nonzero,
                actual_sparsity,
            });

            pruned_map.insert(name.clone(), pruned_values);
        }

        let overall_sparsity = if total_params == 0 {
            0.0
        } else {
            pruned_params as f32 / total_params as f32
        };

        let result = PruneResult {
            tensor_results,
            total_params,
            pruned_params,
            overall_sparsity,
        };

        Ok((pruned_map, result))
    }

    /// Cubic sparsity schedule used in gradual magnitude pruning.
    ///
    /// Returns the target sparsity at training `step`:
    ///
    /// ```text
    /// s(t) = s_final + (s_initial − s_final) × (1 − t / T)³
    /// ```
    ///
    /// where `t` is the number of pruning intervals elapsed since `start_step`
    /// and `T` is the total number of such intervals.
    ///
    /// - Before `start_step`: returns `initial_sparsity`.
    /// - After the schedule ends: returns `final_sparsity`.
    pub fn schedule_sparsity(
        initial_sparsity: f32,
        final_sparsity: f32,
        step: usize,
        total_steps: usize,
        start_step: usize,
        prune_freq: usize,
    ) -> f32 {
        if step < start_step {
            return initial_sparsity;
        }
        let freq = prune_freq.max(1);
        let t = ((step - start_step) / freq) as f32;
        let t_total = ((total_steps.saturating_sub(start_step)) / freq) as f32;
        if t_total == 0.0 || t >= t_total {
            return final_sparsity;
        }
        let frac = 1.0 - t / t_total;
        final_sparsity + (initial_sparsity - final_sparsity) * frac * frac * frac
    }

    /// Returns `true` when the tensor with the given `name` should be pruned.
    ///
    /// Exclusion always wins over inclusion.
    fn should_prune(&self, name: &str) -> bool {
        // Exclusion check first.
        for prefix in &self.config.exclude_prefixes {
            if name.starts_with(prefix.as_str()) || name.contains(prefix.as_str()) {
                return false;
            }
        }
        // Inclusion check: empty list ⟹ all remaining tensors are eligible.
        if self.config.include_prefixes.is_empty() {
            return true;
        }
        self.config
            .include_prefixes
            .iter()
            .any(|p| name.starts_with(p.as_str()))
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Prune `weights` using magnitude-based unstructured pruning at `sparsity`.
///
/// Uses the default [`PruneConfig`] with the specified sparsity value.
/// `min_tensor_size` is set to 0 so even small tensors are pruned.
pub fn prune_magnitude(
    weights: &HashMap<String, Vec<f32>>,
    sparsity: f32,
) -> ModelResult<(HashMap<String, Vec<f32>>, PruneResult)> {
    let config = PruneConfig {
        sparsity,
        min_tensor_size: 0,
        ..Default::default()
    };
    ModelPruner::new(config).prune_weights(weights)
}

/// Structured pruning: remove output channels with the lowest L2 norm.
///
/// `weight` is a flat row-major tensor with shape `[out_features, in_features, ...]`.
/// Only the first two dimensions are considered; higher dimensions (e.g., kernel
/// size) are folded into the per-row L2 norm calculation.
///
/// Returns `(pruned_weight, kept_channel_indices)` where `pruned_weight` contains
/// only the rows whose L2 norm was above the pruning threshold, concatenated in
/// their original order.
pub fn prune_structured_channels(
    weight: &[f32],
    shape: &[usize],
    sparsity: f32,
) -> ModelResult<(Vec<f32>, Vec<usize>)> {
    if shape.len() < 2 {
        return Err(ModelError::invalid_config(
            "prune_structured_channels: shape must have at least 2 dimensions [out, in, ...]",
        ));
    }

    let out_channels = shape[0];
    let row_size: usize = shape[1..].iter().product();

    if row_size == 0 {
        return Err(ModelError::invalid_config(
            "prune_structured_channels: inner dimensions must not be zero",
        ));
    }

    if weight.len() != out_channels * row_size {
        return Err(ModelError::invalid_config(format!(
            "prune_structured_channels: weight length {} does not match shape product {}",
            weight.len(),
            out_channels * row_size
        )));
    }

    let n_prune = (out_channels as f32 * sparsity) as usize;

    // Compute L2 norm for each output channel (row).
    let mut norms: Vec<(usize, f32)> = (0..out_channels)
        .map(|ch| {
            let start = ch * row_size;
            let end = start + row_size;
            let norm_sq: f32 = weight[start..end].iter().map(|v| v * v).sum();
            (ch, norm_sq.sqrt())
        })
        .collect();

    // Sort ascending by norm so the smallest-norm channels come first.
    norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // The n_prune channels with the smallest norms are pruned.
    let pruned_set: std::collections::HashSet<usize> = norms[..n_prune.min(out_channels)]
        .iter()
        .map(|&(i, _)| i)
        .collect();

    // Collect kept channel indices in original order.
    let kept_indices: Vec<usize> = (0..out_channels)
        .filter(|ch| !pruned_set.contains(ch))
        .collect();

    // Build the pruned weight tensor.
    let mut pruned_weight: Vec<f32> = Vec::with_capacity(kept_indices.len() * row_size);
    for &ch in &kept_indices {
        let start = ch * row_size;
        let end = start + row_size;
        pruned_weight.extend_from_slice(&weight[start..end]);
    }

    Ok((pruned_weight, kept_indices))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute a magnitude-based mask: prune exactly `n_prune` elements with the
/// smallest absolute values.
///
/// Ties at the threshold are broken by index order (lower indices pruned first)
/// to guarantee exactly `n_prune` elements are masked out.
fn magnitude_mask(values: &[f32], n_prune: usize) -> PruneMask {
    let n = values.len();
    // Build (magnitude, original_index) pairs and sort ascending.
    let mut indexed: Vec<(f32, usize)> = values
        .iter()
        .enumerate()
        .map(|(i, v)| (v.abs(), i))
        .collect();
    indexed.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    // Mark the n_prune smallest as pruned.
    let mut mask = vec![true; n];
    for &(_, idx) in &indexed[..n_prune.min(n)] {
        mask[idx] = false;
    }
    mask
}

/// Deterministic pseudo-random mask using a simple LCG.
/// Elements are chosen uniformly; no external RNG dependency.
fn random_mask(n: usize, n_prune: usize) -> PruneMask {
    // Build a pseudo-random permutation via LCG then mark the first n_prune as pruned.
    let mut indices: Vec<usize> = (0..n).collect();
    // LCG constants (Knuth)
    let mut state: u64 = 0x9e3779b97f4a7c15u64;
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
    let mut mask = vec![true; n];
    for &idx in &indices[..n_prune] {
        mask[idx] = false;
    }
    mask
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prune_config_default() {
        let cfg = PruneConfig::default();
        assert_eq!(cfg.sparsity, 0.5);
        assert!(cfg.exclude_prefixes.contains(&"bias".to_string()));
        assert!(cfg.exclude_prefixes.contains(&"norm".to_string()));
        assert_eq!(cfg.method, PruneMethod::MagnitudeUnstructured);
    }

    #[test]
    fn test_magnitude_mask_correct_sparsity() {
        let pruner = ModelPruner::new(PruneConfig {
            sparsity: 0.5,
            ..Default::default()
        });
        let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mask = pruner.compute_mask(&vals);
        let kept = mask.iter().filter(|&&b| b).count();
        // ~50% kept (4 out of 8, but threshold ties may shift by 1)
        assert!((3..=5).contains(&kept), "kept={kept}");
    }

    #[test]
    fn test_apply_mask_zeros_out() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        let result = ModelPruner::apply_mask(&vals, &mask);
        assert_eq!(result, vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_prune_weights_excludes_bias() {
        let mut weights = HashMap::new();
        weights.insert("layer.weight".to_string(), vec![1.0f32; 32]);
        weights.insert("layer.bias".to_string(), vec![1.0f32; 8]);
        let pruner = ModelPruner::new(PruneConfig {
            sparsity: 0.5,
            ..Default::default()
        });
        let (pruned, result) = pruner.prune_weights(&weights).unwrap();
        // bias should be untouched
        assert_eq!(pruned["layer.bias"], weights["layer.bias"]);
        // weight should have ~50% zeros
        let zeros = pruned["layer.weight"].iter().filter(|&&v| v == 0.0).count();
        assert!((12..=20).contains(&zeros), "zeros={zeros}");
        let _ = result;
    }

    #[test]
    fn test_prune_magnitude_convenience() {
        let mut weights = HashMap::new();
        weights.insert(
            "proj".to_string(),
            vec![0.1f32, 0.5, 0.01, 0.9, 0.2, 0.8, 0.05, 0.7],
        );
        let (pruned, result) = prune_magnitude(&weights, 0.5).unwrap();
        assert!(
            result.overall_sparsity >= 0.4 && result.overall_sparsity <= 0.6,
            "sparsity={}",
            result.overall_sparsity
        );
        assert!(
            result.compression_ratio() > 1.0,
            "ratio={}",
            result.compression_ratio()
        );
        let _ = pruned;
    }

    #[test]
    fn test_prune_result_overall_sparsity() {
        let mut weights = HashMap::new();
        weights.insert(
            "w".to_string(),
            (0..100).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
        );
        let pruner = ModelPruner::new(PruneConfig {
            sparsity: 0.7,
            ..Default::default()
        });
        let (_, result) = pruner.prune_weights(&weights).unwrap();
        assert!(
            result.overall_sparsity >= 0.65 && result.overall_sparsity <= 0.75,
            "sparsity={}",
            result.overall_sparsity
        );
    }

    #[test]
    fn test_schedule_sparsity_bounds() {
        // At step=0: t=0, frac=1 → s = final + (initial - final)*1³ = initial
        let s0 = ModelPruner::schedule_sparsity(0.0, 0.9, 0, 100, 0, 1);
        assert!(
            (s0 - 0.0).abs() < 0.01,
            "s0 should be initial=0.0, got {s0}"
        );
        // At step=total_steps: returns final
        let s_end = ModelPruner::schedule_sparsity(0.0, 0.9, 100, 100, 0, 1);
        assert!(
            (s_end - 0.9).abs() < 0.01,
            "s_end should be 0.9, got {s_end}"
        );
    }

    #[test]
    fn test_structured_channel_pruning() {
        // 4 output channels, 3 input features each
        let weight: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let shape = vec![4, 3];
        // Prune 50% of channels (2 out of 4)
        let (pruned, kept) = prune_structured_channels(&weight, &shape, 0.5).unwrap();
        assert_eq!(kept.len(), 2, "kept={kept:?}");
        assert_eq!(pruned.len(), 6, "pruned.len={}", pruned.len()); // 2 channels × 3 features
    }

    #[test]
    fn test_prune_zero_sparsity_noop() {
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), vec![1.0f32, 2.0, 3.0]);
        let (pruned, result) = prune_magnitude(&weights, 0.0).unwrap();
        assert_eq!(pruned["w"], weights["w"]); // no change
        assert_eq!(result.pruned_params, 0);
    }

    #[test]
    fn test_schedule_sparsity_before_start() {
        // Before start_step, must return initial_sparsity
        let s = ModelPruner::schedule_sparsity(0.1, 0.9, 5, 100, 10, 1);
        assert!((s - 0.1).abs() < 1e-6, "s={s}");
    }

    #[test]
    fn test_prune_method_random() {
        let pruner = ModelPruner::new(PruneConfig {
            method: PruneMethod::RandomUnstructured,
            sparsity: 0.5,
            ..Default::default()
        });
        let vals: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let mask = pruner.compute_mask(&vals);
        let kept = mask.iter().filter(|&&b| b).count();
        assert_eq!(kept, 50, "kept={kept}");
    }

    #[test]
    fn test_compression_ratio_no_pruning() {
        let result = PruneResult {
            tensor_results: vec![],
            total_params: 100,
            pruned_params: 0,
            overall_sparsity: 0.0,
        };
        assert!((result.compression_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_prune_structured_bad_shape() {
        let weight = vec![1.0f32; 12];
        let result = prune_structured_channels(&weight, &[4], 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_should_prune_include_exclude() {
        let pruner = ModelPruner::new(PruneConfig {
            include_prefixes: vec!["proj".to_string()],
            exclude_prefixes: vec!["proj.bias".to_string()],
            ..Default::default()
        });
        assert!(pruner.should_prune("proj.weight"));
        assert!(!pruner.should_prune("proj.bias"));
        assert!(!pruner.should_prune("embed.weight")); // not in include list
    }
}
