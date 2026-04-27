//! Adaptive Early Exit for kizzasi-model
//!
//! Implements adaptive computation where inference can exit at an intermediate layer
//! when the model is sufficiently confident, reducing compute for "easy" inputs.
//!
//! # Criteria
//!
//! Multiple exit criteria are supported:
//! - **Confidence**: Exit when max softmax probability exceeds threshold
//! - **Entropy**: Exit when prediction entropy falls below threshold
//! - **Variance**: Exit when output variance falls below threshold
//! - **Fixed**: Always exit at a predetermined layer
//!
//! # References
//!
//! - Schwartz et al., "The Right Tool for the Job: Matching Model and Instance Complexities" (2020)
//! - Schuster et al., "Confident Adaptive Language Modeling" (2022)

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::Array1;

/// Criterion used to decide whether to exit early at a given layer
#[derive(Debug, Clone, PartialEq)]
pub enum ExitCriterion {
    /// Exit when max softmax probability exceeds threshold
    Confidence {
        /// Minimum confidence (max softmax prob) to trigger exit
        threshold: f32,
    },
    /// Exit when prediction entropy falls below threshold
    Entropy {
        /// Maximum entropy to trigger exit (lower = more certain)
        threshold: f32,
    },
    /// Exit when output variance falls below threshold
    Variance {
        /// Maximum variance to trigger exit
        threshold: f32,
    },
    /// Always exit at a fixed layer index
    EarlyFixed {
        /// Layer index at which to always exit
        exit_layer: usize,
    },
}

/// Configuration for adaptive early exit
#[derive(Debug, Clone)]
pub struct EarlyExitConfig {
    /// The criterion used to decide whether to exit
    pub criterion: ExitCriterion,
    /// Minimum number of layers to always compute (never exit before this)
    pub min_layers: usize,
    /// Maximum number of layers (always exit at or before this)
    pub max_layers: usize,
}

impl EarlyExitConfig {
    /// Create a new early exit configuration
    ///
    /// # Errors
    ///
    /// Returns `ModelError::InvalidConfig` if min_layers > max_layers or max_layers == 0
    pub fn new(
        criterion: ExitCriterion,
        min_layers: usize,
        max_layers: usize,
    ) -> ModelResult<Self> {
        if max_layers == 0 {
            return Err(ModelError::invalid_config("max_layers must be at least 1"));
        }
        if min_layers > max_layers {
            return Err(ModelError::invalid_config(format!(
                "min_layers ({}) must not exceed max_layers ({})",
                min_layers, max_layers
            )));
        }

        // Validate criterion-specific constraints
        match &criterion {
            ExitCriterion::Confidence { threshold } => {
                if *threshold <= 0.0 || *threshold > 1.0 {
                    return Err(ModelError::invalid_config(
                        "confidence threshold must be in (0, 1]",
                    ));
                }
            }
            ExitCriterion::Entropy { threshold } => {
                if *threshold < 0.0 {
                    return Err(ModelError::invalid_config(
                        "entropy threshold must be non-negative",
                    ));
                }
            }
            ExitCriterion::Variance { threshold } => {
                if *threshold < 0.0 {
                    return Err(ModelError::invalid_config(
                        "variance threshold must be non-negative",
                    ));
                }
            }
            ExitCriterion::EarlyFixed { exit_layer } => {
                if *exit_layer >= max_layers {
                    return Err(ModelError::invalid_config(format!(
                        "exit_layer ({}) must be < max_layers ({})",
                        exit_layer, max_layers
                    )));
                }
            }
        }

        Ok(Self {
            criterion,
            min_layers,
            max_layers,
        })
    }
}

/// Statistics tracking for adaptive early exit across multiple inference steps
#[derive(Debug, Clone)]
pub struct ExitStats {
    /// Total number of inference steps processed
    pub total_steps: usize,
    /// Histogram: `exits_per_layer[i]` = count of exits at layer i
    pub exits_per_layer: Vec<usize>,
    /// Average exit layer across all steps
    pub avg_exit_layer: f32,
    /// Percentage of compute saved compared to always running all layers
    pub compute_savings_pct: f32,
}

impl ExitStats {
    /// Create new stats for a model with the given number of layers
    fn new(max_layers: usize) -> Self {
        Self {
            total_steps: 0,
            exits_per_layer: vec![0; max_layers],
            avg_exit_layer: 0.0,
            compute_savings_pct: 0.0,
        }
    }

    /// Record an exit at a given layer and recompute derived statistics
    fn record_exit(&mut self, layer_idx: usize) {
        self.total_steps += 1;

        if layer_idx < self.exits_per_layer.len() {
            self.exits_per_layer[layer_idx] += 1;
        }

        // Recompute average exit layer
        let max_layers = self.exits_per_layer.len();
        let total_steps = self.total_steps as f32;

        let weighted_sum: f32 = self
            .exits_per_layer
            .iter()
            .enumerate()
            .map(|(layer, &count)| layer as f32 * count as f32)
            .sum();

        self.avg_exit_layer = if total_steps > 0.0 {
            weighted_sum / total_steps
        } else {
            0.0
        };

        // Compute savings: fraction of layers saved compared to always using max_layers
        // If avg exit layer is L and max is M, savings = (M - 1 - L) / (M - 1) * 100
        // (M-1 because layers are 0-indexed, so running all layers means exiting at M-1)
        if max_layers > 1 {
            let max_layer_idx = (max_layers - 1) as f32;
            self.compute_savings_pct =
                ((max_layer_idx - self.avg_exit_layer) / max_layer_idx * 100.0).max(0.0);
        } else {
            self.compute_savings_pct = 0.0;
        }
    }
}

/// Adaptive computation engine that decides when to exit early during inference
///
/// Maintains per-inference state (layer outputs) and cumulative statistics.
/// Call `maybe_exit` for each layer during a forward pass. When it returns
/// `Some(output)`, stop processing further layers.
#[derive(Debug, Clone)]
pub struct AdaptiveComputation {
    /// Configuration for this instance
    config: EarlyExitConfig,
    /// Cumulative statistics across all inference steps
    stats: ExitStats,
    /// Layer outputs collected during the current inference step
    layer_outputs: Vec<Array1<f32>>,
}

impl AdaptiveComputation {
    /// Create a new adaptive computation engine
    pub fn new(config: EarlyExitConfig) -> Self {
        let stats = ExitStats::new(config.max_layers);
        Self {
            config,
            stats,
            layer_outputs: Vec::new(),
        }
    }

    /// Check whether to exit at the current layer
    ///
    /// Call this after each layer's forward pass. The function stores the output
    /// and checks the exit criterion. Returns `Some(output)` if the model should
    /// exit at this layer, `None` if it should continue to the next layer.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Zero-based index of the current layer
    /// * `output` - The output of the current layer
    ///
    /// # Errors
    ///
    /// Returns `ModelError::IndexOutOfBounds` if layer_idx >= max_layers
    pub fn maybe_exit(
        &mut self,
        layer_idx: usize,
        output: Array1<f32>,
    ) -> ModelResult<Option<Array1<f32>>> {
        if layer_idx >= self.config.max_layers {
            return Err(ModelError::IndexOutOfBounds {
                index: layer_idx,
                limit: self.config.max_layers,
                context: "layer_idx exceeds max_layers".into(),
            });
        }

        // Store layer output
        if layer_idx == 0 {
            self.layer_outputs.clear();
        }
        self.layer_outputs.push(output.clone());

        // Never exit before min_layers
        if layer_idx < self.config.min_layers {
            return Ok(None);
        }

        // Always exit at the last layer
        if layer_idx >= self.config.max_layers - 1 {
            self.stats.record_exit(layer_idx);
            return Ok(Some(output));
        }

        // Check criterion
        let should_exit = match &self.config.criterion {
            ExitCriterion::Confidence { threshold } => {
                let confidence = compute_confidence(&output)?;
                confidence > *threshold
            }
            ExitCriterion::Entropy { threshold } => {
                let entropy = compute_entropy(&output)?;
                entropy < *threshold
            }
            ExitCriterion::Variance { threshold } => {
                let variance = compute_variance(&output);
                variance < *threshold
            }
            ExitCriterion::EarlyFixed { exit_layer } => layer_idx >= *exit_layer,
        };

        if should_exit {
            self.stats.record_exit(layer_idx);
            Ok(Some(output))
        } else {
            Ok(None)
        }
    }

    /// Get the cumulative exit statistics
    pub fn stats(&self) -> &ExitStats {
        &self.stats
    }

    /// Reset cumulative statistics (e.g., between evaluation runs)
    pub fn reset_stats(&mut self) {
        self.stats = ExitStats::new(self.config.max_layers);
    }

    /// Get the layer outputs collected during the current inference step
    pub fn layer_outputs(&self) -> &[Array1<f32>] {
        &self.layer_outputs
    }

    /// Get the configuration
    pub fn config(&self) -> &EarlyExitConfig {
        &self.config
    }
}

/// Compute confidence as the maximum softmax probability
///
/// Applies softmax to the output and returns the maximum probability value.
/// Higher confidence means the model is more certain about its prediction.
fn compute_confidence(output: &Array1<f32>) -> ModelResult<f32> {
    let probs = stable_softmax(output)?;
    let max_prob = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Ok(max_prob)
}

/// Compute Shannon entropy of the softmax distribution
///
/// H = -sum(p * ln(p)) for p > 0
/// Lower entropy means the distribution is more peaked (model is more certain).
fn compute_entropy(output: &Array1<f32>) -> ModelResult<f32> {
    let probs = stable_softmax(output)?;
    let mut entropy = 0.0_f32;

    for &p in probs.iter() {
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }

    Ok(entropy)
}

/// Compute variance of the raw output values
///
/// Uses the standard variance formula: E[x^2] - E[x]^2
fn compute_variance(output: &Array1<f32>) -> f32 {
    let n = output.len();
    if n == 0 {
        return 0.0;
    }

    let n_f = n as f32;
    let mean: f32 = output.iter().sum::<f32>() / n_f;
    let variance: f32 = output.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n_f;
    variance
}

/// Numerically stable softmax
///
/// Subtracts the maximum value before computing exponentials to prevent overflow.
fn stable_softmax(logits: &Array1<f32>) -> ModelResult<Array1<f32>> {
    let n = logits.len();
    if n == 0 {
        return Err(ModelError::invalid_config(
            "cannot compute softmax of empty array",
        ));
    }

    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if !max_val.is_finite() {
        return Err(ModelError::numerical_instability(
            "softmax",
            "logits contain non-finite values",
        ));
    }

    let mut probs = logits.mapv(|x| (x - max_val).exp());
    let sum: f32 = probs.iter().sum();

    if sum < 1e-30 {
        return Err(ModelError::numerical_instability(
            "softmax",
            "sum of exponentials is near zero",
        ));
    }

    probs /= sum;
    Ok(probs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_early_exit_confidence_triggers() {
        // Create output with one very dominant logit -> high confidence
        let config = EarlyExitConfig::new(
            ExitCriterion::Confidence { threshold: 0.5 },
            0,  // min_layers = 0 (can exit immediately)
            10, // max_layers
        )
        .expect("valid config");

        let mut ac = AdaptiveComputation::new(config);

        // Very peaked distribution -> high confidence
        let peaked_output = array![10.0, 0.0, 0.0, 0.0, 0.0];
        let result = ac.maybe_exit(1, peaked_output).expect("should succeed");

        assert!(
            result.is_some(),
            "should exit early with high-confidence output"
        );
    }

    #[test]
    fn test_early_exit_entropy_triggers() {
        let config = EarlyExitConfig::new(ExitCriterion::Entropy { threshold: 0.5 }, 0, 10)
            .expect("valid config");

        let mut ac = AdaptiveComputation::new(config);

        // Very peaked distribution -> low entropy
        let peaked_output = array![20.0, 0.0, 0.0, 0.0, 0.0];
        let result = ac.maybe_exit(1, peaked_output).expect("should succeed");

        assert!(
            result.is_some(),
            "should exit early with low-entropy output"
        );
    }

    #[test]
    fn test_early_exit_min_layers_respected() {
        let config = EarlyExitConfig::new(
            ExitCriterion::Confidence { threshold: 0.01 }, // very low threshold
            5, // min_layers = 5 (must compute at least 5 layers)
            10,
        )
        .expect("valid config");

        let mut ac = AdaptiveComputation::new(config);

        // Even with very confident output, should not exit before min_layers
        let peaked_output = array![100.0, 0.0, 0.0, 0.0, 0.0];

        for layer in 0..5 {
            let result = ac
                .maybe_exit(layer, peaked_output.clone())
                .expect("should succeed");
            assert!(
                result.is_none(),
                "should NOT exit at layer {} (before min_layers=5)",
                layer
            );
        }

        // At layer 5, should exit
        let result = ac.maybe_exit(5, peaked_output).expect("should succeed");
        assert!(result.is_some(), "should exit at layer 5 (>= min_layers)");
    }

    #[test]
    fn test_early_exit_stats_accumulate() {
        let config = EarlyExitConfig::new(ExitCriterion::EarlyFixed { exit_layer: 3 }, 0, 10)
            .expect("valid config");

        let mut ac = AdaptiveComputation::new(config);

        let output = array![1.0, 2.0, 3.0];

        // Run 5 inference steps, each going through layers 0..=3
        for _step in 0..5 {
            for layer in 0..=3 {
                let result = ac
                    .maybe_exit(layer, output.clone())
                    .expect("should succeed");
                if result.is_some() {
                    break;
                }
            }
        }

        let stats = ac.stats();
        assert_eq!(stats.total_steps, 5, "should have 5 total steps");

        // All exits should be at layer 3
        assert_eq!(stats.exits_per_layer[3], 5);

        // Sum of exits_per_layer should equal total_steps
        let sum: usize = stats.exits_per_layer.iter().sum();
        assert_eq!(
            sum, stats.total_steps,
            "exits_per_layer sum ({}) should equal total_steps ({})",
            sum, stats.total_steps
        );
    }

    #[test]
    fn test_early_exit_fixed() {
        let config = EarlyExitConfig::new(ExitCriterion::EarlyFixed { exit_layer: 2 }, 0, 8)
            .expect("valid config");

        let mut ac = AdaptiveComputation::new(config);
        let output = array![1.0, 1.0, 1.0, 1.0];

        // Layers before exit_layer should not exit
        let r0 = ac.maybe_exit(0, output.clone()).expect("ok");
        let r1 = ac.maybe_exit(1, output.clone()).expect("ok");
        assert!(r0.is_none(), "should not exit at layer 0");
        assert!(r1.is_none(), "should not exit at layer 1");

        // At exit_layer=2, should exit
        let r2 = ac.maybe_exit(2, output.clone()).expect("ok");
        assert!(r2.is_some(), "should exit at layer 2 (EarlyFixed)");
    }

    #[test]
    fn test_early_exit_max_layers_forces_exit() {
        // Even with very high threshold (would never trigger), exits at max_layers - 1
        let config = EarlyExitConfig::new(ExitCriterion::Confidence { threshold: 0.9999 }, 0, 4)
            .expect("valid config");

        let mut ac = AdaptiveComputation::new(config);

        // Uniform distribution -> low confidence
        let uniform_output = array![1.0, 1.0, 1.0, 1.0];

        for layer in 0..3 {
            let result = ac.maybe_exit(layer, uniform_output.clone()).expect("ok");
            assert!(result.is_none(), "should not exit at layer {}", layer);
        }

        // At layer 3 (max_layers - 1), must exit regardless
        let result = ac.maybe_exit(3, uniform_output).expect("ok");
        assert!(result.is_some(), "must exit at max_layers - 1");
    }

    #[test]
    fn test_early_exit_variance_criterion() {
        let config = EarlyExitConfig::new(ExitCriterion::Variance { threshold: 0.01 }, 0, 10)
            .expect("valid config");

        let mut ac = AdaptiveComputation::new(config);

        // Very low variance output -> should trigger exit
        let low_var_output = array![1.0, 1.001, 0.999, 1.0, 1.0];
        let result = ac.maybe_exit(1, low_var_output).expect("ok");
        assert!(result.is_some(), "low variance should trigger exit");
    }

    #[test]
    fn test_early_exit_config_validation() {
        // max_layers = 0
        assert!(EarlyExitConfig::new(ExitCriterion::Confidence { threshold: 0.5 }, 0, 0,).is_err());

        // min > max
        assert!(
            EarlyExitConfig::new(ExitCriterion::Confidence { threshold: 0.5 }, 10, 5,).is_err()
        );

        // exit_layer >= max_layers
        assert!(
            EarlyExitConfig::new(ExitCriterion::EarlyFixed { exit_layer: 10 }, 0, 10,).is_err()
        );

        // invalid confidence threshold
        assert!(
            EarlyExitConfig::new(ExitCriterion::Confidence { threshold: 0.0 }, 0, 10,).is_err()
        );
    }

    #[test]
    fn test_compute_savings_percentage() {
        let config = EarlyExitConfig::new(ExitCriterion::EarlyFixed { exit_layer: 2 }, 0, 10)
            .expect("valid config");

        let mut ac = AdaptiveComputation::new(config);
        let output = array![1.0, 2.0, 3.0];

        // Exit at layer 2 of 10 layers
        for layer in 0..=2 {
            let _ = ac.maybe_exit(layer, output.clone()).expect("ok");
        }

        let stats = ac.stats();
        // avg_exit_layer = 2, max_layers = 10
        // savings = (9 - 2) / 9 * 100 = 77.8%
        assert!(
            stats.compute_savings_pct > 70.0,
            "savings {} should be > 70%",
            stats.compute_savings_pct
        );
    }
}
