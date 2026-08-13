//! Automatic performance tuning and workload profiling
//!
//! This module provides sophisticated auto-tuning capabilities:
//! - Workload profiling and analysis
//! - Automatic parameter optimization based on performance metrics
//! - Adaptive batch size selection
//! - Dynamic model selection
//! - Latency-driven configuration adjustments
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi::autotuning::{AutoTuner, TuningConfig, WorkloadProfile};
//!
//! let tuner = AutoTuner::new(TuningConfig::default());
//! let optimized_config = tuner.tune_for_latency(target_latency_ms)?;
//! ```

use crate::error::{KizzasiError, KizzasiResult};
use crate::predictor::{Kizzasi, KizzasiBuilder};
use kizzasi_core::ModelType;
use scirs2_core::ndarray::Array1;
use std::collections::VecDeque;
use std::time::Instant;

/// Configuration for auto-tuning
#[derive(Debug, Clone)]
pub struct TuningConfig {
    /// Number of warmup iterations before profiling
    pub warmup_iterations: usize,

    /// Number of profiling iterations
    pub profiling_iterations: usize,

    /// Target latency in microseconds (for latency-driven tuning)
    pub target_latency_us: Option<u64>,

    /// Target throughput in predictions/sec (for throughput-driven tuning)
    pub target_throughput: Option<f64>,

    /// Maximum memory usage in bytes
    pub max_memory_bytes: Option<usize>,

    /// Enable aggressive tuning (may sacrifice accuracy for speed)
    pub aggressive_tuning: bool,

    /// Enable conservative tuning (prioritize accuracy over speed)
    pub conservative_tuning: bool,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            profiling_iterations: 100,
            target_latency_us: None,
            target_throughput: None,
            max_memory_bytes: None,
            aggressive_tuning: false,
            conservative_tuning: false,
        }
    }
}

impl TuningConfig {
    /// Create a new tuning configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set warmup iterations
    pub fn with_warmup(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    /// Set profiling iterations
    pub fn with_profiling(mut self, iterations: usize) -> Self {
        self.profiling_iterations = iterations;
        self
    }

    /// Set target latency
    pub fn with_target_latency_us(mut self, latency_us: u64) -> Self {
        self.target_latency_us = Some(latency_us);
        self
    }

    /// Set target throughput
    pub fn with_target_throughput(mut self, throughput: f64) -> Self {
        self.target_throughput = Some(throughput);
        self
    }

    /// Set maximum memory usage
    pub fn with_max_memory(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = Some(bytes);
        self
    }

    /// Enable aggressive tuning
    pub fn aggressive(mut self) -> Self {
        self.aggressive_tuning = true;
        self.conservative_tuning = false;
        self
    }

    /// Enable conservative tuning
    pub fn conservative(mut self) -> Self {
        self.conservative_tuning = true;
        self.aggressive_tuning = false;
        self
    }
}

/// Workload characteristics discovered through profiling
#[derive(Debug, Clone)]
pub struct WorkloadProfile {
    /// Average input dimension
    pub avg_input_dim: usize,

    /// Average output dimension
    pub avg_output_dim: usize,

    /// Average prediction latency in microseconds
    pub avg_latency_us: u64,

    /// Latency standard deviation
    pub latency_std_dev_us: f64,

    /// p50, p95, p99 latencies
    pub p50_latency_us: u64,
    pub p95_latency_us: u64,
    pub p99_latency_us: u64,

    /// Throughput in predictions per second
    pub throughput_pps: f64,

    /// Average prediction error (if available)
    pub avg_error: Option<f64>,

    /// Estimated memory usage in bytes
    pub estimated_memory_bytes: usize,
}

/// Recommendation for model configuration
#[derive(Debug, Clone)]
pub struct TuningRecommendation {
    /// Recommended model type
    pub model_type: ModelType,

    /// Recommended hidden dimension
    pub hidden_dim: usize,

    /// Recommended number of layers
    pub num_layers: usize,

    /// Recommended context window
    pub context_window: usize,

    /// Expected performance characteristics
    pub expected_latency_us: u64,
    pub expected_throughput_pps: f64,
    pub expected_memory_bytes: usize,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// Reasoning for this recommendation
    pub reasoning: String,
}

/// Auto-tuner for performance optimization
pub struct AutoTuner {
    config: TuningConfig,
    profiling_history: VecDeque<WorkloadProfile>,
}

impl AutoTuner {
    /// Create a new auto-tuner
    pub fn new(config: TuningConfig) -> Self {
        Self {
            config,
            profiling_history: VecDeque::with_capacity(10),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(TuningConfig::default())
    }

    /// Profile a predictor's performance
    ///
    /// The predictor's recurrent state is captured before the synthetic
    /// warmup/profiling samples are pushed through it and restored afterwards,
    /// so profiling a *live* predictor (as [`AdaptiveTuner`] does) does not
    /// poison the sequence the caller is streaming.
    ///
    /// One exception, inherited from `kizzasi-model`: for
    /// [`kizzasi_core::ModelType::S4`] the snapshot covers the SSM state but
    /// not each layer's 3-tap causal-convolution history, which `kizzasi-model`
    /// does not expose. Profiling an S4-backed predictor therefore leaves up
    /// to two synthetic frames in that short history. This matters here
    /// because [`AutoTuner::build_for_latency`] deliberately recommends S4 for
    /// the tightest latency budgets.
    pub fn profile(
        &mut self,
        predictor: &mut Kizzasi,
        input_dim: usize,
    ) -> KizzasiResult<WorkloadProfile> {
        if self.config.profiling_iterations == 0 {
            return Err(KizzasiError::config(
                "profiling_iterations must be > 0 to profile a predictor",
            ));
        }
        if input_dim == 0 {
            return Err(KizzasiError::config("input_dim must be > 0 to profile"));
        }

        let mut latencies = Vec::with_capacity(self.config.profiling_iterations);

        // Snapshot the live hidden state so the synthetic samples below can be
        // rolled back; without this every `AdaptiveTuner::adapt` would inject
        // `warmup + profiling` all-ones samples into the user's stream.
        let saved_state = predictor.snapshot_state();

        let test_input = Array1::from_vec(vec![1.0; input_dim]);
        let profiled = (|| -> KizzasiResult<()> {
            // Warmup phase
            for _ in 0..self.config.warmup_iterations {
                let _ = predictor.step(&test_input)?;
            }

            // Profiling phase
            for _ in 0..self.config.profiling_iterations {
                let start = Instant::now();
                let _ = predictor.step(&test_input)?;
                let elapsed = start.elapsed().as_micros() as u64;
                latencies.push(elapsed);
            }
            Ok(())
        })();

        // Restore the caller's state whether or not profiling succeeded.
        let restored = predictor.restore_state(saved_state);
        profiled?;
        restored?;

        // Calculate statistics. `latencies` is non-empty: profiling_iterations
        // was validated above and every iteration pushes exactly one sample.
        let sample_count = latencies.len().max(1) as u64;
        let avg_latency_us = latencies.iter().sum::<u64>() / sample_count;

        let variance: f64 = latencies
            .iter()
            .map(|&lat| {
                let diff = lat as f64 - avg_latency_us as f64;
                diff * diff
            })
            .sum::<f64>()
            / sample_count as f64;
        let latency_std_dev_us = variance.sqrt();

        // Calculate percentiles
        let mut sorted_latencies = latencies.clone();
        sorted_latencies.sort_unstable();
        let last_index = sorted_latencies.len().saturating_sub(1);
        let percentile = |fraction: f64| -> u64 {
            let index = ((sorted_latencies.len() as f64 * fraction) as usize).min(last_index);
            sorted_latencies.get(index).copied().unwrap_or(0)
        };
        let p50_latency_us = percentile(0.5);
        let p95_latency_us = percentile(0.95);
        let p99_latency_us = percentile(0.99);

        // A sub-microsecond step measures as 0 µs; clamp so throughput stays
        // finite instead of reporting +inf predictions/sec.
        let throughput_pps = 1_000_000.0 / (avg_latency_us.max(1) as f64);

        // Estimate memory usage (rough approximation)
        let hidden_dim = predictor.hidden_dim();
        let num_layers = predictor.num_layers();
        let state_dim = predictor.state_dim();
        let estimated_memory_bytes = (hidden_dim * state_dim * num_layers * 4) + // State
            (hidden_dim * hidden_dim * num_layers * 4 * 4); // Weights (A, B, C, D)

        let profile = WorkloadProfile {
            avg_input_dim: input_dim,
            avg_output_dim: predictor.output_dim(),
            avg_latency_us,
            latency_std_dev_us,
            p50_latency_us,
            p95_latency_us,
            p99_latency_us,
            throughput_pps,
            avg_error: None,
            estimated_memory_bytes,
        };

        // Store in history
        self.profiling_history.push_back(profile.clone());
        if self.profiling_history.len() > 10 {
            self.profiling_history.pop_front();
        }

        Ok(profile)
    }

    /// Generate tuning recommendations for latency target
    pub fn recommend_for_latency(
        &self,
        input_dim: usize,
        output_dim: usize,
        target_latency_us: u64,
    ) -> TuningRecommendation {
        // Decision tree for model selection based on latency requirements
        let (model_type, hidden_dim, num_layers, context_window, reasoning) =
            if target_latency_us < 100 {
                // Ultra low latency: < 100 microseconds
                (
                    ModelType::S4,
                    32,
                    1,
                    64,
                    "Ultra-low latency requirement - using minimal S4 model".to_string(),
                )
            } else if target_latency_us < 500 {
                // Low latency: 100-500 microseconds
                (
                    ModelType::Mamba2,
                    64,
                    2,
                    128,
                    "Low latency requirement - using compact Mamba2 model".to_string(),
                )
            } else if target_latency_us < 2000 {
                // Medium latency: 500-2000 microseconds
                (
                    ModelType::Mamba2,
                    128,
                    4,
                    256,
                    "Medium latency budget - using balanced Mamba2 model".to_string(),
                )
            } else if target_latency_us < 10000 {
                // High latency: 2-10 milliseconds
                (
                    ModelType::Rwkv,
                    256,
                    6,
                    512,
                    "High latency budget - using larger RWKV model for better accuracy".to_string(),
                )
            } else {
                // Very high latency: > 10 milliseconds
                (
                    ModelType::Rwkv,
                    512,
                    8,
                    1024,
                    "Very high latency budget - using large model for maximum accuracy".to_string(),
                )
            };

        // Adjust for conservative/aggressive tuning
        let (hidden_dim, num_layers) = if self.config.aggressive_tuning {
            let adjusted_layers = if num_layers > 1 { num_layers - 1 } else { 1 };
            (hidden_dim / 2, adjusted_layers)
        } else if self.config.conservative_tuning {
            (hidden_dim * 2, num_layers + 2)
        } else {
            (hidden_dim, num_layers)
        };

        // Estimate expected performance
        let base_latency: usize = match model_type {
            ModelType::S4 => 50,
            ModelType::Mamba => 100,
            ModelType::Mamba2 => 80,
            ModelType::Rwkv => 150,
        };

        // Computed in f64: `hidden_dim / 64` in integer arithmetic collapses
        // the whole product to 0 for every model narrower than 64 channels
        // (which is exactly what the ultra-low-latency and aggressive branches
        // above produce), yielding a "0 µs, infinite throughput" recommendation.
        let expected_latency_us = ((base_latency as f64)
            * (hidden_dim as f64 / 64.0)
            * num_layers as f64
            * ((input_dim + output_dim) as f64 / input_dim.max(1) as f64))
            .max(1.0) as u64;

        let expected_throughput_pps = 1_000_000.0 / expected_latency_us.max(1) as f64;

        let expected_memory_bytes =
            (hidden_dim * 64 * num_layers * 4) + (hidden_dim * hidden_dim * num_layers * 16);

        // Calculate confidence based on how well we can meet the target
        let latency_ratio = expected_latency_us as f64 / target_latency_us.max(1) as f64;
        let confidence = if latency_ratio <= 0.8 {
            0.95 // Very confident we can meet target
        } else if latency_ratio <= 1.0 {
            0.85 // Confident
        } else if latency_ratio <= 1.2 {
            0.7 // Somewhat confident
        } else {
            0.5 // Less confident
        };

        TuningRecommendation {
            model_type,
            hidden_dim,
            num_layers,
            context_window,
            expected_latency_us,
            expected_throughput_pps,
            expected_memory_bytes,
            confidence,
            reasoning,
        }
    }

    /// Generate tuning recommendations for throughput target
    pub fn recommend_for_throughput(
        &self,
        input_dim: usize,
        output_dim: usize,
        target_throughput_pps: f64,
    ) -> TuningRecommendation {
        // Convert throughput to latency target. A non-positive or non-finite
        // target would otherwise produce a nonsensical (or saturating) budget.
        let target_latency_us = if target_throughput_pps.is_finite() && target_throughput_pps > 0.0
        {
            (1_000_000.0 / target_throughput_pps).max(1.0) as u64
        } else {
            1
        };
        self.recommend_for_latency(input_dim, output_dim, target_latency_us)
    }

    /// Generate balanced recommendations
    ///
    /// Uses `TuningConfig::target_latency_us` when one was configured, falling
    /// back to `TuningConfig::target_throughput` and finally to a 1 ms budget,
    /// then caps the recommendation with `TuningConfig::max_memory_bytes`.
    pub fn recommend_balanced(&self, input_dim: usize, output_dim: usize) -> TuningRecommendation {
        let mut rec = match (self.config.target_latency_us, self.config.target_throughput) {
            (Some(latency_us), _) => self.recommend_for_latency(input_dim, output_dim, latency_us),
            (None, Some(throughput)) => {
                self.recommend_for_throughput(input_dim, output_dim, throughput)
            }
            (None, None) => self.recommend_for_latency(input_dim, output_dim, 1000),
        };
        rec.reasoning = "Balanced configuration for general-purpose use".to_string();
        self.apply_memory_cap(&mut rec);
        rec
    }

    /// Shrink a recommendation until its estimated footprint fits
    /// `TuningConfig::max_memory_bytes`, if one was configured.
    ///
    /// Without this the field would be a builder option that gates nothing.
    fn apply_memory_cap(&self, rec: &mut TuningRecommendation) {
        let Some(limit) = self.config.max_memory_bytes else {
            return;
        };

        while rec.expected_memory_bytes > limit && (rec.hidden_dim > 8 || rec.num_layers > 1) {
            if rec.num_layers > 1 {
                rec.num_layers -= 1;
            } else {
                rec.hidden_dim /= 2;
            }
            rec.expected_memory_bytes = (rec.hidden_dim * 64 * rec.num_layers * 4)
                + (rec.hidden_dim * rec.hidden_dim * rec.num_layers * 16);
        }

        if rec.expected_memory_bytes > limit {
            rec.confidence = rec.confidence.min(0.5);
            rec.reasoning
                .push_str(" (could not fit max_memory_bytes even at the smallest supported size)");
        } else {
            rec.reasoning
                .push_str(" (capped to fit the configured max_memory_bytes)");
        }
    }

    /// Auto-tune and build a predictor for latency target
    pub fn build_for_latency(
        &self,
        input_dim: usize,
        output_dim: usize,
        target_latency_us: u64,
    ) -> KizzasiResult<Kizzasi> {
        let rec = self.recommend_for_latency(input_dim, output_dim, target_latency_us);

        KizzasiBuilder::new()
            .model_type(rec.model_type)
            .input_dim(input_dim)
            .output_dim(output_dim)
            .hidden_dim(rec.hidden_dim)
            .num_layers(rec.num_layers)
            .context_window(rec.context_window)
            .build()
    }

    /// Auto-tune and build a predictor for throughput target
    pub fn build_for_throughput(
        &self,
        input_dim: usize,
        output_dim: usize,
        target_throughput_pps: f64,
    ) -> KizzasiResult<Kizzasi> {
        let rec = self.recommend_for_throughput(input_dim, output_dim, target_throughput_pps);

        KizzasiBuilder::new()
            .model_type(rec.model_type)
            .input_dim(input_dim)
            .output_dim(output_dim)
            .hidden_dim(rec.hidden_dim)
            .num_layers(rec.num_layers)
            .context_window(rec.context_window)
            .build()
    }

    /// Get profiling history
    pub fn history(&self) -> &VecDeque<WorkloadProfile> {
        &self.profiling_history
    }

    /// Get the most recent profile
    pub fn latest_profile(&self) -> Option<&WorkloadProfile> {
        self.profiling_history.back()
    }

    /// Clear profiling history
    pub fn clear_history(&mut self) {
        self.profiling_history.clear();
    }
}

/// Adaptive tuner that continuously adjusts configuration based on workload
pub struct AdaptiveTuner {
    tuner: AutoTuner,
    current_predictor: Option<Kizzasi>,
    input_dim: usize,
    output_dim: usize,
    adaptation_interval: usize,
    predictions_since_adaptation: usize,
}

impl AdaptiveTuner {
    /// Create a new adaptive tuner
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        config: TuningConfig,
        adaptation_interval: usize,
    ) -> Self {
        Self {
            tuner: AutoTuner::new(config),
            current_predictor: None,
            input_dim,
            output_dim,
            adaptation_interval,
            predictions_since_adaptation: 0,
        }
    }

    /// Initialize with a predictor
    pub fn with_predictor(mut self, predictor: Kizzasi) -> Self {
        self.current_predictor = Some(predictor);
        self
    }

    /// Perform a prediction and potentially adapt
    pub fn predict(&mut self, input: &Array1<f32>) -> KizzasiResult<Array1<f32>> {
        // Initialize predictor if needed
        if self.current_predictor.is_none() {
            let predictor =
                KizzasiBuilder::lightweight_preset(self.input_dim, self.output_dim).build()?;
            self.current_predictor = Some(predictor);
        }

        let predictor =
            self.current_predictor
                .as_mut()
                .ok_or_else(|| KizzasiError::InvalidState {
                    reason: "Predictor not initialized".to_string(),
                    recovery: None,
                })?;

        let output = predictor.step(input)?;
        self.predictions_since_adaptation += 1;

        // Check if we should adapt
        if self.predictions_since_adaptation >= self.adaptation_interval {
            self.adapt()?;
            self.predictions_since_adaptation = 0;
        }

        Ok(output)
    }

    /// Trigger adaptation based on current performance
    ///
    /// Profiling restores the live hidden state (see [`AutoTuner::profile`]).
    /// Switching architectures does **not**: the replacement predictor starts
    /// from a fresh hidden state and freshly initialised weights, because no
    /// two architectures share a state layout. Guardrails configured on the
    /// outgoing predictor are carried over; plugins are not (they are not
    /// required to be cloneable) and must be re-registered by the caller.
    fn adapt(&mut self) -> KizzasiResult<()> {
        let Some(predictor) = &mut self.current_predictor else {
            return Ok(());
        };

        // Profile current predictor
        let profile = self.tuner.profile(predictor, self.input_dim)?;

        // Determine if we need to adapt
        let Some(target_latency_us) = self.tuner.config.target_latency_us else {
            return Ok(());
        };
        if profile.avg_latency_us <= target_latency_us * 120 / 100 {
            return Ok(());
        }

        // More than 20% over target - switch to faster model
        let rec =
            self.tuner
                .recommend_for_latency(self.input_dim, self.output_dim, target_latency_us);
        if rec.confidence <= 0.7 {
            return Ok(());
        }

        let mut builder = KizzasiBuilder::new()
            .model_type(rec.model_type)
            .input_dim(self.input_dim)
            .output_dim(self.output_dim)
            .hidden_dim(rec.hidden_dim)
            .num_layers(rec.num_layers)
            .context_window(rec.context_window);

        // Carry safety constraints across the switch; dropping them silently
        // would remove enforcement the caller explicitly asked for.
        #[cfg(feature = "logic")]
        if let Some(guardrails) = predictor.guardrails() {
            builder = builder.guardrails(guardrails.clone());
        }

        self.current_predictor = Some(builder.build()?);

        Ok(())
    }

    /// Get the current predictor
    pub fn predictor(&self) -> Option<&Kizzasi> {
        self.current_predictor.as_ref()
    }

    /// Get mutable reference to current predictor
    pub fn predictor_mut(&mut self) -> Option<&mut Kizzasi> {
        self.current_predictor.as_mut()
    }

    /// Get the most recent profile
    pub fn latest_profile(&self) -> Option<&WorkloadProfile> {
        self.tuner.latest_profile()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuning_config() {
        let config = TuningConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.profiling_iterations, 100);

        let custom = TuningConfig::new()
            .with_target_latency_us(1000)
            .with_warmup(20)
            .aggressive();

        assert_eq!(custom.target_latency_us, Some(1000));
        assert_eq!(custom.warmup_iterations, 20);
        assert!(custom.aggressive_tuning);
    }

    #[test]
    fn test_autotuner_creation() {
        let tuner = AutoTuner::with_defaults();
        assert_eq!(tuner.profiling_history.len(), 0);
    }

    #[test]
    fn test_latency_recommendations() {
        let tuner = AutoTuner::with_defaults();

        // Ultra-low latency
        let rec1 = tuner.recommend_for_latency(2, 2, 50);
        assert_eq!(rec1.model_type, ModelType::S4);
        assert!(rec1.hidden_dim <= 64);

        // Medium latency
        let rec2 = tuner.recommend_for_latency(64, 64, 1000);
        assert_eq!(rec2.model_type, ModelType::Mamba2);

        // High latency
        let rec3 = tuner.recommend_for_latency(128, 128, 5000);
        assert_eq!(rec3.model_type, ModelType::Rwkv);
        assert!(rec3.hidden_dim >= 128);
    }

    #[test]
    fn test_throughput_recommendations() {
        let tuner = AutoTuner::with_defaults();

        // High throughput = low latency. The estimate must be strictly
        // positive: integer division used to collapse it to 0 for every model
        // narrower than 64 channels (the ultra-low-latency branch picks
        // hidden_dim = 32), which reported +inf throughput at 0.95 confidence.
        // The old bound `< 20` passed only because the value was 0.
        let rec1 = tuner.recommend_for_throughput(2, 2, 100000.0); // 100K pps
        assert!(rec1.expected_latency_us > 0, "latency estimate must be > 0");
        assert!(rec1.expected_throughput_pps.is_finite());

        // Low throughput = can use larger model
        let rec2 = tuner.recommend_for_throughput(64, 64, 100.0); // 100 pps
        assert!(rec2.expected_latency_us > 1000);
        assert!(rec2.expected_throughput_pps.is_finite());

        // A tighter budget must still yield a cheaper model.
        assert!(rec1.expected_latency_us < rec2.expected_latency_us);
    }

    #[test]
    fn test_zero_profiling_iterations_is_rejected() {
        let mut tuner = AutoTuner::new(TuningConfig::new().with_profiling(0));
        let mut predictor = KizzasiBuilder::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16)
            .build()
            .unwrap();

        // Must return an error rather than dividing by zero / indexing an
        // empty percentile vector.
        assert!(tuner.profile(&mut predictor, 2).is_err());
    }

    #[test]
    fn test_profile_restores_live_hidden_state() {
        let mut predictor = KizzasiBuilder::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(1)
            .build()
            .unwrap();

        let input = Array1::from_vec(vec![0.25, -0.5]);
        predictor.step(&input).unwrap();

        // Baseline: what the live stream would produce next.
        let mut reference = predictor.fork().unwrap();
        reference.step(&input).unwrap();
        let expected = reference.step(&input).unwrap();

        let mut tuner = AutoTuner::new(TuningConfig::new().with_warmup(2).with_profiling(5));
        tuner.profile(&mut predictor, 2).unwrap();

        // The synthetic profiling samples must not have advanced the caller's
        // hidden state.
        let actual = predictor.step(&input).unwrap();
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "profiling poisoned the hidden state");
        }
    }

    #[test]
    fn test_max_memory_caps_recommendation() {
        let tuner = AutoTuner::new(TuningConfig::new().with_max_memory(64 * 1024));
        let rec = tuner.recommend_balanced(8, 8);
        assert!(
            rec.expected_memory_bytes <= 64 * 1024 || rec.confidence <= 0.5,
            "max_memory_bytes must actually constrain the recommendation"
        );
    }

    #[test]
    fn test_balanced_recommendations() {
        let tuner = AutoTuner::with_defaults();
        let rec = tuner.recommend_balanced(32, 32);

        assert!(rec.hidden_dim >= 64);
        assert!(rec.num_layers >= 2);
        assert!(rec.confidence >= 0.0); // Confidence is calculated based on latency target match
        assert!(rec.confidence <= 1.0);
    }

    #[test]
    fn test_build_for_latency() -> KizzasiResult<()> {
        let tuner = AutoTuner::with_defaults();
        let predictor = tuner.build_for_latency(4, 4, 1000)?;

        assert_eq!(predictor.input_dim(), 4);
        assert_eq!(predictor.output_dim(), 4);

        Ok(())
    }

    #[test]
    fn test_profiling() -> KizzasiResult<()> {
        let config = TuningConfig::new().with_warmup(2).with_profiling(10);
        let mut tuner = AutoTuner::new(config);

        let mut predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let profile = tuner.profile(&mut predictor, 2)?;

        assert_eq!(profile.avg_input_dim, 2);
        assert!(profile.avg_latency_us > 0);
        assert!(profile.throughput_pps > 0.0);
        assert_eq!(tuner.profiling_history.len(), 1);

        Ok(())
    }

    #[test]
    fn test_adaptive_tuner() -> KizzasiResult<()> {
        let config = TuningConfig::new().with_target_latency_us(10000);
        let mut adaptive = AdaptiveTuner::new(2, 2, config, 100);

        let input = Array1::from_vec(vec![1.0, 2.0]);

        // First prediction should initialize
        let output = adaptive.predict(&input)?;
        assert_eq!(output.len(), 2);

        // Subsequent predictions
        for _ in 0..10 {
            let _ = adaptive.predict(&input)?;
        }

        assert!(adaptive.predictor().is_some());

        Ok(())
    }

    #[test]
    fn test_aggressive_tuning() {
        let config = TuningConfig::default().aggressive();
        let tuner = AutoTuner::new(config);

        let rec1 = tuner.recommend_for_latency(64, 64, 1000);

        let config_normal = TuningConfig::default();
        let tuner_normal = AutoTuner::new(config_normal);
        let rec2 = tuner_normal.recommend_for_latency(64, 64, 1000);

        // Aggressive should use smaller model
        assert!(rec1.hidden_dim <= rec2.hidden_dim);
    }

    #[test]
    fn test_conservative_tuning() {
        let config = TuningConfig::default().conservative();
        let tuner = AutoTuner::new(config);

        let rec1 = tuner.recommend_for_latency(64, 64, 5000);

        let config_normal = TuningConfig::default();
        let tuner_normal = AutoTuner::new(config_normal);
        let rec2 = tuner_normal.recommend_for_latency(64, 64, 5000);

        // Conservative should use larger model
        assert!(rec1.hidden_dim >= rec2.hidden_dim);
    }
}
