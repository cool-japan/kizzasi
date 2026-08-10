//! Sampling strategies for inference
//!
//! This module provides various sampling strategies for autoregressive prediction:
//! - **Greedy**: Always select the highest probability value
//! - **Temperature**: Scale logits to control randomness
//! - **Top-k**: Sample from the k most likely candidates
//! - **Top-p (nucleus)**: Sample from the smallest set with cumulative probability >= p
//! - **Beam search**: Maintain multiple hypotheses for multi-step prediction

use crate::error::{InferenceError, InferenceResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Construct a `Send`-safe RNG from an optional seed.
///
/// When a seed is provided the RNG is deterministic (reproducible across runs).
/// When no seed is provided a random `u64` seed is drawn from the thread-local RNG and
/// used to initialise a `StdRng` — random but `Send`-safe, unlike `ThreadRng` which is
/// `!Send`.
///
/// Uses `scirs2_core::random::StdRng` (= `Random<rand::rngs::StdRng>`) which is `Send`.
fn make_sampler_rng(seed: Option<u64>) -> scirs2_core::random::StdRng {
    // `Random::<rand::rngs::ThreadRng>::seed(s)` returns `Random<rand::rngs::StdRng>`
    let s = match seed {
        Some(s) => s,
        None => {
            // Draw a random seed from the thread-local RNG so the sampler is
            // non-deterministic by default but still `Send`-safe.
            use scirs2_core::random::RngExt;
            scirs2_core::random::rng().random::<u64>()
        }
    };
    scirs2_core::random::ThreadRng::seed(s)
}

/// Configuration for sampling strategies
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SamplingConfig {
    /// Sampling strategy to use
    pub strategy: SamplingStrategy,
    /// Temperature for scaling (1.0 = no scaling, <1.0 = sharper, >1.0 = smoother)
    pub temperature: f32,
    /// For top-k sampling: number of top candidates to consider
    pub top_k: Option<usize>,
    /// For top-p sampling: cumulative probability threshold
    pub top_p: Option<f32>,
    /// For beam search: beam width
    pub beam_width: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            beam_width: 1,
            seed: None,
        }
    }
}

impl SamplingConfig {
    /// Create a new sampling configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the sampling strategy
    pub fn strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set temperature (1.0 = no scaling)
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Enable top-k sampling
    pub fn top_k(mut self, k: usize) -> Self {
        self.strategy = SamplingStrategy::TopK;
        self.top_k = Some(k);
        self
    }

    /// Enable top-p (nucleus) sampling
    pub fn top_p(mut self, p: f32) -> Self {
        self.strategy = SamplingStrategy::TopP;
        self.top_p = Some(p);
        self
    }

    /// Enable beam search
    pub fn beam_search(mut self, width: usize) -> Self {
        self.strategy = SamplingStrategy::BeamSearch;
        self.beam_width = width;
        self
    }

    /// Set random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Available sampling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SamplingStrategy {
    /// Always select the highest value (deterministic)
    Greedy,
    /// Sample from scaled distribution
    Temperature,
    /// Sample from top k candidates
    TopK,
    /// Sample from nucleus (cumulative probability threshold)
    TopP,
    /// Maintain multiple beams for multi-step prediction
    BeamSearch,
    /// Custom sampling function
    Custom,
}

/// Custom sampling function type
/// Takes logits and temperature, returns sampled index
pub type CustomSamplingFn = Arc<dyn Fn(&Array1<f32>, f32) -> InferenceResult<f32> + Send + Sync>;

/// Sampler for generating predictions from model outputs
pub struct Sampler {
    config: SamplingConfig,
    /// Custom sampling function (if strategy is Custom)
    custom_fn: Option<CustomSamplingFn>,
    /// Internal RNG — deterministic when `config.seed` is `Some`, OS-seeded otherwise.
    /// Using `StdRng` (= `Random<rand::rngs::StdRng>`, not `ThreadRng`) keeps `Sampler: Send`.
    rng: scirs2_core::random::StdRng,
}

impl Sampler {
    /// Create a new sampler with given configuration
    pub fn new(config: SamplingConfig) -> Self {
        let rng = make_sampler_rng(config.seed);
        Self {
            config,
            custom_fn: None,
            rng,
        }
    }

    /// Create a sampler with a custom sampling function
    pub fn with_custom_fn(mut config: SamplingConfig, custom_fn: CustomSamplingFn) -> Self {
        config.strategy = SamplingStrategy::Custom;
        let rng = make_sampler_rng(config.seed);
        Self {
            config,
            custom_fn: Some(custom_fn),
            rng,
        }
    }

    /// Set custom sampling function
    pub fn set_custom_fn(&mut self, custom_fn: CustomSamplingFn) {
        self.custom_fn = Some(custom_fn);
        self.config.strategy = SamplingStrategy::Custom;
    }

    /// Sample a single value from logits
    ///
    /// # Arguments
    /// * `logits` - Raw model outputs (unnormalized)
    ///
    /// # Returns
    /// The sampled value
    pub fn sample(&mut self, logits: &Array1<f32>) -> InferenceResult<f32> {
        if logits.is_empty() {
            return Err(InferenceError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }

        match self.config.strategy {
            SamplingStrategy::Greedy => Ok(self.greedy_sample(logits)),
            SamplingStrategy::Temperature => self.temperature_sample(logits),
            SamplingStrategy::TopK => self.top_k_sample(logits),
            SamplingStrategy::TopP => self.top_p_sample(logits),
            SamplingStrategy::BeamSearch => {
                // Beam search requires multi-step context, use greedy for single-step
                Ok(self.greedy_sample(logits))
            }
            SamplingStrategy::Custom => {
                if let Some(ref custom_fn) = self.custom_fn {
                    custom_fn(logits, self.config.temperature)
                } else {
                    // Fallback to greedy if no custom function is set
                    Ok(self.greedy_sample(logits))
                }
            }
        }
    }

    /// Sample multiple values from a batch of logits
    pub fn sample_batch(&mut self, logits: &Array2<f32>) -> InferenceResult<Array1<f32>> {
        let batch_size = logits.nrows();
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let logit_row = logits.row(i).to_owned();
            results.push(self.sample(&logit_row)?);
        }

        Ok(Array1::from_vec(results))
    }

    /// Greedy sampling: select the maximum value
    fn greedy_sample(&self, logits: &Array1<f32>) -> f32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as f32)
            .unwrap_or(0.0)
    }

    /// Temperature sampling with optional scaling.
    ///
    /// When temperature is at or near zero the distribution collapses to a point
    /// mass on the argmax.  Dividing by a near-zero temperature would produce
    /// `±inf` values that propagate as `NaN` through softmax, so we short-circuit
    /// to greedy sampling instead.
    fn temperature_sample(&mut self, logits: &Array1<f32>) -> InferenceResult<f32> {
        if self.config.temperature <= 1e-6_f32 {
            return Ok(self.greedy_sample(logits));
        }

        let scaled = if (self.config.temperature - 1.0).abs() > 1e-6 {
            logits.mapv(|x| x / self.config.temperature)
        } else {
            logits.clone()
        };

        let probs = softmax(&scaled);
        self.sample_categorical(&probs)
    }

    /// Top-k sampling: sample from k most likely candidates
    fn top_k_sample(&mut self, logits: &Array1<f32>) -> InferenceResult<f32> {
        let k = self.config.top_k.unwrap_or(10);

        // Get top-k indices
        let mut indexed: Vec<_> = logits.iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top_k_indices: Vec<usize> = indexed.iter().take(k).map(|(idx, _)| *idx).collect();

        // Create filtered logits
        let mut filtered = Array1::from_elem(logits.len(), f32::NEG_INFINITY);
        for &idx in &top_k_indices {
            filtered[idx] = logits[idx];
        }

        let probs = softmax(&filtered);
        self.sample_categorical(&probs)
    }

    /// Top-p (nucleus) sampling: sample from cumulative probability threshold
    fn top_p_sample(&mut self, logits: &Array1<f32>) -> InferenceResult<f32> {
        let p = self.config.top_p.unwrap_or(0.9);

        // Sort by probability (descending)
        let probs = softmax(logits);
        let mut indexed: Vec<_> = probs.iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Find nucleus (smallest set with cumulative prob >= p)
        let mut cumsum = 0.0;
        let mut nucleus_size = 0;
        for (_, &prob) in &indexed {
            cumsum += prob;
            nucleus_size += 1;
            if cumsum >= p {
                break;
            }
        }

        // Create filtered logits
        let nucleus_indices: Vec<usize> = indexed
            .iter()
            .take(nucleus_size)
            .map(|(idx, _)| *idx)
            .collect();
        let mut filtered = Array1::from_elem(logits.len(), f32::NEG_INFINITY);
        for &idx in &nucleus_indices {
            filtered[idx] = logits[idx];
        }

        let filtered_probs = softmax(&filtered);
        self.sample_categorical(&filtered_probs)
    }

    /// Sample from a categorical distribution using the stored RNG.
    ///
    /// The RNG is either seeded deterministically (when `SamplingConfig::seed` is
    /// `Some`) or seeded from OS entropy, ensuring that seeded samplers produce
    /// fully reproducible sequences across calls.
    fn sample_categorical(&mut self, probs: &Array1<f32>) -> InferenceResult<f32> {
        use scirs2_core::random::RngExt;
        let uniform: f32 = self.rng.random::<f32>();
        let mut cumsum = 0.0;
        for (idx, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if uniform < cumsum {
                return Ok(idx as f32);
            }
        }
        // Fallback to last index (handles floating-point rounding where cumsum < 1.0)
        Ok((probs.len() - 1) as f32)
    }

    /// Get the current configuration
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }
}

/// Apply softmax to convert logits to probabilities
fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    // Subtract max for numerical stability
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits = logits.mapv(|x| (x - max_logit).exp());
    let sum_exp: f32 = exp_logits.sum();

    if sum_exp > 0.0 {
        exp_logits / sum_exp
    } else {
        // All zeros case - uniform distribution
        Array1::from_elem(logits.len(), 1.0 / logits.len() as f32)
    }
}

/// Beam search state for multi-step prediction
#[derive(Debug, Clone)]
pub struct Beam {
    /// Sequence of values
    pub sequence: Vec<f32>,
    /// Cumulative log probability
    pub log_prob: f32,
    /// Current hidden states
    pub states: Vec<kizzasi_core::HiddenState>,
}

impl Beam {
    /// Create a new beam
    pub fn new() -> Self {
        Self {
            sequence: Vec::new(),
            log_prob: 0.0,
            states: Vec::new(),
        }
    }

    /// Add a value to the beam
    pub fn extend(&mut self, value: f32, log_prob: f32) {
        self.sequence.push(value);
        self.log_prob += log_prob;
    }

    /// Get the average log probability (normalized by length)
    pub fn avg_log_prob(&self) -> f32 {
        if self.sequence.is_empty() {
            0.0
        } else {
            self.log_prob / self.sequence.len() as f32
        }
    }
}

impl Default for Beam {
    fn default() -> Self {
        Self::new()
    }
}

/// Beam search manager
pub struct BeamSearch {
    /// Number of beams to maintain
    beam_width: usize,
    /// Current beams
    beams: Vec<Beam>,
}

impl BeamSearch {
    /// Create a new beam search with given width
    pub fn new(beam_width: usize) -> Self {
        let beams = vec![Beam::new()];
        Self { beam_width, beams }
    }

    /// Expand beams with new candidates
    pub fn expand(&mut self, logits: &Array2<f32>) -> InferenceResult<()> {
        if logits.nrows() != self.beams.len() {
            return Err(InferenceError::DimensionMismatch {
                expected: self.beams.len(),
                got: logits.nrows(),
            });
        }

        let mut candidates = Vec::new();

        for (beam_idx, beam) in self.beams.iter().enumerate() {
            let beam_logits = logits.row(beam_idx).to_owned();
            let probs = softmax(&beam_logits);

            // Get top-k candidates for this beam
            let mut indexed: Vec<_> = probs.iter().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            for (idx, &prob) in indexed.iter().take(self.beam_width) {
                let mut new_beam = beam.clone();
                new_beam.extend(*idx as f32, prob.ln());
                candidates.push(new_beam);
            }
        }

        // Select top beam_width candidates
        candidates.sort_by(|a, b| {
            b.avg_log_prob()
                .partial_cmp(&a.avg_log_prob())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.beams = candidates.into_iter().take(self.beam_width).collect();

        Ok(())
    }

    /// Get the best beam
    pub fn best(&self) -> Option<&Beam> {
        self.beams.first()
    }

    /// Get all beams
    pub fn beams(&self) -> &[Beam] {
        &self.beams
    }
}

/// Constraint function type for constrained beam search
/// Returns true if the sequence satisfies the constraint
pub type ConstraintFn = Arc<dyn Fn(&[f32]) -> bool + Send + Sync>;

/// Constrained beam search that only keeps beams satisfying constraints
pub struct ConstrainedBeamSearch {
    /// Underlying beam search
    beam_search: BeamSearch,
    /// Constraint functions to check
    constraints: Vec<ConstraintFn>,
    /// Whether to use soft constraints (prefer but don't require)
    soft_constraints: bool,
    /// Penalty for violating soft constraints
    constraint_penalty: f32,
}

impl ConstrainedBeamSearch {
    /// Create a new constrained beam search
    pub fn new(beam_width: usize) -> Self {
        Self {
            beam_search: BeamSearch::new(beam_width),
            constraints: Vec::new(),
            soft_constraints: false,
            constraint_penalty: 1.0,
        }
    }

    /// Add a hard constraint (must be satisfied)
    pub fn add_constraint(mut self, constraint: ConstraintFn) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Enable soft constraints with penalty
    pub fn with_soft_constraints(mut self, penalty: f32) -> Self {
        self.soft_constraints = true;
        self.constraint_penalty = penalty;
        self
    }

    /// Check if a sequence satisfies all constraints
    fn satisfies_constraints(&self, sequence: &[f32]) -> bool {
        self.constraints.iter().all(|c| c(sequence))
    }

    /// Expand beams with constraint checking
    pub fn expand(&mut self, logits: &Array2<f32>) -> InferenceResult<()> {
        // First, perform standard beam expansion
        self.beam_search.expand(logits)?;

        // Then filter or penalize beams based on constraints
        if self.soft_constraints {
            // Soft constraints: penalize violating beams
            // First collect which beams violate constraints
            let violations: Vec<bool> = self
                .beam_search
                .beams
                .iter()
                .map(|beam| !self.satisfies_constraints(&beam.sequence))
                .collect();

            // Then apply penalties
            let penalty = self.constraint_penalty;
            for (beam, &violates) in self.beam_search.beams.iter_mut().zip(violations.iter()) {
                if violates {
                    beam.log_prob -= penalty;
                }
            }

            // Re-sort by modified scores
            self.beam_search.beams.sort_by(|a, b| {
                b.log_prob
                    .partial_cmp(&a.log_prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // Hard constraints: filter out violating beams
            let valid_beams: Vec<Beam> = self
                .beam_search
                .beams
                .iter()
                .filter(|beam| self.satisfies_constraints(&beam.sequence))
                .cloned()
                .collect();

            if !valid_beams.is_empty() {
                self.beam_search.beams = valid_beams;
            }
            // If all beams violate constraints, keep original beams
            // (fallback behavior - could also raise error)
        }

        Ok(())
    }

    /// Get the best beam
    pub fn best(&self) -> Option<&Beam> {
        self.beam_search.best()
    }

    /// Get all beams
    pub fn beams(&self) -> &[Beam] {
        self.beam_search.beams()
    }

    /// Get number of active constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
}

use std::sync::Arc;

// ============================================================================
// Rejection Sampling with Constraints
// ============================================================================

/// Rejection sampler that rejects samples violating constraints
pub struct RejectionSampler {
    /// Base sampler for generating candidates
    base_sampler: Sampler,
    /// Constraint functions to check
    constraints: Vec<ConstraintFn>,
    /// Maximum number of rejection attempts before giving up
    max_attempts: usize,
    /// Fallback strategy when all attempts fail
    fallback_strategy: FallbackStrategy,
}

/// Fallback strategy when rejection sampling fails
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackStrategy {
    /// Return the best candidate that violates constraints least
    BestCandidate,
    /// Return greedy sample
    Greedy,
    /// Return an error
    Error,
}

impl RejectionSampler {
    /// Create a new rejection sampler
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            base_sampler: Sampler::new(config),
            constraints: Vec::new(),
            max_attempts: 100,
            fallback_strategy: FallbackStrategy::BestCandidate,
        }
    }

    /// Add a constraint function
    pub fn add_constraint(mut self, constraint: ConstraintFn) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Set maximum number of rejection attempts
    pub fn max_attempts(mut self, attempts: usize) -> Self {
        self.max_attempts = attempts;
        self
    }

    /// Set fallback strategy
    pub fn fallback_strategy(mut self, strategy: FallbackStrategy) -> Self {
        self.fallback_strategy = strategy;
        self
    }

    /// Sample with constraint checking and rejection
    ///
    /// # Arguments
    /// * `logits` - Model output logits
    /// * `context` - Current sequence context for constraint checking
    ///
    /// # Returns
    /// Sampled value that satisfies constraints, or fallback value
    pub fn sample_with_rejection(
        &mut self,
        logits: &Array1<f32>,
        context: &[f32],
    ) -> InferenceResult<f32> {
        if self.constraints.is_empty() {
            // No constraints, just sample normally
            return self.base_sampler.sample(logits);
        }

        let mut best_candidate = None;
        let mut min_violations = usize::MAX;

        for attempt in 0..self.max_attempts {
            let candidate = self.base_sampler.sample(logits)?;

            // Build test sequence
            let mut test_sequence = context.to_vec();
            test_sequence.push(candidate);

            // Check constraints
            let violations = self.count_violations(&test_sequence);

            if violations == 0 {
                // Found a valid sample!
                return Ok(candidate);
            }

            // Track best candidate
            if violations < min_violations {
                min_violations = violations;
                best_candidate = Some(candidate);
            }

            // Early exit if we're making progress
            if attempt > self.max_attempts / 2 && violations < self.constraints.len() / 2 {
                break;
            }
        }

        // All attempts failed, use fallback
        match self.fallback_strategy {
            FallbackStrategy::BestCandidate => best_candidate.ok_or_else(|| {
                InferenceError::ForwardError(
                    "Rejection sampling failed: no candidates generated".to_string(),
                )
            }),
            FallbackStrategy::Greedy => {
                let greedy_config = SamplingConfig::new().strategy(SamplingStrategy::Greedy);
                let mut greedy_sampler = Sampler::new(greedy_config);
                greedy_sampler.sample(logits)
            }
            FallbackStrategy::Error => Err(InferenceError::ForwardError(format!(
                "Rejection sampling failed after {} attempts",
                self.max_attempts
            ))),
        }
    }

    /// Count how many constraints are violated
    fn count_violations(&self, sequence: &[f32]) -> usize {
        self.constraints
            .iter()
            .filter(|constraint| !constraint(sequence))
            .count()
    }

    /// Get the base sampler
    pub fn base_sampler(&self) -> &Sampler {
        &self.base_sampler
    }

    /// Get mutable base sampler
    pub fn base_sampler_mut(&mut self) -> &mut Sampler {
        &mut self.base_sampler
    }

    /// Get number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
}

/// Adaptive rejection sampler that learns from rejections
pub struct AdaptiveRejectionSampler {
    /// Base rejection sampler
    rejection_sampler: RejectionSampler,
    /// Rejection history for learning
    rejection_counts: Vec<usize>,
    /// Total samples attempted
    total_samples: usize,
}

impl AdaptiveRejectionSampler {
    /// Create a new adaptive rejection sampler
    pub fn new(config: SamplingConfig, vocab_size: usize) -> Self {
        Self {
            rejection_sampler: RejectionSampler::new(config),
            rejection_counts: vec![0; vocab_size],
            total_samples: 0,
        }
    }

    /// Add a constraint
    pub fn add_constraint(mut self, constraint: ConstraintFn) -> Self {
        self.rejection_sampler = self.rejection_sampler.add_constraint(constraint);
        self
    }

    /// Sample with adaptive biasing away from frequently rejected values
    pub fn sample_adaptive(
        &mut self,
        logits: &Array1<f32>,
        context: &[f32],
    ) -> InferenceResult<f32> {
        self.total_samples += 1;

        // Bias logits away from frequently rejected values
        let mut adjusted_logits = logits.clone();
        if self.total_samples > 10 {
            let max_rejections = *self.rejection_counts.iter().max().unwrap_or(&1) as f32;
            for (i, &count) in self.rejection_counts.iter().enumerate() {
                if i < adjusted_logits.len() && count > 0 {
                    // Penalize frequently rejected values
                    let penalty = (count as f32 / max_rejections) * 2.0;
                    adjusted_logits[i] -= penalty;
                }
            }
        }

        // Try to sample with rejection
        let result = self
            .rejection_sampler
            .sample_with_rejection(&adjusted_logits, context);

        // Record statistics even on success (for learning)
        if let Ok(value) = result {
            Ok(value)
        } else {
            // On failure, try greedy as fallback and record
            let greedy_config = SamplingConfig::new().strategy(SamplingStrategy::Greedy);
            let mut greedy_sampler = Sampler::new(greedy_config);
            if let Ok(fallback) = greedy_sampler.sample(&adjusted_logits) {
                let idx = fallback as usize;
                if idx < self.rejection_counts.len() {
                    self.rejection_counts[idx] += 1;
                }
            }
            Err(InferenceError::ForwardError(
                "Adaptive rejection sampling failed".to_string(),
            ))
        }
    }

    /// Get rejection statistics
    pub fn rejection_rate(&self) -> f32 {
        if self.total_samples == 0 {
            return 0.0;
        }
        let total_rejections: usize = self.rejection_counts.iter().sum();
        total_rejections as f32 / self.total_samples as f32
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.rejection_counts.fill(0);
        self.total_samples = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let config = SamplingConfig::new().strategy(SamplingStrategy::Greedy);
        let mut sampler = Sampler::new(config);

        let logits = Array1::from_vec(vec![0.1, 0.5, 0.3, 0.8, 0.2]);
        let result = sampler.sample(&logits).unwrap();
        assert_eq!(result, 3.0); // Index of max value
    }

    #[test]
    fn test_temperature_sampling() {
        let config = SamplingConfig::new()
            .strategy(SamplingStrategy::Temperature)
            .temperature(0.5)
            .seed(42);
        let mut sampler = Sampler::new(config);

        let logits = Array1::from_vec(vec![0.1, 0.5, 0.3, 0.8, 0.2]);
        let result = sampler.sample(&logits);
        assert!(result.is_ok());
    }

    #[test]
    fn test_top_k_sampling() {
        let config = SamplingConfig::new().top_k(3).seed(42);
        let mut sampler = Sampler::new(config);

        let logits = Array1::from_vec(vec![0.1, 0.5, 0.3, 0.8, 0.2]);
        let result = sampler.sample(&logits);
        assert!(result.is_ok());
    }

    #[test]
    fn test_top_p_sampling() {
        let config = SamplingConfig::new().top_p(0.9).seed(42);
        let mut sampler = Sampler::new(config);

        let logits = Array1::from_vec(vec![0.1, 0.5, 0.3, 0.8, 0.2]);
        let result = sampler.sample(&logits);
        assert!(result.is_ok());
    }

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);

        // Probabilities should sum to 1
        let sum: f32 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Highest logit should have highest probability
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(max_idx, 2);
    }

    #[test]
    fn test_beam_search() {
        let mut bs = BeamSearch::new(2);

        // First expansion
        let logits1 = Array2::from_shape_vec((1, 3), vec![0.5, 0.3, 0.2]).unwrap();
        bs.expand(&logits1).unwrap();
        assert_eq!(bs.beams().len(), 2);

        // Second expansion
        let logits2 = Array2::from_shape_vec((2, 3), vec![0.4, 0.3, 0.3, 0.5, 0.3, 0.2]).unwrap();
        bs.expand(&logits2).unwrap();
        assert_eq!(bs.beams().len(), 2);

        let best = bs.best().unwrap();
        assert_eq!(best.sequence.len(), 2);
    }

    #[test]
    fn test_beam_avg_log_prob() {
        let mut beam = Beam::new();
        beam.extend(1.0, -0.5);
        beam.extend(2.0, -0.3);

        let avg = beam.avg_log_prob();
        assert!((avg - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn test_sample_batch() {
        let config = SamplingConfig::new().strategy(SamplingStrategy::Greedy);
        let mut sampler = Sampler::new(config);

        let logits = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.1, 0.5, 0.3, 0.2, // Row 0: max at index 1
                0.8, 0.2, 0.1, 0.3, // Row 1: max at index 0
                0.2, 0.3, 0.9, 0.1, // Row 2: max at index 2
            ],
        )
        .unwrap();

        let results = sampler.sample_batch(&logits).unwrap();
        assert_eq!(results[0], 1.0);
        assert_eq!(results[1], 0.0);
        assert_eq!(results[2], 2.0);
    }

    #[test]
    fn test_seeded_sampling_reproducible() {
        // Two samplers with the same seed must produce identical sequences.
        let logits = Array1::from_vec(vec![1.0_f32, 2.0, 0.5, 1.5]);
        let mut s1 = Sampler::new(
            SamplingConfig::new()
                .strategy(SamplingStrategy::Temperature)
                .temperature(0.8)
                .seed(42),
        );
        let mut s2 = Sampler::new(
            SamplingConfig::new()
                .strategy(SamplingStrategy::Temperature)
                .temperature(0.8)
                .seed(42),
        );
        for _ in 0..20 {
            let r1 = s1.sample(&logits).expect("s1 sample");
            let r2 = s2.sample(&logits).expect("s2 sample");
            assert_eq!(r1.to_bits(), r2.to_bits(), "Seeded samplers diverged");
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        // Uniform logits → pure randomness; different seeds should yield different sequences.
        let logits = Array1::from_vec(vec![1.0_f32, 1.0, 1.0, 1.0]);
        let mut s1 = Sampler::new(
            SamplingConfig::new()
                .strategy(SamplingStrategy::Temperature)
                .temperature(1.0)
                .seed(1),
        );
        let mut s2 = Sampler::new(
            SamplingConfig::new()
                .strategy(SamplingStrategy::Temperature)
                .temperature(1.0)
                .seed(99999),
        );
        let results1: Vec<f32> = (0..20).map(|_| s1.sample(&logits).unwrap()).collect();
        let results2: Vec<f32> = (0..20).map(|_| s2.sample(&logits).unwrap()).collect();
        assert!(
            results1 != results2,
            "Different seeds produced identical sequences"
        );
    }

    #[test]
    fn test_temperature_zero_is_greedy() {
        // At T=0, temperature_sample must always return the argmax index.
        let logits = Array1::from_vec(vec![0.1_f32, 5.0, 0.3, 0.2]); // argmax → index 1
        let config = SamplingConfig::new()
            .strategy(SamplingStrategy::Temperature)
            .temperature(0.0);
        let mut sampler = Sampler::new(config);
        for _ in 0..5 {
            let result = sampler.sample(&logits).expect("sample");
            assert_eq!(
                result as usize, 1,
                "T=0 should pick argmax (index 1), got {}",
                result
            );
        }
    }
}
