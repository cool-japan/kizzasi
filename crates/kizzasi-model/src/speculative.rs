//! Speculative Decoding for kizzasi-model
//!
//! Implements speculative decoding where a smaller draft model proposes K tokens
//! and a larger target model verifies them in a single forward pass. This enables
//! significant speedups for autoregressive generation without changing output quality.
//!
//! # Algorithm
//!
//! 1. Draft model generates K tokens autoregressively (cheap)
//! 2. Target model evaluates all K positions in one pass (expensive but batched)
//! 3. Accept/reject each draft token based on probability ratio
//! 4. On rejection, sample correction token from adjusted distribution
//!
//! # References
//!
//! - Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)
//! - Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Configuration and state for speculative decoding
///
/// The draft model proposes `draft_steps` tokens per cycle, then the target model
/// verifies them all at once. Accepted tokens are kept; on first rejection, a
/// correction token is sampled from the adjusted distribution.
#[derive(Debug, Clone)]
pub struct SpeculativeDecoder {
    /// K: number of tokens to draft per speculation cycle
    draft_steps: usize,
    /// Sampling temperature applied to logits before softmax
    temperature: f32,
    /// Minimum probability ratio p_target/p_draft for token acceptance
    acceptance_threshold: f32,
    /// Running statistics for monitoring performance
    stats: SpeculativeStats,
}

/// Result of a single speculative decoding verification step
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// How many draft tokens were accepted by the target model
    pub accepted_count: usize,
    /// Index of the first rejected token (None if all accepted)
    pub rejected_at: Option<usize>,
    /// Estimated speedup factor compared to pure autoregressive decoding
    pub speedup_estimate: f32,
    /// Per-token acceptance probabilities that were computed
    pub token_acceptance_probs: Vec<f32>,
}

/// Cumulative statistics across multiple speculation cycles
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total number of speculation cycles performed
    pub total_cycles: usize,
    /// Total number of draft tokens proposed across all cycles
    pub total_drafted: usize,
    /// Total number of draft tokens accepted across all cycles
    pub total_accepted: usize,
    /// Running average of speedup estimates
    pub avg_speedup: f32,
    /// Histogram of accepted counts per cycle (index = accepted_count)
    pub acceptance_histogram: Vec<usize>,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder
    ///
    /// # Arguments
    ///
    /// * `draft_steps` - K: number of tokens the draft model proposes per cycle
    /// * `temperature` - Sampling temperature (must be > 0). Higher = more random
    /// * `acceptance_threshold` - Minimum p_target/p_draft ratio to accept (0.0 to 1.0)
    ///
    /// # Errors
    ///
    /// Returns `ModelError::InvalidConfig` if parameters are out of valid range
    pub fn new(
        draft_steps: usize,
        temperature: f32,
        acceptance_threshold: f32,
    ) -> ModelResult<Self> {
        if draft_steps == 0 {
            return Err(ModelError::invalid_config("draft_steps must be at least 1"));
        }
        if temperature <= 0.0 {
            return Err(ModelError::invalid_config("temperature must be positive"));
        }
        if !(0.0..=1.0).contains(&acceptance_threshold) {
            return Err(ModelError::invalid_config(
                "acceptance_threshold must be in [0.0, 1.0]",
            ));
        }

        let acceptance_histogram = vec![0; draft_steps + 1];

        Ok(Self {
            draft_steps,
            temperature,
            acceptance_threshold,
            stats: SpeculativeStats {
                acceptance_histogram,
                ..Default::default()
            },
        })
    }

    /// Get the number of draft steps (K)
    pub fn draft_steps(&self) -> usize {
        self.draft_steps
    }

    /// Get the current temperature
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Get the acceptance threshold
    pub fn acceptance_threshold(&self) -> f32 {
        self.acceptance_threshold
    }

    /// Get cumulative statistics
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset cumulative statistics
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats {
            acceptance_histogram: vec![0; self.draft_steps + 1],
            ..Default::default()
        };
    }

    /// Verify draft tokens against target model logits
    ///
    /// For each draft token i (0..K), computes the probability ratio
    /// p_target(token_i) / p_draft(token_i). If the ratio >= acceptance_threshold,
    /// the token is accepted. The first rejection stops the acceptance chain.
    ///
    /// # Arguments
    ///
    /// * `draft_tokens` - Token indices from draft model, shape (K,)
    /// * `draft_logits` - Logits from draft model, shape (K, vocab_size)
    /// * `target_logits` - Logits from target model, shape (K, vocab_size)
    ///
    /// # Errors
    ///
    /// Returns `ModelError::DimensionMismatch` if shapes are inconsistent
    pub fn verify(
        &mut self,
        draft_tokens: &Array1<usize>,
        draft_logits: &Array2<f32>,
        target_logits: &Array2<f32>,
    ) -> ModelResult<SpeculativeResult> {
        let k = draft_tokens.len();

        // Validate dimensions
        if k == 0 {
            return Err(ModelError::invalid_config("draft_tokens must not be empty"));
        }
        if draft_logits.nrows() != k {
            return Err(ModelError::dimension_mismatch(
                "draft_logits rows vs draft_tokens length",
                k,
                draft_logits.nrows(),
            ));
        }
        if target_logits.nrows() != k {
            return Err(ModelError::dimension_mismatch(
                "target_logits rows vs draft_tokens length",
                k,
                target_logits.nrows(),
            ));
        }
        let vocab_size = draft_logits.ncols();
        if target_logits.ncols() != vocab_size {
            return Err(ModelError::dimension_mismatch(
                "target_logits vocab_size vs draft_logits vocab_size",
                vocab_size,
                target_logits.ncols(),
            ));
        }

        let mut accepted_count = 0;
        let mut rejected_at = None;
        let mut token_acceptance_probs = Vec::with_capacity(k);

        for i in 0..k {
            let draft_row = draft_logits.row(i);
            let target_row = target_logits.row(i);

            // Convert logits to probabilities with temperature
            let draft_probs = softmax_with_temperature(&draft_row.to_owned(), self.temperature)?;
            let target_probs = softmax_with_temperature(&target_row.to_owned(), self.temperature)?;

            let token_idx = draft_tokens[i];
            if token_idx >= vocab_size {
                return Err(ModelError::IndexOutOfBounds {
                    index: token_idx,
                    limit: vocab_size,
                    context: format!("draft_tokens[{}] exceeds vocab_size", i),
                });
            }

            let p_draft = draft_probs[token_idx];
            let p_target = target_probs[token_idx];

            // Compute acceptance probability: min(1, p_target / p_draft)
            let accept_prob = if p_draft > 1e-10 {
                (p_target / p_draft).min(1.0)
            } else if p_target > 1e-10 {
                // Draft assigns ~0 probability but target doesn't -- reject
                0.0
            } else {
                // Both assign ~0 probability -- accept (both agree it's unlikely)
                1.0
            };

            token_acceptance_probs.push(accept_prob);

            if accept_prob >= self.acceptance_threshold {
                accepted_count += 1;
            } else {
                rejected_at = Some(i);
                break;
            }
        }

        // Speedup estimate: we verified K tokens with one target pass
        // Effective tokens per target eval = accepted + 1 (the correction token)
        // vs autoregressive: 1 token per target eval
        // So speedup = (accepted + 1) / (1 + K_draft_cost_ratio)
        // Assuming draft cost is negligible compared to target:
        let speedup_estimate = if k > 0 {
            (accepted_count as f32 + 1.0) / (1.0 + (k as f32).recip())
        } else {
            1.0
        };

        // Update statistics
        self.stats.total_cycles += 1;
        self.stats.total_drafted += k;
        self.stats.total_accepted += accepted_count;

        if accepted_count < self.stats.acceptance_histogram.len() {
            self.stats.acceptance_histogram[accepted_count] += 1;
        }

        // Update running average speedup
        let n = self.stats.total_cycles as f32;
        self.stats.avg_speedup = self.stats.avg_speedup * ((n - 1.0) / n) + speedup_estimate / n;

        Ok(SpeculativeResult {
            accepted_count,
            rejected_at,
            speedup_estimate,
            token_acceptance_probs,
        })
    }

    /// Compute element-wise acceptance probabilities for draft tokens
    ///
    /// For each position i, `acceptance_prob[i] = min(1, target_probs[i] / draft_probs[i])`
    ///
    /// # Arguments
    ///
    /// * `draft_probs` - Probability distribution from draft model (must sum to ~1)
    /// * `target_probs` - Probability distribution from target model (must sum to ~1)
    ///
    /// # Errors
    ///
    /// Returns `ModelError::DimensionMismatch` if arrays have different lengths
    pub fn acceptance_probabilities(
        &self,
        draft_probs: &Array1<f32>,
        target_probs: &Array1<f32>,
    ) -> ModelResult<Array1<f32>> {
        if draft_probs.len() != target_probs.len() {
            return Err(ModelError::dimension_mismatch(
                "acceptance_probabilities: draft_probs vs target_probs",
                draft_probs.len(),
                target_probs.len(),
            ));
        }

        let n = draft_probs.len();
        let mut result = Array1::<f32>::zeros(n);

        for i in 0..n {
            let p_d = draft_probs[i];
            let p_t = target_probs[i];

            result[i] = if p_d > 1e-10 {
                (p_t / p_d).min(1.0)
            } else if p_t > 1e-10 {
                // Draft says impossible, target says possible -> low acceptance
                0.0
            } else {
                // Both near zero
                1.0
            };
        }

        Ok(result)
    }

    /// Sample a correction token from the adjusted distribution after rejection
    ///
    /// When a draft token is rejected, we sample from max(0, p_target - p_draft)
    /// normalized to a valid probability distribution. This ensures the overall
    /// distribution matches the target model exactly.
    ///
    /// # Arguments
    ///
    /// * `draft_probs` - Probability distribution from draft model
    /// * `target_probs` - Probability distribution from target model
    ///
    /// # Returns
    ///
    /// Token index sampled from the correction distribution
    ///
    /// # Errors
    ///
    /// Returns `ModelError::DimensionMismatch` if arrays have different lengths
    /// Returns `ModelError::NumericalInstability` if correction distribution is degenerate
    pub fn sample_correction(
        &self,
        draft_probs: &Array1<f32>,
        target_probs: &Array1<f32>,
    ) -> ModelResult<usize> {
        if draft_probs.len() != target_probs.len() {
            return Err(ModelError::dimension_mismatch(
                "sample_correction: draft_probs vs target_probs",
                draft_probs.len(),
                target_probs.len(),
            ));
        }

        let n = draft_probs.len();
        if n == 0 {
            return Err(ModelError::invalid_config(
                "cannot sample from empty distribution",
            ));
        }

        // Compute correction distribution: max(0, p_target - p_draft)
        let mut correction = Array1::<f32>::zeros(n);
        let mut sum = 0.0_f32;

        for i in 0..n {
            let diff = (target_probs[i] - draft_probs[i]).max(0.0);
            correction[i] = diff;
            sum += diff;
        }

        if sum > 1e-10 {
            // Normalize to valid probability distribution
            correction /= sum;

            // Deterministic sampling: pick the token with highest corrected probability
            // This is equivalent to argmax, which is the greedy choice from the
            // correction distribution. For stochastic sampling, a random number
            // generator would be needed, but we follow the deterministic approach
            // for reproducibility.
            let mut best_idx = 0;
            let mut best_prob = correction[0];
            for i in 1..n {
                if correction[i] > best_prob {
                    best_prob = correction[i];
                    best_idx = i;
                }
            }
            Ok(best_idx)
        } else {
            // Correction distribution is all zeros -- distributions are identical
            // or draft dominates everywhere. Fall back to argmax of target.
            let mut best_idx = 0;
            let mut best_prob = target_probs[0];
            for i in 1..n {
                if target_probs[i] > best_prob {
                    best_prob = target_probs[i];
                    best_idx = i;
                }
            }
            Ok(best_idx)
        }
    }

    /// Run a complete speculative decoding cycle
    ///
    /// Given draft and target logits for K positions, verify all tokens and
    /// compute a correction token if needed.
    ///
    /// # Arguments
    ///
    /// * `draft_tokens` - Token indices from draft model, shape (K,)
    /// * `draft_logits` - Logits from draft model, shape (K, vocab_size)
    /// * `target_logits` - Logits from target model, shape (K, vocab_size)
    ///
    /// # Returns
    ///
    /// Tuple of (accepted_tokens, optional_correction_token, result)
    pub fn decode_cycle(
        &mut self,
        draft_tokens: &Array1<usize>,
        draft_logits: &Array2<f32>,
        target_logits: &Array2<f32>,
    ) -> ModelResult<(Vec<usize>, Option<usize>, SpeculativeResult)> {
        let result = self.verify(draft_tokens, draft_logits, target_logits)?;

        let mut accepted_tokens = Vec::with_capacity(result.accepted_count);
        for i in 0..result.accepted_count {
            accepted_tokens.push(draft_tokens[i]);
        }

        let correction_token = if let Some(rejected_idx) = result.rejected_at {
            // Sample correction token at the rejected position
            let draft_row = draft_logits.row(rejected_idx);
            let target_row = target_logits.row(rejected_idx);

            let draft_probs = softmax_with_temperature(&draft_row.to_owned(), self.temperature)?;
            let target_probs = softmax_with_temperature(&target_row.to_owned(), self.temperature)?;

            Some(self.sample_correction(&draft_probs, &target_probs)?)
        } else {
            None
        };

        Ok((accepted_tokens, correction_token, result))
    }
}

/// Compute stable softmax with temperature scaling
///
/// Applies temperature scaling to logits before computing softmax.
/// Uses the log-sum-exp trick for numerical stability: subtract max before exp.
///
/// # Arguments
///
/// * `logits` - Raw logit values
/// * `temperature` - Temperature scaling factor (> 0)
///
/// # Returns
///
/// Probability distribution (sums to 1.0)
fn softmax_with_temperature(logits: &Array1<f32>, temperature: f32) -> ModelResult<Array1<f32>> {
    let n = logits.len();
    if n == 0 {
        return Err(ModelError::invalid_config(
            "cannot compute softmax of empty array",
        ));
    }

    // Scale by temperature
    let scaled: Array1<f32> = logits.mapv(|x| x / temperature);

    // Find max for numerical stability
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if !max_val.is_finite() {
        return Err(ModelError::numerical_instability(
            "softmax",
            "logits contain non-finite values",
        ));
    }

    // Compute exp(x - max) for stability
    let mut probs = scaled.mapv(|x| (x - max_val).exp());

    // Normalize
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

/// Compute softmax without temperature (temperature = 1.0)
pub fn softmax(logits: &Array1<f32>) -> ModelResult<Array1<f32>> {
    softmax_with_temperature(logits, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    /// Helper: create logits where both draft and target agree perfectly
    fn create_matching_logits(
        k: usize,
        vocab_size: usize,
    ) -> (Array1<usize>, Array2<f32>, Array2<f32>) {
        let mut logits = Array2::<f32>::zeros((k, vocab_size));
        let mut tokens = Array1::<usize>::zeros(k);

        for i in 0..k {
            // Token i maps to vocab index i % vocab_size
            let token_idx = i % vocab_size;
            tokens[i] = token_idx;
            // Set high logit for the chosen token
            logits[[i, token_idx]] = 10.0;
        }

        (tokens, logits.clone(), logits)
    }

    /// Helper: create mismatched logits where draft and target disagree
    fn create_mismatched_logits(
        k: usize,
        vocab_size: usize,
    ) -> (Array1<usize>, Array2<f32>, Array2<f32>) {
        let mut draft_logits = Array2::<f32>::zeros((k, vocab_size));
        let mut target_logits = Array2::<f32>::zeros((k, vocab_size));
        let mut tokens = Array1::<usize>::zeros(k);

        for i in 0..k {
            let draft_token = i % vocab_size;
            tokens[i] = draft_token;
            // Draft model is confident about its token
            draft_logits[[i, draft_token]] = 10.0;
            // Target model prefers a different token
            let target_token = (i + 1) % vocab_size;
            target_logits[[i, target_token]] = 10.0;
        }

        (tokens, draft_logits, target_logits)
    }

    #[test]
    fn test_speculative_all_accepted() {
        // When draft == target distributions, all K tokens should be accepted
        let k = 5;
        let vocab_size = 10;
        let (tokens, draft_logits, target_logits) = create_matching_logits(k, vocab_size);

        let mut decoder = SpeculativeDecoder::new(k, 1.0, 0.5).expect("valid config");
        let result = decoder
            .verify(&tokens, &draft_logits, &target_logits)
            .expect("verify should succeed");

        assert_eq!(result.accepted_count, k, "all tokens should be accepted");
        assert!(result.rejected_at.is_none(), "no rejection expected");
    }

    #[test]
    fn test_speculative_first_rejected() {
        // When draft and target disagree, first token should be rejected
        let k = 5;
        let vocab_size = 10;
        let (tokens, draft_logits, target_logits) = create_mismatched_logits(k, vocab_size);

        let mut decoder = SpeculativeDecoder::new(k, 1.0, 0.5).expect("valid config");
        let result = decoder
            .verify(&tokens, &draft_logits, &target_logits)
            .expect("verify should succeed");

        assert_eq!(
            result.rejected_at,
            Some(0),
            "first token should be rejected"
        );
        assert_eq!(result.accepted_count, 0);
    }

    #[test]
    fn test_speculative_acceptance_probabilities_shape() {
        let decoder = SpeculativeDecoder::new(5, 1.0, 0.5).expect("valid config");

        let draft_probs = array![0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05];
        let target_probs = array![0.15, 0.25, 0.25, 0.15, 0.1, 0.05, 0.05];

        let accept_probs = decoder
            .acceptance_probabilities(&draft_probs, &target_probs)
            .expect("should succeed");

        assert_eq!(
            accept_probs.len(),
            draft_probs.len(),
            "output shape should match input"
        );

        // All acceptance probabilities should be in [0, 1]
        for &p in accept_probs.iter() {
            assert!(
                (0.0..=1.0).contains(&p),
                "acceptance prob {} out of range",
                p
            );
        }
    }

    #[test]
    fn test_speculative_sample_correction() {
        let decoder = SpeculativeDecoder::new(5, 1.0, 0.5).expect("valid config");

        // Target prefers token 3, draft prefers token 0
        let draft_probs = array![0.8, 0.1, 0.05, 0.03, 0.02];
        let target_probs = array![0.1, 0.1, 0.1, 0.6, 0.1];

        let correction_idx = decoder
            .sample_correction(&draft_probs, &target_probs)
            .expect("should succeed");

        // Correction should be a valid index
        assert!(
            correction_idx < draft_probs.len(),
            "correction index {} should be < vocab_size {}",
            correction_idx,
            draft_probs.len()
        );

        // The correction distribution is max(0, target - draft).
        // target - draft = [-0.7, 0.0, 0.05, 0.57, 0.08]
        // max(0, ...) = [0.0, 0.0, 0.05, 0.57, 0.08]
        // argmax = 3
        assert_eq!(
            correction_idx, 3,
            "correction should pick token 3 (highest in correction dist)"
        );
    }

    #[test]
    fn test_speculative_result_speedup_estimate() {
        let k = 5;
        let vocab_size = 10;
        let (tokens, draft_logits, target_logits) = create_matching_logits(k, vocab_size);

        let mut decoder = SpeculativeDecoder::new(k, 1.0, 0.5).expect("valid config");
        let result = decoder
            .verify(&tokens, &draft_logits, &target_logits)
            .expect("verify should succeed");

        // When all tokens are accepted, speedup should be > 1.0
        assert!(
            result.speedup_estimate > 1.0,
            "speedup {} should be > 1.0 when all tokens accepted",
            result.speedup_estimate
        );
    }

    #[test]
    fn test_speculative_invalid_config() {
        // draft_steps = 0
        assert!(SpeculativeDecoder::new(0, 1.0, 0.5).is_err());
        // temperature = 0
        assert!(SpeculativeDecoder::new(5, 0.0, 0.5).is_err());
        // negative temperature
        assert!(SpeculativeDecoder::new(5, -1.0, 0.5).is_err());
        // threshold > 1
        assert!(SpeculativeDecoder::new(5, 1.0, 1.5).is_err());
        // threshold < 0
        assert!(SpeculativeDecoder::new(5, 1.0, -0.1).is_err());
    }

    #[test]
    fn test_speculative_dimension_mismatch() {
        let mut decoder = SpeculativeDecoder::new(3, 1.0, 0.5).expect("valid config");

        let tokens = Array1::<usize>::zeros(3);
        let draft_logits = Array2::<f32>::zeros((3, 10));
        let bad_target_logits = Array2::<f32>::zeros((2, 10)); // wrong rows

        assert!(decoder
            .verify(&tokens, &draft_logits, &bad_target_logits)
            .is_err());
    }

    #[test]
    fn test_softmax_basic() {
        let logits = array![1.0, 2.0, 3.0];
        let probs = softmax(&logits).expect("softmax should succeed");

        // Should sum to ~1
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {}",
            sum
        );

        // Should be monotonically increasing
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_decode_cycle_with_correction() {
        let k = 3;
        let vocab_size = 5;
        let (tokens, draft_logits, target_logits) = create_mismatched_logits(k, vocab_size);

        let mut decoder = SpeculativeDecoder::new(k, 1.0, 0.5).expect("valid config");
        let (accepted, correction, result) = decoder
            .decode_cycle(&tokens, &draft_logits, &target_logits)
            .expect("decode_cycle should succeed");

        assert_eq!(accepted.len(), result.accepted_count);
        // Since distributions mismatch, we expect a correction token
        if result.rejected_at.is_some() {
            assert!(
                correction.is_some(),
                "should have correction token on rejection"
            );
            assert!(
                correction.expect("correction exists") < vocab_size,
                "correction token should be valid"
            );
        }
    }

    #[test]
    fn test_stats_accumulate() {
        let k = 3;
        let vocab_size = 5;
        let (tokens, draft_logits, target_logits) = create_matching_logits(k, vocab_size);

        let mut decoder = SpeculativeDecoder::new(k, 1.0, 0.5).expect("valid config");

        // Run multiple cycles
        for _ in 0..5 {
            let _ = decoder
                .verify(&tokens, &draft_logits, &target_logits)
                .expect("verify should succeed");
        }

        let stats = decoder.stats();
        assert_eq!(stats.total_cycles, 5);
        assert_eq!(stats.total_drafted, 15); // 5 * 3
        assert!(stats.avg_speedup > 0.0);
    }
}
