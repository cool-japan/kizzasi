//! Speculative decoding for faster inference
//!
//! Speculative decoding uses a smaller "draft" model to generate candidate
//! tokens quickly, then verifies them with the main model. This can significantly
//! speed up autoregressive generation while maintaining quality.
//!
//! ## Algorithm
//!
//! 1. Draft model generates K candidate tokens
//! 2. Main model scores all K candidates in parallel
//! 3. Accept candidates that match main model predictions
//! 4. Reject at first mismatch and continue from there
//!
//! This achieves speedups of 2-3x for compatible draft/main model pairs.

use crate::error::{InferenceError, InferenceResult};
use crate::sampling::{Sampler, SamplingConfig};
use kizzasi_model::AutoregressiveModel;
use scirs2_core::ndarray::Array1;

/// Configuration for speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculatively generate
    pub num_draft_tokens: usize,
    /// Temperature for draft model sampling
    pub draft_temperature: f32,
    /// Temperature for main model verification
    pub main_temperature: f32,
    /// Whether to use greedy decoding for verification
    pub greedy_verification: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_draft_tokens: 4,
            draft_temperature: 1.0,
            main_temperature: 1.0,
            greedy_verification: true,
        }
    }
}

impl SpeculativeConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of draft tokens
    pub fn num_draft_tokens(mut self, n: usize) -> Self {
        self.num_draft_tokens = n;
        self
    }

    /// Set draft temperature
    pub fn draft_temperature(mut self, temp: f32) -> Self {
        self.draft_temperature = temp;
        self
    }

    /// Set main temperature
    pub fn main_temperature(mut self, temp: f32) -> Self {
        self.main_temperature = temp;
        self
    }

    /// Enable/disable greedy verification
    pub fn greedy_verification(mut self, greedy: bool) -> Self {
        self.greedy_verification = greedy;
        self
    }
}

/// Speculative decoding engine
///
/// Uses a small draft model to generate candidates, verified by a larger main model
pub struct SpeculativeDecoder {
    /// Main (larger) model for verification
    main_model: Box<dyn AutoregressiveModel>,
    /// Draft (smaller, faster) model for candidate generation
    draft_model: Box<dyn AutoregressiveModel>,
    /// Configuration
    config: SpeculativeConfig,
    /// Sampler for draft model
    draft_sampler: Sampler,
    /// Sampler for main model
    main_sampler: Sampler,
    /// Statistics
    total_tokens: usize,
    accepted_tokens: usize,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder
    ///
    /// # Arguments
    /// * `main_model` - The main (high-quality) model
    /// * `draft_model` - The draft (fast) model for generating candidates
    /// * `config` - Configuration for speculative decoding
    pub fn new(
        main_model: Box<dyn AutoregressiveModel>,
        draft_model: Box<dyn AutoregressiveModel>,
        config: SpeculativeConfig,
    ) -> Self {
        let draft_sampler = Sampler::new(
            SamplingConfig::new()
                .temperature(config.draft_temperature)
                .strategy(crate::sampling::SamplingStrategy::Temperature),
        );

        let main_sampler = Sampler::new(
            SamplingConfig::new()
                .temperature(config.main_temperature)
                .strategy(if config.greedy_verification {
                    crate::sampling::SamplingStrategy::Greedy
                } else {
                    crate::sampling::SamplingStrategy::Temperature
                }),
        );

        Self {
            main_model,
            draft_model,
            config,
            draft_sampler,
            main_sampler,
            total_tokens: 0,
            accepted_tokens: 0,
        }
    }

    /// Generate tokens with speculative decoding
    ///
    /// # Arguments
    /// * `input` - Initial input
    /// * `max_tokens` - Maximum number of tokens to generate
    ///
    /// # Returns
    /// Generated sequence
    pub fn generate(
        &mut self,
        input: &Array1<f32>,
        max_tokens: usize,
    ) -> InferenceResult<Vec<Array1<f32>>> {
        let mut sequence = Vec::with_capacity(max_tokens);
        let mut current = input.clone();

        while sequence.len() < max_tokens {
            // Generate draft tokens
            let draft_candidates = self.generate_draft_tokens(&current)?;

            // Verify with main model
            let (accepted, next_token) = self.verify_candidates(&current, &draft_candidates)?;

            // Add accepted tokens to sequence
            for token in draft_candidates.iter().take(accepted) {
                sequence.push(token.clone());
                if sequence.len() >= max_tokens {
                    break;
                }
            }

            // Update statistics
            self.total_tokens += self.config.num_draft_tokens;
            self.accepted_tokens += accepted;

            // Continue from the next token
            if sequence.len() < max_tokens {
                sequence.push(next_token.clone());
                current = next_token;
            }
        }

        Ok(sequence)
    }

    /// Generate candidate tokens using the draft model
    fn generate_draft_tokens(&mut self, input: &Array1<f32>) -> InferenceResult<Vec<Array1<f32>>> {
        let mut candidates = Vec::with_capacity(self.config.num_draft_tokens);
        let mut current = input.clone();

        for _ in 0..self.config.num_draft_tokens {
            let logits = self
                .draft_model
                .step(&current)
                .map_err(|e| InferenceError::ForwardError(e.to_string()))?;

            let sampled = self.draft_sampler.sample(&logits)?;
            let token = Array1::from_elem(1, sampled);
            candidates.push(token.clone());
            current = token;
        }

        Ok(candidates)
    }

    /// Verify candidates using the main model
    ///
    /// Returns (number of accepted tokens, next token to use)
    fn verify_candidates(
        &mut self,
        input: &Array1<f32>,
        candidates: &[Array1<f32>],
    ) -> InferenceResult<(usize, Array1<f32>)> {
        let mut current = input.clone();
        let mut accepted = 0;

        for candidate in candidates {
            // Get main model prediction
            let main_logits = self
                .main_model
                .step(&current)
                .map_err(|e| InferenceError::ForwardError(e.to_string()))?;

            let main_prediction = self.main_sampler.sample(&main_logits)?;

            // Check if candidate matches
            let candidate_value = candidate[0];
            let matches = if self.config.greedy_verification {
                // Greedy: must match exactly
                (candidate_value - main_prediction).abs() < 1e-6
            } else {
                // Sampling: accept with some probability
                (candidate_value - main_prediction).abs() < 0.5
            };

            if matches {
                accepted += 1;
                current = candidate.clone();
            } else {
                // Rejection: use main model prediction instead
                let next_token = Array1::from_elem(1, main_prediction);
                return Ok((accepted, next_token));
            }
        }

        // All candidates accepted, generate one more with main model
        let main_logits = self
            .main_model
            .step(&current)
            .map_err(|e| InferenceError::ForwardError(e.to_string()))?;

        let main_prediction = self.main_sampler.sample(&main_logits)?;
        let next_token = Array1::from_elem(1, main_prediction);

        Ok((accepted, next_token))
    }

    /// Get acceptance rate (ratio of accepted to total draft tokens)
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.total_tokens as f32
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.total_tokens = 0;
        self.accepted_tokens = 0;
    }

    /// Get configuration
    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kizzasi_model::s4::{S4Config, S4D};

    #[test]
    fn test_speculative_config() {
        let config = SpeculativeConfig::new()
            .num_draft_tokens(5)
            .draft_temperature(0.8)
            .main_temperature(1.2)
            .greedy_verification(false);

        assert_eq!(config.num_draft_tokens, 5);
        assert!((config.draft_temperature - 0.8).abs() < 1e-6);
        assert!((config.main_temperature - 1.2).abs() < 1e-6);
        assert!(!config.greedy_verification);
    }

    #[test]
    fn test_speculative_decoder_creation() {
        // Create a small draft model
        let draft_config = S4Config::new()
            .input_dim(1)
            .hidden_dim(32)
            .state_dim(8)
            .num_layers(1)
            .diagonal(true);
        let draft_model = S4D::new(draft_config).unwrap();

        // Create a larger main model
        let main_config = S4Config::new()
            .input_dim(1)
            .hidden_dim(64)
            .state_dim(16)
            .num_layers(2)
            .diagonal(true);
        let main_model = S4D::new(main_config).unwrap();

        let config = SpeculativeConfig::new().num_draft_tokens(3);

        let decoder = SpeculativeDecoder::new(Box::new(main_model), Box::new(draft_model), config);

        assert_eq!(decoder.config().num_draft_tokens, 3);
        assert_eq!(decoder.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_speculative_generation() {
        // Create models
        let draft_config = S4Config::new()
            .input_dim(1)
            .hidden_dim(32)
            .state_dim(8)
            .num_layers(1)
            .diagonal(true);
        let draft_model = S4D::new(draft_config).unwrap();

        let main_config = S4Config::new()
            .input_dim(1)
            .hidden_dim(64)
            .state_dim(16)
            .num_layers(2)
            .diagonal(true);
        let main_model = S4D::new(main_config).unwrap();

        let config = SpeculativeConfig::new().num_draft_tokens(2);

        let mut decoder =
            SpeculativeDecoder::new(Box::new(main_model), Box::new(draft_model), config);

        let input = Array1::from_vec(vec![0.5]);
        let result = decoder.generate(&input, 10);

        assert!(result.is_ok());
        let sequence = result.unwrap();
        assert_eq!(sequence.len(), 10);

        // Acceptance rate should be between 0 and 1
        let acc_rate = decoder.acceptance_rate();
        assert!((0.0..=1.0).contains(&acc_rate));
    }
}
