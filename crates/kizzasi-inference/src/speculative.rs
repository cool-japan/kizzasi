//! Speculative decoding
//!
//! Speculative decoding uses a smaller "draft" model to propose candidate tokens,
//! then verifies them with the main model, keeping the main model's output
//! distribution intact.
//!
//! ## Algorithm
//!
//! 1. The draft model generates K candidate tokens, recording its proposal
//!    distribution `q` and a state snapshot per step.
//! 2. The main model scores the candidates one position at a time, producing the
//!    target distribution `p` at each position.
//! 3. Candidate `x` is accepted with probability `min(1, p(x) / q(x))`; on
//!    rejection a replacement is drawn from the normalised residual
//!    `max(p - q, 0)`. This is the acceptance rule of Leviathan et al. (2023) and
//!    Chen et al. (2023) and makes the generated sequence distributed exactly as
//!    the main model alone would generate it.
//! 4. The draft model's state is rolled back to the accepted prefix, so rejected
//!    candidates never leak into subsequent rounds.
//!
//! With `greedy_verification` the acceptance test degenerates to "the candidate
//! equals the main model's argmax", matching greedy decoding by the main model.
//!
//! ## Performance
//!
//! Speculative decoding pays off when the main model can score the whole K-token
//! candidate prefix in **one** forward pass. The [`AutoregressiveModel`] interface
//! consumes one signal vector per call, so verification here costs one main-model
//! step per candidate position: this implementation buys **no wall-clock speedup**
//! over plain autoregressive generation with the recurrent backends in this
//! workspace. What it does provide is the correct speculative sampling semantics —
//! draft proposals, the acceptance ratio, and residual resampling — so that a
//! backend gaining batched prefix scoring becomes faster without changing the
//! statistics of the output.

use crate::error::{InferenceError, InferenceResult};
use crate::sampling::{make_sampler_rng, softmax, Sampler, SamplingConfig};
use kizzasi_core::HiddenState;
use kizzasi_model::AutoregressiveModel;
use scirs2_core::ndarray::Array1;
use scirs2_core::random::RngExt;

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
    /// Seed for the acceptance-test RNG (reproducible when set)
    pub seed: Option<u64>,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_draft_tokens: 4,
            draft_temperature: 1.0,
            main_temperature: 1.0,
            greedy_verification: true,
            seed: None,
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

    /// Seed the acceptance-test RNG for reproducible runs
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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
    /// RNG driving the acceptance test and residual resampling
    rng: scirs2_core::random::StdRng,
    /// Statistics
    total_tokens: usize,
    accepted_tokens: usize,
}

/// One draft round: candidates, their proposal distributions, and the draft
/// model's state after each step (used to roll back rejected candidates).
struct DraftRound {
    candidates: Vec<Array1<f32>>,
    proposals: Vec<Array1<f32>>,
    states: Vec<Vec<HiddenState>>,
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

        let rng = make_sampler_rng(config.seed);

        Self {
            main_model,
            draft_model,
            config,
            draft_sampler,
            main_sampler,
            rng,
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
        if self.config.num_draft_tokens == 0 {
            return Err(InferenceError::InvalidConfiguration(
                "num_draft_tokens must be at least 1".to_string(),
            ));
        }

        let mut sequence = Vec::with_capacity(max_tokens);
        let mut current = input.clone();

        while sequence.len() < max_tokens {
            // Generate draft tokens
            let round = self.generate_draft_tokens(&current)?;

            // Verify with main model
            let (accepted, next_token) = self.verify_candidates(&current, &round)?;

            // Roll the draft model back to the accepted prefix so rejected
            // candidates do not permanently poison its state.
            self.rollback_draft(&round, accepted)?;

            // Add accepted tokens to sequence
            for token in round.candidates.iter().take(accepted) {
                sequence.push(token.clone());
                if sequence.len() >= max_tokens {
                    break;
                }
            }

            // Update statistics
            self.total_tokens += round.candidates.len();
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
    ///
    /// Records the proposal distribution `q` behind each candidate and a snapshot
    /// of the draft model's state after each step, both of which the verification
    /// step needs.
    fn generate_draft_tokens(&mut self, input: &Array1<f32>) -> InferenceResult<DraftRound> {
        let num_tokens = self.config.num_draft_tokens;
        let mut candidates = Vec::with_capacity(num_tokens);
        let mut proposals = Vec::with_capacity(num_tokens);
        let mut states = Vec::with_capacity(num_tokens);
        let mut current = input.clone();

        for _ in 0..num_tokens {
            let logits = self
                .draft_model
                .step(&current)
                .map_err(|e| InferenceError::ForwardError(e.to_string()))?;

            let sampled = self.draft_sampler.sample(&logits)?;
            let token = Array1::from_elem(1, sampled);

            proposals.push(scaled_softmax(&logits, self.config.draft_temperature));
            states.push(self.draft_model.get_states());
            candidates.push(token.clone());
            current = token;
        }

        Ok(DraftRound {
            candidates,
            proposals,
            states,
        })
    }

    /// Restore the draft model's state to the accepted prefix
    ///
    /// `states[i]` is the state after the draft model consumed the prefix ending at
    /// candidate `i - 1`, which is exactly the state the next round must start from
    /// when `i` candidates were accepted. When every candidate is accepted the live
    /// state is already correct.
    fn rollback_draft(&mut self, round: &DraftRound, accepted: usize) -> InferenceResult<()> {
        if accepted >= round.states.len() {
            return Ok(());
        }
        self.draft_model
            .set_states(round.states[accepted].clone())
            .map_err(|e| InferenceError::ForwardError(e.to_string()))?;
        Ok(())
    }

    /// Verify candidates using the main model
    ///
    /// Implements the speculative-sampling acceptance test: candidate `x` proposed
    /// with probability `q(x)` is accepted with probability `min(1, p(x) / q(x))`
    /// where `p` is the main model's distribution at that position; on rejection the
    /// replacement is drawn from the normalised residual `max(p - q, 0)`. The
    /// resulting sequence is distributed exactly as the main model's own sampling.
    ///
    /// Returns (number of accepted tokens, next token to use).
    fn verify_candidates(
        &mut self,
        input: &Array1<f32>,
        round: &DraftRound,
    ) -> InferenceResult<(usize, Array1<f32>)> {
        let mut current = input.clone();
        let mut accepted = 0;

        for (candidate, proposal) in round.candidates.iter().zip(round.proposals.iter()) {
            // Get main model prediction
            let main_logits = self
                .main_model
                .step(&current)
                .map_err(|e| InferenceError::ForwardError(e.to_string()))?;

            if self.config.greedy_verification {
                // Greedy verification reproduces greedy decoding by the main model:
                // the candidate must be the main model's own choice.
                let main_prediction = self.main_sampler.sample(&main_logits)?;
                if (candidate[0] - main_prediction).abs() < 1e-6 {
                    accepted += 1;
                    current = candidate.clone();
                    continue;
                }
                return Ok((accepted, Array1::from_elem(1, main_prediction)));
            }

            let target = scaled_softmax(&main_logits, self.config.main_temperature);
            if target.len() != proposal.len() {
                return Err(InferenceError::InvalidConfiguration(format!(
                    "draft and main models produce different output spaces ({} vs {}); \
                     speculative verification requires a shared output space",
                    proposal.len(),
                    target.len()
                )));
            }

            let index = token_index(candidate[0], target.len())?;
            let proposal_prob = proposal[index];
            let target_prob = target[index];

            // min(1, p(x) / q(x)); a candidate the draft could not have produced
            // (q(x) == 0) is rejected rather than dividing by zero.
            let acceptance = if proposal_prob > 0.0 {
                (target_prob / proposal_prob).min(1.0)
            } else {
                0.0
            };

            if self.rng.random::<f32>() < acceptance {
                accepted += 1;
                current = candidate.clone();
                continue;
            }

            let next_token = self.sample_residual(&target, proposal)?;
            return Ok((accepted, next_token));
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

    /// Draw a replacement token from the normalised residual `max(p - q, 0)`
    ///
    /// When the residual carries no mass (the two distributions coincide on the
    /// support) the replacement is drawn from the target distribution `p`, which is
    /// the limit of the residual rule and keeps the output distributed as `p`.
    fn sample_residual(
        &mut self,
        target: &Array1<f32>,
        proposal: &Array1<f32>,
    ) -> InferenceResult<Array1<f32>> {
        let residual = target - proposal;
        let residual = residual.mapv(|x| if x > 0.0 { x } else { 0.0 });
        let mass: f32 = residual.sum();

        let distribution = if mass > f32::EPSILON {
            residual / mass
        } else {
            target.clone()
        };

        let uniform: f32 = self.rng.random::<f32>();
        let mut cumulative = 0.0;
        for (index, &probability) in distribution.iter().enumerate() {
            cumulative += probability;
            if uniform < cumulative {
                return Ok(Array1::from_elem(1, index as f32));
            }
        }

        // Rounding can leave the cumulative sum just below 1.0.
        let last = distribution.len().saturating_sub(1);
        Ok(Array1::from_elem(1, last as f32))
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

    /// Snapshot the draft model's hidden states
    pub fn draft_states(&self) -> Vec<HiddenState> {
        self.draft_model.get_states()
    }

    /// Snapshot the main model's hidden states
    pub fn main_states(&self) -> Vec<HiddenState> {
        self.main_model.get_states()
    }
}

/// Softmax with a temperature applied to the logits
///
/// A temperature at or below zero collapses the distribution onto the argmax,
/// mirroring [`Sampler`]'s handling and avoiding `inf`/`NaN` from dividing by zero.
fn scaled_softmax(logits: &Array1<f32>, temperature: f32) -> Array1<f32> {
    if temperature <= 1e-6 {
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let mut point_mass = Array1::zeros(logits.len());
        if let Some(slot) = point_mass.get_mut(argmax) {
            *slot = 1.0;
        }
        return point_mass;
    }

    if (temperature - 1.0).abs() <= 1e-6 {
        softmax(logits)
    } else {
        softmax(&logits.mapv(|x| x / temperature))
    }
}

/// Interpret a sampled token value as an index into a distribution of `len` entries
fn token_index(value: f32, len: usize) -> InferenceResult<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value as usize >= len {
        return Err(InferenceError::InvalidConfiguration(format!(
            "candidate token {} is not a valid index into a {}-entry distribution; \
             speculative decoding requires samplers that emit categorical indices",
            value, len
        )));
    }
    Ok(value as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kizzasi_model::s4::{S4Config, S4D};
    use scirs2_core::ndarray::Array2;

    /// Deterministic model over a fixed categorical distribution.
    ///
    /// Real signal models in this workspace map a `d`-dimensional input to a
    /// `d`-dimensional output, which degenerates to a one-symbol vocabulary for the
    /// scalar tokens speculative decoding feeds back. This stub decouples the two so
    /// the acceptance rule can be exercised over a real multi-symbol distribution.
    /// Its hidden state records how many steps it has consumed, which makes draft
    /// rollback observable.
    struct StubModel {
        logits: Array1<f32>,
        steps: usize,
    }

    impl StubModel {
        fn new(logits: Vec<f32>) -> Self {
            Self {
                logits: Array1::from_vec(logits),
                steps: 0,
            }
        }
    }

    impl kizzasi_core::SignalPredictor for StubModel {
        fn step(&mut self, _input: &Array1<f32>) -> kizzasi_core::CoreResult<Array1<f32>> {
            self.steps += 1;
            Ok(self.logits.clone())
        }

        fn reset(&mut self) {
            self.steps = 0;
        }

        fn context_window(&self) -> usize {
            usize::MAX
        }
    }

    impl AutoregressiveModel for StubModel {
        fn hidden_dim(&self) -> usize {
            1
        }

        fn state_dim(&self) -> usize {
            1
        }

        fn num_layers(&self) -> usize {
            1
        }

        fn model_type(&self) -> kizzasi_model::ModelType {
            kizzasi_model::ModelType::S4D
        }

        fn get_states(&self) -> Vec<HiddenState> {
            let mut state = HiddenState::new(1, 1);
            state.update(Array2::from_elem((1, 1), self.steps as f32));
            vec![state]
        }

        fn set_states(&mut self, states: Vec<HiddenState>) -> kizzasi_model::ModelResult<()> {
            if let Some(first) = states.first() {
                self.steps = first.state()[[0, 0]] as usize;
            }
            Ok(())
        }
    }

    fn consumed_steps(states: &[HiddenState]) -> usize {
        states
            .first()
            .map(|s| s.state()[[0, 0]] as usize)
            .unwrap_or(0)
    }

    /// Regression: acceptance used a `|candidate - prediction| < 0.5` distance test
    /// on sampled *indices*, which is neither the speculative-sampling acceptance
    /// rule nor distribution preserving.
    #[test]
    fn test_matching_distributions_accept_every_candidate() {
        let main_model = Box::new(StubModel::new(vec![0.0, 0.0, 20.0]));
        let draft_model = Box::new(StubModel::new(vec![0.0, 0.0, 20.0]));

        let config = SpeculativeConfig::new()
            .num_draft_tokens(4)
            .greedy_verification(false)
            .seed(7);
        let mut decoder = SpeculativeDecoder::new(main_model, draft_model, config);

        let sequence = decoder
            .generate(&Array1::from_vec(vec![0.0]), 8)
            .expect("generation must succeed");

        assert_eq!(sequence.len(), 8);
        for token in &sequence {
            assert_eq!(token[0], 2.0, "every token must come from the shared mode");
        }
        assert!(
            (decoder.acceptance_rate() - 1.0).abs() < 1e-6,
            "identical distributions must accept everything, got {}",
            decoder.acceptance_rate()
        );
    }

    /// Regression: a candidate the target model assigns (almost) no mass to used to
    /// be accepted whenever the sampled indices happened to be within 0.5.
    #[test]
    fn test_disjoint_distributions_reject_and_resample_from_residual() {
        // Draft always proposes index 2, the target concentrates on index 0.
        let main_model = Box::new(StubModel::new(vec![20.0, 0.0, 0.0]));
        let draft_model = Box::new(StubModel::new(vec![0.0, 0.0, 20.0]));

        let config = SpeculativeConfig::new()
            .num_draft_tokens(4)
            .greedy_verification(false)
            .seed(11);
        let mut decoder = SpeculativeDecoder::new(main_model, draft_model, config);

        let sequence = decoder
            .generate(&Array1::from_vec(vec![0.0]), 6)
            .expect("generation must succeed");

        assert_eq!(sequence.len(), 6);
        for token in &sequence {
            assert_eq!(
                token[0], 0.0,
                "rejected candidates must be replaced from the target's support"
            );
        }
        assert_eq!(
            decoder.acceptance_rate(),
            0.0,
            "candidates outside the target support must never be accepted"
        );
    }

    /// Regression: the draft model kept the state of rejected candidates, so its
    /// history diverged permanently from the accepted sequence.
    #[test]
    fn test_draft_state_is_rolled_back_to_accepted_prefix() {
        let main_model = Box::new(StubModel::new(vec![20.0, 0.0, 0.0]));
        let draft_model = Box::new(StubModel::new(vec![0.0, 0.0, 20.0]));

        let num_draft = 4;
        let config = SpeculativeConfig::new()
            .num_draft_tokens(num_draft)
            .greedy_verification(false)
            .seed(3);
        let mut decoder = SpeculativeDecoder::new(main_model, draft_model, config);

        let max_tokens = 5;
        decoder
            .generate(&Array1::from_vec(vec![0.0]), max_tokens)
            .expect("generation must succeed");

        // Every round rejects the first candidate, so the draft model must retain
        // exactly one consumed step per emitted token — not `num_draft` of them.
        assert_eq!(
            consumed_steps(&decoder.draft_states()),
            max_tokens,
            "draft state must follow the accepted prefix, not every drafted candidate"
        );
        assert!(consumed_steps(&decoder.draft_states()) < max_tokens * num_draft);
    }

    #[test]
    fn test_token_index_rejects_non_indices() {
        assert!(token_index(1.0, 3).is_ok());
        assert!(token_index(2.5, 3).is_err());
        assert!(token_index(-1.0, 3).is_err());
        assert!(token_index(3.0, 3).is_err());
        assert!(token_index(f32::NAN, 3).is_err());
    }

    #[test]
    fn test_zero_draft_tokens_is_rejected() {
        let main_model = Box::new(StubModel::new(vec![1.0, 2.0]));
        let draft_model = Box::new(StubModel::new(vec![1.0, 2.0]));
        let config = SpeculativeConfig::new().num_draft_tokens(0);
        let mut decoder = SpeculativeDecoder::new(main_model, draft_model, config);

        assert!(matches!(
            decoder.generate(&Array1::from_vec(vec![0.0]), 4),
            Err(InferenceError::InvalidConfiguration(_))
        ));
    }

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
