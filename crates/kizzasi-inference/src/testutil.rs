//! Deterministic models used by the crate's own unit tests
//!
//! The production backends are unsuitable for asserting properties of the
//! *inference* layer:
//!
//! - `S4D`/`RWKV` initialise their weights from an unseeded RNG, so two
//!   separately constructed models never agree;
//! - not every backend round-trips its whole recurrent state through
//!   [`HiddenState`] (S4D's short-convolution history is not part of it), which
//!   would mask — or fake — the isolation this crate provides.
//!
//! The models here are exact, reproducible and fully snapshot-able, so a failing
//! assertion points at this crate rather than at a model implementation.

use kizzasi_core::{CoreError, CoreResult, HiddenState, SignalPredictor};
use kizzasi_model::{AutoregressiveModel, ModelError, ModelResult, ModelType};
use scirs2_core::ndarray::{Array1, Array2};

/// One-dimensional leaky accumulator: `state <- decay * state + input`
///
/// The output is the new state, so any leakage of state between requests changes
/// the output immediately and visibly.
#[derive(Debug, Clone)]
pub struct CountingModel {
    accumulated: f32,
    decay: f32,
    failures_remaining: usize,
}

impl CountingModel {
    /// Create a model with the default decay of `0.5`
    pub fn new() -> Self {
        Self {
            accumulated: 0.0,
            decay: 0.5,
            failures_remaining: 0,
        }
    }

    /// Create a model whose first `count` steps fail
    ///
    /// Used to exercise the engine's error paths without relying on a real model
    /// misbehaving.
    pub fn failing_for(count: usize) -> Self {
        Self {
            failures_remaining: count,
            ..Self::new()
        }
    }

    /// Current accumulated value
    pub fn accumulated(&self) -> f32 {
        self.accumulated
    }
}

impl Default for CountingModel {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalPredictor for CountingModel {
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        if self.failures_remaining > 0 {
            self.failures_remaining -= 1;
            return Err(CoreError::InferenceError(
                "CountingModel was asked to fail".to_string(),
            ));
        }

        let value = input.first().copied().unwrap_or(0.0);
        self.accumulated = self.accumulated * self.decay + value;
        Ok(Array1::from_elem(1, self.accumulated))
    }

    fn reset(&mut self) {
        self.accumulated = 0.0;
    }

    fn context_window(&self) -> usize {
        usize::MAX
    }
}

impl AutoregressiveModel for CountingModel {
    fn hidden_dim(&self) -> usize {
        1
    }

    fn state_dim(&self) -> usize {
        1
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn model_type(&self) -> ModelType {
        ModelType::S4D
    }

    fn get_states(&self) -> Vec<HiddenState> {
        let mut state = HiddenState::new(1, 1);
        state.update(Array2::from_elem((1, 1), self.accumulated));
        vec![state]
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        match states.first() {
            Some(first) => {
                self.accumulated = first.state()[[0, 0]];
                Ok(())
            }
            None => Err(ModelError::state_count_mismatch("CountingModel", 1, 0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counting_model_state_round_trips() {
        let mut model = CountingModel::new();
        model
            .step(&Array1::from_elem(1, 1.0))
            .expect("step must succeed");
        let snapshot = model.get_states();

        model
            .step(&Array1::from_elem(1, 1.0))
            .expect("step must succeed");
        assert!((model.accumulated() - 1.5).abs() < 1e-6);

        model
            .set_states(snapshot)
            .expect("state restore must succeed");
        assert!((model.accumulated() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_counting_model_can_be_asked_to_fail() {
        let mut model = CountingModel::failing_for(1);
        assert!(model.step(&Array1::from_elem(1, 1.0)).is_err());
        assert!(model.step(&Array1::from_elem(1, 1.0)).is_ok());
    }
}
