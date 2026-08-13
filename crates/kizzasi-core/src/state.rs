//! Hidden state management for SSM
//!
//! A [`HiddenState`] carries **one recurrent state matrix per SSM layer**.
//! Layer 0's matrix is stored inline and is the one returned by [`HiddenState::state`]
//! / [`HiddenState::state_mut`], so single-layer callers (and every checkpoint
//! written before per-layer state existed) keep working unchanged. Additional
//! layers live in `extra_layer_states`, which deserialises to an empty vector for
//! older checkpoints.

use crate::error::{CoreError, CoreResult};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Represents the hidden state of the SSM
///
/// The state is per layer: a model with `L` layers owns `L` matrices of shape
/// `(hidden_dim, state_dim)`. Sharing a single matrix across layers destroys
/// every layer's memory except the last one, so [`SelectiveSSM`] always sizes
/// the state to its layer count via [`HiddenState::ensure_layers`].
///
/// [`SelectiveSSM`]: crate::SelectiveSSM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenState {
    /// Recurrent state matrix of layer 0, shape `(hidden_dim, state_dim)`
    state: Array2<f32>,
    /// Number of steps processed
    step_count: usize,
    /// Optional convolution history buffer (for causal conv layers)
    /// Note: The `default` attribute ensures backward compatibility with older checkpoints
    #[serde(default)]
    conv_history: Option<Vec<Vec<f32>>>,
    /// Recurrent state matrices of layers `1..num_layers`.
    ///
    /// Empty for single-layer models. Declared last and `#[serde(default)]` so
    /// checkpoints written before per-layer state existed still deserialise.
    #[serde(default)]
    extra_layer_states: Vec<Array2<f32>>,
}

impl HiddenState {
    /// Create a new single-layer hidden state with given dimensions
    pub fn new(hidden_dim: usize, state_dim: usize) -> Self {
        Self {
            state: Array2::zeros((hidden_dim, state_dim)),
            step_count: 0,
            conv_history: None,
            extra_layer_states: Vec::new(),
        }
    }

    /// Create a hidden state holding one `(hidden_dim, state_dim)` matrix per layer.
    ///
    /// `num_layers` is clamped to at least 1 — a state with zero layers cannot
    /// represent anything and would make [`state`](Self::state) unusable.
    pub fn new_layered(num_layers: usize, hidden_dim: usize, state_dim: usize) -> Self {
        let extra = num_layers.saturating_sub(1);
        let mut extra_layer_states = Vec::with_capacity(extra);
        for _ in 0..extra {
            extra_layer_states.push(Array2::zeros((hidden_dim, state_dim)));
        }
        Self {
            state: Array2::zeros((hidden_dim, state_dim)),
            step_count: 0,
            conv_history: None,
            extra_layer_states,
        }
    }

    /// Number of layers this state can hold (always >= 1)
    pub fn num_layers(&self) -> usize {
        1 + self.extra_layer_states.len()
    }

    /// Grow or shrink the state so it holds exactly `num_layers` matrices.
    ///
    /// Newly added layers are zero-initialised with the same shape as layer 0.
    /// `num_layers` is clamped to at least 1. Existing layers are left untouched
    /// so a resize mid-stream never silently discards live history for the
    /// layers that survive.
    pub fn ensure_layers(&mut self, num_layers: usize) {
        let wanted_extra = num_layers.saturating_sub(1);
        if self.extra_layer_states.len() > wanted_extra {
            self.extra_layer_states.truncate(wanted_extra);
            return;
        }
        let shape = (self.state.nrows(), self.state.ncols());
        while self.extra_layer_states.len() < wanted_extra {
            self.extra_layer_states.push(Array2::zeros(shape));
        }
    }

    /// Reset every layer's state to zeros
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        for layer in self.extra_layer_states.iter_mut() {
            layer.fill(0.0);
        }
        self.step_count = 0;
        if let Some(ref mut hist) = self.conv_history {
            for h in hist {
                h.fill(0.0);
            }
        }
    }

    /// Set the convolution history
    pub fn set_conv_history(&mut self, history: Vec<Vec<f32>>) {
        self.conv_history = Some(history);
    }

    /// Get the convolution history
    pub fn conv_history(&self) -> Option<&Vec<Vec<f32>>> {
        self.conv_history.as_ref()
    }

    /// Take the convolution history (moves ownership)
    pub fn take_conv_history(&mut self) -> Option<Vec<Vec<f32>>> {
        self.conv_history.take()
    }

    /// Replace layer 0's state and advance the step counter
    ///
    /// Equivalent to `update_layer(0, new_state)` followed by [`advance`](Self::advance).
    pub fn update(&mut self, new_state: Array2<f32>) {
        self.state = new_state;
        self.step_count += 1;
    }

    /// Replace the state matrix of layer `layer_idx` without touching the step counter
    ///
    /// # Errors
    /// Returns [`CoreError::DimensionMismatch`] when `layer_idx` is out of range.
    pub fn update_layer(&mut self, layer_idx: usize, new_state: Array2<f32>) -> CoreResult<()> {
        match self.layer_state_mut(layer_idx) {
            Some(slot) => {
                *slot = new_state;
                Ok(())
            }
            None => Err(CoreError::DimensionMismatch {
                expected: self.num_layers(),
                got: layer_idx,
            }),
        }
    }

    /// Advance the step counter by one without replacing any state matrix.
    ///
    /// Used by in-place recurrences that mutate the layer matrices directly via
    /// [`layer_state_mut`](Self::layer_state_mut) and therefore have nothing to
    /// hand back to [`update`](Self::update).
    pub fn advance(&mut self) {
        self.step_count += 1;
    }

    /// Get the current state of layer 0
    pub fn state(&self) -> &Array2<f32> {
        &self.state
    }

    /// Get mutable reference to layer 0's state
    pub fn state_mut(&mut self) -> &mut Array2<f32> {
        &mut self.state
    }

    /// Borrow the state matrix of layer `layer_idx`, or `None` when out of range
    pub fn layer_state(&self, layer_idx: usize) -> Option<&Array2<f32>> {
        match layer_idx {
            0 => Some(&self.state),
            n => self.extra_layer_states.get(n - 1),
        }
    }

    /// Mutably borrow the state matrix of layer `layer_idx`, or `None` when out of range
    pub fn layer_state_mut(&mut self, layer_idx: usize) -> Option<&mut Array2<f32>> {
        match layer_idx {
            0 => Some(&mut self.state),
            n => self.extra_layer_states.get_mut(n - 1),
        }
    }

    /// Get the step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get a row of layer 0's state as 1D array
    pub fn get_row(&self, idx: usize) -> Array1<f32> {
        self.state.row(idx).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hidden_state() {
        let mut state = HiddenState::new(256, 16);
        assert_eq!(state.step_count(), 0);
        assert_eq!(state.state().shape(), &[256, 16]);

        state.reset();
        assert_eq!(state.step_count(), 0);
    }

    #[test]
    fn test_single_layer_by_default() {
        let state = HiddenState::new(8, 4);
        assert_eq!(state.num_layers(), 1);
        assert!(state.layer_state(0).is_some());
        assert!(state.layer_state(1).is_none());
    }

    #[test]
    fn test_layered_state_is_independent_per_layer() {
        let mut state = HiddenState::new_layered(3, 4, 2);
        assert_eq!(state.num_layers(), 3);

        for layer_idx in 0..3 {
            let layer = state
                .layer_state_mut(layer_idx)
                .expect("layer must exist within num_layers");
            layer.fill(layer_idx as f32 + 1.0);
        }

        for layer_idx in 0..3 {
            let layer = state
                .layer_state(layer_idx)
                .expect("layer must exist within num_layers");
            assert!(layer.iter().all(|&v| v == layer_idx as f32 + 1.0));
        }

        // Layer 0 must be the one exposed by the legacy accessor.
        assert!(state.state().iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_reset_clears_every_layer() {
        let mut state = HiddenState::new_layered(3, 4, 2);
        for layer_idx in 0..3 {
            if let Some(layer) = state.layer_state_mut(layer_idx) {
                layer.fill(7.0);
            }
        }
        state.advance();
        state.reset();

        assert_eq!(state.step_count(), 0);
        for layer_idx in 0..3 {
            let layer = state
                .layer_state(layer_idx)
                .expect("layer must exist within num_layers");
            assert!(
                layer.iter().all(|&v| v == 0.0),
                "layer {layer_idx} not zeroed"
            );
        }
    }

    #[test]
    fn test_ensure_layers_grows_and_shrinks() {
        let mut state = HiddenState::new(4, 2);
        state.ensure_layers(4);
        assert_eq!(state.num_layers(), 4);
        assert_eq!(
            state
                .layer_state(3)
                .expect("layer 3 exists after ensure_layers(4)")
                .shape(),
            &[4, 2]
        );

        state.ensure_layers(2);
        assert_eq!(state.num_layers(), 2);
        assert!(state.layer_state(2).is_none());

        // Zero is clamped to one usable layer.
        state.ensure_layers(0);
        assert_eq!(state.num_layers(), 1);
    }

    #[test]
    fn test_advance_bumps_step_count_only() {
        let mut state = HiddenState::new_layered(2, 3, 2);
        if let Some(layer) = state.layer_state_mut(1) {
            layer.fill(2.0);
        }
        state.advance();
        assert_eq!(state.step_count(), 1);
        assert!(state
            .layer_state(1)
            .expect("layer 1 exists")
            .iter()
            .all(|&v| v == 2.0));
    }

    #[test]
    fn test_update_layer_reports_out_of_range() {
        let mut state = HiddenState::new_layered(2, 3, 2);
        assert!(state.update_layer(1, Array2::zeros((3, 2))).is_ok());
        assert!(state.update_layer(5, Array2::zeros((3, 2))).is_err());
    }

    #[test]
    fn test_legacy_checkpoint_deserialises_without_extra_layers() {
        // Older checkpoints carry only `state` / `step_count`; the per-layer
        // vector must default to empty rather than failing to parse. Build the
        // legacy payload by stripping the newer fields from a current one so the
        // test does not hard-code ndarray's serialisation format.
        let mut current = HiddenState::new_layered(3, 2, 2);
        current.advance();
        let value: serde_json::Value =
            serde_json::to_value(&current).expect("HiddenState must serialise");
        let mut map = match value {
            serde_json::Value::Object(map) => map,
            other => panic!("HiddenState must serialise to a JSON object, got {other:?}"),
        };
        map.remove("extra_layer_states");
        map.remove("conv_history");

        let state: HiddenState = serde_json::from_value(serde_json::Value::Object(map))
            .expect("legacy checkpoint must still deserialise");
        assert_eq!(state.num_layers(), 1);
        assert_eq!(state.step_count(), 1);
        assert_eq!(state.state().shape(), &[2, 2]);
    }
}
