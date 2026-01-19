//! Hidden state management for SSM

use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Represents the hidden state of the SSM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenState {
    /// The recurrent state tensor (batch, layers, hidden_dim, state_dim)
    state: Array2<f32>,
    /// Number of steps processed
    step_count: usize,
    /// Optional convolution history buffer (for causal conv layers)
    /// Note: The `default` attribute ensures backward compatibility with older checkpoints
    #[serde(default)]
    conv_history: Option<Vec<Vec<f32>>>,
}

impl HiddenState {
    /// Create a new hidden state with given dimensions
    pub fn new(hidden_dim: usize, state_dim: usize) -> Self {
        Self {
            state: Array2::zeros((hidden_dim, state_dim)),
            step_count: 0,
            conv_history: None,
        }
    }

    /// Reset the state to zeros
    pub fn reset(&mut self) {
        self.state.fill(0.0);
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

    /// Update the state with new values
    pub fn update(&mut self, new_state: Array2<f32>) {
        self.state = new_state;
        self.step_count += 1;
    }

    /// Get the current state
    pub fn state(&self) -> &Array2<f32> {
        &self.state
    }

    /// Get mutable reference to state
    pub fn state_mut(&mut self) -> &mut Array2<f32> {
        &mut self.state
    }

    /// Get the step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get a row of the state as 1D array
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
}
