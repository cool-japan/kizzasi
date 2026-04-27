//! H3: Hungry Hungry Hippos
//!
//! H3 is a state space model that uses shift SSMs with multiplicative interactions,
//! achieving strong performance on language modeling while maintaining linear complexity.
//!
//! # Key Features
//!
//! - **Shift SSM**: Simple shift operation instead of complex SSM computations
//! - **Multiplicative interactions**: Gating mechanisms for improved expressiveness
//! - **Linear complexity**: O(L) for sequence length L
//! - **Hardware-efficient**: Optimized for modern accelerators
//!
//! # Architecture
//!
//! ```text
//! Input → [Linear] → [ShiftSSM] → [Mult Gate] → [Linear] → Output
//!                        ↓
//!                    [Shift Buffer]
//! ```
//!
//! # Shift SSM Formulation
//!
//! Instead of complex state space dynamics, H3 uses:
//! ```text
//! y[t] = shift(x[t-k..t]) ⊙ gate(x[t])
//! ```
//!
//! Where shift is a learned linear combination of shifted inputs.
//!
//! # References
//!
//! - H3 paper: "Hungry Hungry Hippos: Towards Language Modeling with State Space Models"
//! - <https://arxiv.org/abs/2212.14052>

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{silu, CoreResult, HiddenState, LayerNorm, NormType, SignalPredictor};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, RngExt};
use std::collections::VecDeque;

#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

/// Configuration for H3 model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct H3Config {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// SSM state dimension
    pub ssm_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Shift distance (how far back to look)
    pub shift_distance: usize,
    /// Number of heads for multi-head shift SSM
    pub num_heads: usize,
}

impl H3Config {
    /// Create default H3 configuration
    pub fn new(input_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            ssm_dim: 64,
            num_layers,
            shift_distance: 4,
            num_heads: 4,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> ModelResult<()> {
        if self.hidden_dim == 0 {
            return Err(ModelError::invalid_config("hidden_dim must be > 0"));
        }
        if self.ssm_dim == 0 {
            return Err(ModelError::invalid_config("ssm_dim must be > 0"));
        }
        if self.num_layers == 0 {
            return Err(ModelError::invalid_config("num_layers must be > 0"));
        }
        if self.shift_distance == 0 {
            return Err(ModelError::invalid_config("shift_distance must be > 0"));
        }
        if self.num_heads == 0 {
            return Err(ModelError::invalid_config("num_heads must be > 0"));
        }
        if !self.hidden_dim.is_multiple_of(self.num_heads) {
            return Err(ModelError::invalid_config(
                "hidden_dim must be divisible by num_heads",
            ));
        }
        Ok(())
    }
}

/// Shift SSM block - core of H3
struct ShiftSSM {
    /// Head dimension
    head_dim: usize,
    /// Shift distance
    shift_distance: usize,
    /// Shift weights for each position [shift_distance, head_dim]
    shift_weights: Array2<f32>,
    /// History buffer for shifts
    history: VecDeque<Array1<f32>>,
}

impl ShiftSSM {
    /// Create new Shift SSM
    fn new(head_dim: usize, shift_distance: usize) -> Self {
        let mut rng = rng();

        // Initialize shift weights
        let scale = (1.0 / shift_distance as f32).sqrt();
        let shift_weights = Array2::from_shape_fn((shift_distance, head_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Initialize history buffer
        let history = VecDeque::with_capacity(shift_distance);

        Self {
            head_dim,
            shift_distance,
            shift_weights,
            history,
        }
    }

    /// Forward pass through shift SSM
    fn forward(&mut self, x: &Array1<f32>) -> Array1<f32> {
        // Add current input to history
        self.history.push_back(x.clone());

        // Keep only the last shift_distance elements
        while self.history.len() > self.shift_distance {
            self.history.pop_front();
        }

        // Compute weighted sum of shifted inputs
        let mut output = Array1::zeros(self.head_dim);
        for (i, hist_x) in self.history.iter().enumerate() {
            let weight_row = self.shift_weights.row(i);
            output = output + hist_x * &weight_row;
        }

        output
    }

    /// Reset history
    fn reset(&mut self) {
        self.history.clear();
    }
}

/// H3 layer with multi-head shift SSM
struct H3Layer {
    /// Number of heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Input projection
    input_proj: Array2<f32>,
    /// Shift SSMs (one per head)
    shift_ssms: Vec<ShiftSSM>,
    /// Gate projection for multiplicative interaction
    gate_proj: Array2<f32>,
    /// Output projection
    output_proj: Array2<f32>,
    /// Layer normalization
    layer_norm: LayerNorm,
}

impl H3Layer {
    /// Create a new H3 layer
    fn new(config: &H3Config) -> Self {
        let mut rng = rng();
        let num_heads = config.num_heads;
        let head_dim = config.hidden_dim / num_heads;

        // Input projection
        let scale = (2.0 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let input_proj = Array2::from_shape_fn((config.input_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Create shift SSMs for each head
        let shift_ssms = (0..num_heads)
            .map(|_| ShiftSSM::new(head_dim, config.shift_distance))
            .collect();

        // Gate projection
        let scale = (2.0 / (config.hidden_dim + config.hidden_dim) as f32).sqrt();
        let gate_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Output projection
        let scale = (2.0 / (config.hidden_dim + config.input_dim) as f32).sqrt();
        let output_proj = Array2::from_shape_fn((config.hidden_dim, config.input_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Layer normalization
        let layer_norm = LayerNorm::new(config.hidden_dim, NormType::RMSNorm);

        Self {
            num_heads,
            head_dim,
            input_proj,
            shift_ssms,
            gate_proj,
            output_proj,
            layer_norm,
        }
    }

    /// Forward pass
    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Project input to hidden dimension
        let hidden = x.dot(&self.input_proj);

        // Split into heads and process through shift SSMs
        let mut ssm_outputs = Vec::with_capacity(self.num_heads);
        for (head_idx, ssm) in self.shift_ssms.iter_mut().enumerate() {
            let start = head_idx * self.head_dim;
            let end = start + self.head_dim;
            let head_input = hidden.slice(s![start..end]).to_owned();
            ssm_outputs.push(ssm.forward(&head_input));
        }

        // Concatenate head outputs
        let mut ssm_output = Array1::zeros(self.num_heads * self.head_dim);
        for (head_idx, head_out) in ssm_outputs.iter().enumerate() {
            let start = head_idx * self.head_dim;
            let end = start + self.head_dim;
            ssm_output.slice_mut(s![start..end]).assign(head_out);
        }

        // Multiplicative gating
        let gate = hidden.dot(&self.gate_proj);
        let gate_activated = silu(&gate);
        let gated = &ssm_output * &gate_activated;

        // Layer normalization
        let normed = self.layer_norm.forward(&gated);

        // Output projection with residual
        let output = normed.dot(&self.output_proj) + x;

        Ok(output)
    }

    /// Reset layer state
    fn reset(&mut self) {
        for ssm in &mut self.shift_ssms {
            ssm.reset();
        }
    }
}

/// H3 model with multiple layers
pub struct H3 {
    config: H3Config,
    layers: Vec<H3Layer>,
}

impl H3 {
    /// Create a new H3 model
    #[instrument(skip(config), fields(input_dim = config.input_dim, hidden_dim = config.hidden_dim, num_layers = config.num_layers))]
    pub fn new(config: H3Config) -> ModelResult<Self> {
        debug!("Creating new H3 model");
        config.validate()?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            trace!("Initializing H3 layer {}", layer_idx);
            layers.push(H3Layer::new(&config));
        }
        debug!("Initialized {} H3 layers", layers.len());

        debug!("H3 model created successfully");
        Ok(Self { config, layers })
    }

    /// Get configuration
    pub fn config(&self) -> &H3Config {
        &self.config
    }
}

impl SignalPredictor for H3 {
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let mut x = input.clone();

        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }

        Ok(x)
    }

    #[instrument(skip(self))]
    fn reset(&mut self) {
        debug!("Resetting H3 model state");
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    fn context_window(&self) -> usize {
        // H3 has limited context via shift buffer
        self.config.shift_distance * self.config.num_layers
    }
}

impl AutoregressiveModel for H3 {
    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn state_dim(&self) -> usize {
        self.config.ssm_dim
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn model_type(&self) -> ModelType {
        ModelType::S4 // H3 is an SSM variant, closest to S4
    }

    fn get_states(&self) -> Vec<HiddenState> {
        self.layers
            .iter()
            .map(|layer| {
                // Collect shift buffer histories into a state
                let total_size =
                    layer.shift_ssms.len() * layer.head_dim * self.config.shift_distance;
                let mut state_vec = vec![0.0; total_size];

                let mut offset = 0;
                for ssm in &layer.shift_ssms {
                    for hist in &ssm.history {
                        if let Some(hist_slice) = hist.as_slice() {
                            state_vec[offset..offset + hist.len()].copy_from_slice(hist_slice);
                        } else {
                            for (i, &val) in hist.iter().enumerate() {
                                state_vec[offset + i] = val;
                            }
                        }
                        offset += hist.len();
                    }
                    // Pad if history is shorter than shift_distance
                    offset += (self.config.shift_distance - ssm.history.len()) * layer.head_dim;
                }

                let state_1d = Array1::from_vec(state_vec);
                let state_2d = state_1d.insert_axis(scirs2_core::ndarray::Axis(0));
                let mut hidden_state = HiddenState::new(
                    self.config.hidden_dim,
                    state_2d.len_of(scirs2_core::ndarray::Axis(1)),
                );
                hidden_state.update(state_2d);
                hidden_state
            })
            .collect()
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != self.config.num_layers {
            return Err(ModelError::state_count_mismatch(
                "H3",
                self.config.num_layers,
                states.len(),
            ));
        }

        for (layer, state) in self.layers.iter_mut().zip(states.iter()) {
            let state_2d = state.state();
            if state_2d.nrows() > 0 {
                let state_1d = state_2d.row(0).to_owned();
                let mut offset = 0;

                for ssm in &mut layer.shift_ssms {
                    ssm.history.clear();
                    for _ in 0..self
                        .config
                        .shift_distance
                        .min(state_1d.len() / layer.head_dim)
                    {
                        if offset + layer.head_dim <= state_1d.len() {
                            let hist_vec: Vec<f32> =
                                state_1d.slice(s![offset..offset + layer.head_dim]).to_vec();
                            ssm.history.push_back(Array1::from_vec(hist_vec));
                            offset += layer.head_dim;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// Import ndarray slice macro
use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h3_creation() {
        let config = H3Config::new(32, 64, 2);
        let model = H3::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_h3_forward() {
        let config = H3Config::new(32, 64, 2);
        let mut model = H3::new(config).expect("Failed to create H3 model");

        let input = Array1::from_vec(vec![1.0; 32]);
        let output = model.step(&input);
        assert!(output.is_ok());
        assert_eq!(output.expect("Failed to get output").len(), 32);
    }

    #[test]
    fn test_h3_reset() {
        let config = H3Config::new(32, 64, 2);
        let mut model = H3::new(config).expect("Failed to create H3 model");

        let input = Array1::from_vec(vec![1.0; 32]);
        let _output1 = model.step(&input).expect("Failed to get output1");

        model.reset();

        let output2 = model.step(&input).expect("Failed to get output2");
        assert_eq!(output2.len(), 32);
    }

    #[test]
    fn test_invalid_config() {
        let mut config = H3Config::new(32, 64, 2);
        config.num_heads = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_h3_context_window() {
        let config = H3Config::new(32, 64, 3);
        let model = H3::new(config.clone()).expect("Failed to create H3 model");
        assert_eq!(
            model.context_window(),
            config.shift_distance * config.num_layers
        );
    }

    #[test]
    fn test_h3_state_management() {
        let config = H3Config::new(32, 64, 2);
        let mut model = H3::new(config).expect("Failed to create H3 model");

        // Run a few steps
        let input = Array1::from_vec(vec![0.5; 32]);
        for _ in 0..5 {
            let _ = model.step(&input).expect("Failed to step H3 model");
        }

        // Get states
        let states = model.get_states();
        assert_eq!(states.len(), 2);

        // Reset and set states
        model.reset();
        let result = model.set_states(states);
        assert!(result.is_ok());
    }
}
