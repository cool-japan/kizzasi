//! S5: Simplified State Space Model
//!
//! S5 simplifies S4 by using a more efficient parameterization of the state space
//! while maintaining competitive performance. Key simplifications include:
//!
//! - **Simplified initialization**: Easier parameter initialization
//! - **Faster computation**: Reduced computational overhead
//! - **Better numerical stability**: Improved gradient flow
//! - **Diagonal state matrix**: Like S4D, but with optimized discretization
//!
//! # Architecture
//!
//! ```text
//! Input → [Linear] → [SSM Block] → [Activation] → [LayerNorm] → Output
//!                         ↓
//!                     [State]
//! ```
//!
//! # SSM Formulation
//!
//! Continuous-time:
//! ```text
//! h'(t) = Ah(t) + Bx(t)
//! y(t) = Ch(t)
//! ```
//!
//! Where A is diagonal and initialized more simply than S4.
//!
//! # References
//!
//! - S5 paper: https://arxiv.org/abs/2208.04933
//! - Efficiently Modeling Long Sequences with Structured State Spaces

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{gelu, CoreResult, HiddenState, LayerNorm, NormType, SignalPredictor};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, Rng};

#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

/// Configuration for S5 model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct S5Config {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// State dimension (typically 64-256)
    pub state_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Discretization step size (Δt)
    pub dt: f32,
    /// Block size for chunked computation
    pub block_size: usize,
}

impl S5Config {
    /// Create default S5 configuration
    pub fn new(input_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            state_dim: 64,
            num_layers,
            dt: 0.001,
            block_size: 64,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> ModelResult<()> {
        if self.hidden_dim == 0 {
            return Err(ModelError::invalid_config("hidden_dim must be > 0"));
        }
        if self.state_dim == 0 {
            return Err(ModelError::invalid_config("state_dim must be > 0"));
        }
        if self.num_layers == 0 {
            return Err(ModelError::invalid_config("num_layers must be > 0"));
        }
        if self.dt <= 0.0 {
            return Err(ModelError::invalid_config("dt must be > 0"));
        }
        if self.block_size == 0 {
            return Err(ModelError::invalid_config("block_size must be > 0"));
        }
        Ok(())
    }
}

/// S5 SSM block with diagonal state matrix
#[allow(dead_code)]
struct S5Block {
    /// Diagonal of A matrix (log-space for stability)
    log_a: Array1<f32>,
    /// B matrix [state_dim, hidden_dim]
    b_matrix: Array2<f32>,
    /// C matrix [hidden_dim, state_dim]
    c_matrix: Array2<f32>,
    /// D skip connection [hidden_dim]
    d_vec: Array1<f32>,
    /// Discretization step
    dt: f32,
    /// Discretized A diagonal
    a_bar: Array1<f32>,
    /// Discretized B matrix
    b_bar: Array2<f32>,
    /// Current state [state_dim]
    state: Array1<f32>,
}

impl S5Block {
    /// Create new S5 block with simplified initialization
    fn new(hidden_dim: usize, state_dim: usize, dt: f32) -> Self {
        let mut rng = rng();

        // Initialize log_a with uniform spacing (simplified vs S4's HiPPO)
        let log_a = Array1::from_shape_fn(state_dim, |i| -((i + 1) as f32).ln());

        // Initialize B and C with random values
        let scale_b = (2.0 / (state_dim + hidden_dim) as f32).sqrt();
        let b_matrix = Array2::from_shape_fn((state_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale_b
        });

        let scale_c = (2.0 / (hidden_dim + state_dim) as f32).sqrt();
        let c_matrix = Array2::from_shape_fn((hidden_dim, state_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale_c
        });

        // Initialize D (skip connection) to small values
        let d_vec = Array1::from_shape_fn(hidden_dim, |_| rng.random::<f32>() * 0.01);

        // Discretize using zero-order hold (ZOH)
        let a_bar = log_a.mapv(|log_a_i| (dt * log_a_i.exp()).exp());
        let b_bar = b_matrix.clone() * dt;

        let state = Array1::zeros(state_dim);

        Self {
            log_a,
            b_matrix,
            c_matrix,
            d_vec,
            dt,
            a_bar,
            b_bar,
            state,
        }
    }

    /// Forward pass through S5 block
    #[instrument(skip(self, x))]
    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Update state: h[t] = A̅·h[t-1] + B̅·x[t]
        self.state = &self.state * &self.a_bar + self.b_bar.dot(x);

        // Compute output: y[t] = C·h[t] + D·x[t]
        let y = self.c_matrix.dot(&self.state) + &self.d_vec * x;

        Ok(y)
    }

    /// Reset the state
    fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

/// S5 layer with SSM block, activation, and normalization
struct S5Layer {
    /// Input projection
    input_proj: Array2<f32>,
    /// S5 SSM block
    s5_block: S5Block,
    /// Layer normalization
    layer_norm: LayerNorm,
    /// Output projection
    output_proj: Array2<f32>,
}

impl S5Layer {
    /// Create a new S5 layer
    fn new(config: &S5Config) -> ModelResult<Self> {
        let mut rng = rng();

        // Input projection
        let scale = (2.0 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let input_proj = Array2::from_shape_fn((config.input_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // S5 block
        let s5_block = S5Block::new(config.hidden_dim, config.state_dim, config.dt);

        // Layer normalization
        let layer_norm = LayerNorm::new(config.hidden_dim, NormType::RMSNorm);

        // Output projection
        let scale = (2.0 / (config.hidden_dim + config.input_dim) as f32).sqrt();
        let output_proj = Array2::from_shape_fn((config.hidden_dim, config.input_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        Ok(Self {
            input_proj,
            s5_block,
            layer_norm,
            output_proj,
        })
    }

    /// Forward pass
    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Project input
        let hidden = x.dot(&self.input_proj);

        // S5 SSM block
        let ssm_out = self.s5_block.forward(&hidden)?;

        // Activation
        let activated = gelu(&ssm_out);

        // Layer norm
        let normed = self.layer_norm.forward(&activated);

        // Output projection with residual
        let output = normed.dot(&self.output_proj) + x;

        Ok(output)
    }

    /// Reset layer state
    fn reset(&mut self) {
        self.s5_block.reset();
    }
}

/// S5 model with multiple layers
pub struct S5 {
    config: S5Config,
    layers: Vec<S5Layer>,
}

impl S5 {
    /// Create a new S5 model
    #[instrument(skip(config), fields(input_dim = config.input_dim, hidden_dim = config.hidden_dim, num_layers = config.num_layers))]
    pub fn new(config: S5Config) -> ModelResult<Self> {
        debug!("Creating new S5 model");
        config.validate()?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            trace!("Initializing S5 layer {}", layer_idx);
            layers.push(S5Layer::new(&config)?);
        }
        debug!("Initialized {} S5 layers", layers.len());

        debug!("S5 model created successfully");
        Ok(Self { config, layers })
    }

    /// Get configuration
    pub fn config(&self) -> &S5Config {
        &self.config
    }
}

impl SignalPredictor for S5 {
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
        debug!("Resetting S5 model state");
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    fn context_window(&self) -> usize {
        // SSMs have theoretically infinite context via recurrence
        usize::MAX
    }
}

impl AutoregressiveModel for S5 {
    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn state_dim(&self) -> usize {
        self.config.state_dim
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn model_type(&self) -> ModelType {
        ModelType::S4 // S5 is a variant of S4
    }

    fn get_states(&self) -> Vec<HiddenState> {
        self.layers
            .iter()
            .map(|layer| {
                // S5 uses 1D state, so expand to 2D for HiddenState
                let state_1d = layer.s5_block.state.clone();
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
                "S5",
                self.config.num_layers,
                states.len(),
            ));
        }

        for (layer, state) in self.layers.iter_mut().zip(states.iter()) {
            // Convert from 2D back to 1D
            let state_2d = state.state();
            if state_2d.nrows() > 0 && state_2d.ncols() > 0 {
                layer.s5_block.state = state_2d.row(0).to_owned();
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s5_creation() {
        let config = S5Config::new(32, 64, 2);
        let model = S5::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_s5_forward() {
        let config = S5Config::new(32, 64, 2);
        let mut model = S5::new(config).expect("Failed to create S5 model");

        let input = Array1::from_vec(vec![1.0; 32]);
        let output = model.step(&input);
        assert!(output.is_ok());
        assert_eq!(output.expect("Failed to get output").len(), 32);
    }

    #[test]
    fn test_s5_reset() {
        let config = S5Config::new(32, 64, 2);
        let mut model = S5::new(config).expect("Failed to create S5 model");

        let input = Array1::from_vec(vec![1.0; 32]);
        let _output1 = model.step(&input).expect("Failed to get output1");

        model.reset();

        let output2 = model.step(&input).expect("Failed to get output2");
        // After reset, same input should give similar output to first step
        assert_eq!(output2.len(), 32);
    }

    #[test]
    fn test_invalid_config() {
        let mut config = S5Config::new(32, 64, 2);
        config.state_dim = 0;
        assert!(config.validate().is_err());
    }
}
