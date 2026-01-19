//! Configuration types for the SSM engine

#[cfg(not(feature = "std"))]
use alloc::string::String;

use serde::{Deserialize, Serialize};

/// Type of state space model to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelType {
    /// Mamba selective SSM (original)
    Mamba,
    /// Mamba-2 with improved efficiency
    #[default]
    Mamba2,
    /// Structured State Space (S4)
    S4,
    /// RWKV linear attention
    Rwkv,
}

/// Configuration for the Kizzasi AGSP engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KizzasiConfig {
    model_type: ModelType,
    context_window: usize,
    hidden_dim: usize,
    state_dim: usize,
    num_layers: usize,
    input_dim: usize,
    output_dim: usize,
    dt_rank: usize,
    weights_path: Option<String>,
}

impl Default for KizzasiConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::default(),
            context_window: 8192,
            hidden_dim: 256,
            state_dim: 16,
            num_layers: 4,
            input_dim: 1,
            output_dim: 1,
            dt_rank: 8,
            weights_path: None,
        }
    }
}

impl KizzasiConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model type
    pub fn model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Set the context window size
    pub fn context_window(mut self, size: usize) -> Self {
        self.context_window = size;
        self
    }

    /// Set the hidden dimension
    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self
    }

    /// Set the state dimension
    pub fn state_dim(mut self, dim: usize) -> Self {
        self.state_dim = dim;
        self
    }

    /// Set the number of layers
    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Set input dimension
    pub fn input_dim(mut self, dim: usize) -> Self {
        self.input_dim = dim;
        self
    }

    /// Set output dimension
    pub fn output_dim(mut self, dim: usize) -> Self {
        self.output_dim = dim;
        self
    }

    /// Load weights from a file path
    pub fn load_weights(mut self, path: &str) -> Self {
        self.weights_path = Some(path.to_string());
        self
    }

    // Getters
    pub fn get_model_type(&self) -> ModelType {
        self.model_type
    }

    pub fn get_context_window(&self) -> usize {
        self.context_window
    }

    pub fn get_hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    pub fn get_state_dim(&self) -> usize {
        self.state_dim
    }

    pub fn get_num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn get_input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn get_output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn get_dt_rank(&self) -> usize {
        self.dt_rank
    }

    pub fn get_weights_path(&self) -> Option<&str> {
        self.weights_path.as_deref()
    }
}
