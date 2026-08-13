//! Configuration types for the SSM engine

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
    /// Inner-dimension expansion factor. Only the gated architectures
    /// (`ModelType::Mamba`) consume it; see [`KizzasiConfig::expansion_factor`].
    #[serde(default)]
    expansion_factor: Option<usize>,
    /// Multi-head head dimension. See [`KizzasiConfig::head_dim`].
    #[serde(default)]
    head_dim: Option<usize>,
    /// Multi-head head count. See [`KizzasiConfig::num_heads`].
    #[serde(default)]
    num_heads: Option<usize>,
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
            expansion_factor: None,
            head_dim: None,
            num_heads: None,
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

    /// Set the rank of the Δ (time-step) projection used by the selective scan.
    ///
    /// The selective mechanism produces one Δ per hidden channel from a
    /// rank-`dt_rank` projection of the layer input. Values are clamped to at
    /// least 1 — a rank-0 projection would make Δ constant and collapse the
    /// selectivity.
    pub fn dt_rank(mut self, rank: usize) -> Self {
        self.dt_rank = rank.max(1);
        self
    }

    /// Set the inner-dimension expansion factor.
    ///
    /// Mamba-style blocks expand the model dimension by this factor inside the
    /// gated branch (`d_inner = expand · d_model`). It is consumed only by the
    /// architectures that actually have such a branch — the selective-scan
    /// engine used for [`ModelType::Mamba2`] has none, and constructing that
    /// engine with an explicit expansion factor is rejected rather than
    /// silently ignored.
    pub fn expansion_factor(mut self, factor: usize) -> Self {
        self.expansion_factor = Some(factor);
        self
    }

    /// Set the per-head dimension for multi-head architectures.
    ///
    /// Consumed by the multi-head architectures ([`ModelType::Rwkv`]); the
    /// single-head selective-scan engine rejects it rather than ignoring it.
    pub fn head_dim(mut self, dim: usize) -> Self {
        self.head_dim = Some(dim);
        self
    }

    /// Set the number of heads for multi-head architectures.
    ///
    /// Must divide `hidden_dim`. Consumed by the multi-head architectures
    /// ([`ModelType::Rwkv`]); the single-head selective-scan engine rejects it
    /// rather than ignoring it.
    pub fn num_heads(mut self, heads: usize) -> Self {
        self.num_heads = Some(heads);
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

    /// Explicitly configured inner-dimension expansion factor, if any.
    pub fn get_expansion_factor(&self) -> Option<usize> {
        self.expansion_factor
    }

    /// Explicitly configured per-head dimension, if any.
    pub fn get_head_dim(&self) -> Option<usize> {
        self.head_dim
    }

    /// Explicitly configured head count, if any.
    pub fn get_num_heads(&self) -> Option<usize> {
        self.num_heads
    }
}
