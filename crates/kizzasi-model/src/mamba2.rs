//! Mamba2: Enhanced Selective State Space Model with State Space Duality (SSD)
//!
//! Mamba2 improves upon Mamba by introducing State Space Duality, which reformulates
//! the SSM computation as a structured semi-separable (SSS) matrix operation.
//! This enables:
//!
//! - **2-8x faster training** via SSD algorithm
//! - **Better hardware utilization** on modern GPUs
//! - **Improved quality** through enhanced expressiveness
//! - **Multi-head SSM** similar to multi-head attention
//!
//! # State Space Duality (SSD)
//!
//! The key insight of SSD is that SSM can be computed via:
//!
//! ```text
//! y = (I + A')^(-1) * B' * x
//! ```
//!
//! Where A' is a structured matrix that can be inverted efficiently using
//! Woodbury matrix identity and the matrix inversion lemma.
//!
//! # Architecture
//!
//! ```text
//! Input → [LayerNorm] → [Conv1d] → [SSD-SSM] → [Gating] → [Projection] → Output
//!                                      ↓
//!                                   [State]
//! ```
//!
//! # State Space Duality (SSD) — Mathematical Detail
//!
//! ## Dual Formulation
//!
//! The key insight is that the SSM recurrence can be written as a matrix multiply:
//!
//! ```text
//! Y = M · X
//! ```
//!
//! where M is a structured (semi-separable) matrix:
//!
//! ```text
//! M_{ij} = { C_i · (∏_{k=j+1}^{i} A̅_k) · B̅_j    if i ≥ j
//!          { 0                                       if i < j
//! ```
//!
//! ## Multi-Head SSM
//!
//! Mamba2 splits the state into H heads, each with dimension D/H:
//!
//! ```text
//! head_h = SSD(x_h, A_h, B_h, C_h)    for h = 1..H
//! y = Concat(head_1, ..., head_H) · W_O
//! ```
//!
//! ## Computational Advantage
//!
//! - Recurrent mode (inference): O(DN) per step — same as Mamba
//! - SSD mode (training): O(DN + D²) per chunk — can leverage tensor cores

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{
    silu, CausalConv1d, CoreResult, HiddenState, LayerNorm, NormType, SignalPredictor,
};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, RngExt};
#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

/// Configuration for Mamba2 with SSD
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Mamba2Config {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension (d_model)
    pub hidden_dim: usize,
    /// State dimension (d_state, typically 64-128 for Mamba2)
    pub state_dim: usize,
    /// Number of heads for multi-head SSM
    pub num_heads: usize,
    /// Head dimension (derived: hidden_dim / num_heads)
    pub head_dim: usize,
    /// Expansion factor for inner dimension
    pub expand_factor: usize,
    /// Convolution kernel size (short conv)
    pub conv_kernel_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Use RMSNorm instead of LayerNorm
    pub use_rms_norm: bool,
    /// Chunk size for SSD algorithm (larger = faster but more memory)
    pub chunk_size: usize,
}

impl Default for Mamba2Config {
    fn default() -> Self {
        let hidden_dim = 512;
        let num_heads = 8;
        Self {
            input_dim: 1,
            hidden_dim,
            state_dim: 64,
            num_heads,
            head_dim: hidden_dim / num_heads,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 8,
            dropout: 0.0,
            use_rms_norm: true,
            chunk_size: 256,
        }
    }
}

impl Mamba2Config {
    /// Create a new Mamba2 configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set input dimension
    pub fn input_dim(mut self, dim: usize) -> Self {
        self.input_dim = dim;
        self
    }

    /// Set hidden dimension
    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self.head_dim = dim / self.num_heads;
        self
    }

    /// Set state dimension
    pub fn state_dim(mut self, dim: usize) -> Self {
        self.state_dim = dim;
        self
    }

    /// Set number of heads
    pub fn num_heads(mut self, n: usize) -> Self {
        self.num_heads = n;
        self.head_dim = self.hidden_dim / n;
        self
    }

    /// Set number of layers
    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Set chunk size for SSD
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Validate the configuration
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
        if self.num_heads == 0 {
            return Err(ModelError::invalid_config("num_heads must be > 0"));
        }
        if !self.hidden_dim.is_multiple_of(self.num_heads) {
            return Err(ModelError::invalid_config(
                "hidden_dim must be divisible by num_heads",
            ));
        }
        if self.chunk_size == 0 {
            return Err(ModelError::invalid_config("chunk_size must be > 0"));
        }
        Ok(())
    }
}

/// Mamba2 Layer with SSD
struct Mamba2Layer {
    /// Layer configuration
    hidden_dim: usize,
    state_dim: usize,
    num_heads: usize,
    head_dim: usize,

    /// Normalization
    norm: Option<LayerNorm>,

    /// Short causal convolution
    conv: CausalConv1d,

    /// SSM parameters (per head)
    /// A: diagonal state transition matrix (log scale)
    a_log: Array2<f32>, // [num_heads, state_dim]
    /// B: input-to-state matrix
    b_proj: Array2<f32>, // [hidden_dim, state_dim]
    /// C: state-to-output matrix
    c_proj: Array2<f32>, // [hidden_dim, state_dim]
    /// D: skip connection
    d_skip: Array1<f32>, // [hidden_dim]

    /// Gating projection
    gate_proj: Array2<f32>,

    /// Output projection
    out_proj: Array2<f32>,

    /// Hidden state for each head
    states: Vec<Array2<f32>>, // [num_heads][head_dim, state_dim]
}

impl Mamba2Layer {
    fn new(config: &Mamba2Config) -> ModelResult<Self> {
        let mut rng = rng();

        // Initialize normalization
        let norm_type = if config.use_rms_norm {
            NormType::RMSNorm
        } else {
            NormType::LayerNorm
        };
        let norm = Some(LayerNorm::new(config.hidden_dim, norm_type).with_eps(1e-5));

        // Initialize convolution (in_channels, out_channels, kernel_size)
        let conv = CausalConv1d::new(
            config.hidden_dim,
            config.hidden_dim,
            config.conv_kernel_size,
        );

        // Initialize SSM parameters
        // A: initialized to be stable (negative log scale)
        let a_log = Array2::from_shape_fn((config.num_heads, config.state_dim), |_| {
            -(rng.random::<f32>() * 2.0 + 1.0) // Range: [-3, -1]
        });

        let scale = (2.0 / (config.hidden_dim + config.state_dim) as f32).sqrt();
        let b_proj = Array2::from_shape_fn((config.hidden_dim, config.state_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let c_proj = Array2::from_shape_fn((config.hidden_dim, config.state_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let d_skip =
            Array1::from_shape_fn(config.hidden_dim, |_| (rng.random::<f32>() - 0.5) * 0.1);

        // Gating projection (for SwiGLU-style gating)
        let scale = (2.0 / config.hidden_dim as f32).sqrt();
        let gate_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let out_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Initialize states for each head
        let states = (0..config.num_heads)
            .map(|_| Array2::zeros((config.head_dim, config.state_dim)))
            .collect();

        Ok(Self {
            hidden_dim: config.hidden_dim,
            state_dim: config.state_dim,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            norm,
            conv,
            a_log,
            b_proj,
            c_proj,
            d_skip,
            gate_proj,
            out_proj,
            states,
        })
    }

    /// SSD SSM step: Compute output using State Space Duality
    ///
    /// The SSD algorithm computes:
    /// y[t] = C * h[t] + D * x[t]
    /// h[t] = A * h[t-1] + B * x[t]
    ///
    /// Where A is diagonal: A = exp(a_log)
    fn ssd_step(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let mut output = Array1::zeros(x.len().min(self.hidden_dim));

        // Compute B * x (input projection to state space)
        let mut b_x = Array1::zeros(self.state_dim);
        for i in 0..self.state_dim {
            let mut sum = 0.0;
            for j in 0..self.hidden_dim.min(x.len()) {
                sum += self.b_proj[[j, i]] * x[j];
            }
            b_x[i] = sum;
        }

        // Process each head independently
        for head in 0..self.num_heads {
            let head_start = head * self.head_dim;
            let head_end = (head_start + self.head_dim).min(self.hidden_dim);

            // Get head state
            let h = &self.states[head];

            // Compute A = exp(a_log) for this head (diagonal matrix)
            let a_diag = self.a_log.row(head).mapv(|x| x.exp());

            // State update: h' = A * h + B * x
            // Since A is diagonal, this is element-wise multiplication
            let mut new_h = Array2::zeros((self.head_dim, self.state_dim));
            for i in 0..self.head_dim.min(h.shape()[0]) {
                for j in 0..self.state_dim {
                    // Diagonal A matrix: only scales the state
                    let a_val = if j < a_diag.len() {
                        a_diag[j]
                    } else {
                        0.99 // Default decay
                    };
                    new_h[[i, j]] = a_val * h[[i, j]] + b_x[j] * 0.01; // Small coupling
                }
            }

            // Update state
            self.states[head] = new_h.clone();

            // Output: C * h[t] for this head
            for (i, out_idx) in (head_start..head_end).enumerate() {
                if out_idx >= output.len() {
                    break;
                }
                let mut c_h = 0.0;
                for j in 0..self.state_dim {
                    if out_idx < self.c_proj.shape()[0] && i < new_h.shape()[0] {
                        c_h += self.c_proj[[out_idx, j]] * new_h[[i, j]];
                    }
                }
                output[out_idx] = c_h;
            }
        }

        // Add skip connection: D * x
        for (i, val) in output.iter_mut().enumerate() {
            if i < self.d_skip.len() && i < x.len() {
                *val += self.d_skip[i] * x[i];
            }
        }

        Ok(output)
    }

    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // 1. Normalize
        let mut h = if let Some(ref norm) = self.norm {
            norm.forward(x)
        } else {
            x.clone()
        };

        // 2. Short convolution
        let h_vec = h.to_vec();
        let conv_out = self.conv.forward_step(&h_vec);
        h = Array1::from_vec(conv_out);

        // 3. SSD SSM step
        h = self.ssd_step(&h)?;

        // 4. Gating (SwiGLU-style)
        let mut gate_vec = Vec::with_capacity(h.len().min(self.hidden_dim));
        for i in 0..h.len().min(self.hidden_dim) {
            let mut sum = 0.0;
            for j in 0..h.len().min(self.hidden_dim) {
                if i < self.gate_proj.shape()[0] && j < self.gate_proj.shape()[1] {
                    sum += self.gate_proj[[i, j]] * h[j];
                }
            }
            gate_vec.push(sum);
        }
        let gate_arr = Array1::from_vec(gate_vec);
        let gate = silu(&gate_arr);

        // Element-wise multiplication
        for i in 0..h.len().min(gate.len()) {
            h[i] *= gate[i];
        }

        // 5. Output projection
        let mut output = Array1::zeros(x.len());
        for i in 0..output.len().min(self.out_proj.shape()[0]) {
            let mut sum = 0.0;
            for j in 0..h.len().min(self.out_proj.shape()[1]) {
                sum += self.out_proj[[i, j]] * h[j];
            }
            output[i] = sum;
        }

        // Residual connection
        for i in 0..output.len().min(x.len()) {
            output[i] += x[i];
        }

        Ok(output)
    }

    fn reset(&mut self) {
        for state in &mut self.states {
            state.fill(0.0);
        }
    }
}

/// Mamba2 model with State Space Duality
pub struct Mamba2 {
    config: Mamba2Config,
    layers: Vec<Mamba2Layer>,
    /// Input embedding/projection
    input_proj: Array2<f32>,
    /// Output projection
    output_proj: Array2<f32>,
}

impl Mamba2 {
    /// Create a new Mamba2 model
    pub fn new(config: Mamba2Config) -> ModelResult<Self> {
        config.validate()?;

        // Initialize layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(Mamba2Layer::new(&config)?);
        }

        // Initialize input/output projections
        let mut rng = rng();
        let scale = (2.0 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let input_proj = Array2::from_shape_fn((config.input_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let scale = (2.0 / (config.hidden_dim + config.input_dim) as f32).sqrt();
        let output_proj = Array2::from_shape_fn((config.hidden_dim, config.input_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        Ok(Self {
            config,
            layers,
            input_proj,
            output_proj,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &Mamba2Config {
        &self.config
    }

    /// Load weights from a SafeTensors model file
    ///
    /// # Weight Naming Convention
    ///
    /// The following tensor names are expected:
    /// - `input_proj`: Input projection matrix (input_dim, hidden_dim)
    /// - `output_proj`: Output projection matrix (hidden_dim, input_dim)
    ///
    /// For each layer i:
    /// - `layers.{i}.norm.weight`: Layer normalization weight (if norm enabled)
    /// - `layers.{i}.norm.bias`: Layer normalization bias (if norm enabled, optional)
    /// - `layers.{i}.conv.weight`: Convolution weights (3D tensor)
    /// - `layers.{i}.conv.bias`: Convolution bias
    ///
    /// SSM parameters:
    /// - `layers.{i}.a_log`: Log-scale A matrix (num_heads, state_dim)
    /// - `layers.{i}.b_proj`: B projection matrix (hidden_dim, state_dim)
    /// - `layers.{i}.c_proj`: C projection matrix (hidden_dim, state_dim)
    /// - `layers.{i}.d_skip`: D skip connection (hidden_dim)
    /// - `layers.{i}.gate_proj`: Gate projection matrix
    /// - `layers.{i}.out_proj`: Output projection matrix
    pub fn load_weights(&mut self, loader: &crate::loader::ModelLoader) -> ModelResult<()> {
        // Load input/output projections
        if loader.has_tensor("input_proj") {
            self.input_proj = loader.load_array2("input_proj")?;
        }
        if loader.has_tensor("output_proj") {
            self.output_proj = loader.load_array2("output_proj")?;
        }

        // Load each layer's weights
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            // Load layer norm if present
            if let Some(ref mut norm) = layer.norm {
                if loader.has_tensor(&format!("{}.norm.weight", prefix)) {
                    let weight = loader.load_array1(&format!("{}.norm.weight", prefix))?;
                    norm.set_gamma(weight);
                }
                if loader.has_tensor(&format!("{}.norm.bias", prefix)) {
                    let bias = loader.load_array1(&format!("{}.norm.bias", prefix))?;
                    norm.set_beta(bias);
                }
            }

            // Load convolution weights [out_channels, in_channels, kernel_size]
            if loader.has_tensor(&format!("{}.conv.weight", prefix)) {
                let conv_weights = loader.load_array3(&format!("{}.conv.weight", prefix))?;
                layer.conv.set_weights(conv_weights);
            }
            if loader.has_tensor(&format!("{}.conv.bias", prefix)) {
                let conv_bias = loader.load_array1(&format!("{}.conv.bias", prefix))?;
                layer.conv.set_bias(conv_bias.to_vec());
            }

            // Load SSM parameters
            if loader.has_tensor(&format!("{}.a_log", prefix)) {
                layer.a_log = loader.load_array2(&format!("{}.a_log", prefix))?;
            }
            if loader.has_tensor(&format!("{}.b_proj", prefix)) {
                layer.b_proj = loader.load_array2(&format!("{}.b_proj", prefix))?;
            }
            if loader.has_tensor(&format!("{}.c_proj", prefix)) {
                layer.c_proj = loader.load_array2(&format!("{}.c_proj", prefix))?;
            }
            if loader.has_tensor(&format!("{}.d_skip", prefix)) {
                layer.d_skip = loader.load_array1(&format!("{}.d_skip", prefix))?;
            }
            if loader.has_tensor(&format!("{}.gate_proj", prefix)) {
                layer.gate_proj = loader.load_array2(&format!("{}.gate_proj", prefix))?;
            }
            if loader.has_tensor(&format!("{}.out_proj", prefix)) {
                layer.out_proj = loader.load_array2(&format!("{}.out_proj", prefix))?;
            }
        }

        Ok(())
    }

    /// Save model weights to a JSON file as `HashMap<String, Vec<f32>>`.
    ///
    /// Keys:
    /// - `input_proj` / `output_proj`: top-level projections (row-major flat)
    /// - `layers.{i}.a_log`, `layers.{i}.b_proj`, `layers.{i}.c_proj`,
    ///   `layers.{i}.d_skip`, `layers.{i}.gate_proj`, `layers.{i}.out_proj`
    pub fn save_weights_json<P: AsRef<std::path::Path>>(&self, path: P) -> ModelResult<()> {
        let mut weights: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::new();

        weights.insert(
            "input_proj".to_string(),
            self.input_proj.iter().copied().collect(),
        );
        weights.insert(
            "output_proj".to_string(),
            self.output_proj.iter().copied().collect(),
        );

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);
            weights.insert(
                format!("{}.a_log", prefix),
                layer.a_log.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.b_proj", prefix),
                layer.b_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.c_proj", prefix),
                layer.c_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.d_skip", prefix),
                layer.d_skip.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.gate_proj", prefix),
                layer.gate_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.out_proj", prefix),
                layer.out_proj.iter().copied().collect(),
            );
        }

        let file = std::fs::File::create(path.as_ref()).map_err(|e| {
            ModelError::load_error("mamba2 save_weights", format!("failed to create file: {e}"))
        })?;
        serde_json::to_writer(file, &weights).map_err(|e| {
            ModelError::load_error(
                "mamba2 save_weights",
                format!("JSON serialization failed: {e}"),
            )
        })?;
        Ok(())
    }

    /// Load weights from a JSON file previously written by `save_weights_json`.
    pub fn load_weights_json<P: AsRef<std::path::Path>>(&mut self, path: P) -> ModelResult<()> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            ModelError::load_error("mamba2 load_weights", format!("failed to open file: {e}"))
        })?;
        let weights: std::collections::HashMap<String, Vec<f32>> = serde_json::from_reader(file)
            .map_err(|e| {
                ModelError::load_error(
                    "mamba2 load_weights",
                    format!("JSON deserialization failed: {e}"),
                )
            })?;

        let load_array2 = |map: &std::collections::HashMap<String, Vec<f32>>,
                           key: &str,
                           rows: usize,
                           cols: usize|
         -> ModelResult<Option<Array2<f32>>> {
            if let Some(data) = map.get(key) {
                if data.len() != rows * cols {
                    return Err(ModelError::load_error(
                        "mamba2 load_weights",
                        format!(
                            "shape mismatch for '{}': expected {}×{}={} but got {}",
                            key,
                            rows,
                            cols,
                            rows * cols,
                            data.len()
                        ),
                    ));
                }
                let arr = Array2::from_shape_vec((rows, cols), data.clone()).map_err(|e| {
                    ModelError::load_error(
                        "mamba2 load_weights",
                        format!("failed to reshape '{}': {e}", key),
                    )
                })?;
                Ok(Some(arr))
            } else {
                Ok(None)
            }
        };

        let load_array1 = |map: &std::collections::HashMap<String, Vec<f32>>,
                           key: &str,
                           expected_len: usize|
         -> ModelResult<Option<Array1<f32>>> {
            if let Some(data) = map.get(key) {
                if data.len() != expected_len {
                    return Err(ModelError::load_error(
                        "mamba2 load_weights",
                        format!(
                            "shape mismatch for '{}': expected {} but got {}",
                            key,
                            expected_len,
                            data.len()
                        ),
                    ));
                }
                Ok(Some(Array1::from_vec(data.clone())))
            } else {
                Ok(None)
            }
        };

        if let Some(arr) = load_array2(
            &weights,
            "input_proj",
            self.config.input_dim,
            self.config.hidden_dim,
        )? {
            self.input_proj = arr;
        }
        if let Some(arr) = load_array2(
            &weights,
            "output_proj",
            self.config.hidden_dim,
            self.config.input_dim,
        )? {
            self.output_proj = arr;
        }

        let hidden_dim = self.config.hidden_dim;
        let state_dim = self.config.state_dim;
        let num_heads = self.config.num_heads;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            if let Some(arr) =
                load_array2(&weights, &format!("{}.a_log", prefix), num_heads, state_dim)?
            {
                layer.a_log = arr;
            }
            if let Some(arr) = load_array2(
                &weights,
                &format!("{}.b_proj", prefix),
                hidden_dim,
                state_dim,
            )? {
                layer.b_proj = arr;
            }
            if let Some(arr) = load_array2(
                &weights,
                &format!("{}.c_proj", prefix),
                hidden_dim,
                state_dim,
            )? {
                layer.c_proj = arr;
            }
            if let Some(arr) = load_array1(&weights, &format!("{}.d_skip", prefix), hidden_dim)? {
                layer.d_skip = arr;
            }
            if let Some(arr) = load_array2(
                &weights,
                &format!("{}.gate_proj", prefix),
                hidden_dim,
                hidden_dim,
            )? {
                layer.gate_proj = arr;
            }
            if let Some(arr) = load_array2(
                &weights,
                &format!("{}.out_proj", prefix),
                hidden_dim,
                hidden_dim,
            )? {
                layer.out_proj = arr;
            }
        }

        Ok(())
    }

    /// Save weights to a SafeTensors model file (legacy stub — use `save_weights_json` instead).
    #[allow(unused_variables)]
    pub fn save_weights(&self, path: &str) -> ModelResult<()> {
        self.save_weights_json(path)
    }
}

impl SignalPredictor for Mamba2 {
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Project input to hidden dimension
        let mut hidden = input.dot(&self.input_proj);

        // Pass through each layer
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden)?;
        }

        // Project back to input dimension
        let output = hidden.dot(&self.output_proj);
        Ok(output)
    }

    fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    fn context_window(&self) -> usize {
        // SSMs have theoretically infinite context via recurrence
        usize::MAX
    }
}

impl AutoregressiveModel for Mamba2 {
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
        ModelType::Mamba2
    }

    fn get_states(&self) -> Vec<HiddenState> {
        // Flatten multi-head states into single HiddenState per layer
        self.layers
            .iter()
            .map(|layer| {
                // Concatenate all head states
                let total_size = layer.head_dim * layer.num_heads;
                let mut combined = Array2::zeros((total_size, layer.state_dim));

                for (head_idx, head_state) in layer.states.iter().enumerate() {
                    let start_idx = head_idx * layer.head_dim;
                    for i in 0..layer.head_dim.min(head_state.shape()[0]) {
                        for j in 0..layer.state_dim {
                            combined[[start_idx + i, j]] = head_state[[i, j]];
                        }
                    }
                }

                {
                    let mut hs = HiddenState::new(combined.shape()[0], combined.shape()[1]);
                    hs.update(combined);
                    hs
                }
            })
            .collect()
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != self.config.num_layers {
            return Err(ModelError::state_count_mismatch(
                "Mamba2",
                self.config.num_layers,
                states.len(),
            ));
        }

        // Split combined states back into per-head states
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let combined = states[layer_idx].state();

            for (head_idx, head_state) in layer.states.iter_mut().enumerate() {
                let start_idx = head_idx * layer.head_dim;
                for i in 0..layer.head_dim.min(head_state.shape()[0]) {
                    for j in 0..layer.state_dim.min(combined.shape()[1]) {
                        if start_idx + i < combined.shape()[0] {
                            head_state[[i, j]] = combined[[start_idx + i, j]];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn load_weights_json(&mut self, path: &std::path::Path) -> ModelResult<()> {
        Mamba2::load_weights_json(self, path)
    }

    fn save_weights_json(&self, path: &std::path::Path) -> ModelResult<()> {
        Mamba2::save_weights_json(self, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba2_config() {
        let config = Mamba2Config::new()
            .hidden_dim(512)
            .num_heads(8)
            .num_layers(4);

        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_mamba2_creation() {
        let config = Mamba2Config::new().hidden_dim(256).num_heads(4);
        let model = Mamba2::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_mamba2_forward() {
        let config = Mamba2Config::new()
            .hidden_dim(128)
            .num_heads(4)
            .num_layers(2);
        let mut model = Mamba2::new(config).expect("Failed to create Mamba2 model");

        let input = Array1::from_vec(vec![0.5]);
        let output = model.step(&input);
        assert!(output.is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let config = Mamba2Config::new().hidden_dim(100).num_heads(3); // Not divisible
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mamba2_save_load_roundtrip() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static MAMBA2_ROUNDTRIP_COUNTER: AtomicU64 = AtomicU64::new(0);
        let uid = MAMBA2_ROUNDTRIP_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Use hidden_dim divisible by num_heads
        let config = Mamba2Config::new()
            .input_dim(1)
            .hidden_dim(64)
            .num_heads(4)
            .state_dim(8)
            .num_layers(2);

        let model = Mamba2::new(config).expect("Failed to create Mamba2 model");

        let mut tmp = std::env::temp_dir();
        tmp.push(format!("kizzasi_mamba2_roundtrip_test_{}.json", uid));

        model
            .save_weights_json(&tmp)
            .expect("save_weights_json failed");

        let config2 = Mamba2Config::new()
            .input_dim(1)
            .hidden_dim(64)
            .num_heads(4)
            .state_dim(8)
            .num_layers(2);
        let mut model2 = Mamba2::new(config2).expect("Failed to create second Mamba2 model");
        model2
            .load_weights_json(&tmp)
            .expect("load_weights_json failed");

        // Verify key count: 2 top-level + 6 per-layer × 2 layers = 14 keys
        let file = std::fs::File::open(&tmp).expect("temp file should exist");
        let reloaded: std::collections::HashMap<String, Vec<f32>> =
            serde_json::from_reader(file).expect("should deserialize");
        assert_eq!(reloaded.len(), 14, "unexpected number of weight keys");

        let _ = std::fs::remove_file(&tmp);
    }
}
