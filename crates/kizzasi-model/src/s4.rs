//! S4 and S4D: Structured State Space Models
//!
//! S4 (Structured State Space Sequence model) is a deep learning architecture
//! that leverages properties of continuous-time linear time-invariant (LTI) systems
//! for efficient and effective sequence modeling.
//!
//! # S4D Variant
//!
//! S4D simplifies S4 by using diagonal state matrices, which:
//! - Reduces computation from O(N²) to O(N)
//! - Simplifies implementation while maintaining quality
//! - Makes the model more interpretable
//!
//! # SSM Formulation
//!
//! Continuous-time:
//! ```text
//! h'(t) = A h(t) + B u(t)
//! y(t) = C h(t) + D u(t)
//! ```
//!
//! Discrete-time (after discretization):
//! ```text
//! h[k] = A̅ h[k-1] + B̅ u[k]
//! y[k] = C̅ h[k] + D̅ u[k]
//! ```
//!
//! Where:
//! - A̅ = exp(Δ·A) for ZOH (Zero-Order Hold)
//! - B̅ = (A̅ - I) A^(-1) B
//!
//! # S4D Architecture
//!
//! For S4D, A is diagonal: A = diag(-exp(α₁), -exp(α₂), ..., -exp(αₙ))
//! This makes discretization and computation much simpler.
//!
//! # Discretization Methods Comparison
//!
//! S4/S4D supports multiple discretization methods:
//!
//! ## Zero-Order Hold (ZOH) — Default
//! ```text
//! A̅ = exp(Δ · A)
//! B̅ = A⁻¹(A̅ - I) · B = (exp(Δ·A) - I) · A⁻¹ · B
//! ```
//! Best for: piecewise-constant inputs (sampled signals)
//!
//! ## Bilinear (Tustin's method)
//! ```text
//! A̅ = (I + Δ/2 · A)(I - Δ/2 · A)⁻¹
//! B̅ = Δ · (I - Δ/2 · A)⁻¹ · B
//! ```
//! Best for: frequency-domain preservation, stability guarantees
//!
//! ## Forward Euler
//! ```text
//! A̅ = I + Δ · A
//! B̅ = Δ · B
//! ```
//! Simplest but may be unstable for large Δ·A
//!
//! ## HiPPO Initialization
//!
//! S4D uses HiPPO-LegS initialization for the diagonal A matrix:
//! ```text
//! A_n = -(n + 1/2)    for n = 0, 1, ..., N-1
//! ```
//! This captures a compressed history of the input via Legendre polynomial projections.
//!
//! # References
//!
//! - S4 paper: <https://arxiv.org/abs/2111.00396>
//! - S4D paper: <https://arxiv.org/abs/2206.11893>

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{
    gelu, CausalConv1d, CoreResult, HiddenState, LayerNorm, NormType, SignalPredictor,
};
use safetensors::tensor::{Dtype, TensorView};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, RngExt};
#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

/// Configuration for S4D
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct S4Config {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension (d_model)
    pub hidden_dim: usize,
    /// State dimension (N)
    pub state_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Discretization step size (Δ)
    pub dt_min: f32,
    pub dt_max: f32,
    /// Use diagonal state matrix (S4D)
    pub use_diagonal: bool,
    /// Use RMSNorm instead of LayerNorm
    pub use_rms_norm: bool,
}

impl Default for S4Config {
    fn default() -> Self {
        Self {
            input_dim: 1,
            hidden_dim: 512,
            state_dim: 64,
            num_layers: 6,
            dropout: 0.0,
            dt_min: 0.001,
            dt_max: 0.1,
            use_diagonal: true, // S4D by default
            use_rms_norm: true,
        }
    }
}

impl S4Config {
    /// Create a new S4 configuration
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
        self
    }

    /// Set state dimension
    pub fn state_dim(mut self, dim: usize) -> Self {
        self.state_dim = dim;
        self
    }

    /// Set number of layers
    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Use diagonal state matrix (S4D)
    pub fn diagonal(mut self, use_diagonal: bool) -> Self {
        self.use_diagonal = use_diagonal;
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
        if self.dt_min <= 0.0 || self.dt_max <= 0.0 {
            return Err(ModelError::invalid_config("dt_min and dt_max must be > 0"));
        }
        if self.dt_min > self.dt_max {
            return Err(ModelError::invalid_config("dt_min must be <= dt_max"));
        }
        Ok(())
    }
}

/// S4D kernel: Diagonal state space model
struct S4DKernel {
    hidden_dim: usize,
    state_dim: usize,

    /// Log of diagonal A matrix elements
    /// A = diag(-exp(log_a[0]), -exp(log_a[1]), ..., -exp(log_a[N-1]))
    log_a: Array1<f32>,

    /// B matrix (input-to-state) [state_dim, hidden_dim]
    b_matrix: Array2<f32>,

    /// C matrix (state-to-output) [hidden_dim, state_dim]
    c_matrix: Array2<f32>,

    /// D matrix (skip connection) [hidden_dim]
    d_skip: Array1<f32>,

    /// Discretization step size (learnable)
    log_dt: Array1<f32>,

    /// Hidden state
    state: Array2<f32>, // [hidden_dim, state_dim]
}

impl S4DKernel {
    fn new(config: &S4Config) -> ModelResult<Self> {
        let mut rng = rng();

        // Initialize diagonal A with HiPPO initialization
        // A = diag(-1/2, -3/2, -5/2, ..., -(2N-1)/2)
        // Store log of absolute value since we negate later: A[n] = -exp(log_a[n])
        let log_a = Array1::from_shape_fn(config.state_dim, |n| ((2 * n + 1) as f32 / 2.0).ln());

        // Initialize B with random values
        let scale = (1.0 / config.state_dim as f32).sqrt();
        let b_matrix = Array2::from_shape_fn((config.state_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Initialize C with random values
        let c_matrix = Array2::from_shape_fn((config.hidden_dim, config.state_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Initialize D (skip connection)
        let d_skip = Array1::ones(config.hidden_dim);

        // Initialize discretization step (log scale)
        let log_dt = Array1::from_shape_fn(config.hidden_dim, |_| {
            let dt = config.dt_min + rng.random::<f32>() * (config.dt_max - config.dt_min);
            dt.ln()
        });

        // Initialize state
        let state = Array2::zeros((config.hidden_dim, config.state_dim));

        Ok(Self {
            hidden_dim: config.hidden_dim,
            state_dim: config.state_dim,
            log_a,
            b_matrix,
            c_matrix,
            d_skip,
            log_dt,
            state,
        })
    }

    /// Discretize continuous SSM to discrete SSM using ZOH (Zero-Order Hold)
    ///
    /// For diagonal A:
    /// A̅[i] = exp(Δ·A[i])
    /// B̅[i] = B[i] * (1 - A̅[i]) / (-A[i])
    fn discretize(&self, dt: f32) -> (Array1<f32>, Array2<f32>) {
        let mut a_bar = Array1::zeros(self.state_dim);
        let mut b_bar = Array2::zeros(self.b_matrix.raw_dim());

        for i in 0..self.state_dim {
            // A[i] = -exp(log_a[i])
            let a_i = -self.log_a[i].exp();

            // A̅[i] = exp(Δ·A[i])
            a_bar[i] = (dt * a_i).exp();

            // B̅[i, :] = B[i, :] * (1 - A̅[i]) / (-A[i])
            let scale = (1.0 - a_bar[i]) / (-a_i);
            for j in 0..self.hidden_dim {
                b_bar[[i, j]] = self.b_matrix[[i, j]] * scale;
            }
        }

        (a_bar, b_bar)
    }

    /// Forward step: compute next state and output
    fn forward_step(&mut self, u: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let batch_size = u.len().min(self.hidden_dim);

        // Get discretization parameters (per dimension)
        let mut output = Array1::zeros(batch_size);

        for dim in 0..batch_size {
            let dt = self.log_dt[dim].exp();
            let (a_bar, b_bar) = self.discretize(dt);

            // Update state: h[k] = A̅ ⊙ h[k-1] + B̅ u[k]
            // Since A̅ is diagonal, this is element-wise multiplication
            for i in 0..self.state_dim {
                let bu = if dim < b_bar.shape()[1] {
                    b_bar[[i, dim]] * u[dim]
                } else {
                    0.0
                };
                self.state[[dim, i]] = a_bar[i] * self.state[[dim, i]] + bu;
            }

            // Compute output: y[k] = C h[k] + D u[k]
            let mut c_h = 0.0;
            for i in 0..self.state_dim {
                c_h += self.c_matrix[[dim, i]] * self.state[[dim, i]];
            }
            output[dim] = c_h + self.d_skip[dim] * u[dim];
        }

        Ok(output)
    }

    fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

/// S4D Layer
struct S4DLayer {
    norm: LayerNorm,
    s4_kernel: S4DKernel,
    conv: CausalConv1d,
    output_proj: Array2<f32>,
}

impl S4DLayer {
    fn new(config: &S4Config) -> ModelResult<Self> {
        let norm_type = if config.use_rms_norm {
            NormType::RMSNorm
        } else {
            NormType::LayerNorm
        };

        let norm = LayerNorm::new(config.hidden_dim, norm_type).with_eps(1e-5);
        let s4_kernel = S4DKernel::new(config)?;

        // Short convolution for local context
        let conv = CausalConv1d::new(config.hidden_dim, config.hidden_dim, 3);

        // Output projection
        let mut rng = rng();
        let scale = (2.0 / config.hidden_dim as f32).sqrt();
        let output_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        Ok(Self {
            norm,
            s4_kernel,
            conv,
            output_proj,
        })
    }

    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // 1. Normalize
        let x_norm = self.norm.forward(x);

        // 2. Short convolution
        let x_vec = x_norm.to_vec();
        let conv_out = self.conv.forward_step(&x_vec);
        let x_conv = Array1::from_vec(conv_out);

        // 3. S4D SSM
        let ssm_out = self.s4_kernel.forward_step(&x_conv)?;

        // 4. Activation
        let activated = gelu(&ssm_out);

        // 5. Output projection
        let mut projected = Array1::zeros(x.len().min(self.output_proj.shape()[0]));
        for i in 0..projected.len() {
            let mut sum = 0.0;
            for j in 0..activated.len().min(self.output_proj.shape()[1]) {
                sum += self.output_proj[[i, j]] * activated[j];
            }
            projected[i] = sum;
        }

        // 6. Residual connection
        let mut output = x.clone();
        for i in 0..output.len().min(projected.len()) {
            output[i] += projected[i];
        }

        Ok(output)
    }

    fn reset(&mut self) {
        self.s4_kernel.reset();
    }
}

/// S4D model
pub struct S4D {
    config: S4Config,
    layers: Vec<S4DLayer>,
    ln_out: LayerNorm,
    input_proj: Array2<f32>,
    output_proj: Array2<f32>,
}

impl S4D {
    /// Create a new S4D model
    pub fn new(config: S4Config) -> ModelResult<Self> {
        config.validate()?;

        // Initialize layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(S4DLayer::new(&config)?);
        }

        // Output layer normalization
        let norm_type = if config.use_rms_norm {
            NormType::RMSNorm
        } else {
            NormType::LayerNorm
        };
        let ln_out = LayerNorm::new(config.hidden_dim, norm_type).with_eps(1e-5);

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
            ln_out,
            input_proj,
            output_proj,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &S4Config {
        &self.config
    }

    /// Load weights from a SafeTensors model file
    ///
    /// # Weight Naming Convention
    ///
    /// The following tensor names are expected:
    /// - `input_proj`: Input projection matrix (input_dim, hidden_dim)
    /// - `output_proj`: Output projection matrix (hidden_dim, input_dim)
    /// - `ln_out.weight`: Output layer norm weight (gamma)
    /// - `ln_out.bias`: Output layer norm bias (beta, optional)
    ///
    /// For each layer i:
    /// - `layers.{i}.norm.weight`: Layer normalization weight
    /// - `layers.{i}.norm.bias`: Layer normalization bias (optional)
    /// - `layers.{i}.output_proj`: Output projection matrix
    ///
    /// S4D kernel parameters:
    /// - `layers.{i}.s4_kernel.log_a`: Log of diagonal A matrix
    /// - `layers.{i}.s4_kernel.b_matrix`: B matrix (input-to-state)
    /// - `layers.{i}.s4_kernel.c_matrix`: C matrix (state-to-output)
    /// - `layers.{i}.s4_kernel.d_skip`: D skip connection
    /// - `layers.{i}.s4_kernel.log_dt`: Log of discretization step size
    pub fn load_weights(&mut self, loader: &crate::loader::ModelLoader) -> ModelResult<()> {
        // Load input/output projections
        if loader.has_tensor("input_proj") {
            self.input_proj = loader.load_array2("input_proj")?;
        }
        if loader.has_tensor("output_proj") {
            self.output_proj = loader.load_array2("output_proj")?;
        }

        // Load output layer norm
        if loader.has_tensor("ln_out.weight") {
            let weight = loader.load_array1("ln_out.weight")?;
            self.ln_out.set_gamma(weight);
        }
        if loader.has_tensor("ln_out.bias") {
            let bias = loader.load_array1("ln_out.bias")?;
            self.ln_out.set_beta(bias);
        }

        // Load each layer's weights
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            // Load layer norm
            if loader.has_tensor(&format!("{}.norm.weight", prefix)) {
                let weight = loader.load_array1(&format!("{}.norm.weight", prefix))?;
                layer.norm.set_gamma(weight);
            }
            if loader.has_tensor(&format!("{}.norm.bias", prefix)) {
                let bias = loader.load_array1(&format!("{}.norm.bias", prefix))?;
                layer.norm.set_beta(bias);
            }

            // Load output projection
            if loader.has_tensor(&format!("{}.output_proj", prefix)) {
                layer.output_proj = loader.load_array2(&format!("{}.output_proj", prefix))?;
            }

            // Load S4D kernel parameters
            let kernel_prefix = format!("{}.s4_kernel", prefix);
            if loader.has_tensor(&format!("{}.log_a", kernel_prefix)) {
                layer.s4_kernel.log_a = loader.load_array1(&format!("{}.log_a", kernel_prefix))?;
            }
            if loader.has_tensor(&format!("{}.b_matrix", kernel_prefix)) {
                layer.s4_kernel.b_matrix =
                    loader.load_array2(&format!("{}.b_matrix", kernel_prefix))?;
            }
            if loader.has_tensor(&format!("{}.c_matrix", kernel_prefix)) {
                layer.s4_kernel.c_matrix =
                    loader.load_array2(&format!("{}.c_matrix", kernel_prefix))?;
            }
            if loader.has_tensor(&format!("{}.d_skip", kernel_prefix)) {
                layer.s4_kernel.d_skip =
                    loader.load_array1(&format!("{}.d_skip", kernel_prefix))?;
            }
            if loader.has_tensor(&format!("{}.log_dt", kernel_prefix)) {
                layer.s4_kernel.log_dt =
                    loader.load_array1(&format!("{}.log_dt", kernel_prefix))?;
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
        }

        Ok(())
    }

    /// Save model weights to a JSON file as `HashMap<String, Vec<f32>>`.
    ///
    /// Keys:
    /// - `input_proj` / `output_proj`: top-level projections
    /// - `layers.{i}.output_proj`
    /// - `layers.{i}.s4_kernel.log_a`, `.b_matrix`, `.c_matrix`, `.d_skip`, `.log_dt`
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
            let kp = format!("{}.s4_kernel", prefix);

            weights.insert(
                format!("{}.output_proj", prefix),
                layer.output_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.log_a", kp),
                layer.s4_kernel.log_a.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.b_matrix", kp),
                layer.s4_kernel.b_matrix.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.c_matrix", kp),
                layer.s4_kernel.c_matrix.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.d_skip", kp),
                layer.s4_kernel.d_skip.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.log_dt", kp),
                layer.s4_kernel.log_dt.iter().copied().collect(),
            );
        }

        let file = std::fs::File::create(path.as_ref()).map_err(|e| {
            ModelError::load_error("s4d save_weights", format!("failed to create file: {e}"))
        })?;
        serde_json::to_writer(file, &weights).map_err(|e| {
            ModelError::load_error(
                "s4d save_weights",
                format!("JSON serialization failed: {e}"),
            )
        })?;
        Ok(())
    }

    /// Load weights from a JSON file previously written by `save_weights_json`.
    pub fn load_weights_json<P: AsRef<std::path::Path>>(&mut self, path: P) -> ModelResult<()> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            ModelError::load_error("s4d load_weights", format!("failed to open file: {e}"))
        })?;
        let weights: std::collections::HashMap<String, Vec<f32>> = serde_json::from_reader(file)
            .map_err(|e| {
                ModelError::load_error(
                    "s4d load_weights",
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
                        "s4d load_weights",
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
                        "s4d load_weights",
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
                        "s4d load_weights",
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

        let hidden = self.config.hidden_dim;
        let state = self.config.state_dim;

        if let Some(arr) = load_array2(&weights, "input_proj", self.config.input_dim, hidden)? {
            self.input_proj = arr;
        }
        if let Some(arr) = load_array2(&weights, "output_proj", hidden, self.config.input_dim)? {
            self.output_proj = arr;
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);
            let kp = format!("{}.s4_kernel", prefix);

            if let Some(arr) =
                load_array2(&weights, &format!("{}.output_proj", prefix), hidden, hidden)?
            {
                layer.output_proj = arr;
            }
            if let Some(arr) = load_array1(&weights, &format!("{}.log_a", kp), state)? {
                layer.s4_kernel.log_a = arr;
            }
            if let Some(arr) = load_array2(&weights, &format!("{}.b_matrix", kp), state, hidden)? {
                layer.s4_kernel.b_matrix = arr;
            }
            if let Some(arr) = load_array2(&weights, &format!("{}.c_matrix", kp), hidden, state)? {
                layer.s4_kernel.c_matrix = arr;
            }
            if let Some(arr) = load_array1(&weights, &format!("{}.d_skip", kp), hidden)? {
                layer.s4_kernel.d_skip = arr;
            }
            if let Some(arr) = load_array1(&weights, &format!("{}.log_dt", kp), hidden)? {
                layer.s4_kernel.log_dt = arr;
            }
        }

        Ok(())
    }

    /// Save weights to a SafeTensors model file.
    ///
    /// Serializes all model parameters (input projection, output projection, and per-layer
    /// S4D kernel matrices) into the SafeTensors binary format at the given path.
    pub fn save_weights(&self, path: &str) -> ModelResult<()> {
        // Collect (name, shape, raw-f32-bytes) so byte vecs stay alive long enough
        // to be borrowed by TensorView below.
        let mut entries: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();

        // Helper: push a 2-D tensor entry.
        fn push_arr2(
            entries: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
            name: String,
            arr: &Array2<f32>,
        ) {
            let bytes: Vec<u8> = arr.iter().flat_map(|v| v.to_le_bytes()).collect();
            entries.push((name, vec![arr.nrows(), arr.ncols()], bytes));
        }

        // Helper: push a 1-D tensor entry.
        fn push_arr1(
            entries: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
            name: String,
            arr: &Array1<f32>,
        ) {
            let bytes: Vec<u8> = arr.iter().flat_map(|v| v.to_le_bytes()).collect();
            entries.push((name, vec![arr.len()], bytes));
        }

        push_arr2(&mut entries, "input_proj".to_owned(), &self.input_proj);
        push_arr2(&mut entries, "output_proj".to_owned(), &self.output_proj);

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{i}");
            let kp = format!("{prefix}.s4_kernel");
            push_arr2(
                &mut entries,
                format!("{prefix}.output_proj"),
                &layer.output_proj,
            );
            push_arr1(&mut entries, format!("{kp}.log_a"), &layer.s4_kernel.log_a);
            push_arr2(
                &mut entries,
                format!("{kp}.b_matrix"),
                &layer.s4_kernel.b_matrix,
            );
            push_arr2(
                &mut entries,
                format!("{kp}.c_matrix"),
                &layer.s4_kernel.c_matrix,
            );
            push_arr1(
                &mut entries,
                format!("{kp}.d_skip"),
                &layer.s4_kernel.d_skip,
            );
            push_arr1(
                &mut entries,
                format!("{kp}.log_dt"),
                &layer.s4_kernel.log_dt,
            );
        }

        // Build TensorView slice — borrows from `entries` which lives until end of fn.
        let tensor_views: Vec<(String, TensorView<'_>)> = entries
            .iter()
            .map(|(name, shape, bytes)| {
                let view = TensorView::new(Dtype::F32, shape.clone(), bytes).map_err(|e| {
                    ModelError::load_error("s4d save_weights", format!("TensorView failed: {e}"))
                })?;
                Ok((name.clone(), view))
            })
            .collect::<ModelResult<Vec<_>>>()?;

        safetensors::tensor::serialize_to_file(tensor_views, None, std::path::Path::new(path))
            .map_err(|e| ModelError::load_error("s4d save_weights", format!("write failed: {e}")))
    }
}

impl SignalPredictor for S4D {
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Project input to hidden dimension
        let mut hidden = input.dot(&self.input_proj);

        // Pass through each layer
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden)?;
        }

        // Final layer normalization
        hidden = self.ln_out.forward(&hidden);

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
        // S4D has theoretically infinite context via recurrence
        usize::MAX
    }
}

impl AutoregressiveModel for S4D {
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
        ModelType::S4D
    }

    fn get_states(&self) -> Vec<HiddenState> {
        self.layers
            .iter()
            .map(|layer| {
                let state = layer.s4_kernel.state.clone();
                let mut hs = HiddenState::new(state.shape()[0], state.shape()[1]);
                hs.update(state);
                hs
            })
            .collect()
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != self.config.num_layers {
            return Err(ModelError::state_count_mismatch(
                "S4D",
                self.config.num_layers,
                states.len(),
            ));
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            layer.s4_kernel.state = states[layer_idx].state().clone();
        }

        Ok(())
    }

    fn load_weights_json(&mut self, path: &std::path::Path) -> ModelResult<()> {
        S4D::load_weights_json(self, path)
    }

    fn save_weights_json(&self, path: &std::path::Path) -> ModelResult<()> {
        S4D::save_weights_json(self, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s4d_config() {
        let config = S4Config::new().hidden_dim(256).state_dim(64).num_layers(4);

        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.state_dim, 64);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_s4d_creation() {
        let config = S4Config::new().hidden_dim(128).state_dim(32);
        let model = S4D::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_s4d_forward() {
        let config = S4Config::new().hidden_dim(64).state_dim(16).num_layers(2);
        let mut model = S4D::new(config).expect("Failed to create S4D model");

        let input = Array1::from_vec(vec![0.5]);
        let output = model.step(&input);
        assert!(output.is_ok());
    }

    #[test]
    fn test_invalid_dt() {
        let config = S4Config {
            dt_min: 0.1,
            dt_max: 0.01, // max < min
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_s4_save_load_roundtrip() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static S4_ROUNDTRIP_COUNTER: AtomicU64 = AtomicU64::new(0);
        let uid = S4_ROUNDTRIP_COUNTER.fetch_add(1, Ordering::Relaxed);

        let config = S4Config::new().hidden_dim(32).state_dim(8).num_layers(2);
        let model = S4D::new(config).expect("Failed to create S4D model");

        let mut tmp = std::env::temp_dir();
        tmp.push(format!("kizzasi_s4_roundtrip_test_{}.json", uid));

        model
            .save_weights_json(&tmp)
            .expect("save_weights_json failed");

        let config2 = S4Config::new().hidden_dim(32).state_dim(8).num_layers(2);
        let mut model2 = S4D::new(config2).expect("Failed to create second S4D model");
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

    #[test]
    fn test_save_weights_roundtrip_safetensors() {
        let config = S4Config {
            input_dim: 4,
            hidden_dim: 8,
            state_dim: 4,
            num_layers: 1,
            ..Default::default()
        };
        let model = S4D::new(config).expect("model creation failed");
        let tmp = std::env::temp_dir().join("test_s4d_save_weights.safetensors");
        model
            .save_weights(tmp.to_str().expect("path to str"))
            .expect("save_weights failed");
        let data = std::fs::read(&tmp).expect("read file");
        let tensors = safetensors::SafeTensors::deserialize(&data).expect("deserialize");
        assert!(!tensors.names().is_empty(), "no tensors written");
        std::fs::remove_file(&tmp).ok();
    }
}
