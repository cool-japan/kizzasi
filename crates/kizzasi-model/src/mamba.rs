//! Mamba: Selective State Space Model
//!
//! Mamba is a selective SSM that uses input-dependent state transitions,
//! allowing it to selectively remember or forget information based on content.
//!
//! # Key Features
//!
//! - **O(1) inference**: Constant time per token during autoregressive generation
//! - **Selectivity**: Input-dependent Δ, B, C parameters
//! - **Hardware-efficient**: Parallel scan for training, recurrent for inference
//! - **Continuous native**: No discrete vocabulary needed for signal prediction
//!
//! # Architecture
//!
//! ```text
//! Input → [Expand] → [Conv1D] → [SSM] → [Gate] → [Project] → Output
//!                                  ↓
//!                               [State]
//! ```
//!
//! # Selective SSM Formulation
//!
//! Unlike traditional SSMs with fixed parameters, Mamba computes:
//!
//! ```text
//! Δ, B, C = Linear(x)  // Input-dependent parameters
//! A̅ = exp(Δ·A)         // Discretized A matrix
//! B̅ = (A̅ - I)·A^(-1)·B // Discretized B matrix
//! h[t] = A̅·h[t-1] + B̅·x[t]
//! y[t] = C·h[t]
//! ```
//!
//! # Detailed Mathematical Formulation
//!
//! ## Input-Dependent Parameters
//!
//! Mamba's selectivity comes from computing SSM parameters as functions of the input:
//!
//! ```text
//! Δ_t = softplus(Linear_Δ(x_t) + bias_Δ)     ∈ ℝ^D
//! B_t = Linear_B(x_t)                          ∈ ℝ^{D×N}
//! C_t = Linear_C(x_t)                          ∈ ℝ^{D×N}
//! ```
//!
//! where D is the expanded dimension and N is the state dimension.
//!
//! ## Zero-Order Hold (ZOH) Discretization
//!
//! The continuous SSM is discretized using ZOH:
//!
//! ```text
//! A̅_t = exp(Δ_t ⊙ A)                         (element-wise for diagonal A)
//! B̅_t = (A̅_t - I) ⊘ A ⊙ (Δ_t ⊙ B_t)        (simplified for diagonal A)
//! ```
//!
//! For numerical stability, when |Δ·A| < ε, a first-order Taylor approximation is used:
//!
//! ```text
//! B̅_t ≈ Δ_t ⊙ B_t    (when Δ·A → 0)
//! ```
//!
//! ## Gating Mechanism
//!
//! The output is gated using a SiLU (Swish) activation:
//!
//! ```text
//! z_t = SiLU(Linear_z(x_t))
//! y_t = (C_t · h_t) ⊙ z_t
//! ```
//!
//! # References
//!
//! - Mamba paper: <https://arxiv.org/abs/2312.00752>
//! - Efficient Implementation: Parallel prefix scan for training

use crate::error::{ModelError, ModelResult};
use crate::AutoregressiveModel;
use kizzasi_core::{
    silu, CausalConv1d, CoreResult, HiddenState, LayerNorm, NormType, SignalPredictor,
};
use safetensors::tensor::{Dtype, TensorView};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, RngExt};
use tracing::{debug, instrument, trace, warn};

/// Configuration for Mamba model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MambaConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension (d_model)
    pub hidden_dim: usize,
    /// State dimension (d_state, typically 16)
    pub state_dim: usize,
    /// Expansion factor for inner dimension
    pub expand_factor: usize,
    /// Convolution kernel size
    pub conv_kernel_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout rate applied to each layer output while training mode is
    /// enabled via `set_training(true)` (inverted dropout; inert at inference)
    pub dropout: f32,
    /// Use Mamba2 architecture (SSD)
    pub use_mamba2: bool,
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            input_dim: 1,
            hidden_dim: 256,
            state_dim: 16,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 4,
            dropout: 0.0,
            use_mamba2: true,
        }
    }
}

impl MambaConfig {
    /// Create a new Mamba configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Mamba-Tiny: Lightweight configuration for fast inference and low memory
    ///
    /// Optimized for:
    /// - Edge devices
    /// - Real-time streaming applications
    /// - Low-latency inference
    ///
    /// # Parameters
    /// - Hidden dim: 128
    /// - State dim: 8
    /// - Layers: 2
    /// - Target latency: <50μs per step
    /// - Memory: <10MB
    pub fn tiny(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 128,
            state_dim: 8,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 2,
            dropout: 0.0,
            use_mamba2: false, // Use simpler Mamba for speed
        }
    }

    /// Mamba-Small: Balanced configuration for moderate capacity
    ///
    /// Optimized for:
    /// - General-purpose applications
    /// - Moderate accuracy requirements
    /// - Resource-constrained servers
    ///
    /// # Parameters
    /// - Hidden dim: 256
    /// - State dim: 16
    /// - Layers: 4
    /// - Target latency: <100μs per step
    /// - Memory: <50MB
    pub fn small(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 256,
            state_dim: 16,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 4,
            dropout: 0.1,
            use_mamba2: true,
        }
    }

    /// Mamba-Base: Standard configuration (default)
    ///
    /// Optimized for:
    /// - Standard applications
    /// - Good accuracy/speed tradeoff
    /// - Server deployment
    ///
    /// # Parameters
    /// - Hidden dim: 512
    /// - State dim: 16
    /// - Layers: 6
    /// - Target latency: <200μs per step
    /// - Memory: <200MB
    pub fn base(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 512,
            state_dim: 16,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 6,
            dropout: 0.1,
            use_mamba2: true,
        }
    }

    /// Mamba-Large: High-capacity configuration for maximum accuracy
    ///
    /// Optimized for:
    /// - High-accuracy applications
    /// - Complex sequence modeling
    /// - GPU deployment
    ///
    /// # Parameters
    /// - Hidden dim: 1024
    /// - State dim: 32
    /// - Layers: 12
    /// - Target latency: <500μs per step
    /// - Memory: <1GB
    pub fn large(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 1024,
            state_dim: 32,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 12,
            dropout: 0.1,
            use_mamba2: true,
        }
    }

    /// Mamba-XLarge: Experimental extra-large configuration
    ///
    /// Optimized for:
    /// - Research and experimentation
    /// - Maximum model capacity
    /// - Multi-GPU deployment
    ///
    /// # Parameters
    /// - Hidden dim: 2048
    /// - State dim: 64
    /// - Layers: 24
    /// - Target latency: <1ms per step
    /// - Memory: <4GB
    pub fn xlarge(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 2048,
            state_dim: 64,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 24,
            dropout: 0.2,
            use_mamba2: true,
        }
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

    /// Use Mamba2 (SSD) architecture
    pub fn mamba2(mut self, use_mamba2: bool) -> Self {
        self.use_mamba2 = use_mamba2;
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
        if self.expand_factor == 0 {
            return Err(ModelError::invalid_config("expand_factor must be > 0"));
        }
        Ok(())
    }
}

/// Selective SSM block with input-dependent parameters
struct SelectiveSSM {
    state_dim: usize,
    inner_dim: usize,

    /// Fixed diagonal A matrix (in log space for stability)
    /// A = -exp(log_a), initialized with HiPPO
    log_a: Array1<f32>,

    /// Projections for selective parameters
    /// Δ (delta): discretization step size
    delta_proj: Array2<f32>, // [inner_dim, inner_dim]
    delta_bias: Array1<f32>, // [inner_dim]

    /// B: input-to-state projection (selective)
    b_proj: Array2<f32>, // [inner_dim, state_dim]

    /// C: state-to-output projection (selective)
    c_proj: Array2<f32>, // [inner_dim, state_dim]

    /// D: skip connection
    d_skip: Array1<f32>, // [inner_dim]

    /// Current state
    state: Array2<f32>, // [inner_dim, state_dim]
}

impl SelectiveSSM {
    fn new(config: &MambaConfig) -> ModelResult<Self> {
        let mut rng = rng();
        let inner_dim = config.hidden_dim * config.expand_factor;

        // Initialize diagonal A with HiPPO initialization
        // A[n] = -(n + 1) for improved long-range modeling
        // Store log of the absolute value since we'll negate later
        let log_a = Array1::from_shape_fn(config.state_dim, |n| ((n + 1) as f32).ln());

        // Initialize projections
        let scale = (2.0 / inner_dim as f32).sqrt();

        let delta_proj = Array2::from_shape_fn((inner_dim, inner_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let delta_bias = Array1::from_shape_fn(inner_dim, |_| rng.random::<f32>() * 0.1);

        let b_proj = Array2::from_shape_fn((inner_dim, config.state_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let c_proj = Array2::from_shape_fn((inner_dim, config.state_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let d_skip = Array1::ones(inner_dim);

        let state = Array2::zeros((inner_dim, config.state_dim));

        Ok(Self {
            state_dim: config.state_dim,
            inner_dim,
            log_a,
            delta_proj,
            delta_bias,
            b_proj,
            c_proj,
            d_skip,
            state,
        })
    }

    /// Selective SSM forward step
    ///
    /// Computes input-dependent parameters and performs state update
    fn forward_step(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let _span = tracing::trace_span!(
            "ssm_step",
            state_dim = self.state_dim,
            inner_dim = self.inner_dim
        )
        .entered();
        let batch_size = x.len().min(self.inner_dim);

        // 1. Compute input-dependent Δ (discretization step size)
        // Δ = Softplus(Linear(x) + bias)
        let mut delta = Array1::zeros(batch_size);
        for i in 0..batch_size {
            let mut sum = self.delta_bias[i];
            for j in 0..batch_size {
                sum += self.delta_proj[[i, j]] * x[j];
            }
            // Softplus activation to ensure Δ > 0
            // Clamp input to avoid overflow in exp
            let clamped = sum.clamp(-20.0, 20.0);
            delta[i] = (1.0 + clamped.exp()).ln().clamp(1e-6, 0.1);
        }

        // 2. Compute input-dependent B (input-to-state)
        // B = Linear_B(x), not just copying weights.
        //
        // `b_row[n] = sum_j b_proj[j, n] * x[j]` does not depend on the batch
        // row `i` at all (see consumption below), so it is computed once as a
        // length-`state_dim` row instead of redundantly recomputing the same
        // value for every one of the `batch_size` rows of a
        // `(batch_size, state_dim)` matrix. The bound checks are hoisted out
        // of the inner loop too: `b_row[n]` for `n >= b_bound_n` stays at its
        // zero-init value, matching the original `else { 0.0 }` branch.
        let b_bound_j = self.b_proj.shape()[0].min(batch_size);
        let b_bound_n = self.b_proj.shape()[1].min(self.state_dim);
        let mut b_row = vec![0.0f32; self.state_dim];
        for (n, slot) in b_row.iter_mut().enumerate().take(b_bound_n) {
            let mut sum = 0.0;
            for j in 0..b_bound_j {
                sum += self.b_proj[[j, n]] * x[j];
            }
            *slot = sum;
        }

        // 3. Compute input-dependent C (state-to-output)
        // C = Linear_C(x), not just copying weights. Same loop-invariance in
        // `i` as `b_row` above.
        let c_bound_j = self.c_proj.shape()[0].min(batch_size);
        let c_bound_n = self.c_proj.shape()[1].min(self.state_dim);
        let mut c_row = vec![0.0f32; self.state_dim];
        for (n, slot) in c_row.iter_mut().enumerate().take(c_bound_n) {
            let mut sum = 0.0;
            for j in 0..c_bound_j {
                sum += self.c_proj[[j, n]] * x[j];
            }
            *slot = sum;
        }

        // 4. Discretize: A̅ = exp(Δ·A)
        // For diagonal A: A̅[n] = exp(Δ · A[n])
        //
        // `a_diag[n] = -exp(log_a[n])` depends only on `n`, not on the batch
        // row `i` or on Δ, so it is computed once here instead of being
        // re-evaluated (with a transcendental `exp()` call each time) inside
        // both the a_bar loop and the b_bar loop below.
        let a_diag: Vec<f32> = (0..self.state_dim)
            .map(|n| -self.log_a[n].exp()) // A[n] = -exp(log_a[n])
            .collect();

        let mut a_bar = Array2::zeros((batch_size, self.state_dim));
        for i in 0..batch_size {
            for n in 0..self.state_dim {
                let delta_a = delta[i] * a_diag[n];
                // Clamp to prevent numerical overflow
                a_bar[[i, n]] = delta_a.clamp(-20.0, 20.0).exp();
            }
        }

        // 5. Discretize: B̅ using ZOH or Taylor approximation
        // Exact: B̅ = (A̅ - I)·A^(-1)·B
        // For small Δ: B̅ ≈ Δ·B (first-order Taylor)
        // For moderate Δ: Use exact formula
        let mut b_bar = Array2::zeros((batch_size, self.state_dim));
        for i in 0..batch_size {
            for n in 0..self.state_dim {
                let a_n = a_diag[n];

                // Use Taylor approximation for small delta (more numerically stable)
                if delta[i].abs() < 0.001 {
                    // First-order: B̅ ≈ Δ·B
                    b_bar[[i, n]] = delta[i] * b_row[n];
                } else {
                    // Exact ZOH discretization
                    // B̅[n] = (exp(Δ·A[n]) - 1) / A[n] · B[n]
                    // As A[n] → 0, L'Hôpital gives the limit (e^{Δa}-1)/a → Δ,
                    // so fall back to the Taylor approximation Δ·B.
                    if a_n.abs() < 1e-8 {
                        b_bar[[i, n]] = delta[i] * b_row[n];
                    } else {
                        b_bar[[i, n]] = (a_bar[[i, n]] - 1.0) / a_n * b_row[n];
                    }
                }
            }
        }

        // 6. State update: h[t] = A̅·h[t-1] + B̅·x[t]
        let mut new_state = Array2::zeros((batch_size, self.state_dim));
        for i in 0..batch_size {
            for n in 0..self.state_dim {
                // Diagonal A: element-wise multiplication
                let decay = a_bar[[i, n]];
                let input_contrib = b_bar[[i, n]] * x[i];

                new_state[[i, n]] = decay * self.state[[i, n]] + input_contrib;
            }
        }

        // Update state
        for i in 0..batch_size.min(self.state.shape()[0]) {
            for n in 0..self.state_dim {
                self.state[[i, n]] = new_state[[i, n]];
            }
        }

        // 7. Output: y = C·h + D·x
        let mut output = Array1::zeros(batch_size);
        for i in 0..batch_size {
            let mut c_h = 0.0;
            for n in 0..self.state_dim {
                c_h += c_row[n] * new_state[[i, n]];
            }
            output[i] = c_h + self.d_skip[i] * x[i];
        }

        Ok(output)
    }

    fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

/// Mamba Layer with Selective SSM
struct MambaLayer {
    hidden_dim: usize,
    inner_dim: usize,

    /// Layer normalization
    norm: LayerNorm,

    /// Expansion projection
    in_proj: Array2<f32>, // [hidden_dim, inner_dim * 2]

    /// Short causal convolution for local context
    conv: CausalConv1d,

    /// Selective SSM
    ssm: SelectiveSSM,

    /// Output projection (contracts inner_dim back to hidden_dim)
    out_proj: Array2<f32>, // [inner_dim, hidden_dim]
}

impl MambaLayer {
    fn new(config: &MambaConfig) -> ModelResult<Self> {
        let inner_dim = config.hidden_dim * config.expand_factor;
        let mut rng = rng();

        // RMSNorm for better stability
        let norm = LayerNorm::new(config.hidden_dim, NormType::RMSNorm).with_eps(1e-5);

        // Input projection (expands and creates gate path)
        let scale = (2.0 / config.hidden_dim as f32).sqrt();
        let in_proj = Array2::from_shape_fn((config.hidden_dim, inner_dim * 2), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Causal convolution
        let conv = CausalConv1d::new(inner_dim, inner_dim, config.conv_kernel_size);

        // Selective SSM
        let ssm = SelectiveSSM::new(config)?;

        // Output projection
        let scale = (2.0 / inner_dim as f32).sqrt();
        let out_proj = Array2::from_shape_fn((inner_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        Ok(Self {
            hidden_dim: config.hidden_dim,
            inner_dim,
            norm,
            in_proj,
            conv,
            ssm,
            out_proj,
        })
    }

    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let _span = tracing::debug_span!(
            "mamba_layer_forward",
            hidden_dim = self.hidden_dim,
            inner_dim = self.inner_dim
        )
        .entered();
        let batch_size = x.len().min(self.hidden_dim);

        // 1. Layer normalization
        let x_norm = self.norm.forward(x);

        // 2. Expansion and gating
        // Project to 2 * inner_dim, then split for SSM path and gate path.
        //
        // `x_norm.dot(&self.in_proj)` is bit-identical to the manual
        // column-strided loop it replaces when `x_norm.len()` matches
        // `in_proj`'s row count (the normal case: `hidden_dim`), and lets
        // ndarray dispatch to a correctly-strided, blocked matmul instead of
        // walking the row-major `in_proj` matrix one column-strided element
        // at a time. `x_norm` is derived from the caller-supplied `x`
        // (nothing upstream guarantees its length), so the original loop is
        // kept as an exact-behavior fallback — it implicitly zero-pads `x`
        // by only summing `j in 0..batch_size` — for the mismatched case.
        let projected = if x_norm.len() == self.in_proj.shape()[0] {
            x_norm.dot(&self.in_proj)
        } else {
            let mut projected = Array1::zeros(self.inner_dim * 2);
            for i in 0..(self.inner_dim * 2).min(self.in_proj.shape()[1]) {
                let mut sum = 0.0;
                for j in 0..batch_size {
                    sum += self.in_proj[[j, i]] * x_norm[j];
                }
                projected[i] = sum;
            }
            projected
        };

        // Split: first half for SSM, second half for gate
        let mut x_ssm = Array1::zeros(self.inner_dim);
        let mut x_gate = Array1::zeros(self.inner_dim);
        for i in 0..self.inner_dim {
            x_ssm[i] = projected[i];
            x_gate[i] = projected[self.inner_dim + i];
        }

        // 3. Short convolution on SSM path. `forward_step` takes `&[f32]`;
        // borrow `x_ssm` directly instead of paying for an extra `to_vec()`
        // allocation+copy when the array is contiguous (always true here,
        // since `x_ssm` was just built via `Array1::zeros` + indexed writes).
        let conv_out = match x_ssm.as_slice() {
            Some(slice) => self.conv.forward_step(slice),
            None => self.conv.forward_step(&x_ssm.to_vec()),
        };
        x_ssm = Array1::from_vec(conv_out);

        // 4. Selective SSM
        let ssm_out = self.ssm.forward_step(&x_ssm)?;

        // 5. Gating with SiLU (Swish)
        let gate = silu(&x_gate);

        // Element-wise multiplication
        let mut gated = Array1::zeros(ssm_out.len().min(gate.len()));
        for i in 0..gated.len() {
            gated[i] = ssm_out[i] * gate[i];
        }

        // 6. Output projection. Same traversal-order fix as step 2 above:
        // `gated.dot(&self.out_proj)` is bit-identical to the manual
        // column-strided loop when shapes line up exactly (the normal
        // case), and correctly-strided; fall back to the original loop
        // (which tolerates `gated`/`batch_size` being shorter than
        // `out_proj`'s dimensions) otherwise.
        let mut output =
            if gated.len() == self.out_proj.shape()[0] && batch_size == self.out_proj.shape()[1] {
                gated.dot(&self.out_proj)
            } else {
                let mut output = Array1::zeros(batch_size);
                for i in 0..batch_size {
                    let mut sum = 0.0;
                    for j in 0..gated.len().min(self.out_proj.shape()[0]) {
                        sum += self.out_proj[[j, i]] * gated[j];
                    }
                    output[i] = sum;
                }
                output
            };

        // 7. Residual connection
        for i in 0..output.len().min(x.len()) {
            output[i] += x[i];
        }

        Ok(output)
    }

    fn reset(&mut self) {
        self.ssm.reset();
        self.conv.reset();
    }
}

/// Mamba: Selective State Space Model
pub struct Mamba {
    config: MambaConfig,
    layers: Vec<MambaLayer>,
    input_proj: Array2<f32>,
    output_proj: Array2<f32>,
    /// Whether `MambaConfig::dropout` is active (see [`Mamba::set_training`]).
    training: bool,
}

impl Mamba {
    /// Create a new Mamba model
    #[instrument(skip(config), fields(input_dim = config.input_dim, hidden_dim = config.hidden_dim, num_layers = config.num_layers))]
    pub fn new(config: MambaConfig) -> ModelResult<Self> {
        debug!("Creating new Mamba model");
        config.validate()?;

        // Initialize layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            trace!("Initializing Mamba layer {}", layer_idx);
            layers.push(MambaLayer::new(&config)?);
        }
        debug!("Initialized {} Mamba layers", layers.len());

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

        debug!("Mamba model created successfully");
        Ok(Self {
            config,
            layers,
            input_proj,
            output_proj,
            training: false,
        })
    }

    /// Load pre-trained weights from a ModelLoader
    ///
    /// # Weight Format
    ///
    /// Expected weight names follow the pattern:
    /// - `input_proj`: Input projection weights
    /// - `output_proj`: Output projection weights
    /// - `layers.{i}.norm.weight`: Layer normalization weights
    /// - `layers.{i}.norm.bias`: Layer normalization bias (optional)
    /// - `layers.{i}.in_proj`: Input projection for layer i
    /// - `layers.{i}.conv.weight`: Convolution weights for layer i
    /// - `layers.{i}.conv.bias`: Convolution bias for layer i (optional)
    /// - `layers.{i}.ssm.log_a`: SSM diagonal A matrix (log space)
    /// - `layers.{i}.ssm.delta_proj`: SSM delta projection weights
    /// - `layers.{i}.ssm.delta_bias`: SSM delta projection bias
    /// - `layers.{i}.ssm.b_proj`: SSM B projection weights
    /// - `layers.{i}.ssm.c_proj`: SSM C projection weights
    /// - `layers.{i}.ssm.d_skip`: SSM skip connection weights
    /// - `layers.{i}.out_proj`: Output projection for layer i
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kizzasi_model::{Mamba, MambaConfig, loader::ModelLoader};
    ///
    /// let config = MambaConfig::new();
    /// let mut model = Mamba::new(config)?;
    /// let loader = ModelLoader::new("mamba_weights.safetensors")?;
    /// model.load_weights(&loader)?;
    /// ```
    pub fn load_weights(&mut self, loader: &crate::loader::ModelLoader) -> ModelResult<()> {
        // Load input/output projections
        if loader.has_tensor("input_proj") {
            self.input_proj = loader.load_array2("input_proj")?;
        }
        if loader.has_tensor("output_proj") {
            self.output_proj = loader.load_array2("output_proj")?;
        }

        // Load layer weights
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            // Load layer norm weights
            if loader.has_tensor(&format!("{}.norm.weight", prefix)) {
                let weight = loader.load_array1(&format!("{}.norm.weight", prefix))?;
                layer.norm.set_gamma(weight);
            }

            // Load input projection
            if loader.has_tensor(&format!("{}.in_proj", prefix)) {
                layer.in_proj = loader.load_array2(&format!("{}.in_proj", prefix))?;
            }

            // Load convolution weights
            if loader.has_tensor(&format!("{}.conv.weight", prefix)) {
                let weights_3d = loader.load_array3(&format!("{}.conv.weight", prefix))?;
                layer.conv.set_weights(weights_3d);
            }

            // Load SSM weights
            if loader.has_tensor(&format!("{}.ssm.log_a", prefix)) {
                layer.ssm.log_a = loader.load_array1(&format!("{}.ssm.log_a", prefix))?;
            }
            if loader.has_tensor(&format!("{}.ssm.delta_proj", prefix)) {
                layer.ssm.delta_proj = loader.load_array2(&format!("{}.ssm.delta_proj", prefix))?;
            }
            if loader.has_tensor(&format!("{}.ssm.delta_bias", prefix)) {
                layer.ssm.delta_bias = loader.load_array1(&format!("{}.ssm.delta_bias", prefix))?;
            }
            if loader.has_tensor(&format!("{}.ssm.b_proj", prefix)) {
                layer.ssm.b_proj = loader.load_array2(&format!("{}.ssm.b_proj", prefix))?;
            }
            if loader.has_tensor(&format!("{}.ssm.c_proj", prefix)) {
                layer.ssm.c_proj = loader.load_array2(&format!("{}.ssm.c_proj", prefix))?;
            }
            if loader.has_tensor(&format!("{}.ssm.d_skip", prefix)) {
                layer.ssm.d_skip = loader.load_array1(&format!("{}.ssm.d_skip", prefix))?;
            }

            // Load output projection
            if loader.has_tensor(&format!("{}.out_proj", prefix)) {
                layer.out_proj = loader.load_array2(&format!("{}.out_proj", prefix))?;
            }
        }

        Ok(())
    }

    /// Save model weights to a JSON file as `HashMap<String, Vec<f32>>`.
    ///
    /// # Format
    ///
    /// Keys follow the pattern:
    /// - `input_proj` / `output_proj`: top-level projection matrices (flattened row-major)
    /// - `layers.{i}.in_proj`, `layers.{i}.out_proj`: layer projections
    /// - `layers.{i}.ssm.log_a`, `layers.{i}.ssm.delta_proj`, `layers.{i}.ssm.delta_bias`,
    ///   `layers.{i}.ssm.b_proj`, `layers.{i}.ssm.c_proj`, `layers.{i}.ssm.d_skip`
    ///
    /// # Limitation: `conv` and `norm` are not saved
    ///
    /// `MambaLayer::conv` (a [`kizzasi_core::conv::CausalConv1d`]) and
    /// `MambaLayer::norm` (a [`kizzasi_core::LayerNorm`]) are **not**
    /// included above, and [`Self::load_weights_json`] never touches them
    /// either — `kizzasi-core` exposes `set_weights`/`set_bias` /
    /// `set_gamma`/`set_beta` setters for both types but no matching
    /// getters, so this crate has no way to read their *current* values
    /// back out to serialize them.
    ///
    /// Both are deterministically derived from layer config at construction
    /// time (Kaiming-style init for the conv kernel, ones/zeros for the norm
    /// gain/bias — no randomness involved), so a save→load round trip
    /// reproduces them correctly *only if* they were never modified after
    /// construction. For a model whose `conv`/`norm` **were** customized
    /// (e.g. loaded from a real trained checkpoint via
    /// [`kizzasi_core::conv::CausalConv1d::set_weights_checked`] or
    /// [`kizzasi_core::LayerNorm::set_gamma`]/`set_beta`), a save→load round
    /// trip silently discards that customization and the reloaded model
    /// reverts to the fresh from-config default instead.
    pub fn save_weights_json<P: AsRef<std::path::Path>>(&self, path: P) -> ModelResult<()> {
        warn!(
            "Mamba::save_weights_json does not persist conv/norm parameters \
             (kizzasi-core exposes no getters for CausalConv1d/LayerNorm); a \
             save\u{2192}load round trip silently drops any conv/norm \
             customization made after construction. See this method's docs."
        );
        let mut weights: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::new();

        // Top-level projections
        weights.insert(
            "input_proj".to_string(),
            self.input_proj.iter().copied().collect(),
        );
        weights.insert(
            "output_proj".to_string(),
            self.output_proj.iter().copied().collect(),
        );

        // Per-layer weights
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);
            weights.insert(
                format!("{}.in_proj", prefix),
                layer.in_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.out_proj", prefix),
                layer.out_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.ssm.log_a", prefix),
                layer.ssm.log_a.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.ssm.delta_proj", prefix),
                layer.ssm.delta_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.ssm.delta_bias", prefix),
                layer.ssm.delta_bias.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.ssm.b_proj", prefix),
                layer.ssm.b_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.ssm.c_proj", prefix),
                layer.ssm.c_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.ssm.d_skip", prefix),
                layer.ssm.d_skip.iter().copied().collect(),
            );
        }

        let file = std::fs::File::create(path.as_ref()).map_err(|e| {
            ModelError::load_error("mamba save_weights", format!("failed to create file: {e}"))
        })?;
        serde_json::to_writer(file, &weights).map_err(|e| {
            ModelError::load_error(
                "mamba save_weights",
                format!("JSON serialization failed: {e}"),
            )
        })?;
        Ok(())
    }

    /// Enable or disable training mode.
    ///
    /// Models are created in inference mode, where `MambaConfig::dropout` is inert
    /// and [`step`](crate::SignalPredictor::step) is deterministic. Set this to
    /// `true` during training so the configured dropout rate is applied to each
    /// layer output (inverted dropout — no rescale is needed when switching
    /// back to inference).
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Whether the model is currently in training mode (dropout active).
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Load weights from a JSON file previously written by `save_weights_json`.
    ///
    /// Only keys present in the file are applied; missing keys leave the current
    /// randomly-initialized values in place (graceful partial loading).
    ///
    /// Note: `conv`/`norm` are never among the applied keys, since
    /// `save_weights_json` never writes them — see that method's doc for why.
    pub fn load_weights_json<P: AsRef<std::path::Path>>(&mut self, path: P) -> ModelResult<()> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            ModelError::load_error("mamba load_weights", format!("failed to open file: {e}"))
        })?;
        let weights: std::collections::HashMap<String, Vec<f32>> = serde_json::from_reader(file)
            .map_err(|e| {
                ModelError::load_error(
                    "mamba load_weights",
                    format!("JSON deserialization failed: {e}"),
                )
            })?;
        self.load_weights_map(&weights).map(|_| ())
    }

    /// Load weights from an in-memory `name → flat f32 values` map.
    ///
    /// This is the in-process counterpart of [`Self::load_weights_json`]: it
    /// applies exactly the same shape checks and partial-loading semantics
    /// without routing the parameters through a serialized file.
    pub fn load_weights_map(
        &mut self,
        weights: &std::collections::HashMap<String, Vec<f32>>,
    ) -> ModelResult<usize> {
        // Number of tensors actually applied. The caller needs this to tell a
        // genuine partial load from a weight map whose names match nothing at
        // all — the latter would otherwise leave the model randomly
        // initialised while reporting success.
        let applied = std::cell::Cell::new(0usize);
        let load_array2 = |map: &std::collections::HashMap<String, Vec<f32>>,
                           key: &str,
                           rows: usize,
                           cols: usize|
         -> ModelResult<Option<Array2<f32>>> {
            if let Some(data) = map.get(key) {
                if data.len() != rows * cols {
                    return Err(ModelError::load_error(
                        "mamba load_weights",
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
                        "mamba load_weights",
                        format!("failed to reshape '{}': {e}", key),
                    )
                })?;
                applied.set(applied.get() + 1);
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
                        "mamba load_weights",
                        format!(
                            "shape mismatch for '{}': expected {} but got {}",
                            key,
                            expected_len,
                            data.len()
                        ),
                    ));
                }
                applied.set(applied.get() + 1);
                Ok(Some(Array1::from_vec(data.clone())))
            } else {
                Ok(None)
            }
        };

        let in_rows = self.config.input_dim;
        let in_cols = self.config.hidden_dim;
        if let Some(arr) = load_array2(weights, "input_proj", in_rows, in_cols)? {
            self.input_proj = arr;
        }
        let out_rows = self.config.hidden_dim;
        let out_cols = self.config.input_dim;
        if let Some(arr) = load_array2(weights, "output_proj", out_rows, out_cols)? {
            self.output_proj = arr;
        }

        let inner_dim = self.config.hidden_dim * self.config.expand_factor;
        let state_dim = self.config.state_dim;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            if let Some(arr) = load_array2(
                weights,
                &format!("{}.in_proj", prefix),
                self.config.hidden_dim,
                inner_dim * 2,
            )? {
                layer.in_proj = arr;
            }
            if let Some(arr) = load_array2(
                weights,
                &format!("{}.out_proj", prefix),
                inner_dim,
                self.config.hidden_dim,
            )? {
                layer.out_proj = arr;
            }
            if let Some(arr) = load_array1(weights, &format!("{}.ssm.log_a", prefix), state_dim)? {
                layer.ssm.log_a = arr;
            }
            if let Some(arr) = load_array2(
                weights,
                &format!("{}.ssm.delta_proj", prefix),
                inner_dim,
                inner_dim,
            )? {
                layer.ssm.delta_proj = arr;
            }
            if let Some(arr) =
                load_array1(weights, &format!("{}.ssm.delta_bias", prefix), inner_dim)?
            {
                layer.ssm.delta_bias = arr;
            }
            if let Some(arr) = load_array2(
                weights,
                &format!("{}.ssm.b_proj", prefix),
                inner_dim,
                state_dim,
            )? {
                layer.ssm.b_proj = arr;
            }
            if let Some(arr) = load_array2(
                weights,
                &format!("{}.ssm.c_proj", prefix),
                inner_dim,
                state_dim,
            )? {
                layer.ssm.c_proj = arr;
            }
            if let Some(arr) = load_array1(weights, &format!("{}.ssm.d_skip", prefix), inner_dim)? {
                layer.ssm.d_skip = arr;
            }
        }

        Ok(applied.get())
    }

    /// Save model weights to SafeTensors format.
    ///
    /// Serialises every parameter tensor (input projection, output projection,
    /// and per-layer SSM weights) as F32 little-endian bytes using the
    /// [SafeTensors](https://github.com/huggingface/safetensors) file format.
    /// The resulting file can be loaded by any SafeTensors-compatible reader.
    pub fn save_weights<P: AsRef<std::path::Path>>(&self, path: P) -> ModelResult<()> {
        // Phase 1: collect (name, raw-bytes, shape) for every tensor.
        // The byte Vecs must outlive the TensorView borrows in phase 2.
        let mut entries: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

        // Helper closures for Array1 / Array2.
        let array1_bytes =
            |a: &Array1<f32>| -> Vec<u8> { a.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let array2_bytes =
            |a: &Array2<f32>| -> Vec<u8> { a.iter().flat_map(|f| f.to_le_bytes()).collect() };

        // Top-level projections.
        entries.push((
            "input_proj".to_string(),
            array2_bytes(&self.input_proj),
            vec![self.config.input_dim, self.config.hidden_dim],
        ));
        entries.push((
            "output_proj".to_string(),
            array2_bytes(&self.output_proj),
            vec![self.config.hidden_dim, self.config.input_dim],
        ));

        // Per-layer weights.
        let inner_dim = self.config.hidden_dim * self.config.expand_factor;
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);
            let ssm = &layer.ssm;

            entries.push((
                format!("{}.in_proj", prefix),
                array2_bytes(&layer.in_proj),
                vec![self.config.hidden_dim, inner_dim * 2],
            ));
            entries.push((
                format!("{}.out_proj", prefix),
                array2_bytes(&layer.out_proj),
                vec![inner_dim, self.config.hidden_dim],
            ));
            entries.push((
                format!("{}.ssm.log_a", prefix),
                array1_bytes(&ssm.log_a),
                vec![self.config.state_dim],
            ));
            entries.push((
                format!("{}.ssm.delta_proj", prefix),
                array2_bytes(&ssm.delta_proj),
                vec![inner_dim, inner_dim],
            ));
            entries.push((
                format!("{}.ssm.delta_bias", prefix),
                array1_bytes(&ssm.delta_bias),
                vec![inner_dim],
            ));
            entries.push((
                format!("{}.ssm.b_proj", prefix),
                array2_bytes(&ssm.b_proj),
                vec![inner_dim, self.config.state_dim],
            ));
            entries.push((
                format!("{}.ssm.c_proj", prefix),
                array2_bytes(&ssm.c_proj),
                vec![inner_dim, self.config.state_dim],
            ));
            entries.push((
                format!("{}.ssm.d_skip", prefix),
                array1_bytes(&ssm.d_skip),
                vec![inner_dim],
            ));
        }

        // Phase 2: build TensorView slice — borrows from `entries`.
        let tensor_views: Vec<(String, TensorView<'_>)> = entries
            .iter()
            .map(|(name, data, shape)| {
                let view =
                    TensorView::new(Dtype::F32, shape.clone(), data.as_slice()).map_err(|e| {
                        ModelError::load_error(
                            "mamba save_weights",
                            format!("TensorView error for '{}': {}", name, e),
                        )
                    })?;
                Ok((name.clone(), view))
            })
            .collect::<ModelResult<Vec<_>>>()?;

        // Phase 3: write the file.
        safetensors::tensor::serialize_to_file(tensor_views, None, path.as_ref()).map_err(|e| {
            ModelError::load_error(
                "mamba save_weights",
                format!("safetensors serialize error: {}", e),
            )
        })?;

        Ok(())
    }

    /// Get the configuration
    pub fn config(&self) -> &MambaConfig {
        &self.config
    }
}

impl SignalPredictor for Mamba {
    #[instrument(skip(self, input), fields(input_size = input.len()))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        crate::check_input_dim(input, self.input_proj.shape()[0])?;
        trace!(
            "Mamba step input range: [{}, {}]",
            input.iter().cloned().fold(f32::INFINITY, f32::min),
            input.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // Project input to hidden dimension
        let mut hidden = input.dot(&self.input_proj);
        trace!("After input projection: hidden_dim={}", hidden.len());

        // Pass through each layer
        let dropout_rate = self.config.dropout;
        let training = self.training;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            trace!("Processing Mamba layer {}", layer_idx);
            hidden = layer.forward(&hidden)?;
            crate::dropout::apply_dropout(&mut hidden, dropout_rate, training);
        }

        // Project back to input dimension
        let output = hidden.dot(&self.output_proj);
        trace!(
            "Mamba step output range: [{}, {}]",
            output.iter().cloned().fold(f32::INFINITY, f32::min),
            output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        Ok(output)
    }

    #[instrument(skip(self))]
    fn reset(&mut self) {
        debug!("Resetting Mamba model state");
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            trace!("Resetting layer {}", layer_idx);
            layer.reset();
        }
    }

    fn context_window(&self) -> usize {
        // SSMs have theoretically infinite context via recurrence
        usize::MAX
    }
}

impl AutoregressiveModel for Mamba {
    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn state_dim(&self) -> usize {
        self.config.state_dim
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn model_type(&self) -> crate::ModelType {
        if self.config.use_mamba2 {
            crate::ModelType::Mamba2
        } else {
            crate::ModelType::Mamba
        }
    }

    fn get_states(&self) -> Vec<HiddenState> {
        self.layers
            .iter()
            .map(|layer| {
                let state = layer.ssm.state.clone();
                let mut hs = HiddenState::new(state.shape()[0], state.shape()[1]);
                hs.update(state);
                // Also save convolution history
                let conv_history = layer.conv.get_history();
                hs.set_conv_history(conv_history);
                hs
            })
            .collect()
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != self.config.num_layers {
            return Err(ModelError::state_count_mismatch(
                "Mamba",
                self.config.num_layers,
                states.len(),
            ));
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            layer.ssm.state = states[layer_idx].state().clone();
            // Also restore convolution history if available. `set_history`
            // validates the history against this layer's kernel size and
            // channel count; propagate a mismatch instead of silently
            // leaving the layer's old (stale or zeroed) history in place.
            if let Some(conv_history) = states[layer_idx].conv_history() {
                layer.conv.set_history(conv_history.clone())?;
            }
        }

        Ok(())
    }

    fn load_weights_json(&mut self, path: &std::path::Path) -> ModelResult<()> {
        Mamba::load_weights_json(self, path)
    }

    fn save_weights_json(&self, path: &std::path::Path) -> ModelResult<()> {
        Mamba::save_weights_json(self, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_weights_map_applies_values_in_memory() {
        let config = MambaConfig::new()
            .input_dim(2)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(1);
        let mut model = Mamba::new(config).expect("Mamba::new");

        let before = model.input_proj.clone();
        let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let mut weights = std::collections::HashMap::new();
        weights.insert("input_proj".to_string(), values.clone());

        model.load_weights_map(&weights).expect("load_weights_map");

        assert_ne!(
            model.input_proj, before,
            "load_weights_map must actually replace the projection"
        );
        for (i, v) in model.input_proj.iter().enumerate() {
            assert!(
                (v - values[i]).abs() < 1e-6,
                "element {i}: {v} != {}",
                values[i]
            );
        }
    }

    #[test]
    fn test_load_weights_map_reports_shape_mismatch() {
        let config = MambaConfig::new()
            .input_dim(2)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(1);
        let mut model = Mamba::new(config).expect("Mamba::new");

        let mut weights = std::collections::HashMap::new();
        weights.insert("input_proj".to_string(), vec![0.0f32; 5]);
        let err = model
            .load_weights_map(&weights)
            .expect_err("wrong-sized weight must be rejected");
        assert!(err.to_string().contains("input_proj"), "got: {err}");
    }

    #[test]
    fn test_dropout_is_applied_only_in_training_mode() {
        let mut config = MambaConfig::tiny(4);
        config.dropout = 0.5;
        let mut model = Mamba::new(config).expect("Mamba::new");
        let input = Array1::from_elem(4, 1.0f32);

        // Inference mode (the default): dropout is inert, `step` is deterministic.
        assert!(!model.is_training());
        let reference = model.step(&input).expect("step");
        model.reset();
        let repeat = model.step(&input).expect("step");
        for (i, (a, b)) in reference.iter().zip(repeat.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "inference must be deterministic; component {i}: {a} != {b}"
            );
        }

        // Training mode: the configured rate must actually perturb the forward
        // pass, *and* the mask must be resampled each time — a fixed mask would
        // be a constant reweighting, not a regularizer.
        model.set_training(true);
        assert!(model.is_training());
        let mut perturbed = false;
        let mut samples: Vec<Vec<f32>> = Vec::new();
        for _ in 0..32 {
            model.reset();
            let out = model.step(&input).expect("step");
            if out
                .iter()
                .zip(reference.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6)
            {
                perturbed = true;
            }
            samples.push(out.to_vec());
        }
        assert!(
            perturbed,
            "config.dropout = 0.5 was set but no forward pass was ever affected"
        );
        let varies = samples
            .iter()
            .any(|s| s.iter().zip(samples[0].iter()).any(|(a, b)| a != b));
        assert!(
            varies,
            "every training-mode step produced an identical result; the dropout mask is not resampled"
        );

        // Switching back restores the deterministic inference path.
        model.set_training(false);
        model.reset();
        let restored = model.step(&input).expect("step");
        for (i, (a, b)) in reference.iter().zip(restored.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "disabling training must restore inference output; component {i}: {a} != {b}"
            );
        }
    }

    #[test]
    fn test_mamba_creation() {
        let config = MambaConfig::new()
            .input_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let mamba = Mamba::new(config);
        assert!(mamba.is_ok());
    }

    #[test]
    fn test_mamba_step() {
        let config = MambaConfig::new()
            .input_dim(3)
            .hidden_dim(32)
            .state_dim(8)
            .num_layers(2);

        let mut mamba = Mamba::new(config).expect("Failed to create Mamba model");
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let output = mamba.step(&input);

        assert!(output.is_ok());
        assert_eq!(output.expect("Failed to get output").len(), 3);
    }

    #[test]
    fn test_mamba_tiny_config() {
        let config = MambaConfig::tiny(4);
        assert_eq!(config.hidden_dim, 128);
        assert_eq!(config.state_dim, 8);
        assert_eq!(config.num_layers, 2);
        assert!(!config.use_mamba2);

        let mut model = Mamba::new(config).expect("Failed to create Mamba model");
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let output = model.step(&input).expect("Failed to get output");
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_mamba_small_config() {
        // Test that small config has correct values
        let config = MambaConfig::small(4);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.state_dim, 16);
        assert_eq!(config.num_layers, 4);
        assert!(config.use_mamba2);

        // Use a minimal model to verify small config is valid (not full model)
        // Full model test is too slow for regular testing
        let minimal_config = MambaConfig::new()
            .input_dim(4)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);
        let mut model = Mamba::new(minimal_config).expect("Failed to create Mamba model");
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let output = model.step(&input).expect("Failed to get output");
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_mamba_base_config() {
        // Test that base config has correct values
        let config = MambaConfig::base(4);
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.state_dim, 16);
        assert_eq!(config.num_layers, 6);
        assert!(config.use_mamba2);

        // Use a minimal model to verify base config is valid (not full model)
        // Full model test is too slow for regular testing
        let minimal_config = MambaConfig::new()
            .input_dim(4)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);
        let mut model = Mamba::new(minimal_config).expect("Failed to create Mamba model");
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let output = model.step(&input).expect("Failed to get output");
        assert_eq!(output.len(), 4);
    }

    #[test]
    #[ignore] // Slow test: ~670s due to large model initialization (hidden_dim=1024, num_layers=12)
    fn test_mamba_large_config() {
        let config = MambaConfig::large(4);
        assert_eq!(config.hidden_dim, 1024);
        assert_eq!(config.state_dim, 32);
        assert_eq!(config.num_layers, 12);
        assert!(config.use_mamba2);

        let mut model = Mamba::new(config).expect("Failed to create Mamba model");
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let output = model.step(&input).expect("Failed to get output");
        assert_eq!(output.len(), 4);
    }

    #[test]
    #[ignore] // Slow test: ~610s due to very large model initialization (hidden_dim=2048, num_layers=24)
    fn test_mamba_xlarge_config() {
        let config = MambaConfig::xlarge(2);
        assert_eq!(config.hidden_dim, 2048);
        assert_eq!(config.state_dim, 64);
        assert_eq!(config.num_layers, 24);
        assert!(config.use_mamba2);

        // Create model to verify configuration is valid
        let model = Mamba::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_preset_configs_size_progression() {
        // Verify that model sizes increase progressively
        let tiny = MambaConfig::tiny(1);
        let small = MambaConfig::small(1);
        let base = MambaConfig::base(1);
        let large = MambaConfig::large(1);
        let xlarge = MambaConfig::xlarge(1);

        assert!(tiny.hidden_dim < small.hidden_dim);
        assert!(small.hidden_dim < base.hidden_dim);
        assert!(base.hidden_dim < large.hidden_dim);
        assert!(large.hidden_dim < xlarge.hidden_dim);

        assert!(tiny.num_layers <= small.num_layers);
        assert!(small.num_layers <= base.num_layers);
        assert!(base.num_layers <= large.num_layers);
        assert!(large.num_layers <= xlarge.num_layers);
    }

    #[test]
    fn test_mamba_save_load_roundtrip() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static MAMBA_ROUNDTRIP_COUNTER: AtomicU64 = AtomicU64::new(0);
        let uid = MAMBA_ROUNDTRIP_COUNTER.fetch_add(1, Ordering::Relaxed);

        let config = MambaConfig::new()
            .input_dim(1)
            .hidden_dim(32)
            .state_dim(8)
            .num_layers(2);

        let model = Mamba::new(config).expect("Failed to create Mamba model");

        let mut tmp = std::env::temp_dir();
        tmp.push(format!("kizzasi_mamba_roundtrip_test_{}.json", uid));

        model
            .save_weights_json(&tmp)
            .expect("save_weights_json failed");

        let config2 = MambaConfig::new()
            .input_dim(1)
            .hidden_dim(32)
            .state_dim(8)
            .num_layers(2);
        let mut model2 = Mamba::new(config2).expect("Failed to create second Mamba model");
        model2
            .load_weights_json(&tmp)
            .expect("load_weights_json failed");

        // Verify key count by re-saving and checking file is valid JSON
        let file = std::fs::File::open(&tmp).expect("temp file should exist");
        let reloaded: std::collections::HashMap<String, Vec<f32>> =
            serde_json::from_reader(file).expect("should deserialize");
        // 2 top-level + 8 per-layer × 2 layers = 18 keys
        assert_eq!(reloaded.len(), 18, "unexpected number of weight keys");

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_mamba_load_weights_shape_mismatch_error() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static MAMBA_SHAPE_MISMATCH_COUNTER: AtomicU64 = AtomicU64::new(0);
        let uid = MAMBA_SHAPE_MISMATCH_COUNTER.fetch_add(1, Ordering::Relaxed);

        let config = MambaConfig::new()
            .input_dim(1)
            .hidden_dim(32)
            .state_dim(8)
            .num_layers(1);

        let mut model = Mamba::new(config).expect("Failed to create Mamba model");

        let mut tmp = std::env::temp_dir();
        tmp.push(format!("kizzasi_mamba_shape_mismatch_test_{}.json", uid));

        // Write deliberately wrong-shaped weights
        let mut bad_weights: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::new();
        // input_proj should be (1, 32) = 32 elements; provide wrong size
        bad_weights.insert("input_proj".to_string(), vec![0.1f32; 5]);
        let file = std::fs::File::create(&tmp).expect("should create temp file");
        serde_json::to_writer(file, &bad_weights).expect("should serialize");

        let result = model.load_weights_json(&tmp);
        assert!(result.is_err(), "expected shape mismatch error");

        let _ = std::fs::remove_file(&tmp);
    }

    /// Verify that when |a_n| < 1e-8 and delta >= 0.001 (the else branch),
    /// the ZOH discretization correctly uses the L'Hôpital limit Δ·B instead
    /// of the former broken substitute `-1.0` that produced near-zero state.
    #[test]
    fn test_mamba_bbar_zoh_limit_near_zero_a() {
        // Use hidden_dim=4, state_dim=4, expand_factor=1 so inner_dim=4
        let config = MambaConfig::new()
            .input_dim(4)
            .hidden_dim(4)
            .state_dim(4)
            .num_layers(1);

        let mut ssm = SelectiveSSM::new(&config).expect("SelectiveSSM::new failed");

        // Force |a_n| ≈ 9.4e-14  (well inside the < 1e-8 guard).
        // log_a = -30 → a_n = -exp(-30) ≈ -9.4e-14
        ssm.log_a.fill(-30.0);

        // Force delta to clamp to 0.1 (>= 0.001), triggering the else branch.
        // softplus(2.0) ≈ 2.127 which the clamp reduces to 0.1.
        ssm.delta_bias.fill(2.0);
        // Zero out delta_proj so the projected contribution is 0 and
        // only the bias term controls delta.
        ssm.delta_proj.fill(0.0);

        // Set b_proj so that b_vec is nonzero and predictable.
        ssm.b_proj.fill(1.0);

        // Simple unit input.
        let x = Array1::from_elem(4, 1.0_f32);

        // After the fix: b_bar = delta * b_vec (Taylor), state ≠ 0.
        // Before the fix: b_bar = (a_bar - 1) / (-1) * b_vec ≈ 0 (since a_bar ≈ 1).
        let _output = ssm.forward_step(&x).expect("forward_step failed");

        // The state must have at least one nonzero entry after absorbing the input.
        assert!(
            ssm.state.iter().any(|&v| v.abs() > 1e-6),
            "state should be nonzero after ZOH limit fix (delta={:.4}, a_n≈{:.2e})",
            0.1_f32,
            (-30.0_f32).exp()
        );
    }

    /// Regression guard: the common path (default HiPPO init, |a_n| >= 1)
    /// must still produce finite outputs after the b_bar change.
    #[test]
    fn test_mamba_bbar_default_init_regression() {
        let config = MambaConfig::new()
            .input_dim(4)
            .hidden_dim(4)
            .state_dim(4)
            .num_layers(1);

        let mut ssm = SelectiveSSM::new(&config).expect("SelectiveSSM::new failed");

        let x = Array1::from_vec(vec![0.1_f32, 0.2, 0.3, 0.4]);
        let output = ssm.forward_step(&x).expect("forward_step failed");

        assert!(
            output.iter().all(|v| v.is_finite()),
            "all output elements must be finite with default HiPPO init"
        );
    }

    #[test]
    fn test_save_weights_roundtrip_safetensors() {
        let config = MambaConfig {
            input_dim: 4,
            hidden_dim: 8,
            state_dim: 4,
            num_layers: 1,
            expand_factor: 2,
            ..Default::default()
        };
        let model = Mamba::new(config).expect("model creation failed");
        let tmp = std::env::temp_dir().join("test_mamba_save_weights.safetensors");
        model.save_weights(&tmp).expect("save_weights failed");
        let data = std::fs::read(&tmp).expect("read file");
        let tensors = safetensors::SafeTensors::deserialize(&data).expect("deserialize");
        assert!(!tensors.names().is_empty(), "no tensors written");
        std::fs::remove_file(&tmp).ok();
    }
}
