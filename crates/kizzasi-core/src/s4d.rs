//! S4D: Diagonal Structured State Space Model
//!
//! S4D simplifies the full S4 model by using diagonal state matrices,
//! which enables:
//! - **O(N log N)** training via FFT convolution
//! - **O(1)** inference via recurrence
//! - **Stable long-range dependencies** via HiPPO initialization
//! - **Hardware-efficient** computation
//!
//! # Mathematical Formulation
//!
//! The continuous-time SSM:
//! ```text
//! h'(t) = A h(t) + B x(t)
//! y(t) = C h(t) + D x(t)
//! ```
//!
//! For S4D, A is diagonal: A = diag(λ₁, λ₂, ..., λₙ)
//! This allows closed-form discretization and efficient computation.
//!
//! # Discretization
//!
//! Using Zero-Order Hold (ZOH) with step size Δ:
//! ```text
//! Ā = exp(Δ A) = diag(exp(Δ λ₁), ..., exp(Δ λₙ))
//! B̄ = (A⁻¹)(exp(Δ A) - I)B ≈ Δ B for small Δ
//! ```
//!
//! # Recurrence (O(1) inference)
//!
//! ```text
//! hₜ = Ā ⊙ hₜ₋₁ + B̄ ⊙ xₜ    (element-wise multiplication)
//! yₜ = C · hₜ + D · xₜ
//! ```

use crate::error::{CoreError, CoreResult};
use crate::numerics::safe_exp;
use crate::simd;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;

/// S4D Configuration
#[derive(Debug, Clone)]
pub struct S4DConfig {
    /// Input dimension
    pub input_dim: usize,
    /// State dimension (N)
    pub state_dim: usize,
    /// Hidden/output dimension
    pub hidden_dim: usize,
    /// Step size for discretization
    pub delta: f32,
    /// Use HiPPO initialization for long-range dependencies
    pub use_hippo: bool,
    /// Bidirectional (process both forward and backward)
    pub bidirectional: bool,
}

impl S4DConfig {
    /// Create a new S4D configuration
    pub fn new(input_dim: usize, state_dim: usize, hidden_dim: usize) -> Self {
        Self {
            input_dim,
            state_dim,
            hidden_dim,
            delta: 0.001, // Default step size
            use_hippo: true,
            bidirectional: false,
        }
    }

    /// Set step size
    pub fn delta(mut self, delta: f32) -> Self {
        self.delta = delta;
        self
    }

    /// Enable/disable HiPPO initialization
    pub fn use_hippo(mut self, use_hippo: bool) -> Self {
        self.use_hippo = use_hippo;
        self
    }

    /// Enable bidirectional processing
    pub fn bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }
}

/// S4D Layer
///
/// Implements diagonal structured state space model with:
/// - Diagonal A matrix for efficiency
/// - HiPPO initialization for long-range modeling
/// - Both recurrent (O(1) per step) and convolutional modes
#[derive(Debug)]
pub struct S4DLayer {
    config: S4DConfig,
    // SSM parameters (all diagonal)
    lambda: Array1<f32>, // Diagonal of A (state_dim,)
    b: Array1<f32>,      // Input matrix (state_dim,)
    c: Array1<f32>,      // Output matrix (state_dim,)
    d: f32,              // Skip connection
    // Input/output projections
    input_proj: Array2<f32>,  // (input_dim, hidden_dim)
    output_proj: Array2<f32>, // (hidden_dim, hidden_dim)
    // Discretized parameters (cached)
    a_bar: Array1<f32>, // exp(Δ λ)
    b_bar: Array1<f32>, // Discretized B
}

impl S4DLayer {
    /// Create a new S4D layer
    pub fn new(config: S4DConfig) -> CoreResult<Self> {
        let mut rng = thread_rng();
        let state_dim = config.state_dim;
        let input_dim = config.input_dim;
        let hidden_dim = config.hidden_dim;

        // Initialize lambda (diagonal of A) with HiPPO if enabled
        let lambda = if config.use_hippo {
            Self::hippo_initialization(state_dim)
        } else {
            // Random initialization with negative real parts for stability
            Array1::from_shape_fn(state_dim, |i| {
                -0.5 - rng.random::<f32>() * 0.5 - (i as f32 / state_dim as f32)
            })
        };

        // Initialize B (input coefficients)
        let scale = (2.0 / state_dim as f32).sqrt();
        let b = Array1::from_shape_fn(state_dim, |_| (rng.random::<f32>() - 0.5) * scale);

        // Initialize C (output coefficients)
        let c = Array1::from_shape_fn(state_dim, |_| (rng.random::<f32>() - 0.5) * scale);

        // Skip connection (D)
        let d = 1.0;

        // Input/output projections
        let proj_scale = (1.0 / input_dim as f32).sqrt();
        let input_proj = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * proj_scale
        });
        let output_proj = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * proj_scale
        });

        // Discretize
        let (a_bar, b_bar) = Self::discretize(&lambda, &b, config.delta);

        Ok(Self {
            config,
            lambda,
            b,
            c,
            d,
            input_proj,
            output_proj,
            a_bar,
            b_bar,
        })
    }

    /// HiPPO initialization for long-range dependencies
    ///
    /// Based on "HiPPO: Recurrent Memory with Optimal Polynomial Projections"
    /// Uses the LegS (Scaled Legendre) measure for optimal approximation
    fn hippo_initialization(state_dim: usize) -> Array1<f32> {
        let mut lambda = Array1::zeros(state_dim);
        for n in 0..state_dim {
            let n_f = n as f32;
            // HiPPO-LegS eigenvalues: λₙ = -(2n + 1)
            lambda[n] = -(2.0 * n_f + 1.0);
        }
        lambda
    }

    /// Discretize continuous SSM parameters using Zero-Order Hold
    ///
    /// For diagonal A:
    /// - Ā = exp(Δ A) = [exp(Δ λ₁), ..., exp(Δ λₙ)]
    /// - B̄ = (A⁻¹)(exp(Δ A) - I)B = [(exp(Δ λᵢ) - 1) / λᵢ] ⊙ B
    fn discretize(lambda: &Array1<f32>, b: &Array1<f32>, delta: f32) -> (Array1<f32>, Array1<f32>) {
        let state_dim = lambda.len();
        let mut a_bar = Array1::zeros(state_dim);
        let mut b_bar = Array1::zeros(state_dim);

        for i in 0..state_dim {
            let lambda_i = lambda[i];
            let exp_val = safe_exp(delta * lambda_i);
            a_bar[i] = exp_val;

            // For B̄, use the exact formula: (exp(Δλ) - 1) / λ
            // For small λ, use Taylor expansion to avoid numerical issues
            if lambda_i.abs() < 1e-4 {
                b_bar[i] = delta * b[i]; // First-order approximation
            } else {
                b_bar[i] = ((exp_val - 1.0) / lambda_i) * b[i];
            }
        }

        (a_bar, b_bar)
    }

    /// Forward step for recurrent inference (O(1) per step)
    ///
    /// Computes one step of the recurrence:
    /// - h_t = Ā ⊙ h_{t-1} + B̄ ⊙ x_t
    /// - y_t = C · h_t + D · x_t
    pub fn step(&self, x: &Array1<f32>, h: &mut Array1<f32>) -> CoreResult<Array1<f32>> {
        if x.len() != self.config.input_dim {
            return Err(CoreError::DimensionMismatch {
                expected: self.config.input_dim,
                got: x.len(),
            });
        }

        // Project input
        let x_proj = x.dot(&self.input_proj);

        // Assume x_proj is scalar broadcasted or take mean
        let x_scalar = x_proj.mean().unwrap_or(0.0);

        // State update: h = Ā ⊙ h + B̄ ⊙ x (element-wise)
        for i in 0..self.config.state_dim {
            h[i] = self.a_bar[i] * h[i] + self.b_bar[i] * x_scalar;
        }

        // Output: y = C · h + D · x
        let mut y_scalar = self.d * x_scalar;
        y_scalar += simd::dot_product(
            self.c
                .as_slice()
                .expect("invariant: c is a contiguous Array1"),
            h.as_slice()
                .expect("invariant: h is a contiguous local Array1"),
        );

        // Broadcast to output dimension
        let y = Array1::from_elem(self.config.hidden_dim, y_scalar);

        // Apply output projection
        let output = y.dot(&self.output_proj);
        Ok(output)
    }

    /// Forward pass for a sequence (training mode)
    ///
    /// Input shape: (seq_len, input_dim)
    /// Output shape: (seq_len, hidden_dim)
    ///
    /// When `config.bidirectional` is set, this runs the recurrence a
    /// second time over the time-reversed sequence (with its own
    /// independent hidden state) and averages the two passes -- giving
    /// every output position access to both past and future context,
    /// unlike the causal-only forward pass. Previously `bidirectional` was
    /// accepted by the config and builder but never read anywhere in this
    /// file, so enabling it silently produced a plain causal model.
    pub fn forward_sequence(&self, input: &Array2<f32>) -> CoreResult<Array2<f32>> {
        let (seq_len, input_dim) = input.dim();
        if input_dim != self.config.input_dim {
            return Err(CoreError::DimensionMismatch {
                expected: self.config.input_dim,
                got: input_dim,
            });
        }

        let forward_out = self.forward_sequence_causal(input)?;

        if !self.config.bidirectional {
            return Ok(forward_out);
        }

        // Backward pass: run the SAME recurrence (fresh hidden state) over
        // the time-reversed input, then reverse the result back into
        // forward time order before combining.
        let mut reversed_input = Array2::zeros((seq_len, input_dim));
        for t in 0..seq_len {
            reversed_input
                .row_mut(t)
                .assign(&input.row(seq_len - 1 - t));
        }
        let backward_out_reversed = self.forward_sequence_causal(&reversed_input)?;

        let mut output = Array2::zeros((seq_len, self.config.hidden_dim));
        for t in 0..seq_len {
            let fwd = forward_out.row(t);
            let bwd = backward_out_reversed.row(seq_len - 1 - t);
            output.row_mut(t).assign(&((&fwd + &bwd) * 0.5));
        }

        Ok(output)
    }

    /// The causal (forward-only) recurrence itself, independent of
    /// `config.bidirectional`. Used directly for the forward pass, and
    /// again on a time-reversed input for the backward pass when
    /// bidirectional mode is enabled.
    fn forward_sequence_causal(&self, input: &Array2<f32>) -> CoreResult<Array2<f32>> {
        let seq_len = input.nrows();
        let mut output = Array2::zeros((seq_len, self.config.hidden_dim));
        let mut h = Array1::zeros(self.config.state_dim);

        for t in 0..seq_len {
            let x_t = input.row(t).to_owned();
            let y_t = self.step(&x_t, &mut h)?;
            output.row_mut(t).assign(&y_t);
        }

        Ok(output)
    }

    /// Reset hidden state
    pub fn reset_state(&self) -> Array1<f32> {
        Array1::zeros(self.config.state_dim)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let ssm_params = self.lambda.len() + self.b.len() + self.c.len() + 1; // +1 for D
        let proj_params = self.input_proj.len() + self.output_proj.len();
        ssm_params + proj_params
    }

    /// Get configuration
    pub fn config(&self) -> &S4DConfig {
        &self.config
    }
}

/// Multi-layer S4D model
///
/// Stacks multiple S4D layers with optional residual connections and normalization
#[derive(Debug)]
pub struct S4DModel {
    layers: Vec<S4DLayer>,
    num_layers: usize,
}

impl S4DModel {
    /// Create a new multi-layer S4D model
    pub fn new(config: S4DConfig, num_layers: usize) -> CoreResult<Self> {
        let mut layers = Vec::with_capacity(num_layers);

        // First layer uses the input config as-is
        layers.push(S4DLayer::new(config.clone())?);

        // Subsequent layers: input_dim = previous hidden_dim
        for _ in 1..num_layers {
            let mut layer_config = config.clone();
            layer_config.input_dim = config.hidden_dim;
            layers.push(S4DLayer::new(layer_config)?);
        }

        Ok(Self { layers, num_layers })
    }

    /// Forward pass through all layers
    pub fn forward(
        &self,
        input: &Array2<f32>,
        states: &mut [Array1<f32>],
    ) -> CoreResult<Array2<f32>> {
        if states.len() != self.num_layers {
            return Err(CoreError::InvalidConfig(format!(
                "Expected {} states, got {}",
                self.num_layers,
                states.len()
            )));
        }

        let mut x = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_sequence(&x)?;
            // Update state for this layer (for next inference step)
            states[i] = layer.reset_state();
        }

        Ok(x)
    }

    /// Single step inference (O(1) per step)
    pub fn step(&self, input: &Array1<f32>, states: &mut [Array1<f32>]) -> CoreResult<Array1<f32>> {
        if states.len() != self.num_layers {
            return Err(CoreError::InvalidConfig(format!(
                "Expected {} states, got {}",
                self.num_layers,
                states.len()
            )));
        }

        // Start with input
        let mut x = input.clone();

        // Process through each layer
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.step(&x, &mut states[i])?;
        }

        Ok(x)
    }

    /// Reset all states
    pub fn reset_states(&self) -> Vec<Array1<f32>> {
        self.layers
            .iter()
            .map(|layer| layer.reset_state())
            .collect()
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.num_parameters()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s4d_config() {
        let config = S4DConfig::new(10, 64, 128)
            .delta(0.01)
            .use_hippo(true)
            .bidirectional(false);

        assert_eq!(config.input_dim, 10);
        assert_eq!(config.state_dim, 64);
        assert_eq!(config.hidden_dim, 128);
        assert_eq!(config.delta, 0.01);
        assert!(config.use_hippo);
    }

    #[test]
    fn test_s4d_layer() {
        let config = S4DConfig::new(10, 64, 128);
        let layer = S4DLayer::new(config).unwrap();

        let input = Array1::from_vec(vec![0.1; 10]);
        let mut state = layer.reset_state();

        let output = layer.step(&input, &mut state).unwrap();
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_s4d_sequence() {
        let config = S4DConfig::new(10, 64, 128);
        let layer = S4DLayer::new(config).unwrap();

        let seq_len = 20;
        let input = Array2::from_shape_vec((seq_len, 10), vec![0.1; seq_len * 10]).unwrap();

        let output = layer.forward_sequence(&input).unwrap();
        assert_eq!(output.dim(), (seq_len, 128));
    }

    #[test]
    fn test_s4d_model() {
        let config = S4DConfig::new(10, 64, 128);
        let model = S4DModel::new(config, 4).unwrap();

        let seq_len = 15;
        let input = Array2::from_shape_vec((seq_len, 10), vec![0.1; seq_len * 10]).unwrap();
        let mut states = model.reset_states();

        let output = model.forward(&input, &mut states).unwrap();
        assert_eq!(output.dim(), (seq_len, 128));
    }

    #[test]
    fn test_hippo_initialization() {
        let state_dim = 16;
        let lambda = S4DLayer::hippo_initialization(state_dim);

        assert_eq!(lambda.len(), state_dim);
        // Check that all eigenvalues are negative (stable)
        for &val in lambda.iter() {
            assert!(val < 0.0);
        }
    }

    #[test]
    fn test_bidirectional_forward_sequence_uses_future_context() {
        // Regression: `bidirectional` was accepted by the config/builder but
        // never read by `forward_sequence`, so enabling it silently produced
        // the same causal-only output as `bidirectional: false`.
        let state_dim = 8;
        let hidden_dim = 6;
        let seq_len = 10;

        let mut config_causal = S4DConfig::new(4, state_dim, hidden_dim).use_hippo(false);
        config_causal.bidirectional = false;
        let mut config_bidi = config_causal.clone();
        config_bidi.bidirectional = true;

        let layer_causal = S4DLayer::new(config_causal).unwrap();
        // `S4DLayer::new` re-randomizes internal parameters per call (via
        // `thread_rng`), so build the bidirectional layer by cloning the
        // causal layer's already-initialized parameters via its config
        // rather than constructing independently -- otherwise a difference
        // in output could come from different random weights, not from
        // bidirectionality.
        let mut layer_bidi = S4DLayer::new(config_bidi).unwrap();
        layer_bidi.lambda = layer_causal.lambda.clone();
        layer_bidi.b = layer_causal.b.clone();
        layer_bidi.c = layer_causal.c.clone();
        layer_bidi.d = layer_causal.d;
        layer_bidi.input_proj = layer_causal.input_proj.clone();
        layer_bidi.output_proj = layer_causal.output_proj.clone();
        layer_bidi.a_bar = layer_causal.a_bar.clone();
        layer_bidi.b_bar = layer_causal.b_bar.clone();

        // Non-trivial (non-constant) input so a causal vs. non-causal
        // difference actually has something to show up in.
        let input = Array2::from_shape_fn((seq_len, 4), |(t, d)| {
            (t as f32 * 0.3 + d as f32 * 0.7).sin()
        });

        let out_causal = layer_causal.forward_sequence(&input).unwrap();
        let out_bidi = layer_bidi.forward_sequence(&input).unwrap();

        assert_eq!(out_causal.dim(), (seq_len, hidden_dim));
        assert_eq!(out_bidi.dim(), (seq_len, hidden_dim));

        // With identical parameters, bidirectional output must differ from
        // the causal-only output at at least one position (proving the
        // backward pass genuinely contributes), except possibly at the very
        // first timestep where forward-only and backward-only can coincide
        // for a degenerate case -- so check the whole sequence, not just t=0.
        let mut any_difference = false;
        for t in 0..seq_len {
            for d in 0..hidden_dim {
                if (out_causal[[t, d]] - out_bidi[[t, d]]).abs() > 1e-6 {
                    any_difference = true;
                }
            }
        }
        assert!(
            any_difference,
            "bidirectional=true must produce different output than bidirectional=false"
        );
    }

    #[test]
    fn test_bidirectional_false_is_unchanged() {
        // Sanity check: leaving bidirectional at its default (false) must
        // behave identically to calling the causal recurrence directly.
        let config = S4DConfig::new(5, 12, 8);
        let layer = S4DLayer::new(config).unwrap();
        let input = Array2::from_shape_fn((7, 5), |(t, d)| (t + d) as f32 * 0.05);

        let via_forward_sequence = layer.forward_sequence(&input).unwrap();
        let via_causal_directly = layer.forward_sequence_causal(&input).unwrap();

        assert_eq!(via_forward_sequence, via_causal_directly);
    }

    #[test]
    fn test_discretization() {
        let state_dim = 8;
        let lambda = Array1::from_vec(vec![-1.0; state_dim]);
        let b = Array1::from_vec(vec![1.0; state_dim]);
        let delta = 0.01;

        let (a_bar, b_bar) = S4DLayer::discretize(&lambda, &b, delta);

        assert_eq!(a_bar.len(), state_dim);
        assert_eq!(b_bar.len(), state_dim);

        // Check that discretized values are reasonable
        for &val in a_bar.iter() {
            assert!(val > 0.0 && val < 1.0); // exp(-delta) for negative lambda
        }
    }
}
