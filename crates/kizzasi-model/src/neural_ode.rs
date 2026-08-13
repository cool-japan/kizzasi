//! Neural ODE / Continuous-Time Models
//!
//! Neural ODEs model dynamics as dx/dt = f(x, t), where f is a neural network.
//! This allows continuous-time evolution of hidden states for signal prediction,
//! providing a fundamentally different approach from discrete-time SSMs.
//!
//! # Architecture
//!
//! ```text
//! Input -> [Input Projection] -> x₀
//!   x₀ -> ODE Solver(dx/dt = f(x, t), t₀, t₁) -> x₁
//!   x₁ -> [Output Projection] -> Prediction
//! ```
//!
//! The dynamics function f is a multi-layer neural network with tanh activations
//! that takes the current state and time as inputs and produces the rate of change.
//!
//! # Solver Methods
//!
//! - **Euler**: First-order, simple but less accurate
//! - **Midpoint**: Second-order Runge-Kutta, better accuracy
//! - **RK4**: Fourth-order Runge-Kutta, high accuracy for smooth dynamics
//! - **Adaptive RK45**: Dormand-Prince with automatic step size control
//!
//! # Augmented Neural ODE
//!
//! The augmented variant pads the state with extra zero-initialized dimensions,
//! giving the ODE more capacity to learn complex dynamics without crossing
//! trajectories in the original state space.
//!
//! # References
//!
//! - "Neural Ordinary Differential Equations" (Chen et al., 2018)
//! - "Augmented Neural ODEs" (Dupont et al., 2019)

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{CoreResult, HiddenState, SignalPredictor};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

// ---------------------------------------------------------------------------
// Seeded deterministic RNG for reproducible weight initialization
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG for deterministic weight initialization.
struct SeededRng {
    state: u64,
}

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    /// Returns a float in [-1, 1)
    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
    }
}

// ---------------------------------------------------------------------------
// ODE Solver enum
// ---------------------------------------------------------------------------

/// ODE solver methods for numerical integration
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub enum OdeSolver {
    /// Forward Euler method (1st order)
    Euler,
    /// Explicit midpoint method (2nd order Runge-Kutta)
    Midpoint,
    /// Classic 4th order Runge-Kutta
    #[default]
    Rk4,
    /// Adaptive Dormand-Prince RK45 with error-based step control
    AdaptiveRk45 {
        /// Error tolerance for step size adaptation
        tol: f32,
    },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Neural ODE model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralOdeConfig {
    /// Input dimension (signal width)
    pub input_dim: usize,
    /// Hidden dimension of the ODE state
    pub hidden_dim: usize,
    /// Number of layers in the dynamics network f
    pub num_layers: usize,
    /// ODE solver method
    pub solver: OdeSolver,
    /// Integration step size
    pub dt: f32,
    /// Number of integration steps per forward call
    pub integration_steps: usize,
    /// Context length (for SignalPredictor)
    pub context_length: usize,
}

impl Default for NeuralOdeConfig {
    fn default() -> Self {
        Self {
            input_dim: 1,
            hidden_dim: 256,
            num_layers: 3,
            solver: OdeSolver::Rk4,
            dt: 0.01,
            integration_steps: 10,
            context_length: 4096,
        }
    }
}

impl NeuralOdeConfig {
    /// Create a small configuration for quick experiments
    pub fn small(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 64,
            num_layers: 2,
            solver: OdeSolver::Rk4,
            dt: 0.01,
            integration_steps: 5,
            context_length: 1024,
        }
    }

    /// Create a base configuration
    pub fn base(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 256,
            num_layers: 3,
            solver: OdeSolver::Rk4,
            dt: 0.01,
            integration_steps: 10,
            context_length: 4096,
        }
    }

    /// Create a large configuration
    pub fn large(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 512,
            num_layers: 4,
            solver: OdeSolver::AdaptiveRk45 { tol: 1e-5 },
            dt: 0.005,
            integration_steps: 20,
            context_length: 8192,
        }
    }

    /// Validate the configuration
    fn validate(&self) -> ModelResult<()> {
        if self.hidden_dim == 0 {
            return Err(ModelError::invalid_config("hidden_dim must be > 0"));
        }
        if self.num_layers == 0 {
            return Err(ModelError::invalid_config("num_layers must be > 0"));
        }
        if self.dt <= 0.0 {
            return Err(ModelError::invalid_config("dt must be positive"));
        }
        if self.integration_steps == 0 {
            return Err(ModelError::invalid_config("integration_steps must be > 0"));
        }
        if self.input_dim == 0 {
            return Err(ModelError::invalid_config("input_dim must be > 0"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ODE Dynamics: the neural network f(x, t)
// ---------------------------------------------------------------------------

/// Neural network that defines the ODE dynamics: dx/dt = f(x, t).
///
/// The network consists of multiple layers with tanh activations.
/// Time information is embedded via sin/cos features added to the input.
pub struct OdeDynamics {
    /// Layer weights and biases: (weight, bias) pairs
    layers: Vec<(Array2<f32>, Array1<f32>)>,
    /// Hidden dimension of the state
    hidden_dim: usize,
}

impl OdeDynamics {
    /// Create new ODE dynamics network
    ///
    /// The network maps R^(hidden_dim + 2) -> R^hidden_dim, where the +2
    /// comes from sin(t) and cos(t) time embeddings.
    pub fn new(hidden_dim: usize, num_layers: usize) -> ModelResult<Self> {
        if hidden_dim == 0 {
            return Err(ModelError::invalid_config(
                "OdeDynamics: hidden_dim must be > 0",
            ));
        }
        if num_layers == 0 {
            return Err(ModelError::invalid_config(
                "OdeDynamics: num_layers must be > 0",
            ));
        }

        let mut rng = SeededRng::new(42_u64);
        let mut layers = Vec::with_capacity(num_layers);

        // Input dimension includes time embedding (sin(t), cos(t))
        let input_with_time = hidden_dim + 2;

        for layer_idx in 0..num_layers {
            let (in_dim, out_dim) = if layer_idx == 0 {
                (input_with_time, hidden_dim)
            } else {
                (hidden_dim, hidden_dim)
            };

            // Xavier/Glorot initialization
            let scale = (2.0 / (in_dim + out_dim) as f32).sqrt();
            let mut weight = Array2::zeros((in_dim, out_dim));
            let mut bias = Array1::zeros(out_dim);

            for i in 0..in_dim {
                for j in 0..out_dim {
                    weight[[i, j]] = rng.next_f32() * scale;
                }
            }
            for j in 0..out_dim {
                bias[j] = rng.next_f32() * 0.01;
            }

            layers.push((weight, bias));
        }

        Ok(Self { layers, hidden_dim })
    }

    /// Evaluate the dynamics function: dx/dt = f(x, t)
    ///
    /// Time embedding (sin(t), cos(t)) is concatenated to the state vector
    /// before being passed through the network.
    pub fn forward(&self, x: &Array1<f32>, t: f32) -> ModelResult<Array1<f32>> {
        if x.len() != self.hidden_dim {
            return Err(ModelError::dimension_mismatch(
                "OdeDynamics::forward",
                self.hidden_dim,
                x.len(),
            ));
        }

        // Build input with time embedding: [x, sin(t), cos(t)]
        let mut input = Array1::zeros(self.hidden_dim + 2);
        for i in 0..self.hidden_dim {
            input[i] = x[i];
        }
        input[self.hidden_dim] = t.sin();
        input[self.hidden_dim + 1] = t.cos();

        let mut hidden = input;

        for (layer_idx, (weight, bias)) in self.layers.iter().enumerate() {
            // Linear: h = input @ weight + bias
            let pre_activation = hidden.dot(weight) + bias;

            // Tanh activation for all layers except the last
            // Last layer has no activation to allow unbounded derivatives
            if layer_idx < self.layers.len() - 1 {
                hidden = pre_activation.mapv(f32::tanh);
            } else {
                hidden = pre_activation;
            }
        }

        // Check for numerical issues
        for &val in hidden.iter() {
            if val.is_nan() || val.is_infinite() {
                return Err(ModelError::numerical_instability(
                    "OdeDynamics::forward",
                    format!("NaN or Inf detected in dynamics output at t={t}"),
                ));
            }
        }

        Ok(hidden)
    }
}

// ---------------------------------------------------------------------------
// ODE Integrator
// ---------------------------------------------------------------------------

/// Numerical ODE integrator implementing various solver methods.
pub struct OdeIntegrator {
    solver: OdeSolver,
    dt: f32,
    steps: usize,
}

impl OdeIntegrator {
    /// Create a new integrator
    pub fn new(solver: OdeSolver, dt: f32, steps: usize) -> Self {
        Self { solver, dt, steps }
    }

    /// Integrate from t0 over `steps` time steps.
    ///
    /// Returns the final state after integration.
    pub fn integrate<F>(&self, x0: &Array1<f32>, t0: f32, dynamics: &F) -> ModelResult<Array1<f32>>
    where
        F: Fn(&Array1<f32>, f32) -> ModelResult<Array1<f32>>,
    {
        let mut x = x0.clone();
        let mut t = t0;

        match &self.solver {
            OdeSolver::Euler => {
                for _ in 0..self.steps {
                    x = Self::euler_step(&x, t, self.dt, dynamics)?;
                    t += self.dt;
                }
            }
            OdeSolver::Midpoint => {
                for _ in 0..self.steps {
                    x = Self::midpoint_step(&x, t, self.dt, dynamics)?;
                    t += self.dt;
                }
            }
            OdeSolver::Rk4 => {
                for _ in 0..self.steps {
                    x = Self::rk4_step(&x, t, self.dt, dynamics)?;
                    t += self.dt;
                }
            }
            OdeSolver::AdaptiveRk45 { tol } => {
                let tol = *tol;
                let t_end = t0 + self.dt * self.steps as f32;
                let mut current_dt = self.dt;
                let min_dt = self.dt * 1e-6;
                let max_dt = self.dt * 10.0;
                let max_iterations = self.steps * 1000; // safety limit
                let t_eps = self.dt * 1e-8; // floating-point tolerance for end check

                let mut iterations = 0;
                while t < t_end - t_eps && iterations < max_iterations {
                    // Don't overshoot the end time
                    let remaining = t_end - t;
                    if current_dt > remaining {
                        current_dt = remaining;
                    }
                    if current_dt < min_dt {
                        current_dt = min_dt;
                    }

                    let (x_new, new_dt) =
                        Self::adaptive_rk45_step(&x, t, current_dt, tol, dynamics)?;
                    x = x_new;
                    t += current_dt;
                    current_dt = new_dt.clamp(min_dt, max_dt);
                    iterations += 1;
                }
            }
        }

        Ok(x)
    }

    /// Forward Euler step: x_{n+1} = x_n + dt * f(x_n, t_n)
    fn euler_step<F>(x: &Array1<f32>, t: f32, dt: f32, f: &F) -> ModelResult<Array1<f32>>
    where
        F: Fn(&Array1<f32>, f32) -> ModelResult<Array1<f32>>,
    {
        let k1 = f(x, t)?;
        Ok(x + &(&k1 * dt))
    }

    /// Explicit midpoint step (2nd order Runge-Kutta):
    /// k1 = f(x, t)
    /// k2 = f(x + dt/2 * k1, t + dt/2)
    /// x_{n+1} = x_n + dt * k2
    fn midpoint_step<F>(x: &Array1<f32>, t: f32, dt: f32, f: &F) -> ModelResult<Array1<f32>>
    where
        F: Fn(&Array1<f32>, f32) -> ModelResult<Array1<f32>>,
    {
        let k1 = f(x, t)?;
        let x_mid = x + &(&k1 * (dt * 0.5));
        let k2 = f(&x_mid, t + dt * 0.5)?;
        Ok(x + &(&k2 * dt))
    }

    /// Classic 4th order Runge-Kutta:
    /// k1 = f(x, t)
    /// k2 = f(x + dt/2 * k1, t + dt/2)
    /// k3 = f(x + dt/2 * k2, t + dt/2)
    /// k4 = f(x + dt * k3, t + dt)
    /// x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    fn rk4_step<F>(x: &Array1<f32>, t: f32, dt: f32, f: &F) -> ModelResult<Array1<f32>>
    where
        F: Fn(&Array1<f32>, f32) -> ModelResult<Array1<f32>>,
    {
        let k1 = f(x, t)?;
        let x2 = x + &(&k1 * (dt * 0.5));
        let k2 = f(&x2, t + dt * 0.5)?;
        let x3 = x + &(&k2 * (dt * 0.5));
        let k3 = f(&x3, t + dt * 0.5)?;
        let x4 = x + &(&k3 * dt);
        let k4 = f(&x4, t + dt)?;

        // x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        let increment = (&k1 + &(&k2 * 2.0) + &(&k3 * 2.0) + &k4) * (dt / 6.0);
        Ok(x + &increment)
    }

    /// Adaptive Dormand-Prince RK45 step with error estimation.
    ///
    /// Returns (new_state, suggested_next_dt).
    ///
    /// Uses the Dormand-Prince tableau for embedded RK4(5) pair.
    /// The error is estimated as the difference between the 4th and 5th order solutions.
    fn adaptive_rk45_step<F>(
        x: &Array1<f32>,
        t: f32,
        dt: f32,
        tol: f32,
        f: &F,
    ) -> ModelResult<(Array1<f32>, f32)>
    where
        F: Fn(&Array1<f32>, f32) -> ModelResult<Array1<f32>>,
    {
        // Dormand-Prince coefficients (simplified for practical use)
        // We compute a 4th order and 5th order solution and use the difference as error.

        let k1 = f(x, t)?;

        let x2 = x + &(&k1 * (dt * (1.0 / 5.0)));
        let k2 = f(&x2, t + dt * (1.0 / 5.0))?;

        let x3 = x + &(&(&k1 * (3.0 / 40.0) + &(&k2 * (9.0 / 40.0))) * dt);
        let k3 = f(&x3, t + dt * (3.0 / 10.0))?;

        let x4 =
            x + &(&(&k1 * (44.0 / 45.0) + &(&k2 * (-56.0 / 15.0)) + &(&k3 * (32.0 / 9.0))) * dt);
        let k4 = f(&x4, t + dt * (4.0 / 5.0))?;

        let x5 = x + &(&(&k1 * (19372.0 / 6561.0)
            + &(&k2 * (-25360.0 / 2187.0))
            + &(&k3 * (64448.0 / 6561.0))
            + &(&k4 * (-212.0 / 729.0)))
            * dt);
        let k5 = f(&x5, t + dt * (8.0 / 9.0))?;

        let x6 = x + &(&(&k1 * (9017.0 / 3168.0)
            + &(&k2 * (-355.0 / 33.0))
            + &(&k3 * (46732.0 / 5247.0))
            + &(&k4 * (49.0 / 176.0))
            + &(&k5 * (-5103.0 / 18656.0)))
            * dt);
        let k6 = f(&x6, t + dt)?;

        // 5th order solution (Dormand-Prince)
        let y5 = x + &(&(&k1 * (35.0 / 384.0)
            + &(&k3 * (500.0 / 1113.0))
            + &(&k4 * (125.0 / 192.0))
            + &(&k5 * (-2187.0 / 6784.0))
            + &(&k6 * (11.0 / 84.0)))
            * dt);

        // 4th order solution for error estimation
        let y4 = x + &(&(&k1 * (5179.0 / 57600.0)
            + &(&k3 * (7571.0 / 16695.0))
            + &(&k4 * (393.0 / 640.0))
            + &(&k5 * (-92097.0 / 339200.0))
            + &(&k6 * (187.0 / 2100.0)))
            * dt);

        // Error estimate
        let error_vec = &y5 - &y4;
        let error_norm = error_vec
            .iter()
            .map(|&e| e * e)
            .sum::<f32>()
            .sqrt()
            .max(1e-10);

        // Compute optimal step size using standard PI controller
        let safety = 0.9_f32;
        let order = 5.0_f32;
        let scale = safety * (tol / error_norm).powf(1.0 / order);
        let new_dt = dt * scale.clamp(0.2, 5.0);

        Ok((y5, new_dt))
    }
}

// ---------------------------------------------------------------------------
// Neural ODE Model
// ---------------------------------------------------------------------------

/// Complete Neural ODE model for signal prediction.
///
/// Combines an input projection, ODE dynamics network, ODE integrator,
/// and output projection into a complete autoregressive model.
pub struct NeuralOdeModel {
    /// Model configuration
    pub config: NeuralOdeConfig,
    /// Dynamics function: dx/dt = f(x, t)
    dynamics: OdeDynamics,
    /// ODE solver/integrator
    integrator: OdeIntegrator,
    /// Input projection: input_dim -> hidden_dim
    input_proj: Array2<f32>,
    /// Output projection: hidden_dim -> input_dim
    output_proj: Array2<f32>,
    /// Current hidden state
    state: Array1<f32>,
    /// Current time in the ODE integration
    current_time: f32,
}

impl NeuralOdeModel {
    /// Create a new Neural ODE model from configuration
    pub fn new(config: NeuralOdeConfig) -> ModelResult<Self> {
        config.validate()?;

        let dynamics = OdeDynamics::new(config.hidden_dim, config.num_layers)?;
        let integrator =
            OdeIntegrator::new(config.solver.clone(), config.dt, config.integration_steps);

        let mut rng = SeededRng::new(12345_u64);

        // Input projection: input_dim -> hidden_dim (Xavier init)
        let scale_in = (2.0 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let mut input_proj = Array2::zeros((config.input_dim, config.hidden_dim));
        for i in 0..config.input_dim {
            for j in 0..config.hidden_dim {
                input_proj[[i, j]] = rng.next_f32() * scale_in;
            }
        }

        // Output projection: hidden_dim -> input_dim (Xavier init)
        let scale_out = (2.0 / (config.hidden_dim + config.input_dim) as f32).sqrt();
        let mut output_proj = Array2::zeros((config.hidden_dim, config.input_dim));
        for i in 0..config.hidden_dim {
            for j in 0..config.input_dim {
                output_proj[[i, j]] = rng.next_f32() * scale_out;
            }
        }

        let state = Array1::zeros(config.hidden_dim);

        Ok(Self {
            config,
            dynamics,
            integrator,
            input_proj,
            output_proj,
            state,
            current_time: 0.0,
        })
    }

    /// Create a small model preset
    pub fn small() -> ModelResult<Self> {
        Self::new(NeuralOdeConfig::small(1))
    }

    /// Create a base model preset
    pub fn base() -> ModelResult<Self> {
        Self::new(NeuralOdeConfig::base(1))
    }

    /// Create a large model preset
    pub fn large() -> ModelResult<Self> {
        Self::new(NeuralOdeConfig::large(1))
    }

    /// Get the current ODE time
    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    /// Set the ODE time (e.g., for restarting integration from a different point)
    pub fn set_time(&mut self, t: f32) {
        self.current_time = t;
    }
}

impl SignalPredictor for NeuralOdeModel {
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        crate::check_input_dim(input, self.input_proj.shape()[0])?;

        // 1. Project input to hidden space
        let h = input.dot(&self.input_proj);

        // 2. Add projected input to current state
        let x = &self.state + &h;

        // 3. Integrate the ODE: state = solve(dynamics, x, t, t + dt*steps)
        let t0 = self.current_time;
        let dynamics = &self.dynamics;
        let new_state = self
            .integrator
            .integrate(&x, t0, &|state, t| dynamics.forward(state, t))
            .map_err(|e| {
                kizzasi_core::CoreError::InferenceError(format!("NeuralODE integration: {e}"))
            })?;

        // 4. Clamp state for numerical stability
        self.state = new_state.mapv(|v| v.clamp(-100.0, 100.0));

        // 5. Advance time
        self.current_time += self.config.dt * self.config.integration_steps as f32;

        // 6. Project state to output space
        let output = self.state.dot(&self.output_proj);
        Ok(output)
    }

    fn reset(&mut self) {
        self.state = Array1::zeros(self.config.hidden_dim);
        self.current_time = 0.0;
    }

    fn context_window(&self) -> usize {
        self.config.context_length
    }
}

impl AutoregressiveModel for NeuralOdeModel {
    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn state_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn model_type(&self) -> ModelType {
        ModelType::NeuralOde
    }

    fn get_states(&self) -> Vec<HiddenState> {
        // Neural ODE has a single layer of state (the hidden state vector)
        let mut hs = HiddenState::new(self.config.hidden_dim, 1);
        let state_2d = self
            .state
            .clone()
            .into_shape_with_order((self.config.hidden_dim, 1));
        if let Ok(s) = state_2d {
            hs.update(s);
        }
        vec![hs]
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != 1 {
            return Err(ModelError::state_count_mismatch(
                "NeuralODE",
                1,
                states.len(),
            ));
        }
        let hs = &states[0];
        let state_2d = hs.state();
        // Extract the first column as the hidden state
        for i in 0..self.config.hidden_dim.min(state_2d.nrows()) {
            self.state[i] = state_2d[[i, 0]];
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Augmented Neural ODE
// ---------------------------------------------------------------------------

/// Augmented Neural ODE: augments the state space with extra dimensions.
///
/// This gives the ODE more capacity to learn complex dynamics by adding
/// extra zero-initialized dimensions to the state vector, preventing
/// trajectory crossings in the augmented space.
pub struct AugmentedNeuralOde {
    /// Inner Neural ODE model (operates in augmented space)
    inner: NeuralOdeModel,
    /// Number of extra augmented dimensions
    augment_dim: usize,
    /// Original hidden dim (before augmentation)
    original_hidden_dim: usize,
}

impl AugmentedNeuralOde {
    /// Create an augmented Neural ODE.
    ///
    /// The inner model operates on hidden_dim + augment_dim dimensions.
    pub fn new(mut config: NeuralOdeConfig, augment_dim: usize) -> ModelResult<Self> {
        if augment_dim == 0 {
            return Err(ModelError::invalid_config(
                "AugmentedNeuralOde: augment_dim must be > 0",
            ));
        }

        let original_hidden_dim = config.hidden_dim;
        // Augment the hidden dimension
        config.hidden_dim += augment_dim;

        // We need a custom input/output projection to handle the augmented space
        let inner = NeuralOdeModel::new(config)?;

        Ok(Self {
            inner,
            augment_dim,
            original_hidden_dim,
        })
    }

    /// Get the effective (augmented) dimension
    pub fn effective_dim(&self) -> usize {
        self.original_hidden_dim + self.augment_dim
    }

    /// Get the original hidden dimension (before augmentation)
    pub fn original_dim(&self) -> usize {
        self.original_hidden_dim
    }

    /// Get the augmentation dimension
    pub fn augment_dim(&self) -> usize {
        self.augment_dim
    }

    /// Get the current ODE time
    pub fn current_time(&self) -> f32 {
        self.inner.current_time()
    }

    /// Set the ODE time
    pub fn set_time(&mut self, t: f32) {
        self.inner.set_time(t);
    }
}

impl SignalPredictor for AugmentedNeuralOde {
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // The inner model already has an augmented hidden dim.
        // The input projection from inner handles input_dim -> (hidden_dim + augment_dim).
        // We call the inner step directly, and the output projection handles
        // (hidden_dim + augment_dim) -> input_dim.
        self.inner.step(input)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn context_window(&self) -> usize {
        self.inner.context_window()
    }
}

impl AutoregressiveModel for AugmentedNeuralOde {
    fn hidden_dim(&self) -> usize {
        self.original_hidden_dim
    }

    fn state_dim(&self) -> usize {
        self.effective_dim()
    }

    fn num_layers(&self) -> usize {
        self.inner.config.num_layers
    }

    fn model_type(&self) -> ModelType {
        ModelType::NeuralOde
    }

    fn get_states(&self) -> Vec<HiddenState> {
        self.inner.get_states()
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        self.inner.set_states(states)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    /// Test Euler solver on dx/dt = -x (exponential decay)
    #[test]
    fn test_euler_simple_ode() -> ModelResult<()> {
        let integrator = OdeIntegrator::new(OdeSolver::Euler, 0.001, 1000);
        let x0 = Array1::from_vec(vec![1.0]);

        // dx/dt = -x => x(t) = exp(-t)
        let result =
            integrator.integrate(&x0, 0.0, &|x: &Array1<f32>, _t: f32| Ok(x.mapv(|v| -v)))?;

        // After t=1.0, x should be close to exp(-1) ≈ 0.3679
        let expected = (-1.0_f32).exp();
        let error = (result[0] - expected).abs();
        assert!(
            error < 0.01,
            "Euler error too large: got {}, expected {}, error={}",
            result[0],
            expected,
            error
        );
        Ok(())
    }

    /// Test midpoint solver is more accurate than Euler for same dt
    #[test]
    fn test_midpoint_accuracy() -> ModelResult<()> {
        let dt = 0.01;
        let steps = 100; // t=0 to t=1

        let euler_integrator = OdeIntegrator::new(OdeSolver::Euler, dt, steps);
        let midpoint_integrator = OdeIntegrator::new(OdeSolver::Midpoint, dt, steps);

        let x0 = Array1::from_vec(vec![1.0]);
        let dynamics =
            |x: &Array1<f32>, _t: f32| -> ModelResult<Array1<f32>> { Ok(x.mapv(|v| -v)) };

        let euler_result = euler_integrator.integrate(&x0, 0.0, &dynamics)?;
        let midpoint_result = midpoint_integrator.integrate(&x0, 0.0, &dynamics)?;

        let expected = (-1.0_f32).exp();
        let euler_error = (euler_result[0] - expected).abs();
        let midpoint_error = (midpoint_result[0] - expected).abs();

        assert!(
            midpoint_error < euler_error,
            "Midpoint ({}) should be more accurate than Euler ({})",
            midpoint_error,
            euler_error
        );
        Ok(())
    }

    /// Test RK4 is the most accurate fixed-step solver
    #[test]
    fn test_rk4_accuracy() -> ModelResult<()> {
        let dt = 0.01;
        let steps = 100;

        let midpoint_integrator = OdeIntegrator::new(OdeSolver::Midpoint, dt, steps);
        let rk4_integrator = OdeIntegrator::new(OdeSolver::Rk4, dt, steps);

        let x0 = Array1::from_vec(vec![1.0]);
        let dynamics =
            |x: &Array1<f32>, _t: f32| -> ModelResult<Array1<f32>> { Ok(x.mapv(|v| -v)) };

        let midpoint_result = midpoint_integrator.integrate(&x0, 0.0, &dynamics)?;
        let rk4_result = rk4_integrator.integrate(&x0, 0.0, &dynamics)?;

        let expected = (-1.0_f32).exp();
        let midpoint_error = (midpoint_result[0] - expected).abs();
        let rk4_error = (rk4_result[0] - expected).abs();

        assert!(
            rk4_error < midpoint_error,
            "RK4 ({}) should be more accurate than Midpoint ({})",
            rk4_error,
            midpoint_error
        );
        // RK4 should be very accurate (f32 precision limits to ~1e-7)
        assert!(
            rk4_error < 1e-6,
            "RK4 error should be very small, got {}",
            rk4_error
        );
        Ok(())
    }

    /// Test adaptive RK45 solver adjusts step size
    #[test]
    fn test_adaptive_rk45() -> ModelResult<()> {
        let integrator = OdeIntegrator::new(OdeSolver::AdaptiveRk45 { tol: 1e-6 }, 0.01, 100);

        let x0 = Array1::from_vec(vec![1.0]);
        let result =
            integrator.integrate(&x0, 0.0, &|x: &Array1<f32>, _t: f32| Ok(x.mapv(|v| -v)))?;

        let expected = (-1.0_f32).exp();
        let error = (result[0] - expected).abs();
        assert!(
            error < 0.01,
            "Adaptive RK45 error too large: got {}, expected {}, error={}",
            result[0],
            expected,
            error
        );
        Ok(())
    }

    /// Test Neural ODE model creation with small preset
    #[test]
    fn test_neural_ode_model_creation() -> ModelResult<()> {
        let model = NeuralOdeModel::small()?;

        assert_eq!(model.config.hidden_dim, 64);
        assert_eq!(model.config.num_layers, 2);
        assert_eq!(model.config.input_dim, 1);
        assert_eq!(model.state.len(), 64);
        assert_eq!(model.current_time(), 0.0);
        Ok(())
    }

    /// Test single forward step produces correct output shape
    #[test]
    fn test_neural_ode_forward() -> ModelResult<()> {
        let mut model = NeuralOdeModel::small()?;
        let input = Array1::from_vec(vec![0.5]);

        let output = model
            .step(&input)
            .map_err(|e| ModelError::forward_error(0, e.to_string()))?;

        assert_eq!(output.len(), 1, "Output should match input_dim=1");
        assert!(!output[0].is_nan(), "Output should not be NaN");
        assert!(!output[0].is_infinite(), "Output should not be infinite");
        // Time should have advanced
        assert!(model.current_time() > 0.0);
        Ok(())
    }

    /// Test 10 consecutive steps without NaN
    #[test]
    fn test_neural_ode_multi_step() -> ModelResult<()> {
        let mut model = NeuralOdeModel::small()?;
        let input = Array1::from_vec(vec![0.1]);

        for step_idx in 0..10 {
            let output = model
                .step(&input)
                .map_err(|e| ModelError::forward_error(step_idx, e.to_string()))?;
            assert!(
                !output[0].is_nan(),
                "Step {step_idx}: output should not be NaN"
            );
            assert!(
                !output[0].is_infinite(),
                "Step {step_idx}: output should not be infinite"
            );
        }

        // Verify time has advanced appropriately
        let expected_time = 10.0 * model.config.dt * model.config.integration_steps as f32;
        let time_diff = (model.current_time() - expected_time).abs();
        assert!(
            time_diff < 1e-5,
            "Time mismatch: got {}, expected {}",
            model.current_time(),
            expected_time
        );
        Ok(())
    }

    /// Test that reset produces deterministic output
    #[test]
    fn test_neural_ode_state_reset() -> ModelResult<()> {
        let mut model = NeuralOdeModel::small()?;
        let input = Array1::from_vec(vec![0.5]);

        // Run a few steps
        for _ in 0..5 {
            let _ = model
                .step(&input)
                .map_err(|e| ModelError::forward_error(0, e.to_string()))?;
        }

        // Reset
        model.reset();
        assert_eq!(model.current_time(), 0.0);

        // First step after reset
        let output_after_reset = model
            .step(&input)
            .map_err(|e| ModelError::forward_error(0, e.to_string()))?;

        // Create a fresh model and take one step
        let mut fresh_model = NeuralOdeModel::small()?;
        let output_fresh = fresh_model
            .step(&input)
            .map_err(|e| ModelError::forward_error(0, e.to_string()))?;

        // Outputs should match
        let diff = (output_after_reset[0] - output_fresh[0]).abs();
        assert!(
            diff < 1e-6,
            "Reset output ({}) should match fresh model output ({}), diff={}",
            output_after_reset[0],
            output_fresh[0],
            diff
        );
        Ok(())
    }

    /// Test SignalPredictor trait is implemented
    #[test]
    fn test_neural_ode_signal_predictor() -> ModelResult<()> {
        let model = NeuralOdeModel::small()?;

        // Verify the trait methods
        assert_eq!(model.context_window(), 1024);

        // Use as trait object
        let predictor: Box<dyn SignalPredictor> = Box::new(model);
        assert_eq!(predictor.context_window(), 1024);
        Ok(())
    }

    /// Test AutoregressiveModel get_states/set_states roundtrip
    #[test]
    fn test_neural_ode_autoregressive() -> ModelResult<()> {
        let mut model = NeuralOdeModel::small()?;
        let input = Array1::from_vec(vec![0.3]);

        // Run a few steps to build up state
        for _ in 0..3 {
            let _ = model
                .step(&input)
                .map_err(|e| ModelError::forward_error(0, e.to_string()))?;
        }

        // Get states
        let states = model.get_states();
        assert_eq!(states.len(), 1, "Neural ODE should have 1 state");

        // Create a new model and set the states
        let mut model2 = NeuralOdeModel::small()?;
        model2.set_states(states)?;

        // Both models should produce the same output for the same input
        let out1 = model
            .step(&input)
            .map_err(|e| ModelError::forward_error(0, e.to_string()))?;

        // Set time to match
        model2.set_time(
            model.current_time() - model.config.dt * model.config.integration_steps as f32,
        );
        let out2 = model2
            .step(&input)
            .map_err(|e| ModelError::forward_error(0, e.to_string()))?;

        let diff = (out1[0] - out2[0]).abs();
        assert!(
            diff < 1e-4,
            "States roundtrip: outputs should be close, got diff={}",
            diff
        );
        Ok(())
    }

    /// Test augmented Neural ODE has larger effective dimension
    #[test]
    fn test_augmented_ode() -> ModelResult<()> {
        let config = NeuralOdeConfig::small(1);
        let original_dim = config.hidden_dim;
        let augment_dim = 16;

        let mut aug_model = AugmentedNeuralOde::new(config, augment_dim)?;

        assert_eq!(aug_model.effective_dim(), original_dim + augment_dim);
        assert_eq!(aug_model.original_dim(), original_dim);
        assert_eq!(aug_model.augment_dim(), augment_dim);

        // Should be able to step
        let input = Array1::from_vec(vec![0.5]);
        let output = aug_model
            .step(&input)
            .map_err(|e| ModelError::forward_error(0, e.to_string()))?;

        assert_eq!(output.len(), 1);
        assert!(!output[0].is_nan());
        assert!(!output[0].is_infinite());
        Ok(())
    }

    /// Test numerical stability with large and small inputs
    #[test]
    fn test_neural_ode_numerical_stability() -> ModelResult<()> {
        let mut model = NeuralOdeModel::small()?;

        // Large input
        let large_input = Array1::from_vec(vec![1000.0]);
        let output_large = model
            .step(&large_input)
            .map_err(|e| ModelError::forward_error(0, e.to_string()))?;
        assert!(
            !output_large[0].is_nan(),
            "Large input should not produce NaN"
        );
        assert!(
            !output_large[0].is_infinite(),
            "Large input should not produce Inf"
        );

        model.reset();

        // Small input
        let small_input = Array1::from_vec(vec![1e-10]);
        let output_small = model
            .step(&small_input)
            .map_err(|e| ModelError::forward_error(0, e.to_string()))?;
        assert!(
            !output_small[0].is_nan(),
            "Small input should not produce NaN"
        );
        assert!(
            !output_small[0].is_infinite(),
            "Small input should not produce Inf"
        );

        model.reset();

        // Negative input
        let neg_input = Array1::from_vec(vec![-500.0]);
        let output_neg = model
            .step(&neg_input)
            .map_err(|e| ModelError::forward_error(0, e.to_string()))?;
        assert!(
            !output_neg[0].is_nan(),
            "Negative input should not produce NaN"
        );
        assert!(
            !output_neg[0].is_infinite(),
            "Negative input should not produce Inf"
        );
        Ok(())
    }

    /// Test OdeDynamics directly
    #[test]
    fn test_ode_dynamics_forward() -> ModelResult<()> {
        let dynamics = OdeDynamics::new(8, 2)?;
        let x = Array1::zeros(8);
        let result = dynamics.forward(&x, 0.0)?;

        assert_eq!(result.len(), 8);
        for &val in result.iter() {
            assert!(!val.is_nan(), "Dynamics output should not be NaN");
        }
        Ok(())
    }

    /// Test OdeDynamics dimension mismatch error
    #[test]
    fn test_ode_dynamics_dimension_mismatch() {
        let dynamics = OdeDynamics::new(8, 2).expect("should create dynamics");
        let wrong_input = Array1::zeros(4);
        let result = dynamics.forward(&wrong_input, 0.0);
        assert!(result.is_err(), "Should error on dimension mismatch");
    }

    /// Test config validation
    #[test]
    fn test_config_validation() {
        let config = NeuralOdeConfig {
            hidden_dim: 0,
            ..NeuralOdeConfig::default()
        };
        assert!(NeuralOdeModel::new(config).is_err());

        let config2 = NeuralOdeConfig {
            dt: -0.01,
            ..NeuralOdeConfig::default()
        };
        assert!(NeuralOdeModel::new(config2).is_err());

        let config3 = NeuralOdeConfig {
            integration_steps: 0,
            ..NeuralOdeConfig::default()
        };
        assert!(NeuralOdeModel::new(config3).is_err());
    }

    /// Test model presets
    #[test]
    fn test_model_presets() -> ModelResult<()> {
        let small = NeuralOdeModel::small()?;
        let base = NeuralOdeModel::base()?;
        let large = NeuralOdeModel::large()?;

        assert!(small.config.hidden_dim < base.config.hidden_dim);
        assert!(base.config.hidden_dim < large.config.hidden_dim);
        Ok(())
    }
}
