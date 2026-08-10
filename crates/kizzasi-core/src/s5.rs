//! # S5 (Simplified State Space Layers)
//!
//! Implementation of S5, a simplified and efficient state space model architecture.
//!
//! ## Key Features
//!
//! - **Simplified Structure**: More straightforward than S4D with better scaling
//! - **Parallel Training**: Efficient parallelization during training
//! - **Multi-Input Multi-Output (MIMO)**: Support for multiple input/output channels
//! - **Stable Parameterization**: Numerically stable initialization and updates
//! - **Efficient Inference**: O(1) recurrent mode for deployment
//!
//! ## References
//!
//! - "Simplified State Space Layers for Sequence Modeling" (Smith et al., 2023)

use crate::{numerics::zoh_discretize, CoreError, CoreResult, HiddenState};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Configuration for S5 layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S5Config {
    /// Input dimension
    pub d_input: usize,
    /// Model dimension (hidden size)
    pub d_model: usize,
    /// State dimension (N)
    pub d_state: usize,
    /// Number of blocks (P)
    pub n_blocks: usize,
    /// Whether to use complex-valued state spaces
    pub use_complex: bool,
    /// Discretization timestep
    pub dt: f32,
    /// Use bias in linear projections
    pub use_bias: bool,
    /// Dropout rate (for training)
    pub dropout: f32,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
}

impl Default for S5Config {
    fn default() -> Self {
        Self {
            d_input: 1,
            d_model: 256,
            d_state: 64,
            n_blocks: 8,
            use_complex: false,
            dt: 0.001,
            use_bias: false,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
        }
    }
}

impl S5Config {
    /// Create a new S5 configuration
    pub fn new(d_input: usize, d_model: usize, d_state: usize) -> Self {
        Self {
            d_input,
            d_model,
            d_state,
            ..Default::default()
        }
    }

    /// Set number of blocks
    pub fn with_blocks(mut self, n_blocks: usize) -> Self {
        self.n_blocks = n_blocks;
        self
    }

    /// Set timestep
    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Use complex-valued state spaces
    pub fn with_complex(mut self, use_complex: bool) -> Self {
        self.use_complex = use_complex;
        self
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Block size (d_model / n_blocks)
    pub fn block_size(&self) -> usize {
        self.d_model / self.n_blocks
    }

    /// Validate configuration
    pub fn validate(&self) -> CoreResult<()> {
        if self.d_input == 0 {
            return Err(CoreError::InvalidConfig("d_input must be > 0".into()));
        }
        if self.d_model == 0 {
            return Err(CoreError::InvalidConfig("d_model must be > 0".into()));
        }
        if self.d_state == 0 {
            return Err(CoreError::InvalidConfig("d_state must be > 0".into()));
        }
        if self.n_blocks == 0 {
            return Err(CoreError::InvalidConfig("n_blocks must be > 0".into()));
        }
        if !self.d_model.is_multiple_of(self.n_blocks) {
            return Err(CoreError::InvalidConfig(
                "d_model must be divisible by n_blocks".into(),
            ));
        }
        if self.dt <= 0.0 {
            return Err(CoreError::InvalidConfig("dt must be > 0".into()));
        }
        Ok(())
    }
}

/// S5 Layer with block-diagonal state space structure
///
/// Uses a block-diagonal parameterization where each block operates independently,
/// enabling efficient parallelization and better scaling.
pub struct S5Layer {
    config: S5Config,
    /// Input projection (d_input -> d_model)
    in_proj_w: Array2<f32>,
    in_proj_b: Option<Array1<f32>>,
    /// Output projection (d_model -> d_model)
    out_proj_w: Array2<f32>,
    out_proj_b: Option<Array1<f32>>,
    /// Block-diagonal A matrices (n_blocks, d_state, d_state) - stored for reference
    #[allow(dead_code)]
    a_matrices: Vec<Array2<f32>>,
    /// B matrices per block (n_blocks, d_state, block_size) - stored for reference
    #[allow(dead_code)]
    b_matrices: Vec<Array2<f32>>,
    /// C matrices per block (n_blocks, block_size, d_state)
    c_matrices: Vec<Array2<f32>>,
    /// D matrices (skip connections) per block (n_blocks, block_size)
    d_vectors: Vec<Array1<f32>>,
    /// Discretized A matrices (cached)
    a_bar: Vec<Array2<f32>>,
    /// Discretized B matrices (cached)
    b_bar: Vec<Array2<f32>>,
    /// Layer norm weights
    norm_w: Array1<f32>,
    norm_b: Array1<f32>,
    /// Hidden states per block
    hidden_states: Vec<HiddenState>,
}

impl S5Layer {
    /// Create a new S5 layer with random initialization
    pub fn new(config: S5Config) -> CoreResult<Self> {
        config.validate()?;

        use scirs2_core::random::thread_rng;
        let mut rng = thread_rng();
        let init_scale = (2.0 / config.d_model as f32).sqrt();
        let block_size = config.block_size();

        // Input projection
        let in_proj_w = Array2::from_shape_fn((config.d_model, config.d_input), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * init_scale
        });
        let in_proj_b = if config.use_bias {
            Some(Array1::zeros(config.d_model))
        } else {
            None
        };

        // Output projection
        let out_proj_w = Array2::from_shape_fn((config.d_model, config.d_model), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * init_scale
        });
        let out_proj_b = if config.use_bias {
            Some(Array1::zeros(config.d_model))
        } else {
            None
        };

        // Initialize block-diagonal SSM parameters
        let mut a_matrices = Vec::with_capacity(config.n_blocks);
        let mut b_matrices = Vec::with_capacity(config.n_blocks);
        let mut c_matrices = Vec::with_capacity(config.n_blocks);
        let mut d_vectors = Vec::with_capacity(config.n_blocks);
        let mut hidden_states = Vec::with_capacity(config.n_blocks);

        for _ in 0..config.n_blocks {
            // Initialize A with HiPPO-like parameterization
            let a = Self::init_a_matrix(config.d_state)?;
            a_matrices.push(a);

            // Initialize B, C with random values
            let b = Array2::from_shape_fn((config.d_state, block_size), |_| {
                (rng.random::<f32>() - 0.5) * 2.0 * init_scale
            });
            b_matrices.push(b);

            let c = Array2::from_shape_fn((block_size, config.d_state), |_| {
                (rng.random::<f32>() - 0.5) * 2.0 * init_scale
            });
            c_matrices.push(c);

            // D (skip connection) initialized to small values
            let d = Array1::from_elem(block_size, 0.01);
            d_vectors.push(d);

            // Hidden state for this block
            hidden_states.push(HiddenState::new(1, config.d_state));
        }

        // Discretize all blocks
        let (a_bar, b_bar) = Self::discretize_all(&a_matrices, &b_matrices, config.dt);

        // Layer norm parameters
        let norm_w = Array1::ones(config.d_model);
        let norm_b = Array1::zeros(config.d_model);

        Ok(Self {
            config,
            in_proj_w,
            in_proj_b,
            out_proj_w,
            out_proj_b,
            a_matrices,
            b_matrices,
            c_matrices,
            d_vectors,
            a_bar,
            b_bar,
            norm_w,
            norm_b,
            hidden_states,
        })
    }

    /// Initialize A matrix with stable diagonal parameterization
    fn init_a_matrix(d_state: usize) -> CoreResult<Array2<f32>> {
        use scirs2_core::random::thread_rng;

        // Initialize diagonal A with HiPPO-inspired values
        // Diagonal elements are negative for stability
        let mut rng = thread_rng();
        let mut a = Array2::zeros((d_state, d_state));

        for i in 0..d_state {
            // Diagonal: negative values for stable dynamics (Re(A) < 0 is required)
            // log_lambda ∈ [-5, -1] so exp(log_lambda) ∈ (exp(-5), exp(-1)) > 0
            // Negate to ensure a[i,i] < 0 — the SSM stability condition
            let log_lambda = -rng.random::<f32>() * 4.0 - 1.0; // Range: [-5, -1]
            a[[i, i]] = -log_lambda.exp(); // negative diagonal for stable dynamics

            // Add small off-diagonal elements for richer dynamics
            if i > 0 {
                a[[i, i - 1]] = (rng.random::<f32>() - 0.5) * 0.1;
            }
            if i < d_state - 1 {
                a[[i, i + 1]] = (rng.random::<f32>() - 0.5) * 0.1;
            }
        }

        Ok(a)
    }

    /// Discretize all A and B matrices using zero-order hold
    fn discretize_all(
        a_matrices: &[Array2<f32>],
        b_matrices: &[Array2<f32>],
        dt: f32,
    ) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let mut a_bar_vec = Vec::with_capacity(a_matrices.len());
        let mut b_bar_vec = Vec::with_capacity(b_matrices.len());

        for (a, b) in a_matrices.iter().zip(b_matrices.iter()) {
            let (a_bar, b_bar) = Self::discretize_block(a, b, dt);
            a_bar_vec.push(a_bar);
            b_bar_vec.push(b_bar);
        }

        (a_bar_vec, b_bar_vec)
    }

    /// Discretize a single block using exact Zero-Order Hold (ZOH) via matrix exponential
    fn discretize_block(a: &Array2<f32>, b: &Array2<f32>, dt: f32) -> (Array2<f32>, Array2<f32>) {
        zoh_discretize(a, b, dt)
    }

    /// Apply layer normalization
    fn layer_norm(&self, x: &Array1<f32>) -> Array1<f32> {
        let mean = x.mean().unwrap_or(0.0);
        let variance = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>() / (x.len() as f32);
        let std = (variance + self.config.layer_norm_eps).sqrt();

        let normalized = x.mapv(|xi| (xi - mean) / std);
        &normalized * &self.norm_w + &self.norm_b
    }

    /// SSM step for a single block: h' = A_bar * h + B_bar * u, y = C * h' + D * u
    fn block_step(&mut self, block_idx: usize, u: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let a_bar = &self.a_bar[block_idx];
        let b_bar = &self.b_bar[block_idx];
        let c = &self.c_matrices[block_idx];
        let d = &self.d_vectors[block_idx];

        // Get current hidden state for this block
        let h_state = self.hidden_states[block_idx].state();
        let h = h_state.row(0); // Shape: (1, d_state) -> get first row

        // Update state: h' = A_bar * h + B_bar * u
        let h_new_1d = a_bar.dot(&h) + b_bar.dot(u);
        let mut h_new = Array2::zeros((1, h_new_1d.len()));
        h_new.row_mut(0).assign(&h_new_1d);

        // Output: y = C * h' + D * u
        let y = c.dot(&h_new_1d) + d * u;

        // Update hidden state
        self.hidden_states[block_idx].update(h_new);

        Ok(y)
    }

    /// Forward pass (recurrent mode for O(1) inference)
    pub fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        if x.len() != self.config.d_input {
            return Err(CoreError::DimensionMismatch {
                expected: self.config.d_input,
                got: x.len(),
            });
        }

        // 1. Input projection
        let mut u = self.in_proj_w.dot(x);
        if let Some(ref bias) = self.in_proj_b {
            u = &u + bias;
        }

        // 2. Layer normalization
        let u_norm = self.layer_norm(&u);

        // 3. Process each block in parallel (sequential here, but parallelizable)
        let block_size = self.config.block_size();
        let mut y = Array1::zeros(self.config.d_model);

        for block_idx in 0..self.config.n_blocks {
            let start = block_idx * block_size;
            let end = start + block_size;

            // Extract input for this block
            let u_block = u_norm.slice(s![start..end]).to_owned();

            // Run SSM for this block
            let y_block = self.block_step(block_idx, &u_block)?;

            // Assign output
            y.slice_mut(s![start..end]).assign(&y_block);
        }

        // 4. Output projection
        let mut out = self.out_proj_w.dot(&y);
        if let Some(ref bias) = self.out_proj_b {
            out = &out + bias;
        }

        // 5. Residual connection (if dimensions match)
        if x.len() == out.len() {
            out = &out + x;
        }

        Ok(out)
    }

    /// Batch forward pass (for training)
    pub fn forward_batch(&mut self, x: &Array2<f32>) -> CoreResult<Array2<f32>> {
        let batch_size = x.nrows();
        let mut outputs = Array2::zeros((batch_size, self.config.d_model));

        for (i, input) in x.axis_iter(Axis(0)).enumerate() {
            let output = self.forward(&input.to_owned())?;
            outputs.row_mut(i).assign(&output);
        }

        Ok(outputs)
    }

    /// Reset all hidden states
    pub fn reset(&mut self) {
        for state in &mut self.hidden_states {
            state.reset();
        }
    }

    /// Get configuration
    pub fn config(&self) -> &S5Config {
        &self.config
    }
}

/// Full S5 model with multiple layers
pub struct S5Model {
    layers: Vec<S5Layer>,
    config: S5Config,
}

impl S5Model {
    /// Create a new S5 model with specified number of layers
    pub fn new(config: S5Config, n_layers: usize) -> CoreResult<Self> {
        let mut layers = Vec::with_capacity(n_layers);

        // First layer
        layers.push(S5Layer::new(config.clone())?);

        // Subsequent layers: d_input = d_model from previous layer
        for _ in 1..n_layers {
            let mut layer_config = config.clone();
            layer_config.d_input = config.d_model;
            layers.push(S5Layer::new(layer_config)?);
        }

        Ok(Self { layers, config })
    }

    /// Forward pass through all layers
    pub fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let mut hidden = x.clone();
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden)?;
        }
        Ok(hidden)
    }

    /// Batch forward pass
    pub fn forward_batch(&mut self, x: &Array2<f32>) -> CoreResult<Array2<f32>> {
        let mut hidden = x.clone();
        for layer in &mut self.layers {
            hidden = layer.forward_batch(&hidden)?;
        }
        Ok(hidden)
    }

    /// Reset all layers
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Get number of layers
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get configuration
    pub fn config(&self) -> &S5Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s5_config() {
        let config = S5Config::new(10, 256, 64);
        assert_eq!(config.d_input, 10);
        assert_eq!(config.d_model, 256);
        assert_eq!(config.d_state, 64);
        assert_eq!(config.block_size(), 32); // 256 / 8
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_s5_config_validation() {
        let mut config = S5Config::new(10, 256, 64);
        config.d_model = 0;
        assert!(config.validate().is_err());

        let mut config = S5Config::new(10, 255, 64);
        config.n_blocks = 8;
        assert!(config.validate().is_err()); // Not divisible

        let mut config = S5Config::new(10, 256, 64);
        config.dt = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_s5_layer_creation() {
        let config = S5Config::new(10, 128, 32);
        let result = S5Layer::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_s5_forward() {
        let config = S5Config::new(10, 64, 16);
        let mut layer = S5Layer::new(config).unwrap();
        let input = Array1::from_vec(vec![0.1; 10]);

        let output = layer.forward(&input);
        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.len(), 64);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_s5_reset() {
        let config = S5Config::new(10, 64, 16);
        let mut layer = S5Layer::new(config).unwrap();
        let input = Array1::from_vec(vec![0.1; 10]);

        // Process some inputs
        layer.forward(&input).unwrap();
        layer.forward(&input).unwrap();

        // Reset
        layer.reset();

        // All states should be reset
        for state in &layer.hidden_states {
            let h = state.state();
            assert!(h.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn test_s5_model() {
        let config = S5Config::new(10, 64, 16);
        let mut model = S5Model::new(config, 3).unwrap();
        assert_eq!(model.n_layers(), 3);

        let input = Array1::from_vec(vec![0.1; 10]);
        let output = model.forward(&input);
        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.len(), 64);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_s5_batch() {
        let config = S5Config::new(10, 64, 16);
        let mut layer = S5Layer::new(config).unwrap();

        let batch = Array2::from_shape_fn((4, 10), |(i, j)| 0.1 * (i as f32 + j as f32));

        let output = layer.forward_batch(&batch);
        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.shape(), &[4, 64]);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_s5_no_nan() {
        let config = S5Config::new(10, 64, 16);
        let mut layer = S5Layer::new(config).unwrap();

        // Test with various inputs
        for _ in 0..10 {
            let input = Array1::from_elem(10, 0.5);
            let output = layer.forward(&input).unwrap();
            assert!(output.iter().all(|&x| !x.is_nan()));
        }
    }

    #[test]
    fn test_layer_norm() {
        let config = S5Config::new(10, 64, 16);
        let layer = S5Layer::new(config).unwrap();

        let input = Array1::from_vec((0..64).map(|i| i as f32).collect());
        let normalized = layer.layer_norm(&input);

        // Check mean ≈ 0, std ≈ 1 (within tolerance)
        let mean = normalized.mean().unwrap();
        let std = (normalized.iter().map(|&x| x.powi(2)).sum::<f32>() / 64.0).sqrt();

        assert!((mean.abs()) < 1e-5, "Mean should be close to 0");
        assert!((std - 1.0).abs() < 1e-4, "Std should be close to 1");
    }

    #[test]
    fn test_discretization() {
        let a = Array2::from_shape_fn((4, 4), |(i, j)| if i == j { -0.5 } else { 0.0 });
        let b = Array2::from_shape_fn((4, 2), |_| 0.1);
        let dt = 0.01_f32;

        let (a_bar, b_bar) = S5Layer::discretize_block(&a, &b, dt);

        // A_bar diagonal must equal exp(a[i,i] * dt) = exp(-0.5 * 0.01) ≈ 0.99501
        // zoh_discretize uses 8-term Taylor series which is exact to < 1e-5 here
        let expected_a = (-0.5_f32 * dt).exp();
        for i in 0..4 {
            assert!(
                (a_bar[[i, i]] - expected_a).abs() < 1e-5,
                "a_bar[{i},{i}]={} expected {}",
                a_bar[[i, i]],
                expected_a
            );
        }

        // B_bar under ZOH: b_d = (I + A*dt/2) * B * dt
        // For diagonal A = -0.5, dt = 0.01: scale = (1 - 0.5*0.01/2) = 0.9975
        // So b_bar[i,j] = 0.9975 * 0.1 * 0.01 = 0.0009975
        let expected_b = 0.9975_f32 * 0.1_f32 * dt;
        for i in 0..4 {
            for j in 0..2 {
                assert!(
                    (b_bar[[i, j]] - expected_b).abs() < 1e-5,
                    "b_bar[{i},{j}]={} expected {}",
                    b_bar[[i, j]],
                    expected_b
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Stability correctness tests (Track 2 bug fix)
    // -----------------------------------------------------------------------

    #[test]
    fn test_s5_core_diagonal_negative_after_init() {
        // Every diagonal of every continuous-time A matrix must be negative
        // so that the eigenvalues have negative real parts (stability requirement)
        let config = S5Config::new(4, 8, 4).with_dt(0.01);
        let layer = S5Layer::new(config).expect("S5Layer::new failed");
        for (block_idx, a) in layer.a_matrices.iter().enumerate() {
            for i in 0..a.nrows() {
                assert!(
                    a[[i, i]] < 0.0,
                    "a_matrices[{block_idx}][{i},{i}] = {} is not negative",
                    a[[i, i]]
                );
            }
        }
    }

    #[test]
    fn test_s5_core_a_bar_stable() {
        // After ZOH discretization, diagonal of A_bar must be in (0, 1)
        // This corresponds to exp(dt * a[i,i]) with a[i,i] < 0 and dt > 0
        let config = S5Config::new(4, 8, 4).with_dt(0.01);
        let layer = S5Layer::new(config).expect("S5Layer::new failed");
        for (block_idx, a_bar) in layer.a_bar.iter().enumerate() {
            let n = a_bar.nrows();
            for i in 0..n {
                let val = a_bar[[i, i]];
                assert!(
                    val > 0.0 && val < 1.0,
                    "a_bar[{block_idx}][{i},{i}] = {val} not in (0,1)"
                );
            }
        }
    }

    #[test]
    fn test_s5_core_state_decays() {
        // With zero input, all hidden states must decay (not grow) over 200 steps.
        // We use d_state=1 so each block is a 1×1 diagonal — no off-diagonal
        // coupling can perturb stability, making the test deterministic.
        // d_model=8, n_blocks=8 (default) => block_size=1
        let config = S5Config::new(1, 8, 1).with_dt(0.01);
        let mut layer = S5Layer::new(config).expect("S5Layer::new failed");

        // Set all block states to 1.0
        for state in &mut layer.hidden_states {
            let shape = state.state().shape().to_vec();
            let ones = Array2::ones((shape[0], shape[1]));
            state.update(ones);
        }

        let initial_norm: f32 = layer
            .hidden_states
            .iter()
            .flat_map(|s| {
                let h = s.state();
                h.iter().map(|&x| x * x).collect::<Vec<_>>()
            })
            .sum::<f32>()
            .sqrt();

        // 200 steps of zero input
        let zero_input = Array1::zeros(1);
        for _ in 0..200 {
            layer.forward(&zero_input).expect("forward failed");
        }

        let final_norm: f32 = layer
            .hidden_states
            .iter()
            .flat_map(|s| {
                let h = s.state();
                h.iter().map(|&x| x * x).collect::<Vec<_>>()
            })
            .sum::<f32>()
            .sqrt();

        assert!(
            final_norm <= initial_norm + 1e-3,
            "State grew from {} to {} — A_bar eigenvalues must be < 1",
            initial_norm,
            final_norm
        );
    }

    #[test]
    fn test_s5_core_zoh_matches_diagonal_formula() {
        // For a purely diagonal A with negative entries,
        // zoh_discretize a_bar[i,i] must equal exp(dt * A[i,i]) to high precision
        use crate::numerics::zoh_discretize;
        let dt = 0.01_f32;
        let n = 4_usize;
        let mut a = Array2::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = -((i + 1) as f32); // strictly negative diagonal
        }
        let b = Array2::ones((n, 2));
        let (a_bar, _) = zoh_discretize(&a, &b, dt);
        for i in 0..n {
            let expected = (dt * a[[i, i]]).exp();
            let got = a_bar[[i, i]];
            assert!(
                (got - expected).abs() < 1e-4,
                "a_bar[{i},{i}]={got} expected {expected} (diff {})",
                (got - expected).abs()
            );
        }
    }
}
