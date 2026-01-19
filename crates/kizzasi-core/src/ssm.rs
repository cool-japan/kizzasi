//! State Space Model implementations
//!
//! Implements Mamba-style selective SSM for O(1) inference steps.
//! Uses SIMD-optimized operations for high performance.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::config::KizzasiConfig;
use crate::embedding::ContinuousEmbedding;
use crate::error::CoreResult;
use crate::simd;
use crate::state::HiddenState;
use crate::SignalPredictor;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;
use serde::{Deserialize, Serialize};

/// Trait for state space model implementations
pub trait StateSpaceModel {
    /// Perform a single recurrence step
    fn recurrence_step(
        &self,
        input: &Array1<f32>,
        state: &mut HiddenState,
    ) -> CoreResult<Array1<f32>>;

    /// Get model configuration
    fn config(&self) -> &KizzasiConfig;
}

/// Selective State Space Model (Mamba-style)
///
/// Implements the selective scan mechanism from Mamba for
/// content-aware state transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectiveSSM {
    config: KizzasiConfig,
    embedding: ContinuousEmbedding,
    state: HiddenState,
    // SSM parameters (A, B, C, D matrices per layer)
    a_matrices: Vec<Array2<f32>>,
    b_matrices: Vec<Array2<f32>>,
    c_matrices: Vec<Array2<f32>>,
    d_vectors: Vec<Array1<f32>>,
    // Output projection
    output_proj: Array2<f32>,
}

impl SelectiveSSM {
    /// Create a new SelectiveSSM from configuration
    pub fn new(config: KizzasiConfig) -> CoreResult<Self> {
        let hidden_dim = config.get_hidden_dim();
        let state_dim = config.get_state_dim();
        let num_layers = config.get_num_layers();
        let input_dim = config.get_input_dim();
        let output_dim = config.get_output_dim();

        // Initialize embedding layer
        let embedding = ContinuousEmbedding::new(input_dim, hidden_dim);

        // Initialize hidden state
        let state = HiddenState::new(hidden_dim, state_dim);

        // Initialize SSM matrices for each layer
        let mut rng = thread_rng();
        let scale = 0.01;
        let mut a_matrices = Vec::with_capacity(num_layers);
        let mut b_matrices = Vec::with_capacity(num_layers);
        let mut c_matrices = Vec::with_capacity(num_layers);
        let mut d_vectors = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            // A matrix: state transition (initialized for stability)
            let a = Array2::from_shape_fn((hidden_dim, state_dim), |_| {
                -0.5 + rng.random::<f32>() * scale
            });
            a_matrices.push(a);

            // B matrix: input projection to state
            let b = Array2::from_shape_fn((hidden_dim, state_dim), |_| {
                (rng.random::<f32>() - 0.5) * scale
            });
            b_matrices.push(b);

            // C matrix: state to output projection
            let c = Array2::from_shape_fn((hidden_dim, state_dim), |_| {
                (rng.random::<f32>() - 0.5) * scale
            });
            c_matrices.push(c);

            // D vector: skip connection
            let d = Array1::ones(hidden_dim);
            d_vectors.push(d);
        }

        // Output projection
        let output_proj = Array2::from_shape_fn((hidden_dim, output_dim), |_| {
            (rng.random::<f32>() - 0.5) * scale
        });

        Ok(Self {
            config,
            embedding,
            state,
            a_matrices,
            b_matrices,
            c_matrices,
            d_vectors,
            output_proj,
        })
    }

    /// Get a reference to the hidden state
    pub fn get_state(&self) -> &HiddenState {
        &self.state
    }

    /// Get a mutable reference to the hidden state
    pub fn get_state_mut(&mut self) -> &mut HiddenState {
        &mut self.state
    }

    /// Set the hidden state
    pub fn set_state(&mut self, state: HiddenState) {
        self.state = state;
    }

    /// Get the step count from the hidden state
    pub fn step_count(&self) -> usize {
        self.state.step_count()
    }

    /// Get a reference to the embedding layer
    pub fn embedding(&self) -> &ContinuousEmbedding {
        &self.embedding
    }

    /// Get a reference to the A matrices
    pub fn a_matrices(&self) -> &Vec<Array2<f32>> {
        &self.a_matrices
    }

    /// Get a reference to the B matrices
    pub fn b_matrices(&self) -> &Vec<Array2<f32>> {
        &self.b_matrices
    }

    /// Get a reference to the C matrices
    pub fn c_matrices(&self) -> &Vec<Array2<f32>> {
        &self.c_matrices
    }

    /// Get a reference to the D vectors
    pub fn d_vectors(&self) -> &Vec<Array1<f32>> {
        &self.d_vectors
    }

    /// Get a reference to the output projection matrix
    pub fn output_proj(&self) -> &Array2<f32> {
        &self.output_proj
    }

    /// Discretize continuous SSM parameters (standard precision)
    #[allow(dead_code)]
    fn discretize(
        &self,
        delta: f32,
        a: &Array2<f32>,
        b: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        // Zero-order hold discretization
        // A_bar = exp(delta * A)
        // B_bar = (A^-1) * (A_bar - I) * B ≈ delta * B for small delta
        let a_bar = a.mapv(|x| (delta * x).exp());
        let b_bar = b.mapv(|x| delta * x);
        (a_bar, b_bar)
    }

    /// Selective scan step for a single layer (SIMD-optimized)
    fn selective_scan_step(
        &self,
        layer_idx: usize,
        x: &Array1<f32>,
        h: &mut Array2<f32>,
    ) -> Array1<f32> {
        let a = &self.a_matrices[layer_idx];
        let b = &self.b_matrices[layer_idx];
        let c = &self.c_matrices[layer_idx];
        let d = &self.d_vectors[layer_idx];

        // Compute input-dependent delta (simplified)
        let delta = 0.1; // In full implementation, this is learned

        // Discretize using SIMD-optimized exp
        let (a_bar, b_bar) = self.discretize_simd(delta, a, b);

        // State update: h = A_bar * h + B_bar * x (SIMD-optimized per row)
        for i in 0..h.nrows() {
            let x_val = x[i];
            let row_len = h.ncols();
            let mut h_row = h.row_mut(i);
            let a_row = a_bar.row(i);
            let b_row = b_bar.row(i);

            // Use SIMD FMA for each row element
            for j in 0..row_len {
                h_row[j] = a_row[j].mul_add(h_row[j], b_row[j] * x_val);
            }
        }

        // Output: y = C * h + D * x (SIMD-optimized dot products)
        let mut y = Array1::zeros(x.len());
        for i in 0..y.len() {
            let h_row = h.row(i);
            let c_row = c.row(i);
            y[i] = simd::dot_view(h_row, c_row) + d[i] * x[i];
        }

        y
    }

    /// SIMD-optimized discretization using fast_exp
    fn discretize_simd(
        &self,
        delta: f32,
        a: &Array2<f32>,
        b: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        // Zero-order hold discretization with fast exp approximation
        let a_bar = a.mapv(|x| simd::fast_exp(delta * x));
        let b_bar = b.mapv(|x| delta * x);
        (a_bar, b_bar)
    }
}

impl StateSpaceModel for SelectiveSSM {
    fn recurrence_step(
        &self,
        input: &Array1<f32>,
        state: &mut HiddenState,
    ) -> CoreResult<Array1<f32>> {
        // Embed input
        let mut x = self.embedding.embed(input)?;

        // Apply layer normalization
        x = ContinuousEmbedding::layer_norm(&x, 1e-5);

        // Process through each layer
        let mut h = state.state().clone();
        for layer_idx in 0..self.config.get_num_layers() {
            x = self.selective_scan_step(layer_idx, &x, &mut h);
            x = ContinuousEmbedding::layer_norm(&x, 1e-5);
        }

        // Update state
        state.update(h);

        // Project to output dimension
        let output = x.dot(&self.output_proj);
        Ok(output)
    }

    fn config(&self) -> &KizzasiConfig {
        &self.config
    }
}

impl SignalPredictor for SelectiveSSM {
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let mut state = self.state.clone();
        let output = self.recurrence_step(input, &mut state)?;
        self.state = state;
        Ok(output)
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn context_window(&self) -> usize {
        self.config.get_context_window()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selective_ssm() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let mut ssm = SelectiveSSM::new(config).expect("SSM creation should succeed");
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let output = ssm.step(&input).expect("SSM step should succeed");
        assert_eq!(output.len(), 3);
    }
}
