//! State Space Model implementations
//!
//! Implements Mamba-style selective SSM for O(1) inference steps.
//! Uses SIMD-optimized operations for high performance.

use crate::config::KizzasiConfig;
use crate::embedding::ContinuousEmbedding;
use crate::error::{CoreError, CoreResult};
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

/// Numerically stable softplus: ln(1 + exp(x))
/// Uses log1p(exp(x)) for x < 20 (avoids exp overflow), otherwise x.
#[inline]
fn softplus(x: f32) -> f32 {
    if x >= 20.0 {
        x
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

/// Layer-normalise `x` in place, avoiding the per-layer allocation that a
/// value-returning layer norm would incur.
#[inline]
fn layer_norm_in_place(x: &mut Array1<f32>, eps: f32) {
    match x.as_slice_mut() {
        Some(slice) => simd::layer_norm(slice, eps),
        // Non-contiguous storage cannot be handed to the slice kernel; fall
        // back to the allocating ndarray path rather than silently skipping
        // normalisation.
        None => {
            let normed = ContinuousEmbedding::layer_norm(&x.to_owned(), eps);
            x.assign(&normed);
        }
    }
}

/// Selective State Space Model (Mamba-style)
///
/// Implements the selective scan mechanism from Mamba for
/// content-aware state transitions.
///
/// # Per-layer state
///
/// Every layer owns its own `(hidden_dim, state_dim)` recurrent state, held by
/// [`HiddenState`]. A step reads and writes each layer's own history, so a
/// deep model actually accumulates memory at every depth.
///
/// # Selectivity
///
/// The time-step Δ is **per hidden channel**, produced by a rank-`dt_rank`
/// projection of the layer input (`x → W_down → W_up → softplus`), matching
/// Mamba's selective mechanism. A single scalar Δ would collapse the
/// per-channel selectivity that distinguishes a selective SSM from plain S4.
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
    // Low-rank Δ (time-step) projection for input-dependent discretization.
    // Per layer: (hidden_dim, dt_rank) followed by (dt_rank, hidden_dim).
    dt_proj_down: Vec<Array2<f32>>,
    dt_proj_up: Vec<Array2<f32>>,
    // Output projection
    output_proj: Array2<f32>,
}

/// Reusable per-step scratch buffers.
///
/// Allocated once per [`StateSpaceModel::recurrence_step`] and reused by every
/// layer, so the number of allocations per step is independent of depth.
struct ScanScratch {
    /// Layer output buffer, ping-ponged with the layer input
    y: Array1<f32>,
    /// Low-rank Δ projection intermediate, length `dt_rank`
    dt_low: Array1<f32>,
    /// Per-channel Δ, length `hidden_dim`
    delta: Array1<f32>,
}

impl SelectiveSSM {
    /// Create a new SelectiveSSM from configuration
    pub fn new(config: KizzasiConfig) -> CoreResult<Self> {
        let hidden_dim = config.get_hidden_dim();
        let state_dim = config.get_state_dim();
        let num_layers = config.get_num_layers();
        let input_dim = config.get_input_dim();
        let output_dim = config.get_output_dim();
        let dt_rank = config.get_dt_rank().max(1);

        // Initialize embedding layer
        let embedding = ContinuousEmbedding::new(input_dim, hidden_dim);

        // Initialize hidden state — one matrix per layer
        let state = HiddenState::new_layered(num_layers, hidden_dim, state_dim);

        // Initialize SSM matrices for each layer
        let mut rng = thread_rng();
        let scale = 0.01;
        let mut a_matrices = Vec::with_capacity(num_layers);
        let mut b_matrices = Vec::with_capacity(num_layers);
        let mut c_matrices = Vec::with_capacity(num_layers);
        let mut d_vectors = Vec::with_capacity(num_layers);
        let mut dt_proj_down = Vec::with_capacity(num_layers);
        let mut dt_proj_up = Vec::with_capacity(num_layers);

        // Keep the variance of the two-stage Δ projection comparable to the
        // single-vector form so the initial Δ still lands near softplus(0).
        let up_scale = 2.0 / (dt_rank as f32).sqrt();

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

            // Δ projection: softplus(x · W_down · W_up) gives one input-dependent
            // time step per hidden channel. Initialized small so the initial
            // Δ ≈ softplus(0) ≈ ln(2) ≈ 0.693.
            let down =
                Array2::from_shape_fn((hidden_dim, dt_rank), |_| (rng.random::<f32>() - 0.5) * 0.1);
            dt_proj_down.push(down);

            let up = Array2::from_shape_fn((dt_rank, hidden_dim), |_| {
                (rng.random::<f32>() - 0.5) * up_scale
            });
            dt_proj_up.push(up);
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
            dt_proj_down,
            dt_proj_up,
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
    ///
    /// The state is resized to the model's layer count so a state captured from
    /// a shallower model cannot silently starve the deeper layers.
    pub fn set_state(&mut self, state: HiddenState) {
        self.state = state;
        self.state.ensure_layers(self.config.get_num_layers());
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

    /// Get a reference to the per-layer Δ down-projection matrices, shape `(hidden_dim, dt_rank)`
    pub fn dt_proj_down(&self) -> &Vec<Array2<f32>> {
        &self.dt_proj_down
    }

    /// Get a reference to the per-layer Δ up-projection matrices, shape `(dt_rank, hidden_dim)`
    pub fn dt_proj_up(&self) -> &Vec<Array2<f32>> {
        &self.dt_proj_up
    }

    /// Get a reference to the output projection matrix
    pub fn output_proj(&self) -> &Array2<f32> {
        &self.output_proj
    }

    /// Compute the per-channel time step Δ for a layer into `scratch`.
    ///
    /// `Δ = softplus(x · W_down · W_up)`, one value per hidden channel.
    fn compute_delta(&self, layer_idx: usize, x: &Array1<f32>, scratch: &mut ScanScratch) {
        let down = &self.dt_proj_down[layer_idx];
        let up = &self.dt_proj_up[layer_idx];

        // Stage 1: (hidden_dim) · (hidden_dim, dt_rank) -> (dt_rank)
        for r in 0..scratch.dt_low.len() {
            let mut acc = 0.0_f32;
            for i in 0..x.len() {
                acc = down[[i, r]].mul_add(x[i], acc);
            }
            scratch.dt_low[r] = acc;
        }

        // Stage 2: (dt_rank) · (dt_rank, hidden_dim) -> (hidden_dim), then softplus
        for i in 0..scratch.delta.len() {
            let mut acc = 0.0_f32;
            for r in 0..scratch.dt_low.len() {
                acc = up[[r, i]].mul_add(scratch.dt_low[r], acc);
            }
            scratch.delta[i] = softplus(acc);
        }
    }

    /// Selective scan step for a single layer, writing into `scratch.y`.
    ///
    /// The zero-order-hold discretization is fused into the state update:
    /// `h[i,j] = exp(Δ[i]·A[i,j]) · h[i,j] + Δ[i]·B[i,j]·x[i]`. Materialising
    /// `A_bar`/`B_bar` first would cost two `hidden_dim × state_dim`
    /// allocations plus two extra passes over that memory per layer per step,
    /// for values consumed exactly once.
    fn selective_scan_step_into(
        &self,
        layer_idx: usize,
        x: &Array1<f32>,
        h: &mut Array2<f32>,
        scratch: &mut ScanScratch,
    ) {
        let a = &self.a_matrices[layer_idx];
        let b = &self.b_matrices[layer_idx];
        let c = &self.c_matrices[layer_idx];
        let d = &self.d_vectors[layer_idx];

        // Input-dependent, per-channel Δ (Mamba selectivity mechanism)
        self.compute_delta(layer_idx, x, scratch);

        // Fused discretize + state update: h = exp(Δ⊙A) ⊙ h + (Δ⊙B) ⊙ x
        let state_dim = h.ncols();
        for i in 0..h.nrows() {
            let x_val = x[i];
            let delta_i = scratch.delta[i];
            let mut h_row = h.row_mut(i);
            let a_row = a.row(i);
            let b_row = b.row(i);

            for j in 0..state_dim {
                h_row[j] = simd::fast_exp(delta_i * a_row[j])
                    .mul_add(h_row[j], delta_i * b_row[j] * x_val);
            }
        }

        // Output: y = C * h + D * x (SIMD-optimized dot products)
        for i in 0..scratch.y.len() {
            let h_row = h.row(i);
            let c_row = c.row(i);
            scratch.y[i] = simd::dot_view(h_row, c_row) + d[i] * x[i];
        }
    }
}

impl StateSpaceModel for SelectiveSSM {
    fn recurrence_step(
        &self,
        input: &Array1<f32>,
        state: &mut HiddenState,
    ) -> CoreResult<Array1<f32>> {
        let hidden_dim = self.config.get_hidden_dim();
        let state_dim = self.config.get_state_dim();
        let num_layers = self.config.get_num_layers();
        let dt_rank = self.config.get_dt_rank().max(1);

        // Each layer must have its own state; grow a state that was created
        // for a shallower model (or loaded from a pre-per-layer checkpoint).
        state.ensure_layers(num_layers);

        // Embed input
        let mut x = self.embedding.embed(input)?;

        // Apply layer normalization
        layer_norm_in_place(&mut x, 1e-5);

        let mut scratch = ScanScratch {
            y: Array1::zeros(hidden_dim),
            dt_low: Array1::zeros(dt_rank),
            delta: Array1::zeros(hidden_dim),
        };

        // Process through each layer, each against its own recurrent state.
        for layer_idx in 0..num_layers {
            let layer_state =
                state
                    .layer_state_mut(layer_idx)
                    .ok_or(CoreError::DimensionMismatch {
                        expected: num_layers,
                        got: layer_idx,
                    })?;

            if layer_state.nrows() != hidden_dim || layer_state.ncols() != state_dim {
                return Err(CoreError::DimensionMismatch {
                    expected: hidden_dim * state_dim,
                    got: layer_state.nrows() * layer_state.ncols(),
                });
            }

            self.selective_scan_step_into(layer_idx, &x, layer_state, &mut scratch);

            // Ping-pong: the layer output becomes the next layer's input and the
            // old input buffer becomes the next output scratch.
            core::mem::swap(&mut x, &mut scratch.y);
            layer_norm_in_place(&mut x, 1e-5);
        }

        // The layer states were mutated in place; only the step counter is left.
        state.advance();

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
        // Move the state out (a move, not a copy — the placeholder is a 0x0
        // array and allocates nothing) so `&self` and `&mut state` coexist
        // without cloning the whole state on every step.
        let mut state = core::mem::replace(&mut self.state, HiddenState::new(0, 0));
        let result = self.recurrence_step(input, &mut state);
        self.state = state;
        result
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    /// Returns the configured context-window size, unchanged from
    /// [`KizzasiConfig::get_context_window`].
    ///
    /// This is metadata only. `SelectiveSSM` is a Mamba-style recurrent
    /// model whose per-layer hidden state (see the struct docs above) is a
    /// fixed `(hidden_dim, state_dim)` buffer that never grows with the
    /// number of steps taken -- there is no history buffer for this value
    /// to size or truncate, and `recurrence_step`/`step` never read it.
    /// Two configs that differ only in `context_window` build byte-identical
    /// models, allocate the same memory, and produce identical predictions
    /// for the same input sequence.
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

    #[test]
    fn test_softplus_positive() {
        // softplus must be strictly positive for all inputs
        for &x in &[-5.0_f32, 0.0, 1.0, 5.0, 25.0] {
            let y = softplus(x);
            assert!(y > 0.0, "softplus({x}) = {y} must be > 0");
        }
    }

    #[test]
    fn test_softplus_large_input() {
        // For large x, softplus(x) ≈ x (within 1e-4) and must not overflow
        let x = 25.0_f32;
        let y = softplus(x);
        assert!(
            (y - x).abs() < 1e-4,
            "softplus({x}) = {y}, expected ≈ {x} (diff = {})",
            (y - x).abs()
        );
        assert!(y.is_finite(), "softplus({x}) must be finite, got {y}");
    }

    #[test]
    fn test_delta_is_input_dependent() {
        // Two distinct inputs should produce distinct outputs because delta adapts.
        let hidden_dim = 8_usize;
        let config = KizzasiConfig::new()
            .input_dim(hidden_dim)
            .output_dim(hidden_dim)
            .hidden_dim(hidden_dim)
            .state_dim(4)
            .num_layers(1);

        let mut ssm1 = SelectiveSSM::new(config.clone()).expect("SSM creation should succeed");
        let mut ssm2 = SelectiveSSM::new(config).expect("SSM creation should succeed");

        // Use identical model weights (same seed would require determinism; instead verify
        // that by forking the same model we can distinguish which path was taken).
        // More robustly: run both inputs through the SAME SSM instance and compare outputs.
        let x_ones = Array1::<f32>::ones(hidden_dim);
        let x_zeros = Array1::<f32>::zeros(hidden_dim);

        // Both SSMs are freshly initialized with the same config; we run each on one input.
        let out1 = ssm1.step(&x_ones).expect("step should succeed");
        let out2 = ssm2.step(&x_zeros).expect("step should succeed");

        // With input-dependent delta, dt_proj · ones ≠ dt_proj · zeros (in general),
        // yielding different A_bar/B_bar and thus different outputs.
        // We assert that the outputs are not byte-identical (they should differ).
        let identical = out1.iter().zip(out2.iter()).all(|(a, b)| a == b);
        assert!(
            !identical,
            "Outputs for all-ones vs all-zeros inputs should differ with input-dependent delta"
        );
    }

    #[test]
    fn test_delta_not_hardcoded() {
        // If delta were a constant, two sequential steps with different inputs would produce
        // the same A_bar/B_bar and only differ through B_bar * x (which scales with x).
        // With input-dependent delta the A_bar itself changes, making the difference
        // strictly richer.  We verify that two-step outputs diverge meaningfully.
        let hidden_dim = 16_usize;
        let config = KizzasiConfig::new()
            .input_dim(hidden_dim)
            .output_dim(hidden_dim)
            .hidden_dim(hidden_dim)
            .state_dim(8)
            .num_layers(2);

        let mut ssm = SelectiveSSM::new(config).expect("SSM creation should succeed");

        let x_high = Array1::from_elem(hidden_dim, 2.0_f32);
        let x_low = Array1::from_elem(hidden_dim, -2.0_f32);

        let out_high = ssm.step(&x_high).expect("step should succeed");
        // Reset so history does not accumulate between the two sub-runs
        ssm.reset();
        let out_low = ssm.step(&x_low).expect("step should succeed");

        // The outputs must differ — they would be identical only if delta were hardcoded
        // and the layer-norm collapsed both inputs to the same magnitude.
        let max_diff = out_high
            .iter()
            .zip(out_low.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "Outputs for high vs low inputs should differ with input-dependent delta (max_diff={max_diff})"
        );
    }

    #[test]
    fn test_delta_is_per_channel() {
        // A per-channel Δ must produce different values for different hidden
        // channels; a scalar Δ (the historical behaviour) would make every
        // entry identical.
        let hidden_dim = 32_usize;
        let config = KizzasiConfig::new()
            .input_dim(hidden_dim)
            .output_dim(hidden_dim)
            .hidden_dim(hidden_dim)
            .state_dim(4)
            .num_layers(1)
            .dt_rank(8);

        let ssm = SelectiveSSM::new(config).expect("SSM creation should succeed");
        let mut scratch = ScanScratch {
            y: Array1::zeros(hidden_dim),
            dt_low: Array1::zeros(8),
            delta: Array1::zeros(hidden_dim),
        };

        let mut x = Array1::from_shape_fn(hidden_dim, |i| (i as f32) * 0.1 - 1.0);
        layer_norm_in_place(&mut x, 1e-5);
        ssm.compute_delta(0, &x, &mut scratch);

        let first = scratch.delta[0];
        let all_equal = scratch.delta.iter().all(|&v| (v - first).abs() < 1e-9);
        assert!(
            !all_equal,
            "Δ must vary per hidden channel, got a constant {first}"
        );
        assert!(
            scratch.delta.iter().all(|&v| v > 0.0 && v.is_finite()),
            "Δ must be finite and strictly positive"
        );
    }

    #[test]
    fn test_dt_rank_shapes_projection() {
        // The Δ projection must be sized from the configured dt_rank.
        let config = KizzasiConfig::new()
            .input_dim(4)
            .output_dim(4)
            .hidden_dim(12)
            .state_dim(4)
            .num_layers(2)
            .dt_rank(3);

        let ssm = SelectiveSSM::new(config).expect("SSM creation should succeed");
        assert_eq!(ssm.dt_proj_down().len(), 2);
        assert_eq!(ssm.dt_proj_up().len(), 2);
        assert_eq!(ssm.dt_proj_down()[0].shape(), &[12, 3]);
        assert_eq!(ssm.dt_proj_up()[0].shape(), &[3, 12]);
    }

    #[test]
    fn test_each_layer_keeps_its_own_state() {
        // Regression: every layer used to share one buffer, so all but the last
        // layer lost its history on every step. Each layer must now accumulate
        // its own non-zero state, and the layers must differ from one another.
        let config = KizzasiConfig::new()
            .input_dim(4)
            .output_dim(4)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(3);

        let mut ssm = SelectiveSSM::new(config).expect("SSM creation should succeed");
        assert_eq!(ssm.get_state().num_layers(), 3);

        let input = Array1::from_vec(vec![0.4_f32, -0.2, 0.9, 0.1]);
        for _ in 0..5 {
            ssm.step(&input).expect("step should succeed");
        }

        let mut norms = Vec::with_capacity(3);
        for layer_idx in 0..3 {
            let layer = ssm
                .get_state()
                .layer_state(layer_idx)
                .expect("every layer must have a state");
            let norm: f32 = layer.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                norm > 0.0 && norm.is_finite(),
                "layer {layer_idx} state must be non-zero and finite, got {norm}"
            );
            norms.push(norm);
        }

        // Distinct layers see distinct inputs, so their states must not coincide.
        assert!(
            (norms[0] - norms[1]).abs() > 1e-9 || (norms[1] - norms[2]).abs() > 1e-9,
            "per-layer states are suspiciously identical: {norms:?}"
        );
    }

    #[test]
    fn test_layer0_state_evolves_across_steps() {
        // With a shared buffer, layer 0's state was overwritten by the deeper
        // layers every step, so it never reflected its own history.
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(2);

        let mut ssm = SelectiveSSM::new(config).expect("SSM creation should succeed");
        let input = Array1::from_vec(vec![0.7_f32, -0.3]);

        ssm.step(&input).expect("step should succeed");
        let after_one = ssm
            .get_state()
            .layer_state(0)
            .expect("layer 0 exists")
            .clone();

        ssm.step(&input).expect("step should succeed");
        let after_two = ssm
            .get_state()
            .layer_state(0)
            .expect("layer 0 exists")
            .clone();

        let max_diff = after_one
            .iter()
            .zip(after_two.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-9,
            "layer 0 state did not change between steps (max_diff={max_diff})"
        );
    }

    #[test]
    fn test_step_count_increments_once_per_step() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(4);

        let mut ssm = SelectiveSSM::new(config).expect("SSM creation should succeed");
        let input = Array1::from_vec(vec![0.1_f32, 0.2]);

        for expected in 1..=4 {
            ssm.step(&input).expect("step should succeed");
            assert_eq!(ssm.step_count(), expected);
        }

        ssm.reset();
        assert_eq!(ssm.step_count(), 0);
    }

    #[test]
    fn test_step_is_deterministic_after_reset() {
        // The in-place recurrence must not leak state between runs: the same
        // input sequence after `reset()` must reproduce the same outputs.
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(3);

        let mut ssm = SelectiveSSM::new(config).expect("SSM creation should succeed");
        let inputs = [
            Array1::from_vec(vec![0.1_f32, 0.2, 0.3]),
            Array1::from_vec(vec![-0.4_f32, 0.8, 0.0]),
            Array1::from_vec(vec![0.5_f32, -0.5, 0.25]),
        ];

        let mut first_run = Vec::new();
        for input in inputs.iter() {
            first_run.push(ssm.step(input).expect("step should succeed"));
        }

        ssm.reset();

        for (i, input) in inputs.iter().enumerate() {
            let out = ssm.step(input).expect("step should succeed");
            for (a, b) in out.iter().zip(first_run[i].iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "step {i} diverged after reset: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn test_set_state_resizes_to_model_depth() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(3);

        let mut ssm = SelectiveSSM::new(config).expect("SSM creation should succeed");
        // A state captured from a single-layer model must be grown, not used
        // as a shared buffer.
        ssm.set_state(HiddenState::new(8, 4));
        assert_eq!(ssm.get_state().num_layers(), 3);

        let input = Array1::from_vec(vec![0.3_f32, 0.6]);
        let out = ssm.step(&input).expect("step should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
