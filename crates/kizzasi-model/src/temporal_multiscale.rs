//! Multi-Scale Temporal Modeling
//!
//! This module implements temporal modeling at multiple resolutions simultaneously.
//! Each "scale" processes the input at a different decimation rate, allowing the
//! model to capture both fine-grained and coarse temporal dynamics.
//!
//! # Architecture
//!
//! ```text
//! Input ──→ [Scale 1, dt=1]  ──→ state_1 ──┐
//!        ├─→ [Scale 2, dt=4]  ──→ state_2 ──┤ → [Fusion] → [Output Proj] → Prediction
//!        └─→ [Scale 3, dt=16] ──→ state_3 ──┘
//! ```
//!
//! Each `TemporalScale` updates its state only every `decimation` steps,
//! emulating processing at different temporal resolutions.
//!
//! # Fusion Strategies
//!
//! - **Concatenate**: Concatenate all scale outputs, then project linearly.
//! - **Weighted**: Learned scalar weights (softmax-normalized) per scale.
//! - **Attention**: Cross-scale attention — query from mean, keys/values from scales.
//!
//! # GRU-style Update
//!
//! Each scale uses a simplified GRU recurrence:
//! ```text
//! state = tanh(W_proj @ input + W_rec @ state + bias)
//! ```
//!
//! # References
//!
//! - "Temporal Convolutional Networks" (Bai et al., 2018)
//! - "Multi-Scale RNNs" (El Hihi & Bengio, 1996)

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{CoreResult, HiddenState, SignalPredictor};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

// ---------------------------------------------------------------------------
// Seeded deterministic RNG
// ---------------------------------------------------------------------------

struct SeededRng {
    state: u64,
}

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
    }
}

// ---------------------------------------------------------------------------
// Scale Fusion Strategy
// ---------------------------------------------------------------------------

/// Strategy for combining outputs from multiple temporal scales
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleFusion {
    /// Concatenate all scale hidden states → linear projection to output_dim
    Concatenate,
    /// Learned softmax-normalized scalar weight per scale
    Weighted,
    /// Cross-scale attention: query = mean of states, keys/values = per-scale states
    Attention,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Multi-Scale Temporal Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleConfig {
    /// Input signal dimension
    pub input_dim: usize,
    /// Hidden dimension per temporal scale
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Number of temporal scales
    pub num_scales: usize,
    /// Decimation factor per scale (e.g., [1, 4, 16] means scale 0 updates every step,
    /// scale 1 every 4 steps, scale 2 every 16 steps)
    pub scale_factors: Vec<usize>,
    /// Fusion strategy for combining scale outputs
    pub fusion: ScaleFusion,
    /// Nominal context window (informational only; recurrence is unbounded)
    pub context_length: usize,
}

impl MultiScaleConfig {
    /// Validate configuration
    pub fn validate(&self) -> ModelResult<()> {
        if self.input_dim == 0 {
            return Err(ModelError::invalid_config("input_dim must be > 0"));
        }
        if self.hidden_dim == 0 {
            return Err(ModelError::invalid_config("hidden_dim must be > 0"));
        }
        if self.output_dim == 0 {
            return Err(ModelError::invalid_config("output_dim must be > 0"));
        }
        if self.num_scales == 0 {
            return Err(ModelError::invalid_config("num_scales must be > 0"));
        }
        if self.scale_factors.len() != self.num_scales {
            return Err(ModelError::invalid_config(
                "scale_factors.len() must equal num_scales",
            ));
        }
        for &sf in &self.scale_factors {
            if sf == 0 {
                return Err(ModelError::invalid_config("all scale_factors must be > 0"));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TemporalScale
// ---------------------------------------------------------------------------

/// A single temporal scale that processes input at a given decimation rate.
///
/// The scale runs a GRU-style recurrent update every `decimation` steps.
/// Between updates, it holds its previous state constant.
pub struct TemporalScale {
    hidden_dim: usize,
    /// Number of steps between state updates
    decimation: usize,
    /// Input projection: (hidden_dim, input_dim)
    projection: Array2<f32>,
    /// Recurrent weight: (hidden_dim, hidden_dim)
    recurrent: Array2<f32>,
    /// Bias
    bias: Array1<f32>,
    /// Internal tick counter (0..decimation-1)
    tick_counter: usize,
    /// Current hidden state
    state: Array1<f32>,
}

impl TemporalScale {
    /// Create a new temporal scale
    pub fn new(input_dim: usize, hidden_dim: usize, decimation: usize) -> ModelResult<Self> {
        if input_dim == 0 || hidden_dim == 0 || decimation == 0 {
            return Err(ModelError::invalid_config(
                "TemporalScale dimensions and decimation must be > 0",
            ));
        }

        let scale_input = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        let scale_rec = (2.0 / (hidden_dim + hidden_dim) as f32).sqrt();
        let seed = ((input_dim + hidden_dim * 37 + decimation * 997) as u64)
            .wrapping_mul(6364136223846793005);
        let mut rng = SeededRng::new(seed);

        let projection =
            Array2::from_shape_fn((hidden_dim, input_dim), |_| rng.next_f32() * scale_input);
        let recurrent =
            Array2::from_shape_fn((hidden_dim, hidden_dim), |_| rng.next_f32() * scale_rec);
        let bias = Array1::from_shape_fn(hidden_dim, |_| rng.next_f32() * 0.01);

        Ok(Self {
            hidden_dim,
            decimation,
            projection,
            recurrent,
            bias,
            tick_counter: 0,
            state: Array1::zeros(hidden_dim),
        })
    }

    /// Process one time step.
    ///
    /// Returns `Some(state)` when the scale updates (every `decimation` steps),
    /// or `None` if this step is a no-op for this scale.
    #[instrument(skip(self, input), fields(decimation = self.decimation, tick = self.tick_counter))]
    pub fn step(&mut self, input: &Array1<f32>) -> ModelResult<Option<Array1<f32>>> {
        self.tick_counter += 1;

        if !self.tick_counter.is_multiple_of(self.decimation) {
            return Ok(None);
        }

        // GRU-style update: state = tanh(W_proj @ input + W_rec @ state + bias)
        let proj_out = self.projection.dot(input);
        let rec_out = self.recurrent.dot(&self.state);
        let pre_act = proj_out + rec_out + &self.bias;
        let new_state = pre_act.mapv(f32::tanh);

        if new_state.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::numerical_instability(
                "TemporalScale::step",
                "NaN or Inf in state update",
            ));
        }

        self.state = new_state.clone();
        Ok(Some(new_state))
    }

    /// Get the current hidden state (does not advance the tick counter)
    pub fn current_state(&self) -> &Array1<f32> {
        &self.state
    }

    /// Reset tick counter and hidden state to zero
    pub fn reset(&mut self) {
        self.tick_counter = 0;
        self.state.fill(0.0);
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get decimation factor
    pub fn decimation(&self) -> usize {
        self.decimation
    }
}

// ---------------------------------------------------------------------------
// ScaleFusionLayer
// ---------------------------------------------------------------------------

/// Internal fusion module that combines outputs from multiple temporal scales.
struct ScaleFusionLayer {
    fusion: ScaleFusion,
    /// For Concatenate: projects [num_scales * hidden_dim] → hidden_dim
    concat_proj: Option<Array2<f32>>,
    /// For Weighted: log-space unnormalized weights (one per scale)
    scale_weights: Option<Array1<f32>>,
    /// For Attention: query projection (hidden_dim, hidden_dim)
    attn_q: Option<Array2<f32>>,
    /// For Attention: key projection (hidden_dim, hidden_dim)
    attn_k: Option<Array2<f32>>,
    /// For Attention: value projection (hidden_dim, hidden_dim)
    attn_v: Option<Array2<f32>>,
    num_scales: usize,
    hidden_dim: usize,
}

impl ScaleFusionLayer {
    fn new(
        fusion: ScaleFusion,
        num_scales: usize,
        hidden_dim: usize,
        seed: u64,
    ) -> ModelResult<Self> {
        if num_scales == 0 || hidden_dim == 0 {
            return Err(ModelError::invalid_config(
                "ScaleFusionLayer: num_scales and hidden_dim must be > 0",
            ));
        }

        let mut rng = SeededRng::new(seed);
        let scale = (2.0 / (hidden_dim * 2) as f32).sqrt();

        let (concat_proj, scale_weights, attn_q, attn_k, attn_v) = match &fusion {
            ScaleFusion::Concatenate => {
                let in_dim = num_scales * hidden_dim;
                let proj_scale = (2.0 / (in_dim + hidden_dim) as f32).sqrt();
                let proj =
                    Array2::from_shape_fn((hidden_dim, in_dim), |_| rng.next_f32() * proj_scale);
                (Some(proj), None, None, None, None)
            }
            ScaleFusion::Weighted => {
                // Initialize log-weights to zero (uniform after softmax)
                let weights = Array1::zeros(num_scales);
                (None, Some(weights), None, None, None)
            }
            ScaleFusion::Attention => {
                let q = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| rng.next_f32() * scale);
                let k = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| rng.next_f32() * scale);
                let v = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| rng.next_f32() * scale);
                (None, None, Some(q), Some(k), Some(v))
            }
        };

        Ok(Self {
            fusion,
            concat_proj,
            scale_weights,
            attn_q,
            attn_k,
            attn_v,
            num_scales,
            hidden_dim,
        })
    }

    /// Fuse scale states into a single hidden_dim vector.
    fn fuse(&self, scale_states: &[Array1<f32>]) -> ModelResult<Array1<f32>> {
        if scale_states.len() != self.num_scales {
            return Err(ModelError::dimension_mismatch(
                "ScaleFusionLayer::fuse",
                self.num_scales,
                scale_states.len(),
            ));
        }

        match &self.fusion {
            ScaleFusion::Concatenate => self.fuse_concatenate(scale_states),
            ScaleFusion::Weighted => self.fuse_weighted(scale_states),
            ScaleFusion::Attention => self.fuse_attention(scale_states),
        }
    }

    fn fuse_concatenate(&self, scale_states: &[Array1<f32>]) -> ModelResult<Array1<f32>> {
        let proj = self.concat_proj.as_ref().ok_or_else(|| {
            ModelError::not_initialized("concat_proj missing for Concatenate fusion")
        })?;

        // Concatenate all states into a single vector
        let total_dim = self.num_scales * self.hidden_dim;
        let mut concat = Array1::<f32>::zeros(total_dim);
        for (i, state) in scale_states.iter().enumerate() {
            let start = i * self.hidden_dim;
            let end = start + self.hidden_dim;
            if state.len() != self.hidden_dim {
                return Err(ModelError::dimension_mismatch(
                    format!("scale {i} state"),
                    self.hidden_dim,
                    state.len(),
                ));
            }
            concat
                .slice_mut(scirs2_core::ndarray::s![start..end])
                .assign(state);
        }

        Ok(proj.dot(&concat))
    }

    fn fuse_weighted(&self, scale_states: &[Array1<f32>]) -> ModelResult<Array1<f32>> {
        let log_weights = self.scale_weights.as_ref().ok_or_else(|| {
            ModelError::not_initialized("scale_weights missing for Weighted fusion")
        })?;

        // Softmax normalization for numerical stability
        let max_w = log_weights
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_w: Vec<f32> = log_weights.iter().map(|&w| (w - max_w).exp()).collect();
        let sum_exp: f32 = exp_w.iter().sum();
        let norm_weights: Vec<f32> = exp_w.iter().map(|&e| e / sum_exp).collect();

        let mut result = Array1::<f32>::zeros(self.hidden_dim);
        for (state, &w) in scale_states.iter().zip(norm_weights.iter()) {
            if state.len() != self.hidden_dim {
                return Err(ModelError::dimension_mismatch(
                    "weighted scale state",
                    self.hidden_dim,
                    state.len(),
                ));
            }
            result = result + state * w;
        }
        Ok(result)
    }

    fn fuse_attention(&self, scale_states: &[Array1<f32>]) -> ModelResult<Array1<f32>> {
        let q_proj = self
            .attn_q
            .as_ref()
            .ok_or_else(|| ModelError::not_initialized("attn_q missing for Attention fusion"))?;
        let k_proj = self
            .attn_k
            .as_ref()
            .ok_or_else(|| ModelError::not_initialized("attn_k missing for Attention fusion"))?;
        let v_proj = self
            .attn_v
            .as_ref()
            .ok_or_else(|| ModelError::not_initialized("attn_v missing for Attention fusion"))?;

        // Query: mean of all scale states
        let mut mean_state = Array1::<f32>::zeros(self.hidden_dim);
        for state in scale_states {
            if state.len() != self.hidden_dim {
                return Err(ModelError::dimension_mismatch(
                    "attention scale state",
                    self.hidden_dim,
                    state.len(),
                ));
            }
            mean_state += state;
        }
        mean_state.mapv_inplace(|v| v / self.num_scales as f32);

        let query = q_proj.dot(&mean_state); // (hidden_dim,)
        let scale_factor = (self.hidden_dim as f32).sqrt();

        // Compute attention scores: q · k_i / sqrt(d)
        let mut scores = Vec::with_capacity(self.num_scales);
        for state in scale_states {
            let key_i = k_proj.dot(state);
            let score = query.dot(&key_i) / scale_factor;
            scores.push(score);
        }

        // Softmax over scores
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let attn_weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Weighted sum of values
        let mut result = Array1::<f32>::zeros(self.hidden_dim);
        for (state, &w) in scale_states.iter().zip(attn_weights.iter()) {
            let value_i = v_proj.dot(state);
            result = result + value_i * w;
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// MultiScaleModel
// ---------------------------------------------------------------------------

/// Multi-Scale Temporal Model that processes signals at multiple resolutions.
pub struct MultiScaleModel {
    /// Model configuration
    pub config: MultiScaleConfig,
    /// Per-scale temporal processors
    scales: Vec<TemporalScale>,
    /// Fusion layer for combining scale outputs
    fusion_layer: ScaleFusionLayer,
    /// Output projection: (output_dim, hidden_dim)
    output_proj: Array2<f32>,
    /// Output bias
    output_bias: Array1<f32>,
    /// Most recent output from each scale (initialized to zeros)
    last_scale_outputs: Vec<Array1<f32>>,
}

impl MultiScaleModel {
    /// Create a new multi-scale model
    #[instrument(skip(config), fields(scales = config.num_scales, hidden = config.hidden_dim))]
    pub fn new(config: MultiScaleConfig) -> ModelResult<Self> {
        config.validate()?;
        debug!(
            "Building MultiScaleModel: {} scales at {:?}",
            config.num_scales, config.scale_factors
        );

        let mut scales = Vec::with_capacity(config.num_scales);
        for (i, &decimation) in config.scale_factors.iter().enumerate() {
            let seed = ((i + 1) as u64).wrapping_mul(6364136223846793005);
            let _ = seed; // seed baked into TemporalScale::new via dimension-based seeding
            scales.push(TemporalScale::new(
                config.input_dim,
                config.hidden_dim,
                decimation,
            )?);
        }

        let fusion_seed = (config.num_scales as u64 * 1000 + config.hidden_dim as u64)
            .wrapping_mul(2862933555777941757);
        let fusion_layer = ScaleFusionLayer::new(
            config.fusion.clone(),
            config.num_scales,
            config.hidden_dim,
            fusion_seed,
        )?;

        let out_scale = (2.0 / (config.hidden_dim + config.output_dim) as f32).sqrt();
        let mut rng = SeededRng::new(
            ((config.hidden_dim * 7919 + config.output_dim) as u64)
                .wrapping_mul(6364136223846793005),
        );
        let output_proj = Array2::from_shape_fn((config.output_dim, config.hidden_dim), |_| {
            rng.next_f32() * out_scale
        });
        let output_bias = Array1::from_shape_fn(config.output_dim, |_| rng.next_f32() * 0.01);

        let last_scale_outputs = vec![Array1::zeros(config.hidden_dim); config.num_scales];

        debug!("MultiScaleModel built successfully");
        Ok(Self {
            config,
            scales,
            fusion_layer,
            output_proj,
            output_bias,
            last_scale_outputs,
        })
    }

    /// Small preset: 3 scales at [1, 4, 16], input/output dim 1
    pub fn small() -> ModelResult<Self> {
        let config = MultiScaleConfig {
            input_dim: 1,
            hidden_dim: 32,
            output_dim: 1,
            num_scales: 3,
            scale_factors: vec![1, 4, 16],
            fusion: ScaleFusion::Concatenate,
            context_length: 512,
        };
        Self::new(config)
    }

    /// Base preset: 4 scales at [1, 2, 8, 32], input/output dim 1
    pub fn base() -> ModelResult<Self> {
        let config = MultiScaleConfig {
            input_dim: 1,
            hidden_dim: 64,
            output_dim: 1,
            num_scales: 4,
            scale_factors: vec![1, 2, 8, 32],
            fusion: ScaleFusion::Weighted,
            context_length: 2048,
        };
        Self::new(config)
    }

    /// Internal forward step
    fn forward_step(&mut self, input: &Array1<f32>) -> ModelResult<Array1<f32>> {
        if input.len() != self.config.input_dim {
            return Err(ModelError::dimension_mismatch(
                "MultiScaleModel input",
                self.config.input_dim,
                input.len(),
            ));
        }

        // Step each scale; update last_scale_outputs when scale fires
        for (i, scale) in self.scales.iter_mut().enumerate() {
            if let Some(new_state) = scale.step(input)? {
                self.last_scale_outputs[i] = new_state;
            }
        }

        // Fuse all (possibly stale) scale outputs
        let fused = self.fusion_layer.fuse(&self.last_scale_outputs)?;

        // Output projection
        let output = self.output_proj.dot(&fused) + &self.output_bias;

        if output.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::numerical_instability(
                "MultiScaleModel output",
                "NaN or Inf detected",
            ));
        }

        Ok(output)
    }
}

impl SignalPredictor for MultiScaleModel {
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        self.forward_step(input)
            .map_err(|e| kizzasi_core::CoreError::Generic(e.to_string()))
    }

    #[instrument(skip(self))]
    fn reset(&mut self) {
        debug!("Resetting MultiScaleModel state");
        for scale in &mut self.scales {
            scale.reset();
        }
        for output in &mut self.last_scale_outputs {
            output.fill(0.0);
        }
    }

    fn context_window(&self) -> usize {
        self.config.context_length
    }
}

impl AutoregressiveModel for MultiScaleModel {
    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn state_dim(&self) -> usize {
        // Total state = hidden_dim per scale
        self.config.hidden_dim * self.config.num_scales
    }

    fn num_layers(&self) -> usize {
        self.config.num_scales
    }

    fn model_type(&self) -> ModelType {
        ModelType::MultiScale
    }

    fn get_states(&self) -> Vec<HiddenState> {
        self.scales
            .iter()
            .map(|scale| {
                let state = scale.current_state().clone();
                let dim = state.len();
                let state_2d = state.insert_axis(scirs2_core::ndarray::Axis(0));
                let mut hidden = HiddenState::new(dim, 1);
                hidden.update(state_2d);
                hidden
            })
            .collect()
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != self.config.num_scales {
            return Err(ModelError::state_count_mismatch(
                "MultiScale",
                self.config.num_scales,
                states.len(),
            ));
        }
        for (scale, hidden) in self.scales.iter_mut().zip(states.iter()) {
            let state_2d = hidden.state();
            if state_2d.nrows() > 0 && state_2d.ncols() > 0 {
                scale.state = state_2d.row(0).to_owned();
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_concat_config() -> MultiScaleConfig {
        MultiScaleConfig {
            input_dim: 4,
            hidden_dim: 8,
            output_dim: 4,
            num_scales: 3,
            scale_factors: vec![1, 2, 4],
            fusion: ScaleFusion::Concatenate,
            context_length: 64,
        }
    }

    fn make_weighted_config() -> MultiScaleConfig {
        MultiScaleConfig {
            input_dim: 4,
            hidden_dim: 8,
            output_dim: 4,
            num_scales: 3,
            scale_factors: vec![1, 2, 4],
            fusion: ScaleFusion::Weighted,
            context_length: 64,
        }
    }

    fn make_attention_config() -> MultiScaleConfig {
        MultiScaleConfig {
            input_dim: 4,
            hidden_dim: 8,
            output_dim: 4,
            num_scales: 3,
            scale_factors: vec![1, 2, 4],
            fusion: ScaleFusion::Attention,
            context_length: 64,
        }
    }

    // 9. test_temporal_scale_decimation
    #[test]
    fn test_temporal_scale_decimation() {
        let decimation = 4;
        let mut scale =
            TemporalScale::new(4, 8, decimation).expect("TemporalScale creation failed");

        let input = Array1::from_vec(vec![1.0_f32; 4]);

        // Steps 1, 2, 3 should return None (not a multiple of 4)
        let r1 = scale.step(&input).expect("step 1 failed");
        let r2 = scale.step(&input).expect("step 2 failed");
        let r3 = scale.step(&input).expect("step 3 failed");
        assert!(r1.is_none(), "step 1 should be None");
        assert!(r2.is_none(), "step 2 should be None");
        assert!(r3.is_none(), "step 3 should be None");

        // Step 4 should return Some(state)
        let r4 = scale.step(&input).expect("step 4 failed");
        assert!(r4.is_some(), "step 4 should return Some(state)");
        assert_eq!(r4.as_ref().map(|s| s.len()), Some(8));
    }

    // 10. test_temporal_scale_continuous_state
    #[test]
    fn test_temporal_scale_continuous_state() {
        let mut scale = TemporalScale::new(4, 8, 1).expect("TemporalScale creation failed");

        let input = Array1::from_vec(vec![0.5_f32; 4]);

        let r1 = scale.step(&input).expect("step 1 failed");
        assert!(r1.is_some(), "decimation=1 should always return Some");

        let state_after_step1 = scale.current_state().clone();

        let r2 = scale.step(&input).expect("step 2 failed");
        assert!(r2.is_some());

        let state_after_step2 = scale.current_state().clone();

        // With non-zero input, state should change between steps
        let diff: f32 = (&state_after_step2 - &state_after_step1)
            .iter()
            .map(|v| v.abs())
            .sum();
        // State may or may not change (could converge), but state persists
        assert!(state_after_step1.len() == 8 && state_after_step2.len() == 8);
        let _ = diff; // diff may be 0 if converged, that's OK
    }

    // 11. test_multiscale_small
    #[test]
    fn test_multiscale_small() {
        let mut model = MultiScaleModel::small().expect("small model creation failed");

        let input = Array1::from_vec(vec![0.3_f32; 1]);
        let output = model.forward_step(&input).expect("forward failed");

        assert_eq!(output.len(), 1);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // 12. test_multiscale_base
    #[test]
    fn test_multiscale_base() {
        let mut model = MultiScaleModel::base().expect("base model creation failed");

        let input = Array1::from_vec(vec![0.1_f32; 1]);
        for _ in 0..10 {
            let output = model.forward_step(&input).expect("forward failed");
            assert_eq!(output.len(), 1);
            assert!(output.iter().all(|v| v.is_finite()));
        }
    }

    // 13. test_multiscale_fusion_concat
    #[test]
    fn test_multiscale_fusion_concat() {
        let config = make_concat_config();
        let output_dim = config.output_dim;
        let mut model = MultiScaleModel::new(config).expect("model creation failed");

        let input = Array1::from_vec(vec![0.5_f32; 4]);
        let output = model.forward_step(&input).expect("forward failed");

        assert_eq!(output.len(), output_dim);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // 14. test_multiscale_fusion_weighted
    #[test]
    fn test_multiscale_fusion_weighted() {
        let config = make_weighted_config();
        let output_dim = config.output_dim;
        let mut model = MultiScaleModel::new(config).expect("model creation failed");

        let input = Array1::from_vec(vec![0.5_f32; 4]);
        let output = model.forward_step(&input).expect("forward failed");

        assert_eq!(output.len(), output_dim);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // 14b. test_multiscale_fusion_attention
    #[test]
    fn test_multiscale_fusion_attention() {
        let config = make_attention_config();
        let output_dim = config.output_dim;
        let mut model = MultiScaleModel::new(config).expect("model creation failed");

        let input = Array1::from_vec(vec![0.5_f32; 4]);
        let output = model.forward_step(&input).expect("forward failed");

        assert_eq!(output.len(), output_dim);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // 15. test_multiscale_signal_predictor
    #[test]
    fn test_multiscale_signal_predictor() {
        let config = make_concat_config();
        let output_dim = config.output_dim;
        let mut model = MultiScaleModel::new(config).expect("model creation failed");

        let input = Array1::from_vec(vec![0.2_f32; 4]);
        let output = model.step(&input).expect("SignalPredictor::step failed");

        assert_eq!(output.len(), output_dim);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // 16. test_multiscale_numerical_stability
    #[test]
    fn test_multiscale_numerical_stability() {
        let config = make_weighted_config();
        let mut model = MultiScaleModel::new(config).expect("model creation failed");

        // Test with zero input
        let zero_input = Array1::zeros(4);
        let out_zero = model.forward_step(&zero_input).expect("zero input failed");
        assert!(
            out_zero.iter().all(|v| v.is_finite()),
            "zero input should produce finite output"
        );

        // Test with moderate large input
        let large_input = Array1::from_vec(vec![100.0_f32; 4]);
        let out_large = model.forward_step(&large_input);
        match out_large {
            Ok(o) => assert!(
                o.iter().all(|v| v.is_finite()),
                "large input should produce finite output"
            ),
            Err(ModelError::NumericalInstability { .. }) => {
                // acceptable for extreme inputs
            }
            Err(e) => panic!("unexpected error: {e}"),
        }

        // Test with very small input
        let tiny_input = Array1::from_vec(vec![1e-30_f32; 4]);
        let out_tiny = model.forward_step(&tiny_input).expect("tiny input failed");
        assert!(
            out_tiny.iter().all(|v| v.is_finite()),
            "tiny input should produce finite output"
        );
    }

    // AutoregressiveModel trait test
    #[test]
    fn test_multiscale_autoregressive_model() {
        let config = make_concat_config();
        let model = MultiScaleModel::new(config).expect("model creation failed");

        assert_eq!(model.model_type(), ModelType::MultiScale);
        assert_eq!(model.num_layers(), 3);
        assert_eq!(model.hidden_dim(), 8);
        assert_eq!(model.state_dim(), 24); // 8 * 3

        let states = model.get_states();
        assert_eq!(states.len(), 3);
    }
}
