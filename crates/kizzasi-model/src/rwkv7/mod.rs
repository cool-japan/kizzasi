//! RWKV v7: Next Generation Receptance Weighted Key Value
//!
//! RWKV v7 introduces several key innovations over v6:
//!
//! - **Data-dependent time decay**: Decay is computed per-token from input,
//!   not just a learned static parameter. This allows the model to dynamically
//!   control how much historical state to retain based on current context.
//!
//! - **Value gate**: An additional SiLU gate on the value path provides
//!   finer-grained control over information flow.
//!
//! - **Bonus gate**: An extra learned attention-like offset that enriches
//!   the query-key interaction beyond simple recurrence.
//!
//! # Architecture
//!
//! ```text
//! Input -> [LayerNorm] -> [Time-Mixing v7] -> [Add] ->
//!            |                                   |
//!         [LayerNorm] -> [Channel-Mixing]  -> [Add] -> Output
//! ```
//!
//! ## Time-Mixing v7 Forward Pass
//!
//! 1. Token shift: `dx = x - prev_x`, update shift state
//! 2. Receptance: `r = sigmoid(w_r @ (x + lerp_r * dx))`
//! 3. Data-dependent decay: `w = sigmoid(w_w @ (x + lerp_w * dx))`
//! 4. Key: `k = w_k @ (x + lerp_k * dx)`
//! 5. Value: `v = w_v @ (x + lerp_v * dx)`
//! 6. Value gate (v7): `g = silu(w_g @ x)`
//! 7. Bonus gate (v7): `a = sigmoid(w_a @ x)`
//! 8. Decay gate (v7): `b = sigmoid(w_b @ x)`
//! 9. Per-head WKV with data-dependent decay and bonus attention
//! 10. Apply value gate: `output = g * ln_x(concat(heads))`
//! 11. Output projection: `out = w_o @ output`
//!
//! # Data-Dependent Decay — Mathematical Detail
//!
//! ## Dynamic Time Decay
//!
//! Unlike v6's static decay `w`, v7 computes decay from the input:
//!
//! ```text
//! w_t = σ(W_w · (x_t + μ_w ⊙ (x_t - x_{t-1})))    ∈ (0, 1)^D
//! ```
//!
//! where σ is sigmoid, making the decay data-dependent per token.
//!
//! ## Per-Head WKV Update (v7)
//!
//! For each head h with state S_h ∈ ℝ^{d_h × d_h}:
//!
//! ```text
//! S_h ← diag(w_t^h) · S_h + k_t^h · (v_t^h)^T     (rank-1 outer product update)
//! o_t^h = r_t^h · (S_h · 1 + a_t^h ⊙ k_t^h)        (with bonus attention)
//! ```
//!
//! ## Value Gate
//!
//! ```text
//! g_t = SiLU(W_g · x_t)
//! output_t = g_t ⊙ GroupNorm(Concat(o_t^1, ..., o_t^H))
//! ```
//!
//! # References
//!
//! - RWKV: <https://github.com/BlinkDL/RWKV-LM>
//! - RWKV v7 paper: <https://arxiv.org/abs/2503.14456>

pub mod channel_mixing;
pub mod time_mixing;

pub use time_mixing::Rwkv7TimeMixing;

use channel_mixing::Rwkv7ChannelMixing;
use time_mixing::SeededRng;

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{CoreResult, HiddenState, LayerNorm, NormType, SignalPredictor};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_json;

#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// RWKV v7 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rwkv7Config {
    /// Input dimension (signal width)
    pub input_dim: usize,
    /// Hidden dimension (d_model)
    pub hidden_dim: usize,
    /// Number of transformer-like layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Per-head dimension (`hidden_dim / num_heads`)
    pub head_dim: usize,
    /// FFN expansion factor (default 3.5x)
    pub expand_factor: f32,
    /// Maximum context length (theoretical; RNN has infinite via recurrence)
    pub context_length: usize,
    /// Time decay initialization bias
    pub time_decay_init: f32,
}

impl Default for Rwkv7Config {
    fn default() -> Self {
        let hidden_dim = 768;
        let num_heads = 12;
        Self {
            input_dim: 1,
            hidden_dim,
            num_layers: 24,
            num_heads,
            head_dim: hidden_dim / num_heads,
            expand_factor: 3.5,
            context_length: 16384,
            time_decay_init: -6.0,
        }
    }
}

impl Rwkv7Config {
    /// Create default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Small v7 model for quick experiments
    pub fn small(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            head_dim: 64,
            expand_factor: 3.5,
            context_length: 4096,
            time_decay_init: -5.0,
        }
    }

    /// Base v7 model
    pub fn base(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            head_dim: 64,
            expand_factor: 3.5,
            context_length: 8192,
            time_decay_init: -6.0,
        }
    }

    /// Large v7 model (7B-class)
    pub fn large(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            expand_factor: 3.5,
            context_length: 16384,
            time_decay_init: -6.0,
        }
    }

    /// Builder: set input dimension
    pub fn input_dim(mut self, dim: usize) -> Self {
        self.input_dim = dim;
        self
    }

    /// Builder: set hidden dimension (recomputes head_dim)
    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        if let Some(d) = dim.checked_div(self.num_heads) {
            self.head_dim = d;
        }
        self
    }

    /// Builder: set number of layers
    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Builder: set number of heads (recomputes head_dim)
    pub fn num_heads(mut self, n: usize) -> Self {
        self.num_heads = n;
        if let Some(d) = self.hidden_dim.checked_div(n) {
            self.head_dim = d;
        }
        self
    }

    /// Builder: set maximum context length
    pub fn context_length(mut self, len: usize) -> Self {
        self.context_length = len;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> ModelResult<()> {
        if self.hidden_dim == 0 {
            return Err(ModelError::invalid_config("hidden_dim must be > 0"));
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
        if self.expand_factor <= 0.0 {
            return Err(ModelError::invalid_config("expand_factor must be > 0"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RWKV-v7 State
// ---------------------------------------------------------------------------

/// Per-layer recurrent state for RWKV v7
pub struct Rwkv7State {
    /// Per-head WKV state matrices: `(head_dim, head_dim)` per head per layer.
    /// Outer vec is layers, inner vec is heads.
    pub wkv_states: Vec<Vec<Array2<f32>>>,
    /// Token shift state for each layer (previous token embedding)
    pub shift_states: Vec<Array1<f32>>,
}

impl Rwkv7State {
    /// Create a fresh zero state for the given config
    pub fn new(config: &Rwkv7Config) -> Self {
        let wkv_states = (0..config.num_layers)
            .map(|_| {
                (0..config.num_heads)
                    .map(|_| Array2::zeros((config.head_dim, config.head_dim)))
                    .collect()
            })
            .collect();
        let shift_states = (0..config.num_layers)
            .map(|_| Array1::zeros(config.hidden_dim))
            .collect();
        Self {
            wkv_states,
            shift_states,
        }
    }

    /// Reset all states to zero
    pub fn reset(&mut self) {
        for layer_states in &mut self.wkv_states {
            for head_state in layer_states.iter_mut() {
                head_state.fill(0.0);
            }
        }
        for shift in &mut self.shift_states {
            shift.fill(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Rwkv7 Layer
// ---------------------------------------------------------------------------

/// A single RWKV v7 layer (time-mixing + channel-mixing with residuals)
struct Rwkv7Layer {
    ln1: LayerNorm,
    ln2: LayerNorm,
    time_mixing: Rwkv7TimeMixing,
    channel_mixing: Rwkv7ChannelMixing,
}

impl Rwkv7Layer {
    fn new(config: &Rwkv7Config) -> ModelResult<Self> {
        let ln1 = LayerNorm::new(config.hidden_dim, NormType::RMSNorm).with_eps(1e-5);
        let ln2 = LayerNorm::new(config.hidden_dim, NormType::RMSNorm).with_eps(1e-5);
        let time_mixing = Rwkv7TimeMixing::new(config)?;
        let channel_mixing = Rwkv7ChannelMixing::new(config)?;
        Ok(Self {
            ln1,
            ln2,
            time_mixing,
            channel_mixing,
        })
    }

    fn forward(
        &mut self,
        x: &Array1<f32>,
        state: &mut Rwkv7State,
        layer_idx: usize,
    ) -> ModelResult<Array1<f32>> {
        // Time-mixing with residual
        let x_norm = self.ln1.forward(x);
        let tm_out = self.time_mixing.forward(&x_norm, state, layer_idx)?;
        let x_after_tm = x + &tm_out;

        // Channel-mixing with residual
        let x_norm2 = self.ln2.forward(&x_after_tm);
        let cm_out = self
            .channel_mixing
            .forward(&x_norm2)
            .map_err(|e| ModelError::forward_error(layer_idx, format!("channel mixing: {e}")))?;
        let output = &x_after_tm + &cm_out;

        Ok(output)
    }

    fn reset_channel_mixing(&mut self) {
        self.channel_mixing.reset();
    }
}

// ---------------------------------------------------------------------------
// Rwkv7Model
// ---------------------------------------------------------------------------

/// Full RWKV v7 model
pub struct Rwkv7Model {
    /// Public configuration
    pub config: Rwkv7Config,
    layers: Vec<Rwkv7Layer>,
    ln_out: LayerNorm,
    pub(crate) input_proj: Array2<f32>,
    output_proj: Array2<f32>,
    state: Rwkv7State,
}

impl Rwkv7Model {
    /// Create a new RWKV v7 model from config
    pub fn new(config: Rwkv7Config) -> ModelResult<Self> {
        config.validate()?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(Rwkv7Layer::new(&config)?);
        }

        let ln_out = LayerNorm::new(config.hidden_dim, NormType::RMSNorm).with_eps(1e-5);

        let mut rng = SeededRng::new(7777 + config.hidden_dim as u64);
        let scale = (2.0 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let input_proj = Array2::from_shape_fn((config.input_dim, config.hidden_dim), |_| {
            rng.next_f32() * scale
        });
        let output_proj = Array2::from_shape_fn((config.hidden_dim, config.input_dim), |_| {
            rng.next_f32() * scale
        });

        let state = Rwkv7State::new(&config);

        debug!(
            "Created RWKV v7 model: {} layers, {} hidden, {} heads",
            config.num_layers, config.hidden_dim, config.num_heads
        );

        Ok(Self {
            config,
            layers,
            ln_out,
            input_proj,
            output_proj,
            state,
        })
    }

    /// Create a small model for testing/benchmarking
    pub fn small() -> ModelResult<Self> {
        Self::new(Rwkv7Config::small(1))
    }

    /// Create a base model
    pub fn base() -> ModelResult<Self> {
        Self::new(Rwkv7Config::base(1))
    }

    /// Create a large model
    pub fn large() -> ModelResult<Self> {
        Self::new(Rwkv7Config::large(1))
    }

    /// Initialize a fresh state for this model
    pub fn init_state(&self) -> Rwkv7State {
        Rwkv7State::new(&self.config)
    }

    /// Get the configuration
    pub fn config(&self) -> &Rwkv7Config {
        &self.config
    }

    /// Save model weights to a JSON file as `HashMap<String, Vec<f32>>`.
    ///
    /// Keys serialised:
    /// - `input_proj` / `output_proj`: top-level input/output projections
    /// - Per-layer time-mixing: `layers.{i}.time_mixing.{w_r,w_w,w_k,w_v,w_o,w_g,w_a,w_b,lerp_r,lerp_w,lerp_k,lerp_v}`
    /// - Per-layer channel-mixing: `layers.{i}.channel_mixing.{time_mix_k,time_mix_r,key_proj,value_proj,receptance_proj}`
    pub fn save_weights_json<P: AsRef<std::path::Path>>(&self, path: P) -> ModelResult<()> {
        use std::collections::HashMap;
        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();

        weights.insert(
            "input_proj".to_string(),
            self.input_proj.iter().copied().collect(),
        );
        weights.insert(
            "output_proj".to_string(),
            self.output_proj.iter().copied().collect(),
        );

        for (i, layer) in self.layers.iter().enumerate() {
            let tm = format!("layers.{}.time_mixing", i);
            let cm = format!("layers.{}.channel_mixing", i);

            // Time-mixing projection weights (2D)
            weights.insert(
                format!("{}.w_r", tm),
                layer.time_mixing.w_r.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.w_w", tm),
                layer.time_mixing.w_w.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.w_k", tm),
                layer.time_mixing.w_k.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.w_v", tm),
                layer.time_mixing.w_v.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.w_o", tm),
                layer.time_mixing.w_o.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.w_g", tm),
                layer.time_mixing.w_g.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.w_a", tm),
                layer.time_mixing.w_a.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.w_b", tm),
                layer.time_mixing.w_b.iter().copied().collect(),
            );
            // Lerp coefficients (1D)
            weights.insert(
                format!("{}.lerp_r", tm),
                layer.time_mixing.lerp_r.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.lerp_w", tm),
                layer.time_mixing.lerp_w.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.lerp_k", tm),
                layer.time_mixing.lerp_k.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.lerp_v", tm),
                layer.time_mixing.lerp_v.iter().copied().collect(),
            );

            // Channel-mixing lerp coefficients (1D)
            weights.insert(
                format!("{}.time_mix_k", cm),
                layer.channel_mixing.time_mix_k.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.time_mix_r", cm),
                layer.channel_mixing.time_mix_r.iter().copied().collect(),
            );
            // Channel-mixing projections (2D)
            weights.insert(
                format!("{}.key_proj", cm),
                layer.channel_mixing.key_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.value_proj", cm),
                layer.channel_mixing.value_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.receptance_proj", cm),
                layer
                    .channel_mixing
                    .receptance_proj
                    .iter()
                    .copied()
                    .collect(),
            );
        }

        let file = std::fs::File::create(path.as_ref()).map_err(|e| {
            ModelError::load_error("rwkv7 save_weights", format!("failed to create file: {e}"))
        })?;
        serde_json::to_writer(file, &weights).map_err(|e| {
            ModelError::load_error(
                "rwkv7 save_weights",
                format!("JSON serialization failed: {e}"),
            )
        })?;
        Ok(())
    }

    /// Load weights from a JSON file previously written by `save_weights_json`.
    pub fn load_weights_json<P: AsRef<std::path::Path>>(&mut self, path: P) -> ModelResult<()> {
        use std::collections::HashMap;
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            ModelError::load_error("rwkv7 load_weights", format!("failed to open file: {e}"))
        })?;
        let weights: HashMap<String, Vec<f32>> = serde_json::from_reader(file).map_err(|e| {
            ModelError::load_error(
                "rwkv7 load_weights",
                format!("JSON deserialization failed: {e}"),
            )
        })?;

        let load_2d = |map: &HashMap<String, Vec<f32>>,
                       key: &str,
                       rows: usize,
                       cols: usize|
         -> ModelResult<Option<Array2<f32>>> {
            if let Some(data) = map.get(key) {
                if data.len() != rows * cols {
                    return Err(ModelError::load_error(
                        "rwkv7 load_weights",
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
                        "rwkv7 load_weights",
                        format!("failed to reshape '{}': {e}", key),
                    )
                })?;
                Ok(Some(arr))
            } else {
                Ok(None)
            }
        };

        let load_1d = |map: &HashMap<String, Vec<f32>>,
                       key: &str,
                       expected_len: usize|
         -> ModelResult<Option<Array1<f32>>> {
            if let Some(data) = map.get(key) {
                if data.len() != expected_len {
                    return Err(ModelError::load_error(
                        "rwkv7 load_weights",
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

        let d = self.config.hidden_dim;
        let inter = (d as f32 * self.config.expand_factor) as usize;

        if let Some(arr) = load_2d(&weights, "input_proj", self.config.input_dim, d)? {
            self.input_proj = arr;
        }
        if let Some(arr) = load_2d(&weights, "output_proj", d, self.config.input_dim)? {
            self.output_proj = arr;
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let tm = format!("layers.{}.time_mixing", i);
            let cm = format!("layers.{}.channel_mixing", i);

            if let Some(arr) = load_2d(&weights, &format!("{}.w_r", tm), d, d)? {
                layer.time_mixing.w_r = arr;
            }
            if let Some(arr) = load_2d(&weights, &format!("{}.w_w", tm), d, d)? {
                layer.time_mixing.w_w = arr;
            }
            if let Some(arr) = load_2d(&weights, &format!("{}.w_k", tm), d, d)? {
                layer.time_mixing.w_k = arr;
            }
            if let Some(arr) = load_2d(&weights, &format!("{}.w_v", tm), d, d)? {
                layer.time_mixing.w_v = arr;
            }
            if let Some(arr) = load_2d(&weights, &format!("{}.w_o", tm), d, d)? {
                layer.time_mixing.w_o = arr;
            }
            if let Some(arr) = load_2d(&weights, &format!("{}.w_g", tm), d, d)? {
                layer.time_mixing.w_g = arr;
            }
            if let Some(arr) = load_2d(&weights, &format!("{}.w_a", tm), d, d)? {
                layer.time_mixing.w_a = arr;
            }
            if let Some(arr) = load_2d(&weights, &format!("{}.w_b", tm), d, d)? {
                layer.time_mixing.w_b = arr;
            }
            if let Some(arr) = load_1d(&weights, &format!("{}.lerp_r", tm), d)? {
                layer.time_mixing.lerp_r = arr;
            }
            if let Some(arr) = load_1d(&weights, &format!("{}.lerp_w", tm), d)? {
                layer.time_mixing.lerp_w = arr;
            }
            if let Some(arr) = load_1d(&weights, &format!("{}.lerp_k", tm), d)? {
                layer.time_mixing.lerp_k = arr;
            }
            if let Some(arr) = load_1d(&weights, &format!("{}.lerp_v", tm), d)? {
                layer.time_mixing.lerp_v = arr;
            }

            if let Some(arr) = load_1d(&weights, &format!("{}.time_mix_k", cm), d)? {
                layer.channel_mixing.time_mix_k = arr;
            }
            if let Some(arr) = load_1d(&weights, &format!("{}.time_mix_r", cm), d)? {
                layer.channel_mixing.time_mix_r = arr;
            }
            if let Some(arr) = load_2d(&weights, &format!("{}.key_proj", cm), d, inter)? {
                layer.channel_mixing.key_proj = arr;
            }
            if let Some(arr) = load_2d(&weights, &format!("{}.value_proj", cm), inter, d)? {
                layer.channel_mixing.value_proj = arr;
            }
            if let Some(arr) = load_2d(&weights, &format!("{}.receptance_proj", cm), d, d)? {
                layer.channel_mixing.receptance_proj = arr;
            }
        }

        Ok(())
    }
}

impl SignalPredictor for Rwkv7Model {
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Project input to hidden dim
        let mut hidden = input.dot(&self.input_proj);

        // Forward through layers
        for layer_idx in 0..self.layers.len() {
            // We need to pass `&mut self.state` and `&mut self.layers[layer_idx]`
            // simultaneously. Split the borrow by indexing.
            let layer = &mut self.layers[layer_idx];
            hidden = layer
                .forward(&hidden, &mut self.state, layer_idx)
                .map_err(|e| {
                    kizzasi_core::CoreError::InferenceError(format!("rwkv7 layer {layer_idx}: {e}"))
                })?;
        }

        // Final norm + output projection
        hidden = self.ln_out.forward(&hidden);
        let output = hidden.dot(&self.output_proj);
        Ok(output)
    }

    fn reset(&mut self) {
        self.state.reset();
        for layer in &mut self.layers {
            layer.reset_channel_mixing();
        }
    }

    fn context_window(&self) -> usize {
        // RNN-style: theoretically unlimited context via recurrence
        usize::MAX
    }
}

impl AutoregressiveModel for Rwkv7Model {
    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn state_dim(&self) -> usize {
        self.config.head_dim * self.config.num_heads
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn model_type(&self) -> ModelType {
        ModelType::Rwkv
    }

    fn get_states(&self) -> Vec<HiddenState> {
        self.state
            .wkv_states
            .iter()
            .map(|layer_heads| {
                // Flatten all heads into a single (hidden_dim, head_dim) matrix
                let total_rows = self.config.num_heads * self.config.head_dim;
                let cols = self.config.head_dim;
                let mut combined = Array2::zeros((total_rows, cols));

                for (h, head_state) in layer_heads.iter().enumerate() {
                    let row_start = h * self.config.head_dim;
                    for i in 0..self.config.head_dim {
                        for j in 0..cols {
                            combined[[row_start + i, j]] = head_state[[i, j]];
                        }
                    }
                }

                let mut hs = HiddenState::new(total_rows, cols);
                hs.update(combined);
                hs
            })
            .collect()
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != self.config.num_layers {
            return Err(ModelError::state_count_mismatch(
                "RWKV7",
                self.config.num_layers,
                states.len(),
            ));
        }

        for (layer_idx, hs) in states.iter().enumerate() {
            let combined = hs.state();
            for h in 0..self.config.num_heads {
                let row_start = h * self.config.head_dim;
                let head_state = &mut self.state.wkv_states[layer_idx][h];
                for i in 0..self.config.head_dim {
                    for j in 0..self.config.head_dim {
                        if row_start + i < combined.shape()[0] && j < combined.shape()[1] {
                            head_state[[i, j]] = combined[[row_start + i, j]];
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Backward-compatible Rwkv7 alias (matches the old scaffolding API)
// ---------------------------------------------------------------------------

/// Backward-compatible type alias: `Rwkv7` delegates to `Rwkv7Model`.
pub type Rwkv7 = Rwkv7Model;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Rwkv7Config {
        Rwkv7Config {
            input_dim: 1,
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            head_dim: 16,
            expand_factor: 2.0,
            context_length: 256,
            time_decay_init: -5.0,
        }
    }

    #[test]
    fn test_rwkv7_config_valid() {
        let config = Rwkv7Config::new();
        assert!(config.validate().is_ok());

        let bad = Rwkv7Config {
            hidden_dim: 0,
            ..Rwkv7Config::default()
        };
        assert!(bad.validate().is_err());

        let bad2 = Rwkv7Config {
            hidden_dim: 100,
            num_heads: 3,
            ..Rwkv7Config::default()
        };
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn test_rwkv7_small_forward() {
        let config = tiny_config();
        let mut model = Rwkv7Model::new(config).expect("model creation");
        let input = Array1::from_vec(vec![0.5]);
        let output = model.step(&input).expect("forward step");
        assert_eq!(output.len(), 1, "output should match input_dim");
        assert!(output[0].is_finite(), "output must be finite");
    }

    #[test]
    fn test_rwkv7_state_persistence() {
        let config = tiny_config();
        let mut model = Rwkv7Model::new(config).expect("model creation");

        let input = Array1::from_vec(vec![0.1]);
        for _ in 0..10 {
            let out = model.step(&input).expect("step");
            for &v in out.iter() {
                assert!(v.is_finite(), "output should stay finite over 10 steps");
                assert!(!v.is_nan(), "no NaN values");
            }
        }
    }

    #[test]
    fn test_rwkv7_state_reset() {
        let config = tiny_config();
        let mut model = Rwkv7Model::new(config).expect("model creation");

        let input = Array1::from_vec(vec![0.3]);

        // Run some steps
        for _ in 0..5 {
            let _ = model.step(&input).expect("step");
        }

        // Capture output after reset at step 1
        model.reset();
        let out_after_reset = model.step(&input).expect("step after reset");

        // Create a brand-new model (same deterministic weights)
        let config2 = tiny_config();
        let mut fresh = Rwkv7Model::new(config2).expect("fresh model creation");
        let out_fresh = fresh.step(&input).expect("fresh step");

        // They should be identical because weights are deterministically seeded
        for (a, b) in out_after_reset.iter().zip(out_fresh.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "reset output should match fresh model: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_rwkv7_multi_layer() {
        let mut config = tiny_config();
        config.num_layers = 4;
        let mut model = Rwkv7Model::new(config).expect("4-layer model");

        let input = Array1::from_vec(vec![0.42]);
        let out = model.step(&input).expect("forward");
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite());
    }

    #[test]
    fn test_rwkv7_signal_predictor_trait() {
        let config = tiny_config();
        let mut model = Rwkv7Model::new(config).expect("model");

        // step
        let input = Array1::from_vec(vec![1.0]);
        let out = model.step(&input).expect("step");
        assert_eq!(out.len(), 1);

        // reset
        model.reset();

        // context_window
        assert_eq!(model.context_window(), usize::MAX);
    }

    #[test]
    fn test_rwkv7_autoregressive_trait() {
        let config = tiny_config();
        let mut model = Rwkv7Model::new(config.clone()).expect("model");

        // Run a step to populate state
        let input = Array1::from_vec(vec![0.7]);
        let _ = model.step(&input).expect("step");

        // get_states / set_states roundtrip
        let states = model.get_states();
        assert_eq!(states.len(), config.num_layers);

        // Set states on a fresh model
        let mut model2 = Rwkv7Model::new(config).expect("model2");
        model2.set_states(states.clone()).expect("set_states");

        let states2 = model2.get_states();
        assert_eq!(states.len(), states2.len());

        // Verify state values match
        for (s1, s2) in states.iter().zip(states2.iter()) {
            let a = s1.state();
            let b = s2.state();
            assert_eq!(a.shape(), b.shape());
            for (va, vb) in a.iter().zip(b.iter()) {
                assert!((va - vb).abs() < 1e-6, "state roundtrip mismatch");
            }
        }
    }

    #[test]
    fn test_rwkv7_numerical_stability() {
        let config = tiny_config();
        let mut model = Rwkv7Model::new(config).expect("model");

        // Test with large input
        let large_input = Array1::from_vec(vec![1000.0]);
        let out_large = model.step(&large_input).expect("large input step");
        for &v in out_large.iter() {
            assert!(
                v.is_finite(),
                "output should be finite for large input: {v}"
            );
        }

        model.reset();

        // Test with very small input
        let small_input = Array1::from_vec(vec![1e-10]);
        let out_small = model.step(&small_input).expect("small input step");
        for &v in out_small.iter() {
            assert!(
                v.is_finite(),
                "output should be finite for small input: {v}"
            );
        }

        model.reset();

        // Test with negative input
        let neg_input = Array1::from_vec(vec![-500.0]);
        let out_neg = model.step(&neg_input).expect("negative input step");
        for &v in out_neg.iter() {
            assert!(
                v.is_finite(),
                "output should be finite for negative input: {v}"
            );
        }
    }

    #[test]
    fn test_rwkv7_hidden_dim_state_dim() {
        let config = tiny_config();
        let model = Rwkv7Model::new(config).expect("model");

        assert_eq!(model.hidden_dim(), 64);
        assert_eq!(model.state_dim(), 64); // head_dim * num_heads = 16 * 4
        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.model_type(), ModelType::Rwkv);
    }

    #[test]
    fn test_rwkv7_save_load_weights_roundtrip() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        // Build two models: one with deterministic seed weights (the "fresh" model
        // that load_weights_json will target) and a *mutated* reference whose
        // input_proj has been scaled by 2×.  Because Rwkv7Model::new uses a fixed
        // SeededRng, both would normally produce identical outputs — scaling the
        // reference projection makes them genuinely diverge before loading.
        let config = tiny_config();
        let mut reference = Rwkv7Model::new(config.clone()).expect("reference model");

        // Mutate reference.input_proj so reference ≠ a default-seeded model.
        reference.input_proj = &reference.input_proj * 2.0_f32;

        // Save the *mutated* weights to a unique temp file
        let uid = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let mut path = std::env::temp_dir();
        path.push(format!("kizzasi_rwkv7_test_{}_{}.json", pid, uid));

        reference
            .save_weights_json(&path)
            .expect("save_weights_json");

        // Load into a fresh model — starts with original (unscaled) weights
        let mut loaded = Rwkv7Model::new(config).expect("loaded model");

        // Verify the models genuinely differ BEFORE loading
        let probe = Array1::from_vec(vec![0.5]);
        let mut ref_clone = reference;
        let out_before = loaded
            .step(&probe.clone())
            .expect("loaded step before load");
        loaded.reset();

        // Load the saved (mutated) weights
        loaded.load_weights_json(&path).expect("load_weights_json");
        let _ = std::fs::remove_file(&path);

        // After loading, outputs must match the (mutated) reference
        let out_ref = ref_clone.step(&probe.clone()).expect("ref step");
        let out_after = loaded.step(&probe).expect("loaded step after load");

        assert_eq!(out_ref.len(), out_after.len());

        // The outputs before and after load must differ (non-vacuousness check)
        let diverged_before_load = out_before
            .iter()
            .zip(out_ref.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            diverged_before_load,
            "pre-load outputs must differ from reference; check that input_proj mutation took effect"
        );

        // After loading, outputs must match
        for (a, b) in out_ref.iter().zip(out_after.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "weight roundtrip: outputs diverge after load: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_rwkv7_factory_weight_injection() {
        use crate::dynamic_quantization::QuantizedWeightStorage;
        use crate::factory::ModelFactory;
        use scirs2_core::ndarray::Array2;
        use std::collections::HashMap;
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let config = tiny_config();
        let reference = Rwkv7Model::new(config.clone()).expect("reference");

        // Save reference weights to temp JSON
        let uid = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let mut tmp = std::env::temp_dir();
        tmp.push(format!("kizzasi_rwkv7_factory_{}_{}.json", pid, uid));
        reference.save_weights_json(&tmp).expect("save");

        // Read back and build QuantizedWeightStorage map
        let file = std::fs::File::open(&tmp).expect("open temp JSON");
        let f32_map: HashMap<String, Vec<f32>> =
            serde_json::from_reader(file).expect("deserialise JSON");
        let _ = std::fs::remove_file(&tmp);

        assert!(!f32_map.is_empty(), "saved weights must be non-empty");

        let mut quant_weights: HashMap<String, QuantizedWeightStorage> = HashMap::new();
        for (k, v) in f32_map {
            let len = v.len();
            let arr = Array2::from_shape_vec((1, len), v).expect("reshape to Array2");
            quant_weights.insert(k, QuantizedWeightStorage::FP32(arr));
        }

        // ModelFactory::create_rwkv7 must succeed with weight injection
        let result = ModelFactory::create_rwkv7(config, quant_weights);
        assert!(
            result.is_ok(),
            "create_rwkv7 with weights should succeed: {:?}",
            result.err()
        );

        // Verify the created model is functional
        let mut factory_model = result.expect("model from factory");
        let input = Array1::from_vec(vec![0.5]);
        let out = factory_model.step(&input).expect("factory model step");
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite(), "factory model output must be finite");
    }
}
