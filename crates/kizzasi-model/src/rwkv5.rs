//! RWKV v5: Multi-Head WKV Compatibility Layer
//!
//! RWKV v5 sits between RWKV v4 (scalar WKV) and RWKV v6 (data-dependent decay).
//! The key innovation in v5 is **multi-head WKV** with a per-head learned static
//! time-decay and an explicit time-first bonus for the current token.
//!
//! # Architecture Differences from v6
//!
//! | Feature           | v5                          | v6                        |
//! |-------------------|-----------------------------|---------------------------|
//! | Decay             | Static (learned per head)   | Data-dependent (per token)|
//! | Gate              | SiLU gate on output         | Value gate + bonus gate   |
//! | WKV state         | (head_dim × head_dim)       | Same                      |
//! | Norm              | Group norm on heads         | Group norm on heads       |
//!
//! # Forward Pass (single token, recurrent mode)
//!
//! 1. Token shift: `dx = x - prev_x`, update shift state
//! 2. Receptance: `r = sigmoid(w_r @ (x + lerp_r * dx))`
//! 3. Key:        `k = w_k @ (x + lerp_k * dx)`
//! 4. Value:      `v = w_v @ (x + lerp_v * dx)`
//! 5. Gate:       `g = silu(w_g @ x)`
//! 6. Per-head WKV (linear recurrence):
//!    ```text
//!    state_h = diag(exp(time_decay_h)) @ state_h + outer(k_h, v_h)
//!    wkv_h   = r_h @ state_h + exp(time_first_h) * (r_h ⊙ k_h) ⊙ v_h
//!    ```
//! 7. `out = group_norm(concat(wkv_h)) ⊙ g`
//! 8. `w_o @ out`
//!
//! # References
//!
//! - RWKV-LM: <https://github.com/BlinkDL/RWKV-LM>
//! - Eagle/Finch (RWKV-v5): <https://arxiv.org/abs/2404.05892>

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{sigmoid, silu, CoreResult, HiddenState, LayerNorm, NormType, SignalPredictor};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

// ---------------------------------------------------------------------------
// Deterministic weight initializer (same pattern as rwkv7.rs)
// ---------------------------------------------------------------------------

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
// Configuration
// ---------------------------------------------------------------------------

/// RWKV v5 model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rwkv5Config {
    /// Input signal dimension
    pub input_dim: usize,
    /// Hidden model dimension (d_model)
    pub hidden_dim: usize,
    /// Number of transformer-like layers
    pub num_layers: usize,
    /// Number of WKV heads
    pub num_heads: usize,
    /// Per-head dimension (`hidden_dim / num_heads`)
    pub head_dim: usize,
    /// Channel-mixing expansion factor (typically 4×)
    pub intermediate_dim: usize,
    /// Maximum context length (theoretical; RNN is infinite via recurrence)
    pub context_length: usize,
    /// Use RMSNorm instead of standard LayerNorm
    pub use_rms_norm: bool,
}

impl Default for Rwkv5Config {
    fn default() -> Self {
        let hidden_dim = 512;
        let num_heads = 8;
        Self {
            input_dim: 1,
            hidden_dim,
            num_layers: 12,
            num_heads,
            head_dim: hidden_dim / num_heads,
            intermediate_dim: hidden_dim * 4,
            context_length: 8192,
            use_rms_norm: true,
        }
    }
}

impl Rwkv5Config {
    /// Create default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Small configuration for testing / quick experiments.
    pub fn small(input_dim: usize) -> Self {
        let hidden_dim = 256;
        let num_heads = 4;
        Self {
            input_dim,
            hidden_dim,
            num_layers: 4,
            num_heads,
            head_dim: hidden_dim / num_heads,
            intermediate_dim: hidden_dim * 4,
            context_length: 4096,
            use_rms_norm: true,
        }
    }

    /// Validate configuration consistency.
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
        if self.head_dim != self.hidden_dim / self.num_heads {
            return Err(ModelError::invalid_config(
                "head_dim must equal hidden_dim / num_heads",
            ));
        }
        if self.intermediate_dim == 0 {
            return Err(ModelError::invalid_config("intermediate_dim must be > 0"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Recurrent State
// ---------------------------------------------------------------------------

/// Per-layer recurrent state for RWKV v5.
pub struct Rwkv5State {
    /// Per-layer, per-head WKV state: outer vec = layers, inner vec = heads, each (head_dim, head_dim).
    pub wkv_states: Vec<Vec<Array2<f32>>>,
    /// Per-layer token shift state (previous token's hidden representation).
    pub shift_states: Vec<Array1<f32>>,
}

impl Rwkv5State {
    /// Construct a fresh zero state for the given config.
    pub fn new(config: &Rwkv5Config) -> Self {
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

    /// Reset all state to zero.
    pub fn reset(&mut self) {
        for layer in &mut self.wkv_states {
            for head in layer.iter_mut() {
                head.fill(0.0);
            }
        }
        for shift in &mut self.shift_states {
            shift.fill(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// RWKV v5 Time Mixing
// ---------------------------------------------------------------------------

/// Multi-head WKV time-mixing block for RWKV v5.
pub struct Rwkv5TimeMixing {
    w_r: Array2<f32>, // receptance  (hidden_dim × hidden_dim)
    w_k: Array2<f32>, // key         (hidden_dim × hidden_dim)
    w_v: Array2<f32>, // value       (hidden_dim × hidden_dim)
    w_o: Array2<f32>, // output      (hidden_dim × hidden_dim)
    w_g: Array2<f32>, // SiLU gate   (hidden_dim × hidden_dim)

    lerp_r: Array1<f32>,
    lerp_k: Array1<f32>,
    lerp_v: Array1<f32>,

    /// Per-head static time decay: one value per head, in log space.
    /// Actual decay = exp(time_decay[h]).  Typically initialised to negative values
    /// so exp(time_decay) < 1.
    time_decay: Array1<f32>,
    /// Per-head "bonus" for current token: `exp(time_first[h])`.
    time_first: Array1<f32>,

    /// Group normalisation applied to the concatenated head outputs.
    ln_x: LayerNorm,

    num_heads: usize,
    head_dim: usize,
}

impl Rwkv5TimeMixing {
    fn new(config: &Rwkv5Config) -> ModelResult<Self> {
        let d = config.hidden_dim;
        let h = config.num_heads;
        let mut rng = SeededRng::new(55 + d as u64 + h as u64);
        let scale = (2.0 / d as f32).sqrt();

        let make_proj = |rng: &mut SeededRng| -> Array2<f32> {
            Array2::from_shape_fn((d, d), |_| rng.next_f32() * scale)
        };

        let w_r = make_proj(&mut rng);
        let w_k = make_proj(&mut rng);
        let w_v = make_proj(&mut rng);
        let w_o = make_proj(&mut rng);
        let w_g = make_proj(&mut rng);

        let lerp_r = Array1::from_shape_fn(d, |_| rng.next_f32().abs() * 0.5 + 0.25);
        let lerp_k = Array1::from_shape_fn(d, |_| rng.next_f32().abs() * 0.5 + 0.25);
        let lerp_v = Array1::from_shape_fn(d, |_| rng.next_f32().abs() * 0.5 + 0.25);

        // Static time decay: one scalar per head, initialised to ~ -5 (slow decay)
        let time_decay = Array1::from_shape_fn(h, |_| -5.0 + rng.next_f32() * 0.5);
        // Time-first bonus: small positive values
        let time_first = Array1::from_shape_fn(h, |_| rng.next_f32().abs() * 0.1);

        let norm_type = if config.use_rms_norm {
            NormType::RMSNorm
        } else {
            NormType::LayerNorm
        };
        let ln_x = LayerNorm::new(d, norm_type).with_eps(1e-5);

        Ok(Self {
            w_r,
            w_k,
            w_v,
            w_o,
            w_g,
            lerp_r,
            lerp_k,
            lerp_v,
            time_decay,
            time_first,
            ln_x,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
        })
    }

    /// Single-step forward pass, mutating `state` for layer `layer_idx`.
    fn forward(
        &self,
        x: &Array1<f32>,
        state: &mut Rwkv5State,
        layer_idx: usize,
    ) -> ModelResult<Array1<f32>> {
        let d = x.len();

        // 1. Token shift
        let prev = &state.shift_states[layer_idx];
        let dx = x - prev;
        state.shift_states[layer_idx] = x.clone();

        // 2. Lerp-mixed projections inputs
        let xr = x + &(&self.lerp_r * &dx);
        let xk = x + &(&self.lerp_k * &dx);
        let xv = x + &(&self.lerp_v * &dx);

        // 3. Linear projections
        let r_raw = matvec(&self.w_r, &xr);
        let k_raw = matvec(&self.w_k, &xk);
        let v_raw = matvec(&self.w_v, &xv);

        // 4. Gate
        let r = sigmoid(&r_raw);
        let g = silu(&matvec(&self.w_g, x));

        // 5. Per-head WKV
        let mut output = Array1::zeros(d);

        for h in 0..self.num_heads {
            let lo = h * self.head_dim;
            let hi = lo + self.head_dim;

            let r_h = r.slice(scirs2_core::ndarray::s![lo..hi]).to_owned();
            let k_h = k_raw.slice(scirs2_core::ndarray::s![lo..hi]).to_owned();
            let v_h = v_raw.slice(scirs2_core::ndarray::s![lo..hi]).to_owned();

            let decay_h = self.time_decay[h].exp().clamp(0.0, 1.0);
            let first_h = self.time_first[h].exp();

            let head_state = &mut state.wkv_states[layer_idx][h];

            // state_h = diag(decay_h) @ state_h  +  outer(k_h, v_h)
            for i in 0..self.head_dim {
                for j in 0..self.head_dim {
                    head_state[[i, j]] = decay_h * head_state[[i, j]] + k_h[i] * v_h[j];
                }
            }

            // wkv_h = r_h @ state_h + first_h * (r_h ⊙ k_h) ⊙ v_h
            let state_r = matvec(head_state, &r_h);
            for i in 0..self.head_dim {
                let direct = first_h * r_h[i] * k_h[i] * v_h[i];
                output[lo + i] = state_r[i] + direct;
            }
        }

        // 6. Group norm then gate
        let normed = self.ln_x.forward(&output);
        let gated = &g * &normed;

        // 7. Output projection
        let out = matvec(&self.w_o, &gated);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// RWKV v5 Channel Mixing
// ---------------------------------------------------------------------------

/// Channel-mixing (FFN) block for RWKV v5.
struct Rwkv5ChannelMixing {
    hidden_dim: usize,
    intermediate_dim: usize,
    time_mix_k: Array1<f32>,
    time_mix_r: Array1<f32>,
    key_proj: Array2<f32>,        // (hidden_dim, intermediate_dim)
    value_proj: Array2<f32>,      // (intermediate_dim, hidden_dim)
    receptance_proj: Array2<f32>, // (hidden_dim, hidden_dim)
    prev_x: Array1<f32>,
}

impl Rwkv5ChannelMixing {
    fn new(config: &Rwkv5Config) -> ModelResult<Self> {
        let d = config.hidden_dim;
        let inter = config.intermediate_dim;
        let mut rng = SeededRng::new(137 + d as u64 + inter as u64);
        let scale = (2.0 / d as f32).sqrt();

        let time_mix_k = Array1::from_shape_fn(d, |_| rng.next_f32().abs() * 0.5 + 0.25);
        let time_mix_r = Array1::from_shape_fn(d, |_| rng.next_f32().abs() * 0.5 + 0.25);
        let key_proj = Array2::from_shape_fn((d, inter), |_| rng.next_f32() * scale);
        let value_proj = Array2::from_shape_fn((inter, d), |_| rng.next_f32() * scale);
        let receptance_proj = Array2::from_shape_fn((d, d), |_| rng.next_f32() * scale);

        Ok(Self {
            hidden_dim: d,
            intermediate_dim: inter,
            time_mix_k,
            time_mix_r,
            key_proj,
            value_proj,
            receptance_proj,
            prev_x: Array1::zeros(d),
        })
    }

    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let d = x.len().min(self.hidden_dim);

        let mut xk = Array1::zeros(d);
        let mut xr = Array1::zeros(d);
        for i in 0..d {
            let prev = if i < self.prev_x.len() {
                self.prev_x[i]
            } else {
                0.0
            };
            xk[i] = self.time_mix_k[i] * x[i] + (1.0 - self.time_mix_k[i]) * prev;
            xr[i] = self.time_mix_r[i] * x[i] + (1.0 - self.time_mix_r[i]) * prev;
        }

        // Key: squared ReLU expansion
        let k = project(&self.key_proj, &xk, self.intermediate_dim);
        let k_act = k.mapv(|v| {
            let relu = v.max(0.0);
            relu * relu
        });
        let vk = project_t(&self.value_proj, &k_act, self.hidden_dim);

        // Receptance gating
        let r_raw = project(&self.receptance_proj, &xr, self.hidden_dim.min(d));
        let r_sig = sigmoid(&r_raw);

        let mut output = Array1::zeros(d);
        for i in 0..d.min(vk.len()).min(r_sig.len()) {
            output[i] = r_sig[i] * vk[i];
        }

        self.prev_x = x.slice(scirs2_core::ndarray::s![..d]).to_owned();
        Ok(output)
    }

    fn reset(&mut self) {
        self.prev_x.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// RWKV v5 Layer
// ---------------------------------------------------------------------------

struct Rwkv5Layer {
    ln1: LayerNorm,
    ln2: LayerNorm,
    time_mixing: Rwkv5TimeMixing,
    channel_mixing: Rwkv5ChannelMixing,
}

impl Rwkv5Layer {
    fn new(config: &Rwkv5Config) -> ModelResult<Self> {
        let norm_type = if config.use_rms_norm {
            NormType::RMSNorm
        } else {
            NormType::LayerNorm
        };
        let ln1 = LayerNorm::new(config.hidden_dim, norm_type).with_eps(1e-5);
        let ln2 = LayerNorm::new(config.hidden_dim, norm_type).with_eps(1e-5);
        let time_mixing = Rwkv5TimeMixing::new(config)?;
        let channel_mixing = Rwkv5ChannelMixing::new(config)?;
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
        state: &mut Rwkv5State,
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
        Ok(&x_after_tm + &cm_out)
    }

    fn reset_channel_mixing(&mut self) {
        self.channel_mixing.reset();
    }
}

// ---------------------------------------------------------------------------
// RWKV v5 Model
// ---------------------------------------------------------------------------

/// Full RWKV v5 model: embedding → layers → output projection.
pub struct Rwkv5Model {
    /// Public model configuration.
    pub config: Rwkv5Config,
    layers: Vec<Rwkv5Layer>,
    ln_out: LayerNorm,
    input_proj: Array2<f32>,  // (input_dim, hidden_dim)
    output_proj: Array2<f32>, // (hidden_dim, input_dim)
    state: Rwkv5State,
}

impl Rwkv5Model {
    /// Create a new RWKV v5 model from the given configuration.
    pub fn new(config: Rwkv5Config) -> ModelResult<Self> {
        config.validate()?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(Rwkv5Layer::new(&config)?);
        }

        let norm_type = if config.use_rms_norm {
            NormType::RMSNorm
        } else {
            NormType::LayerNorm
        };
        let ln_out = LayerNorm::new(config.hidden_dim, norm_type).with_eps(1e-5);

        let mut rng = SeededRng::new(5555 + config.hidden_dim as u64);
        let scale = (2.0 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let input_proj = Array2::from_shape_fn((config.input_dim, config.hidden_dim), |_| {
            rng.next_f32() * scale
        });
        let output_proj = Array2::from_shape_fn((config.hidden_dim, config.input_dim), |_| {
            rng.next_f32() * scale
        });

        let state = Rwkv5State::new(&config);

        debug!(
            "Created RWKV v5 model: {} layers, hidden={}, heads={}",
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

    /// Create a small model for testing.
    pub fn small() -> ModelResult<Self> {
        Self::new(Rwkv5Config::small(1))
    }

    /// Get a fresh zero recurrent state for this model.
    pub fn init_state(&self) -> Rwkv5State {
        Rwkv5State::new(&self.config)
    }
}

impl SignalPredictor for Rwkv5Model {
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        crate::check_input_dim(input, self.input_proj.shape()[0])?;

        // Project input into hidden space
        let mut hidden = input.dot(&self.input_proj);

        // Forward through all layers
        for layer_idx in 0..self.layers.len() {
            let layer = &mut self.layers[layer_idx];
            hidden = layer
                .forward(&hidden, &mut self.state, layer_idx)
                .map_err(|e| {
                    kizzasi_core::CoreError::InferenceError(format!("rwkv5 layer {layer_idx}: {e}"))
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
        // RNN-style recurrence → theoretically unlimited context
        usize::MAX
    }
}

impl AutoregressiveModel for Rwkv5Model {
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
        ModelType::Rwkv5
    }

    fn get_states(&self) -> Vec<HiddenState> {
        self.state
            .wkv_states
            .iter()
            .map(|layer_heads| {
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
                "RWKV5",
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
// Internal helpers
// ---------------------------------------------------------------------------

/// Matrix-vector multiply: `y = W @ x` (W is rows × cols, x is cols-vector)
///
/// `w.dot(x)` dispatches to ndarray/matrixmultiply with the correct
/// (row-major) traversal and no per-element `Index` bounds-check overhead,
/// instead of walking `w[[i, j]]` through ndarray's checked 2D indexing one
/// scalar at a time. It requires `x.len() == w.ncols()` exactly (ndarray
/// panics otherwise), which holds for every call site in this module; the
/// original clamped loop is kept as a fallback for any case where that
/// invariant doesn't hold, so a shape mismatch degrades to the old (slower,
/// silently truncated) behavior instead of panicking.
fn matvec(w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
    if x.len() == w.shape()[1] {
        return w.dot(x);
    }
    let rows = w.shape()[0];
    let cols = w.shape()[1];
    let xlen = x.len();
    let mut out = Array1::zeros(rows);
    for i in 0..rows {
        let mut sum = 0.0_f32;
        for j in 0..cols.min(xlen) {
            sum += w[[i, j]] * x[j];
        }
        out[i] = sum;
    }
    out
}

/// Project `x` through `w` (rows × out_dim layout) up to `out_dim` outputs:
/// `out[j] = sum_i w[i, j] * x[i]`, i.e. `out = w^T @ x` truncated to
/// `out_dim`.
///
/// `w.t().dot(x)` is bit-identical to the manual loop below when shapes line
/// up (`x.len() == w.nrows()` and `out_dim >= w.ncols()`, the normal case)
/// and traverses each column of `w` as a genuine strided ndarray view
/// instead of walking `w[[i, j]]` with `i` (the row index) as the inner,
/// stride-`cols` loop variable — the worst-case traversal for a row-major
/// array. The manual loop remains as a fallback for the truncated/padded
/// case the `.min()` clamps exist for.
fn project(w: &Array2<f32>, x: &Array1<f32>, out_dim: usize) -> Array1<f32> {
    let rows = w.shape()[0];
    let cols = w.shape()[1];
    let xlen = x.len();
    let n_out = out_dim.min(cols);
    if xlen == rows && n_out == cols {
        return w.t().dot(x);
    }
    let mut out = Array1::zeros(n_out);
    for j in 0..n_out {
        let mut sum = 0.0_f32;
        for i in 0..rows.min(xlen) {
            sum += w[[i, j]] * x[i];
        }
        out[j] = sum;
    }
    out
}

/// Project `x` through `w` (rows × out_dim layout, transposed access) for
/// down-projection. Computation is identical to [`project`] — kept as a
/// separate name purely for call-site clarity (down- vs up-projection).
fn project_t(w: &Array2<f32>, x: &Array1<f32>, out_dim: usize) -> Array1<f32> {
    project(w, x, out_dim)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kizzasi_core::SignalPredictor;

    fn tiny_config() -> Rwkv5Config {
        Rwkv5Config {
            input_dim: 1,
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            head_dim: 16,
            intermediate_dim: 128,
            context_length: 256,
            use_rms_norm: true,
        }
    }

    // -----------------------------------------------------------------------
    // Test 9: Config validity
    // -----------------------------------------------------------------------
    #[test]
    fn test_rwkv5_config_valid() {
        let config = tiny_config();
        assert!(config.validate().is_ok());

        // Bad: hidden_dim == 0
        let bad = Rwkv5Config {
            hidden_dim: 0,
            ..tiny_config()
        };
        assert!(bad.validate().is_err());

        // Bad: not divisible by num_heads
        let bad2 = Rwkv5Config {
            hidden_dim: 65,
            num_heads: 4,
            head_dim: 16,
            ..tiny_config()
        };
        assert!(bad2.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // Test 10: Single step produces correct output shape
    // -----------------------------------------------------------------------
    #[test]
    fn test_rwkv5_small_forward() {
        let mut model = Rwkv5Model::new(tiny_config()).expect("model");
        let input = scirs2_core::ndarray::array![0.5_f32];
        let output = model.step(&input).expect("step");
        assert_eq!(
            output.len(),
            tiny_config().input_dim,
            "output must match input_dim"
        );
        assert!(output[0].is_finite(), "output must be finite");
    }

    // -----------------------------------------------------------------------
    // Test 11: 10 steps without NaN
    // -----------------------------------------------------------------------
    #[test]
    fn test_rwkv5_multi_step() {
        let mut model = Rwkv5Model::new(tiny_config()).expect("model");
        let input = scirs2_core::ndarray::array![0.1_f32];
        for step in 0..10 {
            let out = model.step(&input).expect("step");
            for &v in out.iter() {
                assert!(v.is_finite(), "NaN/Inf at step {step}: {v}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 12: State reset gives same first output as fresh model
    // -----------------------------------------------------------------------
    #[test]
    fn test_rwkv5_state_reset() {
        let mut model = Rwkv5Model::new(tiny_config()).expect("model");
        let input = scirs2_core::ndarray::array![0.3_f32];

        // Warm up
        for _ in 0..5 {
            let _ = model.step(&input).expect("warm-up step");
        }

        model.reset();
        let out_reset = model.step(&input).expect("step after reset");

        // Fresh model (same deterministic seed)
        let mut fresh = Rwkv5Model::new(tiny_config()).expect("fresh model");
        let out_fresh = fresh.step(&input).expect("fresh step");

        for (&a, &b) in out_reset.iter().zip(out_fresh.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "reset output should match fresh model: {a} vs {b}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 13: SignalPredictor trait (step / reset / context_window)
    // -----------------------------------------------------------------------
    #[test]
    fn test_rwkv5_signal_predictor() {
        let mut model = Rwkv5Model::new(tiny_config()).expect("model");
        let input = scirs2_core::ndarray::array![1.0_f32];

        let out = model.step(&input).expect("step");
        assert_eq!(out.len(), 1);

        model.reset();
        assert_eq!(model.context_window(), usize::MAX);
    }

    // -----------------------------------------------------------------------
    // Test 14: AutoregressiveModel – get_states / set_states roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn test_rwkv5_autoregressive() {
        let config = tiny_config();
        let mut model = Rwkv5Model::new(config.clone()).expect("model");

        let input = scirs2_core::ndarray::array![0.7_f32];
        let _ = model.step(&input).expect("step");

        let states = model.get_states();
        assert_eq!(states.len(), config.num_layers);

        let mut model2 = Rwkv5Model::new(config.clone()).expect("model2");
        model2.set_states(states.clone()).expect("set_states");

        let states2 = model2.get_states();
        assert_eq!(states.len(), states2.len());
        for (s1, s2) in states.iter().zip(states2.iter()) {
            let a = s1.state();
            let b = s2.state();
            assert_eq!(a.shape(), b.shape());
            for (&va, &vb) in a.iter().zip(b.iter()) {
                assert!((va - vb).abs() < 1e-6, "state roundtrip mismatch");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 15: Numerical stability under extreme inputs
    // -----------------------------------------------------------------------
    #[test]
    fn test_rwkv5_numerical_stability() {
        let mut model = Rwkv5Model::new(tiny_config()).expect("model");

        let large = scirs2_core::ndarray::array![1000.0_f32];
        let out = model.step(&large).expect("large input");
        for &v in out.iter() {
            assert!(v.is_finite(), "large input produced non-finite: {v}");
        }

        model.reset();
        let tiny = scirs2_core::ndarray::array![1e-10_f32];
        let out = model.step(&tiny).expect("tiny input");
        for &v in out.iter() {
            assert!(v.is_finite(), "tiny input produced non-finite: {v}");
        }

        model.reset();
        let neg = scirs2_core::ndarray::array![-500.0_f32];
        let out = model.step(&neg).expect("negative input");
        for &v in out.iter() {
            assert!(v.is_finite(), "negative input produced non-finite: {v}");
        }
    }
}
