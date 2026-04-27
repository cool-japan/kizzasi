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

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{sigmoid, silu, CoreResult, HiddenState, LayerNorm, NormType, SignalPredictor};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

// ---------------------------------------------------------------------------
// Seeded deterministic RNG for reproducible weight initialization
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG for deterministic weight initialization.
/// This avoids platform-dependent randomness in tests and benchmarks.
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
        // Map u64 to [-1, 1)
        (self.state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
    }
}

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
// Time Mixing v7
// ---------------------------------------------------------------------------

/// RWKV v7 time-mixing block with data-dependent decay, value gate, and bonus attention
pub struct Rwkv7TimeMixing {
    // Projection weights
    w_r: Array2<f32>, // receptance
    w_w: Array2<f32>, // decay input projection (data-dependent decay)
    w_k: Array2<f32>, // key
    w_v: Array2<f32>, // value
    w_o: Array2<f32>, // output
    w_g: Array2<f32>, // value gate
    w_a: Array2<f32>, // bonus/attention gate
    w_b: Array2<f32>, // decay gate

    // Learned interpolation coefficients for token shift
    lerp_r: Array1<f32>,
    lerp_w: Array1<f32>,
    lerp_k: Array1<f32>,
    lerp_v: Array1<f32>,

    // Group normalization applied to concatenated head outputs
    ln_x: LayerNorm,

    num_heads: usize,
    head_dim: usize,
}

impl Rwkv7TimeMixing {
    /// Create a new time-mixing block
    pub fn new(config: &Rwkv7Config) -> ModelResult<Self> {
        let d = config.hidden_dim;
        let mut rng = SeededRng::new(42 + d as u64);
        let scale = (2.0 / d as f32).sqrt();

        let make_proj = |rng: &mut SeededRng| -> Array2<f32> {
            Array2::from_shape_fn((d, d), |_| rng.next_f32() * scale)
        };

        let w_r = make_proj(&mut rng);
        let w_w = make_proj(&mut rng);
        let w_k = make_proj(&mut rng);
        let w_v = make_proj(&mut rng);
        let w_o = make_proj(&mut rng);
        let w_g = make_proj(&mut rng);
        let w_a = make_proj(&mut rng);
        let w_b = make_proj(&mut rng);

        let lerp_r = Array1::from_shape_fn(d, |_| rng.next_f32().abs() * 0.5 + 0.25);
        let lerp_w = Array1::from_shape_fn(d, |_| rng.next_f32().abs() * 0.5 + 0.25);
        let lerp_k = Array1::from_shape_fn(d, |_| rng.next_f32().abs() * 0.5 + 0.25);
        let lerp_v = Array1::from_shape_fn(d, |_| rng.next_f32().abs() * 0.5 + 0.25);

        let ln_x = LayerNorm::new(d, NormType::RMSNorm).with_eps(1e-5);

        Ok(Self {
            w_r,
            w_w,
            w_k,
            w_v,
            w_o,
            w_g,
            w_a,
            w_b,
            lerp_r,
            lerp_w,
            lerp_k,
            lerp_v,
            ln_x,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
        })
    }

    /// Single-step forward pass for layer `layer_idx`.
    ///
    /// Reads and mutates the corresponding layer in `state`.
    pub fn forward(
        &self,
        x: &Array1<f32>,
        state: &mut Rwkv7State,
        layer_idx: usize,
    ) -> ModelResult<Array1<f32>> {
        let d = x.len();

        // 1. Token shift
        let prev = &state.shift_states[layer_idx];
        let dx = x - prev;
        state.shift_states[layer_idx] = x.clone();

        // 2. Mixed inputs for each projection path
        let xr = x + &(&self.lerp_r * &dx);
        let xw = x + &(&self.lerp_w * &dx);
        let xk = x + &(&self.lerp_k * &dx);
        let xv = x + &(&self.lerp_v * &dx);

        // 3. Linear projections
        let r_raw = self.matvec(&self.w_r, &xr);
        let w_raw = self.matvec(&self.w_w, &xw);
        let k_raw = self.matvec(&self.w_k, &xk);
        let v_raw = self.matvec(&self.w_v, &xv);

        // 4. Activations
        let r = sigmoid(&r_raw); // receptance
        let w = sigmoid(&w_raw); // data-dependent decay (v7)
        let g = silu(&self.matvec(&self.w_g, x)); // value gate (v7)
        let a = sigmoid(&self.matvec(&self.w_a, x)); // bonus gate (v7)
        let b = sigmoid(&self.matvec(&self.w_b, x)); // decay gate (v7)

        // 5. Per-head WKV computation
        let mut output_heads = Array1::zeros(d);

        for h in 0..self.num_heads {
            let lo = h * self.head_dim;
            let hi = lo + self.head_dim;

            // Extract per-head slices
            let r_h = r.slice(scirs2_core::ndarray::s![lo..hi]).to_owned();
            let k_h = k_raw.slice(scirs2_core::ndarray::s![lo..hi]).to_owned();
            let v_h = v_raw.slice(scirs2_core::ndarray::s![lo..hi]).to_owned();
            let w_h = w.slice(scirs2_core::ndarray::s![lo..hi]).to_owned();
            let a_h = a.slice(scirs2_core::ndarray::s![lo..hi]).to_owned();
            let b_h = b.slice(scirs2_core::ndarray::s![lo..hi]).to_owned();

            let head_state = &mut state.wkv_states[layer_idx][h];

            // state_h = diag(w_h) @ state_h  (data-dependent decay)
            // Then add rank-1 update: + outer(k_h, v_h)
            for i in 0..self.head_dim {
                let decay = w_h[i].clamp(0.0, 1.0);
                for j in 0..self.head_dim {
                    head_state[[i, j]] = decay * head_state[[i, j]] + k_h[i] * v_h[j];
                }
            }

            // output_h = r_h * (state_h @ b_h + a_h * v_h)
            // The bonus attention term `a_h * v_h` provides direct value bypass
            let state_b = self.matvec_small(head_state, &b_h);
            for i in 0..self.head_dim {
                let val = r_h[i] * (state_b[i] + a_h[i] * v_h[i]);
                output_heads[lo + i] = val;
            }
        }

        // 6. Apply group normalization then value gate
        let normed = self.ln_x.forward(&output_heads);
        let gated = &g * &normed;

        // 7. Output projection
        let out = self.matvec(&self.w_o, &gated);
        Ok(out)
    }

    // Matrix-vector multiply: y = W @ x
    fn matvec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
        let rows = w.shape()[0];
        let cols = w.shape()[1];
        let xlen = x.len();
        let mut out = Array1::zeros(rows);
        for i in 0..rows {
            let mut sum = 0.0f32;
            for j in 0..cols.min(xlen) {
                sum += w[[i, j]] * x[j];
            }
            out[i] = sum;
        }
        out
    }

    fn matvec_small(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
        let rows = w.shape()[0];
        let cols = w.shape()[1];
        let xlen = x.len();
        let mut out = Array1::zeros(rows);
        for i in 0..rows {
            let mut sum = 0.0f32;
            for j in 0..cols.min(xlen) {
                sum += w[[i, j]] * x[j];
            }
            out[i] = sum;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Channel Mixing v7
// ---------------------------------------------------------------------------

/// Channel mixing (FFN) block for RWKV v7 with expanded intermediate dim
struct Rwkv7ChannelMixing {
    hidden_dim: usize,
    intermediate_dim: usize,

    time_mix_k: Array1<f32>,
    time_mix_r: Array1<f32>,

    key_proj: Array2<f32>,        // (hidden_dim, intermediate_dim)
    value_proj: Array2<f32>,      // (intermediate_dim, hidden_dim)
    receptance_proj: Array2<f32>, // (hidden_dim, hidden_dim)

    prev_x: Array1<f32>,
}

impl Rwkv7ChannelMixing {
    fn new(config: &Rwkv7Config) -> ModelResult<Self> {
        let d = config.hidden_dim;
        let inter = (d as f32 * config.expand_factor) as usize;
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

        // Time-mixed inputs
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

        // Key path: project up, squared ReLU, project back down
        let k = self.project_up(&xk);
        let k_act = k.mapv(|v| {
            let relu = v.max(0.0);
            relu * relu
        });
        let vk = self.project_down(&k_act);

        // Receptance gating
        let r = self.project_r(&xr);
        let r_sig = sigmoid(&r);

        let mut output = Array1::zeros(d);
        for i in 0..d.min(vk.len()).min(r_sig.len()) {
            output[i] = r_sig[i] * vk[i];
        }

        self.prev_x = x.slice(scirs2_core::ndarray::s![..d]).to_owned();
        Ok(output)
    }

    fn project_up(&self, x: &Array1<f32>) -> Array1<f32> {
        let out_dim = self.intermediate_dim;
        let mut output = Array1::zeros(out_dim);
        for i in 0..out_dim {
            let mut sum = 0.0f32;
            for j in 0..x.len().min(self.key_proj.shape()[0]) {
                sum += self.key_proj[[j, i]] * x[j];
            }
            output[i] = sum;
        }
        output
    }

    fn project_down(&self, x: &Array1<f32>) -> Array1<f32> {
        let out_dim = self.hidden_dim;
        let mut output = Array1::zeros(out_dim);
        for i in 0..out_dim {
            let mut sum = 0.0f32;
            for j in 0..x.len().min(self.value_proj.shape()[0]) {
                sum += self.value_proj[[j, i]] * x[j];
            }
            output[i] = sum;
        }
        output
    }

    fn project_r(&self, x: &Array1<f32>) -> Array1<f32> {
        let out_dim = self.receptance_proj.shape()[0];
        let mut output = Array1::zeros(out_dim.min(x.len()));
        for i in 0..output.len() {
            let mut sum = 0.0f32;
            for j in 0..x.len().min(self.receptance_proj.shape()[1]) {
                sum += self.receptance_proj[[i, j]] * x[j];
            }
            output[i] = sum;
        }
        output
    }

    fn reset(&mut self) {
        self.prev_x.fill(0.0);
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
    input_proj: Array2<f32>,
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
}
