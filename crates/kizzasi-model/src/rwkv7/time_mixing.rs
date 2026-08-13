// ---------------------------------------------------------------------------
// Seeded deterministic RNG for reproducible weight initialization
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG for deterministic weight initialization.
/// This avoids platform-dependent randomness in tests and benchmarks.
pub(super) struct SeededRng {
    state: u64,
}

impl SeededRng {
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    /// Returns a float in [-1, 1)
    pub(super) fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        // Map u64 to [-1, 1)
        (self.state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
    }
}

// ---------------------------------------------------------------------------
// Time Mixing v7
// ---------------------------------------------------------------------------

use crate::error::ModelResult;
use kizzasi_core::{sigmoid, silu, LayerNorm, NormType};
use scirs2_core::ndarray::{Array1, Array2};

use super::Rwkv7Config;
use super::Rwkv7State;

/// RWKV v7 time-mixing block with data-dependent decay, value gate, and bonus attention
pub struct Rwkv7TimeMixing {
    // Projection weights
    pub(super) w_r: Array2<f32>, // receptance
    pub(super) w_w: Array2<f32>, // decay input projection (data-dependent decay)
    pub(super) w_k: Array2<f32>, // key
    pub(super) w_v: Array2<f32>, // value
    pub(super) w_o: Array2<f32>, // output
    pub(super) w_g: Array2<f32>, // value gate
    pub(super) w_a: Array2<f32>, // bonus/attention gate
    pub(super) w_b: Array2<f32>, // decay gate

    // Learned interpolation coefficients for token shift
    pub(super) lerp_r: Array1<f32>,
    pub(super) lerp_w: Array1<f32>,
    pub(super) lerp_k: Array1<f32>,
    pub(super) lerp_v: Array1<f32>,

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
            let state_b = self.matvec(head_state, &b_h);
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
    //
    // `w.dot(x)` dispatches to ndarray/matrixmultiply with the correct
    // (row-major) traversal and no per-element `Index` bounds-check
    // overhead, instead of walking `w[[i, j]]` through ndarray's checked 2D
    // indexing one scalar at a time. It requires `x.len() == w.ncols()`
    // exactly (ndarray panics otherwise), which always holds in this
    // module's call sites (every input here is sized to `hidden_dim`, and
    // every weight matrix is `(hidden_dim, hidden_dim)`) — the original
    // clamped loop is kept as a fallback for any case where that invariant
    // doesn't hold, so a shape mismatch degrades to the old (slower, silently
    // truncated) behavior instead of panicking.
    fn matvec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
        if x.len() == w.shape()[1] {
            return w.dot(x);
        }
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
