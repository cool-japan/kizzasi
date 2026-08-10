// ---------------------------------------------------------------------------
// Channel Mixing v7
// ---------------------------------------------------------------------------

use crate::error::ModelResult;
use kizzasi_core::{sigmoid, CoreResult};
use scirs2_core::ndarray::{Array1, Array2};

use super::time_mixing::SeededRng;
use super::Rwkv7Config;

/// Channel mixing (FFN) block for RWKV v7 with expanded intermediate dim
pub(super) struct Rwkv7ChannelMixing {
    pub(super) hidden_dim: usize,
    pub(super) intermediate_dim: usize,

    pub(super) time_mix_k: Array1<f32>,
    pub(super) time_mix_r: Array1<f32>,

    pub(super) key_proj: Array2<f32>, // (hidden_dim, intermediate_dim)
    pub(super) value_proj: Array2<f32>, // (intermediate_dim, hidden_dim)
    pub(super) receptance_proj: Array2<f32>, // (hidden_dim, hidden_dim)

    prev_x: Array1<f32>,
}

impl Rwkv7ChannelMixing {
    pub(super) fn new(config: &Rwkv7Config) -> ModelResult<Self> {
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

    pub(super) fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
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

    pub(super) fn reset(&mut self) {
        self.prev_x.fill(0.0);
    }
}
