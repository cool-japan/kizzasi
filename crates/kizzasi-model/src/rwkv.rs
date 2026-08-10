//! RWKV v6: Receptance Weighted Key Value
//!
//! RWKV is a novel RNN architecture that combines the efficient parallelizable training
//! of Transformers with the efficient inference of RNNs. Unlike attention, RWKV uses
//! linear attention with time-mixing, achieving O(1) per-step inference complexity.
//!
//! # RWKV v6 Features
//!
//! - **Time-Mixing**: Linear attention with exponential decay
//! - **Channel-Mixing**: Token-shift with gated linear units
//! - **Efficient Training**: Parallelizable via WKV algorithm
//! - **O(1) Inference**: Constant memory and time per step
//! - **No Positional Encoding**: Time awareness through mixing
//!
//! # Architecture
//!
//! ```text
//! Input → [LayerNorm] → [Time-Mixing] → [Add] →
//!           ↓                                   ↓
//!        [LayerNorm] → [Channel-Mixing] → [Add] → Output
//! ```
//!
//! # WKV Attention Formula
//!
//! The core WKV (Weighted Key-Value) computation for RWKV v6:
//!
//! ```text
//! wkv_t = (∑_{i=1}^{t-1} e^{-(t-1-i)·w + k_i} · v_i + e^{u+k_t} · v_t)
//!       / (∑_{i=1}^{t-1} e^{-(t-1-i)·w + k_i}     + e^{u+k_t})
//! ```
//!
//! where:
//! - `w` is the learned time decay (per-channel)
//! - `u` is the learned bonus term for current token
//! - `k_i`, `v_i` are key and value at position i
//!
//! ## Efficient Recurrence
//!
//! The WKV sum is maintained as running state:
//!
//! ```text
//! num_t = e^{-w} · num_{t-1} + e^{k_t} · v_t
//! den_t = e^{-w} · den_{t-1} + e^{k_t}
//! wkv_t = (num_t + e^{u+k_t} · v_t) / (den_t + e^{u+k_t})
//! ```
//!
//! This gives O(1) per-step inference with constant memory.
//!
//! # References
//!
//! - RWKV paper: <https://arxiv.org/abs/2305.13048>
//! - RWKV v6 improvements: Enhanced stability and performance

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{sigmoid, silu, CoreResult, HiddenState, LayerNorm, NormType, SignalPredictor};
use safetensors::tensor::{Dtype, TensorView};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, RngExt};
#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

/// Configuration for RWKV v6
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RwkvConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension (d_model)
    pub hidden_dim: usize,
    /// Intermediate dimension for FFN (typically 4x hidden_dim)
    pub intermediate_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads (v6 uses multi-head)
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Time decay initialization
    pub time_decay_init: f32,
    /// Use RMSNorm instead of LayerNorm
    pub use_rms_norm: bool,
}

impl Default for RwkvConfig {
    fn default() -> Self {
        let hidden_dim = 512;
        let num_heads = 8;
        Self {
            input_dim: 1,
            hidden_dim,
            intermediate_dim: hidden_dim * 4,
            num_layers: 12,
            num_heads,
            head_dim: hidden_dim / num_heads,
            dropout: 0.0,
            time_decay_init: -5.0,
            use_rms_norm: true,
        }
    }
}

impl RwkvConfig {
    /// Create a new RWKV configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set input dimension
    pub fn input_dim(mut self, dim: usize) -> Self {
        self.input_dim = dim;
        self
    }

    /// Set hidden dimension
    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self.head_dim = dim / self.num_heads;
        self
    }

    /// Set intermediate dimension
    pub fn intermediate_dim(mut self, dim: usize) -> Self {
        self.intermediate_dim = dim;
        self
    }

    /// Set number of layers
    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Set number of heads
    pub fn num_heads(mut self, n: usize) -> Self {
        self.num_heads = n;
        self.head_dim = self.hidden_dim / n;
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
        Ok(())
    }
}

/// RWKV Time-Mixing block
///
/// Implements linear attention with time decay:
/// wkv[t] = (w * wkv[t-1] + k[t] * v[t]) / (w * aa[t-1] + k[t])
struct TimeMixing {
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,

    /// Time-mixing parameters
    time_mix_k: Array1<f32>,
    #[allow(dead_code)]
    time_mix_v: Array1<f32>, // Reserved for future use
    time_mix_r: Array1<f32>,
    time_mix_g: Array1<f32>,

    /// Time decay (per head)
    time_decay: Array2<f32>, // [num_heads, head_dim]

    /// Projection matrices
    key_proj: Array2<f32>,
    value_proj: Array2<f32>,
    receptance_proj: Array2<f32>,
    gate_proj: Array2<f32>,
    output_proj: Array2<f32>,

    /// State: WKV accumulator and normalizer per head
    wkv_state: Vec<Array1<f32>>, // [num_heads][head_dim]
    wkv_norm: Vec<f32>, // [num_heads]
    prev_x: Array1<f32>,
}

impl TimeMixing {
    fn new(config: &RwkvConfig) -> ModelResult<Self> {
        let mut rng = rng();

        // Initialize time-mixing parameters (learnable interpolation)
        let time_mix_k = Array1::from_shape_fn(config.hidden_dim, |_| rng.random::<f32>());
        let time_mix_v = Array1::from_shape_fn(config.hidden_dim, |_| rng.random::<f32>());
        let time_mix_r = Array1::from_shape_fn(config.hidden_dim, |_| rng.random::<f32>());
        let time_mix_g = Array1::from_shape_fn(config.hidden_dim, |_| rng.random::<f32>());

        // Initialize time decay (log scale, negative for decay)
        let time_decay = Array2::from_shape_fn((config.num_heads, config.head_dim), |(h, i)| {
            // Different decay rates per head and dimension
            config.time_decay_init - (h as f32 * 0.1) - (i as f32 * 0.01)
        });

        // Initialize projection matrices
        let scale = (2.0 / config.hidden_dim as f32).sqrt();
        let key_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let value_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let receptance_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let gate_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let output_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Initialize states
        let wkv_state = (0..config.num_heads)
            .map(|_| Array1::zeros(config.head_dim))
            .collect();
        let wkv_norm = vec![0.0; config.num_heads];
        let prev_x = Array1::zeros(config.hidden_dim);

        Ok(Self {
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            time_mix_k,
            time_mix_v,
            time_mix_r,
            time_mix_g,
            time_decay,
            key_proj,
            value_proj,
            receptance_proj,
            gate_proj,
            output_proj,
            wkv_state,
            wkv_norm,
            prev_x,
        })
    }

    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let batch_size = x.len().min(self.hidden_dim);

        // Time-mixing: interpolate between current and previous input
        let mut xx = Array1::zeros(batch_size);
        for i in 0..batch_size {
            let prev_val = if i < self.prev_x.len() {
                self.prev_x[i]
            } else {
                0.0
            };
            xx[i] = self.time_mix_k[i] * x[i] + (1.0 - self.time_mix_k[i]) * prev_val;
        }

        // Compute K, V, R, G projections
        let k = self.project(&xx, &self.key_proj);
        let v = self.project(&xx, &self.value_proj);

        let mut xr = Array1::zeros(batch_size);
        for i in 0..batch_size {
            let prev_val = if i < self.prev_x.len() {
                self.prev_x[i]
            } else {
                0.0
            };
            xr[i] = self.time_mix_r[i] * x[i] + (1.0 - self.time_mix_r[i]) * prev_val;
        }
        let r = self.project(&xr, &self.receptance_proj);

        let mut xg = Array1::zeros(batch_size);
        for i in 0..batch_size {
            let prev_val = if i < self.prev_x.len() {
                self.prev_x[i]
            } else {
                0.0
            };
            xg[i] = self.time_mix_g[i] * x[i] + (1.0 - self.time_mix_g[i]) * prev_val;
        }
        let g = self.project(&xg, &self.gate_proj);

        // WKV: Weighted Key-Value with time decay (per head)
        let mut wkv_output = Array1::zeros(batch_size);

        for head in 0..self.num_heads {
            let head_start = head * self.head_dim;
            let head_end = (head_start + self.head_dim).min(batch_size);

            for i in 0..(head_end - head_start) {
                let idx = head_start + i;
                if idx >= k.len() || idx >= v.len() {
                    break;
                }

                // Get time decay for this head and dimension
                let w = self.time_decay[[head, i]].exp();

                // Update WKV state: num_t = e^{-w}·num_{t-1} + e^{k_t}·v_t
                let exp_k = k[idx].exp();
                let new_wkv = w * self.wkv_state[head][i] + exp_k * v[idx];
                self.wkv_state[head][i] = new_wkv;

                // Update normalizer: den_t = e^{-w}·den_{t-1} + e^{k_t}
                self.wkv_norm[head] = w * self.wkv_norm[head] + exp_k;

                // Output: wkv / norm
                let norm = self.wkv_norm[head].max(1e-8);
                wkv_output[idx] = new_wkv / norm;
            }
        }

        // Apply receptance (gating)
        let r_sigmoid = sigmoid(&r);
        for i in 0..wkv_output.len().min(r_sigmoid.len()) {
            wkv_output[i] *= r_sigmoid[i];
        }

        // Apply group normalization (v6 feature)
        let g_silu = silu(&g);
        for i in 0..wkv_output.len().min(g_silu.len()) {
            wkv_output[i] *= g_silu[i];
        }

        // Output projection
        let output = self.project(&wkv_output, &self.output_proj);

        // Update previous input
        self.prev_x = Array1::from_vec(x.iter().take(self.hidden_dim).copied().collect());

        Ok(output)
    }

    fn project(&self, x: &Array1<f32>, weight: &Array2<f32>) -> Array1<f32> {
        let out_dim = weight.shape()[0];
        let mut output = Array1::zeros(out_dim.min(x.len()));
        for i in 0..output.len() {
            let mut sum = 0.0;
            for j in 0..x.len().min(weight.shape()[1]) {
                sum += weight[[i, j]] * x[j];
            }
            output[i] = sum;
        }
        output
    }

    fn reset(&mut self) {
        for state in &mut self.wkv_state {
            state.fill(0.0);
        }
        self.wkv_norm.fill(0.0);
        self.prev_x.fill(0.0);
    }
}

/// RWKV Channel-Mixing block
///
/// Token-shifted feed-forward network with gated linear units
struct ChannelMixing {
    hidden_dim: usize,
    intermediate_dim: usize,

    /// Time-mixing parameter for channel mixing
    time_mix_k: Array1<f32>,
    time_mix_r: Array1<f32>,

    /// Projection matrices
    key_proj: Array2<f32>,
    value_proj: Array2<f32>,
    receptance_proj: Array2<f32>,

    /// Previous input
    prev_x: Array1<f32>,
}

impl ChannelMixing {
    fn new(config: &RwkvConfig) -> ModelResult<Self> {
        let mut rng = rng();

        // Initialize time-mixing parameters
        let time_mix_k = Array1::from_shape_fn(config.hidden_dim, |_| rng.random::<f32>());
        let time_mix_r = Array1::from_shape_fn(config.hidden_dim, |_| rng.random::<f32>());

        // Initialize projection matrices
        let scale = (2.0 / config.hidden_dim as f32).sqrt();
        let key_proj = Array2::from_shape_fn((config.hidden_dim, config.intermediate_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let value_proj =
            Array2::from_shape_fn((config.intermediate_dim, config.hidden_dim), |_| {
                (rng.random::<f32>() - 0.5) * 2.0 * scale
            });

        let receptance_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let prev_x = Array1::zeros(config.hidden_dim);

        Ok(Self {
            hidden_dim: config.hidden_dim,
            intermediate_dim: config.intermediate_dim,
            time_mix_k,
            time_mix_r,
            key_proj,
            value_proj,
            receptance_proj,
            prev_x,
        })
    }

    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let batch_size = x.len().min(self.hidden_dim);

        // Time-mixing for key
        let mut xk = Array1::zeros(batch_size);
        for i in 0..batch_size {
            let prev_val = if i < self.prev_x.len() {
                self.prev_x[i]
            } else {
                0.0
            };
            xk[i] = self.time_mix_k[i] * x[i] + (1.0 - self.time_mix_k[i]) * prev_val;
        }

        // Time-mixing for receptance
        let mut xr = Array1::zeros(batch_size);
        for i in 0..batch_size {
            let prev_val = if i < self.prev_x.len() {
                self.prev_x[i]
            } else {
                0.0
            };
            xr[i] = self.time_mix_r[i] * x[i] + (1.0 - self.time_mix_r[i]) * prev_val;
        }

        // Project and apply activation
        let k = self.project(&xk, &self.key_proj);
        let k_squared = k.mapv(|v| v * v); // Squared ReLU
        let vk = self.project_back(&k_squared, &self.value_proj);

        // Apply receptance gating
        let r = self.project_r(&xr, &self.receptance_proj);
        let r_sigmoid = sigmoid(&r);

        let mut output = Array1::zeros(batch_size);
        for i in 0..output.len().min(vk.len()).min(r_sigmoid.len()) {
            output[i] = r_sigmoid[i] * vk[i];
        }

        // Update previous input
        self.prev_x = Array1::from_vec(x.iter().take(self.hidden_dim).copied().collect());

        Ok(output)
    }

    fn project(&self, x: &Array1<f32>, weight: &Array2<f32>) -> Array1<f32> {
        let out_dim = weight.shape()[1].min(self.intermediate_dim);
        let mut output = Array1::zeros(out_dim);
        for i in 0..out_dim {
            let mut sum = 0.0;
            for j in 0..x.len().min(weight.shape()[0]) {
                sum += weight[[j, i]] * x[j];
            }
            output[i] = sum;
        }
        output
    }

    fn project_back(&self, x: &Array1<f32>, weight: &Array2<f32>) -> Array1<f32> {
        let out_dim = weight.shape()[1].min(self.hidden_dim);
        let mut output = Array1::zeros(out_dim);
        for i in 0..out_dim {
            let mut sum = 0.0;
            for j in 0..x.len().min(weight.shape()[0]) {
                sum += weight[[j, i]] * x[j];
            }
            output[i] = sum;
        }
        output
    }

    fn project_r(&self, x: &Array1<f32>, weight: &Array2<f32>) -> Array1<f32> {
        let out_dim = weight.shape()[0];
        let mut output = Array1::zeros(out_dim.min(x.len()));
        for i in 0..output.len() {
            let mut sum = 0.0;
            for j in 0..x.len().min(weight.shape()[1]) {
                sum += weight[[i, j]] * x[j];
            }
            output[i] = sum;
        }
        output
    }

    fn reset(&mut self) {
        self.prev_x.fill(0.0);
    }
}

/// RWKV Layer combining time-mixing and channel-mixing
struct RwkvLayer {
    ln1: LayerNorm,
    ln2: LayerNorm,
    time_mixing: TimeMixing,
    channel_mixing: ChannelMixing,
}

impl RwkvLayer {
    fn new(config: &RwkvConfig) -> ModelResult<Self> {
        let norm_type = if config.use_rms_norm {
            NormType::RMSNorm
        } else {
            NormType::LayerNorm
        };

        let ln1 = LayerNorm::new(config.hidden_dim, norm_type).with_eps(1e-5);
        let ln2 = LayerNorm::new(config.hidden_dim, norm_type).with_eps(1e-5);
        let time_mixing = TimeMixing::new(config)?;
        let channel_mixing = ChannelMixing::new(config)?;

        Ok(Self {
            ln1,
            ln2,
            time_mixing,
            channel_mixing,
        })
    }

    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Time-mixing with residual
        let x_norm = self.ln1.forward(x);
        let tm_out = self.time_mixing.forward(&x_norm)?;
        let mut x_tm = x.clone();
        for i in 0..x_tm.len().min(tm_out.len()) {
            x_tm[i] += tm_out[i];
        }

        // Channel-mixing with residual
        let x_norm2 = self.ln2.forward(&x_tm);
        let cm_out = self.channel_mixing.forward(&x_norm2)?;
        let mut output = x_tm;
        for i in 0..output.len().min(cm_out.len()) {
            output[i] += cm_out[i];
        }

        Ok(output)
    }

    fn reset(&mut self) {
        self.time_mixing.reset();
        self.channel_mixing.reset();
    }
}

/// RWKV v6 model
pub struct Rwkv {
    config: RwkvConfig,
    layers: Vec<RwkvLayer>,
    ln_out: LayerNorm,
    input_proj: Array2<f32>,
    output_proj: Array2<f32>,
}

impl Rwkv {
    /// Create a new RWKV model
    pub fn new(config: RwkvConfig) -> ModelResult<Self> {
        config.validate()?;

        // Initialize layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(RwkvLayer::new(&config)?);
        }

        // Output layer normalization
        let norm_type = if config.use_rms_norm {
            NormType::RMSNorm
        } else {
            NormType::LayerNorm
        };
        let ln_out = LayerNorm::new(config.hidden_dim, norm_type).with_eps(1e-5);

        // Initialize input/output projections
        let mut rng = rng();
        let scale = (2.0 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let input_proj = Array2::from_shape_fn((config.input_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let scale = (2.0 / (config.hidden_dim + config.input_dim) as f32).sqrt();
        let output_proj = Array2::from_shape_fn((config.hidden_dim, config.input_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        Ok(Self {
            config,
            layers,
            ln_out,
            input_proj,
            output_proj,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &RwkvConfig {
        &self.config
    }

    /// Load weights from a SafeTensors model file
    ///
    /// # Weight Naming Convention
    ///
    /// The following tensor names are expected:
    /// - `input_proj`: Input projection matrix (input_dim, hidden_dim)
    /// - `output_proj`: Output projection matrix (hidden_dim, input_dim)
    /// - `ln_out.weight`: Output layer norm weight
    /// - `ln_out.bias`: Output layer norm bias (optional)
    ///
    /// For each layer i:
    /// - `layers.{i}.ln1.weight`: Time-mixing layer norm weight
    /// - `layers.{i}.ln1.bias`: Time-mixing layer norm bias (optional)
    /// - `layers.{i}.ln2.weight`: Channel-mixing layer norm weight
    /// - `layers.{i}.ln2.bias`: Channel-mixing layer norm bias (optional)
    ///
    /// Time-mixing parameters:
    /// - `layers.{i}.time_mixing.time_mix_k`: Time mixing for key
    /// - `layers.{i}.time_mixing.time_mix_v`: Time mixing for value
    /// - `layers.{i}.time_mixing.time_mix_r`: Time mixing for receptance
    /// - `layers.{i}.time_mixing.time_mix_g`: Time mixing for gate
    /// - `layers.{i}.time_mixing.time_decay`: Time decay matrix
    /// - `layers.{i}.time_mixing.key_proj`: Key projection
    /// - `layers.{i}.time_mixing.value_proj`: Value projection
    /// - `layers.{i}.time_mixing.receptance_proj`: Receptance projection
    /// - `layers.{i}.time_mixing.gate_proj`: Gate projection
    /// - `layers.{i}.time_mixing.output_proj`: Output projection
    ///
    /// Channel-mixing parameters:
    /// - `layers.{i}.channel_mixing.time_mix_k`: Time mixing for key
    /// - `layers.{i}.channel_mixing.time_mix_r`: Time mixing for receptance
    /// - `layers.{i}.channel_mixing.key_proj`: Key projection
    /// - `layers.{i}.channel_mixing.value_proj`: Value projection
    /// - `layers.{i}.channel_mixing.receptance_proj`: Receptance projection
    pub fn load_weights(&mut self, loader: &crate::loader::ModelLoader) -> ModelResult<()> {
        // Load input/output projections
        if loader.has_tensor("input_proj") {
            self.input_proj = loader.load_array2("input_proj")?;
        }
        if loader.has_tensor("output_proj") {
            self.output_proj = loader.load_array2("output_proj")?;
        }

        // Load output layer norm
        if loader.has_tensor("ln_out.weight") {
            let weight = loader.load_array1("ln_out.weight")?;
            self.ln_out.set_gamma(weight);
        }
        if loader.has_tensor("ln_out.bias") {
            let bias = loader.load_array1("ln_out.bias")?;
            self.ln_out.set_beta(bias);
        }

        // Load each layer's weights
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            // Load layer norm 1
            if loader.has_tensor(&format!("{}.ln1.weight", prefix)) {
                let weight = loader.load_array1(&format!("{}.ln1.weight", prefix))?;
                layer.ln1.set_gamma(weight);
            }
            if loader.has_tensor(&format!("{}.ln1.bias", prefix)) {
                let bias = loader.load_array1(&format!("{}.ln1.bias", prefix))?;
                layer.ln1.set_beta(bias);
            }

            // Load layer norm 2
            if loader.has_tensor(&format!("{}.ln2.weight", prefix)) {
                let weight = loader.load_array1(&format!("{}.ln2.weight", prefix))?;
                layer.ln2.set_gamma(weight);
            }
            if loader.has_tensor(&format!("{}.ln2.bias", prefix)) {
                let bias = loader.load_array1(&format!("{}.ln2.bias", prefix))?;
                layer.ln2.set_beta(bias);
            }

            // Load time-mixing parameters
            let tm_prefix = format!("{}.time_mixing", prefix);
            if loader.has_tensor(&format!("{}.time_mix_k", tm_prefix)) {
                layer.time_mixing.time_mix_k =
                    loader.load_array1(&format!("{}.time_mix_k", tm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.time_mix_v", tm_prefix)) {
                layer.time_mixing.time_mix_v =
                    loader.load_array1(&format!("{}.time_mix_v", tm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.time_mix_r", tm_prefix)) {
                layer.time_mixing.time_mix_r =
                    loader.load_array1(&format!("{}.time_mix_r", tm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.time_mix_g", tm_prefix)) {
                layer.time_mixing.time_mix_g =
                    loader.load_array1(&format!("{}.time_mix_g", tm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.time_decay", tm_prefix)) {
                layer.time_mixing.time_decay =
                    loader.load_array2(&format!("{}.time_decay", tm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.key_proj", tm_prefix)) {
                layer.time_mixing.key_proj =
                    loader.load_array2(&format!("{}.key_proj", tm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.value_proj", tm_prefix)) {
                layer.time_mixing.value_proj =
                    loader.load_array2(&format!("{}.value_proj", tm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.receptance_proj", tm_prefix)) {
                layer.time_mixing.receptance_proj =
                    loader.load_array2(&format!("{}.receptance_proj", tm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.gate_proj", tm_prefix)) {
                layer.time_mixing.gate_proj =
                    loader.load_array2(&format!("{}.gate_proj", tm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.output_proj", tm_prefix)) {
                layer.time_mixing.output_proj =
                    loader.load_array2(&format!("{}.output_proj", tm_prefix))?;
            }

            // Load channel-mixing parameters
            let cm_prefix = format!("{}.channel_mixing", prefix);
            if loader.has_tensor(&format!("{}.time_mix_k", cm_prefix)) {
                layer.channel_mixing.time_mix_k =
                    loader.load_array1(&format!("{}.time_mix_k", cm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.time_mix_r", cm_prefix)) {
                layer.channel_mixing.time_mix_r =
                    loader.load_array1(&format!("{}.time_mix_r", cm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.key_proj", cm_prefix)) {
                layer.channel_mixing.key_proj =
                    loader.load_array2(&format!("{}.key_proj", cm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.value_proj", cm_prefix)) {
                layer.channel_mixing.value_proj =
                    loader.load_array2(&format!("{}.value_proj", cm_prefix))?;
            }
            if loader.has_tensor(&format!("{}.receptance_proj", cm_prefix)) {
                layer.channel_mixing.receptance_proj =
                    loader.load_array2(&format!("{}.receptance_proj", cm_prefix))?;
            }
        }

        Ok(())
    }

    /// Save model weights to a JSON file as `HashMap<String, Vec<f32>>`.
    ///
    /// Keys:
    /// - `input_proj` / `output_proj`: top-level projections
    /// - Per-layer time-mixing and channel-mixing parameters
    pub fn save_weights_json<P: AsRef<std::path::Path>>(&self, path: P) -> ModelResult<()> {
        let mut weights: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::new();

        weights.insert(
            "input_proj".to_string(),
            self.input_proj.iter().copied().collect(),
        );
        weights.insert(
            "output_proj".to_string(),
            self.output_proj.iter().copied().collect(),
        );

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);
            let tm = format!("{}.time_mixing", prefix);
            let cm = format!("{}.channel_mixing", prefix);

            // Time-mixing parameters
            weights.insert(
                format!("{}.time_mix_k", tm),
                layer.time_mixing.time_mix_k.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.time_mix_v", tm),
                layer.time_mixing.time_mix_v.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.time_mix_r", tm),
                layer.time_mixing.time_mix_r.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.time_mix_g", tm),
                layer.time_mixing.time_mix_g.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.time_decay", tm),
                layer.time_mixing.time_decay.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.key_proj", tm),
                layer.time_mixing.key_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.value_proj", tm),
                layer.time_mixing.value_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.receptance_proj", tm),
                layer.time_mixing.receptance_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.gate_proj", tm),
                layer.time_mixing.gate_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.output_proj", tm),
                layer.time_mixing.output_proj.iter().copied().collect(),
            );

            // Channel-mixing parameters
            weights.insert(
                format!("{}.time_mix_k", cm),
                layer.channel_mixing.time_mix_k.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.time_mix_r", cm),
                layer.channel_mixing.time_mix_r.iter().copied().collect(),
            );
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
            ModelError::load_error("rwkv save_weights", format!("failed to create file: {e}"))
        })?;
        serde_json::to_writer(file, &weights).map_err(|e| {
            ModelError::load_error(
                "rwkv save_weights",
                format!("JSON serialization failed: {e}"),
            )
        })?;
        Ok(())
    }

    /// Load weights from a JSON file previously written by `save_weights_json`.
    pub fn load_weights_json<P: AsRef<std::path::Path>>(&mut self, path: P) -> ModelResult<()> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            ModelError::load_error("rwkv load_weights", format!("failed to open file: {e}"))
        })?;
        let weights: std::collections::HashMap<String, Vec<f32>> = serde_json::from_reader(file)
            .map_err(|e| {
                ModelError::load_error(
                    "rwkv load_weights",
                    format!("JSON deserialization failed: {e}"),
                )
            })?;

        let load_array2 = |map: &std::collections::HashMap<String, Vec<f32>>,
                           key: &str,
                           rows: usize,
                           cols: usize|
         -> ModelResult<Option<Array2<f32>>> {
            if let Some(data) = map.get(key) {
                if data.len() != rows * cols {
                    return Err(ModelError::load_error(
                        "rwkv load_weights",
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
                        "rwkv load_weights",
                        format!("failed to reshape '{}': {e}", key),
                    )
                })?;
                Ok(Some(arr))
            } else {
                Ok(None)
            }
        };

        let load_array1 = |map: &std::collections::HashMap<String, Vec<f32>>,
                           key: &str,
                           expected_len: usize|
         -> ModelResult<Option<Array1<f32>>> {
            if let Some(data) = map.get(key) {
                if data.len() != expected_len {
                    return Err(ModelError::load_error(
                        "rwkv load_weights",
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

        let hidden = self.config.hidden_dim;
        let intermediate = self.config.intermediate_dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        if let Some(arr) = load_array2(&weights, "input_proj", self.config.input_dim, hidden)? {
            self.input_proj = arr;
        }
        if let Some(arr) = load_array2(&weights, "output_proj", hidden, self.config.input_dim)? {
            self.output_proj = arr;
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);
            let tm = format!("{}.time_mixing", prefix);
            let cm = format!("{}.channel_mixing", prefix);

            if let Some(arr) = load_array1(&weights, &format!("{}.time_mix_k", tm), hidden)? {
                layer.time_mixing.time_mix_k = arr;
            }
            if let Some(arr) = load_array1(&weights, &format!("{}.time_mix_v", tm), hidden)? {
                layer.time_mixing.time_mix_v = arr;
            }
            if let Some(arr) = load_array1(&weights, &format!("{}.time_mix_r", tm), hidden)? {
                layer.time_mixing.time_mix_r = arr;
            }
            if let Some(arr) = load_array1(&weights, &format!("{}.time_mix_g", tm), hidden)? {
                layer.time_mixing.time_mix_g = arr;
            }
            if let Some(arr) =
                load_array2(&weights, &format!("{}.time_decay", tm), num_heads, head_dim)?
            {
                layer.time_mixing.time_decay = arr;
            }
            if let Some(arr) = load_array2(&weights, &format!("{}.key_proj", tm), hidden, hidden)? {
                layer.time_mixing.key_proj = arr;
            }
            if let Some(arr) = load_array2(&weights, &format!("{}.value_proj", tm), hidden, hidden)?
            {
                layer.time_mixing.value_proj = arr;
            }
            if let Some(arr) =
                load_array2(&weights, &format!("{}.receptance_proj", tm), hidden, hidden)?
            {
                layer.time_mixing.receptance_proj = arr;
            }
            if let Some(arr) = load_array2(&weights, &format!("{}.gate_proj", tm), hidden, hidden)?
            {
                layer.time_mixing.gate_proj = arr;
            }
            if let Some(arr) =
                load_array2(&weights, &format!("{}.output_proj", tm), hidden, hidden)?
            {
                layer.time_mixing.output_proj = arr;
            }

            if let Some(arr) = load_array1(&weights, &format!("{}.time_mix_k", cm), hidden)? {
                layer.channel_mixing.time_mix_k = arr;
            }
            if let Some(arr) = load_array1(&weights, &format!("{}.time_mix_r", cm), hidden)? {
                layer.channel_mixing.time_mix_r = arr;
            }
            if let Some(arr) =
                load_array2(&weights, &format!("{}.key_proj", cm), hidden, intermediate)?
            {
                layer.channel_mixing.key_proj = arr;
            }
            if let Some(arr) = load_array2(
                &weights,
                &format!("{}.value_proj", cm),
                intermediate,
                hidden,
            )? {
                layer.channel_mixing.value_proj = arr;
            }
            if let Some(arr) =
                load_array2(&weights, &format!("{}.receptance_proj", cm), hidden, hidden)?
            {
                layer.channel_mixing.receptance_proj = arr;
            }
        }

        Ok(())
    }

    /// Save model weights to a SafeTensors file.
    ///
    /// All tensors are serialised as `F32` in little-endian byte order.
    /// The resulting file can be loaded with any SafeTensors-compatible reader.
    pub fn save_weights(&self, path: &str) -> ModelResult<()> {
        // ── 1. Collect (name, flat-f32-bytes, shape) for every tensor ──────
        let mut raw: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

        let f32_bytes =
            |slice: &[f32]| -> Vec<u8> { slice.iter().flat_map(|f| f.to_le_bytes()).collect() };

        // Top-level projections
        {
            let arr = &self.input_proj;
            let shape = vec![arr.nrows(), arr.ncols()];
            let bytes = f32_bytes(arr.as_slice().ok_or_else(|| {
                ModelError::load_error("rwkv save_weights", "input_proj is not contiguous")
            })?);
            raw.push(("input_proj".to_string(), bytes, shape));
        }
        {
            let arr = &self.output_proj;
            let shape = vec![arr.nrows(), arr.ncols()];
            let bytes = f32_bytes(arr.as_slice().ok_or_else(|| {
                ModelError::load_error("rwkv save_weights", "output_proj is not contiguous")
            })?);
            raw.push(("output_proj".to_string(), bytes, shape));
        }

        // Per-layer tensors
        for (i, layer) in self.layers.iter().enumerate() {
            let tm = &layer.time_mixing;
            let cm = &layer.channel_mixing;
            let tm_prefix = format!("layers.{i}.time_mixing");
            let cm_prefix = format!("layers.{i}.channel_mixing");

            // ── Time-mixing Array1 fields ──
            for (suffix, arr) in [
                ("time_mix_k", &tm.time_mix_k),
                ("time_mix_v", &tm.time_mix_v),
                ("time_mix_r", &tm.time_mix_r),
                ("time_mix_g", &tm.time_mix_g),
            ] {
                let shape = vec![arr.len()];
                let bytes = f32_bytes(arr.as_slice().ok_or_else(|| {
                    ModelError::load_error(
                        "rwkv save_weights",
                        format!("{tm_prefix}.{suffix} is not contiguous"),
                    )
                })?);
                raw.push((format!("{tm_prefix}.{suffix}"), bytes, shape));
            }

            // ── Time-mixing Array2 fields ──
            {
                let arr = &tm.time_decay;
                let shape = vec![arr.nrows(), arr.ncols()];
                let bytes = f32_bytes(arr.as_slice().ok_or_else(|| {
                    ModelError::load_error(
                        "rwkv save_weights",
                        format!("{tm_prefix}.time_decay is not contiguous"),
                    )
                })?);
                raw.push((format!("{tm_prefix}.time_decay"), bytes, shape));
            }
            for (suffix, arr) in [
                ("key_proj", &tm.key_proj),
                ("value_proj", &tm.value_proj),
                ("receptance_proj", &tm.receptance_proj),
                ("gate_proj", &tm.gate_proj),
                ("output_proj", &tm.output_proj),
            ] {
                let shape = vec![arr.nrows(), arr.ncols()];
                let bytes = f32_bytes(arr.as_slice().ok_or_else(|| {
                    ModelError::load_error(
                        "rwkv save_weights",
                        format!("{tm_prefix}.{suffix} is not contiguous"),
                    )
                })?);
                raw.push((format!("{tm_prefix}.{suffix}"), bytes, shape));
            }

            // ── Channel-mixing Array1 fields ──
            for (suffix, arr) in [
                ("time_mix_k", &cm.time_mix_k),
                ("time_mix_r", &cm.time_mix_r),
            ] {
                let shape = vec![arr.len()];
                let bytes = f32_bytes(arr.as_slice().ok_or_else(|| {
                    ModelError::load_error(
                        "rwkv save_weights",
                        format!("{cm_prefix}.{suffix} is not contiguous"),
                    )
                })?);
                raw.push((format!("{cm_prefix}.{suffix}"), bytes, shape));
            }

            // ── Channel-mixing Array2 fields ──
            for (suffix, arr) in [
                ("key_proj", &cm.key_proj),
                ("value_proj", &cm.value_proj),
                ("receptance_proj", &cm.receptance_proj),
            ] {
                let shape = vec![arr.nrows(), arr.ncols()];
                let bytes = f32_bytes(arr.as_slice().ok_or_else(|| {
                    ModelError::load_error(
                        "rwkv save_weights",
                        format!("{cm_prefix}.{suffix} is not contiguous"),
                    )
                })?);
                raw.push((format!("{cm_prefix}.{suffix}"), bytes, shape));
            }
        }

        // ── 2. Build TensorView slice (borrows bytes stored in `raw`) ───────
        let views: Vec<(String, TensorView<'_>)> = raw
            .iter()
            .map(|(name, bytes, shape)| {
                TensorView::new(Dtype::F32, shape.clone(), bytes)
                    .map(|view| (name.clone(), view))
                    .map_err(|e| {
                        ModelError::load_error(
                            "rwkv save_weights",
                            format!("TensorView for {name}: {e}"),
                        )
                    })
            })
            .collect::<ModelResult<Vec<_>>>()?;

        // ── 3. Write to disk ─────────────────────────────────────────────────
        safetensors::tensor::serialize_to_file(views, None, std::path::Path::new(path)).map_err(
            |e| ModelError::load_error("rwkv save_weights", format!("serialize_to_file: {e}")),
        )?;

        Ok(())
    }
}

impl SignalPredictor for Rwkv {
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Project input to hidden dimension
        let mut hidden = input.dot(&self.input_proj);

        // Pass through each layer
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden)?;
        }

        // Final layer normalization
        hidden = self.ln_out.forward(&hidden);

        // Project back to input dimension
        let output = hidden.dot(&self.output_proj);
        Ok(output)
    }

    fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    fn context_window(&self) -> usize {
        // RWKV has theoretically infinite context via recurrence
        usize::MAX
    }
}

impl AutoregressiveModel for Rwkv {
    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn state_dim(&self) -> usize {
        self.config.head_dim
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn model_type(&self) -> ModelType {
        ModelType::Rwkv
    }

    fn get_states(&self) -> Vec<HiddenState> {
        // Collect WKV states from each layer
        self.layers
            .iter()
            .map(|layer| {
                // Flatten multi-head WKV states
                let total_size = layer.time_mixing.num_heads * layer.time_mixing.head_dim;
                let mut combined = Array2::zeros((total_size, 1));

                for (head_idx, head_state) in layer.time_mixing.wkv_state.iter().enumerate() {
                    let start_idx = head_idx * layer.time_mixing.head_dim;
                    for i in 0..layer.time_mixing.head_dim.min(head_state.len()) {
                        combined[[start_idx + i, 0]] = head_state[i];
                    }
                }

                let mut hs = HiddenState::new(combined.shape()[0], combined.shape()[1]);
                hs.update(combined);
                hs
            })
            .collect()
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != self.config.num_layers {
            return Err(ModelError::state_count_mismatch(
                "RWKV",
                self.config.num_layers,
                states.len(),
            ));
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let combined = states[layer_idx].state();

            // Split combined state back into per-head WKV states
            for (head_idx, head_state) in layer.time_mixing.wkv_state.iter_mut().enumerate() {
                let start_idx = head_idx * layer.time_mixing.head_dim;
                for i in 0..layer.time_mixing.head_dim.min(head_state.len()) {
                    if start_idx + i < combined.shape()[0] && 0 < combined.shape()[1] {
                        head_state[i] = combined[[start_idx + i, 0]];
                    }
                }
            }
        }

        Ok(())
    }

    fn load_weights_json(&mut self, path: &std::path::Path) -> ModelResult<()> {
        Rwkv::load_weights_json(self, path)
    }

    fn save_weights_json(&self, path: &std::path::Path) -> ModelResult<()> {
        Rwkv::save_weights_json(self, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rwkv_config() {
        let config = RwkvConfig::new().hidden_dim(512).num_heads(8).num_layers(6);

        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_rwkv_creation() {
        // Use smaller configuration for faster test
        // Default has num_layers=12 which is slow to initialize
        let config = RwkvConfig::new().hidden_dim(128).num_heads(4).num_layers(2);
        let model = Rwkv::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_rwkv_forward() {
        let config = RwkvConfig::new().hidden_dim(128).num_heads(4).num_layers(2);
        let mut model = Rwkv::new(config).expect("Failed to create RWKV model");

        let input = Array1::from_vec(vec![0.5]);
        let output = model.step(&input);
        assert!(output.is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let config = RwkvConfig::new().hidden_dim(100).num_heads(3); // Not divisible
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_rwkv_save_load_roundtrip() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static RWKV_ROUNDTRIP_COUNTER: AtomicU64 = AtomicU64::new(0);
        let uid = RWKV_ROUNDTRIP_COUNTER.fetch_add(1, Ordering::Relaxed);

        let hidden = 64usize;
        let config = RwkvConfig {
            input_dim: 1,
            hidden_dim: hidden,
            intermediate_dim: hidden * 4,
            num_layers: 2,
            num_heads: 4,
            head_dim: hidden / 4,
            dropout: 0.0,
            time_decay_init: -5.0,
            use_rms_norm: true,
        };

        let model = Rwkv::new(config).expect("Failed to create RWKV model");

        let mut tmp = std::env::temp_dir();
        tmp.push(format!("kizzasi_rwkv_roundtrip_test_{}.json", uid));

        model
            .save_weights_json(&tmp)
            .expect("save_weights_json failed");

        let config2 = RwkvConfig {
            input_dim: 1,
            hidden_dim: hidden,
            intermediate_dim: hidden * 4,
            num_layers: 2,
            num_heads: 4,
            head_dim: hidden / 4,
            dropout: 0.0,
            time_decay_init: -5.0,
            use_rms_norm: true,
        };
        let mut model2 = Rwkv::new(config2).expect("Failed to create second RWKV model");
        model2
            .load_weights_json(&tmp)
            .expect("load_weights_json failed");

        // Verify the saved file is valid JSON with expected keys
        let file = std::fs::File::open(&tmp).expect("temp file should exist");
        let reloaded: std::collections::HashMap<String, Vec<f32>> =
            serde_json::from_reader(file).expect("should deserialize");
        // 2 top-level + (10 time_mixing + 5 channel_mixing) × 2 layers = 32 keys
        assert_eq!(reloaded.len(), 32, "unexpected number of weight keys");

        let _ = std::fs::remove_file(&tmp);
    }

    // ─── WKV exp(k) correctness ──────────────────────────────────────────

    #[test]
    fn test_rwkv_wkv_exp_k_bounded_output() {
        // key_proj filled with -1.0 → k = Σ(-1)*1 = -hidden_dim for unit input
        // Bug: wkv_norm += k (negative) → clamped to 1e-8 → output ≈ |k|*|v|/1e-8 ~ 1e8
        // Fix: wkv_norm += exp(k) = exp(-4) ≈ 0.018 → output bounded O(1)
        let config = RwkvConfig {
            input_dim: 1,
            hidden_dim: 4,
            intermediate_dim: 16,
            num_layers: 1,
            num_heads: 2,
            head_dim: 2,
            dropout: 0.0,
            time_decay_init: -5.0,
            use_rms_norm: false,
        };
        let mut tm = TimeMixing::new(&config).expect("TimeMixing::new");
        tm.key_proj.fill(-1.0);
        tm.time_mix_k.fill(1.0); // xx = x (no prev mixing)

        let x = Array1::from_vec(vec![1.0f32; 4]);
        let output = tm.forward(&x).expect("TimeMixing::forward");

        assert!(
            output.iter().all(|&v| v.is_finite() && v.abs() < 1e6),
            "WKV output must be bounded; missing exp(k) gives ~1e8. Got: {:?}",
            output
        );
    }

    #[test]
    fn test_rwkv_wkv_positive_denominator() {
        // exp(k) > 0 ensures the WKV denominator is always positive through the public API.
        let config = RwkvConfig::new().hidden_dim(8).num_heads(2).num_layers(1);
        let mut model = Rwkv::new(config).expect("Rwkv::new");
        let input = Array1::from_vec(vec![0.5f32]);
        let output = model.step(&input).expect("step");
        assert!(
            output.iter().all(|&v| v.is_finite()),
            "step output must be finite; got: {:?}",
            output
        );
    }

    #[test]
    fn test_save_weights_roundtrip_safetensors() {
        let config = RwkvConfig {
            input_dim: 4,
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
            intermediate_dim: 16,
            dropout: 0.0,
            time_decay_init: -5.0,
            use_rms_norm: true,
        };
        let model = Rwkv::new(config).expect("model creation failed");
        let tmp = std::env::temp_dir().join("test_rwkv_save_weights.safetensors");
        model
            .save_weights(tmp.to_str().expect("path to str"))
            .expect("save_weights failed");
        let data = std::fs::read(&tmp).expect("read file");
        let tensors = safetensors::SafeTensors::deserialize(&data).expect("deserialize");
        assert!(!tensors.names().is_empty(), "no tensors written");
        std::fs::remove_file(&tmp).ok();
    }
}
