//! Transformer: Standard Multi-Head Attention Baseline
//!
//! This module implements a standard Transformer architecture for comparison
//! with State Space Models. While Transformers require O(N²) attention computation
//! and O(N) memory per step during inference, they serve as a strong baseline
//! for quality comparison.
//!
//! # Architecture
//!
//! ```text
//! Input → [Embedding] → [LayerNorm] → [Multi-Head Attention] → [Add] →
//!                          ↓                                      ↓
//!                       [LayerNorm] → [Feed Forward] → [Add] → Output
//! ```
//!
//! # Comparison with SSMs
//!
//! | Model       | Per-Step Time | Per-Step Memory | Training  | Context |
//! |-------------|---------------|-----------------|-----------|---------|
//! | Transformer | O(N)          | O(N)            | O(N²)     | Limited |
//! | Mamba/RWKV  | O(1)          | O(1)            | O(N)      | ∞       |
//! | S4/S4D      | O(1)          | O(1)            | O(N log N)| ∞       |
//!
//! # Purpose
//!
//! This implementation serves as a quality baseline to validate that SSM
//! architectures (Mamba2, RWKV, S4D) achieve competitive or better performance
//! while maintaining their efficiency advantages.

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{gelu, softmax, CoreResult, HiddenState, LayerNorm, NormType, SignalPredictor};
use safetensors::tensor::{Dtype, TensorView};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, RngExt};
use std::collections::VecDeque;
#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

/// Configuration for Transformer
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformerConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension (d_model)
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension (derived: hidden_dim / num_heads)
    pub head_dim: usize,
    /// Feed-forward intermediate dimension (typically 4x hidden_dim)
    pub ff_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Maximum context window
    pub max_seq_len: usize,
    /// Dropout rate applied to each layer output while training mode is
    /// enabled via `set_training(true)` (inverted dropout; inert at inference)
    pub dropout: f32,
    /// Use RMSNorm instead of LayerNorm
    pub use_rms_norm: bool,
    /// Reserved: currently has no runtime effect.
    ///
    /// `Transformer::step` (and the `MultiHeadAttention` it drives) is a
    /// single-token-at-a-time streaming path: each step appends exactly one
    /// new position to the KV cache and never removes from the front except
    /// to enforce `max_seq_len`. Every cached position is therefore already
    /// `<= ` the current token by construction, so there is no future
    /// position an explicit causal mask would ever need to hide — the
    /// architecture is unconditionally causal regardless of this flag.
    pub causal: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        let hidden_dim = 512;
        let num_heads = 8;
        Self {
            input_dim: 1,
            hidden_dim,
            num_heads,
            head_dim: hidden_dim / num_heads,
            ff_dim: hidden_dim * 4,
            num_layers: 6,
            max_seq_len: 2048,
            dropout: 0.1,
            use_rms_norm: true,
            causal: true,
        }
    }
}

impl TransformerConfig {
    /// Create a new Transformer configuration
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

    /// Set number of heads
    pub fn num_heads(mut self, n: usize) -> Self {
        self.num_heads = n;
        self.head_dim = self.hidden_dim / n;
        self
    }

    /// Set number of layers
    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Set maximum sequence length
    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> ModelResult<()> {
        if self.hidden_dim == 0 {
            return Err(ModelError::invalid_config("hidden_dim must be > 0"));
        }
        if self.num_heads == 0 {
            return Err(ModelError::invalid_config("num_heads must be > 0"));
        }
        if !self.hidden_dim.is_multiple_of(self.num_heads) {
            return Err(ModelError::invalid_config(
                "hidden_dim must be divisible by num_heads",
            ));
        }
        if self.num_layers == 0 {
            return Err(ModelError::invalid_config("num_layers must be > 0"));
        }
        if self.max_seq_len == 0 {
            return Err(ModelError::invalid_config("max_seq_len must be > 0"));
        }
        Ok(())
    }
}

/// Apply rotary position embeddings (RoPE) to one attention head's slice of
/// `vec`, in place, at absolute position `pos`.
///
/// Uses the "rotate half" convention (splitting the head into two halves and
/// rotating them as a complex pair per dimension), matching most
/// open-weight implementations (LLaMA, GPT-NeoX, ...), with the standard
/// base of 10000. An odd `head_dim` leaves its final, unpaired element
/// untouched.
fn apply_rope(vec: &mut Array1<f32>, head_start: usize, head_dim: usize, pos: usize) {
    let half = head_dim / 2;
    for i in 0..half {
        let inv_freq = 1.0 / 10000f32.powf(2.0 * i as f32 / head_dim as f32);
        let theta = pos as f32 * inv_freq;
        let (sin_t, cos_t) = theta.sin_cos();
        let idx0 = head_start + i;
        let idx1 = head_start + half + i;
        if idx1 < vec.len() {
            let x0 = vec[idx0];
            let x1 = vec[idx1];
            vec[idx0] = x0 * cos_t - x1 * sin_t;
            vec[idx1] = x0 * sin_t + x1 * cos_t;
        }
    }
}

/// Multi-Head Self-Attention
struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    hidden_dim: usize,

    /// Query, Key, Value projections
    q_proj: Array2<f32>,
    k_proj: Array2<f32>,
    v_proj: Array2<f32>,

    /// Output projection
    o_proj: Array2<f32>,

    /// Cached keys and values for autoregressive generation
    key_cache: VecDeque<Array1<f32>>,
    value_cache: VecDeque<Array1<f32>>,
    max_cache_len: usize,

    /// Absolute position of the next token to be processed by `forward`.
    /// Drives the RoPE rotation applied to `q`/`k`; reset to 0 in
    /// [`Self::reset`]. Keys are rotated once, at insertion time, before
    /// being cached — not re-rotated on every subsequent step — so eviction
    /// from the cache (once `max_cache_len` is exceeded) does not disturb
    /// the positional encoding of the entries that remain.
    position: usize,
}

impl MultiHeadAttention {
    fn new(config: &TransformerConfig) -> ModelResult<Self> {
        let mut rng = rng();
        let scale = (2.0 / config.hidden_dim as f32).sqrt();

        let q_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let k_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let v_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let o_proj = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        Ok(Self {
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            hidden_dim: config.hidden_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            key_cache: VecDeque::new(),
            value_cache: VecDeque::new(),
            max_cache_len: config.max_seq_len,
            position: 0,
        })
    }

    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let batch_size = x.len().min(self.hidden_dim);
        let current_pos = self.position;

        // Project to Q, K, V
        let mut q = self.project(x, &self.q_proj);
        let mut k = self.project(x, &self.k_proj);
        let v = self.project(x, &self.v_proj);

        // Rotary position embeddings: rotate q (this token) and k (before
        // caching, so every cached key stays permanently tagged with its
        // own absolute position) per head. This is what makes attention
        // depend on token order at all: previously there was no positional
        // signal anywhere in this module, so permuting the order of
        // previously-seen inputs left the output completely unchanged.
        for h in 0..self.num_heads {
            let head_start = h * self.head_dim;
            apply_rope(&mut q, head_start, self.head_dim, current_pos);
            apply_rope(&mut k, head_start, self.head_dim, current_pos);
        }

        // Add to cache
        self.key_cache.push_back(k.clone());
        self.value_cache.push_back(v.clone());

        // Maintain cache size
        while self.key_cache.len() > self.max_cache_len {
            self.key_cache.pop_front();
            self.value_cache.pop_front();
        }

        // Compute attention over cached context
        let seq_len = self.key_cache.len();
        let scale = (self.head_dim as f32).sqrt();

        let mut attention_output = Array1::zeros(batch_size);

        // For each head
        for h in 0..self.num_heads {
            let head_start = h * self.head_dim;

            // Compute attention scores with all cached positions
            let mut scores = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                let k_cached = &self.key_cache[pos];
                let mut score = 0.0;

                // Q · K^T for this head (both already RoPE-rotated)
                for i in 0..self.head_dim {
                    let q_idx = head_start + i;
                    let k_idx = head_start + i;
                    if q_idx < q.len() && k_idx < k_cached.len() {
                        score += q[q_idx] * k_cached[k_idx];
                    }
                }
                score /= scale;

                // No causal mask is applied here (and none is needed):
                // `key_cache`/`value_cache` hold exactly the positions
                // `forward` has already appended, oldest-evicted-first, so
                // every cached position is by construction <= the current
                // token — there is no future position for a mask to ever
                // need to hide in this single-token-at-a-time streaming
                // design. `TransformerConfig::causal` therefore has no
                // effect on this step-wise path.
                scores.push(score);
            }

            // Softmax over positions
            let attention_weights = softmax(&Array1::from_vec(scores));

            // Weighted sum of values
            for i in 0..self.head_dim {
                let out_idx = head_start + i;
                if out_idx >= attention_output.len() {
                    break;
                }

                let mut weighted_value = 0.0;
                for (pos, &weight) in attention_weights.iter().enumerate() {
                    let v_cached = &self.value_cache[pos];
                    let v_idx = head_start + i;
                    if v_idx < v_cached.len() {
                        weighted_value += weight * v_cached[v_idx];
                    }
                }
                attention_output[out_idx] = weighted_value;
            }
        }

        self.position += 1;

        // Output projection
        let output = self.project(&attention_output, &self.o_proj);
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
        self.key_cache.clear();
        self.value_cache.clear();
        self.position = 0;
    }
}

/// Feed-Forward Network
struct FeedForward {
    fc1: Array2<f32>,
    fc2: Array2<f32>,
}

impl FeedForward {
    fn new(config: &TransformerConfig) -> ModelResult<Self> {
        let mut rng = rng();
        let scale1 = (2.0 / config.hidden_dim as f32).sqrt();
        let scale2 = (2.0 / config.ff_dim as f32).sqrt();

        let fc1 = Array2::from_shape_fn((config.hidden_dim, config.ff_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale1
        });
        let fc2 = Array2::from_shape_fn((config.ff_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale2
        });

        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // First layer
        let mut hidden = Array1::zeros(self.fc1.shape()[1]);
        for i in 0..hidden.len() {
            let mut sum = 0.0;
            for j in 0..x.len().min(self.fc1.shape()[0]) {
                sum += self.fc1[[j, i]] * x[j];
            }
            hidden[i] = sum;
        }

        // Activation (GELU)
        hidden = gelu(&hidden);

        // Second layer
        let mut output = Array1::zeros(x.len().min(self.fc2.shape()[1]));
        for i in 0..output.len() {
            let mut sum = 0.0;
            for j in 0..hidden.len().min(self.fc2.shape()[0]) {
                sum += self.fc2[[j, i]] * hidden[j];
            }
            output[i] = sum;
        }

        Ok(output)
    }
}

/// Transformer Layer
struct TransformerLayer {
    ln1: LayerNorm,
    ln2: LayerNorm,
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
}

impl TransformerLayer {
    fn new(config: &TransformerConfig) -> ModelResult<Self> {
        let norm_type = if config.use_rms_norm {
            NormType::RMSNorm
        } else {
            NormType::LayerNorm
        };

        let ln1 = LayerNorm::new(config.hidden_dim, norm_type).with_eps(1e-5);
        let ln2 = LayerNorm::new(config.hidden_dim, norm_type).with_eps(1e-5);
        let attention = MultiHeadAttention::new(config)?;
        let feed_forward = FeedForward::new(config)?;

        Ok(Self {
            ln1,
            ln2,
            attention,
            feed_forward,
        })
    }

    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Pre-norm: LayerNorm → Attention → Residual
        let x_norm = self.ln1.forward(x);
        // `TransformerConfig::causal` is not threaded through: see
        // `MultiHeadAttention::forward`'s doc on why a causal mask is
        // structurally unnecessary (and was previously dead code) for this
        // single-token-at-a-time streaming path.
        let attn_out = self.attention.forward(&x_norm)?;
        let mut x_attn = x.clone();
        for i in 0..x_attn.len().min(attn_out.len()) {
            x_attn[i] += attn_out[i];
        }

        // Pre-norm: LayerNorm → FFN → Residual
        let x_norm2 = self.ln2.forward(&x_attn);
        let ff_out = self.feed_forward.forward(&x_norm2)?;
        let mut output = x_attn;
        for i in 0..output.len().min(ff_out.len()) {
            output[i] += ff_out[i];
        }

        Ok(output)
    }

    fn reset(&mut self) {
        self.attention.reset();
    }
}

/// Transformer model
pub struct Transformer {
    config: TransformerConfig,
    layers: Vec<TransformerLayer>,
    ln_out: LayerNorm,
    input_proj: Array2<f32>,
    output_proj: Array2<f32>,
    /// Whether `TransformerConfig::dropout` is active (see [`Transformer::set_training`]).
    training: bool,
}

impl Transformer {
    /// Create a new Transformer model
    pub fn new(config: TransformerConfig) -> ModelResult<Self> {
        config.validate()?;

        // Initialize layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(TransformerLayer::new(&config)?);
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
            training: false,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Load weights from a SafeTensors model file
    ///
    /// # Weight Naming Convention
    ///
    /// The following tensor names are expected:
    /// - `input_proj`: Input projection matrix (input_dim, hidden_dim)
    /// - `output_proj`: Output projection matrix (hidden_dim, input_dim)
    /// - `ln_out.weight`: Output layer norm weight (gamma)
    /// - `ln_out.bias`: Output layer norm bias (beta, optional)
    ///
    /// For each layer i:
    /// - `layers.{i}.ln1.weight`: Attention layer norm weight
    /// - `layers.{i}.ln1.bias`: Attention layer norm bias (optional)
    /// - `layers.{i}.ln2.weight`: Feed-forward layer norm weight
    /// - `layers.{i}.ln2.bias`: Feed-forward layer norm bias (optional)
    ///
    /// Multi-head attention parameters:
    /// - `layers.{i}.attention.q_proj`: Query projection
    /// - `layers.{i}.attention.k_proj`: Key projection
    /// - `layers.{i}.attention.v_proj`: Value projection
    /// - `layers.{i}.attention.o_proj`: Output projection
    ///
    /// Feed-forward parameters:
    /// - `layers.{i}.feed_forward.fc1`: First linear layer
    /// - `layers.{i}.feed_forward.fc2`: Second linear layer
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

            // Load layer norm 1 (attention)
            if loader.has_tensor(&format!("{}.ln1.weight", prefix)) {
                let weight = loader.load_array1(&format!("{}.ln1.weight", prefix))?;
                layer.ln1.set_gamma(weight);
            }
            if loader.has_tensor(&format!("{}.ln1.bias", prefix)) {
                let bias = loader.load_array1(&format!("{}.ln1.bias", prefix))?;
                layer.ln1.set_beta(bias);
            }

            // Load layer norm 2 (feed-forward)
            if loader.has_tensor(&format!("{}.ln2.weight", prefix)) {
                let weight = loader.load_array1(&format!("{}.ln2.weight", prefix))?;
                layer.ln2.set_gamma(weight);
            }
            if loader.has_tensor(&format!("{}.ln2.bias", prefix)) {
                let bias = loader.load_array1(&format!("{}.ln2.bias", prefix))?;
                layer.ln2.set_beta(bias);
            }

            // Load attention parameters
            let attn_prefix = format!("{}.attention", prefix);
            if loader.has_tensor(&format!("{}.q_proj", attn_prefix)) {
                layer.attention.q_proj = loader.load_array2(&format!("{}.q_proj", attn_prefix))?;
            }
            if loader.has_tensor(&format!("{}.k_proj", attn_prefix)) {
                layer.attention.k_proj = loader.load_array2(&format!("{}.k_proj", attn_prefix))?;
            }
            if loader.has_tensor(&format!("{}.v_proj", attn_prefix)) {
                layer.attention.v_proj = loader.load_array2(&format!("{}.v_proj", attn_prefix))?;
            }
            if loader.has_tensor(&format!("{}.o_proj", attn_prefix)) {
                layer.attention.o_proj = loader.load_array2(&format!("{}.o_proj", attn_prefix))?;
            }

            // Load feed-forward parameters
            let ff_prefix = format!("{}.feed_forward", prefix);
            if loader.has_tensor(&format!("{}.fc1", ff_prefix)) {
                layer.feed_forward.fc1 = loader.load_array2(&format!("{}.fc1", ff_prefix))?;
            }
            if loader.has_tensor(&format!("{}.fc2", ff_prefix)) {
                layer.feed_forward.fc2 = loader.load_array2(&format!("{}.fc2", ff_prefix))?;
            }
        }

        Ok(())
    }

    /// Save model weights to a JSON file as `HashMap<String, Vec<f32>>`.
    ///
    /// Keys:
    /// - `input_proj` / `output_proj`: top-level projections
    /// - `layers.{i}.attention.q_proj`, `k_proj`, `v_proj`, `o_proj`
    /// - `layers.{i}.feed_forward.fc1`, `fc2`
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
            let attn = format!("{}.attention", prefix);
            let ff = format!("{}.feed_forward", prefix);

            weights.insert(
                format!("{}.q_proj", attn),
                layer.attention.q_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.k_proj", attn),
                layer.attention.k_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.v_proj", attn),
                layer.attention.v_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.o_proj", attn),
                layer.attention.o_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.fc1", ff),
                layer.feed_forward.fc1.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.fc2", ff),
                layer.feed_forward.fc2.iter().copied().collect(),
            );
        }

        let file = std::fs::File::create(path.as_ref()).map_err(|e| {
            ModelError::load_error(
                "transformer save_weights",
                format!("failed to create file: {e}"),
            )
        })?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut writer, &weights).map_err(|e| {
            ModelError::load_error(
                "transformer save_weights",
                format!("JSON serialization failed: {e}"),
            )
        })?;
        // Explicitly flush the BufWriter so all buffered data reaches the OS before the
        // file handle is closed. Without this, data still in the BufWriter's internal
        // buffer would be silently discarded if the drop-flush encountered an error,
        // resulting in a truncated file and an EOF error on the subsequent read.
        use std::io::Write as _;
        writer.flush().map_err(|e| {
            ModelError::load_error(
                "transformer save_weights",
                format!("failed to flush JSON to file: {e}"),
            )
        })?;
        Ok(())
    }

    /// Enable or disable training mode.
    ///
    /// Models are created in inference mode, where `TransformerConfig::dropout` is inert
    /// and [`step`](crate::SignalPredictor::step) is deterministic. Set this to
    /// `true` during training so the configured dropout rate is applied to each
    /// layer output (inverted dropout — no rescale is needed when switching
    /// back to inference).
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Whether the model is currently in training mode (dropout active).
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Load weights from a JSON file previously written by `save_weights_json`.
    pub fn load_weights_json<P: AsRef<std::path::Path>>(&mut self, path: P) -> ModelResult<()> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            ModelError::load_error(
                "transformer load_weights",
                format!("failed to open file: {e}"),
            )
        })?;
        let weights: std::collections::HashMap<String, Vec<f32>> = serde_json::from_reader(file)
            .map_err(|e| {
                ModelError::load_error(
                    "transformer load_weights",
                    format!("JSON deserialization failed: {e}"),
                )
            })?;
        self.load_weights_map(&weights).map(|_| ())
    }

    /// Load weights from an in-memory `name → flat f32 values` map.
    ///
    /// This is the in-process counterpart of [`Self::load_weights_json`]: it
    /// applies exactly the same shape checks and partial-loading semantics
    /// without routing the parameters through a serialized file.
    pub fn load_weights_map(
        &mut self,
        weights: &std::collections::HashMap<String, Vec<f32>>,
    ) -> ModelResult<usize> {
        // Number of tensors actually applied. The caller needs this to tell a
        // genuine partial load from a weight map whose names match nothing at
        // all — the latter would otherwise leave the model randomly
        // initialised while reporting success.
        let applied = std::cell::Cell::new(0usize);
        let load_array2 = |map: &std::collections::HashMap<String, Vec<f32>>,
                           key: &str,
                           rows: usize,
                           cols: usize|
         -> ModelResult<Option<Array2<f32>>> {
            if let Some(data) = map.get(key) {
                if data.len() != rows * cols {
                    return Err(ModelError::load_error(
                        "transformer load_weights",
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
                        "transformer load_weights",
                        format!("failed to reshape '{}': {e}", key),
                    )
                })?;
                applied.set(applied.get() + 1);
                Ok(Some(arr))
            } else {
                Ok(None)
            }
        };

        let hidden = self.config.hidden_dim;
        let ff_dim = self.config.ff_dim;

        if let Some(arr) = load_array2(weights, "input_proj", self.config.input_dim, hidden)? {
            self.input_proj = arr;
        }
        if let Some(arr) = load_array2(weights, "output_proj", hidden, self.config.input_dim)? {
            self.output_proj = arr;
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);
            let attn = format!("{}.attention", prefix);
            let ff = format!("{}.feed_forward", prefix);

            if let Some(arr) = load_array2(weights, &format!("{}.q_proj", attn), hidden, hidden)? {
                layer.attention.q_proj = arr;
            }
            if let Some(arr) = load_array2(weights, &format!("{}.k_proj", attn), hidden, hidden)? {
                layer.attention.k_proj = arr;
            }
            if let Some(arr) = load_array2(weights, &format!("{}.v_proj", attn), hidden, hidden)? {
                layer.attention.v_proj = arr;
            }
            if let Some(arr) = load_array2(weights, &format!("{}.o_proj", attn), hidden, hidden)? {
                layer.attention.o_proj = arr;
            }
            if let Some(arr) = load_array2(weights, &format!("{}.fc1", ff), hidden, ff_dim)? {
                layer.feed_forward.fc1 = arr;
            }
            if let Some(arr) = load_array2(weights, &format!("{}.fc2", ff), ff_dim, hidden)? {
                layer.feed_forward.fc2 = arr;
            }
        }

        Ok(applied.get())
    }

    /// Save all model weights to a SafeTensors file at `path`.
    ///
    /// Each named tensor is serialised as a row-major `F32` tensor. The file
    /// can be reloaded with any SafeTensors-compatible loader.
    pub fn save_weights(&self, path: &str) -> ModelResult<()> {
        let hidden = self.config.hidden_dim;
        let input_dim = self.config.input_dim;
        let ff_dim = self.config.ff_dim;

        // Collect (name, raw-bytes, shape) for every tensor.
        let mut entries: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

        let to_bytes =
            |arr: &Array2<f32>| -> Vec<u8> { arr.iter().flat_map(|f| f.to_le_bytes()).collect() };

        entries.push((
            "input_proj".to_owned(),
            to_bytes(&self.input_proj),
            vec![input_dim, hidden],
        ));
        entries.push((
            "output_proj".to_owned(),
            to_bytes(&self.output_proj),
            vec![hidden, input_dim],
        ));

        for (i, layer) in self.layers.iter().enumerate() {
            let attn = &layer.attention;
            let ff = &layer.feed_forward;

            entries.push((
                format!("layers.{i}.attention.q_proj"),
                to_bytes(&attn.q_proj),
                vec![hidden, hidden],
            ));
            entries.push((
                format!("layers.{i}.attention.k_proj"),
                to_bytes(&attn.k_proj),
                vec![hidden, hidden],
            ));
            entries.push((
                format!("layers.{i}.attention.v_proj"),
                to_bytes(&attn.v_proj),
                vec![hidden, hidden],
            ));
            entries.push((
                format!("layers.{i}.attention.o_proj"),
                to_bytes(&attn.o_proj),
                vec![hidden, hidden],
            ));
            entries.push((
                format!("layers.{i}.feed_forward.fc1"),
                to_bytes(&ff.fc1),
                vec![hidden, ff_dim],
            ));
            entries.push((
                format!("layers.{i}.feed_forward.fc2"),
                to_bytes(&ff.fc2),
                vec![ff_dim, hidden],
            ));
        }

        // Build TensorViews that borrow from `entries`.
        let views: Vec<(String, TensorView<'_>)> = entries
            .iter()
            .map(|(name, bytes, shape)| {
                TensorView::new(Dtype::F32, shape.clone(), bytes)
                    .map(|view| (name.clone(), view))
                    .map_err(|e| {
                        ModelError::load_error(
                            "save_weights",
                            format!("failed to create TensorView for '{name}': {e}"),
                        )
                    })
            })
            .collect::<ModelResult<Vec<_>>>()?;

        safetensors::tensor::serialize_to_file(views, None, std::path::Path::new(path))
            .map_err(|e| ModelError::load_error("save_weights", e.to_string()))
    }
}

impl SignalPredictor for Transformer {
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        crate::check_input_dim(input, self.input_proj.shape()[0])?;

        // Project input to hidden dimension
        let mut hidden = input.dot(&self.input_proj);

        // Pass through each layer
        let dropout_rate = self.config.dropout;
        let training = self.training;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden)?;
            crate::dropout::apply_dropout(&mut hidden, dropout_rate, training);
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
        self.config.max_seq_len
    }
}

impl AutoregressiveModel for Transformer {
    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn state_dim(&self) -> usize {
        // Transformers use KV cache, which grows with sequence length
        self.config.hidden_dim
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn model_type(&self) -> ModelType {
        ModelType::Transformer
    }

    fn get_states(&self) -> Vec<HiddenState> {
        // Pack K, V, and the RoPE position counter into a single Array2 per
        // layer. Layout: rows [0, cache_len) hold K entries;
        // [cache_len, 2*cache_len) hold V entries; the final row (index
        // 2*cache_len) is a position row whose column 0 holds
        // `attention.position` (as f32) — that counter is part of the
        // layer's real recurrent state now that q/k are RoPE-rotated by it,
        // and must round-trip through get/set_states just like the cache
        // does, or a restored model would rotate its next token by the
        // wrong absolute position and silently diverge from the
        // continuation it is supposed to reproduce.
        //
        // An empty cache (cache_len == 0) is encoded as a 1-row sentinel with
        // step_count == 0 (update() is NOT called), so set_states can distinguish
        // "fresh model" from "one step of history". `position` is always 0
        // when the cache is empty (nothing has been pushed yet), so the
        // sentinel does not need to carry it separately.
        self.layers
            .iter()
            .map(|layer| {
                let cache_len = layer.attention.key_cache.len();
                if cache_len == 0 {
                    // Sentinel: HiddenState with step_count == 0; one zero-row is the
                    // minimum that HiddenState::new accepts without needing .max(1).
                    HiddenState::new(1, self.config.hidden_dim)
                } else {
                    let total_rows = cache_len * 2 + 1;
                    let mut combined = Array2::zeros((total_rows, self.config.hidden_dim));
                    for (i, k) in layer.attention.key_cache.iter().enumerate() {
                        for j in 0..k.len().min(self.config.hidden_dim) {
                            combined[[i, j]] = k[j];
                        }
                    }
                    for (i, v) in layer.attention.value_cache.iter().enumerate() {
                        for j in 0..v.len().min(self.config.hidden_dim) {
                            combined[[cache_len + i, j]] = v[j];
                        }
                    }
                    combined[[2 * cache_len, 0]] = layer.attention.position as f32;
                    let mut hs = HiddenState::new(total_rows, self.config.hidden_dim);
                    hs.update(combined);
                    hs
                }
            })
            .collect()
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != self.config.num_layers {
            return Err(ModelError::state_count_mismatch(
                "Transformer",
                self.config.num_layers,
                states.len(),
            ));
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            layer.attention.key_cache.clear();
            layer.attention.value_cache.clear();

            // step_count == 0 means the sentinel for an empty cache; nothing to restore.
            if states[layer_idx].step_count() == 0 {
                layer.attention.position = 0;
                continue;
            }

            let combined = states[layer_idx].state();
            let nrows = combined.nrows();

            if nrows == 0 || nrows.is_multiple_of(2) {
                return Err(ModelError::load_error(
                    "Transformer set_states",
                    format!(
                        "layer {layer_idx}: KV cache state has row count {nrows}; \
                         expected an odd number >= 1 (K rows concatenated with V \
                         rows, plus one trailing position row)"
                    ),
                ));
            }

            let cache_len = (nrows - 1) / 2;
            for i in 0..cache_len {
                let mut k = Array1::zeros(self.config.hidden_dim);
                for j in 0..self.config.hidden_dim.min(combined.ncols()) {
                    k[j] = combined[[i, j]];
                }
                layer.attention.key_cache.push_back(k);
            }
            for i in 0..cache_len {
                let mut v = Array1::zeros(self.config.hidden_dim);
                for j in 0..self.config.hidden_dim.min(combined.ncols()) {
                    v[j] = combined[[cache_len + i, j]];
                }
                layer.attention.value_cache.push_back(v);
            }
            layer.attention.position = if combined.ncols() > 0 {
                combined[[2 * cache_len, 0]].max(0.0) as usize
            } else {
                0
            };
        }

        Ok(())
    }

    fn load_weights_json(&mut self, path: &std::path::Path) -> ModelResult<()> {
        Transformer::load_weights_json(self, path)
    }

    fn save_weights_json(&self, path: &std::path::Path) -> ModelResult<()> {
        Transformer::save_weights_json(self, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_config() {
        let config = TransformerConfig::new()
            .hidden_dim(256)
            .num_heads(8)
            .num_layers(4);

        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 32);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_transformer_creation() {
        let config = TransformerConfig::new().hidden_dim(128).num_heads(4);
        let model = Transformer::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_transformer_forward() {
        let config = TransformerConfig::new()
            .hidden_dim(64)
            .num_heads(4)
            .num_layers(2)
            .max_seq_len(128);
        let mut model = Transformer::new(config).expect("Failed to create Transformer");

        let input = Array1::from_vec(vec![0.5]);
        let output = model.step(&input);
        assert!(output.is_ok());
    }

    #[test]
    fn test_transformer_output_depends_on_token_order() {
        // Regression test for the missing-positional-encoding bug: before
        // RoPE, attention had no positional signal at all, so permuting the
        // order of previously-seen inputs left the output for a fixed final
        // token completely unchanged. That must no longer hold.
        let config = TransformerConfig::new()
            .hidden_dim(64)
            .num_heads(4)
            .num_layers(2)
            .max_seq_len(128);
        let mut model = Transformer::new(config).expect("Failed to create Transformer");

        let a = Array1::from_vec(vec![0.3]);
        let b = Array1::from_vec(vec![0.7]);
        let c = Array1::from_vec(vec![-0.4]);

        // Sequence 1: a, b, then c.
        model.step(&a).expect("step a");
        model.step(&b).expect("step b");
        let out1 = model.step(&c).expect("step c");

        model.reset();

        // Sequence 2: b, a, then the SAME final token c -- same multiset of
        // prior tokens, different order.
        model.step(&b).expect("step b");
        model.step(&a).expect("step a");
        let out2 = model.step(&c).expect("step c");

        let differs = out1
            .iter()
            .zip(out2.iter())
            .any(|(x, y)| (x - y).abs() > 1e-5);
        assert!(
            differs,
            "reordering previously-seen tokens must change the output now \
             that RoPE provides a positional signal"
        );
    }

    #[test]
    fn test_invalid_heads() {
        let config = TransformerConfig::new().hidden_dim(100).num_heads(3); // Not divisible
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_context_window() {
        // Use smaller configuration for faster test
        // Default has hidden_dim=512, num_layers=6 which is slow to initialize
        let config = TransformerConfig::new()
            .hidden_dim(64)
            .num_heads(4)
            .num_layers(2)
            .max_seq_len(512);
        let model = Transformer::new(config).expect("Failed to create Transformer");
        assert_eq!(model.context_window(), 512);
    }

    #[test]
    fn test_transformer_save_load_roundtrip() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TRANSFORMER_ROUNDTRIP_COUNTER: AtomicU64 = AtomicU64::new(0);
        let uid = TRANSFORMER_ROUNDTRIP_COUNTER.fetch_add(1, Ordering::Relaxed);

        let config = TransformerConfig::new()
            .hidden_dim(64)
            .num_heads(4)
            .num_layers(2)
            .max_seq_len(128);

        let model = Transformer::new(config).expect("Failed to create Transformer");

        let mut tmp = std::env::temp_dir();
        tmp.push(format!("kizzasi_transformer_roundtrip_test_{}.json", uid));

        model
            .save_weights_json(&tmp)
            .expect("save_weights_json failed");

        let config2 = TransformerConfig::new()
            .hidden_dim(64)
            .num_heads(4)
            .num_layers(2)
            .max_seq_len(128);
        let mut model2 = Transformer::new(config2).expect("Failed to create second Transformer");
        model2
            .load_weights_json(&tmp)
            .expect("load_weights_json failed");

        // Verify key count: 2 top-level + 6 per-layer × 2 layers = 14 keys
        let file = std::fs::File::open(&tmp).expect("temp file should exist");
        let reloaded: std::collections::HashMap<String, Vec<f32>> =
            serde_json::from_reader(file).expect("should deserialize");
        assert_eq!(reloaded.len(), 14, "unexpected number of weight keys");

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_save_weights_roundtrip_safetensors() {
        let config = TransformerConfig {
            input_dim: 4,
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 2,
            ff_dim: 16,
            ..Default::default()
        };
        let model = Transformer::new(config).expect("model creation failed");
        let tmp = std::env::temp_dir().join("test_transformer_save_weights.safetensors");
        model
            .save_weights(tmp.to_str().expect("path to str"))
            .expect("save_weights failed");
        let data = std::fs::read(&tmp).expect("read file");
        let tensors = safetensors::SafeTensors::deserialize(&data).expect("deserialize");
        assert!(!tensors.names().is_empty(), "no tensors written");
        std::fs::remove_file(&tmp).ok();
    }
}
