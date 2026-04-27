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
    /// Dropout rate
    pub dropout: f32,
    /// Use RMSNorm instead of LayerNorm
    pub use_rms_norm: bool,
    /// Use causal masking (autoregressive)
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
        })
    }

    fn forward(&mut self, x: &Array1<f32>, causal: bool) -> CoreResult<Array1<f32>> {
        let batch_size = x.len().min(self.hidden_dim);

        // Project to Q, K, V
        let q = self.project(x, &self.q_proj);
        let k = self.project(x, &self.k_proj);
        let v = self.project(x, &self.v_proj);

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
            let _head_end = (head_start + self.head_dim).min(batch_size);

            // Compute attention scores with all cached positions
            let mut scores = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                let k_cached = &self.key_cache[pos];
                let mut score = 0.0;

                // Q · K^T for this head
                for i in 0..self.head_dim {
                    let q_idx = head_start + i;
                    let k_idx = head_start + i;
                    if q_idx < q.len() && k_idx < k_cached.len() {
                        score += q[q_idx] * k_cached[k_idx];
                    }
                }
                score /= scale;

                // Causal masking: only attend to current and past positions
                if !causal || pos < seq_len {
                    scores.push(score);
                } else {
                    scores.push(f32::NEG_INFINITY);
                }
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
    causal: bool,
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
            causal: config.causal,
        })
    }

    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Pre-norm: LayerNorm → Attention → Residual
        let x_norm = self.ln1.forward(x);
        let attn_out = self.attention.forward(&x_norm, self.causal)?;
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
                Ok(Some(arr))
            } else {
                Ok(None)
            }
        };

        let hidden = self.config.hidden_dim;
        let ff_dim = self.config.ff_dim;

        if let Some(arr) = load_array2(&weights, "input_proj", self.config.input_dim, hidden)? {
            self.input_proj = arr;
        }
        if let Some(arr) = load_array2(&weights, "output_proj", hidden, self.config.input_dim)? {
            self.output_proj = arr;
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);
            let attn = format!("{}.attention", prefix);
            let ff = format!("{}.feed_forward", prefix);

            if let Some(arr) = load_array2(&weights, &format!("{}.q_proj", attn), hidden, hidden)? {
                layer.attention.q_proj = arr;
            }
            if let Some(arr) = load_array2(&weights, &format!("{}.k_proj", attn), hidden, hidden)? {
                layer.attention.k_proj = arr;
            }
            if let Some(arr) = load_array2(&weights, &format!("{}.v_proj", attn), hidden, hidden)? {
                layer.attention.v_proj = arr;
            }
            if let Some(arr) = load_array2(&weights, &format!("{}.o_proj", attn), hidden, hidden)? {
                layer.attention.o_proj = arr;
            }
            if let Some(arr) = load_array2(&weights, &format!("{}.fc1", ff), hidden, ff_dim)? {
                layer.feed_forward.fc1 = arr;
            }
            if let Some(arr) = load_array2(&weights, &format!("{}.fc2", ff), ff_dim, hidden)? {
                layer.feed_forward.fc2 = arr;
            }
        }

        Ok(())
    }

    /// Save weights to a SafeTensors model file (legacy stub — use `save_weights_json` instead).
    #[allow(unused_variables)]
    pub fn save_weights(&self, path: &str) -> ModelResult<()> {
        self.save_weights_json(path)
    }
}

impl SignalPredictor for Transformer {
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
        // Return KV cache state for each layer
        self.layers
            .iter()
            .map(|layer| {
                let cache_len = layer.attention.key_cache.len();
                let mut combined = Array2::zeros((cache_len.max(1), self.config.hidden_dim));

                // Store key cache (value cache could be stored similarly)
                for (i, k) in layer.attention.key_cache.iter().enumerate() {
                    for j in 0..k.len().min(self.config.hidden_dim) {
                        combined[[i, j]] = k[j];
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
                "Transformer",
                self.config.num_layers,
                states.len(),
            ));
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let combined = states[layer_idx].state();

            // Restore key cache (simplified - in practice would restore both K and V)
            layer.attention.key_cache.clear();
            for i in 0..combined.shape()[0] {
                let mut k = Array1::zeros(self.config.hidden_dim);
                for j in 0..self.config.hidden_dim.min(combined.shape()[1]) {
                    k[j] = combined[[i, j]];
                }
                layer.attention.key_cache.push_back(k);
            }
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
}
