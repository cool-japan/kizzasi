//! Multi-Modal Input Fusion
//!
//! This module enables cross-modal signal processing by combining multiple
//! signal streams (audio, vision, sensors, control, text) into a unified
//! representation for prediction.
//!
//! # Architecture
//!
//! ```text
//! Audio  ──→ [Encoder_A] ──┐
//! Vision ──→ [Encoder_V] ──┼──→ [FusionLayer] ──→ [OutputProj] ──→ Prediction
//! Sensor ──→ [Encoder_S] ──┘
//! ```
//!
//! # Fusion Strategies
//!
//! - **Concatenation**: Concat all encoded modalities, project to fusion_dim
//! - **Addition**: Element-wise sum (all modalities share same projection_dim)
//! - **Gated**: Learned sigmoid gates control each modality's contribution
//! - **CrossAttention**: Each modality attends to all others via scaled dot-product
//! - **Bottleneck**: Concat → compress → expand through bottleneck layer
//!
//! # Missing Modalities
//!
//! The model gracefully handles missing modalities by substituting zero vectors,
//! enabling robust inference when some sensor streams are unavailable.
//!
//! # Stream Alignment
//!
//! [`ModalityAligner`] synchronizes streams with different sample rates by
//! buffering and releasing aligned frames based on sample rate ratios.

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{sigmoid, CoreResult, HiddenState, SignalPredictor};
use scirs2_core::ndarray::{Array1, Array2};

#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

// ---------------------------------------------------------------------------
// Seeded deterministic RNG (same pattern as rwkv7.rs)
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG for deterministic weight initialization.
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
// Modality type
// ---------------------------------------------------------------------------

/// Modality type identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Audio signal stream
    Audio,
    /// Vision / image stream
    Vision,
    /// Generic sensor data
    Sensor,
    /// Control / action signals
    Control,
    /// Text / token embeddings
    Text,
    /// User-defined modality
    Custom(String),
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Modality::Audio => write!(f, "Audio"),
            Modality::Vision => write!(f, "Vision"),
            Modality::Sensor => write!(f, "Sensor"),
            Modality::Control => write!(f, "Control"),
            Modality::Text => write!(f, "Text"),
            Modality::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Modality encoder
// ---------------------------------------------------------------------------

/// Configuration for a single modality encoder
#[derive(Debug, Clone)]
pub struct ModalityEncoderConfig {
    /// Which modality this encoder handles
    pub modality: Modality,
    /// Raw input dimension for this modality
    pub input_dim: usize,
    /// Dimension after projection (should match fusion_dim)
    pub projection_dim: usize,
    /// Depth of the encoder MLP
    pub num_layers: usize,
}

/// Encodes a single modality's input into a shared representation space
pub struct ModalityEncoder {
    config: ModalityEncoderConfig,
    /// (weight, bias) for each MLP layer
    layers: Vec<(Array2<f32>, Array1<f32>)>,
    /// Optional layer normalization (gamma, beta)
    norm: Option<(Array1<f32>, Array1<f32>)>,
}

impl ModalityEncoder {
    /// Create a new modality encoder with deterministic weight initialization.
    pub fn new(config: ModalityEncoderConfig) -> ModelResult<Self> {
        if config.input_dim == 0 {
            return Err(ModelError::invalid_config("input_dim must be > 0"));
        }
        if config.projection_dim == 0 {
            return Err(ModelError::invalid_config("projection_dim must be > 0"));
        }
        if config.num_layers == 0 {
            return Err(ModelError::invalid_config("num_layers must be > 0"));
        }

        let mut rng =
            SeededRng::new(42 + config.input_dim as u64 * 7 + config.projection_dim as u64 * 13);

        let mut layers = Vec::with_capacity(config.num_layers);

        for i in 0..config.num_layers {
            let (in_dim, out_dim) = if i == 0 {
                (config.input_dim, config.projection_dim)
            } else {
                (config.projection_dim, config.projection_dim)
            };

            let scale = (2.0 / in_dim as f32).sqrt();
            let weight = Array2::from_shape_fn((in_dim, out_dim), |_| rng.next_f32() * scale);
            let bias = Array1::zeros(out_dim);
            layers.push((weight, bias));
        }

        // Layer normalization on the final output
        let gamma = Array1::ones(config.projection_dim);
        let beta = Array1::zeros(config.projection_dim);
        let norm = Some((gamma, beta));

        Ok(Self {
            config,
            layers,
            norm,
        })
    }

    /// Encode an input vector into the shared representation space.
    pub fn encode(&self, input: &Array1<f32>) -> ModelResult<Array1<f32>> {
        if input.len() != self.config.input_dim {
            return Err(ModelError::dimension_mismatch(
                format!("ModalityEncoder({}) input", self.config.modality),
                self.config.input_dim,
                input.len(),
            ));
        }

        let mut x = input.clone();
        for (i, (weight, bias)) in self.layers.iter().enumerate() {
            x = x.dot(weight) + bias;
            // Apply ReLU activation on all but the last layer
            if i + 1 < self.layers.len() {
                x.mapv_inplace(|v| v.max(0.0));
            }
        }

        // Apply layer normalization
        if let Some((gamma, beta)) = &self.norm {
            x = layer_norm_1d(&x, gamma, beta);
        }

        Ok(x)
    }

    /// Get the input dimension.
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Get the output (projection) dimension.
    pub fn output_dim(&self) -> usize {
        self.config.projection_dim
    }
}

/// Simple layer normalization for a 1-D vector.
fn layer_norm_1d(x: &Array1<f32>, gamma: &Array1<f32>, beta: &Array1<f32>) -> Array1<f32> {
    let n = x.len() as f32;
    let mean = x.sum() / n;
    let var = x.mapv(|v| (v - mean).powi(2)).sum() / n;
    let std_inv = 1.0 / (var + 1e-5_f32).sqrt();
    let normalized = x.mapv(|v| (v - mean) * std_inv);
    &normalized * gamma + beta
}

// ---------------------------------------------------------------------------
// Fusion strategy
// ---------------------------------------------------------------------------

/// Fusion strategy for combining multiple modalities
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// Simple concatenation followed by linear projection
    Concatenation,
    /// Element-wise addition (all modalities must share same projection_dim)
    Addition,
    /// Gated fusion: learned sigmoid gates control each modality's contribution
    Gated,
    /// Cross-attention: each modality attends to all others
    CrossAttention {
        /// Number of attention heads
        num_heads: usize,
    },
    /// Bottleneck: concat → compress → expand
    Bottleneck {
        /// Inner bottleneck dimension
        bottleneck_dim: usize,
    },
}

// ---------------------------------------------------------------------------
// Fusion layer
// ---------------------------------------------------------------------------

/// Multi-modal fusion layer implementing various fusion strategies
pub struct FusionLayer {
    strategy: FusionStrategy,
    fusion_dim: usize,
    num_modalities: usize,
    // Concatenation parameters
    concat_proj: Option<Array2<f32>>,
    // Gated fusion parameters
    gate_weights: Option<Vec<Array2<f32>>>,
    // Cross-attention parameters
    attention_q: Option<Vec<Array2<f32>>>,
    attention_k: Option<Vec<Array2<f32>>>,
    attention_v: Option<Vec<Array2<f32>>>,
    // Bottleneck parameters
    bottleneck_down: Option<Array2<f32>>,
    bottleneck_up: Option<Array2<f32>>,
}

impl FusionLayer {
    /// Create a new fusion layer for the given strategy.
    pub fn new(
        strategy: FusionStrategy,
        num_modalities: usize,
        fusion_dim: usize,
    ) -> ModelResult<Self> {
        if num_modalities == 0 {
            return Err(ModelError::invalid_config("num_modalities must be > 0"));
        }
        if fusion_dim == 0 {
            return Err(ModelError::invalid_config("fusion_dim must be > 0"));
        }

        let mut rng = SeededRng::new(1337 + num_modalities as u64 * 11 + fusion_dim as u64 * 3);

        let mut layer = Self {
            strategy: strategy.clone(),
            fusion_dim,
            num_modalities,
            concat_proj: None,
            gate_weights: None,
            attention_q: None,
            attention_k: None,
            attention_v: None,
            bottleneck_down: None,
            bottleneck_up: None,
        };

        match &strategy {
            FusionStrategy::Concatenation => {
                let concat_dim = fusion_dim * num_modalities;
                let scale = (2.0 / concat_dim as f32).sqrt();
                let proj =
                    Array2::from_shape_fn((concat_dim, fusion_dim), |_| rng.next_f32() * scale);
                layer.concat_proj = Some(proj);
            }
            FusionStrategy::Addition => {
                // No extra parameters needed
            }
            FusionStrategy::Gated => {
                let scale = (2.0 / fusion_dim as f32).sqrt();
                let gates: Vec<Array2<f32>> = (0..num_modalities)
                    .map(|_| {
                        Array2::from_shape_fn((fusion_dim, fusion_dim), |_| rng.next_f32() * scale)
                    })
                    .collect();
                layer.gate_weights = Some(gates);
            }
            FusionStrategy::CrossAttention { num_heads } => {
                if !fusion_dim.is_multiple_of(*num_heads) {
                    return Err(ModelError::invalid_config(format!(
                        "fusion_dim ({fusion_dim}) must be divisible by num_heads ({num_heads})"
                    )));
                }
                let scale = (2.0 / fusion_dim as f32).sqrt();
                let make_projs = |rng: &mut SeededRng| -> Vec<Array2<f32>> {
                    (0..num_modalities)
                        .map(|_| {
                            Array2::from_shape_fn((fusion_dim, fusion_dim), |_| {
                                rng.next_f32() * scale
                            })
                        })
                        .collect()
                };
                layer.attention_q = Some(make_projs(&mut rng));
                layer.attention_k = Some(make_projs(&mut rng));
                layer.attention_v = Some(make_projs(&mut rng));
            }
            FusionStrategy::Bottleneck { bottleneck_dim } => {
                if *bottleneck_dim == 0 {
                    return Err(ModelError::invalid_config("bottleneck_dim must be > 0"));
                }
                let concat_dim = fusion_dim * num_modalities;
                let scale_down = (2.0 / concat_dim as f32).sqrt();
                let scale_up = (2.0 / *bottleneck_dim as f32).sqrt();
                layer.bottleneck_down =
                    Some(Array2::from_shape_fn((concat_dim, *bottleneck_dim), |_| {
                        rng.next_f32() * scale_down
                    }));
                layer.bottleneck_up =
                    Some(Array2::from_shape_fn((*bottleneck_dim, fusion_dim), |_| {
                        rng.next_f32() * scale_up
                    }));
            }
        }

        Ok(layer)
    }

    /// Fuse encoded modality representations into a single vector.
    pub fn fuse(&self, encoded_modalities: &[Array1<f32>]) -> ModelResult<Array1<f32>> {
        if encoded_modalities.len() != self.num_modalities {
            return Err(ModelError::dimension_mismatch(
                "FusionLayer modality count",
                self.num_modalities,
                encoded_modalities.len(),
            ));
        }

        // Validate dimensions
        for (i, enc) in encoded_modalities.iter().enumerate() {
            if enc.len() != self.fusion_dim {
                return Err(ModelError::dimension_mismatch(
                    format!("FusionLayer modality {i} dim"),
                    self.fusion_dim,
                    enc.len(),
                ));
            }
        }

        match &self.strategy {
            FusionStrategy::Concatenation => self.fuse_concatenation(encoded_modalities),
            FusionStrategy::Addition => self.fuse_addition(encoded_modalities),
            FusionStrategy::Gated => self.fuse_gated(encoded_modalities),
            FusionStrategy::CrossAttention { num_heads } => {
                self.fuse_cross_attention(encoded_modalities, *num_heads)
            }
            FusionStrategy::Bottleneck { .. } => self.fuse_bottleneck(encoded_modalities),
        }
    }

    fn fuse_concatenation(&self, encoded_modalities: &[Array1<f32>]) -> ModelResult<Array1<f32>> {
        let concat_dim = self.fusion_dim * self.num_modalities;
        let mut concat = Array1::zeros(concat_dim);
        for (i, enc) in encoded_modalities.iter().enumerate() {
            let start = i * self.fusion_dim;
            for (j, &val) in enc.iter().enumerate() {
                concat[start + j] = val;
            }
        }
        let proj = self
            .concat_proj
            .as_ref()
            .ok_or_else(|| ModelError::not_initialized("concat_proj"))?;
        Ok(concat.dot(proj))
    }

    fn fuse_addition(&self, encoded_modalities: &[Array1<f32>]) -> ModelResult<Array1<f32>> {
        let mut result = Array1::zeros(self.fusion_dim);
        for enc in encoded_modalities {
            result += enc;
        }
        Ok(result)
    }

    fn fuse_gated(&self, encoded_modalities: &[Array1<f32>]) -> ModelResult<Array1<f32>> {
        let gate_weights = self
            .gate_weights
            .as_ref()
            .ok_or_else(|| ModelError::not_initialized("gate_weights"))?;

        let mut result = Array1::zeros(self.fusion_dim);
        for (i, enc) in encoded_modalities.iter().enumerate() {
            let pre_gate = enc.dot(&gate_weights[i]);
            let gate = sigmoid(&pre_gate);
            result += &(enc * &gate);
        }
        Ok(result)
    }

    fn fuse_cross_attention(
        &self,
        encoded_modalities: &[Array1<f32>],
        num_heads: usize,
    ) -> ModelResult<Array1<f32>> {
        let q_projs = self
            .attention_q
            .as_ref()
            .ok_or_else(|| ModelError::not_initialized("attention_q"))?;
        let k_projs = self
            .attention_k
            .as_ref()
            .ok_or_else(|| ModelError::not_initialized("attention_k"))?;
        let v_projs = self
            .attention_v
            .as_ref()
            .ok_or_else(|| ModelError::not_initialized("attention_v"))?;

        let head_dim = self.fusion_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n = self.num_modalities;

        // For each modality, compute attention over all other modalities and sum
        let mut fused = Array1::zeros(self.fusion_dim);

        for i in 0..n {
            let q = encoded_modalities[i].dot(&q_projs[i]);

            // Compute attention scores against all modalities (including self)
            let mut attn_output: Array1<f32> = Array1::zeros(self.fusion_dim);

            // Per-head attention
            for h in 0..num_heads {
                let h_start = h * head_dim;
                let h_end = h_start + head_dim;

                let q_h = q.slice(scirs2_core::ndarray::s![h_start..h_end]);

                // Compute scores for all modalities
                let mut scores = Vec::with_capacity(n);
                let mut values = Vec::with_capacity(n);
                for j in 0..n {
                    let k = encoded_modalities[j].dot(&k_projs[j]);
                    let v = encoded_modalities[j].dot(&v_projs[j]);
                    let k_h = k.slice(scirs2_core::ndarray::s![h_start..h_end]);
                    let score = q_h.dot(&k_h) * scale;
                    scores.push(score);
                    values.push(v.slice(scirs2_core::ndarray::s![h_start..h_end]).to_owned());
                }

                // Softmax over scores
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let sum_exp_safe = if sum_exp.abs() < 1e-10 {
                    1e-10
                } else {
                    sum_exp
                };

                // Weighted sum of values
                for (j, v_h) in values.iter().enumerate() {
                    let weight = exp_scores[j] / sum_exp_safe;
                    for (k, &val) in v_h.iter().enumerate() {
                        attn_output[h_start + k] += weight * val;
                    }
                }
            }

            fused = fused + attn_output;
        }

        // Average over modalities
        let divisor = n as f32;
        fused.mapv_inplace(|v| v / divisor);

        Ok(fused)
    }

    fn fuse_bottleneck(&self, encoded_modalities: &[Array1<f32>]) -> ModelResult<Array1<f32>> {
        let down = self
            .bottleneck_down
            .as_ref()
            .ok_or_else(|| ModelError::not_initialized("bottleneck_down"))?;
        let up = self
            .bottleneck_up
            .as_ref()
            .ok_or_else(|| ModelError::not_initialized("bottleneck_up"))?;

        // Concatenate
        let concat_dim = self.fusion_dim * self.num_modalities;
        let mut concat = Array1::zeros(concat_dim);
        for (i, enc) in encoded_modalities.iter().enumerate() {
            let start = i * self.fusion_dim;
            for (j, &val) in enc.iter().enumerate() {
                concat[start + j] = val;
            }
        }

        // Down-project → ReLU → Up-project
        let bottleneck = concat.dot(down);
        let activated = bottleneck.mapv(|v| v.max(0.0));
        Ok(activated.dot(up))
    }
}

// ---------------------------------------------------------------------------
// Multi-modal configuration and model
// ---------------------------------------------------------------------------

/// Configuration for the multi-modal fusion model
#[derive(Debug, Clone)]
pub struct MultiModalConfig {
    /// Shared representation dimension (all encoders project to this)
    pub fusion_dim: usize,
    /// Fusion strategy
    pub fusion_strategy: FusionStrategy,
    /// Final output dimension
    pub output_dim: usize,
    /// Per-modality encoder configurations
    pub modalities: Vec<ModalityEncoderConfig>,
    /// Context window length
    pub context_length: usize,
}

/// Complete multi-modal model combining encoders, fusion, and output projection
pub struct MultiModalModel {
    /// Model configuration
    pub config: MultiModalConfig,
    encoders: Vec<ModalityEncoder>,
    fusion: FusionLayer,
    output_proj: Array2<f32>,
    output_bias: Array1<f32>,
    /// Fused hidden state
    state: Array1<f32>,
}

impl MultiModalModel {
    /// Create a new multi-modal model from configuration.
    pub fn new(config: MultiModalConfig) -> ModelResult<Self> {
        if config.modalities.is_empty() {
            return Err(ModelError::invalid_config(
                "at least one modality is required",
            ));
        }
        if config.fusion_dim == 0 {
            return Err(ModelError::invalid_config("fusion_dim must be > 0"));
        }
        if config.output_dim == 0 {
            return Err(ModelError::invalid_config("output_dim must be > 0"));
        }
        if config.context_length == 0 {
            return Err(ModelError::invalid_config("context_length must be > 0"));
        }

        // Validate that all modalities project to fusion_dim
        for mc in &config.modalities {
            if mc.projection_dim != config.fusion_dim {
                return Err(ModelError::invalid_config(format!(
                    "modality {} projection_dim ({}) must match fusion_dim ({})",
                    mc.modality, mc.projection_dim, config.fusion_dim
                )));
            }
        }

        // Build encoders
        let encoders: Vec<ModalityEncoder> = config
            .modalities
            .iter()
            .map(|mc| ModalityEncoder::new(mc.clone()))
            .collect::<ModelResult<Vec<_>>>()?;

        // Build fusion layer
        let fusion = FusionLayer::new(
            config.fusion_strategy.clone(),
            config.modalities.len(),
            config.fusion_dim,
        )?;

        // Output projection
        let mut rng = SeededRng::new(99 + config.fusion_dim as u64 * 5);
        let scale = (2.0 / config.fusion_dim as f32).sqrt();
        let output_proj = Array2::from_shape_fn((config.fusion_dim, config.output_dim), |_| {
            rng.next_f32() * scale
        });
        let output_bias = Array1::zeros(config.output_dim);

        let state = Array1::zeros(config.fusion_dim);

        Ok(Self {
            config,
            encoders,
            fusion,
            output_proj,
            output_bias,
            state,
        })
    }

    /// Forward pass with all modality inputs present.
    ///
    /// `inputs` must contain one `Array1<f32>` per modality, in the same order
    /// as `config.modalities`.
    pub fn forward_multimodal(&mut self, inputs: &[Array1<f32>]) -> ModelResult<Array1<f32>> {
        if inputs.len() != self.encoders.len() {
            return Err(ModelError::dimension_mismatch(
                "MultiModalModel input count",
                self.encoders.len(),
                inputs.len(),
            ));
        }

        // Encode each modality
        let encoded: Vec<Array1<f32>> = self
            .encoders
            .iter()
            .zip(inputs.iter())
            .map(|(enc, inp)| enc.encode(inp))
            .collect::<ModelResult<Vec<_>>>()?;

        // Fuse
        let fused = self.fusion.fuse(&encoded)?;

        // Check for NaN / Inf
        if fused.iter().any(|v| v.is_nan() || v.is_infinite()) {
            return Err(ModelError::numerical_instability(
                "forward_multimodal",
                "NaN or Inf detected after fusion",
            ));
        }

        // Update state
        self.state = fused.clone();

        // Output projection
        let output = fused.dot(&self.output_proj) + &self.output_bias;
        Ok(output)
    }

    /// Forward pass with optional modalities.
    ///
    /// Missing modalities (None) are replaced with zero vectors.
    pub fn forward_with_missing(
        &mut self,
        inputs: &[Option<Array1<f32>>],
    ) -> ModelResult<Array1<f32>> {
        if inputs.len() != self.encoders.len() {
            return Err(ModelError::dimension_mismatch(
                "MultiModalModel input count",
                self.encoders.len(),
                inputs.len(),
            ));
        }

        // Encode each present modality; use zeros for missing ones
        let encoded: Vec<Array1<f32>> = self
            .encoders
            .iter()
            .zip(inputs.iter())
            .map(|(enc, maybe_inp)| match maybe_inp {
                Some(inp) => enc.encode(inp),
                None => Ok(Array1::zeros(enc.output_dim())),
            })
            .collect::<ModelResult<Vec<_>>>()?;

        // Fuse
        let fused = self.fusion.fuse(&encoded)?;

        if fused.iter().any(|v| v.is_nan() || v.is_infinite()) {
            return Err(ModelError::numerical_instability(
                "forward_with_missing",
                "NaN or Inf detected after fusion",
            ));
        }

        self.state = fused.clone();

        let output = fused.dot(&self.output_proj) + &self.output_bias;
        Ok(output)
    }

    /// Number of modalities.
    pub fn num_modalities(&self) -> usize {
        self.encoders.len()
    }

    /// References to each modality identifier.
    pub fn modality_names(&self) -> Vec<&Modality> {
        self.config
            .modalities
            .iter()
            .map(|mc| &mc.modality)
            .collect()
    }

    /// Total trainable parameter count.
    pub fn total_params(&self) -> usize {
        let mut count = 0usize;

        // Encoder parameters
        for enc in &self.encoders {
            for (w, b) in &enc.layers {
                count += w.len() + b.len();
            }
            if let Some((g, b)) = &enc.norm {
                count += g.len() + b.len();
            }
        }

        // Fusion parameters
        if let Some(p) = &self.fusion.concat_proj {
            count += p.len();
        }
        if let Some(gates) = &self.fusion.gate_weights {
            for g in gates {
                count += g.len();
            }
        }
        if let Some(qs) = &self.fusion.attention_q {
            for q in qs {
                count += q.len();
            }
        }
        if let Some(ks) = &self.fusion.attention_k {
            for k in ks {
                count += k.len();
            }
        }
        if let Some(vs) = &self.fusion.attention_v {
            for v in vs {
                count += v.len();
            }
        }
        if let Some(d) = &self.fusion.bottleneck_down {
            count += d.len();
        }
        if let Some(u) = &self.fusion.bottleneck_up {
            count += u.len();
        }

        // Output projection
        count += self.output_proj.len() + self.output_bias.len();

        count
    }
}

impl SignalPredictor for MultiModalModel {
    /// Step with a single concatenated input.
    ///
    /// The input is split into per-modality chunks based on each encoder's
    /// `input_dim`, concatenated in the order of `config.modalities`.
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Split input into per-modality slices
        let total_input_dim: usize = self.encoders.iter().map(|e| e.input_dim()).sum();

        if input.len() != total_input_dim {
            return Err(kizzasi_core::CoreError::DimensionMismatch {
                expected: total_input_dim,
                got: input.len(),
            });
        }

        let mut offset = 0;
        let mut per_modality = Vec::with_capacity(self.encoders.len());
        for enc in &self.encoders {
            let dim = enc.input_dim();
            let slice = input
                .slice(scirs2_core::ndarray::s![offset..offset + dim])
                .to_owned();
            per_modality.push(slice);
            offset += dim;
        }

        self.forward_multimodal(&per_modality)
            .map_err(|e| kizzasi_core::CoreError::Generic(e.to_string()))
    }

    #[instrument(skip(self))]
    fn reset(&mut self) {
        debug!("Resetting MultiModalModel state");
        self.state = Array1::zeros(self.config.fusion_dim);
    }

    fn context_window(&self) -> usize {
        self.config.context_length
    }
}

impl AutoregressiveModel for MultiModalModel {
    fn hidden_dim(&self) -> usize {
        self.config.fusion_dim
    }

    fn state_dim(&self) -> usize {
        self.config.fusion_dim
    }

    fn num_layers(&self) -> usize {
        // 1 fusion layer
        1
    }

    fn model_type(&self) -> ModelType {
        ModelType::MultiModal
    }

    fn get_states(&self) -> Vec<HiddenState> {
        // `self.state` is the real fused hidden vector from the last
        // forward pass (see `forward_multimodal`/`forward_with_missing`),
        // stored as a `(fusion_dim, 1)` column so it round-trips through
        // `set_states` exactly.
        let mut hs = HiddenState::new(self.config.fusion_dim, 1);
        {
            let mat = hs.state_mut();
            for (i, &v) in self.state.iter().enumerate() {
                mat[[i, 0]] = v;
            }
        }
        vec![hs]
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != 1 {
            return Err(ModelError::state_count_mismatch(
                "MultiModal",
                1,
                states.len(),
            ));
        }
        let mat = states[0].state();
        if mat.shape() != [self.config.fusion_dim, 1] {
            return Err(ModelError::dimension_mismatch(
                "MultiModalModel::set_states",
                self.config.fusion_dim,
                mat.shape().first().copied().unwrap_or(0),
            ));
        }
        for i in 0..self.config.fusion_dim {
            self.state[i] = mat[[i, 0]];
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Modality aligner
// ---------------------------------------------------------------------------

/// Synchronizes multiple modality streams with different sample rates.
///
/// The aligner buffers incoming samples and produces aligned frames only when
/// all modalities have accumulated enough data relative to the reference rate.
pub struct ModalityAligner {
    reference_rate: f32,
    modality_rates: Vec<f32>,
    buffers: Vec<Vec<Array1<f32>>>,
}

impl ModalityAligner {
    /// Create a new aligner with the given reference and per-modality sample rates.
    pub fn new(reference_rate: f32, modality_rates: Vec<f32>) -> Self {
        let buffers = modality_rates.iter().map(|_| Vec::new()).collect();
        Self {
            reference_rate,
            modality_rates,
            buffers,
        }
    }

    /// Push a sample for the given modality.
    pub fn push(&mut self, modality_idx: usize, sample: Array1<f32>) {
        if modality_idx < self.buffers.len() {
            self.buffers[modality_idx].push(sample);
        }
    }

    /// Try to produce an aligned frame.
    ///
    /// Returns `Some(vec)` containing one sample per modality when all
    /// modalities have buffered enough data. Otherwise returns `None`.
    pub fn try_align(&mut self) -> Option<Vec<Array1<f32>>> {
        // For each modality, determine how many samples are needed per
        // reference frame: ceil(modality_rate / reference_rate).
        // When all modalities have at least that many samples, we take the
        // most recent sample from each and drain the consumed portion.

        let mut required: Vec<usize> = Vec::with_capacity(self.modality_rates.len());
        for rate in &self.modality_rates {
            let ratio = rate / self.reference_rate;
            let need = ratio.ceil().max(1.0) as usize;
            required.push(need);
        }

        // Check if all modalities have enough buffered data
        for (i, &need) in required.iter().enumerate() {
            if self.buffers[i].len() < need {
                return None;
            }
        }

        // Consume and return the last sample of each consumed chunk
        let mut aligned = Vec::with_capacity(self.buffers.len());
        for (i, &need) in required.iter().enumerate() {
            // Take the last sample from the consumed window
            let sample = self.buffers[i][need - 1].clone();
            // Drain consumed samples
            self.buffers[i].drain(..need);
            aligned.push(sample);
        }

        Some(aligned)
    }

    /// Clear all internal buffers.
    pub fn clear(&mut self) {
        for buf in &mut self.buffers {
            buf.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder_config(
        modality: Modality,
        input_dim: usize,
        proj_dim: usize,
    ) -> ModalityEncoderConfig {
        ModalityEncoderConfig {
            modality,
            input_dim,
            projection_dim: proj_dim,
            num_layers: 2,
        }
    }

    fn make_default_config() -> MultiModalConfig {
        MultiModalConfig {
            fusion_dim: 16,
            fusion_strategy: FusionStrategy::Addition,
            output_dim: 4,
            modalities: vec![
                make_encoder_config(Modality::Audio, 8, 16),
                make_encoder_config(Modality::Vision, 12, 16),
                make_encoder_config(Modality::Sensor, 6, 16),
            ],
            context_length: 512,
        }
    }

    // 1. test_modality_encoder_creation
    #[test]
    fn test_modality_encoder_creation() {
        let cfg = make_encoder_config(Modality::Audio, 8, 16);
        let enc = ModalityEncoder::new(cfg).expect("failed to create encoder");
        assert_eq!(enc.input_dim(), 8);
        assert_eq!(enc.output_dim(), 16);
    }

    // 2. test_modality_encoder_forward
    #[test]
    fn test_modality_encoder_forward() {
        let cfg = make_encoder_config(Modality::Vision, 12, 16);
        let enc = ModalityEncoder::new(cfg).expect("failed to create encoder");
        let input = Array1::from_vec(vec![0.1; 12]);
        let output = enc.encode(&input).expect("encode failed");
        assert_eq!(output.len(), 16);
        // Output should be finite
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // 3. test_fusion_concatenation
    #[test]
    fn test_fusion_concatenation() {
        let fusion_dim = 8;
        let n = 3;
        let layer = FusionLayer::new(FusionStrategy::Concatenation, n, fusion_dim)
            .expect("failed to create fusion layer");
        let inputs: Vec<Array1<f32>> = (0..n)
            .map(|_| Array1::from_vec(vec![0.5; fusion_dim]))
            .collect();
        let out = layer.fuse(&inputs).expect("fuse failed");
        assert_eq!(out.len(), fusion_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 4. test_fusion_addition
    #[test]
    fn test_fusion_addition() {
        let fusion_dim = 8;
        let n = 3;
        let layer = FusionLayer::new(FusionStrategy::Addition, n, fusion_dim)
            .expect("failed to create fusion layer");
        let inputs: Vec<Array1<f32>> = (0..n).map(|_| Array1::ones(fusion_dim)).collect();
        let out = layer.fuse(&inputs).expect("fuse failed");
        assert_eq!(out.len(), fusion_dim);
        // Each element should be 3.0 (1+1+1)
        for &v in out.iter() {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    // 5. test_fusion_gated
    #[test]
    fn test_fusion_gated() {
        let fusion_dim = 8;
        let n = 2;
        let layer = FusionLayer::new(FusionStrategy::Gated, n, fusion_dim)
            .expect("failed to create fusion layer");
        let inputs: Vec<Array1<f32>> = (0..n)
            .map(|_| Array1::from_vec(vec![0.3; fusion_dim]))
            .collect();
        let out = layer.fuse(&inputs).expect("fuse failed");
        assert_eq!(out.len(), fusion_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 6. test_fusion_cross_attention
    #[test]
    fn test_fusion_cross_attention() {
        let fusion_dim = 8;
        let n = 2;
        let layer = FusionLayer::new(
            FusionStrategy::CrossAttention { num_heads: 2 },
            n,
            fusion_dim,
        )
        .expect("failed to create fusion layer");
        let inputs: Vec<Array1<f32>> = (0..n)
            .map(|_| Array1::from_vec(vec![0.2; fusion_dim]))
            .collect();
        let out = layer.fuse(&inputs).expect("fuse failed");
        assert_eq!(out.len(), fusion_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 7. test_fusion_bottleneck
    #[test]
    fn test_fusion_bottleneck() {
        let fusion_dim = 8;
        let n = 3;
        let layer = FusionLayer::new(
            FusionStrategy::Bottleneck { bottleneck_dim: 4 },
            n,
            fusion_dim,
        )
        .expect("failed to create fusion layer");
        let inputs: Vec<Array1<f32>> = (0..n)
            .map(|_| Array1::from_vec(vec![0.4; fusion_dim]))
            .collect();
        let out = layer.fuse(&inputs).expect("fuse failed");
        assert_eq!(out.len(), fusion_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 8. test_multimodal_model_creation
    #[test]
    fn test_multimodal_model_creation() {
        let config = make_default_config();
        let model = MultiModalModel::new(config).expect("failed to create model");
        assert_eq!(model.num_modalities(), 3);
        assert_eq!(model.modality_names().len(), 3);
        assert!(model.total_params() > 0);
    }

    // 9. test_multimodal_forward
    #[test]
    fn test_multimodal_forward() {
        let config = make_default_config();
        let mut model = MultiModalModel::new(config).expect("failed to create model");

        let audio = Array1::from_vec(vec![0.1; 8]);
        let vision = Array1::from_vec(vec![0.2; 12]);
        let sensor = Array1::from_vec(vec![0.3; 6]);

        let out = model
            .forward_multimodal(&[audio, vision, sensor])
            .expect("forward failed");
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 10. test_multimodal_missing_modalities
    #[test]
    fn test_multimodal_missing_modalities() {
        let config = make_default_config();
        let mut model = MultiModalModel::new(config).expect("failed to create model");

        let audio = Some(Array1::from_vec(vec![0.1; 8]));
        let vision = None; // missing
        let sensor = Some(Array1::from_vec(vec![0.3; 6]));

        let out = model
            .forward_with_missing(&[audio, vision, sensor])
            .expect("forward_with_missing failed");
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 11. test_multimodal_signal_predictor
    #[test]
    fn test_multimodal_signal_predictor() {
        let config = make_default_config();
        let mut model = MultiModalModel::new(config).expect("failed to create model");

        // Total input dim = 8 + 12 + 6 = 26
        let input = Array1::from_vec(vec![0.1; 26]);
        let out = model.step(&input).expect("step failed");
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));

        // Test reset
        model.reset();
        assert_eq!(model.context_window(), 512);
    }

    // 12. test_modality_aligner
    #[test]
    fn test_modality_aligner() {
        // Reference rate 10 Hz, modality A at 10 Hz, modality B at 20 Hz
        let mut aligner = ModalityAligner::new(10.0, vec![10.0, 20.0]);

        // Push one sample for modality A
        aligner.push(0, Array1::from_vec(vec![1.0, 2.0]));
        // Not enough for modality B yet (needs ceil(20/10) = 2 samples)
        aligner.push(1, Array1::from_vec(vec![3.0, 4.0]));
        assert!(aligner.try_align().is_none());

        // Push second sample for modality B
        aligner.push(1, Array1::from_vec(vec![5.0, 6.0]));
        let aligned = aligner.try_align().expect("should have aligned frame");
        assert_eq!(aligned.len(), 2);
        // Modality A: the first sample [1, 2]
        assert!((aligned[0][0] - 1.0).abs() < 1e-6);
        // Modality B: the last consumed sample [5, 6]
        assert!((aligned[1][0] - 5.0).abs() < 1e-6);

        // After consuming, buffers should be empty
        assert!(aligner.try_align().is_none());
    }

    // 13. test_multimodal_numerical_stability
    #[test]
    fn test_multimodal_numerical_stability() {
        let config = make_default_config();
        let mut model = MultiModalModel::new(config).expect("failed to create model");

        // Test with very large inputs
        let audio_large = Array1::from_vec(vec![1e6; 8]);
        let vision_large = Array1::from_vec(vec![1e6; 12]);
        let sensor_large = Array1::from_vec(vec![1e6; 6]);
        let out = model.forward_multimodal(&[audio_large, vision_large, sensor_large]);
        // Should either succeed with finite values or return a numerical error
        match out {
            Ok(o) => assert!(o.iter().all(|v| v.is_finite()), "output should be finite"),
            Err(ModelError::NumericalInstability { .. }) => {
                // acceptable
            }
            Err(e) => panic!("unexpected error: {e}"),
        }

        // Test with very small inputs
        let audio_small = Array1::from_vec(vec![1e-30; 8]);
        let vision_small = Array1::from_vec(vec![1e-30; 12]);
        let sensor_small = Array1::from_vec(vec![1e-30; 6]);
        let out = model
            .forward_multimodal(&[audio_small, vision_small, sensor_small])
            .expect("small inputs should not cause errors");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output should be finite for small inputs"
        );
    }

    // 14. test_autoregressive_model_trait
    #[test]
    fn test_autoregressive_model_trait() {
        let config = make_default_config();
        let model = MultiModalModel::new(config).expect("failed to create model");
        assert_eq!(model.hidden_dim(), 16);
        assert_eq!(model.state_dim(), 16);
        assert_eq!(model.num_layers(), 1);
        assert_eq!(model.model_type(), ModelType::MultiModal);
        let states = model.get_states();
        assert_eq!(states.len(), 1);
    }

    #[test]
    fn test_get_set_states_round_trip_real_state() {
        let config = make_default_config();
        let mut model = MultiModalModel::new(config).expect("failed to create model");

        // Before any step, `self.state` is all zeros, so exercise a step
        // first so `get_states` has non-trivial data to actually capture.
        let inputs: Vec<Array1<f32>> = vec![
            Array1::from_vec(vec![0.3; 8]),
            Array1::from_vec(vec![0.6; 12]),
            Array1::from_vec(vec![0.9; 6]),
        ];
        model
            .forward_multimodal(&inputs)
            .expect("forward_multimodal failed");

        // The old stub returned a freshly-zeroed HiddenState regardless of
        // the model's real fused state; assert get_states reflects the
        // model's actual (non-zero) internal state.
        assert!(
            model.state.iter().any(|&v| v != 0.0),
            "test setup should have produced a non-zero fused state"
        );
        let states = model.get_states();
        assert_eq!(states.len(), 1);
        let captured: Vec<f32> = (0..model.config.fusion_dim)
            .map(|i| states[0].state()[[i, 0]])
            .collect();
        for (i, &v) in captured.iter().enumerate() {
            assert!(
                (v - model.state[i]).abs() < 1e-7,
                "get_states must capture the real fused state, not zeros"
            );
        }

        // Corrupt the live state, then restore from the snapshot and check
        // set_states actually writes it back (old stub was a silent no-op).
        model.state.fill(-1.0);
        model
            .set_states(states)
            .expect("set_states should accept its own get_states output");
        for (i, &v) in captured.iter().enumerate() {
            assert!(
                (model.state[i] - v).abs() < 1e-7,
                "set_states must restore the exact captured state"
            );
        }

        // Wrong element count -> Err (existing behavior, still required).
        assert!(model.set_states(vec![]).is_err());

        // Right count, wrong inner shape -> Err (this is the part the old
        // no-op stub could never detect, since it never looked at the
        // HiddenState's contents at all).
        let bad = HiddenState::new(model.config.fusion_dim + 1, 1);
        assert!(model.set_states(vec![bad]).is_err());
    }

    // 15. test_modality_display
    #[test]
    fn test_modality_display() {
        assert_eq!(format!("{}", Modality::Audio), "Audio");
        assert_eq!(format!("{}", Modality::Vision), "Vision");
        assert_eq!(
            format!("{}", Modality::Custom("Lidar".to_string())),
            "Custom(Lidar)"
        );
    }

    // 16. test_encoder_dimension_mismatch
    #[test]
    fn test_encoder_dimension_mismatch() {
        let cfg = make_encoder_config(Modality::Audio, 8, 16);
        let enc = ModalityEncoder::new(cfg).expect("failed to create encoder");
        let bad_input = Array1::from_vec(vec![0.1; 5]); // wrong dim
        assert!(enc.encode(&bad_input).is_err());
    }

    // 17. test_aligner_clear
    #[test]
    fn test_aligner_clear() {
        let mut aligner = ModalityAligner::new(10.0, vec![10.0]);
        aligner.push(0, Array1::from_vec(vec![1.0]));
        aligner.clear();
        assert!(aligner.try_align().is_none());
    }
}
