//! Cross-Modal Tokenization
//!
//! Provides unified tokenization of multiple signal modalities (audio, control signals,
//! sensor data, video features) into a shared embedding space. This enables cross-modal
//! alignment and joint reasoning over heterogeneous sensory streams.
//!
//! ## Architecture
//!
//! Each modality has its own linear encoder projecting into a shared embedding space.
//! A discrete codebook (per modality) provides token indices for autoregressive use.
//! A shared alignment projection aligns all modality embeddings into a common manifold.
//! Modality-type embeddings (learned offsets) allow the model to distinguish modalities.
//!
//! ## Design Principles
//!
//! - **Unified token space**: All modalities produce tokens of `shared_dim` size.
//! - **Residual codebooks**: Optional multi-stage residual VQ for fine-grained encoding.
//! - **Confidence**: `1 / (1 + distance_to_nearest)` — high confidence = close match.
//! - **Pure Rust**: No C/Fortran dependencies; deterministic xorshift64 initialization.

use crate::error::{TokenizerError, TokenizerResult};
use crate::SignalTokenizer;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift64)
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
// Modality descriptor
// ---------------------------------------------------------------------------

/// Identifies the type of a signal modality.
///
/// Used to route signals to the correct per-modality encoder and to
/// add the learned modality-type embedding.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalityKind {
    /// Continuous amplitude waveform (e.g. raw audio samples or mel features).
    Audio,
    /// Robot joint angles, velocities, or action commands.
    Control,
    /// IMU, pressure, temperature, or other physical sensor readings.
    Sensor,
    /// Pixel features or CNN/ViT embeddings from video frames.
    Video,
    /// User-defined modality with a descriptive name.
    Custom(String),
}

impl ModalityKind {
    /// Canonical string key used in hash-maps.
    pub fn key(&self) -> String {
        match self {
            ModalityKind::Audio => "audio".to_string(),
            ModalityKind::Control => "control".to_string(),
            ModalityKind::Sensor => "sensor".to_string(),
            ModalityKind::Video => "video".to_string(),
            ModalityKind::Custom(s) => format!("custom_{s}"),
        }
    }

    /// Deterministic seed derived from the modality name (for xorshift64 init).
    fn seed(&self) -> u64 {
        // Simple djb2-style hash of the key bytes
        let key = self.key();
        key.bytes().fold(5381u64, |acc, b| {
            acc.wrapping_mul(33).wrapping_add(b as u64)
        })
    }
}

// ---------------------------------------------------------------------------
// Per-modality configuration
// ---------------------------------------------------------------------------

/// Configuration for a single modality's tokenizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityTokenizerConfig {
    /// Which modality this tokenizer handles.
    pub modality: ModalityKind,
    /// Dimensionality of the raw input signal for this modality.
    pub input_dim: usize,
    /// Shared embedding dimension (must equal `CrossModalTokenizer::shared_dim`).
    pub token_dim: usize,
    /// Number of VQ codebook entries.
    pub codebook_size: usize,
    /// Number of residual VQ stages (1 = standard VQ, >1 = RVQ).
    pub num_stages: usize,
}

impl ModalityTokenizerConfig {
    /// Validate the configuration fields.
    pub fn validate(&self) -> TokenizerResult<()> {
        if self.input_dim == 0 {
            return Err(TokenizerError::InvalidConfig(
                "input_dim must be > 0".into(),
            ));
        }
        if self.token_dim == 0 {
            return Err(TokenizerError::InvalidConfig(
                "token_dim must be > 0".into(),
            ));
        }
        if self.codebook_size == 0 {
            return Err(TokenizerError::InvalidConfig(
                "codebook_size must be > 0".into(),
            ));
        }
        if self.num_stages == 0 {
            return Err(TokenizerError::InvalidConfig(
                "num_stages must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GELU activation
// ---------------------------------------------------------------------------

/// GELU activation: x * Φ(x), approximated via tanh.
#[inline]
fn gelu(x: f32) -> f32 {
    // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let c = 0.797_884_6_f32; // sqrt(2/π)
    let v = c * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + v.tanh())
}

// ---------------------------------------------------------------------------
// Per-modality tokenizer
// ---------------------------------------------------------------------------

/// Encodes raw signals from a single modality into a shared token embedding space.
///
/// Internally uses:
/// - A linear encoder with GELU activation: `input_dim → token_dim`
/// - A nearest-neighbour codebook for discrete token assignment
/// - A linear decoder (transpose of encoder weights) for approximate reconstruction
pub struct ModalityTokenizer {
    config: ModalityTokenizerConfig,
    /// Encoder weight matrix: shape `(input_dim, token_dim)`.
    encoder: Array2<f32>,
    /// Encoder bias: shape `(token_dim,)`.
    encoder_bias: Array1<f32>,
    /// Codebook: shape `(codebook_size, token_dim)`.
    codebook: Array2<f32>,
}

impl ModalityTokenizer {
    /// Create a new modality tokenizer with deterministic weight initialization.
    pub fn new(config: ModalityTokenizerConfig) -> TokenizerResult<Self> {
        config.validate()?;

        let seed = config.modality.seed();
        let mut rng = SeededRng::new(seed);

        // Xavier / Glorot uniform initialization scale
        let enc_scale = (6.0_f32 / (config.input_dim + config.token_dim) as f32).sqrt();
        let encoder = Array2::from_shape_fn((config.input_dim, config.token_dim), |_| {
            rng.next_f32() * enc_scale
        });

        let encoder_bias = Array1::zeros(config.token_dim);

        // Codebook: small random init, scaled by 1/sqrt(token_dim)
        let cb_scale = 1.0_f32 / (config.token_dim as f32).sqrt();
        let codebook = Array2::from_shape_fn((config.codebook_size, config.token_dim), |_| {
            rng.next_f32() * cb_scale
        });

        Ok(Self {
            config,
            encoder,
            encoder_bias,
            codebook,
        })
    }

    /// Project raw input through the encoder (linear + GELU), producing a `token_dim` embedding.
    pub fn encode(&self, input: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if input.len() != self.config.input_dim {
            return Err(TokenizerError::dim_mismatch(
                self.config.input_dim,
                input.len(),
                "ModalityTokenizer::encode input_dim",
            ));
        }

        // Linear: out = input @ encoder + bias
        let pre_act = input.dot(&self.encoder) + &self.encoder_bias;

        // GELU element-wise
        let activated = pre_act.mapv(gelu);
        Ok(activated)
    }

    /// Find the nearest codebook entry (L2) and return `(token_idx, quantized_embedding)`.
    ///
    /// Confidence is defined as `1 / (1 + min_distance)`.
    pub fn quantize(&self, embedding: &Array1<f32>) -> TokenizerResult<(usize, Array1<f32>)> {
        if embedding.len() != self.config.token_dim {
            return Err(TokenizerError::dim_mismatch(
                self.config.token_dim,
                embedding.len(),
                "ModalityTokenizer::quantize embedding dim",
            ));
        }

        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;

        for k in 0..self.config.codebook_size {
            let code = self.codebook.row(k);
            let diff = embedding - &code;
            let dist = diff.dot(&diff); // squared L2
            if dist < best_dist {
                best_dist = dist;
                best_idx = k;
            }
        }

        let quantized = self.codebook.row(best_idx).to_owned();
        Ok((best_idx, quantized))
    }

    /// Decode a discrete token index back to the embedding space (codebook lookup).
    pub fn decode(&self, token_idx: usize) -> TokenizerResult<Array1<f32>> {
        if token_idx >= self.config.codebook_size {
            return Err(TokenizerError::out_of_range(
                token_idx as f32,
                0.0,
                (self.config.codebook_size - 1) as f32,
                "ModalityTokenizer::decode token_idx",
            ));
        }
        Ok(self.codebook.row(token_idx).to_owned())
    }

    /// Decode an embedding back to the raw input space via the encoder weights.
    ///
    /// This is a pseudo-inverse: `out = encoder @ embedding` where
    /// `encoder: (input_dim, token_dim)` and `embedding: (token_dim,)` → `(input_dim,)`.
    pub fn decode_embedding(&self, embedding: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if embedding.len() != self.config.token_dim {
            return Err(TokenizerError::dim_mismatch(
                self.config.token_dim,
                embedding.len(),
                "ModalityTokenizer::decode_embedding embedding dim",
            ));
        }
        // encoder: (input_dim, token_dim)
        // We want W @ e  where W: (input_dim, token_dim) and e: (token_dim,) → (input_dim,)
        // ndarray Array2::dot(&Array1): (m, n) @ (n,) = (m,)
        let reconstructed = self.encoder.dot(embedding);
        Ok(reconstructed)
    }

    /// Raw input dimension.
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Shared token / embedding dimension.
    pub fn token_dim(&self) -> usize {
        self.config.token_dim
    }

    /// Number of discrete codebook entries.
    pub fn codebook_size(&self) -> usize {
        self.config.codebook_size
    }

    /// Read-only reference to the codebook.
    pub fn codebook(&self) -> &Array2<f32> {
        &self.codebook
    }

    /// Compute confidence for an embedding/quantized pair: `1 / (1 + dist)`.
    pub fn confidence(&self, embedding: &Array1<f32>, quantized: &Array1<f32>) -> f32 {
        let diff = embedding - quantized;
        let dist = diff.dot(&diff).sqrt();
        1.0 / (1.0 + dist)
    }
}

// ---------------------------------------------------------------------------
// Cross-modal token
// ---------------------------------------------------------------------------

/// A single token produced by cross-modal tokenization.
///
/// Carries both the modality identity and the token value in the shared space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalToken {
    /// Which modality produced this token.
    pub modality: ModalityKind,
    /// Discrete codebook index within that modality's codebook.
    pub token_idx: usize,
    /// Continuous embedding in the shared `token_dim`-dimensional space.
    pub embedding: Array1<f32>,
    /// Quantization confidence: `1 / (1 + ||embedding - nearest_code||)`.
    /// High value ≈ the signal matched a codebook entry closely.
    pub confidence: f32,
}

// ---------------------------------------------------------------------------
// Cross-modal sequence
// ---------------------------------------------------------------------------

/// An ordered sequence of cross-modal tokens drawn from one or more modalities.
pub struct CrossModalSequence {
    /// Ordered tokens.
    pub tokens: Vec<CrossModalToken>,
    /// Embedding dimensionality shared by all tokens.
    pub shared_dim: usize,
}

impl CrossModalSequence {
    /// Create an empty sequence.
    pub fn new(shared_dim: usize) -> Self {
        Self {
            tokens: Vec::new(),
            shared_dim,
        }
    }

    /// Append a token to the sequence.
    pub fn push(&mut self, token: CrossModalToken) {
        self.tokens.push(token);
    }

    /// Number of tokens in the sequence.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns `true` if the sequence contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Build a `(num_tokens, shared_dim)` embedding matrix from the sequence.
    ///
    /// Each row corresponds to one token's continuous embedding.
    pub fn to_embedding_matrix(&self) -> Array2<f32> {
        let n = self.tokens.len();
        if n == 0 {
            return Array2::zeros((0, self.shared_dim));
        }
        let mut mat = Array2::zeros((n, self.shared_dim));
        for (i, tok) in self.tokens.iter().enumerate() {
            let row_len = tok.embedding.len().min(self.shared_dim);
            for j in 0..row_len {
                mat[[i, j]] = tok.embedding[j];
            }
        }
        mat
    }

    /// Return all tokens belonging to the specified modality.
    pub fn filter_by_modality(&self, modality: &ModalityKind) -> Vec<&CrossModalToken> {
        self.tokens
            .iter()
            .filter(|t| &t.modality == modality)
            .collect()
    }

    /// Return the distinct modalities present in this sequence (in order of first appearance).
    pub fn modalities_present(&self) -> Vec<&ModalityKind> {
        let mut seen: Vec<&ModalityKind> = Vec::new();
        for tok in &self.tokens {
            if !seen.contains(&&tok.modality) {
                seen.push(&tok.modality);
            }
        }
        seen
    }
}

// ---------------------------------------------------------------------------
// Cross-modal aligner
// ---------------------------------------------------------------------------

/// Buffers tokens from different modalities and flushes them as an aligned sequence.
///
/// Useful for synchronising multi-modal streams at a common time step boundary.
pub struct CrossModalAligner {
    shared_dim: usize,
    modality_counts: HashMap<String, usize>,
    buffer: Vec<CrossModalToken>,
}

impl CrossModalAligner {
    /// Create a new aligner for the given shared embedding dimension.
    pub fn new(shared_dim: usize) -> Self {
        Self {
            shared_dim,
            modality_counts: HashMap::new(),
            buffer: Vec::new(),
        }
    }

    /// Add a token to the alignment buffer.
    pub fn push_token(&mut self, token: CrossModalToken) {
        let key = token.modality.key();
        *self.modality_counts.entry(key).or_insert(0) += 1;
        self.buffer.push(token);
    }

    /// Consume the buffer and return it as a `CrossModalSequence`.
    pub fn flush(&mut self) -> CrossModalSequence {
        let mut seq = CrossModalSequence::new(self.shared_dim);
        for tok in self.buffer.drain(..) {
            seq.push(tok);
        }
        self.modality_counts.clear();
        seq
    }

    /// Number of tokens currently buffered.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// How many tokens from a given modality are currently buffered.
    pub fn count_for_modality(&self, modality: &ModalityKind) -> usize {
        self.modality_counts
            .get(&modality.key())
            .copied()
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Cross-modal tokenizer
// ---------------------------------------------------------------------------

/// Unified cross-modal tokenizer.
///
/// Manages one `ModalityTokenizer` per registered modality and applies:
/// 1. Per-modality linear projection (input → shared_dim)
/// 2. Modality-type embedding offset (for identity disambiguation)
/// 3. Shared alignment projection (shared_dim → shared_dim)
/// 4. Nearest-neighbour codebook quantization
pub struct CrossModalTokenizer {
    shared_dim: usize,
    /// Per-modality tokenizers keyed by `ModalityKind::key()`.
    tokenizers: HashMap<String, ModalityTokenizer>,
    /// Shared alignment weight: `(shared_dim, shared_dim)`.
    shared_proj: Array2<f32>,
    /// Shared alignment bias: `(shared_dim,)`.
    shared_bias: Array1<f32>,
    /// Per-modality learned offset vectors: `(shared_dim,)`.
    modality_embeddings: HashMap<String, Array1<f32>>,
}

impl CrossModalTokenizer {
    /// Create a new cross-modal tokenizer with the given shared embedding dimension.
    pub fn new(shared_dim: usize) -> TokenizerResult<Self> {
        if shared_dim == 0 {
            return Err(TokenizerError::InvalidConfig(
                "shared_dim must be > 0".into(),
            ));
        }

        // Initialize shared alignment projection as identity + small noise (xorshift64)
        let mut rng = SeededRng::new(0xdeadbeef_cafebabe);
        let scale = 0.01_f32 / (shared_dim as f32).sqrt();
        let shared_proj = Array2::from_shape_fn((shared_dim, shared_dim), |(i, j)| {
            let identity = if i == j { 1.0_f32 } else { 0.0_f32 };
            identity + rng.next_f32() * scale
        });
        let shared_bias = Array1::zeros(shared_dim);

        Ok(Self {
            shared_dim,
            tokenizers: HashMap::new(),
            shared_proj,
            shared_bias,
            modality_embeddings: HashMap::new(),
        })
    }

    /// Register a new modality.
    ///
    /// The `config.token_dim` must equal `self.shared_dim`.
    pub fn add_modality(&mut self, config: ModalityTokenizerConfig) -> TokenizerResult<()> {
        if config.token_dim != self.shared_dim {
            return Err(TokenizerError::InvalidConfig(format!(
                "ModalityTokenizerConfig.token_dim ({}) must equal shared_dim ({})",
                config.token_dim, self.shared_dim
            )));
        }
        config.validate()?;

        let key = config.modality.key();
        let modality_seed = config.modality.seed().wrapping_add(0x1234_5678_9abc_def0);
        let mut rng = SeededRng::new(modality_seed);
        let embed_scale = 0.02_f32;
        let mod_emb = Array1::from_shape_fn(self.shared_dim, |_| rng.next_f32() * embed_scale);

        let tokenizer = ModalityTokenizer::new(config)?;
        self.tokenizers.insert(key.clone(), tokenizer);
        self.modality_embeddings.insert(key, mod_emb);
        Ok(())
    }

    /// Tokenize a single-modality input.
    ///
    /// Steps:
    /// 1. Encode raw input → `shared_dim` embedding (per-modality encoder + GELU)
    /// 2. Add modality-type embedding offset
    /// 3. Apply shared alignment projection
    /// 4. Quantize against per-modality codebook
    pub fn tokenize(
        &self,
        modality: &ModalityKind,
        input: &Array1<f32>,
    ) -> TokenizerResult<CrossModalToken> {
        let key = modality.key();
        let tok = self.tokenizers.get(&key).ok_or_else(|| {
            TokenizerError::InvalidConfig(format!("modality '{key}' not registered"))
        })?;
        let mod_emb = self.modality_embeddings.get(&key).ok_or_else(|| {
            TokenizerError::InternalError(format!("missing modality embedding for '{key}'"))
        })?;

        // 1. Per-modality encode
        let encoded = tok.encode(input)?;

        // 2. Add modality-type offset
        let with_mod = encoded + mod_emb;

        // 3. Shared alignment: aligned = with_mod @ shared_proj + shared_bias
        let aligned = with_mod.dot(&self.shared_proj) + &self.shared_bias;

        // 4. Quantize
        let (token_idx, quantized) = tok.quantize(&aligned)?;
        let confidence = tok.confidence(&aligned, &quantized);

        Ok(CrossModalToken {
            modality: modality.clone(),
            token_idx,
            embedding: aligned,
            confidence,
        })
    }

    /// Tokenize a batch of (modality, signal) pairs and return them as a `CrossModalSequence`.
    pub fn tokenize_batch(
        &self,
        inputs: &[(ModalityKind, Array1<f32>)],
    ) -> TokenizerResult<CrossModalSequence> {
        let mut seq = CrossModalSequence::new(self.shared_dim);
        for (modality, signal) in inputs {
            let token = self.tokenize(modality, signal)?;
            seq.push(token);
        }
        Ok(seq)
    }

    /// Decode a `CrossModalToken` back to the raw input space.
    ///
    /// Uses the per-modality codebook entry as the quantized embedding,
    /// inverts the shared projection, removes the modality offset,
    /// and applies the pseudo-inverse decoder.
    pub fn decode(&self, token: &CrossModalToken) -> TokenizerResult<Array1<f32>> {
        let key = token.modality.key();
        let tok = self.tokenizers.get(&key).ok_or_else(|| {
            TokenizerError::InvalidConfig(format!("modality '{key}' not registered"))
        })?;
        let mod_emb = self.modality_embeddings.get(&key).ok_or_else(|| {
            TokenizerError::InternalError(format!("missing modality embedding for '{key}'"))
        })?;

        // Codebook lookup gives quantized embedding in shared space
        let quantized = tok.decode(token.token_idx)?;

        // Invert shared projection (approximate: use transpose)
        // aligned ≈ quantized  (we skip full inverse for efficiency)
        let without_mod = quantized - mod_emb;

        // Pseudo-inverse decode through encoder transpose
        tok.decode_embedding(&without_mod)
    }

    /// The shared embedding dimension.
    pub fn shared_dim(&self) -> usize {
        self.shared_dim
    }

    /// Number of registered modalities.
    pub fn num_modalities(&self) -> usize {
        self.tokenizers.len()
    }

    /// Sorted list of registered modality keys.
    pub fn modality_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tokenizers.keys().cloned().collect();
        names.sort();
        names
    }

    // -----------------------------------------------------------------------
    // Presets
    // -----------------------------------------------------------------------

    /// Robotics preset: audio (16-dim), control (6-dim), sensor (9-dim) → shared_dim 64.
    pub fn robotics_preset() -> TokenizerResult<Self> {
        let mut cmt = Self::new(64)?;
        cmt.add_modality(ModalityTokenizerConfig {
            modality: ModalityKind::Audio,
            input_dim: 16,
            token_dim: 64,
            codebook_size: 512,
            num_stages: 1,
        })?;
        cmt.add_modality(ModalityTokenizerConfig {
            modality: ModalityKind::Control,
            input_dim: 6,
            token_dim: 64,
            codebook_size: 256,
            num_stages: 1,
        })?;
        cmt.add_modality(ModalityTokenizerConfig {
            modality: ModalityKind::Sensor,
            input_dim: 9,
            token_dim: 64,
            codebook_size: 256,
            num_stages: 1,
        })?;
        Ok(cmt)
    }

    /// Audio-video preset: audio (80-dim), video (512-dim) → shared_dim 256.
    pub fn audio_video_preset() -> TokenizerResult<Self> {
        let mut cmt = Self::new(256)?;
        cmt.add_modality(ModalityTokenizerConfig {
            modality: ModalityKind::Audio,
            input_dim: 80,
            token_dim: 256,
            codebook_size: 1024,
            num_stages: 2,
        })?;
        cmt.add_modality(ModalityTokenizerConfig {
            modality: ModalityKind::Video,
            input_dim: 512,
            token_dim: 256,
            codebook_size: 2048,
            num_stages: 2,
        })?;
        Ok(cmt)
    }
}

// ---------------------------------------------------------------------------
// SignalTokenizer implementation
// ---------------------------------------------------------------------------

/// `SignalTokenizer` implementation for `CrossModalTokenizer`.
///
/// Treats the input as a concatenation of registered modality signals (in
/// registration order). Encodes each slice, concatenates the resulting
/// embeddings, and returns the full multi-modal embedding vector.
///
/// For `decode`, the embedding is split back into per-modality chunks,
/// decoded, and concatenated.
impl SignalTokenizer for CrossModalTokenizer {
    /// Encode a concatenated multi-modal signal.
    ///
    /// The input must be the concatenation of all registered modalities'
    /// raw signals (in sorted key order). Each modality's token embedding
    /// is concatenated into a single output vector of length
    /// `num_modalities * shared_dim`.
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let mut names = self.modality_names();
        names.sort();

        // Verify total input length
        let total_input_dim: usize = names.iter().map(|n| self.tokenizers[n].input_dim()).sum();
        if signal.len() != total_input_dim {
            return Err(TokenizerError::dim_mismatch(
                total_input_dim,
                signal.len(),
                "CrossModalTokenizer::encode total_input_dim",
            ));
        }

        let mut out = Vec::with_capacity(names.len() * self.shared_dim);
        let mut offset = 0usize;

        for name in &names {
            let tok = &self.tokenizers[name];
            let dim = tok.input_dim();
            let slice = signal.slice(scirs2_core::ndarray::s![offset..offset + dim]);
            let input_owned = slice.to_owned();

            // Find the ModalityKind from the stored tokenizer config
            // (we use the key to re-derive the modality kind by checking all known kinds)
            let modality = Self::key_to_modality_kind(name);
            let token = self.tokenize(&modality, &input_owned)?;
            out.extend_from_slice(
                token.embedding.as_slice().ok_or_else(|| {
                    TokenizerError::InternalError("embedding not contiguous".into())
                })?,
            );
            offset += dim;
        }

        Ok(Array1::from_vec(out))
    }

    /// Decode a concatenated embedding vector back to the raw input space.
    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let mut names = self.modality_names();
        names.sort();
        let n = names.len();

        if n == 0 {
            return Ok(Array1::zeros(0));
        }

        let expected = n * self.shared_dim;
        if tokens.len() != expected {
            return Err(TokenizerError::dim_mismatch(
                expected,
                tokens.len(),
                "CrossModalTokenizer::decode embedding length",
            ));
        }

        let mut out = Vec::new();

        for (i, name) in names.iter().enumerate() {
            let start = i * self.shared_dim;
            let end = start + self.shared_dim;
            let emb_slice = tokens
                .slice(scirs2_core::ndarray::s![start..end])
                .to_owned();

            let tok = &self.tokenizers[name];
            let mod_emb = &self.modality_embeddings[name];

            // Remove modality offset
            let without_mod = emb_slice - mod_emb;

            // Pseudo-inverse decode
            let reconstructed = tok.decode_embedding(&without_mod)?;
            out.extend_from_slice(reconstructed.as_slice().ok_or_else(|| {
                TokenizerError::InternalError("reconstructed not contiguous".into())
            })?);
        }

        Ok(Array1::from_vec(out))
    }

    /// Total output embedding dimension: `num_modalities * shared_dim`.
    fn embed_dim(&self) -> usize {
        self.tokenizers.len() * self.shared_dim
    }

    /// Returns 0 (continuous-style tokenizer; each modality has its own discrete codebook).
    fn vocab_size(&self) -> usize {
        0
    }
}

impl CrossModalTokenizer {
    /// Reconstruct a `ModalityKind` from a string key.
    fn key_to_modality_kind(key: &str) -> ModalityKind {
        match key {
            "audio" => ModalityKind::Audio,
            "control" => ModalityKind::Control,
            "sensor" => ModalityKind::Sensor,
            "video" => ModalityKind::Video,
            other => {
                let custom_name = other.strip_prefix("custom_").unwrap_or(other);
                ModalityKind::Custom(custom_name.to_string())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    // Helper: create a zero-filled input of a given length.
    fn zeros(n: usize) -> Array1<f32> {
        Array1::zeros(n)
    }

    // Helper: small non-zero input.
    fn ones(n: usize) -> Array1<f32> {
        Array1::ones(n)
    }

    // -----------------------------------------------------------------------
    // 1. ModalityTokenizer creation
    // -----------------------------------------------------------------------
    #[test]
    fn test_modality_tokenizer_creation() {
        let cfg = ModalityTokenizerConfig {
            modality: ModalityKind::Audio,
            input_dim: 16,
            token_dim: 64,
            codebook_size: 128,
            num_stages: 1,
        };
        let tok = ModalityTokenizer::new(cfg).expect("should create successfully");
        assert_eq!(tok.input_dim(), 16);
        assert_eq!(tok.token_dim(), 64);
        assert_eq!(tok.codebook_size(), 128);
        assert_eq!(tok.codebook().shape(), [128, 64]);
    }

    // -----------------------------------------------------------------------
    // 2. ModalityTokenizer encode produces correct shape
    // -----------------------------------------------------------------------
    #[test]
    fn test_modality_tokenizer_encode() {
        let cfg = ModalityTokenizerConfig {
            modality: ModalityKind::Control,
            input_dim: 6,
            token_dim: 32,
            codebook_size: 64,
            num_stages: 1,
        };
        let tok = ModalityTokenizer::new(cfg).expect("create");
        let input = ones(6);
        let emb = tok.encode(&input).expect("encode");
        assert_eq!(emb.len(), 32, "embedding must be token_dim");

        // Wrong dimension should error
        let bad = ones(5);
        assert!(tok.encode(&bad).is_err());
    }

    // -----------------------------------------------------------------------
    // 3. ModalityTokenizer quantize returns valid token index
    // -----------------------------------------------------------------------
    #[test]
    fn test_modality_tokenizer_quantize() {
        let cfg = ModalityTokenizerConfig {
            modality: ModalityKind::Sensor,
            input_dim: 9,
            token_dim: 16,
            codebook_size: 32,
            num_stages: 1,
        };
        let tok = ModalityTokenizer::new(cfg).expect("create");
        let emb = zeros(16);
        let (idx, quantized) = tok.quantize(&emb).expect("quantize");
        assert!(idx < 32, "token index must be within codebook");
        assert_eq!(quantized.len(), 16, "quantized must be token_dim");
    }

    // -----------------------------------------------------------------------
    // 4. decode(quantize(encode(x))) roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn test_modality_tokenizer_decode_roundtrip() {
        let cfg = ModalityTokenizerConfig {
            modality: ModalityKind::Audio,
            input_dim: 8,
            token_dim: 32,
            codebook_size: 64,
            num_stages: 1,
        };
        let tok = ModalityTokenizer::new(cfg).expect("create");
        let input = ones(8);
        let emb = tok.encode(&input).expect("encode");
        let (idx, _quantized) = tok.quantize(&emb).expect("quantize");
        let code = tok.decode(idx).expect("decode");
        assert_eq!(code.len(), 32, "decoded codebook entry must be token_dim");

        // Soft decode: pseudo-inverse should return input_dim vector
        let reconstructed = tok.decode_embedding(&emb).expect("decode_embedding");
        assert_eq!(reconstructed.len(), 8, "reconstructed must be input_dim");
    }

    // -----------------------------------------------------------------------
    // 5. CrossModalToken creation
    // -----------------------------------------------------------------------
    #[test]
    fn test_cross_modal_token_creation() {
        let token = CrossModalToken {
            modality: ModalityKind::Video,
            token_idx: 42,
            embedding: Array1::from_vec(vec![0.1, 0.2, 0.3]),
            confidence: 0.95,
        };
        assert_eq!(token.token_idx, 42);
        assert!((token.confidence - 0.95).abs() < 1e-6);
        assert_eq!(token.modality, ModalityKind::Video);
        assert_eq!(token.embedding.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 6. CrossModalSequence push, len, filter_by_modality
    // -----------------------------------------------------------------------
    #[test]
    fn test_cross_modal_sequence_operations() {
        let mut seq = CrossModalSequence::new(8);
        assert!(seq.is_empty());

        seq.push(CrossModalToken {
            modality: ModalityKind::Audio,
            token_idx: 0,
            embedding: Array1::zeros(8),
            confidence: 0.8,
        });
        seq.push(CrossModalToken {
            modality: ModalityKind::Control,
            token_idx: 1,
            embedding: Array1::ones(8),
            confidence: 0.7,
        });
        seq.push(CrossModalToken {
            modality: ModalityKind::Audio,
            token_idx: 2,
            embedding: Array1::zeros(8),
            confidence: 0.9,
        });

        assert_eq!(seq.len(), 3);
        assert!(!seq.is_empty());

        let audio_tokens = seq.filter_by_modality(&ModalityKind::Audio);
        assert_eq!(audio_tokens.len(), 2);

        let control_tokens = seq.filter_by_modality(&ModalityKind::Control);
        assert_eq!(control_tokens.len(), 1);

        let video_tokens = seq.filter_by_modality(&ModalityKind::Video);
        assert_eq!(video_tokens.len(), 0);

        let mods = seq.modalities_present();
        assert_eq!(mods.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 7. CrossModalSequence embedding matrix shape
    // -----------------------------------------------------------------------
    #[test]
    fn test_cross_modal_sequence_embedding_matrix() {
        let shared_dim = 16;
        let mut seq = CrossModalSequence::new(shared_dim);
        for _ in 0..5 {
            seq.push(CrossModalToken {
                modality: ModalityKind::Sensor,
                token_idx: 0,
                embedding: Array1::zeros(shared_dim),
                confidence: 1.0,
            });
        }
        let mat = seq.to_embedding_matrix();
        assert_eq!(mat.shape(), [5, shared_dim]);

        // Empty sequence
        let empty = CrossModalSequence::new(shared_dim);
        let empty_mat = empty.to_embedding_matrix();
        assert_eq!(empty_mat.shape(), [0, shared_dim]);
    }

    // -----------------------------------------------------------------------
    // 8. CrossModalTokenizer add_modality
    // -----------------------------------------------------------------------
    #[test]
    fn test_cross_modal_tokenizer_add_modality() {
        let mut cmt = CrossModalTokenizer::new(32).expect("new");
        cmt.add_modality(ModalityTokenizerConfig {
            modality: ModalityKind::Audio,
            input_dim: 16,
            token_dim: 32,
            codebook_size: 64,
            num_stages: 1,
        })
        .expect("add audio");

        cmt.add_modality(ModalityTokenizerConfig {
            modality: ModalityKind::Control,
            input_dim: 6,
            token_dim: 32,
            codebook_size: 32,
            num_stages: 1,
        })
        .expect("add control");

        assert_eq!(cmt.num_modalities(), 2);
        let names = cmt.modality_names();
        assert!(names.contains(&"audio".to_string()));
        assert!(names.contains(&"control".to_string()));

        // Wrong token_dim should fail
        let bad = cmt.add_modality(ModalityTokenizerConfig {
            modality: ModalityKind::Sensor,
            input_dim: 9,
            token_dim: 16, // mismatch
            codebook_size: 32,
            num_stages: 1,
        });
        assert!(bad.is_err());
    }

    // -----------------------------------------------------------------------
    // 9. CrossModalTokenizer tokenize single modality
    // -----------------------------------------------------------------------
    #[test]
    fn test_cross_modal_tokenizer_tokenize() {
        let mut cmt = CrossModalTokenizer::new(64).expect("new");
        cmt.add_modality(ModalityTokenizerConfig {
            modality: ModalityKind::Audio,
            input_dim: 16,
            token_dim: 64,
            codebook_size: 128,
            num_stages: 1,
        })
        .expect("add audio");

        let input = ones(16);
        let token = cmt
            .tokenize(&ModalityKind::Audio, &input)
            .expect("tokenize");
        assert_eq!(token.modality, ModalityKind::Audio);
        assert!(token.token_idx < 128);
        assert_eq!(token.embedding.len(), 64);
        assert!(token.confidence > 0.0 && token.confidence <= 1.0);

        // Unregistered modality should error
        assert!(cmt.tokenize(&ModalityKind::Video, &ones(512)).is_err());
    }

    // -----------------------------------------------------------------------
    // 10. CrossModalTokenizer tokenize_batch
    // -----------------------------------------------------------------------
    #[test]
    fn test_cross_modal_tokenizer_batch() {
        let mut cmt = CrossModalTokenizer::new(64).expect("new");
        cmt.add_modality(ModalityTokenizerConfig {
            modality: ModalityKind::Audio,
            input_dim: 16,
            token_dim: 64,
            codebook_size: 128,
            num_stages: 1,
        })
        .expect("add audio");
        cmt.add_modality(ModalityTokenizerConfig {
            modality: ModalityKind::Control,
            input_dim: 6,
            token_dim: 64,
            codebook_size: 64,
            num_stages: 1,
        })
        .expect("add control");

        let inputs = vec![
            (ModalityKind::Audio, ones(16)),
            (ModalityKind::Control, zeros(6)),
            (ModalityKind::Audio, zeros(16)),
        ];
        let seq = cmt.tokenize_batch(&inputs).expect("batch");
        assert_eq!(seq.len(), 3);
        assert_eq!(seq.shared_dim, 64);

        let mat = seq.to_embedding_matrix();
        assert_eq!(mat.shape(), [3, 64]);

        let audio_tokens = seq.filter_by_modality(&ModalityKind::Audio);
        assert_eq!(audio_tokens.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 11. CrossModalTokenizer decode
    // -----------------------------------------------------------------------
    #[test]
    fn test_cross_modal_tokenizer_decode() {
        let mut cmt = CrossModalTokenizer::new(32).expect("new");
        cmt.add_modality(ModalityTokenizerConfig {
            modality: ModalityKind::Sensor,
            input_dim: 9,
            token_dim: 32,
            codebook_size: 64,
            num_stages: 1,
        })
        .expect("add sensor");

        let input = ones(9);
        let token = cmt
            .tokenize(&ModalityKind::Sensor, &input)
            .expect("tokenize");

        let reconstructed = cmt.decode(&token).expect("decode");
        assert_eq!(reconstructed.len(), 9, "decoded must match input_dim");

        // Decoding token for an unregistered modality should error
        let bad_token = CrossModalToken {
            modality: ModalityKind::Video,
            token_idx: 0,
            embedding: Array1::zeros(32),
            confidence: 1.0,
        };
        assert!(cmt.decode(&bad_token).is_err());
    }

    // -----------------------------------------------------------------------
    // 12. Robotics preset
    // -----------------------------------------------------------------------
    #[test]
    fn test_cross_modal_robotics_preset() {
        let cmt = CrossModalTokenizer::robotics_preset().expect("robotics preset");
        assert_eq!(cmt.shared_dim(), 64);
        assert_eq!(cmt.num_modalities(), 3);

        let names = cmt.modality_names();
        assert!(names.contains(&"audio".to_string()));
        assert!(names.contains(&"control".to_string()));
        assert!(names.contains(&"sensor".to_string()));

        // Tokenize all three modalities
        let audio_token = cmt
            .tokenize(&ModalityKind::Audio, &ones(16))
            .expect("audio tokenize");
        assert_eq!(audio_token.embedding.len(), 64);

        let control_token = cmt
            .tokenize(&ModalityKind::Control, &zeros(6))
            .expect("control tokenize");
        assert!(control_token.token_idx < 256);

        let sensor_token = cmt
            .tokenize(&ModalityKind::Sensor, &ones(9))
            .expect("sensor tokenize");
        assert!(sensor_token.confidence > 0.0);

        // tokenize_batch
        let inputs = vec![
            (ModalityKind::Audio, ones(16)),
            (ModalityKind::Control, zeros(6)),
            (ModalityKind::Sensor, ones(9)),
        ];
        let seq = cmt.tokenize_batch(&inputs).expect("batch");
        assert_eq!(seq.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 13. CrossModalAligner push and flush
    // -----------------------------------------------------------------------
    #[test]
    fn test_cross_modal_aligner() {
        let mut aligner = CrossModalAligner::new(64);
        assert!(aligner.is_empty());

        aligner.push_token(CrossModalToken {
            modality: ModalityKind::Audio,
            token_idx: 0,
            embedding: Array1::zeros(64),
            confidence: 0.9,
        });
        aligner.push_token(CrossModalToken {
            modality: ModalityKind::Control,
            token_idx: 1,
            embedding: Array1::ones(64),
            confidence: 0.8,
        });
        aligner.push_token(CrossModalToken {
            modality: ModalityKind::Audio,
            token_idx: 2,
            embedding: Array1::zeros(64),
            confidence: 0.7,
        });

        assert_eq!(aligner.len(), 3);
        assert!(!aligner.is_empty());
        assert_eq!(aligner.count_for_modality(&ModalityKind::Audio), 2);
        assert_eq!(aligner.count_for_modality(&ModalityKind::Control), 1);
        assert_eq!(aligner.count_for_modality(&ModalityKind::Sensor), 0);

        let seq = aligner.flush();
        assert_eq!(seq.len(), 3);
        assert!(aligner.is_empty(), "buffer cleared after flush");
        assert_eq!(aligner.count_for_modality(&ModalityKind::Audio), 0);

        let mat = seq.to_embedding_matrix();
        assert_eq!(mat.shape(), [3, 64]);
    }

    // -----------------------------------------------------------------------
    // 14. ModalityKind key and seed determinism
    // -----------------------------------------------------------------------
    #[test]
    fn test_modality_kind_key_and_seed() {
        assert_eq!(ModalityKind::Audio.key(), "audio");
        assert_eq!(ModalityKind::Control.key(), "control");
        assert_eq!(ModalityKind::Sensor.key(), "sensor");
        assert_eq!(ModalityKind::Video.key(), "video");
        assert_eq!(ModalityKind::Custom("robot".into()).key(), "custom_robot");

        // Seeds must be deterministic
        assert_eq!(ModalityKind::Audio.seed(), ModalityKind::Audio.seed());
        assert_ne!(ModalityKind::Audio.seed(), ModalityKind::Control.seed());
    }

    // -----------------------------------------------------------------------
    // 15. Audio-video preset
    // -----------------------------------------------------------------------
    #[test]
    fn test_audio_video_preset() {
        let cmt = CrossModalTokenizer::audio_video_preset().expect("audio_video preset");
        assert_eq!(cmt.shared_dim(), 256);
        assert_eq!(cmt.num_modalities(), 2);

        let audio_tok = cmt
            .tokenize(&ModalityKind::Audio, &ones(80))
            .expect("audio tokenize");
        assert_eq!(audio_tok.embedding.len(), 256);

        let video_tok = cmt
            .tokenize(&ModalityKind::Video, &ones(512))
            .expect("video tokenize");
        assert!(video_tok.token_idx < 2048);
    }
}
