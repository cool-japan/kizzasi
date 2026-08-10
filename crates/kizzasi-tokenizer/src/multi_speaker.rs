//! Multi-speaker tokenization for Kizzasi AGSP
//!
//! This module provides a complete acoustic representation tokenizer that supports
//! multi-speaker scenarios. Unlike [`SpeechTokenizer`], which is not invertible,
//! [`MultiSpeakerTokenizer`] quantizes the mel spectrogram using a per-element
//! linear scheme so that decoding is deterministic and lossless up to quantization error.
//!
//! ## Design
//!
//! The tokenizer follows a three-stage pipeline:
//!
//! 1. **Acoustic feature extraction**: compute mel spectrogram via [`SpeechTokenizer::compute_mel_spectrogram`]
//!    — returns `Array2<f32>` with shape `[n_mels, n_frames]`.
//! 2. **Quantization**: flatten the matrix, record global `(min, max)`, then map each element
//!    to a `u32` index in `[0, 2^bits - 1]`. The original range is stored alongside the token
//!    for lossless dequantization.
//! 3. **Speaker identification**: mean-pool the mel spectrogram over the time axis to obtain
//!    an `n_mels`-dimensional speaker embedding, then do nearest-centroid lookup against an
//!    EMA-trained [`SpeakerCodebook`].
//!
//! ## Decoding note
//!
//! [`MultiSpeakerTokenizer::decode`] returns the dequantized, flattened mel spectrogram
//! (`Array1<f32>` of length `n_mels * n_frames`). It does **not** produce audio — no vocoder
//! is included in this crate. To synthesise audio from mel features use an external vocoder
//! such as Griffin-Lim or WaveGlow.
//!
//! The [`SignalTokenizer`] trait's `decode` method returns an error directing callers to use
//! the struct-based `decode` instead.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::domain_specific::{SpeechTokenizer, SpeechTokenizerConfig};
use crate::error::{TokenizerError, TokenizerResult};
use crate::SignalTokenizer;

// ---------------------------------------------------------------------------
// Tiny deterministic RNG (SplitMix64) — avoids adding `rand` to the workspace
// ---------------------------------------------------------------------------

/// Minimal SplitMix64 PRNG, good enough for k-means++ seeding.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Returns a float in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Samples an index in `[0, n)` using rejection sampling.
    fn next_usize(&mut self, n: usize) -> usize {
        ((self.next_f64() * n as f64) as usize).min(n.saturating_sub(1))
    }
}

// ---------------------------------------------------------------------------
// L2 distance helper
// ---------------------------------------------------------------------------

fn l2_dist(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f32>()
}

// ---------------------------------------------------------------------------
// SpeakerCodebook
// ---------------------------------------------------------------------------

/// EMA speaker codebook storing `num_speakers` centroids in `embed_dim` space.
///
/// Centroids are initialised with k-means++ and updated via EMA (exponential
/// moving average) during supervised fit or online adaptation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerCodebook {
    /// Centroid matrix, shape `[num_speakers, embed_dim]`.
    pub embeddings: Array2<f32>,
    /// Running soft counts used for EMA normalisation.
    pub counts: Array1<f32>,
    /// EMA decay factor (e.g. 0.999).
    pub decay: f32,
}

impl SpeakerCodebook {
    /// Create a zero-initialised codebook. Call `kmeans_plus_plus_init` before use.
    pub fn new(num_speakers: usize, embed_dim: usize, decay: f32) -> Self {
        Self {
            embeddings: Array2::zeros((num_speakers, embed_dim)),
            counts: Array1::zeros(num_speakers),
            decay,
        }
    }

    /// Initialise centroids using k-means++ with the given RNG seed.
    ///
    /// `data` must have at least `num_speakers` elements, otherwise an error is returned.
    pub fn kmeans_plus_plus_init(
        &mut self,
        data: &[Array1<f32>],
        rng_seed: u64,
    ) -> TokenizerResult<()> {
        let k = self.embeddings.shape()[0];
        if data.len() < k {
            return Err(TokenizerError::invalid_input(
                "kmeans_plus_plus_init",
                format!(
                    "Need at least {k} data points to initialise {k} centroids, got {}",
                    data.len()
                ),
            ));
        }

        let mut rng = SplitMix64::new(rng_seed);
        let mut chosen: Vec<usize> = Vec::with_capacity(k);

        // Pick the first centroid uniformly at random
        chosen.push(rng.next_usize(data.len()));

        for c_idx in 1..k {
            // For each data point compute squared distance to nearest chosen centroid
            let dists: Vec<f64> = data
                .iter()
                .map(|x| {
                    chosen
                        .iter()
                        .map(|&ci| l2_dist(x, &data[ci]) as f64)
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();

            let total: f64 = dists.iter().sum();
            let threshold = rng.next_f64() * total;
            let mut cumsum = 0.0f64;
            let mut picked = data.len() - 1; // fallback
            for (i, &d) in dists.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    picked = i;
                    break;
                }
            }
            chosen.push(picked);

            // Store the centroid we just selected
            let src = &data[chosen[c_idx - 1]];
            let embed_dim = self.embeddings.shape()[1];
            let mut row = self.embeddings.row_mut(c_idx - 1);
            for (dst_val, &src_val) in row.iter_mut().zip(src.iter()).take(embed_dim) {
                *dst_val = src_val;
            }
        }

        // Write last centroid
        let last = *chosen.last().ok_or_else(|| {
            TokenizerError::InternalError("Empty chosen set in kmeans++".to_string())
        })?;
        let src = &data[last];
        let embed_dim = self.embeddings.shape()[1];
        let mut row = self.embeddings.row_mut(k - 1);
        for (dst_val, &src_val) in row.iter_mut().zip(src.iter()).take(embed_dim) {
            *dst_val = src_val;
        }

        Ok(())
    }

    /// Return the index of the centroid nearest to `vec` in L2 distance.
    pub fn find_nearest(&self, vec: &Array1<f32>) -> TokenizerResult<u32> {
        let k = self.embeddings.shape()[0];
        if k == 0 {
            return Err(TokenizerError::CodebookNotInitialized {
                hint: "SpeakerCodebook has zero speakers".to_string(),
            });
        }

        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;

        for i in 0..k {
            let centroid = self.embeddings.row(i).to_owned();
            let dist = l2_dist(vec, &centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        Ok(best_idx as u32)
    }

    /// Update the centroid for `speaker_id` via EMA: `embed = decay*embed + (1-decay)*vec`.
    pub fn ema_update(&mut self, speaker_id: u32, vec: &Array1<f32>) {
        let idx = speaker_id as usize;
        let embed_dim = self.embeddings.shape()[1];
        let d = self.decay;
        for j in 0..embed_dim {
            self.embeddings[[idx, j]] = d * self.embeddings[[idx, j]] + (1.0 - d) * vec[j];
        }
        self.counts[idx] = d * self.counts[idx] + (1.0 - d);
    }
}

// ---------------------------------------------------------------------------
// MultiSpeakerConfig
// ---------------------------------------------------------------------------

/// Configuration for [`MultiSpeakerTokenizer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSpeakerConfig {
    /// Configuration forwarded to the inner [`SpeechTokenizer`].
    pub speech_config: SpeechTokenizerConfig,
    /// Number of speaker centroids in the codebook.
    pub num_speakers: usize,
    /// Quantisation precision in bits for mel values (1–16).
    pub bits_per_mel_bin: u32,
    /// EMA decay for the speaker codebook (e.g. 0.999).
    pub codebook_decay: f32,
}

impl Default for MultiSpeakerConfig {
    fn default() -> Self {
        Self {
            speech_config: SpeechTokenizerConfig::default(),
            num_speakers: 16,
            bits_per_mel_bin: 8,
            codebook_decay: 0.999,
        }
    }
}

// ---------------------------------------------------------------------------
// MultiSpeakerToken
// ---------------------------------------------------------------------------

/// A self-contained acoustic + identity token.
///
/// Stores the quantised mel spectrogram together with all the metadata required
/// to dequantise it faithfully, making decoding fully deterministic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSpeakerToken {
    /// Quantised mel spectrogram, flattened to `[n_mels * n_frames]`.
    ///
    /// Entry at index `m * n_frames + t` corresponds to mel bin `m`, time frame `t`.
    pub acoustic: Vec<u32>,
    /// Inferred or explicitly assigned speaker index.
    pub speaker_id: u32,
    /// Number of time frames in the original spectrogram.
    pub num_frames: usize,
    /// Number of mel bins.
    pub n_mels: usize,
    /// Global minimum mel value (for dequantisation).
    pub mel_min: f32,
    /// Global maximum mel value (for dequantisation).
    pub mel_max: f32,
}

// ---------------------------------------------------------------------------
// MultiSpeakerTokenizer
// ---------------------------------------------------------------------------

/// Multi-speaker acoustic tokenizer.
///
/// Encodes audio signals into a decodable mel-spectrogram representation while
/// simultaneously inferring or accepting an explicit speaker identity.
///
/// # Example
///
/// ```rust,no_run
/// use kizzasi_tokenizer::multi_speaker::{MultiSpeakerConfig, MultiSpeakerTokenizer};
/// use scirs2_core::ndarray::Array1;
///
/// let config = MultiSpeakerConfig::default();
/// let mut tok = MultiSpeakerTokenizer::new(config).unwrap();
///
/// // Create a test signal (1 second at 16 kHz)
/// let signal = Array1::from_vec(
///     (0..16000).map(|i| (i as f32 * 0.01).sin()).collect()
/// );
///
/// let token = tok.encode_blind(&signal).unwrap();
/// let mel_flat = tok.decode(&token).unwrap();
/// assert_eq!(mel_flat.len(), token.n_mels * token.num_frames);
/// ```
#[derive(Debug)]
pub struct MultiSpeakerTokenizer {
    /// Inner speech tokenizer (mel spectrogram extractor).
    pub speech: SpeechTokenizer,
    /// Speaker codebook.
    pub speaker: SpeakerCodebook,
    /// Configuration.
    pub config: MultiSpeakerConfig,
}

impl MultiSpeakerTokenizer {
    /// Create a new tokenizer from configuration.
    ///
    /// Returns an error if `bits_per_mel_bin` is 0 or > 16 or if the inner
    /// [`SpeechTokenizer`] fails to initialise.
    pub fn new(config: MultiSpeakerConfig) -> TokenizerResult<Self> {
        if config.bits_per_mel_bin == 0 || config.bits_per_mel_bin > 16 {
            return Err(TokenizerError::InvalidConfig(
                "bits_per_mel_bin must be in [1, 16]".to_string(),
            ));
        }
        if config.num_speakers == 0 {
            return Err(TokenizerError::InvalidConfig(
                "num_speakers must be > 0".to_string(),
            ));
        }

        let n_mels = config.speech_config.n_mels;
        let speech = SpeechTokenizer::new(config.speech_config.clone())?;
        let speaker = SpeakerCodebook::new(config.num_speakers, n_mels, config.codebook_decay);

        Ok(Self {
            speech,
            speaker,
            config,
        })
    }

    /// Fit the speaker codebook using the provided audio signals.
    ///
    /// For each signal, the mel spectrogram is computed and mean-pooled over
    /// the time axis to produce an `n_mels`-dimensional speaker embedding.
    /// k-means++ initialisation followed by one EMA pass assigns counts and
    /// refines centroids.
    pub fn fit_speakers(&mut self, signals: &[Array1<f32>]) -> TokenizerResult<()> {
        if signals.is_empty() {
            return Err(TokenizerError::invalid_input(
                "fit_speakers",
                "signals slice must not be empty",
            ));
        }

        // Extract one embedding per signal
        let embeddings: Vec<Array1<f32>> = signals
            .iter()
            .enumerate()
            .map(|(idx, sig)| {
                let mel = self.speech.compute_mel_spectrogram(sig).map_err(|e| {
                    TokenizerError::EncodingError {
                        operation: format!("fit_speakers[{idx}]"),
                        reason: e.to_string(),
                    }
                })?;
                Ok(mean_pool_time(&mel))
            })
            .collect::<TokenizerResult<Vec<_>>>()?;

        // k-means++ init with fixed seed for reproducibility
        self.speaker
            .kmeans_plus_plus_init(&embeddings, /* rng_seed = */ 42)?;

        // One EMA pass to accumulate counts
        for emb in &embeddings {
            let sid = self.speaker.find_nearest(emb)?;
            self.speaker.ema_update(sid, emb);
        }

        Ok(())
    }

    /// Quantise the mel spectrogram and wrap it as a [`MultiSpeakerToken`].
    pub fn encode_with_speaker(
        &self,
        signal: &Array1<f32>,
        speaker_id: u32,
    ) -> TokenizerResult<MultiSpeakerToken> {
        if speaker_id as usize >= self.config.num_speakers {
            return Err(TokenizerError::out_of_range(
                speaker_id as f32,
                0.0,
                (self.config.num_speakers - 1) as f32,
                "encode_with_speaker: speaker_id",
            ));
        }

        let mel = self.speech.compute_mel_spectrogram(signal)?;
        let (n_mels, n_frames) = mel.dim();

        if n_frames == 0 {
            return Err(TokenizerError::invalid_input(
                "encode_with_speaker",
                "signal too short to produce any mel frames",
            ));
        }

        // Global min/max for quantisation
        let (mel_min, mel_max) = global_minmax(&mel);
        let acoustic = quantize_mel(&mel, mel_min, mel_max, self.config.bits_per_mel_bin);

        Ok(MultiSpeakerToken {
            acoustic,
            speaker_id,
            num_frames: n_frames,
            n_mels,
            mel_min,
            mel_max,
        })
    }

    /// Encode a signal with automatic speaker inference.
    ///
    /// The mel spectrogram is mean-pooled over time to obtain a speaker embedding
    /// and the nearest codebook centroid is selected.
    pub fn encode_blind(&self, signal: &Array1<f32>) -> TokenizerResult<MultiSpeakerToken> {
        let mel = self.speech.compute_mel_spectrogram(signal)?;

        if mel.dim().1 == 0 {
            return Err(TokenizerError::invalid_input(
                "encode_blind",
                "signal too short to produce any mel frames",
            ));
        }

        let embedding = mean_pool_time(&mel);
        let speaker_id = self.speaker.find_nearest(&embedding)?;
        self.encode_with_speaker(signal, speaker_id)
    }

    /// Dequantise a [`MultiSpeakerToken`] and return the flattened mel spectrogram.
    ///
    /// The returned `Array1<f32>` has length `n_mels * num_frames`. This is a mel
    /// representation, **not audio**. No vocoder is included in this crate.
    pub fn decode(&self, token: &MultiSpeakerToken) -> TokenizerResult<Array1<f32>> {
        let expected_len = token.n_mels * token.num_frames;
        if token.acoustic.len() != expected_len {
            return Err(TokenizerError::dim_mismatch(
                expected_len,
                token.acoustic.len(),
                "MultiSpeakerTokenizer::decode: acoustic length",
            ));
        }

        let mel_flat = dequantize_mel(
            &token.acoustic,
            token.mel_min,
            token.mel_max,
            self.config.bits_per_mel_bin,
        );
        Ok(Array1::from_vec(mel_flat))
    }

    /// Clone the token with a new speaker identity.
    ///
    /// The acoustic content is unchanged. Future work: modify the acoustic
    /// representation via speaker embedding arithmetic.
    pub fn re_target(
        &self,
        token: &MultiSpeakerToken,
        target_speaker: u32,
    ) -> TokenizerResult<MultiSpeakerToken> {
        if target_speaker as usize >= self.config.num_speakers {
            return Err(TokenizerError::out_of_range(
                target_speaker as f32,
                0.0,
                (self.config.num_speakers - 1) as f32,
                "re_target: target_speaker",
            ));
        }
        let mut new_token = token.clone();
        new_token.speaker_id = target_speaker;
        Ok(new_token)
    }
}

// ---------------------------------------------------------------------------
// SignalTokenizer impl
// ---------------------------------------------------------------------------

impl SignalTokenizer for MultiSpeakerTokenizer {
    /// Encode a signal using blind speaker inference.
    ///
    /// Returns the quantised mel values cast to `f32`.
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let token = self.encode_blind(signal)?;
        let floats: Vec<f32> = token.acoustic.iter().map(|&q| q as f32).collect();
        Ok(Array1::from_vec(floats))
    }

    /// Not directly supported via the flat `Array1` interface.
    ///
    /// Use [`MultiSpeakerTokenizer::decode`] with a [`MultiSpeakerToken`] instead, which
    /// preserves the metadata (`n_mels`, `num_frames`, `mel_min`, `mel_max`) required for
    /// correct dequantisation.
    fn decode(&self, _tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Err(TokenizerError::decoding(
            "MultiSpeakerTokenizer",
            "use MultiSpeakerTokenizer::decode with a MultiSpeakerToken struct; \
             the flat Array1 interface lacks the metadata needed for dequantisation",
        ))
    }

    /// Embedding dimension is `n_mels`.
    fn embed_dim(&self) -> usize {
        self.config.speech_config.n_mels
    }

    /// Vocabulary size is `2^bits_per_mel_bin`.
    fn vocab_size(&self) -> usize {
        2_usize.pow(self.config.bits_per_mel_bin)
    }
}

// ---------------------------------------------------------------------------
// Quantisation helpers
// ---------------------------------------------------------------------------

/// Compute the global (min, max) of an `Array2<f32>`.
///
/// Returns `(0.0, 1.0)` if the matrix is empty.
fn global_minmax(mat: &Array2<f32>) -> (f32, f32) {
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &v in mat.iter() {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }
    if min_val.is_infinite() {
        (0.0, 1.0)
    } else {
        (min_val, max_val)
    }
}

/// Quantise a mel matrix to `u32` indices in `[0, 2^bits - 1]`.
///
/// Values are row-major: index `m * n_frames + t` corresponds to bin `m`, frame `t`.
fn quantize_mel(mel: &Array2<f32>, min_val: f32, max_val: f32, bits: u32) -> Vec<u32> {
    let levels = (1u64 << bits) - 1;
    let range = max_val - min_val;
    mel.iter()
        .map(|&v| {
            let normalised = (v - min_val) / (range + 1e-8);
            (normalised * levels as f32).round() as u32
        })
        .collect()
}

/// Dequantise a flat `Vec<u32>` back to mel values.
fn dequantize_mel(acoustic: &[u32], min_val: f32, max_val: f32, bits: u32) -> Vec<f32> {
    let levels = (1u64 << bits) - 1;
    let range = max_val - min_val;
    acoustic
        .iter()
        .map(|&q| q as f32 / levels as f32 * range + min_val)
        .collect()
}

/// Mean-pool an `Array2<f32>` over axis 1 (time axis), returning `Array1<f32>`.
///
/// The result has length `n_mels` (the row count). If `n_frames == 0`, zeros are returned.
fn mean_pool_time(mel: &Array2<f32>) -> Array1<f32> {
    let n_frames = mel.dim().1;
    if n_frames == 0 {
        return Array1::zeros(mel.dim().0);
    }
    mel.mean_axis(Axis(1))
        .unwrap_or_else(|| Array1::zeros(mel.dim().0))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Build a simple config with smaller dims for fast tests.
    fn small_config() -> MultiSpeakerConfig {
        MultiSpeakerConfig {
            speech_config: SpeechTokenizerConfig {
                sample_rate: 8000,
                n_mels: 16,
                n_fft: 256,
                hop_length: 128,
                n_phonemes: 44,
                use_delta: false,
                use_delta_delta: false,
            },
            num_speakers: 4,
            bits_per_mel_bin: 8,
            codebook_decay: 0.999,
        }
    }

    /// Generate a deterministic sine-wave signal.
    fn sine(samples: usize, freq: f32, sample_rate: usize) -> Array1<f32> {
        Array1::from_vec(
            (0..samples)
                .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
                .collect(),
        )
    }

    // -----------------------------------------------------------------------
    // 1. Codebook nearest-neighbour
    // -----------------------------------------------------------------------

    #[test]
    fn test_speaker_codebook_find_nearest() {
        let mut cb = SpeakerCodebook::new(2, 4, 0.99);
        // centroid 0 at origin
        for j in 0..4 {
            cb.embeddings[[0, j]] = 0.0;
        }
        // centroid 1 at [1,1,1,1]
        for j in 0..4 {
            cb.embeddings[[1, j]] = 1.0;
        }

        // A vector very close to the origin should map to centroid 0
        let near_zero = Array1::from_vec(vec![0.05, 0.0, 0.0, 0.0]);
        assert_eq!(cb.find_nearest(&near_zero).unwrap(), 0);

        // A vector very close to [1,1,1,1] should map to centroid 1
        let near_one = Array1::from_vec(vec![0.95, 1.0, 1.0, 1.0]);
        assert_eq!(cb.find_nearest(&near_one).unwrap(), 1);
    }

    // -----------------------------------------------------------------------
    // 2. Encode with explicit speaker — check token structure
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_with_speaker_token_structure() {
        let config = small_config();
        let tok = MultiSpeakerTokenizer::new(config).unwrap();

        let signal = sine(1024, 440.0, 8000);
        let token = tok.encode_with_speaker(&signal, 3).unwrap();

        assert_eq!(token.speaker_id, 3);
        assert_eq!(token.n_mels, 16);
        assert_eq!(token.acoustic.len(), 16 * token.num_frames);
        assert!(token.num_frames > 0);
    }

    // -----------------------------------------------------------------------
    // 3. Decode returns the correct flat mel shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_returns_mel_shape() {
        let config = small_config();
        let tok = MultiSpeakerTokenizer::new(config).unwrap();

        let signal = sine(2048, 220.0, 8000);
        let token = tok.encode_with_speaker(&signal, 0).unwrap();
        let mel_flat = tok.decode(&token).unwrap();

        assert_eq!(mel_flat.len(), token.n_mels * token.num_frames);
    }

    // -----------------------------------------------------------------------
    // 4. Blind encoding infers a speaker within {0, 1}
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_blind_infers_speaker() {
        let mut config = small_config();
        config.num_speakers = 2;

        let mut tok = MultiSpeakerTokenizer::new(config).unwrap();

        // Two clearly different signals
        let sig_a = sine(2048, 110.0, 8000);
        let sig_b = sine(2048, 3800.0, 8000);

        tok.fit_speakers(&[sig_a.clone(), sig_b.clone()]).unwrap();

        let tok_a = tok.encode_blind(&sig_a).unwrap();
        let tok_b = tok.encode_blind(&sig_b).unwrap();

        assert!(tok_a.speaker_id < 2);
        assert!(tok_b.speaker_id < 2);
    }

    // -----------------------------------------------------------------------
    // 5. re_target changes only speaker_id
    // -----------------------------------------------------------------------

    #[test]
    fn test_re_target_changes_only_speaker_id() {
        let config = small_config();
        let tok = MultiSpeakerTokenizer::new(config).unwrap();

        let signal = sine(2048, 440.0, 8000);
        let orig = tok.encode_with_speaker(&signal, 0).unwrap();
        let retargeted = tok.re_target(&orig, 2).unwrap();

        assert_eq!(retargeted.speaker_id, 2);
        assert_eq!(retargeted.acoustic, orig.acoustic);
        assert_eq!(retargeted.n_mels, orig.n_mels);
        assert_eq!(retargeted.num_frames, orig.num_frames);
        assert_eq!(retargeted.mel_min, orig.mel_min);
        assert_eq!(retargeted.mel_max, orig.mel_max);
    }

    // -----------------------------------------------------------------------
    // 6. Quantise / dequantise round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let values = [0.1f32, 0.5, 0.9];
        let min_val = 0.0f32;
        let max_val = 1.0f32;
        let bits = 8u32;

        // Build a 1x3 matrix for the helpers
        use scirs2_core::ndarray::Array2;
        let mel = Array2::from_shape_fn((1, values.len()), |(_, j)| values[j]);

        let quantised = quantize_mel(&mel, min_val, max_val, bits);
        let dequantised = dequantize_mel(&quantised, min_val, max_val, bits);

        let mse: f32 = values
            .iter()
            .zip(dequantised.iter())
            .map(|(&orig, &recon)| (orig - recon) * (orig - recon))
            .sum::<f32>()
            / values.len() as f32;

        assert!(
            mse < 0.01,
            "MSE={mse} exceeds threshold 0.01 for 8-bit quantisation"
        );
    }

    // -----------------------------------------------------------------------
    // Extra: SignalTokenizer trait encode path
    // -----------------------------------------------------------------------

    #[test]
    fn test_signal_tokenizer_encode_flat() {
        let config = small_config();
        let tok = MultiSpeakerTokenizer::new(config).unwrap();

        let signal = sine(2048, 440.0, 8000);
        let flat = SignalTokenizer::encode(&tok, &signal).unwrap();
        assert!(!flat.is_empty());
        // All values should be integers in [0, 255] for 8-bit
        for &v in flat.iter() {
            assert!((0.0..=255.0).contains(&v), "unexpected value {v}");
        }
    }

    // -----------------------------------------------------------------------
    // Extra: SignalTokenizer decode returns error with guidance
    // -----------------------------------------------------------------------

    #[test]
    fn test_signal_tokenizer_decode_returns_error() {
        let config = small_config();
        let tok = MultiSpeakerTokenizer::new(config).unwrap();

        let dummy = Array1::from_vec(vec![0.0f32; 16]);
        let result = SignalTokenizer::decode(&tok, &dummy);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("MultiSpeakerToken"),
            "error should mention MultiSpeakerToken, got: {msg}"
        );
    }
}
