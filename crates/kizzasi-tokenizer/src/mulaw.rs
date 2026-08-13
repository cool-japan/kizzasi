//! μ-law companding codec for audio signals
//!
//! μ-law encoding is a logarithmic quantization scheme commonly used
//! for audio signals. It provides better dynamic range preservation
//! than linear quantization, especially for quiet sounds.
//!
//! The μ-law formula is:
//! F(x) = sign(x) * ln(1 + μ|x|) / ln(1 + μ)
//!
//! where μ is typically 255 for 8-bit quantization.

use crate::error::{TokenizerError, TokenizerResult};
use crate::SignalTokenizer;
use scirs2_core::ndarray::Array1;

/// Valid bit-depth range for [`MuLawCodec`]. `bits == 0` makes `mu == 0`,
/// which turns `encode_sample`'s `ln(1 + mu*|x|) / ln(1 + mu)` into `0.0/0.0`
/// (NaN); `bits > 16` overflows `1usize << bits` on any platform where
/// `usize` is narrower than the shift amount implies and produces codecs
/// with unusable level counts. This mirrors the bound every other
/// quantizer in this crate enforces (see `quantizer.rs`, `advanced_quant.rs`).
const MIN_BITS: u8 = 1;
const MAX_BITS: u8 = 16;

/// μ-law companding codec
#[derive(Debug, Clone)]
pub struct MuLawCodec {
    /// μ parameter (typically 255)
    mu: f32,
    /// Number of quantization bits
    bits: u8,
    /// Number of quantization levels
    levels: usize,
}

impl MuLawCodec {
    /// Create a new μ-law codec with specified bits.
    ///
    /// `bits` is clamped into `[1, 16]` and out-of-range input is logged via
    /// `tracing::warn!` rather than silently ignored. This constructor stays
    /// infallible for backward compatibility with existing callers; use
    /// [`MuLawCodec::try_new`] to reject invalid `bits` with a
    /// [`TokenizerError`] instead.
    pub fn new(bits: u8) -> Self {
        let clamped_bits = clamp_bits(bits);
        // `clamped_bits` is always in [MIN_BITS, MAX_BITS], so this can
        // never panic or produce NaN/inf.
        Self::try_new(clamped_bits).unwrap_or_else(|_| {
            // Unreachable given the clamp above, but avoid any panic path
            // in library code: fall back to the narrowest valid codec.
            let levels = 1usize << MIN_BITS;
            Self {
                mu: (levels - 1) as f32,
                bits: MIN_BITS,
                levels,
            }
        })
    }

    /// Create a new μ-law codec, rejecting `bits` outside `[1, 16]`.
    pub fn try_new(bits: u8) -> TokenizerResult<Self> {
        if !(MIN_BITS..=MAX_BITS).contains(&bits) {
            return Err(TokenizerError::InvalidConfig(format!(
                "bits must be in [{MIN_BITS}, {MAX_BITS}], got {bits}"
            )));
        }
        let levels = 1usize << bits;
        let mu = (levels - 1) as f32;
        Ok(Self { mu, bits, levels })
    }

    /// Create with custom μ value.
    ///
    /// `bits` is clamped into `[1, 16]` (logged via `tracing::warn!` when
    /// clamping occurs) and `mu` is clamped to be finite and positive. Use
    /// [`MuLawCodec::try_with_mu`] to reject invalid input instead.
    pub fn with_mu(mu: f32, bits: u8) -> Self {
        let clamped_mu = if mu.is_finite() && mu > 0.0 { mu } else { 1.0 };
        let clamped_bits = clamp_bits(bits);
        // `clamped_mu`/`clamped_bits` are always valid per the clamps above,
        // so `try_with_mu` never actually returns `Err` here; the fallback
        // just avoids any panic path in library code should that invariant
        // ever be violated.
        Self::try_with_mu(clamped_mu, clamped_bits).unwrap_or(Self {
            mu: clamped_mu,
            bits: MIN_BITS,
            levels: 1usize << MIN_BITS,
        })
    }

    /// Create with a custom μ value, rejecting `bits` outside `[1, 16]` or a
    /// non-finite/non-positive `mu`.
    pub fn try_with_mu(mu: f32, bits: u8) -> TokenizerResult<Self> {
        if !(MIN_BITS..=MAX_BITS).contains(&bits) {
            return Err(TokenizerError::InvalidConfig(format!(
                "bits must be in [{MIN_BITS}, {MAX_BITS}], got {bits}"
            )));
        }
        if !mu.is_finite() || mu <= 0.0 {
            return Err(TokenizerError::InvalidConfig(format!(
                "mu must be finite and positive, got {mu}"
            )));
        }
        Ok(Self {
            mu,
            bits,
            levels: 1usize << bits,
        })
    }

    /// Encode a single sample using μ-law
    fn encode_sample(&self, x: f32) -> f32 {
        let x_clamped = x.clamp(-1.0, 1.0);
        let sign = x_clamped.signum();
        let magnitude = (1.0 + self.mu * x_clamped.abs()).ln() / (1.0 + self.mu).ln();
        sign * magnitude
    }

    /// Decode a single sample using μ-law
    fn decode_sample(&self, y: f32) -> f32 {
        let y_clamped = y.clamp(-1.0, 1.0);
        let sign = y_clamped.signum();
        let magnitude = ((1.0 + self.mu).powf(y_clamped.abs()) - 1.0) / self.mu;
        sign * magnitude
    }

    /// Quantize to integer level
    pub fn quantize(&self, x: f32) -> i32 {
        let encoded = self.encode_sample(x);
        let half_levels = (self.levels / 2) as f32;
        ((encoded + 1.0) * half_levels)
            .round()
            .clamp(0.0, (self.levels - 1) as f32) as i32
    }

    /// Dequantize from integer level
    pub fn dequantize(&self, level: i32) -> f32 {
        let half_levels = (self.levels / 2) as f32;
        let encoded = (level as f32 / half_levels) - 1.0;
        self.decode_sample(encoded)
    }

    /// Get the number of bits
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Get μ value
    pub fn mu(&self) -> f32 {
        self.mu
    }

    /// Get the number of quantization levels (`2^bits`).
    ///
    /// Prefer this over re-deriving levels from [`MuLawCodec::bits`] via a
    /// fresh `1 << bits` shift: `bits` is validated to `[1, 16]` internally,
    /// but callers re-shifting an unvalidated bit count (e.g. from
    /// user-controlled input) could still overflow.
    pub fn levels(&self) -> usize {
        self.levels
    }
}

/// Clamp `bits` into `[MIN_BITS, MAX_BITS]`, logging when clamping occurs.
fn clamp_bits(bits: u8) -> u8 {
    let clamped = bits.clamp(MIN_BITS, MAX_BITS);
    if clamped != bits {
        tracing::warn!(
            requested = bits,
            clamped = clamped,
            "MuLawCodec: bits out of [{}, {}], clamped to {}; use try_new/try_with_mu to reject instead",
            MIN_BITS,
            MAX_BITS,
            clamped
        );
    }
    clamped
}

impl SignalTokenizer for MuLawCodec {
    /// Encode a signal into discrete μ-law levels, returned as `f32` token
    /// ids in `0..vocab_size()`.
    ///
    /// This matches [`crate::LinearQuantizer::encode`]'s convention (integer
    /// levels stored as floats) rather than returning the raw companded
    /// value in `[-1, 1]` — generic code over [`SignalTokenizer`] (batch
    /// tokenizers, streaming tokenizers) treats `encode`'s output as token
    /// ids in `0..vocab_size()`, and `vocab_size()` reports `self.levels`.
    /// Use `MuLawCodec::encode_sample`-style companding directly (via
    /// [`MuLawCodec::quantize`]/[`MuLawCodec::dequantize`]) if only the
    /// continuous companded value is needed.
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(signal.mapv(|x| self.quantize(x) as f32))
    }

    /// Decode discrete μ-law levels (as produced by [`MuLawCodec::encode`])
    /// back into a continuous signal.
    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(tokens.mapv(|t| self.dequantize(t.round() as i32)))
    }

    fn embed_dim(&self) -> usize {
        1 // μ-law maintains dimensionality
    }

    fn vocab_size(&self) -> usize {
        self.levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mulaw_encode_decode() {
        let codec = MuLawCodec::new(8);

        // Test roundtrip for various values
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let encoded = codec.encode_sample(x);
            let decoded = codec.decode_sample(encoded);
            assert!((decoded - x).abs() < 0.01, "Roundtrip failed for {}", x);
        }
    }

    #[test]
    fn test_mulaw_quantize() {
        let codec = MuLawCodec::new(8);

        let level = codec.quantize(0.0);
        assert_eq!(level, 128); // Middle of 256 levels

        let level = codec.quantize(-1.0);
        assert_eq!(level, 0);

        let level = codec.quantize(1.0);
        assert_eq!(level, 255);
    }

    #[test]
    fn test_mulaw_quantize_within_vocab() {
        let codec = MuLawCodec::new(8);
        // Boundary values must be within vocab
        assert!((codec.quantize(1.0) as usize) < codec.vocab_size());
        assert!((codec.quantize(-1.0) as usize) < codec.vocab_size());
        // Mid-range values unchanged from before
        assert_eq!(codec.quantize(0.0), 128);
        assert_eq!(codec.quantize(-1.0), 0);
        assert_eq!(codec.quantize(1.0), 255);
    }

    #[test]
    fn test_mulaw_signal() {
        // `SignalTokenizer::encode`/`decode` now round-trip through the
        // *discrete* levels (see `regression_mulaw_signal_tokenizer_is_discrete`
        // below), matching `LinearQuantizer`'s convention instead of the raw
        // continuous companding. That introduces genuine quantization error
        // (largest at the range boundaries, where the companding curve is
        // steepest), so the tolerance is wider than the pure-companding
        // `test_mulaw_encode_decode` roundtrip above.
        let codec = MuLawCodec::new(8);
        let signal = Array1::from_vec(vec![0.0, 0.5, -0.5, 1.0, -1.0]);

        let encoded = codec.encode(&signal).unwrap();
        let decoded = codec.decode(&encoded).unwrap();

        for (orig, dec) in signal.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.05,
                "Signal roundtrip failed: {} vs {}",
                orig,
                dec
            );
        }
    }

    /// Regression: `SignalTokenizer::encode` must return discrete level
    /// indices in `0..vocab_size()` (matching `LinearQuantizer`), not raw
    /// companded floats in `[-1, 1]`. Generic code over `SignalTokenizer`
    /// (batch/streaming tokenizers) relies on `encode`'s output range
    /// matching `vocab_size()`.
    #[test]
    fn regression_mulaw_signal_tokenizer_is_discrete() {
        let codec = MuLawCodec::new(8);
        let signal = Array1::from_vec(vec![-1.0, -0.5, 0.0, 0.5, 1.0]);

        let encoded = codec.encode(&signal).unwrap();
        for &token in encoded.iter() {
            assert!(
                (0.0..codec.vocab_size() as f32).contains(&token),
                "token {token} outside 0..{}",
                codec.vocab_size()
            );
            assert_eq!(
                token,
                token.round(),
                "token {token} is not an integer level"
            );
        }
        // The trait path and the inherent quantize() path must agree exactly.
        for &x in signal.iter() {
            assert_eq!(
                codec.encode(&Array1::from_vec(vec![x])).unwrap()[[0]],
                codec.quantize(x) as f32
            );
        }
    }

    #[test]
    fn test_mulaw_new_clamps_invalid_bits() {
        // bits == 0 used to make mu == 0.0, turning encode_sample into 0.0/0.0 (NaN).
        let codec = MuLawCodec::new(0);
        assert!(codec.bits() >= MIN_BITS);
        assert!(codec.mu().is_finite() && codec.mu() > 0.0);
        assert!(codec.encode_sample(0.5).is_finite());

        // bits >= 64 used to overflow `1usize << bits` (panic in debug builds).
        let codec = MuLawCodec::new(200);
        assert!(codec.bits() <= MAX_BITS);
        assert!(codec.mu().is_finite() && codec.mu() > 0.0);
    }

    #[test]
    fn test_mulaw_try_new_rejects_invalid_bits() {
        assert!(MuLawCodec::try_new(0).is_err());
        assert!(MuLawCodec::try_new(17).is_err());
        assert!(MuLawCodec::try_new(200).is_err());
        assert!(MuLawCodec::try_new(8).is_ok());
        assert!(MuLawCodec::try_new(16).is_ok());
        assert!(MuLawCodec::try_new(1).is_ok());
    }

    #[test]
    fn test_mulaw_try_with_mu_rejects_invalid_mu() {
        assert!(MuLawCodec::try_with_mu(255.0, 8).is_ok());
        assert!(MuLawCodec::try_with_mu(0.0, 8).is_err());
        assert!(MuLawCodec::try_with_mu(-1.0, 8).is_err());
        assert!(MuLawCodec::try_with_mu(f32::NAN, 8).is_err());
        assert!(MuLawCodec::try_with_mu(f32::INFINITY, 8).is_err());
    }
}
