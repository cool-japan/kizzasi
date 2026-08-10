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

use crate::error::TokenizerResult;
use crate::SignalTokenizer;
use scirs2_core::ndarray::Array1;

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
    /// Create a new μ-law codec with specified bits
    pub fn new(bits: u8) -> Self {
        let levels = 1usize << bits;
        let mu = (levels - 1) as f32;
        Self { mu, bits, levels }
    }

    /// Create with custom μ value
    pub fn with_mu(mu: f32, bits: u8) -> Self {
        Self {
            mu,
            bits,
            levels: 1usize << bits,
        }
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
}

impl SignalTokenizer for MuLawCodec {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(signal.mapv(|x| self.encode_sample(x)))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(tokens.mapv(|y| self.decode_sample(y)))
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
        let codec = MuLawCodec::new(8);
        let signal = Array1::from_vec(vec![0.0, 0.5, -0.5, 1.0, -1.0]);

        let encoded = codec.encode(&signal).unwrap();
        let decoded = codec.decode(&encoded).unwrap();

        for (orig, dec) in signal.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.01,
                "Signal roundtrip failed: {} vs {}",
                orig,
                dec
            );
        }
    }
}
