//! Linear quantization for signals
//!
//! Provides simple uniform quantization of continuous signals
//! into discrete levels.

use crate::error::{TokenizerError, TokenizerResult};
use crate::SignalTokenizer;
use scirs2_core::ndarray::Array1;

/// Trait for quantization strategies
pub trait Quantizer {
    /// Quantize a continuous value to discrete level
    fn quantize(&self, value: f32) -> i32;

    /// Dequantize a discrete level back to continuous value
    fn dequantize(&self, level: i32) -> f32;

    /// Get number of quantization levels
    fn num_levels(&self) -> usize;
}

/// Linear uniform quantizer
#[derive(Debug, Clone)]
pub struct LinearQuantizer {
    /// Minimum value of range
    min: f32,
    /// Maximum value of range
    max: f32,
    /// Number of bits
    bits: u8,
    /// Number of levels
    levels: usize,
    /// Step size
    step: f32,
}

impl LinearQuantizer {
    /// Create a new linear quantizer
    pub fn new(min: f32, max: f32, bits: u8) -> TokenizerResult<Self> {
        if min >= max {
            return Err(TokenizerError::InvalidConfig(
                "min must be less than max".into(),
            ));
        }
        if bits == 0 || bits > 16 {
            return Err(TokenizerError::InvalidConfig("bits must be 1-16".into()));
        }

        let levels = 1usize << bits;
        let step = (max - min) / levels as f32;

        Ok(Self {
            min,
            max,
            bits,
            levels,
            step,
        })
    }

    /// Create quantizer for normalized [-1, 1] range
    pub fn normalized(bits: u8) -> TokenizerResult<Self> {
        Self::new(-1.0, 1.0, bits)
    }

    /// Get the range
    pub fn range(&self) -> (f32, f32) {
        (self.min, self.max)
    }

    /// Get number of bits
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Get the quantization step size
    pub fn step_size(&self) -> f32 {
        self.step
    }
}

impl Quantizer for LinearQuantizer {
    fn quantize(&self, value: f32) -> i32 {
        let clamped = value.clamp(self.min, self.max);
        let normalized = (clamped - self.min) / (self.max - self.min);
        (normalized * (self.levels - 1) as f32).round() as i32
    }

    fn dequantize(&self, level: i32) -> f32 {
        let clamped_level = level.clamp(0, (self.levels - 1) as i32);
        let normalized = clamped_level as f32 / (self.levels - 1) as f32;
        self.min + normalized * (self.max - self.min)
    }

    fn num_levels(&self) -> usize {
        self.levels
    }
}

impl SignalTokenizer for LinearQuantizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        // Return quantized values as floats for embedding lookup
        Ok(signal.mapv(|x| self.quantize(x) as f32))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(tokens.mapv(|t| self.dequantize(t.round() as i32)))
    }

    fn embed_dim(&self) -> usize {
        1
    }

    fn vocab_size(&self) -> usize {
        self.levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_quantizer() {
        let q = LinearQuantizer::new(-1.0, 1.0, 8).unwrap();

        assert_eq!(q.num_levels(), 256);

        // Test center value
        let level = q.quantize(0.0);
        assert!((level - 127).abs() <= 1);

        // Test extremes
        assert_eq!(q.quantize(-1.0), 0);
        assert_eq!(q.quantize(1.0), 255);
    }

    #[test]
    fn test_roundtrip() {
        let q = LinearQuantizer::new(0.0, 100.0, 10).unwrap();

        for value in [0.0, 25.0, 50.0, 75.0, 100.0] {
            let level = q.quantize(value);
            let recovered = q.dequantize(level);
            assert!(
                (recovered - value).abs() < 0.2,
                "Roundtrip failed for {} -> {} -> {}",
                value,
                level,
                recovered
            );
        }
    }

    #[test]
    fn test_clamping() {
        let q = LinearQuantizer::new(-1.0, 1.0, 8).unwrap();

        // Values outside range should be clamped
        assert_eq!(q.quantize(-2.0), 0);
        assert_eq!(q.quantize(2.0), 255);
    }
}
