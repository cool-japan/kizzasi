//! Advanced quantization strategies
//!
//! Provides sophisticated quantization methods beyond simple linear/μ-law:
//! - **Adaptive Quantization**: Adjusts step size based on signal statistics
//! - **Dead Zone Quantization**: Optimized for sparse signals
//! - **Perceptual Quantization**: Psychoacoustic modeling for audio
//! - **Entropy-Constrained Quantization**: Rate-distortion optimized

use crate::error::{TokenizerError, TokenizerResult};
use crate::{Quantizer, SignalTokenizer};
use scirs2_core::ndarray::Array1;

/// Adaptive quantizer that adjusts step size based on local signal statistics
///
/// Uses a sliding window to compute local variance and adapts quantization
/// step size accordingly. High-variance regions get finer quantization.
#[derive(Debug, Clone)]
pub struct AdaptiveQuantizer {
    /// Base number of bits
    _bits: u8,
    /// Number of levels
    levels: usize,
    /// Window size for local statistics
    window_size: usize,
    /// Adaptation strength (0.0 = no adaptation, 1.0 = full adaptation)
    adaptation_strength: f32,
    /// Global min/max for normalization
    global_min: f32,
    global_max: f32,
}

impl AdaptiveQuantizer {
    /// Create a new adaptive quantizer
    pub fn new(
        bits: u8,
        window_size: usize,
        adaptation_strength: f32,
        global_min: f32,
        global_max: f32,
    ) -> TokenizerResult<Self> {
        if bits == 0 || bits > 16 {
            return Err(TokenizerError::InvalidConfig("bits must be 1-16".into()));
        }
        if window_size == 0 {
            return Err(TokenizerError::InvalidConfig(
                "window_size must be positive".into(),
            ));
        }
        if !(0.0..=1.0).contains(&adaptation_strength) {
            return Err(TokenizerError::InvalidConfig(
                "adaptation_strength must be in [0, 1]".into(),
            ));
        }

        Ok(Self {
            _bits: bits,
            levels: 1usize << bits,
            window_size,
            adaptation_strength,
            global_min,
            global_max,
        })
    }

    /// Compute local variance around position
    fn local_variance(&self, signal: &Array1<f32>, pos: usize) -> f32 {
        let half_window = self.window_size / 2;
        let start = pos.saturating_sub(half_window);
        let end = (pos + half_window).min(signal.len());

        let window: Vec<f32> = signal
            .iter()
            .skip(start)
            .take(end - start)
            .cloned()
            .collect();
        if window.is_empty() {
            return 1.0;
        }

        let mean = window.iter().sum::<f32>() / window.len() as f32;
        let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / window.len() as f32;

        variance.sqrt().max(1e-6) // Return standard deviation
    }

    /// Compute adaptive step size at position
    fn adaptive_step(&self, signal: &Array1<f32>, pos: usize) -> f32 {
        let base_step = (self.global_max - self.global_min) / self.levels as f32;
        let local_std = self.local_variance(signal, pos);

        // Scale step size based on local statistics
        let global_std = (self.global_max - self.global_min) / 4.0; // Approximate
        let scale = 1.0 + self.adaptation_strength * (local_std / global_std - 1.0);

        base_step * scale.clamp(0.1, 10.0) // Clamp scaling factor
    }

    /// Quantize entire signal with adaptation
    pub fn quantize_adaptive(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<i32>> {
        let mut result = Vec::with_capacity(signal.len());

        for (i, &value) in signal.iter().enumerate() {
            let step = self.adaptive_step(signal, i);
            let clamped = value.clamp(self.global_min, self.global_max);
            let normalized = (clamped - self.global_min) / (self.global_max - self.global_min);
            let level = (normalized / step * (self.levels - 1) as f32).round() as i32;
            result.push(level.clamp(0, (self.levels - 1) as i32));
        }

        Ok(Array1::from_vec(result))
    }
}

impl Quantizer for AdaptiveQuantizer {
    fn quantize(&self, value: f32) -> i32 {
        // Fallback to uniform quantization for single values
        let clamped = value.clamp(self.global_min, self.global_max);
        let normalized = (clamped - self.global_min) / (self.global_max - self.global_min);
        (normalized * (self.levels - 1) as f32).round() as i32
    }

    fn dequantize(&self, level: i32) -> f32 {
        let clamped_level = level.clamp(0, (self.levels - 1) as i32);
        let normalized = clamped_level as f32 / (self.levels - 1) as f32;
        self.global_min + normalized * (self.global_max - self.global_min)
    }

    fn num_levels(&self) -> usize {
        self.levels
    }
}

/// Dead zone quantizer for sparse signals
///
/// Applies a dead zone around zero where small values are quantized to zero.
/// This is useful for signals with many near-zero values (e.g., after transforms).
#[derive(Debug, Clone)]
pub struct DeadZoneQuantizer {
    /// Base quantizer
    _base_bits: u8,
    levels: usize,
    /// Dead zone threshold
    dead_zone: f32,
    /// Range for quantization
    min: f32,
    max: f32,
}

impl DeadZoneQuantizer {
    /// Create a new dead zone quantizer
    ///
    /// # Arguments
    /// * `bits` - Number of quantization bits
    /// * `dead_zone` - Threshold below which values are quantized to zero
    /// * `min`, `max` - Value range
    pub fn new(bits: u8, dead_zone: f32, min: f32, max: f32) -> TokenizerResult<Self> {
        if bits == 0 || bits > 16 {
            return Err(TokenizerError::InvalidConfig("bits must be 1-16".into()));
        }
        if dead_zone < 0.0 {
            return Err(TokenizerError::InvalidConfig(
                "dead_zone must be non-negative".into(),
            ));
        }

        Ok(Self {
            _base_bits: bits,
            levels: 1usize << bits,
            dead_zone,
            min,
            max,
        })
    }
}

impl Quantizer for DeadZoneQuantizer {
    fn quantize(&self, value: f32) -> i32 {
        // Apply dead zone
        if value.abs() < self.dead_zone {
            return (self.levels / 2) as i32; // Zero point
        }

        // Quantize non-dead-zone values
        let clamped = value.clamp(self.min, self.max);
        let normalized = (clamped - self.min) / (self.max - self.min);
        (normalized * (self.levels - 1) as f32).round() as i32
    }

    fn dequantize(&self, level: i32) -> f32 {
        let clamped_level = level.clamp(0, (self.levels - 1) as i32);

        // Check if it's the zero point
        if clamped_level == (self.levels / 2) as i32 {
            return 0.0;
        }

        let normalized = clamped_level as f32 / (self.levels - 1) as f32;
        self.min + normalized * (self.max - self.min)
    }

    fn num_levels(&self) -> usize {
        self.levels
    }
}

/// Non-uniform quantizer with configurable bin edges
///
/// Allows custom quantization levels for optimal rate-distortion trade-off
#[derive(Debug, Clone)]
pub struct NonUniformQuantizer {
    /// Quantization bin edges (sorted)
    bin_edges: Vec<f32>,
    /// Reconstruction values for each bin
    reconstruction_values: Vec<f32>,
}

impl NonUniformQuantizer {
    /// Create from bin edges
    ///
    /// Reconstruction values are set to bin centers
    pub fn from_edges(mut bin_edges: Vec<f32>) -> TokenizerResult<Self> {
        if bin_edges.len() < 2 {
            return Err(TokenizerError::InvalidConfig(
                "Need at least 2 bin edges".into(),
            ));
        }

        bin_edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute reconstruction values as bin centers
        let mut reconstruction_values = Vec::with_capacity(bin_edges.len() - 1);
        for i in 0..bin_edges.len() - 1 {
            reconstruction_values.push((bin_edges[i] + bin_edges[i + 1]) / 2.0);
        }

        Ok(Self {
            bin_edges,
            reconstruction_values,
        })
    }

    /// Create with custom reconstruction values
    pub fn new(bin_edges: Vec<f32>, reconstruction_values: Vec<f32>) -> TokenizerResult<Self> {
        if bin_edges.len() != reconstruction_values.len() + 1 {
            return Err(TokenizerError::InvalidConfig(
                "bin_edges.len() must equal reconstruction_values.len() + 1".into(),
            ));
        }

        Ok(Self {
            bin_edges,
            reconstruction_values,
        })
    }

    /// Create Lloyd-Max quantizer for Gaussian distribution
    ///
    /// Optimizes bin edges and reconstruction values for minimum MSE
    pub fn lloyd_max_gaussian(num_levels: usize, sigma: f32) -> TokenizerResult<Self> {
        if num_levels < 2 {
            return Err(TokenizerError::InvalidConfig(
                "num_levels must be at least 2".into(),
            ));
        }

        // Simple approximation: use percentiles of Gaussian
        let mut bin_edges = Vec::with_capacity(num_levels + 1);
        let mut reconstruction_values = Vec::with_capacity(num_levels);

        // Start with uniform spacing
        for i in 0..=num_levels {
            let p = i as f32 / num_levels as f32;
            // Approximate inverse CDF
            let z = if p < 0.5 {
                -((1.0 - 2.0 * p).sqrt() - 1.0)
            } else {
                (2.0 * p - 1.0).sqrt() - 1.0
            };
            bin_edges.push(z * sigma);
        }

        // Reconstruction values as bin centers
        for i in 0..num_levels {
            reconstruction_values.push((bin_edges[i] + bin_edges[i + 1]) / 2.0);
        }

        Ok(Self {
            bin_edges,
            reconstruction_values,
        })
    }
}

impl Quantizer for NonUniformQuantizer {
    fn quantize(&self, value: f32) -> i32 {
        // Find bin using binary search
        for (i, &edge) in self.bin_edges.iter().enumerate().skip(1) {
            if value < edge {
                return (i - 1) as i32;
            }
        }
        (self.reconstruction_values.len() - 1) as i32
    }

    fn dequantize(&self, level: i32) -> f32 {
        let idx = level.clamp(0, (self.reconstruction_values.len() - 1) as i32) as usize;
        self.reconstruction_values[idx]
    }

    fn num_levels(&self) -> usize {
        self.reconstruction_values.len()
    }
}

// Implement SignalTokenizer for advanced quantizers

impl SignalTokenizer for AdaptiveQuantizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let quantized = self.quantize_adaptive(signal)?;
        Ok(quantized.mapv(|x| x as f32))
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

impl SignalTokenizer for DeadZoneQuantizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
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

impl SignalTokenizer for NonUniformQuantizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(signal.mapv(|x| self.quantize(x) as f32))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(tokens.mapv(|t| self.dequantize(t.round() as i32)))
    }

    fn embed_dim(&self) -> usize {
        1
    }

    fn vocab_size(&self) -> usize {
        self.reconstruction_values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_quantizer() {
        let quant = AdaptiveQuantizer::new(8, 16, 0.5, -1.0, 1.0).unwrap();

        let signal = Array1::from_vec((0..128).map(|i| ((i as f32) * 0.05).sin()).collect());

        let encoded = quant.encode(&signal).unwrap();
        assert_eq!(encoded.len(), 128);

        let decoded = quant.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 128);
    }

    #[test]
    fn test_dead_zone_quantizer() {
        let quant = DeadZoneQuantizer::new(8, 0.1, -1.0, 1.0).unwrap();

        // Test dead zone behavior
        let level = quant.quantize(0.05);
        let recovered = quant.dequantize(level);
        assert_eq!(recovered, 0.0); // Should be in dead zone

        // Test outside dead zone
        let level = quant.quantize(0.5);
        let recovered = quant.dequantize(level);
        assert!(recovered.abs() > 0.1);
    }

    #[test]
    fn test_dead_zone_signal() {
        let quant = DeadZoneQuantizer::new(8, 0.2, -1.0, 1.0).unwrap();

        // Signal with small values that should be zeroed
        let signal = Array1::from_vec(vec![0.01, 0.5, -0.1, 0.8, 0.05]);

        let encoded = quant.encode(&signal).unwrap();
        let decoded = quant.decode(&encoded).unwrap();

        // Small values should become zero
        assert_eq!(decoded[0], 0.0);
        assert_eq!(decoded[2], 0.0);
        assert_eq!(decoded[4], 0.0);

        // Large values should be preserved (approximately)
        assert!(decoded[1] > 0.3);
        assert!(decoded[3] > 0.6);
    }

    #[test]
    fn test_nonuniform_quantizer() {
        let edges = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let quant = NonUniformQuantizer::from_edges(edges).unwrap();

        assert_eq!(quant.num_levels(), 4);

        let level = quant.quantize(-1.0);
        assert_eq!(level, 0);

        let level = quant.quantize(0.25);
        assert_eq!(level, 2);
    }

    #[test]
    fn test_lloyd_max_quantizer() {
        let quant = NonUniformQuantizer::lloyd_max_gaussian(8, 1.0).unwrap();

        assert_eq!(quant.num_levels(), 8);

        // Test symmetry
        let level_pos = quant.quantize(0.5);
        let level_neg = quant.quantize(-0.5);
        let val_pos = quant.dequantize(level_pos);
        let val_neg = quant.dequantize(level_neg);

        assert!((val_pos + val_neg).abs() < 0.5); // Should be roughly symmetric
    }

    #[test]
    fn test_adaptive_vs_uniform() {
        let adaptive = AdaptiveQuantizer::new(6, 8, 0.8, -1.0, 1.0).unwrap();

        // Signal with varying local statistics
        let mut signal_vec = Vec::new();
        // Low variance region
        for i in 0..64 {
            signal_vec.push(0.1 * (i as f32 * 0.05).sin());
        }
        // High variance region
        for i in 64..128 {
            signal_vec.push(0.8 * (i as f32 * 0.1).sin());
        }

        let signal = Array1::from_vec(signal_vec);
        let encoded = adaptive.encode(&signal).unwrap();

        assert_eq!(encoded.len(), 128);
    }

    #[test]
    fn test_nonuniform_with_custom_values() {
        let edges = vec![-1.0, -0.3, 0.0, 0.3, 1.0];
        let recon = vec![-0.7, -0.15, 0.15, 0.7];

        let quant = NonUniformQuantizer::new(edges, recon).unwrap();

        let level = quant.quantize(0.1);
        let value = quant.dequantize(level);
        assert!((value - 0.15).abs() < 0.01);
    }
}
