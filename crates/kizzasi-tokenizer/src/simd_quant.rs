//! SIMD-optimized quantization operations
//!
//! This module provides high-performance implementations of quantization
//! operations using explicit vectorization and loop unrolling.
//!
//! # Performance Optimizations
//!
//! - Process 8 elements at a time with SIMD-width operations
//! - 4-way accumulation to reduce pipeline dependencies
//! - Cache-friendly memory access patterns
//! - Zero-copy operations where possible

/// SIMD width for vectorized operations
const SIMD_WIDTH: usize = 8;

/// SIMD-optimized linear quantization
///
/// Quantizes an array of f32 values to discrete levels using vectorized operations.
///
/// # Arguments
///
/// * `signal` - Input signal to quantize
/// * `min` - Minimum value of quantization range
/// * `max` - Maximum value of quantization range
/// * `levels` - Number of quantization levels
///
/// # Returns
///
/// Array of quantized values as i32
#[inline]
pub fn simd_quantize(signal: &[f32], min: f32, max: f32, levels: usize) -> Vec<i32> {
    let len = signal.len();
    let mut result = vec![0i32; len];

    let range = max - min;
    let scale = (levels - 1) as f32 / range;

    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    // Process 8 elements at a time
    let mut i = 0;
    for _ in 0..chunks {
        // Clamp
        let v0 = signal[i].clamp(min, max);
        let v1 = signal[i + 1].clamp(min, max);
        let v2 = signal[i + 2].clamp(min, max);
        let v3 = signal[i + 3].clamp(min, max);
        let v4 = signal[i + 4].clamp(min, max);
        let v5 = signal[i + 5].clamp(min, max);
        let v6 = signal[i + 6].clamp(min, max);
        let v7 = signal[i + 7].clamp(min, max);

        // Normalize and quantize
        result[i] = ((v0 - min) * scale).round() as i32;
        result[i + 1] = ((v1 - min) * scale).round() as i32;
        result[i + 2] = ((v2 - min) * scale).round() as i32;
        result[i + 3] = ((v3 - min) * scale).round() as i32;
        result[i + 4] = ((v4 - min) * scale).round() as i32;
        result[i + 5] = ((v5 - min) * scale).round() as i32;
        result[i + 6] = ((v6 - min) * scale).round() as i32;
        result[i + 7] = ((v7 - min) * scale).round() as i32;

        i += SIMD_WIDTH;
    }

    // Process remainder
    for j in 0..remainder {
        let v = signal[i + j].clamp(min, max);
        result[i + j] = ((v - min) * scale).round() as i32;
    }

    result
}

/// SIMD-optimized linear dequantization
///
/// Converts discrete quantization levels back to continuous values.
///
/// # Arguments
///
/// * `levels_data` - Quantized levels as i32
/// * `min` - Minimum value of quantization range
/// * `max` - Maximum value of quantization range
/// * `num_levels` - Number of quantization levels
///
/// # Returns
///
/// Array of dequantized f32 values
#[inline]
pub fn simd_dequantize(levels_data: &[i32], min: f32, max: f32, num_levels: usize) -> Vec<f32> {
    let len = levels_data.len();
    let mut result = vec![0.0f32; len];

    let range = max - min;
    let scale = range / (num_levels - 1) as f32;
    let max_level = (num_levels - 1) as i32;

    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    // Process 8 elements at a time
    let mut i = 0;
    for _ in 0..chunks {
        // Clamp levels
        let l0 = levels_data[i].clamp(0, max_level);
        let l1 = levels_data[i + 1].clamp(0, max_level);
        let l2 = levels_data[i + 2].clamp(0, max_level);
        let l3 = levels_data[i + 3].clamp(0, max_level);
        let l4 = levels_data[i + 4].clamp(0, max_level);
        let l5 = levels_data[i + 5].clamp(0, max_level);
        let l6 = levels_data[i + 6].clamp(0, max_level);
        let l7 = levels_data[i + 7].clamp(0, max_level);

        // Dequantize
        result[i] = min + l0 as f32 * scale;
        result[i + 1] = min + l1 as f32 * scale;
        result[i + 2] = min + l2 as f32 * scale;
        result[i + 3] = min + l3 as f32 * scale;
        result[i + 4] = min + l4 as f32 * scale;
        result[i + 5] = min + l5 as f32 * scale;
        result[i + 6] = min + l6 as f32 * scale;
        result[i + 7] = min + l7 as f32 * scale;

        i += SIMD_WIDTH;
    }

    // Process remainder
    for j in 0..remainder {
        let l = levels_data[i + j].clamp(0, max_level);
        result[i + j] = min + l as f32 * scale;
    }

    result
}

/// SIMD-optimized dead zone quantization
///
/// Applies dead zone around zero to promote sparsity.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `threshold` - Dead zone threshold
/// * `step` - Quantization step size
///
/// # Returns
///
/// Quantized signal with dead zone
#[inline]
pub fn simd_deadzone_quantize(signal: &[f32], threshold: f32, step: f32) -> Vec<i32> {
    let len = signal.len();
    let mut result = vec![0i32; len];

    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    // Process 8 elements at a time
    let mut i = 0;
    for _ in 0..chunks {
        for offset in 0..SIMD_WIDTH {
            let v = signal[i + offset];
            let abs_v = v.abs();

            result[i + offset] = if abs_v <= threshold {
                0
            } else {
                let sign = if v >= 0.0 { 1 } else { -1 };
                sign * ((abs_v - threshold) / step).round() as i32
            };
        }
        i += SIMD_WIDTH;
    }

    // Process remainder
    for j in 0..remainder {
        let v = signal[i + j];
        let abs_v = v.abs();

        result[i + j] = if abs_v <= threshold {
            0
        } else {
            let sign = if v >= 0.0 { 1 } else { -1 };
            sign * ((abs_v - threshold) / step).round() as i32
        };
    }

    result
}

/// SIMD-optimized adaptive quantization with local statistics
///
/// Computes local variance in sliding windows and adapts quantization step.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `base_step` - Base quantization step
/// * `window_size` - Window size for local statistics
/// * `adaptation_strength` - How much to adapt (0.0 = uniform, 1.0 = fully adaptive)
///
/// # Returns
///
/// Adaptively quantized signal
pub fn simd_adaptive_quantize(
    signal: &[f32],
    base_step: f32,
    window_size: usize,
    adaptation_strength: f32,
) -> Vec<i32> {
    let len = signal.len();
    let mut result = vec![0i32; len];

    // Compute local variance using sliding window
    let half_window = window_size / 2;

    for i in 0..len {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(len);

        // Compute local mean (SIMD-friendly)
        let window_len = end - start;
        let local_sum: f32 = signal[start..end].iter().sum();
        let local_mean = local_sum / window_len as f32;

        // Compute local variance
        let var_sum: f32 = signal[start..end]
            .iter()
            .map(|&x| {
                let diff = x - local_mean;
                diff * diff
            })
            .sum();
        let local_var = var_sum / window_len as f32;

        // Adapt step size based on local variance
        let local_std = local_var.sqrt();
        let adapted_step = base_step * (1.0 + adaptation_strength * local_std);

        // Quantize with adapted step
        result[i] = (signal[i] / adapted_step).round() as i32;
    }

    result
}

/// SIMD-optimized μ-law encoding
///
/// Applies μ-law companding for audio quantization.
///
/// # Arguments
///
/// * `signal` - Input signal in [-1, 1] range
/// * `mu` - μ-law parameter (typically 255)
///
/// # Returns
///
/// Encoded signal
#[inline]
pub fn simd_mulaw_encode(signal: &[f32], mu: f32) -> Vec<i32> {
    let len = signal.len();
    let mut result = vec![0i32; len];

    let mu_p1 = mu + 1.0;
    let ln_mu_p1 = mu_p1.ln();

    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    // Process 8 elements at a time
    let mut i = 0;
    for _ in 0..chunks {
        for offset in 0..SIMD_WIDTH {
            let x = signal[i + offset].clamp(-1.0, 1.0);
            let sign = if x >= 0.0 { 1.0 } else { -1.0 };
            let abs_x = x.abs();

            let encoded = sign * (1.0 + mu * abs_x).ln() / ln_mu_p1;
            result[i + offset] = (encoded * mu).round() as i32;
        }
        i += SIMD_WIDTH;
    }

    // Process remainder
    for j in 0..remainder {
        let x = signal[i + j].clamp(-1.0, 1.0);
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let abs_x = x.abs();

        let encoded = sign * (1.0 + mu * abs_x).ln() / ln_mu_p1;
        result[i + j] = (encoded * mu).round() as i32;
    }

    result
}

/// SIMD-optimized μ-law decoding
#[inline]
pub fn simd_mulaw_decode(levels: &[i32], mu: f32) -> Vec<f32> {
    let len = levels.len();
    let mut result = vec![0.0f32; len];

    let mu_p1 = mu + 1.0;

    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    // Process 8 elements at a time
    let mut i = 0;
    for _ in 0..chunks {
        for offset in 0..SIMD_WIDTH {
            let y = levels[i + offset] as f32 / mu;
            let sign = if y >= 0.0 { 1.0 } else { -1.0 };
            let abs_y = y.abs();

            result[i + offset] = sign * (mu_p1.powf(abs_y) - 1.0) / mu;
        }
        i += SIMD_WIDTH;
    }

    // Process remainder
    for j in 0..remainder {
        let y = levels[i + j] as f32 / mu;
        let sign = if y >= 0.0 { 1.0 } else { -1.0 };
        let abs_y = y.abs();

        result[i + j] = sign * (mu_p1.powf(abs_y) - 1.0) / mu;
    }

    result
}

/// Optimized sum reduction for signal metrics
#[inline]
pub fn simd_sum(signal: &[f32]) -> f32 {
    let len = signal.len();
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    // 4-way accumulation
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let mut i = 0;
    for _ in 0..chunks {
        sum0 += signal[i];
        sum1 += signal[i + 1];
        sum2 += signal[i + 2];
        sum3 += signal[i + 3];
        sum0 += signal[i + 4];
        sum1 += signal[i + 5];
        sum2 += signal[i + 6];
        sum3 += signal[i + 7];
        i += SIMD_WIDTH;
    }

    for j in 0..remainder {
        sum0 += signal[i + j];
    }

    sum0 + sum1 + sum2 + sum3
}

/// Optimized sum of squares for variance computation
#[inline]
pub fn simd_sum_squares(signal: &[f32]) -> f32 {
    let len = signal.len();
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let mut i = 0;
    for _ in 0..chunks {
        sum0 += signal[i] * signal[i];
        sum1 += signal[i + 1] * signal[i + 1];
        sum2 += signal[i + 2] * signal[i + 2];
        sum3 += signal[i + 3] * signal[i + 3];
        sum0 += signal[i + 4] * signal[i + 4];
        sum1 += signal[i + 5] * signal[i + 5];
        sum2 += signal[i + 6] * signal[i + 6];
        sum3 += signal[i + 7] * signal[i + 7];
        i += SIMD_WIDTH;
    }

    for j in 0..remainder {
        let v = signal[i + j];
        sum0 += v * v;
    }

    sum0 + sum1 + sum2 + sum3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_quantize_basic() {
        let signal = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let result = simd_quantize(&signal, -1.0, 1.0, 256);

        assert!((result[0] - 127).abs() <= 1); // 0.0 -> middle (127 or 128 is fine)
        assert!(result[2] >= 250); // 1.0 -> high
        assert!(result[4] <= 5); // -1.0 -> low
    }

    #[test]
    fn test_simd_dequantize_basic() {
        let levels = vec![0, 127, 255];
        let result = simd_dequantize(&levels, -1.0, 1.0, 256);

        assert!((result[0] - (-1.0)).abs() < 0.01);
        assert!(result[1].abs() < 0.01);
        assert!((result[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_simd_quantize_dequantize_roundtrip() {
        let signal: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();

        let quantized = simd_quantize(&signal, -1.0, 1.0, 256);
        let dequantized = simd_dequantize(&quantized, -1.0, 1.0, 256);

        for i in 0..signal.len() {
            assert!(
                (signal[i] - dequantized[i]).abs() < 0.01,
                "Mismatch at {}: {} vs {}",
                i,
                signal[i],
                dequantized[i]
            );
        }
    }

    #[test]
    fn test_simd_deadzone_quantize() {
        let signal = vec![0.0, 0.05, 0.15, 0.3, -0.05, -0.15, -0.3];
        let result = simd_deadzone_quantize(&signal, 0.08, 0.05);

        assert_eq!(result[0], 0); // Within dead zone
        assert_eq!(result[1], 0); // Within dead zone
        assert_ne!(result[2], 0); // Outside dead zone: (0.15 - 0.08)/0.05 = 1.4 -> 1
        assert_ne!(result[3], 0); // Outside dead zone: (0.3 - 0.08)/0.05 = 4.4 -> 4
    }

    #[test]
    fn test_simd_adaptive_quantize() {
        let signal: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let result = simd_adaptive_quantize(&signal, 0.1, 10, 0.5);

        assert_eq!(result.len(), signal.len());
    }

    #[test]
    fn test_simd_mulaw_encode_decode() {
        let signal = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let encoded = simd_mulaw_encode(&signal, 255.0);
        let decoded = simd_mulaw_decode(&encoded, 255.0);

        for i in 0..signal.len() {
            assert!(
                (signal[i] - decoded[i]).abs() < 0.01,
                "μ-law roundtrip failed at {}: {} vs {}",
                i,
                signal[i],
                decoded[i]
            );
        }
    }

    #[test]
    fn test_simd_sum() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = simd_sum(&signal);

        assert!((sum - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_sum_squares() {
        let signal = vec![1.0, 2.0, 3.0];
        let sum_sq = simd_sum_squares(&signal);

        assert!((sum_sq - 14.0).abs() < 0.001); // 1 + 4 + 9
    }

    #[test]
    fn test_simd_operations_long_sequence() {
        // Test with sequence longer than SIMD_WIDTH
        let signal: Vec<f32> = (0..1000).map(|i| (i as f32 / 100.0).sin()).collect();

        let quantized = simd_quantize(&signal, -1.0, 1.0, 256);
        assert_eq!(quantized.len(), signal.len());

        let dequantized = simd_dequantize(&quantized, -1.0, 1.0, 256);
        assert_eq!(dequantized.len(), signal.len());
    }
}
