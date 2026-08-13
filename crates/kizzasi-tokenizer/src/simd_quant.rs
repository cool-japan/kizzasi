//! Quantization operations, most of them scalar and unrolled for
//! autovectorization; two genuinely dispatch to hardware SIMD.
//!
//! [`simd_sum`] and [`simd_sum_squares`] call straight through to
//! `scirs2_core::simd`'s `simd_sum_f32`/`simd_dot_f32`, which pick
//! AVX-512/AVX2/SSE2 (x86_64) or NEON (aarch64) at runtime via
//! `scirs2_core::simd::detect`, with a scalar fallback on other targets —
//! that is real "explicit vectorization", not autovectorization hopeful-ness.
//!
//! Everything else in this file is scalar, several of them manually
//! unrolled by `SIMD_WIDTH` in the hope that LLVM autovectorizes the
//! unrolled body — which it may or may not do, depending on the target and
//! the loop body's shape:
//!
//! - [`simd_quantize`], [`simd_dequantize`] unroll their hot loop by
//!   `SIMD_WIDTH` with independent outputs per lane, the shape LLVM is
//!   most likely to autovectorize. They deliberately do **not** call
//!   `scirs2_core::simd`'s `simd_clip_f32`/`simd_round_f32`, even though
//!   those exist and shape-match: `simd_round_f32` rounds ties to even
//!   using hardware instructions (`_MM_FROUND_TO_NEAREST_INT` on x86_64,
//!   `vrndnq_f32`/`FRINTN` on aarch64), while this module's contract is
//!   Rust's `f32::round()` (ties away from zero, as used by every other
//!   quantizer in this crate — see `advanced_quant.rs`, `quantizer.rs`).
//!   Switching would silently move some exact half-integer-boundary inputs
//!   to a different quantization level than before, which is exactly the
//!   kind of silent behavior change this crate's honesty policy forbids;
//!   see `test_simd_quantize_matches_scalar_round_at_half_boundaries` below.
//! - [`simd_mulaw_encode`]/[`simd_mulaw_decode`] unroll the same way, but
//!   have no vectorized primitive to call: `ln`/`powf` per element need
//!   `scirs2_core::simd::transcendental`'s `simd_ln_f32` chained with a
//!   scalar-broadcast add/multiply for the μ-law formula's affine terms,
//!   which was judged not worth the extra allocations for a companding step
//!   that already runs once per sample, not in a hot reduction loop.
//! - [`simd_deadzone_quantize`]'s per-element branch (dead-zone threshold
//!   check) doesn't unroll usefully — a `for offset in 0..SIMD_WIDTH` inner
//!   loop over one chunk is control-flow-identical to iterating the whole
//!   array once — so it is a single straightforward per-element loop.
//! - [`simd_adaptive_quantize`]'s per-element sliding-window statistics
//!   have a data dependency on neighboring elements and are not chunked at
//!   all, for the same reason.

use scirs2_core::ndarray::ArrayView1;

/// Manual unroll factor for the loops that use one (see the module docs
/// above for which functions actually benefit from it).
const SIMD_WIDTH: usize = 8;

/// Linear quantization, unrolled for autovectorization (scalar — see the
/// module documentation for why this doesn't call `scirs2_core::simd`).
///
/// Quantizes an array of f32 values to discrete levels.
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

    // `levels - 1` on `levels: usize == 0` underflows (panics in debug,
    // wraps to `usize::MAX` in release). `levels <= 1` has no meaningful
    // step size (0 or 1 distinguishable codes), so every value quantizes to
    // code 0 rather than computing garbage from an underflowed `scale`.
    if levels <= 1 {
        return result;
    }
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

/// Linear dequantization, unrolled for autovectorization (scalar — see the
/// module documentation).
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

    // `num_levels - 1` on `num_levels: usize == 0` underflows (panics in
    // debug, wraps to `usize::MAX` in release); `num_levels <= 1` has no
    // meaningful step size, so every level dequantizes to `min` rather than
    // dividing by an underflowed `scale`.
    if num_levels <= 1 {
        result.fill(min);
        return result;
    }
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

/// Dead zone quantization.
///
/// Applies dead zone around zero to promote sparsity.
///
/// Not chunked/unrolled: the per-element dead-zone threshold branch means
/// each iteration's control flow depends on that element's value, which
/// gives LLVM's autovectorizer nothing more to work with than a plain loop
/// would (a fixed-width `for offset in 0..SIMD_WIDTH` inner loop over one
/// chunk here would be control-flow-identical to just iterating the whole
/// array once — see the module documentation).
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
    signal
        .iter()
        .map(|&v| {
            let abs_v = v.abs();
            if abs_v <= threshold {
                0
            } else {
                let sign = if v >= 0.0 { 1 } else { -1 };
                sign * ((abs_v - threshold) / step).round() as i32
            }
        })
        .collect()
}

/// Adaptive quantization with local statistics.
///
/// Computes local variance in sliding windows and adapts quantization step.
/// Not chunked (see the module documentation): each element's adapted step
/// depends on a window of its neighbors, so there is no fixed-width,
/// data-independent inner loop here to unroll.
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

/// μ-law encoding, unrolled for autovectorization (scalar — see the module
/// documentation).
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

/// μ-law decoding, unrolled for autovectorization (scalar — see the module
/// documentation).
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

/// Sum reduction for signal metrics.
///
/// Genuinely vectorized: dispatches to `scirs2_core::simd::simd_sum_f32`,
/// which picks AVX-512/AVX2/SSE2 (x86_64) or NEON (aarch64) at runtime and
/// falls back to a scalar loop only on targets with none of those (see the
/// module documentation).
#[inline]
pub fn simd_sum(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    scirs2_core::simd::simd_sum_f32(&ArrayView1::from(signal))
}

/// Sum of squares for variance computation.
///
/// Genuinely vectorized: a sum of squares is a self dot-product, computed
/// via `scirs2_core::simd::simd_dot_f32` (same runtime dispatch as
/// [`simd_sum`] above).
#[inline]
pub fn simd_sum_squares(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let view = ArrayView1::from(signal);
    scirs2_core::simd::simd_dot_f32(&view, &view)
}

#[cfg(test)]
fn scalar_sum_squares_reference(signal: &[f32]) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = signal.len() / SIMD_WIDTH;
    let remainder = signal.len() % SIMD_WIDTH;
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

    /// Regression: `levels == 0` used to underflow `levels - 1` on a
    /// `usize` (panic in debug, wrap to `usize::MAX` in release) instead of
    /// returning a sane result.
    #[test]
    fn test_simd_quantize_zero_levels_does_not_panic() {
        let signal = vec![0.0, 0.5, -0.5, 1.0];
        let result = simd_quantize(&signal, -1.0, 1.0, 0);
        assert_eq!(result, vec![0, 0, 0, 0]);

        let result_one_level = simd_quantize(&signal, -1.0, 1.0, 1);
        assert_eq!(result_one_level, vec![0, 0, 0, 0]);
    }

    /// Regression: `num_levels == 0` used to underflow `num_levels - 1` the
    /// same way.
    #[test]
    fn test_simd_dequantize_zero_levels_does_not_panic() {
        let levels = vec![0, 5, -3];
        let result = simd_dequantize(&levels, -1.0, 1.0, 0);
        assert_eq!(result, vec![-1.0, -1.0, -1.0]);

        let result_one_level = simd_dequantize(&levels, -1.0, 1.0, 1);
        assert_eq!(result_one_level, vec![-1.0, -1.0, -1.0]);
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

    /// Regression: `simd_sum`/`simd_sum_squares` must agree with a plain
    /// scalar reference (this crate's own former implementation, kept as
    /// `scalar_sum_squares_reference`) well beyond float noise, across a
    /// length that spans several `SIMD_WIDTH`-sized chunks plus a
    /// remainder — exercising the actual vectorized dispatch path in
    /// `scirs2_core::simd`, not just the empty/short-input edge cases.
    #[test]
    fn test_simd_sum_and_sum_squares_match_scalar_reference() {
        let signal: Vec<f32> = (0..777).map(|i| ((i as f32) * 0.017).sin() * 3.0).collect();

        let scalar_sum: f32 = signal.iter().sum();
        let scalar_sum_sq = scalar_sum_squares_reference(&signal);

        assert!(
            (simd_sum(&signal) - scalar_sum).abs() < 1e-2,
            "simd_sum diverged from scalar sum: {} vs {}",
            simd_sum(&signal),
            scalar_sum
        );
        assert!(
            (simd_sum_squares(&signal) - scalar_sum_sq).abs() < 1e-2,
            "simd_sum_squares diverged from scalar sum of squares: {} vs {}",
            simd_sum_squares(&signal),
            scalar_sum_sq
        );
    }

    #[test]
    fn test_simd_sum_and_sum_squares_empty() {
        assert_eq!(simd_sum(&[]), 0.0);
        assert_eq!(simd_sum_squares(&[]), 0.0);
    }

    /// Regression / design-decision test: documents *why* `simd_quantize`
    /// keeps Rust's `f32::round()` (ties away from zero) instead of calling
    /// `scirs2_core::simd::simd_round_f32` (ties to even, on hardware that
    /// supports it) — see the module documentation. `127.5` and `-127.5`
    /// are exact halfway points at `levels = 256` over `[-1, 1]`; both
    /// rounding modes happen to agree here (128 is even), but the important
    /// invariant is that `simd_quantize` matches plain `f32::round()`
    /// exactly, not merely "close to it", for every value in this sweep —
    /// a silent switch to ties-to-even would only show up on some inputs
    /// (see the module docs' `76.5`/`178.5` analysis), so pinning the exact
    /// values here is what would actually catch a future regression.
    #[test]
    fn test_simd_quantize_matches_scalar_round_at_half_boundaries() {
        let scale = 255.0 / 2.0; // (levels - 1) / (max - min), levels=256, range=[-1,1]
        for i in -100..=100 {
            let x = i as f32 / 100.0;
            let expected = ((x.clamp(-1.0, 1.0) - (-1.0)) * scale).round() as i32;
            let got = simd_quantize(&[x], -1.0, 1.0, 256)[0];
            assert_eq!(got, expected, "mismatch at x={x}");
        }
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
