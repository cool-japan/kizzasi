//! Lightweight INT8 quantization for embedded inference.
//!
//! Provides symmetric per-tensor quantization and dequantization without
//! heap allocation for the core scalar operations. The vectorized
//! `quantize_symmetric` helper is available when `alloc` (or `std`) is enabled.

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::vec::Vec;

use crate::error::{EmbeddedError, EmbeddedResult};
use crate::math::core_math;

/// Maximum absolute value of a slice (no_std compatible).
pub fn max_abs(values: &[f32]) -> f32 {
    values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max)
}

/// Compute scale factor for symmetric INT8 quantization.
///
/// Scale = max_abs / 127. Returns 1.0 for near-zero tensors to avoid
/// division by zero during quantization.
pub fn compute_scale(values: &[f32]) -> f32 {
    let max = max_abs(values);
    if max < 1e-10 {
        1.0
    } else {
        max / 127.0
    }
}

/// Quantize a single f32 value to INT8 given a pre-computed scale.
///
/// Clamps the result to `[-127, 127]` (symmetric range).
pub fn quantize_scalar(value: f32, scale: f32) -> i8 {
    // Routed through `core_math::round` so the call works under both
    // `std` (libstd's `f32::round`) and `no_std + libm` (`libm::roundf`).
    let q = core_math::round(value / scale);
    q.clamp(-127.0, 127.0) as i8
}

/// Dequantize a slice of INT8 values to f32, writing into `output`.
///
/// Requires no heap allocation.
///
/// # Errors
///
/// Returns [`EmbeddedError::BufferTooSmall`] when `output` is shorter than
/// `quantized`, and [`EmbeddedError::DimensionMismatch`] when it is longer.
///
/// The length agreement used to be a `debug_assert_eq!` only, so a release
/// build silently truncated to the shorter slice: handing in a 16-element
/// `quantized` with an 8-element `output` wrote 8 values and returned
/// normally, leaving the caller with a half-dequantised tensor and no signal.
pub fn dequantize_into(quantized: &[i8], scale: f32, output: &mut [f32]) -> EmbeddedResult<()> {
    if output.len() < quantized.len() {
        return Err(EmbeddedError::BufferTooSmall {
            required: quantized.len(),
            available: output.len(),
        });
    }
    if output.len() != quantized.len() {
        return Err(EmbeddedError::DimensionMismatch {
            expected: quantized.len(),
            got: output.len(),
        });
    }
    for (q, o) in quantized.iter().zip(output.iter_mut()) {
        *o = *q as f32 * scale;
    }
    Ok(())
}

/// Quantize a slice of f32 values to INT8 using symmetric per-tensor quantization.
///
/// Returns `(scale, quantized_values)`. Available when `alloc` or `std` feature
/// is enabled.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn quantize_symmetric(values: &[f32]) -> (f32, Vec<i8>) {
    let scale = compute_scale(values);
    let quantized = values
        .iter()
        .map(|&v| quantize_scalar(v, scale))
        .collect::<Vec<i8>>();
    (scale, quantized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_scale() {
        let values = [1.0_f32, -2.0, 3.0, -0.5];
        let scale = compute_scale(&values);
        let expected = 3.0_f32 / 127.0;
        assert!(
            (scale - expected).abs() < 1e-6,
            "scale = {scale}, expected {expected}"
        );
    }

    #[test]
    fn test_quantize_scalar() {
        let scale = 1.0_f32 / 127.0;
        let q = quantize_scalar(1.0, scale);
        assert_eq!(q, 127_i8, "quantize_scalar(1.0, 1/127) should give 127");
    }

    #[test]
    fn test_quantize_scalar_negative() {
        let scale = 1.0_f32 / 127.0;
        let q = quantize_scalar(-1.0, scale);
        assert_eq!(q, -127_i8, "quantize_scalar(-1.0, 1/127) should give -127");
    }

    #[test]
    fn test_dequantize_into() {
        // Round-trip: quantize then dequantize, check error <= 1 LSB
        let original = [0.5_f32, -0.5, 1.0, -1.0];
        let scale = compute_scale(&original);
        // Build quantized array on stack (no heap needed)
        let q: [i8; 4] = [
            quantize_scalar(original[0], scale),
            quantize_scalar(original[1], scale),
            quantize_scalar(original[2], scale),
            quantize_scalar(original[3], scale),
        ];
        let mut reconstructed = [0.0_f32; 4];
        dequantize_into(&q, scale, &mut reconstructed).expect("equal lengths");
        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            // Tolerance: at most 1 LSB
            assert!(
                (orig - recon).abs() <= scale + 1e-6,
                "round-trip error too large: orig={orig}, recon={recon}, tol={scale}"
            );
        }
    }

    #[test]
    fn test_dequantize_into_rejects_short_output() {
        // Release builds used to silently write a partial result here.
        let quantized = [1_i8; 16];
        let mut output = [0.0_f32; 8];
        assert_eq!(
            dequantize_into(&quantized, 0.5, &mut output),
            Err(EmbeddedError::BufferTooSmall {
                required: 16,
                available: 8
            }),
            "a short output buffer must be reported, not silently truncated"
        );
        assert!(
            output.iter().all(|&v| v == 0.0),
            "the output buffer must be left untouched on error"
        );
    }

    #[test]
    fn test_dequantize_into_rejects_long_output() {
        let quantized = [1_i8; 4];
        let mut output = [0.0_f32; 8];
        assert_eq!(
            dequantize_into(&quantized, 0.5, &mut output),
            Err(EmbeddedError::DimensionMismatch {
                expected: 4,
                got: 8
            })
        );
    }

    #[test]
    fn test_max_abs_and_scale_edge_cases() {
        assert_eq!(max_abs(&[]), 0.0, "empty slice has zero max_abs");
        assert_eq!(
            compute_scale(&[0.0; 4]),
            1.0,
            "near-zero tensors must not divide by zero"
        );
        assert!((max_abs(&[-5.0, 2.0]) - 5.0).abs() < 1e-6);
    }
}
