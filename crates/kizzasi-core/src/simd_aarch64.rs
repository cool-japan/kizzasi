//! ARM64 (AArch64) NEON-optimized operations for kizzasi-core.
//!
//! Provides a clean public API surface for NEON-accelerated tensor operations.
//! Delegates to the lower-level `simd_neon` module for architecture-specific
//! primitives and supplies additional higher-level routines.
//!
//! All NEON paths are guarded by `#[cfg(target_arch = "aarch64")]` and use
//! `#[target_feature(enable = "neon")]` for safe dispatch. Scalar fallbacks
//! are always present so this module compiles and runs correctly on x86_64,
//! WASM, and all other targets.
//!
//! # Operations
//!
//! - `dot_product_f32` — dot product with FMA accumulation
//! - `relu_f32` — element-wise ReLU
//! - `add_f32` — element-wise addition
//! - `scale_f32` — element-wise scalar multiply
//! - `l2_norm_f32` — L2 (Euclidean) norm
//! - `normalize_f32` — L2 normalization in-place
//! - `softmax_f32` — numerically stable softmax in-place
//! - `rms_norm_f32` — RMS normalization
//! - `ssm_state_update_f32` — SSM hidden-state update with broadcast scalar

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::simd_neon::{
    neon_dot_product, neon_relu, neon_rms_norm, neon_softmax, neon_ssm_update, neon_vec_add,
};

// ─── dot product ─────────────────────────────────────────────────────────────

/// Compute dot product of two f32 slices.
///
/// Uses NEON FMA on aarch64; falls back to a scalar sum on other targets.
/// `neon_dot_product` already handles a length mismatch internally (falling
/// back to a zip-based scalar sum over the common prefix), so this wrapper
/// does not need its own precondition check.
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    neon_dot_product(a, b)
}

// ─── ReLU ────────────────────────────────────────────────────────────────────

/// Element-wise ReLU: `output[i] = max(0, input[i])`.
///
/// NEON-vectorised on aarch64 (4 lanes per iteration).
///
/// `input`/`output` are truncated to their common length (matching
/// [`crate::simd::dot_product`]'s documented truncation behaviour), so a
/// mismatched length computes the well-defined prefix rather than panicking.
pub fn relu_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    // `neon_relu`'s only error condition is `x.len() != y.len()`; slicing
    // both to the same `len` makes that structurally unreachable here, so
    // discarding the (impossible) error is safe rather than a hedge.
    let _ = neon_relu(&input[..len], &mut output[..len]);
}

// ─── addition ────────────────────────────────────────────────────────────────

/// Element-wise addition: `output[i] = a[i] + b[i]`.
///
/// NEON-vectorised on aarch64 (4 lanes per iteration).
///
/// `a`/`b`/`output` are truncated to their common length (matching
/// [`crate::simd::dot_product`]'s documented truncation behaviour), so a
/// mismatched length computes the well-defined prefix rather than panicking.
pub fn add_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    let len = a.len().min(b.len()).min(output.len());
    // `neon_vec_add`'s only error condition is a length mismatch among its
    // three arguments; slicing all three to the same `len` makes that
    // structurally unreachable here, so discarding the (impossible) error is
    // safe rather than a hedge.
    let _ = neon_vec_add(&a[..len], &b[..len], &mut output[..len]);
}

// ─── scale ───────────────────────────────────────────────────────────────────

/// Element-wise scalar multiply: `output[i] = input[i] * scale`.
///
/// NEON-vectorised on aarch64.
///
/// # Panics
///
/// Panics if `input.len() != output.len()`.
pub fn scale_f32(input: &[f32], scale: f32, output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "scale_f32: length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { scale_neon(input, scale, output) };
            return;
        }
    }

    // Scalar fallback
    for (o, &i) in output.iter_mut().zip(input.iter()) {
        *o = i * scale;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_neon(input: &[f32], scale: f32, output: &mut [f32]) {
    let n = input.len();
    let vscale = vdupq_n_f32(scale);
    let chunks = n / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let v = vld1q_f32(input.as_ptr().add(idx));
        let r = vmulq_f32(v, vscale);
        vst1q_f32(output.as_mut_ptr().add(idx), r);
    }

    for i in (chunks * 4)..n {
        output[i] = input[i] * scale;
    }
}

// ─── L2 norm / normalize ─────────────────────────────────────────────────────

/// Compute the L2 (Euclidean) norm of a f32 slice.
///
/// Returns `sqrt(sum(v[i]^2))`.  Delegates to `dot_product_f32` which is
/// NEON-accelerated on aarch64.
pub fn l2_norm_f32(v: &[f32]) -> f32 {
    dot_product_f32(v, v).sqrt()
}

/// Normalize a mutable f32 slice in-place: `v /= ||v||_2`.
///
/// No-op if the norm is below `1e-12` (prevents division by near-zero).
pub fn normalize_f32(v: &mut [f32]) {
    let norm = l2_norm_f32(v);
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        // Reuse the NEON-accelerated scale path
        let tmp: Vec<f32> = v.iter().map(|&x| x * inv).collect();
        v.copy_from_slice(&tmp);
    }
}

// ─── softmax ─────────────────────────────────────────────────────────────────

/// Numerically stable softmax in-place.
///
/// Implements the online max-tracking algorithm so that large logits do not
/// overflow.  The normalization pass is NEON-vectorised on aarch64.
pub fn softmax_f32(x: &mut [f32]) {
    neon_softmax(x);
}

// ─── RMS normalization ───────────────────────────────────────────────────────

/// RMS normalization: `output[i] = x[i] / sqrt(mean(x^2) + eps)`.
///
/// NEON-vectorised on aarch64 for both the sum-of-squares and the scale pass.
///
/// `x`/`output` are truncated to their common length (matching
/// [`crate::simd::dot_product`]'s documented truncation behaviour): the RMS
/// statistic is computed over the shared prefix rather than panicking on a
/// mismatched length.
pub fn rms_norm_f32(x: &[f32], output: &mut [f32], eps: f32) {
    let len = x.len().min(output.len());
    // `neon_rms_norm`'s only error condition is `x.len() != output.len()`;
    // slicing both to the same `len` makes that structurally unreachable
    // here, so discarding the (impossible) error is safe rather than a hedge.
    let _ = neon_rms_norm(&x[..len], &mut output[..len], eps);
}

// ─── SSM state update ────────────────────────────────────────────────────────

/// SSM hidden-state update with a broadcast scalar input.
///
/// Computes `h[i] = a_bar[i] * h[i] + b_bar[i] * x_val` for all `i`.
///
/// Uses NEON FMA on aarch64 (4 lanes per iteration).
///
/// `a_bar`/`h`/`b_bar` are truncated to their common length (matching
/// [`crate::simd::dot_product`]'s documented truncation behaviour), so a
/// mismatched length updates the well-defined prefix rather than panicking.
pub fn ssm_state_update_f32(a_bar: &[f32], h: &mut [f32], b_bar: &[f32], x_val: f32) {
    let len = a_bar.len().min(h.len()).min(b_bar.len());
    // `neon_ssm_update`'s only error condition is a length mismatch among
    // `a_bar`/`h`/`b_bar`; slicing all three to the same `len` makes that
    // structurally unreachable here, so discarding the (impossible) error is
    // safe rather than a hedge.
    let _ = neon_ssm_update(&a_bar[..len], &mut h[..len], &b_bar[..len], x_val);
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── dot product ──────────────────────────────────────────────────────────

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0];
        let result = dot_product_f32(&a, &b);
        assert!((result - 10.0).abs() < 1e-5, "expected 10.0, got {result}");
    }

    #[test]
    fn test_dot_product_matches_scalar() {
        let a: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..16).map(|i| i as f32 * 0.05).collect();
        let result = dot_product_f32(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (result - expected).abs() < 1e-4,
            "result={result}, expected={expected}"
        );
    }

    #[test]
    fn test_dot_product_large_not_multiple_of_four() {
        let a: Vec<f32> = vec![1.0; 17];
        let b: Vec<f32> = vec![2.0; 17];
        let result = dot_product_f32(&a, &b);
        assert!((result - 34.0).abs() < 1e-4, "expected 34.0, got {result}");
    }

    // ── ReLU ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_relu_zeros_negatives() {
        let input = vec![-1.0f32, 0.0, 1.0, 2.0, -0.5];
        let mut output = vec![0.0f32; 5];
        relu_f32(&input, &mut output);
        assert_eq!(output, vec![0.0, 0.0, 1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_relu_all_positive() {
        let input = vec![0.1f32, 0.5, 1.0, 100.0];
        let mut output = vec![0.0f32; 4];
        relu_f32(&input, &mut output);
        assert_eq!(output, input);
    }

    // ── addition ─────────────────────────────────────────────────────────────

    #[test]
    fn test_add_f32_basic() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![0.5f32, 0.5, 0.5, 0.5];
        let mut out = vec![0.0f32; 4];
        add_f32(&a, &b, &mut out);
        assert_eq!(out, vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_add_f32_non_multiple_of_four() {
        let a = vec![1.0f32; 9];
        let b = vec![2.0f32; 9];
        let mut out = vec![0.0f32; 9];
        add_f32(&a, &b, &mut out);
        assert!(out.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    // ── scale ────────────────────────────────────────────────────────────────

    #[test]
    fn test_scale_f32_basic() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 8];
        scale_f32(&input, 2.0, &mut output);
        assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn test_scale_f32_remainder() {
        let input = vec![1.0f32; 5];
        let mut output = vec![0.0f32; 5];
        scale_f32(&input, 3.0, &mut output);
        assert!(output.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    // ── L2 norm ──────────────────────────────────────────────────────────────

    #[test]
    fn test_l2_norm_pythagorean() {
        let v = vec![3.0f32, 4.0];
        assert!(
            (l2_norm_f32(&v) - 5.0).abs() < 1e-5,
            "expected 5.0, got {}",
            l2_norm_f32(&v)
        );
    }

    #[test]
    fn test_l2_norm_unit() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((l2_norm_f32(&v) - 1.0).abs() < 1e-6);
    }

    // ── normalize ────────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_unit_vector() {
        let mut v = vec![3.0f32, 4.0];
        normalize_f32(&mut v);
        assert!(
            (l2_norm_f32(&v) - 1.0).abs() < 1e-5,
            "norm after normalize = {}",
            l2_norm_f32(&v)
        );
        assert!((v[0] - 0.6).abs() < 1e-5, "v[0] = {}", v[0]);
        assert!((v[1] - 0.8).abs() < 1e-5, "v[1] = {}", v[1]);
    }

    #[test]
    fn test_normalize_zero_vector_no_panic() {
        let mut v = vec![0.0f32; 4];
        normalize_f32(&mut v); // Must not panic or produce NaN
        assert!(v.iter().all(|&x| x == 0.0));
    }

    // ── softmax ──────────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        softmax_f32(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn test_softmax_monotone() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax_f32(&mut x);
        assert!(x[0] < x[1] && x[1] < x[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut x = vec![100.0f32, 101.0, 102.0];
        softmax_f32(&mut x);
        for &v in &x {
            assert!(v.is_finite(), "non-finite: {v}");
        }
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_single_element() {
        let mut x = vec![42.0f32];
        softmax_f32(&mut x);
        assert!((x[0] - 1.0).abs() < 1e-6);
    }

    // ── RMS norm ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rms_norm_basic() {
        let x = vec![3.0f32, 4.0];
        let mut out = vec![0.0f32; 2];
        rms_norm_f32(&x, &mut out, 0.0);
        let rms = (12.5f32).sqrt(); // sqrt((9+16)/2)
        assert!((out[0] - 3.0 / rms).abs() < 1e-5, "out[0] = {}", out[0]);
        assert!((out[1] - 4.0 / rms).abs() < 1e-5, "out[1] = {}", out[1]);
    }

    #[test]
    fn test_rms_norm_ones() {
        let x = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rms_norm_f32(&x, &mut out, 0.0);
        // rms of all-ones = 1.0 → output = input
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_rms_norm_eps_stability() {
        let x = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 4];
        // With eps > 0 and all-zero input, should not produce NaN
        rms_norm_f32(&x, &mut out, 1e-5);
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    // ── SSM state update ─────────────────────────────────────────────────────

    #[test]
    fn test_ssm_state_update_basic() {
        let a_bar = vec![0.5f32, 0.5, 0.5, 0.5];
        let mut h = vec![2.0f32, 4.0, 6.0, 8.0];
        let b_bar = vec![1.0f32, 1.0, 1.0, 1.0];
        ssm_state_update_f32(&a_bar, &mut h, &b_bar, 1.0);
        // h[i] = 0.5 * h_old[i] + 1.0 * 1.0
        let expected = [2.0f32, 3.0, 4.0, 5.0];
        for (i, &v) in h.iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-5,
                "h[{i}] = {v}, expected {}",
                expected[i]
            );
        }
    }

    #[test]
    fn test_ssm_state_update_zero_input() {
        let a_bar = vec![0.9f32; 8];
        let mut h = vec![1.0f32; 8];
        let b_bar = vec![0.1f32; 8];
        ssm_state_update_f32(&a_bar, &mut h, &b_bar, 0.0);
        // h[i] = 0.9 * 1.0 + 0.1 * 0.0 = 0.9
        assert!(h.iter().all(|&v| (v - 0.9).abs() < 1e-5));
    }

    #[test]
    fn test_ssm_state_update_non_multiple_of_four() {
        let n = 7;
        let a_bar = vec![1.0f32; n];
        let mut h = vec![1.0f32; n];
        let b_bar = vec![0.0f32; n];
        ssm_state_update_f32(&a_bar, &mut h, &b_bar, 5.0);
        // h[i] = 1*1 + 0*5 = 1
        assert!(h.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }
}
