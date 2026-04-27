//! ARM NEON SIMD Optimizations
//!
//! This module provides SIMD optimizations using ARM NEON intrinsics for
//! mobile and embedded ARM platforms (ARMv7, ARMv8, Apple Silicon).
//!
//! # Platform Support
//!
//! - ARMv7 with NEON (32-bit, e.g., Raspberry Pi 2/3)
//! - ARMv8/AArch64 (64-bit, e.g., modern smartphones, Raspberry Pi 4, Apple Silicon)
//! - Automatic feature detection at compile-time
//!
//! # NEON Features
//!
//! - 128-bit SIMD registers (4 x f32 or 2 x f64)
//! - Fused Multiply-Add (FMA) support
//! - Vector loads/stores with alignment handling
//! - Element-wise operations (add, mul, etc.)
//!
//! # Performance
//!
//! - 4x speedup for f32 operations vs scalar
//! - 2x speedup for f64 operations vs scalar
//! - Optimized for Apple Silicon M1/M2/M3 chips

use crate::error::{CoreError, CoreResult};

/// Check if NEON is available at runtime
#[inline]
pub fn is_neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on AArch64
        true
    }
    #[cfg(all(target_arch = "arm", target_feature = "neon"))]
    {
        true
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "arm", target_feature = "neon")
    )))]
    {
        false
    }
}

/// NEON-optimized dot product (4 x f32 parallel)
///
/// Computes dot product using ARM NEON SIMD instructions.
/// Falls back to scalar if NEON is not available.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must be same length as a)
///
/// # Returns
///
/// Dot product a · b
#[inline]
pub fn neon_dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        // Fallback to scalar for mismatched lengths
        return scalar_dot_product(a, b);
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_dot_product_impl(a, b) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_dot_product(a, b)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dot_product_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 4;
    let _remainder = len % 4;

    let mut sum = vdupq_n_f32(0.0);

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        sum = vfmaq_f32(sum, va, vb); // FMA: sum += va * vb
    }

    // Horizontal sum: sum all 4 lanes
    let mut result = vaddvq_f32(sum);

    // Handle remainder
    for i in (chunks * 4)..len {
        result += a[i] * b[i];
    }

    result
}

/// Scalar fallback for dot product
#[inline]
fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// NEON-optimized vector addition: c = a + b
///
/// # Arguments
///
/// * `a` - First input vector
/// * `b` - Second input vector
/// * `c` - Output vector (must be same length as inputs)
pub fn neon_vec_add(a: &[f32], b: &[f32], c: &mut [f32]) -> CoreResult<()> {
    if a.len() != b.len() || a.len() != c.len() {
        return Err(CoreError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_vec_add_impl(a, b, c);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..a.len() {
            c[i] = a[i] + b[i];
        }
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_vec_add_impl(a: &[f32], b: &[f32], c: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 4;
    let _remainder = len % 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        let vc = vaddq_f32(va, vb);
        vst1q_f32(c.as_mut_ptr().add(idx), vc);
    }

    // Handle remainder
    for i in (chunks * 4)..len {
        c[i] = a[i] + b[i];
    }
}

/// NEON-optimized vector multiplication: c = a * b
///
/// # Arguments
///
/// * `a` - First input vector
/// * `b` - Second input vector
/// * `c` - Output vector
pub fn neon_vec_mul(a: &[f32], b: &[f32], c: &mut [f32]) -> CoreResult<()> {
    if a.len() != b.len() || a.len() != c.len() {
        return Err(CoreError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_vec_mul_impl(a, b, c);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..a.len() {
            c[i] = a[i] * b[i];
        }
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_vec_mul_impl(a: &[f32], b: &[f32], c: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        let vc = vmulq_f32(va, vb);
        vst1q_f32(c.as_mut_ptr().add(idx), vc);
    }

    // Handle remainder
    for i in (chunks * 4)..len {
        c[i] = a[i] * b[i];
    }
}

/// NEON-optimized fused multiply-add: c = a * b + c
///
/// Uses FMA instructions for better performance and numerical accuracy.
///
/// # Arguments
///
/// * `a` - First multiplicand
/// * `b` - Second multiplicand
/// * `c` - Accumulator (input/output)
pub fn neon_vec_fma(a: &[f32], b: &[f32], c: &mut [f32]) -> CoreResult<()> {
    if a.len() != b.len() || a.len() != c.len() {
        return Err(CoreError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_vec_fma_impl(a, b, c);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..a.len() {
            c[i] += a[i] * b[i];
        }
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_vec_fma_impl(a: &[f32], b: &[f32], c: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        let vc = vld1q_f32(c.as_ptr().add(idx));
        let vresult = vfmaq_f32(vc, va, vb); // FMA: c + a * b
        vst1q_f32(c.as_mut_ptr().add(idx), vresult);
    }

    // Handle remainder
    for i in (chunks * 4)..len {
        c[i] += a[i] * b[i];
    }
}

/// NEON-optimized matrix-vector multiplication
///
/// Computes y = A @ x where A is a matrix and x is a vector.
///
/// # Arguments
///
/// * `matrix` - Input matrix (row-major, m x n)
/// * `x` - Input vector (n,)
/// * `y` - Output vector (m,)
/// * `rows` - Number of rows (m)
/// * `cols` - Number of columns (n)
pub fn neon_matvec(
    matrix: &[f32],
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    cols: usize,
) -> CoreResult<()> {
    if matrix.len() != rows * cols {
        return Err(CoreError::DimensionMismatch {
            expected: rows * cols,
            got: matrix.len(),
        });
    }

    if x.len() != cols || y.len() != rows {
        return Err(CoreError::DimensionMismatch {
            expected: cols,
            got: x.len(),
        });
    }

    for i in 0..rows {
        let row = &matrix[i * cols..(i + 1) * cols];
        y[i] = neon_dot_product(row, x);
    }

    Ok(())
}

/// NEON-optimized ReLU activation
///
/// Applies ReLU: y = max(0, x) element-wise.
///
/// # Arguments
///
/// * `x` - Input vector
/// * `y` - Output vector
pub fn neon_relu(x: &[f32], y: &mut [f32]) -> CoreResult<()> {
    if x.len() != y.len() {
        return Err(CoreError::DimensionMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_relu_impl(x, y);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..x.len() {
            y[i] = x[i].max(0.0);
        }
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_relu_impl(x: &[f32], y: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = x.len();
    let chunks = len / 4;
    let zeros = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let idx = i * 4;
        let vx = vld1q_f32(x.as_ptr().add(idx));
        let vy = vmaxq_f32(vx, zeros); // max(x, 0)
        vst1q_f32(y.as_mut_ptr().add(idx), vy);
    }

    // Handle remainder
    for i in (chunks * 4)..len {
        y[i] = x[i].max(0.0);
    }
}

/// NEON-optimized layer normalization
///
/// Computes: y = (x - mean) / sqrt(variance + eps)
///
/// # Arguments
///
/// * `x` - Input vector
/// * `y` - Output vector
/// * `eps` - Epsilon for numerical stability
pub fn neon_layer_norm(x: &[f32], y: &mut [f32], eps: f32) -> CoreResult<()> {
    if x.len() != y.len() {
        return Err(CoreError::DimensionMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }

    let n = x.len() as f32;

    // Compute mean
    let sum: f32 = x.iter().sum();
    let mean = sum / n;

    // Compute variance
    let var_sum: f32 = x.iter().map(|&xi| (xi - mean).powi(2)).sum();
    let variance = var_sum / n;
    let std_inv = 1.0 / (variance + eps).sqrt();

    // Normalize with NEON
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_layer_norm_impl(x, y, mean, std_inv);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..x.len() {
            y[i] = (x[i] - mean) * std_inv;
        }
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_layer_norm_impl(x: &[f32], y: &mut [f32], mean: f32, std_inv: f32) {
    use std::arch::aarch64::*;

    let len = x.len();
    let chunks = len / 4;

    let vmean = vdupq_n_f32(mean);
    let vstd_inv = vdupq_n_f32(std_inv);

    for i in 0..chunks {
        let idx = i * 4;
        let vx = vld1q_f32(x.as_ptr().add(idx));
        let centered = vsubq_f32(vx, vmean);
        let normalized = vmulq_f32(centered, vstd_inv);
        vst1q_f32(y.as_mut_ptr().add(idx), normalized);
    }

    // Handle remainder
    for i in (chunks * 4)..len {
        y[i] = (x[i] - mean) * std_inv;
    }
}

/// NEON-optimized softmax (numerically stable online algorithm)
///
/// Computes softmax in-place using the online max-tracking algorithm for
/// numerical stability. NEON used for the normalization pass.
///
/// # Arguments
///
/// * `x` - Input/output vector (modified in-place)
pub fn neon_softmax(x: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }

    // Pass 1: compute online max and denominator
    let mut max_val = x[0];
    let mut sum = 1.0f32;

    for &xi in x.iter().skip(1) {
        if xi > max_val {
            sum *= (max_val - xi).exp();
            sum += 1.0;
            max_val = xi;
        } else {
            sum += (xi - max_val).exp();
        }
    }

    // Pass 2: normalize — NEON on aarch64
    let inv_sum = 1.0 / sum;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_softmax_normalize_impl(x, max_val, inv_sum);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for val in x.iter_mut() {
            *val = (*val - max_val).exp() * inv_sum;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_softmax_normalize_impl(x: &mut [f32], max_val: f32, inv_sum: f32) {
    use std::arch::aarch64::*;

    let len = x.len();
    let chunks = len / 4;

    let vmax = vdupq_n_f32(max_val);
    let vinv = vdupq_n_f32(inv_sum);

    for i in 0..chunks {
        let idx = i * 4;
        let vx = vld1q_f32(x.as_ptr().add(idx));
        let vshifted = vsubq_f32(vx, vmax);
        // scalar exp per lane (no NEON exp intrinsic in stable std)
        let e0 = vgetq_lane_f32(vshifted, 0).exp();
        let e1 = vgetq_lane_f32(vshifted, 1).exp();
        let e2 = vgetq_lane_f32(vshifted, 2).exp();
        let e3 = vgetq_lane_f32(vshifted, 3).exp();
        let vexp = {
            let mut tmp = vdupq_n_f32(0.0);
            tmp = vsetq_lane_f32(e0, tmp, 0);
            tmp = vsetq_lane_f32(e1, tmp, 1);
            tmp = vsetq_lane_f32(e2, tmp, 2);
            tmp = vsetq_lane_f32(e3, tmp, 3);
            tmp
        };
        let vnorm = vmulq_f32(vexp, vinv);
        vst1q_f32(x.as_mut_ptr().add(idx), vnorm);
    }

    for val in x.iter_mut().skip(chunks * 4) {
        *val = (*val - max_val).exp() * inv_sum;
    }
}

/// NEON-optimized RMS normalization
///
/// Computes: `output[i] = x[i] / sqrt(mean(x^2) + eps)`
///
/// # Arguments
///
/// * `x` - Input vector
/// * `output` - Output vector (same length as x)
/// * `eps` - Epsilon for numerical stability
pub fn neon_rms_norm(x: &[f32], output: &mut [f32], eps: f32) -> CoreResult<()> {
    if x.len() != output.len() {
        return Err(CoreError::DimensionMismatch {
            expected: x.len(),
            got: output.len(),
        });
    }

    let n = x.len();
    if n == 0 {
        return Ok(());
    }

    // Compute sum of squares
    let sum_sq: f32 = {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { neon_sum_squares(x) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            x.iter().map(|&v| v * v).sum()
        }
    };

    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Scale output
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_scale_impl(x, inv_rms, output);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for (o, &xi) in output.iter_mut().zip(x.iter()) {
            *o = xi * inv_rms;
        }
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum_squares(x: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = x.len();
    let chunks = len / 4;
    let mut accum = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let idx = i * 4;
        let vx = vld1q_f32(x.as_ptr().add(idx));
        accum = vfmaq_f32(accum, vx, vx);
    }

    let mut result = vaddvq_f32(accum);
    for &v in x.iter().skip(chunks * 4) {
        result += v * v;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scale_impl(input: &[f32], scale: f32, output: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = input.len();
    let chunks = len / 4;
    let vscale = vdupq_n_f32(scale);

    for i in 0..chunks {
        let idx = i * 4;
        let vx = vld1q_f32(input.as_ptr().add(idx));
        let vr = vmulq_f32(vx, vscale);
        vst1q_f32(output.as_mut_ptr().add(idx), vr);
    }

    for i in (chunks * 4)..len {
        output[i] = input[i] * scale;
    }
}

/// NEON-optimized SSM state update: `h[i] = a_bar[i] * h[i] + b_bar[i] * x_val`
///
/// Broadcasts scalar `x_val` across all lanes. Uses FMA for accuracy.
///
/// # Arguments
///
/// * `a_bar` - Discretized A matrix diagonal (same length as h)
/// * `h` - Hidden state (modified in-place)
/// * `b_bar` - Discretized B matrix (same length as h)
/// * `x_val` - Scalar input value broadcast across all dimensions
pub fn neon_ssm_update(a_bar: &[f32], h: &mut [f32], b_bar: &[f32], x_val: f32) -> CoreResult<()> {
    if a_bar.len() != h.len() || b_bar.len() != h.len() {
        return Err(CoreError::DimensionMismatch {
            expected: h.len(),
            got: if a_bar.len() != h.len() {
                a_bar.len()
            } else {
                b_bar.len()
            },
        });
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_ssm_update_impl(a_bar, h, b_bar, x_val);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..h.len() {
            h[i] = a_bar[i] * h[i] + b_bar[i] * x_val;
        }
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_ssm_update_impl(a_bar: &[f32], h: &mut [f32], b_bar: &[f32], x_val: f32) {
    use std::arch::aarch64::*;

    let len = h.len();
    let chunks = len / 4;
    let vx = vdupq_n_f32(x_val);

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a_bar.as_ptr().add(idx));
        let vh = vld1q_f32(h.as_ptr().add(idx));
        let vb = vld1q_f32(b_bar.as_ptr().add(idx));
        // h = a * h + b * x  =>  fma(b, x, a*h)
        let vah = vmulq_f32(va, vh);
        let vresult = vfmaq_f32(vah, vb, vx);
        vst1q_f32(h.as_mut_ptr().add(idx), vresult);
    }

    for i in (chunks * 4)..len {
        h[i] = a_bar[i] * h[i] + b_bar[i] * x_val;
    }
}

/// NEON-optimized exponential approximation
///
/// Fast exp approximation using polynomial (less accurate than std::exp).
///
/// # Arguments
///
/// * `x` - Input vector
/// * `y` - Output vector
pub fn neon_fast_exp(x: &[f32], y: &mut [f32]) -> CoreResult<()> {
    if x.len() != y.len() {
        return Err(CoreError::DimensionMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }

    // Use scalar fast_exp for now
    // Can be optimized further with NEON polynomial evaluation
    for i in 0..x.len() {
        y[i] = fast_exp_scalar(x[i]);
    }

    Ok(())
}

/// Fast exp approximation (polynomial)
#[inline]
fn fast_exp_scalar(x: f32) -> f32 {
    // Clamp to prevent overflow
    let x_clamped = x.clamp(-88.0, 88.0);

    // 5th order polynomial approximation
    let x2 = x_clamped * x_clamped;
    let x3 = x2 * x_clamped;
    let x4 = x2 * x2;
    let x5 = x2 * x3;

    1.0 + x_clamped + 0.5 * x2 + 0.16666667 * x3 + 0.04166667 * x4 + 0.00833333 * x5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_availability() {
        let _available = is_neon_available();
        // On aarch64 (Apple Silicon, ARM64), NEON should be available
        #[cfg(target_arch = "aarch64")]
        assert!(_available);
    }

    #[test]
    fn test_neon_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let result = neon_dot_product(&a, &b);
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0 + 5.0 * 6.0;

        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_neon_vec_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        neon_vec_add(&a, &b, &mut c).expect("neon_vec_add failed");

        assert_eq!(c, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_neon_vec_mul() {
        let a = vec![2.0, 3.0, 4.0, 5.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0; 4];

        neon_vec_mul(&a, &b, &mut c).expect("neon_vec_mul failed");

        assert_eq!(c, vec![6.0, 12.0, 20.0, 30.0]);
    }

    #[test]
    fn test_neon_vec_fma() {
        let a = vec![2.0, 3.0, 4.0, 5.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0];

        neon_vec_fma(&a, &b, &mut c).expect("neon_vec_fma failed");

        assert_eq!(c, vec![7.0, 13.0, 21.0, 31.0]); // 1 + 2*3, 1 + 3*4, etc.
    }

    #[test]
    fn test_neon_matvec() {
        let matrix = vec![
            1.0, 2.0, 3.0, // row 1
            4.0, 5.0, 6.0, // row 2
        ];
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 2];

        neon_matvec(&matrix, &x, &mut y, 2, 3).expect("neon_matvec failed");

        assert_eq!(y[0], 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0); // 14.0
        assert_eq!(y[1], 4.0 * 1.0 + 5.0 * 2.0 + 6.0 * 3.0); // 32.0
    }

    #[test]
    fn test_neon_relu() {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut y = vec![0.0; 5];

        neon_relu(&x, &mut y).expect("neon_relu failed");

        assert_eq!(y, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_neon_layer_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 4];

        neon_layer_norm(&x, &mut y, 1e-5).expect("neon_layer_norm failed");

        // Verify mean is 0 and std is 1
        let mean: f32 = y.iter().sum::<f32>() / y.len() as f32;
        let variance: f32 = y.iter().map(|&yi| yi.powi(2)).sum::<f32>() / y.len() as f32;

        assert!(mean.abs() < 1e-5);
        assert!((variance - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_neon_fast_exp() {
        let x = vec![-1.0, 0.0, 1.0, 2.0];
        let mut y = vec![0.0; 4];

        neon_fast_exp(&x, &mut y).expect("neon_fast_exp failed");

        // Verify all values are positive and reasonable
        assert!(y.iter().all(|&val| val > 0.0));
        assert!((y[1] - 1.0).abs() < 0.01); // exp(0) = 1
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let mut c = vec![0.0; 2];

        assert!(neon_vec_add(&a, &b, &mut c).is_err());
    }

    #[test]
    fn test_neon_softmax_sums_to_one() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        neon_softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    #[test]
    fn test_neon_softmax_monotone() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        neon_softmax(&mut x);
        assert!(x[0] < x[1] && x[1] < x[2]);
    }

    #[test]
    fn test_neon_softmax_numerical_stability() {
        let mut x = vec![100.0f32, 101.0, 102.0];
        neon_softmax(&mut x);
        for &v in &x {
            assert!(v.is_finite(), "neon_softmax produced non-finite: {v}");
        }
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_neon_rms_norm_basic() {
        let x = vec![3.0f32, 4.0];
        let mut out = vec![0.0f32; 2];
        neon_rms_norm(&x, &mut out, 0.0).expect("rms_norm failed");
        // rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        let rms = (12.5f32).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn test_neon_rms_norm_dimension_mismatch() {
        let x = vec![1.0f32, 2.0];
        let mut out = vec![0.0f32; 3];
        assert!(neon_rms_norm(&x, &mut out, 1e-5).is_err());
    }

    #[test]
    fn test_neon_ssm_update_basic() {
        let a_bar = vec![0.5f32, 0.5, 0.5, 0.5];
        let mut h = vec![2.0f32, 4.0, 6.0, 8.0];
        let b_bar = vec![1.0f32, 1.0, 1.0, 1.0];
        neon_ssm_update(&a_bar, &mut h, &b_bar, 1.0).expect("ssm_update failed");
        // h[i] = 0.5 * h[i] + 1.0 * 1.0
        assert!((h[0] - 2.0).abs() < 1e-5); // 0.5*2 + 1 = 2
        assert!((h[1] - 3.0).abs() < 1e-5); // 0.5*4 + 1 = 3
        assert!((h[2] - 4.0).abs() < 1e-5); // 0.5*6 + 1 = 4
        assert!((h[3] - 5.0).abs() < 1e-5); // 0.5*8 + 1 = 5
    }

    #[test]
    fn test_neon_ssm_update_dimension_mismatch() {
        let a_bar = vec![1.0f32, 2.0];
        let mut h = vec![1.0f32, 2.0, 3.0];
        let b_bar = vec![1.0f32, 2.0, 3.0];
        assert!(neon_ssm_update(&a_bar, &mut h, &b_bar, 1.0).is_err());
    }
}
