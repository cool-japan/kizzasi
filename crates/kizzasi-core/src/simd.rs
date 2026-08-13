//! SIMD-optimized operations for high-performance matrix computations
//!
//! This module provides optimized implementations for:
//! - Vector dot products
//! - Matrix-vector multiplication
//! - Element-wise operations
//! - State space model recurrence steps
//!
//! Uses explicit SIMD when available, with fallback to optimized scalar code.
//!
//! # Dispatch
//!
//! The entry points here are thin dispatchers. On x86-64 they route to
//! [`crate::simd_avx512`], which performs runtime `avx512f`/`avx2` detection;
//! on aarch64 they route to [`crate::simd_aarch64`]/[`crate::simd_neon`],
//! which use NEON intrinsics. Every other target uses the unrolled scalar
//! kernels in this module, which are also exported directly (`*_scalar`) so
//! cross-backend equality can be asserted in tests.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Process 8 elements at a time for SIMD operations
const SIMD_WIDTH: usize = 8;

/// Dot product, dispatched to the widest vector kernel available on this target.
///
/// The inputs are truncated to their common length, so a caller that passes
/// mismatched slices gets the well-defined prefix product instead of a panic
/// or (as previously) a silent zero.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len().min(b.len());
    dot_product_dispatch(&a[..len], &b[..len])
}

/// x86-64: runtime AVX-512 / AVX2 dispatch with a scalar fallback.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_product_dispatch(a: &[f32], b: &[f32]) -> f32 {
    crate::simd_avx512::dot_product_avx512(a, b)
}

/// aarch64: NEON FMA kernel.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_product_dispatch(a: &[f32], b: &[f32]) -> f32 {
    crate::simd_aarch64::dot_product_f32(a, b)
}

/// Portable fallback for targets without a vector kernel.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dot_product_dispatch(a: &[f32], b: &[f32]) -> f32 {
    dot_product_scalar(a, b)
}

/// Portable dot product with 4-way accumulator unrolling.
///
/// This is the reference implementation the vector backends must agree with.
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len().min(b.len());
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    // Process 8 elements at a time with 4-way accumulation to reduce dependencies
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let mut i = 0;
    for _ in 0..chunks {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        sum0 += a[i + 4] * b[i + 4];
        sum1 += a[i + 5] * b[i + 5];
        sum2 += a[i + 6] * b[i + 6];
        sum3 += a[i + 7] * b[i + 7];
        i += SIMD_WIDTH;
    }

    // Process remainder
    for j in 0..remainder {
        sum0 += a[i + j] * b[i + j];
    }

    sum0 + sum1 + sum2 + sum3
}

/// Optimized dot product for ndarray views
///
/// Contiguous views go straight to the vector kernel. Strided views cannot be
/// viewed as a slice, so they are folded element-wise — the historical
/// `as_slice().unwrap_or_default()` handed the kernel an *empty* slice and
/// silently returned `0.0` for any non-contiguous input.
#[inline]
pub fn dot_view(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    match (a.as_slice(), b.as_slice()) {
        (Some(a_slice), Some(b_slice)) => dot_product(a_slice, b_slice),
        _ => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
    }
}

/// Optimized matrix-vector multiplication: y = M * x
///
/// `y` is written only up to `min(m.nrows(), y.len())` rows; if `y` is
/// shorter than `m` has rows, the extra rows are simply not computed rather
/// than indexing out of bounds. Column-length mismatches between `m` and `x`
/// are handled by [`dot_view`]'s own well-defined prefix truncation.
#[inline]
pub fn matvec(m: &Array2<f32>, x: &Array1<f32>, y: &mut Array1<f32>) {
    let rows = m.nrows().min(y.len());
    let cols = m.ncols();
    debug_assert_eq!(cols, x.len());
    debug_assert_eq!(m.nrows(), y.len());
    // SIMD inner kernel: emit a TRACE-level span so tracing-subscriber can
    // identify it when trace logging is opted into, without adding overhead
    // under the default DEBUG/INFO filter. In a release build there is no
    // cost at all: the workspace enables `tracing`'s `release_max_level_info`
    // feature (root Cargo.toml), which compiles `trace_span!` away entirely.
    // In debug builds it stays a small, subscriber-filtered runtime cost.
    let _span = tracing::trace_span!("simd::matvec", rows = rows, cols = cols).entered();

    // Process 4 rows at a time if possible
    let row_chunks = rows / 4;
    let row_remainder = rows % 4;

    for chunk in 0..row_chunks {
        let base = chunk * 4;
        let r0 = m.row(base);
        let r1 = m.row(base + 1);
        let r2 = m.row(base + 2);
        let r3 = m.row(base + 3);

        y[base] = dot_view(r0, x.view());
        y[base + 1] = dot_view(r1, x.view());
        y[base + 2] = dot_view(r2, x.view());
        y[base + 3] = dot_view(r3, x.view());
    }

    // Handle remaining rows
    for i in 0..row_remainder {
        let row_idx = row_chunks * 4 + i;
        let row = m.row(row_idx);
        y[row_idx] = dot_view(row, x.view());
    }
}

/// Element-wise addition `c = a + b`, dispatched to the target's vector kernel.
#[inline]
pub fn vec_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    let len = a.len().min(b.len()).min(c.len());
    vec_add_dispatch(&a[..len], &b[..len], &mut c[..len]);
}

/// x86-64: AVX-512 element-wise add with a scalar fallback.
#[cfg(target_arch = "x86_64")]
#[inline]
fn vec_add_dispatch(a: &[f32], b: &[f32], c: &mut [f32]) {
    crate::simd_avx512::elementwise::add_avx512(a, b, c);
}

/// aarch64: NEON element-wise add.
#[cfg(target_arch = "aarch64")]
#[inline]
fn vec_add_dispatch(a: &[f32], b: &[f32], c: &mut [f32]) {
    crate::simd_aarch64::add_f32(a, b, c);
}

/// Portable fallback for targets without a vector kernel.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn vec_add_dispatch(a: &[f32], b: &[f32], c: &mut [f32]) {
    vec_add_scalar(a, b, c);
}

/// Portable element-wise addition with loop unrolling.
///
/// Reference implementation the vector backends must agree with.
#[inline]
pub fn vec_add_scalar(a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    let len = a.len().min(b.len()).min(c.len());
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    let mut i = 0;
    for _ in 0..chunks {
        c[i] = a[i] + b[i];
        c[i + 1] = a[i + 1] + b[i + 1];
        c[i + 2] = a[i + 2] + b[i + 2];
        c[i + 3] = a[i + 3] + b[i + 3];
        c[i + 4] = a[i + 4] + b[i + 4];
        c[i + 5] = a[i + 5] + b[i + 5];
        c[i + 6] = a[i + 6] + b[i + 6];
        c[i + 7] = a[i + 7] + b[i + 7];
        i += SIMD_WIDTH;
    }

    for j in 0..remainder {
        c[i + j] = a[i + j] + b[i + j];
    }
}

/// Optimized element-wise multiply: c = a * b
///
/// Inputs are truncated to their common length (matching [`dot_product`]'s
/// documented behaviour), so mismatched slice lengths yield the well-defined
/// prefix product instead of an out-of-bounds index panic.
#[inline]
pub fn vec_mul(a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    let len = a.len().min(b.len()).min(c.len());
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    let mut i = 0;
    for _ in 0..chunks {
        c[i] = a[i] * b[i];
        c[i + 1] = a[i + 1] * b[i + 1];
        c[i + 2] = a[i + 2] * b[i + 2];
        c[i + 3] = a[i + 3] * b[i + 3];
        c[i + 4] = a[i + 4] * b[i + 4];
        c[i + 5] = a[i + 5] * b[i + 5];
        c[i + 6] = a[i + 6] * b[i + 6];
        c[i + 7] = a[i + 7] * b[i + 7];
        i += SIMD_WIDTH;
    }

    for j in 0..remainder {
        c[i + j] = a[i + j] * b[i + j];
    }
}

/// Optimized element-wise fused multiply-add: c = a * b + c
///
/// Inputs are truncated to their common length (matching [`dot_product`]'s
/// documented behaviour), so mismatched slice lengths yield the well-defined
/// prefix result instead of an out-of-bounds index panic.
#[inline]
pub fn vec_fma(a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    let len = a.len().min(b.len()).min(c.len());
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    let mut i = 0;
    for _ in 0..chunks {
        c[i] = a[i].mul_add(b[i], c[i]);
        c[i + 1] = a[i + 1].mul_add(b[i + 1], c[i + 1]);
        c[i + 2] = a[i + 2].mul_add(b[i + 2], c[i + 2]);
        c[i + 3] = a[i + 3].mul_add(b[i + 3], c[i + 3]);
        c[i + 4] = a[i + 4].mul_add(b[i + 4], c[i + 4]);
        c[i + 5] = a[i + 5].mul_add(b[i + 5], c[i + 5]);
        c[i + 6] = a[i + 6].mul_add(b[i + 6], c[i + 6]);
        c[i + 7] = a[i + 7].mul_add(b[i + 7], c[i + 7]);
        i += SIMD_WIDTH;
    }

    for j in 0..remainder {
        c[i + j] = a[i + j].mul_add(b[i + j], c[i + j]);
    }
}

/// Scalar multiply `c = alpha * a`, dispatched to the target's vector kernel.
#[inline]
pub fn vec_scale(a: &[f32], alpha: f32, c: &mut [f32]) {
    debug_assert_eq!(a.len(), c.len());
    let len = a.len().min(c.len());
    vec_scale_dispatch(&a[..len], alpha, &mut c[..len]);
}

/// aarch64: NEON scalar multiply.
#[cfg(target_arch = "aarch64")]
#[inline]
fn vec_scale_dispatch(a: &[f32], alpha: f32, c: &mut [f32]) {
    crate::simd_aarch64::scale_f32(a, alpha, c);
}

/// Portable fallback (x86-64 has no dedicated scale kernel in `simd_avx512`).
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn vec_scale_dispatch(a: &[f32], alpha: f32, c: &mut [f32]) {
    vec_scale_scalar(a, alpha, c);
}

/// Portable scalar multiply with loop unrolling.
///
/// Reference implementation the vector backends must agree with.
#[inline]
pub fn vec_scale_scalar(a: &[f32], alpha: f32, c: &mut [f32]) {
    debug_assert_eq!(a.len(), c.len());
    let len = a.len().min(c.len());
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    let mut i = 0;
    for _ in 0..chunks {
        c[i] = alpha * a[i];
        c[i + 1] = alpha * a[i + 1];
        c[i + 2] = alpha * a[i + 2];
        c[i + 3] = alpha * a[i + 3];
        c[i + 4] = alpha * a[i + 4];
        c[i + 5] = alpha * a[i + 5];
        c[i + 6] = alpha * a[i + 6];
        c[i + 7] = alpha * a[i + 7];
        i += SIMD_WIDTH;
    }

    for j in 0..remainder {
        c[i + j] = alpha * a[i + j];
    }
}

/// Optimized exp for arrays
///
/// Inputs are truncated to their common length (matching [`dot_product`]'s
/// documented behaviour), so a length mismatch yields the well-defined
/// prefix result instead of an out-of-bounds index panic.
#[inline]
pub fn vec_exp(a: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), c.len());
    let len = a.len().min(c.len());
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    let mut i = 0;
    for _ in 0..chunks {
        c[i] = a[i].exp();
        c[i + 1] = a[i + 1].exp();
        c[i + 2] = a[i + 2].exp();
        c[i + 3] = a[i + 3].exp();
        c[i + 4] = a[i + 4].exp();
        c[i + 5] = a[i + 5].exp();
        c[i + 6] = a[i + 6].exp();
        c[i + 7] = a[i + 7].exp();
        i += SIMD_WIDTH;
    }

    for j in 0..remainder {
        c[i + j] = a[i + j].exp();
    }
}

/// Fast approximate exp for performance-critical paths
/// Uses a polynomial approximation valid for x in [-10, 10]
#[inline]
pub fn fast_exp(x: f32) -> f32 {
    use std::f32::consts::{LN_2, LOG2_E};

    // Clamp to valid range
    let x = x.clamp(-10.0, 10.0);

    // Convert to 2^(x * log2(e))
    let t = x * LOG2_E;

    // Split into integer and fractional parts
    let i = t.floor();
    let f = t - i;

    // Polynomial approximation for 2^f where f is in [0, 1]
    // 2^f ≈ 1 + f*(ln2 + f*(ln2^2/2 + f*(ln2^3/6 + f*ln2^4/24)))
    let ln2_sq = LN_2 * LN_2;
    let ln2_cb = ln2_sq * LN_2;
    let ln2_4 = ln2_cb * LN_2;
    let p = 1.0 + f * (LN_2 + f * (ln2_sq / 2.0 + f * (ln2_cb / 6.0 + f * ln2_4 / 24.0)));

    // Combine: 2^i * 2^f
    // Use bit manipulation for 2^i
    let bits = ((127 + i as i32) as u32) << 23;
    let scale = f32::from_bits(bits);

    scale * p
}

/// Optimized SSM state update: h' = A_bar ⊙ h + B_bar ⊙ x
///
/// Uses fused operations to reduce memory bandwidth requirements.
///
/// `x` broadcasts cyclically against `h`/`a_bar`/`b_bar`: pass a
/// single-element slice to broadcast one scalar input across every state
/// dimension (the common case of a scalar per-timestep SSM input), or a
/// slice the same length as `h` for a plain element-wise (non-broadcast)
/// update. `a_bar`/`b_bar` are expected to match `h`'s length exactly (that
/// invariant holds structurally at every real call site, since they are
/// discretized per-state-dimension companions of `h`) and are truncated to
/// `h`'s length defensively, matching this module's other kernels.
///
/// A previous version of this function asserted `b_bar.len() == x.len()`
/// while its body unconditionally indexed `x[i % x.len()]` -- a broadcast
/// idiom that contradicted the assert, and which panicked outright (integer
/// division by zero, even in release builds) if `x` was empty. `x` must now
/// simply be non-empty; an empty `x` leaves `h` unchanged instead of
/// panicking.
#[inline]
pub fn ssm_state_update(a_bar: &[f32], h: &mut [f32], b_bar: &[f32], x: &[f32]) {
    debug_assert_eq!(a_bar.len(), h.len());
    debug_assert_eq!(b_bar.len(), h.len());
    // NOT a debug_assert!(!x.is_empty()): unlike the length checks above
    // (which only affect *which* prefix gets computed if violated), an
    // empty `x` must never panic -- not even in debug builds -- since the
    // whole point of the `xlen == 0` branch below is to make that case a
    // documented, graceful no-op instead of the integer-division-by-zero
    // panic this function used to have.

    let len = h.len().min(a_bar.len()).min(b_bar.len());
    let xlen = x.len();
    if xlen == 0 {
        // Nothing to broadcast; leave `h` untouched instead of panicking on
        // `% 0` for a caller that skipped the (debug-only) precondition.
        return;
    }

    // TRACE-level span: this kernel runs once per recurrence step in the
    // inference loop and would generate huge volumes at DEBUG. TRACE keeps
    // it out of the way under the default DEBUG/INFO subscriber filter, and
    // the workspace's `tracing/release_max_level_info` feature (root
    // Cargo.toml) removes it from release builds at compile time. Only debug
    // builds pay the small subscriber-filtered cost.
    let _span = tracing::trace_span!("simd::ssm_state_update", len = len).entered();

    if xlen == len {
        // Fast, no-broadcast path: no modulo needed at all.
        for i in 0..len {
            h[i] = a_bar[i].mul_add(h[i], b_bar[i] * x[i]);
        }
        return;
    }

    for i in 0..len {
        h[i] = a_bar[i].mul_add(h[i], b_bar[i] * x[i % xlen]);
    }
}

/// Optimized layer normalization
///
/// Computes (x - mean) / sqrt(var + eps) in a single pass using Welford's algorithm
/// for numerical stability.
#[inline]
pub fn layer_norm(x: &mut [f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    // TRACE-level: called once per token/layer in the inference loop, so we
    // emit at the cheapest tracing level. The eps value is included since
    // it's a single f32 and useful for diagnosing numerical issues.
    let _span = tracing::trace_span!("simd::layer_norm", len = n, eps = eps).entered();

    // Compute mean and variance in single pass (Welford's algorithm)
    let mut mean = 0.0f32;
    let mut m2 = 0.0f32;

    for (i, &val) in x.iter().enumerate() {
        let delta = val - mean;
        mean += delta / (i + 1) as f32;
        let delta2 = val - mean;
        m2 += delta * delta2;
    }

    let variance = m2 / n as f32;
    let inv_std = 1.0 / (variance + eps).sqrt();

    // Normalize in place with loop unrolling
    let chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    let mut i = 0;
    for _ in 0..chunks {
        x[i] = (x[i] - mean) * inv_std;
        x[i + 1] = (x[i + 1] - mean) * inv_std;
        x[i + 2] = (x[i + 2] - mean) * inv_std;
        x[i + 3] = (x[i + 3] - mean) * inv_std;
        x[i + 4] = (x[i + 4] - mean) * inv_std;
        x[i + 5] = (x[i + 5] - mean) * inv_std;
        x[i + 6] = (x[i + 6] - mean) * inv_std;
        x[i + 7] = (x[i + 7] - mean) * inv_std;
        i += SIMD_WIDTH;
    }

    for j in 0..remainder {
        x[i + j] = (x[i + j] - mean) * inv_std;
    }
}

/// Optimized softmax computation (two-pass algorithm)
#[inline]
pub fn softmax(x: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    // TRACE-level: invoked per attention head per query position. Avoid
    // anything more verbose than the input length.
    let _span = tracing::trace_span!("simd::softmax", len = n).entered();

    // Find max for numerical stability
    let mut max_val = x[0];
    for &val in x.iter().skip(1) {
        if val > max_val {
            max_val = val;
        }
    }

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for val in x.iter_mut() {
        *val *= inv_sum;
    }
}

/// Online softmax computation using numerically stable single-pass algorithm
///
/// This is more efficient for streaming scenarios and maintains numerical stability
/// using an online max-finding and sum-accumulation technique.
///
/// # Algorithm
/// Uses the online softmax algorithm that maintains running max and sum:
/// - m_i = max(m_{i-1}, x_i)
/// - d_i = d_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)
///
/// This avoids the two-pass requirement of standard softmax while maintaining
/// numerical stability.
#[inline]
pub fn online_softmax(x: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    // TRACE-level: streaming softmax kernel, called often in attention paths.
    let _span = tracing::trace_span!("simd::online_softmax", len = n).entered();

    // First pass: compute running max and denominator
    let mut max_val = x[0];
    let mut sum = 1.0f32; // exp(x[0] - x[0]) = 1.0

    for &x_i in x.iter().skip(1) {
        if x_i > max_val {
            // Update sum for new max
            sum *= (max_val - x_i).exp();
            sum += 1.0; // exp(x_i - x_i)
            max_val = x_i;
        } else {
            sum += (x_i - max_val).exp();
        }
    }

    // Second pass: normalize
    let inv_sum = 1.0 / sum;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp() * inv_sum;
    }
}

/// Fused online softmax with attention weights
///
/// Computes softmax and immediately multiplies with values in a single pass,
/// reducing memory traffic. Useful for attention mechanisms.
///
/// # Arguments
/// * `scores` - Attention scores to softmax (modified in-place)
/// * `values` - Values to weight by softmax scores
/// * `output` - Weighted output
pub fn fused_softmax_attend(scores: &mut [f32], values: &[f32], output: &mut [f32]) {
    let n = scores.len();
    if n == 0 || values.len() != n || output.len() != n {
        return;
    }
    // TRACE-level: fused softmax+attend kernel, called per query position in
    // attention. Records the score vector length only (the values themselves
    // are too large to format cheaply).
    let _span = tracing::trace_span!("simd::fused_softmax_attend", len = n).entered();

    // Online softmax computation
    let mut max_val = scores[0];
    let mut sum = 1.0f32;

    for &s_i in scores.iter().skip(1) {
        if s_i > max_val {
            sum *= (max_val - s_i).exp();
            sum += 1.0;
            max_val = s_i;
        } else {
            sum += (s_i - max_val).exp();
        }
    }

    // Fused normalize and attend
    let inv_sum = 1.0 / sum;
    output.fill(0.0);

    for i in 0..n {
        let weight = (scores[i] - max_val).exp() * inv_sum;
        scores[i] = weight; // Store normalized weights
        output[i] = weight * values[i]; // Apply to values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = dot_product(&a, &b);
        assert!((result - 55.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product_matches_scalar_reference() {
        // The dispatched kernel must agree with the portable reference for
        // every length class, including partial vector tails.
        for len in [0usize, 1, 3, 7, 8, 15, 16, 17, 63, 64, 65, 129, 257] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - 3.0).collect();
            let b: Vec<f32> = (0..len).map(|i| 1.0 - (i as f32) * 0.25).collect();

            let dispatched = dot_product(&a, &b);
            let reference = dot_product_scalar(&a, &b);
            let tol = 1e-3 * reference.abs().max(1.0);
            assert!(
                (dispatched - reference).abs() <= tol,
                "len {len}: dispatched {dispatched} != reference {reference}"
            );
        }
    }

    #[test]
    fn test_vec_add_matches_scalar_reference() {
        for len in [0usize, 1, 5, 8, 17, 64, 65] {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.3).collect();
            let b: Vec<f32> = (0..len).map(|i| 10.0 - i as f32).collect();

            let mut dispatched = vec![0.0f32; len];
            let mut reference = vec![0.0f32; len];
            vec_add(&a, &b, &mut dispatched);
            vec_add_scalar(&a, &b, &mut reference);

            for i in 0..len {
                assert!(
                    (dispatched[i] - reference[i]).abs() < 1e-5,
                    "len {len} idx {i}: {} != {}",
                    dispatched[i],
                    reference[i]
                );
            }
        }
    }

    #[test]
    fn test_vec_scale_matches_scalar_reference() {
        for len in [0usize, 1, 5, 8, 17, 64, 65] {
            let a: Vec<f32> = (0..len).map(|i| i as f32 - 4.0).collect();

            let mut dispatched = vec![0.0f32; len];
            let mut reference = vec![0.0f32; len];
            vec_scale(&a, 1.75, &mut dispatched);
            vec_scale_scalar(&a, 1.75, &mut reference);

            for i in 0..len {
                assert!(
                    (dispatched[i] - reference[i]).abs() < 1e-5,
                    "len {len} idx {i}: {} != {}",
                    dispatched[i],
                    reference[i]
                );
            }
        }
    }

    #[test]
    fn test_dot_view_handles_non_contiguous_views() {
        // Regression: a strided view used to be turned into an empty slice by
        // `unwrap_or_default()`, silently yielding 0.0.
        let a = Array1::from_vec((0..16).map(|i| i as f32).collect::<Vec<_>>());
        let b = Array1::from_vec((0..16).map(|i| 2.0 * i as f32).collect::<Vec<_>>());

        let a_strided = a.slice(scirs2_core::ndarray::s![..;2]);
        let b_strided = b.slice(scirs2_core::ndarray::s![..;2]);
        assert!(
            a_strided.as_slice().is_none(),
            "test precondition: the view must be non-contiguous"
        );

        let expected: f32 = a_strided
            .iter()
            .zip(b_strided.iter())
            .map(|(x, y)| x * y)
            .sum();
        let got = dot_view(a_strided, b_strided);

        assert!(expected > 0.0, "test precondition: non-zero expected value");
        assert!(
            (got - expected).abs() < 1e-4,
            "strided dot_view returned {got}, expected {expected}"
        );
    }

    #[test]
    fn test_matvec_handles_reversed_axes_matrix() {
        // Regression companion to `test_dot_view_handles_non_contiguous_views`:
        // `matvec` calls `dot_view` on each `m.row(i)`. For an owned matrix
        // produced by `reversed_axes()` (a transpose that swaps shape and
        // strides without copying data), every row view is non-contiguous.
        // Before `dot_view` was fixed, this silently produced an all-zeros
        // output vector instead of erroring or computing the real result.
        let m = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        // Transpose to a (4, 3) view sharing the same buffer: row i of
        // `m_t` is column i of `m`, and (unlike `m`'s own rows) is strided.
        let m_t = m.reversed_axes();
        assert!(
            m_t.row(0).as_slice().is_none(),
            "test precondition: rows of a reversed-axes matrix must be non-contiguous"
        );

        let x = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let mut y = Array1::zeros(4);
        matvec(&m_t, &x, &mut y);

        // Column sums of the original matrix: [1+5+9, 2+6+10, 3+7+11, 4+8+12]
        let expected = [15.0, 18.0, 21.0, 24.0];
        for i in 0..4 {
            assert!(
                (y[i] - expected[i]).abs() < 1e-5,
                "y[{i}] = {}, expected {} (matvec must not silently zero non-contiguous rows)",
                y[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_ssm_state_update_broadcast_scalar_x() {
        let mut h = vec![1.0f32, 2.0, 3.0, 4.0];
        let a_bar = vec![0.5f32, 0.5, 0.5, 0.5];
        let b_bar = vec![1.0f32, 1.0, 1.0, 1.0];
        let x = vec![2.0f32]; // Single-element broadcast input.

        ssm_state_update(&a_bar, &mut h, &b_bar, &x);

        // h'[i] = 0.5 * h[i] + 1.0 * 2.0
        assert_eq!(
            h,
            vec![
                0.5 * 1.0 + 2.0,
                0.5 * 2.0 + 2.0,
                0.5 * 3.0 + 2.0,
                0.5 * 4.0 + 2.0
            ]
        );
    }

    #[test]
    fn test_ssm_state_update_no_broadcast_matches_manual() {
        let mut h = vec![1.0f32, 2.0, 3.0];
        let a_bar = vec![0.9f32, 0.8, 0.7];
        let b_bar = vec![0.1f32, 0.2, 0.3];
        let x = vec![1.0f32, 2.0, 3.0]; // Same length as h: no broadcast.

        let expected: Vec<f32> = (0..3).map(|i| a_bar[i] * h[i] + b_bar[i] * x[i]).collect();
        ssm_state_update(&a_bar, &mut h, &b_bar, &x);

        for i in 0..3 {
            assert!(
                (h[i] - expected[i]).abs() < 1e-6,
                "index {i}: {} != {}",
                h[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_ssm_state_update_empty_x_does_not_panic() {
        // Regression: `x[i % x.len()]` with `x.len() == 0` used to panic
        // (integer division by zero) even in release builds.
        let mut h = vec![1.0f32, 2.0, 3.0];
        let h_before = h.clone();
        let a_bar = vec![0.5f32, 0.5, 0.5];
        let b_bar = vec![1.0f32, 1.0, 1.0];
        let x: Vec<f32> = vec![];

        ssm_state_update(&a_bar, &mut h, &b_bar, &x);

        // No panic, and `h` is left untouched since there was nothing to
        // broadcast.
        assert_eq!(h, h_before);
    }

    #[test]
    fn test_vec_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut c = vec![0.0; 9];
        vec_add(&a, &b, &mut c);
        assert!(c.iter().all(|&v| (v - 10.0).abs() < 1e-5));
    }

    #[test]
    fn test_vec_mul() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let mut c = vec![0.0; 4];
        vec_mul(&a, &b, &mut c);
        assert_eq!(c, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_layer_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        layer_norm(&mut x, 1e-5);

        // Mean should be ~0
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        assert!(mean.abs() < 1e-5);

        // Variance should be ~1
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        assert!((var - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);

        // Should sum to 1
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Should be monotonically increasing
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);
    }

    #[test]
    fn test_online_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        online_softmax(&mut x);

        // Should sum to 1
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Should be monotonically increasing
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);
    }

    #[test]
    fn test_online_softmax_matches_standard() {
        // Test that online softmax gives same results as standard softmax
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        let mut x1 = input.clone();
        let mut x2 = input;

        softmax(&mut x1);
        online_softmax(&mut x2);

        for i in 0..x1.len() {
            let diff = (x1[i] - x2[i]).abs();
            assert!(
                diff < 1e-5,
                "Mismatch at index {}: standard={}, online={}",
                i,
                x1[i],
                x2[i]
            );
        }
    }

    #[test]
    fn test_online_softmax_numerical_stability() {
        // Test with large values that would overflow naive implementation
        let mut x = vec![100.0, 101.0, 102.0];
        online_softmax(&mut x);

        // Should not produce NaN or Inf
        for &val in &x {
            assert!(
                val.is_finite(),
                "Softmax produced non-finite value: {}",
                val
            );
        }

        // Should sum to 1
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fused_softmax_attend() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0];
        let values = vec![10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0; 4];

        fused_softmax_attend(&mut scores, &values, &mut output);

        // Scores should be softmaxed
        let scores_sum: f32 = scores.iter().sum();
        assert!((scores_sum - 1.0).abs() < 1e-5);

        // Output should be weighted sum
        let mut expected_output = 0.0;
        for i in 0..4 {
            expected_output += scores[i] * values[i];
        }

        let actual_sum: f32 = output.iter().sum();
        assert!(
            (actual_sum - expected_output).abs() < 1e-4,
            "Output mismatch: expected sum {}, got {}",
            expected_output,
            actual_sum
        );
    }

    #[test]
    fn test_fast_exp() {
        for i in -10..=10 {
            let x = i as f32;
            let expected = x.exp();
            let result = fast_exp(x);
            let rel_error = (result - expected).abs() / expected.max(1e-6);
            assert!(
                rel_error < 0.02,
                "fast_exp({}) = {}, expected {}",
                x,
                result,
                expected
            );
        }
    }

    #[test]
    fn test_matvec() {
        let m = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let x = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let mut y = Array1::zeros(3);

        matvec(&m, &x, &mut y);

        assert!((y[0] - 10.0).abs() < 1e-5);
        assert!((y[1] - 26.0).abs() < 1e-5);
        assert!((y[2] - 42.0).abs() < 1e-5);
    }
}
