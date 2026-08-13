//! Array-oriented Operations for Model Inference
//!
//! Provides `Array1`/`Array2`-oriented implementations of common operations
//! used in SSMs:
//! - Element-wise arithmetic (add, mul, fused multiply-add)
//! - State update kernels
//! - Activation functions (SiLU, Swish, etc.)
//!
//! # Performance
//!
//! These are auto-vectorization-friendly scalar kernels: element-wise
//! functions are written as `ndarray::Zip::for_each` loops (which the
//! compiler can often auto-vectorize) rather than hand-written SIMD
//! intrinsics. There is no explicit `std::simd`/`std::arch` code in this
//! module and no runtime CPU-feature dispatch.
//!
//! This crate's actual explicit-SIMD kernels (with real AVX-512/AVX2/NEON
//! runtime dispatch) live in [`kizzasi_core::simd`] — see that module for a
//! `dot_product`/`matvec`/`ssm_state_update` with genuine per-architecture
//! vector code. Prefer those in a new hot path; the kernels here exist for
//! callers that want a convenient `Array1`-returning (rather than
//! `&mut [f32]`-writing) signature.
//!
//! # Design
//!
//! Every function here is a plain, portable scalar implementation over
//! `scirs2_core::ndarray` arrays — no `#[cfg]`-gated fallback exists because
//! there is only one code path.

use scirs2_core::ndarray::{Array1, Array2, Zip};

/// Fused multiply-add: out = a * b + c
///
/// Element-wise `ndarray::Zip::for_each` loop over large arrays. Not an
/// explicit-SIMD kernel — see the module docs.
///
/// # Arguments
///
/// * `a` - First multiplicand
/// * `b` - Second multiplicand
/// * `c` - Addend
///
/// # Returns
///
/// Result array with `out[i] = a[i] * b[i] + c[i]`
#[inline]
pub fn fused_mul_add(a: &Array1<f32>, b: &Array1<f32>, c: &Array1<f32>) -> Array1<f32> {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());

    let mut result = Array1::zeros(a.len());

    Zip::from(&mut result)
        .and(a)
        .and(b)
        .and(c)
        .for_each(|r, &av, &bv, &cv| {
            *r = av.mul_add(bv, cv);
        });

    result
}

/// Fused multiply-add with scalar: out = a * scalar + b
///
/// # Arguments
///
/// * `a` - Array to multiply
/// * `scalar` - Scalar multiplier
/// * `b` - Array to add
#[inline]
pub fn fused_mul_scalar_add(a: &Array1<f32>, scalar: f32, b: &Array1<f32>) -> Array1<f32> {
    debug_assert_eq!(a.len(), b.len());

    let mut result = Array1::zeros(a.len());

    Zip::from(&mut result)
        .and(a)
        .and(b)
        .for_each(|r, &av, &bv| {
            *r = scalar.mul_add(av, bv);
        });

    result
}

/// Element-wise exponential: `out[i] = exp(a[i])`
///
/// A thin `Array1::mapv` wrapper around `f32::exp` — no vectorized `exp` is
/// implemented here.
#[inline]
pub fn exp_array(a: &Array1<f32>) -> Array1<f32> {
    a.mapv(f32::exp)
}

/// Element-wise natural logarithm: `out[i] = ln(a[i])`
#[inline]
pub fn ln_array(a: &Array1<f32>) -> Array1<f32> {
    a.mapv(f32::ln)
}

/// SiLU activation: `out[i] = x[i] * sigmoid(x[i])`
///
/// Also known as Swish activation. Computes the exact sigmoid formula
/// (`1 / (1 + exp(-x))`), not an approximation.
#[inline]
pub fn silu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| {
        let sigmoid = 1.0 / (1.0 + (-v).exp());
        v * sigmoid
    })
}

/// Sigmoid: `1 / (1 + exp(-x))`
///
/// The exact formula, computed element-wise via `Array1::mapv` — not a
/// SIMD kernel or an approximation.
#[inline]
pub fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Tanh activation with optimized implementation
#[inline]
pub fn tanh_array(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(f32::tanh)
}

/// ReLU activation: max(0, x)
#[inline]
pub fn relu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.max(0.0))
}

/// GELU activation (approximate)
///
/// Uses tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
#[inline]
pub fn gelu(x: &Array1<f32>) -> Array1<f32> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/π)
    const COEFF: f32 = 0.044715;

    x.mapv(|v| {
        let inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    })
}

/// SSM state update kernel (`Array1`-returning; see also
/// [`kizzasi_core::simd::ssm_state_update`] for the crate's explicit-SIMD,
/// in-place equivalent)
///
/// Performs the recurrent state update: `h[t] = a * h[t-1] + b * x[t]`
///
/// # Arguments
///
/// * `h_prev` - Previous state
/// * `a` - Decay factor (per-element)
/// * `b` - Input projection (per-element)
/// * `x` - Current input (per-element)
///
/// # Returns
///
/// Updated state
#[inline]
pub fn ssm_state_update(
    h_prev: &Array1<f32>,
    a: &Array1<f32>,
    b: &Array1<f32>,
    x: &Array1<f32>,
) -> Array1<f32> {
    debug_assert_eq!(h_prev.len(), a.len());
    debug_assert_eq!(h_prev.len(), b.len());
    debug_assert_eq!(h_prev.len(), x.len());

    let mut h_new = Array1::zeros(h_prev.len());

    Zip::from(&mut h_new)
        .and(h_prev)
        .and(a)
        .and(b)
        .and(x)
        .for_each(|h, &h_p, &av, &bv, &xv| {
            // Fused multiply-add: a * h_prev + b * x
            *h = av.mul_add(h_p, bv * xv);
        });

    h_new
}

/// Diagonal SSM state update with scalar input
///
/// For models with diagonal state matrices and scalar inputs
///
/// # Arguments
///
/// * `state` - Previous state `[state_dim]`
/// * `a_diag` - Diagonal A matrix elements `[state_dim]`
/// * `b` - B vector `[state_dim]`
/// * `x_scalar` - Scalar input value
///
/// # Returns
///
/// Updated state
#[inline]
pub fn diagonal_ssm_update(
    state: &Array1<f32>,
    a_diag: &Array1<f32>,
    b: &Array1<f32>,
    x_scalar: f32,
) -> Array1<f32> {
    debug_assert_eq!(state.len(), a_diag.len());
    debug_assert_eq!(state.len(), b.len());

    let mut new_state = Array1::zeros(state.len());

    Zip::from(&mut new_state)
        .and(state)
        .and(a_diag)
        .and(b)
        .for_each(|s, &state_val, &a_val, &b_val| {
            *s = a_val.mul_add(state_val, b_val * x_scalar);
        });

    new_state
}

/// Matrix-vector product: y = A * x
///
/// Row-major traversal (each row is contiguous, matching `Array2`'s default
/// layout) with a 4-way unrolled accumulator per row so the four partial
/// sums have no data dependency on each other — a single running
/// accumulator (`sum = row[j].mul_add(x[j], sum)`) forces the compiler to
/// serialize every multiply-add in the row, which is exactly what blocks
/// auto-vectorization / instruction-level parallelism. This mirrors
/// [`kizzasi_core::simd::dot_product_scalar`]'s accumulator pattern; prefer
/// [`kizzasi_core::simd::matvec`] in a new hot path for genuine
/// AVX-512/AVX2/NEON dispatch.
///
/// # Arguments
///
/// * `matrix` - Matrix `[m, n]`
/// * `vector` - Vector `[n]`
///
/// # Returns
///
/// Result vector `[m]`
#[inline]
pub fn matvec(matrix: &Array2<f32>, vector: &Array1<f32>) -> Array1<f32> {
    let (m, n) = matrix.dim();
    debug_assert_eq!(vector.len(), n);
    let n = n.min(vector.len());

    let mut result = Array1::zeros(m);

    for i in 0..m {
        let row = matrix.row(i);
        let chunks = n / 4;
        let remainder = n % 4;

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let mut j = 0;
        for _ in 0..chunks {
            sum0 = row[j].mul_add(vector[j], sum0);
            sum1 = row[j + 1].mul_add(vector[j + 1], sum1);
            sum2 = row[j + 2].mul_add(vector[j + 2], sum2);
            sum3 = row[j + 3].mul_add(vector[j + 3], sum3);
            j += 4;
        }
        for k in 0..remainder {
            sum0 = row[j + k].mul_add(vector[j + k], sum0);
        }

        result[i] = sum0 + sum1 + sum2 + sum3;
    }

    result
}

/// Softmax activation over array
///
/// Numerically stable implementation using max subtraction
#[inline]
pub fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Subtract max for numerical stability
    let exp_values = x.mapv(|v| (v - max_val).exp());
    let sum: f32 = exp_values.sum();

    exp_values.mapv(|v| v / sum)
}

/// Layer normalization
///
/// Normalizes to zero mean and unit variance, then applies affine transformation
///
/// # Arguments
///
/// * `x` - Input vector
/// * `gamma` - Scale parameter
/// * `beta` - Shift parameter
/// * `eps` - Small constant for numerical stability
#[inline]
pub fn layer_norm(
    x: &Array1<f32>,
    gamma: &Array1<f32>,
    beta: &Array1<f32>,
    eps: f32,
) -> Array1<f32> {
    debug_assert_eq!(x.len(), gamma.len());
    debug_assert_eq!(x.len(), beta.len());

    let mean = x.mean().unwrap_or(0.0);
    let variance = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
    let std_dev = (variance + eps).sqrt();

    let mut result = Array1::zeros(x.len());

    Zip::from(&mut result)
        .and(x)
        .and(gamma)
        .and(beta)
        .for_each(|r, &xv, &g, &b| {
            *r = g * ((xv - mean) / std_dev) + b;
        });

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    fn array_approx_eq(a: &Array1<f32>, b: &Array1<f32>, epsilon: f32) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(&av, &bv)| approx_eq(av, bv, epsilon))
    }

    #[test]
    fn test_fused_mul_add() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let c = Array1::from_vec(vec![1.0, 1.0, 1.0]);

        let result = fused_mul_add(&a, &b, &c);
        let expected = Array1::from_vec(vec![3.0, 7.0, 13.0]);

        assert!(array_approx_eq(&result, &expected, 1e-6));
    }

    #[test]
    fn test_silu() {
        let x = Array1::from_vec(vec![0.0, 1.0, -1.0]);
        let result = silu(&x);

        // SiLU(0) = 0, SiLU(1) ≈ 0.731, SiLU(-1) ≈ -0.269
        assert!(approx_eq(result[0], 0.0, 1e-5));
        assert!(approx_eq(result[1], 0.731, 1e-2));
        assert!(approx_eq(result[2], -0.269, 1e-2));
    }

    #[test]
    fn test_relu() {
        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);
        let result = relu(&x);
        let expected = Array1::from_vec(vec![0.0, 0.0, 1.0, 2.0]);

        assert!(array_approx_eq(&result, &expected, 1e-6));
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = softmax(&x);

        // Check sum to 1
        let sum: f32 = result.sum();
        assert!(approx_eq(sum, 1.0, 1e-5));

        // Check monotonicity (larger input -> larger output for softmax)
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_ssm_state_update() {
        let h_prev = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let a = Array1::from_vec(vec![0.9, 0.8, 0.7]);
        let b = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let x = Array1::from_vec(vec![1.0, 1.0, 1.0]);

        let result = ssm_state_update(&h_prev, &a, &b, &x);

        // h_new[i] = a[i] * h_prev[i] + b[i] * x[i]
        let expected = Array1::from_vec(vec![1.0, 1.8, 2.4]);

        assert!(array_approx_eq(&result, &expected, 1e-5));
    }

    #[test]
    fn test_diagonal_ssm_update() {
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let a_diag = Array1::from_vec(vec![0.9, 0.8, 0.7]);
        let b = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let x_scalar = 2.0;

        let result = diagonal_ssm_update(&state, &a_diag, &b, x_scalar);

        // new_state[i] = a_diag[i] * state[i] + b[i] * x_scalar
        let expected = Array1::from_vec(vec![1.1, 2.0, 2.7]);

        assert!(array_approx_eq(&result, &expected, 1e-5));
    }

    #[test]
    fn test_matvec() {
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Failed to create test matrix");
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = matvec(&matrix, &vector);
        let expected = Array1::from_vec(vec![14.0, 32.0]);

        assert!(array_approx_eq(&result, &expected, 1e-5));
    }

    #[test]
    fn test_layer_norm() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let gamma = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let beta = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);

        let result = layer_norm(&x, &gamma, &beta, 1e-5);

        // After normalization, mean should be ~0 and std ~1
        let mean = result.mean().expect("Failed to compute mean");
        assert!(approx_eq(mean, 0.0, 1e-5));
    }
}
