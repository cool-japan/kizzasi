//! Pure Rust math utilities compatible with no_std
//!
//! Uses software floating point — no libm dependency by default.
//! All functions operate on f32 and use compiler intrinsics available in core.

use core::f32::consts::LN_2;

/// Compute exp(x) via Taylor series with range reduction.
///
/// Accurate for |x| < 88 (IEEE 754 f32 range). Uses range reduction
/// x = k*ln(2) + r to keep Taylor series accurate.
pub fn exp_approx(x: f32) -> f32 {
    // Clamp to avoid overflow/underflow
    let x = x.clamp(-88.0, 88.0);
    // Range reduction: x = k*ln2 + r, |r| <= 0.5*ln2
    let k = (x / LN_2 + 0.5) as i32;
    let r = x - k as f32 * LN_2;
    // 7-term Taylor: e^r = 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5! + r^6/6!
    let r2 = r * r;
    let r4 = r2 * r2;
    let poly = 1.0
        + r
        + r2 * 0.5
        + r * r2 * (1.0 / 6.0)
        + r4 * (1.0 / 24.0)
        + r4 * r * (1.0 / 120.0)
        + r4 * r2 * (1.0 / 720.0);
    // Scale by 2^k using IEEE 754 exponent field manipulation
    let pow2k = if k >= 0 {
        let k = k.min(127) as u32;
        f32::from_bits((127 + k) << 23)
    } else {
        let k = (-k).min(126) as u32;
        f32::from_bits((127 - k) << 23)
    };
    poly * pow2k
}

/// Natural logarithm approximation using IEEE 754 exponent decomposition.
///
/// Decomposes x = m * 2^e where m in [1, 2), then approximates ln(m)
/// using a 4th-order minimax polynomial, returning ln(m) + e*ln(2).
pub fn ln_approx(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa_bits = (bits & 0x7F_FFFF) | 0x3F80_0000;
    let m = f32::from_bits(mantissa_bits); // m in [1.0, 2.0)
                                           // ln(m) minimax approximation for m in [1, 2)
    let t = m - 1.0;
    let ln_m = t * (1.0 - t * (0.5 - t * (1.0 / 3.0 - t * 0.25)));
    ln_m + exp as f32 * LN_2
}

/// Softplus: log(1 + exp(x))
///
/// For x > 20 returns x directly to avoid overflow.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        ln_approx(exp_approx(x) + 1.0)
    }
}

/// Sigmoid: σ(x) = 1 / (1 + e^(-x))
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp_approx(-x))
}

/// SiLU (Swish): x * σ(x)
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Softmax in-place over a slice.
///
/// Applies numerical stability via max subtraction, then normalises.
pub fn softmax_inplace(xs: &mut [f32]) {
    if xs.is_empty() {
        return;
    }
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in xs.iter_mut() {
        *x = exp_approx(*x - max);
        sum += *x;
    }
    if sum > 0.0 {
        for x in xs.iter_mut() {
            *x /= sum;
        }
    }
}

/// Layer normalisation: (x - mean) / sqrt(var + eps) * weight + bias
///
/// `weight` and `bias` may be empty slices, in which case weight defaults
/// to 1.0 and bias to 0.0 for each element.
pub fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    if x.is_empty() {
        return;
    }
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    // f32::sqrt is available as a compiler intrinsic in core (no libm needed)
    let std_dev = (var + eps).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        let w = weight.get(i).copied().unwrap_or(1.0);
        let b = bias.get(i).copied().unwrap_or(0.0);
        *v = (*v - mean) / std_dev * w + b;
    }
}

/// Dot product of two equal-length slices.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector product: y = A * x, A is row-major with shape (m, n).
///
/// Output slice `y` must have length `m`.
pub fn matvec(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), m);
    for i in 0..m {
        y[i] = dot(&a[i * n..(i + 1) * n], x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_approx_near_zero() {
        let result = exp_approx(0.0);
        assert!(
            (result - 1.0).abs() < 1e-4,
            "exp_approx(0) = {result}, expected ~1.0"
        );
    }

    #[test]
    fn test_exp_approx_one() {
        let result = exp_approx(1.0);
        assert!(
            (result - core::f32::consts::E).abs() < 0.01,
            "exp_approx(1) = {result}, expected ~2.718"
        );
    }

    #[test]
    fn test_sigmoid_zero() {
        let result = sigmoid(0.0);
        assert!(
            (result - 0.5).abs() < 1e-4,
            "sigmoid(0) = {result}, expected ~0.5"
        );
    }

    #[test]
    fn test_silu_zero() {
        assert_eq!(silu(0.0), 0.0, "silu(0) must be exactly 0.0");
    }

    #[test]
    fn test_layer_norm_zero_mean() {
        let mut x = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        layer_norm(&mut x, &[], &[], 1e-5);
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        assert!(
            mean.abs() < 1e-4,
            "after layer_norm mean = {mean}, expected ~0"
        );
    }

    #[test]
    fn test_matvec_identity() {
        // 3x3 identity matrix
        let identity = [
            1.0_f32, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
        ];
        let x = [3.0_f32, 7.0, -2.0];
        let mut y = [0.0_f32; 3];
        matvec(&identity, &x, &mut y, 3, 3);
        for (yi, xi) in y.iter().zip(x.iter()) {
            assert!(
                (yi - xi).abs() < 1e-6,
                "identity * x: got {yi}, expected {xi}"
            );
        }
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let result = dot(&a, &b);
        assert!(
            (result - 32.0).abs() < 1e-5,
            "dot([1,2,3],[4,5,6]) = {result}, expected 32"
        );
    }
}
