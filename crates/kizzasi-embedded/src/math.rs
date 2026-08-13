//! Pure Rust math utilities compatible with no_std
//!
//! Most operations are implemented locally as range-reduced polynomial
//! approximations and only rely on `core`. A small handful (currently
//! `sqrt` and `round`) genuinely need an FPU intrinsic or a software-float
//! library, so they are routed through an internal `core_math` module which
//! dispatches to libstd's `f32::sqrt` under the `std` feature and to
//! `libm::sqrtf` under the `libm` feature.
//!
//! # Accuracy contract
//!
//! Every approximation below documents the worst-case error measured by the
//! unit tests in this module against the libstd reference implementation:
//!
//! | Function       | Domain              | Worst relative error |
//! |----------------|---------------------|----------------------|
//! | [`exp_approx`] | `[-88, 88]`         | `< 1e-6`             |
//! | [`ln_approx`]  | `(0, 1e6]`          | `< 1e-6`             |
//! | [`softplus`]   | `[-40, 40]`         | `< 1e-5`             |
//! | [`sigmoid`]    | `[-40, 40]`         | `< 1e-6`             |
//! | [`sin_approx`] | `[-1e3, 1e3]`       | `< 1e-5` (absolute)  |
//! | [`cos_approx`] | `[-1e3, 1e3]`       | `< 1e-5` (absolute)  |

use core::f32::consts::{FRAC_2_PI, LN_2};

use crate::error::{EmbeddedError, EmbeddedResult};

/// `f32` operations that are not directly available in `core`.
///
/// On `std` builds these forward to the corresponding `f32` methods (which
/// in turn use the platform FPU or libstd's software-float). On `no_std`
/// builds with the `libm` feature they forward to `libm`'s `sqrtf` /
/// `roundf` family. When neither feature is enabled a crate-level
/// `compile_error!` in `lib.rs` rejects the build, so the corresponding
/// shim bodies are never compiled.
#[cfg(any(feature = "std", feature = "libm"))]
pub(crate) mod core_math {
    /// Square root of a non-negative `f32`.
    #[inline]
    pub(crate) fn sqrt(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            f32::sqrt(x)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::sqrtf(x)
        }
    }

    /// Round an `f32` to the nearest integer, halves away from zero.
    #[inline]
    pub(crate) fn round(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            f32::round(x)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::roundf(x)
        }
    }
}

/// Exact `2^k` for an integer exponent, including the subnormal range.
///
/// Returns `f32::INFINITY` above the representable range and `0.0` below it.
#[inline]
fn pow2i(k: i32) -> f32 {
    if k > 127 {
        f32::INFINITY
    } else if k >= -126 {
        // Normal range: write the biased exponent directly.
        f32::from_bits(((127 + k) as u32) << 23)
    } else if k >= -252 {
        // Subnormal range: compose two normal scalings so the exponent
        // field never underflows.
        f32::from_bits(((127 + (k + 126)) as u32) << 23) * f32::from_bits(1_u32 << 23)
    } else {
        0.0
    }
}

/// Cody-Waite split of `ln(2)`: `LN2_HI` has its low 9 significand bits zero,
/// so `k * LN2_HI` is exact in `f32` for every `|k| < 512` (we only ever use
/// `|k| <= 127`). `LN2_HI + LN2_LO` reproduces the true `ln 2` to `5.5e-14`.
/// Written as bit patterns because the decimal forms are exact only at full
/// precision, which `clippy::excessive_precision` (rightly) objects to.
const LN2_HI: f32 = f32::from_bits(0x3f31_7200); // 0.693145751953125
const LN2_LO: f32 = f32::from_bits(0x35bf_be8e); // 1.4286067653e-6

/// Cody-Waite split of `pi/2`, same idea: `PI_2_HI` has 16 trailing zero
/// significand bits, so `k * PI_2_HI` is exact for `|k| < 65536`.
const PI_2_HI: f32 = f32::from_bits(0x3fc9_0000); // 1.5703125
const PI_2_LO: f32 = f32::from_bits(0x39fd_aa22); // 4.8382679233e-4

/// Compute `exp(x)` via a range-reduced Taylor series.
///
/// The argument is clamped to `[-88, 88]` (the f32 exponent range) and then
/// reduced as `x = k*ln2 + r` with `k = round(x / ln2)`, which keeps
/// `|r| <= ln2/2 ≈ 0.3466` **for both signs** — a truncating cast would only
/// round correctly for positive `x` and would triple `|r|` on the negative
/// side. The subtraction uses a Cody-Waite two-part `ln 2` so the reduction
/// stays exact out to `|x| = 88`; a single-precision `ln 2` would leak up to
/// `2.6e-6` of absolute error into `r` at large `|k|`. A 7-term Taylor series
/// on `r` then has a truncation error below `1.2e-8`, and the `2^k` factor is
/// applied exactly via an exponent-field write.
pub fn exp_approx(x: f32) -> f32 {
    // Clamp to avoid overflow/underflow.
    let x = x.clamp(-88.0, 88.0);
    // Range reduction: x = k*ln2 + r, |r| <= 0.5*ln2 (round-half-away-from-zero
    // via `core_math::round`, which is correct on both sides of zero).
    let kf = core_math::round(x / LN_2);
    let k = kf as i32;
    let r = (x - kf * LN2_HI) - kf * LN2_LO;
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
    poly * pow2i(k)
}

/// `sqrt(2)`, the split point of the [`ln_approx`] mantissa reduction.
const SQRT_2: f32 = core::f32::consts::SQRT_2;

/// Natural logarithm approximation using IEEE 754 exponent decomposition.
///
/// Decomposes `x = m * 2^e` with `m` in `[1, 2)`, then re-centres the
/// mantissa onto `[sqrt(2)/2, sqrt(2))` so that `s = (m - 1) / (m + 1)`
/// satisfies `|s| <= 0.1716`. On that interval the odd `atanh` series
/// `ln(m) = 2*(s + s^3/3 + s^5/5 + s^7/7)` has a truncation error below
/// `3e-8`, i.e. below `f32` epsilon.
///
/// Returns `f32::NEG_INFINITY` for `x <= 0` and `f32::NAN` for a NaN input.
pub fn ln_approx(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    let bits = x.to_bits();
    let raw_exp = ((bits >> 23) & 0xFF) as i32;
    // Subnormal inputs carry no implicit leading 1; scale them up by 2^24
    // first and correct the exponent afterwards.
    let (bits, mut exp) = if raw_exp == 0 {
        let scaled = x * pow2i(24);
        (scaled.to_bits(), -24)
    } else {
        (bits, 0)
    };
    exp += (((bits >> 23) & 0xFF) as i32) - 127;
    let mantissa_bits = (bits & 0x7F_FFFF) | 0x3F80_0000;
    let mut m = f32::from_bits(mantissa_bits); // m in [1.0, 2.0)
                                               // Re-centre onto [sqrt(2)/2, sqrt(2)) so |s| <= (sqrt2-1)/(sqrt2+1).
                                               // Halving is exact in binary, so this costs no precision.
    if m > SQRT_2 {
        m *= 0.5;
        exp += 1;
    }
    // ln(m) = 2*atanh(s) with s = (m - 1) / (m + 1), |s| <= 0.1716.
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let ln_m = 2.0 * s * (1.0 + s2 * ((1.0 / 3.0) + s2 * ((1.0 / 5.0) + s2 * (1.0 / 7.0))));
    ln_m + exp as f32 * LN_2
}

/// `ln(1 + u)` for `0 <= u <= 1`, accurate in the *relative* sense.
///
/// Below `u = 1/16` the naive `ln_approx(1.0 + u)` is useless: forming
/// `1.0 + u` in `f32` throws away everything under half an ulp of 1.0
/// (`6e-8`), which for `u = 1e-9` is the entire answer. The series is used
/// there instead; its `u^6/6` truncation term is below `1e-7` of `u`.
#[inline]
fn ln_1p_nonneg(u: f32) -> f32 {
    if u < 0.062_5 {
        u * (1.0 - u * (0.5 - u * ((1.0 / 3.0) - u * (0.25 - u * 0.2))))
    } else {
        ln_approx(1.0 + u)
    }
}

/// Softplus: `ln(1 + exp(x))`
///
/// Evaluated in the numerically stable form `max(x, 0) + ln(1 + exp(-|x|))`,
/// so the `exp` argument is never positive and the `ln1p` argument always lies
/// in `(0, 1]`. The result decays to `exp(x)` for large negative `x` with full
/// relative accuracy (no underflow to zero) and to `x` for large positive `x`,
/// without any special-cased branch cut.
pub fn softplus(x: f32) -> f32 {
    let abs_x = if x < 0.0 { -x } else { x };
    let base = if x > 0.0 { x } else { 0.0 };
    base + ln_1p_nonneg(exp_approx(-abs_x))
}

/// Sigmoid: σ(x) = 1 / (1 + e^(-x))
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp_approx(-x))
}

/// SiLU (Swish): x * σ(x)
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `sin(r)` for `|r| <= pi/4` — 5-term odd Taylor series (error `< 2e-9`).
#[inline]
fn sin_kernel(r: f32) -> f32 {
    let r2 = r * r;
    r * (1.0
        + r2 * (-(1.0 / 6.0)
            + r2 * ((1.0 / 120.0) + r2 * (-(1.0 / 5040.0) + r2 * (1.0 / 362_880.0)))))
}

/// `cos(r)` for `|r| <= pi/4` — 5-term even Taylor series (error `< 3e-8`).
#[inline]
fn cos_kernel(r: f32) -> f32 {
    let r2 = r * r;
    1.0 + r2 * (-0.5 + r2 * ((1.0 / 24.0) + r2 * (-(1.0 / 720.0) + r2 * (1.0 / 40_320.0))))
}

/// Reduce `x` to a quadrant index and a remainder near `[-pi/4, pi/4]`.
///
/// Uses the same Cody-Waite trick as [`exp_approx`]: `k * PI_2_HI` is exact,
/// so the only error comes from the tiny `k * PI_2_LO` term.
#[inline]
fn quadrant_reduce(x: f32) -> (i32, f32) {
    let kf = core_math::round(x * FRAC_2_PI);
    // `as i32` saturates rather than wrapping for out-of-range floats, so no
    // overflow is possible here even for absurd arguments.
    let k = kf as i32;
    let r = (x - kf * PI_2_HI) - kf * PI_2_LO;
    (k & 3, r)
}

/// Sine approximation via quadrant reduction plus a Taylor kernel.
///
/// Accurate to `< 1e-5` absolute for `|x| <= 1e3`. Like every finite-precision
/// argument reduction the accuracy degrades for very large `|x|`, because the
/// `x - k*pi/2` subtraction loses significant bits; callers needing large
/// arguments should reduce modulo `2*pi` in a wider type first.
pub fn sin_approx(x: f32) -> f32 {
    let (quadrant, r) = quadrant_reduce(x);
    match quadrant {
        0 => sin_kernel(r),
        1 => cos_kernel(r),
        2 => -sin_kernel(r),
        _ => -cos_kernel(r),
    }
}

/// Cosine approximation via quadrant reduction plus a Taylor kernel.
///
/// Accurate to `< 1e-5` absolute for `|x| <= 1e3`; see [`sin_approx`] for the
/// large-argument caveat.
pub fn cos_approx(x: f32) -> f32 {
    let (quadrant, r) = quadrant_reduce(x);
    match quadrant {
        0 => cos_kernel(r),
        1 => -sin_kernel(r),
        2 => -cos_kernel(r),
        _ => sin_kernel(r),
    }
}

/// Softmax in-place over a slice.
///
/// Applies numerical stability via max subtraction, then normalises.
///
/// # Errors
///
/// Returns [`EmbeddedError::NumericalInstability`] when the input contains a
/// non-finite value, or when the exponential sum is not a usable positive
/// finite number. In that case the slice is left **unmodified** rather than
/// being overwritten with un-normalised exponentials — an all-NaN buffer that
/// still looks like a probability vector is exactly the silent corruption this
/// check exists to prevent.
pub fn softmax_inplace(xs: &mut [f32]) -> EmbeddedResult<()> {
    if xs.is_empty() {
        return Ok(());
    }
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(EmbeddedError::NumericalInstability);
    }
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Compute the sum first so a degenerate input cannot leave the caller's
    // buffer half-written.
    let mut sum = 0.0_f32;
    for &x in xs.iter() {
        sum += exp_approx(x - max);
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(EmbeddedError::NumericalInstability);
    }
    let inv_sum = 1.0 / sum;
    for x in xs.iter_mut() {
        *x = exp_approx(*x - max) * inv_sum;
    }
    Ok(())
}

/// Layer normalisation: (x - mean) / sqrt(var + eps) * weight + bias
///
/// `weight` and `bias` may be empty slices, in which case weight defaults
/// to 1.0 and bias to 0.0 for each element.
///
/// # Errors
///
/// Returns [`EmbeddedError::NumericalInstability`] when `var + eps` is not a
/// strictly positive finite number (e.g. `eps == 0.0` on a constant input),
/// which would otherwise divide by zero and fill the slice with `NaN`/`inf`.
pub fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) -> EmbeddedResult<()> {
    if x.is_empty() {
        return Ok(());
    }
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    // Routed through `core_math::sqrt` so the call works under both
    // `std` (libstd's `f32::sqrt`) and `no_std + libm` (`libm::sqrtf`).
    let std_dev = core_math::sqrt(var + eps);
    if !std_dev.is_finite() || std_dev <= 0.0 {
        return Err(EmbeddedError::NumericalInstability);
    }
    let inv_std = 1.0 / std_dev;
    for (i, v) in x.iter_mut().enumerate() {
        let w = weight.get(i).copied().unwrap_or(1.0);
        let b = bias.get(i).copied().unwrap_or(0.0);
        *v = (*v - mean) * inv_std * w + b;
    }
    Ok(())
}

/// Dot product of two slices, checked for equal length.
///
/// # Errors
///
/// Returns [`EmbeddedError::DimensionMismatch`] when the slices differ in
/// length. The previous `zip`-based implementation silently truncated to the
/// shorter slice and returned a plausible-looking partial sum, which on an
/// embedded target is undetectable numerical corruption.
pub fn dot(a: &[f32], b: &[f32]) -> EmbeddedResult<f32> {
    if a.len() != b.len() {
        return Err(EmbeddedError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(dot_unchecked(a, b))
}

/// Dot product over two slices already known to have equal length.
#[inline]
fn dot_unchecked(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector product: `y = A * x`, `A` row-major with shape `(m, n)`.
///
/// # Errors
///
/// Returns [`EmbeddedError::DimensionMismatch`] unless `a.len() == m * n`,
/// `x.len() == n` and `y.len() == m`, and
/// [`EmbeddedError::InvalidConfig`] when `m * n` overflows `usize`.
///
/// The dimensions used to be checked with `debug_assert_eq!` only, so a
/// release build would either panic on an out-of-range slice (a hard abort
/// on `no_std`, with no unwinding) or — when `x` was too short — silently
/// return a truncated result.
pub fn matvec(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) -> EmbeddedResult<()> {
    let expected = m
        .checked_mul(n)
        .ok_or(EmbeddedError::InvalidConfig("m * n overflows usize"))?;
    if a.len() != expected {
        return Err(EmbeddedError::DimensionMismatch {
            expected,
            got: a.len(),
        });
    }
    if x.len() != n {
        return Err(EmbeddedError::DimensionMismatch {
            expected: n,
            got: x.len(),
        });
    }
    if y.len() != m {
        return Err(EmbeddedError::DimensionMismatch {
            expected: m,
            got: y.len(),
        });
    }
    if n == 0 {
        // Every row is an empty dot product; `chunks_exact(0)` would panic.
        y.iter_mut().for_each(|v| *v = 0.0);
        return Ok(());
    }
    for (row, out) in a.chunks_exact(n).zip(y.iter_mut()) {
        *out = dot_unchecked(row, x);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Worst relative error of `approx` against `reference` over a sweep.
    ///
    /// The reference is evaluated at the **f32-rounded** sample point, so the
    /// measurement isolates the approximation error instead of re-measuring
    /// the `f64 -> f32` quantisation of the sweep argument (which alone is
    /// worth `4e-6` relative for `exp` near `x = 72`).
    fn worst_rel_error(
        lo: f64,
        hi: f64,
        steps: usize,
        approx: impl Fn(f32) -> f32,
        reference: impl Fn(f64) -> f64,
    ) -> (f64, f64) {
        let mut worst = 0.0_f64;
        let mut worst_at = lo;
        for i in 0..=steps {
            let x = (lo + (hi - lo) * (i as f64) / (steps as f64)) as f32;
            let got = approx(x) as f64;
            let want = reference(x as f64);
            let rel = (got - want).abs() / want.abs().max(1e-30);
            if rel > worst {
                worst = rel;
                worst_at = x as f64;
            }
        }
        (worst, worst_at)
    }

    #[test]
    fn test_exp_approx_near_zero() {
        let result = exp_approx(0.0);
        assert!(
            (result - 1.0).abs() < 1e-6,
            "exp_approx(0) = {result}, expected ~1.0"
        );
    }

    #[test]
    fn test_exp_approx_one() {
        let result = exp_approx(1.0);
        assert!(
            (result - core::f32::consts::E).abs() < 1e-5,
            "exp_approx(1) = {result}, expected ~2.718"
        );
    }

    #[test]
    fn test_exp_approx_accuracy_sweep_both_signs() {
        // Regression guard for the truncating range reduction: the previous
        // `(x / LN_2 + 0.5) as i32` cast rounded towards zero, which is a
        // *ceiling* for negative x and pushed |r| up to ~1.04, costing about
        // four decimal digits on the whole negative half-line.
        let (worst, at) = worst_rel_error(-88.0, 88.0, 40_000, exp_approx, f64::exp);
        assert!(
            worst < 1e-6,
            "exp_approx worst relative error {worst:.3e} at x = {at} exceeds 1e-6"
        );

        // And specifically the negative side, which was the degraded branch.
        let (worst_neg, at_neg) = worst_rel_error(-88.0, 0.0, 20_000, exp_approx, f64::exp);
        assert!(
            worst_neg < 1e-6,
            "exp_approx negative-side error {worst_neg:.3e} at x = {at_neg} exceeds 1e-6"
        );
    }

    #[test]
    fn test_exp_approx_extremes_are_finite() {
        assert!(
            exp_approx(88.0).is_finite(),
            "exp_approx(88) must be finite"
        );
        assert!(
            exp_approx(-88.0) >= 0.0,
            "exp_approx(-88) must be non-negative"
        );
        assert!(
            exp_approx(1e30) > 0.0,
            "exp_approx clamps huge arguments instead of producing NaN"
        );
    }

    #[test]
    fn test_ln_approx_accuracy_sweep() {
        // The old body was the 4-term Taylor of ln(1+t) evaluated out to
        // t -> 1, worst-casing at m ~ 1.95 with 13 % relative error.
        let (worst, at) = worst_rel_error(1.0e-3, 1.0e6, 60_000, ln_approx, f64::ln);
        assert!(
            worst < 1e-6,
            "ln_approx worst relative error {worst:.3e} at x = {at} exceeds 1e-6"
        );
    }

    #[test]
    fn test_ln_approx_worst_case_mantissa() {
        // m ~ 1.95 was the historic worst point (0.58092 vs 0.66783).
        for x in [1.95_f32, 1.9999, 127.999, 3.9] {
            let got = ln_approx(x);
            let want = (x as f64).ln() as f32;
            let rel = ((got - want) / want).abs();
            assert!(
                rel < 1e-6,
                "ln_approx({x}) = {got}, expected {want}, rel = {rel:.3e}"
            );
        }
    }

    #[test]
    fn test_ln_approx_edge_cases() {
        assert_eq!(ln_approx(1.0), 0.0, "ln(1) must be exactly 0");
        assert_eq!(
            ln_approx(0.0),
            f32::NEG_INFINITY,
            "ln(0) must be -inf, not NaN"
        );
        assert_eq!(
            ln_approx(-1.0),
            f32::NEG_INFINITY,
            "ln of a negative is -inf"
        );
        assert!(ln_approx(f32::NAN).is_nan(), "ln(NaN) must stay NaN");
        assert_eq!(ln_approx(f32::INFINITY), f32::INFINITY);
        // Subnormal input must not be treated as exponent 0.
        let sub = f32::from_bits(1);
        let got = ln_approx(sub);
        let want = (sub as f64).ln() as f32;
        assert!(
            ((got - want) / want).abs() < 1e-5,
            "ln_approx(subnormal) = {got}, expected {want}"
        );
    }

    #[test]
    fn test_softplus_accuracy_sweep() {
        // softplus inherited ln_approx's 6 % error around x = 1.065.
        // The reference itself must use the stable form: a naive
        // `(1.0 + x.exp()).ln()` collapses to exactly 0.0 below x = -36 even
        // in f64, which would make a *correct* implementation look infinitely
        // wrong.
        let reference = |x: f64| x.max(0.0) + (-x.abs()).exp().ln_1p();
        let (worst, at) = worst_rel_error(-40.0, 40.0, 40_000, softplus, reference);
        assert!(
            worst < 1e-5,
            "softplus worst relative error {worst:.3e} at x = {at} exceeds 1e-5"
        );
    }

    #[test]
    fn test_softplus_known_points() {
        for (x, want) in [
            (1.065_f32, 1.361_f32),
            (0.0, core::f32::consts::LN_2),
            (-1.0, 0.313_261_7),
        ] {
            let got = softplus(x);
            assert!(
                (got - want).abs() < 1e-3,
                "softplus({x}) = {got}, expected ~{want}"
            );
        }
        // Large positive arguments must degrade to x itself.
        assert!(
            (softplus(30.0) - 30.0).abs() < 1e-4,
            "softplus(30) should be ~30"
        );
        // Large negative arguments must degrade to ~0 without going negative.
        assert!(
            softplus(-30.0) >= 0.0 && softplus(-30.0) < 1e-6,
            "softplus(-30) should be a tiny non-negative number"
        );
    }

    #[test]
    fn test_sigmoid_zero() {
        let result = sigmoid(0.0);
        assert!(
            (result - 0.5).abs() < 1e-6,
            "sigmoid(0) = {result}, expected ~0.5"
        );
    }

    #[test]
    fn test_sigmoid_accuracy_sweep() {
        let reference = |x: f64| 1.0 / (1.0 + (-x).exp());
        let (worst, at) = worst_rel_error(-40.0, 40.0, 20_000, sigmoid, reference);
        assert!(
            worst < 1e-6,
            "sigmoid worst relative error {worst:.3e} at x = {at} exceeds 1e-6"
        );
    }

    #[test]
    fn test_silu_zero() {
        assert_eq!(silu(0.0), 0.0, "silu(0) must be exactly 0.0");
    }

    #[test]
    fn test_silu_matches_reference() {
        for x in [-3.0_f32, -1.0, 0.5, 2.0, 6.0] {
            let want = (x as f64) / (1.0 + (-(x as f64)).exp());
            let got = silu(x) as f64;
            assert!(
                (got - want).abs() < 1e-5,
                "silu({x}) = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn test_sin_cos_accuracy() {
        // The reference is evaluated at the f32-rounded sample point: at
        // |x| = 1000 the f32 grid is 6e-5 wide, so comparing against the f64
        // sample would measure that quantisation rather than the kernel.
        let mut worst = 0.0_f64;
        let mut worst_at = 0.0_f64;
        for i in -20_000_i32..=20_000 {
            let x = (i as f64 * 0.05) as f32; // sweep [-1000, 1000]
            let xd = x as f64;
            let s_err = (sin_approx(x) as f64 - xd.sin()).abs();
            let c_err = (cos_approx(x) as f64 - xd.cos()).abs();
            let err = s_err.max(c_err);
            if err > worst {
                worst = err;
                worst_at = xd;
            }
        }
        assert!(
            worst < 1e-5,
            "sin/cos worst absolute error {worst:.3e} at x = {worst_at} exceeds 1e-5"
        );
    }

    #[test]
    fn test_sin_cos_identity() {
        for i in -100_i32..=100 {
            let x = i as f32 * 0.137;
            let s = sin_approx(x);
            let c = cos_approx(x);
            assert!(
                (s * s + c * c - 1.0).abs() < 1e-5,
                "sin^2 + cos^2 != 1 at x = {x}: {}",
                s * s + c * c
            );
        }
    }

    #[test]
    fn test_softmax_normalises() {
        let mut xs = [1.0_f32, 2.0, 3.0];
        softmax_inplace(&mut xs).expect("finite input must succeed");
        let sum: f32 = xs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}, expected 1");
        // Monotonic: larger logits get larger probabilities.
        assert!(xs[0] < xs[1] && xs[1] < xs[2], "softmax must be monotonic");
        // Reference values.
        let denom: f64 = [1.0_f64, 2.0, 3.0].iter().map(|v| v.exp()).sum();
        for (i, v) in xs.iter().enumerate() {
            let want = ((i as f64 + 1.0).exp() / denom) as f32;
            assert!(
                (v - want).abs() < 1e-5,
                "softmax[{i}] = {v}, expected {want}"
            );
        }
    }

    #[test]
    fn test_softmax_all_equal_and_single() {
        let mut xs = [5.0_f32; 4];
        softmax_inplace(&mut xs).expect("uniform input must succeed");
        for v in xs {
            assert!(
                (v - 0.25).abs() < 1e-5,
                "uniform softmax must be 1/n, got {v}"
            );
        }
        let mut one = [42.0_f32];
        softmax_inplace(&mut one).expect("single element must succeed");
        assert!((one[0] - 1.0).abs() < 1e-6, "1-element softmax must be 1.0");
        let mut empty: [f32; 0] = [];
        assert!(
            softmax_inplace(&mut empty).is_ok(),
            "empty slice is a no-op"
        );
    }

    #[test]
    fn test_softmax_rejects_nan_without_corrupting_input() {
        let mut xs = [1.0_f32, f32::NAN, 3.0];
        let result = softmax_inplace(&mut xs);
        assert_eq!(
            result,
            Err(EmbeddedError::NumericalInstability),
            "NaN input must be rejected instead of silently un-normalised"
        );
        // The caller's buffer must be untouched.
        assert_eq!(xs[0], 1.0);
        assert!(xs[1].is_nan());
        assert_eq!(xs[2], 3.0);

        let mut inf = [f32::INFINITY, 0.0];
        assert_eq!(
            softmax_inplace(&mut inf),
            Err(EmbeddedError::NumericalInstability)
        );
    }

    #[test]
    fn test_layer_norm_zero_mean() {
        let mut x = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        layer_norm(&mut x, &[], &[], 1e-5).expect("layer_norm must succeed");
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        assert!(
            mean.abs() < 1e-4,
            "after layer_norm mean = {mean}, expected ~0"
        );
    }

    #[test]
    fn test_layer_norm_rejects_zero_variance_without_eps() {
        let mut x = [2.0_f32; 4];
        assert_eq!(
            layer_norm(&mut x, &[], &[], 0.0),
            Err(EmbeddedError::NumericalInstability),
            "constant input with eps = 0 must not divide by zero"
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
        matvec(&identity, &x, &mut y, 3, 3).expect("dimensions match");
        for (yi, xi) in y.iter().zip(x.iter()) {
            assert!(
                (yi - xi).abs() < 1e-6,
                "identity * x: got {yi}, expected {xi}"
            );
        }
    }

    #[test]
    fn test_matvec_rejects_every_mismatch() {
        let a = [1.0_f32; 6]; // 2x3
        let x = [1.0_f32; 3];
        let mut y = [0.0_f32; 2];

        // a too short
        assert!(
            matvec(&a[..5], &x, &mut y, 2, 3).is_err(),
            "short A rejected"
        );
        // x too short — this used to silently truncate via zip
        assert_eq!(
            matvec(&a, &x[..2], &mut y, 2, 3),
            Err(EmbeddedError::DimensionMismatch {
                expected: 3,
                got: 2
            }),
            "short x must be rejected, not silently truncated"
        );
        // y too short — this used to panic on index in release
        assert_eq!(
            matvec(&a, &x, &mut y[..1], 2, 3),
            Err(EmbeddedError::DimensionMismatch {
                expected: 2,
                got: 1
            }),
            "short y must be rejected, not panic"
        );
        // Overflowing m * n
        assert_eq!(
            matvec(&a, &x, &mut y, usize::MAX, 2),
            Err(EmbeddedError::InvalidConfig("m * n overflows usize")),
            "m * n overflow must be reported, not wrap"
        );
    }

    #[test]
    fn test_matvec_zero_columns() {
        let a: [f32; 0] = [];
        let x: [f32; 0] = [];
        let mut y = [9.0_f32; 3];
        matvec(&a, &x, &mut y, 3, 0).expect("n = 0 is a valid empty product");
        assert_eq!(y, [0.0; 3], "zero-column matvec must zero the output");
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let result = dot(&a, &b).expect("equal lengths");
        assert!(
            (result - 32.0).abs() < 1e-5,
            "dot([1,2,3],[4,5,6]) = {result}, expected 32"
        );
    }

    #[test]
    fn test_dot_rejects_length_mismatch() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0];
        assert_eq!(
            dot(&a, &b),
            Err(EmbeddedError::DimensionMismatch {
                expected: 3,
                got: 2
            }),
            "unequal lengths must error instead of truncating"
        );
    }
}
