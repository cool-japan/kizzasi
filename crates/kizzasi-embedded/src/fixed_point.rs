//! Fixed-point arithmetic: Q16.16 format.
//!
//! The Q16.16 format uses a 32-bit signed integer where the upper 16 bits
//! are the integer part and the lower 16 bits are the fractional part.
//! This is suitable for microcontrollers without an FPU.
//!
//! Representable range: approximately `[-32768.0, 32767.99998]`.
//! Precision: `1 / 65536 ≈ 1.5e-5`.
//!
//! # Overflow contract
//!
//! **Every operation in this module saturates; none of them wraps.** The
//! arithmetic operators (`+`, `-`, `*`, unary `-`), the conversions
//! ([`Q16::from_f32`], [`Q16::from_i32`]), [`Q16::checked_div`] and
//! [`fixed_dot`] all clamp to `[Q16::MIN, Q16::MAX]` when a result leaves the
//! representable range. Saturation is the only sane choice for a signal
//! kernel: a wrapped `+32767.9 -> -32768.0` sign flip inside a recurrence is
//! indistinguishable from a legitimate value and destroys the whole sequence.
//!
//! # Rounding contract
//!
//! Multiplication, [`fixed_dot`], [`fixed_exp_approx`] and [`fixed_ln_approx`]
//! round the fractional result to nearest (halves away from zero) rather than
//! truncating, so repeated products do not accumulate a one-sided downward
//! bias. [`Q16::checked_div`] and [`Q16::to_i32`] truncate towards zero, which
//! is documented on those methods.

use core::ops::{Add, Mul, Neg, Sub};

use crate::error::{EmbeddedError, EmbeddedResult};
use crate::math::core_math;

const FRAC_BITS: i32 = 16;
const SCALE: i32 = 1 << FRAC_BITS; // 65536
const SCALE_I64: i64 = SCALE as i64;
const HALF_I64: i64 = 1 << (FRAC_BITS - 1); // 32768

/// `ln(2)` in Q16.16 (`0.6931472 * 65536 = 45426.09`).
const LN2_Q16: i64 = 45426;
/// `1 / ln(2)` in Q16.16 (`1.4426950 * 65536 = 94548.46`).
const INV_LN2_Q16: i64 = 94548;
/// `1/3`, `1/5`, `1/7`, `1/9` in Q16.16 — the `atanh` series coefficients.
const INV3_Q16: i64 = 21845;
const INV5_Q16: i64 = 13107;
const INV7_Q16: i64 = 9362;
const INV9_Q16: i64 = 7282;

/// Clamp an `i64` into the `i32` range instead of truncating its high bits.
#[inline]
const fn sat_i32(v: i64) -> i32 {
    if v > i32::MAX as i64 {
        i32::MAX
    } else if v < i32::MIN as i64 {
        i32::MIN
    } else {
        v as i32
    }
}

/// `(a * b) >> 16` with round-to-nearest, halves away from zero.
///
/// `a` and `b` are raw Q16.16 words widened to `i64`, so the product is at
/// most `2^62` and cannot overflow.
#[inline]
const fn mul_raw(a: i64, b: i64) -> i64 {
    let prod = a * b;
    if prod >= 0 {
        (prod + HALF_I64) >> FRAC_BITS
    } else {
        -((-prod + HALF_I64) >> FRAC_BITS)
    }
}

/// Q16.16 fixed-point number (16 integer bits, 16 fractional bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Q16(i32);

impl Q16 {
    /// The value 0.0 in Q16.16
    pub const ZERO: Self = Q16(0);
    /// The value 1.0 in Q16.16
    pub const ONE: Self = Q16(SCALE);
    /// The largest representable value (`≈ 32767.99998`).
    pub const MAX: Self = Q16(i32::MAX);
    /// The smallest representable value (`-32768.0`).
    pub const MIN: Self = Q16(i32::MIN);

    /// Wrap a raw Q16.16 word (an `i32` already scaled by `2^16`).
    #[must_use]
    pub const fn from_raw(raw: i32) -> Self {
        Q16(raw)
    }

    /// Unwrap the raw Q16.16 word.
    #[must_use]
    pub const fn to_raw(self) -> i32 {
        self.0
    }

    /// Convert from `f32` to Q16.16, rounding to nearest and **saturating**
    /// outside `[-32768.0, 32767.99998]`. `NaN` maps to [`Q16::ZERO`].
    #[must_use]
    pub fn from_f32(v: f32) -> Self {
        if v.is_nan() {
            return Q16::ZERO;
        }
        // `f32 as i32` already saturates in Rust, so the clamp is belt and
        // braces for readers rather than a correctness requirement.
        let scaled = core_math::round(v * SCALE as f32);
        Q16(scaled.clamp(i32::MIN as f32, i32::MAX as f32) as i32)
    }

    /// Convert from Q16.16 to `f32`.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / SCALE as f32
    }

    /// Convert from an `i32` integer to Q16.16, **saturating** outside
    /// `[-32768, 32767]`.
    ///
    /// The previous `v << FRAC_BITS` silently wrapped for `|v| >= 32768`, so
    /// `from_i32(40000)` decoded as `-25536.0`.
    #[must_use]
    pub fn from_i32(v: i32) -> Self {
        Q16(sat_i32((v as i64) << FRAC_BITS))
    }

    /// Extract the integer part, truncating **towards zero**.
    ///
    /// `Q16::from_f32(-1.5).to_i32() == -1`. The previous arithmetic-shift
    /// implementation floored instead, so every negative non-integral value
    /// came back one too low.
    #[must_use]
    pub fn to_i32(self) -> i32 {
        if self.0 < 0 {
            // Negate in `i64` so `i32::MIN` cannot overflow.
            let magnitude = (-(self.0 as i64)) >> FRAC_BITS;
            sat_i32(-magnitude)
        } else {
            self.0 >> FRAC_BITS
        }
    }

    /// Saturating addition — clamps on overflow. Identical to `+`.
    #[must_use]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Q16(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction — clamps on underflow. Identical to `-`.
    #[must_use]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Q16(self.0.saturating_sub(rhs.0))
    }

    /// Saturating negation — `-Q16::MIN` clamps to [`Q16::MAX`].
    #[must_use]
    pub const fn saturating_neg(self) -> Self {
        Q16(self.0.saturating_neg())
    }

    /// Q16.16 division, truncating towards zero and saturating on overflow.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddedError::NumericalInstability`] if `rhs` is zero.
    pub fn checked_div(self, rhs: Self) -> EmbeddedResult<Self> {
        if rhs.0 == 0 {
            return Err(EmbeddedError::NumericalInstability);
        }
        // `self.0 << 16` peaks at 2^47, so the numerator cannot overflow i64.
        Ok(Q16(sat_i32(((self.0 as i64) << FRAC_BITS) / rhs.0 as i64)))
    }

    /// Absolute value (note: [`Q16::MIN`] has no positive counterpart; saturates).
    #[must_use]
    pub const fn abs(self) -> Self {
        Q16(self.0.saturating_abs())
    }

    /// Returns `true` if the value is strictly negative.
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }
}

impl Mul for Q16 {
    type Output = Self;

    /// Q16.16 multiplication through an `i64` intermediate, rounded to
    /// nearest and **clamped** back into the `i32` range.
    ///
    /// The previous `as i32` cast truncated the high bits of the `i64`
    /// intermediate, defeating exactly the overflow the intermediate existed
    /// to prevent: `300.0 * 300.0` decoded as `≈ 24464.0` instead of
    /// saturating at [`Q16::MAX`].
    fn mul(self, rhs: Self) -> Self::Output {
        Q16(sat_i32(mul_raw(self.0 as i64, rhs.0 as i64)))
    }
}

impl Add for Q16 {
    type Output = Self;

    /// Saturating addition (see the module-level overflow contract).
    fn add(self, rhs: Self) -> Self::Output {
        self.saturating_add(rhs)
    }
}

impl Sub for Q16 {
    type Output = Self;

    /// Saturating subtraction (see the module-level overflow contract).
    fn sub(self, rhs: Self) -> Self::Output {
        self.saturating_sub(rhs)
    }
}

impl Neg for Q16 {
    type Output = Self;

    /// Saturating negation (see the module-level overflow contract).
    fn neg(self) -> Self::Output {
        self.saturating_neg()
    }
}

/// Fixed-point dot product of two equal-length slices.
///
/// The raw `i64` products are accumulated **unshifted** and the single
/// `>> 16` (round-to-nearest) is applied once at the end. Shifting each
/// product before accumulating — as an earlier revision did — discarded the
/// low 16 bits of every term, and because `>>` on a negative `i64` rounds
/// towards `-inf` that loss was a systematic downward bias of up to `n` LSBs
/// rather than zero-mean rounding noise.
///
/// The accumulator saturates, and the result is clamped to
/// `[Q16::MIN, Q16::MAX]`.
///
/// # Errors
///
/// Returns [`EmbeddedError::DimensionMismatch`] when the slices differ in
/// length; the previous `zip` silently computed over the shorter one.
pub fn fixed_dot(a: &[Q16], b: &[Q16]) -> EmbeddedResult<Q16> {
    if a.len() != b.len() {
        return Err(EmbeddedError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let mut acc = 0_i64;
    for (x, y) in a.iter().zip(b.iter()) {
        acc = acc.saturating_add(x.0 as i64 * y.0 as i64);
    }
    // Anything beyond 2^47 clamps to Q16::MAX/MIN after the shift anyway;
    // clamping first keeps the rounding step free of i64 edge cases.
    let bounded = acc.clamp(-(1_i64 << 47), 1_i64 << 47);
    let shifted = if bounded >= 0 {
        (bounded + HALF_I64) >> FRAC_BITS
    } else {
        -((-bounded + HALF_I64) >> FRAC_BITS)
    };
    Ok(Q16(sat_i32(shifted)))
}

/// Fixed-point exponential with full range reduction.
///
/// Reduces `x = k*ln2 + r` with `|r| <= ln2/2`, evaluates a 6th-order Taylor
/// series on `r` in Q16.16, then applies the `2^k` factor as a shift.
///
/// # Accuracy and range
///
/// * Measured error stays within `1e-3 * e^x + 4.6e-5` over `x` in
///   `[-12, 10]`, i.e. `0.1 %` relative wherever the result is large enough
///   to carry that many bits, and within three Q16.16 quanta everywhere else.
///   Q16.16 simply cannot represent `e^-8 = 3.35e-4` to better than `2 %`
///   — that floor is the format, not the polynomial.
/// * Monotonically decreasing for negative `x` — it never rises above 1, so
///   it is safe to use directly as an SSM decay factor `A_bar`. The previous
///   `1 + x + x²/2` parabola turned back upward below `x = -1`, reaching
///   `2.5` at `x = -3` and making the recurrence explode.
/// * Saturates at [`Q16::MAX`] for `x >= ~11.09` (where `e^x` leaves the
///   Q16.16 range) and returns [`Q16::ZERO`] for `x <= ~-12.5` (where `e^x`
///   drops below half a Q16.16 quantum).
#[must_use]
pub fn fixed_exp_approx(x: Q16) -> Q16 {
    let raw = x.0 as i64;
    // k = round(x / ln2). `raw * INV_LN2_Q16` is Q32.32, so round at bit 32.
    let k = (raw * INV_LN2_Q16 + (1_i64 << 31)) >> 32;
    if k >= 16 {
        // e^x >= 2^16 > Q16::MAX
        return Q16::MAX;
    }
    if k <= -18 {
        // e^x < 2^-18, i.e. below a quarter of the Q16.16 quantum.
        return Q16::ZERO;
    }
    let r = raw - k * LN2_Q16; // |r| <= ln2/2 in Q16.16 (~22713)

    // Horner form of e^r = 1 + r(1 + r/2(1 + r/3(1 + r/4(1 + r/5(1 + r/6)))))
    let mut acc = SCALE_I64;
    let mut d = 6_i64;
    while d >= 1 {
        let prod = r * acc; // Q32.32
        let denom = d << FRAC_BITS; // d as Q16.16
        let half = denom >> 1;
        let scaled = if prod >= 0 {
            (prod + half) / denom
        } else {
            (prod - half) / denom
        };
        acc = SCALE_I64 + scaled;
        d -= 1;
    }

    let value = if k >= 0 {
        acc << k
    } else {
        let s = (-k) as u32;
        (acc + (1_i64 << (s - 1))) >> s
    };
    Q16(sat_i32(value))
}

/// Fixed-point natural logarithm.
///
/// Normalises `x = m * 2^e` with `m` in `[1, 2)` via `leading_zeros`, then
/// evaluates the odd `atanh` series `ln(m) = 2*(s + s³/3 + s⁵/5 + s⁷/7 + s⁹/9)`
/// with `s = (m - 1) / (m + 1)` in `[0, 1/3]`, and finally adds `e * ln2`.
///
/// # Accuracy
///
/// Absolute error below `1e-4` over `x` in `[1e-3, 1e4]` (unit-tested); the
/// dominant term is the Q16.16 quantisation of the input itself.
///
/// # Errors
///
/// Returns [`EmbeddedError::NumericalInstability`] for `x <= 0`. Q16.16 has no
/// `-inf` encoding, so an error is the only honest answer.
pub fn fixed_ln_approx(x: Q16) -> EmbeddedResult<Q16> {
    if x.0 <= 0 {
        return Err(EmbeddedError::NumericalInstability);
    }
    let raw = x.0;
    let msb = 31 - raw.leading_zeros() as i32;
    let shift = msb - FRAC_BITS; // x = m * 2^shift, m in [1, 2)
    let m: i64 = if shift > 0 {
        ((raw as i64) + (1_i64 << (shift - 1))) >> shift
    } else {
        (raw as i64) << (-shift)
    };

    // s = (m - 1) / (m + 1), round-to-nearest; numerator is non-negative.
    let denom = m + SCALE_I64;
    let s = (((m - SCALE_I64) << FRAC_BITS) + (denom >> 1)) / denom;
    let s2 = mul_raw(s, s);

    let mut p = INV9_Q16;
    p = INV7_Q16 + mul_raw(s2, p);
    p = INV5_Q16 + mul_raw(s2, p);
    p = INV3_Q16 + mul_raw(s2, p);
    p = SCALE_I64 + mul_raw(s2, p);
    let ln_m = 2 * mul_raw(s, p);

    Ok(Q16(sat_i32(ln_m + (shift as i64) * LN2_Q16)))
}

/// Fixed-point softplus: `ln(1 + e^x)`.
///
/// Evaluated as `max(x, 0) + ln(1 + e^-|x|)`, so the exponential argument is
/// never positive and the logarithm argument always lies in `(1, 2]` — the
/// range where both Q16 kernels are most accurate. The result is always
/// finite and non-negative, and saturates to `x` for large positive `x`.
#[must_use]
pub fn fixed_softplus(x: Q16) -> Q16 {
    let neg_abs = x.abs().saturating_neg();
    let exp_term = fixed_exp_approx(neg_abs);
    let arg = Q16::ONE.saturating_add(exp_term);
    // `arg >= Q16::ONE > 0`, so the logarithm cannot fail; the match keeps
    // the code total without an unwrap.
    let log_term = match fixed_ln_approx(arg) {
        Ok(v) => v,
        Err(_) => Q16::ZERO,
    };
    let base = if x.is_negative() { Q16::ZERO } else { x };
    base.saturating_add(log_term)
}

/// Fixed-point sigmoid: `1 / (1 + e^-x)`.
#[must_use]
pub fn fixed_sigmoid(x: Q16) -> Q16 {
    let exp_term = fixed_exp_approx(x.saturating_neg());
    let denom = Q16::ONE.saturating_add(exp_term);
    // `denom >= Q16::ONE`, so the division cannot fail.
    match Q16::ONE.checked_div(denom) {
        Ok(v) => v,
        Err(_) => Q16::ZERO,
    }
}

/// Fixed-point SiLU / Swish: `x * sigmoid(x)`.
#[must_use]
pub fn fixed_silu(x: Q16) -> Q16 {
    x * fixed_sigmoid(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q16_roundtrip() {
        for original in [0.5_f32, -0.5, 0.0, 1.25, -3.75, 1000.5] {
            let recovered = Q16::from_f32(original).to_f32();
            assert!(
                (recovered - original).abs() < 1e-4,
                "round-trip: from_f32({original}).to_f32() = {recovered}"
            );
        }
    }

    #[test]
    fn test_q16_from_f32_saturates_and_handles_nan() {
        assert_eq!(Q16::from_f32(1.0e9), Q16::MAX, "huge f32 must saturate");
        assert_eq!(
            Q16::from_f32(-1.0e9),
            Q16::MIN,
            "huge negative must saturate"
        );
        assert_eq!(Q16::from_f32(f32::NAN), Q16::ZERO, "NaN must map to zero");
        assert_eq!(Q16::from_f32(f32::INFINITY), Q16::MAX);
    }

    #[test]
    fn test_q16_mul() {
        let product = Q16::from_f32(2.0) * Q16::from_f32(3.0);
        assert!(
            (product.to_f32() - 6.0).abs() < 1e-3,
            "Q16(2.0) * Q16(3.0) = {}, expected ~6.0",
            product.to_f32()
        );
        // Sign handling.
        assert!(((Q16::from_f32(-2.0) * Q16::from_f32(3.0)).to_f32() + 6.0).abs() < 1e-3);
    }

    #[test]
    fn test_q16_mul_saturates_instead_of_wrapping() {
        // 300 * 300 = 90000, far beyond Q16.16. The old `as i32` truncation
        // decoded this as ~24464.0.
        let product = Q16::from_f32(300.0) * Q16::from_f32(300.0);
        assert_eq!(product, Q16::MAX, "overflowing product must saturate");
        assert!(
            product.to_f32() > 32000.0,
            "saturated product must stay positive, got {}",
            product.to_f32()
        );
        let neg = Q16::from_f32(-300.0) * Q16::from_f32(300.0);
        assert_eq!(neg, Q16::MIN, "overflowing negative product must saturate");
    }

    #[test]
    fn test_q16_add_sub_saturate_instead_of_wrapping() {
        // The operator form used to wrap: 32767.0 + 1.0 became ~-32768.0.
        let big = Q16::from_f32(32767.0);
        assert_eq!(big + Q16::ONE, Q16::MAX, "operator + must saturate");
        assert!(
            (big + Q16::ONE).to_f32() > 0.0,
            "saturating add must not flip sign"
        );
        let small = Q16::from_f32(-32767.0);
        assert_eq!(small - Q16::ONE, Q16::MIN, "operator - must saturate");
        assert!(
            (small - Q16::ONE).to_f32() < 0.0,
            "saturating sub must not flip sign"
        );
        // Explicit methods agree with the operators.
        assert_eq!(big + Q16::ONE, big.saturating_add(Q16::ONE));
        assert_eq!(small - Q16::ONE, small.saturating_sub(Q16::ONE));
    }

    #[test]
    fn test_q16_from_i32_saturates() {
        assert_eq!(Q16::from_i32(3).to_f32(), 3.0);
        assert_eq!(Q16::from_i32(-3).to_f32(), -3.0);
        // 40000 << 16 used to wrap to a negative value.
        assert_eq!(Q16::from_i32(40_000), Q16::MAX);
        assert_eq!(Q16::from_i32(-40_000), Q16::MIN);
        assert_eq!(Q16::from_i32(i32::MAX), Q16::MAX);
    }

    #[test]
    fn test_q16_to_i32_truncates_towards_zero() {
        assert_eq!(Q16::from_f32(1.5).to_i32(), 1);
        assert_eq!(Q16::from_f32(-1.5).to_i32(), -1, "must truncate, not floor");
        assert_eq!(Q16::from_f32(-0.5).to_i32(), 0);
        assert_eq!(Q16::from_f32(0.5).to_i32(), 0);
        assert_eq!(Q16::from_f32(-2.0).to_i32(), -2);
        assert_eq!(Q16::from_i32(7).to_i32(), 7);
        // i32::MIN is exactly -32768.0, so truncation and floor agree there.
        assert_eq!(Q16::MIN.to_i32(), -32768);
    }

    #[test]
    fn test_q16_checked_div() {
        let q = Q16::from_f32(3.0)
            .checked_div(Q16::from_f32(2.0))
            .expect("non-zero divisor");
        assert!(
            (q.to_f32() - 1.5).abs() < 1e-4,
            "3 / 2 = {}, expected 1.5",
            q.to_f32()
        );
        let neg = Q16::from_f32(-1.0)
            .checked_div(Q16::from_f32(4.0))
            .expect("non-zero divisor");
        assert!((neg.to_f32() + 0.25).abs() < 1e-4);
    }

    #[test]
    fn test_q16_checked_div_by_zero() {
        assert_eq!(
            Q16::from_f32(1.0).checked_div(Q16::ZERO),
            Err(EmbeddedError::NumericalInstability),
            "division by zero should return NumericalInstability"
        );
    }

    #[test]
    fn test_q16_checked_div_saturates() {
        // 30000 / 0.0001 overflows Q16.16 and used to wrap via `as i32`.
        let q = Q16::from_f32(30000.0)
            .checked_div(Q16::from_raw(1))
            .expect("non-zero divisor");
        assert_eq!(q, Q16::MAX, "overflowing quotient must saturate");
    }

    #[test]
    fn test_q16_saturating_helpers() {
        assert_eq!(Q16::MAX.saturating_add(Q16::ONE), Q16::MAX);
        assert_eq!(Q16::MIN.saturating_sub(Q16::ONE), Q16::MIN);
        assert_eq!(Q16::MIN.saturating_neg(), Q16::MAX);
        assert_eq!(-Q16::ONE, Q16::from_f32(-1.0));
        assert_eq!(Q16::MIN.abs(), Q16::MAX);
        assert_eq!(Q16::from_f32(-3.0).abs(), Q16::from_f32(3.0));
        assert_eq!(Q16::ZERO.abs(), Q16::ZERO);
    }

    #[test]
    fn test_q16_is_negative() {
        assert!(Q16::from_f32(-0.001).is_negative());
        assert!(!Q16::ZERO.is_negative());
        assert!(!Q16::ONE.is_negative());
    }

    #[test]
    fn test_q16_raw_roundtrip() {
        let q = Q16::from_raw(12345);
        assert_eq!(q.to_raw(), 12345);
        assert_eq!(Q16::ONE.to_raw(), 65536);
    }

    #[test]
    fn test_fixed_dot() {
        let a = [Q16::from_f32(1.0), Q16::from_f32(2.0), Q16::from_f32(3.0)];
        let b = [Q16::from_f32(4.0), Q16::from_f32(5.0), Q16::from_f32(6.0)];
        let value = fixed_dot(&a, &b).expect("equal lengths").to_f32();
        assert!(
            (value - 32.0).abs() < 1e-3,
            "fixed_dot([1,2,3],[4,5,6]) = {value}, expected ~32.0"
        );
    }

    #[test]
    fn test_fixed_dot_rejects_length_mismatch() {
        let a = [Q16::ONE; 3];
        let b = [Q16::ONE; 2];
        assert_eq!(
            fixed_dot(&a, &b),
            Err(EmbeddedError::DimensionMismatch {
                expected: 3,
                got: 2
            }),
            "unequal lengths must error instead of truncating"
        );
    }

    #[test]
    fn test_fixed_dot_has_no_downward_bias() {
        // 64 identical small products. Shifting each term before accumulating
        // truncated towards -inf every time, so the negative case drifted low
        // by up to n LSBs. Both signs must now agree in magnitude.
        let a = [Q16::from_raw(3_000); 64];
        let b_pos = [Q16::from_raw(3_000); 64];
        let b_neg = [Q16::from_raw(-3_000); 64];
        let pos = fixed_dot(&a, &b_pos).expect("equal lengths").to_raw();
        let neg = fixed_dot(&a, &b_neg).expect("equal lengths").to_raw();
        assert_eq!(pos, -neg, "positive and negative sums must be symmetric");

        // And the value must match the exact rational answer within 1 LSB.
        let exact = 64_i64 * 3_000 * 3_000 / 65_536;
        assert!(
            (pos as i64 - exact).abs() <= 1,
            "fixed_dot = {pos}, exact = {exact}"
        );
    }

    #[test]
    fn test_fixed_dot_saturates() {
        let a = [Q16::MAX; 8];
        let b = [Q16::MAX; 8];
        assert_eq!(fixed_dot(&a, &b), Ok(Q16::MAX), "huge sums must saturate");
        let c = [Q16::MIN; 8];
        assert_eq!(fixed_dot(&a, &c), Ok(Q16::MIN));
    }

    #[test]
    fn test_fixed_exp_approx_near_zero() {
        assert_eq!(
            fixed_exp_approx(Q16::ZERO),
            Q16::ONE,
            "fixed_exp_approx(0) should equal Q16::ONE"
        );
    }

    #[test]
    fn test_fixed_exp_approx_accuracy_sweep() {
        // Documented contract: |error| <= 1e-3 * e^x + 3 LSB. The old
        // 2nd-order Taylor was 3 % off at x = -0.5 while claiming ~1 %, and
        // diverged completely below x = -1.
        const LSB: f64 = 1.0 / 65_536.0;
        for i in -1200_i32..=1000 {
            let x = i as f64 / 100.0;
            let got = fixed_exp_approx(Q16::from_f32(x as f32)).to_f32() as f64;
            let want = x.exp();
            let tol = 1e-3 * want + 3.0 * LSB;
            assert!(
                (got - want).abs() <= tol,
                "fixed_exp_approx({x}) = {got}, expected {want} (tol {tol:.3e})"
            );
        }
        // Explicit spot checks at the historic failure points.
        for x in [-0.5_f32, -0.4, -1.0, -2.0, -3.0] {
            let got = fixed_exp_approx(Q16::from_f32(x)).to_f32() as f64;
            let want = (x as f64).exp();
            let rel = (got - want).abs() / want;
            assert!(
                rel < 1e-3,
                "fixed_exp_approx({x}) = {got}, expected {want}, rel = {rel:.3e}"
            );
        }
    }

    #[test]
    fn test_fixed_exp_approx_is_monotone_and_bounded_for_negatives() {
        // The old parabola turned upward: exp(-1) -> 0.5, exp(-2) -> 1.0,
        // exp(-3) -> 2.5, which makes an SSM decay factor greater than one.
        let mut prev = f32::INFINITY;
        for i in 0..=1200 {
            let x = -(i as f32) / 100.0;
            let got = fixed_exp_approx(Q16::from_f32(x)).to_f32();
            assert!(
                got <= 1.0 + 1e-4,
                "exp({x}) = {got} must never exceed 1 for negative x"
            );
            assert!(got >= 0.0, "exp({x}) = {got} must stay non-negative");
            assert!(
                got <= prev + 2.0 / 65_536.0,
                "exp must be non-increasing: exp({x}) = {got} > previous {prev}"
            );
            prev = got;
        }
        assert_eq!(
            fixed_exp_approx(Q16::from_f32(-20.0)),
            Q16::ZERO,
            "far-negative arguments underflow to zero"
        );
    }

    #[test]
    fn test_fixed_exp_approx_saturates_on_overflow() {
        assert_eq!(fixed_exp_approx(Q16::from_f32(20.0)), Q16::MAX);
        assert_eq!(fixed_exp_approx(Q16::MAX), Q16::MAX);
        assert_eq!(fixed_exp_approx(Q16::MIN), Q16::ZERO);
    }

    #[test]
    fn test_fixed_ln_approx_accuracy() {
        let mut worst = 0.0_f64;
        let mut worst_at = 0.0_f64;
        let mut x = 0.001_f64;
        while x < 10_000.0 {
            let q = Q16::from_f32(x as f32);
            let got = fixed_ln_approx(q).expect("positive input").to_f32() as f64;
            // Compare against ln of the *quantised* input: the Q16 grid is the
            // real precision floor here.
            let want = (q.to_f32() as f64).ln();
            let err = (got - want).abs();
            if err > worst {
                worst = err;
                worst_at = x;
            }
            x *= 1.01;
        }
        assert!(
            worst < 1e-4,
            "fixed_ln_approx worst absolute error {worst:.3e} at x = {worst_at}"
        );
    }

    #[test]
    fn test_fixed_ln_approx_known_points() {
        for (x, want) in [
            (1.0_f32, 0.0_f32),
            (2.0, core::f32::consts::LN_2),
            (core::f32::consts::E, 1.0),
            (0.5, -core::f32::consts::LN_2),
        ] {
            let got = fixed_ln_approx(Q16::from_f32(x))
                .expect("positive input")
                .to_f32();
            assert!(
                (got - want).abs() < 1e-3,
                "fixed_ln_approx({x}) = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn test_fixed_ln_approx_rejects_non_positive() {
        assert_eq!(
            fixed_ln_approx(Q16::ZERO),
            Err(EmbeddedError::NumericalInstability)
        );
        assert_eq!(
            fixed_ln_approx(Q16::from_f32(-1.0)),
            Err(EmbeddedError::NumericalInstability)
        );
    }

    #[test]
    fn test_fixed_exp_ln_roundtrip() {
        for x in [0.25_f32, 1.0, 2.5, 5.0, 9.0] {
            let e = fixed_exp_approx(Q16::from_f32(x));
            let back = fixed_ln_approx(e).expect("exp is positive").to_f32();
            assert!(
                (back - x).abs() < 5e-3,
                "ln(exp({x})) = {back}, expected {x}"
            );
        }
    }

    #[test]
    fn test_fixed_softplus_matches_f32() {
        for x in [-8.0_f32, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0] {
            let got = fixed_softplus(Q16::from_f32(x)).to_f32();
            let want = (1.0_f64 + (x as f64).exp()).ln() as f32;
            assert!(
                (got - want).abs() < 2e-3,
                "fixed_softplus({x}) = {got}, expected {want}"
            );
        }
        // Always non-negative, never NaN.
        assert!(fixed_softplus(Q16::MIN).to_f32() >= 0.0);
        assert!(fixed_softplus(Q16::from_f32(100.0)).to_f32() > 99.0);
    }

    #[test]
    fn test_fixed_sigmoid_and_silu() {
        for x in [-6.0_f32, -1.0, 0.0, 1.0, 6.0] {
            let got = fixed_sigmoid(Q16::from_f32(x)).to_f32();
            let want = (1.0 / (1.0 + (-(x as f64)).exp())) as f32;
            assert!(
                (got - want).abs() < 2e-3,
                "fixed_sigmoid({x}) = {got}, expected {want}"
            );
            let silu_got = fixed_silu(Q16::from_f32(x)).to_f32();
            let silu_want = x * want;
            assert!(
                (silu_got - silu_want).abs() < 5e-3,
                "fixed_silu({x}) = {silu_got}, expected {silu_want}"
            );
        }
        assert!(
            (fixed_sigmoid(Q16::ZERO).to_f32() - 0.5).abs() < 1e-4,
            "sigmoid(0) must be 0.5"
        );
    }
}
