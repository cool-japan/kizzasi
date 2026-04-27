//! Fixed-point arithmetic: Q16.16 format.
//!
//! The Q16.16 format uses a 32-bit signed integer where the upper 16 bits
//! are the integer part and the lower 16 bits are the fractional part.
//! This is suitable for microcontrollers without an FPU.
//!
//! Representable range: approximately [-32768.0, 32767.99998].
//! Precision: 1 / 65536 ≈ 1.5e-5.

use core::ops::{Add, Mul, Sub};

use crate::error::{EmbeddedError, EmbeddedResult};

const FRAC_BITS: i32 = 16;
const SCALE: i32 = 1 << FRAC_BITS; // 65536

/// Q16.16 fixed-point number (16 integer bits, 16 fractional bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Q16(i32);

impl Q16 {
    /// The value 0.0 in Q16.16
    pub const ZERO: Self = Q16(0);
    /// The value 1.0 in Q16.16
    pub const ONE: Self = Q16(SCALE);

    /// Convert from f32 to Q16.16 (truncates, no overflow check).
    pub fn from_f32(v: f32) -> Self {
        Q16((v * SCALE as f32) as i32)
    }

    /// Convert from Q16.16 to f32.
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / SCALE as f32
    }

    /// Convert from i32 integer to Q16.16 (shift left by FRAC_BITS).
    pub fn from_i32(v: i32) -> Self {
        Q16(v << FRAC_BITS)
    }

    /// Extract the integer part (round towards zero).
    pub fn to_i32(self) -> i32 {
        self.0 >> FRAC_BITS
    }

    /// Saturating addition — clamps on overflow.
    pub fn saturating_add(self, rhs: Self) -> Self {
        Q16(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction — clamps on underflow.
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Q16(self.0.saturating_sub(rhs.0))
    }

    /// Q16.16 division. Returns `Err(NumericalInstability)` if `rhs` is zero.
    pub fn checked_div(self, rhs: Self) -> EmbeddedResult<Self> {
        if rhs.0 == 0 {
            return Err(EmbeddedError::NumericalInstability);
        }
        Ok(Q16(((self.0 as i64 * SCALE as i64) / rhs.0 as i64) as i32))
    }

    /// Absolute value (note: Q16::MIN has no positive counterpart; saturates).
    pub fn abs(self) -> Self {
        if self.0 == i32::MIN {
            Q16(i32::MAX) // saturate
        } else {
            Q16(self.0.abs())
        }
    }

    /// Returns `true` if the value is strictly negative.
    pub fn is_negative(self) -> bool {
        self.0 < 0
    }
}

impl Mul for Q16 {
    type Output = Self;

    /// Q16.16 multiplication using i64 intermediate to avoid overflow.
    fn mul(self, rhs: Self) -> Self::Output {
        Q16(((self.0 as i64 * rhs.0 as i64) >> FRAC_BITS) as i32)
    }
}

impl Add for Q16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Q16(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for Q16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Q16(self.0.wrapping_sub(rhs.0))
    }
}

/// Fixed-point dot product of two equal-length slices.
///
/// Uses i64 accumulator to reduce intermediate overflow risk, then
/// clamps the result to the i32 range.
pub fn fixed_dot(a: &[Q16], b: &[Q16]) -> Q16 {
    let mut acc = 0_i64;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += (x.0 as i64 * y.0 as i64) >> FRAC_BITS;
    }
    Q16(acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
}

/// Fixed-point exponential approximation using 2nd-order Taylor series.
///
/// Approximation: e^x ≈ 1 + x + x²/2
///
/// Accurate to within ~1% for |x| < 0.5. For larger arguments the
/// error grows substantially — this is intended for near-zero step deltas
/// on embedded targets where a full `exp` is too expensive.
pub fn fixed_exp_approx(x: Q16) -> Q16 {
    // e^x ≈ 1 + x + x²/2
    let x_sq = x * x;
    let x_sq_half = Q16(x_sq.0 >> 1); // divide by 2
    Q16::ONE.saturating_add(x).saturating_add(x_sq_half)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q16_roundtrip() {
        let original = 0.5_f32;
        let q = Q16::from_f32(original);
        let recovered = q.to_f32();
        assert!(
            (recovered - original).abs() < 1e-4,
            "round-trip: from_f32(0.5).to_f32() = {recovered}, expected ~0.5"
        );
    }

    #[test]
    fn test_q16_mul() {
        let a = Q16::from_f32(2.0);
        let b = Q16::from_f32(3.0);
        let product = a * b;
        let result = product.to_f32();
        assert!(
            (result - 6.0).abs() < 1e-3,
            "Q16(2.0) * Q16(3.0) = {result}, expected ~6.0"
        );
    }

    #[test]
    fn test_q16_checked_div_by_zero() {
        let a = Q16::from_f32(1.0);
        let result = a.checked_div(Q16::ZERO);
        assert!(
            matches!(result, Err(EmbeddedError::NumericalInstability)),
            "division by zero should return NumericalInstability"
        );
    }

    #[test]
    fn test_q16_saturating_add() {
        let big = Q16(i32::MAX);
        let one = Q16::ONE;
        let result = big.saturating_add(one);
        assert_eq!(
            result.0,
            i32::MAX,
            "saturating_add should clamp at i32::MAX"
        );
    }

    #[test]
    fn test_fixed_dot() {
        let a = [Q16::from_f32(1.0), Q16::from_f32(2.0), Q16::from_f32(3.0)];
        let b = [Q16::from_f32(4.0), Q16::from_f32(5.0), Q16::from_f32(6.0)];
        let result = fixed_dot(&a, &b);
        let value = result.to_f32();
        // [1,2,3]·[4,5,6] = 4 + 10 + 18 = 32
        assert!(
            (value - 32.0).abs() < 0.1,
            "fixed_dot([1,2,3],[4,5,6]) = {value}, expected ~32.0"
        );
    }

    #[test]
    fn test_fixed_exp_approx_near_zero() {
        // e^0 = 1
        let result = fixed_exp_approx(Q16::ZERO);
        assert_eq!(
            result,
            Q16::ONE,
            "fixed_exp_approx(0) should equal Q16::ONE"
        );
    }

    #[test]
    fn test_fixed_exp_approx_small() {
        // e^0.1 ≈ 1.105; 2nd-order Taylor gives 1 + 0.1 + 0.005 = 1.105
        let result = fixed_exp_approx(Q16::from_f32(0.1));
        let value = result.to_f32();
        assert!(
            (value - 1.105).abs() < 0.01,
            "fixed_exp_approx(0.1) = {value}, expected ~1.105"
        );
    }

    #[test]
    fn test_q16_abs() {
        assert_eq!(Q16::from_f32(-3.0).abs(), Q16::from_f32(3.0));
        assert_eq!(Q16::ZERO.abs(), Q16::ZERO);
    }

    #[test]
    fn test_q16_is_negative() {
        assert!(Q16::from_f32(-0.001).is_negative());
        assert!(!Q16::ZERO.is_negative());
        assert!(!Q16::ONE.is_negative());
    }
}
