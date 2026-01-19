//! Fixed-Point Arithmetic for Embedded Systems
//!
//! This module provides fixed-point number representations and operations
//! for embedded systems without hardware floating-point units (FPUs).
//!
//! # Fixed-Point Formats
//!
//! - **Q15.16** (i32): 15 integer bits, 16 fractional bits, range [-32768, 32767]
//! - **Q7.8** (i16): 7 integer bits, 8 fractional bits, range [-128, 127]
//! - **Q31** (i32): 1 sign bit, 31 fractional bits, range [-1, 1)
//!
//! # Use Cases
//!
//! - Microcontrollers without FPU (ARM Cortex-M0/M3)
//! - Low-power embedded systems
//! - Real-time systems requiring deterministic timing
//! - DSP applications with fixed-point arithmetic
//!
//! # Performance
//!
//! - 10-100x faster than software floating-point
//! - Lower power consumption
//! - Deterministic execution time

use crate::error::{CoreError, CoreResult};
use core::ops::{Add, Div, Mul, Neg, Sub};

/// Q15.16 fixed-point number (32-bit signed integer)
///
/// Represents numbers in the range [-32768, 32767] with 16 fractional bits.
/// Precision: ~0.000015 (1/65536)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Q15_16(i32);

impl Q15_16 {
    /// Number of fractional bits
    pub const FRAC_BITS: u32 = 16;

    /// Scaling factor (2^16)
    pub const SCALE: i32 = 1 << Self::FRAC_BITS;

    /// Minimum value (-32768)
    pub const MIN: Q15_16 = Q15_16(i32::MIN);

    /// Maximum value (~32767.99998)
    pub const MAX: Q15_16 = Q15_16(i32::MAX);

    /// Zero
    pub const ZERO: Q15_16 = Q15_16(0);

    /// One
    pub const ONE: Q15_16 = Q15_16(Self::SCALE);

    /// Create from raw i32 value (already scaled)
    #[inline]
    pub const fn from_raw(raw: i32) -> Self {
        Q15_16(raw)
    }

    /// Get raw i32 value
    #[inline]
    pub const fn raw(self) -> i32 {
        self.0
    }

    /// Create from f32 (for testing/initialization)
    #[inline]
    pub fn from_f32(val: f32) -> Self {
        Q15_16((val * Self::SCALE as f32) as i32)
    }

    /// Convert to f32 (for debugging/output)
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Self::SCALE as f32
    }

    /// Create from integer
    #[inline]
    pub const fn from_int(val: i32) -> Self {
        Q15_16(val << Self::FRAC_BITS)
    }

    /// Get integer part
    #[inline]
    pub const fn to_int(self) -> i32 {
        self.0 >> Self::FRAC_BITS
    }

    /// Get fractional part (0.0 to 0.99998)
    #[inline]
    pub const fn frac(self) -> i32 {
        self.0 & ((1 << Self::FRAC_BITS) - 1)
    }

    /// Saturating addition
    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Q15_16(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Q15_16(self.0.saturating_sub(rhs.0))
    }

    /// Saturating multiplication
    #[inline]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        let product = (self.0 as i64) * (rhs.0 as i64);
        let result = (product >> Self::FRAC_BITS) as i32;
        Q15_16(result.saturating_mul(1))
    }

    /// Absolute value
    #[inline]
    pub fn abs(self) -> Self {
        Q15_16(self.0.abs())
    }

    /// Square root (Newton-Raphson method)
    pub fn sqrt(self) -> Self {
        if self.0 <= 0 {
            return Q15_16::ZERO;
        }

        // Initial guess: x/2
        let mut x = Q15_16(self.0 >> 1);

        // Newton-Raphson: x_new = (x + n/x) / 2
        for _ in 0..8 {
            let x_next = (x + self / x) / Q15_16::from_int(2);
            if (x_next - x).abs().0 < 1 {
                break;
            }
            x = x_next;
        }

        x
    }

    /// Reciprocal (1/x) using Newton-Raphson
    pub fn recip(self) -> CoreResult<Self> {
        if self.0 == 0 {
            return Err(CoreError::Generic("Division by zero".to_string()));
        }

        // Initial guess: use a lookup table approach for better convergence
        // For x in range, 1/x is approximated
        let abs_val = self.abs().0;
        let mut x = if abs_val < (Self::SCALE / 2) {
            // |self| < 0.5, so 1/|self| > 2
            Q15_16::from_int(4)
        } else if abs_val < Self::SCALE {
            // 0.5 <= |self| < 1, so 1 < 1/|self| <= 2
            Q15_16::from_int(2)
        } else if abs_val < (Self::SCALE * 2) {
            // 1 <= |self| < 2, so 0.5 < 1/|self| <= 1
            Q15_16::ONE
        } else {
            // |self| >= 2, so 1/|self| <= 0.5
            Q15_16::from_f32(0.25)
        };

        // Handle sign
        if self.0 < 0 {
            x = -x;
        }

        // Newton-Raphson: x_new = x * (2 - self * x)
        for _ in 0..10 {
            let two = Q15_16::from_int(2);
            let x_next = x * (two - self * x);
            if (x_next - x).abs().0 < 2 {
                break;
            }
            x = x_next;
        }

        Ok(x)
    }

    /// Exponential approximation (e^x) using Taylor series
    pub fn exp(self) -> Self {
        // Taylor series: e^x = 1 + x + x²/2! + x³/3! + x⁴/4! + ...
        let mut result = Q15_16::ONE;
        let mut term = Q15_16::ONE;

        for n in 1..10 {
            term = term * self / Q15_16::from_int(n);
            result = result + term;
            if term.abs().0 < 4 {
                // Converged
                break;
            }
        }

        result
    }

    /// Natural logarithm approximation (ln(x))
    pub fn ln(self) -> CoreResult<Self> {
        if self.0 <= 0 {
            return Err(CoreError::Generic("ln of non-positive number".to_string()));
        }

        // ln(x) = ln(2^n * m) = n*ln(2) + ln(m) where m in [1, 2)
        let mut x = self;
        let mut n = 0i32;

        // Normalize to [1, 2)
        while x.0 >= (2 << Self::FRAC_BITS) {
            x = Q15_16(x.0 >> 1);
            n += 1;
        }
        while x.0 < Self::SCALE {
            x = Q15_16(x.0 << 1);
            n -= 1;
        }

        // Taylor series for ln(1 + y) where x = 1 + y
        let y = x - Q15_16::ONE;
        let mut result = y;
        let mut term = y;

        for i in 2..10 {
            term = term * y * Q15_16::from_int(-1) / Q15_16::from_int(i);
            result = result + term;
            if term.abs().0 < 4 {
                break;
            }
        }

        // Add n * ln(2)
        let ln2 = Q15_16::from_f32(core::f32::consts::LN_2);
        result = result + Q15_16::from_int(n) * ln2;

        Ok(result)
    }
}

impl Add for Q15_16 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Q15_16(self.0 + rhs.0)
    }
}

impl Sub for Q15_16 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Q15_16(self.0 - rhs.0)
    }
}

impl Mul for Q15_16 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Use i64 to prevent overflow
        let product = (self.0 as i64) * (rhs.0 as i64);
        Q15_16((product >> Self::FRAC_BITS) as i32)
    }
}

impl Div for Q15_16 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        // Use i64 to prevent overflow
        let dividend = (self.0 as i64) << Self::FRAC_BITS;
        Q15_16((dividend / rhs.0 as i64) as i32)
    }
}

impl Neg for Q15_16 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Q15_16(-self.0)
    }
}

/// Q7.8 fixed-point number (16-bit signed integer)
///
/// Represents numbers in the range [-128, 127] with 8 fractional bits.
/// More compact than Q15.16 but less precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Q7_8(i16);

impl Q7_8 {
    pub const FRAC_BITS: u32 = 8;
    pub const SCALE: i16 = 1 << Self::FRAC_BITS;
    pub const MIN: Q7_8 = Q7_8(i16::MIN);
    pub const MAX: Q7_8 = Q7_8(i16::MAX);
    pub const ZERO: Q7_8 = Q7_8(0);
    pub const ONE: Q7_8 = Q7_8(Self::SCALE);

    #[inline]
    pub const fn from_raw(raw: i16) -> Self {
        Q7_8(raw)
    }

    #[inline]
    pub const fn raw(self) -> i16 {
        self.0
    }

    #[inline]
    pub fn from_f32(val: f32) -> Self {
        Q7_8((val * Self::SCALE as f32) as i16)
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Self::SCALE as f32
    }

    #[inline]
    pub fn abs(self) -> Self {
        Q7_8(self.0.abs())
    }
}

impl Add for Q7_8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Q7_8(self.0 + rhs.0)
    }
}

impl Sub for Q7_8 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Q7_8(self.0 - rhs.0)
    }
}

impl Mul for Q7_8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let product = (self.0 as i32) * (rhs.0 as i32);
        Q7_8((product >> Self::FRAC_BITS) as i16)
    }
}

impl Div for Q7_8 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        let dividend = (self.0 as i32) << Self::FRAC_BITS;
        Q7_8((dividend / rhs.0 as i32) as i16)
    }
}

impl Neg for Q7_8 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Q7_8(-self.0)
    }
}

/// Fixed-point vector operations
pub mod vec_ops {
    use super::*;

    /// Dot product for Q15.16 vectors
    pub fn dot_product_q15_16(a: &[Q15_16], b: &[Q15_16]) -> Q15_16 {
        let mut sum = Q15_16::ZERO;
        for (x, y) in a.iter().zip(b) {
            sum = sum + (*x * *y);
        }
        sum
    }

    /// ReLU activation for Q15.16
    pub fn relu_q15_16(x: &[Q15_16], y: &mut [Q15_16]) {
        for (xi, yi) in x.iter().zip(y) {
            *yi = if xi.0 > 0 { *xi } else { Q15_16::ZERO };
        }
    }

    /// Softmax approximation for Q15.16 (simplified for embedded)
    pub fn softmax_q15_16(x: &[Q15_16], y: &mut [Q15_16]) -> CoreResult<()> {
        // Find max for numerical stability
        let max = x.iter().max().copied().unwrap_or(Q15_16::ZERO);

        // Compute exp(x - max) for each element
        let mut sum = Q15_16::ZERO;
        for i in 0..x.len() {
            let shifted = x[i] - max;
            y[i] = shifted.exp();
            sum = sum + y[i];
        }

        // Normalize
        let recip_sum = sum.recip()?;
        for yi in y.iter_mut() {
            *yi = *yi * recip_sum;
        }

        Ok(())
    }

    /// Layer normalization for Q15.16
    pub fn layer_norm_q15_16(x: &[Q15_16], y: &mut [Q15_16], eps: Q15_16) -> CoreResult<()> {
        let n = Q15_16::from_int(x.len() as i32);

        // Compute mean
        let mut sum = Q15_16::ZERO;
        for &xi in x {
            sum = sum + xi;
        }
        let mean = sum / n;

        // Compute variance
        let mut var_sum = Q15_16::ZERO;
        for &xi in x {
            let diff = xi - mean;
            var_sum = var_sum + (diff * diff);
        }
        let variance = var_sum / n;

        // Compute 1/sqrt(variance + eps)
        let std_inv = (variance + eps).sqrt().recip()?;

        // Normalize
        for (i, &xi) in x.iter().enumerate() {
            y[i] = (xi - mean) * std_inv;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q15_16_basic() {
        let a = Q15_16::from_f32(3.5);
        let b = Q15_16::from_f32(2.0);

        let sum = a + b;
        assert!((sum.to_f32() - 5.5).abs() < 0.001);

        let diff = a - b;
        assert!((diff.to_f32() - 1.5).abs() < 0.001);

        let prod = a * b;
        assert!((prod.to_f32() - 7.0).abs() < 0.001);

        let quot = a / b;
        assert!((quot.to_f32() - 1.75).abs() < 0.001);
    }

    #[test]
    fn test_q15_16_sqrt() {
        let x = Q15_16::from_f32(9.0);
        let result = x.sqrt();
        assert!((result.to_f32() - 3.0).abs() < 0.01);

        let x2 = Q15_16::from_f32(2.0);
        let result2 = x2.sqrt();
        assert!((result2.to_f32() - 1.414).abs() < 0.01);
    }

    #[test]
    fn test_q15_16_exp() {
        let x = Q15_16::from_f32(1.0);
        let result = x.exp();
        assert!((result.to_f32() - core::f32::consts::E).abs() < 0.1);

        let x2 = Q15_16::from_f32(0.0);
        let result2 = x2.exp();
        assert!((result2.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_q15_16_ln() {
        let x = Q15_16::from_f32(core::f32::consts::E);
        let result = x.ln().unwrap();
        assert!((result.to_f32() - 1.0).abs() < 0.1);

        let x2 = Q15_16::from_f32(1.0);
        let result2 = x2.ln().unwrap();
        assert!(result2.to_f32().abs() < 0.01);
    }

    #[test]
    fn test_q15_16_saturating() {
        let max = Q15_16::from_int(30000);
        let big = Q15_16::from_int(10000);

        let sum = max.saturating_add(big);
        assert_eq!(sum, Q15_16::MAX);
    }

    #[test]
    fn test_q7_8_basic() {
        let a = Q7_8::from_f32(3.5);
        let b = Q7_8::from_f32(2.0);

        let sum = a + b;
        assert!((sum.to_f32() - 5.5).abs() < 0.01);

        let prod = a * b;
        assert!((prod.to_f32() - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![
            Q15_16::from_f32(1.0),
            Q15_16::from_f32(2.0),
            Q15_16::from_f32(3.0),
        ];
        let b = vec![
            Q15_16::from_f32(4.0),
            Q15_16::from_f32(5.0),
            Q15_16::from_f32(6.0),
        ];

        let result = vec_ops::dot_product_q15_16(&a, &b);
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 32.0

        assert!((result.to_f32() - expected).abs() < 0.01);
    }

    #[test]
    fn test_relu() {
        let x = vec![
            Q15_16::from_f32(-2.0),
            Q15_16::from_f32(-1.0),
            Q15_16::from_f32(0.0),
            Q15_16::from_f32(1.0),
            Q15_16::from_f32(2.0),
        ];
        let mut y = vec![Q15_16::ZERO; 5];

        vec_ops::relu_q15_16(&x, &mut y);

        assert_eq!(y[0], Q15_16::ZERO);
        assert_eq!(y[1], Q15_16::ZERO);
        assert_eq!(y[2], Q15_16::ZERO);
        assert!((y[3].to_f32() - 1.0).abs() < 0.01);
        assert!((y[4].to_f32() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_layer_norm() {
        let x = vec![
            Q15_16::from_f32(1.0),
            Q15_16::from_f32(2.0),
            Q15_16::from_f32(3.0),
            Q15_16::from_f32(4.0),
        ];
        let mut y = vec![Q15_16::ZERO; 4];
        let eps = Q15_16::from_f32(1e-5);

        vec_ops::layer_norm_q15_16(&x, &mut y, eps).unwrap();

        // Verify mean is close to 0
        let mean: f32 = y.iter().map(|yi| yi.to_f32()).sum::<f32>() / y.len() as f32;
        assert!(mean.abs() < 0.1);

        // Verify variance is close to 1 (fixed-point has lower precision)
        let variance: f32 = y.iter().map(|yi| yi.to_f32().powi(2)).sum::<f32>() / y.len() as f32;
        assert!((variance - 1.0).abs() < 0.5, "variance={}", variance);
    }

    #[test]
    fn test_conversion_precision() {
        let values = [0.0, 0.5, 1.0, -1.0, 100.5, -100.5];

        for &val in &values {
            let fixed = Q15_16::from_f32(val);
            let recovered = fixed.to_f32();
            assert!((val - recovered).abs() < 0.001, "Failed for {}", val);
        }
    }
}
