//! Digital filters for signal processing
//!
//! This module provides FIR (Finite Impulse Response) and IIR (Infinite Impulse Response)
//! digital filters for audio and signal processing applications.

use crate::error::{IoError, IoResult};
use scirs2_core::ndarray::Array1;
use std::f32::consts::PI;

/// FIR (Finite Impulse Response) filter
#[derive(Debug, Clone)]
pub struct FirFilter {
    /// Filter coefficients (impulse response)
    coeffs: Vec<f32>,
    /// Delay line
    buffer: Vec<f32>,
    /// Current position in buffer
    pos: usize,
}

impl FirFilter {
    /// Create a new FIR filter with given coefficients
    pub fn new(coeffs: Vec<f32>) -> IoResult<Self> {
        if coeffs.is_empty() {
            return Err(IoError::SignalError(
                "FIR coefficients cannot be empty".into(),
            ));
        }

        let len = coeffs.len();
        Ok(Self {
            coeffs,
            buffer: vec![0.0; len],
            pos: 0,
        })
    }

    /// Design a windowed sinc low-pass filter
    pub fn sinc_lowpass(cutoff_normalized: f32, num_taps: usize) -> IoResult<Self> {
        validate_normalized_freq(cutoff_normalized, "Normalized cutoff")?;

        if num_taps == 0 || num_taps.is_multiple_of(2) {
            return Err(IoError::SignalError(
                "Number of taps must be odd and > 0".into(),
            ));
        }

        let m = (num_taps - 1) as f32 / 2.0;
        let mut coeffs = Vec::with_capacity(num_taps);

        for i in 0..num_taps {
            let n = i as f32 - m;
            let sinc = if n.abs() < 1e-10 {
                2.0 * cutoff_normalized
            } else {
                (2.0 * PI * cutoff_normalized * n).sin() / (PI * n)
            };

            let window = 0.54 - 0.46 * (2.0 * PI * i as f32 / (num_taps - 1) as f32).cos();
            coeffs.push(sinc * window);
        }

        let sum: f32 = coeffs.iter().sum();
        if !sum.is_finite() || sum.abs() < 1e-10 {
            return Err(IoError::SignalError(
                "sinc_lowpass: coefficient sum is degenerate (too close to zero) and cannot be \
                 normalized"
                    .into(),
            ));
        }
        for c in &mut coeffs {
            *c /= sum;
        }

        Self::new(coeffs)
    }

    /// Design a windowed sinc high-pass filter
    pub fn sinc_highpass(cutoff_normalized: f32, num_taps: usize) -> IoResult<Self> {
        validate_normalized_freq(cutoff_normalized, "Normalized cutoff")?;

        if num_taps == 0 || num_taps.is_multiple_of(2) {
            return Err(IoError::SignalError(
                "Number of taps must be odd and > 0".into(),
            ));
        }

        let mut lpf = Self::sinc_lowpass(cutoff_normalized, num_taps)?;
        let center = num_taps / 2;

        for (i, c) in lpf.coeffs.iter_mut().enumerate() {
            *c = -*c;
            if i == center {
                *c += 1.0;
            }
        }

        Self::new(lpf.coeffs)
    }

    /// Design a moving average filter
    pub fn moving_average(window: usize) -> IoResult<Self> {
        if window == 0 {
            return Err(IoError::SignalError("Window size must be > 0".into()));
        }

        let coeff = 1.0 / window as f32;
        Self::new(vec![coeff; window])
    }

    /// Design a differentiation filter
    pub fn differentiator() -> IoResult<Self> {
        Self::new(vec![1.0, -1.0])
    }

    /// Process a single sample
    pub fn process_sample(&mut self, input: f32) -> f32 {
        self.buffer[self.pos] = input;
        let mut output = 0.0;
        let mut buf_idx = self.pos;

        for &coeff in &self.coeffs {
            output += coeff * self.buffer[buf_idx];
            if buf_idx == 0 {
                buf_idx = self.buffer.len() - 1;
            } else {
                buf_idx -= 1;
            }
        }

        self.pos = (self.pos + 1) % self.buffer.len();
        output
    }

    /// Process an entire signal
    pub fn process(&mut self, signal: &Array1<f32>) -> Array1<f32> {
        let mut output = Array1::zeros(signal.len());
        for (i, &sample) in signal.iter().enumerate() {
            output[i] = self.process_sample(sample);
        }
        output
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.pos = 0;
    }

    /// Get filter coefficients
    pub fn coeffs(&self) -> &[f32] {
        &self.coeffs
    }

    /// Get filter order (number of taps - 1)
    pub fn order(&self) -> usize {
        self.coeffs.len() - 1
    }
}

/// IIR (Infinite Impulse Response) filter
///
/// Implements a Direct Form II transposed structure.
#[derive(Debug, Clone)]
pub struct IirFilter {
    /// Feedforward (numerator) coefficients [b0, b1, b2, ...]
    b: Vec<f32>,
    /// Feedback (denominator) coefficients [a0, a1, a2, ...] (a0 should be 1.0)
    a: Vec<f32>,
    /// State variables
    state: Vec<f32>,
}

impl IirFilter {
    /// Create a new IIR filter with given coefficients
    ///
    /// `b` are the feedforward coefficients (numerator)
    /// `a` are the feedback coefficients (denominator), `a[0]` should be 1.0
    pub fn new(b: Vec<f32>, a: Vec<f32>) -> IoResult<Self> {
        if b.is_empty() || a.is_empty() {
            return Err(IoError::SignalError(
                "Filter coefficients cannot be empty".into(),
            ));
        }

        if (a[0] - 1.0).abs() > 1e-6 {
            return Err(IoError::SignalError(
                "a[0] must be 1.0 for normalized filter".into(),
            ));
        }

        let order = b.len().max(a.len());

        Ok(Self {
            b,
            a,
            state: vec![0.0; order],
        })
    }

    /// Design a 2nd-order Butterworth low-pass filter
    pub fn butterworth_lowpass(cutoff_normalized: f32) -> IoResult<Self> {
        validate_normalized_freq(cutoff_normalized, "Normalized cutoff")?;

        let omega = (PI * cutoff_normalized).tan();
        let omega2 = omega * omega;
        let sqrt2 = 2.0_f32.sqrt();
        let denom = 1.0 + sqrt2 * omega + omega2;

        let b0 = omega2 / denom;
        let b1 = 2.0 * b0;
        let b2 = b0;

        let a1 = 2.0 * (omega2 - 1.0) / denom;
        let a2 = (1.0 - sqrt2 * omega + omega2) / denom;

        Self::new(vec![b0, b1, b2], vec![1.0, a1, a2])
    }

    /// Design a 2nd-order Butterworth high-pass filter
    pub fn butterworth_highpass(cutoff_normalized: f32) -> IoResult<Self> {
        validate_normalized_freq(cutoff_normalized, "Normalized cutoff")?;

        let omega = (PI * cutoff_normalized).tan();
        let omega2 = omega * omega;
        let sqrt2 = 2.0_f32.sqrt();
        let denom = 1.0 + sqrt2 * omega + omega2;

        let b0 = 1.0 / denom;
        let b1 = -2.0 * b0;
        let b2 = b0;

        let a1 = 2.0 * (omega2 - 1.0) / denom;
        let a2 = (1.0 - sqrt2 * omega + omega2) / denom;

        Self::new(vec![b0, b1, b2], vec![1.0, a1, a2])
    }

    /// Design a 2nd-order notch (band-stop) filter
    pub fn notch(center_normalized: f32, q: f32) -> IoResult<Self> {
        validate_normalized_freq(center_normalized, "Normalized center")?;

        if q <= 0.0 {
            return Err(IoError::SignalError("Q factor must be positive".into()));
        }

        let omega0 = 2.0 * PI * center_normalized;
        let alpha = omega0.sin() / (2.0 * q);
        let cos_omega0 = omega0.cos();

        let b0 = 1.0;
        let b1 = -2.0 * cos_omega0;
        let b2 = 1.0;

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega0;
        let a2 = 1.0 - alpha;

        Self::new(vec![b0 / a0, b1 / a0, b2 / a0], vec![1.0, a1 / a0, a2 / a0])
    }

    /// Process a single sample
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let n_b = self.b.len();
        let n_a = self.a.len();

        let output = self.b[0] * input + self.state[0];

        for i in 0..self.state.len() - 1 {
            let b_term = if i + 1 < n_b {
                self.b[i + 1] * input
            } else {
                0.0
            };

            let a_term = if i + 1 < n_a {
                self.a[i + 1] * output
            } else {
                0.0
            };

            self.state[i] = b_term - a_term + self.state[i + 1];
        }

        let last = self.state.len() - 1;
        let b_term = if last + 1 < n_b {
            self.b[last + 1] * input
        } else {
            0.0
        };

        let a_term = if last + 1 < n_a {
            self.a[last + 1] * output
        } else {
            0.0
        };

        self.state[last] = b_term - a_term;

        output
    }

    /// Process an entire signal
    pub fn process(&mut self, signal: &Array1<f32>) -> Array1<f32> {
        let mut output = Array1::zeros(signal.len());
        for (i, &sample) in signal.iter().enumerate() {
            output[i] = self.process_sample(sample);
        }
        output
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

/// Filter types for signal processing
#[derive(Debug, Clone)]
pub enum Filter {
    /// Low-pass Butterworth filter (FFT-based)
    LowPass { cutoff: f32, order: usize },
    /// High-pass Butterworth filter (FFT-based)
    HighPass { cutoff: f32, order: usize },
    /// Band-pass filter (FFT-based)
    BandPass { low: f32, high: f32, order: usize },
    /// Moving average (FIR)
    MovingAverage { window: usize },
    /// Custom IIR filter
    Iir(IirFilter),
    /// Custom FIR filter
    Fir(FirFilter),
}

/// Shared validation for the normalized-frequency designers
/// (`sinc_lowpass`/`sinc_highpass`/`butterworth_lowpass`/
/// `butterworth_highpass`/`notch`).
///
/// The valid range is the OPEN interval `(0, 0.5)`. A previous version used
/// the half-open Rust range `0.0..0.5`, which *includes* 0.0 despite every
/// error message claiming otherwise -- letting `cutoff = 0.0` through
/// produced a zero sinc sum (dividing every FIR coefficient by ~0, yielding
/// NaN) or an all-zero IIR numerator, silently on `Ok`.
fn validate_normalized_freq(value: f32, name: &str) -> IoResult<()> {
    if !(value.is_finite() && value > 0.0 && value < 0.5) {
        return Err(IoError::SignalError(format!(
            "{name} must be in (0, 0.5), got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Regression tests: cutoff/center == 0.0 must be rejected (medium, id=44) ===

    #[test]
    fn test_sinc_lowpass_rejects_zero_cutoff() {
        assert!(FirFilter::sinc_lowpass(0.0, 15).is_err());
    }

    #[test]
    fn test_sinc_highpass_rejects_zero_cutoff() {
        assert!(FirFilter::sinc_highpass(0.0, 15).is_err());
    }

    #[test]
    fn test_sinc_lowpass_rejects_boundary_and_invalid_values() {
        assert!(
            FirFilter::sinc_lowpass(0.5, 15).is_err(),
            "0.5 is out of range"
        );
        assert!(
            FirFilter::sinc_lowpass(-0.1, 15).is_err(),
            "negative is out of range"
        );
        assert!(
            FirFilter::sinc_lowpass(f32::NAN, 15).is_err(),
            "NaN is never a valid cutoff"
        );
    }

    #[test]
    fn test_sinc_lowpass_accepts_valid_cutoff_and_normalizes() {
        let filter = FirFilter::sinc_lowpass(0.25, 15).unwrap();
        let sum: f32 = filter.coeffs().iter().sum();
        // A correctly normalized low-pass filter has unity DC gain.
        assert!((sum - 1.0).abs() < 1e-3, "sum = {sum}");
        assert!(filter.coeffs().iter().all(|c| c.is_finite()));
    }

    #[test]
    fn test_butterworth_lowpass_rejects_zero_cutoff() {
        assert!(IirFilter::butterworth_lowpass(0.0).is_err());
    }

    #[test]
    fn test_butterworth_highpass_rejects_zero_cutoff() {
        assert!(IirFilter::butterworth_highpass(0.0).is_err());
    }

    #[test]
    fn test_butterworth_lowpass_accepts_valid_cutoff() {
        let mut filter = IirFilter::butterworth_lowpass(0.25).unwrap();
        // The filter must produce finite, non-degenerate output (a zero
        // cutoff used to produce an all-zero numerator via a silent Ok,
        // which would process any input into a constant zero).
        let signal = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let output = filter.process(&signal);
        assert!(output.iter().all(|x| x.is_finite()));
        assert!(output.iter().any(|&x| x.abs() > 1e-6));
    }

    #[test]
    fn test_notch_rejects_zero_center() {
        assert!(IirFilter::notch(0.0, 1.0).is_err());
    }

    #[test]
    fn test_notch_accepts_valid_center() {
        assert!(IirFilter::notch(0.25, 1.0).is_ok());
    }

    // === Basic FIR/IIR behavior (test-gap, id=365 in-scope portion) ===

    #[test]
    fn test_fir_moving_average_smooths_step() {
        let mut filter = FirFilter::moving_average(4).unwrap();
        let signal = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let output = filter.process(&signal);
        // Once the window is full of 1.0s, output should approach 1.0.
        assert!((output[7] - 1.0).abs() < 1e-6, "output[7] = {}", output[7]);
        assert!(output[0].abs() < 1e-6, "output[0] = {}", output[0]);
    }

    #[test]
    fn test_fir_differentiator() {
        let mut filter = FirFilter::differentiator().unwrap();
        let signal = Array1::from_vec(vec![0.0, 1.0, 3.0, 6.0]);
        let output = filter.process(&signal);
        // y[n] = x[n] - x[n-1]
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
        assert!((output[2] - 2.0).abs() < 1e-6);
        assert!((output[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_fir_new_rejects_empty_coeffs() {
        assert!(FirFilter::new(vec![]).is_err());
    }

    #[test]
    fn test_fir_reset_clears_state() {
        let mut filter = FirFilter::moving_average(3).unwrap();
        filter.process_sample(5.0);
        filter.process_sample(5.0);
        filter.reset();
        // Right after reset, the delay line is all zeros again, so the
        // very next sample is averaged with two zeros.
        let out = filter.process_sample(3.0);
        assert!((out - 1.0).abs() < 1e-6, "out = {out}");
    }

    #[test]
    fn test_iir_new_requires_normalized_a0() {
        assert!(IirFilter::new(vec![1.0], vec![2.0]).is_err());
        assert!(IirFilter::new(vec![1.0], vec![1.0]).is_ok());
    }

    #[test]
    fn test_iir_notch_attenuates_target_frequency() {
        let mut filter = IirFilter::notch(0.1, 10.0).unwrap();
        let n = 512;
        let signal = Array1::from_vec((0..n).map(|i| (2.0 * PI * 0.1 * i as f32).sin()).collect());
        let output = filter.process(&signal);
        // Skip the transient at the start; the steady-state tail should be
        // strongly attenuated relative to the input.
        let tail_in: f32 = signal.iter().skip(n / 2).map(|x| x * x).sum();
        let tail_out: f32 = output.iter().skip(n / 2).map(|x| x * x).sum();
        assert!(
            tail_out < tail_in * 0.1,
            "notch filter should strongly attenuate its center frequency: in={tail_in}, out={tail_out}"
        );
    }
}
