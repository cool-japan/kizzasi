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
        if !(0.0..0.5).contains(&cutoff_normalized) {
            return Err(IoError::SignalError(
                "Normalized cutoff must be in (0, 0.5)".into(),
            ));
        }

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
        for c in &mut coeffs {
            *c /= sum;
        }

        Self::new(coeffs)
    }

    /// Design a windowed sinc high-pass filter
    pub fn sinc_highpass(cutoff_normalized: f32, num_taps: usize) -> IoResult<Self> {
        if !(0.0..0.5).contains(&cutoff_normalized) {
            return Err(IoError::SignalError(
                "Normalized cutoff must be in (0, 0.5)".into(),
            ));
        }

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
        if !(0.0..0.5).contains(&cutoff_normalized) {
            return Err(IoError::SignalError(
                "Normalized cutoff must be in (0, 0.5)".into(),
            ));
        }

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
        if !(0.0..0.5).contains(&cutoff_normalized) {
            return Err(IoError::SignalError(
                "Normalized cutoff must be in (0, 0.5)".into(),
            ));
        }

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
        if !(0.0..0.5).contains(&center_normalized) {
            return Err(IoError::SignalError(
                "Normalized center must be in (0, 0.5)".into(),
            ));
        }

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
