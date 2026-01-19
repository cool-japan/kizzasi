//! Sample rate conversion and resampling
//!
//! This module provides various resampling algorithms for audio and signal processing:
//! - Linear interpolation (fast, lower quality)
//! - Cubic interpolation (better quality)
//! - Polyphase filtering (high quality, efficient)
//! - Streaming resamplers for real-time processing

use super::functions::bessel_i0;
use std::f32::consts::PI;

/// Simple linear interpolation resampler
///
/// Faster than polyphase but lower quality. Good for preview or
/// when computational resources are limited.
#[derive(Debug, Clone)]
pub struct LinearResampler {
    ratio: f32,
}

impl LinearResampler {
    /// Create a new linear resampler
    pub fn new(from_rate: f32, to_rate: f32) -> Self {
        Self {
            ratio: to_rate / from_rate,
        }
    }

    /// Resample using linear interpolation
    pub fn resample(&self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        let output_len = ((input.len() as f32) * self.ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_pos = i as f32 / self.ratio;
            let idx0 = src_pos.floor() as usize;
            let idx1 = (idx0 + 1).min(input.len() - 1);
            let frac = src_pos - idx0 as f32;

            let sample = input[idx0] * (1.0 - frac) + input[idx1] * frac;
            output.push(sample);
        }

        output
    }

    /// Get the resampling ratio
    pub fn ratio(&self) -> f32 {
        self.ratio
    }
}

/// Cubic interpolation resampler (better quality than linear)
#[derive(Debug, Clone)]
pub struct CubicResampler {
    ratio: f32,
}

impl CubicResampler {
    /// Create a new cubic resampler
    pub fn new(from_rate: f32, to_rate: f32) -> Self {
        Self {
            ratio: to_rate / from_rate,
        }
    }

    /// Resample using cubic interpolation (Catmull-Rom spline)
    pub fn resample(&self, input: &[f32]) -> Vec<f32> {
        if input.len() < 4 {
            return LinearResampler::new(1.0, self.ratio).resample(input);
        }

        let output_len = ((input.len() as f32) * self.ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_pos = i as f32 / self.ratio;
            let idx1 = src_pos.floor() as usize;
            let frac = src_pos - idx1 as f32;

            let idx0 = if idx1 > 0 { idx1 - 1 } else { 0 };
            let idx2 = (idx1 + 1).min(input.len() - 1);
            let idx3 = (idx1 + 2).min(input.len() - 1);

            let p0 = input[idx0];
            let p1 = input[idx1];
            let p2 = input[idx2];
            let p3 = input[idx3];

            let sample = Self::catmull_rom(p0, p1, p2, p3, frac);
            output.push(sample);
        }

        output
    }

    /// Catmull-Rom spline interpolation
    fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
        let t2 = t * t;
        let t3 = t2 * t;
        0.5 * ((2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
    }

    /// Get the resampling ratio
    pub fn ratio(&self) -> f32 {
        self.ratio
    }
}

/// Real-time streaming resampler for audio processing
///
/// Maintains state between calls for seamless real-time resampling.
/// Uses linear interpolation with fractional delay tracking.
///
/// ## Example
/// ```rust
/// use kizzasi_io::StreamingResampler;
///
/// let mut resampler = StreamingResampler::new(44100.0, 48000.0);
///
/// // Process chunks as they arrive
/// let chunk1 = vec![0.0, 0.1, 0.2, 0.3];
/// let output1 = resampler.process(&chunk1);
///
/// let chunk2 = vec![0.4, 0.5, 0.6, 0.7];
/// let output2 = resampler.process(&chunk2);
/// ```
#[derive(Debug, Clone)]
pub struct StreamingResampler {
    /// Sample rate conversion ratio (output/input)
    ratio: f32,
    /// Fractional position in input stream
    position: f32,
    /// Previous sample (for interpolation across chunks)
    prev_sample: f32,
    /// Input sample rate
    input_rate: f32,
    /// Output sample rate
    output_rate: f32,
}

impl StreamingResampler {
    /// Create a new streaming resampler
    pub fn new(input_rate: f32, output_rate: f32) -> Self {
        Self {
            ratio: output_rate / input_rate,
            position: 0.0,
            prev_sample: 0.0,
            input_rate,
            output_rate,
        }
    }

    /// Process a chunk of input samples and produce resampled output
    ///
    /// Maintains internal state for seamless processing across chunks.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        let estimated_output_len = ((input.len() as f32) * self.ratio).ceil() as usize + 1;
        let mut output = Vec::with_capacity(estimated_output_len);

        let mut idx = 0;
        while idx < input.len() {
            let int_pos = self.position.floor() as usize;
            let frac = self.position - self.position.floor();

            if int_pos + 1 < input.len() {
                let sample = input[int_pos] * (1.0 - frac) + input[int_pos + 1] * frac;
                output.push(sample);
            } else if int_pos < input.len() {
                output.push(input[int_pos]);
            } else {
                break;
            }

            self.position += 1.0 / self.ratio;
            idx = self.position.floor() as usize;
        }

        self.position -= input.len() as f32;
        if !input.is_empty() {
            self.prev_sample = *input.last().expect("Input must be non-empty");
        }

        output
    }

    /// Reset the resampler state
    pub fn reset(&mut self) {
        self.position = 0.0;
        self.prev_sample = 0.0;
    }

    /// Get the resampling ratio
    pub fn ratio(&self) -> f32 {
        self.ratio
    }

    /// Get input sample rate
    pub fn input_rate(&self) -> f32 {
        self.input_rate
    }

    /// Get output sample rate
    pub fn output_rate(&self) -> f32 {
        self.output_rate
    }
}

/// High-quality streaming resampler using Sinc interpolation
///
/// Provides better quality than linear interpolation for real-time applications.
/// Uses windowed Sinc interpolation with state preservation.
#[derive(Debug, Clone)]
pub struct SincStreamingResampler {
    /// Sample rate conversion ratio (output/input)
    ratio: f32,
    /// Fractional position in input stream
    position: f32,
    /// History buffer for Sinc interpolation
    history: Vec<f32>,
    /// Sinc filter kernel size
    kernel_size: usize,
    /// Input sample rate
    input_rate: f32,
    /// Output sample rate
    output_rate: f32,
}

impl SincStreamingResampler {
    /// Create a new Sinc streaming resampler
    pub fn new(input_rate: f32, output_rate: f32, kernel_size: usize) -> Self {
        Self {
            ratio: output_rate / input_rate,
            position: 0.0,
            history: vec![0.0; kernel_size],
            kernel_size,
            input_rate,
            output_rate,
        }
    }

    /// Process a chunk of input samples using windowed Sinc interpolation
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        let mut combined = self.history.clone();
        combined.extend_from_slice(input);

        let estimated_output_len = ((input.len() as f32) * self.ratio).ceil() as usize + 1;
        let mut output = Vec::with_capacity(estimated_output_len);

        let half_kernel = self.kernel_size / 2;

        while self.position < input.len() as f32 {
            let center_pos = (self.position + self.kernel_size as f32) as usize;
            if center_pos + half_kernel < combined.len() {
                let sample = self.sinc_interpolate(&combined, center_pos, self.position.fract());
                output.push(sample);
            } else {
                break;
            }
            self.position += 1.0 / self.ratio;
        }

        let history_start = input.len().saturating_sub(self.kernel_size);
        self.history = input[history_start..].to_vec();
        if self.history.len() < self.kernel_size {
            let mut new_history = vec![0.0; self.kernel_size - self.history.len()];
            new_history.extend_from_slice(&self.history);
            self.history = new_history;
        }

        self.position -= input.len() as f32;
        output
    }

    /// Windowed Sinc interpolation
    fn sinc_interpolate(&self, data: &[f32], center: usize, frac: f32) -> f32 {
        let half_kernel = self.kernel_size / 2;
        let mut sum = 0.0;

        for i in 0..self.kernel_size {
            let idx = center + i - half_kernel;
            if idx < data.len() {
                let x = (i as f32) - (half_kernel as f32) - frac;
                let sinc_val = self.sinc(x);
                let window_val = self.blackman_window(i, self.kernel_size);
                sum += data[idx] * sinc_val * window_val;
            }
        }

        sum
    }

    /// Normalized Sinc function
    fn sinc(&self, x: f32) -> f32 {
        if x.abs() < 1e-6 {
            1.0
        } else {
            let pi_x = std::f32::consts::PI * x;
            pi_x.sin() / pi_x
        }
    }

    /// Blackman window function
    fn blackman_window(&self, n: usize, size: usize) -> f32 {
        let a0 = 0.42;
        let a1 = 0.5;
        let a2 = 0.08;
        let n_f = n as f32;
        let size_f = (size - 1) as f32;
        a0 - a1 * (2.0 * std::f32::consts::PI * n_f / size_f).cos()
            + a2 * (4.0 * std::f32::consts::PI * n_f / size_f).cos()
    }

    /// Reset the resampler state
    pub fn reset(&mut self) {
        self.position = 0.0;
        self.history.fill(0.0);
    }

    /// Get the resampling ratio
    pub fn ratio(&self) -> f32 {
        self.ratio
    }

    /// Get the input sample rate
    pub fn input_rate(&self) -> f32 {
        self.input_rate
    }

    /// Get the output sample rate
    pub fn output_rate(&self) -> f32 {
        self.output_rate
    }
}

/// Resampler using polyphase filter implementation
///
/// Provides efficient sample rate conversion using polyphase decomposition
/// of an anti-aliasing/interpolation filter.
#[derive(Debug, Clone)]
pub struct Resampler {
    /// Upsampling factor
    up_factor: usize,
    /// Downsampling factor
    down_factor: usize,
    /// Polyphase filter bank (up_factor phases)
    filter_bank: Vec<Vec<f32>>,
    /// Filter delay line
    delay_line: Vec<f32>,
    /// Current phase
    phase: usize,
    /// Input sample accumulator
    input_idx: usize,
}

impl Resampler {
    /// Create a new resampler for converting from `from_rate` to `to_rate`
    ///
    /// Uses rational resampling with polyphase filter implementation.
    pub fn new(from_rate: u32, to_rate: u32, filter_length: usize) -> Self {
        let gcd = Self::gcd(from_rate, to_rate);
        let up_factor = (to_rate / gcd) as usize;
        let down_factor = (from_rate / gcd) as usize;

        let cutoff = 0.5 / (up_factor.max(down_factor) as f32);
        let filter = Self::design_lowpass(filter_length * up_factor, cutoff);
        let filter_bank = Self::polyphase_decompose(&filter, up_factor);

        let max_filter_len = filter_bank.iter().map(|f| f.len()).max().unwrap_or(0);
        let delay_line = vec![0.0; max_filter_len];

        Self {
            up_factor,
            down_factor,
            filter_bank,
            delay_line,
            phase: 0,
            input_idx: 0,
        }
    }

    /// Create a resampler by a simple integer ratio
    pub fn by_ratio(up: usize, down: usize, filter_length: usize) -> Self {
        let gcd = Self::gcd(up as u32, down as u32);
        let up_factor = up / gcd as usize;
        let down_factor = down / gcd as usize;

        let cutoff = 0.5 / (up_factor.max(down_factor) as f32);
        let filter = Self::design_lowpass(filter_length * up_factor, cutoff);
        let filter_bank = Self::polyphase_decompose(&filter, up_factor);

        let max_filter_len = filter_bank.iter().map(|f| f.len()).max().unwrap_or(0);
        let delay_line = vec![0.0; max_filter_len];

        Self {
            up_factor,
            down_factor,
            filter_bank,
            delay_line,
            phase: 0,
            input_idx: 0,
        }
    }

    /// Get the resampling ratio (output_samples / input_samples)
    pub fn ratio(&self) -> f32 {
        self.up_factor as f32 / self.down_factor as f32
    }

    /// Resample an entire signal
    pub fn resample(&mut self, input: &[f32]) -> Vec<f32> {
        self.reset();
        let expected_len = (input.len() * self.up_factor).div_ceil(self.down_factor);
        let mut output = Vec::with_capacity(expected_len);

        let mut phase_acc = 0usize;

        for &sample in input {
            self.delay_line.rotate_right(1);
            self.delay_line[0] = sample;

            while phase_acc < self.up_factor {
                let y = self.compute_output(phase_acc);
                output.push(y);
                phase_acc += self.down_factor;
            }

            phase_acc -= self.up_factor;
        }

        output
    }

    /// Compute output for a given phase
    fn compute_output(&self, phase: usize) -> f32 {
        if phase >= self.filter_bank.len() {
            return 0.0;
        }

        let filter = &self.filter_bank[phase];
        let mut sum = 0.0f32;

        for (i, &coeff) in filter.iter().enumerate() {
            if i < self.delay_line.len() {
                sum += coeff * self.delay_line[i];
            }
        }

        sum * self.up_factor as f32
    }

    /// Reset the resampler state
    pub fn reset(&mut self) {
        self.delay_line.fill(0.0);
        self.phase = 0;
        self.input_idx = 0;
    }

    /// Design a lowpass FIR filter using windowed sinc
    fn design_lowpass(length: usize, cutoff: f32) -> Vec<f32> {
        let length = if length.is_multiple_of(2) {
            length + 1
        } else {
            length
        };
        let m = (length - 1) as f32 / 2.0;
        let mut filter = Vec::with_capacity(length);

        for i in 0..length {
            let n = i as f32 - m;
            let sinc = if n.abs() < 1e-10 {
                2.0 * cutoff
            } else {
                (2.0 * PI * cutoff * n).sin() / (PI * n)
            };

            let beta = 5.0;
            let x = 2.0 * i as f32 / (length - 1) as f32 - 1.0;
            let window = bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta);
            filter.push(sinc * window);
        }

        let sum: f32 = filter.iter().sum();
        for coeff in &mut filter {
            *coeff /= sum;
        }

        filter
    }

    /// Decompose filter into polyphase components
    fn polyphase_decompose(filter: &[f32], num_phases: usize) -> Vec<Vec<f32>> {
        let mut bank = vec![Vec::new(); num_phases];
        for (i, &coeff) in filter.iter().enumerate() {
            let phase = i % num_phases;
            bank[phase].push(coeff);
        }
        bank
    }

    /// GCD using Euclidean algorithm
    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }
}

/// Farrow structure for fractional delay and arbitrary sample rate conversion
///
/// Provides continuous-time interpolation using polynomial approximation.
/// Efficient for time-varying delays and arbitrary resampling ratios.
///
/// ## References
/// - C. W. Farrow, "A continuously variable digital delay element", 1988
#[derive(Debug, Clone)]
pub struct FarrowResampler {
    /// Polynomial order (typically 3 for cubic, 5 for 5th-order)
    #[allow(dead_code)]
    order: usize,
    /// Filter coefficients for each polynomial
    coefficients: Vec<Vec<f32>>,
    /// Delay line for input samples
    delay_line: Vec<f32>,
    /// Fractional delay accumulator
    fractional_delay: f32,
    /// Resampling ratio (output_rate / input_rate)
    ratio: f32,
}

impl FarrowResampler {
    /// Create a new Farrow resampler with cubic interpolation
    pub fn new_cubic(input_rate: f32, output_rate: f32) -> Self {
        Self::new(input_rate, output_rate, 3)
    }

    /// Create a new Farrow resampler with specified polynomial order
    pub fn new(input_rate: f32, output_rate: f32, order: usize) -> Self {
        let ratio = output_rate / input_rate;

        // Lagrange interpolation coefficients for different polynomial orders
        let coefficients = match order {
            1 => {
                // Linear interpolation
                vec![
                    vec![0.0, 1.0],  // c0: constant term
                    vec![-1.0, 1.0], // c1: linear term
                ]
            }
            3 => {
                // Cubic (Lagrange 3rd order)
                vec![
                    vec![0.0, 1.0, 0.0, 0.0],                           // c0
                    vec![-1.0 / 2.0, 0.0, 1.0 / 2.0, 0.0],              // c1
                    vec![1.0, -5.0 / 2.0, 2.0, -1.0 / 2.0],             // c2
                    vec![-1.0 / 2.0, 3.0 / 2.0, -3.0 / 2.0, 1.0 / 2.0], // c3
                ]
            }
            _ => {
                // Default to cubic
                vec![
                    vec![0.0, 1.0, 0.0, 0.0],
                    vec![-1.0 / 2.0, 0.0, 1.0 / 2.0, 0.0],
                    vec![1.0, -5.0 / 2.0, 2.0, -1.0 / 2.0],
                    vec![-1.0 / 2.0, 3.0 / 2.0, -3.0 / 2.0, 1.0 / 2.0],
                ]
            }
        };

        let filter_len = coefficients[0].len();
        let delay_line = vec![0.0; filter_len];

        Self {
            order,
            coefficients,
            delay_line,
            fractional_delay: 0.0,
            ratio,
        }
    }

    /// Process samples with Farrow structure
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        let estimated_len = ((input.len() as f32) * self.ratio).ceil() as usize + 1;
        let mut output = Vec::with_capacity(estimated_len);

        for &sample in input {
            // Shift delay line
            self.delay_line.rotate_right(1);
            self.delay_line[0] = sample;

            // Generate output samples for this input
            while self.fractional_delay < 1.0 {
                let interpolated = self.farrow_interpolate(self.fractional_delay);
                output.push(interpolated);
                self.fractional_delay += 1.0 / self.ratio;
            }

            self.fractional_delay -= 1.0;
        }

        output
    }

    /// Farrow polynomial interpolation
    fn farrow_interpolate(&self, mu: f32) -> f32 {
        // Evaluate polynomial using Horner's method
        // y(mu) = c0(x) + mu * c1(x) + mu^2 * c2(x) + ...

        let mut result = 0.0;
        let mut mu_power = 1.0;

        for coef_vec in &self.coefficients {
            let mut poly_sum = 0.0;
            for (i, &coef) in coef_vec.iter().enumerate() {
                if i < self.delay_line.len() {
                    poly_sum += coef * self.delay_line[i];
                }
            }
            result += mu_power * poly_sum;
            mu_power *= mu;
        }

        result
    }

    /// Resample entire signal at once
    pub fn resample(&mut self, input: &[f32]) -> Vec<f32> {
        self.reset();
        self.process(input)
    }

    /// Reset the resampler state
    pub fn reset(&mut self) {
        self.delay_line.fill(0.0);
        self.fractional_delay = 0.0;
    }

    /// Get the resampling ratio
    pub fn ratio(&self) -> f32 {
        self.ratio
    }

    /// Set new resampling ratio (for time-varying resampling)
    pub fn set_ratio(&mut self, new_ratio: f32) {
        self.ratio = new_ratio;
    }
}

/// Time-varying resampler for dynamic sample rate conversion
///
/// Allows the resampling ratio to change over time, useful for:
/// - Pitch shifting with time-varying pitch
/// - Synchronization with time-varying clock sources
/// - Adaptive rate control
#[derive(Debug, Clone)]
pub struct TimeVaryingResampler {
    /// Current Farrow resampler
    farrow: FarrowResampler,
    /// Ratio modulation function type
    modulation_type: RatioModulation,
    /// Sample counter
    sample_count: usize,
    /// Base ratio
    base_ratio: f32,
    /// Modulation depth (0.0 to 1.0)
    modulation_depth: f32,
    /// Modulation frequency (in samples)
    modulation_period: f32,
}

/// Type of ratio modulation
#[derive(Debug, Clone, Copy)]
pub enum RatioModulation {
    /// Constant ratio (no modulation)
    Constant,
    /// Sinusoidal modulation
    Sinusoidal,
    /// Linear chirp
    LinearChirp,
    /// Exponential chirp
    ExponentialChirp,
}

impl TimeVaryingResampler {
    /// Create a new time-varying resampler
    pub fn new(
        base_input_rate: f32,
        base_output_rate: f32,
        modulation_type: RatioModulation,
        modulation_depth: f32,
        modulation_freq_hz: f32,
    ) -> Self {
        let base_ratio = base_output_rate / base_input_rate;
        let farrow = FarrowResampler::new_cubic(base_input_rate, base_output_rate);

        let modulation_period = base_input_rate / modulation_freq_hz;

        Self {
            farrow,
            modulation_type,
            sample_count: 0,
            base_ratio,
            modulation_depth,
            modulation_period,
        }
    }

    /// Process samples with time-varying ratio
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::new();

        for &sample in input {
            // Update ratio based on modulation
            let current_ratio = self.compute_current_ratio();
            self.farrow.set_ratio(current_ratio);

            // Process single sample
            let single_input = [sample];
            let chunk_output = self.farrow.process(&single_input);
            output.extend_from_slice(&chunk_output);

            self.sample_count += 1;
        }

        output
    }

    /// Compute current resampling ratio based on modulation
    fn compute_current_ratio(&self) -> f32 {
        match self.modulation_type {
            RatioModulation::Constant => self.base_ratio,

            RatioModulation::Sinusoidal => {
                let phase = 2.0 * PI * (self.sample_count as f32) / self.modulation_period;
                let modulation = self.modulation_depth * phase.sin();
                self.base_ratio * (1.0 + modulation)
            }

            RatioModulation::LinearChirp => {
                let t = (self.sample_count as f32) / self.modulation_period;
                let modulation = self.modulation_depth * t;
                self.base_ratio * (1.0 + modulation)
            }

            RatioModulation::ExponentialChirp => {
                let t = (self.sample_count as f32) / self.modulation_period;
                let modulation = self.modulation_depth * (t.exp() - 1.0);
                self.base_ratio * (1.0 + modulation)
            }
        }
    }

    /// Reset the resampler
    pub fn reset(&mut self) {
        self.farrow.reset();
        self.sample_count = 0;
    }

    /// Get base ratio
    pub fn base_ratio(&self) -> f32 {
        self.base_ratio
    }

    /// Set modulation depth
    pub fn set_modulation_depth(&mut self, depth: f32) {
        self.modulation_depth = depth.clamp(0.0, 1.0);
    }
}

/// Arbitrary sample rate converter using adaptive filtering
///
/// Handles arbitrary (non-rational) conversion ratios efficiently
/// by combining Farrow structure with adaptive filtering.
#[derive(Debug, Clone)]
pub struct ArbitrarySrcResampler {
    farrow: FarrowResampler,
    input_rate: f32,
    output_rate: f32,
}

impl ArbitrarySrcResampler {
    /// Create a new arbitrary SRC resampler
    pub fn new(input_rate: f32, output_rate: f32) -> Self {
        let farrow = FarrowResampler::new_cubic(input_rate, output_rate);

        Self {
            farrow,
            input_rate,
            output_rate,
        }
    }

    /// Process samples
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        self.farrow.process(input)
    }

    /// Resample entire signal
    pub fn resample(&mut self, input: &[f32]) -> Vec<f32> {
        self.farrow.resample(input)
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.farrow.reset();
    }

    /// Get conversion ratio
    pub fn ratio(&self) -> f32 {
        self.output_rate / self.input_rate
    }

    /// Dynamically change sample rates
    pub fn update_rates(&mut self, new_input_rate: f32, new_output_rate: f32) {
        self.input_rate = new_input_rate;
        self.output_rate = new_output_rate;
        let new_ratio = new_output_rate / new_input_rate;
        self.farrow.set_ratio(new_ratio);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_farrow_resampler() {
        let mut resampler = FarrowResampler::new_cubic(44100.0, 48000.0);

        // Test with sine wave
        let input: Vec<f32> = (0..1000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let output = resampler.resample(&input);

        // Output should be longer (upsampling)
        assert!(output.len() > input.len());

        // Check ratio
        let actual_ratio = output.len() as f32 / input.len() as f32;
        let expected_ratio = 48000.0 / 44100.0;
        assert!((actual_ratio - expected_ratio).abs() < 0.1);
    }

    #[test]
    fn test_time_varying_resampler() {
        let mut resampler =
            TimeVaryingResampler::new(44100.0, 48000.0, RatioModulation::Sinusoidal, 0.1, 10.0);

        let input: Vec<f32> = (0..1000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let output = resampler.process(&input);

        assert!(!output.is_empty());
        // Output length varies due to time-varying ratio
    }

    #[test]
    fn test_arbitrary_src() {
        let mut resampler = ArbitrarySrcResampler::new(44100.0, 48000.0);

        let input: Vec<f32> = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let output = resampler.resample(&input);

        assert!(!output.is_empty());
        assert!(output.len() > input.len());
    }

    #[test]
    fn test_farrow_fractional_delay() {
        let mut resampler = FarrowResampler::new_cubic(1000.0, 1000.0);

        // Test with step function
        let mut input = vec![0.0; 10];
        input.extend(vec![1.0; 10]);

        let output = resampler.resample(&input);

        // Should preserve signal characteristics
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_ratio_modulation_types() {
        for modulation in &[
            RatioModulation::Constant,
            RatioModulation::Sinusoidal,
            RatioModulation::LinearChirp,
            RatioModulation::ExponentialChirp,
        ] {
            let mut resampler = TimeVaryingResampler::new(8000.0, 8000.0, *modulation, 0.05, 1.0);

            let input: Vec<f32> = vec![1.0; 100];
            let output = resampler.process(&input);

            assert!(!output.is_empty());
        }
    }
}
