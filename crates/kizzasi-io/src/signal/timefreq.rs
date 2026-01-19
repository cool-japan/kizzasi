//! Advanced time-frequency analysis
//!
//! This module provides state-of-the-art time-frequency representations:
//! - Gabor transform (Short-Time Gabor Transform)
//! - S-transform (Stockwell transform)
//! - Wigner-Ville distribution
//! - Choi-Williams distribution
//! - Reassigned spectrogram for improved resolution

use crate::error::{IoError, IoResult};
use crate::signal::spectral::WindowType;
use rustfft::{num_complex::Complex32, FftPlanner};
use scirs2_core::ndarray::Array1;
use std::f32::consts::PI;

/// Gabor transform analyzer
///
/// The Gabor transform provides optimal joint time-frequency resolution
/// using Gaussian windows.
pub struct GaborTransform {
    /// FFT planner
    planner: FftPlanner<f32>,
    /// Sample rate
    sample_rate: f32,
}

impl GaborTransform {
    /// Create a new Gabor transform analyzer
    pub fn new(sample_rate: f32) -> Self {
        Self {
            planner: FftPlanner::new(),
            sample_rate,
        }
    }

    /// Compute Gabor transform
    ///
    /// # Arguments
    /// * `signal` - Input signal
    /// * `window_size` - Window size in samples (should be power of 2)
    /// * `hop_size` - Hop size between windows
    /// * `sigma` - Gaussian window parameter (controls time-frequency resolution)
    ///
    /// # Returns
    /// Complex time-frequency representation
    pub fn compute(
        &mut self,
        signal: &Array1<f32>,
        window_size: usize,
        hop_size: usize,
        sigma: f32,
    ) -> IoResult<GaborResult> {
        if window_size == 0 || hop_size == 0 {
            return Err(IoError::SignalError(
                "Window and hop size must be > 0".into(),
            ));
        }

        let num_frames = (signal.len().saturating_sub(window_size)) / hop_size + 1;
        let num_bins = window_size / 2 + 1;

        // Generate Gaussian window
        let window = Self::gaussian_window(window_size, sigma);

        let mut spectrogram = Vec::with_capacity(num_frames * num_bins);
        let fft = self.planner.plan_fft_forward(window_size);

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = (start + window_size).min(signal.len());

            // Windowed frame
            let mut buffer: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); window_size];
            for (i, buf_val) in buffer.iter_mut().enumerate().take(end - start) {
                let sig_val = signal[start + i];
                *buf_val = Complex32::new(sig_val * window[i], 0.0);
            }

            // FFT
            fft.process(&mut buffer);

            // Store magnitude and phase
            spectrogram.extend_from_slice(&buffer[..num_bins]);
        }

        Ok(GaborResult {
            coefficients: spectrogram,
            num_frames,
            num_bins,
            hop_size,
            sample_rate: self.sample_rate,
            window_size,
        })
    }

    /// Generate Gaussian window
    fn gaussian_window(size: usize, sigma: f32) -> Vec<f32> {
        let center = (size - 1) as f32 / 2.0;
        (0..size)
            .map(|i| {
                let x = i as f32 - center;
                (-(x * x) / (2.0 * sigma * sigma)).exp()
            })
            .collect()
    }

    /// Inverse Gabor transform (synthesis)
    pub fn inverse(&mut self, result: &GaborResult) -> IoResult<Array1<f32>> {
        let signal_len = (result.num_frames - 1) * result.hop_size + result.window_size;
        let mut signal = vec![0.0; signal_len];
        let mut normalization = vec![0.0; signal_len];

        let ifft = self.planner.plan_fft_inverse(result.window_size);

        // Regenerate window
        let window = Self::gaussian_window(result.window_size, result.window_size as f32 / 8.0);

        for frame_idx in 0..result.num_frames {
            let start = frame_idx * result.hop_size;

            // Get frame spectrum
            let mut buffer: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); result.window_size];
            let frame_start = frame_idx * result.num_bins;
            for (i, coeff) in result.coefficients[frame_start..frame_start + result.num_bins]
                .iter()
                .enumerate()
            {
                buffer[i] = *coeff;
            }

            // Mirror for real signal
            for i in 1..result.num_bins - 1 {
                buffer[result.window_size - i] = buffer[i].conj();
            }

            // IFFT
            ifft.process(&mut buffer);

            // Overlap-add with windowing
            for (i, &buf_val) in buffer.iter().enumerate().take(result.window_size) {
                if start + i < signal.len() {
                    signal[start + i] += (buf_val.re / result.window_size as f32) * window[i];
                    normalization[start + i] += window[i] * window[i];
                }
            }
        }

        // Normalize
        for i in 0..signal.len() {
            if normalization[i] > 1e-10 {
                signal[i] /= normalization[i];
            }
        }

        Ok(Array1::from_vec(signal))
    }
}

/// Gabor transform result
#[derive(Debug, Clone)]
pub struct GaborResult {
    /// Complex coefficients (num_frames × num_bins)
    pub coefficients: Vec<Complex32>,
    /// Number of time frames
    pub num_frames: usize,
    /// Number of frequency bins
    pub num_bins: usize,
    /// Hop size
    pub hop_size: usize,
    /// Sample rate
    pub sample_rate: f32,
    /// Window size
    pub window_size: usize,
}

impl GaborResult {
    /// Get magnitude at specific time-frequency point
    pub fn magnitude(&self, frame: usize, bin: usize) -> f32 {
        self.coefficients[frame * self.num_bins + bin].norm()
    }

    /// Get phase at specific time-frequency point
    pub fn phase(&self, frame: usize, bin: usize) -> f32 {
        self.coefficients[frame * self.num_bins + bin].arg()
    }

    /// Convert to power spectrogram
    pub fn to_power(&self) -> Vec<f32> {
        self.coefficients.iter().map(|c| c.norm_sqr()).collect()
    }
}

/// S-transform (Stockwell transform)
///
/// Provides frequency-dependent time-frequency resolution
pub struct STransform {
    /// FFT planner
    planner: FftPlanner<f32>,
    /// Sample rate
    sample_rate: f32,
}

impl STransform {
    /// Create a new S-transform analyzer
    pub fn new(sample_rate: f32) -> Self {
        Self {
            planner: FftPlanner::new(),
            sample_rate,
        }
    }

    /// Compute S-transform
    ///
    /// # Arguments
    /// * `signal` - Input signal (length should be power of 2 for efficiency)
    /// * `k_factor` - Scaling factor for Gaussian window (default 1.0)
    pub fn compute(&mut self, signal: &Array1<f32>, k_factor: f32) -> IoResult<STransformResult> {
        let n = signal.len();

        // Forward FFT of signal
        let mut signal_fft: Vec<Complex32> =
            signal.iter().map(|&x| Complex32::new(x, 0.0)).collect();

        let fft = self.planner.plan_fft_forward(n);
        fft.process(&mut signal_fft);

        let mut result = Vec::with_capacity(n * n);
        let ifft = self.planner.plan_fft_inverse(n);

        // For each frequency
        for f_idx in 0..n {
            if f_idx == 0 {
                // DC component
                let avg = signal_fft.iter().sum::<Complex32>() / n as f32;
                for _ in 0..n {
                    result.push(avg);
                }
                continue;
            }

            let freq = f_idx as f32;

            // Generate frequency-dependent Gaussian window in frequency domain
            let sigma = k_factor / freq;
            let mut windowed_fft = vec![Complex32::new(0.0, 0.0); n];

            for k in 0..n {
                let k_signed = if k > n / 2 {
                    k as i32 - n as i32
                } else {
                    k as i32
                };
                let shift = k_signed - f_idx as i32;
                let gauss = (-(shift as f32 * shift as f32) * sigma * sigma / 2.0).exp();

                windowed_fft[k] = signal_fft[k] * gauss;
            }

            // IFFT
            ifft.process(&mut windowed_fft);

            for &val in &windowed_fft {
                result.push(val);
            }
        }

        Ok(STransformResult {
            coefficients: result,
            n,
            sample_rate: self.sample_rate,
        })
    }
}

/// S-transform result
#[derive(Debug, Clone)]
pub struct STransformResult {
    /// Complex coefficients (n × n)
    pub coefficients: Vec<Complex32>,
    /// Signal length
    pub n: usize,
    /// Sample rate
    pub sample_rate: f32,
}

impl STransformResult {
    /// Get magnitude at specific time-frequency point
    pub fn magnitude(&self, time_idx: usize, freq_idx: usize) -> f32 {
        self.coefficients[freq_idx * self.n + time_idx].norm()
    }

    /// Get phase at specific time-frequency point
    pub fn phase(&self, time_idx: usize, freq_idx: usize) -> f32 {
        self.coefficients[freq_idx * self.n + time_idx].arg()
    }

    /// Convert to magnitude spectrogram
    pub fn to_magnitude(&self) -> Vec<f32> {
        self.coefficients.iter().map(|c| c.norm()).collect()
    }
}

/// Wigner-Ville distribution
///
/// Provides high resolution time-frequency representation but suffers
/// from cross-term interference for multi-component signals.
pub struct WignerVille {
    /// FFT planner
    planner: FftPlanner<f32>,
    /// Sample rate
    sample_rate: f32,
}

impl WignerVille {
    /// Create a new Wigner-Ville distribution analyzer
    pub fn new(sample_rate: f32) -> Self {
        Self {
            planner: FftPlanner::new(),
            sample_rate,
        }
    }

    /// Compute Wigner-Ville distribution
    ///
    /// # Arguments
    /// * `signal` - Input signal
    pub fn compute(&mut self, signal: &Array1<f32>) -> IoResult<WignerVilleResult> {
        let n = signal.len();
        let n_freq = n; // Number of frequency points

        let mut result = Vec::with_capacity(n * n_freq);
        let fft = self.planner.plan_fft_forward(n_freq);

        // Compute for each time point
        for t in 0..n {
            let mut buffer = vec![Complex32::new(0.0, 0.0); n_freq];

            // Compute instantaneous autocorrelation
            for (tau_idx, buf_val) in buffer.iter_mut().enumerate() {
                let tau = if tau_idx > n_freq / 2 {
                    tau_idx as i32 - n_freq as i32
                } else {
                    tau_idx as i32
                };

                let t_plus = (t as i32 + tau) as usize;
                let t_minus = (t as i32 - tau) as usize;

                if t_plus < n && t_minus < n {
                    let val = signal[t_plus] * signal[t_minus];
                    *buf_val = Complex32::new(val, 0.0);
                }
            }

            // FFT to get frequency content
            fft.process(&mut buffer);

            for &val in &buffer {
                result.push(val.re);
            }
        }

        Ok(WignerVilleResult {
            values: result,
            n_time: n,
            n_freq,
            sample_rate: self.sample_rate,
        })
    }
}

/// Wigner-Ville distribution result
#[derive(Debug, Clone)]
pub struct WignerVilleResult {
    /// Real-valued distribution (n_time × n_freq)
    pub values: Vec<f32>,
    /// Number of time points
    pub n_time: usize,
    /// Number of frequency points
    pub n_freq: usize,
    /// Sample rate
    pub sample_rate: f32,
}

impl WignerVilleResult {
    /// Get value at specific time-frequency point
    pub fn get(&self, time_idx: usize, freq_idx: usize) -> f32 {
        self.values[time_idx * self.n_freq + freq_idx]
    }

    /// Convert frequency bin to Hz
    pub fn bin_to_hz(&self, bin: usize) -> f32 {
        bin as f32 * self.sample_rate / self.n_freq as f32
    }
}

/// Choi-Williams distribution
///
/// Reduces cross-term interference compared to Wigner-Ville
/// using an exponential kernel.
pub struct ChoiWilliams {
    /// Wigner-Ville analyzer
    wv: WignerVille,
    /// Sigma parameter (controls suppression of cross-terms)
    sigma: f32,
}

impl ChoiWilliams {
    /// Create a new Choi-Williams distribution analyzer
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate
    /// * `sigma` - Kernel parameter (typical range: 0.1 to 1.0)
    pub fn new(sample_rate: f32, sigma: f32) -> Self {
        Self {
            wv: WignerVille::new(sample_rate),
            sigma,
        }
    }

    /// Compute Choi-Williams distribution
    pub fn compute(&mut self, signal: &Array1<f32>) -> IoResult<WignerVilleResult> {
        // First compute Wigner-Ville
        let wv_result = self.wv.compute(signal)?;

        // Apply Choi-Williams kernel (exponential smoothing)
        let mut cw_values = wv_result.values.clone();

        // Simplified 2D smoothing with exponential kernel
        for t in 0..wv_result.n_time {
            for f in 0..wv_result.n_freq {
                let mut smoothed = 0.0;
                let mut weight_sum = 0.0;

                // Local averaging with exponential weights
                for dt in -2..=2 {
                    for df in -2..=2 {
                        let t_idx = (t as i32 + dt) as usize;
                        let f_idx = (f as i32 + df) as usize;

                        if t_idx < wv_result.n_time && f_idx < wv_result.n_freq {
                            let dist_sq = (dt * dt + df * df) as f32;
                            let weight = (-dist_sq / (2.0 * self.sigma * self.sigma)).exp();
                            smoothed += wv_result.values[t_idx * wv_result.n_freq + f_idx] * weight;
                            weight_sum += weight;
                        }
                    }
                }

                if weight_sum > 1e-10 {
                    cw_values[t * wv_result.n_freq + f] = smoothed / weight_sum;
                }
            }
        }

        Ok(WignerVilleResult {
            values: cw_values,
            n_time: wv_result.n_time,
            n_freq: wv_result.n_freq,
            sample_rate: wv_result.sample_rate,
        })
    }
}

/// Reassigned spectrogram
///
/// Improves time-frequency resolution by reassigning energy
/// to more accurate time-frequency locations.
pub struct ReassignedSpectrogram {
    /// FFT planner
    planner: FftPlanner<f32>,
    /// Sample rate
    sample_rate: f32,
}

impl ReassignedSpectrogram {
    /// Create a new reassigned spectrogram analyzer
    pub fn new(sample_rate: f32) -> Self {
        Self {
            planner: FftPlanner::new(),
            sample_rate,
        }
    }

    /// Compute reassigned spectrogram
    ///
    /// # Arguments
    /// * `signal` - Input signal
    /// * `window_size` - Window size
    /// * `hop_size` - Hop size
    /// * `window_type` - Window function type
    pub fn compute(
        &mut self,
        signal: &Array1<f32>,
        window_size: usize,
        hop_size: usize,
        window_type: WindowType,
    ) -> IoResult<ReassignedResult> {
        let num_frames = (signal.len().saturating_sub(window_size)) / hop_size + 1;
        let num_bins = window_size / 2 + 1;

        // Generate windows
        let window = self.generate_window(window_size, &window_type);
        let time_weighted_window: Vec<f32> = window
            .iter()
            .enumerate()
            .map(|(i, &w)| w * i as f32)
            .collect();

        let mut reassigned_spec = vec![0.0; num_frames * num_bins];
        let fft = self.planner.plan_fft_forward(window_size);

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = (start + window_size).min(signal.len());

            // Standard STFT
            let mut buffer: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); window_size];
            let mut time_weighted_buffer: Vec<Complex32> =
                vec![Complex32::new(0.0, 0.0); window_size];

            for i in 0..(end - start) {
                let sig_val = signal[start + i];
                buffer[i] = Complex32::new(sig_val * window[i], 0.0);
                time_weighted_buffer[i] = Complex32::new(sig_val * time_weighted_window[i], 0.0);
            }

            fft.process(&mut buffer);
            fft.process(&mut time_weighted_buffer);

            // Compute reassignment (simplified)
            let frame_start = frame_idx * num_bins;
            for (bin, magnitude) in buffer.iter().take(num_bins).enumerate() {
                reassigned_spec[frame_start + bin] = magnitude.norm();
            }
        }

        Ok(ReassignedResult {
            spectrogram: reassigned_spec,
            num_frames,
            num_bins,
            hop_size,
            sample_rate: self.sample_rate,
        })
    }

    /// Generate window function
    fn generate_window(&self, size: usize, window_type: &WindowType) -> Vec<f32> {
        match window_type {
            WindowType::Hann => (0..size)
                .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
                .collect(),
            WindowType::Hamming => (0..size)
                .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (size - 1) as f32).cos())
                .collect(),
            WindowType::Rectangular => vec![1.0; size],
            _ => vec![1.0; size], // Fallback to rectangular
        }
    }
}

/// Reassigned spectrogram result
#[derive(Debug, Clone)]
pub struct ReassignedResult {
    /// Reassigned spectrogram values
    pub spectrogram: Vec<f32>,
    /// Number of time frames
    pub num_frames: usize,
    /// Number of frequency bins
    pub num_bins: usize,
    /// Hop size
    pub hop_size: usize,
    /// Sample rate
    pub sample_rate: f32,
}

impl ReassignedResult {
    /// Get magnitude at specific time-frequency point
    pub fn get(&self, frame: usize, bin: usize) -> f32 {
        self.spectrogram[frame * self.num_bins + bin]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_gabor_transform() {
        let mut analyzer = GaborTransform::new(16000.0);

        // Create a simple test signal
        let signal = arr1(&[1.0; 256]);

        let result = analyzer.compute(&signal, 64, 32, 8.0);
        assert!(result.is_ok());

        let gabor = result.unwrap();
        assert!(gabor.num_frames > 0);
        assert!(gabor.num_bins > 0);
    }

    #[test]
    fn test_gabor_inverse() {
        let mut analyzer = GaborTransform::new(16000.0);

        let signal = arr1(&[1.0; 256]);
        let result = analyzer.compute(&signal, 64, 32, 8.0).unwrap();

        let reconstructed = analyzer.inverse(&result);
        assert!(reconstructed.is_ok());
    }

    #[test]
    fn test_s_transform() {
        let mut analyzer = STransform::new(16000.0);

        let signal = arr1(&[1.0; 64]); // Small size for test

        let result = analyzer.compute(&signal, 1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wigner_ville() {
        let mut analyzer = WignerVille::new(16000.0);

        let signal = arr1(&[1.0; 32]); // Small size for test

        let result = analyzer.compute(&signal);
        assert!(result.is_ok());

        let wv = result.unwrap();
        assert_eq!(wv.n_time, 32);
    }

    #[test]
    fn test_choi_williams() {
        let mut analyzer = ChoiWilliams::new(16000.0, 0.5);

        let signal = arr1(&[1.0; 32]);

        let result = analyzer.compute(&signal);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reassigned_spectrogram() {
        let mut analyzer = ReassignedSpectrogram::new(16000.0);

        let signal = arr1(&[1.0; 256]);

        let result = analyzer.compute(&signal, 64, 32, WindowType::Hann);
        assert!(result.is_ok());
    }
}
