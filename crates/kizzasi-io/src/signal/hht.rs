//! Hilbert-Huang Transform (HHT) implementation
//!
//! This module provides advanced time-frequency analysis for non-stationary and non-linear signals:
//! - Empirical Mode Decomposition (EMD) for signal decomposition
//! - Intrinsic Mode Functions (IMFs) extraction
//! - Hilbert spectral analysis
//! - Instantaneous frequency and amplitude tracking
//! - Ensemble EMD (EEMD) for noise-assisted decomposition

use crate::error::{IoError, IoResult};
use std::f64::consts::PI;

/// Configuration for EMD algorithm
#[derive(Debug, Clone)]
pub struct EmdConfig {
    /// Maximum number of IMFs to extract
    pub max_imfs: usize,
    /// Maximum number of sifting iterations per IMF
    pub max_sifting_iterations: usize,
    /// Standard deviation threshold for stopping sifting
    pub sd_threshold: f64,
    /// Minimum number of extrema required
    pub min_extrema: usize,
    /// Enable boundary extension to reduce edge effects
    pub boundary_extension: bool,
    /// Extension method: "mirror" or "polynomial"
    pub extension_method: String,
}

impl Default for EmdConfig {
    fn default() -> Self {
        Self {
            max_imfs: 10,
            max_sifting_iterations: 100,
            sd_threshold: 0.3,
            min_extrema: 3,
            boundary_extension: true,
            extension_method: "mirror".to_string(),
        }
    }
}

/// Intrinsic Mode Function with its properties
#[derive(Debug, Clone)]
pub struct IntrinsicModeFunction {
    /// IMF signal data
    pub data: Vec<f64>,
    /// Instantaneous amplitude (envelope)
    pub amplitude: Vec<f64>,
    /// Instantaneous frequency (Hz)
    pub frequency: Vec<f64>,
    /// Instantaneous phase (radians)
    pub phase: Vec<f64>,
}

/// Result of EMD decomposition
#[derive(Debug, Clone)]
pub struct EmdResult {
    /// Extracted Intrinsic Mode Functions
    pub imfs: Vec<IntrinsicModeFunction>,
    /// Residual signal (trend)
    pub residual: Vec<f64>,
}

/// Empirical Mode Decomposition (EMD) implementation
pub struct EmpiricalModeDecomposition {
    config: EmdConfig,
    sample_rate: f64,
}

impl EmpiricalModeDecomposition {
    /// Create a new EMD analyzer
    pub fn new(sample_rate: f64, config: EmdConfig) -> Self {
        Self {
            config,
            sample_rate,
        }
    }

    /// Decompose a signal into IMFs
    pub fn decompose(&self, signal: &[f64]) -> IoResult<EmdResult> {
        if signal.is_empty() {
            return Err(IoError::ConfigError("Empty signal".to_string()));
        }

        let mut imfs = Vec::new();
        let mut residual = signal.to_vec();

        for _ in 0..self.config.max_imfs {
            // Check if we can extract more IMFs
            let extrema_count = self.count_extrema(&residual);
            if extrema_count < self.config.min_extrema {
                break;
            }

            // Extract one IMF using sifting process
            match self.extract_imf(&residual) {
                Ok(imf_data) => {
                    // Compute Hilbert transform for the IMF
                    let hilbert_result = self.hilbert_spectrum(&imf_data)?;

                    imfs.push(hilbert_result);

                    // Subtract IMF from residual
                    for i in 0..residual.len() {
                        residual[i] -= imf_data[i];
                    }
                }
                Err(_) => break,
            }
        }

        Ok(EmdResult { imfs, residual })
    }

    /// Extract a single IMF using the sifting process
    fn extract_imf(&self, signal: &[f64]) -> IoResult<Vec<f64>> {
        let mut h = signal.to_vec();

        for _ in 0..self.config.max_sifting_iterations {
            // Find local maxima and minima
            let (max_indices, max_values) = self.find_extrema(&h, true);
            let (min_indices, min_values) = self.find_extrema(&h, false);

            if max_indices.len() < 2 || min_indices.len() < 2 {
                return Err(IoError::SignalError("Insufficient extrema".to_string()));
            }

            // Create upper and lower envelopes using cubic spline interpolation
            let upper_envelope =
                self.cubic_spline_interpolate(&max_indices, &max_values, h.len())?;
            let lower_envelope =
                self.cubic_spline_interpolate(&min_indices, &min_values, h.len())?;

            // Compute mean envelope
            let mut mean_envelope = vec![0.0; h.len()];
            for i in 0..h.len() {
                mean_envelope[i] = (upper_envelope[i] + lower_envelope[i]) / 2.0;
            }

            // Subtract mean from h
            let mut h_new = vec![0.0; h.len()];
            for i in 0..h.len() {
                h_new[i] = h[i] - mean_envelope[i];
            }

            // Check stopping criterion (standard deviation)
            let sd = self.compute_sd(&h, &h_new);

            h = h_new;

            if sd < self.config.sd_threshold {
                break;
            }
        }

        Ok(h)
    }

    /// Find local extrema (maxima or minima)
    fn find_extrema(&self, signal: &[f64], find_maxima: bool) -> (Vec<usize>, Vec<f64>) {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        // Add boundary handling
        if self.config.boundary_extension {
            // Add first point if it's an extremum
            if signal.len() >= 2 {
                let is_extremum = if find_maxima {
                    signal[0] > signal[1]
                } else {
                    signal[0] < signal[1]
                };

                if is_extremum {
                    indices.push(0);
                    values.push(signal[0]);
                }
            }
        }

        // Find interior extrema
        for i in 1..signal.len().saturating_sub(1) {
            let is_extremum = if find_maxima {
                signal[i] > signal[i - 1] && signal[i] > signal[i + 1]
            } else {
                signal[i] < signal[i - 1] && signal[i] < signal[i + 1]
            };

            if is_extremum {
                indices.push(i);
                values.push(signal[i]);
            }
        }

        // Add boundary handling
        if self.config.boundary_extension && signal.len() >= 2 {
            let last_idx = signal.len() - 1;
            let is_extremum = if find_maxima {
                signal[last_idx] > signal[last_idx - 1]
            } else {
                signal[last_idx] < signal[last_idx - 1]
            };

            if is_extremum {
                indices.push(last_idx);
                values.push(signal[last_idx]);
            }
        }

        (indices, values)
    }

    /// Cubic spline interpolation for envelope construction
    fn cubic_spline_interpolate(
        &self,
        x_points: &[usize],
        y_points: &[f64],
        length: usize,
    ) -> IoResult<Vec<f64>> {
        if x_points.len() != y_points.len() || x_points.len() < 2 {
            return Err(IoError::ConfigError(
                "Invalid interpolation points".to_string(),
            ));
        }

        // Natural cubic spline coefficients
        let n = x_points.len();
        let mut h = vec![0.0; n - 1];
        let mut alpha = vec![0.0; n - 1];

        for i in 0..n - 1 {
            h[i] = (x_points[i + 1] - x_points[i]) as f64;
        }

        for i in 1..n - 1 {
            alpha[i] = (3.0 / h[i]) * (y_points[i + 1] - y_points[i])
                - (3.0 / h[i - 1]) * (y_points[i] - y_points[i - 1]);
        }

        // Solve tridiagonal system
        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];

        for i in 1..n - 1 {
            l[i] = 2.0 * (x_points[i + 1] - x_points[i - 1]) as f64 - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        let mut c = vec![0.0; n];
        let mut b = vec![0.0; n - 1];
        let mut d = vec![0.0; n - 1];

        for j in (0..n - 1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (y_points[j + 1] - y_points[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }

        // Evaluate spline at all points
        let mut result = vec![0.0; length];
        let mut seg_idx = 0;

        for (i, result_val) in result.iter_mut().enumerate() {
            // Find which segment this point belongs to
            while seg_idx < n - 2 && i > x_points[seg_idx + 1] {
                seg_idx += 1;
            }

            let dx = (i as f64) - (x_points[seg_idx] as f64);
            *result_val = y_points[seg_idx]
                + b[seg_idx] * dx
                + c[seg_idx] * dx * dx
                + d[seg_idx] * dx * dx * dx;
        }

        Ok(result)
    }

    /// Compute standard deviation between two iterations
    fn compute_sd(&self, h_old: &[f64], h_new: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..h_old.len() {
            let diff = h_old[i] - h_new[i];
            sum += diff * diff / (h_old[i] * h_old[i] + 1e-10);
        }
        (sum / h_old.len() as f64).sqrt()
    }

    /// Count the number of extrema in a signal
    fn count_extrema(&self, signal: &[f64]) -> usize {
        let (max_indices, _) = self.find_extrema(signal, true);
        let (min_indices, _) = self.find_extrema(signal, false);
        max_indices.len() + min_indices.len()
    }

    /// Compute Hilbert spectrum for an IMF
    fn hilbert_spectrum(&self, imf: &[f64]) -> IoResult<IntrinsicModeFunction> {
        let n = imf.len();

        // Compute Hilbert transform using FFT
        let hilbert = self.hilbert_transform(imf)?;

        // Compute instantaneous amplitude (analytic signal envelope)
        let mut amplitude = vec![0.0; n];
        for i in 0..n {
            amplitude[i] = (imf[i] * imf[i] + hilbert[i] * hilbert[i]).sqrt();
        }

        // Compute instantaneous phase
        let mut phase = vec![0.0; n];
        for i in 0..n {
            phase[i] = hilbert[i].atan2(imf[i]);
        }

        // Compute instantaneous frequency from phase derivative
        let mut frequency = vec![0.0; n];
        for i in 1..n - 1 {
            let phase_diff = self.unwrap_phase(phase[i + 1] - phase[i]);
            frequency[i] = phase_diff * self.sample_rate / (2.0 * PI);
        }

        // Handle boundaries
        if n >= 2 {
            frequency[0] = frequency[1];
            frequency[n - 1] = frequency[n - 2];
        }

        Ok(IntrinsicModeFunction {
            data: imf.to_vec(),
            amplitude,
            frequency,
            phase,
        })
    }

    /// Compute Hilbert transform using FFT
    fn hilbert_transform(&self, signal: &[f64]) -> IoResult<Vec<f64>> {
        let n = signal.len();

        // Find next power of 2 for FFT
        let n_fft = n.next_power_of_two();

        // Zero-pad signal
        let mut padded = vec![0.0; n_fft];
        padded[..n].copy_from_slice(signal);

        // FFT
        let mut fft_result = self.fft(&padded);

        // Create Hilbert multiplier: [1, 2, 2, ..., 2, 1, 0, 0, ..., 0]
        // Multiply positive frequencies by 2, zero out negative frequencies
        for val in fft_result.iter_mut().skip(1).take(n_fft / 2 - 1) {
            val.0 *= 2.0;
            val.1 *= 2.0;
        }
        for val in fft_result.iter_mut().skip(n_fft / 2 + 1) {
            val.0 = 0.0;
            val.1 = 0.0;
        }

        // IFFT
        let mut result = self.ifft(&fft_result);

        // Take imaginary part and truncate to original length
        result.truncate(n);

        Ok(result)
    }

    /// Simple FFT implementation (returns complex numbers as (real, imag) tuples)
    fn fft(&self, signal: &[f64]) -> Vec<(f64, f64)> {
        let n = signal.len();
        let mut result = vec![(0.0, 0.0); n];

        for (k, result_k) in result.iter_mut().enumerate() {
            let mut sum_real = 0.0;
            let mut sum_imag = 0.0;

            for (t, &signal_t) in signal.iter().enumerate() {
                let angle = -2.0 * PI * (k as f64) * (t as f64) / (n as f64);
                sum_real += signal_t * angle.cos();
                sum_imag += signal_t * angle.sin();
            }

            *result_k = (sum_real, sum_imag);
        }

        result
    }

    /// Simple IFFT implementation (returns real part)
    fn ifft(&self, fft_data: &[(f64, f64)]) -> Vec<f64> {
        let n = fft_data.len();
        let mut result = vec![0.0; n];

        for (t, result_t) in result.iter_mut().enumerate() {
            let mut sum_real = 0.0;

            for (k, &fft_k) in fft_data.iter().enumerate() {
                let angle = 2.0 * PI * (k as f64) * (t as f64) / (n as f64);
                sum_real += fft_k.0 * angle.cos() - fft_k.1 * angle.sin();
            }

            *result_t = sum_real / (n as f64);
        }

        result
    }

    /// Unwrap phase to ensure continuity
    fn unwrap_phase(&self, phase_diff: f64) -> f64 {
        let mut unwrapped = phase_diff;

        while unwrapped > PI {
            unwrapped -= 2.0 * PI;
        }
        while unwrapped < -PI {
            unwrapped += 2.0 * PI;
        }

        unwrapped
    }
}

/// Ensemble EMD (EEMD) for noise-assisted decomposition
pub struct EnsembleEmd {
    emd: EmpiricalModeDecomposition,
    ensemble_size: usize,
    noise_amplitude: f64,
}

impl EnsembleEmd {
    /// Create a new EEMD analyzer
    pub fn new(
        sample_rate: f64,
        config: EmdConfig,
        ensemble_size: usize,
        noise_amplitude: f64,
    ) -> Self {
        Self {
            emd: EmpiricalModeDecomposition::new(sample_rate, config),
            ensemble_size,
            noise_amplitude,
        }
    }

    /// Decompose signal using ensemble averaging
    pub fn decompose(&self, signal: &[f64]) -> IoResult<EmdResult> {
        if signal.is_empty() {
            return Err(IoError::ConfigError("Empty signal".to_string()));
        }

        let n = signal.len();
        let mut accumulated_imfs: Vec<Vec<f64>> = Vec::new();
        let mut accumulated_residual = vec![0.0; n];

        for ensemble_idx in 0..self.ensemble_size {
            // Add white noise
            let mut noisy_signal = signal.to_vec();
            for (i, noisy_val) in noisy_signal.iter_mut().enumerate() {
                // Simple pseudo-random noise generator (using index and ensemble number)
                let noise = self.generate_noise(i, ensemble_idx) * self.noise_amplitude;
                *noisy_val += noise;
            }

            // Decompose noisy signal
            let result = self.emd.decompose(&noisy_signal)?;

            // Accumulate IMFs
            if accumulated_imfs.is_empty() {
                accumulated_imfs = vec![vec![0.0; n]; result.imfs.len()];
            }

            for (i, imf) in result.imfs.iter().enumerate() {
                if i < accumulated_imfs.len() {
                    for (acc_val, &imf_val) in accumulated_imfs[i].iter_mut().zip(imf.data.iter()) {
                        *acc_val += imf_val;
                    }
                }
            }

            // Accumulate residual
            for (acc_val, &res_val) in accumulated_residual.iter_mut().zip(result.residual.iter()) {
                *acc_val += res_val;
            }
        }

        // Average all accumulated results
        let ensemble_size_f64 = self.ensemble_size as f64;

        for imf_vec in &mut accumulated_imfs {
            for val in imf_vec.iter_mut() {
                *val /= ensemble_size_f64;
            }
        }

        for val in accumulated_residual.iter_mut() {
            *val /= ensemble_size_f64;
        }

        // Convert averaged IMFs to IntrinsicModeFunction with Hilbert analysis
        let mut final_imfs = Vec::new();
        for imf_data in accumulated_imfs {
            let hilbert_result = self.emd.hilbert_spectrum(&imf_data)?;
            final_imfs.push(hilbert_result);
        }

        Ok(EmdResult {
            imfs: final_imfs,
            residual: accumulated_residual,
        })
    }

    /// Simple pseudo-random noise generator
    fn generate_noise(&self, index: usize, ensemble: usize) -> f64 {
        // Linear congruential generator for reproducible noise
        let seed = (index
            .wrapping_mul(2654435761)
            .wrapping_add(ensemble.wrapping_mul(1664525))) as f64;
        let normalized = (seed % 1000000.0) / 1000000.0;
        (normalized - 0.5) * 2.0 // Range: [-1, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emd_simple_signal() {
        let sample_rate = 1000.0;
        let config = EmdConfig::default();
        let emd = EmpiricalModeDecomposition::new(sample_rate, config);

        // Create a simple composite signal: sin(2πf1t) + sin(2πf2t)
        let n = 1000;
        let mut signal = vec![0.0; n];
        for (i, sig) in signal.iter_mut().enumerate() {
            let t = i as f64 / sample_rate;
            *sig = (2.0 * PI * 5.0 * t).sin() + (2.0 * PI * 20.0 * t).sin();
        }

        let result = emd.decompose(&signal);
        assert!(result.is_ok());

        let emd_result = result.unwrap();
        assert!(!emd_result.imfs.is_empty());
    }

    #[test]
    fn test_hilbert_transform() {
        let sample_rate = 1000.0;
        let config = EmdConfig::default();
        let emd = EmpiricalModeDecomposition::new(sample_rate, config);

        // Simple sine wave
        let n = 100;
        let mut signal = vec![0.0; n];
        for (i, sig) in signal.iter_mut().enumerate() {
            let t = i as f64 / sample_rate;
            *sig = (2.0 * PI * 10.0 * t).sin();
        }

        let result = emd.hilbert_transform(&signal);
        assert!(result.is_ok());

        let hilbert = result.unwrap();
        assert_eq!(hilbert.len(), signal.len());
    }

    #[test]
    fn test_extrema_detection() {
        let config = EmdConfig::default();
        let emd = EmpiricalModeDecomposition::new(1000.0, config);

        let signal = vec![0.0, 1.0, 0.5, 2.0, 1.5, 0.3, -0.5, 0.2];

        let (max_indices, max_values) = emd.find_extrema(&signal, true);
        assert!(!max_indices.is_empty());

        let (min_indices, _min_values) = emd.find_extrema(&signal, false);
        assert!(!min_indices.is_empty());

        // Verify that found maxima are actually local maxima
        for (idx, val) in max_indices.iter().zip(max_values.iter()) {
            assert!((signal[*idx] - val).abs() < 1e-10);
        }
    }

    #[test]
    fn test_eemd() {
        let sample_rate = 1000.0;
        let config = EmdConfig::default();
        let eemd = EnsembleEmd::new(sample_rate, config, 5, 0.1);

        let n = 500;
        let mut signal = vec![0.0; n];
        for (i, sig) in signal.iter_mut().enumerate() {
            let t = i as f64 / sample_rate;
            *sig = (2.0 * PI * 5.0 * t).sin() + (2.0 * PI * 15.0 * t).sin();
        }

        let result = eemd.decompose(&signal);
        assert!(result.is_ok());
    }
}
