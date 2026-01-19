//! Cepstral analysis for speech and audio processing
//!
//! This module provides quefrency domain processing including:
//! - Real cepstrum for pitch detection and formant analysis
//! - Complex cepstrum for homomorphic deconvolution
//! - Liftering (cepstral windowing)
//! - Formant tracking and analysis
//! - Pitch detection using cepstrum

use crate::error::{IoError, IoResult};
use rustfft::{num_complex::Complex32, FftPlanner};
use scirs2_core::ndarray::Array1;
use std::f32::consts::PI;

/// Real cepstrum analyzer
///
/// The real cepstrum is the inverse FFT of the log magnitude spectrum.
/// Useful for pitch detection and formant analysis.
pub struct RealCepstrum {
    /// FFT planner
    planner: FftPlanner<f32>,
    /// Minimum value for logarithm (prevents log(0))
    min_log: f32,
}

impl RealCepstrum {
    /// Create a new real cepstrum analyzer
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
            min_log: 1e-10,
        }
    }

    /// Compute real cepstrum of a signal
    ///
    /// # Arguments
    /// * `signal` - Input signal (should be windowed)
    ///
    /// # Returns
    /// Cepstrum coefficients (quefrency domain)
    pub fn compute(&mut self, signal: &Array1<f32>) -> IoResult<Array1<f32>> {
        let n = signal.len();

        // Forward FFT
        let mut buffer: Vec<Complex32> = signal.iter().map(|&x| Complex32::new(x, 0.0)).collect();

        let fft = self.planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        // Compute log magnitude
        for sample in buffer.iter_mut() {
            let mag = sample.norm().max(self.min_log);
            *sample = Complex32::new(mag.ln(), 0.0);
        }

        // Inverse FFT
        let ifft = self.planner.plan_fft_inverse(n);
        ifft.process(&mut buffer);

        // Extract real part and normalize
        let cepstrum = buffer.iter().map(|c| c.re / n as f32).collect::<Vec<_>>();

        Ok(Array1::from_vec(cepstrum))
    }

    /// Apply liftering (cepstral windowing) to emphasize formants
    ///
    /// # Arguments
    /// * `cepstrum` - Input cepstrum
    /// * `lifter_coeff` - Liftering coefficient (typically 22 for speech)
    pub fn lifter(&self, cepstrum: &Array1<f32>, lifter_coeff: f32) -> Array1<f32> {
        let _n = cepstrum.len();
        let mut liftered = cepstrum.clone();

        for (i, coeff) in liftered.iter_mut().enumerate() {
            let lift = 1.0 + (lifter_coeff / 2.0) * (PI * i as f32 / lifter_coeff).sin();
            *coeff *= lift;
        }

        liftered
    }

    /// Detect pitch period from cepstrum
    ///
    /// # Arguments
    /// * `cepstrum` - Input cepstrum
    /// * `sample_rate` - Signal sample rate
    /// * `min_f0` - Minimum expected pitch frequency (Hz)
    /// * `max_f0` - Maximum expected pitch frequency (Hz)
    ///
    /// # Returns
    /// Detected pitch period in samples, or None if no clear pitch
    pub fn detect_pitch(
        &self,
        cepstrum: &Array1<f32>,
        sample_rate: f32,
        min_f0: f32,
        max_f0: f32,
    ) -> Option<f32> {
        let min_period = (sample_rate / max_f0) as usize;
        let max_period = (sample_rate / min_f0) as usize;

        if max_period >= cepstrum.len() {
            return None;
        }

        // Find peak in cepstrum (skip low quefrencies)
        let search_range = &cepstrum
            .as_slice()
            .expect("Array must have contiguous layout")
            [min_period..max_period.min(cepstrum.len())];

        let (max_idx, &max_val) = search_range
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;

        let period = (min_period + max_idx) as f32;

        // Check if peak is significant (simple threshold)
        if max_val > 0.1 {
            Some(period)
        } else {
            None
        }
    }

    /// Convert pitch period to frequency
    pub fn period_to_frequency(&self, period: f32, sample_rate: f32) -> f32 {
        sample_rate / period
    }
}

impl Default for RealCepstrum {
    fn default() -> Self {
        Self::new()
    }
}

/// Complex cepstrum analyzer
///
/// The complex cepstrum is used for homomorphic deconvolution
/// and minimum-phase signal analysis.
pub struct ComplexCepstrum {
    /// FFT planner
    planner: FftPlanner<f32>,
    /// Minimum value for logarithm
    min_log: f32,
}

impl ComplexCepstrum {
    /// Create a new complex cepstrum analyzer
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
            min_log: 1e-10,
        }
    }

    /// Compute complex cepstrum using unwrapped phase
    ///
    /// # Arguments
    /// * `signal` - Input signal
    ///
    /// # Returns
    /// Complex cepstrum coefficients
    pub fn compute(&mut self, signal: &Array1<f32>) -> IoResult<Array1<f32>> {
        let n = signal.len();

        // Forward FFT
        let mut buffer: Vec<Complex32> = signal.iter().map(|&x| Complex32::new(x, 0.0)).collect();

        let fft = self.planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        // Compute log spectrum (magnitude + phase)
        let mut log_spectrum = Vec::with_capacity(n);
        for sample in &buffer {
            let mag = sample.norm().max(self.min_log);
            let phase = sample.arg();
            log_spectrum.push(Complex32::new(mag.ln(), phase));
        }

        // Phase unwrapping (simple algorithm)
        let unwrapped_phase = Self::unwrap_phase(&log_spectrum);

        // Create complex log spectrum
        for (i, sample) in log_spectrum.iter_mut().enumerate() {
            *sample = Complex32::new(sample.re, unwrapped_phase[i]);
        }

        // Inverse FFT
        let mut buffer = log_spectrum;
        let ifft = self.planner.plan_fft_inverse(n);
        ifft.process(&mut buffer);

        // Extract real part
        let cepstrum = buffer.iter().map(|c| c.re / n as f32).collect::<Vec<_>>();

        Ok(Array1::from_vec(cepstrum))
    }

    /// Phase unwrapping algorithm
    fn unwrap_phase(spectrum: &[Complex32]) -> Vec<f32> {
        let mut unwrapped = Vec::with_capacity(spectrum.len());
        let mut cumulative = 0.0;

        for i in 0..spectrum.len() {
            let phase = spectrum[i].im;

            if i > 0 {
                let diff = phase - spectrum[i - 1].im;
                // Detect phase wraps
                if diff > PI {
                    cumulative -= 2.0 * PI;
                } else if diff < -PI {
                    cumulative += 2.0 * PI;
                }
            }

            unwrapped.push(phase + cumulative);
        }

        unwrapped
    }

    /// Compute minimum phase signal from cepstrum
    ///
    /// Useful for creating a minimum-phase version of a signal
    pub fn minimum_phase(&mut self, cepstrum: &Array1<f32>) -> IoResult<Array1<f32>> {
        let n = cepstrum.len();
        let mut min_phase_cep = cepstrum.clone();

        // Apply minimum phase condition: zero out negative quefrencies
        // and double positive quefrencies (except DC)
        for (i, coeff) in min_phase_cep.iter_mut().enumerate() {
            if i == 0 {
                // DC component unchanged
            } else if i < n / 2 {
                // Positive quefrencies doubled
                *coeff *= 2.0;
            } else {
                // Negative quefrencies zeroed
                *coeff = 0.0;
            }
        }

        Ok(min_phase_cep)
    }
}

impl Default for ComplexCepstrum {
    fn default() -> Self {
        Self::new()
    }
}

/// Formant tracker for speech analysis
///
/// Tracks resonant frequencies (formants) in speech signals
pub struct FormantTracker {
    /// Cepstrum analyzer
    cepstrum: RealCepstrum,
    /// Sample rate
    sample_rate: f32,
    /// Liftering coefficient
    lifter_coeff: f32,
}

impl FormantTracker {
    /// Create a new formant tracker
    ///
    /// # Arguments
    /// * `sample_rate` - Signal sample rate
    /// * `lifter_coeff` - Liftering coefficient (default 22)
    pub fn new(sample_rate: f32, lifter_coeff: Option<f32>) -> Self {
        Self {
            cepstrum: RealCepstrum::new(),
            sample_rate,
            lifter_coeff: lifter_coeff.unwrap_or(22.0),
        }
    }

    /// Estimate formant frequencies from a frame
    ///
    /// # Arguments
    /// * `frame` - Windowed speech frame
    /// * `num_formants` - Number of formants to extract (typically 3-5)
    ///
    /// # Returns
    /// Vector of formant frequencies in Hz
    pub fn estimate_formants(
        &mut self,
        frame: &Array1<f32>,
        num_formants: usize,
    ) -> IoResult<Vec<f32>> {
        // Compute cepstrum
        let cepstrum = self.cepstrum.compute(frame)?;

        // Apply liftering to emphasize formants
        let liftered = self.cepstrum.lifter(&cepstrum, self.lifter_coeff);

        // Convert back to frequency domain via FFT
        let mut buffer: Vec<Complex32> = liftered.iter().map(|&x| Complex32::new(x, 0.0)).collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(buffer.len());
        fft.process(&mut buffer);

        // Compute magnitude spectrum
        let n = buffer.len();
        let spectrum: Vec<f32> = buffer.iter().take(n / 2).map(|c| c.norm()).collect();

        // Find peaks in the spectrum (formants)
        let mut formants = Vec::new();
        let mut prev = 0.0;
        let mut prev_prev = 0.0;

        for (i, &val) in spectrum.iter().enumerate().skip(2) {
            // Peak detection: check if current is greater than neighbors
            if prev > prev_prev && prev > val && prev > 0.01 {
                let freq = (i - 1) as f32 * self.sample_rate / n as f32;

                // Formants typically in 200-5000 Hz range for speech
                if (200.0..5000.0).contains(&freq) {
                    formants.push(freq);
                    if formants.len() >= num_formants {
                        break;
                    }
                }
            }
            prev_prev = prev;
            prev = val;
        }

        Ok(formants)
    }
}

/// Cepstral distance measure for speech quality assessment
#[derive(Debug)]
pub struct CepstralDistance;

impl CepstralDistance {
    /// Compute cepstral distance between two signals
    ///
    /// Used for speech quality assessment and speaker verification
    ///
    /// # Arguments
    /// * `cepstrum1` - First cepstrum
    /// * `cepstrum2` - Second cepstrum
    ///
    /// # Returns
    /// Cepstral distance (lower is more similar)
    pub fn compute(cepstrum1: &Array1<f32>, cepstrum2: &Array1<f32>) -> IoResult<f32> {
        if cepstrum1.len() != cepstrum2.len() {
            return Err(IoError::SignalError("Cepstra must have same length".into()));
        }

        let dist: f32 = cepstrum1
            .iter()
            .zip(cepstrum2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        Ok(dist)
    }

    /// Compute weighted cepstral distance
    ///
    /// Applies weights to different cepstral coefficients (lower coefficients
    /// typically more important)
    pub fn compute_weighted(
        cepstrum1: &Array1<f32>,
        cepstrum2: &Array1<f32>,
        weights: &Array1<f32>,
    ) -> IoResult<f32> {
        if cepstrum1.len() != cepstrum2.len() || cepstrum1.len() != weights.len() {
            return Err(IoError::SignalError(
                "All arrays must have same length".into(),
            ));
        }

        let dist: f32 = cepstrum1
            .iter()
            .zip(cepstrum2.iter())
            .zip(weights.iter())
            .map(|((&a, &b), &w)| w * (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        Ok(dist)
    }
}

/// Quefrency domain filter
///
/// Filters signal in the cepstral (quefrency) domain for
/// pitch modification and formant shifting.
pub struct QuefrencyFilter {
    /// Cepstrum analyzer
    cepstrum: RealCepstrum,
    /// FFT planner for reconstruction
    planner: FftPlanner<f32>,
}

impl QuefrencyFilter {
    /// Create a new quefrency domain filter
    pub fn new() -> Self {
        Self {
            cepstrum: RealCepstrum::new(),
            planner: FftPlanner::new(),
        }
    }

    /// Low-pass lifter (low quefrency pass)
    ///
    /// Retains low quefrency components (formant information)
    /// and removes high quefrency (pitch information)
    pub fn lowpass_lifter(
        &mut self,
        signal: &Array1<f32>,
        cutoff_quefrency: usize,
    ) -> IoResult<Array1<f32>> {
        let mut cepstrum = self.cepstrum.compute(signal)?;

        // Zero out high quefrencies
        for i in cutoff_quefrency..cepstrum.len() {
            cepstrum[i] = 0.0;
        }

        self.reconstruct(&cepstrum)
    }

    /// High-pass lifter (high quefrency pass)
    ///
    /// Retains pitch information and removes formant structure
    pub fn highpass_lifter(
        &mut self,
        signal: &Array1<f32>,
        cutoff_quefrency: usize,
    ) -> IoResult<Array1<f32>> {
        let mut cepstrum = self.cepstrum.compute(signal)?;

        // Zero out low quefrencies
        for i in 0..cutoff_quefrency {
            cepstrum[i] = 0.0;
        }

        self.reconstruct(&cepstrum)
    }

    /// Reconstruct signal from cepstrum
    fn reconstruct(&mut self, cepstrum: &Array1<f32>) -> IoResult<Array1<f32>> {
        let n = cepstrum.len();

        // Forward FFT of cepstrum
        let mut buffer: Vec<Complex32> = cepstrum.iter().map(|&x| Complex32::new(x, 0.0)).collect();

        let fft = self.planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        // Exponentiate to get spectrum
        for sample in buffer.iter_mut() {
            let mag = sample.re.exp();
            *sample = Complex32::new(mag, 0.0);
        }

        // Inverse FFT to get signal
        let ifft = self.planner.plan_fft_inverse(n);
        ifft.process(&mut buffer);

        let signal = buffer.iter().map(|c| c.re / n as f32).collect::<Vec<_>>();

        Ok(Array1::from_vec(signal))
    }
}

impl Default for QuefrencyFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_real_cepstrum() {
        let mut analyzer = RealCepstrum::new();

        // Create a simple test signal (impulse)
        let signal = arr1(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let cepstrum = analyzer.compute(&signal).unwrap();

        assert_eq!(cepstrum.len(), signal.len());
    }

    #[test]
    fn test_liftering() {
        let analyzer = RealCepstrum::new();
        let cepstrum = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let liftered = analyzer.lifter(&cepstrum, 22.0);

        assert_eq!(liftered.len(), cepstrum.len());
        // Liftered values should be different from original
        assert_ne!(liftered[1], cepstrum[1]);
    }

    #[test]
    fn test_complex_cepstrum() {
        let mut analyzer = ComplexCepstrum::new();

        let signal = arr1(&[1.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0]);
        let cepstrum = analyzer.compute(&signal).unwrap();

        assert_eq!(cepstrum.len(), signal.len());
    }

    #[test]
    fn test_cepstral_distance() {
        let cep1 = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let cep2 = arr1(&[1.1, 2.1, 3.1, 4.1]);

        let dist = CepstralDistance::compute(&cep1, &cep2).unwrap();

        assert!(dist > 0.0);
        assert!(dist < 1.0); // Should be small for similar signals
    }

    #[test]
    fn test_formant_tracker() {
        let mut tracker = FormantTracker::new(16000.0, None);

        // Create a simple test signal
        let signal = arr1(&[1.0; 256]);

        let formants = tracker.estimate_formants(&signal, 3);

        // Should return result (may be empty for constant signal)
        assert!(formants.is_ok());
    }

    #[test]
    fn test_quefrency_filter() {
        let mut filter = QuefrencyFilter::new();

        let signal = arr1(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let filtered = filter.lowpass_lifter(&signal, 4);
        assert!(filtered.is_ok());

        let filtered = filter.highpass_lifter(&signal, 2);
        assert!(filtered.is_ok());
    }
}
