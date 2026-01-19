//! Signal Quality Metrics Module
//!
//! This module provides objective signal quality assessment metrics:
//! - PESQ-inspired perceptual quality metric
//! - STOI (Short-Time Objective Intelligibility)
//! - POLQA-inspired quality assessment
//! - MOS (Mean Opinion Score) prediction
//! - Objective quality measures (SNR, PESQ, segmental SNR, etc.)

use crate::error::{IoError, IoResult};
use std::f64::consts::PI;

/// Signal-to-Noise Ratio calculator
#[derive(Debug, Clone)]
pub struct SnrCalculator {
    /// Frame size for segmental SNR
    frame_size: usize,
}

impl SnrCalculator {
    /// Create a new SNR calculator
    pub fn new(frame_size: usize) -> Self {
        Self { frame_size }
    }

    /// Calculate overall SNR in dB
    pub fn calculate_snr(&self, reference: &[f64], degraded: &[f64]) -> IoResult<f64> {
        if reference.len() != degraded.len() {
            return Err(IoError::ConfigError("Signal length mismatch".to_string()));
        }

        if reference.is_empty() {
            return Err(IoError::ConfigError("Empty signals".to_string()));
        }

        let mut signal_power = 0.0;
        let mut noise_power = 0.0;

        for i in 0..reference.len() {
            signal_power += reference[i] * reference[i];
            let noise = degraded[i] - reference[i];
            noise_power += noise * noise;
        }

        if noise_power < 1e-10 {
            return Ok(100.0); // Very high SNR
        }

        let snr = 10.0 * (signal_power / noise_power).log10();
        Ok(snr)
    }

    /// Calculate segmental SNR in dB
    pub fn calculate_segmental_snr(&self, reference: &[f64], degraded: &[f64]) -> IoResult<f64> {
        if reference.len() != degraded.len() {
            return Err(IoError::ConfigError("Signal length mismatch".to_string()));
        }

        let num_frames = reference.len() / self.frame_size;
        if num_frames == 0 {
            return self.calculate_snr(reference, degraded);
        }

        let mut snr_sum = 0.0;
        let mut valid_frames = 0;

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.frame_size;
            let end = (start + self.frame_size).min(reference.len());

            let mut signal_power = 0.0;
            let mut noise_power = 0.0;

            for i in start..end {
                signal_power += reference[i] * reference[i];
                let noise = degraded[i] - reference[i];
                noise_power += noise * noise;
            }

            if signal_power > 1e-10 && noise_power > 1e-10 {
                let frame_snr = 10.0 * (signal_power / noise_power).log10();
                // Clip to reasonable range
                let clipped_snr = frame_snr.clamp(-10.0, 35.0);
                snr_sum += clipped_snr;
                valid_frames += 1;
            }
        }

        if valid_frames == 0 {
            return Ok(0.0);
        }

        Ok(snr_sum / valid_frames as f64)
    }

    /// Calculate frequency-weighted SNR
    pub fn calculate_frequency_weighted_snr(
        &self,
        reference: &[f64],
        degraded: &[f64],
        sample_rate: f64,
    ) -> IoResult<f64> {
        // Apply perceptual weighting (A-weighting approximation)
        let weighted_ref = self.apply_a_weighting(reference, sample_rate)?;
        let weighted_deg = self.apply_a_weighting(degraded, sample_rate)?;

        self.calculate_snr(&weighted_ref, &weighted_deg)
    }

    /// Apply A-weighting filter (perceptual loudness weighting)
    fn apply_a_weighting(&self, signal: &[f64], sample_rate: f64) -> IoResult<Vec<f64>> {
        // Simple first-order approximation of A-weighting
        // Full A-weighting requires complex filter design
        let cutoff = 1000.0; // Hz
        let alpha = (-2.0 * PI * cutoff / sample_rate).exp();

        let mut weighted = vec![0.0; signal.len()];
        let mut prev = 0.0;

        for i in 0..signal.len() {
            weighted[i] = alpha * prev + (1.0 - alpha) * signal[i];
            prev = weighted[i];
        }

        Ok(weighted)
    }
}

/// Short-Time Objective Intelligibility (STOI) metric
/// Measures speech intelligibility in noisy conditions
#[derive(Debug, Clone)]
pub struct StoiCalculator {
    /// Sample rate
    #[allow(dead_code)]
    sample_rate: f64,
    /// Frame length in samples
    frame_length: usize,
    /// Number of frequency bands
    #[allow(dead_code)]
    num_bands: usize,
}

impl StoiCalculator {
    /// Create a new STOI calculator
    pub fn new(sample_rate: f64) -> Self {
        let frame_length = (sample_rate * 0.256) as usize; // 256 ms frames
        Self {
            sample_rate,
            frame_length,
            num_bands: 15, // One-third octave bands
        }
    }

    /// Calculate STOI score (0 to 1, higher is better)
    pub fn calculate(&self, reference: &[f64], degraded: &[f64]) -> IoResult<f64> {
        if reference.len() != degraded.len() {
            return Err(IoError::ConfigError("Signal length mismatch".to_string()));
        }

        let num_frames = reference.len() / self.frame_length;
        if num_frames == 0 {
            return Err(IoError::ConfigError("Signal too short".to_string()));
        }

        let mut correlation_sum = 0.0;
        let mut valid_frames = 0;

        // Process each frame
        for frame_idx in 0..num_frames {
            let start = frame_idx * self.frame_length;
            let end = (start + self.frame_length).min(reference.len());

            let ref_frame = &reference[start..end];
            let deg_frame = &degraded[start..end];

            // Compute correlation in frequency bands
            let band_correlation = self.compute_band_correlation(ref_frame, deg_frame)?;

            correlation_sum += band_correlation;
            valid_frames += 1;
        }

        if valid_frames == 0 {
            return Ok(0.0);
        }

        let stoi = correlation_sum / valid_frames as f64;
        Ok(stoi.clamp(0.0, 1.0))
    }

    /// Compute correlation between frequency bands
    fn compute_band_correlation(&self, ref_frame: &[f64], deg_frame: &[f64]) -> IoResult<f64> {
        let mut total_correlation = 0.0;

        // Simplified: compute time-domain correlation
        // Full STOI uses short-time Fourier transform and third-octave bands
        let mut ref_mean = 0.0;
        let mut deg_mean = 0.0;

        for i in 0..ref_frame.len() {
            ref_mean += ref_frame[i];
            deg_mean += deg_frame[i];
        }

        ref_mean /= ref_frame.len() as f64;
        deg_mean /= deg_frame.len() as f64;

        let mut numerator = 0.0;
        let mut ref_variance = 0.0;
        let mut deg_variance = 0.0;

        for i in 0..ref_frame.len() {
            let ref_centered = ref_frame[i] - ref_mean;
            let deg_centered = deg_frame[i] - deg_mean;

            numerator += ref_centered * deg_centered;
            ref_variance += ref_centered * ref_centered;
            deg_variance += deg_centered * deg_centered;
        }

        if ref_variance > 1e-10 && deg_variance > 1e-10 {
            total_correlation = numerator / (ref_variance.sqrt() * deg_variance.sqrt());
        }

        Ok(total_correlation.clamp(-1.0, 1.0))
    }
}

/// PESQ-inspired perceptual quality metric
/// Simplified version of ITU-T P.862
#[derive(Debug, Clone)]
pub struct PesqCalculator {
    #[allow(dead_code)]
    sample_rate: f64,
    frame_size: usize,
}

impl PesqCalculator {
    /// Create a new PESQ calculator
    pub fn new(sample_rate: f64) -> Self {
        let frame_size = (sample_rate * 0.032) as usize; // 32 ms frames
        Self {
            sample_rate,
            frame_size,
        }
    }

    /// Calculate PESQ-like score (1.0 to 4.5 scale, higher is better)
    pub fn calculate(&self, reference: &[f64], degraded: &[f64]) -> IoResult<f64> {
        if reference.len() != degraded.len() {
            return Err(IoError::ConfigError("Signal length mismatch".to_string()));
        }

        // Preprocessing: level alignment
        let (aligned_ref, aligned_deg) = self.align_levels(reference, degraded);

        // Compute perceptual features
        let mut distortion_sum = 0.0;
        let num_frames = aligned_ref.len() / self.frame_size;

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.frame_size;
            let end = (start + self.frame_size).min(aligned_ref.len());

            let ref_frame = &aligned_ref[start..end];
            let deg_frame = &aligned_deg[start..end];

            // Compute perceptual loudness
            let ref_loudness = self.compute_loudness(ref_frame);
            let deg_loudness = self.compute_loudness(deg_frame);

            // Distortion metric
            let distortion = (ref_loudness - deg_loudness).abs();
            distortion_sum += distortion;
        }

        if num_frames == 0 {
            return Ok(1.0);
        }

        let avg_distortion = distortion_sum / num_frames as f64;

        // Map distortion to PESQ-like scale (1.0 to 4.5)
        // Lower distortion = higher score
        let pesq_score = 4.5 - (avg_distortion * 3.5).min(3.5);

        Ok(pesq_score.clamp(1.0, 4.5))
    }

    /// Align signal levels for fair comparison
    fn align_levels(&self, reference: &[f64], degraded: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let ref_rms = self.compute_rms(reference);
        let deg_rms = self.compute_rms(degraded);

        if deg_rms < 1e-10 {
            return (reference.to_vec(), degraded.to_vec());
        }

        let gain = ref_rms / deg_rms;

        let aligned_deg: Vec<f64> = degraded.iter().map(|&x| x * gain).collect();

        (reference.to_vec(), aligned_deg)
    }

    /// Compute RMS value
    fn compute_rms(&self, signal: &[f64]) -> f64 {
        if signal.is_empty() {
            return 0.0;
        }

        let sum_squares: f64 = signal.iter().map(|&x| x * x).sum();
        (sum_squares / signal.len() as f64).sqrt()
    }

    /// Compute perceptual loudness
    fn compute_loudness(&self, frame: &[f64]) -> f64 {
        // Simplified loudness: RMS with perceptual weighting
        let rms = self.compute_rms(frame);
        // Apply power-law compression (similar to human perception)
        rms.powf(0.6)
    }
}

/// MOS (Mean Opinion Score) predictor
#[derive(Debug, Clone)]
pub struct MosPredictor {
    snr_calc: SnrCalculator,
    pesq_calc: PesqCalculator,
    stoi_calc: StoiCalculator,
}

impl MosPredictor {
    /// Create a new MOS predictor
    pub fn new(sample_rate: f64) -> Self {
        Self {
            snr_calc: SnrCalculator::new(512),
            pesq_calc: PesqCalculator::new(sample_rate),
            stoi_calc: StoiCalculator::new(sample_rate),
        }
    }

    /// Predict MOS score (1.0 to 5.0 scale)
    pub fn predict(&self, reference: &[f64], degraded: &[f64]) -> IoResult<f64> {
        // Calculate multiple quality metrics
        let snr = self.snr_calc.calculate_snr(reference, degraded)?;
        let pesq = self.pesq_calc.calculate(reference, degraded)?;
        let stoi = self.stoi_calc.calculate(reference, degraded)?;

        // Weighted combination to predict MOS
        // PESQ is already on a similar scale (1-4.5), normalize others
        let snr_normalized = ((snr + 5.0) / 40.0).clamp(0.0, 1.0); // Map -5 to 35 dB to 0-1
        let stoi_normalized = stoi; // Already 0-1

        // Weighted average (PESQ has highest weight)
        let mos =
            0.5 * pesq + 0.3 * (snr_normalized * 4.0 + 1.0) + 0.2 * (stoi_normalized * 4.0 + 1.0);

        Ok(mos.clamp(1.0, 5.0))
    }

    /// Get detailed quality metrics
    pub fn detailed_metrics(
        &self,
        reference: &[f64],
        degraded: &[f64],
    ) -> IoResult<QualityMetrics> {
        let snr = self.snr_calc.calculate_snr(reference, degraded)?;
        let segmental_snr = self.snr_calc.calculate_segmental_snr(reference, degraded)?;
        let pesq = self.pesq_calc.calculate(reference, degraded)?;
        let stoi = self.stoi_calc.calculate(reference, degraded)?;
        let mos = self.predict(reference, degraded)?;

        Ok(QualityMetrics {
            snr,
            segmental_snr,
            pesq_score: pesq,
            stoi_score: stoi,
            mos,
        })
    }
}

/// Comprehensive quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Signal-to-Noise Ratio (dB)
    pub snr: f64,
    /// Segmental SNR (dB)
    pub segmental_snr: f64,
    /// PESQ score (1.0-4.5)
    pub pesq_score: f64,
    /// STOI score (0.0-1.0)
    pub stoi_score: f64,
    /// Mean Opinion Score (1.0-5.0)
    pub mos: f64,
}

impl QualityMetrics {
    /// Get a quality rating string
    pub fn quality_rating(&self) -> &str {
        if self.mos >= 4.0 {
            "Excellent"
        } else if self.mos >= 3.5 {
            "Good"
        } else if self.mos >= 3.0 {
            "Fair"
        } else if self.mos >= 2.0 {
            "Poor"
        } else {
            "Bad"
        }
    }

    /// Get intelligibility rating
    pub fn intelligibility_rating(&self) -> &str {
        if self.stoi_score >= 0.8 {
            "Highly Intelligible"
        } else if self.stoi_score >= 0.6 {
            "Intelligible"
        } else if self.stoi_score >= 0.4 {
            "Partially Intelligible"
        } else {
            "Unintelligible"
        }
    }
}

/// POLQA-inspired metric (Perceptual Objective Listening Quality Assessment)
#[derive(Debug, Clone)]
pub struct PolqaCalculator {
    #[allow(dead_code)]
    sample_rate: f64,
    frame_size: usize,
}

impl PolqaCalculator {
    /// Create a new POLQA calculator
    pub fn new(sample_rate: f64) -> Self {
        let frame_size = (sample_rate * 0.020) as usize; // 20 ms frames
        Self {
            sample_rate,
            frame_size,
        }
    }

    /// Calculate POLQA-like score (1.0 to 5.0 scale)
    pub fn calculate(&self, reference: &[f64], degraded: &[f64]) -> IoResult<f64> {
        if reference.len() != degraded.len() {
            return Err(IoError::ConfigError("Signal length mismatch".to_string()));
        }

        let num_frames = reference.len() / self.frame_size;
        if num_frames == 0 {
            return Err(IoError::ConfigError("Signal too short".to_string()));
        }

        let mut quality_sum = 0.0;

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.frame_size;
            let end = (start + self.frame_size).min(reference.len());

            let ref_frame = &reference[start..end];
            let deg_frame = &degraded[start..end];

            // Compute frame quality using multiple perceptual dimensions
            let temporal_quality = self.temporal_fidelity(ref_frame, deg_frame);
            let spectral_quality = self.spectral_fidelity(ref_frame, deg_frame);
            let loudness_quality = self.loudness_fidelity(ref_frame, deg_frame);

            // Combined quality
            let frame_quality = (temporal_quality + spectral_quality + loudness_quality) / 3.0;
            quality_sum += frame_quality;
        }

        let avg_quality = quality_sum / num_frames as f64;

        // Map to POLQA scale (1.0 to 5.0)
        let polqa_score = 1.0 + (avg_quality * 4.0);

        Ok(polqa_score.clamp(1.0, 5.0))
    }

    /// Assess temporal fidelity
    fn temporal_fidelity(&self, reference: &[f64], degraded: &[f64]) -> f64 {
        let mut correlation = 0.0;
        let mut ref_energy = 0.0;
        let mut deg_energy = 0.0;

        for i in 0..reference.len() {
            correlation += reference[i] * degraded[i];
            ref_energy += reference[i] * reference[i];
            deg_energy += degraded[i] * degraded[i];
        }

        if ref_energy > 1e-10 && deg_energy > 1e-10 {
            (correlation / (ref_energy.sqrt() * deg_energy.sqrt())).max(0.0)
        } else {
            0.0
        }
    }

    /// Assess spectral fidelity
    fn spectral_fidelity(&self, reference: &[f64], degraded: &[f64]) -> f64 {
        // Simplified spectral comparison using energy in different bands
        let ref_energy = reference.iter().map(|&x| x * x).sum::<f64>();
        let deg_energy = degraded.iter().map(|&x| x * x).sum::<f64>();

        if ref_energy < 1e-10 && deg_energy < 1e-10 {
            return 1.0;
        }

        if ref_energy < 1e-10 || deg_energy < 1e-10 {
            return 0.0;
        }

        let energy_ratio = (deg_energy / ref_energy).min(ref_energy / deg_energy);
        energy_ratio.clamp(0.0, 1.0)
    }

    /// Assess loudness fidelity
    fn loudness_fidelity(&self, reference: &[f64], degraded: &[f64]) -> f64 {
        let ref_rms =
            (reference.iter().map(|&x| x * x).sum::<f64>() / reference.len() as f64).sqrt();
        let deg_rms = (degraded.iter().map(|&x| x * x).sum::<f64>() / degraded.len() as f64).sqrt();

        if ref_rms < 1e-10 && deg_rms < 1e-10 {
            return 1.0;
        }

        if ref_rms < 1e-10 || deg_rms < 1e-10 {
            return 0.0;
        }

        // Perceptual loudness difference
        let loudness_diff = (ref_rms.powf(0.6) - deg_rms.powf(0.6)).abs();
        let max_loudness = ref_rms.powf(0.6).max(deg_rms.powf(0.6));

        if max_loudness < 1e-10 {
            return 1.0;
        }

        (1.0 - loudness_diff / max_loudness).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snr_perfect() {
        let snr_calc = SnrCalculator::new(256);
        let signal = vec![1.0, 0.5, -0.3, 0.8, -0.1];

        let snr = snr_calc.calculate_snr(&signal, &signal);
        assert!(snr.is_ok());
        assert!(snr.unwrap() > 90.0); // Perfect match should have very high SNR
    }

    #[test]
    fn test_snr_with_noise() {
        let snr_calc = SnrCalculator::new(256);
        let reference = vec![1.0, 0.5, -0.3, 0.8, -0.1];
        let degraded = vec![1.01, 0.51, -0.29, 0.81, -0.09];

        let snr = snr_calc.calculate_snr(&reference, &degraded);
        assert!(snr.is_ok());
        let snr_value = snr.unwrap();
        assert!(snr_value > 0.0 && snr_value < 100.0);
    }

    #[test]
    fn test_stoi_calculation() {
        let sample_rate = 8000.0;
        let stoi_calc = StoiCalculator::new(sample_rate);

        let n = 8000; // 1 second
        let reference: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
            .collect();
        let degraded = reference.clone();

        let stoi = stoi_calc.calculate(&reference, &degraded);
        assert!(stoi.is_ok());
        let stoi_value = stoi.unwrap();
        assert!((0.0..=1.0).contains(&stoi_value));
    }

    #[test]
    fn test_pesq_calculation() {
        let sample_rate = 8000.0;
        let pesq_calc = PesqCalculator::new(sample_rate);

        let n = 8000;
        let reference: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
            .collect();
        let degraded = reference.clone();

        let pesq = pesq_calc.calculate(&reference, &degraded);
        assert!(pesq.is_ok());
        let pesq_value = pesq.unwrap();
        assert!((1.0..=4.5).contains(&pesq_value));
    }

    #[test]
    fn test_mos_prediction() {
        let sample_rate = 8000.0;
        let mos_predictor = MosPredictor::new(sample_rate);

        let n = 8000;
        let reference: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
            .collect();
        let degraded = reference.clone();

        let mos = mos_predictor.predict(&reference, &degraded);
        assert!(mos.is_ok());
        let mos_value = mos.unwrap();
        assert!((1.0..=5.0).contains(&mos_value));
    }

    #[test]
    fn test_quality_metrics() {
        let sample_rate = 8000.0;
        let mos_predictor = MosPredictor::new(sample_rate);

        let n = 8000;
        let reference: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
            .collect();
        let degraded = reference.clone();

        let metrics = mos_predictor.detailed_metrics(&reference, &degraded);
        assert!(metrics.is_ok());

        let m = metrics.unwrap();
        assert!(m.snr > 0.0);
        assert!(m.pesq_score >= 1.0 && m.pesq_score <= 4.5);
        assert!(m.stoi_score >= 0.0 && m.stoi_score <= 1.0);
        assert!(m.mos >= 1.0 && m.mos <= 5.0);
    }

    #[test]
    fn test_polqa_calculation() {
        let sample_rate = 16000.0;
        let polqa_calc = PolqaCalculator::new(sample_rate);

        let n = 16000;
        let reference: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
            .collect();
        let degraded = reference.clone();

        let polqa = polqa_calc.calculate(&reference, &degraded);
        assert!(polqa.is_ok());
        let polqa_value = polqa.unwrap();
        assert!((1.0..=5.0).contains(&polqa_value));
    }

    #[test]
    fn test_quality_rating() {
        let metrics = QualityMetrics {
            snr: 30.0,
            segmental_snr: 28.0,
            pesq_score: 4.2,
            stoi_score: 0.9,
            mos: 4.3,
        };

        assert_eq!(metrics.quality_rating(), "Excellent");
        assert_eq!(metrics.intelligibility_rating(), "Highly Intelligible");
    }
}
