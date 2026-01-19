//! Quality metrics for evaluating tokenizer performance
//!
//! This module provides comprehensive metrics for assessing the quality
//! of signal reconstruction, compression efficiency, and rate-distortion tradeoffs.
//!
//! # Metric Categories
//!
//! - **Distortion Metrics**: MSE, MAE, RMSE, SNR, PSNR
//! - **Spectral Metrics**: Spectral convergence, magnitude error
//! - **Compression Metrics**: Compression ratio, bits per sample
//! - **Rate-Distortion**: RD curves, efficiency analysis

use crate::error::{TokenizerError, TokenizerResult};
use scirs2_core::ndarray::Array1;
use std::f32::consts::PI;

/// Comprehensive quality metrics for signal reconstruction
#[derive(Debug, Clone, PartialEq)]
pub struct QualityMetrics {
    /// Mean Squared Error
    pub mse: f32,
    /// Mean Absolute Error
    pub mae: f32,
    /// Root Mean Squared Error
    pub rmse: f32,
    /// Signal-to-Noise Ratio (dB)
    pub snr_db: f32,
    /// Peak Signal-to-Noise Ratio (dB)
    pub psnr_db: f32,
    /// Normalized Mean Squared Error (0-1)
    pub nmse: f32,
}

impl QualityMetrics {
    /// Compute all quality metrics from original and reconstructed signals
    ///
    /// # Arguments
    ///
    /// * `original` - Original signal
    /// * `reconstructed` - Reconstructed signal from tokenizer
    ///
    /// # Returns
    ///
    /// Complete quality metrics
    pub fn compute(original: &Array1<f32>, reconstructed: &Array1<f32>) -> TokenizerResult<Self> {
        if original.len() != reconstructed.len() {
            return Err(TokenizerError::dim_mismatch(
                original.len(),
                reconstructed.len(),
                "dimension validation",
            ));
        }

        let n = original.len() as f32;

        // Mean Squared Error
        let mse: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| (o - r).powi(2))
            .sum::<f32>()
            / n;

        // Mean Absolute Error
        let mae: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| (o - r).abs())
            .sum::<f32>()
            / n;

        // RMSE
        let rmse = mse.sqrt();

        // Signal power
        let signal_power: f32 = original.iter().map(|x| x.powi(2)).sum::<f32>() / n;

        // Noise power
        let noise_power = mse;

        // SNR (dB)
        let snr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f32::INFINITY
        };

        // PSNR (dB) - using max absolute value as peak
        let peak = original
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, |a, b| a.max(b));
        let psnr_db = if mse > 0.0 && peak > 0.0 {
            20.0 * (peak / rmse).log10()
        } else {
            f32::INFINITY
        };

        // Normalized MSE
        let nmse = if signal_power > 0.0 {
            mse / signal_power
        } else {
            0.0
        };

        Ok(Self {
            mse,
            mae,
            rmse,
            snr_db,
            psnr_db,
            nmse,
        })
    }

    /// Check if metrics meet acceptable quality thresholds
    ///
    /// # Arguments
    ///
    /// * `min_snr_db` - Minimum acceptable SNR in dB
    ///
    /// # Returns
    ///
    /// true if quality is acceptable
    pub fn is_acceptable(&self, min_snr_db: f32) -> bool {
        self.snr_db >= min_snr_db && self.snr_db.is_finite()
    }

    /// Get a human-readable quality rating
    pub fn quality_rating(&self) -> &'static str {
        if !self.snr_db.is_finite() {
            "Perfect"
        } else if self.snr_db >= 40.0 {
            "Excellent"
        } else if self.snr_db >= 30.0 {
            "Very Good"
        } else if self.snr_db >= 20.0 {
            "Good"
        } else if self.snr_db >= 10.0 {
            "Fair"
        } else {
            "Poor"
        }
    }
}

/// Spectral distance metrics for frequency-domain analysis
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralMetrics {
    /// Spectral convergence
    pub spectral_convergence: f32,
    /// Magnitude error
    pub magnitude_error: f32,
    /// Phase error (radians)
    pub phase_error: f32,
}

impl SpectralMetrics {
    /// Compute spectral metrics using DFT
    ///
    /// # Arguments
    ///
    /// * `original` - Original signal
    /// * `reconstructed` - Reconstructed signal
    ///
    /// # Returns
    ///
    /// Spectral quality metrics
    pub fn compute(original: &Array1<f32>, reconstructed: &Array1<f32>) -> TokenizerResult<Self> {
        if original.len() != reconstructed.len() {
            return Err(TokenizerError::dim_mismatch(
                original.len(),
                reconstructed.len(),
                "dimension validation",
            ));
        }

        // Compute DFT for both signals
        let orig_spectrum = compute_dft(original);
        let recon_spectrum = compute_dft(reconstructed);

        let n = orig_spectrum.len() as f32;

        // Spectral convergence
        let numerator: f32 = orig_spectrum
            .iter()
            .zip(recon_spectrum.iter())
            .map(|(o, r)| (o.0 - r.0).powi(2) + (o.1 - r.1).powi(2))
            .sum();

        let denominator: f32 = orig_spectrum
            .iter()
            .map(|(re, im)| re.powi(2) + im.powi(2))
            .sum();

        let spectral_convergence = if denominator > 0.0 {
            (numerator / denominator).sqrt()
        } else {
            0.0
        };

        // Magnitude error
        let mag_error: f32 = orig_spectrum
            .iter()
            .zip(recon_spectrum.iter())
            .map(|(o, r)| {
                let mag_o = (o.0.powi(2) + o.1.powi(2)).sqrt();
                let mag_r = (r.0.powi(2) + r.1.powi(2)).sqrt();
                (mag_o - mag_r).abs()
            })
            .sum::<f32>()
            / n;

        // Phase error
        let phase_error: f32 = orig_spectrum
            .iter()
            .zip(recon_spectrum.iter())
            .map(|(o, r)| {
                let phase_o = o.1.atan2(o.0);
                let phase_r = r.1.atan2(r.0);
                let diff = (phase_o - phase_r).abs();
                // Wrap to [-π, π]
                if diff > PI {
                    2.0 * PI - diff
                } else {
                    diff
                }
            })
            .sum::<f32>()
            / n;

        Ok(Self {
            spectral_convergence,
            magnitude_error: mag_error,
            phase_error,
        })
    }
}

/// Simple DFT implementation for spectral analysis
fn compute_dft(signal: &Array1<f32>) -> Vec<(f32, f32)> {
    let n = signal.len();
    let mut spectrum = Vec::with_capacity(n);

    for k in 0..n {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;

        for (t, &x) in signal.iter().enumerate() {
            let angle = -2.0 * PI * (k as f32) * (t as f32) / (n as f32);
            real += x * angle.cos();
            imag += x * angle.sin();
        }

        spectrum.push((real, imag));
    }

    spectrum
}

/// Compression efficiency metrics
#[derive(Debug, Clone, PartialEq)]
pub struct CompressionMetrics {
    /// Original size in bits
    pub original_bits: usize,
    /// Compressed size in bits
    pub compressed_bits: usize,
    /// Compression ratio (original / compressed)
    pub compression_ratio: f64,
    /// Bits per sample
    pub bits_per_sample: f64,
    /// Space savings percentage
    pub space_savings_percent: f64,
}

impl CompressionMetrics {
    /// Compute compression metrics
    ///
    /// # Arguments
    ///
    /// * `num_samples` - Number of samples in original signal
    /// * `bits_per_original_sample` - Bits per sample in original (e.g., 16 for 16-bit audio)
    /// * `compressed_bytes` - Size of compressed representation in bytes
    pub fn compute(
        num_samples: usize,
        bits_per_original_sample: usize,
        compressed_bytes: usize,
    ) -> Self {
        let original_bits = num_samples * bits_per_original_sample;
        let compressed_bits = compressed_bytes * 8;

        let compression_ratio = if compressed_bits > 0 {
            original_bits as f64 / compressed_bits as f64
        } else {
            f64::INFINITY
        };

        let bits_per_sample = if num_samples > 0 {
            compressed_bits as f64 / num_samples as f64
        } else {
            0.0
        };

        let space_savings_percent = if original_bits > 0 {
            ((original_bits - compressed_bits) as f64 / original_bits as f64) * 100.0
        } else {
            0.0
        };

        Self {
            original_bits,
            compressed_bits,
            compression_ratio,
            bits_per_sample,
            space_savings_percent,
        }
    }

    /// Check if compression is effective (ratio > 1)
    pub fn is_effective(&self) -> bool {
        self.compression_ratio > 1.0 && self.compression_ratio.is_finite()
    }
}

/// Rate-distortion point for RD curve analysis
#[derive(Debug, Clone, PartialEq)]
pub struct RateDistortionPoint {
    /// Bit rate (bits per sample)
    pub rate: f64,
    /// Distortion (MSE or other metric)
    pub distortion: f32,
    /// SNR in dB
    pub snr_db: f32,
}

/// Rate-distortion curve for analyzing compression efficiency
#[derive(Debug, Clone)]
pub struct RateDistortionCurve {
    /// Collection of RD points
    points: Vec<RateDistortionPoint>,
}

impl RateDistortionCurve {
    /// Create a new empty RD curve
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Add a rate-distortion point
    pub fn add_point(&mut self, rate: f64, distortion: f32, snr_db: f32) {
        self.points.push(RateDistortionPoint {
            rate,
            distortion,
            snr_db,
        });
    }

    /// Get all points sorted by rate
    pub fn points(&self) -> Vec<RateDistortionPoint> {
        let mut sorted = self.points.clone();
        sorted.sort_by(|a, b| {
            a.rate
                .partial_cmp(&b.rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Find the best operating point for target SNR
    ///
    /// Returns the point with lowest rate that meets the SNR requirement
    pub fn find_best_for_snr(&self, target_snr_db: f32) -> Option<&RateDistortionPoint> {
        self.points
            .iter()
            .filter(|p| p.snr_db >= target_snr_db)
            .min_by(|a, b| {
                a.rate
                    .partial_cmp(&b.rate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Find the best operating point for target rate
    ///
    /// Returns the point with highest SNR under the rate constraint
    pub fn find_best_for_rate(&self, target_rate: f64) -> Option<&RateDistortionPoint> {
        self.points
            .iter()
            .filter(|p| p.rate <= target_rate)
            .max_by(|a, b| {
                a.snr_db
                    .partial_cmp(&b.snr_db)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Compute BD-rate (Bjøntegaard Delta rate) relative to reference curve
    ///
    /// Measures average percentage rate difference for same quality
    pub fn bd_rate(&self, reference: &RateDistortionCurve) -> f64 {
        // Simplified BD-rate calculation
        // In practice, this would use interpolation and integration

        let self_points = self.points();
        let ref_points = reference.points();

        if self_points.is_empty() || ref_points.is_empty() {
            return 0.0;
        }

        // Average rate difference at matching SNR points
        let mut rate_diffs = Vec::new();

        for self_point in &self_points {
            if let Some(ref_point) = ref_points
                .iter()
                .min_by_key(|p| ((p.snr_db - self_point.snr_db).abs() * 1000.0) as i32)
            {
                if (ref_point.snr_db - self_point.snr_db).abs() < 2.0 {
                    let rate_diff = (self_point.rate - ref_point.rate) / ref_point.rate * 100.0;
                    rate_diffs.push(rate_diff);
                }
            }
        }

        if rate_diffs.is_empty() {
            0.0
        } else {
            rate_diffs.iter().sum::<f64>() / rate_diffs.len() as f64
        }
    }
}

impl Default for RateDistortionCurve {
    fn default() -> Self {
        Self::new()
    }
}

/// Perceptual quality metrics
#[derive(Debug, Clone, PartialEq)]
pub struct PerceptualMetrics {
    /// Segmental SNR (dB) - SNR computed over short segments
    pub segmental_snr_db: f32,
    /// Weighted SNR (dB) - frequency-weighted SNR
    pub weighted_snr_db: f32,
}

impl PerceptualMetrics {
    /// Compute perceptual metrics
    ///
    /// # Arguments
    ///
    /// * `original` - Original signal
    /// * `reconstructed` - Reconstructed signal
    /// * `segment_len` - Length of segments for segmental SNR
    pub fn compute(
        original: &Array1<f32>,
        reconstructed: &Array1<f32>,
        segment_len: usize,
    ) -> TokenizerResult<Self> {
        if original.len() != reconstructed.len() {
            return Err(TokenizerError::dim_mismatch(
                original.len(),
                reconstructed.len(),
                "dimension validation",
            ));
        }

        // Segmental SNR
        let num_segments = original.len() / segment_len;
        let mut segment_snrs = Vec::new();

        for i in 0..num_segments {
            let start = i * segment_len;
            let end = start + segment_len;

            let orig_segment = original.slice(s![start..end]);
            let recon_segment = reconstructed.slice(s![start..end]);

            let signal_power: f32 =
                orig_segment.iter().map(|x| x.powi(2)).sum::<f32>() / segment_len as f32;
            let noise_power: f32 = orig_segment
                .iter()
                .zip(recon_segment.iter())
                .map(|(o, r)| (o - r).powi(2))
                .sum::<f32>()
                / segment_len as f32;

            if noise_power > 0.0 && signal_power > 0.0 {
                let snr = 10.0 * (signal_power / noise_power).log10();
                segment_snrs.push(snr);
            }
        }

        let segmental_snr_db = if !segment_snrs.is_empty() {
            segment_snrs.iter().sum::<f32>() / segment_snrs.len() as f32
        } else {
            0.0
        };

        // Weighted SNR (simplified - in practice would use psychoacoustic model)
        // Here we just weight lower frequencies more heavily
        let weighted_snr_db = segmental_snr_db; // Placeholder

        Ok(Self {
            segmental_snr_db,
            weighted_snr_db,
        })
    }
}

use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_metrics_perfect() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let metrics = QualityMetrics::compute(&signal, &signal).unwrap();

        assert_eq!(metrics.mse, 0.0);
        assert_eq!(metrics.mae, 0.0);
        assert_eq!(metrics.rmse, 0.0);
        assert!(metrics.snr_db.is_infinite());
        assert_eq!(metrics.quality_rating(), "Perfect");
    }

    #[test]
    fn test_quality_metrics_noisy() {
        let original = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let reconstructed = Array1::from_vec(vec![1.1, 2.1, 2.9, 4.1, 4.9]);

        let metrics = QualityMetrics::compute(&original, &reconstructed).unwrap();

        assert!(metrics.mse > 0.0);
        assert!(metrics.mae > 0.0);
        assert!(metrics.snr_db.is_finite());
        assert!(metrics.snr_db > 0.0);
    }

    #[test]
    fn test_quality_metrics_rating() {
        let original = Array1::from_vec(vec![1.0; 100]);
        let mut reconstructed = original.clone();
        reconstructed[0] = 1.01; // Slight error

        let metrics = QualityMetrics::compute(&original, &reconstructed).unwrap();

        assert!(metrics.snr_db > 30.0);
        assert!(["Excellent", "Very Good"].contains(&metrics.quality_rating()));
    }

    #[test]
    fn test_spectral_metrics() {
        let signal = Array1::from_vec((0..32).map(|i| (i as f32 * 0.2).sin()).collect());
        let noisy = Array1::from_vec(
            signal
                .iter()
                .map(|&x| x + 0.01 * (x * 10.0).sin())
                .collect(),
        );

        let metrics = SpectralMetrics::compute(&signal, &noisy).unwrap();

        assert!(metrics.spectral_convergence >= 0.0);
        assert!(metrics.magnitude_error >= 0.0);
        assert!(metrics.phase_error >= 0.0);
    }

    #[test]
    fn test_compression_metrics() {
        let metrics = CompressionMetrics::compute(1000, 16, 1000);

        assert_eq!(metrics.original_bits, 16000);
        assert_eq!(metrics.compressed_bits, 8000);
        assert_eq!(metrics.compression_ratio, 2.0);
        assert_eq!(metrics.bits_per_sample, 8.0);
        assert!(metrics.is_effective());
    }

    #[test]
    fn test_compression_metrics_no_compression() {
        let metrics = CompressionMetrics::compute(1000, 16, 2000);

        assert_eq!(metrics.compression_ratio, 1.0);
        assert!(!metrics.is_effective());
    }

    #[test]
    fn test_rate_distortion_curve() {
        let mut curve = RateDistortionCurve::new();

        curve.add_point(1.0, 0.1, 20.0);
        curve.add_point(2.0, 0.05, 25.0);
        curve.add_point(4.0, 0.01, 35.0);

        let best_for_snr = curve.find_best_for_snr(22.0).unwrap();
        assert_eq!(best_for_snr.rate, 2.0);

        let best_for_rate = curve.find_best_for_rate(3.0).unwrap();
        assert_eq!(best_for_rate.rate, 2.0);
    }

    #[test]
    fn test_perceptual_metrics() {
        let signal = Array1::from_vec((0..100).map(|i| (i as f32 * 0.1).sin()).collect());
        let noisy = Array1::from_vec(signal.iter().map(|&x| x + 0.01).collect());

        let metrics = PerceptualMetrics::compute(&signal, &noisy, 10).unwrap();

        assert!(metrics.segmental_snr_db.is_finite());
        assert!(metrics.segmental_snr_db > 0.0);
    }

    #[test]
    fn test_bd_rate() {
        let mut curve1 = RateDistortionCurve::new();
        curve1.add_point(1.0, 0.1, 20.0);
        curve1.add_point(2.0, 0.05, 25.0);

        let mut curve2 = RateDistortionCurve::new();
        curve2.add_point(1.5, 0.1, 20.0);
        curve2.add_point(2.5, 0.05, 25.0);

        let bd = curve2.bd_rate(&curve1);
        assert!(bd > 0.0); // curve2 uses more rate
    }
}
