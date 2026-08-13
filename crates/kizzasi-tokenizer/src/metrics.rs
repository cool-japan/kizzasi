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

    /// Mean percentage rate difference at matching SNR points, relative to
    /// a reference curve.
    ///
    /// This is **not** the Bjøntegaard Delta rate (BD-rate) despite the
    /// name this method used to have: real BD-rate integrates a
    /// piecewise-cubic interpolation of `log10(rate)` vs. quality over the
    /// two curves' overlapping quality interval (Bjøntegaard, VCEG-M33),
    /// which this crate does not implement. This method instead pairs each
    /// of `self`'s points with its nearest-SNR point in `reference` (within
    /// 2 dB) and averages the percentage rate differences — a simpler,
    /// coarser approximation that is only meaningful when both curves were
    /// sampled at similar SNR operating points.
    ///
    /// Returns `None` when there are no comparable points (either curve is
    /// empty, or no pair of points falls within 2 dB of each other) —
    /// distinguishable from `Some(0.0)`, which means the curves were
    /// genuinely identical at every comparable point.
    pub fn mean_rate_difference_pct(&self, reference: &RateDistortionCurve) -> Option<f64> {
        let self_points = self.points();
        let ref_points = reference.points();

        if self_points.is_empty() || ref_points.is_empty() {
            return None;
        }

        // Average rate difference at matching SNR points.
        let mut rate_diffs = Vec::new();

        for self_point in &self_points {
            // `partial_cmp`/`f32::abs` are always finite here (SNR values
            // come from `10*log10(...)` of a non-negative ratio elsewhere in
            // this module), so `unwrap_or(Equal)` never masks a NaN.
            if let Some(ref_point) = ref_points.iter().min_by(|a, b| {
                (a.snr_db - self_point.snr_db)
                    .abs()
                    .partial_cmp(&(b.snr_db - self_point.snr_db).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                if (ref_point.snr_db - self_point.snr_db).abs() < 2.0 && ref_point.rate != 0.0 {
                    let rate_diff = (self_point.rate - ref_point.rate) / ref_point.rate * 100.0;
                    rate_diffs.push(rate_diff);
                }
            }
        }

        if rate_diffs.is_empty() {
            None
        } else {
            Some(rate_diffs.iter().sum::<f64>() / rate_diffs.len() as f64)
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

        // Frequency-domain A-weighted SNR (IEC 61672-1)
        let weighted_snr_db = spectral_weighted_snr(
            original.as_slice().unwrap_or(&[]),
            reconstructed.as_slice().unwrap_or(&[]),
            segment_len,
        );

        Ok(Self {
            segmental_snr_db,
            weighted_snr_db,
        })
    }
}

/// IEC 61672-1 A-weighting transfer function (unnormalized).
///
/// Returns the raw amplitude ratio for the given frequency in Hz.
/// Zero is returned for non-positive frequencies.
fn a_weighting(f_hz: f32) -> f32 {
    if f_hz <= 0.0 {
        return 0.0;
    }
    let f2 = f_hz * f_hz;
    // Pole/zero frequencies squared from IEC 61672-1
    let f1_sq = 20.6_f32 * 20.6_f32; // 20.6 Hz
    let f2_sq = 107.7_f32 * 107.7_f32; // 107.7 Hz
    let f3_sq = 737.9_f32 * 737.9_f32; // 737.9 Hz
    let f4_sq = 12200.0_f32 * 12200.0_f32; // 12200 Hz
    let num = f4_sq * f2 * f2;
    let den = (f2 + f1_sq) * (f2 + f4_sq) * ((f2 + f2_sq) * (f2 + f3_sq)).sqrt();
    (num / den).max(0.0)
}

/// Compute frequency-domain A-weighted SNR by processing the signal in
/// non-overlapping `segment_len`-sample windows.
///
/// A DFT is computed per segment (capped at 256 bins to keep O(N²) tractable),
/// each spectral bin is weighted by the IEC 61672-1 A-weighting curve evaluated
/// at `f = k * 44100 / N` Hz, and the resulting weighted signal/noise powers
/// are accumulated before converting to dB.
fn spectral_weighted_snr(orig: &[f32], recon: &[f32], segment_len: usize) -> f32 {
    if orig.is_empty() || recon.is_empty() || segment_len == 0 {
        return 0.0;
    }

    let n = orig.len().min(recon.len());
    let num_segments = n / segment_len;
    if num_segments == 0 {
        return 0.0;
    }

    // Cap DFT bin loop to 256 to keep O(N²) tractable for large segments.
    let actual_bins = (segment_len / 2 + 1).min(256);

    let mut per_segment_snrs: Vec<f32> = Vec::with_capacity(num_segments);

    for seg_idx in 0..num_segments {
        let start = seg_idx * segment_len;
        let end = start + segment_len;
        let orig_seg = &orig[start..end];
        let recon_seg = &recon[start..end];
        let n_f32 = segment_len as f32;

        // Pre-compute difference signal for the noise DFT
        let diff_seg: Vec<f32> = orig_seg
            .iter()
            .zip(recon_seg.iter())
            .map(|(o, r)| o - r)
            .collect();

        let mut weighted_signal_power = 0.0_f32;
        let mut weighted_noise_power = 0.0_f32;
        let mut total_weight = 0.0_f32;

        // Also accumulate unweighted sums for the zero-weight fallback path
        let mut unweighted_signal_power = 0.0_f32;
        let mut unweighted_noise_power = 0.0_f32;

        for k in 0..actual_bins {
            let mut x_re = 0.0_f32;
            let mut x_im = 0.0_f32;
            let mut d_re = 0.0_f32;
            let mut d_im = 0.0_f32;

            for (t, (&o, &d)) in orig_seg.iter().zip(diff_seg.iter()).enumerate() {
                let angle = 2.0 * PI * (k as f32) * (t as f32) / n_f32;
                let (sin_a, cos_a) = angle.sin_cos();
                x_re += o * cos_a;
                x_im += -o * sin_a;
                d_re += d * cos_a;
                d_im += -d * sin_a;
            }

            let signal_power_k = x_re * x_re + x_im * x_im;
            let noise_power_k = d_re * d_re + d_im * d_im;

            let f_hz = (k as f32) * 44100.0 / n_f32;
            let w = a_weighting(f_hz);

            weighted_signal_power += w * signal_power_k;
            weighted_noise_power += w * noise_power_k;
            total_weight += w;

            unweighted_signal_power += signal_power_k;
            unweighted_noise_power += noise_power_k;
        }

        // Fallback to unweighted when all A-weights are zero
        // (happens for extremely short segments whose bins are all sub-20 Hz DC)
        let (sp, np) = if total_weight <= 0.0 {
            (unweighted_signal_power, unweighted_noise_power)
        } else {
            (weighted_signal_power, weighted_noise_power)
        };

        if sp > 0.0 && np > 0.0 {
            per_segment_snrs.push(10.0 * (sp / np).log10());
        }
    }

    if per_segment_snrs.is_empty() {
        return 0.0;
    }

    per_segment_snrs.iter().sum::<f32>() / per_segment_snrs.len() as f32
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
    fn test_mean_rate_difference_pct() {
        let mut curve1 = RateDistortionCurve::new();
        curve1.add_point(1.0, 0.1, 20.0);
        curve1.add_point(2.0, 0.05, 25.0);

        let mut curve2 = RateDistortionCurve::new();
        curve2.add_point(1.5, 0.1, 20.0);
        curve2.add_point(2.5, 0.05, 25.0);

        let diff = curve2
            .mean_rate_difference_pct(&curve1)
            .expect("points within 2 dB must be comparable");
        assert!(diff > 0.0); // curve2 uses more rate
    }

    /// Regression: no comparable points (SNRs more than 2 dB apart) must
    /// return `None`, distinguishable from `Some(0.0)` ("identical rate").
    #[test]
    fn test_mean_rate_difference_pct_none_when_no_comparable_points() {
        let mut curve1 = RateDistortionCurve::new();
        curve1.add_point(1.0, 0.1, 10.0);

        let mut curve2 = RateDistortionCurve::new();
        curve2.add_point(1.0, 0.1, 50.0); // 40 dB away — not comparable

        assert_eq!(curve2.mean_rate_difference_pct(&curve1), None);
    }

    /// Regression: empty curves must return `None`, not `Some(0.0)`.
    #[test]
    fn test_mean_rate_difference_pct_none_when_empty() {
        let empty = RateDistortionCurve::new();
        let mut other = RateDistortionCurve::new();
        other.add_point(1.0, 0.1, 20.0);

        assert_eq!(empty.mean_rate_difference_pct(&other), None);
        assert_eq!(other.mean_rate_difference_pct(&empty), None);
    }

    /// Regression: a reference point with `rate == 0.0` used to divide by
    /// zero (producing inf/NaN silently folded into the average); it must
    /// now be skipped instead.
    #[test]
    fn test_mean_rate_difference_pct_skips_zero_rate_reference() {
        let mut curve1 = RateDistortionCurve::new();
        curve1.add_point(0.0, 0.1, 20.0); // rate == 0.0

        let mut curve2 = RateDistortionCurve::new();
        curve2.add_point(1.5, 0.1, 20.0);

        // The only reference point has rate == 0.0, so there is nothing
        // comparable left after the guard — must be None, not NaN/inf.
        assert_eq!(curve2.mean_rate_difference_pct(&curve1), None);
    }

    // --- A-weighting and spectral weighted SNR tests ---

    #[test]
    fn test_a_weighting_zero_dc() {
        assert_eq!(a_weighting(0.0), 0.0);
        assert_eq!(a_weighting(-1.0), 0.0);
    }

    #[test]
    fn test_a_weighting_peaks_midrange() {
        // A-weighting should favour mid-range frequencies over sub-bass
        let w100 = a_weighting(100.0);
        let w1000 = a_weighting(1000.0);
        let w3000 = a_weighting(3000.0);
        assert!(
            w1000 > w100,
            "Expected a_weighting(1000) > a_weighting(100), got {} vs {}",
            w1000,
            w100
        );
        assert!(
            w3000 > w100,
            "Expected a_weighting(3000) > a_weighting(100), got {} vs {}",
            w3000,
            w100
        );
    }

    #[test]
    fn test_weighted_snr_differs_from_segmental() {
        // Low-frequency sine: 128 samples at a normalised frequency so that
        // almost all energy sits in the lowest non-DC DFT bins, which are
        // heavily de-emphasised by A-weighting.  Add noise concentrated in
        // the first few samples — segmental SNR sees this noise equally across
        // all segments, but A-weighted SNR will see a different power balance.
        let n = 128usize;
        let orig: Vec<f32> = (0..n)
            .map(|i| {
                // Very low frequency: one full cycle over 128 samples → bin 1
                (2.0 * PI * i as f32 / n as f32).sin()
            })
            .collect();

        let mut recon = orig.clone();
        // Burst noise in the first 8 samples — creates spectrally coloured
        // noise that A-weighting will evaluate differently from flat noise.
        for (i, sample) in recon.iter_mut().enumerate().take(8) {
            *sample += 0.3 * (i as f32 + 1.0).recip();
        }

        let original_arr = Array1::from_vec(orig);
        let recon_arr = Array1::from_vec(recon);
        let segment_len = 64;

        let metrics = PerceptualMetrics::compute(&original_arr, &recon_arr, segment_len).unwrap();

        // The two metrics start from the same underlying signal but weight
        // frequency content differently, so they should differ.
        assert_ne!(
            metrics.weighted_snr_db, metrics.segmental_snr_db,
            "weighted_snr_db and segmental_snr_db should diverge with real A-weighting"
        );
    }

    #[test]
    fn test_weighted_snr_finite() {
        // Build a deterministic pseudo-random signal using a simple LCG
        let n = 128usize;
        let mut state: u64 = 0xdeadbeef_cafebabe;
        let lcg_next = |s: &mut u64| -> f32 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to [-1, 1]
            (*s as i64 as f32) / (i64::MAX as f32)
        };

        let orig: Vec<f32> = (0..n).map(|_| lcg_next(&mut state)).collect();
        let recon: Vec<f32> = orig
            .iter()
            .map(|&x| x + 0.05 * lcg_next(&mut state))
            .collect();

        let original_arr = Array1::from_vec(orig);
        let recon_arr = Array1::from_vec(recon);

        let metrics = PerceptualMetrics::compute(&original_arr, &recon_arr, 32).unwrap();

        assert!(
            metrics.weighted_snr_db.is_finite(),
            "weighted_snr_db should be finite, got {}",
            metrics.weighted_snr_db
        );
        assert!(
            metrics.weighted_snr_db > f32::NEG_INFINITY,
            "weighted_snr_db should be > NEG_INFINITY"
        );
    }
}
