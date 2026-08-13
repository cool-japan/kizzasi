//! Signal Quality Metrics Module
//!
//! This module provides objective signal quality assessment metrics:
//! - STOI (Short-Time Objective Intelligibility) after Taal et al. (2011)
//! - PESQ-inspired perceptual quality metric
//! - POLQA-inspired quality assessment
//! - MOS (Mean Opinion Score) prediction
//! - Objective quality measures (SNR, segmental SNR, frequency-weighted SNR)
//!
//! # Standards conformance
//!
//! Only [`SnrCalculator`] and [`StoiCalculator`] compute a published
//! algorithm. [`PesqCalculator`] and [`PolqaCalculator`] are *inspired by*
//! ITU-T P.862 and ITU-T P.863 respectively but implement neither: they use
//! this crate's own perceptual model, and their output is a relative,
//! comparative figure produced by this crate — never a certified MOS-LQO
//! number. [`StoiCalculator`], [`PesqCalculator`], [`PolqaCalculator`] and
//! [`MosPredictor`] each expose a `STANDARD_CONFORMANT` constant so the
//! distinction is visible from code as well as from docs; [`MosPredictor`]
//! inherits the caveat because it aggregates the other three.

use crate::error::{IoError, IoResult};
use oxifft::{Complex, Direction, Flags, Plan};
use std::f64::consts::PI;

/// Number of one-third octave bands used by STOI (Taal et al., 2011).
const STOI_NUM_BANDS: usize = 15;
/// Centre frequency of the lowest one-third octave band, in Hz.
const THIRD_OCTAVE_LOWEST_CENTER_HZ: f64 = 150.0;
/// STOI analysis window length in seconds (256 samples at 10 kHz).
const STOI_FRAME_SECONDS: f64 = 0.0256;
/// Number of consecutive frames correlated by STOI (384 ms at 10 kHz).
const STOI_SEGMENT_FRAMES: usize = 30;
/// Lower SDR bound in dB used to clip the normalised degraded band energies.
const STOI_BETA_DB: f64 = -15.0;
/// Frames more than this many dB below the loudest frame count as silence.
const STOI_DYNAMIC_RANGE_DB: f64 = 40.0;

/// PESQ analysis window length in seconds (ITU-T P.862 uses 32 ms).
const PESQ_FRAME_SECONDS: f64 = 0.032;
/// One-third octave bands requested for the PESQ loudness spectrum; bands
/// beyond Nyquist are dropped by [`third_octave_bins`].
const PESQ_NUM_BANDS: usize = 21;
/// Zwicker power-law exponent mapping band energy to specific loudness.
const PESQ_LOUDNESS_EXPONENT: f64 = 0.23;
/// Fraction of the smaller loudness treated as masked (ITU-T P.862 uses 0.25).
const PESQ_MASK_FACTOR: f64 = 0.25;
/// Exponent of the asymmetry weight applied when the degraded signal has
/// *more* loudness in a band than the reference.
const PESQ_ASYMMETRY_EXPONENT: f64 = 1.2;
/// Upper bound on the asymmetry weight.
const PESQ_MAX_ASYMMETRY: f64 = 12.0;
/// Floor added to both loudness values before the asymmetry ratio, as a
/// fraction of the mean reference band loudness of the frame.
const PESQ_ASYMMETRY_FLOOR_FRACTION: f64 = 0.01;
/// Maps the mean relative loudness distortion onto the 4.5..1.0 span.
const PESQ_DISTORTION_GAIN: f64 = 3.5;

/// Periodic Hann window (sums to 1.0 under 50 % overlap-add).
fn hann_window(length: usize) -> Vec<f64> {
    if length <= 1 {
        return vec![1.0; length];
    }
    (0..length)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / length as f64).cos())
        .collect()
}

/// One-third octave band edges expressed as half-open FFT bin ranges.
///
/// Centre frequencies are `lowest_center_hz * 2^(k/3)` and the band edges
/// sit at the geometric mean of adjacent centres, i.e. `centre * 2^(±1/6)` —
/// the layout used by the STOI reference implementation. Edges are snapped
/// to the nearest FFT bin, the top band is clipped to Nyquist, and bands
/// that end up empty (below the bin resolution or beyond Nyquist) are
/// dropped, so the returned vector may be shorter than `num_bands`.
fn third_octave_bins(
    sample_rate: f64,
    fft_size: usize,
    num_bands: usize,
    lowest_center_hz: f64,
) -> Vec<(usize, usize)> {
    let mut bands = Vec::with_capacity(num_bands);
    if !sample_rate.is_finite() || sample_rate <= 0.0 || fft_size < 2 {
        return bands;
    }

    let max_bin = fft_size / 2 + 1;
    let bin_of = |hz: f64| -> usize {
        let bin = (hz * fft_size as f64 / sample_rate).round();
        if bin.is_finite() && bin > 0.0 {
            (bin as usize).min(max_bin)
        } else {
            0
        }
    };

    let half_step = 2f64.powf(1.0 / 6.0);
    for k in 0..num_bands {
        let center = lowest_center_hz * 2f64.powf(k as f64 / 3.0);
        let lower = bin_of(center / half_step);
        let upper = bin_of(center * half_step);
        if lower < upper {
            bands.push((lower, upper));
        }
    }

    bands
}

/// Sum the squared magnitudes of `magnitudes[lower..upper]`.
fn band_energy(magnitudes: &[f64], lower: usize, upper: usize) -> f64 {
    magnitudes
        .get(lower..upper.min(magnitudes.len()))
        .map(|slice| slice.iter().map(|m| m * m).sum::<f64>())
        .unwrap_or(0.0)
}

/// Short-time Fourier magnitudes, one row per frame.
///
/// Each row holds `fft_size / 2 + 1` magnitudes for a Hann-windowed frame of
/// `frame_length` samples, zero padded to `fft_size`.
fn stft_magnitudes(
    signal: &[f64],
    frame_length: usize,
    hop_length: usize,
    fft_size: usize,
) -> IoResult<Vec<Vec<f64>>> {
    if frame_length == 0 || hop_length == 0 || fft_size < frame_length {
        return Err(IoError::ConfigError(
            "Invalid STFT geometry: frame/hop/FFT sizes are inconsistent".to_string(),
        ));
    }
    if signal.len() < frame_length {
        return Err(IoError::ConfigError(format!(
            "Signal too short: {} samples, {frame_length} required for one analysis frame",
            signal.len()
        )));
    }

    let plan = Plan::<f64>::dft_1d(fft_size, Direction::Forward, Flags::ESTIMATE)
        .ok_or_else(|| IoError::SignalError(format!("FFT planning failed for size {fft_size}")))?;
    let window = hann_window(frame_length);
    let num_frames = (signal.len() - frame_length) / hop_length + 1;
    let num_bins = fft_size / 2 + 1;

    let mut frames = Vec::with_capacity(num_frames);
    let mut input = vec![Complex::new(0.0f64, 0.0); fft_size];
    let mut output = vec![Complex::new(0.0f64, 0.0); fft_size];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        let Some(frame) = signal.get(start..start + frame_length) else {
            break;
        };

        for value in input.iter_mut() {
            *value = Complex::new(0.0, 0.0);
        }
        for (slot, (&sample, &weight)) in input.iter_mut().zip(frame.iter().zip(window.iter())) {
            *slot = Complex::new(sample * weight, 0.0);
        }

        plan.execute(&input, &mut output);
        frames.push(
            output
                .iter()
                .take(num_bins)
                .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                .collect(),
        );
    }

    Ok(frames)
}

/// Pearson correlation of two equal-length vectors.
///
/// Returns `None` when either vector is constant, i.e. when the correlation
/// is mathematically undefined — callers skip such pairs rather than
/// substituting a fabricated value.
fn pearson_correlation(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len().min(b.len());
    if n < 2 {
        return None;
    }

    let mean_a = a.iter().take(n).sum::<f64>() / n as f64;
    let mean_b = b.iter().take(n).sum::<f64>() / n as f64;

    let mut covariance = 0.0;
    let mut variance_a = 0.0;
    let mut variance_b = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()).take(n) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        covariance += dx * dy;
        variance_a += dx * dx;
        variance_b += dy * dy;
    }

    if variance_a <= f64::MIN_POSITIVE || variance_b <= f64::MIN_POSITIVE {
        return None;
    }

    let rho = covariance / (variance_a.sqrt() * variance_b.sqrt());
    if rho.is_finite() {
        Some(rho.clamp(-1.0, 1.0))
    } else {
        None
    }
}

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
///
/// Measures speech intelligibility in noisy conditions by correlating the
/// one-third octave band energy envelopes of the reference and the degraded
/// signal over short segments, following C. H. Taal, R. C. Hendriks,
/// R. Heusdens and J. Jensen, *"An Algorithm for Intelligibility Prediction
/// of Time-Frequency Weighted Noisy Speech"*, IEEE TASLP 19(7), 2011.
///
/// The stages are the ones described in that paper: silent-frame removal at
/// a 40 dB dynamic range, a Hann-windowed STFT, 15 one-third octave bands
/// starting at 150 Hz, 30-frame segments, per-band energy normalisation of
/// the degraded signal, clipping at a -15 dB SDR, and the mean of the
/// per-(band, segment) correlation coefficients.
///
/// # Status — algorithm implemented, not reference-validated
///
/// Two documented deviations from the reference MATLAB implementation:
///
/// * Signals are analysed at their native sample rate instead of being
///   resampled to 10 kHz. The band layout is defined in Hz and nothing above
///   ~4.3 kHz enters any band, so the band energies are unaffected; only the
///   analysis window length in samples differs (it is rounded to a power of
///   two near 25.6 ms). Sample rates whose Nyquist frequency falls inside
///   the band range clip the topmost band, which *is* a deviation.
/// * The result has not been checked against the reference implementation's
///   published scores, so [`STANDARD_CONFORMANT`](Self::STANDARD_CONFORMANT)
///   is `false`.
///
/// Segments whose band envelope is constant carry no correlation
/// information; they are skipped, and if no segment survives, `calculate`
/// returns an error rather than a fabricated score.
#[derive(Debug, Clone)]
pub struct StoiCalculator {
    /// Sample rate
    sample_rate: f64,
    /// Frame length in samples
    frame_length: usize,
    /// Frame advance in samples (50 % overlap)
    hop_length: usize,
    /// Zero-padded transform size
    fft_size: usize,
    /// Number of frequency bands requested
    num_bands: usize,
    /// Half-open FFT bin range of each one-third octave band
    bands: Vec<(usize, usize)>,
}

impl StoiCalculator {
    /// Whether this implementation has been validated against the reference
    /// STOI implementation. It has not — see the type-level documentation.
    pub const STANDARD_CONFORMANT: bool = false;

    /// Create a new STOI calculator
    pub fn new(sample_rate: f64) -> Self {
        let nominal = (sample_rate * STOI_FRAME_SECONDS).round();
        let frame_length = if nominal.is_finite() && nominal >= 16.0 {
            (nominal as usize).next_power_of_two()
        } else {
            16
        };
        let hop_length = (frame_length / 2).max(1);
        let fft_size = frame_length * 2;
        let bands = third_octave_bins(
            sample_rate,
            fft_size,
            STOI_NUM_BANDS,
            THIRD_OCTAVE_LOWEST_CENTER_HZ,
        );

        Self {
            sample_rate,
            frame_length,
            hop_length,
            fft_size,
            num_bands: STOI_NUM_BANDS,
            bands,
        }
    }

    /// Number of one-third octave bands that fit below Nyquist at this
    /// sample rate (at most the 15 bands STOI asks for).
    pub fn active_bands(&self) -> usize {
        self.bands.len()
    }

    /// Calculate STOI score (0 to 1, higher is better)
    ///
    /// # Errors
    ///
    /// Returns an error when the signals differ in length, when the sample
    /// rate is too low for the one-third octave band layout, when the
    /// reference is silent, when fewer than 30 analysis frames survive
    /// silent-frame removal, or when every band envelope is constant (the
    /// correlation is then undefined).
    pub fn calculate(&self, reference: &[f64], degraded: &[f64]) -> IoResult<f64> {
        if reference.len() != degraded.len() {
            return Err(IoError::ConfigError("Signal length mismatch".to_string()));
        }
        if self.bands.is_empty() {
            return Err(IoError::ConfigError(format!(
                "Sample rate {} Hz is too low for the STOI one-third octave bands ({} bands from {} Hz)",
                self.sample_rate, self.num_bands, THIRD_OCTAVE_LOWEST_CENTER_HZ
            )));
        }

        let (active_reference, active_degraded) = self.remove_silent_frames(reference, degraded)?;
        let reference_bands = self.band_envelopes(&active_reference)?;
        let degraded_bands = self.band_envelopes(&active_degraded)?;

        let num_frames = reference_bands.len().min(degraded_bands.len());
        if num_frames < STOI_SEGMENT_FRAMES {
            return Err(IoError::ConfigError(format!(
                "Signal too short for STOI: {num_frames} analysis frames after silence removal, {STOI_SEGMENT_FRAMES} required"
            )));
        }

        // 10^(-beta/20) with beta = -15 dB: the degraded band energy may not
        // exceed the reference by more than this factor.
        let clip_factor = 1.0 + 10f64.powf(-STOI_BETA_DB / 20.0);

        let mut reference_segment = vec![0.0f64; STOI_SEGMENT_FRAMES];
        let mut degraded_segment = vec![0.0f64; STOI_SEGMENT_FRAMES];
        let mut correlation_sum = 0.0;
        let mut valid_segments = 0usize;

        for start in 0..=(num_frames - STOI_SEGMENT_FRAMES) {
            for band in 0..self.bands.len() {
                for (offset, (reference_slot, degraded_slot)) in reference_segment
                    .iter_mut()
                    .zip(degraded_segment.iter_mut())
                    .enumerate()
                {
                    *reference_slot = reference_bands
                        .get(start + offset)
                        .and_then(|frame| frame.get(band))
                        .copied()
                        .unwrap_or(0.0);
                    *degraded_slot = degraded_bands
                        .get(start + offset)
                        .and_then(|frame| frame.get(band))
                        .copied()
                        .unwrap_or(0.0);
                }

                let reference_power: f64 = reference_segment.iter().map(|v| v * v).sum();
                let degraded_power: f64 = degraded_segment.iter().map(|v| v * v).sum();
                if reference_power <= f64::MIN_POSITIVE || degraded_power <= f64::MIN_POSITIVE {
                    continue;
                }

                // Normalise the degraded envelope to the reference energy,
                // then clip it so a single loud band cannot dominate.
                let alpha = (reference_power / degraded_power).sqrt();
                for (degraded_slot, &reference_value) in
                    degraded_segment.iter_mut().zip(reference_segment.iter())
                {
                    *degraded_slot = (*degraded_slot * alpha).min(reference_value * clip_factor);
                }

                if let Some(rho) = pearson_correlation(&reference_segment, &degraded_segment) {
                    correlation_sum += rho;
                    valid_segments += 1;
                }
            }
        }

        if valid_segments == 0 {
            return Err(IoError::SignalError(
                "STOI is undefined for these signals: every band envelope is constant over all segments".to_string(),
            ));
        }

        let stoi = correlation_sum / valid_segments as f64;
        Ok(stoi.clamp(0.0, 1.0))
    }

    /// Drop frames more than `STOI_DYNAMIC_RANGE_DB` below the loudest
    /// reference frame and overlap-add the survivors back into a shorter
    /// pair of signals (the reference implementation's
    /// `removeSilentFrames`).
    fn remove_silent_frames(
        &self,
        reference: &[f64],
        degraded: &[f64],
    ) -> IoResult<(Vec<f64>, Vec<f64>)> {
        let length = reference.len().min(degraded.len());
        if length < self.frame_length {
            return Err(IoError::ConfigError(format!(
                "Signal too short for STOI: {length} samples, {} required for one analysis frame",
                self.frame_length
            )));
        }

        let window = hann_window(self.frame_length);
        let num_frames = (length - self.frame_length) / self.hop_length + 1;

        let mut frame_levels = Vec::with_capacity(num_frames);
        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_length;
            let Some(frame) = reference.get(start..start + self.frame_length) else {
                break;
            };
            let power: f64 = frame
                .iter()
                .zip(window.iter())
                .map(|(&sample, &weight)| (sample * weight) * (sample * weight))
                .sum();
            let rms = (power / self.frame_length as f64).sqrt();
            frame_levels.push(if rms > 0.0 {
                20.0 * rms.log10()
            } else {
                f64::NEG_INFINITY
            });
        }

        let peak = frame_levels
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !peak.is_finite() {
            return Err(IoError::SignalError(
                "Reference signal is silent: STOI has nothing to score".to_string(),
            ));
        }
        let floor = peak - STOI_DYNAMIC_RANGE_DB;

        let mut active_reference = vec![0.0f64; length];
        let mut active_degraded = vec![0.0f64; length];
        let mut kept = 0usize;

        for (frame_idx, &level) in frame_levels.iter().enumerate() {
            if level <= floor {
                continue;
            }
            let source = frame_idx * self.hop_length;
            let destination = kept * self.hop_length;
            for (offset, &weight) in window.iter().enumerate() {
                let reference_sample = reference.get(source + offset).copied().unwrap_or(0.0);
                let degraded_sample = degraded.get(source + offset).copied().unwrap_or(0.0);
                if let Some(slot) = active_reference.get_mut(destination + offset) {
                    *slot += reference_sample * weight;
                }
                if let Some(slot) = active_degraded.get_mut(destination + offset) {
                    *slot += degraded_sample * weight;
                }
            }
            kept += 1;
        }

        if kept == 0 {
            return Err(IoError::SignalError(
                "Reference signal is silent: STOI has nothing to score".to_string(),
            ));
        }

        let active_length = ((kept - 1) * self.hop_length + self.frame_length).min(length);
        active_reference.truncate(active_length);
        active_degraded.truncate(active_length);

        Ok((active_reference, active_degraded))
    }

    /// One-third octave band energy envelopes, one row per analysis frame.
    fn band_envelopes(&self, signal: &[f64]) -> IoResult<Vec<Vec<f64>>> {
        let spectra = stft_magnitudes(signal, self.frame_length, self.hop_length, self.fft_size)?;
        Ok(spectra
            .into_iter()
            .map(|magnitudes| {
                self.bands
                    .iter()
                    .map(|&(lower, upper)| band_energy(&magnitudes, lower, upper).sqrt())
                    .collect()
            })
            .collect())
    }
}

/// PESQ-inspired perceptual quality metric
///
/// Scores a degraded signal against a reference by comparing their
/// **per-band loudness spectra** frame by frame: the degraded signal is
/// level-aligned to the reference, both are analysed with a 32 ms
/// Hann-windowed STFT at 50 % overlap, band energies are mapped to specific
/// loudness with Zwicker's power law, and the per-band loudness difference
/// is masked and asymmetrically weighted (added energy counts for more than
/// removed energy) before being aggregated onto the 1.0..4.5 scale.
///
/// # Status — not ITU-T P.862 conformant
///
/// [`STANDARD_CONFORMANT`](Self::STANDARD_CONFORMANT) is `false` and the
/// score is **not** a certified PESQ MOS-LQO value. Missing versus P.862:
/// the IRS send/receive filtering, the utterance-based variable time
/// alignment, the Bark-scale auditory transform with per-frame gain
/// equalisation, the full cognitive model, and the published regression onto
/// the MOS-LQO scale. Use the number to compare degradations of the same
/// reference with each other, not as an absolute quality figure.
///
/// The metric is nevertheless sensitive to spectral distortion: band-limited
/// and noise-corrupted signals score below the reference even when their
/// overall RMS matches it exactly, which an envelope-only measure cannot do.
///
/// Known blind spot: the comparison is made on magnitude spectra, so it is
/// phase-blind. Two signals with the same short-time magnitude spectrum but
/// different phase — for example two independent noise realisations at the
/// same level — score close to identical.
#[derive(Debug, Clone)]
pub struct PesqCalculator {
    sample_rate: f64,
    frame_size: usize,
    hop_size: usize,
    fft_size: usize,
    bands: Vec<(usize, usize)>,
}

impl PesqCalculator {
    /// Whether this implementation conforms to ITU-T P.862. It does not —
    /// see the type-level documentation.
    pub const STANDARD_CONFORMANT: bool = false;

    /// Create a new PESQ calculator
    pub fn new(sample_rate: f64) -> Self {
        let nominal = (sample_rate * PESQ_FRAME_SECONDS).round();
        let frame_size = if nominal.is_finite() && nominal >= 16.0 {
            (nominal as usize).next_power_of_two()
        } else {
            16
        };
        let hop_size = (frame_size / 2).max(1);
        let fft_size = frame_size * 2;
        let bands = third_octave_bins(
            sample_rate,
            fft_size,
            PESQ_NUM_BANDS,
            THIRD_OCTAVE_LOWEST_CENTER_HZ,
        );

        Self {
            sample_rate,
            frame_size,
            hop_size,
            fft_size,
            bands,
        }
    }

    /// Number of one-third octave bands that fit below Nyquist at this
    /// sample rate.
    pub fn active_bands(&self) -> usize {
        self.bands.len()
    }

    /// Calculate PESQ-like score (1.0 to 4.5 scale, higher is better)
    ///
    /// # Errors
    ///
    /// Returns an error when the signals differ in length, when the sample
    /// rate is too low for the band layout, when the signals are shorter
    /// than one analysis frame, or when the reference contains no frame with
    /// any energy (there is then nothing to score).
    pub fn calculate(&self, reference: &[f64], degraded: &[f64]) -> IoResult<f64> {
        if reference.len() != degraded.len() {
            return Err(IoError::ConfigError("Signal length mismatch".to_string()));
        }
        if self.bands.is_empty() {
            return Err(IoError::ConfigError(format!(
                "Sample rate {} Hz is too low for the perceptual band layout",
                self.sample_rate
            )));
        }

        // Preprocessing: level alignment (P.862 scales both signals to a
        // fixed listening level before the auditory transform).
        let (aligned_ref, aligned_deg) = self.align_levels(reference, degraded);

        let reference_spectra =
            stft_magnitudes(&aligned_ref, self.frame_size, self.hop_size, self.fft_size)?;
        let degraded_spectra =
            stft_magnitudes(&aligned_deg, self.frame_size, self.hop_size, self.fft_size)?;

        let mut distortion_sum = 0.0;
        let mut scored_frames = 0usize;

        for (reference_magnitudes, degraded_magnitudes) in
            reference_spectra.iter().zip(degraded_spectra.iter())
        {
            let reference_loudness = self.band_loudness(reference_magnitudes);
            let degraded_loudness = self.band_loudness(degraded_magnitudes);

            let reference_total: f64 = reference_loudness.iter().sum();
            if reference_total <= f64::MIN_POSITIVE {
                // Silent reference frame: nothing to be distorted.
                continue;
            }
            let asymmetry_floor = (reference_total / reference_loudness.len().max(1) as f64)
                * PESQ_ASYMMETRY_FLOOR_FRACTION;

            let mut frame_distortion = 0.0;
            for (&loud_ref, &loud_deg) in reference_loudness.iter().zip(degraded_loudness.iter()) {
                // Masking: differences below a quarter of the quieter of the
                // two loudness values are inaudible (ITU-T P.862 §10.3).
                let audible = ((loud_ref - loud_deg).abs()
                    - PESQ_MASK_FACTOR * loud_ref.min(loud_deg))
                .max(0.0);
                // Asymmetry: energy *added* by the degradation (noise,
                // aliasing) is more objectionable than energy removed.
                let asymmetry = ((loud_deg + asymmetry_floor) / (loud_ref + asymmetry_floor))
                    .powf(PESQ_ASYMMETRY_EXPONENT)
                    .clamp(1.0, PESQ_MAX_ASYMMETRY);
                frame_distortion += asymmetry * audible;
            }

            // Normalising by the frame's reference loudness makes the
            // distortion scale invariant.
            distortion_sum += frame_distortion / reference_total;
            scored_frames += 1;
        }

        if scored_frames == 0 {
            return Err(IoError::SignalError(
                "Reference signal has no energy: there is nothing to score".to_string(),
            ));
        }

        let avg_distortion = distortion_sum / scored_frames as f64;

        // Map distortion to PESQ-like scale (1.0 to 4.5)
        // Lower distortion = higher score
        let pesq_score = 4.5 - (avg_distortion * PESQ_DISTORTION_GAIN).min(3.5);

        Ok(pesq_score.clamp(1.0, 4.5))
    }

    /// Specific loudness of every band of one frame (Zwicker's power law).
    fn band_loudness(&self, magnitudes: &[f64]) -> Vec<f64> {
        self.bands
            .iter()
            .map(|&(lower, upper)| {
                let energy = band_energy(magnitudes, lower, upper) / self.fft_size as f64;
                energy.powf(PESQ_LOUDNESS_EXPONENT)
            })
            .collect()
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
}

/// MOS (Mean Opinion Score) predictor
///
/// # Status — not a subjective-listening-test predictor
///
/// This is a fixed weighted blend of the SNR, the PESQ-like score and the
/// STOI score of this crate. It has not been fitted to, or validated
/// against, any subjective listening-test corpus, and it inherits the
/// caveats of [`PesqCalculator`]. Treat the output as a comparative index,
/// not as a predicted MOS.
#[derive(Debug, Clone)]
pub struct MosPredictor {
    snr_calc: SnrCalculator,
    pesq_calc: PesqCalculator,
    stoi_calc: StoiCalculator,
}

impl MosPredictor {
    /// Whether the aggregated metrics conform to their namesake standards.
    /// They do not — see the type-level documentation.
    pub const STANDARD_CONFORMANT: bool = false;

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
    /// PESQ-like score from [`PesqCalculator`] (1.0-4.5, not ITU-T P.862)
    pub pesq_score: f64,
    /// STOI score (0.0-1.0)
    pub stoi_score: f64,
    /// Predicted Mean Opinion Score from [`MosPredictor`] (1.0-5.0)
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
///
/// Scores 20 ms frames on three fidelity dimensions — waveform correlation,
/// broadband energy ratio and perceptual loudness difference — and maps
/// their mean onto a 1.0..5.0 scale.
///
/// # Status — not ITU-T P.863 conformant
///
/// [`STANDARD_CONFORMANT`](Self::STANDARD_CONFORMANT) is `false`. This is
/// **not** POLQA: there is no time alignment, no super-wideband auditory
/// model, no idealisation stage and no regression onto the MOS-LQO scale,
/// and the result carries no ITU-T P.863 licence or certification. Use it to
/// rank degradations of the same reference, not as an absolute quality
/// figure.
#[derive(Debug, Clone)]
pub struct PolqaCalculator {
    #[allow(dead_code)]
    sample_rate: f64,
    frame_size: usize,
}

impl PolqaCalculator {
    /// Whether this implementation conforms to ITU-T P.863. It does not —
    /// see the type-level documentation.
    pub const STANDARD_CONFORMANT: bool = false;

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
            let energy_quality = self.energy_ratio_fidelity(ref_frame, deg_frame);
            let loudness_quality = self.loudness_fidelity(ref_frame, deg_frame);

            // Combined quality
            let frame_quality = (temporal_quality + energy_quality + loudness_quality) / 3.0;
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

    /// Assess broadband energy fidelity
    ///
    /// This is a *broadband* energy ratio, not a spectral comparison: it
    /// cannot see how the energy is distributed across frequency. The
    /// spectral sensitivity of this metric comes from
    /// [`temporal_fidelity`](Self::temporal_fidelity), which correlates the
    /// waveforms sample by sample.
    fn energy_ratio_fidelity(&self, reference: &[f64], degraded: &[f64]) -> f64 {
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

    /// Deterministic pseudo-random sequence in [-1, 1] (xorshift, no `rand`).
    fn pseudo_random(seed: u32, n: usize) -> Vec<f64> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f64 / u32::MAX as f64) * 2.0 - 1.0
            })
            .collect()
    }

    /// Broadband, amplitude-modulated test signal.
    ///
    /// STOI correlates band *envelopes* across 30 frames, so a stationary
    /// tone carries no information for it; speech-like modulation does.
    fn modulated_noise(sample_rate: f64, n: usize, seed: u32) -> Vec<f64> {
        pseudo_random(seed, n)
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let t = i as f64 / sample_rate;
                let envelope = 0.55 + 0.45 * (2.0 * PI * 3.0 * t).sin();
                value * envelope
            })
            .collect()
    }

    fn rms(signal: &[f64]) -> f64 {
        if signal.is_empty() {
            return 0.0;
        }
        (signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64).sqrt()
    }

    /// Rescale `signal` so its overall RMS equals the reference's.
    fn match_rms(reference: &[f64], signal: &[f64]) -> Vec<f64> {
        let target = rms(reference);
        let current = rms(signal);
        if current <= 0.0 {
            return signal.to_vec();
        }
        signal.iter().map(|&x| x * target / current).collect()
    }

    /// Rescale `signal` frame by frame so every frame has exactly the same
    /// RMS as the corresponding reference frame. A degradation built this
    /// way is invisible to any envelope-only metric.
    fn match_frame_rms(reference: &[f64], signal: &[f64], frame: usize) -> Vec<f64> {
        let mut out = signal.to_vec();
        let mut start = 0;
        while start < reference.len() {
            let end = (start + frame).min(reference.len()).min(out.len());
            if start >= end {
                break;
            }
            let (Some(reference_frame), Some(out_frame)) =
                (reference.get(start..end), out.get_mut(start..end))
            else {
                break;
            };
            let target = rms(reference_frame);
            let current = rms(out_frame);
            if current > 0.0 {
                let gain = target / current;
                for value in out_frame.iter_mut() {
                    *value *= gain;
                }
            }
            start = end;
        }
        out
    }

    /// Crude 8-tap moving-average low-pass (first null at `fs / 8`).
    fn low_pass(signal: &[f64]) -> Vec<f64> {
        const TAPS: usize = 8;
        (0..signal.len())
            .map(|i| {
                let start = i.saturating_sub(TAPS - 1);
                let slice = signal.get(start..=i).unwrap_or(&[]);
                if slice.is_empty() {
                    0.0
                } else {
                    slice.iter().sum::<f64>() / slice.len() as f64
                }
            })
            .collect()
    }

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

    // NOTE: the STOI/MOS tests below used to feed a stationary 440 Hz sine.
    // Real STOI correlates band *envelopes* over 30-frame segments, for
    // which a stationary tone carries no information, so they now use a
    // speech-like amplitude-modulated broadband signal instead.

    #[test]
    fn test_stoi_calculation() {
        let sample_rate = 8000.0;
        let stoi_calc = StoiCalculator::new(sample_rate);

        let n = 8000; // 1 second
        let reference = modulated_noise(sample_rate, n, 0x1357_9bdf);
        let degraded = reference.clone();

        let stoi = stoi_calc.calculate(&reference, &degraded);
        assert!(stoi.is_ok(), "{:?}", stoi.err());
        let stoi_value = stoi.unwrap();
        assert!(stoi_value.is_finite());
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
        let reference = modulated_noise(sample_rate, n, 0x2468_ace0);
        let degraded = reference.clone();

        let mos = mos_predictor.predict(&reference, &degraded);
        assert!(mos.is_ok(), "{:?}", mos.err());
        let mos_value = mos.unwrap();
        assert!(mos_value.is_finite());
        assert!((1.0..=5.0).contains(&mos_value));
    }

    #[test]
    fn test_quality_metrics() {
        let sample_rate = 8000.0;
        let mos_predictor = MosPredictor::new(sample_rate);

        let n = 8000;
        let reference = modulated_noise(sample_rate, n, 0x0f0f_0f0f);
        let degraded = reference.clone();

        let metrics = mos_predictor.detailed_metrics(&reference, &degraded);
        assert!(metrics.is_ok(), "{:?}", metrics.err());

        let m = metrics.unwrap();
        assert!(m.snr > 0.0);
        assert!(m.pesq_score >= 1.0 && m.pesq_score <= 4.5);
        assert!(m.stoi_score >= 0.0 && m.stoi_score <= 1.0);
        assert!(m.mos >= 1.0 && m.mos <= 5.0);
        assert!(m.snr.is_finite() && m.mos.is_finite());
    }

    // === STOI: real band-envelope correlation (high, id=300) ===
    //
    // The old kernel was a single time-domain Pearson correlation of the
    // raw 256 ms frame with no band decomposition at all, so it scored a
    // polarity-inverted (perfectly intelligible) signal at ~-1 and ignored
    // the one-third octave structure entirely.

    #[test]
    fn test_stoi_identical_signals_score_one() {
        let sample_rate = 10_000.0;
        let stoi_calc = StoiCalculator::new(sample_rate);
        let reference = modulated_noise(sample_rate, 10_000, 0xdead_1234);

        let score = stoi_calc
            .calculate(&reference, &reference)
            .expect("identical signals must be scorable");
        assert!(
            score > 0.99,
            "identical signals should score ~1.0, got {score}"
        );
    }

    #[test]
    fn test_stoi_is_insensitive_to_polarity_inversion() {
        // Inverting the polarity of a signal does not change what a
        // listener hears; the old broadband waveform correlation scored it
        // at -1 (clamped to 0.0), i.e. "unintelligible".
        let sample_rate = 10_000.0;
        let stoi_calc = StoiCalculator::new(sample_rate);
        let reference = modulated_noise(sample_rate, 10_000, 0xbeef_5678);
        let inverted: Vec<f64> = reference.iter().map(|&x| -x).collect();

        let score = stoi_calc
            .calculate(&reference, &inverted)
            .expect("polarity inversion must be scorable");
        assert!(
            score > 0.99,
            "polarity inversion must not change intelligibility, got {score}"
        );
    }

    #[test]
    fn test_stoi_penalizes_uncorrelated_noise() {
        let sample_rate = 10_000.0;
        let stoi_calc = StoiCalculator::new(sample_rate);
        let reference = modulated_noise(sample_rate, 10_000, 0x1111_2222);
        // A different, unmodulated noise realisation at the same level.
        let unrelated = match_rms(&reference, &pseudo_random(0x3333_4444, reference.len()));

        let clean = stoi_calc.calculate(&reference, &reference).unwrap();
        let noisy = stoi_calc.calculate(&reference, &unrelated).unwrap();
        assert!(
            noisy < clean - 0.3,
            "an unrelated signal must score far below the reference: {noisy} vs {clean}"
        );
    }

    #[test]
    fn test_stoi_rejects_too_short_signals() {
        let stoi_calc = StoiCalculator::new(10_000.0);
        let short = modulated_noise(10_000.0, 1000, 0x5555_6666);
        assert!(stoi_calc.calculate(&short, &short).is_err());
    }

    #[test]
    fn test_stoi_rejects_silent_reference() {
        // A silent reference has no intelligibility to measure; returning a
        // number here would be fabricating one.
        let stoi_calc = StoiCalculator::new(10_000.0);
        let silent = vec![0.0f64; 10_000];
        assert!(stoi_calc.calculate(&silent, &silent).is_err());
    }

    #[test]
    fn test_stoi_uses_all_fifteen_bands_at_speech_rates() {
        assert_eq!(StoiCalculator::new(10_000.0).active_bands(), 15);
        assert_eq!(StoiCalculator::new(16_000.0).active_bands(), 15);
        // Far below the band range there is nothing left to correlate.
        assert!(StoiCalculator::new(100.0).active_bands() < 15);
    }

    // === PESQ: spectral sensitivity (high, id=300) ===
    //
    // `calculate` used to score frames purely on |rms_ref^0.6 - rms_deg^0.6|
    // after a global level alignment, so any degradation that preserved the
    // frame envelope — band limiting, noise, even a completely different
    // signal — scored the maximum 4.5.

    #[test]
    fn test_pesq_identical_signals_score_maximum() {
        let sample_rate = 8000.0;
        let pesq_calc = PesqCalculator::new(sample_rate);
        let reference = modulated_noise(sample_rate, 8000, 0x7777_8888);

        let score = pesq_calc.calculate(&reference, &reference).unwrap();
        assert!(
            (score - 4.5).abs() < 1e-6,
            "identical signals must score 4.5, got {score}"
        );
    }

    #[test]
    fn test_pesq_detects_band_limiting_with_matched_level() {
        let sample_rate = 8000.0;
        let pesq_calc = PesqCalculator::new(sample_rate);
        let reference = modulated_noise(sample_rate, 8000, 0x9999_aaaa);
        let degraded = match_rms(&reference, &low_pass(&reference));

        // The RMS matches the reference exactly, so the old envelope-only
        // metric returned 4.5 here.
        assert!(
            (rms(&degraded) - rms(&reference)).abs() < 1e-9,
            "test setup: levels must match"
        );

        let score = pesq_calc.calculate(&reference, &degraded).unwrap();
        assert!(score < 4.0, "band limiting must be penalised, got {score}");
    }

    #[test]
    fn test_pesq_detects_additive_noise_with_matched_frame_energy() {
        let sample_rate = 8000.0;
        let pesq_calc = PesqCalculator::new(sample_rate);
        let reference: Vec<f64> = (0..8000)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
            .collect();
        let noise = pseudo_random(0xbbbb_cccc, reference.len());
        let noisy: Vec<f64> = reference
            .iter()
            .zip(noise.iter())
            .map(|(&clean, &n)| clean + 0.5 * n)
            .collect();
        // Every 32 ms frame carries exactly the reference's energy, so the
        // old envelope-only metric saw zero distortion here.
        let degraded = match_frame_rms(&reference, &noisy, 256);

        let score = pesq_calc.calculate(&reference, &degraded).unwrap();
        assert!(
            score < 4.0,
            "additive noise with matched frame energy must be penalised, got {score}"
        );
    }

    #[test]
    fn test_pesq_detects_frequency_shift_with_matched_level() {
        // Same waveform family, identical RMS, completely different
        // spectrum — the degradation an envelope-only metric is
        // structurally blind to (it used to return 4.5).
        let sample_rate = 8000.0;
        let pesq_calc = PesqCalculator::new(sample_rate);
        let reference: Vec<f64> = (0..8000)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
            .collect();
        let degraded: Vec<f64> = (0..8000)
            .map(|i| (2.0 * PI * 1800.0 * i as f64 / sample_rate).sin())
            .collect();
        assert!(
            (rms(&degraded) - rms(&reference)).abs() < 1e-3,
            "test setup: levels must match"
        );

        let score = pesq_calc.calculate(&reference, &degraded).unwrap();
        assert!(
            score < 3.5,
            "a spectrally displaced tone must be heavily penalised, got {score}"
        );
    }

    #[test]
    fn test_pesq_ranks_noise_levels_monotonically() {
        let sample_rate = 8000.0;
        let pesq_calc = PesqCalculator::new(sample_rate);
        let reference = modulated_noise(sample_rate, 8000, 0x0123_4567);
        let noise = pseudo_random(0x89ab_cdef, reference.len());

        let mut previous = f64::INFINITY;
        for level in [0.0, 0.05, 0.2, 0.6] {
            let noisy: Vec<f64> = reference
                .iter()
                .zip(noise.iter())
                .map(|(&clean, &n)| clean + level * n)
                .collect();
            let score = pesq_calc.calculate(&reference, &noisy).unwrap();
            assert!(
                score <= previous + 1e-9,
                "more noise must never score higher: level {level} gave {score} after {previous}"
            );
            previous = score;
        }
        assert!(
            previous < 4.0,
            "heavy noise must land clearly below the clean score, got {previous}"
        );
    }

    #[test]
    fn test_pesq_rejects_too_short_signals() {
        // The old code returned Ok(1.0) — a fabricated "Bad" verdict — for
        // signals it never actually measured.
        let pesq_calc = PesqCalculator::new(8000.0);
        let short = vec![0.5f64; 16];
        assert!(pesq_calc.calculate(&short, &short).is_err());
    }

    #[test]
    fn test_pesq_rejects_silent_reference() {
        let pesq_calc = PesqCalculator::new(8000.0);
        let silent = vec![0.0f64; 8000];
        assert!(pesq_calc.calculate(&silent, &silent).is_err());
    }

    // === Honest naming (medium, id=360) ===

    #[test]
    fn test_standard_conformance_flags_are_false() {
        // Compile-time: flipping any of these to `true` without an actual
        // conformance validation must not build.
        const {
            assert!(!PesqCalculator::STANDARD_CONFORMANT);
            assert!(!PolqaCalculator::STANDARD_CONFORMANT);
            assert!(!StoiCalculator::STANDARD_CONFORMANT);
            assert!(!MosPredictor::STANDARD_CONFORMANT);
        }
    }

    // === Shared band / STFT helpers ===

    #[test]
    fn test_third_octave_bins_are_ordered_and_within_nyquist() {
        let bands = third_octave_bins(10_000.0, 512, 15, 150.0);
        assert_eq!(bands.len(), 15);

        let mut previous_lower = 0usize;
        for &(lower, upper) in &bands {
            assert!(lower < upper, "empty band {lower}..{upper}");
            assert!(upper <= 512 / 2 + 1, "band {upper} exceeds Nyquist bin");
            assert!(lower >= previous_lower, "bands must be ordered");
            previous_lower = lower;
        }
    }

    #[test]
    fn test_third_octave_bins_drop_bands_above_nyquist() {
        // At 2 kHz only the bands below 1 kHz survive.
        let bands = third_octave_bins(2000.0, 512, 15, 150.0);
        assert!(!bands.is_empty());
        assert!(bands.len() < 15);
    }

    #[test]
    fn test_stft_magnitudes_finds_a_pure_tone() {
        let sample_rate = 8000.0;
        let fft_size = 512;
        let frame = 256;
        let tone: Vec<f64> = (0..4096)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / sample_rate).sin())
            .collect();

        let spectra = stft_magnitudes(&tone, frame, frame / 2, fft_size).unwrap();
        assert!(!spectra.is_empty());

        let expected_bin = (1000.0 * fft_size as f64 / sample_rate).round() as usize;
        for magnitudes in spectra.iter().skip(1) {
            let peak = magnitudes
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(bin, _)| bin)
                .unwrap_or(usize::MAX);
            assert!(
                peak.abs_diff(expected_bin) <= 1,
                "peak at bin {peak}, expected near {expected_bin}"
            );
        }
    }

    #[test]
    fn test_stft_magnitudes_rejects_short_input() {
        assert!(stft_magnitudes(&[0.0; 10], 256, 128, 512).is_err());
    }

    #[test]
    fn test_pearson_correlation_reports_undefined_for_constants() {
        assert!(pearson_correlation(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
        assert!(pearson_correlation(&[1.0], &[1.0]).is_none());
        let rho = pearson_correlation(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap();
        assert!((rho - 1.0).abs() < 1e-12, "rho = {rho}");
        let anti = pearson_correlation(&[1.0, 2.0, 3.0], &[-2.0, -4.0, -6.0]).unwrap();
        assert!((anti + 1.0).abs() < 1e-12, "rho = {anti}");
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
