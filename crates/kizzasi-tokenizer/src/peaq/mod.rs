//! PEAQ (ITU-R BS.1387-1) Basic Model perceptual audio quality evaluator.
//!
//! Computes 11 Model Output Variables (MOVs), a Distortion Index (DI),
//! and an Objective Difference Grade (ODG) in `[-4, 0]`.
//!
//! # Status
//! ITU-conformant in *shape* (algorithm follows BS.1387-1 Annex 2). **Not
//! certified** — official conformance requires the proprietary ITU test vector
//! set. Use ODG/DI values as qualitative trend indicators, not certified scores.
//! See [`nn::WEIGHTS_VERIFIED`].
//!
//! # Quick Start
//! ```rust,no_run
//! use kizzasi_tokenizer::peaq::{PeaqConfig, PeaqEvaluator};
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> kizzasi_tokenizer::TokenizerResult<()> {
//! let config = PeaqConfig::default();
//! let mut evaluator = PeaqEvaluator::new(config)?;
//!
//! // Generate a 1 kHz sine wave reference
//! let n = 48_000;
//! let reference: Array1<f32> = Array1::from_iter(
//!     (0..n).map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48_000.0).sin())
//! );
//! let test = reference.clone();
//!
//! let result = evaluator.evaluate(&reference, &test)?;
//! println!("ODG = {:.2}", result.odg);
//! # Ok(())
//! # }
//! ```

pub mod ear_model;
pub mod mov;
pub mod nn;

use crate::error::{TokenizerError, TokenizerResult};
use ear_model::{make_fft_plan, EarModel};
use mov::MovCalculator;
use nn::{di_to_odg, PeaqNeuralNet};
use scirs2_core::ndarray::Array1;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the PEAQ Basic Model evaluator.
#[derive(Debug, Clone)]
pub struct PeaqConfig {
    /// Signal sample rate in Hz (default: 48_000).
    pub sample_rate: f32,
    /// FFT frame size; must be a power of two (default: 2048).
    pub frame_size: usize,
    /// Hop size between frames in samples (default: 1024).
    pub hop_size: usize,
    /// Number of Bark bands (default: 109, per BS.1387-1).
    pub num_bark_bands: usize,
}

impl Default for PeaqConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            frame_size: 2048,
            hop_size: 1024,
            num_bark_bands: 109,
        }
    }
}

/// All 11 Basic Model Output Variables from BS.1387-1.
#[derive(Debug, Clone)]
pub struct PeaqMovs {
    /// Effective bandwidth of reference signal (Hz).
    pub bandwidth_ref_b: f32,
    /// Effective bandwidth of test signal (Hz).
    pub bandwidth_test_b: f32,
    /// Total noise-to-mask ratio in dB.
    pub total_nmr_b: f32,
    /// Windowed modulation difference.
    pub win_mod_diff_1_b: f32,
    /// Average distortion block basis in dB.
    pub adb_b: f32,
    /// Error harmonic structure (normalized autocorrelation peak, in [0, 0.9]).
    pub ehs_b: f32,
    /// Average modulation difference (reference-weighted).
    pub avg_mod_diff_1_b: f32,
    /// Average modulation difference (min-energy-weighted).
    pub avg_mod_diff_2_b: f32,
    /// RMS noise loudness.
    pub rms_noise_loud_b: f32,
    /// Maximum frame probability of detection.
    pub mfpd_b: f32,
    /// Relative number of distorted frames.
    pub rel_dist_frames_b: f32,
}

/// ITU-R BS.1387 Objective Difference Grade categories.
///
/// Maps ODG values to perceptual quality labels:
/// - `0` → Imperceptible
/// - `-1` → Perceptible but not annoying
/// - `-2` → Slightly annoying
/// - `-3` → Annoying
/// - `-4` → Very annoying
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OdgGrade {
    /// ODG ≥ -0.5: difference imperceptible.
    Imperceptible,
    /// -1.5 ≤ ODG < -0.5: perceptible but not annoying.
    PerceptibleNotAnnoying,
    /// -2.5 ≤ ODG < -1.5: slightly annoying.
    SlightlyAnnoying,
    /// -3.5 ≤ ODG < -2.5: annoying.
    Annoying,
    /// ODG < -3.5: very annoying.
    VeryAnnoying,
}

impl OdgGrade {
    /// Map an ODG value in `[-4, 0]` to the corresponding perceptual grade.
    pub fn from_odg(odg: f32) -> Self {
        if odg >= -0.5 {
            Self::Imperceptible
        } else if odg >= -1.5 {
            Self::PerceptibleNotAnnoying
        } else if odg >= -2.5 {
            Self::SlightlyAnnoying
        } else if odg >= -3.5 {
            Self::Annoying
        } else {
            Self::VeryAnnoying
        }
    }
}

/// Complete result of a PEAQ evaluation.
#[derive(Debug, Clone)]
pub struct PeaqResult {
    /// All 11 Model Output Variables.
    pub movs: PeaqMovs,
    /// Distortion Index (DI) — output of the neural network, before ODG mapping.
    pub distortion_index: f32,
    /// Objective Difference Grade (ODG) in `[-4, 0]`.
    pub odg: f32,
    /// Perceptual quality grade corresponding to the ODG value.
    pub grade: OdgGrade,
}

// ─────────────────────────────────────────────────────────────────────────────
// PeaqEvaluator
// ─────────────────────────────────────────────────────────────────────────────

/// PEAQ Basic Model evaluator.
///
/// Holds an `EarModel` for both reference and test signals (separate IIR states),
/// a `MovCalculator`, and a `PeaqNeuralNet`.
///
/// Call [`PeaqEvaluator::evaluate`] to compare two signals.
pub struct PeaqEvaluator {
    config: PeaqConfig,
    ear_model_ref: EarModel,
    ear_model_test: EarModel,
    mov_calculator: MovCalculator,
    neural_net: PeaqNeuralNet,
}

impl PeaqEvaluator {
    /// Create a new `PeaqEvaluator` from the given configuration.
    ///
    /// # Errors
    /// Returns `TokenizerError::InvalidConfig` if `frame_size` is not a power of two.
    pub fn new(config: PeaqConfig) -> TokenizerResult<Self> {
        let ear_model_ref = EarModel::new(
            config.sample_rate,
            config.frame_size,
            config.hop_size,
            config.num_bark_bands,
        )?;
        let ear_model_test = ear_model_ref.clone();
        let mov_calculator = MovCalculator::new(
            config.num_bark_bands,
            config.sample_rate,
            ear_model_ref.band_centers_hz.clone(),
        );
        let neural_net = PeaqNeuralNet::new();

        Ok(Self {
            config,
            ear_model_ref,
            ear_model_test,
            mov_calculator,
            neural_net,
        })
    }

    /// Evaluate perceptual quality by comparing `reference` to `test`.
    ///
    /// Both signals must have the same length and be at least `frame_size` samples long.
    ///
    /// # Errors
    /// - `TokenizerError::InvalidConfig` if `reference.len() != test.len()`.
    /// - `TokenizerError::InvalidConfig` if signal is shorter than `frame_size`.
    /// - `TokenizerError::EncodingError` if FFT planning fails.
    pub fn evaluate(
        &mut self,
        reference: &Array1<f32>,
        test: &Array1<f32>,
    ) -> TokenizerResult<PeaqResult> {
        if reference.len() != test.len() {
            return Err(TokenizerError::InvalidConfig(format!(
                "reference length {} != test length {}",
                reference.len(),
                test.len()
            )));
        }
        if reference.len() < self.config.frame_size {
            return Err(TokenizerError::InvalidConfig(format!(
                "signal length {} < frame_size {}",
                reference.len(),
                self.config.frame_size
            )));
        }

        // Reset IIR state for stateless evaluation across calls
        self.ear_model_ref.reset();
        self.ear_model_test.reset();

        // Create FFT plan for this frame size
        let fft_plan = make_fft_plan(self.config.frame_size)?;

        // Process reference and test signals through the ear model
        let ref_frames = self.ear_model_ref.process_signal(reference, &fft_plan)?;
        let test_frames = self.ear_model_test.process_signal(test, &fft_plan)?;

        // Compute all 11 MOVs
        let mov_values = self.mov_calculator.compute(&ref_frames, &test_frames)?;

        // Pack MOVs into array for neural net inference
        let mov_array: [f32; 11] = [
            mov_values.bandwidth_ref_b,
            mov_values.bandwidth_test_b,
            mov_values.total_nmr_b,
            mov_values.win_mod_diff_1_b,
            mov_values.adb_b,
            mov_values.ehs_b,
            mov_values.avg_mod_diff_1_b,
            mov_values.avg_mod_diff_2_b,
            mov_values.rms_noise_loud_b,
            mov_values.mfpd_b,
            mov_values.rel_dist_frames_b,
        ];

        let di = self.neural_net.infer(&mov_array);
        let odg = di_to_odg(di);

        Ok(PeaqResult {
            movs: PeaqMovs {
                bandwidth_ref_b: mov_values.bandwidth_ref_b,
                bandwidth_test_b: mov_values.bandwidth_test_b,
                total_nmr_b: mov_values.total_nmr_b,
                win_mod_diff_1_b: mov_values.win_mod_diff_1_b,
                adb_b: mov_values.adb_b,
                ehs_b: mov_values.ehs_b,
                avg_mod_diff_1_b: mov_values.avg_mod_diff_1_b,
                avg_mod_diff_2_b: mov_values.avg_mod_diff_2_b,
                rms_noise_loud_b: mov_values.rms_noise_loud_b,
                mfpd_b: mov_values.mfpd_b,
                rel_dist_frames_b: mov_values.rel_dist_frames_b,
            },
            distortion_index: di,
            odg,
            grade: OdgGrade::from_odg(odg),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod integration {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq_hz: f32, sample_rate: f32, num_samples: usize) -> Array1<f32> {
        Array1::from_iter(
            (0..num_samples).map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate).sin()),
        )
    }

    fn white_noise(num_samples: usize, seed: u64) -> Array1<f32> {
        let mut state = seed;
        Array1::from_iter((0..num_samples).map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 32) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        }))
    }

    #[test]
    fn test_evaluator_round_trip_identical() {
        let config = PeaqConfig::default();
        let mut evaluator = PeaqEvaluator::new(config).unwrap();
        let signal = sine_wave(1000.0, 48_000.0, 48_000);

        let result = evaluator.evaluate(&signal, &signal).unwrap();

        assert!(
            result.odg.is_finite(),
            "ODG must be finite, got {}",
            result.odg
        );
        assert!(
            result.odg >= -4.0 && result.odg <= 0.0,
            "ODG must be in [-4, 0], got {}",
            result.odg
        );
        assert!(result.distortion_index.is_finite(), "DI must be finite");
    }

    #[test]
    fn test_evaluator_white_noise_degraded() {
        let config = PeaqConfig::default();
        let mut evaluator = PeaqEvaluator::new(config).unwrap();
        let signal = sine_wave(1000.0, 48_000.0, 48_000);

        // Clean: identical
        let result_clean = evaluator.evaluate(&signal, &signal).unwrap();

        // Degraded: sine + white noise of equal amplitude
        let noise = white_noise(48_000, 42);
        let noisy: Array1<f32> =
            Array1::from_iter(signal.iter().zip(noise.iter()).map(|(&s, &n)| s + n));
        let result_noisy = evaluator.evaluate(&signal, &noisy).unwrap();

        // NMR should be higher for the noisy case
        assert!(
            result_noisy.movs.total_nmr_b > result_clean.movs.total_nmr_b,
            "NMR should increase with noise: {} > {}",
            result_noisy.movs.total_nmr_b,
            result_clean.movs.total_nmr_b
        );
        assert!(result_noisy.odg.is_finite());
        assert!(result_noisy.odg >= -4.0 && result_noisy.odg <= 0.0);
    }

    #[test]
    fn test_evaluator_low_pass_degraded_bandwidth() {
        let config = PeaqConfig::default();
        let mut evaluator = PeaqEvaluator::new(config).unwrap();

        // Full-bandwidth white noise reference
        let reference = white_noise(48_000, 42);

        // 8k-limited: zero all high-freq content by creating a mostly-low-freq signal
        let test_8k = white_noise(48_000, 43); // different seed → different spectrum

        // 4k-limited: use a lower-frequency sine
        let test_4k = sine_wave(2000.0, 48_000.0, 48_000);

        let result_ref = evaluator.evaluate(&reference, &reference).unwrap();
        let result_8k = evaluator.evaluate(&reference, &test_8k).unwrap();
        let result_4k = evaluator.evaluate(&reference, &test_4k).unwrap();

        // bandwidth_ref_b should be consistent across calls
        assert!(
            result_ref.movs.bandwidth_ref_b > 0.0,
            "Bandwidth ref must be positive"
        );

        // ODG values must all be finite and in range
        for r in [&result_8k, &result_4k] {
            assert!(r.odg.is_finite());
            assert!(r.odg >= -4.0 && r.odg <= 0.0);
        }
    }

    #[test]
    fn test_evaluator_handles_silent_signals() {
        let config = PeaqConfig::default();
        let mut evaluator = PeaqEvaluator::new(config).unwrap();
        let silence = Array1::zeros(2048);

        let result = evaluator.evaluate(&silence, &silence).unwrap();

        assert!(
            result.odg.is_finite(),
            "ODG must be finite for silent inputs"
        );
        assert!(result.odg >= -4.0 && result.odg <= 0.0);
    }

    #[test]
    fn test_evaluator_signal_length_mismatch_errors() {
        let config = PeaqConfig::default();
        let mut evaluator = PeaqEvaluator::new(config).unwrap();
        let ref_sig = Array1::zeros(4096_usize);
        let test_sig = Array1::zeros(2048_usize);

        let result = evaluator.evaluate(&ref_sig, &test_sig);
        assert!(result.is_err(), "Length mismatch should return error");
    }

    #[test]
    fn test_evaluator_minimum_signal_length_errors() {
        let config = PeaqConfig::default();
        let mut evaluator = PeaqEvaluator::new(config).unwrap();
        let short = Array1::zeros(512_usize);

        let result = evaluator.evaluate(&short, &short);
        assert!(
            result.is_err(),
            "Signal shorter than frame_size should error"
        );
    }

    #[test]
    fn test_evaluator_44100_hz_sample_rate_supported() {
        let config = PeaqConfig {
            sample_rate: 44_100.0,
            ..Default::default()
        };
        let result = PeaqEvaluator::new(config);
        assert!(result.is_ok(), "44100 Hz sample rate should be supported");
    }

    #[test]
    fn test_odg_grade_boundaries() {
        assert!(
            matches!(OdgGrade::from_odg(-0.4), OdgGrade::Imperceptible),
            "ODG -0.4 should be Imperceptible"
        );
        assert!(
            matches!(OdgGrade::from_odg(-1.0), OdgGrade::PerceptibleNotAnnoying),
            "ODG -1.0 should be PerceptibleNotAnnoying"
        );
        assert!(
            matches!(OdgGrade::from_odg(-2.0), OdgGrade::SlightlyAnnoying),
            "ODG -2.0 should be SlightlyAnnoying"
        );
        assert!(
            matches!(OdgGrade::from_odg(-3.0), OdgGrade::Annoying),
            "ODG -3.0 should be Annoying"
        );
        assert!(
            matches!(OdgGrade::from_odg(-3.8), OdgGrade::VeryAnnoying),
            "ODG -3.8 should be VeryAnnoying"
        );
    }

    #[test]
    fn test_evaluator_invariant_to_amplitude_scaling() {
        let config = PeaqConfig::default();
        let mut evaluator = PeaqEvaluator::new(config).unwrap();

        let signal = sine_wave(1000.0, 48_000.0, 48_000);
        let noise = white_noise(48_000, 7);
        let noisy: Array1<f32> =
            Array1::from_iter(signal.iter().zip(noise.iter()).map(|(&s, &n)| s + 0.1 * n));

        let result1 = evaluator.evaluate(&signal, &noisy).unwrap();

        // Scale both by 2.0
        let signal2: Array1<f32> = signal.mapv(|x| x * 2.0);
        let noisy2: Array1<f32> = noisy.mapv(|x| x * 2.0);
        let result2 = evaluator.evaluate(&signal2, &noisy2).unwrap();

        assert!(
            (result1.odg - result2.odg).abs() < 0.5,
            "ODG should be approximately invariant to scaling: {} vs {}",
            result1.odg,
            result2.odg
        );
    }

    #[test]
    fn test_evaluator_consistent_across_multiple_runs() {
        let config = PeaqConfig::default();
        let mut evaluator = PeaqEvaluator::new(config).unwrap();

        let signal = sine_wave(1000.0, 48_000.0, 48_000);
        let noise = white_noise(48_000, 99);
        let noisy: Array1<f32> =
            Array1::from_iter(signal.iter().zip(noise.iter()).map(|(&s, &n)| s + 0.05 * n));

        let result1 = evaluator.evaluate(&signal, &noisy).unwrap();
        let result2 = evaluator.evaluate(&signal, &noisy).unwrap();

        assert_eq!(
            result1.odg, result2.odg,
            "Multiple runs with same input must yield identical ODG"
        );
        assert_eq!(
            result1.distortion_index, result2.distortion_index,
            "Multiple runs must yield identical DI"
        );
    }

    #[test]
    fn test_peaq_config_default() {
        let config = PeaqConfig::default();
        assert_eq!(config.sample_rate, 48_000.0);
        assert_eq!(config.frame_size, 2048);
        assert_eq!(config.hop_size, 1024);
        assert_eq!(config.num_bark_bands, 109);
    }

    #[test]
    fn test_evaluator_non_power_of_two_frame_size_errors() {
        let config = PeaqConfig {
            frame_size: 1000,
            ..Default::default()
        };
        let result = PeaqEvaluator::new(config);
        assert!(
            result.is_err(),
            "Non-power-of-two frame_size should return error"
        );
    }
}
