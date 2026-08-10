//! PEAQ Model Output Variables (MOVs) — all 11 Basic Model MOVs.
//!
//! Computes the 11 MOVs defined in BS.1387-1 Annex 2 from per-frame
//! excitation data produced by the ear model.

use super::ear_model::ExcitationFrame;
use crate::error::{TokenizerError, TokenizerResult};

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/// All 11 Basic Model Output Variables from BS.1387-1.
#[derive(Debug, Clone)]
pub(crate) struct MovValues {
    /// Effective bandwidth of reference signal (Hz).
    pub bandwidth_ref_b: f32,
    /// Effective bandwidth of test signal (Hz).
    pub bandwidth_test_b: f32,
    /// Total noise-to-mask ratio in dB.
    pub total_nmr_b: f32,
    /// Windowed modulation difference (weighted).
    pub win_mod_diff_1_b: f32,
    /// Average distortion block basis in dB.
    pub adb_b: f32,
    /// Error harmonic structure (normalized autocorrelation peak).
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

/// Calculator for all 11 Basic Model MOVs.
pub(crate) struct MovCalculator {
    num_bands: usize,
    _sample_rate: f32,
    /// Band center frequencies in Hz (length: `num_bands`).
    band_centers_hz: Vec<f32>,
}

impl MovCalculator {
    /// Construct a new `MovCalculator`.
    pub fn new(num_bands: usize, sample_rate: f32, band_centers_hz: Vec<f32>) -> Self {
        Self {
            num_bands,
            _sample_rate: sample_rate,
            band_centers_hz,
        }
    }

    /// Compute all 11 MOVs from reference and test excitation frame sequences.
    ///
    /// # Errors
    /// Returns `TokenizerError::InvalidInput` if frame counts or band counts are inconsistent.
    pub fn compute(
        &self,
        ref_frames: &[ExcitationFrame],
        test_frames: &[ExcitationFrame],
    ) -> TokenizerResult<MovValues> {
        if ref_frames.len() != test_frames.len() {
            return Err(TokenizerError::invalid_input(
                "peaq/mov",
                format!(
                    "ref_frames ({}) and test_frames ({}) count mismatch",
                    ref_frames.len(),
                    test_frames.len()
                ),
            ));
        }
        if ref_frames.is_empty() {
            return Err(TokenizerError::invalid_input(
                "peaq/mov",
                "no frames to compute MOVs",
            ));
        }
        // Validate band counts
        for (t, (r, te)) in ref_frames.iter().zip(test_frames.iter()).enumerate() {
            if r.energy.len() != self.num_bands || te.energy.len() != self.num_bands {
                return Err(TokenizerError::invalid_input(
                    "peaq/mov",
                    format!(
                        "frame {t} has wrong band count: ref={}, test={}, expected {}",
                        r.energy.len(),
                        te.energy.len(),
                        self.num_bands
                    ),
                ));
            }
        }

        let num_frames = ref_frames.len();
        let eps = 1e-30_f32;

        // ── Precompute per-frame per-band NMR and error energy ─────────────────
        // error_energy[t][b] = max(ref_energy - test_energy, 0)
        // nmr_linear[t][b]   = error_energy / mask_ref
        let mut error_energy: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
        let mut nmr_linear: Vec<Vec<f32>> = Vec::with_capacity(num_frames);

        for (r, te) in ref_frames.iter().zip(test_frames.iter()) {
            let ef: Vec<f32> = r
                .energy
                .iter()
                .zip(te.energy.iter())
                .map(|(&re, &te)| (re - te).max(0.0))
                .collect();
            let nf: Vec<f32> = ef
                .iter()
                .zip(r.mask.iter())
                .map(|(&e, &m)| e / (m + eps))
                .collect();
            error_energy.push(ef);
            nmr_linear.push(nf);
        }

        // ── MOV 1 & 2: BandwidthRefB / BandwidthTestB ─────────────────────────
        let (bandwidth_ref_b, bandwidth_test_b) = self.compute_bandwidths(ref_frames, test_frames);

        // ── MOV 3: TotalNmrB ───────────────────────────────────────────────────
        let total_nmr_b = self.compute_total_nmr(&nmr_linear);

        // ── Per-frame detection probabilities (used by MOV 10 & 11) ───────────
        let nmr_db_frames: Vec<Vec<f32>> = nmr_linear
            .iter()
            .map(|nf| {
                nf.iter()
                    .map(|&n| 10.0 * (n + eps).log10())
                    .collect::<Vec<f32>>()
            })
            .collect();

        let p_det: Vec<f32> = nmr_db_frames
            .iter()
            .map(|frame_nmr_db| self.frame_detection_probability(frame_nmr_db))
            .collect();

        // ── MOV 10: MFPDB ─────────────────────────────────────────────────────
        let mfpd_b = p_det.iter().cloned().fold(0.0_f32, f32::max);

        // ── MOV 11: RelDistFramesB ─────────────────────────────────────────────
        let dist_count = p_det.iter().filter(|&&p| p > 0.5).count();
        let rel_dist_frames_b = dist_count as f32 / num_frames as f32;

        // ── MOV 5: ADBB ───────────────────────────────────────────────────────
        let adb_b = self.compute_adb(&nmr_db_frames);

        // ── MOV 6: EHSB ───────────────────────────────────────────────────────
        let ehs_b = self.compute_ehs(ref_frames, test_frames);

        // ── Modulation indices (used by MOV 4, 7, 8) ──────────────────────────
        let (win_mod_diff_1_b, avg_mod_diff_1_b, avg_mod_diff_2_b) =
            self.compute_modulation_movs(ref_frames, test_frames);

        // ── MOV 9: RmsNoiseLoudB ──────────────────────────────────────────────
        let rms_noise_loud_b = self.compute_rms_noise_loud(&error_energy, ref_frames);

        Ok(MovValues {
            bandwidth_ref_b,
            bandwidth_test_b,
            total_nmr_b,
            win_mod_diff_1_b,
            adb_b,
            ehs_b,
            avg_mod_diff_1_b,
            avg_mod_diff_2_b,
            rms_noise_loud_b,
            mfpd_b,
            rel_dist_frames_b,
        })
    }

    // ── MOV 1 & 2: Effective Bandwidth ────────────────────────────────────────

    fn compute_bandwidths(
        &self,
        ref_frames: &[ExcitationFrame],
        test_frames: &[ExcitationFrame],
    ) -> (f32, f32) {
        // Compute average per-band raw energy over all frames
        let mut avg_ref = vec![0.0_f32; self.num_bands];
        let mut avg_test = vec![0.0_f32; self.num_bands];
        let n = ref_frames.len() as f32;

        for (r, te) in ref_frames.iter().zip(test_frames.iter()) {
            for b in 0..self.num_bands {
                avg_ref[b] += r.raw_energy.get(b).copied().unwrap_or(0.0) / n;
                avg_test[b] += te.raw_energy.get(b).copied().unwrap_or(0.0) / n;
            }
        }

        // Maximum average energy of reference (in dB) — use as anchor for bandwidth
        let max_ref_db = avg_ref
            .iter()
            .map(|&e| 10.0 * (e + 1e-30).log10())
            .fold(f32::NEG_INFINITY, f32::max);

        // Threshold = max energy - 10 dB (bands within 10 dB of peak are "active")
        let bw_threshold_db = max_ref_db - 10.0;

        let bw_ref = self.effective_bandwidth(&avg_ref, bw_threshold_db);
        let bw_test = self.effective_bandwidth(&avg_test, bw_threshold_db);

        (bw_ref, bw_test)
    }

    fn effective_bandwidth(&self, avg_energy: &[f32], threshold_db: f32) -> f32 {
        // Find highest band where energy in dB > threshold_db
        for b in (0..self.num_bands).rev() {
            let e_db = 10.0 * (avg_energy[b] + 1e-30).log10();
            if e_db > threshold_db {
                return self.band_centers_hz.get(b).copied().unwrap_or(0.0);
            }
        }
        // No band exceeds threshold; return lowest band frequency
        self.band_centers_hz.first().copied().unwrap_or(0.0)
    }

    // ── MOV 3: Total NMR ──────────────────────────────────────────────────────

    fn compute_total_nmr(&self, nmr_linear: &[Vec<f32>]) -> f32 {
        // NMR_frame[t] = mean(nmr_linear[t][:]) in dB
        let eps = 1e-30_f32;
        let frame_nmr_db: Vec<f32> = nmr_linear
            .iter()
            .map(|nf| {
                let mean = nf.iter().sum::<f32>() / (self.num_bands as f32).max(1.0);
                10.0 * (mean + eps).log10()
            })
            .collect();
        frame_nmr_db.iter().sum::<f32>() / (frame_nmr_db.len() as f32).max(1.0)
    }

    // ── MOV 5: ADB (Average Distortion Block Basis) ───────────────────────────

    fn compute_adb(&self, nmr_db_frames: &[Vec<f32>]) -> f32 {
        // Frame NMR dB = mean of per-band NMR dB
        let frame_nmr_db: Vec<f32> = nmr_db_frames
            .iter()
            .map(|nf| nf.iter().sum::<f32>() / (self.num_bands as f32).max(1.0))
            .collect();

        // Average over distorted frames (where mean NMR dB > 0)
        let distorted: Vec<f32> = frame_nmr_db.iter().cloned().filter(|&v| v > 0.0).collect();

        if distorted.is_empty() {
            0.0
        } else {
            distorted.iter().sum::<f32>() / distorted.len() as f32
        }
    }

    // ── MOV 6: EHS (Error Harmonic Structure) ─────────────────────────────────

    fn compute_ehs(&self, ref_frames: &[ExcitationFrame], test_frames: &[ExcitationFrame]) -> f32 {
        // Per-frame band-average error signal
        let e_seq: Vec<f32> = ref_frames
            .iter()
            .zip(test_frames.iter())
            .map(|(r, te)| {
                let diff: f32 = r
                    .energy
                    .iter()
                    .zip(te.energy.iter())
                    .map(|(&re, &te)| (re - te).abs())
                    .sum::<f32>();
                diff / (self.num_bands as f32).max(1.0)
            })
            .collect();

        let n = e_seq.len();
        if n < 2 {
            return 0.0;
        }

        // Compute energy at lag 0 for normalization
        let e0: f32 = e_seq.iter().map(|&e| e * e).sum::<f32>();
        if e0 < 1e-30 {
            return 0.0;
        }

        // Normalized autocorrelation at lags 1..10
        let max_lag = 10.min(n - 1);
        let mut max_corr = 0.0_f32;
        for lag in 1..=max_lag {
            let corr: f32 = e_seq[..n - lag]
                .iter()
                .zip(e_seq[lag..].iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>();
            let normalized = (corr / (e0 + 1e-30)).abs();
            if normalized > max_corr {
                max_corr = normalized;
            }
        }

        max_corr.min(0.9) // clamp to MOV_AMAX[5] = 0.9
    }

    // ── MOV 4, 7, 8: Modulation Differences ───────────────────────────────────

    fn compute_modulation_movs(
        &self,
        ref_frames: &[ExcitationFrame],
        test_frames: &[ExcitationFrame],
    ) -> (f32, f32, f32) {
        let num_frames = ref_frames.len();
        let eps = 1e-6_f32;

        if num_frames < 2 {
            return (0.0, 0.0, 0.0);
        }

        let mut win_acc = 0.0_f32;
        let mut avg1_acc = 0.0_f32;
        let mut avg2_acc = 0.0_f32;
        let mut total_weight1 = 0.0_f32;
        let mut total_weight2 = 0.0_f32;
        let mut win_samples = 0_u64;

        for t in 1..num_frames {
            for b in 0..self.num_bands {
                let re_cur = ref_frames[t].energy[b];
                let re_prev = ref_frames[t - 1].energy[b];
                let te_cur = test_frames[t].energy[b];
                let te_prev = test_frames[t - 1].energy[b];

                // Modulation index: normalized absolute energy change
                let mod_ref = (re_cur - re_prev).abs() / (re_cur + re_prev + eps);
                let mod_test = (te_cur - te_prev).abs() / (te_cur + te_prev + eps);
                let mod_diff = (mod_test - mod_ref).abs();

                // Weight for MOV 4 & 7: energy of reference (soft gating)
                let w1 = re_cur / (re_cur + eps);
                // Weight for MOV 8: min energy
                let min_e = re_cur.min(te_cur);
                let w2 = min_e / (min_e + eps);

                win_acc += mod_diff * w1;
                avg1_acc += mod_diff * w1;
                avg2_acc += mod_diff * w2;

                total_weight1 += w1;
                total_weight2 += w2;
                win_samples += 1;
            }
        }

        let win_mod_diff_1_b = if win_samples > 0 {
            win_acc / win_samples as f32
        } else {
            0.0
        };

        let avg_mod_diff_1_b = if total_weight1 > eps {
            avg1_acc / total_weight1
        } else {
            0.0
        };

        let avg_mod_diff_2_b = if total_weight2 > eps {
            avg2_acc / total_weight2
        } else {
            0.0
        };

        (win_mod_diff_1_b, avg_mod_diff_1_b, avg_mod_diff_2_b)
    }

    // ── MOV 9: RMS Noise Loudness ──────────────────────────────────────────────

    fn compute_rms_noise_loud(
        &self,
        error_energy: &[Vec<f32>],
        ref_frames: &[ExcitationFrame],
    ) -> f32 {
        // Zwicker loudness approximation from error energy
        // loudness_band = (error / (mask + eps))^0.23
        let eps = 1e-30_f32;

        let loudness_sq_sum: f32 = error_energy
            .iter()
            .zip(ref_frames.iter())
            .map(|(ef, r)| {
                let frame_loud: f32 = ef
                    .iter()
                    .zip(r.mask.iter())
                    .map(|(&e, &m)| (e / (m + eps)).powf(0.23))
                    .sum::<f32>();
                frame_loud * frame_loud
            })
            .sum();

        (loudness_sq_sum / (error_energy.len() as f32).max(1.0)).sqrt()
    }

    // ── Per-frame detection probability ───────────────────────────────────────

    fn frame_detection_probability(&self, nmr_db_frame: &[f32]) -> f32 {
        // Per-band sigmoid detection probability (logistic of NMR dB)
        // p_band(b) = sigmoid(10 * NMR_dB[b])
        // Combined: p_det = 1 - product(1 - p_band[b])
        let mut p_not_det = 1.0_f32;
        for &nmr_db in nmr_db_frame {
            let p_band = sigmoid(10.0 * nmr_db).clamp(0.0, 1.0);
            p_not_det *= 1.0 - p_band;
        }
        (1.0 - p_not_det).clamp(0.0, 1.0)
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_frames(
        num_frames: usize,
        num_bands: usize,
        energy: f32,
    ) -> Vec<ExcitationFrame> {
        (0..num_frames)
            .map(|_| ExcitationFrame {
                energy: vec![energy; num_bands],
                mask: vec![energy; num_bands],
                raw_energy: vec![energy; num_bands],
            })
            .collect()
    }

    fn make_calculator() -> MovCalculator {
        let num_bands = 109;
        let band_centers: Vec<f32> = (0..num_bands)
            .map(|b| 80.0 + (18_000.0 - 80.0) * b as f32 / (num_bands - 1) as f32)
            .collect();
        MovCalculator::new(num_bands, 48_000.0, band_centers)
    }

    #[test]
    fn test_identical_signals_yield_low_nmr() {
        let calc = make_calculator();
        let ref_frames = make_uniform_frames(10, 109, 1.0);
        let test_frames = make_uniform_frames(10, 109, 1.0);

        let movs = calc.compute(&ref_frames, &test_frames).unwrap();

        // Identical signals: error_energy = max(1-1, 0) = 0, so NMR should be very low
        assert!(
            movs.total_nmr_b < 0.0,
            "TotalNmrB should be negative dB for identical signals, got {}",
            movs.total_nmr_b
        );
        assert!(
            movs.rel_dist_frames_b < 0.1,
            "RelDistFramesB should be near 0 for identical signals, got {}",
            movs.rel_dist_frames_b
        );
    }

    #[test]
    fn test_bandwidth_low_pass_signal() {
        let calc = make_calculator();
        // ref: all 109 bands active with energy 1.0
        let ref_frames = make_uniform_frames(10, 109, 1.0);
        // test: only first 60 bands active
        let test_frames: Vec<ExcitationFrame> = (0..10)
            .map(|_| {
                let mut energy = vec![0.0_f32; 109];
                let mut raw = vec![0.0_f32; 109];
                for b in 0..60 {
                    energy[b] = 1.0;
                    raw[b] = 1.0;
                }
                ExcitationFrame {
                    energy,
                    mask: vec![1.0; 109],
                    raw_energy: raw,
                }
            })
            .collect();

        let movs = calc.compute(&ref_frames, &test_frames).unwrap();

        assert!(
            movs.bandwidth_test_b < movs.bandwidth_ref_b,
            "Low-pass test should have narrower bandwidth: test={} < ref={}",
            movs.bandwidth_test_b,
            movs.bandwidth_ref_b
        );
    }

    #[test]
    fn test_added_noise_increases_nmr() {
        let calc = make_calculator();

        // Baseline: identical
        let ref_frames = make_uniform_frames(10, 109, 1.0);
        let test_frames_clean = make_uniform_frames(10, 109, 1.0);
        let movs_clean = calc.compute(&ref_frames, &test_frames_clean).unwrap();

        // Noisy: test has lower energy (distorted).
        // Note: error_energy = max(ref - test, 0) = max(1 - 2, 0) = 0 when test > ref.
        // Use test energy < ref energy to get distortion:
        let test_frames_noisy2: Vec<ExcitationFrame> = (0..10)
            .map(|_| ExcitationFrame {
                energy: vec![0.1; 109], // lower → error = max(1 - 0.1, 0) = 0.9
                mask: vec![1.0; 109],
                raw_energy: vec![0.1; 109],
            })
            .collect();
        let movs_noisy = calc.compute(&ref_frames, &test_frames_noisy2).unwrap();

        assert!(
            movs_noisy.total_nmr_b > movs_clean.total_nmr_b,
            "Noisy test should have higher TotalNmrB: {} > {}",
            movs_noisy.total_nmr_b,
            movs_clean.total_nmr_b
        );
    }

    #[test]
    fn test_distorted_frames_fraction_grows_with_distortion() {
        let calc = make_calculator();
        let ref_frames = make_uniform_frames(20, 109, 1.0);
        // Test has much lower energy → high error relative to mask
        let test_frames: Vec<ExcitationFrame> = (0..20)
            .map(|_| ExcitationFrame {
                energy: vec![1e-6; 109], // very low → huge error
                mask: vec![1e-10; 109],  // very low mask → huge NMR
                raw_energy: vec![1e-6; 109],
            })
            .collect();

        let movs = calc.compute(&ref_frames, &test_frames).unwrap();

        assert!(
            movs.rel_dist_frames_b > 0.5,
            "Heavily distorted signal should have rel_dist_frames_b > 0.5, got {}",
            movs.rel_dist_frames_b
        );
    }

    #[test]
    fn test_modulation_difference_responds_to_amplitude_modulation() {
        let calc = make_calculator();
        // ref: constant energy (no modulation)
        let ref_frames = make_uniform_frames(20, 109, 1.0);
        // test: alternating high/low energy (simulating tremolo)
        let test_frames: Vec<ExcitationFrame> = (0..20)
            .map(|t| {
                let energy_val = if t % 2 == 0 { 2.0_f32 } else { 0.5_f32 };
                ExcitationFrame {
                    energy: vec![energy_val; 109],
                    mask: vec![1.0; 109],
                    raw_energy: vec![energy_val; 109],
                }
            })
            .collect();

        let movs = calc.compute(&ref_frames, &test_frames).unwrap();

        assert!(
            movs.avg_mod_diff_1_b > 0.0,
            "AvgModDiff1B should be > 0 for amplitude-modulated test, got {}",
            movs.avg_mod_diff_1_b
        );
    }

    #[test]
    fn test_frame_count_mismatch_errors() {
        let calc = make_calculator();
        let ref_frames = make_uniform_frames(10, 109, 1.0);
        let test_frames = make_uniform_frames(5, 109, 1.0);

        assert!(
            calc.compute(&ref_frames, &test_frames).is_err(),
            "Frame count mismatch should return error"
        );
    }

    #[test]
    fn test_empty_frames_errors() {
        let calc = make_calculator();
        let result = calc.compute(&[], &[]);
        assert!(result.is_err(), "Empty frames should return error");
    }

    #[test]
    fn test_ehs_bounded_by_0_9() {
        let calc = make_calculator();
        let ref_frames = make_uniform_frames(50, 109, 1.0);
        let test_frames = make_uniform_frames(50, 109, 0.5);
        let movs = calc.compute(&ref_frames, &test_frames).unwrap();

        assert!(
            movs.ehs_b >= 0.0 && movs.ehs_b <= 0.9,
            "EHSB must be in [0, 0.9], got {}",
            movs.ehs_b
        );
    }
}
