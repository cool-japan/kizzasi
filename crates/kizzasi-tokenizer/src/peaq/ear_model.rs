//! Outer/middle ear model + 109-band Bark grouping + spreading/masking.
//!
//! Implements the BS.1387-1 Basic Model ear model pipeline:
//!
//! - **109-band Bark grouping**: reconstructed from Traunmüller formula.
//!   Published band edge values in BS.1387-1 Annex 2 Table B.4 require the
//!   paid ITU standard document. The open reconstruction below matches the
//!   same formula used in the perceptual.rs module.
//! - **Outer/middle ear weighting W(f)**: BS.1387-1 Eq. 7, computed per FFT bin.
//! - **Spreading function**: level-dependent two-slope model per Eq. 19 approximation.
//! - **Time spreading**: first-order IIR per band per Eq. 21.
//! - **Internal noise**: Terhardt absolute threshold model per Eq. 12.

use crate::error::{TokenizerError, TokenizerResult};
use oxifft::{Complex, Direction, Flags, Plan};
use scirs2_core::ndarray::{s, Array1};
use std::f32::consts::PI;

/// Type alias for complex single-precision values.
pub(crate) type Complex32 = Complex<f32>;
/// Type alias for single-precision FFT plans.
pub(crate) type Plan32 = Plan<f32>;

// ─────────────────────────────────────────────────────────────────────────────
// Psychoacoustic helper functions
// ─────────────────────────────────────────────────────────────────────────────

/// Convert frequency in Hz to the Bark scale (Traunmüller formula).
#[inline]
fn hz_to_bark(hz: f32) -> f32 {
    26.81 * hz / (1960.0 + hz) - 0.53
}

/// Inverse Traunmüller: Bark → Hz.
#[inline]
fn bark_to_hz(bark: f32) -> f32 {
    let b = bark + 0.53;
    1960.0 * b / (26.81 - b)
}

/// Outer/middle ear weighting in linear scale (not dB) for FFT bin at `hz`.
///
/// W(f) [dB] = -2.184·(f_kHz)^{-0.8} + 6.5·exp(-0.6·(f_kHz - 3.3)²) - 10^{-3}·(f_kHz)^{3.6}
///
/// This implements BS.1387-1 Eq. 7.
fn outer_middle_ear_weight_linear(hz: f32) -> f32 {
    if hz < 1.0 {
        return 0.0; // avoid singularity near DC
    }
    let f_khz = hz / 1000.0;
    let w_db = -2.184 * f_khz.powf(-0.8) + 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp()
        - 1e-3 * f_khz.powf(3.6);
    // Convert dB to linear amplitude ratio: 10^(w_db/20)
    10.0_f32.powf(w_db / 20.0)
}

/// Terhardt (1979) absolute threshold of hearing in linear energy scale.
///
/// Uses the same formula as `perceptual::absolute_threshold_db` but returns
/// linear energy for direct use in noise floor computation (Eq. 12).
fn absolute_threshold_linear(hz: f32) -> f32 {
    if hz < 1.0 {
        return 1e-3;
    }
    let f_khz = hz / 1000.0;
    let db = 3.64 * f_khz.powf(-0.8) - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp()
        + 1e-3 * f_khz.powf(4.0);
    let db_clamped = db.clamp(-60.0, 120.0);
    10.0_f32.powf(db_clamped / 10.0)
}

/// Compute internal noise per band from absolute threshold (BS.1387-1 Eq. 12).
///
/// `EIN(z) = 0.4 · E_threshold(z)`
#[inline]
fn compute_internal_noise(thresh_linear: f32) -> f32 {
    0.4 * thresh_linear
}

/// Compute 109-band Bark-scale band edges and centers from Traunmüller formula.
///
/// Bands span 80 Hz to min(18000 Hz, sample_rate/2 - 1).
fn compute_band_edges(
    sample_rate: f32,
    _frame_size: usize,
    num_bands: usize,
) -> (Vec<f32>, Vec<f32>) {
    let f_min = 80.0_f32;
    let f_max = 18_000.0_f32.min(sample_rate / 2.0 - 1.0);

    let bark_min = hz_to_bark(f_min);
    let bark_max = hz_to_bark(f_max);

    let mut edges = Vec::with_capacity(num_bands + 1);
    for i in 0..=num_bands {
        let bark = bark_min + (bark_max - bark_min) * i as f32 / num_bands as f32;
        let hz = bark_to_hz(bark).clamp(f_min, f_max);
        edges.push(hz);
    }

    // Centers are midpoints of adjacent edges
    let centers: Vec<f32> = (0..num_bands)
        .map(|i| (edges[i] + edges[i + 1]) / 2.0)
        .collect();

    (edges, centers)
}

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/// Frequency-domain excitation for one frame, per band.
#[derive(Debug, Clone)]
pub(crate) struct ExcitationFrame {
    /// Per-band excitation energy (linear, not dB). Length: `num_bark_bands`.
    pub energy: Vec<f32>,
    /// Per-band masking threshold (linear, not dB). Length: `num_bark_bands`.
    pub mask: Vec<f32>,
    /// Total raw energy at each band (before spreading). Length: `num_bark_bands`.
    pub raw_energy: Vec<f32>,
}

/// State for the ear model (maintained across frames for time spreading).
///
/// Create one instance per channel/signal; call [`EarModel::reset`] between
/// independent evaluations.
#[derive(Clone)]
pub(crate) struct EarModel {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// FFT frame size (must be a power of two).
    pub frame_size: usize,
    /// Hop size between frames in samples.
    pub hop_size: usize,
    /// Number of Bark bands.
    pub num_bands: usize,
    /// Band edge frequencies in Hz. Length: `num_bands + 1`.
    pub band_edges_hz: Vec<f32>,
    /// Band center frequencies in Hz. Length: `num_bands`.
    pub band_centers_hz: Vec<f32>,
    /// Bark-scale centers of each band (precomputed for spreading).
    bark_centers: Vec<f32>,
    /// Internal noise per band (linear energy). Length: `num_bands`.
    internal_noise: Vec<f32>,
    /// IIR time-spreading decay coefficient per band. Length: `num_bands`.
    time_iir_alpha: Vec<f32>,
    /// IIR state per band. Length: `num_bands`. Initialized to 0.
    time_state: Vec<f32>,
    /// Outer/middle ear weights per positive-frequency FFT bin. Length: `frame_size/2 + 1`.
    ear_weights: Vec<f32>,
}

impl EarModel {
    /// Construct a new `EarModel`.
    ///
    /// # Arguments
    /// * `sample_rate` – Signal sample rate in Hz.
    /// * `frame_size`  – FFT frame size; must be a power of two.
    /// * `hop_size`    – Hop size between frames in samples.
    /// * `num_bands`   – Number of Bark bands (typically 109 per BS.1387-1).
    ///
    /// # Errors
    /// Returns `TokenizerError::InvalidConfig` if `frame_size` is not a power of two.
    pub fn new(
        sample_rate: f32,
        frame_size: usize,
        hop_size: usize,
        num_bands: usize,
    ) -> TokenizerResult<Self> {
        if !frame_size.is_power_of_two() {
            return Err(TokenizerError::InvalidConfig(format!(
                "frame_size must be power of two, got {frame_size}"
            )));
        }

        let (band_edges_hz, band_centers_hz) =
            compute_band_edges(sample_rate, frame_size, num_bands);

        let num_bins = frame_size / 2 + 1;
        let bin_hz = sample_rate / frame_size as f32;

        // Pre-compute outer/middle ear weights per FFT bin
        let ear_weights: Vec<f32> = (0..num_bins)
            .map(|k| outer_middle_ear_weight_linear(k as f32 * bin_hz))
            .collect();

        // Internal noise per band (Eq. 12)
        let internal_noise: Vec<f32> = band_centers_hz
            .iter()
            .map(|&hz| compute_internal_noise(absolute_threshold_linear(hz)))
            .collect();

        // Time spreading IIR coefficient per band (Eq. 21):
        //   τ(z) = τ_min + 100 * (τ_100 - τ_min) / f_c(z)
        //   α(z) = exp(-hop_size / (sample_rate * τ(z)))
        let tau_100 = 0.030_f32;
        let tau_min = 0.008_f32;
        let time_iir_alpha: Vec<f32> = band_centers_hz
            .iter()
            .map(|&fc| {
                let tau = (tau_min + 100.0 * (tau_100 - tau_min) / fc.max(1.0)).clamp(tau_min, 1.0);
                (-(hop_size as f32) / (sample_rate * tau)).exp()
            })
            .collect();

        // Precompute Bark-scale centers for spreading function
        let bark_centers: Vec<f32> = band_centers_hz.iter().map(|&hz| hz_to_bark(hz)).collect();

        Ok(Self {
            sample_rate,
            frame_size,
            hop_size,
            num_bands,
            band_edges_hz,
            band_centers_hz,
            bark_centers,
            internal_noise,
            time_iir_alpha,
            time_state: vec![0.0_f32; num_bands],
            ear_weights,
        })
    }

    /// Reset the time-spreading IIR state.
    ///
    /// Call this between independent signal evaluations to avoid state leakage.
    pub fn reset(&mut self) {
        self.time_state.fill(0.0);
    }

    /// Process one frame: apply ear weighting, group to Bark bands, spread, time-smear.
    ///
    /// Returns an [`ExcitationFrame`] with excitation energies and masking thresholds.
    ///
    /// # Arguments
    /// * `frame`    – Slice of `frame_size` time-domain samples.
    /// * `fft_plan` – Pre-computed forward FFT plan for `frame_size`.
    ///
    /// # Errors
    /// Returns `TokenizerError::EncodingError` if `frame.len() != frame_size`.
    pub fn process_frame(
        &mut self,
        frame: &[f32],
        fft_plan: &Plan32,
    ) -> TokenizerResult<ExcitationFrame> {
        if frame.len() != self.frame_size {
            return Err(TokenizerError::encoding(
                "peaq",
                format!(
                    "frame length {} != frame_size {}",
                    frame.len(),
                    self.frame_size
                ),
            ));
        }

        let num_bins = self.frame_size / 2 + 1;

        // Apply Hann window and convert to complex
        let hann_denom = (self.frame_size - 1) as f32;
        let input: Vec<Complex32> = frame
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / hann_denom).cos());
                Complex32::new(s * w, 0.0)
            })
            .collect();

        let mut spectrum: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); self.frame_size];
        fft_plan.execute(&input, &mut spectrum);

        // Apply outer/middle ear weighting to magnitude spectrum
        let weighted_mag: Vec<f32> = spectrum
            .iter()
            .enumerate()
            .take(num_bins)
            .map(|(k, c)| {
                let mag = (c.re * c.re + c.im * c.im).sqrt();
                mag * self.ear_weights[k]
            })
            .collect();

        // Group weighted energy per Bark band
        let bin_hz = self.sample_rate / self.frame_size as f32;
        let mut raw_energy = vec![0.0_f32; self.num_bands];
        for (k, &wm) in weighted_mag.iter().enumerate() {
            let freq = k as f32 * bin_hz;
            // Binary search for the band containing `freq`
            let band_idx = self
                .band_edges_hz
                .partition_point(|&edge| edge <= freq)
                .saturating_sub(1)
                .min(self.num_bands - 1);
            if freq >= self.band_edges_hz[band_idx] && freq < self.band_edges_hz[band_idx + 1] {
                raw_energy[band_idx] += wm * wm;
            }
        }

        // Add internal noise before spreading (Eq. 12)
        let energy_with_noise: Vec<f32> = raw_energy
            .iter()
            .zip(self.internal_noise.iter())
            .map(|(&e, &n)| e + n)
            .collect();

        // Apply level-dependent frequency spreading in Bark domain (Eq. 19)
        let spread_energy = self.frequency_spread(&energy_with_noise);

        // Apply time spreading IIR: E_time[t] = max(E_spread[t], α · E_time[t-1])
        // This models forward masking (temporal smearing per Eq. 21)
        for ((state, alpha), spread) in self
            .time_state
            .iter_mut()
            .zip(self.time_iir_alpha.iter())
            .zip(spread_energy.iter())
        {
            let decayed = *alpha * *state;
            *state = spread.max(decayed);
        }
        let excitation = self.time_state.clone();

        // Masking threshold = max(excitation, internal_noise floor)
        let mask: Vec<f32> = excitation
            .iter()
            .zip(self.internal_noise.iter())
            .map(|(&e, &n)| e.max(n))
            .collect();

        Ok(ExcitationFrame {
            energy: excitation,
            mask,
            raw_energy,
        })
    }

    /// Level-dependent two-slope frequency spreading (BS.1387-1 Eq. 19 approximation).
    ///
    /// For masker at band `s` and target band `t`:
    /// - Lower slope (t < s): 27 dB/Bark
    /// - Upper slope (t > s): level-dependent, `24 + 230/(z_s² + 1) - 0.2·L_s` dB/Bark
    fn frequency_spread(&self, energy: &[f32]) -> Vec<f32> {
        let n = energy.len();
        let mut spread = vec![0.0_f32; n];

        for (&e_s, &z_s) in energy.iter().zip(self.bark_centers.iter()) {
            if e_s <= 1e-30 {
                continue;
            }
            let l_s_db = 10.0 * e_s.log10();

            for (spread_t, &z_t) in spread.iter_mut().zip(self.bark_centers.iter()) {
                let delta_z = z_t - z_s;
                let sf_db = if delta_z < 0.0 {
                    // Lower slope: 27 dB/Bark (delta_z < 0, so sf_db < 0)
                    27.0 * delta_z
                } else {
                    // Upper slope: level-dependent (Eq. 19)
                    let slope = (24.0 + 230.0 / (z_s * z_s + 1.0) - 0.2 * l_s_db).max(0.0);
                    -slope * delta_z
                };
                let sf_linear = 10.0_f32.powf(sf_db.max(-100.0) / 10.0);
                *spread_t += e_s * sf_linear;
            }
        }

        spread
    }

    /// Process a full signal: split into overlapping frames and return per-frame excitations.
    ///
    /// # Arguments
    /// * `signal`   – Input signal as a 1-D array.
    /// * `fft_plan` – Pre-computed forward FFT plan for `frame_size`.
    ///
    /// # Errors
    /// Returns `TokenizerError::EncodingError` if the signal is shorter than `frame_size`.
    pub fn process_signal(
        &mut self,
        signal: &Array1<f32>,
        fft_plan: &Plan32,
    ) -> TokenizerResult<Vec<ExcitationFrame>> {
        let n = signal.len();
        if n < self.frame_size {
            return Err(TokenizerError::encoding(
                "peaq",
                format!(
                    "signal length {n} is less than frame_size {}",
                    self.frame_size
                ),
            ));
        }

        let num_frames = (n - self.frame_size) / self.hop_size + 1;
        let mut frames = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let start = i * self.hop_size;
            let end = start + self.frame_size;
            let frame_view = signal.slice(s![start..end]);
            let frame_slice: Vec<f32> = frame_view.iter().cloned().collect();
            frames.push(self.process_frame(&frame_slice, fft_plan)?);
        }

        Ok(frames)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public constructor helper (used by mod.rs)
// ─────────────────────────────────────────────────────────────────────────────

/// Create a forward FFT plan for `frame_size`.
pub(crate) fn make_fft_plan(frame_size: usize) -> TokenizerResult<Plan32> {
    Plan::<f32>::dft_1d(frame_size, Direction::Forward, Flags::MEASURE)
        .ok_or_else(|| TokenizerError::encoding("peaq", "FFT planning failed"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ear_model() -> EarModel {
        EarModel::new(48_000.0, 2048, 1024, 109).unwrap()
    }

    #[test]
    fn test_outer_middle_ear_minimum_around_3500hz() {
        // W(f) has a shape that peaks around 3.5 kHz (ear canal resonance)
        // and rolls off at both ends — BUT as an attenuation model it actually
        // shows that the ear is MOST sensitive (highest weight) around 3.5 kHz.
        // We verify the weight at 3500 Hz is GREATER than at 200 Hz and 10000 Hz.
        let w_200 = outer_middle_ear_weight_linear(200.0);
        let w_3500 = outer_middle_ear_weight_linear(3500.0);
        let w_10000 = outer_middle_ear_weight_linear(10_000.0);
        assert!(
            w_3500 > w_200,
            "Weight at 3500 Hz ({w_3500:.4}) should be > weight at 200 Hz ({w_200:.4})"
        );
        assert!(
            w_3500 > w_10000,
            "Weight at 3500 Hz ({w_3500:.4}) should be > weight at 10000 Hz ({w_10000:.4})"
        );
    }

    #[test]
    fn test_bark_band_count() {
        let em = default_ear_model();
        assert_eq!(
            em.band_edges_hz.len(),
            110,
            "Expected 110 band edges (109+1)"
        );
        assert_eq!(em.band_centers_hz.len(), 109, "Expected 109 band centers");
        // Edges must be monotonically increasing
        for w in em.band_edges_hz.windows(2) {
            assert!(
                w[1] > w[0],
                "Band edges must be strictly increasing: {:.2} >= {:.2}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn test_internal_noise_per_band_positive() {
        let em = default_ear_model();
        for (b, &n) in em.internal_noise.iter().enumerate() {
            assert!(
                n > 0.0,
                "Internal noise at band {b} must be positive, got {n}"
            );
        }
    }

    #[test]
    fn test_frequency_spreading_non_pathological() {
        let em = default_ear_model();
        // Create a unit-impulse energy vector (1.0 in band 50, 0 elsewhere)
        let mut energy = vec![0.0_f32; 109];
        energy[50] = 1.0;

        let spread = em.frequency_spread(&energy);

        // Total spread must be positive
        let total: f32 = spread.iter().sum();
        assert!(total > 0.0, "Spread energy must be positive, got {total}");

        // Source band must have non-zero spread
        assert!(
            spread[50] > 0.0,
            "Source band must have non-zero spread, got {}",
            spread[50]
        );
    }

    #[test]
    fn test_time_spreading_decay_constant() {
        let mut em = default_ear_model();
        let fft_plan = make_fft_plan(2048).unwrap();

        // Process a frame with non-zero signal to build up time state
        let frame_nonzero: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48_000.0).sin())
            .collect();
        em.process_frame(&frame_nonzero, &fft_plan).unwrap();
        let state_after_signal: Vec<f32> = em.time_state.clone();

        // Process a zero frame — state should decay
        let frame_zero = vec![0.0_f32; 2048];
        em.process_frame(&frame_zero, &fft_plan).unwrap();
        let state_after_zero: Vec<f32> = em.time_state.clone();

        // Find at least one band where energy decayed
        let decayed = state_after_signal
            .iter()
            .zip(state_after_zero.iter())
            .any(|(&before, &after)| after < before);
        assert!(decayed, "Time state should decay after a silent frame");
    }

    #[test]
    fn test_silent_input_small_raw_energy() {
        let mut em = default_ear_model();
        let fft_plan = make_fft_plan(2048).unwrap();

        let frame_zero = vec![0.0_f32; 2048];
        let excitation = em.process_frame(&frame_zero, &fft_plan).unwrap();

        // raw_energy comes from FFT of zeros (should be exactly zero or near-zero)
        for (b, &e) in excitation.raw_energy.iter().enumerate() {
            assert!(
                e < 1e-10,
                "raw_energy at band {b} should be near-zero for silent input, got {e}"
            );
        }
    }

    #[test]
    fn test_process_frame_wrong_length_errors() {
        let mut em = default_ear_model();
        let fft_plan = make_fft_plan(2048).unwrap();

        let short_frame = vec![0.0_f32; 1024];
        let result = em.process_frame(&short_frame, &fft_plan);
        assert!(result.is_err(), "Expected error for wrong frame length");
    }

    #[test]
    fn test_process_signal_short_signal_errors() {
        let mut em = default_ear_model();
        let fft_plan = make_fft_plan(2048).unwrap();

        let short_signal = Array1::zeros(512);
        let result = em.process_signal(&short_signal, &fft_plan);
        assert!(
            result.is_err(),
            "Expected error for signal shorter than frame_size"
        );
    }

    #[test]
    fn test_reset_clears_time_state() {
        let mut em = default_ear_model();
        let fft_plan = make_fft_plan(2048).unwrap();

        // Build up state
        let frame: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48_000.0).sin())
            .collect();
        em.process_frame(&frame, &fft_plan).unwrap();

        // Reset and verify
        em.reset();
        for (b, &s) in em.time_state.iter().enumerate() {
            assert_eq!(s, 0.0, "Time state at band {b} must be zero after reset");
        }
    }

    #[test]
    fn test_ear_model_non_power_of_two_errors() {
        let result = EarModel::new(48_000.0, 1000, 500, 109);
        assert!(
            result.is_err(),
            "Expected error for non-power-of-two frame_size"
        );
    }
}
