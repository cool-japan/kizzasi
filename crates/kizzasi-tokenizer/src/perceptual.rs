//! Perceptual / Bark-scale psychoacoustic quantization.
//!
//! This module implements a classic perceptual audio quantizer:
//!
//! * Frequency-to-Bark conversion via the Traunmüller formula.
//! * Terhardt absolute threshold of hearing.
//! * Bark-scale critical-band analysis (Zwicker 1980, 24 bands).
//! * Bit allocation proportional to the exceedance over the absolute threshold.
//! * Mid-tread uniform magnitude quantization per band.
//! * Hann-windowed STFT/ISTFT via OxiFFT (pure Rust, no C/Fortran).
//!
//! The main entry point is [`PerceptualQuantizer`].

use crate::error::{TokenizerError, TokenizerResult};
use crate::SignalTokenizer;
use oxifft::{Complex, Direction, Flags, Plan};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

type Complex32 = Complex<f32>;
type Plan32 = Plan<f32>;

// ─────────────────────────────────────────────────────────────────────────────
// Public psychoacoustic helper functions
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a frequency in Hz to the Bark scale using Traunmüller's formula.
///
/// # Formula
/// `bark = 26.81 * hz / (1960.0 + hz) − 0.53`
///
/// Reference: Traunmüller (1990), "Analytical expressions for the tonotopic
/// sensory scale". *JASA* 88(1).
#[inline]
pub fn frequency_to_bark(hz: f32) -> f32 {
    26.81 * hz / (1960.0 + hz) - 0.53
}

/// Inverse of [`frequency_to_bark`]: convert Bark back to Hz.
///
/// Derived from `bark = 26.81 * hz / (1960 + hz) - 0.53`  →
/// `hz = 1960 * (bark + 0.53) / (26.81 - bark - 0.53)`.
#[inline]
fn bark_to_frequency(bark: f32) -> f32 {
    let b = bark + 0.53;
    1960.0 * b / (26.81 - b)
}

/// Terhardt (1979) absolute threshold of hearing in dB SPL.
///
/// # Formula
/// `T(f) = 3.64 * (f/kHz)^{-0.8}
///         − 6.5 * exp(−0.6 * (f/kHz − 3.3)^2)
///         + 10^{−3} * (f/kHz)^4`
///
/// Represents the minimum audible level for a pure tone in a quiet field.
#[inline]
pub fn absolute_threshold_db(hz: f32) -> f32 {
    let f_khz = hz / 1000.0;
    3.64 * f_khz.powf(-0.8) - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp() + 1e-3 * f_khz.powi(4)
}

// ─────────────────────────────────────────────────────────────────────────────
// BarkBands — Zwicker 1980 critical-band layout
// ─────────────────────────────────────────────────────────────────────────────

/// Description of a Bark-scale critical-band decomposition.
///
/// Band edges are spaced uniformly in Bark and then converted back to Hz.
/// Centers are the midpoints of adjacent edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarkBands {
    /// Number of Bark bands.
    pub num_bands: usize,
    /// Band edge frequencies in Hz; length = `num_bands + 1`.
    pub edges_hz: Vec<f32>,
    /// Band center frequencies in Hz; length = `num_bands`.
    pub centers_hz: Vec<f32>,
}

impl BarkBands {
    /// Construct a Bark-scale band layout.
    ///
    /// # Arguments
    /// * `sample_rate` – Signal sample rate in Hz (determines Nyquist).
    /// * `num_bins`    – FFT positive-frequency bin count (`frame_size / 2 + 1`).
    /// * `num_bands`   – Desired number of Bark bands.
    pub fn new(sample_rate: f32, _num_bins: usize, num_bands: usize) -> Self {
        let nyquist = sample_rate / 2.0;
        let max_bark = frequency_to_bark(nyquist);

        // Evenly spaced edges in Bark: 0, step, 2*step, …, max_bark
        let edges_hz: Vec<f32> = (0..=num_bands)
            .map(|i| {
                let bark = max_bark * i as f32 / num_bands as f32;
                bark_to_frequency(bark).max(0.0)
            })
            .collect();

        // Centers are midpoints of adjacent edges
        let centers_hz: Vec<f32> = (0..num_bands)
            .map(|i| (edges_hz[i] + edges_hz[i + 1]) / 2.0)
            .collect();

        Self {
            num_bands,
            edges_hz,
            centers_hz,
        }
    }

    /// Standard 24-band Bark decomposition (Zwicker 1980).
    ///
    /// Calls [`BarkBands::new`] with `num_bands = 24`.
    pub fn standard_24(sample_rate: f32, num_bins: usize) -> Self {
        Self::new(sample_rate, num_bins, 24)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PerceptualQuantizer
// ─────────────────────────────────────────────────────────────────────────────

/// Psychoacoustic-model-driven signal quantizer.
///
/// Allocates bits per Bark band proportionally to how much each band's energy
/// exceeds the absolute threshold of hearing. Bands below threshold receive
/// zero bits. The encode/decode path uses Hann-windowed STFT (via OxiFFT) and
/// overlap-add synthesis.
///
/// # Reconstruction is magnitude-only, zero-phase
///
/// Phase is not transmitted: every reconstructed bin is placed on the real
/// axis (`Complex::new(mag, 0.0)`), so `decode(encode(x))` is **not**
/// waveform-accurate even when quantization is lossless — only per-band
/// energy is preserved. This is a deliberate scope choice (transmitting
/// quantized phase would need its own bit budget, separate from
/// `total_bits_per_frame`, which currently allocates bits for magnitude
/// only); callers that need phase-accurate reconstruction should use
/// [`FourierTokenizer`](crate::FourierTokenizer) or
/// [`DCTTokenizer`](crate::DCTTokenizer) instead.
///
/// # Example
/// ```ignore
/// use kizzasi_tokenizer::PerceptualQuantizer;
/// let q = PerceptualQuantizer::new(16000.0, 1024, 256).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualQuantizer {
    /// Signal sample rate in Hz.
    pub sample_rate: f32,
    /// Analysis frame size in samples; must be a power of two.
    pub frame_size: usize,
    /// Hop between consecutive frames (default = `frame_size / 2`).
    pub hop_size: usize,
    /// Bark-band descriptor for the chosen sample rate.
    pub bark_bands: BarkBands,
    /// Per-band absolute threshold of hearing in dB (clamped to `min_db`).
    pub threshold_db: Vec<f32>,
    /// Total bits to distribute across bands per frame.
    pub total_bits_per_frame: usize,
    /// Minimum energy floor in dB used when computing band energies.
    pub min_db: f32,
}

impl PerceptualQuantizer {
    /// Construct a `PerceptualQuantizer`.
    ///
    /// # Arguments
    /// * `sample_rate`          – Signal sample rate in Hz.
    /// * `frame_size`           – FFT frame length; **must be a power of two**.
    /// * `total_bits_per_frame` – Total bits shared across all bands per frame.
    ///
    /// # Errors
    /// Returns `Err` if `frame_size` is not a power of two.
    pub fn new(
        sample_rate: f32,
        frame_size: usize,
        total_bits_per_frame: usize,
    ) -> TokenizerResult<Self> {
        if !frame_size.is_power_of_two() {
            return Err(TokenizerError::InvalidConfig(format!(
                "frame_size must be a power of two, got {frame_size}"
            )));
        }
        // `frame_size == 1` is a power of two but makes `hop_size == 0`,
        // which `encode` divides by (`div_ceil(self.hop_size)`), and makes
        // the Hann window's `(frame_size - 1)` divisor zero (NaN window).
        // `frame_size == 2` is the smallest value that keeps both safe.
        if frame_size < 2 {
            return Err(TokenizerError::InvalidConfig(format!(
                "frame_size must be >= 2, got {frame_size}"
            )));
        }

        let hop_size = frame_size / 2;
        let num_bins = frame_size / 2 + 1;
        let bark_bands = BarkBands::standard_24(sample_rate, num_bins);
        let min_db = -80.0_f32;

        let threshold_db: Vec<f32> = bark_bands
            .centers_hz
            .iter()
            .map(|&hz| absolute_threshold_db(hz).max(min_db))
            .collect();

        Ok(Self {
            sample_rate,
            frame_size,
            hop_size,
            bark_bands,
            threshold_db,
            total_bits_per_frame,
            min_db,
        })
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Compute per-band energy in dB from a magnitude-spectrum slice.
    ///
    /// Bins are mapped to bands by their centre frequency
    /// `k * sample_rate / frame_size`.  Energy for each band is the sum of
    /// squared magnitudes of bins whose centre falls in `[edge_b, edge_{b+1})`.
    fn compute_band_energies(&self, spectrum: &[Complex32], num_bins: usize) -> Vec<f32> {
        let mut energies = vec![0.0_f32; self.bark_bands.num_bands];
        let bin_hz = self.sample_rate / self.frame_size as f32;

        for (k, bin) in spectrum.iter().enumerate().take(num_bins) {
            let freq = k as f32 * bin_hz;
            // Find the band this bin belongs to
            for (b, energies_b) in energies.iter_mut().enumerate() {
                let lo = self.bark_bands.edges_hz[b];
                let hi = self.bark_bands.edges_hz[b + 1];
                if freq >= lo && freq < hi {
                    let re = bin.re;
                    let im = bin.im;
                    *energies_b += re * re + im * im;
                    break;
                }
            }
        }

        // Convert to dB and clamp
        energies
            .iter()
            .map(|&e| (10.0 * (e + 1e-12_f32).log10()).max(self.min_db))
            .collect()
    }

    /// Allocate bits across bands proportional to above-threshold exceedance.
    ///
    /// Bands that do not exceed `self.threshold_db[b]` receive zero bits.
    /// The total is guaranteed to equal `self.total_bits_per_frame` (adjusted
    /// on the highest-excess band if rounding drifts).
    fn compute_bit_allocation(&self, band_energies_db: &[f32]) -> Vec<usize> {
        let num_bands = self.bark_bands.num_bands;
        let mut excess: Vec<f32> = band_energies_db
            .iter()
            .zip(self.threshold_db.iter())
            .map(|(&e, &t)| (e - t).max(0.0))
            .collect();

        let total_excess: f32 = excess.iter().sum();

        if total_excess < 1e-9 {
            // All bands below threshold: distribute bits evenly across all bands
            let base = self.total_bits_per_frame / num_bands;
            let mut bits = vec![base; num_bands];
            // Remainder goes to first band
            bits[0] += self.total_bits_per_frame - base * num_bands;
            return bits;
        }

        // Proportional allocation
        let total = self.total_bits_per_frame as f32;
        let mut bits: Vec<usize> = excess
            .iter()
            .map(|&ex| (ex / total_excess * total).round() as usize)
            .collect();

        // Zero out bands with zero excess
        for (b, ex) in excess.iter().enumerate() {
            if *ex == 0.0 {
                bits[b] = 0;
            }
        }

        // Fix-up: adjust the band with largest excess so sum == total_bits_per_frame
        let current_sum: usize = bits.iter().sum();
        let target = self.total_bits_per_frame;

        if current_sum != target {
            // Find band with max excess (guaranteed > 0 since total_excess >= 1e-9)
            let max_band = excess
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            if current_sum < target {
                bits[max_band] = bits[max_band].saturating_add(target - current_sum);
            } else {
                // Subtract from max_band, ensuring we don't underflow
                let over = current_sum - target;
                bits[max_band] = bits[max_band].saturating_sub(over);
            }
        }

        // Zero out excess so we can reuse it (avoid dead_code lint)
        for ex in excess.iter_mut() {
            *ex = 0.0;
        }
        let _ = excess; // consumed

        bits
    }

    /// Return the bin indices that fall within band `b`.
    ///
    /// A bin k belongs to band b iff its centre frequency
    /// `k * sample_rate / frame_size` lies in `[edges_hz[b], edges_hz[b+1])`.
    fn bins_for_band(&self, band: usize, num_bins: usize) -> Vec<usize> {
        let bin_hz = self.sample_rate / self.frame_size as f32;
        let lo = self.bark_bands.edges_hz[band];
        let hi = self.bark_bands.edges_hz[band + 1];
        (0..num_bins)
            .filter(|&k| {
                let f = k as f32 * bin_hz;
                f >= lo && f < hi
            })
            .collect()
    }

    /// Encode one frame: returns a flat list of u32 tokens.
    ///
    /// Token layout per frame:
    /// ```text
    /// [bits[0], bits[1], …, bits[num_bands-1],
    ///  <band_0: scale_bits, quant_indices...> | <0 placeholder if band_0 carries no bits>,
    ///  <band_1: scale_bits, quant_indices...> | <0 placeholder if band_1 carries no bits>,
    ///  …]
    /// ```
    /// `scale_bits` is the band's per-frame magnitude scale (`max_mag +
    /// EPSILON`), bit-cast from `f32` via `f32::to_bits`, so `decode_frame`
    /// can recover real amplitudes instead of values confined to `[0, 1]`.
    fn encode_frame(
        &self,
        frame: &[f32],
        fft_plan: &Plan32,
        num_bins: usize,
    ) -> TokenizerResult<Vec<u32>> {
        // Apply Hann window and build complex input
        let mut input: Vec<Complex32> = frame
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / (self.frame_size - 1) as f32).cos());
                Complex32::new(s * w, 0.0)
            })
            .collect();

        let mut output: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); self.frame_size];
        fft_plan.execute(&input, &mut output);

        // Reuse `input` storage for the next iteration (silence unused-mut warning)
        let _ = &mut input;

        // Compute band energies → bit allocation
        let band_energies = self.compute_band_energies(&output[..num_bins], num_bins);
        let bit_alloc = self.compute_bit_allocation(&band_energies);

        // Header: bit allocation per band
        let mut tokens: Vec<u32> = bit_alloc.iter().map(|&b| b as u32).collect();

        // Quantize each band's magnitudes
        for (b, &bits) in bit_alloc.iter().enumerate() {
            let bin_indices = self.bins_for_band(b, num_bins);
            if bin_indices.is_empty() || bits == 0 {
                // Emit a single zero placeholder so the decoder can recover shape
                tokens.push(0);
                continue;
            }

            let mags: Vec<f32> = bin_indices
                .iter()
                .map(|&k| {
                    let re = output[k].re;
                    let im = output[k].im;
                    (re * re + im * im).sqrt()
                })
                .collect();

            let max_mag = mags.iter().cloned().fold(0.0_f32, f32::max);
            let scale = max_mag + f32::EPSILON;
            let levels = 1usize << bits.min(20); // guard overflow

            // Transmit the per-band scale so `decode_frame` can undo the
            // `mag / scale` normalisation below and recover a real
            // amplitude instead of a value confined to [0, 1].
            tokens.push(scale.to_bits());

            for mag in &mags {
                // Mid-tread uniform quantization in [0, scale]
                let normalised = (mag / scale).clamp(0.0, 1.0);
                // Mid-tread: level centres at (i + 0.5) / levels
                let idx = (normalised * levels as f32).floor() as usize;
                tokens.push(idx.min(levels - 1) as u32);
            }
        }

        Ok(tokens)
    }

    /// Decode one frame from its token stream segment.
    ///
    /// Returns (time-domain samples, tokens_consumed).
    fn decode_frame(
        &self,
        tokens: &[u32],
        ifft_plan: &Plan32,
        num_bins: usize,
    ) -> TokenizerResult<(Vec<f32>, usize)> {
        let num_bands = self.bark_bands.num_bands;

        if tokens.len() < num_bands {
            return Err(TokenizerError::decoding(
                "perceptual",
                "token stream too short for bit-allocation header",
            ));
        }

        let bit_alloc: Vec<usize> = tokens[..num_bands].iter().map(|&b| b as usize).collect();

        let mut cursor = num_bands;
        let mut freq_domain: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); self.frame_size];

        for (b, &bits) in bit_alloc.iter().enumerate() {
            let bin_indices = self.bins_for_band(b, num_bins);
            if bin_indices.is_empty() || bits == 0 {
                // Skip the placeholder
                if cursor >= tokens.len() {
                    return Err(TokenizerError::decoding(
                        "perceptual",
                        "token stream ended prematurely (placeholder)",
                    ));
                }
                cursor += 1;
                continue;
            }

            let levels = 1usize << bits.min(20);
            let num_bins_in_band = bin_indices.len();

            // Read the per-band magnitude scale written by `encode_frame`
            // before the quantized indices.
            if cursor >= tokens.len() {
                return Err(TokenizerError::decoding(
                    "perceptual",
                    "token stream ended prematurely (band scale)",
                ));
            }
            let scale = f32::from_bits(tokens[cursor]);
            cursor += 1;

            if cursor + num_bins_in_band > tokens.len() {
                return Err(TokenizerError::decoding(
                    "perceptual",
                    "token stream ended prematurely (band bins)",
                ));
            }

            // Recover magnitudes (zero-phase reconstruction)
            for (j, &k) in bin_indices.iter().enumerate() {
                let idx = tokens[cursor + j] as usize;
                // Dequantize: centre of the mid-tread cell, rescaled back
                // into the band's original magnitude range.
                let mag = ((idx as f32 + 0.5) / levels as f32) * scale;
                // Zero phase → purely real
                freq_domain[k] = Complex32::new(mag, 0.0);
                // Mirror for IFFT (Hermitian symmetry, real signal)
                if k > 0 && k < self.frame_size - k {
                    freq_domain[self.frame_size - k] = Complex32::new(mag, 0.0);
                }
            }

            cursor += num_bins_in_band;
        }

        // IFFT
        let mut time_domain: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); self.frame_size];
        ifft_plan.execute(&freq_domain, &mut time_domain);

        let norm = self.frame_size as f32;
        let samples: Vec<f32> = time_domain.iter().map(|c| c.re / norm).collect();

        Ok((samples, cursor))
    }

    // ── Public encode / decode ───────────────────────────────────────────────

    /// Encode a continuous signal into a packed `Vec<u32>` token stream.
    ///
    /// The signal is zero-padded so that every frame of length `frame_size`
    /// starting at offsets `0, hop_size, 2*hop_size, …` is fully within bounds.
    pub fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Vec<u32>> {
        let n = signal.len();
        let num_bins = self.frame_size / 2 + 1;

        // Compute how many frames we need and the padded length
        // We need: last_frame_start + frame_size <= padded_len
        // where last_frame_start = k * hop_size for the largest k s.t. k*hop_size < n
        let num_frames = if n <= self.frame_size {
            1
        } else {
            // number of hops that fit: ceil((n - frame_size) / hop_size) + 1
            n.saturating_sub(self.frame_size).div_ceil(self.hop_size) + 1
        };
        let padded_len = (num_frames - 1) * self.hop_size + self.frame_size;

        let mut padded: Vec<f32> = vec![0.0; padded_len];
        for (i, &v) in signal.iter().enumerate() {
            padded[i] = v;
        }

        let fft_plan = Plan::<f32>::dft_1d(self.frame_size, Direction::Forward, Flags::MEASURE)
            .ok_or_else(|| TokenizerError::encoding("perceptual", "FFT planning failed"))?;

        let mut tokens: Vec<u32> = Vec::new();

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let frame = &padded[start..start + self.frame_size];
            let frame_tokens = self.encode_frame(frame, &fft_plan, num_bins)?;
            tokens.extend_from_slice(&frame_tokens);
        }

        Ok(tokens)
    }

    /// Decode a packed `Vec<u32>` token stream back to a continuous signal.
    ///
    /// Uses overlap-add synthesis with a Hann window.
    pub fn decode(&self, tokens: &[u32]) -> TokenizerResult<Array1<f32>> {
        let num_bins = self.frame_size / 2 + 1;

        let ifft_plan =
            Plan::<f32>::dft_1d(self.frame_size, Direction::Backward, Flags::MEASURE)
                .ok_or_else(|| TokenizerError::decoding("perceptual", "IFFT planning failed"))?;

        // Build Hann window for overlap-add normalisation
        let hann: Vec<f32> = (0..self.frame_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (self.frame_size - 1) as f32).cos()))
            .collect();

        // We don't know the final length in advance — accumulate in a Vec
        // and grow as needed.
        let mut output: Vec<f32> = Vec::new();
        let mut window_sum: Vec<f32> = Vec::new();

        let mut cursor = 0;
        let mut frame_idx = 0;

        while cursor < tokens.len() {
            let (samples, consumed) = self.decode_frame(&tokens[cursor..], &ifft_plan, num_bins)?;
            cursor += consumed;

            let start = frame_idx * self.hop_size;
            let end = start + self.frame_size;

            // Grow accumulation buffers if needed
            if end > output.len() {
                output.resize(end, 0.0);
                window_sum.resize(end, 0.0);
            }

            // Overlap-add with Hann window
            for (i, &s) in samples.iter().enumerate() {
                output[start + i] += s * hann[i];
                window_sum[start + i] += hann[i] * hann[i];
            }

            frame_idx += 1;
        }

        // Normalise by the window power sum (WOLA)
        for (o, &ws) in output.iter_mut().zip(window_sum.iter()) {
            if ws > 1e-8 {
                *o /= ws;
            }
        }

        // Trim: round to the nearest multiple of hop_size
        let len = output.len();
        let trimmed_len = if len > self.frame_size {
            // Trim at the last complete OLA position
            ((len.saturating_sub(self.frame_size)) / self.hop_size) * self.hop_size
                + self.frame_size
        } else {
            len
        };
        output.truncate(trimmed_len);

        Ok(Array1::from(output))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SignalTokenizer implementation
// ─────────────────────────────────────────────────────────────────────────────

impl SignalTokenizer for PerceptualQuantizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let tokens = PerceptualQuantizer::encode(self, signal)?;
        let floats: Vec<f32> = tokens.iter().map(|&x| x as f32).collect();
        Ok(Array1::from(floats))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let token_u32: Vec<u32> = tokens.iter().map(|&x| x.round().max(0.0) as u32).collect();
        PerceptualQuantizer::decode(self, &token_u32)
    }

    /// Embedding dimension (num_bands × 2 real/imag placeholder).
    fn embed_dim(&self) -> usize {
        self.bark_bands.num_bands * 2
    }

    /// Vocabulary size: `2^min(total_bits_per_frame, 20)`.
    fn vocab_size(&self) -> usize {
        1usize << self.total_bits_per_frame.min(20)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    // Helper: generate white noise of length `n`.
    fn white_noise(n: usize) -> Array1<f32> {
        // Deterministic Lehmer LCG — no external RNG dependency required.
        let mut state: u64 = 12345;
        let samples: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Map to [-1, 1]
                let u = (state >> 33) as f32 / (u32::MAX as f32);
                u * 2.0 - 1.0
            })
            .collect();
        Array1::from(samples)
    }

    #[test]
    fn test_frequency_to_bark_known_values() {
        // Traunmüller (1990) formula: bark = 26.81 * hz / (1960 + hz) - 0.53
        // This formula produces values that compress the high-frequency range more than
        // some other Bark scales. Expected values are computed directly from the formula.
        //
        // Verify exact formula output (not from external reference tables):
        let b100 = frequency_to_bark(100.0); // 26.81*100/2060 - 0.53 ≈ 0.770
        let b1k = frequency_to_bark(1000.0); // 26.81*1000/2960 - 0.53 ≈ 8.527
        let b4k = frequency_to_bark(4000.0); // 26.81*4000/5960 - 0.53 ≈ 17.46
        let b16k = frequency_to_bark(16000.0); // 26.81*16000/17960 - 0.53 ≈ 23.36

        // Formula properties: values must be strictly increasing (monotone)
        assert!(b100 < b1k, "Bark(100)={b100} should be < Bark(1000)={b1k}");
        assert!(b1k < b4k, "Bark(1000)={b1k} should be < Bark(4000)={b4k}");
        assert!(
            b4k < b16k,
            "Bark(4000)={b4k} should be < Bark(16000)={b16k}"
        );

        // Formula boundary: at 0 Hz → -0.53 (formula gives negative, as expected)
        assert!(
            frequency_to_bark(0.0) < 0.0,
            "bark(0) should be slightly negative"
        );

        // Coarse ordering: 100 Hz < 2 Bark, 1000 Hz in [5, 12], 4000 Hz > 15, 16kHz > 20
        assert!(b100 < 2.0, "100 Hz should be < 2 Bark, got {b100}");
        assert!(
            b1k > 5.0 && b1k < 12.0,
            "1000 Hz should be in [5,12] Bark, got {b1k}"
        );
        assert!(b4k > 15.0, "4000 Hz should be > 15 Bark, got {b4k}");
        assert!(b16k > 20.0, "16000 Hz should be > 20 Bark, got {b16k}");

        // Spot-check 1000 Hz close to published table value (within ±1 Bark)
        assert!(
            (b1k - 8.51).abs() < 1.0,
            "1000 Hz: got {b1k}, expected ~8.51 Bark"
        );
    }

    #[test]
    fn test_absolute_threshold_decreasing_to_3500hz() {
        // Terhardt minimum region: threshold should be non-increasing up to ~3.5 kHz
        let freqs = [100.0_f32, 500.0, 1000.0, 2000.0, 3500.0];
        let thresholds: Vec<f32> = freqs.iter().map(|&f| absolute_threshold_db(f)).collect();
        for i in 0..thresholds.len() - 1 {
            assert!(
                thresholds[i] >= thresholds[i + 1],
                "Threshold should be non-increasing up to 3.5 kHz: \
                 T({}) = {:.2} > T({}) = {:.2}",
                freqs[i],
                thresholds[i],
                freqs[i + 1],
                thresholds[i + 1]
            );
        }
    }

    #[test]
    fn test_bark_bands_energy_conserved() {
        let sample_rate = 16000.0_f32;
        let frame_size = 512_usize;
        let num_bins = frame_size / 2 + 1;
        let pq = PerceptualQuantizer::new(sample_rate, frame_size, 64).unwrap();

        // Flat spectrum: all bins have magnitude 1.0
        let flat: Vec<Complex32> = (0..num_bins).map(|_| Complex32::new(1.0, 0.0)).collect();

        let energies = pq.compute_band_energies(&flat, num_bins);

        // Sum of linear energies (convert dB back to linear)
        let energy_sum: f32 = energies
            .iter()
            .map(|&e_db| 10.0_f32.powf(e_db / 10.0))
            .sum();
        let expected = num_bins as f32; // each bin contributes |1|^2 = 1
        let ratio = energy_sum / expected;

        // Some bins may not land in any band if edges don't cover all bins,
        // so allow up to 90% of total energy to be captured.
        assert!(
            ratio > 0.90,
            "Energy conservation: captured {:.1}% of total (got {:.2}, expected {:.2})",
            ratio * 100.0,
            energy_sum,
            expected
        );
    }

    #[test]
    fn test_encode_roundtrip_length() {
        let sample_rate = 16000.0_f32;
        let signal = white_noise(sample_rate as usize); // 1 second
        let pq = PerceptualQuantizer::new(sample_rate, 1024, 256).unwrap();

        let tokens = pq.encode(&signal).unwrap();
        let reconstructed = pq.decode(&tokens).unwrap();

        let expected = sample_rate as usize; // 16 000
        let tolerance = 1024_usize;
        let diff = (reconstructed.len() as isize - expected as isize).unsigned_abs();
        assert!(
            diff <= tolerance,
            "Roundtrip length: got {}, expected ~{} (diff {}, tol {})",
            reconstructed.len(),
            expected,
            diff,
            tolerance
        );
    }

    /// Regression: decoded amplitude must track the input amplitude.
    ///
    /// Before the fix, `decode_frame` reconstructed every band's magnitudes
    /// as `(idx + 0.5) / levels` — a value confined to `[0, 1]` with no
    /// reference to the band's actual magnitude scale — so a loud signal
    /// and a quiet signal decoded to comparable energy. `encode_frame` now
    /// transmits each band's magnitude scale (`max_mag + EPSILON`, bit-cast
    /// to `u32`) so `decode_frame` can undo the normalisation.
    #[test]
    fn test_decode_amplitude_tracks_input_amplitude() {
        let pq = PerceptualQuantizer::new(8000.0, 256, 1024).unwrap();
        let loud: Array1<f32> =
            Array1::from_vec((0..1024).map(|i| 0.9 * (i as f32 * 0.05).sin()).collect());
        let quiet: Array1<f32> =
            Array1::from_vec((0..1024).map(|i| 0.02 * (i as f32 * 0.05).sin()).collect());

        let loud_tokens = pq.encode(&loud).unwrap();
        let quiet_tokens = pq.encode(&quiet).unwrap();
        let loud_decoded = pq.decode(&loud_tokens).unwrap();
        let quiet_decoded = pq.decode(&quiet_tokens).unwrap();

        let loud_energy: f32 = loud_decoded.iter().map(|x| x * x).sum();
        let quiet_energy: f32 = quiet_decoded.iter().map(|x| x * x).sum();

        assert!(
            loud_energy > quiet_energy * 5.0,
            "decoded energy should scale with input amplitude: loud={loud_energy}, quiet={quiet_energy}"
        );
    }

    /// Regression: with the per-band scale now present in the token stream,
    /// a single band's decoded magnitude should be within one quantization
    /// step of the value that was actually encoded (not an arbitrary value
    /// in `[0, 1]` unrelated to it).
    #[test]
    fn test_encode_frame_scale_roundtrips_through_decode_frame() {
        let pq = PerceptualQuantizer::new(8000.0, 64, 512).unwrap();
        let num_bins = pq.frame_size / 2 + 1;

        // A single sinusoid concentrates energy in a narrow band, giving that
        // band a large, easily-checked magnitude scale.
        let frame: Vec<f32> = (0..pq.frame_size)
            .map(|i| 5.0 * (i as f32 * 0.3).sin())
            .collect();

        let fft_plan = Plan::<f32>::dft_1d(pq.frame_size, Direction::Forward, Flags::MEASURE)
            .expect("FFT plan");
        let ifft_plan = Plan::<f32>::dft_1d(pq.frame_size, Direction::Backward, Flags::MEASURE)
            .expect("IFFT plan");

        let tokens = pq.encode_frame(&frame, &fft_plan, num_bins).unwrap();
        let (decoded, consumed) = pq.decode_frame(&tokens, &ifft_plan, num_bins).unwrap();

        assert_eq!(consumed, tokens.len());
        assert_eq!(decoded.len(), pq.frame_size);

        // The reconstructed frame's peak magnitude should be in the same
        // order of magnitude as the input's (loosely bounded, since this is
        // a lossy psychoacoustic codec) rather than confined to ~[0, 1]
        // regardless of the input scale.
        let input_peak = frame.iter().cloned().fold(0.0_f32, |a, b| a.max(b.abs()));
        let decoded_peak = decoded.iter().cloned().fold(0.0_f32, |a, b| a.max(b.abs()));
        assert!(
            decoded_peak > input_peak * 0.1,
            "decoded peak {decoded_peak} should track input peak {input_peak}, not collapse towards O(1)"
        );
    }

    #[test]
    fn test_bit_allocation_floor() {
        let sample_rate = 16000.0_f32;
        let mut pq = PerceptualQuantizer::new(sample_rate, 1024, 256).unwrap();

        // Push threshold very high for all bands except band 0
        for b in 1..pq.bark_bands.num_bands {
            pq.threshold_db[b] = 100.0;
        }
        pq.threshold_db[0] = -80.0; // well below any signal

        // Energy: band 0 has high energy, rest have low energy
        let mut band_energies = vec![-70.0_f32; pq.bark_bands.num_bands];
        band_energies[0] = 40.0; // well above threshold[0] = -80 dB

        let bits = pq.compute_bit_allocation(&band_energies);

        // Band 0 should get near-all the bits
        assert!(bits[0] > 0, "Band 0 should receive bits; got {:?}", bits);
        // All silenced bands should get 0
        for (b, &alloc) in bits.iter().enumerate().skip(1) {
            assert_eq!(alloc, 0, "Band {b} should have 0 bits; got {alloc}");
        }
    }

    #[test]
    fn test_signal_tokenizer_trait() {
        let pq = PerceptualQuantizer::new(16000.0, 1024, 128).unwrap();
        let signal = white_noise(2048);
        let result = SignalTokenizer::encode(&pq, &signal);
        assert!(
            result.is_ok(),
            "SignalTokenizer::encode failed: {:?}",
            result
        );
        let tokens = result.unwrap();
        assert!(!tokens.is_empty(), "Token array must be non-empty");
    }

    #[test]
    fn test_frame_size_must_be_power_of_two() {
        let result = PerceptualQuantizer::new(16000.0, 1000, 128);
        assert!(
            result.is_err(),
            "Expected Err for non-power-of-two frame_size, got Ok"
        );
    }

    /// Regression: `frame_size == 1` is a power of two but makes
    /// `hop_size == 0`, which used to panic with "attempt to divide by
    /// zero" inside `encode` (and produce a NaN Hann window even before
    /// that). `new` must reject it instead.
    #[test]
    fn test_frame_size_one_is_rejected() {
        assert!(
            PerceptualQuantizer::new(16000.0, 1, 128).is_err(),
            "frame_size = 1 must be rejected, not accepted and later panic in encode()"
        );
    }

    #[test]
    fn test_frame_size_two_is_the_smallest_accepted() {
        assert!(PerceptualQuantizer::new(16000.0, 2, 128).is_ok());
    }
}
