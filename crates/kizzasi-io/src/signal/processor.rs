//! Signal processing and analysis
//!
//! This module provides the main signal processor for audio and signal analysis,
//! including FFT operations, filtering, and feature extraction.

use super::filters::Filter;
use super::functions::bessel_i0;
use super::spectral::{Spectrogram, WindowType};
use crate::error::{IoError, IoResult};
use oxifft::{Complex, Direction, Flags, Plan};
use scirs2_core::ndarray::Array1;
use std::f32::consts::PI;

/// Signal processor for pre-processing sensor data
pub struct SignalProcessor {
    pub(super) buffer_size: usize,
    pub(super) sample_rate: f32,
}

impl SignalProcessor {
    /// Create a new signal processor
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer_size,
            sample_rate: 44100.0,
        }
    }

    /// Set the sample rate
    pub fn with_sample_rate(mut self, rate: f32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Get the buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Apply FFT to signal
    pub fn fft(&mut self, signal: &Array1<f32>) -> IoResult<Vec<Complex<f32>>> {
        let n = signal.len();
        let plan = Plan::dft_1d(n, Direction::Forward, Flags::MEASURE)
            .ok_or_else(|| IoError::SignalError("FFT planning failed: {}".to_string()))?;
        let input: Vec<Complex<f32>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&input, &mut output);
        Ok(output)
    }

    /// Apply inverse FFT
    pub fn ifft(&mut self, spectrum: &mut [Complex<f32>]) -> IoResult<Array1<f32>> {
        let n = spectrum.len();
        let plan = Plan::dft_1d(n, Direction::Backward, Flags::MEASURE)
            .ok_or_else(|| IoError::SignalError("IFFT planning failed: {}".to_string()))?;
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.execute(spectrum, &mut output);
        let scale = 1.0 / n as f32;
        let result: Vec<f32> = output.iter().map(|c| c.re * scale).collect();
        Ok(Array1::from_vec(result))
    }

    /// Check if a number is a power of 2
    fn is_power_of_2(n: usize) -> bool {
        n != 0 && (n & (n - 1)) == 0
    }

    /// Optimized FFT for power-of-2 sizes
    ///
    /// Uses specialized FFT planning for power-of-2 sizes which can be more efficient.
    /// Falls back to standard FFT for non-power-of-2 sizes.
    pub fn fft_pow2(&mut self, signal: &Array1<f32>) -> IoResult<Vec<Complex<f32>>> {
        // OxiFFT automatically optimizes for power-of-2 sizes
        self.fft(signal)
    }

    /// Optimized inverse FFT for power-of-2 sizes
    pub fn ifft_pow2(&mut self, spectrum: &mut [Complex<f32>]) -> IoResult<Array1<f32>> {
        // OxiFFT automatically optimizes for power-of-2 sizes
        self.ifft(spectrum)
    }

    /// Compute power spectrum efficiently for power-of-2 sizes
    ///
    /// Returns the magnitude squared of the FFT.
    pub fn power_spectrum_pow2(&mut self, signal: &Array1<f32>) -> IoResult<Vec<f32>> {
        let spectrum = self.fft_pow2(signal)?;
        Ok(spectrum.iter().map(|c| c.norm_sqr()).collect())
    }

    /// Zero-pad signal to next power of 2 for optimal FFT performance
    pub fn zero_pad_pow2(signal: &Array1<f32>) -> Array1<f32> {
        let n = signal.len();
        if Self::is_power_of_2(n) {
            return signal.clone();
        }
        let next_pow2 = n.next_power_of_two();
        let mut padded = vec![0.0f32; next_pow2];
        let signal_vec: Vec<f32> = signal.iter().copied().collect();
        padded[..n].copy_from_slice(&signal_vec);
        Array1::from_vec(padded)
    }

    /// Apply a filter to the signal
    pub fn apply_filter(&mut self, signal: &Array1<f32>, filter: Filter) -> IoResult<Array1<f32>> {
        match filter {
            Filter::MovingAverage { window } => self.moving_average(signal, window),
            Filter::LowPass { cutoff, .. } => self.lowpass_fft(signal, cutoff),
            Filter::HighPass { cutoff, .. } => self.highpass_fft(signal, cutoff),
            Filter::BandPass { low, high, .. } => self.bandpass_fft(signal, low, high),
            Filter::Iir(mut iir) => Ok(iir.process(signal)),
            Filter::Fir(mut fir) => Ok(fir.process(signal)),
        }
    }

    /// Compute power spectrum (magnitude squared)
    pub fn power_spectrum(&mut self, signal: &Array1<f32>) -> IoResult<Array1<f32>> {
        let spectrum = self.fft(signal)?;
        let power: Vec<f32> = spectrum.iter().map(|c| c.norm_sqr()).collect();
        Ok(Array1::from_vec(power))
    }

    /// Compute magnitude spectrum
    pub fn magnitude_spectrum(&mut self, signal: &Array1<f32>) -> IoResult<Array1<f32>> {
        let spectrum = self.fft(signal)?;
        let magnitude: Vec<f32> = spectrum.iter().map(|c| c.norm()).collect();
        Ok(Array1::from_vec(magnitude))
    }

    /// Compute phase spectrum
    pub fn phase_spectrum(&mut self, signal: &Array1<f32>) -> IoResult<Array1<f32>> {
        let spectrum = self.fft(signal)?;
        let phase: Vec<f32> = spectrum.iter().map(|c| c.arg()).collect();
        Ok(Array1::from_vec(phase))
    }

    /// Compute zero-crossings count
    pub fn zero_crossings(signal: &Array1<f32>) -> usize {
        signal
            .windows(2)
            .into_iter()
            .filter(|w| w[0].signum() != w[1].signum())
            .count()
    }

    /// Compute peak-to-peak amplitude
    pub fn peak_to_peak(signal: &Array1<f32>) -> f32 {
        let max = signal.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = signal.iter().cloned().fold(f32::INFINITY, f32::min);
        max - min
    }

    /// Apply DC removal (subtract mean)
    pub fn remove_dc(signal: &Array1<f32>) -> Array1<f32> {
        let mean = signal.mean().unwrap_or(0.0);
        signal.mapv(|x| x - mean)
    }

    /// Apply envelope detection via Hilbert transform approximation
    pub fn envelope(&mut self, signal: &Array1<f32>) -> IoResult<Array1<f32>> {
        let n = signal.len();
        let mut spectrum = self.fft(signal)?;
        spectrum[0] = Complex::new(0.0, 0.0);
        for item in spectrum.iter_mut().take(n / 2).skip(1) {
            *item *= Complex::new(2.0, 0.0);
        }
        for item in spectrum.iter_mut().skip(n / 2 + 1) {
            *item = Complex::new(0.0, 0.0);
        }
        let plan = Plan::dft_1d(n, Direction::Backward, Flags::MEASURE)
            .ok_or_else(|| IoError::SignalError("IFFT planning failed: {}".to_string()))?;
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&spectrum, &mut output);
        let scale = 1.0 / n as f32;
        let envelope: Vec<f32> = output.iter().map(|c| c.norm() * scale).collect();
        Ok(Array1::from_vec(envelope))
    }

    /// Simple moving average filter
    fn moving_average(&self, signal: &Array1<f32>, window: usize) -> IoResult<Array1<f32>> {
        if window == 0 || window > signal.len() {
            return Err(IoError::SignalError("Invalid window size".into()));
        }
        let mut result = Array1::zeros(signal.len());
        let mut sum: f32 = signal.iter().take(window).sum();
        for i in 0..signal.len() {
            if i >= window {
                sum -= signal[i - window];
                sum += signal[i];
            }
            result[i] = sum / window.min(i + 1) as f32;
        }
        Ok(result)
    }

    /// Low-pass filter using FFT
    fn lowpass_fft(&mut self, signal: &Array1<f32>, cutoff: f32) -> IoResult<Array1<f32>> {
        let mut spectrum = self.fft(signal)?;
        let n = spectrum.len();
        let cutoff_bin = (cutoff / self.sample_rate * n as f32) as usize;
        for item in spectrum.iter_mut().take(n - cutoff_bin).skip(cutoff_bin) {
            *item = Complex::new(0.0, 0.0);
        }
        self.ifft(&mut spectrum)
    }

    /// High-pass filter using FFT
    fn highpass_fft(&mut self, signal: &Array1<f32>, cutoff: f32) -> IoResult<Array1<f32>> {
        let mut spectrum = self.fft(signal)?;
        let n = spectrum.len();
        let cutoff_bin = (cutoff / self.sample_rate * n as f32) as usize;
        for i in 0..cutoff_bin.min(n / 2) {
            spectrum[i] = Complex::new(0.0, 0.0);
            if n - 1 - i < n {
                spectrum[n - 1 - i] = Complex::new(0.0, 0.0);
            }
        }
        self.ifft(&mut spectrum)
    }

    /// Band-pass filter using FFT
    fn bandpass_fft(&mut self, signal: &Array1<f32>, low: f32, high: f32) -> IoResult<Array1<f32>> {
        let mut spectrum = self.fft(signal)?;
        let n = spectrum.len();
        let low_bin = (low / self.sample_rate * n as f32) as usize;
        let high_bin = (high / self.sample_rate * n as f32) as usize;
        for (i, item) in spectrum.iter_mut().enumerate() {
            let freq_bin = if i <= n / 2 { i } else { n - i };
            if freq_bin < low_bin || freq_bin > high_bin {
                *item = Complex::new(0.0, 0.0);
            }
        }
        self.ifft(&mut spectrum)
    }

    /// Normalize signal to [-1, 1] range
    pub fn normalize(signal: &Array1<f32>) -> Array1<f32> {
        let max_abs = signal.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_abs > 0.0 {
            signal.mapv(|x| x / max_abs)
        } else {
            signal.clone()
        }
    }

    /// Compute RMS (Root Mean Square) of signal
    pub fn rms(signal: &Array1<f32>) -> f32 {
        let sum_sq: f32 = signal.iter().map(|x| x * x).sum();
        (sum_sq / signal.len() as f32).sqrt()
    }

    /// SIMD-optimized RMS computation
    ///
    /// Uses chunked processing for better cache locality and potential SIMD vectorization.
    #[cfg(feature = "simd")]
    pub fn rms_simd(signal: &Array1<f32>) -> f32 {
        let owned: Vec<f32> = signal.iter().copied().collect();
        Self::rms_simd_impl(&owned)
    }

    /// SIMD-optimized RMS implementation for slices
    #[cfg(feature = "simd")]
    fn rms_simd_impl(data: &[f32]) -> f32 {
        const CHUNK_SIZE: usize = 8;
        let chunks = data.chunks_exact(CHUNK_SIZE);
        let remainder = chunks.remainder();
        let mut sum_sq = chunks.fold(0.0f32, |acc, chunk| {
            let chunk_sum: f32 = chunk.iter().map(|x| x * x).sum();
            acc + chunk_sum
        });
        sum_sq += remainder.iter().map(|x| x * x).sum::<f32>();
        (sum_sq / data.len() as f32).sqrt()
    }

    /// SIMD-optimized normalization
    ///
    /// Normalizes signal to [-1, 1] range using chunked processing.
    #[cfg(feature = "simd")]
    pub fn normalize_simd(signal: &Array1<f32>) -> Array1<f32> {
        let owned: Vec<f32> = signal.iter().copied().collect();
        let max_abs = Self::max_abs_simd(&owned);
        if max_abs > 0.0 {
            signal.mapv(|x| x / max_abs)
        } else {
            signal.clone()
        }
    }

    /// SIMD-optimized max absolute value
    #[cfg(feature = "simd")]
    fn max_abs_simd(data: &[f32]) -> f32 {
        const CHUNK_SIZE: usize = 8;
        let chunks = data.chunks_exact(CHUNK_SIZE);
        let remainder = chunks.remainder();
        let mut max_val = 0.0f32;
        for chunk in chunks {
            for &val in chunk {
                max_val = max_val.max(val.abs());
            }
        }
        for &val in remainder {
            max_val = max_val.max(val.abs());
        }
        max_val
    }

    /// SIMD-optimized vector addition
    ///
    /// Adds two signals element-wise using chunked processing.
    #[cfg(feature = "simd")]
    pub fn add_simd(a: &Array1<f32>, b: &Array1<f32>) -> IoResult<Array1<f32>> {
        if a.len() != b.len() {
            return Err(IoError::SignalError("Signals must have same length".into()));
        }
        let a_owned: Vec<f32> = a.iter().copied().collect();
        let b_owned: Vec<f32> = b.iter().copied().collect();
        let a_slice = a_owned.as_slice();
        let b_slice = b_owned.as_slice();
        let mut result = vec![0.0f32; a.len()];
        const CHUNK_SIZE: usize = 8;
        let chunks_a = a_slice.chunks_exact(CHUNK_SIZE);
        let chunks_b = b_slice.chunks_exact(CHUNK_SIZE);
        let result_chunks = result.chunks_exact_mut(CHUNK_SIZE);
        for ((chunk_a, chunk_b), result_chunk) in chunks_a.zip(chunks_b).zip(result_chunks) {
            for i in 0..CHUNK_SIZE {
                result_chunk[i] = chunk_a[i] + chunk_b[i];
            }
        }
        let remainder_start = (a.len() / CHUNK_SIZE) * CHUNK_SIZE;
        for i in remainder_start..a.len() {
            result[i] = a_slice[i] + b_slice[i];
        }
        Ok(Array1::from_vec(result))
    }

    /// SIMD-optimized vector multiplication
    ///
    /// Multiplies two signals element-wise using chunked processing.
    #[cfg(feature = "simd")]
    pub fn multiply_simd(a: &Array1<f32>, b: &Array1<f32>) -> IoResult<Array1<f32>> {
        if a.len() != b.len() {
            return Err(IoError::SignalError("Signals must have same length".into()));
        }
        let a_owned: Vec<f32> = a.iter().copied().collect();
        let b_owned: Vec<f32> = b.iter().copied().collect();
        let a_slice = a_owned.as_slice();
        let b_slice = b_owned.as_slice();
        let mut result = vec![0.0f32; a.len()];
        const CHUNK_SIZE: usize = 8;
        let chunks_a = a_slice.chunks_exact(CHUNK_SIZE);
        let chunks_b = b_slice.chunks_exact(CHUNK_SIZE);
        let result_chunks = result.chunks_exact_mut(CHUNK_SIZE);
        for ((chunk_a, chunk_b), result_chunk) in chunks_a.zip(chunks_b).zip(result_chunks) {
            for i in 0..CHUNK_SIZE {
                result_chunk[i] = chunk_a[i] * chunk_b[i];
            }
        }
        let remainder_start = (a.len() / CHUNK_SIZE) * CHUNK_SIZE;
        for i in remainder_start..a.len() {
            result[i] = a_slice[i] * b_slice[i];
        }
        Ok(Array1::from_vec(result))
    }

    /// Compute spectrogram (Short-Time Fourier Transform)
    ///
    /// Returns a 2D array where rows are time frames and columns are frequency bins.
    /// Only the positive frequency bins (0 to n_fft/2) are returned.
    pub fn spectrogram(
        &mut self,
        signal: &Array1<f32>,
        n_fft: usize,
        hop_length: usize,
        window: WindowType,
    ) -> IoResult<Spectrogram> {
        if n_fft == 0 || hop_length == 0 {
            return Err(IoError::SignalError(
                "n_fft and hop_length must be > 0".into(),
            ));
        }
        if signal.len() < n_fft {
            return Err(IoError::SignalError("Signal shorter than n_fft".into()));
        }
        let window_coeffs = Self::create_window(window, n_fft);
        let num_frames = (signal.len() - n_fft) / hop_length + 1;
        let num_bins = n_fft / 2 + 1;
        let mut magnitudes = Vec::with_capacity(num_frames * num_bins);
        let mut phases = Vec::with_capacity(num_frames * num_bins);
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_length;
            let frame: Vec<f32> = signal
                .iter()
                .skip(start)
                .take(n_fft)
                .zip(window_coeffs.iter())
                .map(|(&s, &w)| s * w)
                .collect();
            let spectrum = self.fft(&Array1::from_vec(frame))?;
            for bin in spectrum.iter().take(num_bins) {
                magnitudes.push(bin.norm());
                phases.push(bin.arg());
            }
        }
        Ok(Spectrogram {
            magnitudes,
            phases,
            num_frames,
            num_bins,
            hop_length,
            sample_rate: self.sample_rate,
        })
    }

    /// Create a window function
    pub fn create_window(window_type: WindowType, size: usize) -> Vec<f32> {
        match window_type {
            WindowType::Rectangular => vec![1.0; size],
            WindowType::Hann => (0..size)
                .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
                .collect(),
            WindowType::Hamming => (0..size)
                .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (size - 1) as f32).cos())
                .collect(),
            WindowType::Blackman => (0..size)
                .map(|i| {
                    let n = i as f32 / (size - 1) as f32;
                    0.42 - 0.5 * (2.0 * PI * n).cos() + 0.08 * (4.0 * PI * n).cos()
                })
                .collect(),
            WindowType::Bartlett => (0..size)
                .map(|i| {
                    let half = (size - 1) as f32 / 2.0;
                    1.0 - ((i as f32 - half) / half).abs()
                })
                .collect(),
            WindowType::Kaiser { beta } => {
                let i0_beta = bessel_i0(beta);
                (0..size)
                    .map(|i| {
                        let n = 2.0 * i as f32 / (size - 1) as f32 - 1.0;
                        bessel_i0(beta * (1.0 - n * n).sqrt()) / i0_beta
                    })
                    .collect()
            }
        }
    }

    /// Apply window function to a frame (convenience method)
    pub fn apply_window_to_frame(frame: &[f32], window_type: WindowType) -> Vec<f32> {
        let window = Self::create_window(window_type, frame.len());
        frame
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect()
    }

    /// Compute Mel filterbank
    pub fn mel_filterbank(
        num_filters: usize,
        n_fft: usize,
        sample_rate: f32,
        f_min: f32,
        f_max: f32,
    ) -> Vec<Vec<f32>> {
        let num_bins = n_fft / 2 + 1;
        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);
        let mel_points: Vec<f32> = (0..=num_filters + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (num_filters + 1) as f32)
            .collect();
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&f| ((n_fft as f32 + 1.0) * f / sample_rate).floor() as usize)
            .collect();
        let mut filterbank = Vec::with_capacity(num_filters);
        for m in 0..num_filters {
            let mut filter = vec![0.0; num_bins];
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];
            let rise_denom = (center - left).max(1) as f32;
            for (offset, val) in filter[left..center.min(num_bins)].iter_mut().enumerate() {
                *val = offset as f32 / rise_denom;
            }
            let fall_denom = (right - center).max(1) as f32;
            for (offset, val) in filter[center..right.min(num_bins)].iter_mut().enumerate() {
                *val = (right - center - offset) as f32 / fall_denom;
            }
            filterbank.push(filter);
        }
        filterbank
    }

    /// Convert frequency in Hz to Mel scale
    pub fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert Mel scale to frequency in Hz
    pub fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Compute Mel-frequency cepstral coefficients (MFCCs)
    pub fn mfcc(
        &mut self,
        signal: &Array1<f32>,
        n_mfcc: usize,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
    ) -> IoResult<Vec<Vec<f32>>> {
        let f_max = self.sample_rate / 2.0;
        let filterbank = Self::mel_filterbank(n_mels, n_fft, self.sample_rate, 0.0, f_max);
        let spec = self.spectrogram(signal, n_fft, hop_length, WindowType::Hann)?;
        let mut mfccs = Vec::with_capacity(spec.num_frames);
        for frame_idx in 0..spec.num_frames {
            let power: Vec<f32> = (0..spec.num_bins)
                .map(|bin| {
                    let mag = spec.magnitudes[frame_idx * spec.num_bins + bin];
                    mag * mag
                })
                .collect();
            let mel_energies: Vec<f32> = filterbank
                .iter()
                .map(|filter| {
                    let energy: f32 = filter.iter().zip(power.iter()).map(|(&f, &p)| f * p).sum();
                    (energy + 1e-10).ln()
                })
                .collect();
            let mfcc_frame = Self::dct(&mel_energies, n_mfcc);
            mfccs.push(mfcc_frame);
        }
        Ok(mfccs)
    }

    /// Discrete Cosine Transform (Type-II)
    fn dct(input: &[f32], n_coeffs: usize) -> Vec<f32> {
        let n = input.len();
        (0..n_coeffs)
            .map(|k| {
                let sum: f32 = input
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        x * (PI * k as f32 * (2.0 * i as f32 + 1.0) / (2.0 * n as f32)).cos()
                    })
                    .sum();
                sum * (2.0 / n as f32).sqrt()
            })
            .collect()
    }
}
