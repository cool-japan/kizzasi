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
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::f32::consts::PI;

/// Signal processor for pre-processing sensor data
pub struct SignalProcessor {
    pub(super) buffer_size: usize,
    pub(super) sample_rate: f32,
    /// Cached FFT plans keyed by (transform size, direction). Building a
    /// `Plan` with `Flags::MEASURE` benchmarks candidate algorithms and is
    /// the most expensive planning mode; `spectrogram()`/`mfcc()` call
    /// `fft()` once per frame (~1000+ times for a few seconds of audio at a
    /// typical hop size), so re-planning on every call was the dominant
    /// cost. The cache reuses the plan for repeated calls at the same size.
    plan_cache: HashMap<(usize, Direction), Plan<f32>>,
}

impl SignalProcessor {
    /// Create a new signal processor
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer_size,
            sample_rate: 44100.0,
            plan_cache: HashMap::new(),
        }
    }

    /// Get (building and caching if necessary) the FFT plan for a given
    /// transform size and direction.
    fn get_or_create_plan(&mut self, n: usize, direction: Direction) -> IoResult<&Plan<f32>> {
        match self.plan_cache.entry((n, direction)) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let plan = Plan::dft_1d(n, direction, Flags::MEASURE).ok_or_else(|| {
                    IoError::SignalError(format!("FFT planning failed for size {n}"))
                })?;
                Ok(entry.insert(plan))
            }
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
        let input: Vec<Complex<f32>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        let plan = self.get_or_create_plan(n, Direction::Forward)?;
        plan.execute(&input, &mut output);
        Ok(output)
    }

    /// Apply inverse FFT
    pub fn ifft(&mut self, spectrum: &mut [Complex<f32>]) -> IoResult<Array1<f32>> {
        let n = spectrum.len();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        let plan = self.get_or_create_plan(n, Direction::Backward)?;
        plan.execute(spectrum, &mut output);
        let scale = 1.0 / n as f32;
        let result: Vec<f32> = output.iter().map(|c| c.re * scale).collect();
        Ok(Array1::from_vec(result))
    }

    /// Check if a number is a power of 2
    fn is_power_of_2(n: usize) -> bool {
        n != 0 && (n & (n - 1)) == 0
    }

    /// FFT for power-of-2 sizes, zero-padding the input up to the next
    /// power of two first.
    ///
    /// A previous version was a byte-identical alias for `fft()` -- calling
    /// it made no difference whatsoever, for power-of-2 sizes or otherwise.
    /// Zero-padding (via `zero_pad_pow2`) guarantees the transform size
    /// handed to the planner is always a power of two, which is the
    /// standard technique for forcing fast radix-2/Stockham code paths when
    /// the input length itself isn't already one. This changes the number
    /// of output bins (and hence frequency resolution) for non-power-of-2
    /// inputs versus plain `fft()`; use `fft()` directly if you need
    /// bin-for-bin parity with the original length.
    pub fn fft_pow2(&mut self, signal: &Array1<f32>) -> IoResult<Vec<Complex<f32>>> {
        let padded = Self::zero_pad_pow2(signal);
        self.fft(&padded)
    }

    /// Inverse FFT counterpart to `fft_pow2`.
    ///
    /// The transform size is dictated entirely by `spectrum.len()` (there is
    /// no separate "padding" step to perform on an inverse transform), so
    /// this is intentionally identical to `ifft()`; it exists for API
    /// symmetry with `fft_pow2`. If `spectrum` came from `fft_pow2` on a
    /// non-power-of-2 signal, the reconstructed output will be the padded
    /// (longer) length.
    pub fn ifft_pow2(&mut self, spectrum: &mut [Complex<f32>]) -> IoResult<Array1<f32>> {
        self.ifft(spectrum)
    }

    /// Compute power spectrum of the zero-padded (power-of-2) signal.
    ///
    /// Returns the magnitude squared of `fft_pow2`'s output, so for a
    /// non-power-of-2 input this genuinely differs from `power_spectrum()`
    /// (both in values, due to zero-padding changing bin resolution, and in
    /// length).
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
            Filter::LowPass { cutoff, order } => self.lowpass_fft(signal, cutoff, order),
            Filter::HighPass { cutoff, order } => self.highpass_fft(signal, cutoff, order),
            Filter::BandPass { low, high, order } => self.bandpass_fft(signal, low, high, order),
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
        let mut output = vec![Complex::new(0.0, 0.0); n];
        let plan = self.get_or_create_plan(n, Direction::Backward)?;
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
        // Accumulate incrementally: `sum` always holds the sum of samples
        // `max(0, i - window + 1)..=i`. A previous version seeded `sum`
        // with the sum of the FIRST `window` samples up front and then
        // divided by `window.min(i + 1)` for i < window as if it were a
        // running partial sum -- for i < window - 1 that used samples from
        // the future and produced grossly wrong output (e.g. window=3,
        // signal=[1,2,3,4] gave result[0] = 6.0 instead of 1.0).
        let mut sum: f32 = 0.0;
        for i in 0..signal.len() {
            sum += signal[i];
            if i >= window {
                sum -= signal[i - window];
            }
            result[i] = sum / window.min(i + 1) as f32;
        }
        Ok(result)
    }

    /// Validate a filter cutoff frequency against the Nyquist limit.
    fn validate_cutoff(&self, cutoff: f32, name: &str) -> IoResult<()> {
        let nyquist = self.sample_rate / 2.0;
        if !cutoff.is_finite() || cutoff < 0.0 || cutoff > nyquist {
            return Err(IoError::SignalError(format!(
                "{name} cutoff must be within [0, {nyquist}] Hz (Nyquist for sample_rate={}), got {cutoff}",
                self.sample_rate
            )));
        }
        Ok(())
    }

    /// Validate a Butterworth filter order (must be at least 1).
    fn validate_order(order: usize) -> IoResult<usize> {
        if order == 0 {
            return Err(IoError::SignalError(
                "Filter order must be >= 1".to_string(),
            ));
        }
        Ok(order)
    }

    /// Low-pass Butterworth filter, applied in the frequency domain.
    ///
    /// Shapes each FFT bin by the standard Butterworth magnitude response
    /// `|H(f)| = 1 / sqrt(1 + (f / cutoff)^(2 * order))` instead of hard
    /// zeroing bins outside the passband: a brick-wall mask ignores `order`
    /// entirely (every order produces identical output) and rings in the
    /// time domain. Higher `order` now genuinely yields a steeper rolloff,
    /// approaching a brick wall as `order` grows.
    fn lowpass_fft(
        &mut self,
        signal: &Array1<f32>,
        cutoff: f32,
        order: usize,
    ) -> IoResult<Array1<f32>> {
        self.validate_cutoff(cutoff, "LowPass")?;
        let order = Self::validate_order(order)?;

        let mut spectrum = self.fft(signal)?;
        let n = spectrum.len();
        for (i, item) in spectrum.iter_mut().enumerate() {
            let freq_bin = if i <= n / 2 { i } else { n - i };
            let freq_hz = freq_bin as f32 * self.sample_rate / n as f32;
            let gain = butterworth_lowpass_gain(freq_hz, cutoff, order);
            *item *= Complex::new(gain, 0.0);
        }
        self.ifft(&mut spectrum)
    }

    /// High-pass Butterworth filter, applied in the frequency domain.
    ///
    /// Uses the same conjugate-mirror bin mapping as `bandpass_fft`
    /// (`freq_bin = min(i, n - i)`). A previous version zeroed
    /// `spectrum[n - 1 - i]` as bin `i`'s mirror, which is off by one: the
    /// true mirror of bin `i` (for `i >= 1`) is bin `n - i`, not `n - 1 -
    /// i`. That desynchronized which bins got attenuated from their actual
    /// negative-frequency partners, leaving the spectrum non-conjugate-
    /// symmetric and the filtered (real) output wrong.
    fn highpass_fft(
        &mut self,
        signal: &Array1<f32>,
        cutoff: f32,
        order: usize,
    ) -> IoResult<Array1<f32>> {
        self.validate_cutoff(cutoff, "HighPass")?;
        let order = Self::validate_order(order)?;

        let mut spectrum = self.fft(signal)?;
        let n = spectrum.len();
        for (i, item) in spectrum.iter_mut().enumerate() {
            let freq_bin = if i <= n / 2 { i } else { n - i };
            let freq_hz = freq_bin as f32 * self.sample_rate / n as f32;
            let gain = butterworth_highpass_gain(freq_hz, cutoff, order);
            *item *= Complex::new(gain, 0.0);
        }
        self.ifft(&mut spectrum)
    }

    /// Band-pass filter using FFT (cascaded Butterworth high-pass + low-pass
    /// magnitude response, sharing the low-pass/high-pass validated `order`).
    fn bandpass_fft(
        &mut self,
        signal: &Array1<f32>,
        low: f32,
        high: f32,
        order: usize,
    ) -> IoResult<Array1<f32>> {
        self.validate_cutoff(low, "BandPass low")?;
        self.validate_cutoff(high, "BandPass high")?;
        if low > high {
            return Err(IoError::SignalError(format!(
                "BandPass low ({low}) must not exceed high ({high})"
            )));
        }
        let order = Self::validate_order(order)?;

        let mut spectrum = self.fft(signal)?;
        let n = spectrum.len();
        for (i, item) in spectrum.iter_mut().enumerate() {
            let freq_bin = if i <= n / 2 { i } else { n - i };
            let freq_hz = freq_bin as f32 * self.sample_rate / n as f32;
            let gain = butterworth_highpass_gain(freq_hz, low, order)
                * butterworth_lowpass_gain(freq_hz, high, order);
            *item *= Complex::new(gain, 0.0);
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

    /// SIMD-accelerated RMS computation.
    ///
    /// Dispatches to `scirs2-core`'s vectorised `simd_dot` (AVX2/AVX-512 on
    /// x86-64, NEON on aarch64, scalar fallback elsewhere) over a borrowed
    /// view of `signal`.
    ///
    /// A previous version of every `*_simd` function here contained no
    /// vectorisation whatsoever -- just a scalar loop over `chunks_exact(8)`
    /// -- *and* materialised a full owned `Vec<f32>` copy of each input
    /// first, so enabling the `simd` feature strictly cost an extra O(n)
    /// allocation and copy per call while computing exactly what the scalar
    /// path already computed.
    #[cfg(feature = "simd")]
    pub fn rms_simd(signal: &Array1<f32>) -> f32 {
        use scirs2_core::simd_ops::SimdUnifiedOps;
        let view = signal.view();
        let sum_sq = f32::simd_dot(&view, &view);
        (sum_sq / signal.len() as f32).sqrt()
    }

    /// SIMD-accelerated normalization
    ///
    /// Normalizes signal to [-1, 1] range without copying the input.
    #[cfg(feature = "simd")]
    pub fn normalize_simd(signal: &Array1<f32>) -> Array1<f32> {
        let max_abs = Self::max_abs_simd(signal);
        if max_abs > 0.0 {
            signal.mapv(|x| x / max_abs)
        } else {
            signal.clone()
        }
    }

    /// SIMD-accelerated max absolute value.
    ///
    /// Computed as `max(max_element, -min_element)`, which avoids allocating
    /// the intermediate `|x|` array that a `simd_abs` + `simd_max_element`
    /// pair would need.
    #[cfg(feature = "simd")]
    pub fn max_abs_simd(signal: &Array1<f32>) -> f32 {
        use scirs2_core::simd_ops::SimdUnifiedOps;
        if signal.is_empty() {
            return 0.0;
        }
        let view = signal.view();
        let max_val = f32::simd_max_element(&view);
        let min_val = f32::simd_min_element(&view);
        max_val.max(-min_val)
    }

    /// SIMD-accelerated vector addition
    ///
    /// Adds two signals element-wise via `scirs2-core`'s vectorised `simd_add`.
    #[cfg(feature = "simd")]
    pub fn add_simd(a: &Array1<f32>, b: &Array1<f32>) -> IoResult<Array1<f32>> {
        use scirs2_core::simd_ops::SimdUnifiedOps;
        if a.len() != b.len() {
            return Err(IoError::SignalError("Signals must have same length".into()));
        }
        Ok(f32::simd_add(&a.view(), &b.view()))
    }

    /// SIMD-accelerated vector multiplication
    ///
    /// Multiplies two signals element-wise via `scirs2-core`'s vectorised
    /// `simd_mul`.
    #[cfg(feature = "simd")]
    pub fn multiply_simd(a: &Array1<f32>, b: &Array1<f32>) -> IoResult<Array1<f32>> {
        use scirs2_core::simd_ops::SimdUnifiedOps;
        if a.len() != b.len() {
            return Err(IoError::SignalError("Signals must have same length".into()));
        }
        Ok(f32::simd_mul(&a.view(), &b.view()))
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

/// Butterworth low-pass magnitude response `1 / sqrt(1 + (f / cutoff)^(2n))`.
///
/// `order` is clamped to at least 1 by callers (`SignalProcessor::validate_order`).
fn butterworth_lowpass_gain(freq: f32, cutoff: f32, order: usize) -> f32 {
    if cutoff <= 0.0 {
        // A zero-width passband lets only DC through.
        return if freq <= 0.0 { 1.0 } else { 0.0 };
    }
    let exponent = 2 * i32::try_from(order).unwrap_or(i32::MAX);
    let ratio = (freq / cutoff).powi(exponent);
    1.0 / (1.0 + ratio).sqrt()
}

/// Butterworth high-pass magnitude response `1 / sqrt(1 + (cutoff / f)^(2n))`.
fn butterworth_highpass_gain(freq: f32, cutoff: f32, order: usize) -> f32 {
    if freq <= 0.0 {
        return 0.0;
    }
    if cutoff <= 0.0 {
        return 1.0;
    }
    let exponent = 2 * i32::try_from(order).unwrap_or(i32::MAX);
    let ratio = (cutoff / freq).powi(exponent);
    1.0 / (1.0 + ratio).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, sample_rate: f32, n: usize) -> Array1<f32> {
        Array1::from_vec(
            (0..n)
                .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
                .collect(),
        )
    }

    fn rms(signal: &Array1<f32>) -> f32 {
        let sum_sq: f32 = signal.iter().map(|x| x * x).sum();
        (sum_sq / signal.len() as f32).sqrt()
    }

    // === moving_average (high, id=31) ===

    fn naive_moving_average(signal: &[f32], window: usize) -> Vec<f32> {
        (0..signal.len())
            .map(|i| {
                let start = i.saturating_sub(window - 1);
                let slice = &signal[start..=i];
                slice.iter().sum::<f32>() / slice.len() as f32
            })
            .collect()
    }

    #[test]
    fn test_moving_average_matches_naive_reference() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        for window in [1usize, 2, 3, 4, 8] {
            let mut processor = SignalProcessor::new(signal.len());
            let result = processor
                .apply_filter(&signal, Filter::MovingAverage { window })
                .unwrap();
            let expected = naive_moving_average(signal.as_slice().unwrap(), window);
            for (got, want) in result.iter().zip(expected.iter()) {
                assert!(
                    (got - want).abs() < 1e-5,
                    "window={window}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn test_moving_average_first_sample_is_itself() {
        // Regression for the specific bug: with window=3 and
        // signal=[1,2,3,4], result[0] used to be 6.0 (sum of the first 3
        // samples divided by 1) instead of 1.0 (just the first sample).
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let mut processor = SignalProcessor::new(signal.len());
        let result = processor
            .apply_filter(&signal, Filter::MovingAverage { window: 3 })
            .unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6, "result[0] = {}", result[0]);
        assert!((result[1] - 1.5).abs() < 1e-6, "result[1] = {}", result[1]);
        assert!((result[2] - 2.0).abs() < 1e-6, "result[2] = {}", result[2]);
        assert!((result[3] - 3.0).abs() < 1e-6, "result[3] = {}", result[3]);
    }

    // === FFT filters: mirror bins + Butterworth order (high, id=32/33) ===

    #[test]
    fn test_highpass_attenuates_low_tone_and_passes_high_tone() {
        let sample_rate = 8000.0;
        let n = 2048;
        // Low tone well inside the stopband, high tone well inside the passband.
        let low_tone = sine(100.0, sample_rate, n);
        let high_tone = sine(2000.0, sample_rate, n);
        let mixed = &low_tone + &high_tone;

        let mut processor = SignalProcessor::new(n).with_sample_rate(sample_rate);
        let filtered = processor
            .apply_filter(
                &mixed,
                Filter::HighPass {
                    cutoff: 800.0,
                    order: 4,
                },
            )
            .unwrap();

        // Correlate the filtered output against each pure tone: the high
        // tone should dominate. Before the mirror-bin fix, the wrong bins
        // were attenuated and this relationship did not reliably hold.
        let corr_low: f32 = filtered
            .iter()
            .zip(low_tone.iter())
            .map(|(a, b)| a * b)
            .sum();
        let corr_high: f32 = filtered
            .iter()
            .zip(high_tone.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            corr_high.abs() > corr_low.abs() * 3.0,
            "expected the high tone to dominate after high-pass filtering: corr_low={corr_low}, corr_high={corr_high}"
        );
    }

    #[test]
    fn test_lowpass_attenuates_high_tone_and_passes_low_tone() {
        let sample_rate = 8000.0;
        let n = 2048;
        let low_tone = sine(100.0, sample_rate, n);
        let high_tone = sine(2000.0, sample_rate, n);
        let mixed = &low_tone + &high_tone;

        let mut processor = SignalProcessor::new(n).with_sample_rate(sample_rate);
        let filtered = processor
            .apply_filter(
                &mixed,
                Filter::LowPass {
                    cutoff: 800.0,
                    order: 4,
                },
            )
            .unwrap();

        let corr_low: f32 = filtered
            .iter()
            .zip(low_tone.iter())
            .map(|(a, b)| a * b)
            .sum();
        let corr_high: f32 = filtered
            .iter()
            .zip(high_tone.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            corr_low.abs() > corr_high.abs() * 3.0,
            "expected the low tone to dominate after low-pass filtering: corr_low={corr_low}, corr_high={corr_high}"
        );
    }

    #[test]
    fn test_bandpass_passes_middle_tone_and_rejects_edges() {
        let sample_rate = 8000.0;
        let n = 2048;
        let low_tone = sine(100.0, sample_rate, n);
        let mid_tone = sine(1000.0, sample_rate, n);
        let high_tone = sine(3000.0, sample_rate, n);
        let mixed = &(&low_tone + &mid_tone) + &high_tone;

        let mut processor = SignalProcessor::new(n).with_sample_rate(sample_rate);
        let filtered = processor
            .apply_filter(
                &mixed,
                Filter::BandPass {
                    low: 500.0,
                    high: 1500.0,
                    order: 4,
                },
            )
            .unwrap();

        let corr_mid: f32 = filtered
            .iter()
            .zip(mid_tone.iter())
            .map(|(a, b)| a * b)
            .sum();
        let corr_low: f32 = filtered
            .iter()
            .zip(low_tone.iter())
            .map(|(a, b)| a * b)
            .sum();
        let corr_high: f32 = filtered
            .iter()
            .zip(high_tone.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(corr_mid.abs() > corr_low.abs() * 3.0);
        assert!(corr_mid.abs() > corr_high.abs() * 3.0);
    }

    #[test]
    fn test_filter_order_changes_output() {
        // Before the fix, `order` was destructured with `..` and discarded
        // entirely: every order produced byte-identical output.
        let sample_rate = 8000.0;
        let n = 512;
        let signal = sine(1500.0, sample_rate, n);

        let mut p1 = SignalProcessor::new(n).with_sample_rate(sample_rate);
        let out_order_1 = p1
            .apply_filter(
                &signal,
                Filter::LowPass {
                    cutoff: 1000.0,
                    order: 1,
                },
            )
            .unwrap();

        let mut p8 = SignalProcessor::new(n).with_sample_rate(sample_rate);
        let out_order_8 = p8
            .apply_filter(
                &signal,
                Filter::LowPass {
                    cutoff: 1000.0,
                    order: 8,
                },
            )
            .unwrap();

        assert!(
            rms(&(&out_order_1 - &out_order_8)) > 1e-4,
            "different Butterworth orders must produce different output"
        );

        // A higher order should roll off more sharply, i.e. attenuate a
        // stopband tone (well above cutoff) more strongly.
        assert!(
            rms(&out_order_8) < rms(&out_order_1),
            "order=8 should attenuate a tone above cutoff more than order=1: rms8={}, rms1={}",
            rms(&out_order_8),
            rms(&out_order_1)
        );
    }

    #[test]
    fn test_filter_rejects_cutoff_above_nyquist() {
        let mut processor = SignalProcessor::new(64).with_sample_rate(1000.0);
        let signal = Array1::zeros(64);
        let result = processor.apply_filter(
            &signal,
            Filter::LowPass {
                cutoff: 900.0, // > Nyquist (500 Hz)
                order: 2,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_filter_rejects_zero_order() {
        let mut processor = SignalProcessor::new(64).with_sample_rate(1000.0);
        let signal = Array1::zeros(64);
        let result = processor.apply_filter(
            &signal,
            Filter::HighPass {
                cutoff: 100.0,
                order: 0,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_bandpass_rejects_low_greater_than_high() {
        let mut processor = SignalProcessor::new(64).with_sample_rate(1000.0);
        let signal = Array1::zeros(64);
        let result = processor.apply_filter(
            &signal,
            Filter::BandPass {
                low: 400.0,
                high: 100.0,
                order: 2,
            },
        );
        assert!(result.is_err());
    }

    // === fft/ifft basics + plan caching sanity (used across everything above) ===

    #[test]
    fn test_fft_ifft_round_trip() {
        let mut processor = SignalProcessor::new(256);
        let signal = sine(10.0, 256.0, 256);
        let mut spectrum = processor.fft(&signal).unwrap();
        let reconstructed = processor.ifft(&mut spectrum).unwrap();
        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-3, "a={a}, b={b}");
        }
    }

    // === fft_pow2 / power_spectrum_pow2 genuinely differ from the plain
    // versions for non-power-of-2 input (medium, id=42) ===

    #[test]
    fn test_fft_pow2_zero_pads_non_power_of_two_input() {
        let n = 100; // not a power of two
        let signal = sine(10.0, 100.0, n);
        let mut processor = SignalProcessor::new(n);

        let plain = processor.fft(&signal).unwrap();
        let padded = processor.fft_pow2(&signal).unwrap();

        assert_eq!(plain.len(), 100);
        assert_eq!(
            padded.len(),
            128,
            "fft_pow2 should zero-pad 100 samples up to the next power of two (128)"
        );
    }

    #[test]
    fn test_fft_pow2_matches_fft_for_already_pow2_input() {
        let n = 64;
        let signal = sine(5.0, 64.0, n);
        let mut processor = SignalProcessor::new(n);

        let plain = processor.fft(&signal).unwrap();
        let pow2 = processor.fft_pow2(&signal).unwrap();

        assert_eq!(plain.len(), pow2.len());
        for (a, b) in plain.iter().zip(pow2.iter()) {
            assert!((a.re - b.re).abs() < 1e-3);
            assert!((a.im - b.im).abs() < 1e-3);
        }
    }

    #[test]
    fn test_power_spectrum_pow2_differs_from_plain_for_non_pow2_input() {
        let n = 100;
        let signal = sine(10.0, 100.0, n);
        let mut processor = SignalProcessor::new(n);

        let plain = processor.power_spectrum(&signal).unwrap();
        let padded = processor.power_spectrum_pow2(&signal).unwrap();

        assert_eq!(plain.len(), 100);
        assert_eq!(padded.len(), 128);
    }

    // === FFT plan cache correctness (medium, id=43) ===

    #[test]
    fn test_plan_cache_preserves_correctness_across_repeated_calls() {
        // The cache must not corrupt results on repeated use at the same
        // size with DIFFERENT input data each time.
        let mut processor = SignalProcessor::new(256);
        for freq in [5.0, 11.0, 23.0, 40.0] {
            let signal = sine(freq, 256.0, 256);
            let mut spectrum = processor.fft(&signal).unwrap();
            let reconstructed = processor.ifft(&mut spectrum).unwrap();
            for (a, b) in signal.iter().zip(reconstructed.iter()) {
                assert!((a - b).abs() < 1e-3, "freq={freq}: a={a}, b={b}");
            }
        }
    }

    #[test]
    fn test_envelope_uses_cached_plan_and_stays_correct() {
        let mut processor = SignalProcessor::new(256);
        let signal = sine(20.0, 256.0, 256);
        // Call fft/ifft first so the Backward-direction plan for n=256 is
        // already cached before envelope() requests the same (n, Backward)
        // entry.
        let mut spectrum = processor.fft(&signal).unwrap();
        let _ = processor.ifft(&mut spectrum).unwrap();

        let env = processor.envelope(&signal).unwrap();
        assert_eq!(env.len(), 256);
        assert!(env.iter().all(|x| x.is_finite() && *x >= 0.0));
    }

    #[test]
    fn test_fft_reused_across_calls_with_same_and_different_sizes() {
        // Exercises the plan cache (id=43) with repeated calls at the same
        // size (cache hit) and a different size (cache miss + new entry).
        let mut processor = SignalProcessor::new(128);
        let sig_a = sine(5.0, 128.0, 128);
        let sig_b = sine(7.0, 128.0, 128);
        let sig_c = sine(3.0, 64.0, 64);

        let spec_a = processor.fft(&sig_a).unwrap();
        let spec_b = processor.fft(&sig_b).unwrap();
        let spec_c = processor.fft(&sig_c).unwrap();

        assert_eq!(spec_a.len(), 128);
        assert_eq!(spec_b.len(), 128);
        assert_eq!(spec_c.len(), 64);
    }

    // === Window functions / mel filterbank / DCT (test-gap, id=49) ===

    #[test]
    fn test_create_window_shapes() {
        const SIZE: usize = 64;

        let rectangular = SignalProcessor::create_window(WindowType::Rectangular, SIZE);
        assert_eq!(rectangular.len(), SIZE);
        assert!(rectangular.iter().all(|&w| (w - 1.0).abs() < 1e-6));

        for window_type in [
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::Bartlett,
            WindowType::Kaiser { beta: 8.6 },
        ] {
            let window = SignalProcessor::create_window(window_type, SIZE);
            assert_eq!(window.len(), SIZE, "{window_type:?}");
            assert!(
                window.iter().all(|w| w.is_finite()),
                "{window_type:?} produced a non-finite coefficient"
            );
            // Tapered windows peak in the middle and are symmetric.
            let peak = window.iter().copied().fold(f32::MIN, f32::max);
            let middle = window.get(SIZE / 2).copied().unwrap_or(0.0);
            assert!(
                (peak - middle).abs() < 0.05,
                "{window_type:?}: peak {peak} not near the centre value {middle}"
            );
            for i in 0..SIZE / 2 {
                let left = window.get(i).copied().unwrap_or(0.0);
                let right = window.get(SIZE - 1 - i).copied().unwrap_or(0.0);
                assert!(
                    (left - right).abs() < 1e-4,
                    "{window_type:?} is not symmetric at {i}: {left} vs {right}"
                );
            }
            // Endpoints are attenuated relative to the centre.
            let first = window.first().copied().unwrap_or(0.0);
            assert!(first < middle, "{window_type:?}: first {first} >= {middle}");
        }
    }

    #[test]
    fn test_hann_window_endpoints_are_zero() {
        let window = SignalProcessor::create_window(WindowType::Hann, 32);
        assert!(window.first().copied().unwrap_or(1.0).abs() < 1e-6);
        assert!(window.last().copied().unwrap_or(1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_window_to_frame_scales_the_frame() {
        let frame = vec![1.0f32; 16];
        let windowed = SignalProcessor::apply_window_to_frame(&frame, WindowType::Hann);
        let reference = SignalProcessor::create_window(WindowType::Hann, 16);
        assert_eq!(windowed.len(), reference.len());
        for (got, want) in windowed.iter().zip(reference.iter()) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mel_scale_round_trip() {
        for hz in [0.0f32, 100.0, 700.0, 1000.0, 4000.0, 8000.0] {
            let back = SignalProcessor::mel_to_hz(SignalProcessor::hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-1, "hz {hz} round-tripped to {back}");
        }
        // The mel scale is monotonically increasing.
        assert!(SignalProcessor::hz_to_mel(100.0) < SignalProcessor::hz_to_mel(1000.0));
    }

    #[test]
    fn test_mel_filterbank_geometry() {
        let n_fft = 512;
        let num_filters = 20;
        let filterbank = SignalProcessor::mel_filterbank(num_filters, n_fft, 16_000.0, 0.0, 8000.0);

        assert_eq!(filterbank.len(), num_filters);
        let mut previous_peak = 0usize;
        for (index, filter) in filterbank.iter().enumerate() {
            assert_eq!(filter.len(), n_fft / 2 + 1, "filter {index}");
            assert!(
                filter.iter().all(|&w| (0.0..=1.0).contains(&w)),
                "filter {index} has out-of-range weights"
            );
            assert!(
                filter.iter().any(|&w| w > 0.0),
                "filter {index} is entirely zero"
            );
            let peak = filter
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(bin, _)| bin)
                .unwrap_or(0);
            assert!(
                peak >= previous_peak,
                "filter {index} peaks at bin {peak}, before the previous {previous_peak}"
            );
            previous_peak = peak;
        }
    }

    #[test]
    fn test_dct_matches_naive_reference() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin()).collect();
        let n_coeffs = 8;
        let got = SignalProcessor::dct(&input, n_coeffs);

        let n = input.len() as f32;
        let expected: Vec<f32> = (0..n_coeffs)
            .map(|k| {
                let sum: f32 = input
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x * (PI * k as f32 * (2.0 * i as f32 + 1.0) / (2.0 * n)).cos())
                    .sum();
                sum * (2.0 / n).sqrt()
            })
            .collect();

        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn test_dct_of_constant_signal_is_dc_only() {
        let input = vec![2.0f32; 16];
        let coefficients = SignalProcessor::dct(&input, 8);
        // Only the k = 0 coefficient survives for a constant input.
        assert!(coefficients.first().copied().unwrap_or(0.0).abs() > 1.0);
        for (k, &value) in coefficients.iter().enumerate().skip(1) {
            assert!(value.abs() < 1e-3, "coefficient {k} = {value}");
        }
    }

    // === SIMD paths agree with the scalar reference (high, id=37) ===

    #[cfg(feature = "simd")]
    #[test]
    fn test_rms_simd_matches_scalar_rms() {
        for n in [1usize, 7, 8, 9, 64, 1000] {
            let signal = Array1::from_vec((0..n).map(|i| (i as f32 * 0.31).sin()).collect());
            let scalar = SignalProcessor::rms(&signal);
            let simd = SignalProcessor::rms_simd(&signal);
            assert!(
                (scalar - simd).abs() < 1e-4,
                "n={n}: scalar={scalar}, simd={simd}"
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_max_abs_simd_matches_scalar_reference() {
        for n in [1usize, 5, 8, 33, 257] {
            let signal =
                Array1::from_vec((0..n).map(|i| (i as f32 * 0.7).cos() * 3.0 - 1.0).collect());
            let expected = signal.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let got = SignalProcessor::max_abs_simd(&signal);
            assert!((expected - got).abs() < 1e-5, "n={n}: {expected} vs {got}");
        }
        assert_eq!(
            SignalProcessor::max_abs_simd(&Array1::from_vec(vec![])),
            0.0
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_normalize_simd_matches_scalar_normalize() {
        let signal = Array1::from_vec((0..37).map(|i| (i as f32 * 0.21).sin() * 4.0).collect());
        let scalar = SignalProcessor::normalize(&signal);
        let simd = SignalProcessor::normalize_simd(&signal);
        assert_eq!(scalar.len(), simd.len());
        for (a, b) in scalar.iter().zip(simd.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        // All-zero input must not divide by zero.
        let zeros = Array1::zeros(16);
        assert!(SignalProcessor::normalize_simd(&zeros)
            .iter()
            .all(|x| *x == 0.0));
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_add_and_multiply_simd_match_elementwise_reference() {
        for n in [1usize, 8, 15, 128] {
            let a = Array1::from_vec((0..n).map(|i| i as f32 * 0.5).collect());
            let b = Array1::from_vec((0..n).map(|i| 3.0 - i as f32 * 0.25).collect());

            let sum = SignalProcessor::add_simd(&a, &b).expect("lengths match");
            let product = SignalProcessor::multiply_simd(&a, &b).expect("lengths match");
            assert_eq!(sum.len(), n);
            assert_eq!(product.len(), n);
            for i in 0..n {
                assert!((sum[i] - (a[i] + b[i])).abs() < 1e-5, "add mismatch at {i}");
                assert!(
                    (product[i] - (a[i] * b[i])).abs() < 1e-5,
                    "mul mismatch at {i}"
                );
            }
        }

        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(SignalProcessor::add_simd(&a, &b).is_err());
        assert!(SignalProcessor::multiply_simd(&a, &b).is_err());
    }

    #[test]
    fn test_mfcc_produces_finite_coefficients() {
        let sample_rate = 16_000.0;
        let mut processor = SignalProcessor::new(4096).with_sample_rate(sample_rate);
        let signal = sine(440.0, sample_rate, 4096);

        let mfccs = processor.mfcc(&signal, 13, 512, 256, 26).unwrap();
        assert!(!mfccs.is_empty());
        for frame in &mfccs {
            assert_eq!(frame.len(), 13);
            assert!(frame.iter().all(|c| c.is_finite()));
        }
    }
}
