//! Specialized tokenizers for signal processing
//!
//! This module provides domain-specific tokenization strategies:
//! - Wavelet-based: Multi-resolution time-frequency analysis
//! - Fourier-based: Frequency domain representation via FFT
//! - DCT-based: Discrete Cosine Transform (JPEG-style compression)
//! - K-means: Clustering-based vector quantization

use crate::{SignalTokenizer, TokenizerError, TokenizerResult};
use oxifft::{Complex, Direction, Flags, Plan};
use scirs2_core::ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};

type Complex32 = Complex<f32>;

/// Wavelet family for decomposition
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WaveletFamily {
    /// Haar wavelet (simplest, discontinuous)
    Haar,
    /// Daubechies 4-tap wavelet (smooth, compact support)
    Daubechies4,
}

/// Configuration for wavelet-based tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveletConfig {
    /// Number of decomposition levels
    pub levels: usize,
    /// Wavelet family to use
    pub family: WaveletFamily,
    /// Quantization bits per coefficient
    pub bits: u8,
}

impl Default for WaveletConfig {
    fn default() -> Self {
        Self {
            levels: 3,
            family: WaveletFamily::Haar,
            bits: 8,
        }
    }
}

/// Wavelet-based tokenizer using multi-resolution decomposition
pub struct WaveletTokenizer {
    config: WaveletConfig,
    lowpass: Vec<f32>,
    highpass: Vec<f32>,
}

impl WaveletTokenizer {
    /// Create a new wavelet tokenizer
    pub fn new(config: WaveletConfig) -> TokenizerResult<Self> {
        if config.levels == 0 {
            return Err(TokenizerError::InvalidConfig(
                "Wavelet levels must be > 0".to_string(),
            ));
        }
        if config.bits == 0 || config.bits > 16 {
            return Err(TokenizerError::InvalidConfig(
                "Bits must be in range [1, 16]".to_string(),
            ));
        }

        let (lowpass, highpass) = match config.family {
            WaveletFamily::Haar => {
                // Haar wavelet filters (normalized)
                let sqrt2_inv = 1.0 / 2.0_f32.sqrt();
                (vec![sqrt2_inv, sqrt2_inv], vec![sqrt2_inv, -sqrt2_inv])
            }
            WaveletFamily::Daubechies4 => {
                // Daubechies-4 wavelet coefficients
                let sqrt2 = 2.0_f32.sqrt();
                let sqrt3 = 3.0_f32.sqrt();
                let h0 = (1.0 + sqrt3) / (4.0 * sqrt2);
                let h1 = (3.0 + sqrt3) / (4.0 * sqrt2);
                let h2 = (3.0 - sqrt3) / (4.0 * sqrt2);
                let h3 = (1.0 - sqrt3) / (4.0 * sqrt2);
                (
                    vec![h0, h1, h2, h3],
                    vec![h3, -h2, h1, -h0], // QMF relationship
                )
            }
        };

        Ok(Self {
            config,
            lowpass,
            highpass,
        })
    }

    /// Forward wavelet transform (one level)
    fn dwt_step(&self, signal: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = signal.len();
        let mut approx = Vec::with_capacity(n / 2);
        let mut detail = Vec::with_capacity(n / 2);

        for i in (0..n).step_by(2) {
            let mut low_sum = 0.0;
            let mut high_sum = 0.0;

            for (j, (&l, &h)) in self.lowpass.iter().zip(self.highpass.iter()).enumerate() {
                let idx = (i + j) % n; // Circular boundary
                low_sum += signal[idx] * l;
                high_sum += signal[idx] * h;
            }

            approx.push(low_sum);
            detail.push(high_sum);
        }

        (approx, detail)
    }

    /// Inverse wavelet transform (one level)
    fn idwt_step(&self, approx: &[f32], detail: &[f32]) -> Vec<f32> {
        let n = approx.len() * 2;
        let mut signal = vec![0.0; n];

        for i in 0..approx.len() {
            for (j, (&l, &h)) in self.lowpass.iter().zip(self.highpass.iter()).enumerate() {
                let idx = (2 * i + j) % n;
                signal[idx] += approx[i] * l + detail[i] * h;
            }
        }

        signal
    }

    /// Multi-level decomposition
    fn decompose(&self, signal: &Array1<f32>) -> Vec<Vec<f32>> {
        let mut coeffs = Vec::new();
        let mut current = signal.to_vec();

        for _ in 0..self.config.levels {
            let (approx, detail) = self.dwt_step(&current);
            coeffs.push(detail);
            current = approx;
        }

        // Add final approximation
        coeffs.push(current);
        coeffs.reverse(); // [approx, detail_N, ..., detail_1]
        coeffs
    }

    /// Multi-level reconstruction
    fn reconstruct(&self, coeffs: &[Vec<f32>]) -> Vec<f32> {
        let mut current = coeffs[0].clone();

        for detail in coeffs.iter().skip(1) {
            current = self.idwt_step(&current, detail);
        }

        current
    }

    /// Compute the global max absolute value across all wavelet bands.
    fn coeffs_max_val(coeffs: &[Vec<f32>]) -> f32 {
        coeffs
            .iter()
            .flat_map(|c| c.iter())
            .map(|&x| x.abs())
            .fold(0.0_f32, f32::max)
    }

    /// Quantize coefficients using a caller-supplied max_val for normalization.
    fn quantize_coeffs_with_max(&self, coeffs: &[Vec<f32>], max_val: f32) -> Vec<Vec<i32>> {
        // `1usize << bits` (rather than the default-i32 `1 << bits`) makes the
        // width explicit and matches the validated `bits` bound below.
        let levels = (1usize << self.config.bits) as f32;

        if max_val == 0.0 {
            return coeffs.iter().map(|c| vec![0; c.len()]).collect();
        }

        coeffs
            .iter()
            .map(|band| {
                band.iter()
                    .map(|&x| {
                        let normalized = x / max_val; // [-1, 1]
                        let quantized = (normalized * (levels / 2.0)).round();
                        quantized.clamp(-(levels / 2.0), levels / 2.0 - 1.0) as i32
                    })
                    .collect()
            })
            .collect()
    }

    /// Dequantize coefficients
    fn dequantize_coeffs(&self, quantized: &[Vec<i32>], max_val: f32) -> Vec<Vec<f32>> {
        let levels = (1usize << self.config.bits) as f32;

        quantized
            .iter()
            .map(|band| {
                band.iter()
                    .map(|&q| (q as f32 / (levels / 2.0)) * max_val)
                    .collect()
            })
            .collect()
    }
}

impl SignalTokenizer for WaveletTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let coeffs = self.decompose(signal);
        let max_val = Self::coeffs_max_val(&coeffs);
        let quantized = self.quantize_coeffs_with_max(&coeffs, max_val);

        // Prepend max_val as token[0] so decode can recover the correct scale.
        let mut tokens = vec![max_val];
        tokens.extend(
            quantized
                .iter()
                .flat_map(|band| band.iter().map(|&q| q as f32)),
        );

        Ok(Array1::from_vec(tokens))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if tokens.is_empty() {
            return Ok(Array1::zeros(0));
        }
        // token[0] is the max_val header written by encode.
        let max_val = tokens[0];
        let quantized: Vec<i32> = tokens.iter().skip(1).map(|&t| t as i32).collect();

        // Estimate band sizes (simplified - assumes power-of-2 signal length)
        let mut band_sizes = Vec::new();
        let total_len = quantized.len();
        let mut remaining = total_len;
        for _ in 0..self.config.levels {
            let size = remaining / 2;
            band_sizes.push(size);
            remaining -= size;
        }
        band_sizes.push(remaining);
        band_sizes.reverse();

        let mut offset = 0;
        let mut bands = Vec::new();
        for &size in &band_sizes {
            bands.push(quantized[offset..offset + size].to_vec());
            offset += size;
        }

        let dequantized = self.dequantize_coeffs(&bands, max_val);
        let reconstructed = self.reconstruct(&dequantized);

        Ok(Array1::from_vec(reconstructed))
    }

    fn embed_dim(&self) -> usize {
        // Variable based on signal length and decomposition
        0 // Indicates variable length
    }

    fn vocab_size(&self) -> usize {
        1 << self.config.bits
    }
}

/// Configuration for Fourier-based tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourierConfig {
    /// Number of frequency bins to keep (from the low-frequency end). The
    /// full, lossless spectrum of a real signal of length `n` needs
    /// `n / 2 + 1` bins (the rest are recoverable via Hermitian symmetry);
    /// fewer bins give a low-pass approximation.
    pub num_bins: usize,
    /// Whether to use magnitude only (discard phase). Phase-free
    /// reconstruction assumes zero phase for every kept bin, which is lossy
    /// even when `num_bins` covers the full spectrum.
    pub magnitude_only: bool,
    /// Reserved for a future quantized encoding of the kept bins.
    ///
    /// Currently **unused**: `encode`/`decode` store bins as continuous
    /// `f32` (real, imag) pairs (or magnitudes), matching
    /// [`SignalTokenizer::vocab_size`]'s `0` return, which this crate's
    /// trait documents as "continuous". Setting this field has no effect.
    pub bits: u8,
}

impl Default for FourierConfig {
    fn default() -> Self {
        Self {
            num_bins: 256,
            magnitude_only: false,
            bits: 8,
        }
    }
}

/// Fourier-based tokenizer using FFT
pub struct FourierTokenizer {
    config: FourierConfig,
}

impl FourierTokenizer {
    /// Create a new Fourier tokenizer
    pub fn new(config: FourierConfig) -> TokenizerResult<Self> {
        if config.num_bins == 0 {
            return Err(TokenizerError::InvalidConfig(
                "Number of bins must be > 0".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Compute the full `n`-point forward DFT via `oxifft` (O(n log n)).
    fn fft(&self, signal: &[f32]) -> TokenizerResult<Vec<(f32, f32)>> {
        let n = signal.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let input: Vec<Complex32> = signal.iter().map(|&x| Complex32::new(x, 0.0)).collect();
        let mut output = vec![Complex32::new(0.0, 0.0); n];

        let plan = Plan::<f32>::dft_1d(n, Direction::Forward, Flags::MEASURE)
            .ok_or_else(|| TokenizerError::encoding("fourier", "FFT planning failed"))?;
        plan.execute(&input, &mut output);

        Ok(output.into_iter().map(|c| (c.re, c.im)).collect())
    }

    /// Reconstruct an `n`-sample real signal from `kept` low-frequency bins
    /// of an `n`-point spectrum via the inverse DFT (O(n log n)).
    ///
    /// `kept` need not cover the whole spectrum: missing high-frequency
    /// bins are treated as zero (a low-pass approximation), and every kept
    /// bin `k in 1..kept.len()` whose Hermitian mirror `n - k` was *not*
    /// itself directly kept is reflected into that mirror position
    /// (conjugated) so the spectrum stays Hermitian-symmetric and the
    /// inverse transform of a real-valued signal is real-valued, as a true
    /// (band-limited) DFT of a real signal would be.
    fn ifft(&self, n: usize, kept: &[(f32, f32)]) -> TokenizerResult<Vec<f32>> {
        if n == 0 {
            return Ok(Vec::new());
        }

        let take_n = kept.len().min(n);
        let mut full = vec![(0.0f32, 0.0f32); n];
        full[..take_n].copy_from_slice(&kept[..take_n]);

        for k in 1..take_n {
            let mirror = n - k;
            if mirror != k && mirror >= take_n {
                let (re, im) = full[k];
                full[mirror] = (re, -im);
            }
        }

        let spectrum: Vec<Complex32> = full
            .iter()
            .map(|&(re, im)| Complex32::new(re, im))
            .collect();
        let mut time_domain = vec![Complex32::new(0.0, 0.0); n];

        let plan = Plan::<f32>::dft_1d(n, Direction::Backward, Flags::MEASURE)
            .ok_or_else(|| TokenizerError::decoding("fourier", "IFFT planning failed"))?;
        plan.execute(&spectrum, &mut time_domain);

        // oxifft's inverse transform is unnormalized (IFFT(FFT(x)) = n * x),
        // matching the convention already used in `perceptual.rs`.
        let norm = n as f32;
        Ok(time_domain.iter().map(|c| c.re / norm).collect())
    }
}

impl SignalTokenizer for FourierTokenizer {
    /// Encode a signal into `[n, bin_0, bin_1, ...]`, where `n` is the
    /// original signal length (needed by `decode` to run the inverse
    /// transform at the correct size) and the remaining tokens are the
    /// kept spectrum bins — `(real, imag)` pairs, or magnitudes when
    /// `magnitude_only` is set.
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        // `as_slice` returns `None` for any array that is not in standard
        // layout — a reversed or strided `Array1` is still valid input, so
        // copy the elements out in logical order instead of panicking on a
        // caller that the signature promises to serve with a `Result`.
        let samples: Vec<f32> = signal.iter().copied().collect();
        let n = samples.len();

        let mut tokens: Vec<f32> = Vec::with_capacity(1 + self.config.num_bins * 2);
        tokens.push(n as f32);

        if n == 0 {
            return Ok(Array1::from_vec(tokens));
        }

        let spectrum = self.fft(&samples)?;
        for &(real, imag) in spectrum.iter().take(self.config.num_bins) {
            if self.config.magnitude_only {
                tokens.push((real * real + imag * imag).sqrt());
            } else {
                tokens.push(real);
                tokens.push(imag);
            }
        }

        Ok(Array1::from_vec(tokens))
    }

    /// Decode tokens produced by [`FourierTokenizer::encode`] back into a
    /// signal of the original length.
    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if tokens.is_empty() {
            return Ok(Array1::zeros(0));
        }

        let tokens_vec: Vec<f32> = tokens.iter().copied().collect();
        // Negative or non-finite headers (e.g. from malformed/fuzzed input)
        // collapse to the empty signal rather than producing a bogus huge
        // allocation or a panic in the `as usize` cast.
        let n = if tokens_vec[0].is_finite() && tokens_vec[0] >= 0.0 {
            tokens_vec[0].round() as usize
        } else {
            0
        };

        let rest = &tokens_vec[1..];
        let kept: Vec<(f32, f32)> = if self.config.magnitude_only {
            rest.iter().map(|&mag| (mag, 0.0)).collect() // Zero phase
        } else {
            // Manually iterate in chunks of 2 rather than relying on
            // `as_slice`, so non-standard-layout token arrays still work.
            rest.chunks(2)
                .map(|chunk| {
                    let real = chunk.first().copied().unwrap_or(0.0);
                    let imag = chunk.get(1).copied().unwrap_or(0.0);
                    (real, imag)
                })
                .collect()
        };

        let reconstructed = self.ifft(n, &kept)?;
        Ok(Array1::from_vec(reconstructed))
    }

    fn embed_dim(&self) -> usize {
        // +1 for the length header prepended in encode.
        if self.config.magnitude_only {
            self.config.num_bins + 1
        } else {
            self.config.num_bins * 2 + 1
        }
    }

    fn vocab_size(&self) -> usize {
        0 // Continuous
    }
}

/// Configuration for DCT-based tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DCTConfig {
    /// Number of DCT coefficients to keep
    pub num_coeffs: usize,
    /// Quantization bits
    pub bits: u8,
}

impl Default for DCTConfig {
    fn default() -> Self {
        Self {
            num_coeffs: 64,
            bits: 8,
        }
    }
}

/// DCT-based tokenizer (Type-II DCT, like JPEG)
pub struct DCTTokenizer {
    config: DCTConfig,
}

impl DCTTokenizer {
    /// Create a new DCT tokenizer
    pub fn new(config: DCTConfig) -> TokenizerResult<Self> {
        if config.num_coeffs == 0 {
            return Err(TokenizerError::InvalidConfig(
                "Number of coefficients must be > 0".to_string(),
            ));
        }
        // Matches `WaveletTokenizer::new`'s bound: `levels = 1usize << bits`
        // is used unchecked in `quantize`/`dequantize` below, so `bits` must
        // stay within a range that shift can never overflow.
        if config.bits == 0 || config.bits > 16 {
            return Err(TokenizerError::InvalidConfig(
                "Bits must be in range [1, 16]".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Compute DCT-II via `oxifft`'s O(n log n) Makhoul-reduction implementation.
    fn dct(&self, signal: &[f32]) -> Vec<f32> {
        let n = signal.len();
        if n == 0 {
            return Vec::new();
        }
        let mut coeffs = vec![0.0f32; n];
        oxifft::dct2(signal, &mut coeffs);
        coeffs
    }

    /// Compute inverse DCT-II (DCT-III), rescaled to be the exact inverse of `dct`.
    fn idct(&self, coeffs: &[f32]) -> Vec<f32> {
        let n = coeffs.len();
        if n == 0 {
            return Vec::new();
        }
        let mut signal = vec![0.0f32; n];
        oxifft::dct3(coeffs, &mut signal);
        // `oxifft::dct2`/`dct3` follow the FFTW REDFT10/REDFT01 convention,
        // under which `dct3(dct2(x)) == (n/2) * x` (verified against the
        // crate's own implementation); scaling by `2/n` recovers the exact
        // inverse, equivalent to the orthonormal scale factors the previous
        // hand-rolled implementation applied per-coefficient.
        let scale = 2.0 / n as f32;
        for v in &mut signal {
            *v *= scale;
        }
        signal
    }

    /// Quantize DCT coefficients, returning both the quantized integers and the
    /// max-absolute-value used for normalization so the caller can store it as a header.
    fn quantize(&self, coeffs: &[f32]) -> (Vec<i32>, f32) {
        let levels = (1usize << self.config.bits) as f32;
        let max_val = coeffs
            .iter()
            .take(self.config.num_coeffs)
            .map(|&x| x.abs())
            .fold(0.0_f32, f32::max);

        if max_val == 0.0 {
            return (vec![0; self.config.num_coeffs], 0.0);
        }

        let quantized = coeffs
            .iter()
            .take(self.config.num_coeffs)
            .map(|&x| {
                let normalized = x / max_val;
                let quantized = (normalized * (levels / 2.0)).round();
                quantized.clamp(-(levels / 2.0), levels / 2.0 - 1.0) as i32
            })
            .collect();

        (quantized, max_val)
    }

    /// Dequantize coefficients
    fn dequantize(&self, quantized: &[i32], max_val: f32) -> Vec<f32> {
        let levels = (1usize << self.config.bits) as f32;

        quantized
            .iter()
            .map(|&q| (q as f32 / (levels / 2.0)) * max_val)
            .collect()
    }
}

impl SignalTokenizer for DCTTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        // See `FourierTokenizer::encode`: a non-standard-layout `Array1` is
        // legitimate input, so copy rather than requiring a contiguous slice.
        let samples: Vec<f32> = signal.iter().copied().collect();
        let coeffs = self.dct(&samples);
        let (quantized, max_val) = self.quantize(&coeffs);

        // Prepend max_val as token[0] so decode can recover the correct scale.
        let mut tokens = vec![max_val];
        tokens.extend(quantized.iter().map(|&q| q as f32));
        Ok(Array1::from_vec(tokens))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if tokens.is_empty() {
            return Ok(Array1::zeros(0));
        }
        // token[0] is the max_val header written by encode.
        let max_val = tokens[0];
        let quantized: Vec<i32> = tokens.iter().skip(1).map(|&t| t as i32).collect();
        let coeffs = self.dequantize(&quantized, max_val);

        // Pad with zeros if needed to reach num_coeffs
        let mut full_coeffs = coeffs;
        while full_coeffs.len() < self.config.num_coeffs {
            full_coeffs.push(0.0);
        }

        let reconstructed = self.idct(&full_coeffs);
        Ok(Array1::from_vec(reconstructed))
    }

    fn embed_dim(&self) -> usize {
        // +1 for the max_val header prepended in encode.
        self.config.num_coeffs + 1
    }

    fn vocab_size(&self) -> usize {
        1usize << self.config.bits
    }
}

/// Configuration for K-means clustering tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansConfig {
    /// Number of clusters (codebook size)
    pub num_clusters: usize,
    /// Embedding dimension (window size)
    pub embed_dim: usize,
    /// Maximum iterations for k-means
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f32,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            num_clusters: 256,
            embed_dim: 16,
            max_iterations: 100,
            tolerance: 1e-4,
        }
    }
}

/// K-means clustering tokenizer for vector quantization
pub struct KMeansTokenizer {
    config: KMeansConfig,
    centroids: Array2<f32>,
    trained: bool,
}

impl KMeansTokenizer {
    /// Create a new K-means tokenizer (untrained)
    pub fn new(config: KMeansConfig) -> TokenizerResult<Self> {
        if config.num_clusters == 0 {
            return Err(TokenizerError::InvalidConfig(
                "Number of clusters must be > 0".to_string(),
            ));
        }
        if config.embed_dim == 0 {
            return Err(TokenizerError::InvalidConfig(
                "Embedding dimension must be > 0".to_string(),
            ));
        }

        let centroids = Array2::zeros((config.num_clusters, config.embed_dim));

        Ok(Self {
            config,
            centroids,
            trained: false,
        })
    }

    /// Train the k-means model on data
    pub fn train(&mut self, data: &[Array1<f32>]) -> TokenizerResult<()> {
        if data.is_empty() {
            return Err(TokenizerError::InvalidConfig(
                "No training data".to_string(),
            ));
        }

        // Extract windows from signals
        let mut windows = Vec::new();
        for signal in data {
            for i in 0..=signal.len().saturating_sub(self.config.embed_dim) {
                let window = signal.slice(s![i..i + self.config.embed_dim]).to_owned();
                windows.push(window);
            }
        }

        if windows.len() < self.config.num_clusters {
            return Err(TokenizerError::InvalidConfig(
                "Not enough data for clustering".to_string(),
            ));
        }

        // Initialize centroids with k-means++
        self.kmeans_plus_plus_init(&windows)?;

        // Run k-means iterations
        for iteration in 0..self.config.max_iterations {
            // Assignment step
            let assignments = self.assign_clusters(&windows);

            // Update step
            let old_centroids = self.centroids.clone();
            self.update_centroids(&windows, &assignments)?;

            // Check convergence
            let change = self.compute_centroid_change(&old_centroids);
            if change < self.config.tolerance {
                tracing::debug!("K-means converged at iteration {}", iteration);
                break;
            }
        }

        self.trained = true;
        Ok(())
    }

    /// Initialize centroids using k-means++
    fn kmeans_plus_plus_init(&mut self, windows: &[Array1<f32>]) -> TokenizerResult<()> {
        use scirs2_core::random::quick::{random_f32, random_usize};

        // Choose first centroid randomly
        let first_idx = random_usize(0, windows.len() - 1);
        self.centroids.row_mut(0).assign(&windows[first_idx].view());

        // Choose remaining centroids
        for k in 1..self.config.num_clusters {
            let mut distances = vec![f32::MAX; windows.len()];

            // Compute distances to nearest centroid
            for (i, window) in windows.iter().enumerate() {
                for j in 0..k {
                    let centroid = self.centroids.row(j);
                    let dist = Self::euclidean_distance(window, &centroid.to_owned());
                    distances[i] = distances[i].min(dist);
                }
            }

            // Choose next centroid with probability proportional to distance squared
            let total: f32 = distances.iter().map(|&d| d * d).sum();
            let mut threshold = random_f32() * total;
            let mut chosen_idx = 0;

            for (i, &dist) in distances.iter().enumerate() {
                threshold -= dist * dist;
                if threshold <= 0.0 {
                    chosen_idx = i;
                    break;
                }
            }

            self.centroids
                .row_mut(k)
                .assign(&windows[chosen_idx].view());
        }

        Ok(())
    }

    /// Assign windows to nearest clusters
    fn assign_clusters(&self, windows: &[Array1<f32>]) -> Vec<usize> {
        windows
            .iter()
            .map(|window| self.find_nearest_centroid(window))
            .collect()
    }

    /// Update centroids based on assignments
    fn update_centroids(
        &mut self,
        windows: &[Array1<f32>],
        assignments: &[usize],
    ) -> TokenizerResult<()> {
        let mut counts = vec![0usize; self.config.num_clusters];
        self.centroids.fill(0.0);

        // Accumulate
        for (window, &cluster) in windows.iter().zip(assignments.iter()) {
            for (i, &val) in window.iter().enumerate() {
                self.centroids[[cluster, i]] += val;
            }
            counts[cluster] += 1;
        }

        // Average (handle empty clusters by keeping old centroid)
        for (k, &count) in counts.iter().enumerate().take(self.config.num_clusters) {
            if count > 0 {
                for i in 0..self.config.embed_dim {
                    self.centroids[[k, i]] /= count as f32;
                }
            }
        }

        Ok(())
    }

    /// Find nearest centroid for a window
    ///
    /// `KMeansTokenizer::new` rejects `num_clusters == 0`, so the search range
    /// is never empty; the fallback exists only so the function cannot panic.
    fn find_nearest_centroid(&self, window: &Array1<f32>) -> usize {
        (0..self.config.num_clusters)
            .min_by(|&a, &b| {
                let dist_a = Self::euclidean_distance(window, &self.centroids.row(a).to_owned());
                let dist_b = Self::euclidean_distance(window, &self.centroids.row(b).to_owned());
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    }

    /// Compute Euclidean distance
    fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Compute change in centroids
    fn compute_centroid_change(&self, old_centroids: &Array2<f32>) -> f32 {
        self.centroids
            .iter()
            .zip(old_centroids.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get centroids
    pub fn centroids(&self) -> &Array2<f32> {
        &self.centroids
    }
}

impl SignalTokenizer for KMeansTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if !self.trained {
            return Err(TokenizerError::InvalidConfig(
                "K-means model not trained".to_string(),
            ));
        }

        let mut tokens = Vec::new();

        // Extract windows and assign to clusters
        for i in 0..=signal.len().saturating_sub(self.config.embed_dim) {
            let window = signal.slice(s![i..i + self.config.embed_dim]).to_owned();
            let cluster = self.find_nearest_centroid(&window);
            tokens.push(cluster as f32);
        }

        Ok(Array1::from_vec(tokens))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if !self.trained {
            return Err(TokenizerError::InvalidConfig(
                "K-means model not trained".to_string(),
            ));
        }

        // Simple overlap-add reconstruction
        let output_len = tokens.len() + self.config.embed_dim - 1;
        let mut signal = vec![0.0; output_len];
        let mut counts = vec![0.0; output_len];

        for (i, &token) in tokens.iter().enumerate() {
            let cluster = token as usize;
            if cluster >= self.config.num_clusters {
                return Err(TokenizerError::invalid_input(
                    "decoding",
                    "Invalid cluster index",
                ));
            }

            let centroid = self.centroids.row(cluster);
            for (j, &val) in centroid.iter().enumerate() {
                signal[i + j] += val;
                counts[i + j] += 1.0;
            }
        }

        // Average overlapping regions
        for (s, c) in signal.iter_mut().zip(counts.iter()) {
            if *c > 0.0 {
                *s /= c;
            }
        }

        Ok(Array1::from_vec(signal))
    }

    fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    fn vocab_size(&self) -> usize {
        self.config.num_clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelet_haar_basic() {
        let config = WaveletConfig {
            levels: 2,
            family: WaveletFamily::Haar,
            bits: 8,
        };
        let tokenizer = WaveletTokenizer::new(config).unwrap();

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let tokens = tokenizer.encode(&signal).unwrap();
        assert!(!tokens.is_empty());

        let reconstructed = tokenizer.decode(&tokens).unwrap();
        assert_eq!(reconstructed.len(), signal.len());
    }

    #[test]
    fn test_wavelet_daubechies4() {
        let config = WaveletConfig {
            levels: 1,
            family: WaveletFamily::Daubechies4,
            bits: 8,
        };
        let tokenizer = WaveletTokenizer::new(config).unwrap();

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let tokens = tokenizer.encode(&signal).unwrap();
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_wavelet_invalid_config() {
        let config = WaveletConfig {
            levels: 0,
            family: WaveletFamily::Haar,
            bits: 8,
        };
        assert!(WaveletTokenizer::new(config).is_err());

        let config = WaveletConfig {
            levels: 1,
            family: WaveletFamily::Haar,
            bits: 0,
        };
        assert!(WaveletTokenizer::new(config).is_err());
    }

    #[test]
    fn test_fourier_magnitude_only() {
        let config = FourierConfig {
            num_bins: 8,
            magnitude_only: true,
            bits: 8,
        };
        let tokenizer = FourierTokenizer::new(config).unwrap();

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
        let tokens = tokenizer.encode(&signal).unwrap();
        // token[0] is the length header; tokens[1..] are the 8 magnitude values.
        assert_eq!(tokens.len(), 9);

        let reconstructed = tokenizer.decode(&tokens).unwrap();
        assert_eq!(reconstructed.len(), 8);
    }

    #[test]
    fn test_fourier_complex() {
        let config = FourierConfig {
            num_bins: 4,
            magnitude_only: false,
            bits: 8,
        };
        let tokenizer = FourierTokenizer::new(config).unwrap();

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let tokens = tokenizer.encode(&signal).unwrap();
        // token[0] is the length header; tokens[1..] are 4 bins * 2 (real + imag).
        assert_eq!(tokens.len(), 9);
    }

    /// Regression: `decode(encode(x))` must reproduce `x`, both in length and
    /// in value, whenever `num_bins` covers the full non-redundant spectrum
    /// (`n/2 + 1` bins). Before the fix, `decode` ran an `num_bins`-point
    /// inverse DFT of the first `num_bins` bins of an `n`-point forward DFT
    /// — neither the right transform size nor a valid inverse of anything —
    /// so it produced neither the original signal nor a length-matching
    /// approximation of it.
    #[test]
    fn test_fourier_full_spectrum_roundtrip_is_exact() {
        for &(signal, num_bins) in &[
            (&[1.0f32, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0][..], 5usize), // n=8, n/2+1=5
            (&[1.0, 5.0, -2.0, 3.0, 0.5, 7.0, 2.0][..], 4usize),        // n=7 (odd), n/2+1=4
            (&[3.0, -1.0][..], 2usize),                                 // n=2, n/2+1=2
        ] {
            let config = FourierConfig {
                num_bins,
                magnitude_only: false,
                bits: 8,
            };
            let tokenizer = FourierTokenizer::new(config).unwrap();
            let input = Array1::from_vec(signal.to_vec());

            let tokens = tokenizer.encode(&input).unwrap();
            let decoded = tokenizer.decode(&tokens).unwrap();

            assert_eq!(
                decoded.len(),
                input.len(),
                "decoded length must match the original signal length"
            );
            for (a, b) in decoded.iter().zip(input.iter()) {
                assert!(
                    (a - b).abs() < 1e-3,
                    "full-spectrum roundtrip mismatch: {a} vs {b} (signal={signal:?}, num_bins={num_bins})"
                );
            }
        }
    }

    /// Regression: with `num_bins` smaller than the full spectrum, `decode`
    /// must still return a signal of the *original* length (a low-pass
    /// approximation), not a signal truncated to `num_bins` samples.
    #[test]
    fn test_fourier_partial_spectrum_preserves_length() {
        let config = FourierConfig {
            num_bins: 2,
            magnitude_only: false,
            bits: 8,
        };
        let tokenizer = FourierTokenizer::new(config).unwrap();
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]);

        let tokens = tokenizer.encode(&signal).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();

        assert_eq!(decoded.len(), signal.len());
        assert!(decoded.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_dct_basic() {
        let config = DCTConfig {
            num_coeffs: 8,
            bits: 8,
        };
        let tokenizer = DCTTokenizer::new(config).unwrap();

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let tokens = tokenizer.encode(&signal).unwrap();
        // token[0] is the max_val header; tokens[1..] are the 8 quantized coefficients.
        assert_eq!(tokens.len(), 9);

        let reconstructed = tokenizer.decode(&tokens).unwrap();
        assert_eq!(reconstructed.len(), 8);
    }

    /// Regression: `DCTConfig::bits` used to be unvalidated and fed straight
    /// into `1 << self.config.bits` on an `i32`-defaulted literal —
    /// `bits = 40` overflowed the shift (panic in debug, silently wrapped in
    /// release), and `bits = 31` produced `levels = i32::MIN`, making
    /// `quantized.clamp(-(levels/2.0), levels/2.0 - 1.0)` panic on
    /// `min > max`. `DCTTokenizer::new` must reject out-of-range `bits`
    /// up front instead, mirroring `WaveletTokenizer::new`'s `[1, 16]` bound.
    #[test]
    fn test_dct_invalid_bits_rejected() {
        for &bad_bits in &[0u8, 17, 31, 40, 255] {
            let config = DCTConfig {
                num_coeffs: 4,
                bits: bad_bits,
            };
            assert!(
                DCTTokenizer::new(config).is_err(),
                "bits = {bad_bits} should be rejected, not reach the shift/clamp in quantize()"
            );
        }

        // The valid boundary values must still be accepted.
        for &ok_bits in &[1u8, 8, 16] {
            let config = DCTConfig {
                num_coeffs: 4,
                bits: ok_bits,
            };
            assert!(
                DCTTokenizer::new(config).is_ok(),
                "bits = {ok_bits} should be accepted"
            );
        }
    }

    #[test]
    fn test_dct_compression() {
        let config = DCTConfig {
            num_coeffs: 4,
            bits: 8,
        };
        let tokenizer = DCTTokenizer::new(config).unwrap();

        // Smooth signal should compress well
        let signal = Array1::from_vec(vec![1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8, 0.9]);
        let tokens = tokenizer.encode(&signal).unwrap();
        // token[0] is the max_val header; tokens[1..] are the 4 quantized coefficients.
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_kmeans_training() {
        let config = KMeansConfig {
            num_clusters: 4,
            embed_dim: 4,
            max_iterations: 50,
            tolerance: 1e-3,
        };
        let mut tokenizer = KMeansTokenizer::new(config).unwrap();

        // Generate training data
        let data = vec![
            Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]),
            Array1::from_vec(vec![3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0]),
        ];

        assert!(!tokenizer.is_trained());
        tokenizer.train(&data).unwrap();
        assert!(tokenizer.is_trained());

        let centroids = tokenizer.centroids();
        assert_eq!(centroids.shape(), &[4, 4]);
    }

    #[test]
    fn test_kmeans_encode_decode() {
        let config = KMeansConfig {
            num_clusters: 8,
            embed_dim: 4,
            max_iterations: 100,
            tolerance: 1e-4,
        };
        let mut tokenizer = KMeansTokenizer::new(config).unwrap();

        // Training data
        let data = vec![
            Array1::from_vec((0..32).map(|x| x as f32).collect::<Vec<_>>()),
            Array1::from_vec((0..32).map(|x| (x as f32).sin()).collect::<Vec<_>>()),
        ];

        tokenizer.train(&data).unwrap();

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let tokens = tokenizer.encode(&signal).unwrap();
        assert!(!tokens.is_empty());

        let reconstructed = tokenizer.decode(&tokens).unwrap();
        assert!(!reconstructed.is_empty());
    }

    #[test]
    fn test_kmeans_untrained_error() {
        let config = KMeansConfig::default();
        let tokenizer = KMeansTokenizer::new(config).unwrap();

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(tokenizer.encode(&signal).is_err());
    }

    #[test]
    fn test_kmeans_invalid_config() {
        let config = KMeansConfig {
            num_clusters: 0,
            embed_dim: 4,
            max_iterations: 10,
            tolerance: 1e-3,
        };
        assert!(KMeansTokenizer::new(config).is_err());
    }

    #[test]
    fn test_signal_tokenizer_trait() {
        let tokenizers: Vec<Box<dyn SignalTokenizer>> = vec![
            Box::new(
                WaveletTokenizer::new(WaveletConfig {
                    levels: 1,
                    family: WaveletFamily::Haar,
                    bits: 8,
                })
                .unwrap(),
            ),
            Box::new(
                FourierTokenizer::new(FourierConfig {
                    num_bins: 8,
                    magnitude_only: true,
                    bits: 8,
                })
                .unwrap(),
            ),
            Box::new(
                DCTTokenizer::new(DCTConfig {
                    num_coeffs: 8,
                    bits: 8,
                })
                .unwrap(),
            ),
        ];

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        for tokenizer in tokenizers {
            let tokens = tokenizer.encode(&signal).unwrap();
            assert!(!tokens.is_empty());
            assert!(tokenizer.vocab_size() > 0 || tokenizer.embed_dim() > 0);
        }
    }

    /// Build an owned `Array1` whose memory layout is *not* standard, so
    /// `as_slice()` returns `None`.
    fn reversed_array(values: Vec<f32>) -> Array1<f32> {
        let mut array = Array1::from_vec(values);
        array.invert_axis(scirs2_core::ndarray::Axis(0));
        assert!(
            array.as_slice().is_none(),
            "test fixture must be non-contiguous"
        );
        array
    }

    #[test]
    fn test_transform_tokenizers_accept_non_contiguous_signals() {
        // Regression: these encoders used `as_slice().expect(...)`, so a
        // strided or reversed array — legitimate input for a function
        // returning `TokenizerResult` — panicked instead of encoding.
        let signal = reversed_array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let fourier = FourierTokenizer::new(FourierConfig {
            num_bins: 4,
            magnitude_only: false,
            bits: 8,
        })
        .unwrap();
        let fourier_tokens = fourier
            .encode(&signal)
            .expect("Fourier encode must succeed");
        assert!(fourier_tokens.iter().all(|x| x.is_finite()));

        let dct = DCTTokenizer::new(DCTConfig {
            num_coeffs: 4,
            bits: 8,
        })
        .unwrap();
        let dct_tokens = dct.encode(&signal).expect("DCT encode must succeed");
        assert!(dct_tokens.iter().all(|x| x.is_finite()));

        // The same signal in standard layout must encode identically.
        let contiguous = Array1::from_vec(vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let reference = dct.encode(&contiguous).expect("DCT encode must succeed");
        assert_eq!(dct_tokens.len(), reference.len());
        for (a, b) in dct_tokens.iter().zip(reference.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "non-contiguous input must encode like its contiguous twin: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_fourier_decode_accepts_non_contiguous_tokens() {
        let fourier = FourierTokenizer::new(FourierConfig {
            num_bins: 4,
            magnitude_only: false,
            bits: 8,
        })
        .unwrap();
        let tokens = reversed_array(vec![1.0, 0.5, -0.25, 0.75, 0.0, -1.0, 2.0, 0.125]);
        let decoded = fourier
            .decode(&tokens)
            .expect("Fourier decode must succeed");
        assert!(decoded.iter().all(|x| x.is_finite()));
    }
}
