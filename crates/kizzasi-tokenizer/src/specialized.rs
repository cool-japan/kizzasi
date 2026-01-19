//! Specialized tokenizers for signal processing
//!
//! This module provides domain-specific tokenization strategies:
//! - Wavelet-based: Multi-resolution time-frequency analysis
//! - Fourier-based: Frequency domain representation via FFT
//! - DCT-based: Discrete Cosine Transform (JPEG-style compression)
//! - K-means: Clustering-based vector quantization

use crate::{SignalTokenizer, TokenizerError, TokenizerResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

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

    /// Quantize coefficients
    fn quantize_coeffs(&self, coeffs: &[Vec<f32>]) -> Vec<Vec<i32>> {
        let levels = (1 << self.config.bits) as f32;
        let max_val = coeffs
            .iter()
            .flat_map(|c| c.iter())
            .map(|&x| x.abs())
            .fold(0.0_f32, f32::max);

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
        let levels = (1 << self.config.bits) as f32;

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
        let quantized = self.quantize_coeffs(&coeffs);

        // Flatten into 1D array
        let tokens: Vec<f32> = quantized
            .iter()
            .flat_map(|band| band.iter().map(|&q| q as f32))
            .collect();

        Ok(Array1::from_vec(tokens))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        // Reconstruct coefficient structure
        // This is a simplified version - in practice, we'd need to store band sizes
        let max_val = 1.0; // Simplified - should be stored with coefficients
        let quantized: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();

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
    /// Number of frequency bins to keep
    pub num_bins: usize,
    /// Whether to use magnitude only (discard phase)
    pub magnitude_only: bool,
    /// Quantization bits
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

    /// Compute FFT (simplified DFT for now)
    fn fft(&self, signal: &[f32]) -> Vec<(f32, f32)> {
        let n = signal.len();
        let mut spectrum = Vec::with_capacity(n);

        for k in 0..n {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for (i, &x) in signal.iter().enumerate() {
                let angle = -2.0 * PI * (k as f32) * (i as f32) / (n as f32);
                real_sum += x * angle.cos();
                imag_sum += x * angle.sin();
            }

            spectrum.push((real_sum, imag_sum));
        }

        spectrum
    }

    /// Compute inverse FFT
    fn ifft(&self, spectrum: &[(f32, f32)]) -> Vec<f32> {
        let n = spectrum.len();
        let mut signal = Vec::with_capacity(n);

        for i in 0..n {
            let mut sum = 0.0;

            for (k, &(real, imag)) in spectrum.iter().enumerate() {
                let angle = 2.0 * PI * (k as f32) * (i as f32) / (n as f32);
                sum += real * angle.cos() - imag * angle.sin();
            }

            signal.push(sum / (n as f32));
        }

        signal
    }
}

impl SignalTokenizer for FourierTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let spectrum = self.fft(
            signal
                .as_slice()
                .expect("Signal must have contiguous layout"),
        );

        let tokens: Vec<f32> = spectrum
            .iter()
            .take(self.config.num_bins)
            .flat_map(|&(real, imag)| {
                if self.config.magnitude_only {
                    vec![(real * real + imag * imag).sqrt()]
                } else {
                    vec![real, imag]
                }
            })
            .collect();

        Ok(Array1::from_vec(tokens))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let spectrum: Vec<(f32, f32)> = if self.config.magnitude_only {
            tokens
                .iter()
                .map(|&mag| (mag, 0.0)) // Zero phase
                .collect()
        } else {
            // Manually iterate in chunks of 2
            let mut result = Vec::new();
            let tokens_slice = tokens
                .as_slice()
                .expect("Tokens must have contiguous layout");
            for i in (0..tokens_slice.len()).step_by(2) {
                let real = tokens_slice[i];
                let imag = tokens_slice.get(i + 1).copied().unwrap_or(0.0);
                result.push((real, imag));
            }
            result
        };

        let reconstructed = self.ifft(&spectrum);
        Ok(Array1::from_vec(reconstructed))
    }

    fn embed_dim(&self) -> usize {
        if self.config.magnitude_only {
            self.config.num_bins
        } else {
            self.config.num_bins * 2
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
        Ok(Self { config })
    }

    /// Compute DCT-II
    fn dct(&self, signal: &[f32]) -> Vec<f32> {
        let n = signal.len();
        let mut coeffs = Vec::with_capacity(n);

        for k in 0..n {
            let mut sum = 0.0;
            for (i, &x) in signal.iter().enumerate() {
                sum += x * ((PI * k as f32 * (2 * i + 1) as f32) / (2.0 * n as f32)).cos();
            }

            let scale = if k == 0 {
                (1.0 / n as f32).sqrt()
            } else {
                (2.0 / n as f32).sqrt()
            };

            coeffs.push(sum * scale);
        }

        coeffs
    }

    /// Compute inverse DCT-II (DCT-III)
    fn idct(&self, coeffs: &[f32]) -> Vec<f32> {
        let n = coeffs.len();
        let mut signal = Vec::with_capacity(n);

        for i in 0..n {
            let mut sum = 0.0;

            for (k, &c) in coeffs.iter().enumerate() {
                let scale = if k == 0 {
                    (1.0 / n as f32).sqrt()
                } else {
                    (2.0 / n as f32).sqrt()
                };

                sum += c * scale * ((PI * k as f32 * (2 * i + 1) as f32) / (2.0 * n as f32)).cos();
            }

            signal.push(sum);
        }

        signal
    }

    /// Quantize DCT coefficients (zig-zag scan and quantization)
    fn quantize(&self, coeffs: &[f32]) -> Vec<i32> {
        let levels = (1 << self.config.bits) as f32;
        let max_val = coeffs
            .iter()
            .take(self.config.num_coeffs)
            .map(|&x| x.abs())
            .fold(0.0_f32, f32::max);

        if max_val == 0.0 {
            return vec![0; self.config.num_coeffs];
        }

        coeffs
            .iter()
            .take(self.config.num_coeffs)
            .map(|&x| {
                let normalized = x / max_val;
                let quantized = (normalized * (levels / 2.0)).round();
                quantized.clamp(-(levels / 2.0), levels / 2.0 - 1.0) as i32
            })
            .collect()
    }

    /// Dequantize coefficients
    fn dequantize(&self, quantized: &[i32], max_val: f32) -> Vec<f32> {
        let levels = (1 << self.config.bits) as f32;

        quantized
            .iter()
            .map(|&q| (q as f32 / (levels / 2.0)) * max_val)
            .collect()
    }
}

impl SignalTokenizer for DCTTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let coeffs = self.dct(
            signal
                .as_slice()
                .expect("Signal must have contiguous layout"),
        );
        let quantized = self.quantize(&coeffs);

        let tokens: Vec<f32> = quantized.iter().map(|&q| q as f32).collect();
        Ok(Array1::from_vec(tokens))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let max_val = 1.0; // Simplified - should be stored
        let quantized: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let coeffs = self.dequantize(&quantized, max_val);

        // Pad with zeros if needed
        let mut full_coeffs = coeffs;
        while full_coeffs.len() < tokens.len() {
            full_coeffs.push(0.0);
        }

        let reconstructed = self.idct(&full_coeffs);
        Ok(Array1::from_vec(reconstructed))
    }

    fn embed_dim(&self) -> usize {
        self.config.num_coeffs
    }

    fn vocab_size(&self) -> usize {
        1 << self.config.bits
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
    fn find_nearest_centroid(&self, window: &Array1<f32>) -> usize {
        (0..self.config.num_clusters)
            .min_by(|&a, &b| {
                let dist_a = Self::euclidean_distance(window, &self.centroids.row(a).to_owned());
                let dist_b = Self::euclidean_distance(window, &self.centroids.row(b).to_owned());
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("Range must be non-empty")
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
        assert_eq!(tokens.len(), 8); // 8 magnitude values

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
        assert_eq!(tokens.len(), 8); // 4 bins * 2 (real + imag)
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
        assert_eq!(tokens.len(), 8);

        let reconstructed = tokenizer.decode(&tokens).unwrap();
        assert_eq!(reconstructed.len(), 8);
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
        assert_eq!(tokens.len(), 4); // Compressed to 4 coeffs
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
}
