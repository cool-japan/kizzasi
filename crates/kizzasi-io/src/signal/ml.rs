//! Machine learning utilities for signal processing
//!
//! This module provides neural-network-based signal processing primitives:
//! - `SignalDenoiser`: single-hidden-layer MLP with online Adam adaptation for noise removal
//! - `FeatureExtractor`: MFCC and spectral feature extraction using OxiFFT
//! - `AnomalyDetector`: statistical and reconstruction-based anomaly detection
//! - `MiniAutoencoder`: lightweight autoencoder with encoder transfer and persistence

// All items in this module form the public API surface of kizzasi-io; they are not
// required to be used within this crate itself.
#![allow(dead_code)]

use crate::error::{IoError, IoResult};
use oxifft::{Complex, Direction, Flags, Plan};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{thread_rng, Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Adam optimizer: exponential decay rate for first moment estimates
const ADAM_BETA1: f32 = 0.9;
/// Adam optimizer: exponential decay rate for second moment estimates
const ADAM_BETA2: f32 = 0.999;
/// Adam optimizer: small constant for numerical stability
const ADAM_EPS: f32 = 1e-8;
/// Default Adam learning rate
const ADAM_LR: f32 = 1e-3;
/// Reference minimum energy for log-mel computation
const LOG_MEL_EPS: f32 = 1e-10;
/// Number of Mel filterbank bands used in FeatureExtractor
const N_MELS: usize = 40;

// ---------------------------------------------------------------------------
// Helper activation / linear algebra functions
// ---------------------------------------------------------------------------

/// ReLU activation — element-wise max(0, x)
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Sigmoid activation — 1 / (1 + e^{-x})
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of sigmoid given its *output* value s = σ(x)
fn sigmoid_prime(s: f32) -> f32 {
    s * (1.0 - s)
}

/// Derivative of ReLU given the pre-activation value x
fn relu_prime(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Compute the forward pass of one fully-connected layer.
///
/// `weights` shape: (out, in), `bias` shape: (out,)
/// Returns the pre-activation vector.
fn layer_forward(input: &Array1<f32>, weights: &Array2<f32>, bias: &Array1<f32>) -> Array1<f32> {
    let out_size = weights.nrows();
    let mut result = Array1::zeros(out_size);
    for i in 0..out_size {
        let mut acc = bias[i];
        for j in 0..input.len() {
            acc += weights[[i, j]] * input[j];
        }
        result[i] = acc;
    }
    result
}

/// Xavier / Glorot uniform initialisation — fills a weight matrix of shape (rows, cols).
fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
    let mut rng = thread_rng();
    let limit = (6.0_f32 / (rows + cols) as f32).sqrt();
    let uniform = Normal::new(0.0_f64, (limit as f64) / 3.0_f64)
        .expect("Normal distribution construction failed");
    Array2::from_shape_fn((rows, cols), |_| uniform.sample(&mut rng) as f32)
}

/// Hz → Mel conversion
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Mel → Hz conversion
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Build a triangular Mel filterbank.
///
/// Returns a matrix of shape (n_mels, n_fft/2+1).
fn build_mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: f32) -> Vec<Vec<f32>> {
    let num_bins = n_fft / 2 + 1;
    let f_min = 0.0_f32;
    let f_max = sample_rate / 2.0;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&f| ((n_fft as f32 + 1.0) * f / sample_rate).floor() as usize)
        .collect();

    let mut filterbank: Vec<Vec<f32>> = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let mut filter = vec![0.0_f32; num_bins];
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];
        let rise_denom = (center.saturating_sub(left)).max(1) as f32;
        for (offset, val) in filter[left..center.min(num_bins)].iter_mut().enumerate() {
            *val = offset as f32 / rise_denom;
        }
        let fall_denom = (right.saturating_sub(center)).max(1) as f32;
        for (offset, val) in filter[center..right.min(num_bins)].iter_mut().enumerate() {
            *val = (right - center - offset) as f32 / fall_denom;
        }
        filterbank.push(filter);
    }
    filterbank
}

/// DCT-II: returns first `n_coeffs` coefficients for input slice.
fn dct_ii(input: &[f32], n_coeffs: usize) -> Vec<f32> {
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

/// Hann window of length `size`.
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
        .collect()
}

/// Compute SNR in dB given a clean reference and noisy signal.
fn snr_db_impl(clean: &Array1<f32>, noisy: &Array1<f32>) -> f32 {
    let signal_power: f32 = clean.iter().map(|x| x * x).sum::<f32>() / clean.len() as f32;
    let noise: Array1<f32> = Array1::from_shape_fn(clean.len(), |i| noisy[i] - clean[i]);
    let noise_power: f32 = noise.iter().map(|x| x * x).sum::<f32>() / noise.len() as f32;
    if noise_power < f32::EPSILON {
        return f32::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

// ---------------------------------------------------------------------------
// Adam optimizer state (shared helper)
// ---------------------------------------------------------------------------

/// Flat Adam moment state for a parameter vector of known length.
#[derive(Debug, Clone)]
struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    step: u64,
}

impl AdamState {
    fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
        }
    }

    /// Apply Adam update: `param[i] -= lr * corrected_gradient`.
    fn update(&mut self, param: &mut [f32], grad: &[f32], lr: f32) {
        self.step += 1;
        let t = self.step as f32;
        let bc1 = 1.0 - ADAM_BETA1.powf(t);
        let bc2 = 1.0 - ADAM_BETA2.powf(t);
        for i in 0..param.len() {
            self.m[i] = ADAM_BETA1 * self.m[i] + (1.0 - ADAM_BETA1) * grad[i];
            self.v[i] = ADAM_BETA2 * self.v[i] + (1.0 - ADAM_BETA2) * grad[i] * grad[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            param[i] -= lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
        }
    }
}

// ---------------------------------------------------------------------------
// 1. SignalDenoiser
// ---------------------------------------------------------------------------

/// Single-hidden-layer MLP for neural denoising with online Adam adaptation.
///
/// Architecture: input → hidden (ReLU) → output (linear).
/// Weights are updated online via Adam whenever a clean reference is supplied.
pub struct SignalDenoiser {
    /// Input-to-hidden weight matrix (hidden_size × input_size)
    weights_ih: Array2<f32>,
    /// Hidden-to-output weight matrix (input_size × hidden_size)
    weights_ho: Array2<f32>,
    /// Hidden layer bias (hidden_size)
    bias_h: Array1<f32>,
    /// Output layer bias (input_size)
    bias_o: Array1<f32>,
    /// Adam state for weights_ih (flattened)
    adam_ih: AdamState,
    /// Adam state for weights_ho (flattened)
    adam_ho: AdamState,
    /// Adam state for bias_h
    adam_bh: AdamState,
    /// Adam state for bias_o
    adam_bo: AdamState,
    /// Learning rate
    lr: f32,
}

impl SignalDenoiser {
    /// Create a new denoiser with Xavier-initialised weights.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let weights_ih = xavier_init(hidden_size, input_size);
        let weights_ho = xavier_init(input_size, hidden_size);
        let bias_h = Array1::zeros(hidden_size);
        let bias_o = Array1::zeros(input_size);
        let adam_ih = AdamState::new(hidden_size * input_size);
        let adam_ho = AdamState::new(input_size * hidden_size);
        let adam_bh = AdamState::new(hidden_size);
        let adam_bo = AdamState::new(input_size);
        Self {
            weights_ih,
            weights_ho,
            bias_h,
            bias_o,
            adam_ih,
            adam_ho,
            adam_bh,
            adam_bo,
            lr: ADAM_LR,
        }
    }

    /// Forward pass: noisy → denoised output.
    ///
    /// If `clean_ref` is `Some`, a gradient step is taken (Adam) before returning.
    pub fn denoise(
        &mut self,
        noisy: &Array1<f32>,
        clean_ref: Option<&Array1<f32>>,
    ) -> IoResult<Array1<f32>> {
        if noisy.is_empty() {
            return Err(IoError::SignalError("Empty input to denoiser".into()));
        }
        let input_size = self.weights_ih.ncols();
        let hidden_size = self.weights_ih.nrows();

        if noisy.len() != input_size {
            return Err(IoError::SignalError(format!(
                "Denoiser input size mismatch: expected {input_size}, got {}",
                noisy.len()
            )));
        }

        // --- Forward pass ---
        let pre_h = layer_forward(noisy, &self.weights_ih, &self.bias_h);
        let h: Array1<f32> = pre_h.mapv(relu);
        let output = layer_forward(&h, &self.weights_ho, &self.bias_o);

        // --- Optional backward pass ---
        if let Some(clean) = clean_ref {
            if clean.len() != input_size {
                return Err(IoError::SignalError(
                    "clean_ref length must match input_size".into(),
                ));
            }

            // MSE loss gradient w.r.t output: (output - clean) * 2/N
            let scale = 2.0 / input_size as f32;
            let d_out: Vec<f32> = (0..input_size)
                .map(|i| scale * (output[i] - clean[i]))
                .collect();

            // Gradient for weights_ho and bias_o
            let mut d_who: Vec<f32> = vec![0.0; input_size * hidden_size];
            let d_bo: Vec<f32> = d_out.clone();
            // d_bo holds the bias_o gradient (same as d_out, used below for the Adam step)
            for i in 0..input_size {
                for j in 0..hidden_size {
                    d_who[i * hidden_size + j] = d_out[i] * h[j];
                }
            }

            // Back-propagate through hidden layer
            let mut d_h = vec![0.0_f32; hidden_size];
            for j in 0..hidden_size {
                for (i, &d_out_i) in d_out.iter().enumerate() {
                    d_h[j] += d_out_i * self.weights_ho[[i, j]];
                }
                d_h[j] *= relu_prime(pre_h[j]);
            }

            // Gradient for weights_ih and bias_h
            let mut d_wih: Vec<f32> = vec![0.0; hidden_size * input_size];
            let d_bh: Vec<f32> = d_h.clone();
            for i in 0..hidden_size {
                for j in 0..input_size {
                    d_wih[i * input_size + j] = d_h[i] * noisy[j];
                }
            }

            // Apply Adam updates
            {
                let wih_flat = self
                    .weights_ih
                    .as_slice_mut()
                    .ok_or_else(|| IoError::SignalError("weights_ih not contiguous".into()))?;
                self.adam_ih.update(wih_flat, &d_wih, self.lr);
            }
            {
                let who_flat = self
                    .weights_ho
                    .as_slice_mut()
                    .ok_or_else(|| IoError::SignalError("weights_ho not contiguous".into()))?;
                self.adam_ho.update(who_flat, &d_who, self.lr);
            }
            {
                let bh_flat = self
                    .bias_h
                    .as_slice_mut()
                    .ok_or_else(|| IoError::SignalError("bias_h not contiguous".into()))?;
                self.adam_bh.update(bh_flat, &d_bh, self.lr);
            }
            {
                let bo_flat = self
                    .bias_o
                    .as_slice_mut()
                    .ok_or_else(|| IoError::SignalError("bias_o not contiguous".into()))?;
                self.adam_bo.update(bo_flat, &d_bo, self.lr);
            }

            // Re-run forward with updated weights for a fresh output
            let pre_h2 = layer_forward(noisy, &self.weights_ih, &self.bias_h);
            let h2: Array1<f32> = pre_h2.mapv(relu);
            let output2 = layer_forward(&h2, &self.weights_ho, &self.bias_o);
            return Ok(output2);
        }

        Ok(output)
    }

    /// Compute SNR in dB between clean reference and noisy signal.
    ///
    /// Returns `f32::INFINITY` when the noise power is negligible.
    pub fn snr_db(signal: &Array1<f32>, noisy: &Array1<f32>) -> f32 {
        snr_db_impl(signal, noisy)
    }

    /// Set the Adam learning rate (default: 1e-3).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ---------------------------------------------------------------------------
// 2. FeatureExtractor
// ---------------------------------------------------------------------------

/// MFCC and spectral feature extractor backed by OxiFFT.
pub struct FeatureExtractor {
    /// Audio sample rate in Hz
    sample_rate: u32,
    /// Number of MFCC coefficients to return
    n_mfcc: usize,
    /// FFT size
    n_fft: usize,
    /// Hop length in samples
    hop_length: usize,
    /// Pre-computed Mel filterbank (n_mels × n_bins)
    mel_filterbank: Vec<Vec<f32>>,
    /// Streaming buffer for partial chunks
    stream_buffer: Vec<f32>,
}

impl FeatureExtractor {
    /// Create a new feature extractor.
    ///
    /// # Errors
    /// Returns an error if `n_fft < 2`, `hop_length == 0`, or `n_mfcc > N_MELS`.
    pub fn new(sample_rate: u32, n_mfcc: usize, n_fft: usize, hop_length: usize) -> IoResult<Self> {
        if n_fft < 2 {
            return Err(IoError::SignalError("n_fft must be >= 2".into()));
        }
        if hop_length == 0 {
            return Err(IoError::SignalError("hop_length must be > 0".into()));
        }
        if n_mfcc > N_MELS {
            return Err(IoError::SignalError(format!(
                "n_mfcc ({n_mfcc}) must not exceed N_MELS ({N_MELS})"
            )));
        }
        let mel_filterbank = build_mel_filterbank(N_MELS, n_fft, sample_rate as f32);
        Ok(Self {
            sample_rate,
            n_mfcc,
            n_fft,
            hop_length,
            mel_filterbank,
            stream_buffer: Vec::new(),
        })
    }

    /// Compute MFCCs for the whole signal.
    ///
    /// Returns array of shape `(n_frames, n_mfcc)`.
    pub fn extract_mfcc(&self, signal: &Array1<f32>) -> IoResult<Array2<f32>> {
        if signal.len() < self.n_fft {
            return Err(IoError::SignalError("Signal shorter than n_fft".into()));
        }
        let window = hann_window(self.n_fft);
        let num_frames = (signal.len() - self.n_fft) / self.hop_length + 1;
        let num_bins = self.n_fft / 2 + 1;

        let plan = Plan::dft_1d(self.n_fft, Direction::Forward, Flags::MEASURE)
            .ok_or_else(|| IoError::SignalError("FFT plan creation failed".into()))?;

        let mut rows: Vec<f32> = Vec::with_capacity(num_frames * self.n_mfcc);

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_length;
            let frame: Vec<Complex<f32>> = signal
                .iter()
                .skip(start)
                .take(self.n_fft)
                .zip(window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();
            let mut spectrum = vec![Complex::new(0.0_f32, 0.0_f32); self.n_fft];
            plan.execute(&frame, &mut spectrum);

            let power: Vec<f32> = (0..num_bins)
                .map(|b| {
                    let mag = spectrum[b].norm();
                    mag * mag
                })
                .collect();

            let mel_energies: Vec<f32> = self
                .mel_filterbank
                .iter()
                .map(|filter| {
                    let energy: f32 = filter.iter().zip(power.iter()).map(|(&f, &p)| f * p).sum();
                    (energy + LOG_MEL_EPS).ln()
                })
                .collect();

            let mfcc_frame = dct_ii(&mel_energies, self.n_mfcc);
            rows.extend_from_slice(&mfcc_frame);
        }

        Array2::from_shape_vec((num_frames, self.n_mfcc), rows)
            .map_err(|e| IoError::SignalError(format!("MFCC shape error: {e}")))
    }

    /// Compute per-frame spectral features: centroid, rolloff, flux, flatness.
    ///
    /// Returns array of shape `(n_frames, 4)`.
    pub fn extract_spectral_features(&self, signal: &Array1<f32>) -> IoResult<Array2<f32>> {
        if signal.len() < self.n_fft {
            return Err(IoError::SignalError("Signal shorter than n_fft".into()));
        }
        let window = hann_window(self.n_fft);
        let num_frames = (signal.len() - self.n_fft) / self.hop_length + 1;
        let num_bins = self.n_fft / 2 + 1;

        let plan = Plan::dft_1d(self.n_fft, Direction::Forward, Flags::MEASURE)
            .ok_or_else(|| IoError::SignalError("FFT plan creation failed".into()))?;

        let mut rows: Vec<f32> = Vec::with_capacity(num_frames * 4);
        let mut prev_mag: Option<Vec<f32>> = None;

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_length;
            let frame: Vec<Complex<f32>> = signal
                .iter()
                .skip(start)
                .take(self.n_fft)
                .zip(window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();
            let mut spectrum = vec![Complex::new(0.0_f32, 0.0_f32); self.n_fft];
            plan.execute(&frame, &mut spectrum);

            let magnitudes: Vec<f32> = (0..num_bins).map(|b| spectrum[b].norm()).collect();
            let mag_sum: f32 = magnitudes.iter().sum();

            // Spectral centroid
            let centroid = if mag_sum > f32::EPSILON {
                let weighted: f32 = magnitudes
                    .iter()
                    .enumerate()
                    .map(|(i, &m)| {
                        let freq = i as f32 * self.sample_rate as f32 / self.n_fft as f32;
                        m * freq
                    })
                    .sum();
                weighted / mag_sum
            } else {
                0.0
            };

            // Spectral rolloff (85% threshold)
            let threshold = 0.85 * mag_sum;
            let mut cumsum = 0.0_f32;
            let mut rolloff = 0.0_f32;
            for (i, &m) in magnitudes.iter().enumerate() {
                cumsum += m;
                if cumsum >= threshold {
                    rolloff = i as f32 * self.sample_rate as f32 / self.n_fft as f32;
                    break;
                }
            }

            // Spectral flux (difference from previous frame)
            let flux = if let Some(ref prev) = prev_mag {
                magnitudes
                    .iter()
                    .zip(prev.iter())
                    .map(|(&c, &p)| (c - p).powi(2))
                    .sum::<f32>()
                    .sqrt()
            } else {
                0.0
            };

            // Spectral flatness (geometric mean / arithmetic mean)
            let log_sum: f32 = magnitudes
                .iter()
                .map(|&m| (m + LOG_MEL_EPS).ln())
                .sum::<f32>();
            let geo_mean = (log_sum / num_bins as f32).exp();
            let arith_mean = mag_sum / num_bins as f32;
            let flatness = if arith_mean > f32::EPSILON {
                geo_mean / arith_mean
            } else {
                0.0
            };

            rows.extend_from_slice(&[centroid, rolloff, flux, flatness]);
            prev_mag = Some(magnitudes);
        }

        Array2::from_shape_vec((num_frames, 4), rows)
            .map_err(|e| IoError::SignalError(format!("Spectral feature shape error: {e}")))
    }

    /// Buffered streaming feature extraction.
    ///
    /// Accumulates `chunk` into an internal buffer.  When the buffer contains
    /// at least one full frame (`n_fft` samples), the oldest frame is extracted
    /// and its concatenated MFCC + spectral features are returned.  Returns
    /// `None` when there is not yet enough data.
    pub fn extract_streaming(&mut self, chunk: &[f32]) -> IoResult<Option<Array1<f32>>> {
        self.stream_buffer.extend_from_slice(chunk);

        if self.stream_buffer.len() < self.n_fft {
            return Ok(None);
        }

        // Extract one frame
        let frame_data: Array1<f32> = Array1::from_vec(self.stream_buffer[..self.n_fft].to_vec());

        // Consume hop_length samples from the buffer
        let consume = self.hop_length.min(self.stream_buffer.len());
        self.stream_buffer.drain(..consume);

        let mfcc = self.extract_mfcc(&frame_data)?;
        let spectral = self.extract_spectral_features(&frame_data)?;

        // Flatten: mfcc row 0 + spectral row 0
        let mfcc_row: Vec<f32> = mfcc.row(0).to_vec();
        let spectral_row: Vec<f32> = spectral.row(0).to_vec();
        let mut combined = mfcc_row;
        combined.extend_from_slice(&spectral_row);

        Ok(Some(Array1::from_vec(combined)))
    }

    /// Sample rate accessor
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// n_mfcc accessor
    pub fn n_mfcc(&self) -> usize {
        self.n_mfcc
    }

    /// n_fft accessor
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }
}

// ---------------------------------------------------------------------------
// 3. AnomalyDetector + AnomalyMethod
// ---------------------------------------------------------------------------

/// Method used by `AnomalyDetector` to compute anomaly scores.
#[derive(Debug, Clone)]
pub enum AnomalyMethod {
    /// Z-score thresholding: flag a sample if its maximum absolute z-score exceeds `threshold`.
    ZScore {
        /// Anomaly threshold (number of standard deviations)
        threshold: f32,
    },
    /// Mahalanobis distance approximation (diagonal covariance = z-score L2 norm).
    MahalanobisApprox {
        /// Distance threshold
        threshold: f32,
    },
    /// Autoencoder reconstruction error thresholding.
    AutoencoderReconstruction {
        /// Reconstruction MSE threshold
        threshold: f32,
    },
}

/// Statistical and reconstruction-based anomaly detector.
pub struct AnomalyDetector {
    /// Detection method
    method: AnomalyMethod,
    /// Feature-wise mean (fitted)
    mean: Array1<f32>,
    /// Feature-wise standard deviation (fitted)
    std: Array1<f32>,
    /// Optional autoencoder (used with `AutoencoderReconstruction` method)
    autoencoder: Option<MiniAutoencoder>,
}

impl AnomalyDetector {
    /// Create a new detector with the specified method.
    ///
    /// `fit` must be called before `detect` or `score`.
    pub fn new(method: AnomalyMethod) -> Self {
        Self {
            method,
            mean: Array1::zeros(1),
            std: Array1::ones(1),
            autoencoder: None,
        }
    }

    /// Fit the detector on normal training data.
    ///
    /// For `AutoencoderReconstruction`, a `MiniAutoencoder` is trained for 200 steps.
    ///
    /// # Errors
    /// Returns an error if `data` has zero rows or if FFT-related construction fails.
    pub fn fit(&mut self, data: &Array2<f32>) -> IoResult<()> {
        let (n_samples, n_features) = (data.nrows(), data.ncols());
        if n_samples == 0 {
            return Err(IoError::SignalError("fit: data has zero rows".into()));
        }

        // Compute column-wise mean and std
        let mut mean = vec![0.0_f32; n_features];
        for row in data.rows() {
            for (j, &v) in row.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n_samples as f32;
        }

        let mut std_dev = vec![0.0_f32; n_features];
        for row in data.rows() {
            for (j, &v) in row.iter().enumerate() {
                std_dev[j] += (v - mean[j]).powi(2);
            }
        }
        for s in std_dev.iter_mut() {
            *s = (*s / n_samples as f32).sqrt().max(f32::EPSILON);
        }

        self.mean = Array1::from_vec(mean);
        self.std = Array1::from_vec(std_dev);

        // Train autoencoder if required
        if let AnomalyMethod::AutoencoderReconstruction { .. } = &self.method {
            let hidden_dim = (n_features / 2).max(2);
            let latent_dim = (n_features / 4).max(1);
            let mut ae = MiniAutoencoder::new(n_features, hidden_dim, latent_dim);
            for _ in 0..200 {
                for row in data.rows() {
                    let sample: Array1<f32> = row.to_owned();
                    ae.train_step(&sample, 1e-3);
                }
            }
            self.autoencoder = Some(ae);
        }

        Ok(())
    }

    /// Returns `true` if `sample` is detected as an anomaly.
    ///
    /// # Errors
    /// Returns an error if the detector has not been fitted or if the
    /// `AutoencoderReconstruction` method is selected but no autoencoder is present.
    pub fn detect(&self, sample: &Array1<f32>) -> IoResult<bool> {
        let score = self.score(sample)?;
        let threshold = match &self.method {
            AnomalyMethod::ZScore { threshold } => *threshold,
            AnomalyMethod::MahalanobisApprox { threshold } => *threshold,
            AnomalyMethod::AutoencoderReconstruction { threshold } => *threshold,
        };
        Ok(score > threshold)
    }

    /// Compute a continuous anomaly score for `sample`.
    ///
    /// - `ZScore`: maximum absolute z-score across features.
    /// - `MahalanobisApprox`: L2 norm of z-scores (diagonal Mahalanobis).
    /// - `AutoencoderReconstruction`: MSE between input and reconstruction.
    ///
    /// # Errors
    /// Returns an error if the autoencoder is missing for the reconstruction method.
    pub fn score(&self, sample: &Array1<f32>) -> IoResult<f32> {
        match &self.method {
            AnomalyMethod::ZScore { .. } => {
                let score = sample
                    .iter()
                    .zip(self.mean.iter())
                    .zip(self.std.iter())
                    .map(|((&x, &m), &s)| ((x - m) / s).abs())
                    .fold(f32::NEG_INFINITY, f32::max);
                Ok(score)
            }
            AnomalyMethod::MahalanobisApprox { .. } => {
                let sq_sum: f32 = sample
                    .iter()
                    .zip(self.mean.iter())
                    .zip(self.std.iter())
                    .map(|((&x, &m), &s)| ((x - m) / s).powi(2))
                    .sum();
                Ok(sq_sum.sqrt())
            }
            AnomalyMethod::AutoencoderReconstruction { .. } => {
                let ae = self.autoencoder.as_ref().ok_or_else(|| {
                    IoError::SignalError("Autoencoder not fitted; call fit() first".into())
                })?;
                let recon = ae.reconstruct(sample);
                let mse: f32 = sample
                    .iter()
                    .zip(recon.iter())
                    .map(|(&x, &r)| (x - r).powi(2))
                    .sum::<f32>()
                    / sample.len() as f32;
                Ok(mse)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. MiniAutoencoder
// ---------------------------------------------------------------------------

/// Serializable weight structure for persistence.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct AutoencoderWeights {
    enc_w1: Vec<Vec<f32>>,
    enc_b1: Vec<f32>,
    enc_w2: Vec<Vec<f32>>,
    enc_b2: Vec<f32>,
    dec_w1: Vec<Vec<f32>>,
    dec_b1: Vec<f32>,
    dec_w2: Vec<Vec<f32>>,
    dec_b2: Vec<f32>,
    input_dim: usize,
    hidden_dim: usize,
    latent_dim: usize,
}

/// Lightweight autoencoder: encoder (n → hidden → latent), decoder (latent → hidden → n).
///
/// Activation: sigmoid in encoder hidden layer, linear in bottleneck and output.
pub struct MiniAutoencoder {
    // Encoder
    enc_w1: Array2<f32>,
    enc_b1: Array1<f32>,
    enc_w2: Array2<f32>,
    enc_b2: Array1<f32>,
    // Decoder
    dec_w1: Array2<f32>,
    dec_b1: Array1<f32>,
    dec_w2: Array2<f32>,
    dec_b2: Array1<f32>,
    // Dimensions
    input_dim: usize,
    hidden_dim: usize,
    latent_dim: usize,
    // Adam states (flat)
    adam_ew1: AdamState,
    adam_eb1: AdamState,
    adam_ew2: AdamState,
    adam_eb2: AdamState,
    adam_dw1: AdamState,
    adam_db1: AdamState,
    adam_dw2: AdamState,
    adam_db2: AdamState,
}

impl MiniAutoencoder {
    /// Construct a new `MiniAutoencoder` with Xavier initialisation.
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        let enc_w1 = xavier_init(hidden_dim, input_dim);
        let enc_b1 = Array1::zeros(hidden_dim);
        let enc_w2 = xavier_init(latent_dim, hidden_dim);
        let enc_b2 = Array1::zeros(latent_dim);
        let dec_w1 = xavier_init(hidden_dim, latent_dim);
        let dec_b1 = Array1::zeros(hidden_dim);
        let dec_w2 = xavier_init(input_dim, hidden_dim);
        let dec_b2 = Array1::zeros(input_dim);

        let adam_ew1 = AdamState::new(hidden_dim * input_dim);
        let adam_eb1 = AdamState::new(hidden_dim);
        let adam_ew2 = AdamState::new(latent_dim * hidden_dim);
        let adam_eb2 = AdamState::new(latent_dim);
        let adam_dw1 = AdamState::new(hidden_dim * latent_dim);
        let adam_db1 = AdamState::new(hidden_dim);
        let adam_dw2 = AdamState::new(input_dim * hidden_dim);
        let adam_db2 = AdamState::new(input_dim);

        Self {
            enc_w1,
            enc_b1,
            enc_w2,
            enc_b2,
            dec_w1,
            dec_b1,
            dec_w2,
            dec_b2,
            input_dim,
            hidden_dim,
            latent_dim,
            adam_ew1,
            adam_eb1,
            adam_ew2,
            adam_eb2,
            adam_dw1,
            adam_db1,
            adam_dw2,
            adam_db2,
        }
    }

    /// Encode input to latent representation.
    pub fn encode(&self, x: &Array1<f32>) -> Array1<f32> {
        let h = layer_forward(x, &self.enc_w1, &self.enc_b1).mapv(sigmoid);
        layer_forward(&h, &self.enc_w2, &self.enc_b2)
    }

    /// Decode latent vector to reconstruction.
    pub fn decode(&self, z: &Array1<f32>) -> Array1<f32> {
        let h = layer_forward(z, &self.dec_w1, &self.dec_b1).mapv(sigmoid);
        layer_forward(&h, &self.dec_w2, &self.dec_b2)
    }

    /// Reconstruct input through encode → decode.
    pub fn reconstruct(&self, x: &Array1<f32>) -> Array1<f32> {
        let z = self.encode(x);
        self.decode(&z)
    }

    /// Perform one gradient step (Adam) and return the reconstruction MSE loss.
    pub fn train_step(&mut self, x: &Array1<f32>, lr: f32) -> f32 {
        // Forward
        let pre_eh = layer_forward(x, &self.enc_w1, &self.enc_b1);
        let enc_h: Array1<f32> = pre_eh.mapv(sigmoid);
        let z = layer_forward(&enc_h, &self.enc_w2, &self.enc_b2);

        let pre_dh = layer_forward(&z, &self.dec_w1, &self.dec_b1);
        let dec_h: Array1<f32> = pre_dh.mapv(sigmoid);
        let recon = layer_forward(&dec_h, &self.dec_w2, &self.dec_b2);

        // MSE loss
        let n = self.input_dim as f32;
        let loss: f32 = x
            .iter()
            .zip(recon.iter())
            .map(|(&xi, &ri)| (xi - ri).powi(2))
            .sum::<f32>()
            / n;

        // Backward: gradient of MSE w.r.t recon
        let scale = 2.0 / n;
        let d_recon: Vec<f32> = x
            .iter()
            .zip(recon.iter())
            .map(|(&xi, &ri)| scale * (ri - xi))
            .collect();

        // --- Decoder layer 2 (dec_w2, dec_b2) ---
        let mut d_dw2: Vec<f32> = vec![0.0; self.input_dim * self.hidden_dim];
        let d_db2: Vec<f32> = d_recon.clone();
        for i in 0..self.input_dim {
            for j in 0..self.hidden_dim {
                d_dw2[i * self.hidden_dim + j] = d_recon[i] * dec_h[j];
            }
        }
        // gradient w.r.t dec_h (pre-sigmoid activation)
        let mut d_dec_h = vec![0.0_f32; self.hidden_dim];
        for j in 0..self.hidden_dim {
            for (i, &d_recon_i) in d_recon.iter().enumerate() {
                d_dec_h[j] += d_recon_i * self.dec_w2[[i, j]];
            }
            d_dec_h[j] *= sigmoid_prime(dec_h[j]);
        }

        // --- Decoder layer 1 (dec_w1, dec_b1) ---
        let mut d_dw1: Vec<f32> = vec![0.0; self.hidden_dim * self.latent_dim];
        let d_db1: Vec<f32> = d_dec_h.clone();
        for i in 0..self.hidden_dim {
            for j in 0..self.latent_dim {
                d_dw1[i * self.latent_dim + j] = d_dec_h[i] * z[j];
            }
        }
        let mut d_z = vec![0.0_f32; self.latent_dim];
        for (j, d_z_j) in d_z.iter_mut().enumerate() {
            for (i, &d_dec_h_i) in d_dec_h.iter().enumerate() {
                *d_z_j += d_dec_h_i * self.dec_w1[[i, j]];
            }
        }

        // --- Encoder layer 2 (enc_w2, enc_b2) ---
        let mut d_ew2: Vec<f32> = vec![0.0; self.latent_dim * self.hidden_dim];
        let d_eb2: Vec<f32> = d_z.clone();
        for i in 0..self.latent_dim {
            for j in 0..self.hidden_dim {
                d_ew2[i * self.hidden_dim + j] = d_z[i] * enc_h[j];
            }
        }
        let mut d_enc_h = vec![0.0_f32; self.hidden_dim];
        for j in 0..self.hidden_dim {
            for (i, &d_z_i) in d_z.iter().enumerate() {
                d_enc_h[j] += d_z_i * self.enc_w2[[i, j]];
            }
            d_enc_h[j] *= sigmoid_prime(enc_h[j]);
        }

        // --- Encoder layer 1 (enc_w1, enc_b1) ---
        let mut d_ew1: Vec<f32> = vec![0.0; self.hidden_dim * self.input_dim];
        let d_eb1: Vec<f32> = d_enc_h.clone();
        for i in 0..self.hidden_dim {
            for j in 0..self.input_dim {
                d_ew1[i * self.input_dim + j] = d_enc_h[i] * x[j];
            }
        }

        // Apply Adam updates — ignore errors (contiguity guaranteed by construction)
        if let Some(flat) = self.enc_w1.as_slice_mut() {
            self.adam_ew1.update(flat, &d_ew1, lr);
        }
        if let Some(flat) = self.enc_b1.as_slice_mut() {
            self.adam_eb1.update(flat, &d_eb1, lr);
        }
        if let Some(flat) = self.enc_w2.as_slice_mut() {
            self.adam_ew2.update(flat, &d_ew2, lr);
        }
        if let Some(flat) = self.enc_b2.as_slice_mut() {
            self.adam_eb2.update(flat, &d_eb2, lr);
        }
        if let Some(flat) = self.dec_w1.as_slice_mut() {
            self.adam_dw1.update(flat, &d_dw1, lr);
        }
        if let Some(flat) = self.dec_b1.as_slice_mut() {
            self.adam_db1.update(flat, &d_db1, lr);
        }
        if let Some(flat) = self.dec_w2.as_slice_mut() {
            self.adam_dw2.update(flat, &d_dw2, lr);
        }
        if let Some(flat) = self.dec_b2.as_slice_mut() {
            self.adam_db2.update(flat, &d_db2, lr);
        }

        loss
    }

    /// Serialise weights to JSON at `path`.
    ///
    /// # Errors
    /// Returns an error if file I/O or JSON serialisation fails.
    pub fn save_weights(&self, path: &std::path::Path) -> IoResult<()> {
        let to_nested = |arr: &Array2<f32>| -> Vec<Vec<f32>> {
            arr.rows().into_iter().map(|row| row.to_vec()).collect()
        };
        let weights = AutoencoderWeights {
            enc_w1: to_nested(&self.enc_w1),
            enc_b1: self.enc_b1.to_vec(),
            enc_w2: to_nested(&self.enc_w2),
            enc_b2: self.enc_b2.to_vec(),
            dec_w1: to_nested(&self.dec_w1),
            dec_b1: self.dec_b1.to_vec(),
            dec_w2: to_nested(&self.dec_w2),
            dec_b2: self.dec_b2.to_vec(),
            input_dim: self.input_dim,
            hidden_dim: self.hidden_dim,
            latent_dim: self.latent_dim,
        };
        let json = serde_json::to_string(&weights)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Deserialise weights from JSON at `path`.
    ///
    /// # Errors
    /// Returns an error if file I/O, JSON parsing, or shape reconstruction fails.
    pub fn load_weights(path: &std::path::Path) -> IoResult<Self> {
        let data = std::fs::read_to_string(path)?;
        let w: AutoencoderWeights = serde_json::from_str(&data)?;

        let from_nested =
            |nested: Vec<Vec<f32>>, rows: usize, cols: usize| -> IoResult<Array2<f32>> {
                let flat: Vec<f32> = nested.into_iter().flatten().collect();
                Array2::from_shape_vec((rows, cols), flat)
                    .map_err(|e| IoError::SignalError(format!("Weight shape error: {e}")))
            };

        let mut ae = MiniAutoencoder::new(w.input_dim, w.hidden_dim, w.latent_dim);
        ae.enc_w1 = from_nested(w.enc_w1, w.hidden_dim, w.input_dim)?;
        ae.enc_b1 = Array1::from_vec(w.enc_b1);
        ae.enc_w2 = from_nested(w.enc_w2, w.latent_dim, w.hidden_dim)?;
        ae.enc_b2 = Array1::from_vec(w.enc_b2);
        ae.dec_w1 = from_nested(w.dec_w1, w.hidden_dim, w.latent_dim)?;
        ae.dec_b1 = Array1::from_vec(w.dec_b1);
        ae.dec_w2 = from_nested(w.dec_w2, w.input_dim, w.hidden_dim)?;
        ae.dec_b2 = Array1::from_vec(w.dec_b2);
        Ok(ae)
    }

    /// Copy encoder weights (`enc_w1`, `enc_b1`, `enc_w2`, `enc_b2`) to `target`.
    ///
    /// The target must have the same `hidden_dim` and `latent_dim` as `self`.
    pub fn transfer_encoder(&self, target: &mut MiniAutoencoder) {
        target.enc_w1 = self.enc_w1.clone();
        target.enc_b1 = self.enc_b1.clone();
        target.enc_w2 = self.enc_w2.clone();
        target.enc_b2 = self.enc_b2.clone();
    }

    /// Input dimension accessor
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Hidden dimension accessor
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Latent dimension accessor
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }
}

// Suppress unused import warning for ArrayView1 — it is part of the public API surface
// that callers may use alongside these types.
#[allow(dead_code)]
fn _use_array_view1(_v: ArrayView1<f32>) {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: generate a sine wave of length `n` at normalised frequency `freq`.
    fn sine_wave(n: usize, freq: f32) -> Array1<f32> {
        Array1::from_shape_fn(n, |i| (2.0 * PI * freq * i as f32).sin())
    }

    /// Helper: add Gaussian noise with given standard deviation.
    fn add_noise(signal: &Array1<f32>, std_dev: f32) -> Array1<f32> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0_f64, std_dev as f64).expect("Normal init");
        signal.mapv(|x| x + normal.sample(&mut rng) as f32)
    }

    // -----------------------------------------------------------------------
    // Test 1: SNR improves after 50 training steps
    // -----------------------------------------------------------------------
    #[test]
    fn test_denoiser_snr_improvement() {
        let n = 64;
        let clean = sine_wave(n, 0.05);
        let noisy = add_noise(&clean, 0.5);

        let initial_snr = SignalDenoiser::snr_db(&clean, &noisy);

        let mut denoiser = SignalDenoiser::new(n, 32);
        let mut last_output = noisy.clone();
        for _ in 0..50 {
            last_output = denoiser
                .denoise(&noisy, Some(&clean))
                .expect("denoise failed");
        }
        let final_snr = SignalDenoiser::snr_db(&clean, &last_output);
        assert!(
            final_snr > initial_snr,
            "SNR should improve: initial={initial_snr:.2} dB, final={final_snr:.2} dB"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: Denoiser output has same shape as input
    // -----------------------------------------------------------------------
    #[test]
    fn test_denoiser_forward_shape() {
        let n = 128;
        let noisy = add_noise(&sine_wave(n, 0.1), 0.3);
        let mut denoiser = SignalDenoiser::new(n, 64);
        let output = denoiser.denoise(&noisy, None).expect("denoise failed");
        assert_eq!(output.len(), n, "Output length must match input length");
    }

    // -----------------------------------------------------------------------
    // Test 3: MFCC output has correct shape
    // -----------------------------------------------------------------------
    #[test]
    fn test_feature_extractor_mfcc_shape() {
        let sample_rate: u32 = 16000;
        let n_fft = 512;
        let hop_length = 256;
        let n_mfcc = 13;
        let fe = FeatureExtractor::new(sample_rate, n_mfcc, n_fft, hop_length)
            .expect("FeatureExtractor::new failed");

        // 1 second of audio
        let signal = sine_wave(sample_rate as usize, 440.0 / sample_rate as f32);
        let mfccs = fe.extract_mfcc(&signal).expect("extract_mfcc failed");

        let expected_frames = (signal.len() - n_fft) / hop_length + 1;
        assert_eq!(mfccs.nrows(), expected_frames, "Wrong number of frames");
        assert_eq!(mfccs.ncols(), n_mfcc, "Wrong number of MFCC coefficients");
    }

    // -----------------------------------------------------------------------
    // Test 4: Spectral feature extraction dimensions
    // -----------------------------------------------------------------------
    #[test]
    fn test_feature_extractor_spectral_features() {
        let sample_rate: u32 = 8000;
        let n_fft = 256;
        let hop_length = 128;
        let fe = FeatureExtractor::new(sample_rate, 13, n_fft, hop_length)
            .expect("FeatureExtractor::new failed");

        let signal = sine_wave(sample_rate as usize, 440.0 / sample_rate as f32);
        let features = fe
            .extract_spectral_features(&signal)
            .expect("extract_spectral_features failed");

        let expected_frames = (signal.len() - n_fft) / hop_length + 1;
        assert_eq!(features.nrows(), expected_frames, "Wrong frame count");
        assert_eq!(
            features.ncols(),
            4,
            "Expected 4 spectral features per frame"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: Streaming extraction — at least one Some result from 3 chunks
    // -----------------------------------------------------------------------
    #[test]
    fn test_streaming_extraction() {
        let sample_rate: u32 = 8000;
        let n_fft = 256;
        let hop_length = 128;
        let mut fe = FeatureExtractor::new(sample_rate, 13, n_fft, hop_length)
            .expect("FeatureExtractor::new failed");

        let chunk_size = 128;
        let chunk: Vec<f32> = (0..chunk_size)
            .map(|i| (2.0 * PI * 0.05 * i as f32).sin())
            .collect();

        let mut got_some = false;
        for _ in 0..3 {
            let result = fe.extract_streaming(&chunk).expect("streaming failed");
            if result.is_some() {
                got_some = true;
            }
        }
        assert!(
            got_some,
            "At least one streaming chunk should produce features"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: Z-score anomaly detector flags outlier
    // -----------------------------------------------------------------------
    #[test]
    fn test_anomaly_zscore() {
        let n_features = 8;
        let n_samples = 50;
        let mut rng = thread_rng();
        let normal_dist = Normal::new(0.0_f64, 1.0_f64).expect("Normal init");

        // Normal training data around 0
        let data_vec: Vec<f32> = (0..n_samples * n_features)
            .map(|_| normal_dist.sample(&mut rng) as f32)
            .collect();
        let data = Array2::from_shape_vec((n_samples, n_features), data_vec).expect("shape");

        let mut detector = AnomalyDetector::new(AnomalyMethod::ZScore { threshold: 3.0 });
        detector.fit(&data).expect("fit failed");

        // Normal sample — should not be anomaly (usually)
        let normal_sample = Array1::zeros(n_features);
        let normal_result = detector.detect(&normal_sample).expect("detect failed");
        assert!(!normal_result, "Zero vector should not be an anomaly");

        // Extreme outlier
        let outlier = Array1::from_elem(n_features, 100.0_f32);
        let outlier_result = detector.detect(&outlier).expect("detect failed");
        assert!(outlier_result, "Extreme outlier must be detected");
    }

    // -----------------------------------------------------------------------
    // Test 7: Normal sample score < outlier score
    // -----------------------------------------------------------------------
    #[test]
    fn test_anomaly_score_ordering() {
        let n_features = 6;
        let n_samples = 40;
        let mut rng = thread_rng();
        let normal_dist = Normal::new(0.0_f64, 0.5_f64).expect("Normal init");

        let data_vec: Vec<f32> = (0..n_samples * n_features)
            .map(|_| normal_dist.sample(&mut rng) as f32)
            .collect();
        let data = Array2::from_shape_vec((n_samples, n_features), data_vec).expect("shape");

        let mut detector =
            AnomalyDetector::new(AnomalyMethod::MahalanobisApprox { threshold: 5.0 });
        detector.fit(&data).expect("fit failed");

        let normal_sample = Array1::zeros(n_features);
        let outlier = Array1::from_elem(n_features, 50.0_f32);

        let normal_score = detector.score(&normal_sample).expect("score failed");
        let outlier_score = detector.score(&outlier).expect("score failed");

        assert!(
            normal_score < outlier_score,
            "Normal score ({normal_score}) must be less than outlier score ({outlier_score})"
        );
    }

    // -----------------------------------------------------------------------
    // Test 8: Autoencoder reconstruction loss < 1.0 after 100 training steps
    // -----------------------------------------------------------------------
    #[test]
    fn test_autoencoder_roundtrip() {
        let input_dim = 16;
        let hidden_dim = 8;
        let latent_dim = 4;
        let mut ae = MiniAutoencoder::new(input_dim, hidden_dim, latent_dim);

        // Simple constant input
        let x = Array1::from_elem(input_dim, 0.5_f32);
        let mut loss = f32::INFINITY;
        for _ in 0..100 {
            loss = ae.train_step(&x, 1e-2);
        }
        assert!(
            loss < 1.0,
            "Reconstruction loss should be < 1.0 after 100 steps, got {loss}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: Save and load weights — reconstructions match
    // -----------------------------------------------------------------------
    #[test]
    fn test_autoencoder_save_load() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static AE_SAVE_LOAD_COUNTER: AtomicU64 = AtomicU64::new(0);
        let uid = AE_SAVE_LOAD_COUNTER.fetch_add(1, Ordering::Relaxed);

        let input_dim = 12;
        let hidden_dim = 6;
        let latent_dim = 3;
        let mut ae = MiniAutoencoder::new(input_dim, hidden_dim, latent_dim);

        let x = Array1::from_shape_fn(input_dim, |i| i as f32 / input_dim as f32);
        // Brief training
        for _ in 0..20 {
            ae.train_step(&x, 1e-3);
        }

        let recon_before = ae.reconstruct(&x);

        let mut tmp = std::env::temp_dir();
        tmp.push(format!("kizzasi_ae_test_weights_{}.json", uid));
        ae.save_weights(&tmp).expect("save_weights failed");

        let ae_loaded = MiniAutoencoder::load_weights(&tmp).expect("load_weights failed");
        let recon_after = ae_loaded.reconstruct(&x);

        // Reconstructions must be numerically identical after save/load
        for (a, b) in recon_before.iter().zip(recon_after.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Reconstruction mismatch after save/load: {a} vs {b}"
            );
        }

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    // -----------------------------------------------------------------------
    // Test 10: Encoder transfer copies weights correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_transfer_encoder() {
        let input_dim = 10;
        let hidden_dim = 5;
        let latent_dim = 2;

        let mut src = MiniAutoencoder::new(input_dim, hidden_dim, latent_dim);
        let x = Array1::from_shape_fn(input_dim, |i| i as f32);
        // Train source briefly so weights differ from zero init
        for _ in 0..10 {
            src.train_step(&x, 1e-3);
        }

        let mut tgt = MiniAutoencoder::new(input_dim, hidden_dim, latent_dim);
        src.transfer_encoder(&mut tgt);

        // Encoder outputs must be identical
        let enc_src = src.encode(&x);
        let enc_tgt = tgt.encode(&x);
        for (a, b) in enc_src.iter().zip(enc_tgt.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Encoder output mismatch after transfer: {a} vs {b}"
            );
        }
    }
}
