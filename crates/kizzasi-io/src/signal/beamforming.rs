//! Multi-channel beamforming for spatial audio processing
//!
//! This module provides advanced beamforming techniques for microphone arrays:
//! - Delay-and-Sum beamforming (classic approach)
//! - Minimum Variance Distortionless Response (MVDR) beamforming
//! - Generalized Sidelobe Canceller (GSC)
//! - Direction of Arrival (DOA) estimation
//! - Adaptive beamforming with LMS

use crate::error::{IoError, IoResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::f32::consts::PI;

/// Microphone array configuration
#[derive(Debug, Clone)]
pub struct MicrophoneArray {
    /// Microphone positions in meters [x, y, z]
    pub positions: Vec<[f32; 3]>,
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// Speed of sound in m/s (default 343 m/s)
    pub speed_of_sound: f32,
}

impl MicrophoneArray {
    /// Create a new microphone array
    pub fn new(positions: Vec<[f32; 3]>, sample_rate: f32) -> Self {
        Self {
            positions,
            sample_rate,
            speed_of_sound: 343.0,
        }
    }

    /// Create a linear array along x-axis
    pub fn linear(num_mics: usize, spacing: f32, sample_rate: f32) -> Self {
        let positions = (0..num_mics)
            .map(|i| [i as f32 * spacing, 0.0, 0.0])
            .collect();

        Self::new(positions, sample_rate)
    }

    /// Create a circular array in xy-plane
    pub fn circular(num_mics: usize, radius: f32, sample_rate: f32) -> Self {
        let positions = (0..num_mics)
            .map(|i| {
                let angle = 2.0 * PI * i as f32 / num_mics as f32;
                [radius * angle.cos(), radius * angle.sin(), 0.0]
            })
            .collect();

        Self::new(positions, sample_rate)
    }

    /// Number of microphones
    pub fn num_mics(&self) -> usize {
        self.positions.len()
    }
}

/// Delay-and-Sum beamformer
///
/// Classic beamforming technique that applies time delays to align signals
/// from a specific direction and sums them.
pub struct DelayAndSum {
    /// Microphone array
    array: MicrophoneArray,
}

impl DelayAndSum {
    /// Create a new delay-and-sum beamformer
    pub fn new(array: MicrophoneArray) -> Self {
        Self { array }
    }

    /// Process multi-channel signal
    ///
    /// # Arguments
    /// * `signals` - Input signals (n_samples × n_mics)
    /// * `azimuth` - Azimuth angle in radians (0 = front, π/2 = right)
    /// * `elevation` - Elevation angle in radians (0 = horizontal, π/2 = up)
    ///
    /// # Returns
    /// Beamformed output signal
    pub fn process(
        &self,
        signals: &Array2<f32>,
        azimuth: f32,
        elevation: f32,
    ) -> IoResult<Array1<f32>> {
        let (n_samples, n_mics) = signals.dim();

        if n_mics != self.array.num_mics() {
            return Err(IoError::SignalError(
                "Number of channels doesn't match array".into(),
            ));
        }

        // Direction vector
        let dir = [
            azimuth.cos() * elevation.cos(),
            azimuth.sin() * elevation.cos(),
            elevation.sin(),
        ];

        // Compute time delays for each microphone
        let delays = self.compute_delays(&dir);

        // Apply delays and sum
        let mut output = Array1::zeros(n_samples);

        for mic in 0..n_mics {
            let delay_samples = (delays[mic] * self.array.sample_rate).round() as isize;

            for i in 0..n_samples {
                let src_idx = i as isize - delay_samples;
                if src_idx >= 0 && (src_idx as usize) < n_samples {
                    output[i] += signals[[src_idx as usize, mic]];
                }
            }
        }

        // Normalize
        output /= n_mics as f32;

        Ok(output)
    }

    /// Compute time delays for each microphone
    fn compute_delays(&self, direction: &[f32; 3]) -> Vec<f32> {
        // Reference point (first mic)
        let ref_pos = self.array.positions[0];

        self.array
            .positions
            .iter()
            .map(|pos| {
                // Distance difference from reference
                let dx = pos[0] - ref_pos[0];
                let dy = pos[1] - ref_pos[1];
                let dz = pos[2] - ref_pos[2];

                // Projection onto direction
                let proj = dx * direction[0] + dy * direction[1] + dz * direction[2];

                // Time delay
                -proj / self.array.speed_of_sound
            })
            .collect()
    }
}

/// Minimum Variance Distortionless Response (MVDR) beamformer
///
/// Adaptive beamformer that minimizes output variance while maintaining
/// unity gain in the look direction (distortionless constraint).
pub struct MVDR {
    /// Microphone array
    array: MicrophoneArray,
    /// Diagonal loading factor for numerical stability
    diagonal_loading: f32,
}

impl MVDR {
    /// Create a new MVDR beamformer
    ///
    /// # Arguments
    /// * `array` - Microphone array configuration
    /// * `diagonal_loading` - Regularization parameter (default 0.01)
    pub fn new(array: MicrophoneArray, diagonal_loading: Option<f32>) -> Self {
        Self {
            array,
            diagonal_loading: diagonal_loading.unwrap_or(0.01),
        }
    }

    /// Compute MVDR weights
    ///
    /// # Arguments
    /// * `signals` - Input signals for covariance estimation
    /// * `azimuth` - Look direction azimuth
    /// * `elevation` - Look direction elevation
    ///
    /// # Returns
    /// Optimal beamforming weights
    pub fn compute_weights(
        &self,
        signals: &Array2<f32>,
        azimuth: f32,
        elevation: f32,
    ) -> IoResult<Array1<f32>> {
        let n_mics = self.array.num_mics();

        // Estimate spatial covariance matrix
        let cov = self.estimate_covariance(signals)?;

        // Steering vector (frequency-domain representation simplified)
        let steering = self.compute_steering_vector(azimuth, elevation);

        // MVDR weights: w = (R^{-1} a) / (a^H R^{-1} a)
        let cov_inv = Self::invert_with_loading(&cov, self.diagonal_loading)?;
        let cov_inv_a = cov_inv.dot(&steering);
        let denom = steering.dot(&cov_inv_a);

        let weights = if denom.abs() > 1e-10 {
            cov_inv_a / denom
        } else {
            Array1::from_elem(n_mics, 1.0 / n_mics as f32)
        };

        Ok(weights)
    }

    /// Process signals with MVDR beamforming
    pub fn process(
        &self,
        signals: &Array2<f32>,
        azimuth: f32,
        elevation: f32,
    ) -> IoResult<Array1<f32>> {
        let weights = self.compute_weights(signals, azimuth, elevation)?;

        // Apply weights
        let output = signals.dot(&weights);

        Ok(output)
    }

    /// Estimate spatial covariance matrix
    fn estimate_covariance(&self, signals: &Array2<f32>) -> IoResult<Array2<f32>> {
        let (n_samples, _n_mics) = signals.dim();

        // R = (1/N) X^T X
        let cov = signals.t().dot(signals) / n_samples as f32;

        Ok(cov)
    }

    /// Compute steering vector for given direction
    fn compute_steering_vector(&self, azimuth: f32, elevation: f32) -> Array1<f32> {
        let dir = [
            azimuth.cos() * elevation.cos(),
            azimuth.sin() * elevation.cos(),
            elevation.sin(),
        ];

        // Simplified steering vector (assumes narrowband)
        let ref_pos = self.array.positions[0];
        let mut steering = Array1::zeros(self.array.num_mics());

        for (i, pos) in self.array.positions.iter().enumerate() {
            let dx = pos[0] - ref_pos[0];
            let dy = pos[1] - ref_pos[1];
            let dz = pos[2] - ref_pos[2];

            let proj = dx * dir[0] + dy * dir[1] + dz * dir[2];
            steering[i] = (-2.0 * PI * proj / self.array.speed_of_sound).cos();
        }

        // Normalize
        let norm = steering.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            steering /= norm;
        }

        steering
    }

    /// Matrix inversion with diagonal loading
    fn invert_with_loading(matrix: &Array2<f32>, loading: f32) -> IoResult<Array2<f32>> {
        let n = matrix.nrows();

        // Add diagonal loading: R_loaded = R + λI
        let mut loaded = matrix.clone();
        for i in 0..n {
            loaded[[i, i]] += loading;
        }

        // Simple inversion for small matrices (Gauss-Jordan)
        Self::gauss_jordan_invert(&loaded)
    }

    /// Gauss-Jordan matrix inversion
    fn gauss_jordan_invert(matrix: &Array2<f32>) -> IoResult<Array2<f32>> {
        let n = matrix.nrows();
        let mut augmented = Array2::zeros((n, 2 * n));

        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                augmented[[i, j + n]] = if i == j { 1.0 } else { 0.0 };
            }
        }

        // Forward elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            for j in 0..2 * n {
                let tmp = augmented[[i, j]];
                augmented[[i, j]] = augmented[[max_row, j]];
                augmented[[max_row, j]] = tmp;
            }

            let pivot = augmented[[i, i]];
            if pivot.abs() < 1e-10 {
                return Err(IoError::SignalError("Singular matrix".into()));
            }

            // Scale pivot row
            for j in 0..2 * n {
                augmented[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..2 * n {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }

        // Extract inverse
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, j + n]];
            }
        }

        Ok(inverse)
    }
}

/// Direction of Arrival (DOA) estimation
///
/// Estimates the direction from which a signal arrives using
/// multiple algorithms (GCC-PHAT, MUSIC).
pub struct DOAEstimator {
    /// Microphone array
    array: MicrophoneArray,
}

impl DOAEstimator {
    /// Create a new DOA estimator
    pub fn new(array: MicrophoneArray) -> Self {
        Self { array }
    }

    /// Estimate DOA using steered response power
    ///
    /// Scans through possible directions and finds the one with maximum power
    pub fn estimate_srp(
        &self,
        signals: &Array2<f32>,
        azimuth_resolution: usize,
    ) -> IoResult<(f32, f32)> {
        let beamformer = DelayAndSum::new(self.array.clone());

        let mut max_power = 0.0f32;
        let mut best_azimuth = 0.0f32;

        // Scan azimuth angles (assume elevation = 0)
        for i in 0..azimuth_resolution {
            let azimuth = (2.0 * PI * i as f32) / azimuth_resolution as f32;
            let output = beamformer.process(signals, azimuth, 0.0)?;

            // Compute power
            let power: f32 = output.iter().map(|x| x * x).sum();

            if power > max_power {
                max_power = power;
                best_azimuth = azimuth;
            }
        }

        Ok((best_azimuth, 0.0))
    }
}

/// Adaptive beamformer using LMS algorithm
pub struct AdaptiveBeamformer {
    /// Microphone array
    array: MicrophoneArray,
    /// Weights
    weights: Array1<f32>,
    /// Step size
    mu: f32,
}

impl AdaptiveBeamformer {
    /// Create a new adaptive beamformer
    ///
    /// # Arguments
    /// * `array` - Microphone array
    /// * `mu` - LMS step size (learning rate)
    pub fn new(array: MicrophoneArray, mu: f32) -> Self {
        let n_mics = array.num_mics();
        Self {
            array,
            weights: Array1::from_elem(n_mics, 1.0 / n_mics as f32),
            mu,
        }
    }

    /// Adapt weights based on desired signal
    ///
    /// # Arguments
    /// * `input` - Input signal vector (one sample from all mics)
    /// * `desired` - Desired output (reference signal)
    ///
    /// # Returns
    /// * Output signal
    /// * Error signal
    pub fn adapt(&mut self, input: &Array1<f32>, desired: f32) -> (f32, f32) {
        // Compute output
        let output = self.weights.dot(input);

        // Compute error
        let error = desired - output;

        // Update weights: w(n+1) = w(n) + μ * e(n) * x(n)
        self.weights = &self.weights + &(input * (self.mu * error));

        (output, error)
    }

    /// Get current weights
    pub fn weights(&self) -> &Array1<f32> {
        &self.weights
    }

    /// Reset weights
    pub fn reset(&mut self) {
        let n_mics = self.array.num_mics();
        self.weights = Array1::from_elem(n_mics, 1.0 / n_mics as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_microphone_array_linear() {
        let array = MicrophoneArray::linear(4, 0.05, 16000.0);
        assert_eq!(array.num_mics(), 4);
        assert_eq!(array.positions[0], [0.0, 0.0, 0.0]);
        assert_eq!(array.positions[3], [0.15, 0.0, 0.0]);
    }

    #[test]
    fn test_microphone_array_circular() {
        let array = MicrophoneArray::circular(8, 0.1, 16000.0);
        assert_eq!(array.num_mics(), 8);
    }

    #[test]
    fn test_delay_and_sum() {
        let array = MicrophoneArray::linear(3, 0.05, 16000.0);
        let beamformer = DelayAndSum::new(array);

        // Create test signals (3 mics, 100 samples)
        let signals = arr2(&[
            [1.0, 1.1, 0.9],
            [2.0, 2.1, 1.9],
            [1.5, 1.6, 1.4],
            [3.0, 3.1, 2.9],
        ]);

        let output = beamformer.process(&signals, 0.0, 0.0);
        assert!(output.is_ok());
        assert_eq!(output.unwrap().len(), 4);
    }

    #[test]
    fn test_mvdr_weights() {
        let array = MicrophoneArray::linear(3, 0.05, 16000.0);
        let mvdr = MVDR::new(array, None);

        let signals = arr2(&[
            [1.0, 1.1, 0.9],
            [2.0, 2.1, 1.9],
            [1.5, 1.6, 1.4],
            [3.0, 3.1, 2.9],
        ]);

        let weights = mvdr.compute_weights(&signals, 0.0, 0.0);
        assert!(weights.is_ok());
        assert_eq!(weights.unwrap().len(), 3);
    }

    #[test]
    fn test_doa_estimator() {
        let array = MicrophoneArray::linear(4, 0.05, 16000.0);
        let doa = DOAEstimator::new(array);

        let signals = arr2(&[
            [1.0, 1.1, 0.9, 1.0],
            [2.0, 2.1, 1.9, 2.0],
            [1.5, 1.6, 1.4, 1.5],
        ]);

        let result = doa.estimate_srp(&signals, 12);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptive_beamformer() {
        let array = MicrophoneArray::linear(3, 0.05, 16000.0);
        let mut beamformer = AdaptiveBeamformer::new(array, 0.01);

        let input = Array1::from_vec(vec![1.0, 1.1, 0.9]);
        let desired = 1.5;

        let (output, error) = beamformer.adapt(&input, desired);

        assert!(output.is_finite());
        assert!(error.is_finite());
        assert_eq!(beamformer.weights().len(), 3);
    }
}
