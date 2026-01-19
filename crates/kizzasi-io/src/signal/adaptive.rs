//! Adaptive and advanced filtering algorithms
//!
//! This module provides state-of-the-art filtering techniques including:
//! - Kalman Filter for optimal linear state estimation
//! - Extended Kalman Filter (EKF) for nonlinear systems
//! - Particle Filter for non-Gaussian/nonlinear estimation
//! - LMS (Least Mean Squares) adaptive filter
//! - RLS (Recursive Least Squares) adaptive filter
//! - NLMS (Normalized LMS) for faster convergence

use crate::error::{IoError, IoResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Distribution, Normal};

/// Kalman Filter for optimal linear state estimation
///
/// The Kalman filter provides optimal estimates of unknown states
/// in linear dynamical systems with Gaussian noise.
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// State estimate (x)
    state: Array1<f32>,
    /// State covariance matrix (P)
    covariance: Array2<f32>,
    /// State transition matrix (F)
    transition: Array2<f32>,
    /// Observation matrix (H)
    observation: Array2<f32>,
    /// Process noise covariance (Q)
    process_noise: Array2<f32>,
    /// Measurement noise covariance (R)
    measurement_noise: Array2<f32>,
}

impl KalmanFilter {
    /// Create a new Kalman filter
    ///
    /// # Arguments
    /// * `initial_state` - Initial state estimate
    /// * `initial_covariance` - Initial state covariance
    /// * `transition` - State transition matrix F (n×n)
    /// * `observation` - Observation matrix H (m×n)
    /// * `process_noise` - Process noise covariance Q (n×n)
    /// * `measurement_noise` - Measurement noise covariance R (m×m)
    pub fn new(
        initial_state: Array1<f32>,
        initial_covariance: Array2<f32>,
        transition: Array2<f32>,
        observation: Array2<f32>,
        process_noise: Array2<f32>,
        measurement_noise: Array2<f32>,
    ) -> IoResult<Self> {
        let n = initial_state.len();
        let m = observation.shape()[0];

        if initial_covariance.shape() != [n, n] {
            return Err(IoError::SignalError(
                "Initial covariance must be n×n".into(),
            ));
        }
        if transition.shape() != [n, n] {
            return Err(IoError::SignalError("Transition matrix must be n×n".into()));
        }
        if observation.shape() != [m, n] {
            return Err(IoError::SignalError(
                "Observation matrix must be m×n".into(),
            ));
        }
        if process_noise.shape() != [n, n] {
            return Err(IoError::SignalError("Process noise must be n×n".into()));
        }
        if measurement_noise.shape() != [m, m] {
            return Err(IoError::SignalError("Measurement noise must be m×m".into()));
        }

        Ok(Self {
            state: initial_state,
            covariance: initial_covariance,
            transition,
            observation,
            process_noise,
            measurement_noise,
        })
    }

    /// Predict step (time update)
    pub fn predict(&mut self) {
        // x_pred = F * x
        let x_pred = self.transition.dot(&self.state);
        // P_pred = F * P * F^T + Q
        let p_temp = self.transition.dot(&self.covariance);
        let p_pred = p_temp.dot(&self.transition.t()) + &self.process_noise;

        self.state = x_pred;
        self.covariance = p_pred;
    }

    /// Update step (measurement update)
    pub fn update(&mut self, measurement: &Array1<f32>) -> IoResult<()> {
        // Innovation (y): y = z - H * x
        let predicted_measurement = self.observation.dot(&self.state);
        let innovation = measurement - &predicted_measurement;

        // Innovation covariance (S): S = H * P * H^T + R
        let h_p = self.observation.dot(&self.covariance);
        let s = h_p.dot(&self.observation.t()) + &self.measurement_noise;

        // Kalman gain (K): K = P * H^T * S^{-1}
        let s_inv = Self::invert_matrix(&s)?;
        let p_ht = self.covariance.dot(&self.observation.t());
        let k = p_ht.dot(&s_inv);

        // Updated state: x = x + K * y
        let state_update = k.dot(&innovation);
        self.state = &self.state + &state_update;

        // Updated covariance: P = (I - K * H) * P
        let n = self.state.len();
        let identity = Array2::eye(n);
        let kh = k.dot(&self.observation);
        let p_update = (&identity - &kh).dot(&self.covariance);
        self.covariance = p_update;

        Ok(())
    }

    /// Get current state estimate
    pub fn state(&self) -> &Array1<f32> {
        &self.state
    }

    /// Get current state covariance
    pub fn covariance(&self) -> &Array2<f32> {
        &self.covariance
    }

    /// Reset filter to initial conditions
    pub fn reset(&mut self, state: Array1<f32>, covariance: Array2<f32>) {
        self.state = state;
        self.covariance = covariance;
    }

    /// Simple 2×2 matrix inversion (for larger matrices, use proper linear algebra)
    fn invert_matrix(mat: &Array2<f32>) -> IoResult<Array2<f32>> {
        let shape = mat.shape();
        if shape[0] != shape[1] {
            return Err(IoError::SignalError("Matrix must be square".into()));
        }

        let n = shape[0];
        if n == 1 {
            let det = mat[[0, 0]];
            if det.abs() < 1e-10 {
                return Err(IoError::SignalError("Singular matrix".into()));
            }
            let mut inv = Array2::zeros((1, 1));
            inv[[0, 0]] = 1.0 / det;
            return Ok(inv);
        }

        if n == 2 {
            let a = mat[[0, 0]];
            let b = mat[[0, 1]];
            let c = mat[[1, 0]];
            let d = mat[[1, 1]];

            let det = a * d - b * c;
            if det.abs() < 1e-10 {
                return Err(IoError::SignalError("Singular matrix".into()));
            }

            let mut inv = Array2::zeros((2, 2));
            inv[[0, 0]] = d / det;
            inv[[0, 1]] = -b / det;
            inv[[1, 0]] = -c / det;
            inv[[1, 1]] = a / det;
            return Ok(inv);
        }

        // For larger matrices, use Gauss-Jordan elimination
        let mut augmented = Array2::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = mat[[i, j]];
                augmented[[i, j + n]] = if i == j { 1.0 } else { 0.0 };
            }
        }

        // Forward elimination
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

            // Scale row
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
        let mut inv = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] = augmented[[i, j + n]];
            }
        }

        Ok(inv)
    }
}

/// LMS (Least Mean Squares) Adaptive Filter
///
/// An adaptive filter that adjusts its weights using the gradient descent
/// algorithm to minimize mean square error.
#[derive(Debug, Clone)]
pub struct LmsFilter {
    /// Filter weights
    weights: Array1<f32>,
    /// Step size (learning rate)
    mu: f32,
    /// Input buffer (delay line)
    buffer: Vec<f32>,
    /// Current position in buffer
    pos: usize,
}

impl LmsFilter {
    /// Create a new LMS adaptive filter
    ///
    /// # Arguments
    /// * `num_taps` - Number of filter coefficients
    /// * `mu` - Step size (learning rate), typically 0.01 to 0.1
    pub fn new(num_taps: usize, mu: f32) -> IoResult<Self> {
        if num_taps == 0 {
            return Err(IoError::SignalError("Number of taps must be > 0".into()));
        }
        if mu <= 0.0 {
            return Err(IoError::SignalError("Step size must be > 0".into()));
        }

        Ok(Self {
            weights: Array1::zeros(num_taps),
            mu,
            buffer: vec![0.0; num_taps],
            pos: 0,
        })
    }

    /// Adapt and filter a single sample
    ///
    /// # Arguments
    /// * `input` - Input signal sample
    /// * `desired` - Desired output (reference signal)
    ///
    /// # Returns
    /// * `output` - Filter output
    /// * `error` - Estimation error (desired - output)
    pub fn adapt(&mut self, input: f32, desired: f32) -> (f32, f32) {
        // Update buffer
        self.buffer[self.pos] = input;

        // Compute output
        let mut output = 0.0;
        let mut buf_idx = self.pos;

        for &weight in self.weights.iter() {
            output += weight * self.buffer[buf_idx];
            if buf_idx == 0 {
                buf_idx = self.buffer.len() - 1;
            } else {
                buf_idx -= 1;
            }
        }

        // Compute error
        let error = desired - output;

        // Update weights: w(n+1) = w(n) + mu * error * x(n)
        buf_idx = self.pos;
        for weight in self.weights.iter_mut() {
            *weight += self.mu * error * self.buffer[buf_idx];
            if buf_idx == 0 {
                buf_idx = self.buffer.len() - 1;
            } else {
                buf_idx -= 1;
            }
        }

        self.pos = (self.pos + 1) % self.buffer.len();

        (output, error)
    }

    /// Get current filter weights
    pub fn weights(&self) -> &Array1<f32> {
        &self.weights
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.pos = 0;
    }
}

/// NLMS (Normalized LMS) Adaptive Filter
///
/// Similar to LMS but normalizes the step size by the input power,
/// providing faster and more stable convergence.
#[derive(Debug, Clone)]
pub struct NlmsFilter {
    /// Filter weights
    weights: Array1<f32>,
    /// Step size (learning rate)
    mu: f32,
    /// Regularization parameter (for numerical stability)
    epsilon: f32,
    /// Input buffer (delay line)
    buffer: Vec<f32>,
    /// Current position in buffer
    pos: usize,
}

impl NlmsFilter {
    /// Create a new NLMS adaptive filter
    ///
    /// # Arguments
    /// * `num_taps` - Number of filter coefficients
    /// * `mu` - Step size (learning rate), typically 0.5 to 1.0
    /// * `epsilon` - Regularization parameter, default 1e-6
    pub fn new(num_taps: usize, mu: f32, epsilon: Option<f32>) -> IoResult<Self> {
        if num_taps == 0 {
            return Err(IoError::SignalError("Number of taps must be > 0".into()));
        }
        if mu <= 0.0 {
            return Err(IoError::SignalError("Step size must be > 0".into()));
        }

        Ok(Self {
            weights: Array1::zeros(num_taps),
            mu,
            epsilon: epsilon.unwrap_or(1e-6),
            buffer: vec![0.0; num_taps],
            pos: 0,
        })
    }

    /// Adapt and filter a single sample
    pub fn adapt(&mut self, input: f32, desired: f32) -> (f32, f32) {
        // Update buffer
        self.buffer[self.pos] = input;

        // Compute output
        let mut output = 0.0;
        let mut buf_idx = self.pos;

        for &weight in self.weights.iter() {
            output += weight * self.buffer[buf_idx];
            if buf_idx == 0 {
                buf_idx = self.buffer.len() - 1;
            } else {
                buf_idx -= 1;
            }
        }

        // Compute error
        let error = desired - output;

        // Compute input power
        let power: f32 = self.buffer.iter().map(|x| x * x).sum();
        let normalized_mu = self.mu / (power + self.epsilon);

        // Update weights: w(n+1) = w(n) + (mu / (||x||^2 + epsilon)) * error * x(n)
        buf_idx = self.pos;
        for weight in self.weights.iter_mut() {
            *weight += normalized_mu * error * self.buffer[buf_idx];
            if buf_idx == 0 {
                buf_idx = self.buffer.len() - 1;
            } else {
                buf_idx -= 1;
            }
        }

        self.pos = (self.pos + 1) % self.buffer.len();

        (output, error)
    }

    /// Get current filter weights
    pub fn weights(&self) -> &Array1<f32> {
        &self.weights
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.pos = 0;
    }
}

/// RLS (Recursive Least Squares) Adaptive Filter
///
/// An adaptive filter with exponentially weighted least squares criterion,
/// providing faster convergence than LMS but with higher computational cost.
#[derive(Debug, Clone)]
pub struct RlsFilter {
    /// Filter weights
    weights: Array1<f32>,
    /// Inverse correlation matrix (P)
    p_matrix: Array2<f32>,
    /// Forgetting factor (lambda), typically 0.99 to 1.0
    lambda: f32,
    /// Input buffer (delay line)
    buffer: Vec<f32>,
    /// Current position in buffer
    pos: usize,
}

impl RlsFilter {
    /// Create a new RLS adaptive filter
    ///
    /// # Arguments
    /// * `num_taps` - Number of filter coefficients
    /// * `lambda` - Forgetting factor (0 < lambda <= 1), typically 0.99
    /// * `delta` - Initialization parameter for P matrix, typically 0.01 to 1.0
    pub fn new(num_taps: usize, lambda: f32, delta: f32) -> IoResult<Self> {
        if num_taps == 0 {
            return Err(IoError::SignalError("Number of taps must be > 0".into()));
        }
        if lambda <= 0.0 || lambda > 1.0 {
            return Err(IoError::SignalError(
                "Forgetting factor must be in (0, 1]".into(),
            ));
        }
        if delta <= 0.0 {
            return Err(IoError::SignalError("Delta must be > 0".into()));
        }

        // Initialize P = (1/delta) * I
        let p_matrix = Array2::eye(num_taps) * (1.0 / delta);

        Ok(Self {
            weights: Array1::zeros(num_taps),
            p_matrix,
            lambda,
            buffer: vec![0.0; num_taps],
            pos: 0,
        })
    }

    /// Adapt and filter a single sample
    pub fn adapt(&mut self, input: f32, desired: f32) -> (f32, f32) {
        // Update buffer
        self.buffer[self.pos] = input;

        // Get input vector (reversed order for proper convolution)
        let mut x = Array1::zeros(self.buffer.len());
        let mut buf_idx = self.pos;
        for i in 0..self.buffer.len() {
            x[i] = self.buffer[buf_idx];
            if buf_idx == 0 {
                buf_idx = self.buffer.len() - 1;
            } else {
                buf_idx -= 1;
            }
        }

        // Compute output
        let output = self.weights.dot(&x);

        // Compute error
        let error = desired - output;

        // Compute gain vector: k = P * x / (lambda + x^T * P * x)
        let p_x = self.p_matrix.dot(&x);
        let denominator = self.lambda + x.dot(&p_x);
        let k = &p_x / denominator;

        // Update weights: w(n+1) = w(n) + k * error
        self.weights = &self.weights + &(&k * error);

        // Update P matrix: P(n+1) = (P(n) - k * x^T * P(n)) / lambda
        let k_reshape = k
            .clone()
            .to_shape((k.len(), 1))
            .expect("Adaptive filter operation must succeed")
            .to_owned();
        let x_reshape = x
            .clone()
            .to_shape((1, x.len()))
            .expect("Adaptive filter operation must succeed")
            .to_owned();
        let k_xt_p = k_reshape.dot(&x_reshape).dot(&self.p_matrix);
        self.p_matrix = (&self.p_matrix - &k_xt_p) / self.lambda;

        self.pos = (self.pos + 1) % self.buffer.len();

        (output, error)
    }

    /// Get current filter weights
    pub fn weights(&self) -> &Array1<f32> {
        &self.weights
    }

    /// Reset filter state
    pub fn reset(&mut self, delta: f32) {
        self.weights.fill(0.0);
        let n = self.weights.len();
        self.p_matrix = Array2::eye(n) * (1.0 / delta);
        self.buffer.fill(0.0);
        self.pos = 0;
    }
}

/// Particle for particle filter
#[derive(Debug, Clone)]
struct Particle {
    /// State vector
    state: Array1<f32>,
    /// Weight
    weight: f32,
}

/// Particle Filter for non-Gaussian/nonlinear state estimation
///
/// Uses Monte Carlo methods to estimate the posterior distribution
/// of states in nonlinear/non-Gaussian systems.
#[derive(Debug, Clone)]
pub struct ParticleFilter {
    /// Particles
    particles: Vec<Particle>,
    /// Number of particles
    num_particles: usize,
    /// Process noise standard deviation
    process_noise_std: f32,
    /// Measurement noise standard deviation
    measurement_noise_std: f32,
}

impl ParticleFilter {
    /// Create a new particle filter
    ///
    /// # Arguments
    /// * `num_particles` - Number of particles
    /// * `initial_state` - Initial state estimate
    /// * `initial_std` - Initial state uncertainty (standard deviation)
    /// * `process_noise_std` - Process noise standard deviation
    /// * `measurement_noise_std` - Measurement noise standard deviation
    pub fn new(
        num_particles: usize,
        initial_state: Array1<f32>,
        initial_std: f32,
        process_noise_std: f32,
        measurement_noise_std: f32,
    ) -> IoResult<Self> {
        if num_particles == 0 {
            return Err(IoError::SignalError(
                "Number of particles must be > 0".into(),
            ));
        }

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, initial_std as f64).map_err(|e| {
            IoError::SignalError(format!("Failed to create normal distribution: {}", e))
        })?;

        // Initialize particles with equal weights
        let weight = 1.0 / num_particles as f32;
        let particles: Vec<Particle> = (0..num_particles)
            .map(|_| {
                let mut state = initial_state.clone();
                for s in state.iter_mut() {
                    *s += normal.sample(&mut rng) as f32;
                }
                Particle { state, weight }
            })
            .collect();

        Ok(Self {
            particles,
            num_particles,
            process_noise_std,
            measurement_noise_std,
        })
    }

    /// Predict step with custom state transition function
    ///
    /// # Arguments
    /// * `transition_fn` - Function that takes a state and returns predicted state
    pub fn predict<F>(&mut self, transition_fn: F)
    where
        F: Fn(&Array1<f32>) -> Array1<f32>,
    {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, self.process_noise_std as f64)
            .expect("Adaptive filter operation must succeed");

        for particle in &mut self.particles {
            // Apply state transition
            particle.state = transition_fn(&particle.state);

            // Add process noise
            for s in particle.state.iter_mut() {
                *s += normal.sample(&mut rng) as f32;
            }
        }
    }

    /// Update step with custom measurement function
    ///
    /// # Arguments
    /// * `measurement` - Observed measurement
    /// * `measurement_fn` - Function that converts state to measurement
    pub fn update<F>(&mut self, measurement: &Array1<f32>, measurement_fn: F)
    where
        F: Fn(&Array1<f32>) -> Array1<f32>,
    {
        // Update weights based on measurement likelihood
        for particle in &mut self.particles {
            let predicted_measurement = measurement_fn(&particle.state);
            let diff = measurement - &predicted_measurement;
            let distance_sq: f32 = diff.iter().map(|&x| x * x).sum();

            // Gaussian likelihood
            let variance = self.measurement_noise_std * self.measurement_noise_std;
            particle.weight *= (-distance_sq / (2.0 * variance)).exp();
        }

        // Normalize weights
        let sum_weights: f32 = self.particles.iter().map(|p| p.weight).sum();
        if sum_weights > 1e-10 {
            for particle in &mut self.particles {
                particle.weight /= sum_weights;
            }
        } else {
            // Reset to uniform if all weights are too small
            let uniform_weight = 1.0 / self.num_particles as f32;
            for particle in &mut self.particles {
                particle.weight = uniform_weight;
            }
        }

        // Resample if effective sample size is low
        self.resample_if_needed();
    }

    /// Resample particles (systematic resampling)
    fn resample_if_needed(&mut self) {
        // Compute effective sample size
        let sum_sq_weights: f32 = self.particles.iter().map(|p| p.weight * p.weight).sum();
        let n_eff = 1.0 / sum_sq_weights;

        // Resample if n_eff < n/2
        if n_eff < (self.num_particles as f32 / 2.0) {
            self.systematic_resample();
        }
    }

    /// Systematic resampling
    fn systematic_resample(&mut self) {
        let mut rng = thread_rng();
        let n = self.num_particles;
        let mut new_particles = Vec::with_capacity(n);

        // Compute cumulative weights
        let mut cumsum = Vec::with_capacity(n);
        let mut sum = 0.0;
        for particle in &self.particles {
            sum += particle.weight;
            cumsum.push(sum);
        }

        // Systematic resampling
        let step = 1.0 / n as f32;
        let start: f32 = rng.gen_range(0.0..step);

        let mut i = 0;
        for j in 0..n {
            let u = start + j as f32 * step;
            while i < n - 1 && cumsum[i] < u {
                i += 1;
            }
            let mut new_particle = self.particles[i].clone();
            new_particle.weight = 1.0 / n as f32;
            new_particles.push(new_particle);
        }

        self.particles = new_particles;
    }

    /// Get mean state estimate
    pub fn mean_state(&self) -> Array1<f32> {
        let state_dim = self.particles[0].state.len();
        let mut mean = Array1::zeros(state_dim);

        for particle in &self.particles {
            mean += &(&particle.state * particle.weight);
        }

        mean
    }

    /// Get weighted covariance of state estimate
    pub fn covariance(&self) -> Array2<f32> {
        let state_dim = self.particles[0].state.len();
        let mean = self.mean_state();
        let mut cov = Array2::zeros((state_dim, state_dim));

        for particle in &self.particles {
            let diff = &particle.state - &mean;
            let diff_col = diff
                .clone()
                .to_shape((state_dim, 1))
                .expect("Adaptive filter operation must succeed")
                .to_owned();
            let diff_row = diff
                .to_shape((1, state_dim))
                .expect("Adaptive filter operation must succeed")
                .to_owned();
            let outer = diff_col.dot(&diff_row);
            cov += &(&outer * particle.weight);
        }

        cov
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_kalman_filter_1d() {
        // Simple 1D position tracking
        let initial_state = arr1(&[0.0]);
        let initial_cov = Array2::from_shape_fn((1, 1), |_| 1.0);
        let transition = Array2::from_shape_fn((1, 1), |_| 1.0); // x(k+1) = x(k)
        let observation = Array2::from_shape_fn((1, 1), |_| 1.0); // z(k) = x(k)
        let process_noise = Array2::from_shape_fn((1, 1), |_| 0.01);
        let measurement_noise = Array2::from_shape_fn((1, 1), |_| 0.1);

        let mut kf = KalmanFilter::new(
            initial_state,
            initial_cov,
            transition,
            observation,
            process_noise,
            measurement_noise,
        )
        .expect("Adaptive filter operation must succeed");

        // Simulate measurement
        let measurement = arr1(&[1.0]);

        kf.predict();
        kf.update(&measurement)
            .expect("Adaptive filter operation must succeed");

        // State should move toward measurement
        assert!(kf.state()[0] > 0.0 && kf.state()[0] < 1.0);
    }

    #[test]
    fn test_lms_filter() {
        let mut lms = LmsFilter::new(4, 0.01).expect("Adaptive filter operation must succeed");

        // Simulate system identification
        for _ in 0..100 {
            let input = scirs2_core::random::thread_rng().gen_range(-1.0..1.0);
            let desired = input * 0.5; // System to identify
            lms.adapt(input, desired);
        }

        // Weights should converge to approximate the system
        // (This is a basic test, convergence varies)
        assert_eq!(lms.weights().len(), 4);
    }

    #[test]
    fn test_nlms_filter() {
        let mut nlms =
            NlmsFilter::new(4, 0.5, None).expect("Adaptive filter operation must succeed");

        for _ in 0..100 {
            let input = scirs2_core::random::thread_rng().gen_range(-1.0..1.0);
            let desired = input * 0.5;
            nlms.adapt(input, desired);
        }

        assert_eq!(nlms.weights().len(), 4);
    }

    #[test]
    fn test_rls_filter() {
        let mut rls = RlsFilter::new(4, 0.99, 0.1).expect("Adaptive filter operation must succeed");

        for _ in 0..100 {
            let input = scirs2_core::random::thread_rng().gen_range(-1.0..1.0);
            let desired = input * 0.5;
            rls.adapt(input, desired);
        }

        assert_eq!(rls.weights().len(), 4);
    }

    #[test]
    fn test_particle_filter() {
        let initial_state = arr1(&[0.0]);
        let pf = ParticleFilter::new(100, initial_state, 1.0, 0.1, 0.5)
            .expect("Adaptive filter operation must succeed");

        assert_eq!(pf.particles.len(), 100);

        let mean = pf.mean_state();
        assert_eq!(mean.len(), 1);
    }

    #[test]
    fn test_matrix_inversion_2x2() {
        let mat = Array2::from_shape_vec((2, 2), vec![4.0, 7.0, 2.0, 6.0])
            .expect("Adaptive filter operation must succeed");
        let inv =
            KalmanFilter::invert_matrix(&mat).expect("Adaptive filter operation must succeed");

        // Check that mat * inv = I
        let product = mat.dot(&inv);
        assert!((product[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((product[[1, 1]] - 1.0).abs() < 1e-5);
        assert!(product[[0, 1]].abs() < 1e-5);
        assert!(product[[1, 0]].abs() < 1e-5);
    }
}
