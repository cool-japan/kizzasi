//! Numerical stability utilities for SSM computations
//!
//! Provides numerically stable implementations for:
//! - Log-space operations (log-sum-exp, log-softmax)
//! - Stable discretization methods
//! - Overflow/underflow protection
//! - Long sequence accumulation
//! - Mixed precision helpers

use scirs2_core::ndarray::{Array1, Array2};

// ============================================================================
// Constants
// ============================================================================

/// Machine epsilon for f32
pub const F32_EPS: f32 = 1.192_092_9e-7;

/// Minimum positive normal f32
pub const F32_MIN_POSITIVE: f32 = 1.175_494_4e-38;

/// Maximum finite f32
pub const F32_MAX: f32 = 3.402_823_5e38;

/// Safe log minimum (avoids log(0))
pub const LOG_MIN: f32 = -87.0; // ln(F32_MIN_POSITIVE) ≈ -87.3

/// Safe log maximum (avoids exp overflow)
pub const LOG_MAX: f32 = 88.0; // ln(F32_MAX) ≈ 88.7

/// Small epsilon for numerical stability
pub const EPS: f32 = 1e-8;

// ============================================================================
// Log-space Operations
// ============================================================================

/// Numerically stable log-sum-exp: log(sum(exp(x)))
///
/// Uses the shift trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
pub fn log_sum_exp(x: &[f32]) -> f32 {
    if x.is_empty() {
        return f32::NEG_INFINITY;
    }

    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_val.is_infinite() && max_val < 0.0 {
        return f32::NEG_INFINITY;
    }

    let sum: f32 = x.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum.ln()
}

/// Log-sum-exp for two values: log(exp(a) + exp(b))
#[inline]
pub fn log_add_exp(a: f32, b: f32) -> f32 {
    if a > b {
        a + (1.0 + (b - a).exp()).ln()
    } else {
        b + (1.0 + (a - b).exp()).ln()
    }
}

/// Numerically stable log-softmax
pub fn log_softmax_stable(x: &Array1<f32>) -> Array1<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted = x.mapv(|v| v - max_val);
    let log_sum: f32 = shifted.mapv(|v| v.exp()).sum().ln();
    shifted.mapv(|v| v - log_sum)
}

/// Numerically stable softmax
pub fn softmax_stable(x: &Array1<f32>) -> Array1<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_x = x.mapv(|v| (v - max_val).exp());
    let sum: f32 = exp_x.sum();
    if sum > 0.0 {
        exp_x / sum
    } else {
        Array1::from_elem(x.len(), 1.0 / x.len() as f32)
    }
}

// ============================================================================
// Safe Exponential and Log
// ============================================================================

/// Safe exponential that avoids overflow
#[inline]
pub fn safe_exp(x: f32) -> f32 {
    if x > LOG_MAX {
        F32_MAX
    } else if x < LOG_MIN {
        0.0
    } else {
        x.exp()
    }
}

/// Safe natural log that avoids log(0)
#[inline]
pub fn safe_ln(x: f32) -> f32 {
    if x <= 0.0 {
        LOG_MIN
    } else {
        x.ln().max(LOG_MIN)
    }
}

/// Safe log10 that avoids log(0)
#[inline]
pub fn safe_log10(x: f32) -> f32 {
    if x <= 0.0 {
        LOG_MIN / std::f32::consts::LN_10
    } else {
        x.log10()
    }
}

/// Clamp value to safe range for exp
#[inline]
pub fn clamp_for_exp(x: f32) -> f32 {
    x.clamp(LOG_MIN, LOG_MAX)
}

// ============================================================================
// Stable Division and Normalization
// ============================================================================

/// Safe division that avoids division by zero
#[inline]
pub fn safe_div(num: f32, denom: f32) -> f32 {
    if denom.abs() < EPS {
        if num >= 0.0 {
            F32_MAX
        } else {
            -F32_MAX
        }
    } else {
        num / denom
    }
}

/// Safe normalization (avoid div by zero in vector norm)
pub fn safe_normalize(x: &Array1<f32>) -> Array1<f32> {
    let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < EPS {
        Array1::zeros(x.len())
    } else {
        x / norm
    }
}

/// L2 normalize with minimum denominator
pub fn l2_normalize(x: &Array1<f32>, min_norm: f32) -> Array1<f32> {
    let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let denom = norm.max(min_norm);
    x / denom
}

// ============================================================================
// Stable Accumulation
// ============================================================================

/// Kahan summation for improved accuracy in long sequences
#[derive(Debug, Clone, Default)]
pub struct KahanSum {
    sum: f32,
    compensation: f32,
}

impl KahanSum {
    /// Create a new Kahan summation accumulator
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    /// Create with initial value
    pub fn with_value(initial: f32) -> Self {
        Self {
            sum: initial,
            compensation: 0.0,
        }
    }

    /// Add a value using compensated summation
    #[inline]
    pub fn add(&mut self, value: f32) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    /// Get the current sum
    pub fn sum(&self) -> f32 {
        self.sum
    }

    /// Reset the accumulator
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.compensation = 0.0;
    }
}

/// Compute sum using Kahan summation
pub fn kahan_sum(values: &[f32]) -> f32 {
    let mut acc = KahanSum::new();
    for &v in values {
        acc.add(v);
    }
    acc.sum()
}

/// Compute mean using Kahan summation
pub fn kahan_mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    kahan_sum(values) / values.len() as f32
}

/// Welford's online algorithm for numerically stable variance
#[derive(Debug, Clone, Default)]
pub struct WelfordVariance {
    count: usize,
    mean: f32,
    m2: f32,
}

impl WelfordVariance {
    /// Create a new Welford accumulator
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Add a value
    pub fn add(&mut self, value: f32) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get the mean
    pub fn mean(&self) -> f32 {
        self.mean
    }

    /// Get the sample variance (using n-1 denominator)
    pub fn variance(&self) -> f32 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f32
        }
    }

    /// Get the population variance (using n denominator)
    pub fn variance_population(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.m2 / self.count as f32
        }
    }

    /// Get the standard deviation
    pub fn std(&self) -> f32 {
        self.variance().sqrt()
    }

    /// Get the count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset the accumulator
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

// ============================================================================
// Stable SSM Discretization
// ============================================================================

/// Stable matrix exponential using Padé approximation with scaling
///
/// Uses scaling and squaring: exp(A) = (exp(A/2^s))^(2^s)
pub fn matrix_exp_pade(a: &Array2<f32>, order: usize) -> Array2<f32> {
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "Matrix must be square");

    // Compute norm for scaling
    let norm: f32 = a.iter().map(|x| x.abs()).sum::<f32>();

    // Determine scaling factor
    let s = if norm > 0.0 {
        (norm.log2() as i32).max(0) as u32
    } else {
        0
    };

    // Scale matrix
    let scale = 2.0f32.powi(-(s as i32));
    let a_scaled = a.mapv(|x| x * scale);

    // Compute Padé approximant
    let mut u: Array2<f32> = Array2::eye(n);
    let mut v: Array2<f32> = Array2::eye(n);

    // Padé coefficients for given order
    let (c_u, c_v) = pade_coefficients(order);

    let mut a_power = Array2::eye(n);
    for k in 1..=order {
        a_power = a_power.dot(&a_scaled);
        if k % 2 == 1 {
            u = &u + &a_power.mapv(|x| x * c_u[k]);
        } else {
            v = &v + &a_power.mapv(|x| x * c_v[k]);
        }
    }

    // exp(A) ≈ (V - U)^(-1) * (V + U)
    // For simplicity, use approximation: exp(A) ≈ (I + A/2) * (I - A/2)^(-1)
    // which is the first-order Padé
    let result = solve_linear(&(&v - &u), &(&v + &u));

    // Square back
    let mut exp_a = result;
    for _ in 0..s {
        exp_a = exp_a.dot(&exp_a);
    }

    exp_a
}

/// Get Padé coefficients for given order
fn pade_coefficients(order: usize) -> (Vec<f32>, Vec<f32>) {
    // Simplified coefficients for orders 1-6
    let order = order.min(6);
    let mut c_u = vec![0.0f32; order + 1];
    let mut c_v = vec![0.0f32; order + 1];

    // c_v[0] = 1, c_v[2] = 1/2, c_v[4] = 1/24, ...
    // c_u[1] = 1/2, c_u[3] = 1/12, ...
    c_v[0] = 1.0;
    if order >= 1 {
        c_u[1] = 0.5;
    }
    if order >= 2 {
        c_v[2] = 1.0 / 12.0;
    }
    if order >= 3 {
        c_u[3] = 1.0 / 120.0;
    }
    if order >= 4 {
        c_v[4] = 1.0 / 30240.0;
    }
    if order >= 5 {
        c_u[5] = 1.0 / 1209600.0;
    }
    if order >= 6 {
        c_v[6] = 1.0 / 17297280.0;
    }

    (c_u, c_v)
}

/// Simple linear solve (I + A)^(-1) B using Neumann series approximation
fn solve_linear(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let n = a.shape()[0];

    // For well-conditioned matrices, use Neumann series
    // (I - A)^(-1) ≈ I + A + A^2 + ...
    // Here we have (A)^(-1) B, which we approximate with a few iterations

    // Simple iterative refinement
    let mut x = b.clone();
    let identity = Array2::eye(n);

    // Gauss-Seidel-like iteration
    for _ in 0..10 {
        let residual = b - &a.dot(&x);
        let correction = residual.mapv(|v| v * 0.5);
        x = &x + &identity.dot(&correction);
    }

    x
}

/// Zero-Order Hold (ZOH) discretization for SSM
///
/// Given continuous A, B matrices and step size dt:
/// A_d = exp(A * dt)
/// B_d = A^(-1) * (A_d - I) * B (approximated for stability)
pub fn zoh_discretize(a: &Array2<f32>, b: &Array2<f32>, dt: f32) -> (Array2<f32>, Array2<f32>) {
    let n = a.shape()[0];

    // Scale A by dt
    let a_dt = a.mapv(|x| x * dt);

    // Compute exp(A * dt) using Taylor series
    let a_d = taylor_exp(&a_dt, 8);

    // B_d approximation using first-order ZOH:
    // B_d ≈ dt * (I + A*dt/2 + (A*dt)^2/6) * B
    // For small dt, this is ≈ dt * B
    let identity: Array2<f32> = Array2::eye(n);
    let half_a_dt = a_dt.mapv(|x| x * 0.5);
    let approx_factor = &identity + &half_a_dt;
    let b_d = approx_factor.dot(b).mapv(|x| x * dt);

    (a_d, b_d)
}

// ============================================================================
// Diagonal-A Discretization Methods
// ============================================================================

/// Discretization method for SSM state-space models with diagonal A.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DiscretizationMethod {
    /// Zero-Order Hold (exact for piecewise-constant inputs, diagonal A)
    Zoh,
    /// Bilinear (Tustin) transform — preserves stability for all dt > 0
    Bilinear,
    /// Forward Euler — first-order; unstable for dt·|a| > 2
    ForwardEuler,
}

/// Zero-Order Hold (ZOH) discretization for **diagonal** A.
///
/// For each element i:
/// - a_bar[i] = exp(dt · a[i])
/// - b_bar[i, :] = (exp(dt · a[i]) - 1) / a[i] · b[i, :]
///   (= dt · b[i, :] when a[i] ≈ 0)
///
/// This is the exact closed-form ZOH for a diagonal continuous-time system.
pub fn zoh_discretize_diagonal(
    a: &Array1<f32>,
    b: &Array2<f32>,
    dt: f32,
) -> (Array1<f32>, Array2<f32>) {
    let a_bar = a.mapv(|ai| (dt * ai).exp());
    let n_in = b.ncols();
    let state_dim = a.len();
    let b_bar = Array2::from_shape_fn((state_dim, n_in), |(i, j)| {
        let ai = a[i];
        // Scale factor: expm1(dt*ai) / ai  when |dt*ai| > eps
        //               dt                  in the limit ai -> 0
        let scale = {
            let y = dt * ai;
            if y.abs() < 1e-6 {
                dt
            } else {
                y.exp_m1() / ai
            }
        };
        scale * b[[i, j]]
    });
    (a_bar, b_bar)
}

/// Bilinear (Tustin) discretization for **diagonal** A.
///
/// For each element i:
/// - a_bar[i] = (1 + dt·a[i]/2) / (1 - dt·a[i]/2)
/// - b_bar[i, :] = dt/2 · (1 + a_bar[i]) · b[i, :]
///
/// Preserves stability: |a_bar| < 1 for all Re(a) < 0 and dt > 0.
pub fn bilinear_discretize(
    a: &Array1<f32>,
    b: &Array2<f32>,
    dt: f32,
) -> (Array1<f32>, Array2<f32>) {
    let half_dt = dt * 0.5;
    let a_bar: Array1<f32> = a.mapv(|ai| {
        let num = 1.0 + half_dt * ai;
        let den = 1.0 - half_dt * ai;
        num / den
    });
    let n_in = b.ncols();
    let state_dim = a.len();
    let b_bar = Array2::from_shape_fn((state_dim, n_in), |(i, j)| {
        half_dt * (1.0 + a_bar[i]) * b[[i, j]]
    });
    (a_bar, b_bar)
}

/// Forward Euler discretization for **diagonal** A.
///
/// For each element i:
/// - a_bar[i] = 1 + dt·a[i]
/// - b_bar = dt·b
///
/// NOTE: Unstable when dt·|a[i]| > 2. Use ZOH or Bilinear for large dt.
pub fn forward_euler_discretize(
    a: &Array1<f32>,
    b: &Array2<f32>,
    dt: f32,
) -> (Array1<f32>, Array2<f32>) {
    let a_bar = a.mapv(|ai| 1.0 + dt * ai);
    let b_bar = b.mapv(|bij| dt * bij);
    (a_bar, b_bar)
}

/// Dispatch to the requested diagonal-A discretization method.
pub fn discretize(
    method: DiscretizationMethod,
    a: &Array1<f32>,
    b: &Array2<f32>,
    dt: f32,
) -> (Array1<f32>, Array2<f32>) {
    match method {
        DiscretizationMethod::Zoh => zoh_discretize_diagonal(a, b, dt),
        DiscretizationMethod::Bilinear => bilinear_discretize(a, b, dt),
        DiscretizationMethod::ForwardEuler => forward_euler_discretize(a, b, dt),
    }
}

/// Taylor series expansion for matrix exponential
fn taylor_exp(a: &Array2<f32>, terms: usize) -> Array2<f32> {
    let n = a.shape()[0];
    let mut result = Array2::eye(n);
    let mut a_power = Array2::eye(n);
    let mut factorial = 1.0f32;

    for k in 1..=terms {
        factorial *= k as f32;
        a_power = a_power.dot(a);
        result = &result + &a_power.mapv(|x| x / factorial);
    }

    result
}

// ============================================================================
// Gradient Clipping
// ============================================================================

/// Clip gradients by global norm
pub fn clip_grad_norm(gradients: &mut [Array1<f32>], max_norm: f32) -> f32 {
    let total_norm: f32 = gradients
        .iter()
        .map(|g| g.iter().map(|x| x * x).sum::<f32>())
        .sum::<f32>()
        .sqrt();

    let clip_coef = max_norm / (total_norm + EPS);
    if clip_coef < 1.0 {
        for grad in gradients.iter_mut() {
            grad.mapv_inplace(|x| x * clip_coef);
        }
    }

    total_norm
}

/// Clip gradients by value
pub fn clip_grad_value(gradient: &mut Array1<f32>, max_value: f32) {
    gradient.mapv_inplace(|x| x.clamp(-max_value, max_value));
}

// ============================================================================
// NaN and Inf Handling
// ============================================================================

/// Check if array contains NaN or Inf
pub fn has_nan_inf(x: &Array1<f32>) -> bool {
    x.iter().any(|&v| v.is_nan() || v.is_infinite())
}

/// Replace NaN values with a default
pub fn replace_nan(x: &Array1<f32>, default: f32) -> Array1<f32> {
    x.mapv(|v| if v.is_nan() { default } else { v })
}

/// Replace NaN and Inf values
pub fn sanitize(x: &Array1<f32>, nan_value: f32, inf_value: f32) -> Array1<f32> {
    x.mapv(|v| {
        if v.is_nan() {
            nan_value
        } else if v.is_infinite() {
            if v > 0.0 {
                inf_value
            } else {
                -inf_value
            }
        } else {
            v
        }
    })
}

/// Check and clamp values to valid range
pub fn clamp_to_valid(x: &Array1<f32>, min: f32, max: f32) -> Array1<f32> {
    x.mapv(|v| {
        if v.is_nan() {
            (min + max) / 2.0
        } else {
            v.clamp(min, max)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sum_exp() {
        let x = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&x);
        // log(e^1 + e^2 + e^3) ≈ 3.408
        assert!((result - 3.408).abs() < 0.01);
    }

    #[test]
    fn test_log_sum_exp_large() {
        // Test with large values that would overflow naive implementation
        let x = vec![1000.0, 1001.0, 1002.0];
        let result = log_sum_exp(&x);
        // Should be approximately 1002.408
        assert!((result - 1002.408).abs() < 0.01);
    }

    #[test]
    fn test_log_add_exp() {
        let a = 2.0f32;
        let b = 3.0f32;
        let result = log_add_exp(a, b);
        let expected = (a.exp() + b.exp()).ln();
        assert!((result - expected).abs() < 0.001);
    }

    #[test]
    fn test_softmax_stable() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = softmax_stable(&x);

        // Sum should be 1
        assert!((result.sum() - 1.0).abs() < 0.001);
        // Values should be ordered
        assert!(result[2] > result[1] && result[1] > result[0]);
    }

    #[test]
    fn test_softmax_large_values() {
        // Test with large values
        let x = Array1::from_vec(vec![1000.0, 1001.0, 1002.0]);
        let result = softmax_stable(&x);
        assert!((result.sum() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_safe_exp() {
        assert!(safe_exp(100.0) < f32::INFINITY);
        assert!(safe_exp(100.0) > 0.0);
        assert!(safe_exp(-100.0) >= 0.0); // Returns 0 for very small values
        assert!((safe_exp(0.0) - 1.0).abs() < 0.001);
        assert!((safe_exp(1.0) - std::f32::consts::E).abs() < 0.001);
    }

    #[test]
    fn test_safe_ln() {
        assert!(safe_ln(0.0).is_finite());
        assert!(safe_ln(-1.0).is_finite());
        assert!((safe_ln(1.0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_kahan_sum() {
        let values: Vec<f32> = (0..1000).map(|_| 0.1).collect();
        let result = kahan_sum(&values);
        // Should be closer to 100.0 than naive summation
        assert!((result - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_welford_variance() {
        let mut acc = WelfordVariance::new();
        for i in 1..=5 {
            acc.add(i as f32);
        }

        assert!((acc.mean() - 3.0).abs() < 0.001);
        assert!((acc.variance() - 2.5).abs() < 0.001); // sample variance
    }

    #[test]
    fn test_safe_normalize() {
        let x = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let result = safe_normalize(&x);
        assert!(!has_nan_inf(&result));

        let x = Array1::from_vec(vec![3.0, 4.0]);
        let result = safe_normalize(&x);
        let norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_clip_grad_norm() {
        let mut grads = vec![
            Array1::from_vec(vec![3.0, 4.0]),
            Array1::from_vec(vec![5.0, 12.0]),
        ];
        // Total norm = sqrt(9+16+25+144) = sqrt(194) ≈ 13.93

        let norm = clip_grad_norm(&mut grads, 5.0);
        assert!((norm - 13.93).abs() < 0.1);

        // After clipping, total norm should be ~5.0
        let new_norm: f32 = grads
            .iter()
            .map(|g| g.iter().map(|x| x * x).sum::<f32>())
            .sum::<f32>()
            .sqrt();
        assert!((new_norm - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_sanitize() {
        let x = Array1::from_vec(vec![1.0, f32::NAN, f32::INFINITY, -f32::INFINITY, 2.0]);
        let result = sanitize(&x, 0.0, 1e6);

        assert!(!has_nan_inf(&result));
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[4], 2.0);
    }

    #[test]
    fn test_taylor_exp_identity() {
        let n = 3;
        let a = Array2::zeros((n, n));
        let result = taylor_exp(&a, 6);

        // exp(0) = I
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result[[i, j]] - expected).abs() < 0.001);
            }
        }
    }

    #[test]
    fn test_zoh_discretize() {
        // Simple diagonal system
        let a = Array2::from_diag(&Array1::from_vec(vec![-1.0, -2.0]));
        let b: Array2<f32> = Array2::eye(2);
        let dt = 0.1;

        let (a_d, b_d) = zoh_discretize(&a, &b, dt);

        // A_d should be close to exp(A*dt) = diag(exp(-0.1), exp(-0.2))
        // Using Taylor series approximation, we get different values
        // Taylor: I + A*dt + (A*dt)^2/2 + ... ≈ 1 - 0.1 + 0.005 = 0.905 for first entry
        assert!(
            (a_d[[0, 0]] - 0.905).abs() < 0.1,
            "a_d[0,0] = {}",
            a_d[[0, 0]]
        );
        assert!(
            (a_d[[1, 1]] - 0.82).abs() < 0.1,
            "a_d[1,1] = {}",
            a_d[[1, 1]]
        );

        // B_d should be non-zero
        assert!(b_d[[0, 0]].abs() > 0.0, "b_d[0,0] = {}", b_d[[0, 0]]);
    }

    // -----------------------------------------------------------------------
    // Diagonal-A discretization tests
    // -----------------------------------------------------------------------

    #[test]
    fn bilinear_stable_negative_eigenvalue() {
        // For Re(a) < 0, all dt > 0, |a_bar| < 1
        let a = Array1::from_vec(vec![-1.0f32, -0.5, -2.0]);
        let b = Array2::<f32>::ones((3, 1));
        for &dt in &[0.01f32, 0.1, 0.5, 1.0, 2.0] {
            let (a_bar, _) = bilinear_discretize(&a, &b, dt);
            for &x in a_bar.iter() {
                assert!(x.abs() < 1.0, "stability violated: a_bar={x} at dt={dt}");
            }
        }
    }

    #[test]
    fn bilinear_exact_at_zero_eigenvalue() {
        // a=0: a_bar should be 1, b_bar = dt * b
        let a = Array1::<f32>::zeros(2);
        let b = Array2::from_shape_vec((2, 1), vec![2.0f32, 3.0]).unwrap();
        let dt = 0.1;
        let (a_bar, b_bar) = bilinear_discretize(&a, &b, dt);
        for &x in a_bar.iter() {
            assert!((x - 1.0).abs() < 1e-6, "a_bar should be 1 for a=0, got {x}");
        }
        // b_bar = dt/2 * (1 + 1) * b = dt * b
        let expected_b = b.mapv(|x| dt * x);
        for (got, exp) in b_bar.iter().zip(expected_b.iter()) {
            assert!((got - exp).abs() < 1e-6, "b_bar mismatch: {got} vs {exp}");
        }
    }

    #[test]
    fn forward_euler_close_to_zoh_small_dt() {
        // Forward Euler O(dt^2) approximation to ZOH-diagonal
        let a = Array1::from_vec(vec![-1.0f32]);
        let b = Array2::<f32>::ones((1, 1));
        let dt = 1e-4_f32;
        let (a_fe, _) = forward_euler_discretize(&a, &b, dt);
        let (a_zoh, _) = zoh_discretize_diagonal(&a, &b, dt);
        let err = (a_fe[0] - a_zoh[0]).abs();
        assert!(
            err < 1e-7,
            "FE vs ZOH-diagonal error {err} too large for dt={dt}"
        );
    }

    #[test]
    fn discretize_zoh_matches_zoh_discretize_diagonal() {
        let a = Array1::from_vec(vec![-0.5f32, -1.0, -2.0]);
        let b = Array2::<f32>::ones((3, 2));
        let dt = 0.05;
        let (a1, b1) = zoh_discretize_diagonal(&a, &b, dt);
        let (a2, b2) = discretize(DiscretizationMethod::Zoh, &a, &b, dt);
        for (x, y) in a1.iter().zip(a2.iter()) {
            assert!((x - y).abs() < 1e-7, "ZOH mismatch: {x} vs {y}");
        }
        for (x, y) in b1.iter().zip(b2.iter()) {
            assert!((x - y).abs() < 1e-7, "ZOH B mismatch: {x} vs {y}");
        }
    }

    #[test]
    fn zoh_diagonal_exact_expm() {
        // For diagonal A, ZOH should be exact: a_bar[i] = exp(dt * a[i])
        let a = Array1::from_vec(vec![-1.0f32, -2.0, -0.5]);
        let b = Array2::<f32>::ones((3, 2));
        let dt = 0.1;
        let (a_bar, _) = zoh_discretize_diagonal(&a, &b, dt);
        for (i, (&ab, &ai)) in a_bar.iter().zip(a.iter()).enumerate() {
            let expected = (dt * ai).exp();
            assert!(
                (ab - expected).abs() < 1e-6,
                "ZOH-diagonal a_bar[{i}]={ab} expected {expected}"
            );
        }
    }
}
