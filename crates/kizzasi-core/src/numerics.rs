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

/// Stable matrix exponential using Padé\[13,13\] approximation with scaling and squaring.
///
/// Uses scaling and squaring: exp(A) = (exp(A/2^s))^(2^s)
///
/// The approximant is:
/// - `U = sum_{k odd}  b[k] * A^k`
/// - `V = sum_{k even} b[k] * A^k`
/// - `exp(A) ≈ (V - U)^{-1} * (V + U)`
///
/// Coefficients from Higham 2005 Table 10.4, normalized so `b[0]=1`.
pub fn matrix_exp_pade(a: &Array2<f32>, order: usize) -> Array2<f32> {
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "Matrix must be square");

    // 1-norm (max absolute column sum) for scaling heuristic
    let norm: f32 = {
        let mut col_sums = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                col_sums[j] += a[[i, j]].abs();
            }
        }
        col_sums.into_iter().fold(0.0f32, f32::max)
    };

    // Determine scaling factor: scale so that ||A/2^s|| <= 1
    let s = if norm > 1.0 {
        (norm.log2().ceil() as i32).max(0) as u32
    } else {
        0
    };

    // Scale matrix
    let scale = 2.0f32.powi(-(s as i32));
    let a_scaled = a.mapv(|x| x * scale);

    // Get Padé[13,13] coefficients (normalized so b[0]=1)
    let b = pade_coefficients(order);
    let p = b.len() - 1; // polynomial degree = 13

    // Accumulate even-degree terms in V and odd-degree terms in U
    // V = b[0]*I + b[2]*A^2 + b[4]*A^4 + ...
    // U = b[1]*A + b[3]*A^3 + b[5]*A^5 + ...
    let mut v: Array2<f32> = Array2::eye(n);
    v.mapv_inplace(|x| x * b[0]);
    let mut u: Array2<f32> = Array2::zeros((n, n));

    // Build powers A^1, A^2, ..., A^p iteratively
    let mut a_power = a_scaled.clone(); // A^1
    for (k, &coeff) in b.iter().enumerate().skip(1) {
        if k % 2 == 1 {
            // odd power -> U
            u = &u + &a_power.mapv(|x| x * coeff);
        } else {
            // even power -> V
            v = &v + &a_power.mapv(|x| x * coeff);
        }
        if k < p {
            a_power = a_power.dot(&a_scaled);
        }
    }

    // exp(A) ≈ (V - U)^{-1} * (V + U)
    let v_minus_u = &v - &u;
    let v_plus_u = &v + &u;
    let result = solve_linear(&v_minus_u, &v_plus_u);

    // Square back s times: exp(A) = exp(A/2^s)^(2^s)
    let mut exp_a = result;
    for _ in 0..s {
        let tmp = exp_a.clone();
        exp_a = exp_a.dot(&tmp);
    }

    exp_a
}

/// Padé[13,13] scalar coefficients from Higham (2005) Table 10.4.
///
/// Returns the 14 coefficients `b[0..=13]` normalized so that `b[0] = 1`.
/// The same table serves both numerator (U, odd indices) and denominator (V, even indices).
/// The `_order` parameter is accepted for API compatibility but always uses degree 13.
fn pade_coefficients(_order: usize) -> Vec<f32> {
    // Higham 2005, Table 10.4 — exact rational values cast to f64 then f32.
    // The polynomial is: sum_{k=0}^{13} b[k] * A^k
    // Numerator U uses odd-k terms; denominator V uses even-k terms.
    // Raw (un-normalized) values:
    const RAW: [f64; 14] = [
        64_764_752_532_480_000.0, // b[0]
        32_382_376_266_240_000.0, // b[1]
        7_771_770_303_897_600.0,  // b[2]
        1_187_353_796_428_800.0,  // b[3]
        129_060_195_264_000.0,    // b[4]
        10_559_470_521_600.0,     // b[5]
        670_442_572_800.0,        // b[6]
        33_522_128_640.0,         // b[7]
        1_323_241_920.0,          // b[8]
        40_840_800.0,             // b[9]
        960_960.0,                // b[10]
        16_380.0,                 // b[11]
        182.0,                    // b[12]
        1.0,                      // b[13]
    ];
    let scale = 1.0 / RAW[0];
    RAW.iter().map(|&x| (x * scale) as f32).collect()
}

/// Solve the linear system `A X = B` using LU decomposition with partial pivoting.
///
/// Implements the Doolittle algorithm with row pivoting.  Near-zero pivots are
/// regularised by adding `1e-8` to the diagonal — this is safe because the
/// caller always inverts `V - U`, which is invertible by Padé construction.
///
/// Returns `X` such that `A X ≈ B`.
fn solve_linear(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let n = a.shape()[0];
    let m = b.shape()[1];

    // -----------------------------------------------------------------------
    // LU decomposition with partial (row) pivoting
    // -----------------------------------------------------------------------
    // Work on a mutable copy; combine L and U in-place (Doolittle).
    let mut lu = a.clone();
    let mut piv = vec![0usize; n]; // pivot row indices

    for col in 0..n {
        // Find pivot row: argmax |lu[row, col]| for row >= col
        let mut max_abs = lu[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = lu[[row, col]].abs();
            if v > max_abs {
                max_abs = v;
                max_row = row;
            }
        }
        piv[col] = max_row;

        // Swap rows col and max_row in lu
        if max_row != col {
            for j in 0..n {
                let tmp = lu[[col, j]];
                lu[[col, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
        }

        // Regularise near-zero pivot to avoid division by zero
        if lu[[col, col]].abs() < 1e-12 {
            lu[[col, col]] = if lu[[col, col]] >= 0.0 { 1e-8 } else { -1e-8 };
        }

        // Eliminate below
        let pivot_val = lu[[col, col]];
        for row in (col + 1)..n {
            lu[[row, col]] /= pivot_val;
            let multiplier = lu[[row, col]];
            for j in (col + 1)..n {
                let sub = multiplier * lu[[col, j]];
                lu[[row, j]] -= sub;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Solve for each column of B separately
    // -----------------------------------------------------------------------
    let mut x = Array2::<f32>::zeros((n, m));

    for col in 0..m {
        // Apply row permutations to this RHS column
        let mut rhs: Vec<f32> = (0..n).map(|i| b[[i, col]]).collect();
        for (i, &pivot_row) in piv.iter().enumerate() {
            rhs.swap(i, pivot_row);
        }

        // Forward substitution: solve L y = rhs  (L has unit diagonal)
        let mut y = rhs;
        for i in 0..n {
            for j in 0..i {
                let sub = lu[[i, j]] * y[j];
                y[i] -= sub;
            }
        }

        // Back substitution: solve U x_col = y
        let mut x_col = y;
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                let sub = lu[[i, j]] * x_col[j];
                x_col[i] -= sub;
            }
            x_col[i] /= lu[[i, i]];
        }

        for i in 0..n {
            x[[i, col]] = x_col[i];
        }
    }

    x
}

/// Zero-Order Hold (ZOH) discretization for SSM
///
/// Given continuous A, B matrices and step size dt:
/// - `A_d = exp(A * dt)`
/// - `B_d = A^{-1} * (A_d - I) * B`  (exact ZOH, solved via LU)
///
/// For near-zero `||A * dt||`, falls back to the zeroth-order series `B_d ≈ dt * B`
/// to avoid numerical issues in the linear solve.
pub fn zoh_discretize(a: &Array2<f32>, b: &Array2<f32>, dt: f32) -> (Array2<f32>, Array2<f32>) {
    let n = a.shape()[0];

    // Scale A by dt
    let a_dt = a.mapv(|x| x * dt);

    // Compute exp(A * dt) using the corrected Padé[13,13] approximant
    let a_d = matrix_exp_pade(&a_dt, 13);

    // Compute B_d = A^{-1} * (A_d - I) * B via solve_linear(A*dt, (A_d - I) * B)
    // This is equivalent to solving (A*dt) * B_d = (A_d - I) * B and then
    // multiplying by 1/dt on both sides — but we directly use:
    //   B_d = A^{-1} * (A_d - I) * B  =>  A * B_d = (A_d - I) * B
    //   => (A*dt) * B_d = (A_d - I) * B * dt / dt   -- no, keep it clean:
    //   solve_linear(A*dt, (A_d - I) * B) gives (A*dt)^{-1} * (A_d - I) * B
    //   which equals A^{-1}/dt * (A_d - I) * B.
    // The correct ZOH formula is B_d = A^{-1}*(A_d - I)*B, so:
    //   B_d = dt * solve_linear(A*dt, (A_d - I) * B)

    // 1-norm of A*dt as a proxy for whether A is near-zero
    let norm_a_dt: f32 = {
        let mut col_sums = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                col_sums[j] += a_dt[[i, j]].abs();
            }
        }
        col_sums.into_iter().fold(0.0f32, f32::max)
    };

    let b_d = if norm_a_dt < 1e-9 {
        // Degenerate case: A ≈ 0, exact limit gives B_d = dt * B
        b.mapv(|x| x * dt)
    } else {
        // Exact ZOH: B_d = A^{-1} * (A_d - I) * B
        // Equivalently:  A * B_d = (A_d - I) * B
        //             (A*dt) * B_d = dt * (A_d - I) * B
        // So:  B_d = solve_linear(A*dt, dt*(A_d-I)*B)
        let a_d_minus_i = &a_d - &Array2::<f32>::eye(n);
        let rhs = a_d_minus_i.dot(b).mapv(|x| x * dt);
        solve_linear(&a_dt, &rhs)
    };

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
/// For each element `i`:
/// - `a_bar[i] = exp(dt · a[i])`
/// - `b_bar[i, :] = (exp(dt · a[i]) - 1) / a[i] · b[i, :]`
///   (= `dt · b[i, :]` when `a[i] ≈ 0`)
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
/// For each element `i`:
/// - `a_bar[i] = (1 + dt·a[i]/2) / (1 - dt·a[i]/2)`
/// - `b_bar[i, :] = dt/2 · (1 + a_bar[i]) · b[i, :]`
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
/// For each element `i`:
/// - `a_bar[i] = 1 + dt·a[i]`
/// - b_bar = dt·b
///
/// NOTE: Unstable when `dt·|a[i]| > 2`. Use ZOH or Bilinear for large dt.
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

/// Taylor series expansion for matrix exponential (used in tests)
#[cfg(test)]
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
        // Simple diagonal system: A = diag(-1, -2), B = I, dt = 0.1
        let a = Array2::from_diag(&Array1::from_vec(vec![-1.0, -2.0]));
        let b: Array2<f32> = Array2::eye(2);
        let dt = 0.1f32;

        let (a_d, b_d) = zoh_discretize(&a, &b, dt);

        // Exact ZOH: A_d = diag(exp(-0.1), exp(-0.2))
        let exp_neg01 = (-0.1f32).exp();
        let exp_neg02 = (-0.2f32).exp();
        assert!(
            (a_d[[0, 0]] - exp_neg01).abs() < 1e-4,
            "a_d[0,0] = {}, expected {}",
            a_d[[0, 0]],
            exp_neg01
        );
        assert!(
            (a_d[[1, 1]] - exp_neg02).abs() < 1e-4,
            "a_d[1,1] = {}, expected {}",
            a_d[[1, 1]],
            exp_neg02
        );

        // Exact ZOH for diagonal: B_d[i,i] = (exp(a[i]*dt) - 1) / a[i]
        // B_d[0,0] = (exp(-0.1) - 1) / (-1) = 1 - exp(-0.1)
        let b_d_00_exact = (1.0 - exp_neg01) / 1.0; // divide by |a[0]| with correct sign
        let b_d_11_exact = (1.0 - exp_neg02) / 2.0;
        assert!(
            (b_d[[0, 0]] - b_d_00_exact).abs() < 1e-4,
            "b_d[0,0] = {}, expected {}",
            b_d[[0, 0]],
            b_d_00_exact
        );
        assert!(
            (b_d[[1, 1]] - b_d_11_exact).abs() < 1e-4,
            "b_d[1,1] = {}, expected {}",
            b_d[[1, 1]],
            b_d_11_exact
        );
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

    // -----------------------------------------------------------------------
    // matrix_exp_pade correctness tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_matrix_exp_identity() {
        // exp(zero matrix) = I
        let n = 3usize;
        let a: Array2<f32> = Array2::zeros((n, n));
        let result = matrix_exp_pade(&a, 13);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0f32 } else { 0.0f32 };
                assert!(
                    (result[[i, j]] - expected).abs() < 1e-6,
                    "exp(0)[{i},{j}] = {}, expected {expected}",
                    result[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_matrix_exp_diagonal() {
        // exp(diag(-1,-2)) ≈ diag(e^{-1}, e^{-2})
        let a = Array2::from_diag(&Array1::from_vec(vec![-1.0f32, -2.0f32]));
        let result = matrix_exp_pade(&a, 13);
        let e1 = (-1.0f32).exp();
        let e2 = (-2.0f32).exp();
        assert!(
            (result[[0, 0]] - e1).abs() < 1e-5,
            "exp(diag)[0,0] = {}, expected {e1}",
            result[[0, 0]]
        );
        assert!(
            (result[[1, 1]] - e2).abs() < 1e-5,
            "exp(diag)[1,1] = {}, expected {e2}",
            result[[1, 1]]
        );
        // Off-diagonal should be ~0
        assert!(result[[0, 1]].abs() < 1e-6);
        assert!(result[[1, 0]].abs() < 1e-6);
    }

    #[test]
    fn test_matrix_exp_nilpotent() {
        // exp([[0,1],[0,0]]) = [[1,1],[0,1]]
        let a = Array2::from_shape_vec((2, 2), vec![0.0f32, 1.0, 0.0, 0.0]).unwrap();
        let result = matrix_exp_pade(&a, 13);
        let expected = [[1.0f32, 1.0], [0.0, 1.0]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (result[[i, j]] - expected[i][j]).abs() < 1e-6,
                    "nilpotent exp[{i},{j}] = {}, expected {}",
                    result[[i, j]],
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn test_matrix_exp_skew_symmetric() {
        // exp([[0,-0.5],[0.5,0]]) = rotation by 0.5 rad
        // = [[cos(0.5), -sin(0.5)], [sin(0.5), cos(0.5)]]
        let theta = 0.5f32;
        let a = Array2::from_shape_vec((2, 2), vec![0.0, -theta, theta, 0.0]).unwrap();
        let result = matrix_exp_pade(&a, 13);
        let c = theta.cos();
        let s = theta.sin();
        assert!(
            (result[[0, 0]] - c).abs() < 1e-5,
            "rotation[0,0]={}, expected {c}",
            result[[0, 0]]
        );
        assert!(
            (result[[0, 1]] - (-s)).abs() < 1e-5,
            "rotation[0,1]={}, expected {}",
            result[[0, 1]],
            -s
        );
        assert!(
            (result[[1, 0]] - s).abs() < 1e-5,
            "rotation[1,0]={}, expected {s}",
            result[[1, 0]]
        );
        assert!(
            (result[[1, 1]] - c).abs() < 1e-5,
            "rotation[1,1]={}, expected {c}",
            result[[1, 1]]
        );
    }

    #[test]
    fn test_matrix_exp_agrees_with_taylor() {
        // For small-norm A, Padé should agree closely with Taylor series
        let a = Array2::from_shape_vec((2, 2), vec![0.005f32, -0.003, 0.002, -0.007]).unwrap();
        let pade_result = matrix_exp_pade(&a, 13);
        let taylor_result = taylor_exp(&a, 16);
        for i in 0..2 {
            for j in 0..2 {
                let diff = (pade_result[[i, j]] - taylor_result[[i, j]]).abs();
                assert!(
                    diff < 1e-4,
                    "Padé vs Taylor [{i},{j}]: diff={diff}, pade={}, taylor={}",
                    pade_result[[i, j]],
                    taylor_result[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_matrix_exp_scaling_squaring() {
        // exp(5*I) = e^5 * I  (tests the scaling-and-squaring path)
        let n = 2usize;
        let a: Array2<f32> = Array2::<f32>::eye(n).mapv(|x| x * 5.0f32);
        let result = matrix_exp_pade(&a, 13);
        let e5 = 5.0f32.exp(); // e^5 ≈ 148.41
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { e5 } else { 0.0f32 };
                assert!(
                    (result[[i, j]] - expected).abs() < e5 * 1e-4,
                    "exp(5I)[{i},{j}] = {}, expected {expected}",
                    result[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_solve_linear_2x2() {
        // Solve [[2,1],[5,7]] X = [[11,4],[13,1]], known solution X = [[3,1],[5,2]]
        // Check: [[2,1],[5,7]] . [[3,1],[5,2]] = [[11,4],[50,19]] -- let's verify exact solution
        // Actually: [2*3+1*5, 2*1+1*2] = [11, 4] ✓
        //           [5*3+7*5, 5*1+7*2] = [50, 19] -- that doesn't match b[1] = [13, 1]
        // Use a well-conditioned example with known solution:
        // A = [[2,1],[1,3]], b = [[5,7],[10,14]] => x = [[1,1],[3,5]]
        // Check: [2*1+1*3, 2*1+1*5] = [5, 7] ✓; [1*1+3*3, 1*1+3*5] = [10, 16] -- nope
        // Use the original given example: [[2,1],[5,7]] X = [[11],[13]]
        let a = Array2::from_shape_vec((2, 2), vec![2.0f32, 1.0, 5.0, 7.0]).unwrap();
        // Solution to A x = [11,13]^T:  det(A) = 14-5 = 9
        //   x1 = (11*7 - 1*13)/9 = (77-13)/9 = 64/9
        //   x2 = (2*13 - 5*11)/9 = (26-55)/9 = -29/9
        let b_vec = Array2::from_shape_vec((2, 1), vec![11.0f32, 13.0]).unwrap();
        let x = solve_linear(&a, &b_vec);
        let residual = a.dot(&x);
        assert!(
            (residual[[0, 0]] - 11.0).abs() < 1e-4,
            "residual[0] = {}",
            residual[[0, 0]]
        );
        assert!(
            (residual[[1, 0]] - 13.0).abs() < 1e-4,
            "residual[1] = {}",
            residual[[1, 0]]
        );
    }

    #[test]
    fn test_zoh_discretize_updated() {
        // Scalar system A=[-1], B=[1], dt=0.1
        // Exact ZOH: A_d = exp(-0.1), B_d = 1 - exp(-0.1)
        let a = Array2::from_shape_vec((1, 1), vec![-1.0f32]).unwrap();
        let b = Array2::from_shape_vec((1, 1), vec![1.0f32]).unwrap();
        let dt = 0.1f32;

        let (a_d, b_d) = zoh_discretize(&a, &b, dt);

        let exp_neg01 = (-dt).exp();
        assert!(
            (a_d[[0, 0]] - exp_neg01).abs() < 1e-5,
            "A_d = {}, expected exp(-0.1) = {exp_neg01}",
            a_d[[0, 0]]
        );
        let b_d_exact = 1.0 - exp_neg01;
        assert!(
            (b_d[[0, 0]] - b_d_exact).abs() < 1e-5,
            "B_d = {}, expected 1-exp(-0.1) = {b_d_exact}",
            b_d[[0, 0]]
        );
    }
}
