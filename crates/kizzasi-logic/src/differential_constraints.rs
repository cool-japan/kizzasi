//! Differential Constraints
//!
//! This module provides constraints on derivatives and integrals:
//! - Higher-order derivative constraints
//! - Integral constraints over time windows
//! - Differential-algebraic constraints
//! - Path integral constraints

use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Order of derivative
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DerivativeOrder {
    /// First derivative (velocity)
    First,
    /// Second derivative (acceleration)
    Second,
    /// Third derivative (jerk)
    Third,
    /// Higher order
    Custom(usize),
}

impl DerivativeOrder {
    /// Get numeric order
    pub fn order(&self) -> usize {
        match self {
            Self::First => 1,
            Self::Second => 2,
            Self::Third => 3,
            Self::Custom(n) => *n,
        }
    }
}

/// Higher-order derivative constraint
#[derive(Debug, Clone)]
pub struct DerivativeConstraint {
    /// Name of the constraint
    name: String,
    /// Order of derivative
    order: DerivativeOrder,
    /// Time step for numerical differentiation
    dt: f32,
    /// Upper bound on derivative magnitude
    max_magnitude: f32,
    /// Historical values for computing derivatives
    history: VecDeque<(f32, Array1<f32>)>, // (time, value)
    /// Maximum history length
    max_history: usize,
}

impl DerivativeConstraint {
    /// Create a new derivative constraint
    pub fn new(
        name: impl Into<String>,
        order: DerivativeOrder,
        dt: f32,
        max_magnitude: f32,
    ) -> Self {
        let max_history = order.order() + 2;
        Self {
            name: name.into(),
            order,
            dt,
            max_magnitude,
            history: VecDeque::new(),
            max_history,
        }
    }

    /// Add a new observation
    pub fn observe(&mut self, time: f32, value: Array1<f32>) {
        self.history.push_back((time, value));
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Compute numerical derivative of given order
    fn compute_derivative(&self) -> Option<Array1<f32>> {
        let n = self.order.order();
        if self.history.len() < n + 1 {
            return None; // Not enough data
        }

        // Use finite differences to compute nth derivative
        // For simplicity, use backward differences
        let values: Vec<_> = self.history.iter().rev().take(n + 1).collect();

        match n {
            1 => {
                // First derivative: (v[0] - v[1]) / dt
                let (t0, v0) = values[0];
                let (t1, v1) = values[1];
                let dt = t0 - t1;
                Some((v0 - v1) / dt)
            }
            2 => {
                // Second derivative: ((v[0] - v[1]) - (v[1] - v[2])) / dt^2
                let (t0, v0) = values[0];
                let (t1, v1) = values[1];
                let (t2, v2) = values[2];
                let dt = (t0 - t1 + t1 - t2) / 2.0;
                Some(((v0 - v1) - (v1 - v2)) / (dt * dt))
            }
            3 => {
                // Third derivative (jerk)
                if values.len() < 4 {
                    return None;
                }
                let (_, v0) = values[0];
                let (_, v1) = values[1];
                let (_, v2) = values[2];
                let (_, v3) = values[3];
                Some((v0 - &(v1 * 3.0) + &(v2 * 3.0) - v3) / (self.dt * self.dt * self.dt))
            }
            _ => {
                // General n-th order backward finite difference
                // ∇ⁿ x[t] = sum_{k=0}^{n} (-1)^k * C(n,k) * x[t - k*dt]
                // Needs n+1 values (already checked above).
                //
                // values[0] = most recent (x[t]), values[k] = x[t - k*dt]
                // (collected via .iter().rev().take(n+1) above)
                //
                // Clamp to order 8 to prevent u64 overflow in binomial
                // coefficients and catastrophic numeric instability.
                let effective_n = n.min(8);

                if values.len() < effective_n + 1 {
                    return None;
                }

                // Compute binomial coefficients C(effective_n, k) via the
                // multiplicative recurrence:
                //   C(n, 0) = 1
                //   C(n, k) = C(n, k-1) * (n - k + 1) / k
                let mut binom = vec![0_u64; effective_n + 1];
                binom[0] = 1;
                for k in 1..=effective_n {
                    binom[k] = binom[k - 1].saturating_mul((effective_n - k + 1) as u64) / k as u64;
                }

                let dt_n = self.dt.powi(effective_n as i32);

                let dim = values[0].1.len();
                let mut result = Array1::<f32>::zeros(dim);
                for k in 0..=effective_n {
                    let sign = if k % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
                    let coeff = sign * binom[k] as f32;
                    result = result + &values[k].1 * coeff;
                }

                Some(result / dt_n)
            }
        }
    }

    /// Check if derivative constraint is satisfied
    pub fn check(&self) -> bool {
        if let Some(derivative) = self.compute_derivative() {
            let magnitude = derivative.iter().map(|x| x * x).sum::<f32>().sqrt();
            magnitude <= self.max_magnitude
        } else {
            true // Not enough data, trivially satisfied
        }
    }

    /// Compute violation amount
    pub fn violation(&self) -> f32 {
        if let Some(derivative) = self.compute_derivative() {
            let magnitude = derivative.iter().map(|x| x * x).sum::<f32>().sqrt();
            (magnitude - self.max_magnitude).max(0.0)
        } else {
            0.0
        }
    }

    /// Get current derivative estimate
    pub fn get_derivative(&self) -> Option<Array1<f32>> {
        self.compute_derivative()
    }

    /// Get name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Reset history
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

/// Integral constraint over a time window
#[derive(Debug, Clone)]
pub struct IntegralConstraint {
    /// Name of the constraint
    name: String,
    /// Time window for integration
    window_duration: f32,
    /// Upper bound on integral value
    max_integral: f32,
    /// Lower bound on integral value
    min_integral: f32,
    /// Historical values for integration
    history: VecDeque<(f32, Array1<f32>)>, // (time, value)
}

impl IntegralConstraint {
    /// Create a new integral constraint
    pub fn new(
        name: impl Into<String>,
        window_duration: f32,
        min_integral: f32,
        max_integral: f32,
    ) -> Self {
        Self {
            name: name.into(),
            window_duration,
            max_integral,
            min_integral,
            history: VecDeque::new(),
        }
    }

    /// Add a new observation
    pub fn observe(&mut self, time: f32, value: Array1<f32>) {
        self.history.push_back((time, value));

        // Remove values outside the window
        let cutoff_time = time - self.window_duration;
        while let Some((t, _)) = self.history.front() {
            if *t < cutoff_time {
                self.history.pop_front();
            } else {
                break;
            }
        }
    }

    /// Compute integral using trapezoidal rule
    fn compute_integral(&self) -> Option<Array1<f32>> {
        if self.history.len() < 2 {
            return None;
        }

        let dim = self.history[0].1.len();
        let mut integral = Array1::zeros(dim);

        for i in 0..self.history.len() - 1 {
            let (t1, v1) = &self.history[i];
            let (t2, v2) = &self.history[i + 1];
            let dt = t2 - t1;
            // Trapezoidal rule: (v1 + v2) / 2 * dt
            integral += &((v1 + v2) * (dt / 2.0));
        }

        Some(integral)
    }

    /// Check if integral constraint is satisfied
    pub fn check(&self) -> bool {
        if let Some(integral) = self.compute_integral() {
            integral
                .iter()
                .all(|&x| x >= self.min_integral && x <= self.max_integral)
        } else {
            true
        }
    }

    /// Compute violation amount
    pub fn violation(&self) -> f32 {
        if let Some(integral) = self.compute_integral() {
            let mut total_violation = 0.0;
            for &x in integral.iter() {
                if x < self.min_integral {
                    total_violation += self.min_integral - x;
                } else if x > self.max_integral {
                    total_violation += x - self.max_integral;
                }
            }
            total_violation
        } else {
            0.0
        }
    }

    /// Get current integral estimate
    pub fn get_integral(&self) -> Option<Array1<f32>> {
        self.compute_integral()
    }

    /// Get name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Reset history
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

/// Differential-algebraic constraint (DAE)
/// Represents constraints of the form: F(x, dx/dt, t) = 0
#[derive(Debug, Clone)]
pub struct DifferentialAlgebraicConstraint {
    /// Name of the constraint
    name: String,
    /// Constraint function: F(x, dx/dt, t) -> residual
    constraint_fn: fn(&Array1<f32>, &Array1<f32>, f32) -> Array1<f32>,
    /// Tolerance for residual
    tolerance: f32,
    /// Historical data for computing derivative
    history: VecDeque<(f32, Array1<f32>)>,
    /// Time step
    #[allow(dead_code)]
    dt: f32,
}

impl DifferentialAlgebraicConstraint {
    /// Create a new DAE constraint
    pub fn new(
        name: impl Into<String>,
        constraint_fn: fn(&Array1<f32>, &Array1<f32>, f32) -> Array1<f32>,
        tolerance: f32,
        dt: f32,
    ) -> Self {
        Self {
            name: name.into(),
            constraint_fn,
            tolerance,
            history: VecDeque::new(),
            dt,
        }
    }

    /// Add observation
    pub fn observe(&mut self, time: f32, value: Array1<f32>) {
        self.history.push_back((time, value));
        if self.history.len() > 2 {
            self.history.pop_front();
        }
    }

    /// Compute current derivative estimate
    fn compute_derivative(&self) -> Option<Array1<f32>> {
        if self.history.len() < 2 {
            return None;
        }

        let (t1, v1) = &self.history[0];
        let (t2, v2) = &self.history[1];
        let dt = t2 - t1;
        Some((v2 - v1) / dt)
    }

    /// Check constraint
    pub fn check(&self) -> bool {
        if self.history.is_empty() {
            return true;
        }

        let (t, x) = &self.history[self.history.len() - 1];

        if let Some(dx_dt) = self.compute_derivative() {
            let residual = (self.constraint_fn)(x, &dx_dt, *t);
            let residual_norm = residual.iter().map(|r| r * r).sum::<f32>().sqrt();
            residual_norm <= self.tolerance
        } else {
            true
        }
    }

    /// Compute violation
    pub fn violation(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }

        let (t, x) = &self.history[self.history.len() - 1];

        if let Some(dx_dt) = self.compute_derivative() {
            let residual = (self.constraint_fn)(x, &dx_dt, *t);
            let residual_norm = residual.iter().map(|r| r * r).sum::<f32>().sqrt();
            (residual_norm - self.tolerance).max(0.0)
        } else {
            0.0
        }
    }

    /// Get name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Reset
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

/// Path integral constraint for trajectory optimization
#[derive(Debug, Clone)]
pub struct PathIntegralConstraint {
    /// Name of the constraint
    name: String,
    /// Cost function: cost(x, dx/dt, t) -> scalar
    cost_fn: fn(&Array1<f32>, &Array1<f32>, f32) -> f32,
    /// Maximum allowed path integral (total cost)
    max_cost: f32,
    /// Historical trajectory data
    trajectory: VecDeque<(f32, Array1<f32>)>,
}

impl PathIntegralConstraint {
    /// Create a new path integral constraint
    pub fn new(
        name: impl Into<String>,
        cost_fn: fn(&Array1<f32>, &Array1<f32>, f32) -> f32,
        max_cost: f32,
    ) -> Self {
        Self {
            name: name.into(),
            cost_fn,
            max_cost,
            trajectory: VecDeque::new(),
        }
    }

    /// Add trajectory point
    pub fn observe(&mut self, time: f32, state: Array1<f32>) {
        self.trajectory.push_back((time, state));
    }

    /// Compute path integral
    fn compute_path_integral(&self) -> f32 {
        if self.trajectory.len() < 2 {
            return 0.0;
        }

        let mut total_cost = 0.0;

        for i in 0..self.trajectory.len() - 1 {
            let (t1, x1) = &self.trajectory[i];
            let (t2, x2) = &self.trajectory[i + 1];

            let dt = t2 - t1;
            let dx_dt = (x2 - x1) / dt;

            // Evaluate cost at midpoint
            let t_mid = (t1 + t2) / 2.0;
            let x_mid = (x1 + x2) / 2.0;

            let cost = (self.cost_fn)(&x_mid, &dx_dt, t_mid);
            total_cost += cost * dt;
        }

        total_cost
    }

    /// Check constraint
    pub fn check(&self) -> bool {
        let cost = self.compute_path_integral();
        cost <= self.max_cost
    }

    /// Compute violation
    pub fn violation(&self) -> f32 {
        let cost = self.compute_path_integral();
        (cost - self.max_cost).max(0.0)
    }

    /// Get current path cost
    pub fn get_path_cost(&self) -> f32 {
        self.compute_path_integral()
    }

    /// Get name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Reset trajectory
    pub fn reset(&mut self) {
        self.trajectory.clear();
    }

    /// Get trajectory length
    pub fn trajectory_length(&self) -> usize {
        self.trajectory.len()
    }
}

/// Constraint set for differential constraints
#[derive(Debug, Clone)]
pub struct DifferentialConstraintSet {
    /// Derivative constraints
    derivative_constraints: Vec<DerivativeConstraint>,
    /// Integral constraints
    integral_constraints: Vec<IntegralConstraint>,
    /// DAE constraints
    dae_constraints: Vec<DifferentialAlgebraicConstraint>,
    /// Path integral constraints
    path_integral_constraints: Vec<PathIntegralConstraint>,
}

impl DifferentialConstraintSet {
    /// Create a new differential constraint set
    pub fn new() -> Self {
        Self {
            derivative_constraints: Vec::new(),
            integral_constraints: Vec::new(),
            dae_constraints: Vec::new(),
            path_integral_constraints: Vec::new(),
        }
    }

    /// Add a derivative constraint
    pub fn add_derivative(&mut self, constraint: DerivativeConstraint) {
        self.derivative_constraints.push(constraint);
    }

    /// Add an integral constraint
    pub fn add_integral(&mut self, constraint: IntegralConstraint) {
        self.integral_constraints.push(constraint);
    }

    /// Add a DAE constraint
    pub fn add_dae(&mut self, constraint: DifferentialAlgebraicConstraint) {
        self.dae_constraints.push(constraint);
    }

    /// Add a path integral constraint
    pub fn add_path_integral(&mut self, constraint: PathIntegralConstraint) {
        self.path_integral_constraints.push(constraint);
    }

    /// Observe new state at time t
    pub fn observe(&mut self, time: f32, state: Array1<f32>) {
        for constraint in &mut self.derivative_constraints {
            constraint.observe(time, state.clone());
        }
        for constraint in &mut self.integral_constraints {
            constraint.observe(time, state.clone());
        }
        for constraint in &mut self.dae_constraints {
            constraint.observe(time, state.clone());
        }
        for constraint in &mut self.path_integral_constraints {
            constraint.observe(time, state.clone());
        }
    }

    /// Check all constraints
    pub fn check_all(&self) -> bool {
        self.derivative_constraints.iter().all(|c| c.check())
            && self.integral_constraints.iter().all(|c| c.check())
            && self.dae_constraints.iter().all(|c| c.check())
            && self.path_integral_constraints.iter().all(|c| c.check())
    }

    /// Compute total violation
    pub fn total_violation(&self) -> f32 {
        let mut total = 0.0;
        for c in &self.derivative_constraints {
            total += c.violation();
        }
        for c in &self.integral_constraints {
            total += c.violation();
        }
        for c in &self.dae_constraints {
            total += c.violation();
        }
        for c in &self.path_integral_constraints {
            total += c.violation();
        }
        total
    }

    /// Reset all constraints
    pub fn reset(&mut self) {
        for c in &mut self.derivative_constraints {
            c.reset();
        }
        for c in &mut self.integral_constraints {
            c.reset();
        }
        for c in &mut self.dae_constraints {
            c.reset();
        }
        for c in &mut self.path_integral_constraints {
            c.reset();
        }
    }

    /// Get number of constraints
    pub fn num_constraints(&self) -> usize {
        self.derivative_constraints.len()
            + self.integral_constraints.len()
            + self.dae_constraints.len()
            + self.path_integral_constraints.len()
    }
}

impl Default for DifferentialConstraintSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_constraint() {
        let mut constraint =
            DerivativeConstraint::new("velocity_limit", DerivativeOrder::First, 0.1, 10.0);

        // Add observations
        constraint.observe(0.0, Array1::from_vec(vec![0.0]));
        constraint.observe(0.1, Array1::from_vec(vec![0.5])); // velocity = 5.0

        assert!(constraint.check()); // 5.0 <= 10.0

        constraint.observe(0.2, Array1::from_vec(vec![2.0])); // velocity = 15.0
        assert!(!constraint.check()); // 15.0 > 10.0
    }

    #[test]
    fn test_integral_constraint() {
        let mut constraint = IntegralConstraint::new("energy_limit", 1.0, 0.0, 100.0);

        constraint.observe(0.0, Array1::from_vec(vec![10.0]));
        constraint.observe(0.5, Array1::from_vec(vec![20.0]));
        constraint.observe(1.0, Array1::from_vec(vec![10.0]));

        assert!(constraint.check());
        assert!(constraint.get_integral().is_some());
    }

    #[test]
    fn test_dae_constraint() {
        // Simple DAE: x + dx/dt = 0
        fn dae_fn(x: &Array1<f32>, dx_dt: &Array1<f32>, _t: f32) -> Array1<f32> {
            x + dx_dt
        }

        let mut constraint = DifferentialAlgebraicConstraint::new("simple_dae", dae_fn, 1.0, 0.1);

        constraint.observe(0.0, Array1::from_vec(vec![1.0]));
        constraint.observe(0.1, Array1::from_vec(vec![0.9])); // dx/dt = -1.0, satisfies x + dx/dt ≈ 0

        assert!(constraint.check());
    }

    #[test]
    fn test_path_integral_constraint() {
        // Cost function: ||dx/dt||^2
        fn cost_fn(_x: &Array1<f32>, dx_dt: &Array1<f32>, _t: f32) -> f32 {
            dx_dt.iter().map(|v| v * v).sum()
        }

        let mut constraint = PathIntegralConstraint::new("min_energy", cost_fn, 100.0);

        constraint.observe(0.0, Array1::from_vec(vec![0.0]));
        constraint.observe(0.1, Array1::from_vec(vec![1.0]));
        constraint.observe(0.2, Array1::from_vec(vec![2.0]));

        assert!(constraint.check());
        assert_eq!(constraint.trajectory_length(), 3);
    }

    #[test]
    fn test_differential_constraint_set() {
        let mut set = DifferentialConstraintSet::new();

        set.add_derivative(DerivativeConstraint::new(
            "velocity",
            DerivativeOrder::First,
            0.1,
            10.0,
        ));

        set.observe(0.0, Array1::from_vec(vec![0.0]));
        set.observe(0.1, Array1::from_vec(vec![0.5]));

        assert!(set.check_all());
        assert_eq!(set.num_constraints(), 1);
    }

    #[test]
    fn test_derivative_constraint_4th_order() {
        // 4th derivative of a quadratic x(t) = t^2 is 0.
        // Feed samples at dt=1.0 so x[k] = k^2: [0, 1, 4, 9, 16, 25].
        // max_history = 4+2 = 6, exactly fits all 6 points.
        let dt = 1.0_f32;
        let mut constraint =
            DerivativeConstraint::new("jounce", DerivativeOrder::Custom(4), dt, 1000.0);

        for step in 0_i32..6 {
            constraint.observe(step as f32 * dt, Array1::from_vec(vec![step.pow(2) as f32]));
        }

        let deriv = constraint.get_derivative();
        assert!(
            deriv.is_some(),
            "expected Some derivative after 6 observations"
        );
        let d = deriv.unwrap();
        // 4th derivative of quadratic is exactly 0; allow generous float tolerance
        assert!(
            d[0].abs() < 1.0,
            "4th derivative of quadratic should be ~0, got {}",
            d[0]
        );
    }

    #[test]
    fn test_derivative_constraint_insufficient_history() {
        // Custom(4) requires at least 5 observations; feeding only 3 must yield None.
        let mut constraint =
            DerivativeConstraint::new("jounce", DerivativeOrder::Custom(4), 1.0, 1000.0);

        for step in 0_i32..3 {
            constraint.observe(step as f32, Array1::from_vec(vec![step as f32]));
        }

        assert!(
            constraint.get_derivative().is_none(),
            "should return None when fewer than n+1 observations are available"
        );
    }

    #[test]
    fn test_derivative_constraint_custom_check() {
        // Feed a near-impulse sequence [0,0,0,0,0,1] with tight max_magnitude.
        // The 4th-order finite difference over this will produce a large value,
        // causing a violation when max_magnitude = 0.001.
        let dt = 0.1_f32;
        let mut constraint =
            DerivativeConstraint::new("jounce_check", DerivativeOrder::Custom(4), dt, 0.001);

        let signal: [f32; 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for (i, &val) in signal.iter().enumerate() {
            constraint.observe(i as f32 * dt, Array1::from_vec(vec![val]));
        }

        // The derivative must be Some (we have 6 >= 5 observations).
        let deriv = constraint.get_derivative();
        assert!(
            deriv.is_some(),
            "expected Some derivative for impulse signal"
        );

        // The 4th-order backward difference of [0,0,0,0,0,1] relative to dt=0.1
        // is substantial; violation must be positive and check() must be false.
        assert!(
            constraint.violation() > 0.0,
            "expected positive violation for impulse signal with tight max_magnitude"
        );
        assert!(
            !constraint.check(),
            "check() should fail for impulse signal with max_magnitude=0.001"
        );
    }
}
