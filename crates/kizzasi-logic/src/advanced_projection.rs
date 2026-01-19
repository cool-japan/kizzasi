//! Advanced projection algorithms for constraint satisfaction
//!
//! This module implements sophisticated projection methods for finding
//! points that satisfy multiple constraints simultaneously.

use crate::constraint::{
    LinearConstraint, NonlinearConstraint, SetMembershipConstraint, ViolationComputable,
};
use crate::error::LogicResult;
use scirs2_core::ndarray::Array1;

// ============================================================================
// Dykstra's Alternating Projection Algorithm
// ============================================================================

/// Dykstra's alternating projection algorithm for convex sets
///
/// Finds the closest point in the intersection of multiple convex sets
/// by alternating projections with incremental corrections.
pub struct DykstraProjection<C> {
    constraints: Vec<C>,
    max_iterations: usize,
    tolerance: f32,
}

impl<C: ViolationComputable> DykstraProjection<C> {
    /// Create a new Dykstra projection solver
    pub fn new(constraints: Vec<C>) -> Self {
        Self {
            constraints,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    /// Get the number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
}

impl DykstraProjection<LinearConstraint> {
    /// Project onto intersection using Dykstra's algorithm for linear constraints
    pub fn project(&self, x: &Array1<f32>) -> LogicResult<Array1<f32>> {
        let n = x.len();
        let m = self.constraints.len();

        // Current iterate
        let mut y = x.clone();

        // Increment vectors (one per constraint)
        let mut increments: Vec<Array1<f32>> = vec![Array1::zeros(n); m];

        for _iter in 0..self.max_iterations {
            let y_old = y.clone();

            // Cycle through all constraints
            for (i, constraint) in self.constraints.iter().enumerate() {
                // Add increment
                let z = &y + &increments[i];

                // Project onto constraint
                let projected = constraint.project(
                    z.as_slice()
                        .expect("Array must have contiguous layout for projection"),
                );
                let p = Array1::from_vec(projected);

                // Update increment
                increments[i] = &z - &p;

                // Update iterate
                y = p;
            }

            // Check convergence
            let diff = (&y - &y_old)
                .iter()
                .map(|&d| d.abs())
                .fold(0.0f32, |a, b| a.max(b));

            if diff < self.tolerance {
                break;
            }
        }

        Ok(y)
    }
}

impl DykstraProjection<SetMembershipConstraint> {
    /// Project onto intersection using Dykstra's algorithm for set constraints
    pub fn project(&self, x: &Array1<f32>) -> LogicResult<Array1<f32>> {
        let n = x.len();
        let m = self.constraints.len();

        let mut y = x.clone();
        let mut increments: Vec<Array1<f32>> = vec![Array1::zeros(n); m];

        for _iter in 0..self.max_iterations {
            let y_old = y.clone();

            for (i, constraint) in self.constraints.iter().enumerate() {
                let z = &y + &increments[i];
                let projected = constraint.project(
                    z.as_slice()
                        .expect("Array must have contiguous layout for projection"),
                );
                let p = Array1::from_vec(projected);
                increments[i] = &z - &p;
                y = p;
            }

            let diff = (&y - &y_old)
                .iter()
                .map(|&d| d.abs())
                .fold(0.0f32, |a, b| a.max(b));

            if diff < self.tolerance {
                break;
            }
        }

        Ok(y)
    }
}

// ============================================================================
// Gradient-Based Projection
// ============================================================================

/// Gradient-based projection for smooth nonlinear constraints
///
/// Uses gradient descent to find the closest point satisfying constraints.
pub struct GradientProjection {
    max_iterations: usize,
    step_size: f32,
    tolerance: f32,
}

impl GradientProjection {
    /// Create a new gradient projection solver
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            step_size: 0.01,
            tolerance: 1e-6,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set step size for gradient descent
    pub fn with_step_size(mut self, step: f32) -> Self {
        self.step_size = step;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    /// Project onto constraints using gradient descent
    pub fn project(
        &self,
        x: &Array1<f32>,
        constraints: &[NonlinearConstraint],
    ) -> LogicResult<Array1<f32>> {
        let mut result = x.clone();

        for _iter in 0..self.max_iterations {
            // Check if all constraints satisfied
            if constraints.iter().all(|c| {
                c.check(
                    result
                        .as_slice()
                        .expect("Array must have contiguous layout for projection"),
                )
            }) {
                break;
            }

            let prev = result.clone();

            // Compute total gradient from all violated constraints
            let mut total_grad: Array1<f32> = Array1::zeros(x.len());
            let mut has_gradient = false;

            for constraint in constraints {
                if !constraint.check(
                    result
                        .as_slice()
                        .expect("Array must have contiguous layout for projection"),
                ) {
                    if let Some(grad) = constraint.gradient(
                        result
                            .as_slice()
                            .expect("Array must have contiguous layout for projection"),
                    ) {
                        let violation = constraint.violation(
                            result
                                .as_slice()
                                .expect("Array must have contiguous layout for projection"),
                        );
                        for (i, &gi) in grad.iter().enumerate() {
                            total_grad[i] += violation * gi;
                        }
                        has_gradient = true;
                    }
                }
            }

            if !has_gradient {
                // No gradients available, cannot continue
                break;
            }

            // Gradient descent step
            for (ri, &gi) in result.iter_mut().zip(total_grad.iter()) {
                *ri -= self.step_size * gi;
            }

            // Check convergence
            let diff = (&result - &prev)
                .iter()
                .map(|&d| d.abs())
                .fold(0.0f32, |a, b| a.max(b));

            if diff < self.tolerance {
                break;
            }
        }

        Ok(result)
    }

    /// Project with adaptive step size (line search)
    pub fn project_adaptive(
        &self,
        x: &Array1<f32>,
        constraints: &[NonlinearConstraint],
    ) -> LogicResult<Array1<f32>> {
        let mut result = x.clone();
        let mut step_size = self.step_size;

        for _iter in 0..self.max_iterations {
            if constraints.iter().all(|c| {
                c.check(
                    result
                        .as_slice()
                        .expect("Array must have contiguous layout for projection"),
                )
            }) {
                break;
            }

            // Compute gradient
            let mut total_grad: Array1<f32> = Array1::zeros(x.len());
            let mut current_violation = 0.0;

            for constraint in constraints {
                let viol = constraint.violation(
                    result
                        .as_slice()
                        .expect("Array must have contiguous layout for projection"),
                );
                current_violation += viol;

                if viol > 0.0 {
                    if let Some(grad) = constraint.gradient(
                        result
                            .as_slice()
                            .expect("Array must have contiguous layout for projection"),
                    ) {
                        for (i, &gi) in grad.iter().enumerate() {
                            total_grad[i] += viol * gi;
                        }
                    }
                }
            }

            // Backtracking line search
            let mut alpha = step_size;
            for _ in 0..10 {
                let mut candidate = result.clone();
                for (ci, &gi) in candidate.iter_mut().zip(total_grad.iter()) {
                    *ci -= alpha * gi;
                }

                // Check if violation decreased
                let new_violation: f32 = constraints
                    .iter()
                    .map(|c| {
                        c.violation(
                            candidate
                                .as_slice()
                                .expect("Array must have contiguous layout for projection"),
                        )
                    })
                    .sum();

                if new_violation < current_violation {
                    result = candidate;
                    step_size = (alpha * 1.1).min(1.0); // Increase step size
                    break;
                }

                alpha *= 0.5; // Reduce step size
            }
        }

        Ok(result)
    }
}

impl Default for GradientProjection {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Augmented Lagrangian Method
// ============================================================================

/// Augmented Lagrangian method for constrained optimization
///
/// Solves: min f(x) subject to g(x) <= 0
/// Using penalty + Lagrange multipliers
pub struct AugmentedLagrangian {
    max_outer_iterations: usize,
    max_inner_iterations: usize,
    penalty_parameter: f32,
    penalty_increase_factor: f32,
    tolerance: f32,
    step_size: f32,
}

impl AugmentedLagrangian {
    /// Create a new augmented Lagrangian solver
    pub fn new() -> Self {
        Self {
            max_outer_iterations: 20,
            max_inner_iterations: 100,
            penalty_parameter: 1.0,
            penalty_increase_factor: 10.0,
            tolerance: 1e-5,
            step_size: 0.01,
        }
    }

    /// Set maximum outer iterations
    pub fn with_max_outer_iterations(mut self, max_iter: usize) -> Self {
        self.max_outer_iterations = max_iter;
        self
    }

    /// Set penalty parameter
    pub fn with_penalty_parameter(mut self, rho: f32) -> Self {
        self.penalty_parameter = rho;
        self
    }

    /// Project x to satisfy constraints using augmented Lagrangian
    ///
    /// Minimizes ||x - x0||² subject to constraints
    pub fn project(
        &self,
        x0: &Array1<f32>,
        constraints: &[NonlinearConstraint],
    ) -> LogicResult<Array1<f32>> {
        let n = x0.len();
        let m = constraints.len();

        let mut x = x0.clone();
        let mut lambda = vec![0.0f32; m]; // Lagrange multipliers
        let mut rho = self.penalty_parameter;

        for _outer in 0..self.max_outer_iterations {
            // Minimize augmented Lagrangian using gradient descent
            for _inner in 0..self.max_inner_iterations {
                // Compute gradient of augmented Lagrangian
                let mut grad = Array1::zeros(n);

                // Gradient of ||x - x0||²: 2(x - x0)
                for (i, (&xi, &x0i)) in x.iter().zip(x0.iter()).enumerate() {
                    grad[i] = 2.0 * (xi - x0i);
                }

                // Add constraint gradients
                for (j, constraint) in constraints.iter().enumerate() {
                    let g_j = constraint.evaluate(
                        x.as_slice()
                            .expect("Array must have contiguous layout for projection"),
                    );

                    if let Some(grad_g) = constraint.gradient(
                        x.as_slice()
                            .expect("Array must have contiguous layout for projection"),
                    ) {
                        let factor = lambda[j] + rho * g_j.max(0.0);
                        for (i, &dg) in grad_g.iter().enumerate() {
                            grad[i] += factor * dg;
                        }
                    }
                }

                // Gradient descent step
                for (xi, &gi) in x.iter_mut().zip(grad.iter()) {
                    *xi -= self.step_size * gi;
                }
            }

            // Update Lagrange multipliers
            for (j, constraint) in constraints.iter().enumerate() {
                let g_j = constraint.evaluate(
                    x.as_slice()
                        .expect("Array must have contiguous layout for projection"),
                );
                lambda[j] = (lambda[j] + rho * g_j).max(0.0);
            }

            // Check convergence
            let max_violation: f32 = constraints
                .iter()
                .map(|c| {
                    c.violation(
                        x.as_slice()
                            .expect("Array must have contiguous layout for projection"),
                    )
                })
                .fold(0.0f32, |a, b| a.max(b));

            if max_violation < self.tolerance {
                break;
            }

            // Increase penalty parameter
            rho *= self.penalty_increase_factor;
        }

        Ok(x)
    }
}

impl Default for AugmentedLagrangian {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::{GeometricSet, LinearConstraint};

    #[test]
    fn test_dykstra_linear_constraints() {
        // Two intersecting halfspaces: x <= 5 and x >= 0
        let c1 = LinearConstraint::less_eq(vec![1.0], 5.0);
        let c2 = LinearConstraint::greater_eq(vec![1.0], 0.0);

        let dykstra = DykstraProjection::new(vec![c1, c2]).with_tolerance(1e-6);

        // Point outside: x = -1 should project to 0
        let x = Array1::from_vec(vec![-1.0]);
        let projected = dykstra.project(&x).unwrap();
        assert!((projected[0] - 0.0).abs() < 1e-5);

        // Point outside: x = 10 should project to 5
        let x = Array1::from_vec(vec![10.0]);
        let projected = dykstra.project(&x).unwrap();
        assert!((projected[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_gradient_projection() {
        // Nonlinear constraint: x² <= 1 (i.e., x² - 1 <= 0)
        let constraint =
            NonlinearConstraint::inequality("x_squared", |x: &[f32]| x[0] * x[0] - 1.0)
                .with_gradient(|x: &[f32]| vec![2.0 * x[0]]);

        let proj = GradientProjection::new()
            .with_max_iterations(100)
            .with_step_size(0.1);

        // Point outside: x = 2 should project closer to 1
        let x = Array1::from_vec(vec![2.0]);
        let projected = proj.project(&x, &[constraint]).unwrap();
        // After projection, should be closer to feasible region
        assert!(projected[0] < 2.0); // Should move toward constraint
        assert!((projected[0] * projected[0] - 1.0).abs() < 0.5); // Should be closer to x²=1
    }

    #[test]
    fn test_dykstra_set_constraints() {
        // Two balls: one centered at origin with radius 2, another at (3, 0) with radius 2
        let set1 = GeometricSet::ball(vec![0.0, 0.0], 2.0);
        let set2 = GeometricSet::ball(vec![3.0, 0.0], 2.0);

        let c1 = SetMembershipConstraint::new("ball1", set1);
        let c2 = SetMembershipConstraint::new("ball2", set2);

        let dykstra = DykstraProjection::new(vec![c1, c2]).with_tolerance(1e-5);

        // Point in the middle should stay roughly in the middle
        let x = Array1::from_vec(vec![1.5, 0.0]);
        let projected = dykstra.project(&x).unwrap();

        // Should be close to intersection region
        assert!(projected[0] >= 1.0 && projected[0] <= 2.0);
        assert!(projected[1].abs() < 0.5);
    }
}
