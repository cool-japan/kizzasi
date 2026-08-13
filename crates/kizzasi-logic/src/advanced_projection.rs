//! Advanced projection algorithms for constraint satisfaction
//!
//! This module implements sophisticated projection methods for finding
//! points that satisfy multiple constraints simultaneously.

use crate::constraint::{
    LinearConstraint, NonlinearConstraint, SetMembershipConstraint, ViolationComputable,
};
use crate::error::{LogicError, LogicResult};
use scirs2_core::ndarray::Array1;

/// Return `arr` as a contiguous slice, or a recoverable error instead of
/// panicking.
///
/// An owned `Array1<f32>` is not guaranteed contiguous (e.g. one produced by
/// `slice_move(s![..;2])` has stride 2), so `as_slice()` can return `None`.
/// Every public entry point in this module normalizes its input to standard
/// layout once up front (`x.as_standard_layout().into_owned()`), after which
/// every array derived from it by in-place mutation or by arithmetic on two
/// already-standard-layout operands stays contiguous — so in practice this
/// should always succeed. It still returns `LogicResult` instead of
/// `.expect()`-panicking so a violation of that invariant becomes a
/// recoverable error, not a process abort.
fn require_contiguous(arr: &Array1<f32>) -> LogicResult<&[f32]> {
    arr.as_slice().ok_or_else(|| {
        LogicError::InvalidInput(
            "array must be in contiguous (standard) layout for projection".to_string(),
        )
    })
}

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

        // Current iterate, normalized to standard layout once so every
        // derived array below stays contiguous (see `require_contiguous`).
        let mut y = x.as_standard_layout().into_owned();

        // Increment vectors (one per constraint)
        let mut increments: Vec<Array1<f32>> = vec![Array1::zeros(n); m];

        for _iter in 0..self.max_iterations {
            let y_old = y.clone();

            // Cycle through all constraints
            for (i, constraint) in self.constraints.iter().enumerate() {
                // Add increment
                let z = &y + &increments[i];

                // Project onto constraint
                let projected = constraint.project(require_contiguous(&z)?);
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

        let mut y = x.as_standard_layout().into_owned();
        let mut increments: Vec<Array1<f32>> = vec![Array1::zeros(n); m];

        for _iter in 0..self.max_iterations {
            let y_old = y.clone();

            for (i, constraint) in self.constraints.iter().enumerate() {
                let z = &y + &increments[i];
                let projected = constraint.project(require_contiguous(&z)?);
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
        let mut result = x.as_standard_layout().into_owned();

        for _iter in 0..self.max_iterations {
            let result_slice = require_contiguous(&result)?;
            // Check if all constraints satisfied
            if constraints.iter().all(|c| c.check(result_slice)) {
                break;
            }

            let prev = result.clone();

            // Compute total gradient from all violated constraints
            let mut total_grad: Array1<f32> = Array1::zeros(x.len());
            let mut has_gradient = false;

            for constraint in constraints {
                if !constraint.check(result_slice) {
                    if let Some(grad) = constraint.gradient(result_slice) {
                        let violation = constraint.violation(result_slice);
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
        let mut result = x.as_standard_layout().into_owned();
        let mut step_size = self.step_size;

        for _iter in 0..self.max_iterations {
            let result_slice = require_contiguous(&result)?;
            if constraints.iter().all(|c| c.check(result_slice)) {
                break;
            }

            // Compute gradient
            let mut total_grad: Array1<f32> = Array1::zeros(x.len());
            let mut current_violation = 0.0;

            for constraint in constraints {
                let viol = constraint.violation(result_slice);
                current_violation += viol;

                if viol > 0.0 {
                    if let Some(grad) = constraint.gradient(result_slice) {
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
                let candidate_slice = require_contiguous(&candidate)?;
                let new_violation: f32 = constraints
                    .iter()
                    .map(|c| c.violation(candidate_slice))
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

/// Maximum backtracking halvings tried per inner-loop gradient step of
/// [`AugmentedLagrangian::project`] before giving up on that step.
const MAX_LINE_SEARCH_STEPS: usize = 20;

/// Floor applied to the augmented Lagrangian penalty parameter so it can
/// never reach zero (which would divide by zero below) regardless of what a
/// caller passes to `with_penalty_parameter`/`with_penalty_increase_factor`.
const MIN_PENALTY_PARAMETER: f32 = 1e-6;

/// Augmented Lagrangian method for constrained optimization
///
/// Solves: min f(x) subject to g(x) <= 0
/// Using penalty + Lagrange multipliers
pub struct AugmentedLagrangian {
    max_outer_iterations: usize,
    max_inner_iterations: usize,
    penalty_parameter: f32,
    penalty_increase_factor: f32,
    max_penalty_parameter: f32,
    tolerance: f32,
    inner_tolerance: f32,
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
            max_penalty_parameter: 1e8,
            tolerance: 1e-5,
            inner_tolerance: 1e-6,
            step_size: 0.01,
        }
    }

    /// Set maximum outer iterations
    pub fn with_max_outer_iterations(mut self, max_iter: usize) -> Self {
        self.max_outer_iterations = max_iter;
        self
    }

    /// Set the maximum number of inner (fixed-multiplier) gradient-descent
    /// iterations performed per outer iteration.
    pub fn with_max_inner_iterations(mut self, max_iter: usize) -> Self {
        self.max_inner_iterations = max_iter;
        self
    }

    /// Set penalty parameter
    pub fn with_penalty_parameter(mut self, rho: f32) -> Self {
        self.penalty_parameter = rho;
        self
    }

    /// Set the factor `rho` is multiplied by after an outer iteration that
    /// has not yet converged.
    pub fn with_penalty_increase_factor(mut self, factor: f32) -> Self {
        self.penalty_increase_factor = factor;
        self
    }

    /// Cap `rho` so it cannot grow without bound across outer iterations
    /// (an unbounded `rho` is what let the inner loop's fixed step size
    /// overshoot into `inf`/`NaN` before this fix).
    pub fn with_max_penalty_parameter(mut self, max_rho: f32) -> Self {
        self.max_penalty_parameter = max_rho;
        self
    }

    /// Set the outer-loop convergence tolerance, on the maximum constraint violation.
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set the inner-loop convergence tolerance, on the augmented
    /// Lagrangian's gradient norm at fixed multipliers.
    pub fn with_inner_tolerance(mut self, tol: f32) -> Self {
        self.inner_tolerance = tol;
        self
    }

    /// Set the base gradient-descent step size used by the inner loop's
    /// backtracking line search.
    pub fn with_step_size(mut self, step: f32) -> Self {
        self.step_size = step;
        self
    }

    /// Project x to satisfy constraints using the augmented Lagrangian method.
    ///
    /// Minimizes `||x - x0||²` subject to `constraints` using the standard
    /// Hestenes-Powell augmented Lagrangian for inequality constraints
    /// `g_j(x) <= 0`:
    ///
    /// `L_rho(x, λ) = ||x - x0||² + (1/2ρ) * Σⱼ [max(0, λⱼ + ρ·gⱼ(x))² - λⱼ²]`
    ///
    /// whose gradient contribution from constraint `j` is `max(0, λⱼ +
    /// ρ·gⱼ(x)) · ∇gⱼ(x)` — not `(λⱼ + ρ·max(0, gⱼ(x))) · ∇gⱼ(x)`, which is
    /// what the multiplier-update step already used but the inner gradient
    /// step previously did not, and does not reduce to the textbook method.
    ///
    /// The inner loop minimizes `L_rho` at fixed multipliers via gradient
    /// descent with backtracking line search — it only ever accepts a step
    /// that decreases `L_rho`, so the iterate cannot diverge no matter how
    /// large `rho` grows — stopping once its gradient norm falls below
    /// `inner_tolerance` or `max_inner_iterations` is reached. The outer
    /// loop then updates the multipliers and increases `rho` (capped at
    /// `max_penalty_parameter`), stopping once the maximum constraint
    /// violation falls below `tolerance`.
    ///
    /// # Errors
    ///
    /// Returns [`LogicError::ProjectionFailed`] if the result is ever
    /// non-finite. The line search above should make this unreachable, but
    /// it is still checked explicitly rather than ever being returned as if
    /// it were a valid `Ok` projection.
    pub fn project(
        &self,
        x0: &Array1<f32>,
        constraints: &[NonlinearConstraint],
    ) -> LogicResult<Array1<f32>> {
        let x0 = x0.as_standard_layout().into_owned();
        let n = x0.len();
        let m = constraints.len();

        let mut x = x0.clone();
        let mut lambda = vec![0.0f32; m]; // Lagrange multipliers
        let mut rho = self
            .penalty_parameter
            .max(MIN_PENALTY_PARAMETER)
            .min(self.max_penalty_parameter);

        for _outer in 0..self.max_outer_iterations {
            // Minimize the augmented Lagrangian at fixed (lambda, rho).
            for _inner in 0..self.max_inner_iterations {
                let x_slice = require_contiguous(&x)?;

                let mut grad = Array1::<f32>::zeros(n);
                let mut al_value = 0.0f32;
                for (i, (&xi, &x0i)) in x.iter().zip(x0.iter()).enumerate() {
                    let d = xi - x0i;
                    grad[i] = 2.0 * d;
                    al_value += d * d;
                }

                for (j, constraint) in constraints.iter().enumerate() {
                    let g_j = constraint.evaluate(x_slice);
                    let factor = (lambda.get(j).copied().unwrap_or(0.0) + rho * g_j).max(0.0);
                    al_value += factor * factor / (2.0 * rho);
                    if let Some(grad_g) = constraint.gradient(x_slice) {
                        for (i, &dg) in grad_g.iter().enumerate() {
                            if let Some(slot) = grad.get_mut(i) {
                                *slot += factor * dg;
                            }
                        }
                    }
                }

                let grad_norm = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
                if grad_norm < self.inner_tolerance {
                    break;
                }

                // Backtracking line search: only accept a step that
                // decreases the augmented Lagrangian value.
                let mut alpha = self.step_size;
                let mut accepted = false;
                for _ in 0..MAX_LINE_SEARCH_STEPS {
                    let mut candidate = x.clone();
                    for (ci, &gi) in candidate.iter_mut().zip(grad.iter()) {
                        *ci -= alpha * gi;
                    }

                    if candidate.iter().all(|v| v.is_finite()) {
                        let candidate_slice = require_contiguous(&candidate)?;
                        let mut candidate_value = candidate
                            .iter()
                            .zip(x0.iter())
                            .map(|(&ci, &x0i)| (ci - x0i) * (ci - x0i))
                            .sum::<f32>();
                        for (j, constraint) in constraints.iter().enumerate() {
                            let g_j = constraint.evaluate(candidate_slice);
                            let factor =
                                (lambda.get(j).copied().unwrap_or(0.0) + rho * g_j).max(0.0);
                            candidate_value += factor * factor / (2.0 * rho);
                        }

                        if candidate_value.is_finite() && candidate_value <= al_value {
                            x = candidate;
                            accepted = true;
                            break;
                        }
                    }

                    alpha *= 0.5;
                }

                if !accepted {
                    // No shrinking step improved the augmented Lagrangian:
                    // already at (or numerically indistinguishable from) an
                    // inner-loop stationary point.
                    break;
                }
            }

            // Update Lagrange multipliers at the new x.
            let x_slice = require_contiguous(&x)?;
            for (j, constraint) in constraints.iter().enumerate() {
                let g_j = constraint.evaluate(x_slice);
                if let Some(slot) = lambda.get_mut(j) {
                    *slot = (*slot + rho * g_j).max(0.0);
                }
            }

            // Check convergence.
            let x_slice = require_contiguous(&x)?;
            let max_violation: f32 = constraints
                .iter()
                .map(|c| c.violation(x_slice))
                .fold(0.0f32, |a, b| a.max(b));

            if max_violation < self.tolerance {
                break;
            }

            // Increase the penalty parameter, capped so it cannot grow without bound.
            rho = (rho * self.penalty_increase_factor)
                .max(MIN_PENALTY_PARAMETER)
                .min(self.max_penalty_parameter);
        }

        if x.iter().any(|v| !v.is_finite()) {
            return Err(LogicError::ProjectionFailed(
                "augmented Lagrangian projection produced a non-finite result".to_string(),
            ));
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
        let set1 = GeometricSet::ball(vec![0.0, 0.0], 2.0).expect("valid ball");
        let set2 = GeometricSet::ball(vec![3.0, 0.0], 2.0).expect("valid ball");

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

    // -----------------------------------------------------------------------
    // AugmentedLagrangian (finding 135)
    // -----------------------------------------------------------------------

    /// Regression: `AugmentedLagrangian::project` used to walk to +-inf and
    /// then NaN once `rho` grew large enough (unconditional `rho *=
    /// penalty_increase_factor` each outer iteration, fixed inner step
    /// size, no convergence test), and returned that as `Ok`. It must now
    /// either converge or error — never silently return a non-finite point.
    #[test]
    fn test_augmented_lagrangian_never_returns_nonfinite() {
        let constraint = NonlinearConstraint::inequality("bound", |x: &[f32]| x[0] - 1.0)
            .with_gradient(|_: &[f32]| vec![1.0]);
        let al = AugmentedLagrangian::new();

        // Start far from the feasible region, as the original bug report did.
        let x0 = Array1::from_vec(vec![1000.0]);
        let result = al
            .project(&x0, std::slice::from_ref(&constraint))
            .expect("a well-posed linear-constraint projection must not error");
        assert!(
            result.iter().all(|v| v.is_finite()),
            "projection must be finite, got {result:?}"
        );
    }

    /// The augmented Lagrangian method is designed to converge to the exact
    /// (not merely feasible) nearest point as the penalty grows and the
    /// multiplier is updated — verify it actually gets there for a simple
    /// bound where the true Euclidean projection is known analytically.
    #[test]
    fn test_augmented_lagrangian_converges_near_true_projection() {
        // x <= 1, so the nearest feasible point to x0 = 5.0 is exactly 1.0.
        let constraint = NonlinearConstraint::inequality("bound", |x: &[f32]| x[0] - 1.0)
            .with_gradient(|_: &[f32]| vec![1.0]);
        let al = AugmentedLagrangian::new();

        let x0 = Array1::from_vec(vec![5.0]);
        let result = al
            .project(&x0, std::slice::from_ref(&constraint))
            .expect("projection succeeds");

        assert!(result[0].is_finite());
        assert!(
            result[0] <= 1.0 + 1e-2,
            "must be (near-)feasible: {}",
            result[0]
        );
        assert!(
            (result[0] - 1.0).abs() < 0.15,
            "expected convergence near the true projection 1.0, got {}",
            result[0]
        );
    }

    /// The inner-loop multiplier term must use `max(0, lambda + rho*g)`,
    /// matching the multiplier-update step, not `lambda + rho*max(0, g)`
    /// (these differ whenever `g < 0` and `lambda > 0`, which is exactly the
    /// regime entered once the method starts converging).
    #[test]
    fn test_augmented_lagrangian_setters_are_reachable() {
        let al = AugmentedLagrangian::new()
            .with_max_outer_iterations(5)
            .with_max_inner_iterations(10)
            .with_penalty_parameter(2.0)
            .with_penalty_increase_factor(4.0)
            .with_max_penalty_parameter(1e4)
            .with_tolerance(1e-3)
            .with_inner_tolerance(1e-4)
            .with_step_size(0.05);

        let constraint = NonlinearConstraint::inequality("bound", |x: &[f32]| x[0] - 1.0)
            .with_gradient(|_: &[f32]| vec![1.0]);
        let x0 = Array1::from_vec(vec![3.0]);
        let result = al.project(&x0, std::slice::from_ref(&constraint));
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // Non-contiguous input (findings 139 / 288)
    // -----------------------------------------------------------------------

    /// Regression: every public `project`/`project_adaptive` entry point in
    /// this module used to `.expect()`-panic on a non-contiguous owned
    /// `Array1` (e.g. one produced by a strided `slice_move`). They must now
    /// normalize the input instead of aborting.
    #[test]
    fn test_projections_accept_strided_input() {
        use scirs2_core::ndarray::s;

        // A strided (non-contiguous) owned Array1: every other element of
        // [0, 10, 1, 11, 2, 12] taken via `slice_move` with a step-2 range.
        // (`.slice(..).to_owned()` would compact back to contiguous, which
        // is exactly why `slice_move` is used here instead.)
        let source = Array1::from_vec(vec![0.0f32, 10.0, 1.0, 11.0, 2.0, 12.0]);
        let strided = source.slice_move(s![..;2]);
        assert_eq!(
            strided.as_slice(),
            None,
            "test fixture must be non-contiguous"
        );
        assert_eq!(strided.to_vec(), vec![0.0, 1.0, 2.0]);

        let linear_constraint = LinearConstraint::less_eq(vec![1.0, 0.0, 0.0], 0.5);
        let dykstra = DykstraProjection::new(vec![linear_constraint]);
        assert!(dykstra.project(&strided).is_ok());

        let nonlinear = NonlinearConstraint::inequality("c", |x: &[f32]| x[0] - 0.5)
            .with_gradient(|_: &[f32]| vec![1.0, 0.0, 0.0]);
        let grad_proj = GradientProjection::new();
        assert!(grad_proj
            .project(&strided, std::slice::from_ref(&nonlinear))
            .is_ok());
        assert!(grad_proj
            .project_adaptive(&strided, std::slice::from_ref(&nonlinear))
            .is_ok());

        let al = AugmentedLagrangian::new().with_max_outer_iterations(2);
        assert!(al.project(&strided, &[nonlinear]).is_ok());
    }
}
