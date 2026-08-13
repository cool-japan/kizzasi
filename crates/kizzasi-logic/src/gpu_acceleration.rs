//! Batch-Oriented Constraint Checking
//!
//! This module provides batch-oriented implementations of constraint
//! checking and projection operations, sized for large point batches and
//! high-dimensional spaces.
//!
//! # GPU status
//!
//! **No GPU backend is currently implemented.** [`check_gpu_availability`],
//! and every type's `is_gpu_available()`, always return `false`; every
//! `*_batch` method always runs on CPU. Earlier revisions of this module
//! claimed GPU acceleration in both the type names and this doc comment
//! while every "GPU" code path silently ran the identical CPU
//! implementation underneath a feature flag that gated nothing — that was
//! never true and has been corrected here rather than actually
//! implemented, because a genuine GPU backend is out of scope for this
//! pass: `ViolationComputable` is an arbitrary Rust trait, and there is no
//! way to compile arbitrary user-supplied trait-object logic to a compute
//! shader. A real backend would need to be built against a concrete,
//! GPU-shaped constraint representation (e.g. the WGSL kernels in
//! `kizzasi-webgpu`), which is a larger, separate design effort.
//!
//! # Features
//!
//! - Batch constraint evaluation
//! - Batch projection onto constraint sets
//! - Batch gradient computation for constraint violations
//!
//! # Performance
//!
//! These batch APIs amortize per-call overhead across many points; they are
//! most useful for large batch sizes, high-dimensional spaces, and complex
//! constraint sets, independent of the (currently CPU-only) execution
//! backend.

use crate::constraint::ViolationComputable;
use crate::error::LogicResult;
use scirs2_core::ndarray::{Array1, Array2};

/// Batch-oriented constraint checker.
///
/// Evaluates constraints against many points. See the module doc comment:
/// there is currently no GPU backend, so [`Self::is_gpu_available`] always
/// returns `false` and every `*_batch` call runs on CPU.
pub struct GPUConstraintChecker<C> {
    /// Constraints to check
    constraints: Vec<C>,
    /// Whether GPU is available (always `false` — see the module doc comment)
    gpu_available: bool,
    /// Batch size threshold that previously selected the (non-existent) GPU
    /// path; retained on the public API for compatibility with
    /// [`Self::with_gpu_threshold`], but no longer changes behavior.
    gpu_threshold: usize,
}

/// Check if a GPU backend is available.
///
/// Always returns `false`: no GPU backend is implemented in this module
/// (see the module doc comment). This performs no hardware probing.
fn check_gpu_availability() -> bool {
    false
}

impl<C: ViolationComputable + Clone> GPUConstraintChecker<C> {
    /// Create a new GPU constraint checker
    pub fn new(constraints: Vec<C>) -> Self {
        Self {
            constraints,
            gpu_available: check_gpu_availability(),
            gpu_threshold: 1000, // Use GPU for batches larger than this
        }
    }

    /// Set the batch size threshold for GPU usage
    pub fn with_gpu_threshold(mut self, threshold: usize) -> Self {
        self.gpu_threshold = threshold;
        self
    }

    /// Check constraints for a batch of points.
    ///
    /// Always runs on CPU — see the module doc comment for why. Kept as a
    /// separate entry point from `Self::check_batch_cpu` (rather than
    /// merging the two) so `Self::gpu_threshold`/[`Self::is_gpu_available`]
    /// stay meaningful hooks for a future real backend without another
    /// public-API break.
    pub fn check_batch(&self, points: &Array2<f32>) -> Vec<bool> {
        self.check_batch_cpu(points)
    }

    /// CPU-based batch checking
    fn check_batch_cpu(&self, points: &Array2<f32>) -> Vec<bool> {
        let (n_points, _) = points.dim();
        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let point = points.row(i);
            let point_slice: Vec<f32> = point.iter().copied().collect();
            let satisfied = self.constraints.iter().all(|c| c.check(&point_slice));
            results.push(satisfied);
        }

        results
    }

    /// Compute violations for a batch of points.
    ///
    /// Always runs on CPU — see the module doc comment.
    pub fn violation_batch(&self, points: &Array2<f32>) -> Vec<f32> {
        self.violation_batch_cpu(points)
    }

    /// CPU-based batch violation computation
    fn violation_batch_cpu(&self, points: &Array2<f32>) -> Vec<f32> {
        let (n_points, _) = points.dim();
        let mut violations = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let point = points.row(i);
            let point_slice: Vec<f32> = point.iter().copied().collect();

            let total_violation: f32 = self
                .constraints
                .iter()
                .map(|c| c.violation(&point_slice))
                .sum();

            violations.push(total_violation);
        }

        violations
    }

    /// Get GPU availability status. Always `false` — see the module doc comment.
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// Get number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
}

/// Batch-oriented projection onto constraint sets.
///
/// Always runs on CPU — see the module doc comment.
pub struct GPUProjector {
    /// Maximum iterations for projection
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f32,
    /// Whether GPU is available (always `false` — see the module doc comment)
    gpu_available: bool,
}

impl GPUProjector {
    /// Create a new GPU projector
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            gpu_available: check_gpu_availability(),
        }
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set tolerance
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    /// Project a batch of points onto a constraint set.
    ///
    /// Always runs on CPU — see the module doc comment. Each point is
    /// projected independently via `Self::project_point`, which performs
    /// *feasibility* descent (minimizes summed constraint violation with a
    /// backtracking line search), not a metric projection: it returns *a*
    /// feasible point efficiently, not necessarily the *nearest* one. Use
    /// [`crate::AugmentedLagrangian::project`] when the nearest feasible
    /// point specifically matters — it explicitly minimizes `||x - x0||²`
    /// subject to the constraints.
    pub fn project_batch<C: ViolationComputable + Clone>(
        &self,
        points: &Array2<f32>,
        constraints: &[C],
    ) -> LogicResult<Array2<f32>> {
        let (n_points, n_dims) = points.dim();

        if n_points == 0 {
            return Ok(points.clone());
        }

        let mut projected = points.clone();

        for i in 0..n_points {
            let point = projected.row(i).to_owned();
            let projected_point = self.project_point(&point, constraints)?;

            for j in 0..n_dims {
                if let Some(slot) = projected.get_mut([i, j]) {
                    *slot = projected_point.get(j).copied().unwrap_or(*slot);
                }
            }
        }

        Ok(projected)
    }

    /// Feasibility descent for a single point: minimizes the summed
    /// constraint violation via a numerical gradient and a backtracking
    /// line search.
    ///
    /// The line search grows the step when it helps and shrinks it when it
    /// doesn't, so a point far from feasible (e.g. `x = 100` against
    /// `less_eq(5.0)`) reaches the boundary in a bounded number of
    /// iterations instead of creeping toward it at a fixed step size and
    /// stopping while still infeasible. The per-dimension finite difference
    /// mutates and restores a single scratch buffer element instead of
    /// cloning a full `x_plus` vector per dimension per constraint per
    /// iteration.
    fn project_point<C: ViolationComputable>(
        &self,
        point: &Array1<f32>,
        constraints: &[C],
    ) -> LogicResult<Array1<f32>> {
        let n = point.len();
        let mut x: Vec<f32> = point.iter().copied().collect();
        let mut gradient = vec![0.0f32; n];
        let mut step_size = 0.1f32;
        const EPS: f32 = 1e-4;

        for _iter in 0..self.max_iterations {
            let total_violation: f32 = constraints.iter().map(|c| c.violation(&x)).sum();
            if total_violation < self.tolerance {
                break;
            }

            for slot in gradient.iter_mut() {
                *slot = 0.0;
            }
            for constraint in constraints {
                let base = constraint.violation(&x);
                if base <= 0.0 {
                    continue;
                }
                for i in 0..n {
                    let Some(&original) = x.get(i) else {
                        continue;
                    };
                    if let Some(slot) = x.get_mut(i) {
                        *slot = original + EPS;
                    }
                    let perturbed = constraint.violation(&x);
                    if let Some(slot) = x.get_mut(i) {
                        *slot = original; // restore before the next dimension
                    }
                    if let Some(slot) = gradient.get_mut(i) {
                        *slot += (perturbed - base) / EPS;
                    }
                }
            }

            // Backtracking line search on the summed violation.
            let mut alpha = step_size;
            let mut accepted = false;
            for _ in 0..GPU_PROJECTOR_LINE_SEARCH_STEPS {
                let mut candidate = x.clone();
                for (c, &g) in candidate.iter_mut().zip(gradient.iter()) {
                    *c -= alpha * g;
                }

                if candidate.iter().all(|v| v.is_finite()) {
                    let candidate_violation: f32 =
                        constraints.iter().map(|c| c.violation(&candidate)).sum();
                    if candidate_violation.is_finite() && candidate_violation < total_violation {
                        x = candidate;
                        step_size = (alpha * 1.5).min(10.0);
                        accepted = true;
                        break;
                    }
                }
                alpha *= 0.5;
            }

            if !accepted {
                // No shrinking step reduced the violation further: already
                // at (or numerically indistinguishable from) a stationary
                // point of the feasibility objective.
                break;
            }
        }

        Ok(Array1::from_vec(x))
    }

    /// Check if GPU is available. Always `false` — see the module doc comment.
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }
}

/// Maximum backtracking halvings tried per gradient step of
/// [`GPUProjector::project_point`] before giving up on that step.
const GPU_PROJECTOR_LINE_SEARCH_STEPS: usize = 20;

impl Default for GPUProjector {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch-oriented constraint gradient computation (finite differences).
///
/// Always runs on CPU — see the module doc comment.
pub struct GPUGradientComputer {
    /// Finite difference epsilon
    epsilon: f32,
    /// Whether GPU is available (always `false` — see the module doc comment)
    gpu_available: bool,
}

impl GPUGradientComputer {
    /// Create a new GPU gradient computer
    pub fn new() -> Self {
        Self {
            epsilon: 1e-5,
            gpu_available: check_gpu_availability(),
        }
    }

    /// Set epsilon for finite differences
    pub fn with_epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }

    /// Compute gradient of constraint violation for a batch
    ///
    /// Returns Array2 of shape (n_points, n_dims) containing gradients
    pub fn compute_batch_gradients<C: ViolationComputable + Clone>(
        &self,
        points: &Array2<f32>,
        constraint: &C,
    ) -> LogicResult<Array2<f32>> {
        let (n_points, n_dims) = points.dim();
        let mut gradients = Array2::zeros((n_points, n_dims));

        for i in 0..n_points {
            let point = points.row(i);
            let point_slice: Vec<f32> = point.iter().copied().collect();
            let base_violation = constraint.violation(&point_slice);

            for j in 0..n_dims {
                let mut perturbed = point_slice.clone();
                perturbed[j] += self.epsilon;
                let perturbed_violation = constraint.violation(&perturbed);

                gradients[[i, j]] = (perturbed_violation - base_violation) / self.epsilon;
            }
        }

        Ok(gradients)
    }

    /// Compute Hessian of constraint violation (second-order information)
    ///
    /// Useful for Newton-based optimization
    pub fn compute_hessian<C: ViolationComputable>(
        &self,
        point: &[f32],
        constraint: &C,
    ) -> LogicResult<Array2<f32>> {
        let n_dims = point.len();
        let mut hessian = Array2::zeros((n_dims, n_dims));

        // Compute second-order finite differences
        for i in 0..n_dims {
            for j in 0..n_dims {
                let mut x_ij = point.to_vec();
                let mut x_i = point.to_vec();
                let mut x_j = point.to_vec();

                x_ij[i] += self.epsilon;
                x_ij[j] += self.epsilon;
                x_i[i] += self.epsilon;
                x_j[j] += self.epsilon;

                let f_ij = constraint.violation(&x_ij);
                let f_i = constraint.violation(&x_i);
                let f_j = constraint.violation(&x_j);
                let f_0 = constraint.violation(point);

                hessian[[i, j]] = (f_ij - f_i - f_j + f_0) / (self.epsilon * self.epsilon);
            }
        }

        Ok(hessian)
    }

    /// Check if GPU is available. Always `false` — see the module doc comment.
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }
}

impl Default for GPUGradientComputer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintBuilder;

    #[test]
    fn test_gpu_constraint_checker() {
        let constraint1 = ConstraintBuilder::new()
            .name("c1")
            .less_than(10.0)
            .build()
            .unwrap();

        let constraint2 = ConstraintBuilder::new()
            .name("c2")
            .greater_than(0.0)
            .build()
            .unwrap();

        let checker = GPUConstraintChecker::new(vec![constraint1, constraint2]);

        assert_eq!(checker.num_constraints(), 2);

        // Test batch checking
        let points = Array2::from_shape_vec((3, 1), vec![5.0, 15.0, -1.0]).unwrap();
        let results = checker.check_batch(&points);

        assert_eq!(results.len(), 3);
        assert!(results[0]); // 5.0 satisfies both
        assert!(!results[1]); // 15.0 violates c1
        assert!(!results[2]); // -1.0 violates c2
    }

    #[test]
    fn test_gpu_violation_batch() {
        let constraint = ConstraintBuilder::new()
            .name("c1")
            .less_than(5.0)
            .build()
            .unwrap();

        let checker = GPUConstraintChecker::new(vec![constraint]);

        let points = Array2::from_shape_vec((3, 1), vec![3.0, 5.0, 7.0]).unwrap();
        let violations = checker.violation_batch(&points);

        assert_eq!(violations.len(), 3);
        assert!(violations[0] < 1e-5); // 3.0 < 5.0, no violation
        assert!(violations[1] < 1e-5); // 5.0 = 5.0, no violation
        assert!(violations[2] > 0.0); // 7.0 > 5.0, violation
    }

    #[test]
    fn test_gpu_projector() {
        // Use more iterations for better convergence
        let projector = GPUProjector::new().with_max_iterations(1000);

        let constraint = ConstraintBuilder::new()
            .name("c1")
            .less_than(5.0)
            .build()
            .unwrap();

        // Point at 7.0 should be projected to near 5.0
        let point = Array1::from_vec(vec![7.0]);
        let projected = projector.project_point(&point, &[constraint]).unwrap();

        println!("Projected value: {}", projected[0]);
        assert!(
            projected[0] <= 5.1,
            "Projected value {} is greater than 5.1",
            projected[0]
        );
    }

    #[test]
    fn test_gpu_gradient_computer() {
        let grad_computer = GPUGradientComputer::new();

        let constraint = ConstraintBuilder::new()
            .name("c1")
            .less_than(5.0)
            .build()
            .unwrap();

        let points = Array2::from_shape_vec((2, 1), vec![3.0, 7.0]).unwrap();
        let gradients = grad_computer
            .compute_batch_gradients(&points, &constraint)
            .unwrap();

        assert_eq!(gradients.dim(), (2, 1));
        // Gradient should be 0 for satisfied constraint, positive for violated
        assert!(gradients[[0, 0]].abs() < 0.1);
        assert!(gradients[[1, 0]] > 0.0);
    }

    #[test]
    fn test_gpu_hessian() {
        let grad_computer = GPUGradientComputer::new();

        let constraint = ConstraintBuilder::new()
            .name("c1")
            .less_than(5.0)
            .build()
            .unwrap();

        let point = vec![3.0];
        let hessian = grad_computer.compute_hessian(&point, &constraint).unwrap();

        assert_eq!(hessian.dim(), (1, 1));
        // For linear constraint, Hessian should be near zero
        assert!(hessian[[0, 0]].abs() < 0.1);
    }

    #[test]
    fn test_gpu_threshold() {
        let constraint = ConstraintBuilder::new()
            .name("c1")
            .less_than(10.0)
            .build()
            .unwrap();

        let checker = GPUConstraintChecker::new(vec![constraint]).with_gpu_threshold(500);

        // Small batch should use CPU
        let small_batch = Array2::from_shape_vec((10, 1), vec![1.0; 10]).unwrap();
        let _results = checker.check_batch(&small_batch);

        // Large batch would use GPU if available
        let large_batch = Array2::from_shape_vec((1000, 1), vec![1.0; 1000]).unwrap();
        let _results = checker.check_batch(&large_batch);
    }

    /// Regression (findings 301/345/141): `is_gpu_available()` on all three
    /// types must honestly report `false` — no GPU backend is implemented,
    /// so it must never claim otherwise regardless of what feature flags a
    /// build enables (there is no `gpu` feature anymore for exactly this
    /// reason: it used to make this return `true` unconditionally with no
    /// hardware probe behind it).
    #[test]
    fn test_is_gpu_available_is_always_false() {
        let constraint = ConstraintBuilder::new()
            .name("c")
            .less_than(1.0)
            .build()
            .unwrap();
        assert!(!GPUConstraintChecker::new(vec![constraint]).is_gpu_available());
        assert!(!GPUProjector::new().is_gpu_available());
        assert!(!GPUGradientComputer::new().is_gpu_available());
    }

    /// A point far from a feasible half-line must actually become feasible
    /// within the iteration budget, not merely move a fixed 0.01 step and
    /// stop (finding 143: `project_point` used to be unregularized penalty
    /// descent with a fixed step size and no real termination rule).
    #[test]
    fn test_gpu_projector_reaches_feasibility_from_far_start() {
        let constraint = ConstraintBuilder::new()
            .name("c1")
            .less_eq(5.0)
            .build()
            .unwrap();

        let projector = GPUProjector::new().with_max_iterations(200);
        let point = Array1::from_vec(vec![100.0]);
        let projected = projector
            .project_point(&point, &[constraint])
            .expect("projection succeeds");

        assert!(
            projected[0] <= 5.0 + 1e-3,
            "must actually reach feasibility, got {}",
            projected[0]
        );
    }
}
