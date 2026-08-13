use serde::{Deserialize, Serialize};

use super::{
    Constraint, GeometricSet, LinearConstraint, NonlinearConstraint, QuadraticConstraint,
    SetMembershipConstraint,
};

// ============================================================================
// Soft vs Hard Constraint Wrapper
// ============================================================================

/// Constraint enforcement mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ConstraintMode {
    /// Hard constraint - must be strictly satisfied, used for projection
    #[default]
    Hard,
    /// Soft constraint - contributes to loss, violation allowed during training
    Soft,
}

/// Penalty function type for soft constraints
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum PenaltyFunction {
    /// L1 penalty: weight * |violation|
    L1,
    /// L2 penalty: weight * violation²
    #[default]
    L2,
    /// Huber penalty: L2 for small violations, L1 for large
    Huber { delta: f32 },
    /// Log barrier: -weight * log(slack)
    LogBarrier { slack: f32 },
    /// Exact penalty: infinite if violated, 0 otherwise (approximates hard)
    Exact { threshold: f32 },
}

impl PenaltyFunction {
    /// Compute penalty value for a given violation
    pub fn compute(&self, violation: f32, weight: f32) -> f32 {
        if violation <= 0.0 {
            return 0.0;
        }

        match self {
            Self::L1 => weight * violation,
            Self::L2 => weight * violation * violation,
            Self::Huber { delta } => {
                if violation <= *delta {
                    weight * 0.5 * violation * violation
                } else {
                    weight * (*delta * violation - 0.5 * delta * delta)
                }
            }
            Self::LogBarrier { slack } => {
                let s = *slack - violation;
                if s > 0.0 {
                    -weight * s.ln()
                } else {
                    f32::MAX
                }
            }
            Self::Exact { threshold } => {
                if violation > *threshold {
                    f32::MAX
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute gradient of penalty with respect to violation
    pub fn gradient(&self, violation: f32, weight: f32) -> f32 {
        if violation <= 0.0 {
            return 0.0;
        }

        match self {
            Self::L1 => weight,
            Self::L2 => 2.0 * weight * violation,
            Self::Huber { delta } => {
                if violation <= *delta {
                    weight * violation
                } else {
                    weight * *delta
                }
            }
            Self::LogBarrier { slack } => {
                let s = *slack - violation;
                if s > 0.0 {
                    weight / s
                } else {
                    f32::MAX
                }
            }
            Self::Exact { .. } => 0.0, // Non-differentiable
        }
    }
}

/// Wrapper for any constraint with soft/hard mode
#[derive(Debug, Clone)]
pub struct SoftHardConstraint<C> {
    /// The underlying constraint
    constraint: C,
    /// Enforcement mode
    mode: ConstraintMode,
    /// Penalty function for soft mode
    penalty: PenaltyFunction,
    /// Weight for loss computation
    weight: f32,
    /// Priority level (higher = more important)
    priority: u32,
}

impl<C> SoftHardConstraint<C> {
    /// Create a new hard constraint
    pub fn hard(constraint: C) -> Self {
        Self {
            constraint,
            mode: ConstraintMode::Hard,
            penalty: PenaltyFunction::default(),
            weight: 1.0,
            priority: 0,
        }
    }

    /// Create a new soft constraint
    pub fn soft(constraint: C) -> Self {
        Self {
            constraint,
            mode: ConstraintMode::Soft,
            penalty: PenaltyFunction::default(),
            weight: 1.0,
            priority: 0,
        }
    }

    /// Set the penalty function
    pub fn with_penalty(mut self, penalty: PenaltyFunction) -> Self {
        self.penalty = penalty;
        self
    }

    /// Set the weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Set the priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Get the mode
    pub fn mode(&self) -> ConstraintMode {
        self.mode
    }

    /// Is this a hard constraint?
    pub fn is_hard(&self) -> bool {
        self.mode == ConstraintMode::Hard
    }

    /// Is this a soft constraint?
    pub fn is_soft(&self) -> bool {
        self.mode == ConstraintMode::Soft
    }

    /// Get the inner constraint
    pub fn inner(&self) -> &C {
        &self.constraint
    }

    /// Get the weight
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Get the priority
    pub fn priority(&self) -> u32 {
        self.priority
    }

    /// Get the penalty function
    pub fn penalty(&self) -> PenaltyFunction {
        self.penalty
    }
}

/// Number of gradient steps used by the iterative `project_onto` implementations.
const ITERATIVE_PROJECTION_STEPS: usize = 200;

/// Step size used by the iterative `project_onto` implementations.
const ITERATIVE_PROJECTION_STEP_SIZE: f32 = 0.1;

/// Trait for constraints that can compute violation
///
/// # Out-of-range dimension convention
///
/// Implementations in this crate (see `impl ViolationComputable for
/// Constraint` below) that target a specific `dimension()` treat an input
/// `x` shorter than that dimension as **vacuously satisfied**: `check`
/// returns `true` and `violation` returns `0.0`, on the reasoning that a
/// constraint tagged for a dimension the vector doesn't have simply does
/// not apply to it. This is a deliberate, crate-wide convention — it is
/// *not* shared by [`crate::ConstraintConsistencyChecker`]'s sampling-based
/// analysis in `constraint_analysis`, which treats the analogous case as
/// unsatisfied instead; that module's exact interval analysis documents its
/// own convention separately.
///
/// This convention only applies to a length **shorter** than the tagged
/// dimension. It does not license silently treating a same-length but
/// non-contiguous array as empty — see `crate::array_utils::contiguous`,
/// which every call site in this crate now uses instead of
/// `x.as_slice().unwrap_or(&[])` for exactly that reason.
pub trait ViolationComputable {
    /// Compute violation for given input
    fn violation(&self, x: &[f32]) -> f32;
    /// Check if constraint is satisfied
    fn check(&self, x: &[f32]) -> bool;

    /// Project `x` onto this constraint's feasible set.
    ///
    /// Returns `Some(projected)` when this constraint type provides a
    /// projection operator, and `None` when it does not (for example a
    /// polytope, whose exact projection needs a QP solve, or a user type that
    /// has not overridden the default).
    ///
    /// The default implementation returns `None`. Callers that must enforce a
    /// constraint — such as the control saturation inside
    /// [`crate::MPCController::solve`] — are required to translate `None` into
    /// an error rather than silently accepting an unprojected point.
    ///
    /// Implementations must return a vector with the same length as `x`.
    fn project_onto(&self, _x: &[f32]) -> Option<Vec<f32>> {
        None
    }
}

// Implement ViolationComputable for basic Constraint
impl ViolationComputable for Constraint {
    fn violation(&self, x: &[f32]) -> f32 {
        if let Some(dim) = self.dimension() {
            if dim < x.len() {
                Constraint::violation(self, x[dim])
            } else {
                0.0
            }
        } else {
            // Apply to all dimensions and sum
            x.iter().map(|&v| Constraint::violation(self, v)).sum()
        }
    }
    fn check(&self, x: &[f32]) -> bool {
        if let Some(dim) = self.dimension() {
            if dim < x.len() {
                Constraint::check(self, x[dim])
            } else {
                true
            }
        } else {
            // All dimensions must satisfy
            x.iter().all(|&v| Constraint::check(self, v))
        }
    }

    fn project_onto(&self, x: &[f32]) -> Option<Vec<f32>> {
        let mut projected = x.to_vec();
        match self.dimension() {
            Some(dim) => {
                // `dim >= x.len()` makes the constraint vacuous here (see the
                // `violation` implementation above), so the identity map is the
                // correct projection for that case.
                if let Some(slot) = projected.get_mut(dim) {
                    *slot = Constraint::project(self, *slot);
                }
            }
            None => {
                for value in projected.iter_mut() {
                    *value = Constraint::project(self, *value);
                }
            }
        }
        Some(projected)
    }
}

// Implement ViolationComputable for LinearConstraint
impl ViolationComputable for LinearConstraint {
    fn violation(&self, x: &[f32]) -> f32 {
        LinearConstraint::violation(self, x)
    }
    fn check(&self, x: &[f32]) -> bool {
        LinearConstraint::check(self, x)
    }

    fn project_onto(&self, x: &[f32]) -> Option<Vec<f32>> {
        // `LinearConstraint::project` zips `x` with the coefficient vector, so
        // when there are fewer coefficients than dimensions the result would be
        // truncated. `violation` zips the same way — the missing coefficients
        // behave as zeros — so the consistent projection leaves those trailing
        // dimensions untouched instead of dropping them.
        let mut projected = LinearConstraint::project(self, x);
        if projected.len() < x.len() {
            if let Some(tail) = x.get(projected.len()..) {
                projected.extend_from_slice(tail);
            }
        }
        if projected.len() == x.len() {
            Some(projected)
        } else {
            None
        }
    }
}

// Implement ViolationComputable for QuadraticConstraint
impl ViolationComputable for QuadraticConstraint {
    fn violation(&self, x: &[f32]) -> f32 {
        QuadraticConstraint::violation(self, x)
    }
    fn check(&self, x: &[f32]) -> bool {
        QuadraticConstraint::check(self, x)
    }

    fn project_onto(&self, x: &[f32]) -> Option<Vec<f32>> {
        let projected = QuadraticConstraint::project(
            self,
            x,
            ITERATIVE_PROJECTION_STEPS,
            ITERATIVE_PROJECTION_STEP_SIZE,
        );
        if projected.len() == x.len() {
            Some(projected)
        } else {
            None
        }
    }
}

// Implement ViolationComputable for NonlinearConstraint
impl ViolationComputable for NonlinearConstraint {
    fn violation(&self, x: &[f32]) -> f32 {
        NonlinearConstraint::violation(self, x)
    }
    fn check(&self, x: &[f32]) -> bool {
        NonlinearConstraint::check(self, x)
    }

    fn project_onto(&self, x: &[f32]) -> Option<Vec<f32>> {
        // Without a gradient the underlying `project` returns its input
        // unchanged; report "no projection available" instead of that no-op.
        if !self.has_gradient() {
            return None;
        }
        let projected = NonlinearConstraint::project(
            self,
            x,
            ITERATIVE_PROJECTION_STEPS,
            ITERATIVE_PROJECTION_STEP_SIZE,
        );
        if projected.len() == x.len() {
            Some(projected)
        } else {
            None
        }
    }
}

// Implement ViolationComputable for SetMembershipConstraint
impl ViolationComputable for SetMembershipConstraint {
    fn violation(&self, x: &[f32]) -> f32 {
        SetMembershipConstraint::violation(self, x)
    }
    fn check(&self, x: &[f32]) -> bool {
        SetMembershipConstraint::check(self, x)
    }

    fn project_onto(&self, x: &[f32]) -> Option<Vec<f32>> {
        let dimension_matches = match self.set() {
            GeometricSet::Box { lower, upper } => lower.len() == x.len() && upper.len() == x.len(),
            GeometricSet::Ball { center, .. } | GeometricSet::LInfBall { center, .. } => {
                center.len() == x.len()
            }
            GeometricSet::Simplex { dimension } => *dimension == x.len(),
            // `GeometricSet::project` returns the input unchanged for these two
            // sets (an exact projection needs a QP solve), so advertising a
            // projection here would be a silent no-op.
            GeometricSet::Ellipsoid { .. } | GeometricSet::Polytope { .. } => return None,
        };
        if !dimension_matches {
            return None;
        }
        let projected = SetMembershipConstraint::project(self, x);
        if projected.len() == x.len() {
            Some(projected)
        } else {
            None
        }
    }
}

impl<C: ViolationComputable> SoftHardConstraint<C> {
    /// Check if constraint is satisfied
    pub fn check(&self, x: &[f32]) -> bool {
        self.constraint.check(x)
    }

    /// Compute violation amount
    pub fn violation(&self, x: &[f32]) -> f32 {
        self.constraint.violation(x)
    }

    /// Compute penalty/loss for this constraint
    pub fn loss(&self, x: &[f32]) -> f32 {
        let viol = self.constraint.violation(x);
        match self.mode {
            ConstraintMode::Hard => {
                // Hard constraints: infinite loss if violated
                if viol > 0.0 {
                    f32::MAX
                } else {
                    0.0
                }
            }
            ConstraintMode::Soft => self.penalty.compute(viol, self.weight),
        }
    }

    /// Compute gradient of loss with respect to violation
    pub fn loss_gradient(&self, x: &[f32]) -> f32 {
        let viol = self.constraint.violation(x);
        match self.mode {
            ConstraintMode::Hard => 0.0, // Non-differentiable
            ConstraintMode::Soft => self.penalty.gradient(viol, self.weight),
        }
    }
}

/// Collection of soft and hard constraints
#[derive(Debug, Default)]
pub struct ConstraintSet<C> {
    constraints: Vec<SoftHardConstraint<C>>,
}

impl<C: ViolationComputable> ConstraintSet<C> {
    /// Create a new constraint set
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Add a hard constraint
    pub fn add_hard(&mut self, constraint: C) {
        self.constraints.push(SoftHardConstraint::hard(constraint));
    }

    /// Add a soft constraint
    pub fn add_soft(&mut self, constraint: C, penalty: PenaltyFunction, weight: f32) {
        self.constraints.push(
            SoftHardConstraint::soft(constraint)
                .with_penalty(penalty)
                .with_weight(weight),
        );
    }

    /// Add a constraint with full configuration
    pub fn add(&mut self, constraint: SoftHardConstraint<C>) {
        self.constraints.push(constraint);
    }

    /// Check if all hard constraints are satisfied
    pub fn all_hard_satisfied(&self, x: &[f32]) -> bool {
        self.constraints
            .iter()
            .filter(|c| c.is_hard())
            .all(|c| c.check(x))
    }

    /// Check if all constraints (hard and soft) are satisfied
    pub fn all_satisfied(&self, x: &[f32]) -> bool {
        self.constraints.iter().all(|c| c.check(x))
    }

    /// Compute total loss from soft constraints
    pub fn soft_loss(&self, x: &[f32]) -> f32 {
        self.constraints
            .iter()
            .filter(|c| c.is_soft())
            .map(|c| c.loss(x))
            .sum()
    }

    /// Compute total loss (soft constraints only, hard are binary)
    pub fn total_loss(&self, x: &[f32]) -> f32 {
        let mut loss = 0.0;
        for c in &self.constraints {
            let l = c.loss(x);
            if l == f32::MAX {
                return f32::MAX;
            }
            loss += l;
        }
        loss
    }

    /// Get hard constraints
    pub fn hard_constraints(&self) -> impl Iterator<Item = &SoftHardConstraint<C>> {
        self.constraints.iter().filter(|c| c.is_hard())
    }

    /// Get soft constraints
    pub fn soft_constraints(&self) -> impl Iterator<Item = &SoftHardConstraint<C>> {
        self.constraints.iter().filter(|c| c.is_soft())
    }

    /// Get all constraints sorted by priority (highest first)
    pub fn by_priority(&self) -> Vec<&SoftHardConstraint<C>> {
        let mut sorted: Vec<_> = self.constraints.iter().collect();
        sorted.sort_by_key(|c| std::cmp::Reverse(c.priority()));
        sorted
    }

    /// Number of constraints
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintBuilder;

    #[test]
    fn test_constraint_project_onto_honours_dimension_tag() {
        let tagged = ConstraintBuilder::new()
            .name("dim1")
            .dimension(1)
            .less_eq(1.0)
            .build()
            .expect("constraint builds");

        let projected = ViolationComputable::project_onto(&tagged, &[5.0, 9.0, 7.0])
            .expect("basic constraints provide a projection");
        assert_eq!(projected.len(), 3);
        assert!((projected[0] - 5.0).abs() < 1e-6, "dim 0 untouched");
        assert!((projected[1] - 1.0).abs() < 1e-6, "dim 1 projected");
        assert!((projected[2] - 7.0).abs() < 1e-6, "dim 2 untouched");
    }

    #[test]
    fn test_untagged_constraint_projects_every_dimension() {
        let untagged = ConstraintBuilder::new()
            .name("all")
            .less_eq(1.0)
            .build()
            .expect("constraint builds");

        let projected = ViolationComputable::project_onto(&untagged, &[5.0, 0.0])
            .expect("projection available");
        assert!((projected[0] - 1.0).abs() < 1e-6);
        assert!((projected[1] - 0.0).abs() < 1e-6);
    }

    /// A linear constraint with fewer coefficients than dimensions treats the
    /// missing ones as zero in `violation`; `project_onto` must agree and keep
    /// the vector length instead of truncating it.
    #[test]
    fn test_linear_projection_pads_missing_coefficients() {
        let constraint = LinearConstraint::less_eq(vec![1.0], 2.0);
        let point = [5.0_f32, 42.0, -7.0];

        let projected = ViolationComputable::project_onto(&constraint, &point)
            .expect("linear constraints provide a projection");
        assert_eq!(projected.len(), point.len(), "length must be preserved");
        assert!(
            ViolationComputable::violation(&constraint, &projected) <= 1e-5,
            "projected point must satisfy the constraint"
        );
        assert!(
            (projected[1] - 42.0).abs() < 1e-6,
            "unconstrained dims kept"
        );
        assert!((projected[2] + 7.0).abs() < 1e-6, "unconstrained dims kept");
    }

    /// A polytope has no closed-form projection in this crate, so the trait
    /// must report `None` rather than silently returning the input.
    #[test]
    fn test_polytope_reports_no_projection() {
        let set = GeometricSet::polytope(vec![1.0, 0.0, 0.0, 1.0], vec![1.0, 1.0], 2, 2)
            .expect("valid polytope");
        let constraint = SetMembershipConstraint::new("poly", set);
        assert!(ViolationComputable::project_onto(&constraint, &[5.0, 5.0]).is_none());
    }
}
