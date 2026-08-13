//! Constraint Repair Algorithms
//!
//! This module provides algorithms for handling infeasible constraint sets by:
//! - Identifying minimal infeasible subsets (IIS)
//! - Computing minimal relaxations to restore feasibility
//! - Conflict resolution strategies
//! - Constraint priority-based repair
//!
//! # Use Cases
//!
//! - Over-constrained optimization problems
//! - Conflict resolution in multi-agent systems
//! - Fault recovery in control systems
//! - Interactive constraint debugging

use crate::constraint::{BoundType, Constraint, ViolationComputable};
use crate::error::{LogicError, LogicResult};
use scirs2_core::ndarray::Array1;

/// Maximum number of cyclic-projection sweeps used when moving a point.
const PROJECTION_SWEEPS: usize = 200;

/// Result of constraint repair operation
///
/// The exact meaning of the fields depends on the [`RepairStrategy`] used —
/// see that enum for the per-strategy contract. In all cases the invariant is:
/// when `success` is `true`, `repaired_point` holds a point that satisfies
/// every constraint after the reported constraints are relaxed by the reported
/// amounts. When `success` is `false`, `repaired_point` is `None` and no
/// feasible (relaxed) configuration was found within `max_relaxation`.
#[derive(Debug, Clone)]
pub struct RepairResult {
    /// Indices of constraints that were relaxed
    pub relaxed_constraints: Vec<usize>,
    /// Amount each constraint was relaxed (parallel to `relaxed_constraints`)
    pub relaxation_amounts: Vec<f32>,
    /// Total cost of repair
    pub repair_cost: f32,
    /// Whether repair was successful
    pub success: bool,
    /// Repaired point (only present when `success`)
    pub repaired_point: Option<Array1<f32>>,
}

impl RepairResult {
    /// Number of constraints relaxed
    pub fn num_relaxed(&self) -> usize {
        self.relaxed_constraints.len()
    }

    /// Check if repair was minimal (only relaxed necessary constraints)
    pub fn is_minimal(&self) -> bool {
        self.relaxation_amounts.iter().all(|&amount| amount > 0.0)
    }
}

/// Strategy for repairing infeasible constraints
///
/// The first three strategies treat the *point* as the hard requirement and
/// repair by relaxing constraints around it (the point is returned unchanged
/// when the relaxation alone restores feasibility). `ElasticProgramming`
/// instead treats the *constraints* as the requirement and moves the point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RepairStrategy {
    /// Relax all violated constraints by the same amount (the largest
    /// violation, capped at `max_relaxation`)
    Uniform,
    /// Relax constraints with lowest priority first; each violated constraint
    /// is relaxed by its own violation and the report is ordered by ascending
    /// priority
    PriorityBased,
    /// Relax each violated constraint by exactly its own violation — the
    /// smallest total relaxation that admits the point
    MinimalRelaxation,
    /// Use elastic programming: move the point onto the constraint set and
    /// report the residual slack that could not be removed
    ElasticProgramming,
}

/// Constraint repairer for handling infeasible constraint sets
pub struct ConstraintRepairer {
    /// Repair strategy
    strategy: RepairStrategy,
    /// Maximum relaxation allowed per constraint
    max_relaxation: f32,
    /// Tolerance for feasibility
    tolerance: f32,
}

impl ConstraintRepairer {
    /// Create a new constraint repairer
    pub fn new(strategy: RepairStrategy) -> Self {
        Self {
            strategy,
            max_relaxation: 1e3,
            tolerance: 1e-6,
        }
    }

    /// Set maximum relaxation
    pub fn with_max_relaxation(mut self, max_relax: f32) -> Self {
        self.max_relaxation = max_relax;
        self
    }

    /// Set tolerance
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    /// Repair infeasible constraints at a given point
    ///
    /// Every constraint is evaluated on the dimension it governs
    /// (`Constraint::dimension()`); an untagged constraint applies to all
    /// dimensions, matching the [`ViolationComputable`] implementation for
    /// [`Constraint`].
    ///
    /// # Errors
    ///
    /// * [`LogicError::DimensionMismatch`] when a constraint names a dimension
    ///   the point does not have. (This is deliberately stricter than the
    ///   `ViolationComputable` implementation for `Constraint`, which reports
    ///   such a constraint as satisfied: silently reporting an out-of-range
    ///   constraint as feasible would hide exactly the infeasibility this
    ///   module exists to diagnose.)
    /// * [`LogicError::InvalidInput`] when the point is empty but constraints
    ///   were supplied.
    /// * [`LogicError::InfeasibleConstraint`] when the priority vector length
    ///   does not match the constraint count.
    pub fn repair(
        &self,
        point: &[f32],
        constraints: &[Constraint],
        priorities: Option<&[f32]>,
    ) -> LogicResult<RepairResult> {
        validate_point(point, constraints)?;

        match self.strategy {
            RepairStrategy::Uniform => self.repair_uniform(point, constraints),
            RepairStrategy::PriorityBased => {
                self.repair_priority_based(point, constraints, priorities)
            }
            RepairStrategy::MinimalRelaxation => self.repair_minimal(point, constraints),
            RepairStrategy::ElasticProgramming => self.repair_elastic(point, constraints),
        }
    }

    /// Uniform relaxation: relax every violated constraint by the same amount.
    ///
    /// The shared relaxation is the largest violation present, capped at
    /// `max_relaxation`. When the cap bites, the point is additionally moved so
    /// that the residual fits inside the reported relaxation.
    fn repair_uniform(
        &self,
        point: &[f32],
        constraints: &[Constraint],
    ) -> LogicResult<RepairResult> {
        let violations = self.violations(point, constraints);
        let violated: Vec<usize> = self.violated_indices(&violations);

        let largest = violated
            .iter()
            .filter_map(|&i| violations.get(i).copied())
            .fold(0.0_f32, f32::max);
        let shared = largest.min(self.max_relaxation);

        let mut budgets = vec![0.0_f32; constraints.len()];
        for &i in &violated {
            if let Some(slot) = budgets.get_mut(i) {
                *slot = shared;
            }
        }

        let amounts = vec![shared; violated.len()];
        let cost = shared * violated.len() as f32;
        Ok(self.finalize(point, constraints, violated, amounts, cost, &budgets))
    }

    /// Priority-based relaxation: relax low-priority constraints first.
    fn repair_priority_based(
        &self,
        point: &[f32],
        constraints: &[Constraint],
        priorities: Option<&[f32]>,
    ) -> LogicResult<RepairResult> {
        let default_priorities: Vec<f32> = vec![1.0; constraints.len()];
        let priorities = priorities.unwrap_or(&default_priorities);

        if priorities.len() != constraints.len() {
            return Err(LogicError::InfeasibleConstraint(
                "Priority vector length mismatch".to_string(),
            ));
        }

        let violations = self.violations(point, constraints);

        // (index, priority, violation) for every violated constraint
        let mut ranked: Vec<(usize, f32, f32)> = self
            .violated_indices(&violations)
            .into_iter()
            .map(|i| {
                (
                    i,
                    priorities.get(i).copied().unwrap_or(1.0),
                    violations.get(i).copied().unwrap_or(0.0),
                )
            })
            .collect();

        // Sort by priority (lowest first) then by violation (largest first):
        // the cheapest constraints to give up come first.
        ranked.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });

        let mut budgets = vec![0.0_f32; constraints.len()];
        let mut relaxed = Vec::with_capacity(ranked.len());
        let mut amounts = Vec::with_capacity(ranked.len());
        let mut cost = 0.0_f32;

        for (index, priority, violation) in ranked {
            let amount = violation.min(self.max_relaxation);
            if let Some(slot) = budgets.get_mut(index) {
                *slot = amount;
            }
            relaxed.push(index);
            amounts.push(amount);
            cost += priority * amount; // Weight by priority
        }

        Ok(self.finalize(point, constraints, relaxed, amounts, cost, &budgets))
    }

    /// Minimal relaxation: relax each violated constraint by exactly its own
    /// violation — the smallest total relaxation that admits the point.
    fn repair_minimal(
        &self,
        point: &[f32],
        constraints: &[Constraint],
    ) -> LogicResult<RepairResult> {
        let violations = self.violations(point, constraints);

        let mut violated: Vec<(usize, f32)> = self
            .violated_indices(&violations)
            .into_iter()
            .map(|i| (i, violations.get(i).copied().unwrap_or(0.0)))
            .collect();

        // Sort by violation (largest first for greedy selection)
        violated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut budgets = vec![0.0_f32; constraints.len()];
        let mut relaxed = Vec::with_capacity(violated.len());
        let mut amounts = Vec::with_capacity(violated.len());
        let mut cost = 0.0_f32;

        for (index, violation) in violated {
            let amount = violation.min(self.max_relaxation);
            if let Some(slot) = budgets.get_mut(index) {
                *slot = amount;
            }
            relaxed.push(index);
            amounts.push(amount);
            cost += amount;
        }

        Ok(self.finalize(point, constraints, relaxed, amounts, cost, &budgets))
    }

    /// Elastic programming: solve with slack variables
    ///
    /// ```text
    /// min Σ slackᵢ
    /// s.t. cᵢ(x) <= slackᵢ,  slackᵢ >= 0
    /// ```
    ///
    /// The point is moved onto the constraint set by cyclic projection, which
    /// drives the slacks to zero whenever the constraints admit a common point.
    /// Whatever violation survives is reported as the residual slack.
    fn repair_elastic(
        &self,
        point: &[f32],
        constraints: &[Constraint],
    ) -> LogicResult<RepairResult> {
        // Target zero slack for every constraint and let the projection tell us
        // what actually remains.
        let zero_budgets = vec![0.0_f32; constraints.len()];
        let repaired = self.compute_repaired_point(point, constraints, &zero_budgets);

        let residuals = self.violations(&repaired, constraints);
        let violated = self.violated_indices(&residuals);

        let mut relaxed = Vec::with_capacity(violated.len());
        let mut slacks = Vec::with_capacity(violated.len());
        let mut total_slack = 0.0_f32;
        let mut success = true;

        for index in violated {
            let residual = residuals.get(index).copied().unwrap_or(0.0);
            let slack = residual.min(self.max_relaxation);
            if residual > slack + self.tolerance {
                // The residual violation exceeds the configured relaxation cap,
                // so no admissible elastic program exists at this point.
                success = false;
            }
            relaxed.push(index);
            slacks.push(slack);
            total_slack += slack;
        }

        Ok(RepairResult {
            relaxed_constraints: relaxed,
            relaxation_amounts: slacks,
            repair_cost: total_slack,
            success,
            repaired_point: if success {
                Some(Array1::from_vec(repaired))
            } else {
                None
            },
        })
    }

    /// Violation of every constraint at `point`, honouring the dimension tag.
    fn violations(&self, point: &[f32], constraints: &[Constraint]) -> Vec<f32> {
        constraints
            .iter()
            .map(|c| ViolationComputable::violation(c, point))
            .collect()
    }

    /// Indices whose violation exceeds the feasibility tolerance.
    fn violated_indices(&self, violations: &[f32]) -> Vec<usize> {
        violations
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > self.tolerance)
            .map(|(i, _)| i)
            .collect()
    }

    /// Turn a relaxation plan into a [`RepairResult`], verifying that the plan
    /// really does restore feasibility.
    ///
    /// `budgets` is indexed by constraint and holds the relaxation granted to
    /// each one. If the original point already fits inside those budgets it is
    /// returned unchanged; otherwise the point is moved by cyclic projection
    /// onto the relaxed constraint set and the outcome is re-verified.
    fn finalize(
        &self,
        point: &[f32],
        constraints: &[Constraint],
        relaxed: Vec<usize>,
        amounts: Vec<f32>,
        cost: f32,
        budgets: &[f32],
    ) -> RepairResult {
        let mut candidate: Vec<f32> = point.to_vec();

        if !self.fits_budgets(&candidate, constraints, budgets) {
            candidate = self.compute_repaired_point(point, constraints, budgets);
        }

        let success = self.fits_budgets(&candidate, constraints, budgets);

        RepairResult {
            relaxed_constraints: relaxed,
            relaxation_amounts: amounts,
            repair_cost: cost,
            success,
            repaired_point: if success {
                Some(Array1::from_vec(candidate))
            } else {
                None
            },
        }
    }

    /// `true` when every constraint's violation at `point` is inside its budget.
    fn fits_budgets(&self, point: &[f32], constraints: &[Constraint], budgets: &[f32]) -> bool {
        constraints.iter().enumerate().all(|(i, constraint)| {
            let budget = budgets.get(i).copied().unwrap_or(0.0);
            ViolationComputable::violation(constraint, point) <= budget + self.tolerance
        })
    }

    /// Move `initial` onto the constraint set by cyclic projection (POCS).
    ///
    /// `budgets[i]` is the slack granted to constraint `i`: the projection
    /// targets the budget-relaxed feasible region, so a constraint with a
    /// generous budget does not pull the point further than necessary. For a
    /// single constraint this is the exact Euclidean projection; when the
    /// (relaxed) constraints have no common point the sweep stalls and the
    /// caller's re-verification reports the failure.
    fn compute_repaired_point(
        &self,
        initial: &[f32],
        constraints: &[Constraint],
        budgets: &[f32],
    ) -> Vec<f32> {
        let mut x = initial.to_vec();

        for _ in 0..PROJECTION_SWEEPS {
            let mut max_move = 0.0_f32;

            for (i, constraint) in constraints.iter().enumerate() {
                let budget = budgets.get(i).copied().unwrap_or(0.0);
                if ViolationComputable::violation(constraint, &x) <= budget + self.tolerance {
                    continue;
                }

                match constraint.dimension() {
                    Some(dim) => {
                        if let Some(slot) = x.get_mut(dim) {
                            let projected = project_relaxed(constraint.bound(), *slot, budget);
                            max_move = max_move.max((projected - *slot).abs());
                            *slot = projected;
                        }
                    }
                    None => {
                        for slot in x.iter_mut() {
                            let projected = project_relaxed(constraint.bound(), *slot, budget);
                            max_move = max_move.max((projected - *slot).abs());
                            *slot = projected;
                        }
                    }
                }
            }

            if max_move <= self.tolerance {
                break;
            }
        }

        x
    }
}

/// Reject inputs that cannot be evaluated consistently.
fn validate_point(point: &[f32], constraints: &[Constraint]) -> LogicResult<()> {
    if constraints.is_empty() {
        return Ok(());
    }
    if point.is_empty() {
        return Err(LogicError::InvalidInput(
            "repair requires a non-empty point when constraints are supplied".to_string(),
        ));
    }
    for constraint in constraints {
        if let Some(dim) = constraint.dimension() {
            if dim >= point.len() {
                return Err(LogicError::DimensionMismatch {
                    expected: dim + 1,
                    got: point.len(),
                });
            }
        }
    }
    Ok(())
}

/// Project a scalar onto a bound that has been relaxed by `slack`.
///
/// Relaxing by `s` widens the feasible interval by `s` on every finite side,
/// so the projection of an infeasible value lands exactly on the widened edge.
fn project_relaxed(bound: &BoundType, value: f32, slack: f32) -> f32 {
    let slack = slack.max(0.0);
    match bound {
        BoundType::LessThan(b) => value.min(b + slack - f32::EPSILON),
        BoundType::LessEq(b) => value.min(b + slack),
        BoundType::GreaterThan(b) => value.max(b - slack + f32::EPSILON),
        BoundType::GreaterEq(b) => value.max(b - slack),
        BoundType::Equal(target, tol) => {
            let widened = tol + slack;
            value.clamp(target - widened, target + widened)
        }
        BoundType::InRange(lo, hi) => {
            let (low, high) = (lo - slack, hi + slack);
            if low <= high {
                value.clamp(low, high)
            } else {
                // Degenerate (empty) range: fall back to the midpoint.
                0.5 * (low + high)
            }
        }
    }
}

impl Default for ConstraintRepairer {
    fn default() -> Self {
        Self::new(RepairStrategy::MinimalRelaxation)
    }
}

/// Minimal Infeasible Subset (IIS) finder
///
/// Identifies smallest subsets of constraints that are infeasible
pub struct IISFinder {
    /// Tolerance for feasibility
    tolerance: f32,
}

impl IISFinder {
    /// Create a new IIS finder
    pub fn new() -> Self {
        Self { tolerance: 1e-6 }
    }

    /// Set tolerance
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    /// Find all minimal infeasible subsets
    ///
    /// Uses deletion filter algorithm. Constraints are evaluated on the
    /// dimension they govern; see [`ConstraintRepairer::repair`] for the
    /// validation rules and errors.
    pub fn find_all_iis(
        &self,
        point: &[f32],
        constraints: &[Constraint],
    ) -> LogicResult<Vec<Vec<usize>>> {
        validate_point(point, constraints)?;

        let mut iis_sets = Vec::new();

        // Start with all violated constraints
        let violated: Vec<usize> = constraints
            .iter()
            .enumerate()
            .filter(|(_, c)| ViolationComputable::violation(*c, point) > self.tolerance)
            .map(|(i, _)| i)
            .collect();

        if violated.is_empty() {
            return Ok(iis_sets);
        }

        // Try to find minimal subsets using deletion filter
        let iis = self.deletion_filter(point, constraints, &violated)?;
        if !iis.is_empty() {
            iis_sets.push(iis);
        }

        Ok(iis_sets)
    }

    /// Deletion filter algorithm for finding IIS
    fn deletion_filter(
        &self,
        point: &[f32],
        constraints: &[Constraint],
        candidate: &[usize],
    ) -> LogicResult<Vec<usize>> {
        let mut current = candidate.to_vec();

        // Try removing each constraint to see if still infeasible
        let mut i = 0;
        while i < current.len() {
            let mut test_set = current.clone();
            test_set.remove(i);

            // Check if test_set is still infeasible
            if self.is_infeasible(point, constraints, &test_set) {
                // Can remove current[i]
                current = test_set;
            } else {
                // Cannot remove current[i], it's necessary
                i += 1;
            }
        }

        Ok(current)
    }

    /// Check if a subset of constraints is infeasible
    fn is_infeasible(
        &self,
        point: &[f32],
        all_constraints: &[Constraint],
        subset: &[usize],
    ) -> bool {
        subset.iter().any(|&i| {
            all_constraints
                .get(i)
                .map(|c| ViolationComputable::violation(c, point) > self.tolerance)
                .unwrap_or(false)
        })
    }
}

impl Default for IISFinder {
    fn default() -> Self {
        Self::new()
    }
}

/// A one-sided bound (lower or upper) of an interval, with strictness.
#[derive(Clone, Copy)]
struct OneSidedBound {
    value: f32,
    /// true = closed (≤ or ≥), false = open (< or >)
    inclusive: bool,
}

/// Extract the feasible interval [lower, upper] for a `Constraint`.
///
/// Returns `(lower_bound, upper_bound)` where `None` means unbounded
/// (−∞ for lower, +∞ for upper).
fn feasible_interval(c: &Constraint) -> (Option<OneSidedBound>, Option<OneSidedBound>) {
    match c.bound() {
        BoundType::LessThan(b) => (
            None,
            Some(OneSidedBound {
                value: *b,
                inclusive: false,
            }),
        ),
        BoundType::LessEq(b) => (
            None,
            Some(OneSidedBound {
                value: *b,
                inclusive: true,
            }),
        ),
        BoundType::GreaterThan(b) => (
            Some(OneSidedBound {
                value: *b,
                inclusive: false,
            }),
            None,
        ),
        BoundType::GreaterEq(b) => (
            Some(OneSidedBound {
                value: *b,
                inclusive: true,
            }),
            None,
        ),
        BoundType::Equal(v, tol) => (
            Some(OneSidedBound {
                value: v - tol,
                inclusive: true,
            }),
            Some(OneSidedBound {
                value: v + tol,
                inclusive: true,
            }),
        ),
        BoundType::InRange(lo, hi) => (
            Some(OneSidedBound {
                value: *lo,
                inclusive: true,
            }),
            Some(OneSidedBound {
                value: *hi,
                inclusive: true,
            }),
        ),
    }
}

/// Return `true` when upper bound `hi` is strictly less than lower bound `lo`,
/// meaning the intervals are disjoint.
///
/// Strict disjointness:
/// - If both bounds are inclusive: hi < lo
/// - If either bound is exclusive: hi <= lo (since hi and lo cannot both be reached)
fn intervals_disjoint(
    hi: OneSidedBound, // upper bound of the first interval
    lo: OneSidedBound, // lower bound of the second interval
) -> bool {
    if hi.value < lo.value {
        return true;
    }
    // hi.value == lo.value: disjoint iff at least one bound is exclusive
    hi.value == lo.value && !(hi.inclusive && lo.inclusive)
}

/// Conflict resolution for constraint sets
pub struct ConflictResolver {
    /// Strategy for resolving conflicts
    #[allow(dead_code)]
    resolution_strategy: RepairStrategy,
    /// Repairer instance
    repairer: ConstraintRepairer,
}

impl ConflictResolver {
    /// Create a new conflict resolver
    pub fn new(strategy: RepairStrategy) -> Self {
        Self {
            resolution_strategy: strategy,
            repairer: ConstraintRepairer::new(strategy),
        }
    }

    /// Resolve conflicts at a given point
    ///
    /// Returns a repaired point and list of relaxed constraints
    pub fn resolve(
        &self,
        point: &[f32],
        constraints: &[Constraint],
        priorities: Option<&[f32]>,
    ) -> LogicResult<RepairResult> {
        self.repairer.repair(point, constraints, priorities)
    }

    /// Get conflicting constraint pairs
    ///
    /// Identifies pairs of constraints that conflict with each other
    pub fn find_conflicts(&self, constraints: &[Constraint]) -> Vec<(usize, usize)> {
        let mut conflicts = Vec::new();

        // Structural conflict detection: pairs whose feasible intervals are
        // geometrically disjoint on the same variable cannot be satisfied
        // simultaneously.
        for i in 0..constraints.len() {
            for j in (i + 1)..constraints.len() {
                if self.might_conflict(&constraints[i], &constraints[j]) {
                    conflicts.push((i, j));
                }
            }
        }

        conflicts
    }

    /// Detect structural conflicts between two constraints by checking whether
    /// their feasible intervals on the same variable are geometrically disjoint.
    ///
    /// Detected cases:
    /// - Disjoint bound pairs: e.g. x ≥ 10 and x ≤ 5 on the same dimension
    /// - Strict boundary contradictions: x < 5 and x > 5
    /// - Equality contradictions: x == 3 (±0.01) and x == 7 (±0.01)
    /// - InRange intervals that do not overlap
    ///
    /// Conservative: returns `false` for constraints on different dimensions or
    /// when feasibility cannot be determined from structure alone.
    fn might_conflict(&self, c1: &Constraint, c2: &Constraint) -> bool {
        // If both constraints name an explicit dimension and they differ,
        // they constrain independent variables — no pairwise conflict possible.
        if let (Some(d1), Some(d2)) = (c1.dimension(), c2.dimension()) {
            if d1 != d2 {
                return false;
            }
        }

        let (lo1, hi1) = feasible_interval(c1);
        let (lo2, hi2) = feasible_interval(c2);

        // Check: does [lo1, hi1] ∩ [lo2, hi2] = ∅ ?
        //
        // Disjoint if hi1 < lo2  OR  hi2 < lo1
        // (where "less than" respects strictness of the bounds)

        if let (Some(hi), Some(lo)) = (hi1, lo2) {
            if intervals_disjoint(hi, lo) {
                return true;
            }
        }

        if let (Some(hi), Some(lo)) = (hi2, lo1) {
            if intervals_disjoint(hi, lo) {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintBuilder;

    #[test]
    fn test_uniform_repair() {
        let repairer = ConstraintRepairer::new(RepairStrategy::Uniform);

        let c1 = ConstraintBuilder::new()
            .name("c1")
            .less_than(5.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("c2")
            .greater_than(10.0)
            .build()
            .unwrap();

        // Point at 7.0 violates c2 (should be >= 10.0)
        let point = vec![7.0];
        let result = repairer.repair(&point, &[c1, c2], None).unwrap();

        assert!(result.success);
        assert!(!result.relaxed_constraints.is_empty());
    }

    #[test]
    fn test_priority_repair() {
        let repairer = ConstraintRepairer::new(RepairStrategy::PriorityBased);

        let c1 = ConstraintBuilder::new()
            .name("c1")
            .less_than(5.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("c2")
            .greater_than(10.0)
            .build()
            .unwrap();

        let point = vec![7.0];
        let priorities = vec![1.0, 2.0]; // c1 has lower priority

        let result = repairer
            .repair(&point, &[c1, c2], Some(&priorities))
            .unwrap();

        assert!(result.success);
    }

    #[test]
    fn test_minimal_repair() {
        let repairer = ConstraintRepairer::new(RepairStrategy::MinimalRelaxation);

        let c1 = ConstraintBuilder::new()
            .name("c1")
            .less_than(5.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("c2")
            .less_than(10.0)
            .build()
            .unwrap();

        // Point at 12.0 violates both constraints
        let point = vec![12.0];
        let result = repairer.repair(&point, &[c1, c2], None).unwrap();

        assert!(result.success);
        assert_eq!(result.num_relaxed(), 2);
    }

    #[test]
    fn test_elastic_repair() {
        let repairer = ConstraintRepairer::new(RepairStrategy::ElasticProgramming);

        let c1 = ConstraintBuilder::new()
            .name("c1")
            .less_than(5.0)
            .build()
            .unwrap();

        let point = vec![7.0];
        let result = repairer.repair(&point, &[c1], None).unwrap();

        assert!(result.success);
        assert!(result.repaired_point.is_some());
    }

    #[test]
    fn test_iis_finder() {
        let finder = IISFinder::new();

        let c1 = ConstraintBuilder::new()
            .name("c1")
            .less_than(5.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("c2")
            .greater_than(10.0)
            .build()
            .unwrap();

        // Point at 7.0: c1 wants <= 5.0, c2 wants >= 10.0
        let point = vec![7.0];
        let iis_sets = finder.find_all_iis(&point, &[c1, c2]).unwrap();

        // Should find at least one IIS
        assert!(!iis_sets.is_empty());
    }

    #[test]
    fn test_conflict_resolver() {
        let resolver = ConflictResolver::new(RepairStrategy::MinimalRelaxation);

        let c1 = ConstraintBuilder::new()
            .name("c1")
            .less_than(5.0)
            .build()
            .unwrap();

        let point = vec![7.0];
        let result = resolver.resolve(&point, &[c1], None).unwrap();

        assert!(result.success);
    }

    #[test]
    fn test_repair_cost() {
        let repairer = ConstraintRepairer::new(RepairStrategy::Uniform);

        let c1 = ConstraintBuilder::new()
            .name("c1")
            .less_than(5.0)
            .build()
            .unwrap();

        let point = vec![7.0];
        let result = repairer.repair(&point, &[c1], None).unwrap();

        // Repair cost should reflect the violation amount
        assert!(result.repair_cost > 0.0);
    }

    /// Verify that the gradient-descent repair moves the **tagged** dimension
    /// toward feasibility while leaving unrelated dimensions unchanged.
    ///
    /// Before the fix, `compute_repaired_point` always perturbed `x_plus[0]`
    /// and read back `x_plus[0]`, producing a zero gradient for `i > 0`, so
    /// dimension 1 was never updated.
    #[test]
    fn test_suggestions_move_tagged_dimension() {
        // 2-D initial: dim-0 = 0.0 (feasible), dim-1 = 3.0 (violates x <= 1.0)
        let initial = vec![0.0_f32, 3.0_f32];

        let constraint = ConstraintBuilder::new()
            .name("upper_dim1")
            .dimension(1)
            .less_than(1.0)
            .build()
            .expect("constraint build failed");

        let repairer = ConstraintRepairer::new(RepairStrategy::ElasticProgramming);
        let result = repairer
            .repair(&initial, &[constraint], None)
            .expect("repair failed");

        let repaired = result.repaired_point.expect("expected a repaired point");

        // Dimension 1 must have moved toward the bound (< 2.9 means meaningful progress)
        assert!(
            repaired[1] < 2.9,
            "dim-1 should have moved toward 1.0, got {}",
            repaired[1]
        );
        // Dimension 0 must stay at 0.0 – the constraint never touches it
        assert!(
            (repaired[0] - 0.0_f32).abs() < 1e-4,
            "dim-0 should remain ~0.0, got {}",
            repaired[0]
        );
    }

    /// Regression: a 1-D constraint with no dimension tag (defaults to dim-0)
    /// must still drive the repair toward feasibility.
    #[test]
    fn test_suggestions_dimension_zero_regression() {
        let initial = vec![3.0_f32];

        let constraint = ConstraintBuilder::new()
            .name("upper_dim0")
            .less_than(1.0)
            .build()
            .expect("constraint build failed");

        let repairer = ConstraintRepairer::new(RepairStrategy::ElasticProgramming);
        let result = repairer
            .repair(&initial, &[constraint], None)
            .expect("repair failed");

        let repaired = result.repaired_point.expect("expected a repaired point");

        assert!(
            repaired[0] < 3.0,
            "dim-0 should have moved toward 1.0, got {}",
            repaired[0]
        );
    }

    #[test]
    fn test_conflict_detection_disjoint_bounds() {
        // x >= 10 and x <= 5 on the same dimension — clearly infeasible
        let resolver = ConflictResolver::new(RepairStrategy::MinimalRelaxation);
        let c1 = ConstraintBuilder::new()
            .name("lb")
            .greater_eq(10.0)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("ub")
            .less_eq(5.0)
            .build()
            .unwrap();
        let conflicts = resolver.find_conflicts(&[c1, c2]);
        assert!(
            !conflicts.is_empty(),
            "disjoint bounds should be detected as conflict"
        );
        assert_eq!(conflicts[0], (0, 1));
    }

    #[test]
    fn test_conflict_detection_compatible() {
        // x >= 1 and x <= 10 — feasible
        let resolver = ConflictResolver::new(RepairStrategy::MinimalRelaxation);
        let c1 = ConstraintBuilder::new()
            .name("lb")
            .greater_eq(1.0)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("ub")
            .less_eq(10.0)
            .build()
            .unwrap();
        let conflicts = resolver.find_conflicts(&[c1, c2]);
        assert!(
            conflicts.is_empty(),
            "compatible bounds must not be flagged as conflict"
        );
    }

    #[test]
    fn test_conflict_detection_equality_contradiction() {
        // x == 3.0 (tol=0.01) and x == 7.0 (tol=0.01) — infeasible
        let resolver = ConflictResolver::new(RepairStrategy::MinimalRelaxation);
        let c1 = ConstraintBuilder::new()
            .name("eq1")
            .equal(3.0, 0.01)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("eq2")
            .equal(7.0, 0.01)
            .build()
            .unwrap();
        let conflicts = resolver.find_conflicts(&[c1, c2]);
        assert!(
            !conflicts.is_empty(),
            "contradictory equalities should be detected as conflict"
        );
    }

    #[test]
    fn test_conflict_detection_strict_bounds_at_boundary() {
        // x < 5.0 and x > 5.0 — no x satisfies both (strict gap at boundary)
        let resolver = ConflictResolver::new(RepairStrategy::MinimalRelaxation);
        let c1 = ConstraintBuilder::new()
            .name("strict_ub")
            .less_than(5.0)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("strict_lb")
            .greater_than(5.0)
            .build()
            .unwrap();
        let conflicts = resolver.find_conflicts(&[c1, c2]);
        assert!(
            !conflicts.is_empty(),
            "strict boundary contradiction should be detected"
        );
    }

    /// Regression (finding 127): every strategy must evaluate a constraint on
    /// the dimension it governs, not on `point[0]`.
    ///
    /// Constraints `x0 <= 1` and `x3 <= 1` with the point `[0.5, 0, 0, 99]`:
    /// only dimension 3 is violated. The old code computed both violations at
    /// `point[0] = 0.5`, reported "no violation" and still claimed success.
    #[test]
    fn test_repair_uses_tagged_dimension_not_first_element() {
        let c0 = ConstraintBuilder::new()
            .name("x0")
            .dimension(0)
            .less_eq(1.0)
            .build()
            .expect("c0 builds");
        let c3 = ConstraintBuilder::new()
            .name("x3")
            .dimension(3)
            .less_eq(1.0)
            .build()
            .expect("c3 builds");

        let point = vec![0.5_f32, 0.0, 0.0, 99.0];

        for strategy in [
            RepairStrategy::Uniform,
            RepairStrategy::PriorityBased,
            RepairStrategy::MinimalRelaxation,
        ] {
            let repairer = ConstraintRepairer::new(strategy);
            let result = repairer
                .repair(&point, &[c0.clone(), c3.clone()], None)
                .unwrap_or_else(|e| panic!("{strategy:?} failed: {e}"));

            assert_eq!(
                result.relaxed_constraints,
                vec![1],
                "{strategy:?} must report the dim-3 constraint as violated"
            );
            let amount = result.relaxation_amounts[0];
            assert!(
                (amount - 98.0).abs() < 1e-3,
                "{strategy:?} relaxation must equal the dim-3 violation (98), got {amount}"
            );
        }

        // Elastic must move dimension 3 and leave the others alone.
        let elastic = ConstraintRepairer::new(RepairStrategy::ElasticProgramming);
        let result = elastic
            .repair(&point, &[c0, c3], None)
            .expect("elastic repair");
        let repaired = result.repaired_point.expect("elastic point");
        assert!(
            (repaired[3] - 1.0).abs() < 1e-3,
            "dim-3 must be projected onto its bound, got {}",
            repaired[3]
        );
        assert!((repaired[0] - 0.5).abs() < 1e-5, "dim-0 must be untouched");
    }

    /// Regression (finding 127): `max_relaxation` must cap the reported
    /// relaxation *and* be reflected in `success` for every strategy, not only
    /// for elastic programming.
    #[test]
    fn test_repair_respects_max_relaxation_cap() {
        let constraint = ConstraintBuilder::new()
            .name("tight")
            .less_eq(1.0)
            .build()
            .expect("constraint builds");

        // Violation of 99 against a cap of 2.0.
        let point = vec![100.0_f32];

        for strategy in [
            RepairStrategy::Uniform,
            RepairStrategy::PriorityBased,
            RepairStrategy::MinimalRelaxation,
        ] {
            let repairer = ConstraintRepairer::new(strategy).with_max_relaxation(2.0);
            let result = repairer
                .repair(&point, std::slice::from_ref(&constraint), None)
                .unwrap_or_else(|e| panic!("{strategy:?} failed: {e}"));

            assert!(
                result.relaxation_amounts.iter().all(|&a| a <= 2.0 + 1e-6),
                "{strategy:?} must not report a relaxation above max_relaxation: {:?}",
                result.relaxation_amounts
            );
            // The point cannot stay where it is, so the repairer must move it
            // into the 2.0-relaxed region (x <= 3.0) to claim success.
            assert!(
                result.success,
                "{strategy:?} should repair by moving the point"
            );
            let repaired = result
                .repaired_point
                .unwrap_or_else(|| panic!("{strategy:?} must return a point on success"));
            assert!(
                repaired[0] <= 3.0 + 1e-3,
                "{strategy:?} point must fit inside the capped relaxation, got {}",
                repaired[0]
            );
        }
    }

    /// Regression (finding 127): a feasible point needs no repair, and
    /// `MinimalRelaxation` must report that as success (it used to return
    /// `success = !relaxed.is_empty()`, i.e. `false`).
    #[test]
    fn test_minimal_repair_success_on_feasible_point() {
        let repairer = ConstraintRepairer::new(RepairStrategy::MinimalRelaxation);
        let c = ConstraintBuilder::new()
            .name("c")
            .less_eq(5.0)
            .build()
            .expect("constraint builds");

        let result = repairer.repair(&[1.0], &[c], None).expect("repair");
        assert!(result.success, "a feasible point is trivially repaired");
        assert!(result.relaxed_constraints.is_empty());
        assert_eq!(result.repair_cost, 0.0);
    }

    /// Regression (finding 127): a constraint naming a dimension the point does
    /// not have must be rejected rather than silently treated as satisfied.
    #[test]
    fn test_repair_rejects_out_of_range_dimension() {
        let repairer = ConstraintRepairer::new(RepairStrategy::Uniform);
        let c = ConstraintBuilder::new()
            .name("x7")
            .dimension(7)
            .less_eq(1.0)
            .build()
            .expect("constraint builds");

        let result = repairer.repair(&[0.0, 1.0], &[c], None);
        assert!(
            matches!(result, Err(LogicError::DimensionMismatch { .. })),
            "out-of-range dimension must be a DimensionMismatch error"
        );
    }

    /// Uniform relaxation must be uniform: both violated constraints get the
    /// same relaxation (the largest violation).
    #[test]
    fn test_uniform_relaxation_is_equal_across_constraints() {
        let repairer = ConstraintRepairer::new(RepairStrategy::Uniform);
        let c1 = ConstraintBuilder::new()
            .name("c1")
            .less_eq(5.0)
            .build()
            .expect("c1");
        let c2 = ConstraintBuilder::new()
            .name("c2")
            .less_eq(10.0)
            .build()
            .expect("c2");

        // Violations are 7 and 2 → uniform relaxation of 7 for both.
        let result = repairer.repair(&[12.0], &[c1, c2], None).expect("repair");
        assert_eq!(result.num_relaxed(), 2);
        assert!(
            result
                .relaxation_amounts
                .iter()
                .all(|&a| (a - 7.0).abs() < 1e-4),
            "all amounts must equal the largest violation: {:?}",
            result.relaxation_amounts
        );
    }

    #[test]
    fn test_no_conflict_different_dimensions() {
        // x[0] >= 10 and x[1] <= 5 — on DIFFERENT dimensions, no conflict
        let resolver = ConflictResolver::new(RepairStrategy::MinimalRelaxation);
        let c1 = ConstraintBuilder::new()
            .name("dim0_lb")
            .dimension(0)
            .greater_eq(10.0)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("dim1_ub")
            .dimension(1)
            .less_eq(5.0)
            .build()
            .unwrap();
        let conflicts = resolver.find_conflicts(&[c1, c2]);
        assert!(
            conflicts.is_empty(),
            "constraints on different dimensions must not conflict"
        );
    }
}
