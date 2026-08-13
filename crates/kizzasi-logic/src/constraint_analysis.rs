//! Constraint Analysis and Verification
//!
//! This module provides tools for analyzing constraint sets to detect:
//! - Inconsistent (unsatisfiable) constraints
//! - Redundant constraints
//! - Constraint dependencies
//! - Minimal constraint sets
//!
//! # Examples
//!
//! ```
//! use kizzasi_logic::{ConstraintBuilder, ConstraintConsistencyChecker};
//! use scirs2_core::ndarray::Array1;
//!
//! // Create constraints
//! let c1 = ConstraintBuilder::new()
//!     .name("lower_bound")
//!     .greater_than(0.0)
//!     .build()
//!     .unwrap();
//!
//! let c2 = ConstraintBuilder::new()
//!     .name("upper_bound")
//!     .less_than(10.0)
//!     .build()
//!     .unwrap();
//!
//! // Check consistency
//! let checker = ConstraintConsistencyChecker::new();
//! let analysis = checker.analyze(&[c1, c2]);
//!
//! assert!(analysis.is_consistent);
//! ```

use crate::constraint::BoundInterval;
use crate::{Constraint, LogicError, LogicResult};
use scirs2_core::ndarray::Array1;
use std::collections::{HashMap, HashSet};

/// Split `constraints` into the feasible interval implied by the "global"
/// (untagged, `dimension() == None`) constraints, and a map from each
/// explicitly tagged dimension to the interval implied by the constraints
/// tagged for it. A global constraint applies to *every* dimension, so a
/// tagged dimension's true feasible interval is that map entry intersected
/// with the global interval (not the map entry alone).
///
/// This is `O(constraints.len())` regardless of how large a `dimension()`
/// tag is — no array sized by the tag is ever allocated — which is what
/// lets [`ConstraintConsistencyChecker::analyze`] decide feasibility exactly
/// instead of falling back to bounded random sampling.
fn dimension_intervals(
    constraints: &[Constraint],
) -> (BoundInterval, HashMap<usize, BoundInterval>) {
    let mut global = BoundInterval::UNBOUNDED;
    let mut tagged: HashMap<usize, BoundInterval> = HashMap::new();
    for c in constraints {
        let interval = c.bound().interval();
        match c.dimension() {
            None => global = global.intersect(&interval),
            Some(d) => {
                let entry = tagged.entry(d).or_insert(BoundInterval::UNBOUNDED);
                *entry = entry.intersect(&interval);
            }
        }
    }
    (global, tagged)
}

/// Result of constraint consistency analysis
#[derive(Debug, Clone)]
pub struct ConsistencyAnalysis {
    /// Whether the constraint set is consistent (satisfiable)
    pub is_consistent: bool,

    /// Indices of redundant constraints that can be removed
    pub redundant_constraints: Vec<usize>,

    /// Minimal set of constraint indices needed (non-redundant subset)
    pub minimal_set: Vec<usize>,

    /// Constraint dependency graph: constraint_id -> dependencies
    pub dependencies: HashMap<usize, Vec<usize>>,

    /// If inconsistent, minimal unsatisfiable subset
    pub minimal_unsatisfiable_subset: Option<Vec<usize>>,

    /// Sample point that satisfies constraints (if consistent)
    pub sample_point: Option<Array1<f32>>,
}

impl ConsistencyAnalysis {
    /// Create a new analysis result
    pub fn new(is_consistent: bool) -> Self {
        Self {
            is_consistent,
            redundant_constraints: Vec::new(),
            minimal_set: Vec::new(),
            dependencies: HashMap::new(),
            minimal_unsatisfiable_subset: None,
            sample_point: None,
        }
    }

    /// Get the number of redundant constraints
    pub fn redundancy_count(&self) -> usize {
        self.redundant_constraints.len()
    }

    /// Get the minimal set size
    pub fn minimal_size(&self) -> usize {
        self.minimal_set.len()
    }

    /// Check if a specific constraint is redundant
    pub fn is_redundant(&self, index: usize) -> bool {
        self.redundant_constraints.contains(&index)
    }
}

/// Constraint Consistency Checker
///
/// Analyzes constraint sets for consistency, redundancy, and dependencies.
pub struct ConstraintConsistencyChecker {
    /// Number of sample points to test for consistency
    sample_count: usize,

    /// Search space bounds for sampling
    search_bounds: (f32, f32),

    /// Maximum dimension to analyze
    max_dimension: usize,

    /// Tolerance for constraint satisfaction
    tolerance: f32,
}

impl Default for ConstraintConsistencyChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintConsistencyChecker {
    /// Create a new consistency checker with default parameters
    pub fn new() -> Self {
        Self {
            sample_count: 1000,
            search_bounds: (-100.0, 100.0),
            max_dimension: 100,
            tolerance: 1e-6,
        }
    }

    /// Set the number of sample points for consistency checking
    pub fn with_sample_count(mut self, count: usize) -> Self {
        self.sample_count = count;
        self
    }

    /// Set the search bounds for sampling
    pub fn with_search_bounds(mut self, lower: f32, upper: f32) -> Self {
        self.search_bounds = (lower, upper);
        self
    }

    /// Set the maximum dimension to analyze
    pub fn with_max_dimension(mut self, dim: usize) -> Self {
        self.max_dimension = dim;
        self
    }

    /// Set the tolerance for constraint satisfaction
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    /// Analyze a set of constraints
    pub fn analyze(&self, constraints: &[Constraint]) -> ConsistencyAnalysis {
        if constraints.is_empty() {
            let mut analysis = ConsistencyAnalysis::new(true);
            analysis.sample_point = Some(Array1::zeros(1));
            return analysis;
        }

        // Infer dimensionality from constraints
        let dimension = self.infer_dimension(constraints);

        // Try to find a satisfying point
        let satisfying_point = self.find_satisfying_point(constraints, dimension);

        let is_consistent = satisfying_point.is_some();
        let mut analysis = ConsistencyAnalysis::new(is_consistent);
        analysis.sample_point = satisfying_point;

        if is_consistent {
            // Analyze redundancy for consistent constraint sets
            self.analyze_redundancy(constraints, &mut analysis);
            self.build_dependency_graph(constraints, &mut analysis);
        } else {
            // Find minimal unsatisfiable subset
            analysis.minimal_unsatisfiable_subset =
                self.find_minimal_unsatisfiable_subset(constraints);
        }

        analysis
    }

    /// Check if constraints are consistent (satisfiable)
    pub fn is_consistent(&self, constraints: &[Constraint]) -> bool {
        self.analyze(constraints).is_consistent
    }

    /// Find redundant constraints
    pub fn find_redundant(&self, constraints: &[Constraint]) -> Vec<usize> {
        self.analyze(constraints).redundant_constraints
    }

    /// Infer dimension from constraints
    fn infer_dimension(&self, constraints: &[Constraint]) -> usize {
        // Find the maximum dimension referenced in constraints
        let max_dim = constraints
            .iter()
            .filter_map(|c| c.dimension())
            .max()
            .unwrap_or(0);

        // Return at least 1, and the max dimension + 1 (since dimensions are 0-indexed)
        (max_dim + 1).max(1).min(self.max_dimension)
    }

    /// Decide whether `constraints` is satisfiable, and if so, produce a
    /// witness point.
    ///
    /// `Constraint` is 1-D interval arithmetic over a
    /// [`BoundType`](crate::constraint::BoundType): this decides
    /// satisfiability exactly,
    /// by intersecting the per-dimension intervals every constraint implies
    /// (see [`dimension_intervals`]), rather than drawing bounded random
    /// samples from `self.search_bounds` and hoping one lands in the
    /// feasible region. That removes two false-negative failure modes the
    /// old sampling approach had: a feasible region entirely outside
    /// `search_bounds` (e.g. a single `greater_eq(1000.0)` constraint with
    /// the default `[-100, 100]` window), and a `dimension()` tag at or
    /// beyond `self.max_dimension` (the verdict below never depends on that
    /// cap; only the returned witness array's length is bounded by it, to
    /// avoid materializing an array sized by a pathological tag).
    fn find_satisfying_point(
        &self,
        constraints: &[Constraint],
        dimension: usize,
    ) -> Option<Array1<f32>> {
        let (global, tagged) = dimension_intervals(constraints);

        if global.is_empty() {
            return None;
        }
        for interval in tagged.values() {
            if interval.intersect(&global).is_empty() {
                return None;
            }
        }

        // Feasible: build a concrete witness. Its length is capped by
        // `self.max_dimension` purely to bound the allocation for a
        // pathologically large dimension tag — the feasibility verdict
        // above never depended on that cap.
        let highest_tagged = tagged.keys().copied().max().map_or(0, |d| d + 1);
        let out_dim = dimension
            .max(highest_tagged)
            .max(1)
            .min(self.max_dimension.max(1));

        let mut point = Array1::<f32>::zeros(out_dim);
        for d in 0..out_dim {
            let interval = tagged.get(&d).map_or(global, |t| t.intersect(&global));
            if let Some(slot) = point.get_mut(d) {
                *slot = interval.witness().unwrap_or(0.0);
            }
        }
        Some(point)
    }

    /// Check if a point satisfies all constraints, treating a constraint as
    /// satisfied whenever its violation is at most `self.tolerance` (not
    /// only when it is exactly zero). This is what wires the `tolerance`
    /// field into behavior — it was previously stored by `with_tolerance`
    /// and read nowhere.
    fn satisfies_all(&self, constraints: &[Constraint], point: &Array1<f32>) -> bool {
        constraints.iter().all(|c| {
            let value = match c.dimension() {
                Some(dim) => point.get(dim).copied(),
                // If no dimension specified, check against first element.
                None => point.first().copied(),
            };
            match value {
                Some(v) => c.violation(v) <= self.tolerance,
                None => false,
            }
        })
    }

    /// Analyze redundancy in constraint set
    fn analyze_redundancy(&self, constraints: &[Constraint], analysis: &mut ConsistencyAnalysis) {
        let n = constraints.len();
        let mut redundant = HashSet::new();
        let mut minimal = Vec::new();

        // For each constraint, check if it's implied by others
        for i in 0..n {
            if redundant.contains(&i) {
                continue;
            }

            // Create subset without constraint i
            let subset: Vec<_> = constraints
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i && !redundant.contains(j))
                .map(|(_, c)| c)
                .collect();

            // Check if constraint i is redundant (implied by subset)
            if self.is_implied(&constraints[i], &subset) {
                redundant.insert(i);
            } else {
                minimal.push(i);
            }
        }

        analysis.redundant_constraints = redundant.into_iter().collect();
        analysis.redundant_constraints.sort_unstable();
        analysis.minimal_set = minimal;
    }

    /// Check if a constraint is implied by a subset of constraints
    fn is_implied(&self, constraint: &Constraint, subset: &[&Constraint]) -> bool {
        use scirs2_core::random::thread_rng;

        if subset.is_empty() {
            return false;
        }

        // Sample points that satisfy the subset
        let mut rng = thread_rng();
        let (lower, upper) = self.search_bounds;

        // Infer dimension
        let subset_owned: Vec<Constraint> = subset.iter().map(|c| (*c).clone()).collect();
        let dimension = self.infer_dimension(&subset_owned);

        // If we can find a point that satisfies subset but violates constraint,
        // then constraint is not implied
        for _ in 0..self.sample_count {
            let point: Array1<f32> =
                Array1::from_iter((0..dimension).map(|_| rng.gen_range(lower..upper)));

            // Check if point satisfies subset
            let satisfies_subset = self.satisfies_all(&subset_owned, &point);

            if satisfies_subset {
                // Check if it also satisfies the constraint
                let value = match constraint.dimension() {
                    Some(dim) => point.get(dim).copied(),
                    None => point.first().copied(),
                };
                let satisfies_constraint = value.is_some_and(|v| constraint.check(v));

                if !satisfies_constraint {
                    // Found a counterexample: constraint is not implied
                    return false;
                }
            }
        }

        // No counterexample found: likely implied
        true
    }

    /// Build constraint dependency graph
    fn build_dependency_graph(
        &self,
        constraints: &[Constraint],
        analysis: &mut ConsistencyAnalysis,
    ) {
        let n = constraints.len();

        for i in 0..n {
            let mut deps = Vec::new();

            for j in 0..n {
                if i == j {
                    continue;
                }

                // Check if constraint i depends on constraint j
                // (removing j would make i inconsistent or change its effect)
                if self.has_dependency(i, j, constraints) {
                    deps.push(j);
                }
            }

            if !deps.is_empty() {
                analysis.dependencies.insert(i, deps);
            }
        }
    }

    /// Check if constraint `i` has a dependency on constraint `j`.
    ///
    /// Two constraints are considered dependent when they can affect the
    /// same value — they target the same explicit `dimension()`, or either
    /// one is "global" (`dimension() == None`, so it applies to every
    /// dimension including whichever the other one targets) — *and* their
    /// [`BoundType`](crate::constraint::BoundType) feasible intervals
    /// intersect. This is closed-form from each constraint's bound, unlike
    /// the placeholder this replaced, which always returned `false`
    /// regardless of its (unread) parameters.
    fn has_dependency(&self, i: usize, j: usize, constraints: &[Constraint]) -> bool {
        let (Some(ci), Some(cj)) = (constraints.get(i), constraints.get(j)) else {
            return false;
        };

        let same_target = match (ci.dimension(), cj.dimension()) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        };
        if !same_target {
            return false;
        }

        !ci.bound()
            .interval()
            .intersect(&cj.bound().interval())
            .is_empty()
    }

    /// Find minimal unsatisfiable subset
    fn find_minimal_unsatisfiable_subset(&self, constraints: &[Constraint]) -> Option<Vec<usize>> {
        // Try to find a minimal subset that is unsatisfiable
        let n = constraints.len();

        // Start with all constraints
        let mut current_subset: Vec<usize> = (0..n).collect();

        // Try removing each constraint to see if subset becomes satisfiable
        loop {
            let mut reduced = false;

            for i in 0..current_subset.len() {
                let test_subset: Vec<Constraint> = current_subset
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &idx)| constraints[idx].clone())
                    .collect();

                // Check if test subset is still unsatisfiable
                let dimension = self.infer_dimension(&test_subset);
                if self
                    .find_satisfying_point(&test_subset, dimension)
                    .is_none()
                {
                    // Still unsatisfiable, remove this constraint
                    current_subset.remove(i);
                    reduced = true;
                    break;
                }
            }

            if !reduced {
                break;
            }
        }

        if current_subset.len() < n {
            Some(current_subset)
        } else {
            Some((0..n).collect())
        }
    }
}

/// Helper to validate constraint sets before use
pub fn validate_constraint_set(constraints: &[Constraint]) -> LogicResult<()> {
    let checker = ConstraintConsistencyChecker::new();
    let analysis = checker.analyze(constraints);

    if !analysis.is_consistent {
        return Err(LogicError::InfeasibleConstraint(format!(
            "Constraint set is inconsistent. Minimal unsatisfiable subset: {:?}",
            analysis.minimal_unsatisfiable_subset
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConstraintBuilder;

    #[test]
    fn test_consistent_constraints() {
        let c1 = ConstraintBuilder::new()
            .name("lower")
            .greater_than(0.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("upper")
            .less_than(10.0)
            .build()
            .unwrap();

        let checker = ConstraintConsistencyChecker::new();
        let analysis = checker.analyze(&[c1, c2]);

        assert!(analysis.is_consistent);
        assert!(analysis.sample_point.is_some());
    }

    #[test]
    fn test_inconsistent_constraints() {
        let c1 = ConstraintBuilder::new()
            .name("lower")
            .greater_than(10.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("upper")
            .less_than(5.0)
            .build()
            .unwrap();

        let checker = ConstraintConsistencyChecker::new();
        let analysis = checker.analyze(&[c1, c2]);

        assert!(!analysis.is_consistent);
        assert!(analysis.minimal_unsatisfiable_subset.is_some());
    }

    #[test]
    fn test_redundant_constraints() {
        let c1 = ConstraintBuilder::new()
            .name("lower")
            .greater_than(0.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("upper")
            .less_than(10.0)
            .build()
            .unwrap();

        // This constraint is redundant (implied by c1 and c2)
        let c3 = ConstraintBuilder::new()
            .name("middle")
            .in_range(0.0, 10.0)
            .build()
            .unwrap();

        let checker = ConstraintConsistencyChecker::new();
        let analysis = checker.analyze(&[c1, c2, c3]);

        assert!(analysis.is_consistent);
        // Note: redundancy detection may vary based on sampling
    }

    #[test]
    fn test_empty_constraint_set() {
        let checker = ConstraintConsistencyChecker::new();
        let analysis = checker.analyze(&[]);

        assert!(analysis.is_consistent);
        assert_eq!(analysis.redundancy_count(), 0);
    }

    #[test]
    fn test_single_constraint() {
        let c = ConstraintBuilder::new()
            .name("simple")
            .greater_than(0.0)
            .build()
            .unwrap();

        let checker = ConstraintConsistencyChecker::new();
        let analysis = checker.analyze(&[c]);

        assert!(analysis.is_consistent);
        assert_eq!(analysis.minimal_size(), 1);
    }

    #[test]
    fn test_validate_constraint_set_success() {
        let c1 = ConstraintBuilder::new()
            .name("c1")
            .greater_than(0.0)
            .build()
            .unwrap();

        assert!(validate_constraint_set(&[c1]).is_ok());
    }

    #[test]
    fn test_validate_constraint_set_failure() {
        let c1 = ConstraintBuilder::new()
            .name("c1")
            .greater_than(10.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("c2")
            .less_than(5.0)
            .build()
            .unwrap();

        assert!(validate_constraint_set(&[c1, c2]).is_err());
    }

    // -----------------------------------------------------------------------
    // Regression (finding 134): exact feasibility, not bounded sampling.
    // -----------------------------------------------------------------------

    /// The old bounded-sampling `find_satisfying_point` drew candidates only
    /// from `search_bounds` (default `[-100, 100]`) — a feasible region
    /// entirely outside that window, like `x >= 1000`, was reported
    /// inconsistent even though it is trivially satisfiable.
    #[test]
    fn test_feasible_region_far_outside_default_search_bounds() {
        let c = ConstraintBuilder::new()
            .name("far")
            .greater_eq(1000.0)
            .build()
            .unwrap();

        let checker = ConstraintConsistencyChecker::new();
        let analysis = checker.analyze(&[c]);

        assert!(
            analysis.is_consistent,
            "x >= 1000 is trivially satisfiable and must not be reported inconsistent"
        );
        let point = analysis.sample_point.expect("witness point");
        assert!(point[0] >= 1000.0);
    }

    /// A constraint dimension tag beyond `max_dimension` (default 100) used
    /// to make `infer_dimension` truncate the candidate vector, so
    /// `satisfies_all` always saw the tagged constraint as out of range and
    /// failed it — reporting a perfectly satisfiable set as inconsistent.
    #[test]
    fn test_feasible_region_with_dimension_tag_beyond_default_cap() {
        let c = ConstraintBuilder::new()
            .name("late_dim")
            .dimension(150)
            .in_range(0.0, 10.0)
            .build()
            .unwrap();

        let checker = ConstraintConsistencyChecker::new();
        let analysis = checker.analyze(&[c]);

        assert!(
            analysis.is_consistent,
            "a constraint tagged for dimension 150 is still satisfiable even though the \
             default max_dimension is 100"
        );
    }

    /// The exact interval intersection must still correctly report a
    /// genuinely empty region as inconsistent (not just stop reporting
    /// false negatives for the cases above).
    #[test]
    fn test_genuinely_inconsistent_region_on_the_same_dimension() {
        let c1 = ConstraintBuilder::new()
            .name("c1")
            .dimension(0)
            .greater_than(10.0)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("c2")
            .dimension(0)
            .less_than(5.0)
            .build()
            .unwrap();

        let checker = ConstraintConsistencyChecker::new();
        let analysis = checker.analyze(&[c1, c2]);
        assert!(!analysis.is_consistent);
    }

    /// A global (untagged) constraint intersected with a per-dimension tag
    /// must be honored: `x <= 5` globally plus `dimension(3) >= 10` is
    /// infeasible on dimension 3 even though dimension 3 has no bound of
    /// its own besides the global one.
    #[test]
    fn test_global_constraint_intersects_with_tagged_dimension() {
        let global = ConstraintBuilder::new()
            .name("global")
            .less_eq(5.0)
            .build()
            .unwrap();
        let tagged = ConstraintBuilder::new()
            .name("tagged")
            .dimension(3)
            .greater_eq(10.0)
            .build()
            .unwrap();

        let checker = ConstraintConsistencyChecker::new();
        let analysis = checker.analyze(&[global, tagged]);
        assert!(!analysis.is_consistent);
    }

    /// `tolerance` used to be stored by `with_tolerance` and read nowhere.
    /// It now relaxes `satisfies_all`'s (and therefore `is_implied`'s)
    /// satisfaction check.
    #[test]
    fn test_tolerance_is_applied_in_redundancy_analysis() {
        let checker = ConstraintConsistencyChecker::new().with_tolerance(0.5);
        // A point exactly at the boundary of `less_eq(10.0)` plus a tiny
        // violation (10.2) must be treated as satisfied within tolerance 0.5.
        let c = ConstraintBuilder::new()
            .name("c")
            .less_eq(10.0)
            .build()
            .unwrap();
        let point = Array1::from_vec(vec![10.2]);
        assert!(checker.satisfies_all(&[c], &point));
    }

    // -----------------------------------------------------------------------
    // Regression (finding 133): has_dependency is no longer hardcoded false.
    // -----------------------------------------------------------------------

    #[test]
    fn test_dependencies_detected_for_overlapping_same_dimension_constraints() {
        // Two constraints on the same dimension with overlapping feasible
        // intervals: [0, 10] and [5, 15] overlap on [5, 10].
        let c1 = ConstraintBuilder::new()
            .name("c1")
            .dimension(0)
            .in_range(0.0, 10.0)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("c2")
            .dimension(0)
            .in_range(5.0, 15.0)
            .build()
            .unwrap();

        let checker = ConstraintConsistencyChecker::new();
        let analysis = checker.analyze(&[c1, c2]);

        assert!(analysis.is_consistent);
        assert!(
            !analysis.dependencies.is_empty(),
            "overlapping same-dimension constraints must be reported as dependent, not an \
             always-empty map"
        );
        assert_eq!(analysis.dependencies.get(&0), Some(&vec![1]));
        assert_eq!(analysis.dependencies.get(&1), Some(&vec![0]));
    }

    #[test]
    fn test_no_dependency_for_disjoint_dimensions() {
        let c1 = ConstraintBuilder::new()
            .name("c1")
            .dimension(0)
            .in_range(0.0, 10.0)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("c2")
            .dimension(1)
            .in_range(0.0, 10.0)
            .build()
            .unwrap();

        let checker = ConstraintConsistencyChecker::new();
        let analysis = checker.analyze(&[c1, c2]);
        assert!(analysis.is_consistent);
        assert!(
            analysis.dependencies.is_empty(),
            "constraints on unrelated dimensions must not be reported as dependent"
        );
    }

    #[test]
    fn test_global_constraint_depends_on_every_tagged_dimension_it_overlaps() {
        let global = ConstraintBuilder::new()
            .name("global")
            .less_eq(20.0)
            .build()
            .unwrap();
        let tagged = ConstraintBuilder::new()
            .name("tagged")
            .dimension(5)
            .greater_eq(0.0)
            .build()
            .unwrap();

        let checker = ConstraintConsistencyChecker::new();
        let analysis = checker.analyze(&[global, tagged]);
        assert!(analysis.is_consistent);
        assert_eq!(analysis.dependencies.get(&0), Some(&vec![1]));
        assert_eq!(analysis.dependencies.get(&1), Some(&vec![0]));
    }
}
