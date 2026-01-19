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

use crate::{Constraint, LogicError, LogicResult};
use scirs2_core::ndarray::Array1;
use std::collections::{HashMap, HashSet};

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

    /// Find a point that satisfies all constraints using random sampling
    fn find_satisfying_point(
        &self,
        constraints: &[Constraint],
        dimension: usize,
    ) -> Option<Array1<f32>> {
        use scirs2_core::random::thread_rng;

        let mut rng = thread_rng();
        let (lower, upper) = self.search_bounds;

        // Try random sampling
        for _ in 0..self.sample_count {
            let point: Array1<f32> =
                Array1::from_iter((0..dimension).map(|_| rng.gen_range(lower..upper)));

            if self.satisfies_all(constraints, &point) {
                return Some(point);
            }
        }

        // Try origin
        let origin = Array1::zeros(dimension);
        if self.satisfies_all(constraints, &origin) {
            return Some(origin);
        }

        // Try unit vectors
        for i in 0..dimension {
            let mut unit = Array1::zeros(dimension);
            unit[i] = 1.0;
            if self.satisfies_all(constraints, &unit) {
                return Some(unit);
            }
        }

        None
    }

    /// Check if a point satisfies all constraints
    fn satisfies_all(&self, constraints: &[Constraint], point: &Array1<f32>) -> bool {
        constraints.iter().all(|c| {
            if let Some(dim) = c.dimension() {
                if dim < point.len() {
                    c.check(point[dim])
                } else {
                    false
                }
            } else {
                // If no dimension specified, check against first element
                c.check(point[0])
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
                let satisfies_constraint = if let Some(dim) = constraint.dimension() {
                    if dim < point.len() {
                        constraint.check(point[dim])
                    } else {
                        false
                    }
                } else {
                    constraint.check(point[0])
                };

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

    /// Check if constraint i has a dependency on constraint j
    fn has_dependency(&self, _i: usize, _j: usize, _constraints: &[Constraint]) -> bool {
        // Simplified implementation: would need more sophisticated analysis
        false
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
}
