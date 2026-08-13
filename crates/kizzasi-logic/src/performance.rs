//! Performance optimizations for constraint checking
//!
//! This module provides batch operations, caching, and SIMD-friendly
//! implementations for efficient constraint evaluation.

use crate::constraint::ViolationComputable;
use rayon::prelude::*;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

// ============================================================================
// Batch Constraint Checking
// ============================================================================

/// Batch constraint checker for evaluating multiple points efficiently
pub struct BatchConstraintChecker<C> {
    constraints: Vec<C>,
    cache_enabled: bool,
    cache: HashMap<Vec<i32>, bool>, // Discretized cache key -> satisfaction result
    cache_resolution: f32,
}

impl<C: ViolationComputable> BatchConstraintChecker<C> {
    /// Create a new batch checker
    pub fn new(constraints: Vec<C>) -> Self {
        Self {
            constraints,
            cache_enabled: false,
            cache: HashMap::new(),
            cache_resolution: 0.1,
        }
    }

    /// Enable caching with specified resolution
    pub fn with_caching(mut self, resolution: f32) -> Self {
        self.cache_enabled = true;
        self.cache_resolution = resolution;
        self
    }

    /// Check multiple points in batch
    pub fn check_batch(&mut self, points: &Array2<f32>) -> Vec<bool> {
        let (n_points, _) = points.dim();
        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let point = points.row(i);
            let point_slice: Vec<f32> = point.iter().copied().collect();

            if self.cache_enabled {
                let key = self.discretize(&point_slice);
                if let Some(&cached) = self.cache.get(&key) {
                    results.push(cached);
                    continue;
                }

                let satisfied = self.check_point(&point_slice);
                self.cache.insert(key, satisfied);
                results.push(satisfied);
            } else {
                results.push(self.check_point(&point_slice));
            }
        }

        results
    }

    /// Check if a single point satisfies all constraints
    fn check_point(&self, point: &[f32]) -> bool {
        self.constraints.iter().all(|c| c.check(point))
    }

    /// Compute violations for batch of points
    pub fn violation_batch(&self, points: &Array2<f32>) -> Vec<f32> {
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

    /// Discretize point for caching
    fn discretize(&self, point: &[f32]) -> Vec<i32> {
        point
            .iter()
            .map(|&x| (x / self.cache_resolution).round() as i32)
            .collect()
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            entries: self.cache.len(),
            enabled: self.cache_enabled,
        }
    }

    /// Get constraint count
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub enabled: bool,
}

// ============================================================================
// Parallel Constraint Checking
// ============================================================================

/// Parallel constraint checker using rayon.
///
/// Splits the point batch across rayon's global thread pool; each point is
/// evaluated against every constraint independently, so the outer loop over
/// points is embarrassingly parallel.
pub struct ParallelConstraintChecker<C> {
    constraints: Vec<C>,
}

impl<C: ViolationComputable + Send + Sync> ParallelConstraintChecker<C> {
    /// Create a new parallel checker
    pub fn new(constraints: Vec<C>) -> Self {
        Self { constraints }
    }

    /// Check points in parallel across rayon's thread pool.
    pub fn check_batch(&self, points: &Array2<f32>) -> Vec<bool> {
        let (n_points, _) = points.dim();

        (0..n_points)
            .into_par_iter()
            .map(|i| {
                let point = points.row(i);
                let point_slice: Vec<f32> = point.iter().copied().collect();
                self.constraints.iter().all(|c| c.check(&point_slice))
            })
            .collect()
    }

    /// Compute violations in parallel across rayon's thread pool.
    pub fn violation_batch(&self, points: &Array2<f32>) -> Vec<f32> {
        let (n_points, _) = points.dim();

        (0..n_points)
            .into_par_iter()
            .map(|i| {
                let point = points.row(i);
                let point_slice: Vec<f32> = point.iter().copied().collect();
                self.constraints
                    .iter()
                    .map(|c| c.violation(&point_slice))
                    .sum()
            })
            .collect()
    }
}

// ============================================================================
// Lazy Constraint Evaluation
// ============================================================================

/// Lazy constraint evaluator that skips evaluation when not needed
pub struct LazyConstraintEvaluator<C> {
    constraints: Vec<(C, bool)>, // (constraint, is_critical)
}

impl<C: ViolationComputable> LazyConstraintEvaluator<C> {
    /// Create a new lazy evaluator
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Add a constraint with criticality flag
    pub fn add_constraint(&mut self, constraint: C, is_critical: bool) {
        self.constraints.push((constraint, is_critical));
    }

    /// Check constraints lazily: stop immediately on the first *critical*
    /// violation, but still evaluate every non-critical constraint so the
    /// returned `bool` reflects true full satisfaction.
    ///
    /// Returns `(all_satisfied, stopped_at)`. `stopped_at` is the index of
    /// the first critical violation, or `self.constraints.len()` if none was
    /// found (regardless of whether a non-critical constraint was violated).
    ///
    /// Previously this returned `true` whenever no *critical* constraint was
    /// violated, silently reporting "satisfied" for a point that violated
    /// every non-critical constraint — indistinguishable from genuine full
    /// satisfaction to a caller that only looked at the `bool`.
    pub fn check_lazy(&self, point: &[f32]) -> (bool, usize) {
        let mut non_critical_violated = false;
        for (i, (constraint, is_critical)) in self.constraints.iter().enumerate() {
            if !constraint.check(point) {
                if *is_critical {
                    // Critical constraint violated, stop immediately.
                    return (false, i);
                }
                non_critical_violated = true;
            }
        }
        (!non_critical_violated, self.constraints.len())
    }

    /// Compute violation with early stopping
    pub fn violation_lazy(&self, point: &[f32], threshold: f32) -> (f32, bool) {
        let mut total_violation = 0.0;

        for (constraint, is_critical) in &self.constraints {
            let viol = constraint.violation(point);
            total_violation += viol;

            // Early stop if violation exceeds threshold
            if *is_critical && viol > threshold {
                return (total_violation, true);
            }
        }

        (total_violation, false)
    }
}

impl<C: ViolationComputable> Default for LazyConstraintEvaluator<C> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Vectorized Constraint Operations
// ============================================================================

/// Vectorized operations for efficient batch processing
pub struct VectorizedConstraints<C> {
    constraints: Vec<C>,
}

impl<C: ViolationComputable> VectorizedConstraints<C> {
    /// Create vectorized constraint evaluator
    pub fn new(constraints: Vec<C>) -> Self {
        Self { constraints }
    }

    /// Evaluate all constraints on all points, returning matrix of violations
    pub fn violation_matrix(&self, points: &Array2<f32>) -> Array2<f32> {
        let (n_points, _dim) = points.dim();
        let n_constraints = self.constraints.len();

        let mut violations = Array2::zeros((n_points, n_constraints));

        for i in 0..n_points {
            let point = points.row(i);
            let point_slice: Vec<f32> = point.iter().copied().collect();

            for (j, constraint) in self.constraints.iter().enumerate() {
                violations[[i, j]] = constraint.violation(&point_slice);
            }
        }

        violations
    }

    /// Get satisfaction matrix (bool for each point and constraint)
    pub fn satisfaction_matrix(&self, points: &Array2<f32>) -> Vec<Vec<bool>> {
        let (n_points, _) = points.dim();
        let mut satisfaction = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let point = points.row(i);
            let point_slice: Vec<f32> = point.iter().copied().collect();

            let row: Vec<bool> = self
                .constraints
                .iter()
                .map(|c| c.check(&point_slice))
                .collect();

            satisfaction.push(row);
        }

        satisfaction
    }

    /// Count violations per constraint across all points
    pub fn violation_counts(&self, points: &Array2<f32>) -> Vec<usize> {
        let (n_points, _) = points.dim();
        let mut counts = vec![0; self.constraints.len()];

        for i in 0..n_points {
            let point = points.row(i);
            let point_slice: Vec<f32> = point.iter().copied().collect();

            for (j, constraint) in self.constraints.iter().enumerate() {
                if !constraint.check(&point_slice) {
                    counts[j] += 1;
                }
            }
        }

        counts
    }
}

// ============================================================================
// Adaptive Constraint Ordering
// ============================================================================

/// Adaptive constraint ordering for efficient early termination
///
/// The constraint permutation is maintained incrementally (see
/// [`Self::check_adaptive`]) rather than re-sorted from scratch on every
/// call, so the ordering overhead does not dominate the checks it exists to
/// reduce.
pub struct AdaptiveConstraintOrder<C> {
    constraints: Vec<C>,
    violation_counts: Vec<usize>,
    /// Permutation of `0..constraints.len()`, kept sorted by descending
    /// `violation_counts` so the most-frequently-violated constraint is
    /// checked first.
    order: Vec<usize>,
    check_count: usize,
}

impl<C: ViolationComputable> AdaptiveConstraintOrder<C> {
    /// Create new adaptive ordering
    pub fn new(constraints: Vec<C>) -> Self {
        let n = constraints.len();
        Self {
            constraints,
            violation_counts: vec![0; n],
            order: (0..n).collect(),
            check_count: 0,
        }
    }

    /// Check constraints in adaptive order.
    ///
    /// On a violation, the violated constraint's count is incremented and it
    /// is bubbled toward the front of `order` only as far as its new count
    /// warrants — amortized O(1) bookkeeping per call instead of a full
    /// `O(n log n)` re-sort plus a fresh heap allocation every time.
    pub fn check_adaptive(&mut self, point: &[f32]) -> bool {
        self.check_count += 1;

        for pos in 0..self.order.len() {
            let i = self.order[pos];
            let Some(constraint) = self.constraints.get(i) else {
                continue;
            };
            if !constraint.check(point) {
                if let Some(count) = self.violation_counts.get_mut(i) {
                    *count += 1;
                }
                self.promote(pos);
                return false;
            }
        }

        true
    }

    /// Move `order[pos]` toward the front while its violation count exceeds
    /// its predecessor's, keeping `order` sorted by descending count.
    fn promote(&mut self, mut pos: usize) {
        while pos > 0 {
            let prev = pos - 1;
            let current = self.order[pos];
            let predecessor = self.order[prev];
            if self.violation_counts[current] > self.violation_counts[predecessor] {
                self.order.swap(pos, prev);
                pos = prev;
            } else {
                break;
            }
        }
    }

    /// Get violation statistics
    pub fn get_statistics(&self) -> Vec<(usize, f32)> {
        self.violation_counts
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let rate = if self.check_count > 0 {
                    count as f32 / self.check_count as f32
                } else {
                    0.0
                };
                (i, rate)
            })
            .collect()
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.violation_counts.fill(0);
        self.check_count = 0;
        // The stale counts backing `order` are gone; restore identity order
        // (O(n), but only on an explicit reset — not per check).
        for (idx, slot) in self.order.iter_mut().enumerate() {
            *slot = idx;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintBuilder;

    #[test]
    fn test_batch_checking() {
        let c1 = ConstraintBuilder::new()
            .name("x_positive")
            .greater_eq(0.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("x_bounded")
            .less_eq(10.0)
            .build()
            .unwrap();

        let mut checker = BatchConstraintChecker::new(vec![c1, c2]);

        // Create batch of points
        let points = Array2::from_shape_vec(
            (4, 1),
            vec![
                -1.0, // violates c1
                5.0,  // satisfies both
                15.0, // violates c2
                3.0,  // satisfies both
            ],
        )
        .unwrap();

        let results = checker.check_batch(&points);
        assert_eq!(results, vec![false, true, false, true]);
    }

    /// Regression (finding 128): `ParallelConstraintChecker` used to be a
    /// plain sequential loop despite its name, docs, and `Send + Sync`
    /// bounds. Assert parity with `BatchConstraintChecker` on the same
    /// batch (this is also the crate's first test coverage of the type at
    /// all — it previously had none).
    #[test]
    fn test_parallel_constraint_checker_matches_batch_checker() {
        let c1 = ConstraintBuilder::new()
            .name("x_positive")
            .greater_eq(0.0)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("x_bounded")
            .less_eq(10.0)
            .build()
            .unwrap();
        let c1b = ConstraintBuilder::new()
            .name("x_positive")
            .greater_eq(0.0)
            .build()
            .unwrap();
        let c2b = ConstraintBuilder::new()
            .name("x_bounded")
            .less_eq(10.0)
            .build()
            .unwrap();

        let points = Array2::from_shape_vec(
            (5, 1),
            vec![
                -1.0, // violates c1
                5.0,  // satisfies both
                15.0, // violates c2
                3.0,  // satisfies both
                0.0,  // boundary: satisfies both
            ],
        )
        .unwrap();

        let mut batch_checker = BatchConstraintChecker::new(vec![c1, c2]);
        let parallel_checker = ParallelConstraintChecker::new(vec![c1b, c2b]);

        assert_eq!(
            parallel_checker.check_batch(&points),
            batch_checker.check_batch(&points)
        );

        let batch_violations = batch_checker.violation_batch(&points);
        let parallel_violations = parallel_checker.violation_batch(&points);
        assert_eq!(batch_violations.len(), parallel_violations.len());
        for (b, p) in batch_violations.iter().zip(parallel_violations.iter()) {
            assert!((b - p).abs() < 1e-6, "batch={b} parallel={p}");
        }
    }

    #[test]
    fn test_batch_violations() {
        let c = ConstraintBuilder::new()
            .name("bound")
            .less_eq(5.0)
            .build()
            .unwrap();

        let checker = BatchConstraintChecker::new(vec![c]);

        let points = Array2::from_shape_vec((3, 1), vec![3.0, 7.0, 10.0]).unwrap();
        let violations = checker.violation_batch(&points);

        assert_eq!(violations[0], 0.0); // 3 <= 5, no violation
        assert_eq!(violations[1], 2.0); // 7 - 5 = 2
        assert_eq!(violations[2], 5.0); // 10 - 5 = 5
    }

    #[test]
    fn test_caching() {
        let c = ConstraintBuilder::new()
            .name("test")
            .in_range(0.0, 10.0)
            .build()
            .unwrap();

        let mut checker = BatchConstraintChecker::new(vec![c]).with_caching(0.1);

        let points = Array2::from_shape_vec((2, 1), vec![5.0, 5.05]).unwrap();
        let _ = checker.check_batch(&points);

        let stats = checker.cache_stats();
        assert!(stats.enabled);
        // Both points discretize to same bucket with resolution 0.1
        assert!(stats.entries >= 1);
    }

    #[test]
    fn test_lazy_evaluation() {
        let c1 = ConstraintBuilder::new()
            .name("critical")
            .greater_eq(0.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("non_critical")
            .less_eq(100.0)
            .build()
            .unwrap();

        let mut evaluator = LazyConstraintEvaluator::new();
        evaluator.add_constraint(c1, true); // critical
        evaluator.add_constraint(c2, false); // non-critical

        // Violates critical constraint, should stop early
        let (satisfied, stopped_at) = evaluator.check_lazy(&[-1.0]);
        assert!(!satisfied);
        assert_eq!(stopped_at, 0);

        // Satisfies all
        let (satisfied, stopped_at) = evaluator.check_lazy(&[5.0]);
        assert!(satisfied);
        assert_eq!(stopped_at, 2);
    }

    /// Regression (finding 142): a violated *non-critical* constraint must
    /// not be reported as "satisfied". Previously `check_lazy` returned
    /// `true` whenever no critical constraint was violated, discarding the
    /// non-critical result entirely.
    #[test]
    fn test_lazy_evaluation_reports_non_critical_violation() {
        let c1 = ConstraintBuilder::new()
            .name("critical")
            .greater_eq(0.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("non_critical")
            .less_eq(100.0)
            .build()
            .unwrap();

        let mut evaluator = LazyConstraintEvaluator::new();
        evaluator.add_constraint(c1, true); // critical
        evaluator.add_constraint(c2, false); // non-critical

        // Satisfies the critical constraint but violates the non-critical
        // one: must NOT be reported as fully satisfied.
        let (satisfied, stopped_at) = evaluator.check_lazy(&[500.0]);
        assert!(
            !satisfied,
            "a violated non-critical constraint must not read as satisfied"
        );
        // No *critical* violation was found, so the scan still ran to completion.
        assert_eq!(stopped_at, 2);
    }

    #[test]
    fn test_adaptive_ordering() {
        let c1 = ConstraintBuilder::new()
            .name("rarely_violated")
            .greater_eq(-100.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("often_violated")
            .less_eq(5.0)
            .build()
            .unwrap();

        let mut adaptive = AdaptiveConstraintOrder::new(vec![c1, c2]);

        // Check several points, c2 violated more often
        adaptive.check_adaptive(&[10.0]); // violates c2
        adaptive.check_adaptive(&[3.0]); // satisfies both
        adaptive.check_adaptive(&[15.0]); // violates c2

        let stats = adaptive.get_statistics();
        assert!(stats[1].1 > stats[0].1); // c2 violated more frequently
    }
}
