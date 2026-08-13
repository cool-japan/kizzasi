//! Constraint Violation Explanation
//!
//! This module provides tools for explaining why constraints were violated:
//! - Minimal violating subset identification
//! - Constraint violation attribution
//! - Human-readable violation reports
//! - Counterfactual constraint analysis

use crate::error::{LogicError, LogicResult};
use crate::ViolationComputable;
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Explanation for a constraint violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationExplanation {
    /// Human-readable description
    pub description: String,
    /// Violated constraints (index -> violation amount)
    pub violated_constraints: HashMap<usize, f32>,
    /// Most responsible feature dimensions
    pub responsible_dimensions: Vec<(usize, f32)>, // (dimension, contribution)
    /// Suggested fixes
    pub suggestions: Vec<String>,
    /// Counterfactual: nearest feasible point
    pub nearest_feasible: Option<Array1<f32>>,
}

impl ViolationExplanation {
    /// Create a new violation explanation
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            violated_constraints: HashMap::new(),
            responsible_dimensions: Vec::new(),
            suggestions: Vec::new(),
            nearest_feasible: None,
        }
    }

    /// Add a violated constraint
    pub fn add_violated_constraint(&mut self, index: usize, violation: f32) {
        self.violated_constraints.insert(index, violation);
    }

    /// Add a responsible dimension
    pub fn add_responsible_dimension(&mut self, dim: usize, contribution: f32) {
        self.responsible_dimensions.push((dim, contribution));
        // Sort by contribution (descending)
        self.responsible_dimensions
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Add a suggestion
    pub fn add_suggestion(&mut self, suggestion: impl Into<String>) {
        self.suggestions.push(suggestion.into());
    }

    /// Set nearest feasible point
    pub fn set_nearest_feasible(&mut self, point: Array1<f32>) {
        self.nearest_feasible = Some(point);
    }

    /// Generate a human-readable report
    pub fn to_report(&self) -> String {
        let mut report = format!("=== Violation Explanation ===\n{}\n\n", self.description);

        if !self.violated_constraints.is_empty() {
            report.push_str("Violated Constraints:\n");
            let mut violations: Vec<_> = self.violated_constraints.iter().collect();
            violations.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (idx, viol) in violations {
                report.push_str(&format!(
                    "  - Constraint #{}: violation = {:.4}\n",
                    idx, viol
                ));
            }
            report.push('\n');
        }

        if !self.responsible_dimensions.is_empty() {
            report.push_str("Most Responsible Dimensions:\n");
            for (dim, contrib) in self.responsible_dimensions.iter().take(5) {
                report.push_str(&format!(
                    "  - Dimension {}: contribution = {:.4}\n",
                    dim, contrib
                ));
            }
            report.push('\n');
        }

        if !self.suggestions.is_empty() {
            report.push_str("Suggestions:\n");
            for suggestion in &self.suggestions {
                report.push_str(&format!("  - {}\n", suggestion));
            }
            report.push('\n');
        }

        if let Some(ref nearest) = self.nearest_feasible {
            report.push_str(&format!(
                "Nearest Feasible Point: {:?}\n",
                nearest.as_slice()
            ));
        }

        report
    }
}

/// Minimal violating subset finder
#[derive(Debug, Clone)]
pub struct MinimalViolatingSubsetFinder<C: ViolationComputable> {
    /// All constraints
    constraints: Vec<C>,
    /// Constraint names for reporting
    names: Vec<String>,
}

impl<C: ViolationComputable + Clone> MinimalViolatingSubsetFinder<C> {
    /// Create a new MVS finder
    ///
    /// # Errors
    ///
    /// [`LogicError::DimensionMismatch`] when `names` does not have one entry
    /// per constraint.
    pub fn new(constraints: Vec<C>, names: Vec<String>) -> LogicResult<Self> {
        if constraints.len() != names.len() {
            return Err(LogicError::DimensionMismatch {
                expected: constraints.len(),
                got: names.len(),
            });
        }
        Ok(Self { constraints, names })
    }

    /// Compute violation gradient for constraint `idx` at `point` via central finite differences.
    ///
    /// Returns a `Vec<f32>` of length `point.len()`. If the point is 0-dimensional, returns
    /// an empty vector. Uses `eps = 1e-4` as the finite difference step size.
    fn compute_violation_gradient(&self, idx: usize, point: &[f32]) -> Vec<f32> {
        let n = point.len();
        if n == 0 {
            return Vec::new();
        }
        let eps = 1e-4_f32;
        let mut grad = vec![0.0_f32; n];
        let c = &self.constraints[idx];
        for k in 0..n {
            let mut p_plus = point.to_vec();
            let mut p_minus = point.to_vec();
            p_plus[k] += eps;
            p_minus[k] -= eps;
            let v_plus = c.violation(&p_plus);
            let v_minus = c.violation(&p_minus);
            grad[k] = (v_plus - v_minus) / (2.0 * eps);
        }
        grad
    }

    /// Compute cosine similarity between two gradient vectors.
    ///
    /// Returns a value in [-1, 1]. Near-zero norms are regularised with `eps = 1e-10` to
    /// prevent division by zero.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b + 1e-10)
    }

    /// Find minimal violating subset for a given point.
    ///
    /// Uses gradient-based greedy shrink: sorts violated constraints by ascending violation
    /// magnitude, then removes any constraint whose gradient direction is dominated (cosine
    /// similarity > 0.9) by another constraint already in the subset.
    pub fn find_mvs(&self, point: &Array1<f32>) -> Vec<usize> {
        let point_slice = crate::array_utils::contiguous(point);

        // Step 1: collect all violated constraint indices.
        let violated: Vec<usize> = self
            .constraints
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.check(&point_slice))
            .map(|(i, _)| i)
            .collect();

        // Trivially minimal cases.
        if violated.len() <= 1 {
            return violated;
        }

        // Step 2: compute violation magnitudes and gradients for each violated constraint.
        let violations: Vec<(usize, f32, Vec<f32>)> = violated
            .iter()
            .map(|&idx| {
                let mag = self.constraints[idx].violation(&point_slice);
                let grad = self.compute_violation_gradient(idx, &point_slice);
                (idx, mag, grad)
            })
            .collect();

        // Step 3: sort by violation magnitude ascending (smallest violation first —
        // those are the best removal candidates for redundancy).
        let mut sorted = violations;
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Step 4: greedy shrink — maintain running `minimal` set.
        // We work with indices into `sorted`.
        let mut minimal_indices: Vec<usize> = (0..sorted.len()).collect();

        let mut remove_positions: Vec<usize> = Vec::new();

        for pos in 0..sorted.len() {
            if minimal_indices.len() == 1 {
                break; // Cannot shrink further.
            }

            // Check whether `pos` is still in minimal_indices.
            if !minimal_indices.contains(&pos) {
                continue;
            }

            let (_, _, ref grad_i) = sorted[pos];

            // A constraint with zero-norm gradient cannot be assessed — keep it.
            let norm_i: f32 = grad_i.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_i < 1e-10 {
                continue;
            }

            // Check if there exists any other j in minimal_indices with cos(i,j) > 0.9.
            let dominated = minimal_indices.iter().any(|&other_pos| {
                if other_pos == pos {
                    return false;
                }
                let (_, _, ref grad_j) = sorted[other_pos];
                let norm_j: f32 = grad_j.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_j < 1e-10 {
                    return false;
                }
                let cos = Self::cosine_similarity(grad_i, grad_j);
                cos > 0.9
            });

            if dominated {
                remove_positions.push(pos);
                minimal_indices.retain(|&p| p != pos);
            }
        }

        // Step 5: collect surviving constraint indices preserving original order.
        let mut result: Vec<usize> = minimal_indices.iter().map(|&pos| sorted[pos].0).collect();
        result.sort_unstable();

        // Safety invariant: never return empty when violations were non-empty.
        if result.is_empty() {
            // Fallback: return the constraint with the largest violation magnitude.
            if let Some((idx, _, _)) = sorted.last() {
                result.push(*idx);
            }
        }

        result
    }

    /// Explain violation with minimal subset
    pub fn explain(&self, point: &Array1<f32>) -> ViolationExplanation {
        let point_slice = crate::array_utils::contiguous(point);
        let mvs = self.find_mvs(point);

        let mut explanation = ViolationExplanation::new(format!(
            "Point violates {} out of {} constraints",
            mvs.len(),
            self.constraints.len()
        ));

        for &idx in &mvs {
            let violation = self.constraints[idx].violation(&point_slice);
            explanation.add_violated_constraint(idx, violation);
            explanation.add_suggestion(format!(
                "Fix constraint '{}' (violation: {:.4})",
                self.names[idx], violation
            ));
        }

        explanation
    }

    /// Get number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
}

/// Violation attribution analyzer
#[derive(Debug, Clone)]
pub struct ViolationAttributionAnalyzer<C: ViolationComputable> {
    /// Constraints to analyze
    constraints: Vec<C>,
    /// Feature names for reporting
    feature_names: Vec<String>,
}

impl<C: ViolationComputable + Clone> ViolationAttributionAnalyzer<C> {
    /// Create a new attribution analyzer
    pub fn new(constraints: Vec<C>, feature_names: Vec<String>) -> Self {
        Self {
            constraints,
            feature_names,
        }
    }

    /// Compute feature attribution for violations using finite differences
    pub fn attribute_violations(&self, point: &Array1<f32>) -> Vec<(usize, f32)> {
        let point_slice = crate::array_utils::contiguous(point);
        let mut attributions = Vec::new();

        // Compute total violation
        let total_violation: f32 = self
            .constraints
            .iter()
            .map(|c| c.violation(&point_slice).max(0.0))
            .sum();

        if total_violation < 1e-8 {
            return attributions; // No violations
        }

        // For each dimension, compute how much changing it would reduce violations
        for dim in 0..point.len() {
            let mut perturbed = point.clone();
            let epsilon = 0.01;

            // Try positive perturbation
            perturbed[dim] += epsilon;
            let viol_plus: f32 = self
                .constraints
                .iter()
                .map(|c| {
                    c.violation(&crate::array_utils::contiguous(&perturbed))
                        .max(0.0)
                })
                .sum();

            // Try negative perturbation
            perturbed[dim] = point[dim] - epsilon;
            let viol_minus: f32 = self
                .constraints
                .iter()
                .map(|c| {
                    c.violation(&crate::array_utils::contiguous(&perturbed))
                        .max(0.0)
                })
                .sum();

            // Sensitivity: how much violation changes with this dimension
            let sensitivity = ((viol_plus - total_violation).abs()
                + (viol_minus - total_violation).abs())
                / (2.0 * epsilon);

            attributions.push((dim, sensitivity));
        }

        // Sort by attribution (descending)
        attributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        attributions
    }

    /// Create explanation with attributions
    pub fn explain(&self, point: &Array1<f32>) -> ViolationExplanation {
        let point_slice = crate::array_utils::contiguous(point);
        let attributions = self.attribute_violations(point);

        let total_violation: f32 = self
            .constraints
            .iter()
            .map(|c| c.violation(&point_slice).max(0.0))
            .sum();

        let mut explanation =
            ViolationExplanation::new(format!("Total violation: {:.4}", total_violation));

        for (dim, attr) in &attributions {
            explanation.add_responsible_dimension(*dim, *attr);
        }

        // Add suggestions based on top attributions
        for (dim, _) in attributions.iter().take(3) {
            let feature_name = self
                .feature_names
                .get(*dim)
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            explanation.add_suggestion(format!(
                "Adjust feature '{}' (dimension {})",
                feature_name, dim
            ));
        }

        explanation
    }
}

/// Counterfactual analyzer for finding nearest feasible points
#[derive(Debug, Clone)]
pub struct CounterfactualAnalyzer<C: ViolationComputable> {
    /// Constraints defining feasible region
    constraints: Vec<C>,
    /// Maximum iterations for search
    max_iterations: usize,
    /// Step size for gradient descent
    step_size: f32,
}

impl<C: ViolationComputable + Clone> CounterfactualAnalyzer<C> {
    /// Create a new counterfactual analyzer
    pub fn new(constraints: Vec<C>, max_iterations: usize, step_size: f32) -> Self {
        Self {
            constraints,
            max_iterations,
            step_size,
        }
    }

    /// Find nearest feasible point using gradient descent
    pub fn find_nearest_feasible(&self, point: &Array1<f32>) -> Option<Array1<f32>> {
        let mut current = point.clone();
        let epsilon = 1e-4;

        for _ in 0..self.max_iterations {
            let current_slice = crate::array_utils::contiguous(&current);

            // Check if feasible
            let is_feasible = self.constraints.iter().all(|c| c.check(&current_slice));
            if is_feasible {
                return Some(current);
            }

            // Compute gradient of total violation
            let total_violation: f32 = self
                .constraints
                .iter()
                .map(|c| c.violation(&current_slice).max(0.0))
                .sum();

            if total_violation < 1e-6 {
                return Some(current);
            }

            // Compute gradient using finite differences
            for dim in 0..current.len() {
                let mut perturbed = current.clone();
                perturbed[dim] += epsilon;

                let viol_plus: f32 = self
                    .constraints
                    .iter()
                    .map(|c| {
                        c.violation(&crate::array_utils::contiguous(&perturbed))
                            .max(0.0)
                    })
                    .sum();

                let grad = (viol_plus - total_violation) / epsilon;

                // Gradient descent step
                current[dim] -= self.step_size * grad;
            }
        }

        // Failed to converge
        None
    }

    /// Generate counterfactual explanation
    pub fn explain(&self, point: &Array1<f32>) -> ViolationExplanation {
        let point_slice = crate::array_utils::contiguous(point);
        let total_violation: f32 = self
            .constraints
            .iter()
            .map(|c| c.violation(&point_slice).max(0.0))
            .sum();

        let mut explanation = ViolationExplanation::new(format!(
            "Searching for nearest feasible point (current violation: {:.4})",
            total_violation
        ));

        if let Some(nearest) = self.find_nearest_feasible(point) {
            // Compute distance
            let distance = point
                .iter()
                .zip(nearest.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();

            explanation.set_nearest_feasible(nearest.clone());
            explanation.add_suggestion(format!(
                "Move to nearest feasible point (distance: {:.4})",
                distance
            ));

            // Identify which dimensions changed most
            for dim in 0..point.len() {
                let change = (point[dim] - nearest[dim]).abs();
                if change > 1e-4 {
                    explanation.add_responsible_dimension(dim, change);
                }
            }
        } else {
            explanation
                .add_suggestion("Could not find feasible point in search radius".to_string());
        }

        explanation
    }
}

/// Unified violation explainer combining multiple strategies
#[derive(Debug, Clone)]
pub struct ViolationExplainer<C: ViolationComputable> {
    /// Minimal violating subset finder
    mvs_finder: Option<MinimalViolatingSubsetFinder<C>>,
    /// Attribution analyzer
    attribution_analyzer: Option<ViolationAttributionAnalyzer<C>>,
    /// Counterfactual analyzer
    counterfactual_analyzer: Option<CounterfactualAnalyzer<C>>,
}

impl<C: ViolationComputable + Clone> ViolationExplainer<C> {
    /// Create a new violation explainer
    pub fn new() -> Self {
        Self {
            mvs_finder: None,
            attribution_analyzer: None,
            counterfactual_analyzer: None,
        }
    }

    /// Set minimal violating subset finder
    ///
    /// # Errors
    ///
    /// [`LogicError::DimensionMismatch`] when `names` does not have one entry
    /// per constraint.
    pub fn with_mvs_finder(mut self, constraints: Vec<C>, names: Vec<String>) -> LogicResult<Self> {
        self.mvs_finder = Some(MinimalViolatingSubsetFinder::new(constraints, names)?);
        Ok(self)
    }

    /// Set attribution analyzer
    pub fn with_attribution_analyzer(
        mut self,
        constraints: Vec<C>,
        feature_names: Vec<String>,
    ) -> Self {
        self.attribution_analyzer = Some(ViolationAttributionAnalyzer::new(
            constraints,
            feature_names,
        ));
        self
    }

    /// Set counterfactual analyzer
    pub fn with_counterfactual_analyzer(
        mut self,
        constraints: Vec<C>,
        max_iterations: usize,
        step_size: f32,
    ) -> Self {
        self.counterfactual_analyzer = Some(CounterfactualAnalyzer::new(
            constraints,
            max_iterations,
            step_size,
        ));
        self
    }

    /// Generate comprehensive explanation
    pub fn explain(&self, point: &Array1<f32>) -> ViolationExplanation {
        let mut explanation = ViolationExplanation::new("Comprehensive Violation Analysis");

        // MVS analysis
        if let Some(ref mvs) = self.mvs_finder {
            let mvs_exp = mvs.explain(point);
            explanation.violated_constraints = mvs_exp.violated_constraints;
            explanation.suggestions.extend(mvs_exp.suggestions);
        }

        // Attribution analysis
        if let Some(ref attr) = self.attribution_analyzer {
            let attr_exp = attr.explain(point);
            explanation.responsible_dimensions = attr_exp.responsible_dimensions;
            explanation.suggestions.extend(attr_exp.suggestions);
        }

        // Counterfactual analysis
        if let Some(ref cf) = self.counterfactual_analyzer {
            let cf_exp = cf.explain(point);
            explanation.nearest_feasible = cf_exp.nearest_feasible;
            explanation.suggestions.extend(cf_exp.suggestions);
        }

        explanation
    }
}

impl<C: ViolationComputable + Clone> Default for ViolationExplainer<C> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearConstraint;

    #[test]
    fn test_violation_explanation() {
        let mut exp = ViolationExplanation::new("Test violation");
        exp.add_violated_constraint(0, 1.5);
        exp.add_responsible_dimension(0, 0.8);
        exp.add_suggestion("Reduce dimension 0");

        let report = exp.to_report();
        assert!(report.contains("Test violation"));
        assert!(report.contains("Constraint #0"));
    }

    #[test]
    fn test_mvs_finder() {
        // Create constraints: x <= 5 and x <= 3
        let constraints = vec![
            LinearConstraint::less_eq(vec![1.0], 5.0),
            LinearConstraint::less_eq(vec![1.0], 3.0),
        ];
        let names = vec!["c1".to_string(), "c2".to_string()];
        let finder = MinimalViolatingSubsetFinder::new(constraints, names).expect("matched names");
        assert!(
            MinimalViolatingSubsetFinder::new(finder.constraints.clone(), vec![]).is_err(),
            "a names/constraints length mismatch must be a recoverable error"
        );

        let point = Array1::from_vec(vec![7.0]); // Violates both
        let mvs = finder.find_mvs(&point);

        assert!(!mvs.is_empty());
        assert_eq!(finder.num_constraints(), 2);
    }

    #[test]
    fn test_attribution_analyzer() {
        let constraints = vec![LinearConstraint::less_eq(vec![1.0, 1.0], 5.0)];
        let features = vec!["x".to_string(), "y".to_string()];
        let analyzer = ViolationAttributionAnalyzer::new(constraints, features);

        let point = Array1::from_vec(vec![4.0, 3.0]); // sum = 7, violates <= 5
        let exp = analyzer.explain(&point);

        assert!(!exp.responsible_dimensions.is_empty());
    }

    #[test]
    fn test_counterfactual_analyzer() {
        let constraints = vec![LinearConstraint::less_eq(vec![1.0], 5.0)];
        let analyzer = CounterfactualAnalyzer::new(constraints, 100, 0.1);

        let point = Array1::from_vec(vec![10.0]); // Violates x <= 5
        let nearest = analyzer.find_nearest_feasible(&point);

        assert!(nearest.is_some());
        if let Some(feasible) = nearest {
            assert!(feasible[0] <= 5.0 + 1e-2); // Should be close to 5
        }
    }

    #[test]
    fn test_unified_explainer() {
        let constraints = vec![LinearConstraint::less_eq(vec![1.0], 5.0)];
        let names = vec!["x_max".to_string()];
        let features = vec!["x".to_string()];

        let explainer = ViolationExplainer::new()
            .with_mvs_finder(constraints.clone(), names)
            .expect("matched names")
            .with_attribution_analyzer(constraints.clone(), features)
            .with_counterfactual_analyzer(constraints, 100, 0.1);

        let point = Array1::from_vec(vec![10.0]);
        let exp = explainer.explain(&point);

        assert!(!exp.violated_constraints.is_empty());
        assert!(!exp.suggestions.is_empty());
    }

    // ---------------------------------------------------------------------------
    // Helper: simple per-dimension upper-bound constraint for MVS tests.
    // ---------------------------------------------------------------------------
    #[derive(Clone)]
    struct SimpleBound {
        dim: usize,
        upper: f32,
    }

    impl ViolationComputable for SimpleBound {
        fn check(&self, point: &[f32]) -> bool {
            point.get(self.dim).is_none_or(|&v| v <= self.upper)
        }
        fn violation(&self, point: &[f32]) -> f32 {
            point
                .get(self.dim)
                .map_or(0.0, |&v| (v - self.upper).max(0.0))
        }
    }

    // Test 1: feasible point produces empty MVS.
    #[test]
    fn test_mvs_returns_empty_for_feasible_point() {
        let constraints = vec![
            SimpleBound {
                dim: 0,
                upper: 10.0,
            },
            SimpleBound {
                dim: 1,
                upper: 10.0,
            },
        ];
        let names = vec!["c0".to_string(), "c1".to_string()];
        let finder = MinimalViolatingSubsetFinder::new(constraints, names).expect("matched names");

        // Point (3.0, 3.0) satisfies both x0 <= 10 and x1 <= 10.
        let point = Array1::from_vec(vec![3.0_f32, 3.0]);
        let mvs = finder.find_mvs(&point);
        assert!(
            mvs.is_empty(),
            "Expected empty MVS for feasible point, got {:?}",
            mvs
        );
    }

    // Test 2: exactly one violated constraint is returned as-is.
    #[test]
    fn test_mvs_single_violated_returned() {
        let constraints = vec![
            SimpleBound { dim: 0, upper: 5.0 },
            SimpleBound {
                dim: 1,
                upper: 10.0,
            },
        ];
        let names = vec!["c0".to_string(), "c1".to_string()];
        let finder = MinimalViolatingSubsetFinder::new(constraints, names).expect("matched names");

        // Point (8.0, 2.0): only constraint 0 (x0 <= 5) is violated.
        let point = Array1::from_vec(vec![8.0_f32, 2.0]);
        let mvs = finder.find_mvs(&point);
        assert_eq!(
            mvs.len(),
            1,
            "Expected exactly 1 element in MVS, got {:?}",
            mvs
        );
        assert_eq!(
            mvs[0], 0,
            "Expected violated constraint index 0, got {:?}",
            mvs
        );
    }

    // Test 3: two orthogonal constraints — both should remain in MVS.
    //
    // Constraint 0: x0 <= 3  (gradient direction: [1, 0])
    // Constraint 1: x1 <= 3  (gradient direction: [0, 1])
    // cos([1,0],[0,1]) == 0.0, well below 0.9 threshold → neither is dominated.
    #[test]
    fn test_mvs_independent_constraints_all_kept() {
        let constraints = vec![
            SimpleBound { dim: 0, upper: 3.0 },
            SimpleBound { dim: 1, upper: 3.0 },
        ];
        let names = vec!["c0".to_string(), "c1".to_string()];
        let finder = MinimalViolatingSubsetFinder::new(constraints, names).expect("matched names");

        // Both violated: x0 = 5 > 3, x1 = 5 > 3.
        let point = Array1::from_vec(vec![5.0_f32, 5.0]);
        let mut mvs = finder.find_mvs(&point);
        mvs.sort_unstable();
        assert_eq!(
            mvs.len(),
            2,
            "Expected both orthogonal constraints kept in MVS, got {:?}",
            mvs
        );
        assert_eq!(mvs, vec![0, 1]);
    }

    // Test 4: two parallel constraints on the same dimension — the lesser one is redundant.
    //
    // Constraint 0: x0 <= 6  (violation at x0=8: 2.0)  — smaller magnitude
    // Constraint 1: x0 <= 4  (violation at x0=8: 4.0)  — larger magnitude
    //
    // Both gradients are [1.0] (same direction). When sorted ascending by violation,
    // constraint 0 (viol=2) comes first. Its gradient is dominated by constraint 1 (cos=1).
    // So constraint 0 is removed; MVS = [1].
    #[test]
    fn test_mvs_parallel_constraints_redundant_removed() {
        let constraints = vec![
            SimpleBound { dim: 0, upper: 6.0 }, // index 0, smaller violation
            SimpleBound { dim: 0, upper: 4.0 }, // index 1, larger violation
        ];
        let names = vec!["c0".to_string(), "c1".to_string()];
        let finder = MinimalViolatingSubsetFinder::new(constraints, names).expect("matched names");

        // x0 = 8 violates both.
        let point = Array1::from_vec(vec![8.0_f32]);
        let mvs = finder.find_mvs(&point);
        assert_eq!(
            mvs.len(),
            1,
            "Expected one constraint after redundancy removal, got {:?}",
            mvs
        );
        // The surviving constraint must be constraint 1 (tighter bound, larger violation).
        assert_eq!(
            mvs[0], 1,
            "Expected constraint index 1 (tighter) to survive, got {:?}",
            mvs
        );
    }

    // Test 5: three violated constraints — 2 parallel (same dim) + 1 orthogonal.
    //
    // Constraint 0: x0 <= 6  (violation at (8,5): 2.0, grad=[1,0])
    // Constraint 1: x0 <= 4  (violation at (8,5): 4.0, grad=[1,0])  — parallel to 0
    // Constraint 2: x1 <= 3  (violation at (8,5): 2.0, grad=[0,1])  — orthogonal
    //
    // Sorted ascending by violation: {0: 2.0, 2: 2.0, 1: 4.0} (ties broken arbitrarily).
    // The lesser of the two parallel constraints (index 0, viol=2.0) is dominated by
    // index 1 and removed. Constraint 2 is orthogonal to all — not dominated.
    // Result: 2 constraints (indices 1 and 2).
    #[test]
    fn test_mvs_mixed_three_constraints() {
        let constraints = vec![
            SimpleBound { dim: 0, upper: 6.0 }, // index 0
            SimpleBound { dim: 0, upper: 4.0 }, // index 1 (tighter, same direction)
            SimpleBound { dim: 1, upper: 3.0 }, // index 2 (orthogonal)
        ];
        let names = vec!["c0".to_string(), "c1".to_string(), "c2".to_string()];
        let finder = MinimalViolatingSubsetFinder::new(constraints, names).expect("matched names");

        let point = Array1::from_vec(vec![8.0_f32, 5.0]);
        let mut mvs = finder.find_mvs(&point);
        mvs.sort_unstable();
        assert_eq!(
            mvs.len(),
            2,
            "Expected 2 constraints in MVS (one parallel removed), got {:?}",
            mvs
        );
        assert!(
            mvs.contains(&1),
            "MVS must contain tighter dim-0 constraint (idx 1), got {:?}",
            mvs
        );
        assert!(
            mvs.contains(&2),
            "MVS must contain orthogonal dim-1 constraint (idx 2), got {:?}",
            mvs
        );
    }
}
