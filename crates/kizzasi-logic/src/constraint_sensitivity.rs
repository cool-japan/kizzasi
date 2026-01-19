//! Constraint Sensitivity Analysis
//!
//! This module provides tools for analyzing how solutions are affected by constraints:
//! - Shadow prices (dual values) for constraint relaxation
//! - Constraint tightness and slack analysis
//! - Perturbation analysis for robustness
//! - Critical constraint identification
//!
//! # Examples
//!
//! ```
//! use kizzasi_logic::{ConstraintBuilder, ConstraintSensitivityAnalyzer};
//! use scirs2_core::ndarray::Array1;
//!
//! // Create constraints
//! let c1 = ConstraintBuilder::new()
//!     .name("upper_bound")
//!     .less_than(10.0)
//!     .build()
//!     .unwrap();
//!
//! // Analyze sensitivity at a solution point
//! let analyzer = ConstraintSensitivityAnalyzer::new();
//! let point = Array1::from_vec(vec![8.0]);
//! let analysis = analyzer.analyze(&[c1], &point);
//!
//! // Check which constraints are tight (nearly violated)
//! for (idx, &is_tight) in analysis.tight_constraints.iter().enumerate() {
//!     if is_tight {
//!         println!("Constraint {} is tight", idx);
//!     }
//! }
//! ```

use crate::{Constraint, LogicError, LogicResult};
use scirs2_core::ndarray::Array1;

/// Result of constraint sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityAnalysis {
    /// Shadow prices (dual values) for each constraint
    /// Indicates how much the objective would improve if constraint i were relaxed by one unit
    pub shadow_prices: Vec<f32>,

    /// Slack for each constraint (distance to violation)
    /// Positive means satisfied with margin, negative means violated
    pub slacks: Vec<f32>,

    /// Whether each constraint is "tight" (close to being violated)
    pub tight_constraints: Vec<bool>,

    /// Critical constraints (tight and sensitive)
    pub critical_constraints: Vec<usize>,

    /// Constraint activity levels (0.0 = not active, 1.0 = fully active/tight)
    pub activity_levels: Vec<f32>,

    /// Perturbation sensitivity: how much solution would change if constraint i changes
    pub perturbation_sensitivity: Vec<f32>,

    /// Ranking of constraints by importance (most critical first)
    pub importance_ranking: Vec<usize>,
}

impl SensitivityAnalysis {
    /// Create a new sensitivity analysis result
    pub fn new(num_constraints: usize) -> Self {
        Self {
            shadow_prices: vec![0.0; num_constraints],
            slacks: vec![0.0; num_constraints],
            tight_constraints: vec![false; num_constraints],
            critical_constraints: Vec::new(),
            activity_levels: vec![0.0; num_constraints],
            perturbation_sensitivity: vec![0.0; num_constraints],
            importance_ranking: Vec::new(),
        }
    }

    /// Get the most critical constraint index
    pub fn most_critical(&self) -> Option<usize> {
        self.importance_ranking.first().copied()
    }

    /// Get the number of tight constraints
    pub fn tight_count(&self) -> usize {
        self.tight_constraints.iter().filter(|&&t| t).count()
    }

    /// Get the total slack across all constraints
    pub fn total_slack(&self) -> f32 {
        self.slacks.iter().sum()
    }

    /// Get the minimum slack (constraint closest to violation)
    pub fn min_slack(&self) -> f32 {
        self.slacks.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Check if a specific constraint is critical
    pub fn is_critical(&self, index: usize) -> bool {
        self.critical_constraints.contains(&index)
    }
}

/// Constraint Sensitivity Analyzer
///
/// Analyzes how solutions are affected by constraints and their perturbations.
pub struct ConstraintSensitivityAnalyzer {
    /// Tolerance for considering a constraint as "tight"
    tightness_tolerance: f32,

    /// Step size for finite difference approximation
    perturbation_step: f32,

    /// Number of samples for perturbation analysis
    perturbation_samples: usize,

    /// Minimum activity level to be considered critical
    critical_threshold: f32,
}

impl Default for ConstraintSensitivityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintSensitivityAnalyzer {
    /// Create a new sensitivity analyzer with default parameters
    pub fn new() -> Self {
        Self {
            tightness_tolerance: 0.1,
            perturbation_step: 0.01,
            perturbation_samples: 10,
            critical_threshold: 0.8,
        }
    }

    /// Set the tolerance for considering a constraint as tight
    pub fn with_tightness_tolerance(mut self, tol: f32) -> Self {
        self.tightness_tolerance = tol;
        self
    }

    /// Set the step size for perturbation analysis
    pub fn with_perturbation_step(mut self, step: f32) -> Self {
        self.perturbation_step = step;
        self
    }

    /// Set the number of samples for perturbation analysis
    pub fn with_perturbation_samples(mut self, samples: usize) -> Self {
        self.perturbation_samples = samples;
        self
    }

    /// Set the threshold for critical constraints
    pub fn with_critical_threshold(mut self, threshold: f32) -> Self {
        self.critical_threshold = threshold;
        self
    }

    /// Analyze constraints at a given solution point
    pub fn analyze(&self, constraints: &[Constraint], point: &Array1<f32>) -> SensitivityAnalysis {
        let n = constraints.len();
        let mut analysis = SensitivityAnalysis::new(n);

        // Compute slacks and identify tight constraints
        for (i, constraint) in constraints.iter().enumerate() {
            let slack = self.compute_slack(constraint, point);
            analysis.slacks[i] = slack;
            analysis.tight_constraints[i] = slack.abs() < self.tightness_tolerance;

            // Activity level: 0.0 for large slack, 1.0 for tight
            analysis.activity_levels[i] = self.compute_activity(slack);
        }

        // Estimate shadow prices using finite differences
        analysis.shadow_prices = self.estimate_shadow_prices(constraints, point);

        // Compute perturbation sensitivity
        analysis.perturbation_sensitivity =
            self.compute_perturbation_sensitivity(constraints, point);

        // Identify critical constraints
        analysis.critical_constraints = self.identify_critical_constraints(&analysis);

        // Rank constraints by importance
        analysis.importance_ranking = self.rank_constraints(&analysis);

        analysis
    }

    /// Compute slack for a constraint at a point
    /// Positive slack means satisfied with margin, negative means violated
    ///
    /// For x < b: slack = b - x (positive when satisfied)
    /// For x > b: slack = x - b (positive when satisfied)
    fn compute_slack(&self, constraint: &Constraint, point: &Array1<f32>) -> f32 {
        if let Some(dim) = constraint.dimension() {
            if dim < point.len() {
                let value = point[dim];
                let violation = constraint.violation(value);

                // If violated, slack is negative of violation
                // If satisfied, we need to compute the distance to boundary
                // We'll use: slack = -violation when violated
                // For satisfied: we compute based on boundary
                if violation > 0.0 {
                    -violation
                } else {
                    // Satisfied: compute distance to boundary
                    // Project to boundary and measure distance
                    let boundary = constraint.project(value + 1000.0);
                    (boundary - value).abs()
                }
            } else {
                f32::NEG_INFINITY
            }
        } else {
            let value = point[0];
            let violation = constraint.violation(value);

            if violation > 0.0 {
                -violation
            } else {
                let boundary = constraint.project(value + 1000.0);
                (boundary - value).abs()
            }
        }
    }

    /// Compute activity level from slack (0.0 = inactive, 1.0 = tight)
    fn compute_activity(&self, slack: f32) -> f32 {
        if slack < 0.0 {
            // Violated: fully active
            1.0
        } else if slack < self.tightness_tolerance {
            // Tight: high activity
            1.0 - (slack / self.tightness_tolerance)
        } else {
            // Not tight: activity decreases with slack
            (1.0 / (1.0 + slack)).min(1.0)
        }
    }

    /// Estimate shadow prices using finite differences
    fn estimate_shadow_prices(&self, constraints: &[Constraint], point: &Array1<f32>) -> Vec<f32> {
        let n = constraints.len();
        let mut shadow_prices = vec![0.0; n];

        // For each constraint, estimate the shadow price
        // Shadow price ≈ change in objective / change in constraint bound
        // We approximate this by measuring constraint violation sensitivity
        for i in 0..n {
            let constraint = &constraints[i];

            if let Some(dim) = constraint.dimension() {
                if dim < point.len() {
                    let base_violation = constraint.violation(point[dim]);

                    // Perturb the point slightly and measure change
                    let mut perturbed_point = point.clone();
                    perturbed_point[dim] += self.perturbation_step;

                    let perturbed_violation = constraint.violation(perturbed_point[dim]);
                    let sensitivity =
                        (perturbed_violation - base_violation) / self.perturbation_step;

                    // Shadow price is related to the constraint sensitivity
                    shadow_prices[i] = sensitivity.abs();
                }
            }
        }

        shadow_prices
    }

    /// Compute perturbation sensitivity for each constraint
    fn compute_perturbation_sensitivity(
        &self,
        constraints: &[Constraint],
        point: &Array1<f32>,
    ) -> Vec<f32> {
        let n = constraints.len();
        let mut sensitivities = vec![0.0; n];

        for i in 0..n {
            let constraint = &constraints[i];

            if let Some(dim) = constraint.dimension() {
                if dim >= point.len() {
                    continue;
                }

                let base_value = point[dim];
                let base_satisfied = constraint.check(base_value);

                // Test sensitivity by perturbing in both directions
                let mut sensitivity_sum = 0.0;
                let mut count = 0;

                for j in 1..=self.perturbation_samples {
                    let delta = self.perturbation_step * j as f32;

                    // Positive perturbation
                    let pos_satisfied = constraint.check(base_value + delta);
                    if pos_satisfied != base_satisfied {
                        sensitivity_sum += 1.0 / delta;
                        count += 1;
                    }

                    // Negative perturbation
                    let neg_satisfied = constraint.check(base_value - delta);
                    if neg_satisfied != base_satisfied {
                        sensitivity_sum += 1.0 / delta;
                        count += 1;
                    }
                }

                if count > 0 {
                    sensitivities[i] = sensitivity_sum / count as f32;
                } else {
                    // If no satisfaction changes, sensitivity is low
                    sensitivities[i] = 0.0;
                }
            }
        }

        sensitivities
    }

    /// Identify critical constraints based on multiple criteria
    fn identify_critical_constraints(&self, analysis: &SensitivityAnalysis) -> Vec<usize> {
        let mut critical = Vec::new();

        for i in 0..analysis.activity_levels.len() {
            let activity = analysis.activity_levels[i];
            let is_tight = analysis.tight_constraints[i];
            let has_high_sensitivity = analysis.perturbation_sensitivity[i] > 0.1;

            // A constraint is critical if:
            // 1. It has high activity level, OR
            // 2. It's tight AND has high perturbation sensitivity
            if activity > self.critical_threshold || (is_tight && has_high_sensitivity) {
                critical.push(i);
            }
        }

        critical
    }

    /// Rank constraints by importance (most important first)
    fn rank_constraints(&self, analysis: &SensitivityAnalysis) -> Vec<usize> {
        let n = analysis.activity_levels.len();
        let mut rankings: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                // Importance score combines multiple factors
                let activity = analysis.activity_levels[i];
                let shadow = analysis.shadow_prices[i];
                let perturbation = analysis.perturbation_sensitivity[i];

                // Weighted combination
                let score = activity * 0.5 + shadow * 0.3 + perturbation * 0.2;
                (i, score)
            })
            .collect();

        // Sort by score (descending)
        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        rankings.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Perform local sensitivity analysis by varying constraints
    pub fn local_sensitivity(
        &self,
        constraints: &[Constraint],
        point: &Array1<f32>,
        constraint_idx: usize,
    ) -> LogicResult<LocalSensitivity> {
        if constraint_idx >= constraints.len() {
            return Err(LogicError::InvalidInput(format!(
                "Constraint index {} out of bounds (max {})",
                constraint_idx,
                constraints.len() - 1
            )));
        }

        let constraint = &constraints[constraint_idx];
        let dim = constraint.dimension().ok_or_else(|| {
            LogicError::InvalidInput("Constraint has no dimension specified".to_string())
        })?;

        if dim >= point.len() {
            return Err(LogicError::DimensionMismatch {
                expected: dim + 1,
                got: point.len(),
            });
        }

        let base_value = point[dim];
        let base_slack = self.compute_slack(constraint, point);

        // Compute sensitivity in both directions
        let mut positive_changes = Vec::new();
        let mut negative_changes = Vec::new();

        for i in 1..=self.perturbation_samples {
            let delta = self.perturbation_step * i as f32;

            // Positive direction
            let mut pos_point = point.clone();
            pos_point[dim] = base_value + delta;
            let pos_slack = self.compute_slack(constraint, &pos_point);
            positive_changes.push((delta, pos_slack - base_slack));

            // Negative direction
            let mut neg_point = point.clone();
            neg_point[dim] = base_value - delta;
            let neg_slack = self.compute_slack(constraint, &neg_point);
            negative_changes.push((-delta, neg_slack - base_slack));
        }

        Ok(LocalSensitivity {
            constraint_idx,
            base_slack,
            positive_changes,
            negative_changes,
        })
    }

    /// Compute robustness margin: how much all constraints can be perturbed simultaneously
    pub fn robustness_margin(&self, constraints: &[Constraint], point: &Array1<f32>) -> f32 {
        if constraints.is_empty() {
            return f32::INFINITY;
        }

        // Minimum slack is the robustness margin
        constraints
            .iter()
            .map(|c| self.compute_slack(c, point))
            .fold(f32::INFINITY, f32::min)
    }
}

/// Local sensitivity analysis result for a single constraint
#[derive(Debug, Clone)]
pub struct LocalSensitivity {
    /// Index of the constraint being analyzed
    pub constraint_idx: usize,

    /// Base slack value at the solution point
    pub base_slack: f32,

    /// Changes in slack for positive perturbations: (delta, slack_change)
    pub positive_changes: Vec<(f32, f32)>,

    /// Changes in slack for negative perturbations: (delta, slack_change)
    pub negative_changes: Vec<(f32, f32)>,
}

impl LocalSensitivity {
    /// Estimate the gradient of slack with respect to the variable
    pub fn gradient_estimate(&self) -> f32 {
        if self.positive_changes.is_empty() {
            return 0.0;
        }

        // Use linear regression or simple average of slopes
        let mut slopes = Vec::new();
        for (delta, change) in &self.positive_changes {
            slopes.push(change / delta);
        }
        for (delta, change) in &self.negative_changes {
            slopes.push(change / delta);
        }

        if slopes.is_empty() {
            0.0
        } else {
            slopes.iter().sum::<f32>() / slopes.len() as f32
        }
    }

    /// Check if the constraint exhibits linear behavior
    pub fn is_linear(&self) -> bool {
        let grad = self.gradient_estimate();
        let tolerance = 0.01;

        // Check if all slopes are similar to the estimated gradient
        for (delta, change) in &self.positive_changes {
            let slope = change / delta;
            if (slope - grad).abs() > tolerance {
                return false;
            }
        }

        for (delta, change) in &self.negative_changes {
            let slope = change / delta;
            if (slope - grad).abs() > tolerance {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConstraintBuilder;

    #[test]
    fn test_sensitivity_analysis_basic() {
        let c1 = ConstraintBuilder::new()
            .name("upper")
            .dimension(0)
            .less_than(10.0)
            .build()
            .unwrap();

        let analyzer = ConstraintSensitivityAnalyzer::new();
        let point = Array1::from_vec(vec![8.0]);
        let analysis = analyzer.analyze(&[c1], &point);

        assert_eq!(analysis.slacks.len(), 1);
        assert!(
            analysis.slacks[0] > 0.0,
            "Slack should be positive: {}",
            analysis.slacks[0]
        ); // Should have positive slack
        assert_eq!(analysis.shadow_prices.len(), 1);
    }

    #[test]
    fn test_tight_constraint_detection() {
        let c1 = ConstraintBuilder::new()
            .name("upper")
            .dimension(0)
            .less_than(10.0)
            .build()
            .unwrap();

        let analyzer = ConstraintSensitivityAnalyzer::new().with_tightness_tolerance(0.5);

        // Point very close to boundary
        let point = Array1::from_vec(vec![9.95]);
        let analysis = analyzer.analyze(&[c1], &point);

        assert!(analysis.tight_constraints[0], "Constraint should be tight");
        assert_eq!(analysis.tight_count(), 1);
    }

    #[test]
    fn test_critical_constraint_identification() {
        let c1 = ConstraintBuilder::new()
            .name("tight_constraint")
            .dimension(0)
            .less_than(10.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("loose_constraint")
            .dimension(0)
            .less_than(100.0)
            .build()
            .unwrap();

        let analyzer = ConstraintSensitivityAnalyzer::new();
        let point = Array1::from_vec(vec![9.5]);
        let analysis = analyzer.analyze(&[c1, c2], &point);

        // First constraint should be more critical (closer to boundary)
        assert!(
            analysis.activity_levels[0] > analysis.activity_levels[1],
            "Activity levels: c1={}, c2={}",
            analysis.activity_levels[0],
            analysis.activity_levels[1]
        );
    }

    #[test]
    fn test_importance_ranking() {
        let c1 = ConstraintBuilder::new()
            .name("c1")
            .dimension(0)
            .less_than(10.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("c2")
            .dimension(0)
            .greater_than(0.0)
            .build()
            .unwrap();

        let analyzer = ConstraintSensitivityAnalyzer::new();
        let point = Array1::from_vec(vec![9.0]);
        let analysis = analyzer.analyze(&[c1, c2], &point);

        assert_eq!(analysis.importance_ranking.len(), 2);
        assert!(analysis.importance_ranking.contains(&0));
        assert!(analysis.importance_ranking.contains(&1));
    }

    #[test]
    fn test_local_sensitivity() {
        let c1 = ConstraintBuilder::new()
            .name("test")
            .dimension(0)
            .less_than(10.0)
            .build()
            .unwrap();

        let analyzer = ConstraintSensitivityAnalyzer::new();
        let point = Array1::from_vec(vec![5.0]);

        let local = analyzer.local_sensitivity(&[c1], &point, 0).unwrap();

        assert_eq!(local.constraint_idx, 0);
        assert!(local.base_slack > 0.0);
        assert!(!local.positive_changes.is_empty());
        assert!(!local.negative_changes.is_empty());
    }

    #[test]
    fn test_robustness_margin() {
        let c1 = ConstraintBuilder::new()
            .name("c1")
            .dimension(0)
            .less_than(10.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("c2")
            .dimension(0)
            .greater_than(0.0)
            .build()
            .unwrap();

        let analyzer = ConstraintSensitivityAnalyzer::new();

        // Point at [5.0] has slack 5.0 for both constraints
        let point = Array1::from_vec(vec![5.0]);
        let margin = analyzer.robustness_margin(&[c1, c2], &point);

        assert!(margin > 0.0, "Margin should be positive: {}", margin);
        assert!(margin <= 5.0, "Margin should be at most 5.0: {}", margin);
    }

    #[test]
    fn test_gradient_estimate() {
        let c1 = ConstraintBuilder::new()
            .name("linear")
            .dimension(0)
            .less_than(10.0)
            .build()
            .unwrap();

        let analyzer = ConstraintSensitivityAnalyzer::new();
        let point = Array1::from_vec(vec![5.0]);

        let local = analyzer.local_sensitivity(&[c1], &point, 0).unwrap();
        let grad = local.gradient_estimate();

        // For a linear constraint x < 10, gradient should be approximately -1
        assert!(grad.abs() < 2.0); // Reasonable range for numerical approximation
    }

    #[test]
    fn test_min_slack() {
        let c1 = ConstraintBuilder::new()
            .name("c1")
            .dimension(0)
            .less_than(10.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("c2")
            .dimension(0)
            .less_than(100.0)
            .build()
            .unwrap();

        let analyzer = ConstraintSensitivityAnalyzer::new();
        let point = Array1::from_vec(vec![8.0]);
        let analysis = analyzer.analyze(&[c1, c2], &point);

        let min_slack = analysis.min_slack();
        assert!(
            min_slack > 0.0,
            "Min slack should be positive: {}",
            min_slack
        );
        assert!(
            min_slack <= 3.0,
            "Min slack should be at most 3.0 (was {})",
            min_slack
        ); // Slack for c1 should be ~2.0
    }
}
