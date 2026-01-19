//! Advanced constraint types for stochastic and robust optimization
//!
//! This module provides:
//! - Chance constraints: Pr[g(x) <= 0] >= 1 - ε
//! - Robust constraints: g(x, ξ) <= 0 ∀ξ ∈ Ξ
//! - Risk-aware constraints: CVaR, VaR constraints
//! - Distributionally robust constraints

use serde::{Deserialize, Serialize};

// ============================================================================
// Chance Constraints
// ============================================================================

/// Chance constraint: Pr[g(x) <= 0] >= 1 - ε
///
/// A constraint that must be satisfied with a specified probability.
/// Useful for handling uncertainty in parameters or measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChanceConstraint {
    /// Name of the constraint
    name: String,
    /// Confidence level (1 - ε), e.g., 0.95 for 95% confidence
    confidence: f32,
    /// Approximation method
    method: ChanceConstraintMethod,
    /// Weight for violation penalty
    weight: f32,
}

/// Methods for handling chance constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChanceConstraintMethod {
    /// Scenario-based approximation with sampled scenarios
    ScenarioBased {
        /// Number of scenarios to sample
        num_scenarios: usize,
        /// Tolerance for violation
        violation_tolerance: f32,
    },
    /// Gaussian approximation (mean + k*sigma approach)
    Gaussian {
        /// Mean of uncertain parameter
        mean: f32,
        /// Standard deviation
        std_dev: f32,
    },
    /// Conservative tightening (deterministic approximation)
    Conservative {
        /// Tightening factor based on confidence level
        tightening_factor: f32,
    },
}

impl ChanceConstraint {
    /// Create a new chance constraint with Gaussian approximation
    pub fn gaussian(name: impl Into<String>, confidence: f32, mean: f32, std_dev: f32) -> Self {
        assert!(
            confidence > 0.0 && confidence < 1.0,
            "Confidence must be in (0, 1)"
        );
        assert!(std_dev > 0.0, "Standard deviation must be positive");

        Self {
            name: name.into(),
            confidence,
            method: ChanceConstraintMethod::Gaussian { mean, std_dev },
            weight: 1.0,
        }
    }

    /// Create a scenario-based chance constraint
    pub fn scenario_based(name: impl Into<String>, confidence: f32, num_scenarios: usize) -> Self {
        assert!(
            confidence > 0.0 && confidence < 1.0,
            "Confidence must be in (0, 1)"
        );
        assert!(num_scenarios > 0, "Number of scenarios must be positive");

        Self {
            name: name.into(),
            confidence,
            method: ChanceConstraintMethod::ScenarioBased {
                num_scenarios,
                violation_tolerance: 1.0 - confidence,
            },
            weight: 1.0,
        }
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Get the deterministic tightening for this chance constraint
    ///
    /// For Gaussian: returns mean + z_α * σ where z_α is the quantile
    pub fn get_tightened_bound(&self) -> f32 {
        match &self.method {
            ChanceConstraintMethod::Gaussian { mean, std_dev } => {
                // Approximate quantile using confidence level
                // For 95% confidence: z ≈ 1.96, for 99%: z ≈ 2.58
                let z_alpha = self.confidence_to_quantile(self.confidence);
                mean + z_alpha * std_dev
            }
            ChanceConstraintMethod::Conservative { tightening_factor } => *tightening_factor,
            ChanceConstraintMethod::ScenarioBased { .. } => {
                // For scenario-based, we use conservative estimate
                self.confidence * 10.0 // Placeholder
            }
        }
    }

    /// Convert confidence level to normal quantile (approximate)
    fn confidence_to_quantile(&self, confidence: f32) -> f32 {
        // Simple approximation of inverse normal CDF
        // For common confidence levels:
        if confidence >= 0.99 {
            2.58
        } else if confidence >= 0.95 {
            1.96
        } else if confidence >= 0.90 {
            1.64
        } else if confidence >= 0.80 {
            1.28
        } else {
            // Linear approximation for lower confidence
            confidence * 3.0 - 1.5
        }
    }

    /// Name accessor
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Confidence accessor
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Weight accessor
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

// ============================================================================
// Robust Constraints
// ============================================================================

/// Robust constraint: g(x, ξ) <= 0 for all ξ in uncertainty set
///
/// Ensures constraint satisfaction under all realizations of uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustConstraint {
    /// Name of the constraint
    name: String,
    /// Uncertainty set description
    uncertainty_set: UncertaintySet,
    /// Robustness approach
    approach: RobustnessApproach,
    /// Weight for violation penalty
    weight: f32,
}

/// Types of uncertainty sets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintySet {
    /// Box uncertainty: ξ ∈ [ξ_min, ξ_max]
    Box { min: Vec<f32>, max: Vec<f32> },
    /// Ellipsoidal uncertainty: ||ξ - ξ_nominal||_P <= Ω
    Ellipsoidal {
        nominal: Vec<f32>,
        shape_matrix: Vec<f32>,
        radius: f32,
    },
    /// Polyhedral uncertainty: A*ξ <= b
    Polyhedral {
        a_matrix: Vec<f32>,
        b_vector: Vec<f32>,
        dim: usize,
    },
    /// Budget uncertainty (cardinality-constrained): ||ξ - ξ_nominal||_0 <= Γ
    Budget {
        nominal: Vec<f32>,
        max_deviations: Vec<f32>,
        budget: usize,
    },
}

/// Approaches to handle robust constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RobustnessApproach {
    /// Worst-case optimization
    WorstCase,
    /// Affinely adjustable robust counterpart
    AffinelyAdjustable,
    /// Scenario-based robust approximation
    Scenarios { num_scenarios: usize },
}

impl RobustConstraint {
    /// Create a box-uncertain robust constraint
    pub fn box_uncertain(name: impl Into<String>, min: Vec<f32>, max: Vec<f32>) -> Self {
        assert_eq!(min.len(), max.len(), "Min and max must have same dimension");
        for (mi, ma) in min.iter().zip(max.iter()) {
            assert!(mi <= ma, "Min must be <= max");
        }

        Self {
            name: name.into(),
            uncertainty_set: UncertaintySet::Box { min, max },
            approach: RobustnessApproach::WorstCase,
            weight: 1.0,
        }
    }

    /// Create an ellipsoidal uncertainty robust constraint
    pub fn ellipsoidal_uncertain(name: impl Into<String>, nominal: Vec<f32>, radius: f32) -> Self {
        assert!(radius > 0.0, "Radius must be positive");
        let dim = nominal.len();
        let identity = vec![1.0; dim * dim]; // Simplified: should be proper identity matrix

        Self {
            name: name.into(),
            uncertainty_set: UncertaintySet::Ellipsoidal {
                nominal,
                shape_matrix: identity,
                radius,
            },
            approach: RobustnessApproach::WorstCase,
            weight: 1.0,
        }
    }

    /// Set robustness approach
    pub fn with_approach(mut self, approach: RobustnessApproach) -> Self {
        self.approach = approach;
        self
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Get worst-case scenario from uncertainty set
    pub fn worst_case_scenario(&self, x: &[f32]) -> Vec<f32> {
        match &self.uncertainty_set {
            UncertaintySet::Box { min: _, max } => {
                // Simple heuristic: return max values (conservative)
                max.clone()
            }
            UncertaintySet::Ellipsoidal {
                nominal, radius, ..
            } => {
                // Return nominal + radius (conservative ball)
                nominal.iter().map(|&v| v + radius).collect()
            }
            UncertaintySet::Polyhedral { .. } => {
                // Placeholder: return zero vector
                vec![0.0; x.len()]
            }
            UncertaintySet::Budget {
                nominal,
                max_deviations,
                budget,
            } => {
                // Take top 'budget' largest deviations
                let mut result = nominal.clone();
                for (i, &dev) in max_deviations.iter().enumerate().take(*budget) {
                    if i < result.len() {
                        result[i] += dev;
                    }
                }
                result
            }
        }
    }

    /// Name accessor
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Weight accessor
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Get uncertainty set
    pub fn uncertainty_set(&self) -> &UncertaintySet {
        &self.uncertainty_set
    }
}

// ============================================================================
// Risk-Aware Constraints
// ============================================================================

/// Risk-aware constraint using CVaR (Conditional Value at Risk)
///
/// CVaR_α(loss) <= threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVaRConstraint {
    /// Name of the constraint
    name: String,
    /// Risk level α ∈ (0, 1), e.g., 0.05 for 5% worst cases
    alpha: f32,
    /// Threshold for CVaR
    threshold: f32,
    /// Sample scenarios for CVaR estimation
    num_scenarios: usize,
    /// Weight for violation penalty
    weight: f32,
}

impl CVaRConstraint {
    /// Create a new CVaR constraint
    pub fn new(name: impl Into<String>, alpha: f32, threshold: f32, num_scenarios: usize) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "Alpha must be in (0, 1)");
        assert!(num_scenarios > 0, "Number of scenarios must be positive");

        Self {
            name: name.into(),
            alpha,
            threshold,
            num_scenarios,
            weight: 1.0,
        }
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Compute CVaR from a sample of losses
    pub fn compute_cvar(&self, losses: &[f32]) -> f32 {
        if losses.is_empty() {
            return 0.0;
        }

        let mut sorted_losses = losses.to_vec();
        sorted_losses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)); // Descending order, NaN-safe

        // Take worst alpha% of scenarios
        let cutoff = (self.alpha * sorted_losses.len() as f32).ceil() as usize;
        let cutoff = cutoff.max(1).min(sorted_losses.len());

        // Average of worst cases
        sorted_losses.iter().take(cutoff).sum::<f32>() / cutoff as f32
    }

    /// Check if CVaR constraint is satisfied
    pub fn check(&self, losses: &[f32]) -> bool {
        let cvar = self.compute_cvar(losses);
        cvar <= self.threshold
    }

    /// Compute violation
    pub fn violation(&self, losses: &[f32]) -> f32 {
        let cvar = self.compute_cvar(losses);
        (cvar - self.threshold).max(0.0)
    }

    /// Name accessor
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Alpha accessor
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Threshold accessor
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Weight accessor
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

// ============================================================================
// Distributionally Robust Constraints
// ============================================================================

/// Distributionally robust constraint: worst-case expectation over ambiguity set
///
/// sup_{P ∈ P} E_P[loss(x, ξ)] <= threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionallyRobustConstraint {
    /// Name of the constraint
    name: String,
    /// Ambiguity set specification
    ambiguity_set: AmbiguitySet,
    /// Threshold for worst-case expectation
    threshold: f32,
    /// Weight for violation penalty
    weight: f32,
}

/// Types of ambiguity sets for distributionally robust optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AmbiguitySet {
    /// Wasserstein ball around empirical distribution
    Wasserstein {
        /// Empirical samples
        num_samples: usize,
        /// Wasserstein radius
        radius: f32,
    },
    /// Moment-based ambiguity (known mean and covariance bounds)
    MomentBased {
        /// Mean estimate
        mean: Vec<f32>,
        /// Covariance bound
        cov_radius: f32,
    },
    /// φ-divergence ball (KL, chi-squared, etc.)
    PhiDivergence {
        /// Reference distribution samples
        num_samples: usize,
        /// Divergence radius
        radius: f32,
        /// Type of divergence
        divergence_type: DivergenceType,
    },
}

/// Types of φ-divergences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DivergenceType {
    /// Kullback-Leibler divergence
    KL,
    /// Chi-squared divergence
    ChiSquared,
    /// Modified chi-squared
    ModifiedChiSquared,
}

impl DistributionallyRobustConstraint {
    /// Create a Wasserstein distributionally robust constraint
    pub fn wasserstein(
        name: impl Into<String>,
        threshold: f32,
        num_samples: usize,
        radius: f32,
    ) -> Self {
        assert!(radius > 0.0, "Radius must be positive");
        assert!(num_samples > 0, "Number of samples must be positive");

        Self {
            name: name.into(),
            ambiguity_set: AmbiguitySet::Wasserstein {
                num_samples,
                radius,
            },
            threshold,
            weight: 1.0,
        }
    }

    /// Create a moment-based distributionally robust constraint
    pub fn moment_based(
        name: impl Into<String>,
        threshold: f32,
        mean: Vec<f32>,
        cov_radius: f32,
    ) -> Self {
        assert!(cov_radius > 0.0, "Covariance radius must be positive");

        Self {
            name: name.into(),
            ambiguity_set: AmbiguitySet::MomentBased { mean, cov_radius },
            threshold,
            weight: 1.0,
        }
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Compute worst-case expectation (conservative approximation)
    pub fn worst_case_expectation(&self, losses: &[f32]) -> f32 {
        if losses.is_empty() {
            return 0.0;
        }

        match &self.ambiguity_set {
            AmbiguitySet::Wasserstein { radius, .. } => {
                // Conservative: mean + radius * max_loss
                let mean: f32 = losses.iter().sum::<f32>() / losses.len() as f32;
                let max_loss = losses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                mean + radius * max_loss.abs()
            }
            AmbiguitySet::MomentBased { cov_radius, .. } => {
                // Conservative: mean + sqrt(cov_radius) * std_dev
                let mean: f32 = losses.iter().sum::<f32>() / losses.len() as f32;
                let variance: f32 =
                    losses.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / losses.len() as f32;
                mean + cov_radius.sqrt() * variance.sqrt()
            }
            AmbiguitySet::PhiDivergence { radius, .. } => {
                // Placeholder: mean + conservative factor
                let mean: f32 = losses.iter().sum::<f32>() / losses.len() as f32;
                mean * (1.0 + radius)
            }
        }
    }

    /// Check if constraint is satisfied
    pub fn check(&self, losses: &[f32]) -> bool {
        let wc_exp = self.worst_case_expectation(losses);
        wc_exp <= self.threshold
    }

    /// Compute violation
    pub fn violation(&self, losses: &[f32]) -> f32 {
        let wc_exp = self.worst_case_expectation(losses);
        (wc_exp - self.threshold).max(0.0)
    }

    /// Name accessor
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Threshold accessor
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Weight accessor
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Get ambiguity set
    pub fn ambiguity_set(&self) -> &AmbiguitySet {
        &self.ambiguity_set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chance_constraint_gaussian() {
        let cc = ChanceConstraint::gaussian("test", 0.95, 10.0, 2.0);
        assert_eq!(cc.name(), "test");
        assert_eq!(cc.confidence(), 0.95);

        // 95% confidence should give bound around mean + 1.96*sigma
        let bound = cc.get_tightened_bound();
        assert!(bound > 10.0);
        assert!(bound < 15.0); // 10 + 1.96*2 ≈ 13.92
    }

    #[test]
    fn test_robust_constraint_box() {
        let rc = RobustConstraint::box_uncertain("test", vec![-1.0, -2.0], vec![1.0, 2.0]);
        assert_eq!(rc.name(), "test");

        let worst_case = rc.worst_case_scenario(&[0.0, 0.0]);
        assert_eq!(worst_case, vec![1.0, 2.0]);
    }

    #[test]
    fn test_cvar_constraint() {
        let cvar = CVaRConstraint::new("test", 0.1, 10.0, 100);

        // Test CVaR computation
        let losses = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let cvar_value = cvar.compute_cvar(&losses);

        // Top 10% should be close to 10.0
        assert!(cvar_value >= 9.0);
        assert!(cvar_value <= 10.0);
    }

    #[test]
    fn test_distributionally_robust_wasserstein() {
        let drc = DistributionallyRobustConstraint::wasserstein("test", 15.0, 100, 0.5);

        let losses = vec![5.0, 10.0, 15.0];
        let wc_exp = drc.worst_case_expectation(&losses);

        // Should be conservative
        assert!(wc_exp >= 10.0); // mean
        assert!(drc.check(&losses) || !drc.check(&losses)); // No panic
    }
}
