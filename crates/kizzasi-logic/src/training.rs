//! Training integration for constraint-aware neural networks
//!
//! This module provides differentiable projections, constraint-aware loss functions,
//! and methods for integrating constraints into neural network training.

use crate::constraint::{PenaltyFunction, ViolationComputable};
use crate::error::{LogicError, LogicResult};

// ============================================================================
// Differentiable Projection
// ============================================================================

/// Differentiable projection using soft projection with temperature parameter
///
/// Instead of hard projection, uses a smooth approximation that is differentiable.
/// As temperature → 0, converges to hard projection.
pub struct DifferentiableProjection {
    temperature: f32,
    max_iterations: usize,
}

impl DifferentiableProjection {
    /// Create a new differentiable projection
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidInput`] when `temperature` is not strictly
    /// positive — the soft projection divides by it.
    pub fn new(temperature: f32) -> LogicResult<Self> {
        if temperature.is_nan() || temperature <= 0.0 {
            return Err(LogicError::InvalidInput(format!(
                "differentiable projection temperature must be positive, got {temperature}"
            )));
        }
        Ok(Self {
            temperature,
            max_iterations: 10,
        })
    }

    /// Set maximum iterations for iterative soft projection
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Soft projection onto box constraints [lower, upper]
    ///
    /// Uses smooth clamp: softclamp(x, a, b) ≈ clamp(x, a, b)
    pub fn soft_project_box(&self, x: f32, lower: f32, upper: f32) -> f32 {
        let tau = self.temperature;

        // Soft lower bound: x + tau * log(1 + exp((lower - x) / tau))
        let soft_lower = x + tau * ((lower - x) / tau).exp().ln_1p();

        // Soft upper bound: x - tau * log(1 + exp((x - upper) / tau))
        let soft_upper = x - tau * ((x - upper) / tau).exp().ln_1p();

        soft_lower.min(soft_upper)
    }

    /// Soft projection with barrier function
    ///
    /// For constraint g(x) <= 0, adds barrier: -tau * log(-g(x))
    pub fn barrier_project(&self, value: f32, gradient: f32, constraint_val: f32) -> f32 {
        if constraint_val >= 0.0 {
            // Outside constraint, push back
            value - self.temperature * gradient * constraint_val
        } else {
            // Inside constraint, barrier effect
            let barrier_grad = -self.temperature / constraint_val;
            value - self.temperature * gradient * barrier_grad
        }
    }

    /// Get the temperature parameter
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
}

// ============================================================================
// Constraint-Aware Loss Functions
// ============================================================================

/// Constraint-aware loss function for training
///
/// Combines task loss with constraint violation penalties
pub struct ConstraintAwareLoss<C> {
    constraints: Vec<C>,
    penalty_function: PenaltyFunction,
    constraint_weight: f32,
}

impl<C: ViolationComputable> ConstraintAwareLoss<C> {
    /// Create a new constraint-aware loss
    pub fn new(constraints: Vec<C>, penalty: PenaltyFunction, weight: f32) -> Self {
        Self {
            constraints,
            penalty_function: penalty,
            constraint_weight: weight,
        }
    }

    /// Compute total loss: task_loss + constraint_weight * penalty(violations)
    pub fn compute_loss(&self, prediction: &[f32], task_loss: f32) -> f32 {
        let violation: f32 = self
            .constraints
            .iter()
            .map(|c| c.violation(prediction))
            .sum();

        task_loss
            + self
                .penalty_function
                .compute(violation, self.constraint_weight)
    }

    /// Compute constraint violation penalty only
    pub fn constraint_penalty(&self, prediction: &[f32]) -> f32 {
        let violation: f32 = self
            .constraints
            .iter()
            .map(|c| c.violation(prediction))
            .sum();

        self.penalty_function
            .compute(violation, self.constraint_weight)
    }

    /// Check if all constraints are satisfied
    pub fn all_satisfied(&self, prediction: &[f32]) -> bool {
        self.constraints.iter().all(|c| c.check(prediction))
    }

    /// Get constraint weight
    pub fn constraint_weight(&self) -> f32 {
        self.constraint_weight
    }

    /// Set constraint weight (for dynamic weighting during training)
    pub fn set_constraint_weight(&mut self, weight: f32) {
        self.constraint_weight = weight;
    }
}

// ============================================================================
// Lagrangian Relaxation for Constrained Training
// ============================================================================

/// Lagrangian relaxation for constrained optimization during training
///
/// Converts constrained problem into unconstrained using Lagrange multipliers:
/// L(x, λ) = f(x) + Σ λ_i * g_i(x)
pub struct LagrangianRelaxation {
    multipliers: Vec<f32>,
    learning_rate_multipliers: f32,
    max_multiplier: f32,
}

impl LagrangianRelaxation {
    /// Create a new Lagrangian relaxation with given number of constraints
    pub fn new(num_constraints: usize) -> Self {
        Self {
            multipliers: vec![0.0; num_constraints],
            learning_rate_multipliers: 0.01,
            max_multiplier: 100.0,
        }
    }

    /// Set learning rate for multiplier updates
    pub fn with_multiplier_lr(mut self, lr: f32) -> Self {
        self.learning_rate_multipliers = lr;
        self
    }

    /// Set maximum multiplier value (for numerical stability)
    pub fn with_max_multiplier(mut self, max_val: f32) -> Self {
        self.max_multiplier = max_val;
        self
    }

    /// Compute Lagrangian: task_loss + Σ λ_i * g_i(x)
    pub fn compute_lagrangian<C: ViolationComputable>(
        &self,
        prediction: &[f32],
        constraints: &[C],
        task_loss: f32,
    ) -> f32 {
        let constraint_terms: f32 = self
            .multipliers
            .iter()
            .zip(constraints.iter())
            .map(|(&lambda, c)| lambda * c.violation(prediction))
            .sum();

        task_loss + constraint_terms
    }

    /// Update Lagrange multipliers using gradient ascent
    ///
    /// λ_i ← max(0, λ_i + lr * g_i(x))
    pub fn update_multipliers<C: ViolationComputable>(
        &mut self,
        prediction: &[f32],
        constraints: &[C],
    ) {
        for (lambda, constraint) in self.multipliers.iter_mut().zip(constraints.iter()) {
            let violation = constraint.violation(prediction);
            *lambda = (*lambda + self.learning_rate_multipliers * violation)
                .clamp(0.0, self.max_multiplier);
        }
    }

    /// Get current multipliers
    pub fn multipliers(&self) -> &[f32] {
        &self.multipliers
    }

    /// Reset multipliers to zero
    pub fn reset_multipliers(&mut self) {
        self.multipliers.fill(0.0);
    }

    /// Get average multiplier (indicates constraint tightness)
    pub fn average_multiplier(&self) -> f32 {
        if self.multipliers.is_empty() {
            0.0
        } else {
            self.multipliers.iter().sum::<f32>() / self.multipliers.len() as f32
        }
    }
}

// ============================================================================
// Penalty Method for Constrained Training
// ============================================================================

/// Penalty method for constrained optimization
///
/// Solves constrained problem by adding penalty terms and increasing penalty weight
pub struct PenaltyMethod {
    penalty_weight: f32,
    penalty_increase_factor: f32,
    penalty_function: PenaltyFunction,
}

impl PenaltyMethod {
    /// Create a new penalty method
    pub fn new(penalty_function: PenaltyFunction) -> Self {
        Self {
            penalty_weight: 1.0,
            penalty_increase_factor: 10.0,
            penalty_function,
        }
    }

    /// Set initial penalty weight
    pub fn with_penalty_weight(mut self, weight: f32) -> Self {
        self.penalty_weight = weight;
        self
    }

    /// Set penalty increase factor
    pub fn with_increase_factor(mut self, factor: f32) -> Self {
        self.penalty_increase_factor = factor;
        self
    }

    /// Compute penalized loss: task_loss + penalty_weight * penalty(violations)
    pub fn compute_loss<C: ViolationComputable>(
        &self,
        prediction: &[f32],
        constraints: &[C],
        task_loss: f32,
    ) -> f32 {
        let total_violation: f32 = constraints.iter().map(|c| c.violation(prediction)).sum();

        task_loss
            + self
                .penalty_function
                .compute(total_violation, self.penalty_weight)
    }

    /// Increase penalty weight (called after each epoch/iteration)
    pub fn increase_penalty(&mut self) {
        self.penalty_weight *= self.penalty_increase_factor;
    }

    /// Get current penalty weight
    pub fn penalty_weight(&self) -> f32 {
        self.penalty_weight
    }

    /// Reset penalty weight
    pub fn reset_penalty(&mut self, initial_weight: f32) {
        self.penalty_weight = initial_weight;
    }
}

// ============================================================================
// Barrier Method for Constrained Training
// ============================================================================

/// Barrier method for inequality constrained optimization
///
/// Uses logarithmic barrier: -μ * Σ log(-g_i(x)) for g_i(x) < 0
pub struct BarrierMethod {
    barrier_weight: f32,
    barrier_decrease_factor: f32,
}

impl BarrierMethod {
    /// Create a new barrier method
    pub fn new() -> Self {
        Self {
            barrier_weight: 1.0,
            barrier_decrease_factor: 0.1,
        }
    }

    /// Set initial barrier weight
    pub fn with_barrier_weight(mut self, weight: f32) -> Self {
        self.barrier_weight = weight;
        self
    }

    /// Set barrier decrease factor (μ ← μ * factor after each iteration)
    pub fn with_decrease_factor(mut self, factor: f32) -> Self {
        self.barrier_decrease_factor = factor;
        self
    }

    /// Compute barrier loss: task_loss - barrier_weight * Σ log(-g_i(x))
    ///
    /// Note: Only works if all constraints are satisfied (g_i(x) < 0)
    pub fn compute_loss<C: ViolationComputable>(
        &self,
        prediction: &[f32],
        constraints: &[C],
        task_loss: f32,
    ) -> LogicResult<f32> {
        let mut barrier_term = 0.0;

        for constraint in constraints {
            // For barrier to work, we need g(x) < 0
            // We compute violation, so constraint value = -violation
            let violation = constraint.violation(prediction);

            if violation > 0.0 {
                // Constraint violated, barrier not applicable
                return Ok(f32::MAX); // Infinite loss
            }

            // g(x) = -violation (negative when satisfied)
            // But we want strict interior: g(x) < 0
            // Use small epsilon for numerical stability
            let epsilon = 1e-6;
            let g_x = -violation - epsilon;

            if g_x >= 0.0 {
                return Ok(f32::MAX);
            }

            barrier_term -= g_x.ln();
        }

        Ok(task_loss + self.barrier_weight * barrier_term)
    }

    /// Decrease barrier weight (called after each iteration to approach boundary)
    pub fn decrease_barrier(&mut self) {
        self.barrier_weight *= self.barrier_decrease_factor;
    }

    /// Get current barrier weight
    pub fn barrier_weight(&self) -> f32 {
        self.barrier_weight
    }

    /// Reset barrier weight
    pub fn reset_barrier(&mut self, initial_weight: f32) {
        self.barrier_weight = initial_weight;
    }
}

impl Default for BarrierMethod {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Adaptive Constraint Weighting
// ============================================================================

/// Adaptive constraint weighting for balanced training
///
/// Adjusts constraint weights during training to balance task and constraint losses
pub struct AdaptiveWeighting {
    constraint_weights: Vec<f32>,
    learning_rate: f32,
    min_weight: f32,
    max_weight: f32,
}

impl AdaptiveWeighting {
    /// Create adaptive weighting for given number of constraints
    pub fn new(num_constraints: usize) -> Self {
        Self {
            constraint_weights: vec![1.0; num_constraints],
            learning_rate: 0.01,
            min_weight: 0.01,
            max_weight: 100.0,
        }
    }

    /// Set learning rate for weight adaptation
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set weight bounds
    pub fn with_weight_bounds(mut self, min: f32, max: f32) -> Self {
        self.min_weight = min;
        self.max_weight = max;
        self
    }

    /// Update weights based on constraint violations
    ///
    /// Increases weight for frequently violated constraints
    pub fn update_weights<C: ViolationComputable>(
        &mut self,
        prediction: &[f32],
        constraints: &[C],
    ) {
        for (weight, constraint) in self.constraint_weights.iter_mut().zip(constraints.iter()) {
            let violation = constraint.violation(prediction);

            // Increase weight if violated, decrease if satisfied
            let adjustment = if violation > 0.0 {
                self.learning_rate * violation
            } else {
                -self.learning_rate * 0.1 // Slow decrease when satisfied
            };

            *weight = (*weight + adjustment).clamp(self.min_weight, self.max_weight);
        }
    }

    /// Get constraint weights
    pub fn weights(&self) -> &[f32] {
        &self.constraint_weights
    }

    /// Compute weighted constraint penalty
    pub fn weighted_penalty<C: ViolationComputable>(
        &self,
        prediction: &[f32],
        constraints: &[C],
        penalty: PenaltyFunction,
    ) -> f32 {
        self.constraint_weights
            .iter()
            .zip(constraints.iter())
            .map(|(&weight, c)| {
                let violation = c.violation(prediction);
                penalty.compute(violation, weight)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintBuilder;

    #[test]
    fn test_differentiable_projection() {
        let proj = DifferentiableProjection::new(0.1).expect("valid temperature");
        assert!(DifferentiableProjection::new(0.0).is_err());
        assert!(DifferentiableProjection::new(f32::NAN).is_err());

        // Soft clamp to [0, 1]
        let result = proj.soft_project_box(-0.5, 0.0, 1.0);
        assert!(result < 0.2); // Should be close to lower bound

        let result = proj.soft_project_box(1.5, 0.0, 1.0);
        assert!(result > 0.8); // Should be close to upper bound

        let result = proj.soft_project_box(0.5, 0.0, 1.0);
        assert!((result - 0.5).abs() < 0.2); // Should stay near 0.5
    }

    #[test]
    fn test_constraint_aware_loss() {
        let constraint = ConstraintBuilder::new()
            .name("max_val")
            .less_eq(10.0)
            .build()
            .unwrap();

        let loss = ConstraintAwareLoss::new(vec![constraint], PenaltyFunction::L2, 1.0);

        // Within constraint
        assert!(loss.all_satisfied(&[5.0]));
        let penalty = loss.constraint_penalty(&[5.0]);
        assert_eq!(penalty, 0.0);

        // Violating constraint: prediction = 15, violation = 5
        assert!(!loss.all_satisfied(&[15.0]));
        let penalty = loss.constraint_penalty(&[15.0]);
        assert!((penalty - 25.0).abs() < 0.01); // L2: 1.0 * 5² = 25
    }

    #[test]
    fn test_lagrangian_relaxation() {
        let constraint = ConstraintBuilder::new()
            .name("upper")
            .less_eq(5.0)
            .build()
            .unwrap();

        let mut lagrangian = LagrangianRelaxation::new(1).with_multiplier_lr(0.1);

        // Initial lagrangian with λ=0
        let task_loss = 1.0;
        let lag =
            lagrangian.compute_lagrangian(&[3.0], std::slice::from_ref(&constraint), task_loss);
        assert_eq!(lag, task_loss); // No constraint contribution

        // Update multipliers with violation
        lagrangian.update_multipliers(&[10.0], std::slice::from_ref(&constraint)); // violation = 5

        // Now lagrangian should include constraint term
        let lag = lagrangian.compute_lagrangian(&[10.0], &[constraint], task_loss);
        assert!(lag > task_loss); // Multiplier * violation added
    }

    #[test]
    fn test_penalty_method() {
        let mut penalty = PenaltyMethod::new(PenaltyFunction::L2).with_penalty_weight(1.0);

        let constraint = ConstraintBuilder::new()
            .name("bound")
            .less_eq(10.0)
            .build()
            .unwrap();

        // Compute loss with violation
        let task_loss = 2.0;
        let total = penalty.compute_loss(&[15.0], &[constraint], task_loss);
        // violation = 5, L2 penalty = 1.0 * 5² = 25
        assert!((total - 27.0).abs() < 0.01);

        // Increase penalty
        penalty.increase_penalty();
        assert_eq!(penalty.penalty_weight(), 10.0);
    }

    #[test]
    fn test_adaptive_weighting() {
        let constraint1 = ConstraintBuilder::new()
            .name("c1")
            .less_eq(5.0)
            .build()
            .unwrap();

        let constraint2 = ConstraintBuilder::new()
            .name("c2")
            .greater_eq(0.0)
            .build()
            .unwrap();

        let mut adaptive = AdaptiveWeighting::new(2).with_learning_rate(0.1);

        // Initial weights are 1.0
        assert_eq!(adaptive.weights(), &[1.0, 1.0]);

        // Update with violations: [10.0] violates c1 (violation=5) but satisfies c2
        adaptive.update_weights(&[10.0], &[constraint1.clone(), constraint2.clone()]);

        // Weight for c1 should increase, c2 should decrease slightly
        assert!(adaptive.weights()[0] > 1.0);
        assert!(adaptive.weights()[1] <= 1.0);
    }
}
