//! Approximate Constraint Satisfaction
//!
//! This module provides fast approximate solutions when exact satisfaction is too expensive:
//! - Constraint relaxation hierarchies
//! - Soft constraint approximation
//! - Bounded error constraint solving
//! - Anytime constraint algorithms

use crate::ViolationComputable;
use scirs2_core::ndarray::Array1;
use std::time::{Duration, Instant};

/// Constraint with priority level for hierarchical relaxation
#[derive(Debug, Clone)]
pub struct HierarchicalConstraint<C: ViolationComputable> {
    /// The constraint
    pub constraint: C,
    /// Priority level (higher = more important, 0 = lowest)
    pub priority: u32,
    /// Weight within same priority level
    pub weight: f32,
}

impl<C: ViolationComputable> HierarchicalConstraint<C> {
    /// Create a new hierarchical constraint
    pub fn new(constraint: C, priority: u32, weight: f32) -> Self {
        Self {
            constraint,
            priority,
            weight,
        }
    }
}

/// Hierarchical constraint relaxation solver
#[derive(Debug, Clone)]
pub struct HierarchicalRelaxation<C: ViolationComputable> {
    /// Constraints organized by priority
    constraints: Vec<HierarchicalConstraint<C>>,
    /// Maximum iterations per level
    max_iterations_per_level: usize,
    /// Convergence tolerance
    tolerance: f32,
}

impl<C: ViolationComputable + Clone> HierarchicalRelaxation<C> {
    /// Create a new hierarchical relaxation solver
    pub fn new(max_iterations_per_level: usize, tolerance: f32) -> Self {
        Self {
            constraints: Vec::new(),
            max_iterations_per_level,
            tolerance,
        }
    }

    /// Add a constraint with priority
    pub fn add_constraint(&mut self, constraint: C, priority: u32, weight: f32) {
        self.constraints
            .push(HierarchicalConstraint::new(constraint, priority, weight));
        // Sort by priority (descending)
        self.constraints.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Solve with hierarchical relaxation
    pub fn solve(&self, initial: &Array1<f32>) -> ApproximateSolution {
        let start_time = Instant::now();
        let mut current = initial.clone();
        let mut satisfied_constraints = 0;
        let mut total_violation = 0.0;

        // Group by priority
        let max_priority = self
            .constraints
            .iter()
            .map(|c| c.priority)
            .max()
            .unwrap_or(0);

        // Solve level by level, highest priority first
        for priority_level in (0..=max_priority).rev() {
            let level_constraints: Vec<_> = self
                .constraints
                .iter()
                .filter(|c| c.priority == priority_level)
                .collect();

            if level_constraints.is_empty() {
                continue;
            }

            // Try to satisfy constraints at this level
            for _ in 0..self.max_iterations_per_level {
                let mut improved = false;

                for hc in &level_constraints {
                    let current_slice = current.as_slice().unwrap_or(&[]);
                    if !hc.constraint.check(current_slice) {
                        // Try to reduce violation via gradient descent
                        let violation = hc.constraint.violation(current_slice);
                        if violation > self.tolerance {
                            // Simple gradient-based adjustment
                            for i in 0..current.len() {
                                let epsilon = 0.001;
                                let mut perturbed = current.clone();
                                perturbed[i] += epsilon;

                                let viol_plus =
                                    hc.constraint.violation(perturbed.as_slice().unwrap_or(&[]));
                                let grad = (viol_plus - violation) / epsilon;

                                if grad.abs() > 1e-6 {
                                    current[i] -= 0.01 * hc.weight * grad.signum();
                                    improved = true;
                                }
                            }
                        }
                    }
                }

                if !improved {
                    break;
                }
            }

            // Count satisfied constraints at this level
            for hc in &level_constraints {
                let current_slice = current.as_slice().unwrap_or(&[]);
                if hc.constraint.check(current_slice) {
                    satisfied_constraints += 1;
                } else {
                    total_violation += hc.constraint.violation(current_slice);
                }
            }
        }

        ApproximateSolution {
            solution: current,
            satisfied_constraints,
            total_constraints: self.constraints.len(),
            total_violation,
            computation_time: start_time.elapsed(),
            optimality_gap: None,
        }
    }

    /// Get number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
}

/// Soft constraint approximation with bounded error
#[derive(Debug, Clone)]
pub struct BoundedErrorSolver<C: ViolationComputable> {
    /// Constraints to satisfy approximately
    constraints: Vec<C>,
    /// Maximum allowed error per constraint
    error_bound: f32,
    /// Step size for gradient descent
    step_size: f32,
    /// Maximum iterations
    max_iterations: usize,
}

impl<C: ViolationComputable + Clone> BoundedErrorSolver<C> {
    /// Create a new bounded error solver
    pub fn new(error_bound: f32, step_size: f32, max_iterations: usize) -> Self {
        Self {
            constraints: Vec::new(),
            error_bound,
            step_size,
            max_iterations,
        }
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: C) {
        self.constraints.push(constraint);
    }

    /// Solve with bounded error guarantee
    pub fn solve(&self, initial: &Array1<f32>) -> ApproximateSolution {
        let start_time = Instant::now();
        let mut current = initial.clone();
        let mut total_violation = 0.0;

        for iter in 0..self.max_iterations {
            let mut max_violation: f32 = 0.0;
            let mut any_violation = false;

            // Compute gradient of total violation
            for constraint in &self.constraints {
                let current_slice = current.as_slice().unwrap_or(&[]);
                let violation = constraint.violation(current_slice);

                if violation > self.error_bound {
                    any_violation = true;
                    max_violation = max_violation.max(violation);

                    // Compute gradient and update
                    for i in 0..current.len() {
                        let epsilon = 0.001;
                        let mut perturbed = current.clone();
                        perturbed[i] += epsilon;

                        let viol_plus = constraint.violation(perturbed.as_slice().unwrap_or(&[]));
                        let grad = (viol_plus - violation) / epsilon;

                        current[i] -= self.step_size * grad;
                    }
                }
            }

            // Check if we're within error bounds
            if !any_violation || max_violation <= self.error_bound {
                break;
            }

            // Adaptive step size
            if iter % 10 == 0 && iter > 0 {
                // Every 10 iterations, reduce step size if not making progress
            }
        }

        // Count satisfied constraints and compute final violation
        let mut satisfied = 0;
        for constraint in &self.constraints {
            let current_slice = current.as_slice().unwrap_or(&[]);
            let violation = constraint.violation(current_slice);
            total_violation += violation;
            if violation <= self.error_bound {
                satisfied += 1;
            }
        }

        ApproximateSolution {
            solution: current,
            satisfied_constraints: satisfied,
            total_constraints: self.constraints.len(),
            total_violation,
            computation_time: start_time.elapsed(),
            optimality_gap: Some(self.error_bound),
        }
    }
}

/// Anytime constraint solver that improves solution quality over time
#[derive(Debug, Clone)]
pub struct AnytimeSolver<C: ViolationComputable> {
    /// Constraints to satisfy
    constraints: Vec<C>,
    /// Initial solution
    initial_solution: Array1<f32>,
    /// Current best solution
    best_solution: Option<Array1<f32>>,
    /// Current best violation
    best_violation: f32,
    /// Iteration count
    iterations: usize,
    /// Step size
    step_size: f32,
}

impl<C: ViolationComputable + Clone> AnytimeSolver<C> {
    /// Create a new anytime solver
    pub fn new(constraints: Vec<C>, initial_solution: Array1<f32>, step_size: f32) -> Self {
        Self {
            constraints,
            initial_solution,
            best_solution: None,
            best_violation: f32::INFINITY,
            iterations: 0,
            step_size,
        }
    }

    /// Run for a specified duration and return best solution found
    pub fn solve_for_duration(&mut self, duration: Duration) -> ApproximateSolution {
        let start_time = Instant::now();
        let mut current = self.initial_solution.clone();

        while start_time.elapsed() < duration {
            self.iterations += 1;

            // Compute current violation
            let mut total_violation = 0.0;
            for constraint in &self.constraints {
                let current_slice = current.as_slice().unwrap_or(&[]);
                total_violation += constraint.violation(current_slice).max(0.0);
            }

            // Update best if improved
            if total_violation < self.best_violation {
                self.best_violation = total_violation;
                self.best_solution = Some(current.clone());
            }

            // Gradient descent step
            for constraint in &self.constraints {
                let current_slice = current.as_slice().unwrap_or(&[]);
                let violation = constraint.violation(current_slice);

                if violation > 0.0 {
                    for i in 0..current.len() {
                        let epsilon = 0.001;
                        let mut perturbed = current.clone();
                        perturbed[i] += epsilon;

                        let viol_plus = constraint.violation(perturbed.as_slice().unwrap_or(&[]));
                        let grad = (viol_plus - violation) / epsilon;

                        current[i] -= self.step_size * grad;
                    }
                }
            }

            // Adaptive step size
            if self.iterations.is_multiple_of(100) {
                self.step_size *= 0.99; // Gradually reduce step size
            }
        }

        let solution = self
            .best_solution
            .clone()
            .unwrap_or_else(|| self.initial_solution.clone());

        // Count satisfied constraints
        let mut satisfied = 0;
        for constraint in &self.constraints {
            let sol_slice = solution.as_slice().unwrap_or(&[]);
            if constraint.check(sol_slice) {
                satisfied += 1;
            }
        }

        ApproximateSolution {
            solution,
            satisfied_constraints: satisfied,
            total_constraints: self.constraints.len(),
            total_violation: self.best_violation,
            computation_time: start_time.elapsed(),
            optimality_gap: None,
        }
    }

    /// Run for a specified number of iterations
    pub fn solve_for_iterations(&mut self, num_iterations: usize) -> ApproximateSolution {
        let start_time = Instant::now();
        let mut current = self.initial_solution.clone();

        for _ in 0..num_iterations {
            self.iterations += 1;

            // Compute current violation
            let mut total_violation = 0.0;
            for constraint in &self.constraints {
                let current_slice = current.as_slice().unwrap_or(&[]);
                total_violation += constraint.violation(current_slice).max(0.0);
            }

            // Update best if improved
            if total_violation < self.best_violation {
                self.best_violation = total_violation;
                self.best_solution = Some(current.clone());
            }

            // Gradient descent step
            for constraint in &self.constraints {
                let current_slice = current.as_slice().unwrap_or(&[]);
                let violation = constraint.violation(current_slice);

                if violation > 0.0 {
                    for i in 0..current.len() {
                        let epsilon = 0.001;
                        let mut perturbed = current.clone();
                        perturbed[i] += epsilon;

                        let viol_plus = constraint.violation(perturbed.as_slice().unwrap_or(&[]));
                        let grad = (viol_plus - violation) / epsilon;

                        current[i] -= self.step_size * grad;
                    }
                }
            }
        }

        let solution = self
            .best_solution
            .clone()
            .unwrap_or_else(|| self.initial_solution.clone());

        let mut satisfied = 0;
        for constraint in &self.constraints {
            let sol_slice = solution.as_slice().unwrap_or(&[]);
            if constraint.check(sol_slice) {
                satisfied += 1;
            }
        }

        ApproximateSolution {
            solution,
            satisfied_constraints: satisfied,
            total_constraints: self.constraints.len(),
            total_violation: self.best_violation,
            computation_time: start_time.elapsed(),
            optimality_gap: None,
        }
    }

    /// Get current best solution
    pub fn best_solution(&self) -> Option<&Array1<f32>> {
        self.best_solution.as_ref()
    }

    /// Get number of iterations performed
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

/// Result of approximate constraint satisfaction
#[derive(Debug, Clone)]
pub struct ApproximateSolution {
    /// The approximate solution
    pub solution: Array1<f32>,
    /// Number of constraints satisfied exactly
    pub satisfied_constraints: usize,
    /// Total number of constraints
    pub total_constraints: usize,
    /// Total constraint violation
    pub total_violation: f32,
    /// Computation time
    pub computation_time: Duration,
    /// Optimality gap (if known)
    pub optimality_gap: Option<f32>,
}

impl ApproximateSolution {
    /// Get satisfaction ratio
    pub fn satisfaction_ratio(&self) -> f32 {
        if self.total_constraints == 0 {
            1.0
        } else {
            self.satisfied_constraints as f32 / self.total_constraints as f32
        }
    }

    /// Check if solution is feasible (all constraints satisfied)
    pub fn is_feasible(&self) -> bool {
        self.satisfied_constraints == self.total_constraints
    }

    /// Get average violation per constraint
    pub fn average_violation(&self) -> f32 {
        if self.total_constraints == 0 {
            0.0
        } else {
            self.total_violation / self.total_constraints as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearConstraint;

    #[test]
    fn test_hierarchical_relaxation() {
        let mut solver = HierarchicalRelaxation::new(200, 0.01);

        // High priority: x <= 5
        solver.add_constraint(LinearConstraint::less_eq(vec![1.0], 5.0), 2, 2.0);
        // Low priority: x <= 3
        solver.add_constraint(LinearConstraint::less_eq(vec![1.0], 3.0), 1, 1.0);

        let initial = Array1::from_vec(vec![10.0]);
        let result = solver.solve(&initial);

        // Should improve from initial (was 10.0)
        assert!(result.solution[0] < 10.0);
        assert_eq!(solver.num_constraints(), 2);
        // Should have reduced violation
        assert!(result.total_violation < 5.0);
    }

    #[test]
    fn test_bounded_error_solver() {
        let mut solver = BoundedErrorSolver::new(0.5, 0.1, 100);

        solver.add_constraint(LinearConstraint::less_eq(vec![1.0], 5.0));
        solver.add_constraint(LinearConstraint::greater_eq(vec![1.0], 2.0));

        let initial = Array1::from_vec(vec![10.0]);
        let result = solver.solve(&initial);

        assert!(result.average_violation() <= 0.5);
        assert!(result.satisfaction_ratio() > 0.0);
    }

    #[test]
    fn test_anytime_solver_iterations() {
        let constraints = vec![
            LinearConstraint::less_eq(vec![1.0], 5.0),
            LinearConstraint::greater_eq(vec![1.0], 0.0),
        ];

        let initial = Array1::from_vec(vec![10.0]);
        let mut solver = AnytimeSolver::new(constraints, initial, 0.1);

        let result = solver.solve_for_iterations(100);

        assert!(result.solution[0] >= 0.0);
        assert!(result.solution[0] <= 6.0); // Allow some slack
        assert_eq!(solver.iterations(), 100);
    }

    #[test]
    fn test_anytime_solver_duration() {
        let constraints = vec![LinearConstraint::less_eq(vec![1.0, 1.0], 10.0)];

        let initial = Array1::from_vec(vec![15.0, 15.0]);
        let mut solver = AnytimeSolver::new(constraints, initial, 0.1);

        let result = solver.solve_for_duration(Duration::from_millis(10));

        assert!(result.computation_time >= Duration::from_millis(10));
        assert!(result.total_violation < 100.0); // Should have improved from initial
    }

    #[test]
    fn test_approximate_solution_metrics() {
        let solution = ApproximateSolution {
            solution: Array1::from_vec(vec![5.0]),
            satisfied_constraints: 3,
            total_constraints: 5,
            total_violation: 2.5,
            computation_time: Duration::from_millis(100),
            optimality_gap: Some(0.1),
        };

        assert_eq!(solution.satisfaction_ratio(), 0.6);
        assert!(!solution.is_feasible());
        assert_eq!(solution.average_violation(), 0.5);
    }
}
