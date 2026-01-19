//! Incremental Constraint Solving
//!
//! This module provides incremental/online constraint solving that efficiently
//! updates solutions when constraints or variables are added/removed dynamically.
//!
//! # Key Features
//!
//! - **Incremental Addition**: Add constraints without full re-solve
//! - **Incremental Removal**: Remove constraints efficiently
//! - **Solution Repair**: Fix violations with minimal changes
//! - **Change Tracking**: Monitor which constraints changed
//! - **Backtracking**: Undo constraint additions
//!
//! # Use Cases
//!
//! - Real-time systems with dynamic constraints
//! - Interactive constraint debugging
//! - Streaming data with evolving constraints
//! - Adaptive control with changing specifications

use crate::constraint::Constraint;
use crate::error::{LogicError, LogicResult};
use scirs2_core::ndarray::Array1;
use std::collections::{HashMap, HashSet};

/// Incremental constraint change type
#[derive(Debug, Clone)]
pub enum ConstraintChange {
    /// Add a new constraint
    Add {
        /// Constraint ID
        id: usize,
        /// The constraint
        constraint: Constraint,
    },
    /// Remove an existing constraint
    Remove {
        /// Constraint ID to remove
        id: usize,
    },
    /// Modify an existing constraint
    Modify {
        /// Constraint ID to modify
        id: usize,
        /// New constraint value
        constraint: Constraint,
    },
}

/// Incremental solver state
#[derive(Debug, Clone)]
pub struct IncrementalState {
    /// Current solution
    pub solution: Array1<f32>,
    /// Active constraints
    pub active_constraints: HashMap<usize, Constraint>,
    /// Constraint violation status
    pub violations: HashMap<usize, f32>,
    /// Total violation
    pub total_violation: f32,
    /// Number of changes applied
    pub change_count: usize,
}

impl IncrementalState {
    /// Check if solution is feasible
    pub fn is_feasible(&self) -> bool {
        self.total_violation < 1e-6
    }

    /// Get number of violated constraints
    pub fn num_violated(&self) -> usize {
        self.violations.values().filter(|&&v| v > 1e-6).count()
    }
}

/// Incremental Constraint Solver
pub struct IncrementalSolver {
    /// Current state
    state: IncrementalState,
    /// Change history (for backtracking)
    history: Vec<IncrementalState>,
    /// Next constraint ID
    next_id: usize,
    /// Maximum repair iterations
    max_repair_iterations: usize,
    /// Repair step size
    repair_step_size: f32,
}

impl IncrementalSolver {
    /// Create a new incremental solver
    pub fn new(initial_solution: Array1<f32>) -> Self {
        Self {
            state: IncrementalState {
                solution: initial_solution,
                active_constraints: HashMap::new(),
                violations: HashMap::new(),
                total_violation: 0.0,
                change_count: 0,
            },
            history: Vec::new(),
            next_id: 0,
            max_repair_iterations: 100,
            repair_step_size: 0.01,
        }
    }

    /// Get next constraint ID
    fn get_next_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Add a constraint incrementally
    pub fn add_constraint(&mut self, constraint: Constraint) -> LogicResult<usize> {
        // Save current state for backtracking
        self.history.push(self.state.clone());

        let id = self.get_next_id();

        // Check if new constraint is violated
        let solution_slice: Vec<f32> = self.state.solution.iter().copied().collect();
        let violation = if solution_slice.is_empty() {
            0.0
        } else {
            constraint.violation(solution_slice[0])
        };

        // Add to active constraints
        self.state.active_constraints.insert(id, constraint);
        self.state.violations.insert(id, violation);
        self.state.total_violation += violation;
        self.state.change_count += 1;

        // If violated, attempt repair
        if violation > 1e-6 {
            self.repair_solution()?;
        }

        Ok(id)
    }

    /// Remove a constraint incrementally
    pub fn remove_constraint(&mut self, id: usize) -> LogicResult<()> {
        if !self.state.active_constraints.contains_key(&id) {
            return Err(LogicError::InvalidInput(format!(
                "Constraint {} does not exist",
                id
            )));
        }

        // Save current state
        self.history.push(self.state.clone());

        // Remove constraint
        self.state.active_constraints.remove(&id);
        if let Some(violation) = self.state.violations.remove(&id) {
            self.state.total_violation -= violation;
        }
        self.state.change_count += 1;

        Ok(())
    }

    /// Modify an existing constraint
    pub fn modify_constraint(&mut self, id: usize, new_constraint: Constraint) -> LogicResult<()> {
        if !self.state.active_constraints.contains_key(&id) {
            return Err(LogicError::InvalidInput(format!(
                "Constraint {} does not exist",
                id
            )));
        }

        // Save current state
        self.history.push(self.state.clone());

        // Remove old violation
        if let Some(old_violation) = self.state.violations.get(&id) {
            self.state.total_violation -= old_violation;
        }

        // Update constraint
        self.state
            .active_constraints
            .insert(id, new_constraint.clone());

        // Recompute violation
        let solution_slice: Vec<f32> = self.state.solution.iter().copied().collect();
        let new_violation = if solution_slice.is_empty() {
            0.0
        } else {
            new_constraint.violation(solution_slice[0])
        };

        self.state.violations.insert(id, new_violation);
        self.state.total_violation += new_violation;
        self.state.change_count += 1;

        // Repair if needed
        if new_violation > 1e-6 {
            self.repair_solution()?;
        }

        Ok(())
    }

    /// Repair solution to satisfy all constraints
    fn repair_solution(&mut self) -> LogicResult<()> {
        for _ in 0..self.max_repair_iterations {
            // Compute gradient of total violation
            let mut gradient = Array1::<f32>::zeros(self.state.solution.len());
            let mut any_violation = false;

            for (id, constraint) in &self.state.active_constraints {
                let solution_slice: Vec<f32> = self.state.solution.iter().copied().collect();
                let violation = if solution_slice.is_empty() {
                    0.0
                } else {
                    constraint.violation(solution_slice[0])
                };

                if violation > 1e-6 {
                    any_violation = true;

                    // Numerical gradient
                    let eps = 1e-5;
                    for i in 0..self.state.solution.len() {
                        let mut perturbed = solution_slice.clone();
                        perturbed[i] += eps;
                        let viol_plus = if perturbed.is_empty() {
                            0.0
                        } else {
                            constraint.violation(perturbed[0])
                        };

                        gradient[i] += (viol_plus - violation) / eps;
                    }

                    // Update violation tracking
                    self.state.violations.insert(*id, violation);
                }
            }

            if !any_violation {
                break;
            }

            // Gradient descent step
            self.state.solution = &self.state.solution - &(&gradient * self.repair_step_size);

            // Recompute total violation
            self.recompute_violations();
        }

        Ok(())
    }

    /// Recompute all violations
    fn recompute_violations(&mut self) {
        self.state.total_violation = 0.0;

        for (id, constraint) in &self.state.active_constraints {
            let solution_slice: Vec<f32> = self.state.solution.iter().copied().collect();
            let violation = if solution_slice.is_empty() {
                0.0
            } else {
                constraint.violation(solution_slice[0])
            };

            self.state.violations.insert(*id, violation);
            self.state.total_violation += violation;
        }
    }

    /// Backtrack to previous state
    pub fn backtrack(&mut self) -> LogicResult<()> {
        if let Some(previous_state) = self.history.pop() {
            self.state = previous_state;
            Ok(())
        } else {
            Err(LogicError::InvalidInput(
                "No previous state to backtrack to".to_string(),
            ))
        }
    }

    /// Get current state
    pub fn state(&self) -> &IncrementalState {
        &self.state
    }

    /// Get current solution
    pub fn solution(&self) -> &Array1<f32> {
        &self.state.solution
    }

    /// Get violated constraint IDs
    pub fn violated_constraints(&self) -> Vec<usize> {
        self.state
            .violations
            .iter()
            .filter(|(_, &v)| v > 1e-6)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Apply a batch of changes
    pub fn apply_batch(&mut self, changes: Vec<ConstraintChange>) -> LogicResult<()> {
        for change in changes {
            match change {
                ConstraintChange::Add { constraint, .. } => {
                    self.add_constraint(constraint)?;
                }
                ConstraintChange::Remove { id } => {
                    self.remove_constraint(id)?;
                }
                ConstraintChange::Modify { id, constraint } => {
                    self.modify_constraint(id, constraint)?;
                }
            }
        }
        Ok(())
    }

    /// Clear history to save memory
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get number of active constraints
    pub fn num_constraints(&self) -> usize {
        self.state.active_constraints.len()
    }
}

/// Constraint change detector
pub struct ConstraintChangeDetector {
    /// Previous constraint set (by hash or ID)
    previous_constraints: HashSet<usize>,
    /// Tolerance for detecting changes
    #[allow(dead_code)]
    tolerance: f32,
}

impl ConstraintChangeDetector {
    /// Create a new change detector
    pub fn new() -> Self {
        Self {
            previous_constraints: HashSet::new(),
            tolerance: 1e-6,
        }
    }

    /// Detect changes between constraint sets
    pub fn detect_changes(&mut self, current_ids: &HashSet<usize>) -> (Vec<usize>, Vec<usize>) {
        // Added constraints
        let added: Vec<usize> = current_ids
            .difference(&self.previous_constraints)
            .copied()
            .collect();

        // Removed constraints
        let removed: Vec<usize> = self
            .previous_constraints
            .difference(current_ids)
            .copied()
            .collect();

        // Update previous
        self.previous_constraints = current_ids.clone();

        (added, removed)
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.previous_constraints.clear();
    }
}

impl Default for ConstraintChangeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintBuilder;

    #[test]
    fn test_incremental_solver_creation() {
        let initial = Array1::from_vec(vec![0.0]);
        let solver = IncrementalSolver::new(initial);

        assert_eq!(solver.state().solution[0], 0.0);
        assert_eq!(solver.num_constraints(), 0);
        assert!(solver.state().is_feasible());
    }

    #[test]
    fn test_add_constraint() {
        let initial = Array1::from_vec(vec![5.0]);
        let mut solver = IncrementalSolver::new(initial);

        let constraint = ConstraintBuilder::new()
            .name("c1")
            .less_than(10.0)
            .build()
            .unwrap();

        let id = solver.add_constraint(constraint).unwrap();
        assert_eq!(solver.num_constraints(), 1);
        assert!(id == 0);
    }

    #[test]
    fn test_remove_constraint() {
        let initial = Array1::from_vec(vec![5.0]);
        let mut solver = IncrementalSolver::new(initial);

        let constraint = ConstraintBuilder::new()
            .name("c1")
            .less_than(10.0)
            .build()
            .unwrap();

        let id = solver.add_constraint(constraint).unwrap();
        assert_eq!(solver.num_constraints(), 1);

        solver.remove_constraint(id).unwrap();
        assert_eq!(solver.num_constraints(), 0);
    }

    #[test]
    fn test_modify_constraint() {
        let initial = Array1::from_vec(vec![5.0]);
        let mut solver = IncrementalSolver::new(initial);

        let constraint1 = ConstraintBuilder::new()
            .name("c1")
            .less_than(10.0)
            .build()
            .unwrap();

        let id = solver.add_constraint(constraint1).unwrap();

        let constraint2 = ConstraintBuilder::new()
            .name("c1_modified")
            .less_than(3.0)
            .build()
            .unwrap();

        solver.modify_constraint(id, constraint2).unwrap();
        assert_eq!(solver.num_constraints(), 1);
    }

    #[test]
    fn test_backtrack() {
        let initial = Array1::from_vec(vec![5.0]);
        let mut solver = IncrementalSolver::new(initial);

        let constraint = ConstraintBuilder::new()
            .name("c1")
            .less_than(10.0)
            .build()
            .unwrap();

        solver.add_constraint(constraint).unwrap();
        assert_eq!(solver.num_constraints(), 1);

        solver.backtrack().unwrap();
        assert_eq!(solver.num_constraints(), 0);
    }

    #[test]
    fn test_violated_constraints() {
        let initial = Array1::from_vec(vec![15.0]);
        let mut solver = IncrementalSolver::new(initial);

        let constraint = ConstraintBuilder::new()
            .name("c1")
            .less_than(10.0)
            .build()
            .unwrap();

        solver.add_constraint(constraint).unwrap();

        // Should have violations (15.0 > 10.0)
        let violated = solver.violated_constraints();
        assert!(!violated.is_empty());
    }

    #[test]
    fn test_batch_changes() {
        let initial = Array1::from_vec(vec![5.0]);
        let mut solver = IncrementalSolver::new(initial);

        let c1 = ConstraintBuilder::new()
            .name("c1")
            .less_than(10.0)
            .build()
            .unwrap();

        let c2 = ConstraintBuilder::new()
            .name("c2")
            .greater_than(0.0)
            .build()
            .unwrap();

        let changes = vec![
            ConstraintChange::Add {
                id: 0,
                constraint: c1,
            },
            ConstraintChange::Add {
                id: 1,
                constraint: c2,
            },
        ];

        solver.apply_batch(changes).unwrap();
        assert_eq!(solver.num_constraints(), 2);
    }

    #[test]
    fn test_change_detector() {
        let mut detector = ConstraintChangeDetector::new();

        let set1: HashSet<usize> = [1, 2, 3].iter().copied().collect();
        let (added, removed) = detector.detect_changes(&set1);

        assert_eq!(added.len(), 3);
        assert_eq!(removed.len(), 0);

        let set2: HashSet<usize> = [2, 3, 4].iter().copied().collect();
        let (added, removed) = detector.detect_changes(&set2);

        assert_eq!(added.len(), 1); // 4 was added
        assert_eq!(removed.len(), 1); // 1 was removed
    }

    #[test]
    fn test_clear_history() {
        let initial = Array1::from_vec(vec![5.0]);
        let mut solver = IncrementalSolver::new(initial);

        let constraint = ConstraintBuilder::new()
            .name("c1")
            .less_than(10.0)
            .build()
            .unwrap();

        solver.add_constraint(constraint).unwrap();
        assert!(!solver.history.is_empty());

        solver.clear_history();
        assert!(solver.history.is_empty());
    }
}
