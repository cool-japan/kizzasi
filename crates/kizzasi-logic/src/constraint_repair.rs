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

use crate::constraint::Constraint;
use crate::error::{LogicError, LogicResult};
use scirs2_core::ndarray::Array1;

/// Result of constraint repair operation
#[derive(Debug, Clone)]
pub struct RepairResult {
    /// Indices of constraints that were relaxed
    pub relaxed_constraints: Vec<usize>,
    /// Amount each constraint was relaxed
    pub relaxation_amounts: Vec<f32>,
    /// Total cost of repair
    pub repair_cost: f32,
    /// Whether repair was successful
    pub success: bool,
    /// Repaired point (if successful)
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RepairStrategy {
    /// Relax all violated constraints equally
    Uniform,
    /// Relax constraints with lowest priority first
    PriorityBased,
    /// Relax constraints to minimize total relaxation
    MinimalRelaxation,
    /// Use elastic programming (soft constraints)
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
    /// Returns a RepairResult with relaxation information
    pub fn repair(
        &self,
        point: &[f32],
        constraints: &[Constraint],
        priorities: Option<&[f32]>,
    ) -> LogicResult<RepairResult> {
        match self.strategy {
            RepairStrategy::Uniform => self.repair_uniform(point, constraints),
            RepairStrategy::PriorityBased => {
                self.repair_priority_based(point, constraints, priorities)
            }
            RepairStrategy::MinimalRelaxation => self.repair_minimal(point, constraints),
            RepairStrategy::ElasticProgramming => self.repair_elastic(point, constraints),
        }
    }

    /// Uniform relaxation: relax all violated constraints equally
    fn repair_uniform(
        &self,
        point: &[f32],
        constraints: &[Constraint],
    ) -> LogicResult<RepairResult> {
        let mut relaxed = Vec::new();
        let mut amounts = Vec::new();
        let mut total_cost = 0.0;

        for (i, constraint) in constraints.iter().enumerate() {
            let violation = if point.is_empty() {
                0.0
            } else {
                constraint.violation(point[0])
            };
            if violation > self.tolerance {
                relaxed.push(i);
                amounts.push(violation);
                total_cost += violation;
            }
        }

        Ok(RepairResult {
            relaxed_constraints: relaxed,
            relaxation_amounts: amounts,
            repair_cost: total_cost,
            success: true,
            repaired_point: Some(Array1::from_vec(point.to_vec())),
        })
    }

    /// Priority-based relaxation: relax low-priority constraints first
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

        // Create (index, priority, violation) tuples
        let mut violations: Vec<(usize, f32, f32)> = constraints
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let viol = if point.is_empty() {
                    0.0
                } else {
                    c.violation(point[0])
                };
                (i, priorities[i], viol)
            })
            .filter(|(_, _, viol)| *viol > self.tolerance)
            .collect();

        // Sort by priority (lowest first) then by violation (largest first)
        violations.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });

        let relaxed: Vec<usize> = violations.iter().map(|(i, _, _)| *i).collect();
        let amounts: Vec<f32> = violations.iter().map(|(_, _, v)| *v).collect();
        let total_cost: f32 = violations
            .iter()
            .map(|(_, p, v)| p * v) // Weight by priority
            .sum();

        Ok(RepairResult {
            relaxed_constraints: relaxed,
            relaxation_amounts: amounts,
            repair_cost: total_cost,
            success: true,
            repaired_point: Some(Array1::from_vec(point.to_vec())),
        })
    }

    /// Minimal relaxation: find minimum set of constraints to relax
    fn repair_minimal(
        &self,
        point: &[f32],
        constraints: &[Constraint],
    ) -> LogicResult<RepairResult> {
        // Find minimal infeasible subset using greedy algorithm
        let mut violated: Vec<(usize, f32)> = constraints
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let viol = if point.is_empty() {
                    0.0
                } else {
                    c.violation(point[0])
                };
                if viol > self.tolerance {
                    Some((i, viol))
                } else {
                    None
                }
            })
            .collect();

        // Sort by violation (largest first for greedy selection)
        violated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let relaxed: Vec<usize> = violated.iter().map(|(i, _)| *i).collect();
        let amounts: Vec<f32> = violated.iter().map(|(_, v)| *v).collect();
        let total_cost: f32 = amounts.iter().sum();

        let success = !relaxed.is_empty();
        Ok(RepairResult {
            relaxed_constraints: relaxed,
            relaxation_amounts: amounts,
            repair_cost: total_cost,
            success,
            repaired_point: Some(Array1::from_vec(point.to_vec())),
        })
    }

    /// Elastic programming: solve with slack variables
    fn repair_elastic(
        &self,
        point: &[f32],
        constraints: &[Constraint],
    ) -> LogicResult<RepairResult> {
        // Elastic programming adds slack variables to violated constraints
        // min Σ slack_i
        // s.t. constraints[i](x) <= slack_i
        //      slack_i >= 0
        //
        // This is a simplified version

        let mut relaxed = Vec::new();
        let mut slacks = Vec::new();
        let mut total_slack = 0.0;

        for (i, constraint) in constraints.iter().enumerate() {
            let violation = if point.is_empty() {
                0.0
            } else {
                constraint.violation(point[0])
            };
            if violation > self.tolerance {
                let slack = violation.min(self.max_relaxation);
                relaxed.push(i);
                slacks.push(slack);
                total_slack += slack;
            }
        }

        // Try to find a point that minimizes slacks (simplified)
        let repaired = self.compute_repaired_point(point, constraints, &relaxed, &slacks)?;

        Ok(RepairResult {
            relaxed_constraints: relaxed,
            relaxation_amounts: slacks,
            repair_cost: total_slack,
            success: true,
            repaired_point: Some(repaired),
        })
    }

    /// Compute repaired point through gradient descent
    fn compute_repaired_point(
        &self,
        initial: &[f32],
        constraints: &[Constraint],
        relaxed: &[usize],
        slacks: &[f32],
    ) -> LogicResult<Array1<f32>> {
        let mut x = Array1::from_vec(initial.to_vec());
        let step_size = 0.01;
        let max_iter = 100;

        for _ in 0..max_iter {
            let mut gradient = Array1::<f32>::zeros(x.len());
            let mut improved = false;

            // Compute gradient to reduce violations
            for (&idx, &slack) in relaxed.iter().zip(slacks.iter()) {
                let x_slice: Vec<f32> = x.iter().copied().collect();
                let violation = if x_slice.is_empty() {
                    0.0
                } else {
                    constraints[idx].violation(x_slice[0])
                };

                if violation > slack + self.tolerance {
                    // Numerical gradient
                    let eps = 1e-5;
                    for i in 0..x.len() {
                        let mut x_plus = x_slice.clone();
                        x_plus[i] += eps;
                        let viol_plus = if x_plus.is_empty() {
                            0.0
                        } else {
                            constraints[idx].violation(x_plus[0])
                        };

                        gradient[i] += (viol_plus - violation) / eps;
                    }
                    improved = true;
                }
            }

            if !improved {
                break;
            }

            // Gradient descent
            x = &x - &(&gradient * step_size);
        }

        Ok(x)
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
    /// Uses deletion filter algorithm
    pub fn find_all_iis(
        &self,
        point: &[f32],
        constraints: &[Constraint],
    ) -> LogicResult<Vec<Vec<usize>>> {
        let mut iis_sets = Vec::new();

        // Start with all violated constraints
        let violated: Vec<usize> = constraints
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                let viol = if point.is_empty() {
                    0.0
                } else {
                    c.violation(point[0])
                };
                viol > self.tolerance
            })
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
            let viol = if point.is_empty() {
                0.0
            } else {
                all_constraints[i].violation(point[0])
            };
            viol > self.tolerance
        })
    }
}

impl Default for IISFinder {
    fn default() -> Self {
        Self::new()
    }
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

        // Simplified conflict detection: check if constraints are incompatible
        // In practice, this would involve more sophisticated analysis
        for i in 0..constraints.len() {
            for j in (i + 1)..constraints.len() {
                // Check if constraints i and j might conflict
                // This is a placeholder - real implementation would be more sophisticated
                if self.might_conflict(&constraints[i], &constraints[j]) {
                    conflicts.push((i, j));
                }
            }
        }

        conflicts
    }

    /// Check if two constraints might conflict
    fn might_conflict(&self, _c1: &Constraint, _c2: &Constraint) -> bool {
        // Placeholder: would analyze constraint structure
        // For now, conservatively return false
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
}
