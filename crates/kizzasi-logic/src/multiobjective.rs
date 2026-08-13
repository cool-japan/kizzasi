//! Multi-Objective Constraint Optimization
//!
//! This module provides algorithms for multi-objective optimization under constraints:
//! - Pareto frontier computation
//! - NSGA-II (Non-dominated Sorting Genetic Algorithm)
//! - Weighted sum methods
//! - ε-constraint method
//! - Hypervolume indicator
//!
//! # Applications
//!
//! - Trade-off analysis between conflicting objectives
//! - Multi-criteria decision making
//! - Resource allocation under constraints
//! - Design optimization with multiple goals

use crate::error::{LogicError, LogicResult};
use scirs2_core::ndarray::Array1;
use std::cmp::Ordering;

/// Multi-objective solution point
#[derive(Debug, Clone)]
pub struct MultiObjectiveSolution {
    /// Decision variables
    pub variables: Array1<f32>,
    /// Objective values (one per objective)
    pub objectives: Vec<f32>,
    /// Constraint violations
    pub violations: Vec<f32>,
    /// Total constraint violation
    pub total_violation: f32,
    /// Pareto rank (0 = non-dominated front)
    pub rank: usize,
    /// Crowding distance (for diversity)
    pub crowding_distance: f32,
}

impl MultiObjectiveSolution {
    /// Check if this solution dominates another
    ///
    /// Solution A dominates B if:
    /// - A is feasible and B is not, OR
    /// - Both are feasible, A is no worse in all objectives, and better in at least one
    pub fn dominates(&self, other: &Self, minimize: &[bool]) -> bool {
        // Feasibility check first
        let self_feasible = self.is_feasible();
        let other_feasible = other.is_feasible();

        if self_feasible && !other_feasible {
            return true;
        }
        if !self_feasible && other_feasible {
            return false;
        }
        if !self_feasible && !other_feasible {
            // Both infeasible: compare constraint violations
            return self.total_violation < other.total_violation;
        }

        // Both feasible: check Pareto dominance
        let mut better_in_at_least_one = false;
        let mut worse_in_any = false;

        for (i, (&obj_a, &obj_b)) in self
            .objectives
            .iter()
            .zip(other.objectives.iter())
            .enumerate()
        {
            let cmp = if minimize[i] {
                obj_a.partial_cmp(&obj_b)
            } else {
                obj_b.partial_cmp(&obj_a)
            };

            match cmp {
                Some(Ordering::Less) => better_in_at_least_one = true,
                Some(Ordering::Greater) => worse_in_any = true,
                _ => {}
            }
        }

        better_in_at_least_one && !worse_in_any
    }

    /// Check if solution is feasible
    pub fn is_feasible(&self) -> bool {
        self.total_violation < 1e-6
    }

    /// Compute Euclidean distance in objective space
    pub fn objective_distance(&self, other: &Self) -> f32 {
        self.objectives
            .iter()
            .zip(other.objectives.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Multi-objective optimizer
pub struct MultiObjectiveOptimizer {
    /// Number of objectives
    num_objectives: usize,
    /// Minimize (true) or maximize (false) for each objective
    minimize: Vec<bool>,
    /// Population size
    population_size: usize,
    /// Maximum generations
    max_generations: usize,
    /// Mutation rate
    #[allow(dead_code)]
    mutation_rate: f32,
}

impl MultiObjectiveOptimizer {
    /// Create a new multi-objective optimizer
    pub fn new(num_objectives: usize, minimize: Vec<bool>) -> Self {
        assert_eq!(
            num_objectives,
            minimize.len(),
            "Minimize vector must match number of objectives"
        );

        Self {
            num_objectives,
            minimize,
            population_size: 100,
            max_generations: 100,
            mutation_rate: 0.1,
        }
    }

    /// Set population size
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set maximum generations
    pub fn with_max_generations(mut self, gens: usize) -> Self {
        self.max_generations = gens;
        self
    }

    /// Compute non-dominated sorting (NSGA-II style)
    pub fn non_dominated_sort(&self, population: &mut [MultiObjectiveSolution]) -> Vec<Vec<usize>> {
        let n = population.len();
        let mut domination_count = vec![0; n];
        let mut dominated_solutions = vec![Vec::new(); n];
        let mut fronts = vec![Vec::new()];

        // Compute domination relationships
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if population[i].dominates(&population[j], &self.minimize) {
                    dominated_solutions[i].push(j);
                } else if population[j].dominates(&population[i], &self.minimize) {
                    domination_count[i] += 1;
                }
            }

            // Solutions with rank 0 are in first front
            if domination_count[i] == 0 {
                population[i].rank = 0;
                fronts[0].push(i);
            }
        }

        // Build subsequent fronts
        let mut current_front = 0;
        while current_front < fronts.len() && !fronts[current_front].is_empty() {
            let mut next_front = Vec::new();

            for &i in &fronts[current_front] {
                for &j in &dominated_solutions[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        population[j].rank = current_front + 1;
                        next_front.push(j);
                    }
                }
            }

            if !next_front.is_empty() {
                fronts.push(next_front);
            }
            current_front += 1;
        }

        fronts
    }

    /// Compute crowding distance for diversity preservation
    pub fn compute_crowding_distance(&self, population: &mut [MultiObjectiveSolution]) {
        let n = population.len();

        if n == 0 {
            return;
        }

        // Initialize crowding distances
        for solution in population.iter_mut() {
            solution.crowding_distance = 0.0;
        }

        // For each objective
        for obj_idx in 0..self.num_objectives {
            // Sort by this objective
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| {
                population[a].objectives[obj_idx]
                    .partial_cmp(&population[b].objectives[obj_idx])
                    .unwrap_or(Ordering::Equal)
            });

            // Boundary points get infinite distance
            population[indices[0]].crowding_distance = f32::INFINITY;
            population[indices[n - 1]].crowding_distance = f32::INFINITY;

            // Compute range
            let obj_min = population[indices[0]].objectives[obj_idx];
            let obj_max = population[indices[n - 1]].objectives[obj_idx];
            let range = obj_max - obj_min;

            if range < 1e-10 {
                continue;
            }

            // Compute crowding distance for interior points
            for i in 1..(n - 1) {
                let prev_obj = population[indices[i - 1]].objectives[obj_idx];
                let next_obj = population[indices[i + 1]].objectives[obj_idx];
                population[indices[i]].crowding_distance += (next_obj - prev_obj) / range;
            }
        }
    }

    /// Get Pareto frontier (non-dominated front)
    pub fn get_pareto_frontier(
        &self,
        population: &[MultiObjectiveSolution],
    ) -> Vec<MultiObjectiveSolution> {
        let mut pop_copy = population.to_vec();
        let fronts = self.non_dominated_sort(&mut pop_copy);

        if fronts.is_empty() || fronts[0].is_empty() {
            return Vec::new();
        }

        fronts[0].iter().map(|&i| pop_copy[i].clone()).collect()
    }
}

/// Weighted sum method for multi-objective optimization
pub struct WeightedSumMethod {
    /// Weights for each objective (must sum to 1.0)
    weights: Vec<f32>,
}

impl WeightedSumMethod {
    /// Create weighted sum with specified weights
    pub fn new(weights: Vec<f32>) -> LogicResult<Self> {
        let sum: f32 = weights.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(LogicError::InvalidInput(
                "Weights must sum to 1.0".to_string(),
            ));
        }

        for &w in &weights {
            if w < 0.0 {
                return Err(LogicError::InvalidInput(
                    "Weights must be non-negative".to_string(),
                ));
            }
        }

        Ok(Self { weights })
    }

    /// Combine multiple objectives into single objective
    pub fn combine_objectives(&self, objectives: &[f32]) -> LogicResult<f32> {
        if objectives.len() != self.weights.len() {
            return Err(LogicError::InvalidInput(
                "Objective count mismatch".to_string(),
            ));
        }

        let combined = objectives
            .iter()
            .zip(self.weights.iter())
            .map(|(obj, weight)| obj * weight)
            .sum();

        Ok(combined)
    }
}

/// Hypervolume indicator for Pareto front quality
pub struct HypervolumeIndicator {
    /// Reference point (worst acceptable point)
    reference_point: Vec<f32>,
}

impl HypervolumeIndicator {
    /// Create hypervolume indicator with reference point
    pub fn new(reference_point: Vec<f32>) -> Self {
        Self { reference_point }
    }

    /// Compute hypervolume of a Pareto front
    ///
    /// Simplified 2D implementation (for 2 objectives)
    pub fn compute_2d(&self, front: &[MultiObjectiveSolution]) -> f32 {
        if front.is_empty() || self.reference_point.len() != 2 {
            return 0.0;
        }

        // Sort by first objective
        let mut sorted_front = front.to_vec();
        sorted_front.sort_by(|a, b| {
            a.objectives[0]
                .partial_cmp(&b.objectives[0])
                .unwrap_or(Ordering::Equal)
        });

        let mut hypervolume = 0.0;
        let mut prev_y = self.reference_point[1];

        for solution in sorted_front.iter() {
            let x = solution.objectives[0];
            let y = solution.objectives[1];

            if x < self.reference_point[0] && y < self.reference_point[1] {
                let width = self.reference_point[0] - x;
                let height = prev_y - y;
                hypervolume += width * height;
                prev_y = y;
            }
        }

        hypervolume
    }

    /// Compute the hypervolume contribution of the solution at `index`
    /// within `front`: `hypervolume(front) - hypervolume(front without that
    /// solution)`.
    ///
    /// Takes an index rather than a `&MultiObjectiveSolution` reference: the
    /// previous signature excluded the solution by pointer identity
    /// (`std::ptr::eq`), so any caller holding a clone — or a solution read
    /// back from storage — silently matched nothing, made
    /// `front_without == front`, and returned a contribution of exactly
    /// `0.0` indistinguishable from a genuinely non-contributing solution.
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidInput`] when `index` is out of bounds for `front`.
    pub fn contribution(&self, index: usize, front: &[MultiObjectiveSolution]) -> LogicResult<f32> {
        if index >= front.len() {
            return Err(LogicError::InvalidInput(format!(
                "contribution: index {index} out of bounds for front of length {}",
                front.len()
            )));
        }

        let hv_with = self.compute_2d(front);

        let front_without: Vec<MultiObjectiveSolution> = front
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(_, s)| s.clone())
            .collect();

        let hv_without = self.compute_2d(&front_without);

        Ok(hv_with - hv_without)
    }
}

/// ε-constraint method for multi-objective optimization
///
/// Optimizes one objective while constraining others
pub struct EpsilonConstraintMethod {
    /// Index of primary objective to optimize
    primary_objective: usize,
    /// Bounds on other objectives
    epsilon_bounds: Vec<f32>,
}

impl EpsilonConstraintMethod {
    /// Create ε-constraint method
    pub fn new(primary_objective: usize, epsilon_bounds: Vec<f32>) -> Self {
        Self {
            primary_objective,
            epsilon_bounds,
        }
    }

    /// Check if solution satisfies ε-constraints
    pub fn satisfies_constraints(&self, solution: &MultiObjectiveSolution) -> bool {
        for (i, &bound) in self.epsilon_bounds.iter().enumerate() {
            if i == self.primary_objective {
                continue;
            }
            if solution.objectives[i] > bound {
                return false;
            }
        }
        true
    }

    /// Get primary objective value
    pub fn primary_value(&self, solution: &MultiObjectiveSolution) -> f32 {
        solution.objectives[self.primary_objective]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominance() {
        let sol1 = MultiObjectiveSolution {
            variables: Array1::from_vec(vec![1.0]),
            objectives: vec![2.0, 3.0],
            violations: vec![],
            total_violation: 0.0,
            rank: 0,
            crowding_distance: 0.0,
        };

        let sol2 = MultiObjectiveSolution {
            variables: Array1::from_vec(vec![2.0]),
            objectives: vec![3.0, 4.0],
            violations: vec![],
            total_violation: 0.0,
            rank: 0,
            crowding_distance: 0.0,
        };

        let minimize = vec![true, true];
        assert!(sol1.dominates(&sol2, &minimize));
        assert!(!sol2.dominates(&sol1, &minimize));
    }

    #[test]
    fn test_non_dominated_sorting() {
        let optimizer = MultiObjectiveOptimizer::new(2, vec![true, true]);

        let mut population = vec![
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![1.0]),
                objectives: vec![1.0, 4.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![2.0]),
                objectives: vec![2.0, 3.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![3.0]),
                objectives: vec![3.0, 2.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![4.0]),
                objectives: vec![4.0, 1.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![5.0]),
                objectives: vec![2.5, 2.5],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        let fronts = optimizer.non_dominated_sort(&mut population);

        // All solutions should be on Pareto front (non-dominated)
        assert!(!fronts.is_empty());
        assert_eq!(fronts[0].len(), 5);
    }

    #[test]
    fn test_crowding_distance() {
        let optimizer = MultiObjectiveOptimizer::new(2, vec![true, true]);

        let mut population = vec![
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![1.0]),
                objectives: vec![1.0, 5.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![2.0]),
                objectives: vec![3.0, 3.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![3.0]),
                objectives: vec![5.0, 1.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        optimizer.compute_crowding_distance(&mut population);

        // Boundary points should have infinite crowding distance
        assert_eq!(population[0].crowding_distance, f32::INFINITY);
        assert_eq!(population[2].crowding_distance, f32::INFINITY);
        // Interior point should have finite crowding distance
        assert!(population[1].crowding_distance.is_finite());
        assert!(population[1].crowding_distance > 0.0);
    }

    #[test]
    fn test_weighted_sum() {
        let method = WeightedSumMethod::new(vec![0.6, 0.4]).unwrap();
        let objectives = vec![10.0, 20.0];
        let combined = method.combine_objectives(&objectives).unwrap();

        assert!((combined - 14.0).abs() < 1e-5); // 0.6*10 + 0.4*20 = 14
    }

    #[test]
    fn test_weighted_sum_validation() {
        // Weights don't sum to 1.0
        let result = WeightedSumMethod::new(vec![0.5, 0.4]);
        assert!(result.is_err());

        // Negative weight
        let result = WeightedSumMethod::new(vec![1.5, -0.5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hypervolume_2d() {
        let indicator = HypervolumeIndicator::new(vec![10.0, 10.0]);

        let front = vec![
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![1.0]),
                objectives: vec![2.0, 8.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![2.0]),
                objectives: vec![5.0, 5.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![3.0]),
                objectives: vec![8.0, 2.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        let hv = indicator.compute_2d(&front);
        assert!(hv > 0.0);
    }

    #[test]
    fn test_epsilon_constraint() {
        let method = EpsilonConstraintMethod::new(0, vec![f32::MAX, 5.0]);

        let sol1 = MultiObjectiveSolution {
            variables: Array1::from_vec(vec![1.0]),
            objectives: vec![2.0, 3.0],
            violations: vec![],
            total_violation: 0.0,
            rank: 0,
            crowding_distance: 0.0,
        };

        let sol2 = MultiObjectiveSolution {
            variables: Array1::from_vec(vec![2.0]),
            objectives: vec![1.0, 6.0],
            violations: vec![],
            total_violation: 0.0,
            rank: 0,
            crowding_distance: 0.0,
        };

        assert!(method.satisfies_constraints(&sol1));
        assert!(!method.satisfies_constraints(&sol2));
        assert_eq!(method.primary_value(&sol1), 2.0);
    }

    #[test]
    fn test_pareto_frontier() {
        let optimizer = MultiObjectiveOptimizer::new(2, vec![true, true]);

        let population = vec![
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![1.0]),
                objectives: vec![1.0, 5.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![2.0]),
                objectives: vec![3.0, 3.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![3.0]),
                objectives: vec![5.0, 1.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![4.0]),
                objectives: vec![2.0, 6.0], // Dominated by sol1 (1.0<2.0 and 5.0<6.0)
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        let frontier = optimizer.get_pareto_frontier(&population);
        assert_eq!(frontier.len(), 3); // First 3 are non-dominated
    }

    /// Regression (finding 144): `contribution` used to exclude the target
    /// solution by pointer identity, so a *cloned* solution (not literally
    /// the same allocation as an element of `front`) silently matched
    /// nothing and always returned `0.0`. It now takes an index instead.
    #[test]
    fn test_contribution_of_cloned_solution_is_nonzero() {
        let indicator = HypervolumeIndicator::new(vec![10.0, 10.0]);
        let front = [
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![0.0]),
                objectives: vec![1.0, 8.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![1.0]),
                objectives: vec![5.0, 5.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                variables: Array1::from_vec(vec![2.0]),
                objectives: vec![8.0, 1.0],
                violations: vec![],
                total_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        // Build a second, independently-allocated front out of clones — the
        // old pointer-identity check would find no match for any element
        // here since none of these allocations are literally `front[i]`.
        let cloned_front: Vec<MultiObjectiveSolution> = front.to_vec();

        let contribution = indicator
            .contribution(1, &cloned_front)
            .expect("index 1 is in bounds");
        assert!(
            contribution > 0.0,
            "the middle solution contributes positive hypervolume, got {contribution}"
        );

        // Out-of-bounds index must error, not panic or silently return 0.0.
        let err = indicator.contribution(cloned_front.len(), &cloned_front);
        assert!(matches!(err, Err(LogicError::InvalidInput(_))));
    }
}
