//! Projection onto constraint manifolds

use crate::constraint::Constraint;
use crate::error::LogicResult;
use scirs2_core::ndarray::Array1;

/// Projects predictions onto valid constraint manifolds
#[derive(Debug, Clone)]
pub struct ConstrainedProjection {
    constraints: Vec<Constraint>,
    max_iterations: usize,
    tolerance: f32,
}

impl ConstrainedProjection {
    /// Create a new constrained projection
    pub fn new(constraints: Vec<Constraint>) -> Self {
        Self {
            constraints,
            max_iterations: 10,
            tolerance: 1e-6,
        }
    }

    /// Set maximum projection iterations
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    /// Project a vector onto the valid manifold using iterative projection
    pub fn project(&self, input: &Array1<f32>) -> LogicResult<Array1<f32>> {
        let mut result = input.clone();

        for _ in 0..self.max_iterations {
            let prev = result.clone();

            // Apply each constraint's projection
            for constraint in &self.constraints {
                if let Some(dim) = constraint.dimension() {
                    if dim < result.len() {
                        result[dim] = constraint.project(result[dim]);
                    }
                } else {
                    // Apply to all dimensions
                    for val in result.iter_mut() {
                        *val = constraint.project(*val);
                    }
                }
            }

            // Check convergence
            let diff: f32 = result
                .iter()
                .zip(prev.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            if diff < self.tolerance {
                break;
            }
        }

        Ok(result)
    }

    /// Compute total violation of all constraints
    pub fn total_violation(&self, input: &Array1<f32>) -> f32 {
        let mut total = 0.0;

        for constraint in &self.constraints {
            if let Some(dim) = constraint.dimension() {
                if dim < input.len() {
                    total += constraint.violation(input[dim]) * constraint.weight();
                }
            } else {
                for val in input.iter() {
                    total += constraint.violation(*val) * constraint.weight();
                }
            }
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintBuilder;

    #[test]
    fn test_projection() {
        let constraints = vec![
            ConstraintBuilder::new()
                .name("bound1")
                .dimension(0)
                .in_range(-1.0, 1.0)
                .build()
                .unwrap(),
            ConstraintBuilder::new()
                .name("bound2")
                .dimension(1)
                .less_eq(0.5)
                .build()
                .unwrap(),
        ];

        let proj = ConstrainedProjection::new(constraints);
        let input = Array1::from_vec(vec![2.0, 1.0, 0.3]);
        let output = proj.project(&input).unwrap();

        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 0.5);
        assert_eq!(output[2], 0.3);
    }
}
