//! Guardrails for runtime constraint enforcement

use crate::constraint::Constraint;
use crate::error::{LogicError, LogicResult};
use crate::ConstrainedInference;
use scirs2_core::ndarray::Array1;

/// A single guardrail combining a constraint with enforcement policy
#[derive(Debug, Clone)]
pub struct Guardrail {
    constraint: Constraint,
    /// Whether to hard-reject violations or soft-project
    hard_reject: bool,
}

impl Guardrail {
    /// Create a new guardrail
    pub fn new(constraint: Constraint, hard_reject: bool) -> Self {
        Self {
            constraint,
            hard_reject,
        }
    }

    /// Check if a value passes this guardrail
    pub fn check(&self, value: f32) -> bool {
        self.constraint.check(value)
    }

    /// Apply guardrail to a value
    pub fn apply(&self, value: f32) -> LogicResult<f32> {
        if self.constraint.check(value) {
            Ok(value)
        } else if self.hard_reject {
            Err(LogicError::ConstraintViolation {
                constraint: self.constraint.name().to_string(),
                value,
                bound: format!("{:?}", self.constraint),
            })
        } else {
            Ok(self.constraint.project(value))
        }
    }

    /// Get the underlying constraint
    pub fn constraint(&self) -> &Constraint {
        &self.constraint
    }
}

/// A set of guardrails for multi-dimensional signals
#[derive(Debug, Clone, Default)]
pub struct GuardrailSet {
    /// Global guardrails applied to all dimensions
    global: Vec<Guardrail>,
    /// Dimension-specific guardrails
    dimensional: Vec<(usize, Guardrail)>,
}

impl GuardrailSet {
    /// Create a new empty guardrail set
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a global guardrail
    pub fn add_global(&mut self, guardrail: Guardrail) {
        self.global.push(guardrail);
    }

    /// Add a dimension-specific guardrail
    pub fn add_dimensional(&mut self, dim: usize, guardrail: Guardrail) {
        self.dimensional.push((dim, guardrail));
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.global.is_empty() && self.dimensional.is_empty()
    }
}

impl ConstrainedInference for GuardrailSet {
    fn constrain(&self, prediction: &Array1<f32>) -> LogicResult<Array1<f32>> {
        let mut result = prediction.clone();

        // Apply global constraints to all dimensions
        for guardrail in &self.global {
            for val in result.iter_mut() {
                *val = guardrail.apply(*val)?;
            }
        }

        // Apply dimension-specific constraints
        for (dim, guardrail) in &self.dimensional {
            if *dim < result.len() {
                result[*dim] = guardrail.apply(result[*dim])?;
            }
        }

        Ok(result)
    }

    fn validate(&self, prediction: &Array1<f32>) -> bool {
        // Check global constraints
        for guardrail in &self.global {
            for val in prediction.iter() {
                if !guardrail.check(*val) {
                    return false;
                }
            }
        }

        // Check dimension-specific constraints
        for (dim, guardrail) in &self.dimensional {
            if *dim < prediction.len() && !guardrail.check(prediction[*dim]) {
                return false;
            }
        }

        true
    }

    fn violation_loss(&self, prediction: &Array1<f32>) -> f32 {
        let mut total_loss = 0.0;

        // Global constraint violations
        for guardrail in &self.global {
            for val in prediction.iter() {
                total_loss +=
                    guardrail.constraint().violation(*val) * guardrail.constraint().weight();
            }
        }

        // Dimension-specific violations
        for (dim, guardrail) in &self.dimensional {
            if *dim < prediction.len() {
                total_loss += guardrail.constraint().violation(prediction[*dim])
                    * guardrail.constraint().weight();
            }
        }

        total_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintBuilder;

    #[test]
    fn test_guardrail_set() {
        let mut set = GuardrailSet::new();

        let constraint = ConstraintBuilder::new()
            .name("max_value")
            .less_eq(1.0)
            .build()
            .unwrap();

        set.add_global(Guardrail::new(constraint, false));

        let input = Array1::from_vec(vec![0.5, 1.5, 0.8]);
        let output = set.constrain(&input).unwrap();

        assert_eq!(output[0], 0.5);
        assert_eq!(output[1], 1.0);
        assert_eq!(output[2], 0.8);
    }
}
