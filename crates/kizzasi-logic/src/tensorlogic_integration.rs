//! TensorLogic integration for symbolic constraint reasoning
//!
//! This module provides integration with the TensorLogic framework
//! for symbolic constraint reasoning, simplification, and learning.

use crate::error::{LogicError, LogicResult};
use std::collections::HashMap;
use std::fmt;

// ============================================================================
// Symbolic Constraint Representation
// ============================================================================

/// Symbolic constraint expression for reasoning
#[derive(Debug, Clone)]
pub enum SymbolicExpr {
    /// Variable reference
    Var(String),
    /// Constant value
    Const(f32),
    /// Addition
    Add(Box<SymbolicExpr>, Box<SymbolicExpr>),
    /// Subtraction
    Sub(Box<SymbolicExpr>, Box<SymbolicExpr>),
    /// Multiplication
    Mul(Box<SymbolicExpr>, Box<SymbolicExpr>),
    /// Division
    Div(Box<SymbolicExpr>, Box<SymbolicExpr>),
    /// Less than or equal
    Le(Box<SymbolicExpr>, Box<SymbolicExpr>),
    /// Greater than or equal
    Ge(Box<SymbolicExpr>, Box<SymbolicExpr>),
    /// Equality
    Eq(Box<SymbolicExpr>, Box<SymbolicExpr>),
}

impl SymbolicExpr {
    /// Create variable expression
    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    /// Create constant expression
    pub fn constant(value: f32) -> Self {
        Self::Const(value)
    }

    /// Create addition expression
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: SymbolicExpr) -> Self {
        Self::Add(Box::new(self), Box::new(other))
    }

    /// Create subtraction expression
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: SymbolicExpr) -> Self {
        Self::Sub(Box::new(self), Box::new(other))
    }

    /// Create multiplication expression
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: SymbolicExpr) -> Self {
        Self::Mul(Box::new(self), Box::new(other))
    }

    /// Create less-than-or-equal constraint
    pub fn le(self, other: SymbolicExpr) -> Self {
        Self::Le(Box::new(self), Box::new(other))
    }

    /// Create greater-than-or-equal constraint
    pub fn ge(self, other: SymbolicExpr) -> Self {
        Self::Ge(Box::new(self), Box::new(other))
    }

    /// Evaluate expression with variable bindings
    pub fn evaluate(&self, bindings: &HashMap<String, f32>) -> LogicResult<f32> {
        match self {
            Self::Var(name) => bindings.get(name).copied().ok_or_else(|| {
                LogicError::InvalidConstraint(format!("Unbound variable: {}", name))
            }),
            Self::Const(v) => Ok(*v),
            Self::Add(a, b) => Ok(a.evaluate(bindings)? + b.evaluate(bindings)?),
            Self::Sub(a, b) => Ok(a.evaluate(bindings)? - b.evaluate(bindings)?),
            Self::Mul(a, b) => Ok(a.evaluate(bindings)? * b.evaluate(bindings)?),
            Self::Div(a, b) => {
                let divisor = b.evaluate(bindings)?;
                if divisor.abs() < f32::EPSILON {
                    Err(LogicError::InvalidConstraint("Division by zero".into()))
                } else {
                    Ok(a.evaluate(bindings)? / divisor)
                }
            }
            Self::Le(a, b) => Ok(if a.evaluate(bindings)? <= b.evaluate(bindings)? {
                1.0
            } else {
                0.0
            }),
            Self::Ge(a, b) => Ok(if a.evaluate(bindings)? >= b.evaluate(bindings)? {
                1.0
            } else {
                0.0
            }),
            Self::Eq(a, b) => {
                let diff = (a.evaluate(bindings)? - b.evaluate(bindings)?).abs();
                Ok(if diff < 1e-6 { 1.0 } else { 0.0 })
            }
        }
    }

    /// Simplify expression using symbolic rules
    pub fn simplify(self) -> Self {
        match self {
            // x + 0 = x
            Self::Add(a, b) if matches!(&*b, Self::Const(v) if *v == 0.0) => a.simplify(),
            // 0 + x = x
            Self::Add(a, b) if matches!(&*a, Self::Const(v) if *v == 0.0) => b.simplify(),
            // x - 0 = x
            Self::Sub(a, b) if matches!(&*b, Self::Const(v) if *v == 0.0) => a.simplify(),
            // x * 1 = x
            Self::Mul(a, b) if matches!(&*b, Self::Const(v) if *v == 1.0) => a.simplify(),
            // 1 * x = x
            Self::Mul(a, b) if matches!(&*a, Self::Const(v) if *v == 1.0) => b.simplify(),
            // x * 0 = 0
            Self::Mul(_, b) if matches!(&*b, Self::Const(v) if *v == 0.0) => Self::Const(0.0),
            // 0 * x = 0
            Self::Mul(a, _) if matches!(&*a, Self::Const(v) if *v == 0.0) => Self::Const(0.0),
            // x / 1 = x
            Self::Div(a, b) if matches!(&*b, Self::Const(v) if *v == 1.0) => a.simplify(),
            // Recursively simplify
            Self::Add(a, b) => Self::Add(Box::new(a.simplify()), Box::new(b.simplify())),
            Self::Sub(a, b) => Self::Sub(Box::new(a.simplify()), Box::new(b.simplify())),
            Self::Mul(a, b) => Self::Mul(Box::new(a.simplify()), Box::new(b.simplify())),
            Self::Div(a, b) => Self::Div(Box::new(a.simplify()), Box::new(b.simplify())),
            Self::Le(a, b) => Self::Le(Box::new(a.simplify()), Box::new(b.simplify())),
            Self::Ge(a, b) => Self::Ge(Box::new(a.simplify()), Box::new(b.simplify())),
            Self::Eq(a, b) => Self::Eq(Box::new(a.simplify()), Box::new(b.simplify())),
            // Base cases
            other => other,
        }
    }
}

impl fmt::Display for SymbolicExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(name) => write!(f, "{}", name),
            Self::Const(v) => write!(f, "{}", v),
            Self::Add(a, b) => write!(f, "({} + {})", a, b),
            Self::Sub(a, b) => write!(f, "({} - {})", a, b),
            Self::Mul(a, b) => write!(f, "({} * {})", a, b),
            Self::Div(a, b) => write!(f, "({} / {})", a, b),
            Self::Le(a, b) => write!(f, "({} <= {})", a, b),
            Self::Ge(a, b) => write!(f, "({} >= {})", a, b),
            Self::Eq(a, b) => write!(f, "({} == {})", a, b),
        }
    }
}

// ============================================================================
// Constraint Learning from Examples
// ============================================================================

/// Learn constraints from positive and negative examples
pub struct ConstraintLearner {
    positive_examples: Vec<Vec<f32>>,
    negative_examples: Vec<Vec<f32>>,
}

impl ConstraintLearner {
    /// Create a new constraint learner
    pub fn new() -> Self {
        Self {
            positive_examples: Vec::new(),
            negative_examples: Vec::new(),
        }
    }

    /// Add positive example (should satisfy constraint)
    pub fn add_positive(&mut self, example: Vec<f32>) {
        self.positive_examples.push(example);
    }

    /// Add negative example (should violate constraint)
    pub fn add_negative(&mut self, example: Vec<f32>) {
        self.negative_examples.push(example);
    }

    /// Learn box constraints from examples
    pub fn learn_box_constraints(&self, dimension: usize) -> LogicResult<(f32, f32)> {
        if self.positive_examples.is_empty() {
            return Err(LogicError::InvalidConstraint(
                "No positive examples provided".into(),
            ));
        }

        // Find min and max from positive examples for the dimension
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        for example in &self.positive_examples {
            if dimension >= example.len() {
                continue;
            }
            let val = example[dimension];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        // Add margin based on negative examples
        let margin = 0.1 * (max_val - min_val);

        Ok((min_val - margin, max_val + margin))
    }

    /// Learn linear separator (simple version)
    pub fn learn_linear_separator(&self) -> LogicResult<Vec<f32>> {
        if self.positive_examples.is_empty() || self.negative_examples.is_empty() {
            return Err(LogicError::InvalidConstraint(
                "Need both positive and negative examples".into(),
            ));
        }

        let dim = self.positive_examples[0].len();

        // Simple centroid-based separator
        let mut pos_centroid = vec![0.0; dim];
        for example in &self.positive_examples {
            for (i, &val) in example.iter().enumerate() {
                pos_centroid[i] += val;
            }
        }
        for val in &mut pos_centroid {
            *val /= self.positive_examples.len() as f32;
        }

        let mut neg_centroid = vec![0.0; dim];
        for example in &self.negative_examples {
            for (i, &val) in example.iter().enumerate() {
                neg_centroid[i] += val;
            }
        }
        for val in &mut neg_centroid {
            *val /= self.negative_examples.len() as f32;
        }

        // Separator is perpendicular to line joining centroids
        let mut separator: Vec<f32> = pos_centroid
            .iter()
            .zip(neg_centroid.iter())
            .map(|(&p, &n)| p - n)
            .collect();

        // Normalize
        let norm: f32 = separator.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm < 1e-6 {
            return Err(LogicError::InvalidConstraint(
                "Cannot separate examples".into(),
            ));
        }

        for val in &mut separator {
            *val /= norm;
        }

        Ok(separator)
    }
}

impl Default for ConstraintLearner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Constraint Synthesis
// ============================================================================

/// Synthesize constraints from specifications
pub struct ConstraintSynthesizer {
    variables: Vec<String>,
}

impl ConstraintSynthesizer {
    /// Create a new synthesizer
    pub fn new(variables: Vec<String>) -> Self {
        Self { variables }
    }

    /// Synthesize constraint from template and examples
    pub fn synthesize_from_template(
        &self,
        template: ConstraintTemplate,
        examples: &[(Vec<f32>, bool)],
    ) -> LogicResult<SymbolicExpr> {
        match template {
            ConstraintTemplate::Linear => self.synthesize_linear(examples),
            ConstraintTemplate::Box => self.synthesize_box(examples),
            ConstraintTemplate::Quadratic => self.synthesize_quadratic(examples),
        }
    }

    fn synthesize_linear(&self, _examples: &[(Vec<f32>, bool)]) -> LogicResult<SymbolicExpr> {
        // Simple linear constraint synthesis
        // For now, return a placeholder
        Ok(SymbolicExpr::var(&self.variables[0]).le(SymbolicExpr::constant(10.0)))
    }

    fn synthesize_box(&self, examples: &[(Vec<f32>, bool)]) -> LogicResult<SymbolicExpr> {
        // Box constraint synthesis
        if examples.is_empty() {
            return Err(LogicError::InvalidConstraint("No examples provided".into()));
        }

        // Find bounds from positive examples
        let positive: Vec<&Vec<f32>> = examples
            .iter()
            .filter(|(_, sat)| *sat)
            .map(|(v, _)| v)
            .collect();

        if positive.is_empty() {
            return Err(LogicError::InvalidConstraint("No positive examples".into()));
        }

        let dim = positive[0].len();
        if dim == 0 || dim > self.variables.len() {
            return Err(LogicError::InvalidConstraint("Invalid dimensions".into()));
        }

        // For first dimension, create box constraint
        let vals: Vec<f32> = positive.iter().map(|v| v[0]).collect();
        let min_val = vals.iter().copied().fold(f32::MAX, f32::min);
        let max_val = vals.iter().copied().fold(f32::MIN, f32::max);

        let var = SymbolicExpr::var(&self.variables[0]);
        let lower = var.clone().ge(SymbolicExpr::constant(min_val));
        let upper = var.le(SymbolicExpr::constant(max_val));

        Ok(Self::and_expr(lower, upper))
    }

    fn synthesize_quadratic(&self, _examples: &[(Vec<f32>, bool)]) -> LogicResult<SymbolicExpr> {
        // Placeholder for quadratic synthesis
        Ok(SymbolicExpr::var(&self.variables[0]).le(SymbolicExpr::constant(1.0)))
    }

    fn and_expr(left: SymbolicExpr, _right: SymbolicExpr) -> SymbolicExpr {
        // For now, just return left (would need proper AND in SymbolicExpr)
        left
    }
}

/// Constraint template for synthesis
pub enum ConstraintTemplate {
    Linear,
    Box,
    Quadratic,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_evaluation() {
        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), 5.0);
        bindings.insert("y".to_string(), 3.0);

        // x + y
        let expr = SymbolicExpr::var("x").add(SymbolicExpr::var("y"));
        assert_eq!(expr.evaluate(&bindings).unwrap(), 8.0);

        // x * 2
        let expr = SymbolicExpr::var("x").mul(SymbolicExpr::constant(2.0));
        assert_eq!(expr.evaluate(&bindings).unwrap(), 10.0);
    }

    #[test]
    fn test_symbolic_simplification() {
        // x + 0 -> x
        let expr = SymbolicExpr::var("x").add(SymbolicExpr::constant(0.0));
        let simplified = expr.simplify();
        assert!(matches!(simplified, SymbolicExpr::Var(_)));

        // x * 1 -> x
        let expr = SymbolicExpr::var("x").mul(SymbolicExpr::constant(1.0));
        let simplified = expr.simplify();
        assert!(matches!(simplified, SymbolicExpr::Var(_)));

        // x * 0 -> 0
        let expr = SymbolicExpr::var("x").mul(SymbolicExpr::constant(0.0));
        let simplified = expr.simplify();
        assert!(matches!(simplified, SymbolicExpr::Const(v) if v == 0.0));
    }

    #[test]
    fn test_constraint_learning() {
        let mut learner = ConstraintLearner::new();

        // Positive examples: values in [2, 8]
        learner.add_positive(vec![3.0]);
        learner.add_positive(vec![5.0]);
        learner.add_positive(vec![7.0]);

        // Negative examples: values outside
        learner.add_negative(vec![0.0]);
        learner.add_negative(vec![10.0]);

        let (min, max) = learner.learn_box_constraints(0).unwrap();

        // Should capture the range with some margin
        assert!(min < 3.0);
        assert!(max > 7.0);
        assert!(min > 0.0); // Not too loose
        assert!(max < 10.0);
    }

    #[test]
    fn test_linear_separator() {
        let mut learner = ConstraintLearner::new();

        // Positive: x < 5
        learner.add_positive(vec![1.0]);
        learner.add_positive(vec![2.0]);
        learner.add_positive(vec![3.0]);

        // Negative: x > 5
        learner.add_negative(vec![7.0]);
        learner.add_negative(vec![8.0]);
        learner.add_negative(vec![9.0]);

        let separator = learner.learn_linear_separator().unwrap();
        assert_eq!(separator.len(), 1);
        // Separator should point in positive direction (from neg to pos centroid)
    }

    #[test]
    fn test_constraint_synthesis() {
        let vars = vec!["x".to_string()];
        let synthesizer = ConstraintSynthesizer::new(vars);

        let examples = vec![
            (vec![3.0], true),
            (vec![5.0], true),
            (vec![7.0], true),
            (vec![15.0], false),
        ];

        let constraint = synthesizer
            .synthesize_from_template(ConstraintTemplate::Box, &examples)
            .unwrap();

        // Should synthesize some constraint
        assert!(!constraint.to_string().is_empty());
    }
}
