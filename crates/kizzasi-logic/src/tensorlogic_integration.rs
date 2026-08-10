//! TensorLogic integration for symbolic constraint reasoning
//!
//! This module integrates with the `tensorlogic-ir` crate for rich symbolic
//! constraint representation, optimization, and learning. It provides:
//!
//! - [`TLExprEvaluator`] — evaluates [`TLExpr`] against f32 variable bindings
//! - [`ConstraintLearner`] — learns constraints from positive/negative examples
//! - [`ConstraintSynthesizer`] — synthesizes constraint expressions from templates

use crate::error::{LogicError, LogicResult};
use std::collections::HashMap;
use tensorlogic_ir::Term;

pub use tensorlogic_ir::{TLExpr, Term as TlTerm, TypeAnnotation};

// ============================================================================
// TLExprEvaluator — f32 numerical evaluation of TLExpr
// ============================================================================

/// Evaluates a [`TLExpr`] symbolically against f32 variable bindings.
///
/// Uses **soft (Gödel) semantics** for logical operators:
/// - `And(a, b)` → `min(a, b)` (Gödel t-norm)
/// - `Or(a, b)` → `max(a, b)` (Gödel t-conorm)
/// - `Not(a)` → `1.0 - a` (clamped to [0, 1])
///
/// Boolean comparison operators (`Eq`, `Lt`, `Gt`, `Lte`, `Gte`) return `1.0`
/// (true) or `0.0` (false).
///
/// # Variable Naming Convention
///
/// Leaf nodes are `TLExpr::Pred { name, args }`:
/// - `args = []`, `name.parse::<f32>().is_ok()` → numeric constant
/// - `args = []`, name in bindings → variable lookup
/// - `args = [Term::Var(v)]` → look up `v` in bindings
/// - `args = [Term::Const(c)]` → parse `c` as f32
///
/// # Example
///
/// ```rust
/// use kizzasi_logic::TLExprEvaluator;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let mut eval = TLExprEvaluator::new();
/// eval.bind("x", 5.0);
///
/// // x <= 10.0
/// let expr = TLExpr::Lte(
///     Box::new(TLExpr::Pred { name: "x".into(), args: vec![] }),
///     Box::new(TLExpr::Pred { name: "10.0".into(), args: vec![] }),
/// );
/// assert_eq!(eval.evaluate(&expr).unwrap(), 1.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct TLExprEvaluator {
    bindings: HashMap<String, f32>,
}

impl TLExprEvaluator {
    /// Create an empty evaluator with no variable bindings.
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Builder-style binding of a variable to a value.
    pub fn with_binding(mut self, name: impl Into<String>, value: f32) -> Self {
        self.bindings.insert(name.into(), value);
        self
    }

    /// Bind a variable name to a f32 value.
    pub fn bind(&mut self, name: impl Into<String>, value: f32) -> &mut Self {
        self.bindings.insert(name.into(), value);
        self
    }

    /// Evaluate a [`TLExpr`] and return the raw f32 result.
    ///
    /// Boolean sub-expressions return `1.0` (true) or `0.0` (false).
    /// Logical connectives use Gödel (min/max) soft semantics over [0, 1].
    pub fn evaluate(&self, expr: &TLExpr) -> LogicResult<f32> {
        match expr {
            TLExpr::Pred { name, args } => self.eval_pred(name, args),

            // Arithmetic
            TLExpr::Add(a, b) => Ok(self.evaluate(a)? + self.evaluate(b)?),
            TLExpr::Sub(a, b) => Ok(self.evaluate(a)? - self.evaluate(b)?),
            TLExpr::Mul(a, b) => Ok(self.evaluate(a)? * self.evaluate(b)?),
            TLExpr::Div(a, b) => {
                let divisor = self.evaluate(b)?;
                if divisor.abs() < f32::EPSILON {
                    Err(LogicError::InvalidConstraint(
                        "TLExprEvaluator: division by zero".into(),
                    ))
                } else {
                    Ok(self.evaluate(a)? / divisor)
                }
            }
            TLExpr::Pow(a, b) => Ok(self.evaluate(a)?.powf(self.evaluate(b)?)),
            TLExpr::Mod(a, b) => Ok(self.evaluate(a)? % self.evaluate(b)?),
            TLExpr::Min(a, b) => Ok(self.evaluate(a)?.min(self.evaluate(b)?)),
            TLExpr::Max(a, b) => Ok(self.evaluate(a)?.max(self.evaluate(b)?)),

            // Unary arithmetic
            TLExpr::Abs(a) => Ok(self.evaluate(a)?.abs()),
            TLExpr::Floor(a) => Ok(self.evaluate(a)?.floor()),
            TLExpr::Ceil(a) => Ok(self.evaluate(a)?.ceil()),
            TLExpr::Round(a) => Ok(self.evaluate(a)?.round()),
            TLExpr::Sqrt(a) => {
                let v = self.evaluate(a)?;
                if v < 0.0 {
                    Err(LogicError::InvalidConstraint(format!(
                        "TLExprEvaluator: sqrt of negative value {v}"
                    )))
                } else {
                    Ok(v.sqrt())
                }
            }
            TLExpr::Exp(a) => Ok(self.evaluate(a)?.exp()),
            TLExpr::Log(a) => {
                let v = self.evaluate(a)?;
                if v <= 0.0 {
                    Err(LogicError::InvalidConstraint(format!(
                        "TLExprEvaluator: log of non-positive value {v}"
                    )))
                } else {
                    Ok(v.ln())
                }
            }
            TLExpr::Sin(a) => Ok((self.evaluate(a)? as f64).sin() as f32),
            TLExpr::Cos(a) => Ok((self.evaluate(a)? as f64).cos() as f32),
            TLExpr::Tan(a) => Ok((self.evaluate(a)? as f64).tan() as f32),

            // Comparisons → 1.0 (true) or 0.0 (false)
            TLExpr::Eq(a, b) => {
                let diff = (self.evaluate(a)? - self.evaluate(b)?).abs();
                Ok(if diff < 1e-6 { 1.0 } else { 0.0 })
            }
            TLExpr::Lt(a, b) => Ok(if self.evaluate(a)? < self.evaluate(b)? {
                1.0
            } else {
                0.0
            }),
            TLExpr::Gt(a, b) => Ok(if self.evaluate(a)? > self.evaluate(b)? {
                1.0
            } else {
                0.0
            }),
            TLExpr::Lte(a, b) => Ok(if self.evaluate(a)? <= self.evaluate(b)? {
                1.0
            } else {
                0.0
            }),
            TLExpr::Gte(a, b) => Ok(if self.evaluate(a)? >= self.evaluate(b)? {
                1.0
            } else {
                0.0
            }),

            // Logic — Gödel (min/max) soft semantics
            TLExpr::And(a, b) => Ok(self.evaluate(a)?.min(self.evaluate(b)?)),
            TLExpr::Or(a, b) => Ok(self.evaluate(a)?.max(self.evaluate(b)?)),
            TLExpr::Not(a) => Ok((1.0_f32 - self.evaluate(a)?).clamp(0.0, 1.0)),
            TLExpr::Imply(a, b) => {
                // Gödel implication: max(1 - a, b)
                Ok((1.0_f32 - self.evaluate(a)?)
                    .max(self.evaluate(b)?)
                    .clamp(0.0, 1.0))
            }
            TLExpr::Score(a) => self.evaluate(a),

            other => Err(LogicError::InvalidConstraint(format!(
                "TLExprEvaluator: unsupported TLExpr variant: {:?}",
                std::mem::discriminant(other)
            ))),
        }
    }

    /// Evaluate to a boolean by thresholding at 0.5.
    pub fn evaluate_bool(&self, expr: &TLExpr) -> LogicResult<bool> {
        Ok(self.evaluate(expr)? >= 0.5)
    }

    fn eval_pred(&self, name: &str, args: &[Term]) -> LogicResult<f32> {
        match args {
            [] => {
                // Try numeric constant first
                if let Ok(v) = name.parse::<f32>() {
                    return Ok(v);
                }
                // Then variable binding
                self.bindings.get(name).copied().ok_or_else(|| {
                    LogicError::InvalidConstraint(format!(
                        "TLExprEvaluator: unbound variable '{name}'"
                    ))
                })
            }
            [Term::Var(v)] => self.bindings.get(v.as_str()).copied().ok_or_else(|| {
                LogicError::InvalidConstraint(format!("TLExprEvaluator: unbound variable '{v}'"))
            }),
            [Term::Const(c)] => c.parse::<f32>().map_err(|_| {
                LogicError::InvalidConstraint(format!(
                    "TLExprEvaluator: cannot parse constant '{c}' as f32"
                ))
            }),
            [Term::Typed { value, .. }] => {
                // Recurse into typed term
                let inner_pred = TLExpr::Pred {
                    name: name.to_string(),
                    args: vec![*value.clone()],
                };
                self.evaluate(&inner_pred)
            }
            _ => Err(LogicError::InvalidConstraint(format!(
                "TLExprEvaluator: Pred '{name}' has unsupported arg list (len={})",
                args.len()
            ))),
        }
    }
}

// ============================================================================
// Helper: build TLExpr leaf nodes
// ============================================================================

/// Build a [`TLExpr`] variable node using a name-as-predicate convention.
///
/// The variable name is stored as the predicate name with no arguments.
/// [`TLExprEvaluator`] will look it up in its bindings map.
pub fn tl_var(name: impl Into<String>) -> TLExpr {
    TLExpr::Pred {
        name: name.into(),
        args: vec![],
    }
}

/// Build a [`TLExpr`] numeric constant node.
///
/// The value is stored as the predicate name (stringified float) with no args.
/// [`TLExprEvaluator`] and [`crate::TlExprCompiler`] will parse it back to f32.
pub fn tl_const(value: f32) -> TLExpr {
    TLExpr::Pred {
        name: value.to_string(),
        args: vec![],
    }
}

// ============================================================================
// ConstraintLearner
// ============================================================================

/// Learn constraints from positive and negative examples.
///
/// Positive examples are signal vectors that satisfy the target constraint.
/// Negative examples are vectors that should violate it.
pub struct ConstraintLearner {
    positive_examples: Vec<Vec<f32>>,
    negative_examples: Vec<Vec<f32>>,
}

impl ConstraintLearner {
    /// Create a new constraint learner.
    pub fn new() -> Self {
        Self {
            positive_examples: Vec::new(),
            negative_examples: Vec::new(),
        }
    }

    /// Add a positive example (should satisfy the constraint).
    pub fn add_positive(&mut self, example: Vec<f32>) {
        self.positive_examples.push(example);
    }

    /// Add a negative example (should violate the constraint).
    pub fn add_negative(&mut self, example: Vec<f32>) {
        self.negative_examples.push(example);
    }

    /// Learn box constraints from positive examples for a given dimension.
    ///
    /// Returns `(min, max)` bounds with a 10% margin.
    pub fn learn_box_constraints(&self, dimension: usize) -> LogicResult<(f32, f32)> {
        if self.positive_examples.is_empty() {
            return Err(LogicError::InvalidConstraint(
                "No positive examples provided".into(),
            ));
        }

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

        if min_val == f32::MAX {
            return Err(LogicError::InvalidConstraint(
                "No examples have data for the requested dimension".into(),
            ));
        }

        let margin = 0.1 * (max_val - min_val);
        Ok((min_val - margin, max_val + margin))
    }

    /// Learn box constraints for dimension 0 and return as a [`TLExpr`].
    ///
    /// Returns `And(Gte(var, lo), Lte(var, hi))` using the given `var_name`
    /// as the symbolic variable.
    ///
    /// The resulting expression can be evaluated with [`TLExprEvaluator`]
    /// by binding `var_name` to the signal value.
    pub fn learn_box_constraints_as_expr(&self, var_name: &str) -> LogicResult<TLExpr> {
        let (lo, hi) = self.learn_box_constraints(0)?;
        let var = tl_var(var_name);
        let lower = TLExpr::Gte(Box::new(var.clone()), Box::new(tl_const(lo)));
        let upper = TLExpr::Lte(Box::new(var), Box::new(tl_const(hi)));
        Ok(TLExpr::And(Box::new(lower), Box::new(upper)))
    }

    /// Learn a linear separator from positive and negative examples.
    ///
    /// Returns the separator normal vector (unit norm).
    pub fn learn_linear_separator(&self) -> LogicResult<Vec<f32>> {
        if self.positive_examples.is_empty() || self.negative_examples.is_empty() {
            return Err(LogicError::InvalidConstraint(
                "Need both positive and negative examples".into(),
            ));
        }

        let dim = self.positive_examples[0].len();

        let mut pos_centroid = vec![0.0_f32; dim];
        for example in &self.positive_examples {
            for (i, &val) in example.iter().enumerate() {
                if i < dim {
                    pos_centroid[i] += val;
                }
            }
        }
        for v in &mut pos_centroid {
            *v /= self.positive_examples.len() as f32;
        }

        let mut neg_centroid = vec![0.0_f32; dim];
        for example in &self.negative_examples {
            for (i, &val) in example.iter().enumerate() {
                if i < dim {
                    neg_centroid[i] += val;
                }
            }
        }
        for v in &mut neg_centroid {
            *v /= self.negative_examples.len() as f32;
        }

        let mut separator: Vec<f32> = pos_centroid
            .iter()
            .zip(neg_centroid.iter())
            .map(|(&p, &n)| p - n)
            .collect();

        let norm: f32 = separator.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm < 1e-6 {
            return Err(LogicError::InvalidConstraint(
                "Cannot separate examples: centroids coincide".into(),
            ));
        }

        for v in &mut separator {
            *v /= norm;
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
// ConstraintSynthesizer
// ============================================================================

/// Synthesize [`TLExpr`] constraints from templates and examples.
pub struct ConstraintSynthesizer {
    variables: Vec<String>,
}

impl ConstraintSynthesizer {
    /// Create a new synthesizer with the given variable names.
    pub fn new(variables: Vec<String>) -> Self {
        Self { variables }
    }

    /// Synthesize a constraint expression from a template and labeled examples.
    ///
    /// Each example is `(values, is_positive)` — positive means the values
    /// should satisfy the synthesized constraint.
    pub fn synthesize_from_template(
        &self,
        template: ConstraintTemplate,
        examples: &[(Vec<f32>, bool)],
    ) -> LogicResult<TLExpr> {
        match template {
            ConstraintTemplate::Linear => self.synthesize_linear(examples),
            ConstraintTemplate::Box => self.synthesize_box(examples),
            ConstraintTemplate::Quadratic => self.synthesize_quadratic(examples),
        }
    }

    fn synthesize_linear(&self, _examples: &[(Vec<f32>, bool)]) -> LogicResult<TLExpr> {
        let var_name = self
            .variables
            .first()
            .ok_or_else(|| LogicError::InvalidConstraint("No variables defined".into()))?;
        Ok(TLExpr::Lte(
            Box::new(tl_var(var_name)),
            Box::new(tl_const(10.0)),
        ))
    }

    fn synthesize_box(&self, examples: &[(Vec<f32>, bool)]) -> LogicResult<TLExpr> {
        if examples.is_empty() {
            return Err(LogicError::InvalidConstraint("No examples provided".into()));
        }
        let var_name = self
            .variables
            .first()
            .ok_or_else(|| LogicError::InvalidConstraint("No variables defined".into()))?;

        let positive: Vec<&Vec<f32>> = examples
            .iter()
            .filter(|(_, sat)| *sat)
            .map(|(v, _)| v)
            .collect();

        if positive.is_empty() {
            return Err(LogicError::InvalidConstraint("No positive examples".into()));
        }

        let vals: Vec<f32> = positive.iter().filter_map(|v| v.first().copied()).collect();
        if vals.is_empty() {
            return Err(LogicError::InvalidConstraint("Empty value vectors".into()));
        }

        let min_val = vals.iter().copied().fold(f32::MAX, f32::min);
        let max_val = vals.iter().copied().fold(f32::MIN, f32::max);

        let var = tl_var(var_name);
        let lower = TLExpr::Gte(Box::new(var.clone()), Box::new(tl_const(min_val)));
        let upper = TLExpr::Lte(Box::new(var), Box::new(tl_const(max_val)));
        Ok(TLExpr::And(Box::new(lower), Box::new(upper)))
    }

    fn synthesize_quadratic(&self, _examples: &[(Vec<f32>, bool)]) -> LogicResult<TLExpr> {
        let var_name = self
            .variables
            .first()
            .ok_or_else(|| LogicError::InvalidConstraint("No variables defined".into()))?;
        // Placeholder: x^2 <= 1  (||x||^2 <= 1)
        let x = tl_var(var_name);
        let x_sq = TLExpr::Mul(Box::new(x.clone()), Box::new(x));
        Ok(TLExpr::Lte(Box::new(x_sq), Box::new(tl_const(1.0))))
    }
}

/// Template type for constraint synthesis.
pub enum ConstraintTemplate {
    /// Linear constraint: a·x ≤ b
    Linear,
    /// Box constraint: lo ≤ x ≤ hi
    Box,
    /// Quadratic constraint: x²  ≤ r²
    Quadratic,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tl_expr_evaluator_arithmetic() {
        let eval = TLExprEvaluator::new()
            .with_binding("x", 5.0)
            .with_binding("y", 3.0);

        // x + y = 8
        let expr = TLExpr::Add(Box::new(tl_var("x")), Box::new(tl_var("y")));
        let result = eval.evaluate(&expr).expect("evaluate failed");
        assert!((result - 8.0).abs() < 1e-5, "expected 8.0, got {result}");

        // x * 2.0 = 10
        let expr = TLExpr::Mul(Box::new(tl_var("x")), Box::new(tl_const(2.0)));
        let result = eval.evaluate(&expr).expect("evaluate failed");
        assert!((result - 10.0).abs() < 1e-5, "expected 10.0, got {result}");
    }

    #[test]
    fn test_tl_expr_evaluator_comparison() {
        let eval = TLExprEvaluator::new().with_binding("x", 5.0);

        // x <= 10.0 → 1.0
        let expr = TLExpr::Lte(Box::new(tl_var("x")), Box::new(tl_const(10.0)));
        assert_eq!(eval.evaluate(&expr).expect("failed"), 1.0);

        // x <= 3.0 → 0.0
        let expr2 = TLExpr::Lte(Box::new(tl_var("x")), Box::new(tl_const(3.0)));
        assert_eq!(eval.evaluate(&expr2).expect("failed"), 0.0);
    }

    #[test]
    fn test_tl_expr_evaluator_soft_and() {
        let eval = TLExprEvaluator::new().with_binding("x", 5.0);

        // And(x >= 0.0, x <= 10.0) → min(1.0, 1.0) = 1.0
        let lower = TLExpr::Gte(Box::new(tl_var("x")), Box::new(tl_const(0.0)));
        let upper = TLExpr::Lte(Box::new(tl_var("x")), Box::new(tl_const(10.0)));
        let both = TLExpr::And(Box::new(lower), Box::new(upper));
        assert_eq!(eval.evaluate(&both).expect("failed"), 1.0);
    }

    #[test]
    fn test_tl_expr_evaluator_bool() {
        let eval = TLExprEvaluator::new().with_binding("x", 5.0);
        let expr = TLExpr::Lte(Box::new(tl_var("x")), Box::new(tl_const(10.0)));
        assert!(eval.evaluate_bool(&expr).expect("failed"));
    }

    #[test]
    fn test_constraint_learner_box() {
        let mut learner = ConstraintLearner::new();
        learner.add_positive(vec![3.0]);
        learner.add_positive(vec![5.0]);
        learner.add_positive(vec![7.0]);
        learner.add_negative(vec![0.0]);
        learner.add_negative(vec![10.0]);

        let (min, max) = learner.learn_box_constraints(0).expect("failed");
        assert!(min < 3.0, "min should have margin below 3.0, got {min}");
        assert!(max > 7.0, "max should have margin above 7.0, got {max}");
        assert!(min > 0.0, "min should not be too loose");
        assert!(max < 10.0, "max should not be too loose");
    }

    #[test]
    fn test_constraint_learner_as_expr() {
        let mut learner = ConstraintLearner::new();
        learner.add_positive(vec![3.0]);
        learner.add_positive(vec![5.0]);
        learner.add_positive(vec![7.0]);

        let expr = learner.learn_box_constraints_as_expr("x").expect("failed");

        // Must be And(Gte(...), Lte(...))
        assert!(matches!(expr, TLExpr::And(_, _)), "expected And node");

        // Evaluate: x=5 should satisfy (min=2.6, max=7.4 with 10% margin)
        let eval = TLExprEvaluator::new().with_binding("x", 5.0);
        assert!(
            eval.evaluate_bool(&expr).expect("eval failed"),
            "x=5 should satisfy"
        );

        // x=100 should fail
        let eval2 = TLExprEvaluator::new().with_binding("x", 100.0);
        assert!(
            !eval2.evaluate_bool(&expr).expect("eval failed"),
            "x=100 should fail"
        );
    }

    #[test]
    fn test_linear_separator() {
        let mut learner = ConstraintLearner::new();
        learner.add_positive(vec![1.0]);
        learner.add_positive(vec![2.0]);
        learner.add_positive(vec![3.0]);
        learner.add_negative(vec![7.0]);
        learner.add_negative(vec![8.0]);
        learner.add_negative(vec![9.0]);

        let sep = learner.learn_linear_separator().expect("failed");
        assert_eq!(sep.len(), 1);
        // The separator should have unit norm
        let norm: f32 = sep.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "separator should be unit norm");
    }

    #[test]
    fn test_constraint_synthesis_box() {
        let vars = vec!["x".to_string()];
        let synthesizer = ConstraintSynthesizer::new(vars);

        let examples = vec![
            (vec![3.0], true),
            (vec![5.0], true),
            (vec![7.0], true),
            (vec![15.0], false),
        ];

        let expr = synthesizer
            .synthesize_from_template(ConstraintTemplate::Box, &examples)
            .expect("failed");

        // Must produce an And(Gte, Lte) structure
        assert!(matches!(expr, TLExpr::And(_, _)));

        // x=5 should satisfy
        let eval = TLExprEvaluator::new().with_binding("x", 5.0);
        assert!(
            eval.evaluate_bool(&expr).expect("eval"),
            "x=5 should satisfy box constraint"
        );

        // x=15 should not satisfy
        let eval2 = TLExprEvaluator::new().with_binding("x", 15.0);
        assert!(
            !eval2.evaluate_bool(&expr).expect("eval"),
            "x=15 should fail box constraint"
        );
    }
}

#[cfg(test)]
mod end_to_end_tests {
    use super::*;
    use crate::compiler::TlExprCompiler;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_synthesizer_to_compiled_box() {
        // ConstraintSynthesizer::synthesize_box uses the positive examples
        // to extract min/max WITHOUT margin: [3.0, 7.0].
        let synthesizer = ConstraintSynthesizer::new(vec!["dim_0".to_string()]);
        let examples = vec![
            (vec![3.0_f32], true),
            (vec![5.0_f32], true),
            (vec![7.0_f32], true),
        ];
        let expr = synthesizer
            .synthesize_from_template(ConstraintTemplate::Box, &examples)
            .expect("synthesis failed");

        assert!(
            matches!(expr, TLExpr::And(_, _)),
            "expected And node from box synthesis"
        );

        let compiler = TlExprCompiler::new();
        let compiled = compiler
            .compile(&expr, "box_constraint", 1)
            .expect("compile failed");

        // 5.0 ∈ [3.0, 7.0] → satisfies
        let x_in = Array1::from_vec(vec![5.0_f32]);
        assert!(
            compiled.evaluate(&x_in).expect("evaluate failed"),
            "x=5.0 should satisfy [3.0, 7.0]"
        );

        // 15.0 ∉ [3.0, 7.0] → rejects
        let x_out = Array1::from_vec(vec![15.0_f32]);
        assert!(
            !compiled.evaluate(&x_out).expect("evaluate failed"),
            "x=15.0 should violate [3.0, 7.0]"
        );
    }

    #[test]
    fn test_learner_to_compiled_box() {
        // learn_box_constraints_as_expr adds 10% margin:
        // positives [3.0, 7.0] → range 4.0 → margin 0.4
        // lo = 3.0 - 0.4 = 2.6, hi = 7.0 + 0.4 = 7.4
        let mut learner = ConstraintLearner::new();
        learner.add_positive(vec![3.0]);
        learner.add_positive(vec![7.0]);

        let expr = learner
            .learn_box_constraints_as_expr("dim_0")
            .expect("learner failed");

        let compiler = TlExprCompiler::new();
        let compiled = compiler
            .compile(&expr, "learned_box", 1)
            .expect("compile failed");

        // 5.0 ∈ [2.6, 7.4] → satisfies
        let x_in = Array1::from_vec(vec![5.0_f32]);
        assert!(
            compiled.evaluate(&x_in).expect("eval failed"),
            "x=5.0 should satisfy learned box [2.6, 7.4]"
        );

        // 100.0 ∉ [2.6, 7.4] → rejects
        let x_out = Array1::from_vec(vec![100.0_f32]);
        assert!(
            !compiled.evaluate(&x_out).expect("eval failed"),
            "x=100.0 should violate learned box [2.6, 7.4]"
        );
    }

    #[test]
    fn test_evaluator_compiler_agreement() {
        // Build: And(Gte(dim_0, -1.0), Lte(dim_0, 1.0))
        // "dim_0" auto-maps to Dim(0) in TlExprCompiler (no with_var needed).
        let lower = TLExpr::Gte(Box::new(tl_var("dim_0")), Box::new(tl_const(-1.0)));
        let upper = TLExpr::Lte(Box::new(tl_var("dim_0")), Box::new(tl_const(1.0)));
        let expr = TLExpr::And(Box::new(lower), Box::new(upper));

        let compiler = TlExprCompiler::new();
        let compiled = compiler
            .compile(&expr, "agreement", 1)
            .expect("compile failed");

        // Samples away from boundaries to avoid Lt/Gt collapse edge
        for x in &[-2.0_f32, -0.5, 0.0, 0.5, 2.0] {
            let eval_result = TLExprEvaluator::new()
                .with_binding("dim_0", *x)
                .evaluate_bool(&expr)
                .expect("evaluator failed");

            let compile_result = compiled
                .evaluate(&Array1::from_vec(vec![*x]))
                .expect("compiled evaluate failed");

            assert_eq!(
                eval_result, compile_result,
                "evaluator and compiler disagree at x={x}: eval={eval_result}, compiled={compile_result}"
            );
        }
    }
}
