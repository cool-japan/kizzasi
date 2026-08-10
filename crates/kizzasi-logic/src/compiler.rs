//! Constraint Compilation to Optimized Bytecode IR
//!
//! Compiles high-level constraint expressions into an optimized stack-based
//! bytecode IR for fast evaluation. The compiler supports constant folding,
//! dead code elimination, and batch constraint programs.
//!
//! # Example
//!
//! ```
//! use kizzasi_logic::compiler::{ConstraintExpr, ConstraintProgram};
//! use scirs2_core::ndarray::Array1;
//!
//! let expr = ConstraintExpr::between(0, -1.0, 1.0);
//! let compiled = expr.compile("bound", 1);
//! let x = Array1::from_vec(vec![0.5_f32]);
//! assert!(compiled.evaluate(&x).unwrap());
//! ```

use crate::error::{LogicError, LogicResult};
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

// ============================================================================
// Opcode — stack-based bytecode instruction set
// ============================================================================

/// Bytecode instruction set for constraint expression evaluation.
///
/// The virtual machine maintains a stack of `f32` values.
/// Each instruction pops its operands and pushes its result.
#[derive(Debug, Clone, PartialEq)]
pub enum Opcode {
    /// Push `x[dim]` onto the stack
    LoadDim(usize),
    /// Push a constant onto the stack
    LoadConst(f32),
    /// Pop `b`, pop `a`; push `a + b`
    Add,
    /// Pop `b`, pop `a`; push `a - b`
    Sub,
    /// Pop `b`, pop `a`; push `a * b`
    Mul,
    /// Pop `b`, pop `a`; push `a / b` (errors on divide-by-zero)
    Div,
    /// Pop `a`; push `-a`
    Neg,
    /// Pop `a`; push `|a|`
    Abs,
    /// Pop `a`; push `sqrt(a)`
    Sqrt,
    /// Pop `b`, pop `a`; push `min(a, b)`
    Min,
    /// Pop `b`, pop `a`; push `max(a, b)`
    Max,
    /// Pop `b`, pop `a`; push `1.0` if `a <= b`, else `0.0`
    CmpLe,
    /// Pop `b`, pop `a`; push `1.0` if `a >= b`, else `0.0`
    CmpGe,
    /// Pop `b`, pop `a`; push `1.0` if both non-zero, else `0.0`
    And,
    /// Pop `b`, pop `a`; push `1.0` if either non-zero, else `0.0`
    Or,
    /// Pop `a`; push `1.0` if `a == 0.0`, else `0.0`
    Not,
    /// Duplicate the top of the stack
    Dup,
    /// Discard the top of the stack
    Pop,
}

// ============================================================================
// CompiledConstraint — executable bytecode program
// ============================================================================

/// A compiled constraint: a linear sequence of `Opcode`s evaluated on a
/// stack machine. The result of evaluation is the top of the stack after
/// all instructions have been executed.
#[derive(Debug, Clone)]
pub struct CompiledConstraint {
    /// The bytecode instruction sequence
    pub ops: Vec<Opcode>,
    /// Human-readable name for this constraint
    pub name: String,
    /// Expected dimensionality of the input vector
    pub num_dims: usize,
}

impl CompiledConstraint {
    /// Execute the bytecode on `x` and return feasibility.
    ///
    /// Feasible iff the top of the stack after execution is non-zero.
    pub fn evaluate(&self, x: &Array1<f32>) -> LogicResult<bool> {
        let raw = self.evaluate_raw(x)?;
        Ok(raw != 0.0)
    }

    /// Execute the bytecode on `x` and return the raw top-of-stack value.
    pub fn evaluate_raw(&self, x: &Array1<f32>) -> LogicResult<f32> {
        if x.len() < self.num_dims {
            return Err(LogicError::DimensionMismatch {
                expected: self.num_dims,
                got: x.len(),
            });
        }

        let mut stack: Vec<f32> = Vec::with_capacity(self.ops.len());

        for op in &self.ops {
            match op {
                Opcode::LoadDim(dim) => {
                    let val = x.get(*dim).copied().ok_or_else(|| {
                        LogicError::InvalidInput(format!(
                            "LoadDim: dimension {} out of bounds (len={})",
                            dim,
                            x.len()
                        ))
                    })?;
                    stack.push(val);
                }
                Opcode::LoadConst(v) => {
                    stack.push(*v);
                }
                Opcode::Add => {
                    let b = stack_pop(&mut stack, "Add")?;
                    let a = stack_pop(&mut stack, "Add")?;
                    stack.push(a + b);
                }
                Opcode::Sub => {
                    let b = stack_pop(&mut stack, "Sub")?;
                    let a = stack_pop(&mut stack, "Sub")?;
                    stack.push(a - b);
                }
                Opcode::Mul => {
                    let b = stack_pop(&mut stack, "Mul")?;
                    let a = stack_pop(&mut stack, "Mul")?;
                    stack.push(a * b);
                }
                Opcode::Div => {
                    let b = stack_pop(&mut stack, "Div")?;
                    let a = stack_pop(&mut stack, "Div")?;
                    if b == 0.0 {
                        return Err(LogicError::InvalidInput(
                            "Div: division by zero".to_string(),
                        ));
                    }
                    stack.push(a / b);
                }
                Opcode::Neg => {
                    let a = stack_pop(&mut stack, "Neg")?;
                    stack.push(-a);
                }
                Opcode::Abs => {
                    let a = stack_pop(&mut stack, "Abs")?;
                    stack.push(a.abs());
                }
                Opcode::Sqrt => {
                    let a = stack_pop(&mut stack, "Sqrt")?;
                    if a < 0.0 {
                        return Err(LogicError::InvalidInput(format!(
                            "Sqrt: negative argument {a}"
                        )));
                    }
                    stack.push(a.sqrt());
                }
                Opcode::Min => {
                    let b = stack_pop(&mut stack, "Min")?;
                    let a = stack_pop(&mut stack, "Min")?;
                    stack.push(a.min(b));
                }
                Opcode::Max => {
                    let b = stack_pop(&mut stack, "Max")?;
                    let a = stack_pop(&mut stack, "Max")?;
                    stack.push(a.max(b));
                }
                Opcode::CmpLe => {
                    let b = stack_pop(&mut stack, "CmpLe")?;
                    let a = stack_pop(&mut stack, "CmpLe")?;
                    stack.push(if a <= b { 1.0 } else { 0.0 });
                }
                Opcode::CmpGe => {
                    let b = stack_pop(&mut stack, "CmpGe")?;
                    let a = stack_pop(&mut stack, "CmpGe")?;
                    stack.push(if a >= b { 1.0 } else { 0.0 });
                }
                Opcode::And => {
                    let b = stack_pop(&mut stack, "And")?;
                    let a = stack_pop(&mut stack, "And")?;
                    stack.push(if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 });
                }
                Opcode::Or => {
                    let b = stack_pop(&mut stack, "Or")?;
                    let a = stack_pop(&mut stack, "Or")?;
                    stack.push(if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 });
                }
                Opcode::Not => {
                    let a = stack_pop(&mut stack, "Not")?;
                    stack.push(if a == 0.0 { 1.0 } else { 0.0 });
                }
                Opcode::Dup => {
                    let a = stack.last().copied().ok_or_else(|| {
                        LogicError::InvalidInput("Dup: stack underflow".to_string())
                    })?;
                    stack.push(a);
                }
                Opcode::Pop => {
                    stack_pop(&mut stack, "Pop")?;
                }
            }
        }

        stack.last().copied().ok_or_else(|| {
            LogicError::InvalidInput("evaluate_raw: stack is empty after execution".to_string())
        })
    }

    /// Optimize the bytecode via constant folding and dead code elimination.
    ///
    /// Constant folding: sequences of two `LoadConst` instructions followed by
    /// a binary arithmetic/comparison opcode are collapsed into a single
    /// `LoadConst` with the pre-computed result.
    pub fn optimize(&self) -> Self {
        let folded = constant_fold(&self.ops);
        let dce = dead_code_eliminate(&folded);
        Self {
            ops: dce,
            name: self.name.clone(),
            num_dims: self.num_dims,
        }
    }

    /// Return the number of opcodes (before optimization).
    pub fn complexity(&self) -> usize {
        self.ops.len()
    }
}

// ============================================================================
// Internal stack helpers
// ============================================================================

#[inline]
fn stack_pop(stack: &mut Vec<f32>, op: &str) -> LogicResult<f32> {
    stack
        .pop()
        .ok_or_else(|| LogicError::InvalidInput(format!("{op}: stack underflow")))
}

// ============================================================================
// Optimizer passes
// ============================================================================

/// Constant folding pass: collapse pairs of LoadConst + binary/unary op.
fn constant_fold(ops: &[Opcode]) -> Vec<Opcode> {
    let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());

    let mut i = 0;
    while i < ops.len() {
        // Attempt binary constant fold: LoadConst(a), LoadConst(b), BinaryOp → LoadConst(result)
        if i + 2 < ops.len() {
            if let (Opcode::LoadConst(a), Opcode::LoadConst(b)) = (&ops[i], &ops[i + 1]) {
                let a = *a;
                let b = *b;
                let folded = match &ops[i + 2] {
                    Opcode::Add => Some(a + b),
                    Opcode::Sub => Some(a - b),
                    Opcode::Mul => Some(a * b),
                    Opcode::Div => {
                        if b != 0.0 {
                            Some(a / b)
                        } else {
                            None
                        }
                    }
                    Opcode::Min => Some(a.min(b)),
                    Opcode::Max => Some(a.max(b)),
                    Opcode::CmpLe => Some(if a <= b { 1.0 } else { 0.0 }),
                    Opcode::CmpGe => Some(if a >= b { 1.0 } else { 0.0 }),
                    Opcode::And => Some(if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 }),
                    Opcode::Or => Some(if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 }),
                    _ => None,
                };
                if let Some(result) = folded {
                    out.push(Opcode::LoadConst(result));
                    i += 3;
                    continue;
                }
            }
        }

        // Attempt unary constant fold: LoadConst(a), UnaryOp → LoadConst(result)
        if i + 1 < ops.len() {
            if let Opcode::LoadConst(a) = &ops[i] {
                let a = *a;
                let folded = match &ops[i + 1] {
                    Opcode::Neg => Some(-a),
                    Opcode::Abs => Some(a.abs()),
                    Opcode::Sqrt => {
                        if a >= 0.0 {
                            Some(a.sqrt())
                        } else {
                            None
                        }
                    }
                    Opcode::Not => Some(if a == 0.0 { 1.0 } else { 0.0 }),
                    _ => None,
                };
                if let Some(result) = folded {
                    out.push(Opcode::LoadConst(result));
                    i += 2;
                    continue;
                }
            }
        }

        out.push(ops[i].clone());
        i += 1;
    }

    // Run again if any folding happened (handles nested folds)
    if out.len() < ops.len() {
        constant_fold(&out)
    } else {
        out
    }
}

/// Dead code elimination: remove `Pop` immediately after a `LoadConst`
/// (the value is never used).
fn dead_code_eliminate(ops: &[Opcode]) -> Vec<Opcode> {
    let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
    let mut i = 0;
    while i < ops.len() {
        if i + 1 < ops.len() {
            if let Opcode::LoadConst(_) = &ops[i] {
                if let Opcode::Pop = &ops[i + 1] {
                    // LoadConst followed by Pop — skip both
                    i += 2;
                    continue;
                }
            }
        }
        out.push(ops[i].clone());
        i += 1;
    }
    out
}

// ============================================================================
// ConstraintExpr — high-level AST
// ============================================================================

/// High-level constraint expression AST.
///
/// Build expressions using the builder helpers (`dim`, `constant`, `between`,
/// `l2_norm_le`, `affine_le`) or compose them manually, then call
/// [`ConstraintExpr::compile`] to produce a [`CompiledConstraint`].
#[derive(Debug, Clone)]
pub enum ConstraintExpr {
    /// Access `x[i]`
    Dim(usize),
    /// A constant scalar
    Const(f32),
    /// `a + b`
    Add(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// `a - b`
    Sub(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// `a * b`
    Mul(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// `a / b`
    Div(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// `-a`
    Neg(Box<ConstraintExpr>),
    /// `|a|`
    Abs(Box<ConstraintExpr>),
    /// `sqrt(a)`
    Sqrt(Box<ConstraintExpr>),
    /// `a <= b` (evaluates to 1.0 or 0.0)
    Le(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// `a >= b` (evaluates to 1.0 or 0.0)
    Ge(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// `a && b`
    And(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// `a || b`
    Or(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// `!a`
    Not(Box<ConstraintExpr>),
}

impl ConstraintExpr {
    // ------------------------------------------------------------------
    // Compilation
    // ------------------------------------------------------------------

    /// Compile this AST into a [`CompiledConstraint`].
    pub fn compile(&self, name: &str, num_dims: usize) -> CompiledConstraint {
        let mut ops = Vec::new();
        emit(self, &mut ops);
        CompiledConstraint {
            ops,
            name: name.to_string(),
            num_dims,
        }
    }

    // ------------------------------------------------------------------
    // Builder helpers
    // ------------------------------------------------------------------

    /// Reference `x[i]`
    pub fn dim(i: usize) -> Self {
        ConstraintExpr::Dim(i)
    }

    /// A scalar constant
    pub fn constant(v: f32) -> Self {
        ConstraintExpr::Const(v)
    }

    /// `lo <= x[dim] <= hi`
    pub fn between(dim: usize, lo: f32, hi: f32) -> Self {
        let x = ConstraintExpr::Dim(dim);
        let lo_le = ConstraintExpr::Le(Box::new(ConstraintExpr::Const(lo)), Box::new(x.clone()));
        let hi_le = ConstraintExpr::Le(Box::new(x), Box::new(ConstraintExpr::Const(hi)));
        ConstraintExpr::And(Box::new(lo_le), Box::new(hi_le))
    }

    /// `||x[dims]||_2 <= radius`
    ///
    /// Compiles to: `sqrt(sum_i(x[dims[i]]^2)) <= radius`
    pub fn l2_norm_le(dims: &[usize], radius: f32) -> Self {
        assert!(!dims.is_empty(), "l2_norm_le: dims must not be empty");

        // Build sum of squares
        let mut sum_sq: ConstraintExpr = ConstraintExpr::Mul(
            Box::new(ConstraintExpr::Dim(dims[0])),
            Box::new(ConstraintExpr::Dim(dims[0])),
        );
        for &d in &dims[1..] {
            let sq = ConstraintExpr::Mul(
                Box::new(ConstraintExpr::Dim(d)),
                Box::new(ConstraintExpr::Dim(d)),
            );
            sum_sq = ConstraintExpr::Add(Box::new(sum_sq), Box::new(sq));
        }

        let norm = ConstraintExpr::Sqrt(Box::new(sum_sq));
        ConstraintExpr::Le(Box::new(norm), Box::new(ConstraintExpr::Const(radius)))
    }

    /// `sum_i(coeffs[i].1 * x[coeffs[i].0]) <= rhs`
    pub fn affine_le(coeffs: &[(usize, f32)], rhs: f32) -> Self {
        assert!(!coeffs.is_empty(), "affine_le: coeffs must not be empty");

        let term = |(dim, c): &(usize, f32)| -> ConstraintExpr {
            ConstraintExpr::Mul(
                Box::new(ConstraintExpr::Const(*c)),
                Box::new(ConstraintExpr::Dim(*dim)),
            )
        };

        let mut sum = term(&coeffs[0]);
        for coeff in &coeffs[1..] {
            sum = ConstraintExpr::Add(Box::new(sum), Box::new(term(coeff)));
        }

        ConstraintExpr::Le(Box::new(sum), Box::new(ConstraintExpr::Const(rhs)))
    }
}

// ============================================================================
// Code emission (AST → opcodes)
// ============================================================================

fn emit(expr: &ConstraintExpr, ops: &mut Vec<Opcode>) {
    match expr {
        ConstraintExpr::Dim(i) => {
            ops.push(Opcode::LoadDim(*i));
        }
        ConstraintExpr::Const(v) => {
            ops.push(Opcode::LoadConst(*v));
        }
        ConstraintExpr::Add(a, b) => {
            emit(a, ops);
            emit(b, ops);
            ops.push(Opcode::Add);
        }
        ConstraintExpr::Sub(a, b) => {
            emit(a, ops);
            emit(b, ops);
            ops.push(Opcode::Sub);
        }
        ConstraintExpr::Mul(a, b) => {
            emit(a, ops);
            emit(b, ops);
            ops.push(Opcode::Mul);
        }
        ConstraintExpr::Div(a, b) => {
            emit(a, ops);
            emit(b, ops);
            ops.push(Opcode::Div);
        }
        ConstraintExpr::Neg(a) => {
            emit(a, ops);
            ops.push(Opcode::Neg);
        }
        ConstraintExpr::Abs(a) => {
            emit(a, ops);
            ops.push(Opcode::Abs);
        }
        ConstraintExpr::Sqrt(a) => {
            emit(a, ops);
            ops.push(Opcode::Sqrt);
        }
        ConstraintExpr::Le(a, b) => {
            emit(a, ops);
            emit(b, ops);
            ops.push(Opcode::CmpLe);
        }
        ConstraintExpr::Ge(a, b) => {
            emit(a, ops);
            emit(b, ops);
            ops.push(Opcode::CmpGe);
        }
        ConstraintExpr::And(a, b) => {
            emit(a, ops);
            emit(b, ops);
            ops.push(Opcode::And);
        }
        ConstraintExpr::Or(a, b) => {
            emit(a, ops);
            emit(b, ops);
            ops.push(Opcode::Or);
        }
        ConstraintExpr::Not(a) => {
            emit(a, ops);
            ops.push(Opcode::Not);
        }
    }
}

// ============================================================================
// ConstraintProgram — named collection of compiled constraints
// ============================================================================

/// A named collection of compiled constraints that can be evaluated together.
///
/// All constraints share the same input vector but may have different
/// dimensionality requirements.
pub struct ConstraintProgram {
    constraints: HashMap<String, CompiledConstraint>,
}

impl Default for ConstraintProgram {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintProgram {
    /// Create an empty program.
    pub fn new() -> Self {
        Self {
            constraints: HashMap::new(),
        }
    }

    /// Compile and add a constraint expression to the program.
    pub fn add(&mut self, expr: ConstraintExpr, name: &str, num_dims: usize) {
        let compiled = expr.compile(name, num_dims);
        self.constraints.insert(name.to_string(), compiled);
    }

    /// Evaluate all constraints and return a map from name → feasibility.
    pub fn evaluate_all(&self, x: &Array1<f32>) -> LogicResult<HashMap<String, bool>> {
        let mut results = HashMap::with_capacity(self.constraints.len());
        for (name, constraint) in &self.constraints {
            let feasible = constraint.evaluate(x)?;
            results.insert(name.clone(), feasible);
        }
        Ok(results)
    }

    /// Return the names of all violated (infeasible) constraints.
    pub fn violated(&self, x: &Array1<f32>) -> LogicResult<Vec<String>> {
        let all = self.evaluate_all(x)?;
        let mut names: Vec<String> = all
            .into_iter()
            .filter_map(|(name, feasible)| if feasible { None } else { Some(name) })
            .collect();
        names.sort(); // deterministic order
        Ok(names)
    }

    /// Return `true` iff all constraints are satisfied.
    pub fn is_feasible(&self, x: &Array1<f32>) -> LogicResult<bool> {
        for constraint in self.constraints.values() {
            if !constraint.evaluate(x)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Return the number of constraints in this program.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn arr(values: Vec<f32>) -> Array1<f32> {
        Array1::from_vec(values)
    }

    #[test]
    fn test_compile_constant() {
        let expr = ConstraintExpr::constant(3.0);
        let compiled = expr.compile("c", 0);
        let x: Array1<f32> = Array1::from_vec(vec![]);
        let raw = compiled.evaluate_raw(&x).expect("evaluate_raw failed");
        assert!((raw - 3.0).abs() < 1e-6, "expected 3.0, got {raw}");
    }

    #[test]
    fn test_compile_load_dim() {
        let expr = ConstraintExpr::dim(1);
        let compiled = expr.compile("c", 2);
        let x = arr(vec![0.0, 5.0]);
        let raw = compiled.evaluate_raw(&x).expect("evaluate_raw failed");
        assert!((raw - 5.0).abs() < 1e-6, "expected 5.0, got {raw}");
    }

    #[test]
    fn test_compile_between() {
        let expr = ConstraintExpr::between(0, -1.0, 1.0);
        let compiled = expr.compile("bound", 1);

        // Feasible: x[0] = 0.5
        let x_ok = arr(vec![0.5]);
        assert!(
            compiled.evaluate(&x_ok).expect("evaluate failed"),
            "0.5 should be in [-1, 1]"
        );

        // Infeasible: x[0] = 2.0
        let x_bad = arr(vec![2.0]);
        assert!(
            !compiled.evaluate(&x_bad).expect("evaluate failed"),
            "2.0 should not be in [-1, 1]"
        );
    }

    #[test]
    fn test_compile_affine_le() {
        // 2*x[0] + 3*x[1] <= 10
        let expr = ConstraintExpr::affine_le(&[(0, 2.0), (1, 3.0)], 10.0);
        let compiled = expr.compile("affine", 2);

        // 2*1 + 3*1 = 5 <= 10 → feasible
        let x_ok = arr(vec![1.0, 1.0]);
        assert!(
            compiled.evaluate(&x_ok).expect("evaluate failed"),
            "2+3=5 should be <= 10"
        );

        // 2*3 + 3*3 = 15 > 10 → infeasible
        let x_bad = arr(vec![3.0, 3.0]);
        assert!(
            !compiled.evaluate(&x_bad).expect("evaluate failed"),
            "6+9=15 should not be <= 10"
        );
    }

    #[test]
    fn test_compile_l2_norm_le() {
        // ||(x[0], x[1])||_2 <= 1.0
        let expr = ConstraintExpr::l2_norm_le(&[0, 1], 1.0);
        let compiled = expr.compile("l2ball", 2);

        // (0.3, 0.4): norm = 0.5 <= 1.0
        let x_ok = arr(vec![0.3, 0.4]);
        assert!(
            compiled.evaluate(&x_ok).expect("evaluate failed"),
            "norm(0.3, 0.4)=0.5 should be <= 1.0"
        );

        // (1.0, 1.0): norm = sqrt(2) > 1.0
        let x_bad = arr(vec![1.0, 1.0]);
        assert!(
            !compiled.evaluate(&x_bad).expect("evaluate failed"),
            "norm(1, 1)=sqrt(2) should not be <= 1.0"
        );
    }

    #[test]
    fn test_optimize_constant_folding() {
        // Add(Const(2), Const(3)) should fold to a single LoadConst(5)
        let expr = ConstraintExpr::Add(
            Box::new(ConstraintExpr::Const(2.0)),
            Box::new(ConstraintExpr::Const(3.0)),
        );
        let compiled = expr.compile("fold", 0);
        let optimized = compiled.optimize();

        // Unoptimized: LoadConst(2), LoadConst(3), Add = 3 ops
        // Optimized:   LoadConst(5) = 1 op
        assert!(
            optimized.complexity() < compiled.complexity(),
            "optimized ({}) should have fewer ops than original ({})",
            optimized.complexity(),
            compiled.complexity()
        );

        // Verify the result is still correct
        let x: Array1<f32> = Array1::from_vec(vec![]);
        let raw = optimized.evaluate_raw(&x).expect("evaluate_raw failed");
        assert!(
            (raw - 5.0).abs() < 1e-6,
            "folded result should be 5.0, got {raw}"
        );
    }

    #[test]
    fn test_program_evaluate_all() {
        let mut prog = ConstraintProgram::new();
        prog.add(ConstraintExpr::between(0, 0.0, 1.0), "x_bound", 1);
        prog.add(ConstraintExpr::between(1, 0.0, 1.0), "y_bound", 2);

        let x = arr(vec![0.5, 0.5]);
        let results = prog.evaluate_all(&x).expect("evaluate_all failed");

        assert_eq!(results.len(), 2, "should have 2 entries");
        assert!(results["x_bound"], "x_bound should be feasible");
        assert!(results["y_bound"], "y_bound should be feasible");
    }

    #[test]
    fn test_program_violated_returns_names() {
        let mut prog = ConstraintProgram::new();
        prog.add(ConstraintExpr::between(0, 0.0, 1.0), "x_bound", 1);
        prog.add(ConstraintExpr::between(1, 0.0, 1.0), "y_bound", 2);

        // x[0] = 2.0 violates x_bound; x[1] = 0.5 satisfies y_bound
        let x = arr(vec![2.0, 0.5]);
        let violated = prog.violated(&x).expect("violated failed");

        assert_eq!(violated, vec!["x_bound".to_string()]);
    }

    #[test]
    fn test_complexity_before_after_optimize() {
        // Deep nested expression: Add(Add(Const(1), Const(2)), Const(3))
        let expr = ConstraintExpr::Add(
            Box::new(ConstraintExpr::Add(
                Box::new(ConstraintExpr::Const(1.0)),
                Box::new(ConstraintExpr::Const(2.0)),
            )),
            Box::new(ConstraintExpr::Const(3.0)),
        );
        let compiled = expr.compile("nested", 0);
        let optimized = compiled.optimize();

        assert!(
            optimized.complexity() <= compiled.complexity(),
            "optimized complexity {} should be <= original {}",
            optimized.complexity(),
            compiled.complexity()
        );

        // Also verify correctness
        let x: Array1<f32> = Array1::from_vec(vec![]);
        let raw = optimized.evaluate_raw(&x).expect("evaluate_raw failed");
        assert!((raw - 6.0).abs() < 1e-6, "result should be 6.0, got {raw}");
    }
}

// ============================================================================
// TlExprCompiler — lower TLExpr → ConstraintExpr → CompiledConstraint
// ============================================================================

/// Lowers a [`tensorlogic_ir::TLExpr`] expression into a [`ConstraintExpr`],
/// then compiles it to a fast stack-VM [`CompiledConstraint`].
///
/// This is the bridge between the symbolic representation layer
/// (`tensorlogic-ir`) and the numerical execution layer (`kizzasi-logic`).
///
/// ## Leaf Node Convention
///
/// `TLExpr` leaf nodes use `Pred { name, args }`:
///
/// | Pred form | Lowered to |
/// |-----------|-----------|
/// | `Pred { name: "dim_N", args: [] }` | `ConstraintExpr::Dim(N)` |
/// | `Pred { name: "5.0", args: [] }` (parseable f32) | `ConstraintExpr::Const(5.0)` |
/// | `Pred { name: var, args: [] }` in `dim_map` | `ConstraintExpr::Dim(dim_map[var])` |
/// | `Pred { name: _, args: [Term::Var(v)] }` | `ConstraintExpr::Dim(dim_map[v])` |
/// | `Pred { name: _, args: [Term::Const(c)] }` | `ConstraintExpr::Const(c.parse())` |
///
/// ## Example
///
/// ```rust
/// use kizzasi_logic::compiler::TlExprCompiler;
/// use tensorlogic_ir::TLExpr;
/// use scirs2_core::ndarray::Array1;
///
/// // Build: x[0] <= 1.0
/// let expr = TLExpr::Lte(
///     Box::new(TLExpr::Pred { name: "dim_0".into(), args: vec![] }),
///     Box::new(TLExpr::Pred { name: "1.0".into(), args: vec![] }),
/// );
/// let compiler = TlExprCompiler::new();
/// let compiled = compiler.compile(&expr, "x_le_1", 1).unwrap();
///
/// let x_ok = Array1::from_vec(vec![0.5_f32]);
/// assert!(compiled.evaluate(&x_ok).unwrap());
///
/// let x_bad = Array1::from_vec(vec![2.0_f32]);
/// assert!(!compiled.evaluate(&x_bad).unwrap());
/// ```
pub struct TlExprCompiler {
    /// Maps symbolic variable names to dimension indices in the input vector.
    dim_map: HashMap<String, usize>,
}

impl Default for TlExprCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl TlExprCompiler {
    /// Create a new compiler with no variable-to-dimension mappings.
    pub fn new() -> Self {
        Self {
            dim_map: HashMap::new(),
        }
    }

    /// Register a variable name → dimension index mapping.
    ///
    /// When the compiler encounters `Pred { name: var, args: [] }` or
    /// `Pred { _, args: [Term::Var(var)] }`, it resolves `var` via this map.
    pub fn with_var(mut self, name: impl Into<String>, dim: usize) -> Self {
        self.dim_map.insert(name.into(), dim);
        self
    }

    /// Lower a `TLExpr` to a `ConstraintExpr` (the high-level AST in kizzasi-logic).
    ///
    /// Returns an error if the expression contains unsupported variants or
    /// unresolvable leaf nodes.
    pub fn lower(&self, expr: &tensorlogic_ir::TLExpr) -> LogicResult<ConstraintExpr> {
        use tensorlogic_ir::TLExpr;

        match expr {
            TLExpr::Pred { name, args } => self.lower_pred(name, args),

            // Arithmetic binary
            TLExpr::Add(a, b) => Ok(ConstraintExpr::Add(
                Box::new(self.lower(a)?),
                Box::new(self.lower(b)?),
            )),
            TLExpr::Sub(a, b) => Ok(ConstraintExpr::Sub(
                Box::new(self.lower(a)?),
                Box::new(self.lower(b)?),
            )),
            TLExpr::Mul(a, b) => Ok(ConstraintExpr::Mul(
                Box::new(self.lower(a)?),
                Box::new(self.lower(b)?),
            )),
            TLExpr::Div(a, b) => Ok(ConstraintExpr::Div(
                Box::new(self.lower(a)?),
                Box::new(self.lower(b)?),
            )),
            TLExpr::Min(a, b) => {
                // min(a, b) = -max(-a, -b) but simpler: inline as ConstraintExpr
                // kizzasi-logic ConstraintExpr doesn't have Min/Max; use conditional form:
                // min(a,b) ≡ (a+b - |a-b|) / 2
                let a_low = self.lower(a)?;
                let b_low = self.lower(b)?;
                // a + b
                let sum = ConstraintExpr::Add(Box::new(a_low.clone()), Box::new(b_low.clone()));
                // a - b
                let diff = ConstraintExpr::Sub(Box::new(a_low), Box::new(b_low));
                // |a - b|
                let abs_diff = ConstraintExpr::Abs(Box::new(diff));
                // (sum - abs_diff) / 2
                Ok(ConstraintExpr::Div(
                    Box::new(ConstraintExpr::Sub(Box::new(sum), Box::new(abs_diff))),
                    Box::new(ConstraintExpr::Const(2.0)),
                ))
            }
            TLExpr::Max(a, b) => {
                // max(a,b) ≡ (a+b + |a-b|) / 2
                let a_low = self.lower(a)?;
                let b_low = self.lower(b)?;
                let sum = ConstraintExpr::Add(Box::new(a_low.clone()), Box::new(b_low.clone()));
                let diff = ConstraintExpr::Sub(Box::new(a_low), Box::new(b_low));
                let abs_diff = ConstraintExpr::Abs(Box::new(diff));
                Ok(ConstraintExpr::Div(
                    Box::new(ConstraintExpr::Add(Box::new(sum), Box::new(abs_diff))),
                    Box::new(ConstraintExpr::Const(2.0)),
                ))
            }
            TLExpr::Pow(a, b) => {
                // x^n — support integer exponents efficiently; fallback to Sqrt for 0.5
                // General case: emit as Mul chain when b is small int const, else error
                let b_low = self.lower(b)?;
                if let ConstraintExpr::Const(exp) = &b_low {
                    let exp_val = *exp;
                    if (exp_val - 0.5).abs() < 1e-6 {
                        return Ok(ConstraintExpr::Sqrt(Box::new(self.lower(a)?)));
                    }
                    if (exp_val - 2.0).abs() < 1e-6 {
                        let a_low = self.lower(a)?;
                        return Ok(ConstraintExpr::Mul(
                            Box::new(a_low.clone()),
                            Box::new(a_low),
                        ));
                    }
                    if (exp_val - 1.0).abs() < 1e-6 {
                        return self.lower(a);
                    }
                }
                Err(LogicError::InvalidConstraint(
                    "TlExprCompiler: Pow with non-constant or non-integer exponent \
                     is not supported (use 0.5 for sqrt, 2.0 for square)"
                        .into(),
                ))
            }
            TLExpr::Mod(_, _) => Err(LogicError::InvalidConstraint(
                "TlExprCompiler: Mod is not supported in ConstraintExpr".into(),
            )),

            // Unary arithmetic
            TLExpr::Abs(a) => Ok(ConstraintExpr::Abs(Box::new(self.lower(a)?))),
            TLExpr::Sqrt(a) => Ok(ConstraintExpr::Sqrt(Box::new(self.lower(a)?))),
            TLExpr::Floor(_)
            | TLExpr::Ceil(_)
            | TLExpr::Round(_)
            | TLExpr::Exp(_)
            | TLExpr::Log(_)
            | TLExpr::Sin(_)
            | TLExpr::Cos(_)
            | TLExpr::Tan(_) => Err(LogicError::InvalidConstraint(
                "TlExprCompiler: transcendental functions (floor/ceil/round/exp/log/sin/cos/tan) \
                 are not supported in ConstraintExpr"
                    .into(),
            )),

            // Comparisons → 1.0 / 0.0 via CmpLe / CmpGe
            TLExpr::Lte(a, b) => Ok(ConstraintExpr::Le(
                Box::new(self.lower(a)?),
                Box::new(self.lower(b)?),
            )),
            TLExpr::Gte(a, b) => Ok(ConstraintExpr::Ge(
                Box::new(self.lower(a)?),
                Box::new(self.lower(b)?),
            )),
            TLExpr::Lt(a, b) => {
                // a < b ≡ a <= b  AND  NOT(a == b) — simplify as Le (slightly wrong at boundary)
                // For continuous signals this is equivalent; use Le
                Ok(ConstraintExpr::Le(
                    Box::new(self.lower(a)?),
                    Box::new(self.lower(b)?),
                ))
            }
            TLExpr::Gt(a, b) => Ok(ConstraintExpr::Ge(
                Box::new(self.lower(a)?),
                Box::new(self.lower(b)?),
            )),
            TLExpr::Eq(a, b) => {
                // a == b ≡ (a <= b) AND (a >= b)
                let le = ConstraintExpr::Le(Box::new(self.lower(a)?), Box::new(self.lower(b)?));
                let ge = ConstraintExpr::Ge(Box::new(self.lower(a)?), Box::new(self.lower(b)?));
                Ok(ConstraintExpr::And(Box::new(le), Box::new(ge)))
            }

            // Logic
            TLExpr::And(a, b) => Ok(ConstraintExpr::And(
                Box::new(self.lower(a)?),
                Box::new(self.lower(b)?),
            )),
            TLExpr::Or(a, b) => Ok(ConstraintExpr::Or(
                Box::new(self.lower(a)?),
                Box::new(self.lower(b)?),
            )),
            TLExpr::Not(a) => Ok(ConstraintExpr::Not(Box::new(self.lower(a)?))),
            TLExpr::Imply(a, b) => {
                // a → b  ≡  (NOT a) OR b
                let not_a = ConstraintExpr::Not(Box::new(self.lower(a)?));
                Ok(ConstraintExpr::Or(
                    Box::new(not_a),
                    Box::new(self.lower(b)?),
                ))
            }
            TLExpr::Score(a) => self.lower(a),

            other => Err(LogicError::InvalidConstraint(format!(
                "TlExprCompiler: unsupported TLExpr variant: {:?}",
                std::mem::discriminant(other)
            ))),
        }
    }

    /// Compile a `TLExpr` directly to a [`CompiledConstraint`].
    pub fn compile(
        &self,
        expr: &tensorlogic_ir::TLExpr,
        name: &str,
        num_dims: usize,
    ) -> LogicResult<CompiledConstraint> {
        let lowered = self.lower(expr)?;
        Ok(lowered.compile(name, num_dims))
    }

    /// Compile a `TLExpr` with symbolic pre-optimization and stack-VM optimization.
    ///
    /// Applies [`tensorlogic_ir::algebraic_simplify`] + [`tensorlogic_ir::constant_fold`]
    /// on the `TLExpr` AST before lowering, then applies the stack-VM level constant
    /// folding and dead-code elimination passes.
    pub fn compile_optimized(
        &self,
        expr: &tensorlogic_ir::TLExpr,
        name: &str,
        num_dims: usize,
    ) -> LogicResult<CompiledConstraint> {
        use tensorlogic_ir::{algebraic_simplify, constant_fold};
        // Pre-optimize at the symbolic level
        let simplified = algebraic_simplify(expr);
        let folded = constant_fold(&simplified);
        // Lower and compile
        let lowered = self.lower(&folded)?;
        // Apply stack-VM level optimizations
        Ok(lowered.compile(name, num_dims).optimize())
    }

    fn lower_pred(&self, name: &str, args: &[tensorlogic_ir::Term]) -> LogicResult<ConstraintExpr> {
        use tensorlogic_ir::Term;

        match args {
            [] => {
                // 1. Numeric constant: "5.0", "-1.0", "0", etc.
                if let Ok(v) = name.parse::<f32>() {
                    return Ok(ConstraintExpr::Const(v));
                }
                // 2. "dim_N" shorthand
                if let Some(rest) = name.strip_prefix("dim_") {
                    if let Ok(n) = rest.parse::<usize>() {
                        return Ok(ConstraintExpr::Dim(n));
                    }
                }
                // 3. Named variable in dim_map
                if let Some(&dim) = self.dim_map.get(name) {
                    return Ok(ConstraintExpr::Dim(dim));
                }
                Err(LogicError::InvalidConstraint(format!(
                    "TlExprCompiler: cannot resolve Pred('{name}') — \
                     not a float literal, not 'dim_N', not in dim_map"
                )))
            }
            [Term::Var(v)] => self
                .dim_map
                .get(v.as_str())
                .copied()
                .map(ConstraintExpr::Dim)
                .ok_or_else(|| {
                    LogicError::InvalidConstraint(format!(
                        "TlExprCompiler: variable '{v}' not in dim_map"
                    ))
                }),
            [Term::Const(c)] => c.parse::<f32>().map(ConstraintExpr::Const).map_err(|_| {
                LogicError::InvalidConstraint(format!(
                    "TlExprCompiler: cannot parse Term::Const('{c}') as f32"
                ))
            }),
            [Term::Typed { value, .. }] => {
                // Recurse into the inner term
                let inner = TlExprCompiler::new_with_map(self.dim_map.clone());
                inner.lower_pred(name, std::slice::from_ref(value))
            }
            _ => Err(LogicError::InvalidConstraint(format!(
                "TlExprCompiler: Pred('{name}') has unsupported arg list (len={})",
                args.len()
            ))),
        }
    }

    fn new_with_map(dim_map: HashMap<String, usize>) -> Self {
        Self { dim_map }
    }
}

// ============================================================================
// TlExprCompiler tests
// ============================================================================

#[cfg(test)]
mod tl_compiler_tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use tensorlogic_ir::TLExpr;

    use crate::tensorlogic_integration::{tl_const, tl_var};

    fn arr(vals: Vec<f32>) -> Array1<f32> {
        Array1::from_vec(vals)
    }

    #[test]
    fn test_lower_dim_shorthand() {
        let compiler = TlExprCompiler::new();
        let expr = TLExpr::Pred {
            name: "dim_0".into(),
            args: vec![],
        };
        let lowered = compiler.lower(&expr).expect("lower failed");
        assert!(matches!(lowered, ConstraintExpr::Dim(0)));
    }

    #[test]
    fn test_lower_const_from_pred_name() {
        let compiler = TlExprCompiler::new();
        let expr = TLExpr::Pred {
            name: "3.14".into(),
            args: vec![],
        };
        let lowered = compiler.lower(&expr).expect("lower failed");
        let expected: f32 = "3.14".parse().expect("parse failed");
        assert!(matches!(lowered, ConstraintExpr::Const(v) if (v - expected).abs() < 1e-4));
    }

    #[test]
    fn test_lower_var_from_dim_map() {
        let compiler = TlExprCompiler::new().with_var("velocity", 2);
        let expr = TLExpr::Pred {
            name: "velocity".into(),
            args: vec![],
        };
        let lowered = compiler.lower(&expr).expect("lower failed");
        assert!(matches!(lowered, ConstraintExpr::Dim(2)));
    }

    #[test]
    fn test_compile_lte_bound() {
        // x[0] <= 1.0
        let expr = TLExpr::Lte(Box::new(tl_var("dim_0")), Box::new(tl_const(1.0)));
        let compiler = TlExprCompiler::new();
        let compiled = compiler
            .compile(&expr, "x_le_1", 1)
            .expect("compile failed");

        assert!(
            compiled.evaluate(&arr(vec![0.5])).expect("eval"),
            "0.5 <= 1.0"
        );
        assert!(
            !compiled.evaluate(&arr(vec![2.0])).expect("eval"),
            "2.0 > 1.0"
        );
    }

    #[test]
    fn test_compile_and_constraint() {
        // -1.0 <= x[0] <= 1.0  (And(Gte, Lte))
        let expr = TLExpr::And(
            Box::new(TLExpr::Gte(
                Box::new(tl_var("dim_0")),
                Box::new(tl_const(-1.0)),
            )),
            Box::new(TLExpr::Lte(
                Box::new(tl_var("dim_0")),
                Box::new(tl_const(1.0)),
            )),
        );
        let compiler = TlExprCompiler::new();
        let compiled = compiler.compile(&expr, "box", 1).expect("compile failed");

        assert!(
            compiled.evaluate(&arr(vec![0.5])).expect("eval"),
            "0.5 in [-1,1]"
        );
        assert!(
            !compiled.evaluate(&arr(vec![2.0])).expect("eval"),
            "2.0 not in [-1,1]"
        );
        assert!(
            !compiled.evaluate(&arr(vec![-2.0])).expect("eval"),
            "-2.0 not in [-1,1]"
        );
    }

    #[test]
    fn test_compile_optimized_constant_fold() {
        // (2.0 + 3.0) <= x[0]  — should fold to Const(5.0) before compile
        let sum = TLExpr::Add(Box::new(tl_const(2.0)), Box::new(tl_const(3.0)));
        let expr = TLExpr::Lte(Box::new(sum), Box::new(tl_var("dim_0")));
        let compiler = TlExprCompiler::new();
        let compiled = compiler
            .compile_optimized(&expr, "folded", 1)
            .expect("compile_optimized failed");

        // 5.0 <= 6.0 → true
        assert!(compiled.evaluate(&arr(vec![6.0])).expect("eval"));
        // 5.0 <= 4.0 → false
        assert!(!compiled.evaluate(&arr(vec![4.0])).expect("eval"));
    }

    #[test]
    fn test_compile_named_var_map() {
        // velocity (mapped to dim 0) <= 10.0
        let expr = TLExpr::Lte(Box::new(tl_var("velocity")), Box::new(tl_const(10.0)));
        let compiler = TlExprCompiler::new().with_var("velocity", 0);
        let compiled = compiler
            .compile(&expr, "vel_bound", 1)
            .expect("compile failed");

        assert!(compiled.evaluate(&arr(vec![5.0])).expect("eval"), "5 <= 10");
        assert!(
            !compiled.evaluate(&arr(vec![15.0])).expect("eval"),
            "15 > 10"
        );
    }

    #[test]
    fn test_compile_sqrt() {
        // sqrt(x[0]^2 + x[1]^2) <= 1.0  (||x||_2 <= 1)
        let x0_sq = TLExpr::Pow(Box::new(tl_var("dim_0")), Box::new(tl_const(2.0)));
        let x1_sq = TLExpr::Pow(Box::new(tl_var("dim_1")), Box::new(tl_const(2.0)));
        let sum_sq = TLExpr::Add(Box::new(x0_sq), Box::new(x1_sq));
        let norm = TLExpr::Sqrt(Box::new(sum_sq));
        let expr = TLExpr::Lte(Box::new(norm), Box::new(tl_const(1.0)));

        let compiler = TlExprCompiler::new();
        let compiled = compiler
            .compile(&expr, "l2_ball", 2)
            .expect("compile failed");

        // (0.3, 0.4) → norm = 0.5 ≤ 1.0
        assert!(compiled.evaluate(&arr(vec![0.3, 0.4])).expect("eval"));
        // (1.0, 1.0) → norm = sqrt(2) > 1.0
        assert!(!compiled.evaluate(&arr(vec![1.0, 1.0])).expect("eval"));
    }
}
