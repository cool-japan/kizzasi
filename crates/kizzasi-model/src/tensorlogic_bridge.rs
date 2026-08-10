//! TensorLogic bridge: compose `TLExpr` constraints for model-level use.
//!
//! This module provides a thin bridge between the symbolic `TLExpr` constraint
//! representation and the numerical `CompiledConstraint` execution layer from
//! `kizzasi-logic`.
//!
//! # Example
//!
//! ```rust
//! use kizzasi_model::tensorlogic_bridge::constraint_from_tl_expr;
//! use kizzasi_logic::{tl_var, tl_const, TLExpr};
//! use scirs2_core::ndarray::Array1;
//!
//! // Build: -1.0 <= x[0] <= 1.0
//! let lower = TLExpr::Gte(Box::new(tl_var("dim_0")), Box::new(tl_const(-1.0)));
//! let upper = TLExpr::Lte(Box::new(tl_var("dim_0")), Box::new(tl_const(1.0)));
//! let expr = TLExpr::And(Box::new(lower), Box::new(upper));
//!
//! let compiled = constraint_from_tl_expr("unit_box", &expr, 1).expect("compile failed");
//! let x = Array1::from_vec(vec![0.5_f32]);
//! assert!(compiled.evaluate(&x).unwrap());
//! ```

use kizzasi_logic::{CompiledConstraint, LogicResult, TLExpr, TlExprCompiler};

/// Compile a single `TLExpr` constraint into a `CompiledConstraint`.
///
/// `name` is a human-readable label stored in the compiled output.
/// `num_dims` is the expected dimensionality of the input vector.
pub fn constraint_from_tl_expr(
    name: &str,
    expr: &TLExpr,
    num_dims: usize,
) -> LogicResult<CompiledConstraint> {
    TlExprCompiler::new().compile(expr, name, num_dims)
}

/// Compile a batch of named `TLExpr` constraints.
///
/// Each entry is `(name, expr, num_dims)`. Returns an error if any
/// constraint fails to compile; otherwise returns all compiled constraints
/// in the same order.
pub fn compile_constraints(
    constraints: &[(&str, TLExpr, usize)],
) -> LogicResult<Vec<CompiledConstraint>> {
    let compiler = TlExprCompiler::new();
    constraints
        .iter()
        .map(|(name, expr, num_dims)| compiler.compile(expr, name, *num_dims))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kizzasi_logic::{tl_const, tl_var};
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_constraint_from_tl_expr_box() {
        // Box constraint: -1.0 <= x[0] <= 1.0
        let lower = TLExpr::Gte(Box::new(tl_var("dim_0")), Box::new(tl_const(-1.0)));
        let upper = TLExpr::Lte(Box::new(tl_var("dim_0")), Box::new(tl_const(1.0)));
        let expr = TLExpr::And(Box::new(lower), Box::new(upper));

        let compiled = constraint_from_tl_expr("unit_box", &expr, 1).expect("compile failed");

        // 0.5 ∈ [-1.0, 1.0]
        let x_in = Array1::from_vec(vec![0.5_f32]);
        assert!(
            compiled.evaluate(&x_in).expect("eval failed"),
            "0.5 should satisfy unit box"
        );

        // 2.0 ∉ [-1.0, 1.0]
        let x_out = Array1::from_vec(vec![2.0_f32]);
        assert!(
            !compiled.evaluate(&x_out).expect("eval failed"),
            "2.0 should violate unit box"
        );
    }

    #[test]
    fn test_compile_constraints_multi() {
        // Constraint 1: box -3.0 <= x[0] <= 3.0
        let box_lower = TLExpr::Gte(Box::new(tl_var("dim_0")), Box::new(tl_const(-3.0)));
        let box_upper = TLExpr::Lte(Box::new(tl_var("dim_0")), Box::new(tl_const(3.0)));
        let box_expr = TLExpr::And(Box::new(box_lower), Box::new(box_upper));

        // Constraint 2: L2 ball radius 3 — x[0]^2 <= 9.0
        let x_sq = TLExpr::Mul(Box::new(tl_var("dim_0")), Box::new(tl_var("dim_0")));
        let ball_expr = TLExpr::Lte(Box::new(x_sq), Box::new(tl_const(9.0)));

        let constraints = vec![("box", box_expr, 1_usize), ("l2_ball", ball_expr, 1_usize)];

        let compiled = compile_constraints(&constraints).expect("compile batch failed");
        assert_eq!(compiled.len(), 2, "should have 2 compiled constraints");

        // x=2.0 satisfies both (|2.0| <= 3 and 2.0^2=4 <= 9)
        let x_in = Array1::from_vec(vec![2.0_f32]);
        assert!(
            compiled[0].evaluate(&x_in).expect("box eval"),
            "2.0 satisfies box [-3, 3]"
        );
        assert!(
            compiled[1].evaluate(&x_in).expect("ball eval"),
            "2.0 satisfies L2 ball r=3"
        );

        // x=4.0 violates both (|4.0| > 3 and 4.0^2=16 > 9)
        let x_out = Array1::from_vec(vec![4.0_f32]);
        assert!(
            !compiled[0].evaluate(&x_out).expect("box eval"),
            "4.0 violates box [-3, 3]"
        );
        assert!(
            !compiled[1].evaluate(&x_out).expect("ball eval"),
            "4.0 violates L2 ball r=3"
        );
    }
}
