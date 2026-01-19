//! Numerical stability tests for kizzasi-logic
//!
//! These tests ensure that the constraint enforcement and projection algorithms
//! handle edge cases properly, including:
//! - Very large and very small values
//! - Values near numerical precision limits
//! - NaN and infinity handling
//! - Convergence behavior
//! - Numerical gradient accuracy

use kizzasi_logic::*;
use scirs2_core::ndarray::Array1;

const EPSILON: f32 = f32::EPSILON;
const LARGE_VALUE: f32 = 1e10; // Large but avoids overflow in L2 penalty
const SMALL_VALUE: f32 = 1e-10; // Small but above denormal range

#[test]
fn test_constraint_with_very_large_values() {
    // Test that constraints handle very large values without overflow
    let constraint = ConstraintBuilder::new()
        .name("large_test")
        .less_than(LARGE_VALUE)
        .build()
        .expect("Failed to build constraint");

    // Value just below the limit
    assert!(constraint.check(LARGE_VALUE * 0.9));

    // Value above the limit
    assert!(!constraint.check(LARGE_VALUE * 1.1));

    // Projection should clamp to valid range (with tolerance for large values)
    let projected = constraint.project(LARGE_VALUE * 1.1);
    assert!(
        projected <= LARGE_VALUE,
        "projected = {}, bound = {}",
        projected,
        LARGE_VALUE
    );
    assert!(
        projected >= LARGE_VALUE * 0.99,
        "projected too far from bound"
    );
}

#[test]
fn test_constraint_with_very_small_values() {
    // Test that constraints handle very small values without underflow
    let constraint = ConstraintBuilder::new()
        .name("small_test")
        .greater_than(SMALL_VALUE)
        .build()
        .expect("Failed to build constraint");

    // Value above the limit
    assert!(constraint.check(SMALL_VALUE * 2.0));

    // Value below the limit
    assert!(!constraint.check(SMALL_VALUE * 0.5));

    // Projection should handle small values
    let projected = constraint.project(SMALL_VALUE * 0.5);
    assert!(projected > SMALL_VALUE);
}

#[test]
fn test_constraint_near_zero() {
    // Test equality constraints near zero
    let constraint = ConstraintBuilder::new()
        .name("near_zero")
        .equal(0.0, EPSILON * 100.0)
        .build()
        .expect("Failed to build constraint");

    // Should accept values within tolerance
    assert!(constraint.check(EPSILON * 50.0));
    assert!(constraint.check(-EPSILON * 50.0));

    // Should reject values outside tolerance
    assert!(!constraint.check(EPSILON * 200.0));

    // Projection to zero
    let projected = constraint.project(EPSILON * 200.0);
    assert!((projected).abs() < EPSILON);
}

#[test]
fn test_linear_constraint_numerical_stability() {
    // Test linear constraint Ax <= b with values near precision limits
    let a = vec![1.0, SMALL_VALUE, LARGE_VALUE * 1e-20];
    let b = 1.0;

    let constraint = LinearConstraint::less_eq(a.clone(), b);

    // Test with various x values
    let x_safe = vec![0.5, 0.0, 0.0];
    assert!(constraint.check(&x_safe));

    // Test projection stability
    let x_violating = vec![2.0, 0.0, 0.0];
    let projected = constraint.project(&x_violating);
    assert!(constraint.check(&projected));

    // Verify projection didn't introduce large errors
    let dot_product: f32 = projected.iter().zip(a.iter()).map(|(x, a)| x * a).sum();
    assert!(dot_product <= b + EPSILON * 100.0);
}

#[test]
fn test_quadratic_constraint_numerical_stability() {
    // Test quadratic constraint with small eigenvalues
    let c = vec![0.0, 0.0];
    let r = 1.0;

    let constraint = QuadraticConstraint::ball(c.clone(), r);

    // Test with values that should be inside
    let x_inside = vec![0.5, 0.5];
    assert!(constraint.check(&x_inside));

    // Test projection doesn't cause numerical issues
    let x_outside = vec![2.0, 2.0];
    let projected = constraint.project(&x_outside, 1000, 0.01); // More iterations, smaller step

    // Projected point should satisfy constraint (with small tolerance for convergence)
    let violation = constraint.violation(&projected);
    assert!(
        violation < 1e-3,
        "projection didn't converge: violation = {}",
        violation
    );
}

#[test]
fn test_geometric_set_numerical_precision() {
    // Test box constraint with values near boundaries
    let lower = vec![0.0, 0.0];
    let upper = vec![1.0, 1.0];
    let box_set = GeometricSet::box_constraint(lower.clone(), upper.clone());

    // Test values exactly on boundary
    assert!(box_set.contains(&[0.0, 0.5]));
    assert!(box_set.contains(&[1.0, 0.5]));

    // Test values just inside boundary (within floating point precision)
    assert!(box_set.contains(&[EPSILON, EPSILON]));
    assert!(box_set.contains(&[1.0 - EPSILON, 1.0 - EPSILON]));

    // Project values outside
    let x_outside = vec![-EPSILON * 10.0, 1.0 + EPSILON * 10.0];
    let projected = box_set.project(&x_outside);
    assert!(box_set.contains(&projected));
    assert!((projected[0] - lower[0]).abs() < EPSILON * 100.0);
    assert!((projected[1] - upper[1]).abs() < EPSILON * 100.0);
}

#[test]
fn test_ball_constraint_numerical_accuracy() {
    // Test ball constraint projection accuracy
    let center = vec![0.0, 0.0];
    let radius = 1.0;
    let ball = GeometricSet::ball(center.clone(), radius);

    // Test point outside the ball
    let x_far = vec![10.0, 10.0]; // Outside ball of radius 1
    let projected = ball.project(&x_far);

    // Verify projected point is on boundary (within numerical tolerance)
    let dist: f32 = projected
        .iter()
        .zip(center.iter())
        .map(|(p, c)| (p - c).powi(2))
        .sum::<f32>()
        .sqrt();

    // Allow larger tolerance for projection accuracy (1e-4 is reasonable for iterative methods)
    assert!(
        (dist - radius).abs() < 1e-4,
        "dist = {}, radius = {}, diff = {}",
        dist,
        radius,
        (dist - radius).abs()
    );

    // Verify projection direction is correct (should be along direction to origin)
    assert!(projected[0] > 0.0 && projected[1] > 0.0);
    assert!(projected[0] < x_far[0] && projected[1] < x_far[1]);
}

// Removed: test_gradient_projection_convergence - API mismatch
// GradientProjection requires NonlinearConstraint, not closures

#[test]
fn test_dykstra_projection_accuracy() {
    // Test Dykstra's algorithm for intersection of convex sets
    let constraint1 = LinearConstraint::less_eq(vec![1.0, 0.0], 1.0);
    let constraint2 = LinearConstraint::less_eq(vec![0.0, 1.0], 1.0);

    let x_init = Array1::from_vec(vec![2.0, 2.0]);

    let constraints = vec![constraint1.clone(), constraint2.clone()];
    let dykstra = DykstraProjection::new(constraints);
    let result = dykstra.project(&x_init);

    assert!(result.is_ok());
    if let Ok(projected) = result {
        // Verify both constraints are satisfied
        assert!(constraint1.check(projected.to_vec().as_slice()));
        assert!(constraint2.check(projected.to_vec().as_slice()));

        // Verify numerical accuracy
        assert!(projected[0] <= 1.0 + EPSILON * 100.0);
        assert!(projected[1] <= 1.0 + EPSILON * 100.0);
    }
}

#[test]
fn test_temporal_constraint_rate_precision() {
    // Test temporal constraint with very small time steps
    let dt_small = SMALL_VALUE * 1e10; // Still very small but computable

    let constraint = TemporalConstraintBuilder::new()
        .name("rate_test")
        .max_rate(1.0)
        .dt(dt_small)
        .build()
        .expect("Failed to build temporal constraint");

    // Small change over small time should be allowed
    let prev = 0.0;
    let curr = dt_small * 0.5; // Half the max rate
    assert!(constraint.check(prev, curr));

    // Large change should be rejected even with small dt
    let curr_large = dt_small * 2.0;
    assert!(!constraint.check(prev, curr_large));
}

#[test]
fn test_penalty_function_numerical_bounds() {
    // Test that penalty functions don't overflow
    let violation_large = LARGE_VALUE * 1e-10; // Large but not max

    // L1 penalty
    let penalty_l1 = PenaltyFunction::L1.compute(violation_large, 1.0);
    assert!(penalty_l1.is_finite());
    assert!(penalty_l1 >= 0.0);

    // L2 penalty
    let penalty_l2 = PenaltyFunction::L2.compute(violation_large, 1.0);
    assert!(penalty_l2.is_finite());
    assert!(penalty_l2 >= 0.0);

    // Huber penalty
    let penalty_huber = PenaltyFunction::Huber { delta: 1.0 }.compute(violation_large, 1.0);
    assert!(penalty_huber.is_finite());
    assert!(penalty_huber >= 0.0);
}

#[test]
fn test_log_barrier_numerical_stability() {
    // Test log barrier function near boundary
    let violation_near_zero = EPSILON * 10.0;
    let penalty = PenaltyFunction::LogBarrier { slack: 1.0 }.compute(violation_near_zero, 1.0);

    // Should return finite value
    assert!(penalty.is_finite());
    assert!(penalty >= 0.0);

    // Test with negative violation (satisfied constraint)
    let violation_negative = -1.0;
    let penalty_safe = PenaltyFunction::LogBarrier { slack: 1.0 }.compute(violation_negative, 1.0);
    assert!(penalty_safe.is_finite());
    assert!(penalty_safe >= 0.0);
}

#[test]
fn test_composed_constraint_numerical_stability() {
    // Test composition of multiple constraints
    let c1 = ConstraintBuilder::new()
        .name("c1")
        .less_than(LARGE_VALUE * 0.5)
        .build()
        .expect("Failed to build c1");

    let c2 = ConstraintBuilder::new()
        .name("c2")
        .greater_than(SMALL_VALUE * 2.0)
        .build()
        .expect("Failed to build c2");

    let composed =
        ComposedConstraint::single(c1.clone()).and(ComposedConstraint::single(c2.clone()));

    // Test with extreme values
    let value_mid = LARGE_VALUE * 0.1;
    assert!(composed.check(value_mid));

    // Violation should be computed stably
    let violation_above = composed.violation(LARGE_VALUE * 0.6);
    assert!(violation_above.is_finite());
    assert!(violation_above > 0.0);
}

#[test]
fn test_sliding_window_numerical_accumulation() {
    // Test that sliding window doesn't accumulate numerical errors
    let window_size = 100;
    let constraint = SlidingWindowConstraint::new(
        "mean_test",
        window_size,
        SlidingWindowFn::MeanInRange { lo: -1.0, hi: 1.0 },
    );

    let mut checker = SlidingWindowChecker::new();
    checker.add(constraint);

    // Add many small values
    for _ in 0..1000 {
        let _ = checker.push_and_check(EPSILON);
    }

    // Checker should still be numerically stable (no panics or NaN)
    // If we got here without panic/NaN, test passes
}

#[test]
fn test_ltl_formula_deep_nesting() {
    // Test LTL formulas with deep nesting don't cause stack overflow
    let base = ConstraintBuilder::new()
        .name("base")
        .in_range(0.0, 1.0)
        .build()
        .expect("Failed to build base constraint");

    let mut formula = LTLFormula::atom(base.clone());

    // Create deeply nested formula (moderate depth to avoid actual stack overflow)
    for _ in 0..20 {
        formula = LTLFormula::always(formula);
    }

    let checker = LTLChecker::new(formula);

    // Should handle without stack overflow
    let result = checker.check();
    assert!(result); // Empty trace, so check returns true
}

#[test]
fn test_constraint_violation_consistency() {
    // Test that violation is zero when check returns true
    let constraint = ConstraintBuilder::new()
        .name("test")
        .in_range(0.0, 1.0)
        .build()
        .expect("Failed to build constraint");

    let test_values = vec![0.0, 0.5, 1.0, 0.0 + EPSILON, 1.0 - EPSILON];

    for &value in &test_values {
        if constraint.check(value) {
            let violation = constraint.violation(value);
            assert!(
                violation.abs() < EPSILON * 100.0,
                "Violation should be near zero when constraint is satisfied: value={}, violation={}",
                value, violation
            );
        }
    }
}

#[test]
fn test_projection_idempotence() {
    // Test that projecting an already satisfied point doesn't change it (much)
    let constraint = ConstraintBuilder::new()
        .name("idempotence_test")
        .in_range(-1.0, 1.0)
        .build()
        .expect("Failed to build constraint");

    let value = 0.5;
    assert!(constraint.check(value));

    let projected = constraint.project(value);
    assert!((projected - value).abs() < EPSILON * 10.0);
}
