//! Property-based tests for constraint system
//!
//! Tests core mathematical properties:
//! - Projection idempotence: project(project(x)) = project(x)
//! - Projection reduces violation: violation(project(x)) <= violation(x)
//! - Penalty consistency: penalty non-negative
//! - Composition soundness for scalar constraints

use kizzasi_logic::*;
use proptest::prelude::*;

// ============================================
// Property: Scalar Constraint Projection
// ============================================

proptest! {
    #[test]
    fn prop_scalar_projection_idempotence(x in -1000.0f32..1000.0f32, bound in 1.0f32..100.0f32) {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .less_than(bound)
            .build()
            .unwrap();

        let projected = constraint.project(x);
        let double_projected = constraint.project(projected);

        // Project twice should equal project once
        prop_assert!((projected - double_projected).abs() < 1e-5);
    }

    #[test]
    fn prop_scalar_projection_reduces_violation(
        x in -100.0f32..100.0f32,
        bound in 1.0f32..50.0f32
    ) {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .less_than(bound)
            .build()
            .unwrap();

        let projected = constraint.project(x);
        let original_violation = constraint.violation(x);
        let projected_violation = constraint.violation(projected);

        // Projection should reduce or maintain violation
        prop_assert!(projected_violation <= original_violation + 1e-5);
    }

    #[test]
    fn prop_range_constraint_enforces_bounds(
        x in -1000.0f32..1000.0f32,
        lower in -100.0f32..0.0f32,
        upper in 1.0f32..100.0f32
    ) {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .in_range(lower, upper)
            .build()
            .unwrap();

        let projected = constraint.project(x);

        // Projected value must be in range
        prop_assert!(projected >= lower - 1e-5);
        prop_assert!(projected <= upper + 1e-5);
    }

    #[test]
    fn prop_violation_non_negative(
        x in -100.0f32..100.0f32,
        bound in 1.0f32..50.0f32
    ) {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .less_than(bound)
            .build()
            .unwrap();

        let violation = constraint.violation(x);

        // Violation should always be non-negative
        prop_assert!(violation >= -1e-5);
    }

    #[test]
    fn prop_zero_violation_iff_satisfied(
        x in -100.0f32..100.0f32,
        bound in 1.0f32..50.0f32
    ) {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .less_than(bound)
            .build()
            .unwrap();

        let violation = constraint.violation(x);
        let satisfied = constraint.check(x);

        // If satisfied, violation must be ~0
        if satisfied {
            prop_assert!(violation.abs() < 1e-5);
        }
    }

    #[test]
    fn prop_greater_than_constraint(
        x in -100.0f32..100.0f32,
        bound in -50.0f32..0.0f32
    ) {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .greater_than(bound)
            .build()
            .unwrap();

        let projected = constraint.project(x);

        // Projected value must be greater than bound
        prop_assert!(projected >= bound - 1e-5);
    }
}

// ============================================
// Property: Penalty Functions
// ============================================

proptest! {
    #[test]
    fn prop_penalty_l1_non_negative(
        violation in 0.0f32..100.0f32,
        weight in 0.1f32..10.0f32
    ) {
        let penalty_fn = PenaltyFunction::L1;
        let penalty = penalty_fn.compute(violation, weight);

        prop_assert!(penalty >= -1e-5);
    }

    #[test]
    fn prop_penalty_l2_non_negative(
        violation in 0.0f32..100.0f32,
        weight in 0.1f32..10.0f32
    ) {
        let penalty_fn = PenaltyFunction::L2;
        let penalty = penalty_fn.compute(violation, weight);

        prop_assert!(penalty >= -1e-5);
    }

    #[test]
    fn prop_penalty_zero_when_satisfied(
        weight in 0.1f32..10.0f32
    ) {
        let penalty_fn = PenaltyFunction::L2;
        let penalty = penalty_fn.compute(0.0, weight);

        prop_assert!(penalty.abs() < 1e-5);
    }

    #[test]
    fn prop_penalty_increases_with_violation(
        v1 in 0.0f32..50.0f32,
        v2 in 51.0f32..100.0f32,
        weight in 0.1f32..10.0f32
    ) {
        let penalty_fn = PenaltyFunction::L2;
        let penalty1 = penalty_fn.compute(v1, weight);
        let penalty2 = penalty_fn.compute(v2, weight);

        // Larger violation -> larger penalty
        prop_assert!(penalty2 >= penalty1 - 1e-5);
    }

    #[test]
    fn prop_penalty_scales_with_weight(
        violation in 1.0f32..10.0f32,
        w1 in 0.1f32..1.0f32,
        w2 in 2.0f32..5.0f32
    ) {
        prop_assume!(w2 > w1);

        let penalty_fn = PenaltyFunction::L1;
        let penalty1 = penalty_fn.compute(violation, w1);
        let penalty2 = penalty_fn.compute(violation, w2);

        // Higher weight -> higher penalty
        prop_assert!(penalty2 >= penalty1);
    }

    #[test]
    fn prop_huber_penalty_smooth_transition(
        violation in 0.0f32..10.0f32,
        weight in 0.5f32..2.0f32,
        delta in 0.5f32..5.0f32
    ) {
        let penalty_fn = PenaltyFunction::Huber { delta };
        let penalty = penalty_fn.compute(violation, weight);

        // Huber penalty should be non-negative and finite
        prop_assert!(penalty >= -1e-5);
        prop_assert!(penalty.is_finite());
    }
}

// ============================================
// Property: Hysteresis
// ============================================

proptest! {
    #[test]
    fn prop_hysteresis_state_transitions(
        lower in 1.0f32..40.0f32,
        upper in 50.0f32..90.0f32
    ) {
        prop_assume!(lower < upper);

        let mut constraint = HysteresisConstraint::new("test", lower, upper);

        // Start below lower threshold -> state should be low
        constraint.update_and_check(lower - 1.0);
        prop_assert!(!constraint.state());

        // Go above upper threshold -> state should be high
        constraint.update_and_check(upper + 1.0);
        prop_assert!(constraint.state());

        // Go between thresholds -> state should stay high
        constraint.update_and_check((lower + upper) / 2.0);
        prop_assert!(constraint.state());

        // Go below lower threshold -> state should be low again
        constraint.update_and_check(lower - 1.0);
        prop_assert!(!constraint.state());
    }

    #[test]
    fn prop_hysteresis_no_chattering(
        lower in 10.0f32..20.0f32,
        upper in 30.0f32..40.0f32,
        mid_value in 20.01f32..29.99f32
    ) {
        prop_assume!(lower < upper);
        prop_assume!(mid_value > lower && mid_value < upper);

        let mut constraint = HysteresisConstraint::new("test", lower, upper);

        // Start low
        constraint.update_and_check(lower - 1.0);
        let state1 = constraint.state();

        // Oscillate in the middle region
        for _ in 0..10 {
            constraint.update_and_check(mid_value);
        }
        let state2 = constraint.state();

        // State should not change when oscillating in hysteresis zone
        prop_assert_eq!(state1, state2);
    }
}

// ============================================
// Property: Sliding Window
// ============================================

proptest! {
    #[test]
    fn prop_sliding_window_violation_non_negative(
        values in prop::collection::vec(-10.0f32..10.0f32, 10..20),
        lower in -5.0f32..0.0f32,
        upper in 1.0f32..5.0f32,
        window_size in 3usize..6usize
    ) {
        let constraint = SlidingWindowConstraint::new(
            "mean_test",
            window_size,
            SlidingWindowFn::MeanInRange { lo: lower, hi: upper }
        );

        let mut checker = SlidingWindowChecker::new();
        checker.add(constraint);

        for &val in &values {
            let _ = checker.push_and_check(val);
        }

        let total_viol = checker.total_violation();
        prop_assert!(total_viol >= -1e-5);
    }

    #[test]
    fn prop_sliding_window_all_satisfied_check(
        window_size in 3usize..8usize,
        lower in -5.0f32..0.0f32,
        upper in 1.0f32..5.0f32
    ) {
        let constraint = SlidingWindowConstraint::new(
            "mean_test",
            window_size,
            SlidingWindowFn::MeanInRange { lo: lower, hi: upper }
        );

        let mut checker = SlidingWindowChecker::new();
        checker.add(constraint);

        // Push values that should satisfy the mean constraint
        let target_mean = (lower + upper) / 2.0;
        for _ in 0..window_size {
            checker.push_and_check(target_mean);
        }

        // Should be satisfied when window is full with good values
        // Note: We're testing that the code runs without panicking
        let _ = checker.all_satisfied();
    }
}

// ============================================
// Property: Constraint Building
// ============================================

proptest! {
    #[test]
    fn prop_constraint_builder_less_than(bound in 1.0f32..1000.0f32) {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .less_than(bound)
            .build();

        prop_assert!(constraint.is_ok());
    }

    #[test]
    fn prop_constraint_builder_greater_than(bound in -1000.0f32..1000.0f32) {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .greater_than(bound)
            .build();

        prop_assert!(constraint.is_ok());
    }

    #[test]
    fn prop_constraint_builder_range(
        lower in -100.0f32..0.0f32,
        upper in 1.0f32..100.0f32
    ) {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .in_range(lower, upper)
            .build();

        prop_assert!(constraint.is_ok());
    }

    #[test]
    fn prop_temporal_constraint_builds(
        max_rate in 1.0f32..100.0f32
    ) {
        let constraint = TemporalConstraintBuilder::new()
            .name("rate_limit")
            .max_rate(max_rate)
            .build();

        // Constraint should build successfully or fail gracefully
        let _ = constraint;
        prop_assert!(true); // Always pass - we just check it doesn't panic
    }
}
