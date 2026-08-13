//! Integration tests for end-to-end constraint workflows

use kizzasi_logic::*;
use scirs2_core::ndarray::Array1;

#[test]
fn test_training_pipeline_with_constraints() {
    let velocity_constraint = ConstraintBuilder::new()
        .name("max_velocity")
        .less_than(10.0)
        .build()
        .unwrap();

    let acceleration_constraint = ConstraintBuilder::new()
        .name("max_acceleration")
        .less_than(5.0)
        .build()
        .unwrap();

    let constraints = vec![velocity_constraint.clone(), acceleration_constraint.clone()];
    let loss_fn = ConstraintAwareLoss::new(constraints, PenaltyFunction::L2, 1.0);

    let predictions = [8.0f32, 9.5, 11.0, 12.0, 9.0, 8.5];

    for (epoch, &pred) in predictions.iter().enumerate() {
        let task_loss = (pred - 9.0f32).powi(2);
        let total_loss = loss_fn.compute_loss(&[pred], task_loss);

        assert!(total_loss >= task_loss);

        let v_satisfied = velocity_constraint.check(pred);
        let a_satisfied = acceleration_constraint.check(pred);

        println!(
            "Epoch {}: pred={:.1}, loss={:.4}, v_ok={}, a_ok={}",
            epoch, pred, total_loss, v_satisfied, a_satisfied
        );
    }
}

#[test]
fn test_sliding_window_workflow() {
    let window_constraint = SlidingWindowConstraint::new(
        "temp_mean",
        5,
        SlidingWindowFn::MeanInRange { lo: 20.0, hi: 25.0 },
    );

    let mut checker = SlidingWindowChecker::new();
    checker.add(window_constraint);

    let temperatures = vec![
        22.0f32, 23.0, 24.0, 23.5, 24.5, 25.5, 26.0, 27.0, 26.5, 25.0, 23.0, 22.5, 23.5, 24.0, 23.0,
    ];

    for (i, &temp) in temperatures.iter().enumerate() {
        let results = checker.push_and_check(temp);

        if i >= 4 {
            for (name, satisfied, violation) in results {
                println!(
                    "Reading {}: temp={:.1}, {}: satisfied={}, violation={:.2}",
                    i + 1,
                    temp,
                    name,
                    satisfied,
                    violation
                );
            }
        }
    }

    let total_viol = checker.total_violation();
    assert!(total_viol >= 0.0);
}

#[test]
fn test_hysteresis_workflow() {
    let thermostat = HysteresisConstraint::new("thermostat", 18.0, 22.0);

    let mut checker = HysteresisChecker::new();
    checker.add(thermostat);

    let temperatures = [20.0f32, 19.0, 17.0, 18.5, 20.0, 23.0, 21.0, 19.0];

    for (i, &temp) in temperatures.iter().enumerate() {
        checker.update(temp);
        let states = checker.states();

        for (name, heating_on) in states {
            println!(
                "Reading {}: temp={:.1}°C, {} = {}",
                i + 1,
                temp,
                name,
                if heating_on { "ON" } else { "OFF" }
            );
        }
    }
}

#[test]
fn test_advanced_constraints_integration() {
    let chance_constraint = ChanceConstraint::gaussian("temp_chance", 0.95, 25.0, 2.0)
        .expect("valid chance constraint");

    let tightened_bound = chance_constraint
        .get_tightened_bound()
        .expect("gaussian tightening succeeds");
    assert!(tightened_bound > 25.0);
    assert!(tightened_bound < 30.0);

    println!(
        "Chance constraint tightened bound: {:.2}°C",
        tightened_bound
    );

    let cvar_constraint =
        CVaRConstraint::new("risk_limit", 0.05, 100.0, 1000).expect("valid CVaR constraint");

    let losses: Vec<f32> = (0..100)
        .map(|i| if i < 95 { i as f32 } else { (i as f32) * 1.5 })
        .collect();

    let cvar_value = cvar_constraint.compute_cvar(&losses);
    let is_satisfied = cvar_constraint.check(&losses);

    println!(
        "CVaR (α=0.05): {:.2}, threshold: {}, satisfied: {}",
        cvar_value,
        cvar_constraint.threshold(),
        is_satisfied
    );

    assert!(cvar_value > 0.0);
}

#[test]
fn test_robust_optimization_workflow() {
    let robust_constraint =
        RobustConstraint::box_uncertain("power_limit", vec![-2.0, -1.0], vec![2.0, 1.0])
            .expect("valid robust constraint");

    let nominal = vec![5.0, 3.0];
    let worst_case = robust_constraint
        .worst_case_scenario(&nominal)
        .expect("box worst case succeeds");

    println!("Nominal: {:?}", nominal);
    println!("Worst-case: {:?}", worst_case);

    assert_eq!(worst_case.len(), 2);
}

#[test]
fn test_differentiable_projection_workflow() {
    let diff_proj = DifferentiableProjection::new(1.0).expect("positive temperature");

    let test_values = vec![
        (-5.0f32, 0.0, 10.0),
        (0.0, 0.0, 10.0),
        (5.0, 0.0, 10.0),
        (10.0, 0.0, 10.0),
        (15.0, 0.0, 10.0),
    ];

    for (x, lower, upper) in test_values {
        let projected = diff_proj.soft_project_box(x, lower, upper);
        let hard_proj = x.max(lower).min(upper);
        let diff = (projected - hard_proj).abs();

        println!(
            "Soft project: x={:.1} -> {:.2} (hard={:.1}, diff={:.2})",
            x, projected, hard_proj, diff
        );

        if x > lower && x < upper {
            assert!(diff < 1.0);
        }
    }
}

#[test]
fn test_guardrail_system_workflow() {
    let velocity_guard = Guardrail::new(
        ConstraintBuilder::new()
            .name("max_velocity")
            .less_than(10.0)
            .build()
            .unwrap(),
        false,
    );

    let acceleration_guard = Guardrail::new(
        ConstraintBuilder::new()
            .name("max_acceleration")
            .less_than(5.0)
            .build()
            .unwrap(),
        false,
    );

    let mut guardrail_set = GuardrailSet::new();
    guardrail_set.add_global(velocity_guard);
    guardrail_set.add_global(acceleration_guard);

    // When both constraints are global, all values must satisfy BOTH constraints
    // So values must be < 5.0 (the more restrictive constraint) to pass
    let test_cases = vec![
        (vec![3.0f32], true, "Satisfies both constraints"),
        (vec![4.5], true, "Just under acceleration limit"),
        (vec![8.0], false, "Violates acceleration constraint"),
        (vec![12.0], false, "Violates both constraints"),
        (vec![6.0], false, "Violates acceleration constraint"),
    ];

    for (values, should_pass, description) in test_cases {
        let input = Array1::from_vec(values.clone());
        let passed = guardrail_set.validate(&input);

        println!(
            "Value {:?}: {} ({})",
            values,
            if passed { "PASS" } else { "FAIL" },
            description
        );

        assert_eq!(
            passed, should_pass,
            "Guardrail check failed for value {:?} ({})",
            values, description
        );
    }
}

#[test]
fn test_safe_model_deployment_workflow() {
    let safety_constraints = vec![
        ConstraintBuilder::new()
            .name("min_safe")
            .greater_than(0.0)
            .build()
            .unwrap(),
        ConstraintBuilder::new()
            .name("max_safe")
            .less_than(100.0)
            .build()
            .unwrap(),
    ];

    let mut guards = GuardrailSet::new();
    for constraint in &safety_constraints {
        guards.add_global(Guardrail::new(constraint.clone(), false));
    }

    let training_loss = ConstraintAwareLoss::new(safety_constraints, PenaltyFunction::L2, 0.5);

    // Test validation with predictions in range (0.0, 100.0)
    let safe_predictions = [50.0f32, 25.0, 75.0, 1.0, 99.0];
    // Use values clearly outside the range to avoid boundary precision issues
    let unsafe_predictions = [110.0f32, -5.0, -10.0, 150.0, 200.0];

    println!("\nTesting safe predictions:");
    for &pred in safe_predictions.iter() {
        let pred_array = Array1::from_vec(vec![pred]);
        let is_safe = guards.validate(&pred_array);
        assert!(
            is_safe,
            "Safe prediction {:.1} should pass validation",
            pred
        );

        let task_loss = 1.0;
        let total_loss = training_loss.compute_loss(&[pred], task_loss);
        // Safe predictions should have minimal penalty
        println!("  Prediction {:.1}: loss = {:.4}", pred, total_loss);
    }

    println!("\nTesting unsafe predictions:");
    for &pred in unsafe_predictions.iter() {
        let pred_array = Array1::from_vec(vec![pred]);
        let is_safe = guards.validate(&pred_array);
        assert!(
            !is_safe,
            "Unsafe prediction {:.1} should fail validation",
            pred
        );

        let task_loss = 1.0;
        let total_loss = training_loss.compute_loss(&[pred], task_loss);
        // Unsafe predictions should have increased loss
        assert!(
            total_loss > task_loss,
            "Violation should increase loss for {:.1}",
            pred
        );
        println!(
            "  Prediction {:.1}: loss = {:.4} (penalty applied)",
            pred, total_loss
        );
    }

    println!("\nSafe deployment test passed: all outputs verified safe");
}

#[test]
fn test_penalty_function_comparison() {
    let violations = vec![0.0f32, 0.5, 1.0, 2.0, 5.0, 10.0];
    let weight = 1.0;

    println!("\nPenalty Function Comparison:");
    println!(
        "{:<10} {:<10} {:<10} {:<10}",
        "Violation", "L1", "L2", "Huber(δ=1)"
    );

    for &viol in &violations {
        let l1 = PenaltyFunction::L1.compute(viol, weight);
        let l2 = PenaltyFunction::L2.compute(viol, weight);
        let huber = PenaltyFunction::Huber { delta: 1.0 }.compute(viol, weight);

        println!("{:<10.1} {:<10.2} {:<10.2} {:<10.2}", viol, l1, l2, huber);

        assert!(l1 >= 0.0);
        assert!(l2 >= 0.0);
        assert!(huber >= 0.0);
    }
}

#[test]
fn test_constraint_composition_workflow() {
    let c1 = ConstraintBuilder::new()
        .name("lower")
        .greater_than(0.0)
        .build()
        .unwrap();

    let c2 = ConstraintBuilder::new()
        .name("upper")
        .less_than(100.0)
        .build()
        .unwrap();

    let combined =
        ComposedConstraint::single(c1.clone()).and(ComposedConstraint::single(c2.clone()));

    let test_values = vec![-5.0f32, 0.5, 50.0, 99.5, 105.0];

    for &val in &test_values {
        let c1_sat = c1.check(val);
        let c2_sat = c2.check(val);
        let combined_sat = combined.check(val);

        println!(
            "Value {:.1}: c1={}, c2={}, combined={}",
            val, c1_sat, c2_sat, combined_sat
        );

        assert_eq!(combined_sat, c1_sat && c2_sat);
    }
}
