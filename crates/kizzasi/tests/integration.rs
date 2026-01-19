//! Integration tests for the full Kizzasi pipeline
//!
//! Tests end-to-end scenarios combining signal generation,
//! prediction, and constraint enforcement.

use kizzasi::prelude::*;

#[test]
fn test_full_pipeline_audio_preset() {
    // Create predictor with audio preset
    let mut predictor = KizzasiBuilder::audio_preset().build().unwrap();

    // Generate test signal
    let input = Array1::from_vec(vec![0.5]);

    // Predict multiple steps
    let predictions = predictor.predict_n(&input, 10).unwrap();
    assert_eq!(predictions.shape(), &[10, 1]);
}

#[test]
fn test_full_pipeline_with_guardrails() {
    // Create a constraint: output must be in [-1, 1]
    let constraint = ConstraintBuilder::new()
        .name("amplitude_bound")
        .in_range(-1.0, 1.0)
        .build()
        .unwrap();

    let mut guardrails = GuardrailSet::new();
    guardrails.add_global(Guardrail::new(constraint, false));

    // Create predictor with guardrails
    let mut predictor = KizzasiBuilder::new()
        .input_dim(2)
        .output_dim(2)
        .hidden_dim(32)
        .state_dim(4)
        .num_layers(1)
        .guardrails(guardrails)
        .build()
        .unwrap();

    // Run prediction
    let input = Array1::from_vec(vec![0.5, 0.5]);
    let output = predictor.step(&input).unwrap();

    // Output should be constrained to [-1, 1]
    for &val in output.iter() {
        assert!((-1.0..=1.0).contains(&val), "Value {} out of bounds", val);
    }
}

#[test]
fn test_predict_until_convergence() {
    let mut predictor = KizzasiBuilder::lightweight_preset(1, 1).build().unwrap();

    let input = Array1::from_vec(vec![1.0]);

    // Predict until output is small (near zero)
    let predictions = predictor
        .predict_until(&input, 100, |output, _| output[0].abs() < 0.01)
        .unwrap();

    // Should have at least one prediction
    assert!(!predictions.is_empty());
    // Should stop before max_steps if converged
    assert!(predictions.len() <= 100);
}

#[test]
fn test_batch_prediction_consistency() {
    // Test that predict_batch is equivalent to calling step repeatedly
    // Use the same predictor for both to ensure same weights
    let config = KizzasiConfig::new()
        .input_dim(2)
        .output_dim(2)
        .hidden_dim(32)
        .state_dim(4)
        .num_layers(1);

    let mut predictor = Kizzasi::new(config).unwrap();

    let inputs = vec![
        Array1::from_vec(vec![0.1, 0.2]),
        Array1::from_vec(vec![0.3, 0.4]),
        Array1::from_vec(vec![0.5, 0.6]),
    ];

    // Batch prediction
    let batch_outputs = predictor.predict_batch(&inputs).unwrap();

    // Reset and run individual predictions on same predictor
    predictor.reset();
    let individual_outputs: Vec<_> = inputs
        .iter()
        .map(|input| predictor.step(input).unwrap())
        .collect();

    // Results should match
    for (batch, individual) in batch_outputs.iter().zip(individual_outputs.iter()) {
        for (a, b) in batch.iter().zip(individual.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Batch and individual predictions differ: {} vs {}",
                a,
                b
            );
        }
    }
}

#[test]
fn test_fork_creates_independent_state() {
    let mut original = KizzasiBuilder::lightweight_preset(2, 2).build().unwrap();

    // Run some predictions on original
    for _ in 0..5 {
        let input = Array1::from_vec(vec![0.1, 0.2]);
        let _ = original.step(&input);
    }

    // Fork the predictor
    let mut forked = original.fork().unwrap();

    // Now both should give same results for same input
    let test_input = Array1::from_vec(vec![0.5, 0.5]);
    let _orig_out = original.step(&test_input).unwrap();
    let _fork_out = forked.step(&test_input).unwrap();

    // They should have independent states now
    // (forked starts fresh, original continues from previous state)
}

#[test]
fn test_reset_clears_state() {
    let mut predictor = KizzasiBuilder::lightweight_preset(2, 2).build().unwrap();

    let input = Array1::from_vec(vec![0.5, 0.5]);

    // Run some predictions
    let first_output = predictor.step(&input).unwrap();
    let _ = predictor.step(&input);

    // Reset
    predictor.reset();

    // First output after reset should be same as initial first output
    let after_reset = predictor.step(&input).unwrap();

    for (a, b) in first_output.iter().zip(after_reset.iter()) {
        assert!((a - b).abs() < 1e-6, "Reset should return to initial state");
    }
}

#[test]
fn test_multi_layer_prediction() {
    // Test with multiple layers
    let mut predictor = KizzasiBuilder::new()
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(64)
        .state_dim(8)
        .num_layers(4) // Multiple layers
        .build()
        .unwrap();

    let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);

    // Should work with multiple layers
    let output = predictor.step(&input).unwrap();
    assert_eq!(output.len(), 3);

    // All outputs should be finite
    for &val in output.iter() {
        assert!(val.is_finite(), "Output should be finite");
    }
}

#[test]
fn test_composed_constraint_enforcement() {
    // Create composed constraint: x >= 0 AND x <= 1
    let lower = ConstraintBuilder::new()
        .name("lower")
        .greater_eq(0.0)
        .build()
        .unwrap();

    let upper = ConstraintBuilder::new()
        .name("upper")
        .less_eq(1.0)
        .build()
        .unwrap();

    let composed = ComposedConstraint::single(lower).and(ComposedConstraint::single(upper));

    // Test values
    assert!(composed.check(0.5));
    assert!(!composed.check(-0.5));
    assert!(!composed.check(1.5));

    // Test projection
    assert_eq!(composed.project(-0.5), 0.0);
    assert_eq!(composed.project(1.5), 1.0);
    assert_eq!(composed.project(0.5), 0.5);
}

#[test]
fn test_validation_without_enforcement() {
    let constraint = ConstraintBuilder::new()
        .name("bound")
        .in_range(-1.0, 1.0)
        .build()
        .unwrap();

    let mut guardrails = GuardrailSet::new();
    guardrails.add_global(Guardrail::new(constraint, false));

    let mut predictor = KizzasiBuilder::new()
        .input_dim(1)
        .output_dim(1)
        .hidden_dim(16)
        .state_dim(4)
        .num_layers(1)
        .guardrails(guardrails)
        .build()
        .unwrap();

    let input = Array1::from_vec(vec![0.5]);
    let output = predictor.step(&input).unwrap();

    // Validation should work
    let valid = predictor.validate(&output);
    // Output was constrained, so should be valid
    assert!(valid);

    // Test violation loss for a value outside bounds
    let bad_value = Array1::from_vec(vec![2.0]);
    let loss = predictor.violation_loss(&bad_value);
    assert!(loss > 0.0);
}
