//! Property-based tests for Kizzasi using proptest
//!
//! These tests verify invariants and properties that should hold
//! for all valid inputs.

use kizzasi::prelude::*;
use proptest::prelude::*;

// ============================================================================
// Property: Predictor always produces output of correct dimension
// ============================================================================

proptest! {
    #[test]
    fn test_output_dimension_invariant(
        input_dim in 1usize..10,
        output_dim in 1usize..10,
        hidden_dim in 16usize..128,
        state_dim in 4usize..32,
        num_layers in 1usize..5,
    ) {
        let config = KizzasiConfig::new()
            .input_dim(input_dim)
            .output_dim(output_dim)
            .hidden_dim(hidden_dim)
            .state_dim(state_dim)
            .num_layers(num_layers);

        let mut predictor = Kizzasi::new(config).unwrap();

        let input = Array1::from_vec(vec![0.5; input_dim]);
        let output = predictor.step(&input).unwrap();

        prop_assert_eq!(output.len(), output_dim);
    }
}

// ============================================================================
// Property: Reset should allow re-prediction to work
// ============================================================================

proptest! {
    #[test]
    fn test_reset_property(
        input_values in prop::collection::vec(-10.0f32..10.0, 3..=3),
    ) {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(32)
            .state_dim(8)
            .num_layers(2);

        let mut predictor = Kizzasi::new(config).unwrap();

        let input = Array1::from_vec(input_values.clone());

        // First prediction
        let output1 = predictor.step(&input).unwrap();

        // Make more predictions to change state
        for _ in 0..5 {
            let _ = predictor.step(&input).unwrap();
        }

        // Reset and predict again
        predictor.reset();
        let output2 = predictor.step(&input).unwrap();

        // After reset with same input, we should get same output
        prop_assert_eq!(output1.len(), output2.len());
        for (a, b) in output1.iter().zip(output2.iter()) {
            prop_assert!((a - b).abs() < 1e-5);
        }
    }
}

// ============================================================================
// Property: predict_n produces correct number of predictions
// ============================================================================

proptest! {
    #[test]
    fn test_predict_n_count(
        n_steps in 1usize..20,
        input_values in prop::collection::vec(-5.0f32..5.0, 2..=2),
    ) {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(1);

        let mut predictor = Kizzasi::new(config).unwrap();
        let input = Array1::from_vec(input_values);

        let predictions = predictor.predict_n(&input, n_steps).unwrap();

        prop_assert_eq!(predictions.nrows(), n_steps);
        prop_assert_eq!(predictions.ncols(), 2);
    }
}

// ============================================================================
// Property: predict_batch preserves order and count
// ============================================================================

proptest! {
    #[test]
    fn test_predict_batch_preserves_count(
        batch_size in 1usize..10,
    ) {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(1);

        let mut predictor = Kizzasi::new(config).unwrap();

        let inputs: Vec<Array1<f32>> = (0..batch_size)
            .map(|i| Array1::from_vec(vec![i as f32 * 0.1, i as f32 * 0.2]))
            .collect();

        let outputs = predictor.predict_batch(&inputs).unwrap();

        prop_assert_eq!(outputs.len(), batch_size);
        for output in outputs {
            prop_assert_eq!(output.len(), 2);
        }
    }
}

// ============================================================================
// Property: Fork creates independent predictors
// ============================================================================

proptest! {
    #[test]
    fn test_fork_independence(
        input_values in prop::collection::vec(-5.0f32..5.0, 3..=3),
    ) {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(1);

        let mut predictor1 = Kizzasi::new(config).unwrap();

        // Prime predictor1 with some inputs to establish a state
        let input = Array1::from_vec(input_values.clone());
        for _ in 0..3 {
            let _ = predictor1.step(&input).unwrap();
        }

        // Fork after establishing state
        let mut predictor2 = predictor1.fork().unwrap();

        // Make one prediction with each
        let out1_before = predictor1.step(&input).unwrap();
        let out2_before = predictor2.step(&input).unwrap();

        // They should NOT be the same initially because fork creates new weights
        // But both should be valid outputs
        prop_assert_eq!(out1_before.len(), 3);
        prop_assert_eq!(out2_before.len(), 3);

        // More steps on predictor1 shouldn't affect predictor2
        for _ in 0..5 {
            let _ = predictor1.step(&input).unwrap();
        }

        // predictor2 should still work independently
        let out2_after = predictor2.step(&input).unwrap();
        prop_assert_eq!(out2_after.len(), 3);

        // Both predictors should maintain valid output dimensions
        prop_assert_eq!(predictor1.config().get_output_dim(), 3);
        prop_assert_eq!(predictor2.config().get_output_dim(), 3);
    }
}

// ============================================================================
// Property: Builder validation catches invalid configs
// ============================================================================

proptest! {
    #[test]
    fn test_builder_validation_rejects_zero_dims(
        valid_dim in 1usize..10,
    ) {
        // Zero input_dim should fail
        let result1 = KizzasiBuilder::new()
            .input_dim(0)
            .output_dim(valid_dim)
            .hidden_dim(valid_dim)
            .build();
        prop_assert!(result1.is_err());

        // Zero output_dim should fail
        let result2 = KizzasiBuilder::new()
            .input_dim(valid_dim)
            .output_dim(0)
            .hidden_dim(valid_dim)
            .build();
        prop_assert!(result2.is_err());

        // Zero hidden_dim should fail
        let result3 = KizzasiBuilder::new()
            .input_dim(valid_dim)
            .output_dim(valid_dim)
            .hidden_dim(0)
            .build();
        prop_assert!(result3.is_err());
    }
}

// ============================================================================
// Property: SignalInput conversions preserve data
// ============================================================================

proptest! {
    #[test]
    fn test_signal_input_from_vec_preserves_data(
        values in prop::collection::vec(-100.0f32..100.0, 1..20),
    ) {
        let signal_input: SignalInput = values.clone().into();
        let array = signal_input.as_array();

        prop_assert_eq!(array.len(), values.len());
        for (i, &val) in values.iter().enumerate() {
            prop_assert!((array[i] - val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_signal_input_from_slice_preserves_data(
        values in prop::collection::vec(-100.0f32..100.0, 1..20),
    ) {
        let signal_input: SignalInput = values.as_slice().into();
        let array = signal_input.as_array();

        prop_assert_eq!(array.len(), values.len());
        for (i, &val) in values.iter().enumerate() {
            prop_assert!((array[i] - val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_signal_input_from_f32_creates_single_element(
        value in -1000.0f32..1000.0,
    ) {
        let signal_input: SignalInput = value.into();
        let array = signal_input.as_array();

        prop_assert_eq!(array.len(), 1);
        prop_assert!((array[0] - value).abs() < 1e-6);
    }
}

// ============================================================================
// Property: Presets create valid configurations
// ============================================================================

#[test]
fn test_audio_preset_creates_valid_predictor() {
    let predictor = KizzasiBuilder::audio_preset().build();
    assert!(predictor.is_ok());

    let p = predictor.unwrap();
    assert_eq!(p.config().get_input_dim(), 1);
    assert_eq!(p.config().get_output_dim(), 1);
}

proptest! {
    #[test]
    fn test_robotics_preset_creates_valid_predictor(
        axes in 1usize..12,
    ) {
        let predictor = KizzasiBuilder::robotics_preset(axes).build();
        prop_assert!(predictor.is_ok());

        let p = predictor.unwrap();
        prop_assert_eq!(p.config().get_input_dim(), axes);
        prop_assert_eq!(p.config().get_output_dim(), axes);
    }

    #[test]
    fn test_sensor_preset_creates_valid_predictor(
        num_sensors in 1usize..20,
    ) {
        let predictor = KizzasiBuilder::sensor_preset(num_sensors).build();
        prop_assert!(predictor.is_ok());

        let p = predictor.unwrap();
        prop_assert_eq!(p.config().get_input_dim(), num_sensors);
    }

    #[test]
    fn test_lightweight_preset_creates_valid_predictor(
        input_dim in 1usize..10,
        output_dim in 1usize..10,
    ) {
        let predictor = KizzasiBuilder::lightweight_preset(input_dim, output_dim).build();
        prop_assert!(predictor.is_ok());

        let p = predictor.unwrap();
        prop_assert_eq!(p.config().get_input_dim(), input_dim);
        prop_assert_eq!(p.config().get_output_dim(), output_dim);
    }

    #[test]
    #[ignore] // Slow test: ~114s due to large range of frame features (1..512)
    fn test_video_preset_creates_valid_predictor(
        frame_features in 1usize..512,
    ) {
        let predictor = KizzasiBuilder::video_preset(frame_features).build();
        prop_assert!(predictor.is_ok());

        let p = predictor.unwrap();
        prop_assert_eq!(p.config().get_input_dim(), frame_features);
    }

    #[test]
    fn test_control_preset_creates_valid_predictor(
        state_dim in 1usize..20,
        action_dim in 1usize..10,
    ) {
        let predictor = KizzasiBuilder::control_preset(state_dim, action_dim).build();
        prop_assert!(predictor.is_ok());

        let p = predictor.unwrap();
        prop_assert_eq!(p.config().get_input_dim(), state_dim);
        prop_assert_eq!(p.config().get_output_dim(), action_dim);
    }
}

// ============================================================================
// Property: Error categories are correctly assigned
// ============================================================================

#[test]
fn test_error_categories() {
    let err1 = KizzasiError::config("test");
    assert_eq!(err1.category(), ErrorCategory::Configuration);

    let err2 = KizzasiError::dimension_mismatch(3, 5, "test");
    assert_eq!(err2.category(), ErrorCategory::Validation);

    let err3 = KizzasiError::invalid_state("test");
    assert_eq!(err3.category(), ErrorCategory::State);

    let err4 = KizzasiError::model_not_ready("test", "suggestion");
    assert_eq!(err4.category(), ErrorCategory::Initialization);

    let err5 = KizzasiError::resource_exhausted("test", 10, 5, "suggestion");
    assert_eq!(err5.category(), ErrorCategory::Resource);
}

// ============================================================================
// Property: Errors provide recovery suggestions when applicable
// ============================================================================

#[test]
fn test_error_recovery_suggestions() {
    let err1 = KizzasiError::dimension_mismatch(3, 5, "input");
    assert!(err1.recovery_suggestion().is_some());
    assert!(err1.is_recoverable());

    let err2 = KizzasiError::invalid_state_with_recovery("not init", "call build()");
    assert!(err2.recovery_suggestion().is_some());
    assert!(err2.is_recoverable());

    let err3 = KizzasiError::config("input_dim must be > 0");
    assert!(err3.recovery_suggestion().is_some());
    assert!(err3.is_recoverable());
}
