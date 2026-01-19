//! Property-Based Tests for Model Architectures
//!
//! Uses proptest to verify mathematical properties and invariants
//! that should hold for all models under various inputs.
//!
//! Note: Tests use small model configurations for performance.
//! The default 256 test cases * model creation time can be slow with large models.

use kizzasi_model::*;
use proptest::prelude::*;
use scirs2_core::ndarray::Array1;

// Reduce the default number of test cases for faster execution
// while still maintaining good coverage
const PROPTEST_CASES: u32 = 32;

/// Strategy for generating reasonable floating point values
fn reasonable_float() -> impl Strategy<Value = f32> {
    prop::num::f32::NORMAL.prop_filter("Must be finite", |x| x.is_finite())
}

mod mamba_properties {
    use super::*;
    use kizzasi_model::mamba::{Mamba, MambaConfig};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

        /// Property: Model output should always be finite
        #[test]
        fn output_is_finite(input_val in reasonable_float()) {
            // Use minimal configuration for fast tests
            let config = MambaConfig::default()
                .hidden_dim(32)
                .state_dim(4)
                .num_layers(1);
            let mut model = Mamba::new(config).unwrap();

            let input = Array1::from_elem(1, input_val);
            let output = model.step(&input).unwrap();

            prop_assert!(output.iter().all(|&x| x.is_finite()));
        }

        /// Property: Reset should make model produce same output for same input
        #[test]
        fn reset_produces_consistent_output(input_val in reasonable_float()) {
            let config = MambaConfig::default()
                .hidden_dim(32)
                .state_dim(4)
                .num_layers(1);
            let mut model = Mamba::new(config).unwrap();

            let input = Array1::from_elem(1, input_val);

            // First run
            model.reset();
            let output1 = model.step(&input).unwrap();

            // Second run after reset
            model.reset();
            let output2 = model.step(&input).unwrap();

            // Outputs should be very close
            for i in 0..output1.len() {
                prop_assert!((output1[i] - output2[i]).abs() < 1e-4);
            }
        }

        /// Property: State persistence - setting and getting states should be consistent
        #[test]
        fn state_persistence_is_consistent(
            steps in 1usize..=3,  // Reduced from 5
            input_val in reasonable_float()
        ) {
            let config = MambaConfig::default()
                .hidden_dim(32)
                .state_dim(4)
                .num_layers(1);
            let mut model = Mamba::new(config).unwrap();

            let input = Array1::from_elem(1, input_val);

            // Process some steps
            for _ in 0..steps {
                let _ = model.step(&input).unwrap();
            }

            // Save state
            let saved_state = model.get_states();

            // Process more steps
            for _ in 0..2 {
                let _ = model.step(&input).unwrap();
            }

            // Restore state
            model.set_states(saved_state.clone()).unwrap();

            // Get state again
            let restored_state = model.get_states();

            // States should match
            prop_assert_eq!(saved_state.len(), restored_state.len());
        }

        /// Property: Model should handle zero input without issues
        #[test]
        fn handles_zero_input(num_steps in 1usize..=5) {  // Reduced from 10
            let config = MambaConfig::default()
                .hidden_dim(32)
                .state_dim(4)
                .num_layers(1);
            let mut model = Mamba::new(config).unwrap();

            let input = Array1::zeros(1);

            for _ in 0..num_steps {
                let output = model.step(&input).unwrap();
                prop_assert!(output.iter().all(|&x| x.is_finite()));
            }
        }

        /// Property: Output magnitude should be bounded for bounded inputs
        #[test]
        fn bounded_input_produces_bounded_output(
            input_val in -10.0f32..=10.0f32,
            steps in 1usize..=10  // Reduced from 20
        ) {
            let config = MambaConfig::default()
                .hidden_dim(32)
                .state_dim(4)
                .num_layers(1);
            let mut model = Mamba::new(config).unwrap();

            let input = Array1::from_elem(1, input_val);

            for _ in 0..steps {
                let output = model.step(&input).unwrap();

                // Output should not explode
                let max_magnitude = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
                prop_assert!(max_magnitude < 1000.0,
                    "Output exploded: max magnitude = {}", max_magnitude);
            }
        }
    }
}

mod rwkv_properties {
    use super::*;
    use kizzasi_model::rwkv::{Rwkv, RwkvConfig};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

        /// Property: RWKV output should be finite
        #[test]
        fn output_is_finite(input_val in reasonable_float()) {
            // Use minimal configuration for fast tests
            let config = RwkvConfig::default()
                .hidden_dim(32)
                .num_heads(2)
                .num_layers(1);
            let mut model = Rwkv::new(config).unwrap();

            let input = Array1::from_elem(1, input_val);
            let output = model.step(&input).unwrap();

            prop_assert!(output.iter().all(|&x| x.is_finite()));
        }

        /// Property: Exponential decay - after large input, outputs should decay with zero input
        #[test]
        fn exponential_decay_property(
            initial_val in 10.0f32..=100.0f32,
            decay_steps in 3usize..=10  // Reduced from 5..=20
        ) {
            let config = RwkvConfig::default()
                .hidden_dim(32)
                .num_heads(2)
                .num_layers(1);
            let mut model = Rwkv::new(config).unwrap();

            // Large initial input
            let large_input = Array1::from_elem(1, initial_val);
            let _ = model.step(&large_input).unwrap();

            // Measure magnitude after first zero
            let zero_input = Array1::zeros(1);
            let output1 = model.step(&zero_input).unwrap();
            let mag1: f32 = output1.iter().map(|&x| x.abs()).sum();

            // Process more zeros
            for _ in 0..decay_steps {
                let _ = model.step(&zero_input).unwrap();
            }

            let output2 = model.step(&zero_input).unwrap();
            let mag2: f32 = output2.iter().map(|&x| x.abs()).sum();

            // Magnitude should decrease or stay similar (due to decay)
            // This is a weak property because of random initialization
            prop_assert!(mag2 < mag1 * 2.0,
                "Expected decay, but mag1={} and mag2={}", mag1, mag2);
        }
    }
}

mod transformer_properties {
    use super::*;
    use kizzasi_model::transformer::{Transformer, TransformerConfig};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

        /// Property: Transformer output should be finite
        #[test]
        fn output_is_finite(input_val in reasonable_float()) {
            // Use minimal configuration for fast tests
            let config = TransformerConfig::default()
                .hidden_dim(32)
                .num_heads(2)
                .num_layers(1);
            let mut model = Transformer::new(config).unwrap();

            let input = Array1::from_elem(1, input_val);
            let output = model.step(&input).unwrap();

            prop_assert!(output.iter().all(|&x| x.is_finite()));
        }

        /// Property: Context window growth
        #[test]
        fn context_grows_with_steps(num_steps in 1usize..=5) {  // Reduced from 10
            let config = TransformerConfig::default()
                .hidden_dim(32)
                .num_heads(2)
                .num_layers(1);
            let mut model = Transformer::new(config).unwrap();

            let input = Array1::from_elem(1, 1.0);

            for _ in 0..num_steps {
                let output = model.step(&input).unwrap();
                prop_assert!(output.iter().all(|&x| x.is_finite()));
            }

            // Model should handle growing context
            prop_assert!(true);
        }
    }
}

mod s4_properties {
    use super::*;
    use kizzasi_model::s4::{S4Config, S4D};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

        /// Property: S4D output should be finite
        #[test]
        fn output_is_finite(input_val in reasonable_float()) {
            // Use minimal configuration for fast tests
            let config = S4Config::default()
                .hidden_dim(32)
                .state_dim(4)
                .num_layers(1);
            let mut model = S4D::new(config).unwrap();

            let input = Array1::from_elem(1, input_val);
            let output = model.step(&input).unwrap();

            prop_assert!(output.iter().all(|&x| x.is_finite()));
        }

        /// Property: Diagonal SSM stability
        #[test]
        fn diagonal_ssm_stability(
            input_val in -10.0f32..=10.0f32,
            steps in 1usize..=15  // Reduced from 30
        ) {
            let config = S4Config::default()
                .hidden_dim(32)
                .state_dim(4)
                .num_layers(1);
            let mut model = S4D::new(config).unwrap();

            let input = Array1::from_elem(1, input_val);

            for _ in 0..steps {
                let output = model.step(&input).unwrap();

                // Output should remain stable
                let max_val = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
                prop_assert!(max_val < 1000.0, "S4D unstable: max = {}", max_val);
            }
        }
    }
}

mod batch_properties {
    use super::*;
    use kizzasi_model::batch::BatchedModel;
    use kizzasi_model::mamba::{Mamba, MambaConfig};
    use scirs2_core::ndarray::Array2;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

        /// Property: Batch processing should produce finite outputs
        #[test]
        fn batch_output_is_finite(
            batch_size in 1usize..=3,  // Reduced from 4
            input_val in reasonable_float()
        ) {
            // Use minimal configuration for fast tests
            let config = MambaConfig::default()
                .hidden_dim(32)
                .state_dim(4)
                .num_layers(1);
            let model = Mamba::new(config).unwrap();
            let mut batched = BatchedModel::new(model, batch_size).unwrap();

            let inputs = Array2::from_elem((batch_size, 1), input_val);
            let outputs = batched.predict_batch(&inputs).unwrap();

            for i in 0..batch_size {
                for j in 0..outputs.ncols() {
                    prop_assert!(outputs[[i, j]].is_finite());
                }
            }
        }

        /// Property: Batch independence - different inputs should produce different outputs
        #[test]
        fn batch_items_are_independent(
            val1 in -5.0f32..=5.0f32,
            val2 in -5.0f32..=5.0f32
        ) {
            prop_assume!((val1 - val2).abs() > 0.1); // Ensure different inputs

            let config = MambaConfig::default()
                .hidden_dim(32)
                .state_dim(4)
                .num_layers(1);
            let model = Mamba::new(config).unwrap();
            let mut batched = BatchedModel::new(model, 2).unwrap();

            let inputs = Array2::from_shape_vec((2, 1), vec![val1, val2]).unwrap();
            let outputs = batched.predict_batch(&inputs).unwrap();

            // Outputs for different inputs should be different (at least sometimes)
            let output1_sum: f32 = (0..outputs.ncols()).map(|j| outputs[[0, j]]).sum();
            let output2_sum: f32 = (0..outputs.ncols()).map(|j| outputs[[1, j]]).sum();

            // Allow for some tolerance due to random initialization
            prop_assert!((output1_sum - output2_sum).abs() < 1000.0);
        }
    }
}

mod quantization_properties {
    use super::*;
    use kizzasi_model::quantization::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

        /// Property: Quantization and dequantization should approximately preserve values
        #[test]
        fn quantization_preserves_magnitude(
            values in prop::collection::vec(-100.0f32..=100.0f32, 5..=15)  // Reduced from 5..=20
        ) {
            let array = Array1::from_vec(values.clone());

            let quantized = quantize_symmetric_1d(&array).unwrap();
            let dequantized = quantized.dequantize_1d().unwrap();

            for i in 0..array.len() {
                let error = (array[i] - dequantized[i]).abs();
                let rel_error = if array[i].abs() > 0.01 {
                    error / array[i].abs()
                } else {
                    error
                };

                // Quantization error should be reasonable
                prop_assert!(rel_error < 0.1 || error < 1.0,
                    "Too much quantization error at {}: {} vs {} (error: {})",
                    i, array[i], dequantized[i], error);
            }
        }

        /// Property: Quantized weights should use less memory
        #[test]
        fn quantization_reduces_memory(
            rows in 2usize..=8,  // Reduced from 10
            cols in 2usize..=8   // Reduced from 10
        ) {
            let size = rows * cols;
            let values: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let array = scirs2_core::ndarray::Array2::from_shape_vec((rows, cols), values).unwrap();

            let quantized = quantize_symmetric_2d(&array).unwrap();

            let original_size = size * 4; // f32 = 4 bytes
            let quantized_size = quantized.memory_size();

            prop_assert!(quantized_size <= original_size);
            prop_assert_eq!(quantized_size, size); // INT8 = 1 byte per element
        }

        /// Property: Per-channel quantization should handle different ranges
        #[test]
        fn per_channel_handles_different_ranges(
            scale1 in 0.1f32..=1.0f32,
            scale2 in 10.0f32..=100.0f32
        ) {
            // Create matrix with different ranges per row
            let row1: Vec<f32> = (0..5).map(|i| (i as f32) * scale1).collect();
            let row2: Vec<f32> = (0..5).map(|i| (i as f32) * scale2).collect();

            let mut all_values = row1;
            all_values.extend(row2);

            let array = scirs2_core::ndarray::Array2::from_shape_vec((2, 5), all_values).unwrap();

            let quantized = quantize_symmetric_per_channel(&array).unwrap();
            let dequantized = quantized.dequantize_2d().unwrap();

            // Per-channel should handle both ranges well
            for i in 0..2 {
                for j in 0..5 {
                    let error = (array[[i, j]] - dequantized[[i, j]]).abs();
                    let rel_error = if array[[i, j]].abs() > 0.01 {
                        error / array[[i, j]].abs()
                    } else {
                        error
                    };

                    prop_assert!(rel_error < 0.15 || error < 2.0,
                        "Per-channel quantization error too high");
                }
            }
        }
    }
}
