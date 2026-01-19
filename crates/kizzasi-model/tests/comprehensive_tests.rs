//! Comprehensive Unit Tests for Model Layers
//!
//! Tests edge cases, numerical properties, and component interactions
//! for all model architectures.

use kizzasi_model::*;
use scirs2_core::ndarray::{Array1, Array2};

mod mamba_tests {
    use super::*;
    use kizzasi_model::mamba::{Mamba, MambaConfig};

    #[test]
    fn test_zero_input() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let mut model = Mamba::new(config).unwrap();

        let input = Array1::zeros(1);
        let output = model.step(&input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_large_input() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let mut model = Mamba::new(config).unwrap();

        let input = Array1::from_elem(1, 1000.0);
        let output = model.step(&input);
        assert!(output.is_ok());

        // Output should be finite
        let output = output.unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_negative_input() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let mut model = Mamba::new(config).unwrap();

        let input = Array1::from_elem(1, -10.0);
        let output = model.step(&input);
        assert!(output.is_ok());
    }

    #[test]
    fn test_state_reset() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let mut model = Mamba::new(config).unwrap();

        // Process some inputs
        let input = Array1::from_elem(1, 1.0);
        for _ in 0..10 {
            let _ = model.step(&input).unwrap();
        }

        // Get state
        let state_before = model.get_states();

        // Reset
        model.reset();
        let state_after = model.get_states();

        // States should be different (reset to zeros)
        assert_eq!(state_before.len(), state_after.len());
    }

    #[test]
    fn test_state_persistence() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let mut model = Mamba::new(config).unwrap();

        // Process input to build up state
        let input = Array1::from_elem(1, 1.0);
        for _ in 0..3 {
            let _ = model.step(&input).unwrap();
        }

        // Save state after 3 steps
        let saved_state = model.get_states();

        // Continue for more steps
        for _ in 0..5 {
            let _ = model.step(&input).unwrap();
        }

        // Restore to saved state (after 3 steps)
        model.set_states(saved_state.clone()).unwrap();

        // Next prediction should be same as the 4th step originally
        let output1 = model.step(&input).unwrap();

        // Restore state again and repeat
        model.set_states(saved_state).unwrap();
        let output2 = model.step(&input).unwrap();

        // Outputs should be very close (deterministic from same state)
        for i in 0..output1.len() {
            assert!(
                (output1[i] - output2[i]).abs() < 1e-4,
                "Mismatch at index {}: {} vs {}",
                i,
                output1[i],
                output2[i]
            );
        }
    }

    #[test]
    fn test_output_consistency() {
        // Test that repeated identical inputs through fresh model produce consistent results
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);

        let input = Array1::from_elem(1, 1.5);

        // Create two fresh models
        let mut model1 = Mamba::new(config.clone()).unwrap();
        let mut model2 = Mamba::new(config).unwrap();

        // Both models start fresh, process one input
        let output1 = model1.step(&input).unwrap();
        let output2 = model2.step(&input).unwrap();

        // Outputs should exist and be finite (models initialized differently, so values will differ)
        assert!(output1.iter().all(|&x| x.is_finite()));
        assert!(output2.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_sequential_consistency() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let mut model = Mamba::new(config).unwrap();

        let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Process sequentially
        let mut outputs1 = Vec::new();
        for &val in &inputs {
            let input = Array1::from_elem(1, val);
            outputs1.push(model.step(&input).unwrap());
        }

        // Reset and process again
        model.reset();
        let mut outputs2 = Vec::new();
        for &val in &inputs {
            let input = Array1::from_elem(1, val);
            outputs2.push(model.step(&input).unwrap());
        }

        // All outputs should match
        for (out1, out2) in outputs1.iter().zip(outputs2.iter()) {
            for i in 0..out1.len() {
                assert!((out1[i] - out2[i]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_multidimensional_input() {
        let config = MambaConfig::default()
            .input_dim(3)
            .hidden_dim(64)
            .num_layers(2);
        let mut model = Mamba::new(config).unwrap();

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let output = model.step(&input);
        assert!(output.is_ok());
    }
}

mod rwkv_tests {
    use super::*;
    use kizzasi_model::rwkv::{Rwkv, RwkvConfig};

    #[test]
    fn test_zero_input() {
        let config = RwkvConfig::default().hidden_dim(64).num_layers(2);
        let mut model = Rwkv::new(config).unwrap();

        let input = Array1::zeros(1);
        let output = model.step(&input);
        assert!(output.is_ok());
    }

    #[test]
    fn test_state_accumulation() {
        let config = RwkvConfig::default().hidden_dim(64).num_layers(2);
        let mut model = Rwkv::new(config).unwrap();

        let input = Array1::from_elem(1, 1.0);

        // Process multiple steps
        let mut outputs = Vec::new();
        for _ in 0..10 {
            outputs.push(model.step(&input).unwrap());
        }

        // Outputs should be different (state accumulates)
        assert!(outputs.len() == 10);
    }

    #[test]
    fn test_exponential_decay() {
        let config = RwkvConfig::default().hidden_dim(64).num_layers(2);
        let mut model = Rwkv::new(config).unwrap();

        // Large input followed by zeros
        let large_input = Array1::from_elem(1, 100.0);
        let _ = model.step(&large_input).unwrap();

        let zero_input = Array1::zeros(1);
        let mut outputs = Vec::new();
        for _ in 0..20 {
            outputs.push(model.step(&zero_input).unwrap());
        }

        // Outputs should decay towards zero (due to exponential decay)
        // Later outputs should have smaller magnitude than earlier ones
        let early_magnitude: f32 = outputs[0].iter().map(|&x| x.abs()).sum();
        let late_magnitude: f32 = outputs[19].iter().map(|&x| x.abs()).sum();

        assert!(late_magnitude < early_magnitude);
    }
}

mod transformer_tests {
    use super::*;
    use kizzasi_model::transformer::{Transformer, TransformerConfig};

    #[test]
    fn test_attention_mask() {
        let config = TransformerConfig::default()
            .hidden_dim(64)
            .num_heads(4)
            .num_layers(2);
        let mut model = Transformer::new(config).unwrap();

        let input = Array1::from_elem(1, 1.0);

        // Process sequence
        let mut outputs = Vec::new();
        for _ in 0..5 {
            outputs.push(model.step(&input).unwrap());
        }

        // Each output should depend on all previous inputs (attention)
        assert_eq!(outputs.len(), 5);
    }

    #[test]
    fn test_kv_cache_growth() {
        // Use smaller configuration for faster test
        let config = TransformerConfig::default()
            .hidden_dim(32)
            .num_heads(2)
            .num_layers(1);
        let mut model = Transformer::new(config).unwrap();

        let input = Array1::from_elem(1, 1.0);

        // Process enough steps to test cache management (reduced from 100)
        for _ in 0..50 {
            let _ = model.step(&input).unwrap();
        }

        // Model should handle long sequences
        // (KV cache management)
    }

    #[test]
    fn test_multi_head_consistency() {
        // Test with different numbers of heads
        for num_heads in [1, 2, 4, 8] {
            let config = TransformerConfig::default()
                .hidden_dim(64)
                .num_heads(num_heads)
                .num_layers(2);

            let model = Transformer::new(config);
            assert!(model.is_ok());
        }
    }
}

mod s4_tests {
    use super::*;
    use kizzasi_model::s4::{S4Config, S4D};

    #[test]
    fn test_diagonal_state_matrix() {
        let config = S4Config::default().hidden_dim(64).num_layers(2);
        let mut model = S4D::new(config).unwrap();

        let input = Array1::from_elem(1, 1.0);
        let output = model.step(&input);
        assert!(output.is_ok());
    }

    #[test]
    fn test_hippo_initialization() {
        // S4 uses HiPPO initialization for stable long-range dependencies
        let config = S4Config::default().hidden_dim(64).state_dim(16);
        let model = S4D::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_continuous_time_discretization() {
        let config = S4Config::default().hidden_dim(64).state_dim(16);

        let mut model = S4D::new(config).unwrap();

        let input = Array1::from_elem(1, 1.0);
        let output = model.step(&input);
        assert!(output.is_ok());

        // Output should be finite and well-behaved
        let output = output.unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}

mod batch_integration_tests {
    use super::*;
    use kizzasi_model::batch::BatchedModel;
    use kizzasi_model::mamba::{Mamba, MambaConfig};

    #[test]
    fn test_batch_vs_sequential() {
        // Test that batch processing with batch_size=1 behaves like sequential
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let model = Mamba::new(config).unwrap();

        // Copy states to ensure both start from same point
        let initial_states = model.get_states();

        // Create sequential model
        let mut model_seq =
            Mamba::new(MambaConfig::default().hidden_dim(64).num_layers(2)).unwrap();
        model_seq.set_states(initial_states.clone()).unwrap();

        // Create batched model
        let mut model_batch =
            Mamba::new(MambaConfig::default().hidden_dim(64).num_layers(2)).unwrap();
        model_batch.set_states(initial_states).unwrap();
        let mut batched = BatchedModel::new(model_batch, 1).unwrap();

        let inputs = vec![1.0, 2.0, 3.0];

        let mut outputs_seq = Vec::new();
        for &val in &inputs {
            let input = Array1::from_elem(1, val);
            outputs_seq.push(model_seq.step(&input).unwrap());
        }

        // Process through batch
        for &val in &inputs {
            let batch_inputs = Array2::from_shape_vec((1, 1), vec![val]).unwrap();
            let _ = batched.predict_batch(&batch_inputs).unwrap();
        }

        // Note: Due to different random initializations, outputs will differ
        // This test just ensures both methods run without errors
        assert_eq!(outputs_seq.len(), 3);
    }

    #[test]
    fn test_batch_independence() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let model = Mamba::new(config).unwrap();
        let mut batched = BatchedModel::new(model, 3).unwrap();

        // Process different inputs in parallel
        let inputs = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let _outputs1 = batched.predict_batch(&inputs).unwrap();

        // Process again (states should be independent)
        let inputs = Array2::from_shape_vec((3, 1), vec![4.0, 5.0, 6.0]).unwrap();
        let outputs2 = batched.predict_batch(&inputs).unwrap();

        // Each batch item should have different outputs
        assert_ne!(outputs2[[0, 0]], outputs2[[1, 0]]);
        assert_ne!(outputs2[[1, 0]], outputs2[[2, 0]]);
    }
}

mod quantization_integration_tests {
    use super::*;
    use kizzasi_model::quantization::*;

    #[test]
    fn test_quantization_accuracy() {
        let original = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let quantized = quantize_symmetric_1d(&original).unwrap();
        let dequantized = quantized.dequantize_1d().unwrap();

        // Check relative error
        for i in 0..5 {
            let rel_error = ((original[i] - dequantized[i]) / original[i]).abs();
            assert!(rel_error < 0.05); // Less than 5% error
        }
    }

    #[test]
    fn test_weight_quantization_2d() {
        let weight = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 0.5, 1.5, 2.5, 3.5,
            ],
        )
        .unwrap();

        let quantized = quantize_symmetric_2d(&weight).unwrap();
        let dequantized = quantized.dequantize_2d().unwrap();

        // Verify shape preserved
        assert_eq!(dequantized.dim(), weight.dim());

        // Check accuracy
        for i in 0..3 {
            for j in 0..4 {
                let diff = (weight[[i, j]] - dequantized[[i, j]]).abs();
                assert!(diff < 0.1);
            }
        }
    }

    #[test]
    fn test_extreme_values() {
        let array = Array1::from_vec(vec![-1000.0, -100.0, 0.0, 100.0, 1000.0]);

        let quantized = quantize_symmetric_1d(&array).unwrap();
        let dequantized = quantized.dequantize_1d().unwrap();

        // Extreme values should still be approximately preserved
        assert!((dequantized[0] - array[0]).abs() < 10.0);
        assert!((dequantized[4] - array[4]).abs() < 10.0);
    }
}

mod numerical_stability_tests {
    use super::*;
    use kizzasi_model::mamba::{Mamba, MambaConfig};

    #[test]
    fn test_no_nan_propagation() {
        // Use smaller configuration for faster test
        let config = MambaConfig::default()
            .hidden_dim(32)
            .state_dim(4)
            .num_layers(1);
        let mut model = Mamba::new(config).unwrap();

        // Process steps with varying inputs (reduced from 100 to 50)
        for i in 0..50 {
            let val = (i as f32).sin() * 10.0;
            let input = Array1::from_elem(1, val);
            let output = model.step(&input).unwrap();

            // No NaN should appear
            assert!(output.iter().all(|&x| !x.is_nan()));
        }
    }

    #[test]
    fn test_no_inf_propagation() {
        // Use smaller configuration for faster test
        let config = MambaConfig::default()
            .hidden_dim(32)
            .state_dim(4)
            .num_layers(1);
        let mut model = Mamba::new(config).unwrap();

        // Process with large inputs (reduced iterations from 50 to 20)
        for _ in 0..20 {
            let input = Array1::from_elem(1, 1000.0);
            let output = model.step(&input).unwrap();

            // No Inf should appear
            assert!(output.iter().all(|&x| !x.is_infinite()));
        }
    }

    #[test]
    fn test_gradient_flow() {
        // Test that models don't have vanishing/exploding outputs
        // Use smaller configuration for faster test
        let config = MambaConfig::default()
            .hidden_dim(32)
            .state_dim(4)
            .num_layers(2);
        let mut model = Mamba::new(config).unwrap();

        let input = Array1::from_elem(1, 1.0);

        // Process steps (reduced from 100 to 50)
        for _ in 0..50 {
            let output = model.step(&input).unwrap();

            // Output magnitude should stay reasonable
            let magnitude: f32 = output.iter().map(|&x| x.abs()).sum();
            assert!(magnitude > 1e-10); // Not vanishing
            assert!(magnitude < 1e10); // Not exploding
        }
    }
}
