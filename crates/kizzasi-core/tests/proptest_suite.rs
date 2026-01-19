//! Property-based tests for kizzasi-core
//!
//! Uses proptest to verify mathematical properties and invariants that should
//! hold for all inputs, catching edge cases that manual tests might miss.

use kizzasi_core::*;
use proptest::prelude::*;
use scirs2_core::ndarray::Array1;

// ============================================================================
// Neural Network Properties
// ============================================================================

proptest! {
    /// Property: Layer norm should have zero mean (within epsilon)
    #[test]
    fn prop_layer_norm_zero_mean(
        values in prop::collection::vec(-100.0f32..100.0f32, 1..=100)
    ) {
        let input = Array1::from_vec(values);
        let normalized = layer_norm(&input, 1e-5);

        let mean = normalized.mean().unwrap();
        prop_assert!(mean.abs() < 1e-4, "Mean should be close to zero, got {}", mean);
    }

    /// Property: Layer norm should have unit variance (within epsilon)
    #[test]
    fn prop_layer_norm_unit_variance(
        values in prop::collection::vec(-100.0f32..100.0f32, 2..=100)
    ) {
        let input = Array1::from_vec(values);
        let normalized = layer_norm(&input, 1e-5);

        let mean = normalized.mean().unwrap();
        let variance = normalized.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / normalized.len() as f32;

        prop_assert!((variance - 1.0).abs() < 0.1,
            "Variance should be close to 1.0, got {}", variance);
    }

    /// Property: RMS norm output magnitude
    #[test]
    fn prop_rms_norm_magnitude(
        values in prop::collection::vec(-100.0f32..100.0f32, 1..=100)
    ) {
        let input = Array1::from_vec(values);
        let normalized = rms_norm(&input, 1e-5);

        // RMS should normalize the magnitude
        let rms = (normalized.iter().map(|&x| x * x).sum::<f32>()
            / normalized.len() as f32).sqrt();

        prop_assert!((rms - 1.0).abs() < 0.1,
            "RMS should be close to 1.0, got {}", rms);
    }

    /// Property: Softmax sums to 1
    #[test]
    fn prop_softmax_sums_to_one(
        values in prop::collection::vec(-50.0f32..50.0f32, 1..=100)
    ) {
        let input = Array1::from_vec(values);
        let output = softmax(&input);

        let sum: f32 = output.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5,
            "Softmax should sum to 1.0, got {}", sum);
    }

    /// Property: Softmax outputs are in [0, 1]
    #[test]
    fn prop_softmax_range(
        values in prop::collection::vec(-50.0f32..50.0f32, 1..=100)
    ) {
        let input = Array1::from_vec(values);
        let output = softmax(&input);

        for &val in output.iter() {
            prop_assert!((0.0..=1.0).contains(&val),
                "Softmax output should be in [0, 1], got {}", val);
        }
    }

    /// Property: ReLU is non-negative
    #[test]
    fn prop_relu_non_negative(
        values in prop::collection::vec(-100.0f32..100.0f32, 1..=100)
    ) {
        let input = Array1::from_vec(values);
        let output = relu(&input);

        for &val in output.iter() {
            prop_assert!(val >= 0.0,
                "ReLU output should be non-negative, got {}", val);
        }
    }

    /// Property: ReLU preserves positive values
    #[test]
    fn prop_relu_preserves_positive(x in 0.0f32..100.0f32) {
        let input = Array1::from_vec(vec![x]);
        let output = relu(&input);
        prop_assert_eq!(output[0], x);
    }

    /// Property: ReLU zeros negative values
    #[test]
    fn prop_relu_zeros_negative(x in -100.0f32..0.0f32) {
        let input = Array1::from_vec(vec![x]);
        let output = relu(&input);
        prop_assert_eq!(output[0], 0.0);
    }

    /// Property: Sigmoid is in [0, 1]
    #[test]
    fn prop_sigmoid_range(
        values in prop::collection::vec(-100.0f32..100.0f32, 1..=100)
    ) {
        let input = Array1::from_vec(values);
        let output = sigmoid(&input);

        for &val in output.iter() {
            prop_assert!((0.0..=1.0).contains(&val),
                "Sigmoid output should be in [0, 1], got {}", val);
        }
    }

    /// Property: Tanh is in [-1, 1]
    #[test]
    fn prop_tanh_range(
        values in prop::collection::vec(-100.0f32..100.0f32, 1..=100)
    ) {
        let input = Array1::from_vec(values);
        let output = tanh(&input);

        for &val in output.iter() {
            prop_assert!((-1.0..=1.0).contains(&val),
                "Tanh output should be in [-1, 1], got {}", val);
        }
    }
}

// ============================================================================
// SIMD Operations Properties
// ============================================================================

proptest! {
    /// Property: SIMD dot product equals naive implementation
    #[test]
    fn prop_simd_dot_correctness(
        values_a in prop::collection::vec(-10.0f32..10.0f32, 1..=100),
        values_b in prop::collection::vec(-10.0f32..10.0f32, 1..=100)
    ) {
        let len = values_a.len().min(values_b.len());
        let a = &values_a[..len];
        let b = &values_b[..len];

        let simd_result = simd::dot_product(a, b);
        let naive_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        let diff = (simd_result - naive_result).abs();
        prop_assert!(diff < 1e-3,
            "SIMD dot should match naive: {} vs {}, diff={}",
            simd_result, naive_result, diff);
    }

    /// Property: SIMD vector addition is commutative
    #[test]
    fn prop_simd_add_commutative(
        values_a in prop::collection::vec(-10.0f32..10.0f32, 8..=64),
        values_b in prop::collection::vec(-10.0f32..10.0f32, 8..=64)
    ) {
        let len = values_a.len().min(values_b.len());
        let a = &values_a[..len];
        let b = &values_b[..len];

        let mut result1 = vec![0.0f32; len];
        let mut result2 = vec![0.0f32; len];

        simd::vec_add(a, b, &mut result1);
        simd::vec_add(b, a, &mut result2);

        for i in 0..len {
            let diff = (result1[i] - result2[i]).abs();
            prop_assert!(diff < 1e-5,
                "Addition should be commutative at index {}: {} vs {}",
                i, result1[i], result2[i]);
        }
    }
}

// ============================================================================
// Numerical Stability Properties
// ============================================================================

proptest! {
    /// Property: Kahan sum should be at least as accurate as naive sum
    #[test]
    fn prop_kahan_sum_accuracy(
        values in prop::collection::vec(-1000.0f32..1000.0f32, 1..=1000)
    ) {
        let kahan_result = numerics::kahan_sum(&values);
        let naive_result: f32 = values.iter().sum();

        // Kahan should be at least as accurate (or very close)
        // This property is hard to test exactly, but we verify it doesn't diverge
        let diff = (kahan_result - naive_result).abs();
        let magnitude = kahan_result.abs().max(1.0);

        prop_assert!(diff / magnitude < 0.1,
            "Kahan sum diverged too much: {} vs {}", kahan_result, naive_result);
    }

    /// Property: Safe exp never returns infinity for reasonable inputs
    #[test]
    fn prop_safe_exp_no_overflow(x in -50.0f32..50.0f32) {
        let result = numerics::safe_exp(x);
        prop_assert!(result.is_finite(), "safe_exp({}) returned non-finite: {}", x, result);
    }

    /// Property: Safe ln never returns NaN for positive inputs
    #[test]
    fn prop_safe_ln_no_nan(x in 1e-10f32..1000.0f32) {
        let result = numerics::safe_ln(x);
        prop_assert!(!result.is_nan(), "safe_ln({}) returned NaN: {}", x, result);
    }
}

// ============================================================================
// SSM Properties
// ============================================================================

proptest! {
    /// Property: SSM step should not produce NaN or Inf
    #[test]
    fn prop_ssm_no_nan_inf(
        input_dim in 1usize..=8,
        hidden_dim in 4usize..=16,
        state_dim in 2usize..=8,
        values in prop::collection::vec(-1.0f32..1.0f32, 1..=8)
    ) {
        let config = KizzasiConfig::new()
            .input_dim(input_dim)
            .output_dim(input_dim)
            .hidden_dim(hidden_dim)
            .state_dim(state_dim)
            .num_layers(1);

        let mut ssm = match SelectiveSSM::new(config) {
            Ok(s) => s,
            Err(_) => return Err(TestCaseError::fail("SSM creation failed")),
        };

        let input = if values.len() >= input_dim {
            Array1::from_vec(values[..input_dim].to_vec())
        } else {
            Array1::from_vec(vec![0.0; input_dim])
        };

        let output = match ssm.step(&input) {
            Ok(o) => o,
            Err(_) => return Err(TestCaseError::fail("SSM step failed")),
        };

        for &val in output.iter() {
            prop_assert!(val.is_finite(), "SSM output should be finite, got {}", val);
        }
    }

    /// Property: SSM reset should produce finite outputs
    #[test]
    fn prop_ssm_reset_finite_output(
        input_dim in 1usize..=4,
        hidden_dim in 4usize..=8
    ) {
        let config = KizzasiConfig::new()
            .input_dim(input_dim)
            .output_dim(input_dim)
            .hidden_dim(hidden_dim)
            .state_dim(4)
            .num_layers(1);

        let mut ssm = match SelectiveSSM::new(config) {
            Ok(s) => s,
            Err(_) => return Err(TestCaseError::fail("SSM creation failed")),
        };

        let input = Array1::from_vec(vec![1.0; input_dim]);

        // Run some steps
        for _ in 0..5 {
            let _ = ssm.step(&input);
        }

        // Reset
        ssm.reset();

        // Output after reset should be finite
        let output = match ssm.step(&input) {
            Ok(o) => o,
            Err(_) => return Err(TestCaseError::fail("SSM step after reset failed")),
        };

        for &val in output.iter() {
            prop_assert!(val.is_finite(),
                "SSM output after reset should be finite, got {}", val);
        }
    }
}

// ============================================================================
// Sequence Operations Properties
// ============================================================================

proptest! {
    /// Property: Padding should create correct output shape
    #[test]
    fn prop_padding_preserves_data(
        seq_len in 5usize..=20,
        n_seqs in 1usize..=5,
        n_features in 1usize..=8
    ) {
        let sequences: Vec<_> = (0..n_seqs)
            .map(|_| scirs2_core::ndarray::Array2::<f32>::zeros((seq_len, n_features)))
            .collect();

        match pad_sequences(&sequences, 0.0, PaddingStrategy::Right) {
            Ok((padded, mask)) => {
                // Check that all sequences are present and padded tensor has correct shape
                prop_assert_eq!(padded.dim().0, n_seqs);
                prop_assert_eq!(padded.dim().1, seq_len);
                prop_assert_eq!(padded.dim().2, n_features);
                prop_assert_eq!(mask.batch_size(), n_seqs);
                prop_assert_eq!(mask.max_len(), seq_len);
            }
            Err(_) => return Err(TestCaseError::fail("pad_sequences failed")),
        }
    }

    /// Property: SequenceMask should track lengths correctly
    #[test]
    fn prop_sequence_mask_lengths(
        lengths in prop::collection::vec(1usize..=20, 1..=10)
    ) {
        match SequenceMask::from_lengths(&lengths) {
            Ok(mask) => {
                prop_assert_eq!(mask.batch_size(), lengths.len());
                let max_len = *lengths.iter().max().unwrap();
                prop_assert_eq!(mask.max_len(), max_len);

                // Check that valid positions are marked correctly
                for (i, &len) in lengths.iter().enumerate() {
                    for j in 0..max_len {
                        let expected = j < len;
                        prop_assert_eq!(mask.is_valid(i, j), expected,
                            "Mask mismatch at ({}, {}): expected {}", i, j, expected);
                    }
                }
            }
            Err(_) => return Err(TestCaseError::fail("SequenceMask creation failed")),
        }
    }
}

// ============================================================================
// Configuration Properties
// ============================================================================

proptest! {
    /// Property: Config builder should preserve set values
    #[test]
    fn prop_config_builder_preserves_values(
        input_dim in 1usize..=32,
        hidden_dim in 4usize..=64,
        state_dim in 2usize..=32
    ) {
        let config = KizzasiConfig::new()
            .input_dim(input_dim)
            .hidden_dim(hidden_dim)
            .state_dim(state_dim);

        prop_assert_eq!(config.get_input_dim(), input_dim);
        prop_assert_eq!(config.get_hidden_dim(), hidden_dim);
        prop_assert_eq!(config.get_state_dim(), state_dim);
    }
}
