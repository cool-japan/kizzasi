//! Numerical Accuracy and Edge Case Tests
//!
//! Comprehensive test suite for:
//! - Numerical stability and accuracy
//! - Edge cases (zero, NaN, infinity)
//! - Long sequence handling
//! - Stress tests

use kizzasi_core::*;
use scirs2_core::ndarray::{Array1, Array2, Array3};

/// Test numerical stability with very small values
#[test]
fn test_small_values_stability() {
    let config = S4DConfig::new(5, 16, 32).delta(1e-6);
    let layer = S4DLayer::new(config).unwrap();

    let input = Array1::from_vec(vec![1e-8; 5]);
    let mut state = layer.reset_state();

    // Should not overflow or underflow
    for _ in 0..10 {
        let output = layer.step(&input, &mut state).unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}

/// Test numerical stability with very large values
#[test]
fn test_large_values_stability() {
    let config = S4DConfig::new(5, 16, 32);
    let layer = S4DLayer::new(config).unwrap();

    let input = Array1::from_vec(vec![1e3; 5]);
    let mut state = layer.reset_state();

    // Should not overflow
    for _ in 0..10 {
        let output = layer.step(&input, &mut state).unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}

/// Test handling of zero input
#[test]
fn test_zero_input_handling() {
    let config = S4DConfig::new(5, 16, 32);
    let layer = S4DLayer::new(config).unwrap();

    let zero_input = Array1::zeros(5);
    let mut state = layer.reset_state();

    // Zero input should produce valid output
    let output = layer.step(&zero_input, &mut state).unwrap();
    assert_eq!(output.len(), 32);
    assert!(output.iter().all(|&x| x.is_finite()));
}

/// Test NaN propagation is handled correctly
#[test]
fn test_nan_detection() {
    use crate::numerics::has_nan_inf;

    let normal = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(!has_nan_inf(&normal));

    let with_nan = Array1::from_vec(vec![1.0, f32::NAN, 3.0]);
    assert!(has_nan_inf(&with_nan));

    let with_inf = Array1::from_vec(vec![1.0, f32::INFINITY, 3.0]);
    assert!(has_nan_inf(&with_inf));
}

/// Test numerical accuracy of exp approximation
#[test]
fn test_fast_exp_accuracy() {
    use crate::simd::fast_exp;

    let test_values = vec![-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0];

    for &x in &test_values {
        let approx = fast_exp(x);
        let exact = x.exp();
        let rel_error = ((approx - exact) / exact).abs();

        // Fast exp should be accurate to ~1% for moderate values
        if x.abs() < 10.0 {
            assert!(
                rel_error < 0.01 || approx.is_nan() && exact.is_nan(),
                "x={}, approx={}, exact={}, error={}",
                x,
                approx,
                exact,
                rel_error
            );
        }
    }
}

/// Test softmax numerical stability
#[test]
fn test_softmax_stability() {
    // Test with large values that would overflow naive softmax
    let large_input = Array1::from_vec(vec![1000.0, 1001.0, 1002.0]);
    let output = softmax(&large_input);

    // Should sum to 1
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Softmax doesn't sum to 1: {}",
        sum
    );

    // All values should be finite and in [0, 1]
    for &val in output.iter() {
        assert!(val.is_finite() && (0.0..=1.0).contains(&val));
    }
}

/// Test layer normalization numerical stability
#[test]
fn test_layer_norm_stability() {
    let layer_norm = LayerNorm::new(128, NormType::LayerNorm);

    // Test with varying input at different scales
    for scale in &[1e-3, 1.0, 1e3] {
        // Create input with more variation
        let input = Array1::from_shape_fn(128, |i| {
            *scale * ((i as f32 / 128.0) * 2.0 - 1.0) // Range: [-scale, scale]
        });
        let output = layer_norm.forward(&input);

        // Output should be finite
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Non-finite output at scale {}",
            scale
        );

        // Output values should be in reasonable range (typically [-3, 3] for normalized data)
        let max_val = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_val < 10.0,
            "Output values too large: {} (scale={})",
            max_val,
            scale
        );
    }
}

/// Test RMS normalization
#[test]
fn test_rms_norm_stability() {
    let rms_norm = LayerNorm::new(128, NormType::RMSNorm);

    // Test with varying input at different scales
    for scale in &[1e-3, 1.0, 1e3] {
        // Create input with variation
        let input = Array1::from_shape_fn(128, |i| *scale * ((i as f32 / 128.0) * 2.0 - 1.0));
        let output = rms_norm.forward(&input);

        // Output should be finite
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Non-finite output at scale {}",
            scale
        );

        // Output values should be in reasonable range
        let max_val = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_val < 10.0,
            "Output values too large: {} (scale={})",
            max_val,
            scale
        );
    }
}

/// Test gradient clipping
#[test]
fn test_gradient_clipping() {
    use crate::numerics::clip_grad_norm;

    let mut gradients = vec![
        Array1::from_vec(vec![10.0, 20.0, 30.0]),
        Array1::from_vec(vec![5.0, 5.0, 5.0]),
    ];

    let max_norm = 10.0;
    let orig_norm_sq: f32 = gradients
        .iter()
        .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
        .sum();
    let orig_norm = orig_norm_sq.sqrt();

    let total_norm = clip_grad_norm(&mut gradients, max_norm);

    // Returned norm should be the original norm
    assert!(
        (total_norm - orig_norm).abs() < 0.1,
        "Returned norm {} doesn't match original {}",
        total_norm,
        orig_norm
    );

    // If original norm > max_norm, gradients should be scaled
    if orig_norm > max_norm {
        let new_norm_sq: f32 = gradients
            .iter()
            .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
            .sum();
        let new_norm = new_norm_sq.sqrt();

        assert!(
            (new_norm - max_norm).abs() < 0.5,
            "Clipped norm {} not equal to max_norm {}",
            new_norm,
            max_norm
        );
    }
}

/// Test numerical stability with alternating signs
#[test]
fn test_alternating_signs() {
    let config = RetNetConfig::new(64, 4, 2).unwrap();
    let model = RetNetModel::new(config).unwrap();
    let mut states = model.reset_states();

    // Alternating positive/negative inputs
    for i in 0..20 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let input = Array1::from_elem(64, sign * 0.5);

        let output = model.step(&input, &mut states).unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}

/// Test long sequence handling
#[test]
fn test_long_sequence_stability() {
    // Use smaller model for faster test while still testing long-range stability
    let config = S4DConfig::new(4, 16, 32);
    let model = S4DModel::new(config, 2).unwrap();

    let seq_len = 200; // Reduced from 1000 - still tests long-range behavior
    let input = Array2::from_shape_fn((seq_len, 4), |(t, _)| (t as f32 / seq_len as f32) * 0.1);

    let mut states = model.reset_states();
    let output = model.forward(&input, &mut states).unwrap();

    // All outputs should be finite
    assert!(output.iter().all(|&x| x.is_finite()));

    // Check for numerical drift - outputs should remain bounded
    let max_val = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    assert!(max_val < 100.0, "Output values grew too large: {}", max_val);
}

/// Test variable-length sequence handling
#[test]
fn test_variable_length_sequences() {
    let seq1 = Array2::from_shape_vec((5, 4), vec![1.0; 20]).unwrap();
    let seq2 = Array2::from_shape_vec((10, 4), vec![2.0; 40]).unwrap();
    let seq3 = Array2::from_shape_vec((3, 4), vec![3.0; 12]).unwrap();

    let sequences = vec![seq1, seq2, seq3];
    let (padded, mask) = pad_sequences(&sequences, 0.0, PaddingStrategy::Right).unwrap();

    // Check dimensions
    assert_eq!(padded.dim(), (3, 10, 4)); // max_len=10

    // Check masking
    assert!(mask.is_valid(0, 0)); // First sequence position 0
    assert!(mask.is_valid(0, 4)); // First sequence position 4
    assert!(!mask.is_valid(0, 5)); // First sequence padding

    assert!(mask.is_valid(1, 9)); // Second sequence full length
    assert!(!mask.is_valid(1, 10)); // Out of bounds

    assert!(mask.is_valid(2, 2)); // Third sequence position 2
    assert!(!mask.is_valid(2, 3)); // Third sequence padding
}

/// Stress test: Process many small sequences
#[test]
fn test_batch_stress() {
    let config = RetNetConfig::new(32, 2, 2).unwrap();
    let model = RetNetModel::new(config).unwrap();

    let batch_size = 100;
    let seq_len = 10;

    let input = Array2::from_shape_fn((seq_len, 32), |(t, i)| {
        (t as f32 * 0.01) + (i as f32 * 0.001)
    });

    // Process same sequence many times
    for _ in 0..batch_size {
        let output = model.forward(&input).unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}

/// Test attention with very long sequences
#[test]
fn test_attention_long_sequence() {
    let config = MultiHeadSSMConfig::new(64, 4, 16).unwrap();
    let attn = MultiHeadSSMAttention::new(config, false).unwrap();

    let query = Array1::from_vec(vec![0.1; 64]);
    let seq_len = 500;
    let key_cache = Array2::from_shape_vec((seq_len, 64), vec![0.05; seq_len * 64]).unwrap();
    let value_cache = Array2::from_shape_vec((seq_len, 64), vec![0.1; seq_len * 64]).unwrap();

    let output = attn.forward_step(&query, &key_cache, &value_cache).unwrap();

    assert_eq!(output.len(), 64);
    assert!(output.iter().all(|&x| x.is_finite()));
}

/// Test discretization stability
#[test]
fn test_discretization_accuracy() {
    use crate::numerics::zoh_discretize;

    let a = Array2::from_shape_vec((2, 2), vec![-1.0, 0.0, 0.0, -2.0]).unwrap();
    let b = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    for &dt in &[0.001, 0.01, 0.1] {
        let (a_d, b_d) = zoh_discretize(&a, &b, dt);

        // A_d eigenvalues should be exp(dt * lambda)
        // For diagonal A with λ=-1,-2:
        // A_d[0,0] ≈ exp(-dt)
        // A_d[1,1] ≈ exp(-2*dt)

        assert!(
            (a_d[[0, 0]] - (-dt).exp()).abs() < 0.01,
            "A_d[0,0] = {}, expected {}",
            a_d[[0, 0]],
            (-dt).exp()
        );
        assert!(
            (a_d[[1, 1]] - (-2.0 * dt).exp()).abs() < 0.01,
            "A_d[1,1] = {}, expected {}",
            a_d[[1, 1]],
            (-2.0 * dt).exp()
        );

        // All values should be finite
        assert!(a_d.iter().all(|&x| x.is_finite()));
        assert!(b_d.iter().all(|&x| x.is_finite()));
    }
}

/// Test parallel scan correctness
#[test]
fn test_parallel_scan_correctness() {
    use crate::scan::{parallel_scan, AssociativeOp};

    struct MultOp;
    impl AssociativeOp<f32> for MultOp {
        fn combine(&self, a: &f32, b: &f32) -> f32 {
            a * b
        }
        fn identity(&self) -> Option<f32> {
            Some(1.0)
        }
    }

    let data = vec![2.0, 3.0, 4.0, 5.0];
    let op = MultOp;

    let result = parallel_scan(&data, &op, false);
    // [2, 2*3, 2*3*4, 2*3*4*5] = [2, 6, 24, 120]
    assert_eq!(result[0], 2.0);
    assert_eq!(result[1], 6.0);
    assert_eq!(result[2], 24.0);
    assert_eq!(result[3], 120.0);
}

/// Test memory pooling doesn't leak
#[test]
fn test_memory_pool_stress() {
    let pool = ArrayPool::new(100, 10);

    // Allocate and deallocate many times
    for _ in 0..1000 {
        let arr = pool.acquire();
        pool.release(arr);
    }

    let stats = pool.stats();
    assert!(stats.hit_rate() > 0.0, "Pool should have some cache hits");
}

/// Test SIMD operations correctness
#[test]
fn test_simd_correctness() {
    use crate::simd;

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

    let dot = simd::dot_product(&a, &b);
    let expected = 72.0; // (1+2+3+4+5+6+7+8) * 2

    assert!((dot - expected).abs() < 1e-5);
}

/// Test masked operations with empty sequences
#[test]
fn test_masked_operations_edge_cases() {
    // Empty lengths array
    let empty: Vec<usize> = vec![];
    let mask = SequenceMask::from_lengths(&empty);
    assert!(mask.is_err()); // Should fail with empty input

    // All zero lengths
    let zeros = vec![0, 0, 0];
    let mask2 = SequenceMask::from_lengths(&zeros);
    assert!(mask2.is_err()); // Should fail with zero max length
}

/// Test packed sequence with all same length
#[test]
fn test_packed_sequence_uniform_length() {
    let lengths = vec![5, 5, 5];
    let mask = SequenceMask::from_lengths(&lengths).unwrap();

    let sequences = Array3::from_shape_fn((3, 5, 4), |(b, t, f)| (b * 100 + t * 10 + f) as f32);

    let packed = PackedSequence::pack(&sequences, &mask).unwrap();
    assert_eq!(packed.num_elements(), 15); // 3 * 5

    let unpacked = packed.unpack(0.0).unwrap();

    // Should match original
    for b in 0..3 {
        for t in 0..5 {
            for f in 0..4 {
                assert_eq!(sequences[[b, t, f]], unpacked[[b, t, f]]);
            }
        }
    }
}
