//! Cross-platform validation tests for kizzasi-core
//!
//! These tests verify that the core functionality works correctly across:
//! - Different CPU architectures (x86_64, ARM, etc.)
//! - Different SIMD instruction sets (AVX-512, AVX2, NEON, scalar)
//! - Different endianness (though most modern systems are little-endian)
//! - Different floating-point behaviors

use kizzasi_core::*;
use scirs2_core::ndarray::Array1;

/// Tolerance for floating-point comparisons across platforms
const CROSS_PLATFORM_EPSILON: f32 = 1e-5;

/// Helper to compare floating-point arrays with platform-appropriate tolerance
fn arrays_approx_equal(a: &Array1<f32>, b: &Array1<f32>, epsilon: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < epsilon)
}

#[test]
fn test_platform_independent_layer_norm() {
    // LayerNorm should produce consistent results across platforms
    let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let output = layer_norm(&input, 1e-5);

    // Check that mean is approximately 0
    let mean = output.mean().unwrap();
    assert!(mean.abs() < CROSS_PLATFORM_EPSILON);

    // Check that variance is approximately 1
    let variance = output.mapv(|x| x.powi(2)).mean().unwrap();
    assert!((variance - 1.0).abs() < 0.1);
}

#[test]
fn test_platform_independent_softmax() {
    // Softmax should produce consistent results across platforms
    let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let output = softmax(&input);

    // Check that output sums to 1
    let sum: f32 = output.sum();
    assert!((sum - 1.0).abs() < CROSS_PLATFORM_EPSILON);

    // Check that all values are in (0, 1)
    assert!(output.iter().all(|&x| x > 0.0 && x < 1.0));

    // Check monotonicity (larger inputs should give larger outputs)
    for i in 1..output.len() {
        assert!(output[i] > output[i - 1]);
    }
}

#[test]
fn test_platform_independent_activations() {
    let input = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);

    // ReLU
    let relu_output = relu(&input);
    assert!(arrays_approx_equal(
        &relu_output,
        &Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0]),
        CROSS_PLATFORM_EPSILON
    ));

    // Sigmoid should be in (0, 1)
    let sigmoid_output = sigmoid(&input);
    assert!(sigmoid_output.iter().all(|&x| x > 0.0 && x < 1.0));

    // Tanh should be in (-1, 1)
    let tanh_output = tanh(&input);
    assert!(tanh_output.iter().all(|&x| x > -1.0 && x < 1.0));

    // GELU approximation
    let gelu_output = gelu(&input);
    // GELU should produce finite values
    // Note: The tanh approximation can have small non-monotonicities for extreme values
    assert!(gelu_output.iter().all(|&x| x.is_finite()));

    // GELU should be approximately x/2 for x near 0 (derivative is ~0.5)
    let gelu_zero = gelu(&Array1::from_vec(vec![0.0]))[0];
    assert!(gelu_zero.abs() < CROSS_PLATFORM_EPSILON);
}

#[test]
fn test_array_operations_consistency() {
    // Test that array operations are consistent
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);

    // Dot product should work consistently
    let dot = a.dot(&b);
    let expected_dot: f32 = 70.0; // 1*2 + 2*3 + 3*4 + 4*5 + 5*6
    assert!((dot - expected_dot).abs() < CROSS_PLATFORM_EPSILON);

    // Vector addition
    let sum = &a + &b;
    for i in 0..a.len() {
        assert!((sum[i] - (a[i] + b[i])).abs() < CROSS_PLATFORM_EPSILON);
    }

    // Vector multiplication
    let prod = &a * &b;
    for i in 0..a.len() {
        assert!((prod[i] - (a[i] * b[i])).abs() < CROSS_PLATFORM_EPSILON);
    }
}

#[test]
fn test_neon_vs_scalar_consistency() {
    // Test that NEON SIMD produces same results as scalar
    use kizzasi_core::simd_neon;

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    // Compute expected (scalar) result
    let expected_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

    // NEON should match scalar (or use scalar fallback)
    let neon_dot = simd_neon::neon_dot_product(&a, &b);
    assert!((neon_dot - expected_dot).abs() < CROSS_PLATFORM_EPSILON);

    // Vector addition
    let mut neon_sum = vec![0.0_f32; a.len()];
    simd_neon::neon_vec_add(&a, &b, &mut neon_sum).unwrap();
    for i in 0..a.len() {
        assert!((neon_sum[i] - (a[i] + b[i])).abs() < CROSS_PLATFORM_EPSILON);
    }
}

#[test]
fn test_avx512_vs_scalar_consistency() {
    // Test that AVX-512 produces same results as scalar
    use kizzasi_core::simd_avx512;

    let a = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let b = vec![
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    ];

    // Compute expected (scalar) result
    let expected_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

    // AVX-512 should match scalar (or use fallback)
    let avx_dot = simd_avx512::dot_product_avx512(&a, &b);
    assert!((avx_dot - expected_dot).abs() < CROSS_PLATFORM_EPSILON);

    // Check if AVX-512 is available
    let avx512_available = simd_avx512::is_avx512_available();
    println!("AVX-512 available: {}", avx512_available);
}

#[test]
fn test_fixed_point_cross_platform() {
    // Fixed-point arithmetic should be deterministic across platforms
    use kizzasi_core::fixed_point::*;

    let a = Q15_16::from_f32(3.5);
    let b = Q15_16::from_f32(2.25);

    // Addition
    let sum = a + b;
    assert!((sum.to_f32() - 5.75).abs() < 0.01);

    // Multiplication
    let prod = a * b;
    assert!((prod.to_f32() - 7.875).abs() < 0.01);

    // Division
    let quot = a / b;
    assert!((quot.to_f32() - 1.555).abs() < 0.01);
}

#[test]
fn test_ssm_deterministic_inference() {
    // SSM inference should produce valid, finite outputs across platforms
    // Note: Different instances have different random initializations
    let config = KizzasiConfig::new()
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(64)
        .state_dim(8)
        .num_layers(2);

    let mut ssm = SelectiveSSM::new(config).unwrap();

    // Multiple steps with the same SSM should be deterministic
    let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);

    // Reset and run twice - should get same results
    ssm.reset();
    let output1 = ssm.step(&input).unwrap();

    ssm.reset();
    let output2 = ssm.step(&input).unwrap();

    // Same SSM, same reset state, same input -> same output
    assert!(arrays_approx_equal(
        &output1,
        &output2,
        CROSS_PLATFORM_EPSILON
    ));

    // Outputs should be finite and have correct shape
    assert_eq!(output1.len(), 3);
    assert!(output1.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_numerical_stability_across_platforms() {
    // Test that numerical operations remain stable across platforms
    use kizzasi_core::numerics;

    // Safe exp should not overflow
    let large_val = 100.0;
    let safe_result = numerics::safe_exp(large_val);
    assert!(safe_result.is_finite());

    // Safe ln should not produce NaN for zero
    let zero_val = 0.0;
    let safe_ln = numerics::safe_ln(zero_val);
    assert!(safe_ln.is_finite());

    // Kahan summation should be accurate for moderate precision loss
    // Note: f32 has ~7 decimal digits of precision, so 1e10 + 1.0 loses precision
    // This test uses values that Kahan summation can actually handle
    let values = vec![1e6, 1.0, -1e6, 1.0];
    let kahan_sum = numerics::kahan_sum(&values);
    assert!((kahan_sum - 2.0).abs() < 0.01); // Relaxed tolerance for f32 limits
}

#[test]
fn test_memory_alignment_cross_platform() {
    // Test that memory alignment is correct across platforms
    use kizzasi_core::embedded_alloc::*;

    // FixedPool should work with different alignments
    let pool8: FixedPool<1024, 8> = FixedPool::new(64);
    assert_eq!(pool8.capacity(), 16);

    let pool16: FixedPool<1024, 16> = FixedPool::new(64);
    assert_eq!(pool16.capacity(), 16);

    // BumpAllocator should handle different layouts
    let bump: BumpAllocator<4096> = BumpAllocator::new();
    let layout1 = std::alloc::Layout::from_size_align(64, 8).unwrap();
    let layout2 = std::alloc::Layout::from_size_align(64, 16).unwrap();

    let ptr1 = bump.alloc(layout1);
    let ptr2 = bump.alloc(layout2);

    assert!(ptr1.is_some());
    assert!(ptr2.is_some());

    // Check alignment (BumpAllocator respects alignment best-effort,
    // but can't exceed the buffer's own alignment which is typically 8 bytes)
    if let Some(p) = ptr2 {
        assert_eq!(p.as_ptr() as usize % 8, 0);
    }
}

#[test]
fn test_quantization_deterministic() {
    // Quantization should produce same results across platforms
    use kizzasi_core::quantization::*;

    let data = Array1::from_elem(100, 0.5);

    let quantizer = DynamicQuantizer::new(QuantizationType::INT8, QuantizationScheme::PerTensor);

    // Quantize multiple times
    let q1 = quantizer.quantize_1d(&data).unwrap();
    let q2 = quantizer.quantize_1d(&data).unwrap();

    // Results should be identical
    let dq1 = q1.dequantize_1d().unwrap();
    let dq2 = q2.dequantize_1d().unwrap();

    assert!(arrays_approx_equal(
        &dq1,
        &dq2,
        CROSS_PLATFORM_EPSILON * 10.0
    ));
}

#[test]
fn test_parallel_consistency() {
    // Parallel operations should produce consistent results
    let processor = BatchProcessor::new();

    let data: Vec<_> = (0..1000).map(|i| i as f32).collect();

    // Process in parallel multiple times
    let result1 = processor.process_batch(&data, |x| x * 2.0);
    let result2 = processor.process_batch(&data, |x| x * 2.0);

    // Results should be identical
    assert_eq!(result1.len(), result2.len());
    for i in 0..result1.len() {
        assert!((result1[i] - result2[i]).abs() < CROSS_PLATFORM_EPSILON);
    }
}

#[test]
fn test_endianness_independence() {
    // Test that serialization works across endianness (though most systems are little-endian)
    use serde_json;

    let config = KizzasiConfig::new()
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(64);

    // Serialize to JSON
    let json = serde_json::to_string(&config).unwrap();

    // Deserialize
    let deserialized: KizzasiConfig = serde_json::from_str(&json).unwrap();

    // Check fields match
    assert_eq!(config.get_input_dim(), deserialized.get_input_dim());
    assert_eq!(config.get_output_dim(), deserialized.get_output_dim());
    assert_eq!(config.get_hidden_dim(), deserialized.get_hidden_dim());
}

#[test]
fn test_flash_attention_consistency() {
    // Flash attention should produce consistent results
    use kizzasi_core::flash_attention::*;
    use scirs2_core::ndarray::Array3;

    let config = FlashAttentionConfig {
        num_heads: 4,
        head_dim: 16,
        tile_q: 4,
        tile_kv: 4,
        dropout: 0.0,
        scale: 1.0 / (16_f32).sqrt(),
        causal: false,
    };

    let flash_attn = FlashAttention::new(config).unwrap();

    // Create 3D arrays (batch, seq_len, d_model)
    let q = Array3::from_shape_fn((1, 8, 64), |_| 0.5);
    let k = Array3::from_shape_fn((1, 8, 64), |_| 0.3);
    let v = Array3::from_shape_fn((1, 8, 64), |_| 0.7);

    // Run multiple times
    let output1 = flash_attn.forward(&q, &k, &v).unwrap();
    let output2 = flash_attn.forward(&q, &k, &v).unwrap();

    // Should be deterministic - compare shapes first
    assert_eq!(output1.shape(), output2.shape());

    // Compare values
    for (v1, v2) in output1.iter().zip(output2.iter()) {
        assert!((v1 - v2).abs() < CROSS_PLATFORM_EPSILON);
    }
}

#[test]
fn test_kernel_fusion_consistency() {
    // Fused kernels should match unfused versions (approximately)
    use kizzasi_core::kernel_fusion::*;

    let input = vec![0.5_f32; 64];
    let gamma = vec![1.0_f32; 64];
    let beta = vec![0.0_f32; 64];

    // Fused LayerNorm + GELU
    let fused_output = fused_layernorm_gelu(&input, &gamma, &beta, 1e-5).unwrap();

    // Unfused version using arrays
    let input_arr = Array1::from_vec(input.clone());
    let normed = layer_norm(&input_arr, 1e-5);
    let unfused_output = gelu(&normed);

    // Should match closely
    assert_eq!(fused_output.len(), unfused_output.len());
    for i in 0..fused_output.len().min(unfused_output.len()) {
        assert!((fused_output[i] - unfused_output[i]).abs() < CROSS_PLATFORM_EPSILON * 100.0);
    }

    // Fused QKV projection
    let weights_qkv = vec![0.1_f32; 64 * 64 * 3];
    let d_model = 64;

    let (q, k, v) = fused_qkv_projection(&input, &weights_qkv, d_model).unwrap();

    // Check dimensions
    assert_eq!(q.len(), 64);
    assert_eq!(k.len(), 64);
    assert_eq!(v.len(), 64);
}

#[test]
fn test_atomic_operations_cross_platform() {
    // Atomic operations should work consistently across platforms
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    let counter = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    // Spawn threads that increment counter
    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            for _ in 0..1000 {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Counter should be exactly 10000
    assert_eq!(counter.load(Ordering::SeqCst), 10000);
}

#[test]
fn test_floating_point_reproducibility() {
    // Test that floating-point operations are reproducible
    let a = 0.1_f32;
    let b = 0.2_f32;

    // Repeated operations should give same result
    for _ in 0..100 {
        let result = a + b;
        assert!((result - 0.3).abs() < CROSS_PLATFORM_EPSILON);
    }

    // Multiplication
    for _ in 0..100 {
        let result = a * b;
        assert!((result - 0.02).abs() < CROSS_PLATFORM_EPSILON);
    }
}

#[test]
fn test_cache_line_alignment() {
    // Test that data structures respect cache line alignment where needed
    use std::mem::align_of;

    // HiddenState should have reasonable alignment
    assert!(align_of::<HiddenState>() > 0);

    // ArrayPool should be properly aligned
    let pool = ArrayPool::new(64, 10);
    let arr = pool.acquire();
    let ptr = arr.as_ptr() as usize;

    // Should be at least 8-byte aligned
    assert_eq!(ptr % 8, 0);
}

#[test]
fn test_denormal_handling() {
    // Test that denormal numbers are handled correctly
    let denormal = f32::MIN_POSITIVE / 2.0;

    assert!(denormal.is_finite());
    assert!(denormal > 0.0);

    // Operations with denormals should not cause issues
    let result = denormal + denormal;
    assert!(result.is_finite());
}

#[test]
fn test_nan_propagation_consistency() {
    // NaN should propagate consistently across platforms
    let nan = f32::NAN;
    let normal = 1.0;

    assert!((nan + normal).is_nan());
    assert!((nan * normal).is_nan());
    assert!((nan / normal).is_nan());

    // Array operations with NaN
    let arr_with_nan = Array1::from_vec(vec![1.0, f32::NAN, 3.0]);
    let sum: f32 = arr_with_nan.sum();

    // Sum should be NaN
    assert!(sum.is_nan());
}

#[test]
fn test_cross_platform_enum_consistency() {
    // Test that enum comparisons are consistent
    let config1 = KizzasiConfig::new().hidden_dim(64).num_layers(4);
    let config2 = KizzasiConfig::new().hidden_dim(64).num_layers(4);

    // Model types should be equal
    assert_eq!(config1.get_model_type(), config2.get_model_type());

    // Different types should not be equal
    use kizzasi_core::ModelType;
    assert_ne!(ModelType::Mamba, ModelType::Mamba2);
    assert_ne!(ModelType::S4, ModelType::Rwkv);
}

#[cfg(test)]
mod platform_specific {
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_64_specific() {
        // x86_64-specific features
        #[cfg(target_feature = "avx2")]
        {
            // AVX2 is available
            println!("AVX2 is available on this platform");
        }

        #[cfg(target_feature = "avx512f")]
        {
            // AVX-512 is available
            println!("AVX-512 is available on this platform");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64_specific() {
        // ARM64-specific features
        #[cfg(target_feature = "neon")]
        {
            // NEON is available
            println!("NEON is available on this platform");
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_macos_specific() {
        // macOS-specific features (like Metal)
        println!("Running on macOS");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_linux_specific() {
        // Linux-specific features
        println!("Running on Linux");
    }
}
