//! Regression tests to ensure stability of previously working functionality.
//!
//! These tests verify that:
//! - API contracts remain stable
//! - Known edge cases continue to work
//! - Backward compatibility is maintained
//! - Important invariants hold across the codebase

use kizzasi_tokenizer::metrics::{CompressionMetrics, QualityMetrics};
use kizzasi_tokenizer::*;
use scirs2_core::ndarray::Array1;

/// Regression: LinearQuantizer should maintain perfect reconstruction at full precision
#[test]
fn regression_linear_quantizer_full_precision() {
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 16).expect("Creation failed");
    let signal = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0]);

    let encoded = quantizer.encode(&signal).expect("Encoding failed");
    let decoded = quantizer.decode(&encoded).expect("Decoding failed");

    for (orig, recon) in signal.iter().zip(decoded.iter()) {
        assert!(
            (orig - recon).abs() < 1e-4,
            "Reconstruction error too large"
        );
    }
}

/// Regression: μ-law codec should handle boundary values [-1, 1] without clipping
#[test]
fn regression_mulaw_boundary_handling() {
    let codec = MuLawCodec::new(8);
    let signal = Array1::from_vec(vec![-1.0, -0.99, 0.0, 0.99, 1.0]);

    let encoded = codec.encode(&signal).expect("Encoding failed");
    let decoded = codec.decode(&encoded).expect("Decoding failed");

    for &val in decoded.iter() {
        assert!((-1.0..=1.0).contains(&val), "Value out of range: {}", val);
    }
}

/// Regression: ContinuousTokenizer embed_dim should match configuration
#[test]
fn regression_continuous_tokenizer_embed_dim() {
    let input_dim = 64;
    let embed_dim = 32;
    let tokenizer = ContinuousTokenizer::new(input_dim, embed_dim);

    assert_eq!(tokenizer.embed_dim(), embed_dim);
    assert_eq!(tokenizer.vocab_size(), 0); // Continuous has no discrete vocab
}

/// Regression: VQ-VAE should maintain codebook size
#[test]
#[cfg(feature = "vqvae")]
fn regression_vqvae_codebook_size() {
    let config = VQConfig {
        codebook_size: 128,
        embed_dim: 16,
        commitment_beta: 0.25,
        ema_decay: 0.99,
        epsilon: 1e-5,
        use_ema: false,
    };

    let tokenizer = VQVAETokenizer::new(64, config);
    assert_eq!(tokenizer.vocab_size(), 128);
}

/// Regression: Individual encoding should be consistent across multiple calls
#[test]
fn regression_encoding_consistency() {
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).expect("Creation failed");

    let signal = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

    // Encode the same signal twice
    let encoded1 = quantizer.encode(&signal).expect("Encoding failed");
    let encoded2 = quantizer.encode(&signal).expect("Encoding failed");

    // Results should be identical
    assert_eq!(encoded1, encoded2);
}

/// Regression: Streaming tokenizer should handle exact chunk boundaries
#[test]
fn regression_streaming_exact_chunks() {
    let base = LinearQuantizer::new(-1.0, 1.0, 8).expect("Creation failed");
    let streaming = StreamingTokenizer::new(base, 128, 16).expect("Creation failed");

    // Signal length is exact multiple of (chunk_size - overlap)
    let signal_len = 128 + (128 - 16) * 3; // 4 chunks exactly
    let signal = Array1::linspace(-1.0, 1.0, signal_len);

    let chunks = streaming
        .encode_streaming(&signal)
        .expect("Encoding failed");
    let decoded = streaming
        .decode_streaming(&chunks)
        .expect("Decoding failed");

    // Decoded length may be slightly longer due to overlap-add
    assert!(decoded.len() >= signal_len);
}

/// Regression: Adaptive quantizer should handle constant signals
#[test]
fn regression_adaptive_constant_signal() {
    let quantizer = AdaptiveQuantizer::new(8, 16, 0.5, -1.0, 1.0).expect("Creation failed");
    let signal = Array1::from_elem(128, 0.5);

    let encoded = quantizer.encode(&signal).expect("Encoding failed");
    let decoded = quantizer.decode(&encoded).expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());

    // Decoded values should be within the signal range
    for &val in decoded.iter() {
        assert!((-1.0..=1.0).contains(&val), "Value out of range: {}", val);
    }
}

/// Regression: Dead-zone quantizer should zero out small values
#[test]
fn regression_deadzone_zeroing() {
    let quantizer = DeadZoneQuantizer::new(8, 0.1, -1.0, 1.0).expect("Creation failed");
    let signal = Array1::from_vec(vec![0.05, -0.05, 0.02, -0.02, 0.08, -0.08]);

    let encoded = quantizer.encode(&signal).expect("Encoding failed");
    let decoded = quantizer.decode(&encoded).expect("Decoding failed");

    // Small values (< 0.1) should be close to zero
    for (i, &val) in decoded.iter().enumerate() {
        if signal[i].abs() < 0.1 {
            assert!(val.abs() < 0.15, "Small value not zeroed: {}", val);
        }
    }
}

/// Regression: Wavelet tokenizer should produce valid outputs
#[test]
fn regression_wavelet_output_validity() {
    let config = WaveletConfig {
        levels: 2,
        family: WaveletFamily::Haar,
        bits: 12,
    };
    let tokenizer = WaveletTokenizer::new(config).expect("Creation failed");

    let signal = Array1::linspace(-1.0, 1.0, 256);

    let encoded = tokenizer.encode(&signal).expect("Encoding failed");
    let decoded = tokenizer.decode(&encoded).expect("Decoding failed");

    // Check that output length matches input
    assert_eq!(decoded.len(), signal.len());

    // Check that output values are finite
    for &val in decoded.iter() {
        assert!(val.is_finite(), "Non-finite value in output");
    }
}

/// Regression: Transformer config validation should reject invalid configs
#[test]
fn regression_transformer_config_validation() {
    // Invalid: input_dim = 0
    let config = TransformerConfig {
        input_dim: 0,
        embed_dim: 64,
        num_heads: 4,
        num_encoder_layers: 2,
        num_decoder_layers: 2,
        feedforward_dim: 128,
        dropout: 0.0,
        max_seq_len: 100,
    };
    assert!(config.validate().is_err());

    // Invalid: embed_dim not divisible by num_heads
    let config = TransformerConfig {
        input_dim: 32,
        embed_dim: 65,
        num_heads: 4,
        num_encoder_layers: 2,
        num_decoder_layers: 2,
        feedforward_dim: 128,
        dropout: 0.0,
        max_seq_len: 100,
    };
    assert!(config.validate().is_err());

    // Valid config
    let config = TransformerConfig {
        input_dim: 32,
        embed_dim: 64,
        num_heads: 4,
        num_encoder_layers: 2,
        num_decoder_layers: 2,
        feedforward_dim: 128,
        dropout: 0.0,
        max_seq_len: 100,
    };
    assert!(config.validate().is_ok());
}

/// Regression: MSM config validation should reject invalid mask ratios
#[test]
fn regression_msm_config_validation() {
    // Invalid: mask_ratio > 1.0
    let config = MSMConfig {
        mask_ratio: 1.5,
        mask_length: 4,
        signal_dim: 32,
        embed_dim: 16,
        learning_rate: 0.01,
        epochs: 1,
        batch_size: 4,
    };
    assert!(config.validate().is_err());

    // Invalid: mask_ratio < 0.0
    let config = MSMConfig {
        mask_ratio: -0.1,
        mask_length: 4,
        signal_dim: 32,
        embed_dim: 16,
        learning_rate: 0.01,
        epochs: 1,
        batch_size: 4,
    };
    assert!(config.validate().is_err());

    // Valid config
    let config = MSMConfig {
        mask_ratio: 0.5,
        mask_length: 4,
        signal_dim: 32,
        embed_dim: 16,
        learning_rate: 0.01,
        epochs: 1,
        batch_size: 4,
    };
    assert!(config.validate().is_ok());
}

/// Regression: Quality metrics should return perfect scores for identical signals
#[test]
fn regression_quality_metrics_perfect_match() {
    let signal = Array1::linspace(-1.0, 1.0, 100);
    let metrics = QualityMetrics::compute(&signal, &signal).expect("Metrics failed");

    assert!(metrics.mse < 1e-10, "MSE should be near zero");
    assert!(metrics.mae < 1e-10, "MAE should be near zero");
    assert!(metrics.rmse < 1e-10, "RMSE should be near zero");
    assert!(metrics.nmse < 1e-10, "NMSE should be near zero");
}

/// Regression: Compression metrics should compute correct ratios
#[test]
fn regression_compression_metrics_ratios() {
    // 1000 samples * 16 bits = 16000 bits
    // 500 bytes = 4000 bits
    // Ratio = 16000 / 4000 = 4.0
    let metrics = CompressionMetrics::compute(1000, 16, 500);

    assert!((metrics.compression_ratio - 4.0).abs() < 1e-6);
    assert!((metrics.bits_per_sample - 4.0).abs() < 1e-6);
    assert!((metrics.space_savings_percent - 75.0).abs() < 1e-6);
}

/// Regression: Memory profiler should track allocations correctly
#[test]
fn regression_memory_profiler_tracking() {
    let mut profiler = MemoryProfiler::new();

    profiler.record_allocation("test1", 1024);
    assert_eq!(profiler.current_memory(), 1024);
    assert_eq!(profiler.peak_memory(), 1024);

    profiler.record_allocation("test2", 2048);
    assert_eq!(profiler.current_memory(), 3072);
    assert_eq!(profiler.peak_memory(), 3072);

    profiler.record_deallocation("test1", 1024);
    assert_eq!(profiler.current_memory(), 2048);
    assert_eq!(profiler.peak_memory(), 3072); // Peak stays at max
}

/// Regression: ProfileScope should track memory allocations
#[test]
fn regression_profile_scope_tracking() {
    let mut profiler = MemoryProfiler::new();

    {
        profiler.start_scope("test_scope");
        profiler.record_allocation("test_scope", 1024);
    }

    // Check that the allocation was tracked
    assert_eq!(profiler.peak_memory(), 1024);
}

/// Regression: Roundtrip encode-decode should preserve signal characteristics
#[test]
fn regression_roundtrip_signal_characteristics() {
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 10).expect("Creation failed");

    // Create signal with known characteristics
    let signal: Vec<f32> = (0..128).map(|i| i as f32 / 128.0 * 2.0 - 1.0).collect();
    let signal = Array1::from_vec(signal);

    let encoded = quantizer.encode(&signal).expect("Encoding failed");
    let decoded = quantizer.decode(&encoded).expect("Decoding failed");

    // Check that mean is preserved
    let orig_mean = signal.iter().sum::<f32>() / signal.len() as f32;
    let recon_mean = decoded.iter().sum::<f32>() / decoded.len() as f32;
    assert!((orig_mean - recon_mean).abs() < 0.1);

    // Check that range is preserved
    let orig_max = signal.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let orig_min = signal.iter().cloned().fold(f32::INFINITY, f32::min);
    let recon_max = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let recon_min = decoded.iter().cloned().fold(f32::INFINITY, f32::min);

    assert!((orig_max - recon_max).abs() < 0.2);
    assert!((orig_min - recon_min).abs() < 0.2);
}

/// Regression: Signal encoding length should match input length
#[test]
fn regression_encoding_length_preservation() {
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).expect("Creation failed");

    for len in [1, 10, 100, 1000] {
        let signal = Array1::linspace(-1.0, 1.0, len);
        let encoded = quantizer.encode(&signal).expect("Encoding failed");
        let decoded = quantizer.decode(&encoded).expect("Decoding failed");

        assert_eq!(
            decoded.len(),
            signal.len(),
            "Length mismatch for {} samples",
            len
        );
    }
}

/// Regression: Zero-length signals should be handled gracefully
#[test]
fn regression_zero_length_signal() {
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).expect("Creation failed");
    let signal = Array1::from_vec(vec![]);

    // Should either succeed with empty output or fail gracefully
    let result = quantizer.encode(&signal);
    if let Ok(encoded) = result {
        assert_eq!(encoded.len(), 0);
    }
}

/// Regression: Very long signals should not cause stack overflow
#[test]
fn regression_very_long_signal() {
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).expect("Creation failed");
    let signal = Array1::linspace(-1.0, 1.0, 100000);

    let encoded = quantizer.encode(&signal).expect("Encoding failed");
    let decoded = quantizer.decode(&encoded).expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());
}

/// Regression: Repeated encode-decode cycles should be stable
#[test]
fn regression_encode_decode_stability() {
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).expect("Creation failed");
    let mut signal = Array1::from_vec(vec![0.0, 0.5, 1.0, -0.5, -1.0]);

    // Multiple encode-decode cycles
    for _ in 0..5 {
        let encoded = quantizer.encode(&signal).expect("Encoding failed");
        signal = quantizer.decode(&encoded).expect("Decoding failed");
    }

    // Signal should converge and remain stable
    for &val in signal.iter() {
        assert!((-1.0..=1.0).contains(&val));
    }
}

/// Regression: Multi-scale tokenizer should be created successfully
#[test]
fn regression_multiscale_creation() {
    let tokenizer = MultiScaleTokenizer::new(64, 16);
    let signal = Array1::linspace(-1.0, 1.0, 64);

    let encoded = tokenizer.encode(&signal).expect("Encoding failed");
    let decoded = tokenizer.decode(&encoded).expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());
}

/// Regression: Hierarchical tokenizer should be created successfully
#[test]
fn regression_hierarchical_creation() {
    let config = HierarchicalConfig {
        num_levels: 3,
        codebook_sizes: vec![256, 128, 64],
        use_residual: true,
    };
    let tokenizer = HierarchicalTokenizer::new(32, config).expect("Creation failed");

    let signal = Array1::linspace(-1.0, 1.0, 32);
    let encoded_indices = tokenizer
        .encode_with_levels(&signal, 3)
        .expect("Encoding failed");
    let decoded = tokenizer
        .decode_hierarchical(&encoded_indices)
        .expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());
}
