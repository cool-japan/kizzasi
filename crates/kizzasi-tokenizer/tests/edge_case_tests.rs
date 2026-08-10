//! Edge case tests for robustness and error handling.
//!
//! These tests verify that tokenizers handle edge cases gracefully,
//! including empty inputs, extreme values, boundary conditions, and error cases.

use kizzasi_tokenizer::metrics::{CompressionMetrics, QualityMetrics};
use kizzasi_tokenizer::*;
use scirs2_core::ndarray::{s, Array1};

/// Test single-element signal
#[test]
fn test_single_element_signal() {
    let tokenizer = ContinuousTokenizer::new(1, 1);
    let signal = Array1::from_vec(vec![0.5]);

    let encoded = tokenizer.encode(&signal).expect("Encoding failed");
    assert!(!encoded.is_empty());

    let decoded = tokenizer.decode(&encoded).expect("Decoding failed");
    assert_eq!(decoded.len(), 1);
}

/// Test extreme large values
#[test]
fn test_extreme_large_values() {
    let tokenizer = ContinuousTokenizer::new(16, 8);
    let signal = Array1::from_vec(vec![
        1000.0, -1000.0, 500.0, -500.0, 0.0, 100.0, -100.0, 0.0, 50.0, -50.0, 200.0, -200.0, 10.0,
        -10.0, 1.0, -1.0,
    ]);

    let encoded = tokenizer.encode(&signal).expect("Encoding failed");
    let decoded = tokenizer.decode(&encoded).expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());
}

/// Test all-zero signal
#[test]
fn test_all_zeros() {
    let tokenizer = ContinuousTokenizer::new(64, 32);
    let signal = Array1::zeros(64);

    let encoded = tokenizer.encode(&signal).expect("Encoding failed");
    let decoded = tokenizer.decode(&encoded).expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());
}

/// Test constant signal
#[test]
fn test_constant_signal() {
    let tokenizer = ContinuousTokenizer::new(64, 32);
    let signal = Array1::from_elem(64, 0.7);

    let encoded = tokenizer.encode(&signal).expect("Encoding failed");
    let decoded = tokenizer.decode(&encoded).expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());
}

/// Test alternating signal
#[test]
fn test_alternating_signal() {
    let tokenizer = ContinuousTokenizer::new(128, 64);
    let signal: Vec<f32> = (0..128)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let signal = Array1::from_vec(signal);

    let encoded = tokenizer.encode(&signal).expect("Encoding failed");
    let decoded = tokenizer.decode(&encoded).expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());
}

/// Test μ-law with boundary values
#[test]
fn test_mulaw_boundaries() {
    let codec = MuLawCodec::new(8);
    let signal = Array1::from_vec(vec![-1.0, -0.99, 0.0, 0.99, 1.0]);

    let encoded = codec.encode(&signal).expect("Encoding failed");
    let decoded = codec.decode(&encoded).expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());
    for &val in decoded.iter() {
        assert!((-1.0..=1.0).contains(&val));
    }
}

/// Test μ-law with out-of-range values
#[test]
fn test_mulaw_out_of_range() {
    let codec = MuLawCodec::new(8);
    let signal = Array1::from_vec(vec![-2.0, 2.0, -1.5, 1.5, 0.0]);

    let encoded = codec.encode(&signal).expect("Encoding failed");
    let decoded = codec.decode(&encoded).expect("Decoding failed");

    for &val in decoded.iter() {
        assert!((-1.0..=1.0).contains(&val));
    }
}

/// Test adaptive quantizer with no variance
#[test]
fn test_adaptive_no_variance() {
    let quantizer = AdaptiveQuantizer::new(8, 16, 0.5, -1.0, 1.0).expect("Creation failed");
    let signal = Array1::from_elem(64, 0.5);

    let encoded = quantizer.encode(&signal).expect("Encoding failed");
    let decoded = quantizer.decode(&encoded).expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());
}

/// Test dead-zone quantizer
#[test]
fn test_deadzone_quantizer() {
    let quantizer = DeadZoneQuantizer::new(8, 0.1, -1.0, 1.0).expect("Creation failed");
    let signal = Array1::from_vec(vec![0.8, -0.8, 0.05, -0.05, 0.9, -0.9, 0.02, -0.02]);

    let encoded = quantizer.encode(&signal).expect("Encoding failed");
    let decoded = quantizer.decode(&encoded).expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());
}

/// Test VQ-VAE with tiny codebook
#[test]
#[cfg(feature = "vqvae")]
fn test_vqvae_tiny_codebook() {
    let config = VQConfig {
        codebook_size: 2,
        embed_dim: 4,
        commitment_beta: 0.25,
        ema_decay: 0.99,
        epsilon: 1e-5,
        use_ema: false,
    };

    let tokenizer = VQVAETokenizer::new(16, config);
    let signal = Array1::linspace(0.0, 1.0, 16);

    let encoded = tokenizer.encode(&signal).expect("Encoding failed");
    let decoded = tokenizer.decode(&encoded).expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());
}

/// Test wavelet with power-of-2
#[test]
fn test_wavelet_power_of_2() {
    let config = WaveletConfig {
        levels: 3,
        family: WaveletFamily::Haar,
        bits: 8,
    };

    let tokenizer = WaveletTokenizer::new(config).expect("Creation failed");
    let signal = Array1::linspace(0.0, 1.0, 128);

    let encoded = tokenizer.encode(&signal).expect("Encoding failed");
    let decoded = tokenizer.decode(&encoded).expect("Decoding failed");

    assert_eq!(decoded.len(), signal.len());
}

/// Test transformer with tiny sequence
#[test]
fn test_transformer_tiny() {
    let config = TransformerConfig {
        input_dim: 8,
        embed_dim: 16,
        num_heads: 2,
        num_encoder_layers: 1,
        num_decoder_layers: 1,
        feedforward_dim: 32,
        dropout: 0.0,
        max_seq_len: 10,
    };

    let tokenizer = TransformerTokenizer::new(config).expect("Creation failed");
    let signal = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

    let encoded = tokenizer.encode(&signal).expect("Encoding failed");
    let decoded = tokenizer.decode(&encoded).expect("Decoding failed");

    assert!(decoded.len() >= signal.len());
}

/// Test MSM with full masking
#[test]
fn test_msm_full_masking() {
    let config = MSMConfig {
        mask_ratio: 1.0,
        mask_length: 4,
        signal_dim: 32,
        embed_dim: 16,
        learning_rate: 0.01,
        epochs: 1,
        batch_size: 4,
    };

    let mut msm = MaskedSignalModeling::new(config).expect("Creation failed");
    let signals: Vec<Array1<f32>> = vec![Array1::linspace(0.0, 1.0, 32)];

    let losses = msm.pretrain(&signals, 1);
    assert!(losses.is_ok());
}

/// Test quality metrics with identical signals
#[test]
fn test_quality_identical() {
    let signal = Array1::linspace(0.0, 1.0, 100);
    let metrics = QualityMetrics::compute(&signal, &signal).expect("Metrics failed");

    assert!(metrics.mse < 1e-10);
    assert!(metrics.mae < 1e-10);
}

/// Test compression metrics
#[test]
fn test_compression_metrics() {
    let metrics = CompressionMetrics::compute(1000, 16, 2000);
    // 1000 samples * 16 bits = 16000 bits original
    // 2000 bytes = 16000 bits compressed
    // compression_ratio = 16000 / 16000 = 1.0
    assert!((metrics.compression_ratio - 1.0).abs() < 1e-6);
}

/// Test memory profiler
#[test]
fn test_profiler_basics() {
    let mut profiler = MemoryProfiler::new();
    profiler.record_allocation("test", 1024);
    assert_eq!(profiler.current_memory(), 1024);
    assert_eq!(profiler.peak_memory(), 1024);
}

/// Test invalid config detection
#[test]
fn test_invalid_configs() {
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
}

/// LinearQuantizer with empty input should return empty output or a clear error.
#[test]
fn test_linear_quantizer_empty_input() {
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).expect("Creation failed");
    let signal = Array1::<f32>::zeros(0);
    match quantizer.encode(&signal) {
        Ok(encoded) => {
            assert!(
                encoded.is_empty(),
                "empty input should yield empty encoding"
            );
        }
        Err(_) => { /* graceful error on empty input is also fine */ }
    }
}

/// μ-law with NaN input — should either propagate NaN or error, not panic.
#[test]
fn test_mulaw_nan_input() {
    let codec = MuLawCodec::new(8);
    let signal = Array1::from_vec(vec![f32::NAN, 0.0, 1.0]);
    // Must not panic. Either encodes (NaN propagation) or errors.
    let _ = codec.encode(&signal);
}

/// μ-law with ±Infinity — must not panic.
#[test]
fn test_mulaw_infinity_input() {
    let codec = MuLawCodec::new(8);
    let signal = Array1::from_vec(vec![f32::INFINITY, f32::NEG_INFINITY, 0.5]);
    let _ = codec.encode(&signal);
}

/// Linear quantizer with NaN input — must not panic.
#[test]
fn test_linear_quantizer_nan_input() {
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).expect("Creation failed");
    let signal = Array1::from_vec(vec![0.0, f32::NAN, 1.0, f32::NAN]);
    let _ = quantizer.encode(&signal);
}

/// LinearQuantizer range boundary: value exactly at min and max.
#[test]
fn test_linear_quantizer_exact_boundaries() {
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 4).expect("Creation failed");
    let signal = Array1::from_vec(vec![-1.0, 1.0]);
    let encoded = quantizer.encode(&signal).expect("Encoding failed");
    let decoded = quantizer.decode(&encoded).expect("Decoding failed");
    assert_eq!(decoded.len(), 2);
    for &v in decoded.iter() {
        assert!(
            (-1.0f32..=1.0).contains(&v),
            "decoded value {v} out of range"
        );
    }
}

/// Very small (subnormal) values should not cause issues.
#[test]
fn test_linear_quantizer_subnormal_values() {
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).expect("Creation failed");
    let signal = Array1::from_vec(vec![f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 0.0]);
    let encoded = quantizer.encode(&signal).expect("Encoding failed");
    let decoded = quantizer.decode(&encoded).expect("Decoding failed");
    assert_eq!(decoded.len(), 3);
}

/// Streaming tokenizer: chunk boundary at exact frame size.
#[test]
fn test_streaming_exact_frame_boundary() {
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).expect("Creation failed");
    // StreamingTokenizer::new(tokenizer, chunk_size, overlap) -> Result
    let streamer =
        StreamingTokenizer::new(quantizer, 4, 0).expect("StreamingTokenizer creation failed");
    // Feed exactly 4 samples (one complete frame).
    let chunk = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let output = streamer
        .encode_streaming(&chunk)
        .expect("Streaming encode failed");
    // Should produce at least one chunk of output.
    assert!(
        !output.is_empty(),
        "exact-frame chunk should produce output"
    );
}

/// AdaptiveQuantizer with max_bits = min_bits (fixed rate).
#[test]
fn test_adaptive_quantizer_fixed_rate() {
    let quantizer = AdaptiveQuantizer::new(8, 8, 0.5, -1.0, 1.0).expect("Creation failed");
    let signal = Array1::linspace(-1.0, 1.0, 32);
    let encoded = quantizer.encode(&signal).expect("Encoding failed");
    let decoded = quantizer.decode(&encoded).expect("Decoding failed");
    assert_eq!(decoded.len(), signal.len());
}

/// Test very long signal processing
#[test]
fn test_very_long_signal() {
    let tokenizer = ContinuousTokenizer::new(1024, 512);
    let signal = Array1::linspace(-1.0, 1.0, 10000);

    let chunk_size = 1024;
    for i in (0..signal.len()).step_by(chunk_size) {
        let end = (i + chunk_size).min(signal.len());
        let chunk = signal.slice(s![i..end]).to_owned();

        if chunk.len() == chunk_size {
            let encoded = tokenizer.encode(&chunk).expect("Encoding failed");
            let decoded = tokenizer.decode(&encoded).expect("Decoding failed");
            assert_eq!(decoded.len(), chunk.len());
        }
    }
}
