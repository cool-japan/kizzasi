//! Property-based tests for kizzasi-tokenizer
//!
//! This module contains comprehensive property-based tests using proptest to ensure
//! mathematical invariants and properties hold for all tokenizers.

#![cfg(feature = "vqvae")]

use kizzasi_tokenizer::*;
use proptest::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};

// ============================================================================
// Property Test Strategies
// ============================================================================

/// Generate a random signal with reasonable bounds
fn signal_strategy() -> impl Strategy<Value = Array1<f32>> {
    prop::collection::vec(-10.0f32..10.0f32, 8..256).prop_map(Array1::from_vec)
}

/// Generate a batch of signals
fn signal_batch_strategy() -> impl Strategy<Value = Vec<Array1<f32>>> {
    prop::collection::vec(prop::collection::vec(-10.0f32..10.0f32, 8..128), 1..16)
        .prop_map(|vecs| vecs.into_iter().map(Array1::from_vec).collect())
}

/// Generate a 2D signal (batch x features)
fn signal_2d_strategy() -> impl Strategy<Value = Array2<f32>> {
    (1usize..32, 4usize..64).prop_flat_map(|(batch, feat)| {
        prop::collection::vec(-10.0f32..10.0f32, batch * feat)
            .prop_map(move |v| Array2::from_shape_vec((batch, feat), v).unwrap())
    })
}

// ============================================================================
// Linear Quantizer Properties
// ============================================================================

proptest! {
    /// Property: Quantize-dequantize should not increase magnitude
    #[test]
    fn linear_quant_bounded_magnitude(signal in signal_strategy(), bits in 2u8..8) {
        // Find min/max of original signal
        let min_orig = signal.iter().copied().fold(f32::INFINITY, f32::min);
        let max_orig = signal.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let quantizer = LinearQuantizer::new(min_orig, max_orig, bits).unwrap();

        let encoded = quantizer.encode(&signal).unwrap();
        let decoded = quantizer.decode(&encoded).unwrap();

        // Decoded should be within original range
        for &val in decoded.iter() {
            prop_assert!(val >= min_orig - 1e-5);
            prop_assert!(val <= max_orig + 1e-5);
        }
    }

    /// Property: Quantization should preserve signal length
    #[test]
    fn linear_quant_length_preservation(signal in signal_strategy(), bits in 2u8..8) {
        let min_val = signal.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = signal.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let quantizer = LinearQuantizer::new(min_val, max_val, bits).unwrap();

        let encoded = quantizer.encode(&signal).unwrap();
        let decoded = quantizer.decode(&encoded).unwrap();

        prop_assert_eq!(signal.len(), decoded.len());
        prop_assert_eq!(signal.len(), encoded.len());
    }

    /// Property: Quantization levels should be used
    #[test]
    fn linear_quant_uses_levels(signal in signal_strategy(), bits in 2u8..8) {
        let min_val = signal.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = signal.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let num_levels = 1u32.wrapping_shl(bits as u32);
        let quantizer = LinearQuantizer::new(min_val, max_val, bits).unwrap();

        let encoded = quantizer.encode(&signal).unwrap();

        // All encoded values should be valid quantization indices
        for &idx in encoded.iter() {
            prop_assert!(idx < num_levels as f32);
        }
    }

    /// Property: Constant signal should encode to constant indices
    #[test]
    fn linear_quant_constant_signal(value in -10.0f32..10.0, len in 8usize..256, bits in 2u8..8) {
        let quantizer = LinearQuantizer::new(value - 0.1, value + 0.1, bits).unwrap();
        let signal = Array1::from_vec(vec![value; len]);

        let encoded = quantizer.encode(&signal).unwrap();

        // All indices should be the same for constant input
        let first = encoded[[0]];
        for i in 1..len {
            prop_assert!((encoded[[i]] - first).abs() < 1e-5);
        }
    }
}

// ============================================================================
// μ-law Codec Properties
// ============================================================================

proptest! {
    /// Property: μ-law should preserve signal length
    #[test]
    fn mulaw_length_preservation(signal in signal_strategy()) {
        let codec = MuLawCodec::new(8);

        let encoded = codec.encode(&signal).unwrap();
        let decoded = codec.decode(&encoded).unwrap();

        prop_assert_eq!(signal.len(), decoded.len());
        prop_assert_eq!(signal.len(), encoded.len());
    }

    /// Property: μ-law encoding should be monotonic for positive values
    #[test]
    fn mulaw_monotonicity_positive(x1 in 0.0f32..1.0, x2 in 0.0f32..1.0) {
        let codec = MuLawCodec::new(8);
        let s1 = Array1::from_vec(vec![x1]);
        let s2 = Array1::from_vec(vec![x2]);

        let e1 = codec.encode(&s1).unwrap()[[0]];
        let e2 = codec.encode(&s2).unwrap()[[0]];

        if x1 < x2 {
            prop_assert!(e1 <= e2);
        } else if x1 > x2 {
            prop_assert!(e1 >= e2);
        }
    }

    /// Property: μ-law should be symmetric around zero
    #[test]
    fn mulaw_symmetry(value in -1.0f32..1.0) {
        let codec = MuLawCodec::new(8);
        let pos = Array1::from_vec(vec![value]);
        let neg = Array1::from_vec(vec![-value]);

        let e_pos = codec.encode(&pos).unwrap()[[0]];
        let e_neg = codec.encode(&neg).unwrap()[[0]];

        // Encoded values should be symmetric (within quantization error)
        prop_assert!((e_pos as i32 + e_neg as i32).abs() <= 1);
    }

    /// Property: Zero input should encode to a consistent value
    #[test]
    fn mulaw_zero_encoding(_dummy in 0..1) {
        let codec = MuLawCodec::new(8);
        let zero_signal = Array1::from_vec(vec![0.0]);

        let encoded = codec.encode(&zero_signal).unwrap();

        // Zero should always encode to the same value
        prop_assert!(encoded[[0]].is_finite());
        prop_assert_eq!(encoded[[0]], 0.0);
    }
}

// ============================================================================
// Continuous Tokenizer Properties
// ============================================================================

proptest! {
    /// Property: Continuous tokenizer should preserve batch dimension
    #[test]
    fn continuous_batch_preservation(
        signal in signal_2d_strategy(),
        embed_dim in 4usize..32
    ) {
        let input_dim = signal.shape()[1];
        let tokenizer = ContinuousTokenizer::new(input_dim, embed_dim);

        let encoded = tokenizer.encode_batch(&signal).unwrap();

        // Batch size should be preserved
        prop_assert_eq!(signal.shape()[0], encoded.shape()[0]);
        // Embedding dimension should match
        prop_assert_eq!(embed_dim, encoded.shape()[1]);
    }

    /// Property: Encode-decode should preserve dimensions
    #[test]
    fn continuous_dimension_preservation(signal in signal_strategy(), embed_dim in 4usize..32) {
        let input_dim = signal.len();
        let tokenizer = ContinuousTokenizer::new(input_dim, embed_dim);

        let encoded = tokenizer.encode(&signal).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        prop_assert_eq!(signal.len(), decoded.len());
        prop_assert_eq!(embed_dim, encoded.len());
    }

    /// Property: Zero signal should encode to bounded embeddings
    #[test]
    fn continuous_zero_signal_bounded(input_dim in 4usize..64, embed_dim in 4usize..32) {
        let tokenizer = ContinuousTokenizer::new(input_dim, embed_dim);
        let zero_signal = Array1::zeros(input_dim);

        let encoded = tokenizer.encode(&zero_signal).unwrap();

        // Embeddings should be finite
        for &val in encoded.iter() {
            prop_assert!(val.is_finite());
        }
    }
}

// ============================================================================
// Multi-Scale Tokenizer Properties
// ============================================================================

proptest! {
    /// Property: Multi-scale should produce finite embeddings
    #[test]
    fn multiscale_finite_embeddings(
        signal in signal_strategy(),
        embed_dim in 4usize..32
    ) {
        let input_dim = signal.len();
        let tokenizer = MultiScaleTokenizer::new(input_dim, embed_dim);

        let encoded = tokenizer.encode(&signal).unwrap();

        // All embeddings should be finite
        for &val in encoded.iter() {
            prop_assert!(val.is_finite());
        }

        // Encoded dimension should be positive
        prop_assert!(!encoded.is_empty());
    }
}

// ============================================================================
// VQ-VAE Properties
// ============================================================================

proptest! {
    /// Property: VQ-VAE should return indices as f32
    #[test]
    fn vqvae_encoding_type(
        signal in signal_strategy(),
        codebook_size in 64usize..256,
        embed_dim in 4usize..16
    ) {
        let input_dim = signal.len();
        let config = VQConfig {
            codebook_size,
            embed_dim,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        };
        let tokenizer = VQVAETokenizer::new(input_dim, config);

        let encoded = tokenizer.encode(&signal).unwrap();

        // All values should be finite f32
        for &val in encoded.iter() {
            prop_assert!(val.is_finite());
        }
    }

    /// Property: VQ-VAE should preserve signal length in reconstruction
    #[test]
    fn vqvae_length_preservation(
        signal in signal_strategy(),
        codebook_size in 64usize..256,
        embed_dim in 4usize..16
    ) {
        let input_dim = signal.len();
        let config = VQConfig {
            codebook_size,
            embed_dim,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        };
        let tokenizer = VQVAETokenizer::new(input_dim, config);

        let encoded = tokenizer.encode(&signal).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        prop_assert_eq!(signal.len(), decoded.len());
    }

    /// Property: Repeated encoding should be deterministic (same input -> same output)
    #[test]
    fn vqvae_deterministic_encoding(
        signal in signal_strategy(),
        codebook_size in 64usize..256,
        embed_dim in 4usize..16
    ) {
        let input_dim = signal.len();
        let config = VQConfig {
            codebook_size,
            embed_dim,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        };
        let tokenizer = VQVAETokenizer::new(input_dim, config);

        let encoded1 = tokenizer.encode(&signal).unwrap();
        let encoded2 = tokenizer.encode(&signal).unwrap();

        // Same input should produce same encoding
        prop_assert_eq!(encoded1.len(), encoded2.len());
        for i in 0..encoded1.len() {
            prop_assert!((encoded1[[i]] - encoded2[[i]]).abs() < 1e-6);
        }
    }

    /// Property: VQ-VAE reconstruction should be bounded
    #[test]
    fn vqvae_reconstruction_bounded(
        signal in signal_strategy(),
        codebook_size in 64usize..256,
        embed_dim in 4usize..16
    ) {
        let input_dim = signal.len();
        let config = VQConfig {
            codebook_size,
            embed_dim,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        };
        let tokenizer = VQVAETokenizer::new(input_dim, config);

        // Initialize with the signal to ensure codebook is reasonable
        let _ = tokenizer.encode(&signal).unwrap();

        let encoded = tokenizer.encode(&signal).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        // Reconstruction should be finite
        for &val in decoded.iter() {
            prop_assert!(val.is_finite());
        }
    }
}

// ============================================================================
// Batch Processing Properties
// ============================================================================

proptest! {
    /// Property: Batch encoding should preserve dimensions
    #[test]
    fn batch_preserves_dimensions(
        signals in signal_batch_strategy(),
        bits in 2u8..8
    ) {
        if !signals.is_empty() {
            // Find global min/max
            let mut global_min = f32::INFINITY;
            let mut global_max = f32::NEG_INFINITY;
            for s in &signals {
                global_min = global_min.min(s.iter().copied().fold(f32::INFINITY, f32::min));
                global_max = global_max.max(s.iter().copied().fold(f32::NEG_INFINITY, f32::max));
            }

            let quantizer = LinearQuantizer::new(global_min, global_max, bits).unwrap();

            // Convert signals to batch format (2D array)
            let max_len = signals.iter().map(|s| s.len()).max().unwrap();
            let batch_size = signals.len();
            let mut batch_data = vec![0.0f32; batch_size * max_len];

            for (i, sig) in signals.iter().enumerate() {
                for (j, &val) in sig.iter().enumerate() {
                    batch_data[i * max_len + j] = val;
                }
            }

            let batch_array = Array2::from_shape_vec((batch_size, max_len), batch_data).unwrap();
            let encoded = quantizer.encode_batch(&batch_array).unwrap();
            let decoded = quantizer.decode_batch(&encoded).unwrap();

            // Check dimensions
            prop_assert_eq!(batch_array.shape(), decoded.shape());
        }
    }
}

// ============================================================================
// Specialized Tokenizers Properties
// ============================================================================

proptest! {
    /// Property: Wavelet tokenizer should preserve length
    #[test]
    fn wavelet_length_preservation(signal in signal_strategy(), levels in 1usize..3) {
        // Ensure signal length is a power of 2 for wavelet
        let len = signal.len();
        let power_of_2 = len.next_power_of_two();
        let mut padded = vec![0.0; power_of_2];
        padded[..len].copy_from_slice(signal.as_slice().unwrap());
        let signal = Array1::from_vec(padded);

        let config = WaveletConfig {
            levels,
            family: WaveletFamily::Haar,
            bits: 8,
        };
        let tokenizer = WaveletTokenizer::new(config).unwrap();

        let encoded = tokenizer.encode(&signal).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        prop_assert_eq!(signal.len(), decoded.len());
    }

    /// Property: DCT tokenizer should produce correct encoding dimension
    #[test]
    fn dct_encoding_dimension(signal in signal_strategy(), num_coeffs in 4usize..32) {
        let num_coeffs = num_coeffs.min(signal.len());
        let config = DCTConfig {
            num_coeffs,
            bits: 8,
        };
        let tokenizer = DCTTokenizer::new(config).unwrap();

        let encoded = tokenizer.encode(&signal).unwrap();

        // Encoded should have num_coeffs elements (quantized DCT coefficients)
        prop_assert_eq!(encoded.len(), num_coeffs);
    }

    /// Property: Fourier tokenizer should produce correct encoding dimension
    #[test]
    fn fourier_encoding_dimension(signal in signal_strategy(), num_bins in 16usize..128) {
        let num_bins = num_bins.min(signal.len());
        let config = FourierConfig {
            num_bins,
            magnitude_only: false,
            bits: 8,
        };
        let tokenizer = FourierTokenizer::new(config).unwrap();

        let encoded = tokenizer.encode(&signal).unwrap();

        // Encoded should have 2 * num_bins elements (magnitude and phase for each bin)
        prop_assert_eq!(encoded.len(), 2 * num_bins);
    }
}

// ============================================================================
// Advanced Quantization Properties
// ============================================================================

proptest! {
    /// Property: Adaptive quantizer should preserve length
    #[test]
    fn adaptive_quant_length_preservation(signal in signal_strategy()) {
        let min_val = signal.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = signal.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let quantizer = AdaptiveQuantizer::new(8, 32, 0.5, min_val, max_val).unwrap();

        let encoded = quantizer.encode(&signal).unwrap();
        let decoded = quantizer.decode(&encoded).unwrap();

        prop_assert_eq!(signal.len(), decoded.len());
        prop_assert_eq!(signal.len(), encoded.len());
    }

    /// Property: Dead zone quantizer should zero out small values
    #[test]
    fn deadzone_quant_zeros_small_values(threshold in 0.01f32..1.0) {
        let quantizer = DeadZoneQuantizer::new(8, threshold, -5.0, 5.0).unwrap();

        // Create signal with small values
        let small_signal = Array1::from_vec(vec![threshold * 0.5; 16]);
        let encoded = quantizer.encode(&small_signal).unwrap();
        let decoded = quantizer.decode(&encoded).unwrap();

        // All values should be quantized to near zero
        for &val in decoded.iter() {
            prop_assert!(val.abs() < threshold * 1.5);
        }
    }

    /// Property: Non-uniform quantizer should preserve length
    #[test]
    fn nonuniform_quant_length_preservation(signal in signal_strategy(), num_bins in 8usize..64) {
        let bin_edges: Vec<f32> = (0..=num_bins)
            .map(|i| -5.0 + 10.0 * i as f32 / num_bins as f32)
            .collect();
        let reconstruction_values: Vec<f32> = (0..num_bins)
            .map(|i| -5.0 + 10.0 * (i as f32 + 0.5) / num_bins as f32)
            .collect();

        let quantizer = NonUniformQuantizer::new(bin_edges, reconstruction_values).unwrap();

        let encoded = quantizer.encode(&signal).unwrap();
        let decoded = quantizer.decode(&encoded).unwrap();

        prop_assert_eq!(signal.len(), decoded.len());
        prop_assert_eq!(signal.len(), encoded.len());
    }
}

// ============================================================================
// Residual VQ Properties
// ============================================================================

proptest! {
    /// Property: RVQ should produce indices for each stage
    #[test]
    fn rvq_produces_stage_indices(
        signal in signal_strategy(),
        num_stages in 2usize..5,
        codebook_size in 32usize..128
    ) {
        let dim = signal.len();
        let config = VQConfig {
            codebook_size,
            embed_dim: dim,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        };
        let rvq = ResidualVQ::new(num_stages, config);

        // Encode with all stages
        let (indices, quantized) = rvq.encode(&signal).unwrap();

        // Should have one index per stage
        prop_assert_eq!(indices.len(), num_stages);
        prop_assert_eq!(quantized.len(), num_stages);

        // All indices should be valid
        for &idx in &indices {
            prop_assert!(idx < codebook_size);
        }
    }

    /// Property: RVQ should preserve signal dimension
    #[test]
    fn rvq_dimension_preservation(
        signal in signal_strategy(),
        num_stages in 2usize..4,
        codebook_size in 32usize..64
    ) {
        let dim = signal.len();
        let config = VQConfig {
            codebook_size,
            embed_dim: dim,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        };
        let rvq = ResidualVQ::new(num_stages, config);

        let (indices, _) = rvq.encode(&signal).unwrap();
        let decoded = rvq.decode(&indices).unwrap();

        prop_assert_eq!(signal.len(), decoded.len());
    }
}
