//! Round-trip correctness tests for scale-preserving transform tokenizers.
//!
//! These tests verify that the max_val header introduced in `encode` is properly
//! round-tripped through `decode`, so that signals with coefficient magnitudes
//! far from 1.0 are faithfully reconstructed.

use kizzasi_tokenizer::specialized::{
    DCTConfig, DCTTokenizer, FourierConfig, FourierTokenizer, WaveletConfig, WaveletFamily,
    WaveletTokenizer,
};
use kizzasi_tokenizer::SignalTokenizer;
use scirs2_core::ndarray::Array1;

/// Build a pure sine signal with the given amplitude `scale`.
fn make_signal(n: usize, scale: f32) -> Array1<f32> {
    Array1::from_shape_fn(n, |i| {
        scale * (i as f32 / n as f32 * 2.0 * std::f32::consts::PI).sin()
    })
}

// ============================================================================
// WaveletTokenizer
// ============================================================================

#[test]
fn test_wavelet_round_trip_non_unit_scale() {
    let config = WaveletConfig {
        bits: 8,
        levels: 3,
        family: WaveletFamily::Haar,
    };
    let bits = config.bits;
    let tokenizer = WaveletTokenizer::new(config).expect("WaveletTokenizer::new failed");

    // Use a power-of-2 length (required for multi-level wavelet decomposition).
    let signal = make_signal(64, 5.0); // scale = 5.0, not 1.0
    let encoded = tokenizer.encode(&signal).expect("encode failed");
    let decoded = tokenizer.decode(&encoded).expect("decode failed");

    assert_eq!(decoded.len(), signal.len());

    // token[0] is the max_val header.  For a scale-5 sinusoid the wavelet
    // coefficients' max absolute value should comfortably exceed 1.0.
    assert!(
        encoded[0] > 1.0,
        "Header max_val={} should be > 1.0 for a scale=5 signal",
        encoded[0]
    );

    // Round-trip error must be bounded by (roughly) one quantization step.
    // step = max_val / (2^(bits-1))
    let step = encoded[0] / (1u32 << (bits - 1)) as f32;
    for (x, y) in signal.iter().zip(decoded.iter()) {
        assert!(
            (x - y).abs() < step * 8.0,
            "Round-trip error too large at a sample: original={} decoded={}",
            x,
            y
        );
    }
}

#[test]
fn test_wavelet_scale_equivariance() {
    let config = WaveletConfig {
        bits: 8,
        levels: 3,
        family: WaveletFamily::Haar,
    };
    let tokenizer = WaveletTokenizer::new(config).expect("WaveletTokenizer::new failed");

    let signal_1 = make_signal(64, 1.0);
    let signal_k = make_signal(64, 5.0);

    let decoded_1 = tokenizer
        .decode(&tokenizer.encode(&signal_1).expect("encode"))
        .expect("decode");
    let decoded_k = tokenizer
        .decode(&tokenizer.encode(&signal_k).expect("encode"))
        .expect("decode");

    // decoded_k should be ≈ 5 × decoded_1 at every sample.
    for (a, b) in decoded_1.iter().zip(decoded_k.iter()) {
        assert!(
            (b - 5.0 * a).abs() < 0.5,
            "Scale equivariance violated: 5.0 * {} ≈ {}, got {}",
            a,
            5.0 * a,
            b
        );
    }
}

#[test]
fn test_wavelet_encode_length_has_header() {
    let config = WaveletConfig {
        bits: 8,
        levels: 2,
        family: WaveletFamily::Haar,
    };
    let tokenizer = WaveletTokenizer::new(config).expect("WaveletTokenizer::new failed");
    let signal = make_signal(32, 1.0);
    let encoded = tokenizer.encode(&signal).expect("encode failed");
    // encoded length == signal.len() + 1  (one header token)
    assert_eq!(
        encoded.len(),
        signal.len() + 1,
        "Wavelet encoded length should be signal.len() + 1 (header)"
    );
}

// ============================================================================
// DCTTokenizer
// ============================================================================

#[test]
fn test_dct_round_trip_non_unit_scale() {
    let config = DCTConfig {
        bits: 8,
        num_coeffs: 16,
    };
    let num_coeffs = config.num_coeffs;
    let tokenizer = DCTTokenizer::new(config).expect("DCTTokenizer::new failed");

    let signal = make_signal(64, 3.0); // scale = 3.0
    let encoded = tokenizer.encode(&signal).expect("encode failed");
    let decoded = tokenizer.decode(&encoded).expect("decode failed");

    // token[0] is max_val; for a scale-3 sinusoid DCT coefficients exceed 1.0.
    assert!(
        encoded[0] > 1.0,
        "DCT header max_val={} should be > 1.0 for scale=3 signal",
        encoded[0]
    );

    // encoded contains 1 header + num_coeffs quantized values.
    assert_eq!(encoded.len(), num_coeffs + 1);

    // decoded reconstructs to num_coeffs samples (lossy truncation).
    assert_eq!(decoded.len(), num_coeffs);
    assert!(
        decoded.iter().all(|x| x.is_finite()),
        "Some decoded DCT values are non-finite"
    );
}

#[test]
fn test_dct_scale_equivariance() {
    let config = DCTConfig {
        bits: 10,
        num_coeffs: 16,
    };
    let tokenizer = DCTTokenizer::new(config).expect("DCTTokenizer::new failed");

    let signal_1 = make_signal(64, 1.0);
    let signal_k = make_signal(64, 4.0);

    let decoded_1 = tokenizer
        .decode(&tokenizer.encode(&signal_1).expect("encode"))
        .expect("decode");
    let decoded_k = tokenizer
        .decode(&tokenizer.encode(&signal_k).expect("encode"))
        .expect("decode");

    // Without scale preservation, decoded_k would equal decoded_1 (the original bug).
    // With the fix, decoded_k ≈ 4 × decoded_1.
    for (a, b) in decoded_1.iter().zip(decoded_k.iter()) {
        assert!(
            (b - 4.0 * a).abs() < 1.0,
            "DCT scale equivariance violated: 4.0 * {} ≈ {}, got {}",
            a,
            4.0 * a,
            b
        );
    }
}

#[test]
fn test_dct_zero_signal_stable() {
    let config = DCTConfig {
        bits: 8,
        num_coeffs: 8,
    };
    let tokenizer = DCTTokenizer::new(config).expect("DCTTokenizer::new failed");
    let signal = Array1::zeros(32_usize);
    let encoded = tokenizer.encode(&signal).expect("encode failed");
    let decoded = tokenizer.decode(&encoded).expect("decode failed");
    assert!(
        decoded.iter().all(|x| x.is_finite()),
        "Zero-signal DCT round-trip produced non-finite values"
    );
}

// ============================================================================
// FourierTokenizer (no quantization — raw complex float storage)
// ============================================================================

#[test]
fn test_fourier_round_trip_non_unit_scale() {
    // FourierTokenizer stores raw FFT coefficients (no quantization), so
    // scale is inherently preserved. Verify the basic round-trip is correct.
    let config = FourierConfig {
        bits: 8,
        num_bins: 8,
        magnitude_only: false,
    };
    let num_bins = config.num_bins;
    let tokenizer = FourierTokenizer::new(config).expect("FourierTokenizer::new failed");

    let signal = make_signal(64, 4.0);
    let encoded = tokenizer.encode(&signal).expect("encode failed");
    let decoded = tokenizer.decode(&encoded).expect("decode failed");

    // token[0] is the original-length header `decode` needs to run the
    // inverse transform at the right size (see `FourierTokenizer::encode`'s
    // docs); tokens[1..] are the real + imag pair per bin.
    assert_eq!(encoded.len(), 1 + num_bins * 2);
    assert_eq!(
        decoded.len(),
        signal.len(),
        "decode must reconstruct the original signal length, not num_bins"
    );
    assert!(
        decoded.iter().all(|x| x.is_finite()),
        "Fourier decoded values must all be finite"
    );
}

#[test]
fn test_fourier_magnitude_only_scale_sensitivity() {
    let config = FourierConfig {
        bits: 8,
        num_bins: 8,
        magnitude_only: true,
    };
    let tokenizer = FourierTokenizer::new(config).expect("FourierTokenizer::new failed");

    let signal_1 = make_signal(64, 1.0);
    let signal_4 = make_signal(64, 4.0);

    let encoded_1 = tokenizer.encode(&signal_1).expect("encode 1");
    let encoded_4 = tokenizer.encode(&signal_4).expect("encode 4");

    // token[0] is the length header (constant across both signals, since
    // both are length 64); tokens[1..] are the magnitudes, the first of
    // which should scale with the signal.
    assert_eq!(encoded_1[0], 64.0);
    assert_eq!(encoded_4[0], 64.0);

    // Magnitudes of a 4× scaled signal should be ≈ 4× larger.
    let ratio = encoded_4[1] / encoded_1[1];
    assert!(
        (ratio - 4.0).abs() < 0.5,
        "Fourier magnitude ratio expected ≈ 4.0, got {}",
        ratio
    );
}
