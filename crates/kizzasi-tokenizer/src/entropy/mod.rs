//! Entropy coding for efficient compression
//!
//! This module provides entropy coding algorithms for compressing quantized
//! signal representations. Entropy coding assigns shorter codes to more
//! frequent symbols, achieving better compression than fixed-length encoding.
//!
//! # Algorithms
//!
//! - **Huffman Coding**: Optimal prefix-free code construction
//! - **Arithmetic Coding**: Near-optimal compression with adaptive probabilities
//! - **Range Coding**: Efficient variant of arithmetic coding
//!
//! # Module layout
//!
//! - The private `encoder` sub-module hosts [`HuffmanEncoder`],
//!   [`ArithmeticEncoder`], and [`RangeEncoder`].
//! - The private `decoder` sub-module hosts [`HuffmanDecoder`],
//!   [`ArithmeticDecoder`], and [`RangeDecoder`].
//! - The private `model` sub-module holds everything the two halves must
//!   agree on bit-for-bit: the arithmetic coder's frequency model and
//!   interval constants, the range coder's cumulative-table quantisation, and
//!   the bit reader/writer.
//! - The shared [`HuffmanNode`] type, frequency utilities, the
//!   [`BitrateController`], and the [`compression_ratio`] helper live here in
//!   `mod.rs`.
//!
//! # Example
//!
//! ```ignore
//! use kizzasi_tokenizer::entropy::{HuffmanEncoder, HuffmanDecoder};
//!
//! // Build encoder from symbol frequencies
//! let mut encoder = HuffmanEncoder::from_frequencies(&frequencies);
//! let compressed = encoder.encode(&symbols)?;
//!
//! // Decode back
//! let mut decoder = HuffmanDecoder::new(encoder.codebook());
//! let decompressed = decoder.decode(&compressed)?;
//! ```

use crate::error::{TokenizerError, TokenizerResult};
use std::collections::HashMap;

mod decoder;
mod encoder;
mod model;

pub use decoder::{ArithmeticDecoder, HuffmanDecoder, RangeDecoder};
pub use encoder::{ArithmeticEncoder, HuffmanEncoder, RangeEncoder};

/// Huffman tree node
///
/// Shared by [`HuffmanEncoder`] and [`HuffmanDecoder`]. Fields are visible to
/// the encoder/decoder sub-modules so they can construct and traverse the
/// tree without going through an additional accessor layer; the type itself
/// remains opaque to downstream crates because no field is `pub`.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct HuffmanNode {
    /// Symbol (None for internal nodes)
    pub(super) symbol: Option<u32>,
    /// Frequency/weight
    pub(super) frequency: u64,
    /// Left child index
    pub(super) left: Option<usize>,
    /// Right child index
    pub(super) right: Option<usize>,
}

impl Ord for HuffmanNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap
        other.frequency.cmp(&self.frequency)
    }
}

impl PartialOrd for HuffmanNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Compute symbol frequencies from a sequence
pub fn compute_frequencies(symbols: &[u32]) -> HashMap<u32, u64> {
    let mut frequencies = HashMap::new();
    for &symbol in symbols {
        *frequencies.entry(symbol).or_insert(0) += 1;
    }
    frequencies
}

/// Bit-rate controller for adaptive quantization
///
/// Dynamically adjusts quantization parameters to achieve a target bit-rate
pub struct BitrateController {
    /// Target bits per symbol
    target_bits_per_symbol: f64,
    /// Current average bits per symbol
    current_bits_per_symbol: f64,
    /// Proportional gain for control
    kp: f64,
    /// Integral gain for control
    ki: f64,
    /// Integral error accumulator
    integral_error: f64,
    /// Quantization step size
    quantization_step: f64,
    /// Minimum step size
    min_step: f64,
    /// Maximum step size
    max_step: f64,
}

impl BitrateController {
    /// Create a new bitrate controller
    ///
    /// # Arguments
    ///
    /// * `target_bits_per_symbol` - Desired average bits per symbol
    /// * `initial_step` - Initial quantization step size
    /// * `kp` - Proportional gain (typical: 0.1)
    /// * `ki` - Integral gain (typical: 0.01)
    pub fn new(
        target_bits_per_symbol: f64,
        initial_step: f64,
        kp: f64,
        ki: f64,
    ) -> TokenizerResult<Self> {
        if target_bits_per_symbol <= 0.0 {
            return Err(TokenizerError::InvalidConfig(
                "Target bits per symbol must be positive".into(),
            ));
        }

        if initial_step <= 0.0 {
            return Err(TokenizerError::InvalidConfig(
                "Initial step must be positive".into(),
            ));
        }

        Ok(Self {
            target_bits_per_symbol,
            current_bits_per_symbol: target_bits_per_symbol,
            kp,
            ki,
            integral_error: 0.0,
            quantization_step: initial_step,
            min_step: initial_step * 0.1,
            max_step: initial_step * 10.0,
        })
    }

    /// Update controller based on observed bit-rate
    ///
    /// # Arguments
    ///
    /// * `actual_bits_per_symbol` - Measured bits per symbol in current frame
    ///
    /// # Returns
    ///
    /// New quantization step size to use
    pub fn update(&mut self, actual_bits_per_symbol: f64) -> f64 {
        // Compute error
        let error = actual_bits_per_symbol - self.target_bits_per_symbol;

        // Update integral
        self.integral_error += error;

        // PI control
        let adjustment = self.kp * error + self.ki * self.integral_error;

        // Update step size (increase step to reduce bits, decrease step to increase bits)
        self.quantization_step *= (1.0 + adjustment).clamp(0.5, 2.0);

        // Clamp step size
        self.quantization_step = self.quantization_step.max(self.min_step).min(self.max_step);

        // Update current estimate
        self.current_bits_per_symbol = actual_bits_per_symbol;

        self.quantization_step
    }

    /// Get current quantization step
    pub fn current_step(&self) -> f64 {
        self.quantization_step
    }

    /// Get target bit-rate
    pub fn target_bitrate(&self) -> f64 {
        self.target_bits_per_symbol
    }

    /// Get current average bit-rate
    pub fn current_bitrate(&self) -> f64 {
        self.current_bits_per_symbol
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.integral_error = 0.0;
        self.current_bits_per_symbol = self.target_bits_per_symbol;
    }

    /// Set new target bit-rate
    pub fn set_target(&mut self, target_bits_per_symbol: f64) -> TokenizerResult<()> {
        if target_bits_per_symbol <= 0.0 {
            return Err(TokenizerError::InvalidConfig(
                "Target bits per symbol must be positive".into(),
            ));
        }
        self.target_bits_per_symbol = target_bits_per_symbol;
        Ok(())
    }
}

/// Compute compression ratio
///
/// # Arguments
///
/// * `original_bits` - Number of bits in original representation
/// * `compressed_bytes` - Number of bytes in compressed representation
///
/// # Returns
///
/// Compression ratio (original / compressed)
pub fn compression_ratio(original_bits: usize, compressed_bytes: usize) -> f64 {
    if compressed_bytes == 0 {
        return f64::INFINITY;
    }
    original_bits as f64 / (compressed_bytes * 8) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_single_symbol() {
        let mut freqs = HashMap::new();
        freqs.insert(42, 100);

        let encoder = HuffmanEncoder::from_frequencies(&freqs).unwrap();
        let symbols = vec![42, 42, 42];
        let encoded = encoder.encode(&symbols).unwrap();

        let decoder = HuffmanDecoder::new(encoder.tree());
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_huffman_basic() {
        let mut freqs = HashMap::new();
        freqs.insert(0, 10);
        freqs.insert(1, 5);
        freqs.insert(2, 2);
        freqs.insert(3, 1);

        let encoder = HuffmanEncoder::from_frequencies(&freqs).unwrap();

        // Symbol 0 should have shortest code (most frequent)
        let code_0 = encoder.codebook().get(&0).unwrap();
        let code_3 = encoder.codebook().get(&3).unwrap();
        assert!(code_0.len() <= code_3.len());

        // Test encode/decode
        let symbols = vec![0, 1, 2, 3, 0, 0, 1];
        let encoded = encoder.encode(&symbols).unwrap();

        let decoder = HuffmanDecoder::new(encoder.tree());
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_huffman_compression() {
        let mut freqs = HashMap::new();
        freqs.insert(0, 50); // Very frequent
        freqs.insert(1, 25);
        freqs.insert(2, 15);
        freqs.insert(3, 10);

        let encoder = HuffmanEncoder::from_frequencies(&freqs).unwrap();

        // Create a sequence with the same distribution
        let symbols: Vec<u32> = (0..100)
            .map(|i| {
                if i < 50 {
                    0
                } else if i < 75 {
                    1
                } else if i < 90 {
                    2
                } else {
                    3
                }
            })
            .collect();

        let encoded = encoder.encode(&symbols).unwrap();

        // Should achieve compression (100 symbols * 2 bits = 200 bits > compressed size)
        let original_bits = symbols.len() * 2; // 2 bits per symbol for 4 symbols
        let compressed_bits = (encoded.len() - 8) * 8; // Subtract metadata

        assert!(compressed_bits < original_bits);

        // Verify correctness
        let decoder = HuffmanDecoder::new(encoder.tree());
        let decoded = decoder.decode(&encoded).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_huffman_average_code_length() {
        let mut freqs = HashMap::new();
        freqs.insert(0, 8);
        freqs.insert(1, 4);
        freqs.insert(2, 2);
        freqs.insert(3, 1);

        let encoder = HuffmanEncoder::from_frequencies(&freqs).unwrap();
        let avg_len = encoder.average_code_length(&freqs);

        // Should be close to entropy
        let entropy = HuffmanEncoder::entropy(&freqs);
        assert!((avg_len - entropy).abs() < 0.5);
    }

    #[test]
    fn test_arithmetic_basic() {
        let mut freqs = HashMap::new();
        freqs.insert(0, 10);
        freqs.insert(1, 5);
        freqs.insert(2, 2);

        let mut encoder = ArithmeticEncoder::from_frequencies(freqs.clone());
        let symbols = vec![0, 1, 2, 0, 0];

        let encoded = encoder.encode(&symbols, false).unwrap();

        let decoder = ArithmeticDecoder::new(freqs);
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    /// Deterministic frequency table over `0..alphabet_size` with the given
    /// per-symbol counts, for building matched encoder/decoder pairs.
    fn freq_table(counts: &[(u32, u64)]) -> HashMap<u32, u64> {
        counts.iter().copied().collect()
    }

    #[test]
    fn test_arithmetic_adaptive() {
        // The adaptive stream must round-trip through a decoder built from the
        // same *initial* model: the decoder replays the identical updates.
        let mut encoder = ArithmeticEncoder::new(4); // 4 symbols
        let symbols = vec![0, 0, 0, 1, 1, 2, 3];

        let encoded = encoder.encode(&symbols, true).unwrap();

        let initial = ArithmeticEncoder::new(4);
        let decoder = ArithmeticDecoder::new(initial.frequencies().clone());
        let decoded = decoder.decode(&encoded).unwrap();
        assert_eq!(decoded, symbols, "adaptive round-trip must be exact");

        // The encoder's model really did adapt: symbol 0 was seen three times.
        assert_eq!(encoder.frequencies().get(&0).copied(), Some(4));
        assert_eq!(encoder.total_count(), 4 + symbols.len() as u64);

        // Non-adaptive streams round-trip too, and both are far smaller than
        // the fixed 12-byte payload the old implementation always emitted.
        let mut encoder2 = ArithmeticEncoder::new(4);
        let encoded2 = encoder2.encode(&symbols, false).unwrap();
        let decoder2 = ArithmeticDecoder::new(ArithmeticEncoder::new(4).frequencies().clone());
        assert_eq!(decoder2.decode(&encoded2).unwrap(), symbols);
        assert!(
            encoded2.len() < 12,
            "7 symbols over a 4-symbol alphabet should not need 12 bytes, got {}",
            encoded2.len()
        );
    }

    #[test]
    fn test_arithmetic_repeated_symbol_does_not_exhaust_precision() {
        // Regression: with no renormalisation the interval collapsed after a
        // handful of symbols and `high = low + range * cum_high / total - 1`
        // underflowed at the 17th repeat of symbol 0.
        let mut encoder = ArithmeticEncoder::new(4);
        let symbols = vec![0u32; 17];

        let encoded = encoder.encode(&symbols, false).unwrap();
        let decoder = ArithmeticDecoder::new(ArithmeticEncoder::new(4).frequencies().clone());
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_arithmetic_long_sequence_static() {
        // 10 000 symbols over a skewed alphabet: the old fixed-width payload
        // decoded to garbage after roughly five symbols.
        let freqs = freq_table(&[(0, 700), (1, 200), (2, 90), (3, 9), (4, 1)]);
        let mut encoder = ArithmeticEncoder::from_frequencies(freqs.clone());
        let decoder = ArithmeticDecoder::new(freqs);

        let symbols: Vec<u32> = (0..10_000u32)
            .map(|i| match i % 1000 {
                0..=699 => 0,
                700..=899 => 1,
                900..=989 => 2,
                990..=998 => 3,
                _ => 4,
            })
            .collect();

        let encoded = encoder.encode(&symbols, false).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
        // Output length must track the input length, not be a constant.
        assert!(
            encoded.len() > 500,
            "10k symbols cannot fit in {} bytes",
            encoded.len()
        );
    }

    #[test]
    fn test_arithmetic_long_sequence_adaptive() {
        let mut encoder = ArithmeticEncoder::new(8);
        let decoder = ArithmeticDecoder::new(ArithmeticEncoder::new(8).frequencies().clone());

        // Deterministic pseudo-random stream with a strong bias towards 0.
        let mut state = 0x1234_5678u64;
        let symbols: Vec<u32> = (0..10_000)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let draw = (state >> 33) % 100;
                if draw < 80 {
                    0
                } else {
                    1 + (draw % 7) as u32
                }
            })
            .collect();

        let encoded = encoder.encode(&symbols, true).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_arithmetic_probability_below_two_to_the_minus_fourteen() {
        // p(rare) = 1 / (2^20 + 1) — far below the 2^-14 resolution that
        // breaks a naively scaled coder. The arithmetic coder codes it exactly.
        let rare_total: u64 = 1 << 20;
        let freqs = freq_table(&[(0, 1), (1, rare_total)]);
        let mut encoder = ArithmeticEncoder::from_frequencies(freqs.clone());
        let decoder = ArithmeticDecoder::new(freqs);

        let mut symbols: Vec<u32> = vec![1; 2_000];
        for idx in [0usize, 1, 2, 500, 999, 1_000, 1_999] {
            if let Some(slot) = symbols.get_mut(idx) {
                *slot = 0;
            }
        }

        let encoded = encoder.encode(&symbols, false).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_arithmetic_randomized_roundtrip() {
        use scirs2_core::random::Random;

        let mut rng = Random::seed(20_240_610);

        for trial in 0..60 {
            let alphabet_size: u32 = rng.gen_range(2u32..24u32);
            let mut freqs: HashMap<u32, u64> = HashMap::new();
            for symbol in 0..alphabet_size {
                // Mix of ordinary and extremely rare symbols.
                let freq: u64 = if symbol % 5 == 0 {
                    1
                } else {
                    rng.gen_range(1u64..100_000u64)
                };
                freqs.insert(symbol, freq);
            }

            let symbols: Vec<u32> = (0..1_500)
                .map(|_| rng.gen_range(0u32..alphabet_size))
                .collect();

            for adaptive in [false, true] {
                let mut encoder = ArithmeticEncoder::from_frequencies(freqs.clone());
                let decoder = ArithmeticDecoder::new(freqs.clone());
                let encoded = encoder.encode(&symbols, adaptive).expect("encode");
                let decoded = decoder.decode(&encoded).expect("decode");
                assert_eq!(
                    decoded, symbols,
                    "arithmetic round-trip mismatch on trial {} (alphabet {}, adaptive {})",
                    trial, alphabet_size, adaptive
                );
            }
        }
    }

    #[test]
    fn test_arithmetic_compression_approaches_entropy() {
        // A heavily skewed source must compress far below 1 byte per symbol.
        let freqs = freq_table(&[(0, 999), (1, 1)]);
        let mut encoder = ArithmeticEncoder::from_frequencies(freqs.clone());

        let symbols: Vec<u32> = (0..5_000u32).map(|i| u32::from(i % 1000 == 0)).collect();
        let encoded = encoder.encode(&symbols, false).unwrap();

        let entropy_bits = HuffmanEncoder::entropy(&freqs) * symbols.len() as f64;
        let coded_bits = (encoded.len() * 8) as f64;
        assert!(
            coded_bits < entropy_bits * 1.5 + 64.0,
            "arithmetic coder used {} bits for an {:.0}-bit source",
            coded_bits,
            entropy_bits
        );

        let decoder = ArithmeticDecoder::new(freqs);
        assert_eq!(decoder.decode(&encoded).unwrap(), symbols);
    }

    #[test]
    fn test_arithmetic_empty_input_roundtrips() {
        let freqs = freq_table(&[(0, 3), (1, 5)]);
        let mut encoder = ArithmeticEncoder::from_frequencies(freqs.clone());
        let encoded = encoder.encode(&[], false).unwrap();

        let decoder = ArithmeticDecoder::new(freqs);
        assert_eq!(decoder.decode(&encoded).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn test_arithmetic_unknown_symbol_is_an_error() {
        let mut encoder = ArithmeticEncoder::new(4);
        let result = encoder.encode(&[0, 1, 99], false);
        assert!(
            matches!(result, Err(TokenizerError::EncodingError { .. })),
            "encoding a symbol outside the alphabet must fail, got {:?}",
            result.map(|bytes| bytes.len())
        );
    }

    #[test]
    fn test_arithmetic_rejects_unknown_flag_bits() {
        let freqs = freq_table(&[(0, 1), (1, 1)]);
        let mut encoder = ArithmeticEncoder::from_frequencies(freqs.clone());
        let mut encoded = encoder.encode(&[0, 1, 0], false).unwrap();
        // Set a flag bit this format version does not define.
        if let Some(flags) = encoded.get_mut(4) {
            *flags |= 0b1000_0000;
        }

        let decoder = ArithmeticDecoder::new(freqs);
        assert!(
            matches!(
                decoder.decode(&encoded),
                Err(TokenizerError::DecodingError { .. })
            ),
            "unknown header flags must be rejected rather than ignored"
        );
    }

    #[test]
    fn test_arithmetic_rejects_truncated_header() {
        let freqs = freq_table(&[(0, 1), (1, 1)]);
        let decoder = ArithmeticDecoder::new(freqs);
        assert!(decoder.decode(&[]).is_err());
        assert!(decoder.decode(&[0, 0, 0, 0]).is_err());
    }

    #[test]
    fn test_arithmetic_rejects_oversized_total() {
        // Above 2^30 the coder can no longer guarantee a non-empty interval
        // per symbol, so the table is rejected instead of silently corrupting.
        let freqs = freq_table(&[(0, 1 << 30), (1, 1 << 30)]);
        let mut encoder = ArithmeticEncoder::from_frequencies(freqs.clone());
        assert!(matches!(
            encoder.encode(&[0, 1], false),
            Err(TokenizerError::InvalidConfig(_))
        ));

        let decoder = ArithmeticDecoder::new(freqs);
        assert!(matches!(
            decoder.decode(&[0, 0, 0, 0, 0]),
            Err(TokenizerError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_compute_frequencies() {
        let symbols = vec![0, 0, 1, 2, 0, 1];
        let freqs = compute_frequencies(&symbols);

        assert_eq!(*freqs.get(&0).unwrap(), 3);
        assert_eq!(*freqs.get(&1).unwrap(), 2);
        assert_eq!(*freqs.get(&2).unwrap(), 1);
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = compression_ratio(800, 50);
        assert!((ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_entropy() {
        let mut freqs = HashMap::new();
        freqs.insert(0, 2);
        freqs.insert(1, 2);

        let entropy = HuffmanEncoder::entropy(&freqs);
        assert!((entropy - 1.0).abs() < 0.01); // Uniform binary = 1 bit
    }

    #[test]
    fn test_range_coding_basic() {
        let mut freqs = HashMap::new();
        freqs.insert(0, 10);
        freqs.insert(1, 5);
        freqs.insert(2, 2);

        let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();
        let symbols = vec![0, 1, 2, 0, 0, 1];

        let encoded = encoder.encode(&symbols).unwrap();

        let decoder = RangeDecoder::from_frequencies(freqs).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_range_coding_single_symbol() {
        let mut freqs = HashMap::new();
        freqs.insert(42, 100);

        let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();
        let symbols = vec![42, 42, 42, 42];

        let encoded = encoder.encode(&symbols).unwrap();

        let decoder = RangeDecoder::from_frequencies(freqs).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_range_coding_compression() {
        let mut freqs = HashMap::new();
        freqs.insert(0, 50);
        freqs.insert(1, 30);
        freqs.insert(2, 15);
        freqs.insert(3, 5);

        let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();

        // Create sequence with same distribution
        let symbols: Vec<u32> = (0..100)
            .map(|i| {
                if i < 50 {
                    0
                } else if i < 80 {
                    1
                } else if i < 95 {
                    2
                } else {
                    3
                }
            })
            .collect();

        let encoded = encoder.encode(&symbols).unwrap();

        // Should achieve good compression.
        // Subtract the 4-byte length prefix and the encoder's 5-byte flush
        // tail (1 cache placeholder byte + 4 bytes initial-code material) so
        // we measure only the entropy-coded body.
        let original_bits = symbols.len() * 2; // 2 bits per symbol for 4 symbols
        let compressed_bytes = encoded.len().saturating_sub(4 + 5);

        // Range coding should be efficient
        assert!(compressed_bytes * 8 < original_bits);

        // Verify correctness
        let decoder = RangeDecoder::from_frequencies(freqs).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_range_coding_long_sequence() {
        let mut freqs = HashMap::new();
        freqs.insert(0, 40);
        freqs.insert(1, 30);
        freqs.insert(2, 20);
        freqs.insert(3, 10);

        let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();

        // Create longer sequence
        let symbols: Vec<u32> = (0..1000).map(|i| (i % 4) as u32).collect();

        let encoded = encoder.encode(&symbols).unwrap();

        let decoder = RangeDecoder::from_frequencies(freqs).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_bitrate_controller_basic() {
        let controller = BitrateController::new(4.0, 1.0, 0.1, 0.01).unwrap();

        assert_eq!(controller.target_bitrate(), 4.0);
        assert_eq!(controller.current_step(), 1.0);
    }

    #[test]
    fn test_bitrate_controller_update_increase() {
        let mut controller = BitrateController::new(4.0, 1.0, 0.1, 0.01).unwrap();

        // If actual bitrate is higher than target, step should increase
        let initial_step = controller.current_step();
        let new_step = controller.update(5.0); // Higher than target

        assert!(new_step > initial_step);
    }

    #[test]
    fn test_bitrate_controller_update_decrease() {
        let mut controller = BitrateController::new(4.0, 1.0, 0.1, 0.01).unwrap();

        // If actual bitrate is lower than target, step should decrease
        let initial_step = controller.current_step();
        let new_step = controller.update(3.0); // Lower than target

        assert!(new_step < initial_step);
    }

    #[test]
    fn test_bitrate_controller_convergence() {
        let mut controller = BitrateController::new(4.0, 1.0, 0.1, 0.01).unwrap();

        // Simulate feedback loop
        for _ in 0..10 {
            controller.update(4.5); // Slightly above target
        }

        // Step should have increased to compensate
        assert!(controller.current_step() > 1.0);
    }

    #[test]
    fn test_bitrate_controller_reset() {
        let mut controller = BitrateController::new(4.0, 1.0, 0.1, 0.01).unwrap();

        controller.update(5.0);
        controller.update(6.0);

        controller.reset();

        assert_eq!(controller.current_bitrate(), 4.0);
    }

    #[test]
    fn test_bitrate_controller_set_target() {
        let mut controller = BitrateController::new(4.0, 1.0, 0.1, 0.01).unwrap();

        controller.set_target(8.0).unwrap();
        assert_eq!(controller.target_bitrate(), 8.0);
    }

    #[test]
    fn test_bitrate_controller_invalid_target() {
        assert!(BitrateController::new(0.0, 1.0, 0.1, 0.01).is_err());
        assert!(BitrateController::new(-1.0, 1.0, 0.1, 0.01).is_err());
    }

    #[test]
    fn test_bitrate_controller_invalid_step() {
        assert!(BitrateController::new(4.0, 0.0, 0.1, 0.01).is_err());
        assert!(BitrateController::new(4.0, -1.0, 0.1, 0.01).is_err());
    }

    #[test]
    fn test_bitrate_controller_step_clamping() {
        let mut controller = BitrateController::new(4.0, 1.0, 0.5, 0.1).unwrap();

        // Try to drive step very high with large errors
        for _ in 0..100 {
            controller.update(20.0); // Very high bitrate
        }

        // Step should be clamped to max_step (10.0)
        assert!(controller.current_step() <= 10.0);

        controller.reset();

        // Try to drive step very low
        for _ in 0..100 {
            controller.update(0.5); // Very low bitrate
        }

        // Step should be clamped to min_step (0.1)
        assert!(controller.current_step() >= 0.1);
    }

    #[test]
    fn test_range_coding_empty() {
        // Empty input must round-trip to an empty symbol vector.
        let mut freqs = HashMap::new();
        freqs.insert(0u32, 3u64);
        freqs.insert(1u32, 5u64);

        let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();
        let symbols: Vec<u32> = Vec::new();
        let encoded = encoder.encode(&symbols).unwrap();

        let decoder = RangeDecoder::from_frequencies(freqs).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, Vec::<u32>::new());
    }

    #[test]
    fn test_range_coding_single_symbol_long() {
        // 5000 copies of the same symbol; exercises long renorm chains.
        let mut freqs = HashMap::new();
        freqs.insert(7u32, 1u64);
        freqs.insert(8u32, 1u64);

        let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();
        let symbols: Vec<u32> = vec![7u32; 5000];
        let encoded = encoder.encode(&symbols).unwrap();

        let decoder = RangeDecoder::from_frequencies(freqs).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_range_coding_skewed_distribution() {
        // 99% symbol A, 1% symbol B over a 5000-symbol sequence.
        let mut freqs = HashMap::new();
        freqs.insert(0u32, 99u64);
        freqs.insert(1u32, 1u64);

        let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();

        let total = 5000usize;
        let b_count = total / 100; // 1% = 50
        let mut symbols: Vec<u32> = Vec::with_capacity(total);
        // Spread the rare symbol roughly evenly through the stream.
        let step = total / b_count;
        for i in 0..total {
            if i % step == step - 1 {
                symbols.push(1);
            } else {
                symbols.push(0);
            }
        }

        let encoded = encoder.encode(&symbols).unwrap();
        let decoder = RangeDecoder::from_frequencies(freqs).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_range_coding_10k_uniform_256() {
        // 10000 symbols drawn from a 256-symbol uniform alphabet.
        let mut freqs: HashMap<u32, u64> = HashMap::new();
        for s in 0u32..256u32 {
            freqs.insert(s, 1u64);
        }

        let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();

        let symbols: Vec<u32> = (0..10_000u32).map(|i| i % 256).collect();
        let encoded = encoder.encode(&symbols).unwrap();

        let decoder = RangeDecoder::from_frequencies(freqs).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_range_coding_boundary_cum_freq() {
        // One symbol has freq = total - 1, the other has 1: extreme skew.
        let mut freqs = HashMap::new();
        freqs.insert(0u32, 9999u64);
        freqs.insert(1u32, 1u64);

        let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();

        // Mix of both symbols including consecutive rare-symbol hits.
        let mut symbols: Vec<u32> = Vec::with_capacity(500);
        for i in 0..500 {
            if i == 100 || i == 101 || i == 200 || i == 400 {
                symbols.push(1);
            } else {
                symbols.push(0);
            }
        }

        let encoded = encoder.encode(&symbols).unwrap();
        let decoder = RangeDecoder::from_frequencies(freqs).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_range_coding_randomized_roundtrip() {
        // 50 random frequency tables, each tested with a random 1000-symbol
        // sequence. Use scirs2_core's seeded RNG for determinism.
        use scirs2_core::random::Random;

        let mut rng = Random::seed(42);

        for trial in 0..50 {
            // Random alphabet size in [2, 16].
            let alphabet_size: u32 = rng.gen_range(2u32..17u32);
            let mut freqs: HashMap<u32, u64> = HashMap::new();
            for s in 0..alphabet_size {
                let f: u64 = rng.gen_range(1u64..50u64);
                freqs.insert(s, f);
            }

            let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();
            let decoder = RangeDecoder::from_frequencies(freqs).unwrap();

            let symbols: Vec<u32> = (0..1000)
                .map(|_| rng.gen_range(0u32..alphabet_size))
                .collect();

            let encoded = encoder.encode(&symbols).expect("encode should succeed");
            let decoded = decoder.decode(&encoded).expect("decode should succeed");

            assert_eq!(
                decoded, symbols,
                "round-trip mismatch on trial {} with alphabet_size {}",
                trial, alphabet_size
            );
        }
    }

    #[test]
    fn test_range_coding_probability_below_two_to_the_minus_fourteen() {
        // Regression: {0: 1, 1: 20000}. Scaling each cumulative bound
        // independently gave symbol 0 the interval [0, 1) and symbol 1 the
        // interval [0, 16384) — overlapping at 0, so every code value of 0
        // decoded as symbol 0 and desynchronised the rest of the stream.
        let mut freqs = HashMap::new();
        freqs.insert(0u32, 1u64);
        freqs.insert(1u32, 20_000u64);

        let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();
        let decoder = RangeDecoder::from_frequencies(freqs).unwrap();

        let mut symbols: Vec<u32> = vec![1; 3_000];
        for idx in [0usize, 1, 2, 37, 1_500, 2_999] {
            if let Some(slot) = symbols.get_mut(idx) {
                *slot = 0;
            }
        }

        let encoded = encoder.encode(&symbols).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_range_coding_uniform_alphabet_of_twenty_thousand() {
        // Regression: with a 2^14 grid, symbols 0 and 1 of a 20 000-entry
        // uniform alphabet both scaled to lower bound 0.
        let mut freqs: HashMap<u32, u64> = HashMap::new();
        for symbol in 0..20_000u32 {
            freqs.insert(symbol, 1u64);
        }

        let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();
        let decoder = RangeDecoder::from_frequencies(freqs).unwrap();

        // Hit the low, middle and high ends of the alphabet.
        let symbols: Vec<u32> = (0..20_000u32)
            .map(|i| (i * 7919) % 20_000)
            .take(5_000)
            .collect();

        let encoded = encoder.encode(&symbols).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_range_coding_alphabet_larger_than_grid_is_rejected() {
        // 65 537 symbols cannot each own a distinct slot of the 2^16 grid.
        // The table must be refused, not quantised into overlapping intervals.
        let mut freqs: HashMap<u32, u64> = HashMap::new();
        for symbol in 0..65_537u32 {
            freqs.insert(symbol, 1u64);
        }

        assert!(
            matches!(
                RangeEncoder::from_frequencies(freqs.clone()),
                Err(TokenizerError::InvalidConfig(_))
            ),
            "oversized alphabets must be rejected by the encoder"
        );
        assert!(
            matches!(
                RangeDecoder::from_frequencies(freqs),
                Err(TokenizerError::InvalidConfig(_))
            ),
            "oversized alphabets must be rejected by the decoder"
        );
    }

    #[test]
    fn test_range_coding_all_zero_frequencies_rejected() {
        let mut freqs: HashMap<u32, u64> = HashMap::new();
        freqs.insert(0u32, 0u64);
        freqs.insert(1u32, 0u64);

        assert!(matches!(
            RangeEncoder::from_frequencies(freqs.clone()),
            Err(TokenizerError::InvalidConfig(_))
        ));
        assert!(matches!(
            RangeDecoder::from_frequencies(freqs),
            Err(TokenizerError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_range_coding_randomized_extreme_skew_roundtrip() {
        // Random tables that deliberately mix counts of 1 with counts near a
        // million, so many symbols fall below the 2^-16 grid resolution.
        use scirs2_core::random::Random;

        let mut rng = Random::seed(1_337);

        for trial in 0..40 {
            let alphabet_size: u32 = rng.gen_range(2u32..64u32);
            let mut freqs: HashMap<u32, u64> = HashMap::new();
            for symbol in 0..alphabet_size {
                let freq: u64 = if symbol % 3 == 0 {
                    1
                } else {
                    rng.gen_range(1u64..1_000_000u64)
                };
                freqs.insert(symbol, freq);
            }

            let encoder = RangeEncoder::from_frequencies(freqs.clone()).unwrap();
            let decoder = RangeDecoder::from_frequencies(freqs).unwrap();

            let symbols: Vec<u32> = (0..2_000)
                .map(|_| rng.gen_range(0u32..alphabet_size))
                .collect();

            let encoded = encoder.encode(&symbols).expect("encode should succeed");
            let decoded = decoder.decode(&encoded).expect("decode should succeed");

            assert_eq!(
                decoded, symbols,
                "extreme-skew round-trip mismatch on trial {} (alphabet {})",
                trial, alphabet_size
            );
        }
    }

    #[test]
    fn test_range_coding_unknown_symbol_is_an_error() {
        let mut freqs = HashMap::new();
        freqs.insert(0u32, 1u64);
        freqs.insert(1u32, 1u64);

        let encoder = RangeEncoder::from_frequencies(freqs).unwrap();
        assert!(matches!(
            encoder.encode(&[0, 1, 42]),
            Err(TokenizerError::EncodingError { .. })
        ));
    }
}
