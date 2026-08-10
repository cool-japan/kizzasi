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

    #[test]
    fn test_arithmetic_adaptive() {
        let mut encoder = ArithmeticEncoder::new(4); // 4 symbols
        let symbols = vec![0, 0, 0, 1, 1, 2, 3];

        let encoded = encoder.encode(&symbols, true).unwrap();

        // Adaptive decoder would need to track the same updates
        // For now, test non-adaptive
        let mut encoder2 = ArithmeticEncoder::new(4);
        let encoded2 = encoder2.encode(&symbols, false).unwrap();

        assert!(encoded.len() >= 12); // At least metadata
        assert!(encoded2.len() >= 12);
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
}
