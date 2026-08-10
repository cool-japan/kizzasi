//! Encoders for the entropy-coding module.
//!
//! This file hosts the encoder half of every entropy-coding algorithm
//! exposed by [`crate::entropy`]:
//!
//! - [`HuffmanEncoder`] – optimal prefix-free codes from symbol frequencies.
//! - [`ArithmeticEncoder`] – adaptive arithmetic coding.
//! - [`RangeEncoder`] – LZMA-style carry-propagating range coder.
//!
//! Decoders for the corresponding formats live in the sibling `decoder`
//! sub-module; the shared internal helper `shift_low` is private to this
//! module and is used only by [`RangeEncoder`].

use super::HuffmanNode;
use crate::error::{TokenizerError, TokenizerResult};
use std::collections::{BinaryHeap, HashMap};

/// Huffman encoder for lossless compression
///
/// Builds an optimal prefix-free code based on symbol frequencies,
/// assigning shorter codes to more frequent symbols.
pub struct HuffmanEncoder {
    /// Symbol to codeword mapping
    codebook: HashMap<u32, Vec<bool>>,
    /// Root of the Huffman tree (for decoder)
    tree_nodes: Vec<HuffmanNode>,
    /// Root node index
    root_idx: usize,
}

impl HuffmanEncoder {
    /// Build a Huffman encoder from symbol frequencies
    ///
    /// # Arguments
    ///
    /// * `frequencies` - Map from symbol to frequency count
    ///
    /// # Returns
    ///
    /// A Huffman encoder with optimal prefix-free codes
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut freqs = HashMap::new();
    /// freqs.insert(0, 10);  // Symbol 0 appears 10 times
    /// freqs.insert(1, 5);   // Symbol 1 appears 5 times
    /// freqs.insert(2, 2);   // Symbol 2 appears 2 times
    ///
    /// let encoder = HuffmanEncoder::from_frequencies(&freqs);
    /// ```
    pub fn from_frequencies(frequencies: &HashMap<u32, u64>) -> TokenizerResult<Self> {
        if frequencies.is_empty() {
            return Err(TokenizerError::encoding(
                "encoding",
                "Cannot build Huffman tree from empty frequencies",
            ));
        }

        // Special case: single symbol
        if frequencies.len() == 1 {
            let symbol = *frequencies
                .keys()
                .next()
                .expect("Frequencies map is non-empty");
            let mut codebook = HashMap::new();
            codebook.insert(symbol, vec![false]); // Single bit code

            let node = HuffmanNode {
                symbol: Some(symbol),
                frequency: *frequencies
                    .get(&symbol)
                    .expect("Symbol exists in frequencies map"),
                left: None,
                right: None,
            };

            return Ok(Self {
                codebook,
                tree_nodes: vec![node],
                root_idx: 0,
            });
        }

        // Build Huffman tree using a min-heap
        #[derive(Eq, PartialEq)]
        struct HeapEntry {
            frequency: u64,
            idx: usize,
        }

        impl Ord for HeapEntry {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Reverse for min-heap, use idx as tiebreaker for stability
                other
                    .frequency
                    .cmp(&self.frequency)
                    .then_with(|| other.idx.cmp(&self.idx))
            }
        }

        impl PartialOrd for HeapEntry {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::new();
        let mut nodes = Vec::new();

        // Initialize leaf nodes
        for (&symbol, &freq) in frequencies {
            let idx = nodes.len();
            nodes.push(HuffmanNode {
                symbol: Some(symbol),
                frequency: freq,
                left: None,
                right: None,
            });
            heap.push(HeapEntry {
                frequency: freq,
                idx,
            });
        }

        // Build tree bottom-up by combining lowest-frequency nodes
        while heap.len() > 1 {
            let entry1 = heap.pop().expect("Heap has at least 2 elements");
            let entry2 = heap.pop().expect("Heap has at least 2 elements");

            let combined_freq = entry1.frequency + entry2.frequency;
            let parent_idx = nodes.len();

            nodes.push(HuffmanNode {
                symbol: None,
                frequency: combined_freq,
                left: Some(entry1.idx),
                right: Some(entry2.idx),
            });

            heap.push(HeapEntry {
                frequency: combined_freq,
                idx: parent_idx,
            });
        }

        let root_idx = heap
            .pop()
            .expect("Heap has exactly 1 root element after loop")
            .idx;

        // Build codebook by traversing tree
        let mut codebook = HashMap::new();
        let mut stack = vec![(root_idx, Vec::new())];

        while let Some((idx, code)) = stack.pop() {
            let node = &nodes[idx];

            if let Some(symbol) = node.symbol {
                // Leaf node - save code
                codebook.insert(symbol, code);
            } else {
                // Internal node - traverse children
                if let Some(left_idx) = node.left {
                    let mut left_code = code.clone();
                    left_code.push(false); // 0
                    stack.push((left_idx, left_code));
                }
                if let Some(right_idx) = node.right {
                    let mut right_code = code.clone();
                    right_code.push(true); // 1
                    stack.push((right_idx, right_code));
                }
            }
        }

        Ok(Self {
            codebook,
            tree_nodes: nodes,
            root_idx,
        })
    }

    /// Encode a sequence of symbols using Huffman coding
    ///
    /// # Arguments
    ///
    /// * `symbols` - Sequence of symbols to encode
    ///
    /// # Returns
    ///
    /// Compressed bitstream as a `Vec<u8>`, with length information prepended
    pub fn encode(&self, symbols: &[u32]) -> TokenizerResult<Vec<u8>> {
        let mut bits = Vec::new();

        // Encode each symbol
        for &symbol in symbols {
            let code = self.codebook.get(&symbol).ok_or_else(|| {
                TokenizerError::encoding("serialization", format!("Unknown symbol: {}", symbol))
            })?;
            bits.extend_from_slice(code);
        }

        // Pack bits into bytes
        let num_bits = bits.len();
        let num_bytes = num_bits.div_ceil(8);
        let mut bytes = vec![0u8; num_bytes];

        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }

        // Prepend metadata: number of symbols (u32) and number of bits (u32)
        let mut result = Vec::new();
        result.extend_from_slice(&(symbols.len() as u32).to_le_bytes());
        result.extend_from_slice(&(num_bits as u32).to_le_bytes());
        result.extend_from_slice(&bytes);

        Ok(result)
    }

    /// Get the codebook (for decoder)
    pub fn codebook(&self) -> &HashMap<u32, Vec<bool>> {
        &self.codebook
    }

    /// Get the Huffman tree (for decoder)
    pub fn tree(&self) -> (&[HuffmanNode], usize) {
        (&self.tree_nodes, self.root_idx)
    }

    /// Compute average code length
    pub fn average_code_length(&self, frequencies: &HashMap<u32, u64>) -> f64 {
        let total: u64 = frequencies.values().sum();
        if total == 0 {
            return 0.0;
        }

        let mut weighted_sum = 0.0;
        for (symbol, freq) in frequencies {
            if let Some(code) = self.codebook.get(symbol) {
                weighted_sum += code.len() as f64 * (*freq as f64);
            }
        }

        weighted_sum / total as f64
    }

    /// Compute entropy of the distribution
    pub fn entropy(frequencies: &HashMap<u32, u64>) -> f64 {
        let total: u64 = frequencies.values().sum();
        if total == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for freq in frequencies.values() {
            if *freq > 0 {
                let p = *freq as f64 / total as f64;
                entropy -= p * p.log2();
            }
        }

        entropy
    }
}

/// Arithmetic encoder for near-optimal compression
///
/// Uses adaptive probability models to achieve compression rates
/// close to the theoretical entropy limit.
pub struct ArithmeticEncoder {
    /// Symbol frequency counts (adaptive)
    frequencies: HashMap<u32, u64>,
    /// Total count
    total_count: u64,
    /// Minimum count for adaptive updates
    min_count: u64,
}

impl ArithmeticEncoder {
    /// Create a new arithmetic encoder with uniform initialization
    ///
    /// # Arguments
    ///
    /// * `alphabet_size` - Number of unique symbols
    pub fn new(alphabet_size: usize) -> Self {
        let mut frequencies = HashMap::new();
        for symbol in 0..alphabet_size as u32 {
            frequencies.insert(symbol, 1);
        }

        Self {
            frequencies,
            total_count: alphabet_size as u64,
            min_count: 1,
        }
    }

    /// Create encoder from existing frequencies
    pub fn from_frequencies(frequencies: HashMap<u32, u64>) -> Self {
        let total_count = frequencies.values().sum();
        Self {
            frequencies,
            total_count,
            min_count: 1,
        }
    }

    /// Update frequency counts (adaptive coding)
    fn update_frequency(&mut self, symbol: u32) {
        *self.frequencies.entry(symbol).or_insert(self.min_count) += 1;
        self.total_count += 1;

        // Prevent overflow by rescaling
        if self.total_count > 1_000_000 {
            self.rescale_frequencies();
        }
    }

    /// Rescale all frequencies by half (prevent overflow)
    fn rescale_frequencies(&mut self) {
        self.total_count = 0;
        for freq in self.frequencies.values_mut() {
            *freq = (*freq / 2).max(self.min_count);
            self.total_count += *freq;
        }
    }

    /// Get cumulative frequency for a symbol
    fn cumulative_frequency(&self, symbol: u32) -> (u64, u64) {
        let mut cumulative = 0u64;

        for s in 0..symbol {
            cumulative += self.frequencies.get(&s).unwrap_or(&0);
        }

        let freq = self.frequencies.get(&symbol).unwrap_or(&self.min_count);
        (cumulative, cumulative + freq)
    }

    /// Encode symbols using arithmetic coding
    ///
    /// # Arguments
    ///
    /// * `symbols` - Sequence of symbols to encode
    /// * `adaptive` - Whether to use adaptive frequency updates
    ///
    /// # Returns
    ///
    /// Compressed representation as bytes
    pub fn encode(&mut self, symbols: &[u32], adaptive: bool) -> TokenizerResult<Vec<u8>> {
        const PRECISION: u64 = 1u64 << 32; // 32-bit precision

        let mut low = 0u64;
        let mut high = PRECISION - 1;

        for &symbol in symbols {
            let range = high - low + 1;
            let (cum_low, cum_high) = self.cumulative_frequency(symbol);

            high = low + (range * cum_high / self.total_count) - 1;
            low += range * cum_low / self.total_count;

            // Adaptive update
            if adaptive {
                self.update_frequency(symbol);
            }

            // Renormalization (emit bits when possible)
            // For simplicity, we'll handle this at the end
        }

        // Final value in [low, high]
        let value = (low + high) / 2;

        // Convert to bytes
        let mut result = Vec::new();
        result.extend_from_slice(&(symbols.len() as u32).to_le_bytes());
        result.extend_from_slice(&value.to_le_bytes());

        Ok(result)
    }

    /// Get the codebook for inspection
    pub fn frequencies(&self) -> &HashMap<u32, u64> {
        &self.frequencies
    }
}

/// Internal helper for the LZMA-style range coder.
///
/// Flushes one byte of `low` to `out`, handling carry propagation through
/// any deferred 0xFF chain. When the top byte of `low` is in the
/// "uncertain" 0xFF state (i.e., `0xFF000000 <= low <= 0xFFFFFFFF`) it
/// defers the decision by incrementing `cache_size`; otherwise it commits
/// the cached byte (plus optional carry) and any 0xFF run, then refills
/// the cache from `low >> 24`.
#[inline]
fn shift_low(low: &mut u64, cache: &mut u8, cache_size: &mut u64, out: &mut Vec<u8>) {
    // Decision committed: low's top byte will not be 0xFF + carry ambiguous.
    if *low < 0xFF000000u64 || *low > 0xFFFFFFFFu64 {
        let carry = (*low >> 32) as u8; // 0 or 1
        out.push(cache.wrapping_add(carry));
        // Drain any pending 0xFF (or 0x00, if carry) bytes.
        let pending = cache_size.saturating_sub(1);
        for _ in 0..pending {
            out.push(0xFFu8.wrapping_add(carry));
        }
        *cache = ((*low >> 24) & 0xFF) as u8;
        *cache_size = 1;
    } else {
        // Top byte is 0xFF and a carry may still arrive — defer the flush.
        *cache_size += 1;
    }
    *low = (*low << 8) & 0xFFFFFFFFu64;
}

/// Range encoder for efficient entropy coding
///
/// Range coding is a variant of arithmetic coding that's more efficient
/// in practice due to simplified renormalization and better bit packing.
pub struct RangeEncoder {
    /// Symbol frequency counts
    frequencies: HashMap<u32, u64>,
    /// Total count
    total_count: u64,
    /// Cumulative frequency table
    cumulative: Vec<(u32, u64, u64)>, // (symbol, low, high)
}

impl RangeEncoder {
    /// Create a new range encoder from frequencies
    pub fn from_frequencies(frequencies: HashMap<u32, u64>) -> TokenizerResult<Self> {
        if frequencies.is_empty() {
            return Err(TokenizerError::encoding(
                "encoding",
                "Cannot create range encoder from empty frequencies",
            ));
        }

        let total_count: u64 = frequencies.values().sum();

        // Build cumulative frequency table
        let mut symbols: Vec<u32> = frequencies.keys().copied().collect();
        symbols.sort_unstable();

        let mut cumulative = Vec::new();
        let mut cum_freq = 0u64;

        for symbol in symbols {
            let freq = frequencies.get(&symbol).unwrap_or(&0);
            if *freq > 0 {
                cumulative.push((symbol, cum_freq, cum_freq + freq));
                cum_freq += freq;
            }
        }

        Ok(Self {
            frequencies,
            total_count,
            cumulative,
        })
    }

    /// Encode symbols using range coding
    ///
    /// Implements an LZMA-style carry-propagating range coder. The encoder
    /// keeps `low` as a 33-bit value (in a `u64`) so a carry produced by
    /// adding `step * cum_low` can be observed and propagated through any
    /// buffered `0xFF` bytes via the cache/cache_size mechanism.
    ///
    /// # Arguments
    ///
    /// * `symbols` - Sequence of symbols to encode
    ///
    /// # Returns
    ///
    /// Compressed bitstream as bytes
    pub fn encode(&self, symbols: &[u32]) -> TokenizerResult<Vec<u8>> {
        // Scale frequencies to fit in reasonable precision.
        // Scaled cumulative frequencies fit in [0, scale] and scale fits in u32.
        let scale: u64 = 1u64 << 14;
        let total = self.total_count;

        // Pre-compute scaled cumulative frequencies.
        // Each (cum_low, cum_high) is in [0, scale] and fits in u32.
        let mut scaled_cum: Vec<(u32, u32, u32)> = Vec::with_capacity(self.cumulative.len());
        for (sym, cum_low, cum_high) in &self.cumulative {
            let scaled_low = ((*cum_low as u128 * scale as u128) / total as u128) as u64;
            let scaled_high = ((*cum_high as u128 * scale as u128) / total as u128) as u64;
            // Ensure at least 1 unit of width for each symbol.
            let scaled_high = scaled_high.max(scaled_low + 1);
            scaled_cum.push((*sym, scaled_low as u32, scaled_high as u32));
        }

        // Range coder state.
        let mut low: u64 = 0; // 33 bits: bit 32 may be a carry
        let mut range: u32 = 0xFFFFFFFF;
        let mut cache: u8 = 0;
        let mut cache_size: u64 = 1;
        let mut output: Vec<u8> = Vec::new();

        let scale_u32 = scale as u32;

        for &symbol in symbols {
            // Find symbol in cumulative table.
            let (_, cum_low, cum_high) = scaled_cum
                .iter()
                .find(|(s, _, _)| *s == symbol)
                .ok_or_else(|| {
                    TokenizerError::encoding("serialization", format!("Unknown symbol: {}", symbol))
                })?;

            // Update range (LZMA-style step computation).
            let step = range / scale_u32;
            low = low.wrapping_add((step as u64).wrapping_mul(*cum_low as u64));
            range = step.wrapping_mul(cum_high.wrapping_sub(*cum_low));

            // Renormalization: emit bytes while range falls below 2^24.
            while range < (1u32 << 24) {
                shift_low(&mut low, &mut cache, &mut cache_size, &mut output);
                range <<= 8;
            }
        }

        // End-of-stream flush: drain `low` (33 bits) through `shift_low`.
        // Five shifts suffice: 5 * 8 = 40 bits, more than enough to push out
        // the full state plus any deferred 0xFF chain.
        for _ in 0..5 {
            shift_low(&mut low, &mut cache, &mut cache_size, &mut output);
        }

        // Prepend metadata: number of symbols.
        let mut result = Vec::with_capacity(4 + output.len());
        result.extend_from_slice(&(symbols.len() as u32).to_le_bytes());
        result.extend_from_slice(&output);

        Ok(result)
    }

    /// Get the frequency table
    pub fn frequencies(&self) -> &HashMap<u32, u64> {
        &self.frequencies
    }
}
