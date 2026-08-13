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

use super::model::{
    build_scaled_table, BitWriter, FreqModel, ARITHMETIC_HEADER_LEN, DEFAULT_MIN_COUNT,
    FLAG_ADAPTIVE, HALF, MAX_TOTAL_COUNT, QUARTER, RANGE_SCALE, THREE_QUARTER, WHOLE,
};
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
            let (&symbol, &frequency) = frequencies.iter().next().ok_or_else(|| {
                TokenizerError::encoding(
                    "encoding",
                    "Frequencies map reported one entry but yielded none",
                )
            })?;
            let mut codebook = HashMap::new();
            codebook.insert(symbol, vec![false]); // Single bit code

            let node = HuffmanNode {
                symbol: Some(symbol),
                frequency,
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
        let heap_underflow = || {
            TokenizerError::InternalError(
                "Huffman heap drained below its own length invariant".into(),
            )
        };
        while heap.len() > 1 {
            let entry1 = heap.pop().ok_or_else(heap_underflow)?;
            let entry2 = heap.pop().ok_or_else(heap_underflow)?;

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

        let root_idx = heap.pop().ok_or_else(heap_underflow)?.idx;

        // Build codebook by traversing tree
        let mut codebook = HashMap::new();
        let mut stack = vec![(root_idx, Vec::new())];

        while let Some((idx, code)) = stack.pop() {
            let node = nodes.get(idx).ok_or_else(|| {
                TokenizerError::InternalError(format!(
                    "Huffman tree references missing node index {}",
                    idx
                ))
            })?;

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
/// Implements the textbook integer arithmetic coder (Witten–Neal–Cleary) with
/// 32-bit `low`/`high` registers and full E1/E2/E3 renormalisation: bits are
/// emitted as soon as the leading bit of the interval is decided, E3
/// (underflow) states are counted and resolved when the interval finally
/// escapes the middle half, and the interval is rescaled after every symbol.
/// The compressed size therefore tracks the entropy of the input instead of
/// being a fixed-width header, and precision is never exhausted no matter how
/// many symbols are coded.
///
/// # Frequency model
///
/// The alphabet is exactly the set of symbols with a non-zero frequency.
/// Encoding a symbol outside that set is an error rather than a silent
/// approximation — approximating it would desynchronise the decoder.
///
/// The total of all counts must not exceed 2^30 so that every symbol is
/// guaranteed a sub-interval of width at least one; larger totals are
/// rejected by [`ArithmeticEncoder::encode`].
///
/// # Adaptive mode
///
/// With `adaptive = true` each coded symbol's count is incremented after it is
/// coded, and all counts are halved once the total passes 1,000,000. The
/// updated counts are written back so consecutive `encode` calls continue to
/// adapt. [`super::ArithmeticDecoder`] applies the identical rule, so an
/// adaptive stream decodes exactly when the decoder is constructed from the
/// same *initial* frequency table the encoder started from.
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
            min_count: DEFAULT_MIN_COUNT,
        }
    }

    /// Create encoder from existing frequencies
    pub fn from_frequencies(frequencies: HashMap<u32, u64>) -> Self {
        let total_count = frequencies.values().sum();
        Self {
            frequencies,
            total_count,
            min_count: DEFAULT_MIN_COUNT,
        }
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
    /// The compressed bitstream: a 4-byte little-endian symbol count, a flag
    /// byte recording whether the stream is adaptive, then the packed bits.
    ///
    /// # Errors
    ///
    /// * [`TokenizerError::InvalidConfig`] if the frequency table is empty or
    ///   its total exceeds the coder's 2^30 limit.
    /// * [`TokenizerError::EncodingError`] if a symbol is not in the alphabet,
    ///   or if more than `u32::MAX` symbols are supplied.
    pub fn encode(&mut self, symbols: &[u32], adaptive: bool) -> TokenizerResult<Vec<u8>> {
        let symbol_count = u32::try_from(symbols.len()).map_err(|_| {
            TokenizerError::encoding(
                "encoding",
                format!(
                    "Arithmetic coder supports at most {} symbols per stream, got {}",
                    u32::MAX,
                    symbols.len()
                ),
            )
        })?;

        let mut model = FreqModel::from_frequencies(&self.frequencies, self.min_count)?;
        self.total_count = model.total();
        if self.total_count > MAX_TOTAL_COUNT {
            return Err(TokenizerError::InvalidConfig(format!(
                "Arithmetic coder total frequency {} exceeds the maximum {}; \
                 rescale the frequency table before encoding",
                self.total_count, MAX_TOTAL_COUNT
            )));
        }

        // Interval registers: `low` and `high` are inclusive bounds inside
        // [0, 2^32). `pending` counts unresolved E3 (underflow) rescalings.
        let mut low: u64 = 0;
        let mut high: u64 = WHOLE - 1;
        let mut pending: u64 = 0;
        let mut writer = BitWriter::new();

        for &symbol in symbols {
            let idx = model.index_of(symbol).ok_or_else(|| {
                TokenizerError::encoding(
                    "serialization",
                    format!("Unknown symbol: {} (not in the coder's alphabet)", symbol),
                )
            })?;
            let (cum_low, cum_high) = model.cumulative(idx).ok_or_else(|| {
                TokenizerError::InternalError(
                    "Arithmetic model index outside its own cumulative table".into(),
                )
            })?;
            let total = model.total();

            // Narrow the interval. `range <= 2^32` and `cum_high <= 2^30`, so
            // the product stays well inside u64.
            let range = high - low + 1;
            high = low + range * cum_high / total - 1;
            low += range * cum_low / total;

            // Renormalisation: E1 (interval in the lower half), E2 (upper
            // half), E3 (straddling the midpoint but inside the middle half).
            loop {
                if high < HALF {
                    writer.write_bit_with_pending(false, &mut pending);
                } else if low >= HALF {
                    writer.write_bit_with_pending(true, &mut pending);
                    low -= HALF;
                    high -= HALF;
                } else if low >= QUARTER && high < THREE_QUARTER {
                    pending += 1;
                    low -= QUARTER;
                    high -= QUARTER;
                } else {
                    break;
                }
                low <<= 1;
                high = (high << 1) | 1;
            }

            if adaptive {
                model.update(idx)?;
            }
        }

        // Terminate: emit one more bit (plus the deferred run) so the decoder's
        // zero-padded reads land inside the final interval.
        pending += 1;
        if low < QUARTER {
            writer.write_bit_with_pending(false, &mut pending);
        } else {
            writer.write_bit_with_pending(true, &mut pending);
        }

        if adaptive {
            self.total_count = model.write_back(&mut self.frequencies);
        }

        let payload = writer.finish();
        let mut result = Vec::with_capacity(ARITHMETIC_HEADER_LEN + payload.len());
        result.extend_from_slice(&symbol_count.to_le_bytes());
        result.push(if adaptive { FLAG_ADAPTIVE } else { 0 });
        result.extend_from_slice(&payload);

        Ok(result)
    }

    /// Get the codebook for inspection
    pub fn frequencies(&self) -> &HashMap<u32, u64> {
        &self.frequencies
    }

    /// Total of all frequency counts currently held by the model
    ///
    /// After an adaptive [`ArithmeticEncoder::encode`] call this reflects the
    /// updated counts.
    pub fn total_count(&self) -> u64 {
        self.total_count
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
    /// Cumulative frequency table quantised onto the shared `[0, 2^16]` grid
    scaled_cum: Vec<(u32, u32, u32)>, // (symbol, scaled_low, scaled_high)
}

impl RangeEncoder {
    /// Create a new range encoder from frequencies
    ///
    /// The frequency table is quantised once, here, onto the coder's fixed
    /// `[0, 2^16]` grid. [`super::RangeDecoder::from_frequencies`] performs
    /// the identical quantisation, so both sides share byte-identical symbol
    /// intervals.
    ///
    /// # Errors
    ///
    /// * [`TokenizerError::EncodingError`] if `frequencies` is empty.
    /// * [`TokenizerError::InvalidConfig`] if every count is zero, or if the
    ///   alphabet has more than 65,536 distinct symbols — the grid cannot give
    ///   each of them a distinct interval, so the table is rejected instead of
    ///   producing a stream that decodes to the wrong symbols.
    pub fn from_frequencies(frequencies: HashMap<u32, u64>) -> TokenizerResult<Self> {
        if frequencies.is_empty() {
            return Err(TokenizerError::encoding(
                "encoding",
                "Cannot create range encoder from empty frequencies",
            ));
        }

        let scaled_cum = build_scaled_table(&frequencies)?;

        Ok(Self {
            frequencies,
            scaled_cum,
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
    ///
    /// # Errors
    ///
    /// [`TokenizerError::EncodingError`] if a symbol is not in the alphabet,
    /// or if more than `u32::MAX` symbols are supplied.
    pub fn encode(&self, symbols: &[u32]) -> TokenizerResult<Vec<u8>> {
        let symbol_count = u32::try_from(symbols.len()).map_err(|_| {
            TokenizerError::encoding(
                "encoding",
                format!(
                    "Range coder supports at most {} symbols per stream, got {}",
                    u32::MAX,
                    symbols.len()
                ),
            )
        })?;

        // Range coder state.
        let mut low: u64 = 0; // 33 bits: bit 32 may be a carry
        let mut range: u32 = 0xFFFFFFFF;
        let mut cache: u8 = 0;
        let mut cache_size: u64 = 1;
        let mut output: Vec<u8> = Vec::new();

        let scale_u32 = RANGE_SCALE as u32;

        for &symbol in symbols {
            // Find symbol in the pre-quantised cumulative table. The table is
            // sorted by symbol, so this is a binary search rather than a scan.
            let idx = self
                .scaled_cum
                .binary_search_by_key(&symbol, |&(s, _, _)| s)
                .map_err(|_| {
                    TokenizerError::encoding("serialization", format!("Unknown symbol: {}", symbol))
                })?;
            let (_, cum_low, cum_high) = self.scaled_cum.get(idx).copied().ok_or_else(|| {
                TokenizerError::InternalError(
                    "Range coder table index outside its own bounds".into(),
                )
            })?;

            // Update range (LZMA-style step computation).
            let step = range / scale_u32;
            low = low.wrapping_add((step as u64).wrapping_mul(cum_low as u64));
            range = step.wrapping_mul(cum_high.wrapping_sub(cum_low));

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
        result.extend_from_slice(&symbol_count.to_le_bytes());
        result.extend_from_slice(&output);

        Ok(result)
    }

    /// Get the frequency table
    pub fn frequencies(&self) -> &HashMap<u32, u64> {
        &self.frequencies
    }
}
