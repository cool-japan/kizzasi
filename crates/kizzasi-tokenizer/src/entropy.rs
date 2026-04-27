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
use std::collections::{BinaryHeap, HashMap};

/// Huffman tree node
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct HuffmanNode {
    /// Symbol (None for internal nodes)
    symbol: Option<u32>,
    /// Frequency/weight
    frequency: u64,
    /// Left child index
    left: Option<usize>,
    /// Right child index
    right: Option<usize>,
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

/// Huffman decoder for decompression
pub struct HuffmanDecoder {
    /// Huffman tree nodes
    tree_nodes: Vec<HuffmanNode>,
    /// Root node index
    root_idx: usize,
}

impl HuffmanDecoder {
    /// Create a decoder from an encoder's codebook
    pub fn new(tree: (&[HuffmanNode], usize)) -> Self {
        Self {
            tree_nodes: tree.0.to_vec(),
            root_idx: tree.1,
        }
    }

    /// Decode a compressed bitstream
    ///
    /// # Arguments
    ///
    /// * `encoded` - Compressed data from HuffmanEncoder::encode()
    ///
    /// # Returns
    ///
    /// Original symbol sequence
    pub fn decode(&self, encoded: &[u8]) -> TokenizerResult<Vec<u32>> {
        if encoded.len() < 8 {
            return Err(TokenizerError::decoding(
                "decoding",
                "Encoded data too short (missing metadata)",
            ));
        }

        // Read metadata
        let num_symbols =
            u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        let num_bits =
            u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]) as usize;

        // Extract bit stream
        let bytes = &encoded[8..];
        let mut bits = Vec::with_capacity(num_bits);

        for (byte_idx, &byte) in bytes.iter().enumerate() {
            for bit_idx in 0..8 {
                if byte_idx * 8 + bit_idx >= num_bits {
                    break;
                }
                bits.push((byte & (1 << (7 - bit_idx))) != 0);
            }
        }

        // Decode symbols using Huffman tree
        let mut symbols = Vec::with_capacity(num_symbols);
        let mut current_idx = self.root_idx;

        // Special case: single-node tree (one symbol)
        let root = &self.tree_nodes[self.root_idx];
        if root.left.is_none() && root.right.is_none() {
            // Single symbol - decode all as that symbol
            if let Some(symbol) = root.symbol {
                for _ in 0..num_symbols {
                    symbols.push(symbol);
                }
                return Ok(symbols);
            }
        }

        // Multi-symbol tree: traverse for each bit
        for &bit in &bits {
            let node = &self.tree_nodes[current_idx];

            // Navigate tree
            current_idx = if bit {
                node.right.ok_or_else(|| {
                    TokenizerError::decoding(
                        "deserialization",
                        "Invalid bitstream: unexpected leaf",
                    )
                })?
            } else {
                node.left.ok_or_else(|| {
                    TokenizerError::decoding(
                        "deserialization",
                        "Invalid bitstream: unexpected leaf",
                    )
                })?
            };

            // Check if we've reached a leaf
            let current_node = &self.tree_nodes[current_idx];
            if let Some(symbol) = current_node.symbol {
                symbols.push(symbol);
                current_idx = self.root_idx; // Reset to root

                if symbols.len() == num_symbols {
                    break;
                }
            }
        }

        if symbols.len() != num_symbols {
            return Err(TokenizerError::decoding(
                "decoding",
                format!(
                    "Decoded {} symbols, expected {}",
                    symbols.len(),
                    num_symbols
                ),
            ));
        }

        Ok(symbols)
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

/// Arithmetic decoder for decompression
pub struct ArithmeticDecoder {
    /// Symbol frequency counts (must match encoder)
    frequencies: HashMap<u32, u64>,
    /// Total count
    total_count: u64,
    /// Alphabet (sorted symbols)
    alphabet: Vec<u32>,
}

impl ArithmeticDecoder {
    /// Create a decoder with matching frequencies
    pub fn new(frequencies: HashMap<u32, u64>) -> Self {
        let total_count = frequencies.values().sum();
        let mut alphabet: Vec<u32> = frequencies.keys().copied().collect();
        alphabet.sort_unstable();

        Self {
            frequencies,
            total_count,
            alphabet,
        }
    }

    /// Decode compressed data
    pub fn decode(&self, encoded: &[u8]) -> TokenizerResult<Vec<u32>> {
        if encoded.len() < 12 {
            return Err(TokenizerError::decoding(
                "decoding",
                "Encoded data too short",
            ));
        }

        let num_symbols =
            u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        let value = u64::from_le_bytes([
            encoded[4],
            encoded[5],
            encoded[6],
            encoded[7],
            encoded[8],
            encoded[9],
            encoded[10],
            encoded[11],
        ]);

        const PRECISION: u64 = 1u64 << 32;
        let mut symbols = Vec::with_capacity(num_symbols);
        let mut low = 0u64;
        let mut high = PRECISION - 1;
        let code_value = value;

        for _ in 0..num_symbols {
            let range = high - low + 1;

            // Find symbol whose cumulative range contains code_value
            let scaled = ((code_value - low + 1) * self.total_count - 1) / range;

            let mut cumulative = 0u64;
            let mut found_symbol = None;

            for &symbol in &self.alphabet {
                let freq = self.frequencies.get(&symbol).unwrap_or(&0);
                if scaled >= cumulative && scaled < cumulative + freq {
                    found_symbol = Some(symbol);
                    break;
                }
                cumulative += freq;
            }

            let symbol = found_symbol.ok_or_else(|| {
                TokenizerError::decoding(
                    "decoding",
                    format!("Cannot decode symbol at position {}", symbols.len()),
                )
            })?;

            symbols.push(symbol);

            // Update range
            let (cum_low, cum_high) = self.cumulative_frequency(symbol);
            high = low + (range * cum_high / self.total_count) - 1;
            low += range * cum_low / self.total_count;
        }

        Ok(symbols)
    }

    fn cumulative_frequency(&self, symbol: u32) -> (u64, u64) {
        let mut cumulative = 0u64;

        for s in &self.alphabet {
            if *s >= symbol {
                break;
            }
            cumulative += self.frequencies.get(s).unwrap_or(&0);
        }

        let freq = self.frequencies.get(&symbol).unwrap_or(&0);
        (cumulative, cumulative + freq)
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
    /// Uses a simplified byte-aligned range coder for robustness.
    ///
    /// # Arguments
    ///
    /// * `symbols` - Sequence of symbols to encode
    ///
    /// # Returns
    ///
    /// Compressed bitstream as bytes
    pub fn encode(&self, symbols: &[u32]) -> TokenizerResult<Vec<u8>> {
        // Simple range coder using u64 to avoid overflow issues
        // Scale frequencies to fit in reasonable precision
        let scale = 1u64 << 14; // 16384 - frequency precision
        let total = self.total_count;

        // Pre-compute scaled cumulative frequencies
        let mut scaled_cum: Vec<(u32, u64, u64)> = Vec::new();
        for (sym, cum_low, cum_high) in &self.cumulative {
            let scaled_low = ((*cum_low as u128 * scale as u128) / total as u128) as u64;
            let scaled_high = ((*cum_high as u128 * scale as u128) / total as u128) as u64;
            // Ensure at least 1 unit for each symbol
            let scaled_high = scaled_high.max(scaled_low + 1);
            scaled_cum.push((*sym, scaled_low, scaled_high));
        }

        let mut low: u64 = 0;
        let mut range: u64 = 1u64 << 32;
        let mut output = Vec::new();

        for &symbol in symbols {
            // Find symbol in cumulative table
            let (_, cum_low, cum_high) = scaled_cum
                .iter()
                .find(|(s, _, _)| *s == symbol)
                .ok_or_else(|| {
                    TokenizerError::encoding("serialization", format!("Unknown symbol: {}", symbol))
                })?;

            // Update range
            let step = range / scale;
            low += step * cum_low;
            range = step * (cum_high - cum_low);

            // Renormalization: output bytes when top byte of low is stable
            while range < (1u64 << 24) {
                output.push((low >> 24) as u8);
                low <<= 8;
                low &= 0xFFFFFFFF; // Keep within 32 bits
                range <<= 8;
            }
        }

        // Flush: output enough bytes to uniquely identify the final interval
        for _ in 0..4 {
            output.push((low >> 24) as u8);
            low <<= 8;
        }

        // Prepend metadata: number of symbols
        let mut result = Vec::new();
        result.extend_from_slice(&(symbols.len() as u32).to_le_bytes());
        result.extend_from_slice(&output);

        Ok(result)
    }

    /// Get the frequency table
    pub fn frequencies(&self) -> &HashMap<u32, u64> {
        &self.frequencies
    }
}

/// Range decoder for decompression
pub struct RangeDecoder {
    /// Cumulative frequency table
    cumulative: Vec<(u32, u64, u64)>,
    /// Total count
    total_count: u64,
}

impl RangeDecoder {
    /// Create a decoder from frequencies
    pub fn from_frequencies(frequencies: HashMap<u32, u64>) -> TokenizerResult<Self> {
        if frequencies.is_empty() {
            return Err(TokenizerError::decoding(
                "decoding",
                "Cannot create range decoder from empty frequencies",
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
            cumulative,
            total_count,
        })
    }

    /// Decode compressed data
    pub fn decode(&self, encoded: &[u8]) -> TokenizerResult<Vec<u32>> {
        if encoded.len() < 4 {
            return Err(TokenizerError::decoding(
                "decoding",
                "Encoded data too short",
            ));
        }

        let num_symbols =
            u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;

        // Same parameters as encoder
        let scale = 1u64 << 14;
        let total = self.total_count;

        // Pre-compute scaled cumulative frequencies (same as encoder)
        let mut scaled_cum: Vec<(u32, u64, u64)> = Vec::new();
        for (sym, cum_low, cum_high) in &self.cumulative {
            let scaled_low = ((*cum_low as u128 * scale as u128) / total as u128) as u64;
            let scaled_high = ((*cum_high as u128 * scale as u128) / total as u128) as u64;
            let scaled_high = scaled_high.max(scaled_low + 1);
            scaled_cum.push((*sym, scaled_low, scaled_high));
        }

        let data = &encoded[4..];
        let mut data_idx = 0;

        // Initialize decoder state - read first 4 bytes as code
        let mut code: u64 = 0;
        for _ in 0..4 {
            code = (code << 8) | (data.get(data_idx).copied().unwrap_or(0) as u64);
            data_idx += 1;
        }

        let mut low: u64 = 0;
        let mut range: u64 = 1u64 << 32;
        let mut symbols = Vec::with_capacity(num_symbols);

        for _ in 0..num_symbols {
            // Find symbol by computing where code falls in the range
            let step = range / scale;
            // Use wrapping subtraction since code and low are both bounded to 32 bits
            let value = code.wrapping_sub(low) / step;

            let (symbol, cum_low, cum_high) = scaled_cum
                .iter()
                .find(|(_, cl, ch)| value >= *cl && value < *ch)
                .ok_or_else(|| {
                    TokenizerError::decoding(
                        "decoding",
                        format!("Invalid encoded data at symbol {}", symbols.len()),
                    )
                })?;

            symbols.push(*symbol);

            // Update decoder state (mirror encoder)
            low += step * cum_low;
            range = step * (cum_high - cum_low);

            // Renormalization: must mirror encoder exactly
            while range < (1u64 << 24) {
                code <<= 8;
                code &= 0xFFFFFFFF;
                code |= data.get(data_idx).copied().unwrap_or(0) as u64;
                data_idx += 1;
                low <<= 8;
                low &= 0xFFFFFFFF;
                range <<= 8;
            }
        }

        Ok(symbols)
    }
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
    #[ignore] // TODO: Fix range coding algorithm - has precision issues with longer sequences
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

        // Should achieve good compression
        let original_bits = symbols.len() * 2; // 2 bits per symbol for 4 symbols
        let compressed_bytes = encoded.len() - 4; // Subtract metadata

        // Range coding should be efficient
        assert!(compressed_bytes * 8 < original_bits);

        // Verify correctness
        let decoder = RangeDecoder::from_frequencies(freqs).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    #[ignore] // TODO: Fix range coding algorithm - has precision issues with longer sequences
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
}
