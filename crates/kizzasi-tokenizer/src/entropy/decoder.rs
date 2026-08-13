//! Decoders for the entropy-coding module.
//!
//! This file hosts the decoder half of every entropy-coding algorithm
//! exposed by [`crate::entropy`]:
//!
//! - [`HuffmanDecoder`] – inverse of [`super::HuffmanEncoder`].
//! - [`ArithmeticDecoder`] – inverse of [`super::ArithmeticEncoder`].
//! - [`RangeDecoder`] – inverse of [`super::RangeEncoder`].
//!
//! Decoder constructors mirror their encoder counterparts so a round-trip
//! `encode → decode` always uses matching parameters.

use super::model::{
    build_scaled_table, BitReader, FreqModel, ARITHMETIC_HEADER_LEN, DEFAULT_MIN_COUNT,
    FLAG_ADAPTIVE, HALF, KNOWN_FLAGS, MAX_TOTAL_COUNT, PRECISION_BITS, QUARTER, RANGE_SCALE,
    THREE_QUARTER, WHOLE,
};
use super::HuffmanNode;
use crate::error::{TokenizerError, TokenizerResult};
use std::collections::HashMap;

/// Upper bound on the symbol capacity pre-allocated from an untrusted header.
///
/// The symbol count in a compressed stream is metadata: a corrupt or hostile
/// header could claim billions of symbols. Decoding still honours the declared
/// count, but the initial allocation is capped so a bad header cannot force a
/// multi-gigabyte reservation before a single bit has been read.
const MAX_PREALLOC_SYMBOLS: usize = 1 << 16;

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

/// Arithmetic decoder for decompression
///
/// Exact inverse of [`super::ArithmeticEncoder`]: 32-bit `low`/`high`
/// registers, the same E1/E2/E3 renormalisation schedule, and the same
/// frequency model. Bits past the end of the payload read as zero, which is
/// what makes the encoder's two-bit termination sequence sufficient.
///
/// # Matching the encoder
///
/// The decoder must be constructed from the *initial* frequency table the
/// encoder started with — for an adaptive stream that means the table before
/// any updates, not the table left behind afterwards. The stream itself
/// records whether it was produced adaptively, and the decoder replays the
/// identical update rule, so no extra flag has to be passed in.
pub struct ArithmeticDecoder {
    /// Symbol frequency counts (must match the encoder's initial table)
    frequencies: HashMap<u32, u64>,
}

impl ArithmeticDecoder {
    /// Create a decoder with matching frequencies
    pub fn new(frequencies: HashMap<u32, u64>) -> Self {
        Self { frequencies }
    }

    /// Decode compressed data
    ///
    /// # Errors
    ///
    /// * [`TokenizerError::DecodingError`] if the header is truncated, sets
    ///   flag bits this version does not understand, or if the payload does
    ///   not resolve to a symbol in the alphabet (a corrupt stream).
    /// * [`TokenizerError::InvalidConfig`] if the frequency table is empty or
    ///   its total exceeds the coder's 2^30 limit.
    pub fn decode(&self, encoded: &[u8]) -> TokenizerResult<Vec<u32>> {
        let count_bytes: [u8; 4] = encoded
            .get(..4)
            .and_then(|slice| slice.try_into().ok())
            .ok_or_else(|| {
                TokenizerError::decoding(
                    "decoding",
                    "Encoded data too short (missing arithmetic-coder header)",
                )
            })?;
        let num_symbols = u32::from_le_bytes(count_bytes) as usize;

        let flags = encoded.get(4).copied().ok_or_else(|| {
            TokenizerError::decoding(
                "decoding",
                "Encoded data too short (missing arithmetic-coder flag byte)",
            )
        })?;
        if flags & !KNOWN_FLAGS != 0 {
            return Err(TokenizerError::decoding(
                "decoding",
                format!(
                    "Unsupported arithmetic-coder flags 0x{:02x}; stream was written by a newer format",
                    flags
                ),
            ));
        }
        let adaptive = flags & FLAG_ADAPTIVE != 0;
        let payload = encoded.get(ARITHMETIC_HEADER_LEN..).unwrap_or(&[]);

        let mut model = FreqModel::from_frequencies(&self.frequencies, DEFAULT_MIN_COUNT)?;
        if model.total() > MAX_TOTAL_COUNT {
            return Err(TokenizerError::InvalidConfig(format!(
                "Arithmetic coder total frequency {} exceeds the maximum {}",
                model.total(),
                MAX_TOTAL_COUNT
            )));
        }

        let mut reader = BitReader::new(payload);
        let mut low: u64 = 0;
        let mut high: u64 = WHOLE - 1;
        let mut value: u64 = 0;
        for _ in 0..PRECISION_BITS {
            value = (value << 1) | u64::from(reader.read_bit());
        }

        let mut symbols = Vec::with_capacity(num_symbols.min(MAX_PREALLOC_SYMBOLS));

        for _ in 0..num_symbols {
            let range = high - low + 1;
            let total = model.total();

            let offset = value.checked_sub(low).ok_or_else(|| {
                TokenizerError::decoding(
                    "decoding",
                    format!(
                        "Corrupt arithmetic stream: code fell below the interval at symbol {}",
                        symbols.len()
                    ),
                )
            })?;
            let scaled = ((offset + 1) * total - 1) / range;

            let idx = model.find_by_cumulative(scaled).ok_or_else(|| {
                TokenizerError::decoding(
                    "decoding",
                    format!("Cannot decode symbol at position {}", symbols.len()),
                )
            })?;
            let (cum_low, cum_high) = model.cumulative(idx).ok_or_else(|| {
                TokenizerError::InternalError(
                    "Arithmetic model index outside its own cumulative table".into(),
                )
            })?;
            let symbol = model.symbol_at(idx).ok_or_else(|| {
                TokenizerError::InternalError(
                    "Arithmetic model index outside its own alphabet".into(),
                )
            })?;
            symbols.push(symbol);

            // Narrow the interval exactly as the encoder did.
            high = low + range * cum_high / total - 1;
            low += range * cum_low / total;

            // Mirror the encoder's E1/E2/E3 renormalisation.
            loop {
                if high < HALF {
                    // E1: nothing to strip from `value`.
                } else if low >= HALF {
                    value -= HALF;
                    low -= HALF;
                    high -= HALF;
                } else if low >= QUARTER && high < THREE_QUARTER {
                    value -= QUARTER;
                    low -= QUARTER;
                    high -= QUARTER;
                } else {
                    break;
                }
                low <<= 1;
                high = (high << 1) | 1;
                value = (value << 1) | u64::from(reader.read_bit());
            }

            if adaptive {
                model.update(idx)?;
            }
        }

        Ok(symbols)
    }
}

/// Range decoder for decompression
pub struct RangeDecoder {
    /// Cumulative frequency table quantised onto the shared `[0, 2^16]` grid
    scaled_cum: Vec<(u32, u32, u32)>,
}

impl RangeDecoder {
    /// Create a decoder from frequencies
    ///
    /// Quantises the frequency table through the same routine
    /// [`super::RangeEncoder::from_frequencies`] uses, so both sides agree on
    /// every symbol interval.
    ///
    /// # Errors
    ///
    /// * [`TokenizerError::DecodingError`] if `frequencies` is empty.
    /// * [`TokenizerError::InvalidConfig`] if every count is zero, or if the
    ///   alphabet has more than 65,536 distinct symbols.
    pub fn from_frequencies(frequencies: HashMap<u32, u64>) -> TokenizerResult<Self> {
        if frequencies.is_empty() {
            return Err(TokenizerError::decoding(
                "decoding",
                "Cannot create range decoder from empty frequencies",
            ));
        }

        let scaled_cum = build_scaled_table(&frequencies)?;

        Ok(Self { scaled_cum })
    }

    /// Decode compressed data
    ///
    /// Mirrors the LZMA-style encoder: skips the encoder's cache placeholder
    /// byte, reads the next four bytes as the initial code, then tracks only
    /// `code: u32` and `range: u32` (no `low`).
    pub fn decode(&self, encoded: &[u8]) -> TokenizerResult<Vec<u32>> {
        let count_bytes: [u8; 4] = encoded
            .get(..4)
            .and_then(|slice| slice.try_into().ok())
            .ok_or_else(|| TokenizerError::decoding("decoding", "Encoded data too short"))?;
        let num_symbols = u32::from_le_bytes(count_bytes) as usize;

        let data = encoded.get(4..).unwrap_or(&[]);
        let mut data_idx = 0usize;

        // Discard the encoder's initial cache placeholder byte.
        if !data.is_empty() {
            data_idx += 1;
        }

        // Initialize the decoder state by reading the next 4 bytes as `code`.
        let mut code: u32 = 0;
        for _ in 0..4 {
            let next = data.get(data_idx).copied().unwrap_or(0);
            code = (code << 8) | (next as u32);
            data_idx += 1;
        }

        let mut range: u32 = 0xFFFFFFFF;
        let mut symbols = Vec::with_capacity(num_symbols.min(MAX_PREALLOC_SYMBOLS));
        let scale_u32 = RANGE_SCALE as u32;

        for _ in 0..num_symbols {
            // Find symbol whose [cum_low, cum_high) contains v = code / step.
            let step = range / scale_u32;
            // Guard: step must be positive; renormalization should ensure this.
            if step == 0 {
                return Err(TokenizerError::decoding(
                    "decoding",
                    format!("Range coder underflow at symbol {}: step==0", symbols.len()),
                ));
            }
            let v = code / step;
            // Clamp to scale-1 so we still find a symbol even if rounding
            // makes v == scale exactly at the top of the alphabet.
            let v_clamped = v.min(scale_u32 - 1);

            // The table is contiguous and strictly increasing, so the owning
            // interval is the last one whose lower bound is <= v.
            let pos = self
                .scaled_cum
                .partition_point(|&(_, cum_low, _)| cum_low <= v_clamped);
            let (symbol, cum_low, cum_high) = pos
                .checked_sub(1)
                .and_then(|idx| self.scaled_cum.get(idx))
                .copied()
                .filter(|&(_, _, cum_high)| v_clamped < cum_high)
                .ok_or_else(|| {
                    TokenizerError::decoding(
                        "decoding",
                        format!("Invalid encoded data at symbol {}", symbols.len()),
                    )
                })?;

            symbols.push(symbol);

            // Update decoder state (mirror encoder).
            code = code.wrapping_sub(step.wrapping_mul(cum_low));
            range = step.wrapping_mul(cum_high.wrapping_sub(cum_low));

            // Renormalization: must mirror encoder exactly. Pull in bytes
            // until `range` is back above 2^24.
            while range < (1u32 << 24) {
                let next = data.get(data_idx).copied().unwrap_or(0);
                data_idx += 1;
                code = (code << 8) | (next as u32);
                range <<= 8;
            }
        }

        Ok(symbols)
    }
}
