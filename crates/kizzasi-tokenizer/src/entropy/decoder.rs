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

use super::HuffmanNode;
use crate::error::{TokenizerError, TokenizerResult};
use std::collections::HashMap;

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
    ///
    /// Mirrors the LZMA-style encoder: skips the encoder's cache placeholder
    /// byte, reads the next four bytes as the initial code, then tracks only
    /// `code: u32` and `range: u32` (no `low`).
    pub fn decode(&self, encoded: &[u8]) -> TokenizerResult<Vec<u32>> {
        if encoded.len() < 4 {
            return Err(TokenizerError::decoding(
                "decoding",
                "Encoded data too short",
            ));
        }

        let num_symbols =
            u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;

        // Same parameters as encoder.
        let scale: u64 = 1u64 << 14;
        let total = self.total_count;

        // Pre-compute scaled cumulative frequencies (same as encoder).
        let mut scaled_cum: Vec<(u32, u32, u32)> = Vec::with_capacity(self.cumulative.len());
        for (sym, cum_low, cum_high) in &self.cumulative {
            let scaled_low = ((*cum_low as u128 * scale as u128) / total as u128) as u64;
            let scaled_high = ((*cum_high as u128 * scale as u128) / total as u128) as u64;
            let scaled_high = scaled_high.max(scaled_low + 1);
            scaled_cum.push((*sym, scaled_low as u32, scaled_high as u32));
        }

        let data = &encoded[4..];
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
        let mut symbols = Vec::with_capacity(num_symbols);
        let scale_u32 = scale as u32;

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

            let (symbol, cum_low, cum_high) = scaled_cum
                .iter()
                .find(|(_, cl, ch)| v_clamped >= *cl && v_clamped < *ch)
                .ok_or_else(|| {
                    TokenizerError::decoding(
                        "decoding",
                        format!("Invalid encoded data at symbol {}", symbols.len()),
                    )
                })?;

            symbols.push(*symbol);

            // Update decoder state (mirror encoder).
            code = code.wrapping_sub(step.wrapping_mul(*cum_low));
            range = step.wrapping_mul(cum_high.wrapping_sub(*cum_low));

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
