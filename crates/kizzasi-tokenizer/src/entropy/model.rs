//! Shared frequency-model and bit-stream primitives for the entropy coders.
//!
//! An entropy coder is correct only when its encoder and decoder agree
//! *exactly* on the probability model and on the renormalisation schedule.
//! Every quantity that both halves must reproduce bit-for-bit therefore lives
//! here rather than being duplicated in `encoder.rs` and `decoder.rs`:
//!
//! - [`FreqModel`] — the sorted symbol/count table used by the arithmetic
//!   coder, including the adaptive update and rescale rules.
//! - [`scaled_cumulative_table`] / [`build_scaled_table`] — the quantisation
//!   of an arbitrary frequency table onto the range coder's fixed
//!   `[0, RANGE_SCALE]` grid, guaranteeing strictly increasing, contiguous,
//!   non-overlapping intervals.
//! - [`BitWriter`] / [`BitReader`] — MSB-first bit packing with the implicit
//!   zero padding the arithmetic decoder relies on at end-of-stream.
//!
//! Nothing in this module is part of the public API.

use crate::error::{TokenizerError, TokenizerResult};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic-coder constants
// ─────────────────────────────────────────────────────────────────────────────

/// Width of the arithmetic coder's `low`/`high` registers, in bits.
///
/// The registers are held in `u64` so that intermediate products never wrap;
/// only the low [`PRECISION_BITS`] bits are significant.
pub(super) const PRECISION_BITS: u32 = 32;

/// One past the largest representable register value (`2^PRECISION_BITS`).
pub(super) const WHOLE: u64 = 1u64 << PRECISION_BITS;

/// Midpoint of the coding interval (the E2 boundary).
pub(super) const HALF: u64 = WHOLE / 2;

/// First quarter of the coding interval (the E1/E3 boundary).
pub(super) const QUARTER: u64 = WHOLE / 4;

/// Third quarter of the coding interval (the E3 boundary).
pub(super) const THREE_QUARTER: u64 = 3 * QUARTER;

/// Largest total frequency count the arithmetic coder can represent exactly.
///
/// After renormalisation the coding interval always satisfies
/// `range > QUARTER`, so a total of at most `QUARTER` guarantees every symbol
/// receives a sub-interval of width at least one — the condition that makes
/// the coder lossless.
pub(super) const MAX_TOTAL_COUNT: u64 = QUARTER;

/// Total count at which an adaptive model halves all of its counts.
///
/// Kept far below [`MAX_TOTAL_COUNT`] so adaptive updates can never drive the
/// model past the representable limit mid-stream.
pub(super) const ADAPTIVE_RESCALE_LIMIT: u64 = 1_000_000;

/// Smallest count an adaptive model will ever assign to a live symbol.
pub(super) const DEFAULT_MIN_COUNT: u64 = 1;

/// Size of the arithmetic bitstream header: `u32` symbol count + flag byte.
pub(super) const ARITHMETIC_HEADER_LEN: usize = 5;

/// Header flag: the payload was produced with adaptive frequency updates.
pub(super) const FLAG_ADAPTIVE: u8 = 0b0000_0001;

/// Every header flag bit this version understands.
pub(super) const KNOWN_FLAGS: u8 = FLAG_ADAPTIVE;

// ─────────────────────────────────────────────────────────────────────────────
// Range-coder constants
// ─────────────────────────────────────────────────────────────────────────────

/// Log2 of the range coder's cumulative-frequency grid.
pub(super) const RANGE_SCALE_BITS: u32 = 16;

/// The range coder quantises every frequency table onto `[0, RANGE_SCALE]`.
///
/// With a 32-bit `range` renormalised to at least `2^24`, the per-step divisor
/// `range / RANGE_SCALE` is always at least `2^8`, so no symbol can collapse
/// to a zero-width interval. The grid size also bounds the largest alphabet
/// the coder can represent: at most [`RANGE_SCALE`] distinct symbols.
pub(super) const RANGE_SCALE: u64 = 1u64 << RANGE_SCALE_BITS;

// ─────────────────────────────────────────────────────────────────────────────
// Frequency model
// ─────────────────────────────────────────────────────────────────────────────

/// Sorted symbol/count table with prefix sums.
///
/// Symbols with a zero count are dropped at construction time: a zero-width
/// interval cannot be coded, and silently coding it would desynchronise the
/// decoder. Because both the encoder and the decoder build the model through
/// this constructor from the same `HashMap`, both observe the same alphabet.
#[derive(Debug, Clone)]
pub(super) struct FreqModel {
    /// Sorted, de-duplicated alphabet (only symbols with a non-zero count).
    symbols: Vec<u32>,
    /// `counts[i]` is the current count of `symbols[i]`; always `>= 1`.
    counts: Vec<u64>,
    /// `prefix[i]` is the sum of `counts[..i]`; length is `counts.len() + 1`.
    prefix: Vec<u64>,
    /// Floor applied to counts when the model is rescaled.
    min_count: u64,
}

impl FreqModel {
    /// Build a model from a symbol/frequency map.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::InvalidConfig`] when no symbol has a
    /// non-zero frequency, or when the total count would overflow `u64`.
    pub(super) fn from_frequencies(
        frequencies: &HashMap<u32, u64>,
        min_count: u64,
    ) -> TokenizerResult<Self> {
        let mut symbols: Vec<u32> = frequencies
            .iter()
            .filter(|(_, &freq)| freq > 0)
            .map(|(&symbol, _)| symbol)
            .collect();
        symbols.sort_unstable();

        if symbols.is_empty() {
            return Err(TokenizerError::InvalidConfig(
                "Entropy model requires at least one symbol with a non-zero frequency".into(),
            ));
        }

        let counts: Vec<u64> = symbols
            .iter()
            .map(|symbol| frequencies.get(symbol).copied().unwrap_or(0))
            .collect();

        let mut model = Self {
            symbols,
            counts,
            prefix: Vec::new(),
            min_count: min_count.max(1),
        };
        model.rebuild_prefix()?;
        Ok(model)
    }

    /// Recompute the prefix-sum table from `counts`.
    fn rebuild_prefix(&mut self) -> TokenizerResult<()> {
        let mut prefix = Vec::with_capacity(self.counts.len() + 1);
        let mut acc: u64 = 0;
        prefix.push(0u64);
        for &count in &self.counts {
            acc = acc.checked_add(count).ok_or_else(|| {
                TokenizerError::InvalidConfig(
                    "Entropy model total frequency overflows a 64-bit counter".into(),
                )
            })?;
            prefix.push(acc);
        }
        self.prefix = prefix;
        Ok(())
    }

    /// Total of all counts in the model.
    pub(super) fn total(&self) -> u64 {
        self.prefix.last().copied().unwrap_or(0)
    }

    /// Number of live symbols in the alphabet.
    pub(super) fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Position of `symbol` in the sorted alphabet, if it is present.
    pub(super) fn index_of(&self, symbol: u32) -> Option<usize> {
        self.symbols.binary_search(&symbol).ok()
    }

    /// Symbol stored at alphabet position `idx`.
    pub(super) fn symbol_at(&self, idx: usize) -> Option<u32> {
        self.symbols.get(idx).copied()
    }

    /// Half-open cumulative-frequency interval `[low, high)` of position `idx`.
    pub(super) fn cumulative(&self, idx: usize) -> Option<(u64, u64)> {
        match (self.prefix.get(idx), self.prefix.get(idx + 1)) {
            (Some(&low), Some(&high)) => Some((low, high)),
            _ => None,
        }
    }

    /// Alphabet position whose cumulative interval contains `scaled`.
    ///
    /// Returns `None` when `scaled >= total()`, which for a decoder means the
    /// bitstream is corrupt rather than that a symbol is merely unusual.
    pub(super) fn find_by_cumulative(&self, scaled: u64) -> Option<usize> {
        // `prefix` is strictly increasing (every count is >= 1) and starts at
        // 0, so this partition point is always at least 1.
        let pos = self.prefix.partition_point(|&bound| bound <= scaled);
        if pos == 0 || pos > self.len() {
            None
        } else {
            Some(pos - 1)
        }
    }

    /// Apply one adaptive update to alphabet position `idx`.
    ///
    /// The encoder and the decoder call this at exactly the same point in the
    /// symbol loop, which is what keeps their models identical.
    pub(super) fn update(&mut self, idx: usize) -> TokenizerResult<()> {
        match self.counts.get_mut(idx) {
            Some(count) => *count = count.saturating_add(1),
            None => return Ok(()),
        }
        for bound in self.prefix.iter_mut().skip(idx + 1) {
            *bound = bound.saturating_add(1);
        }
        if self.total() > ADAPTIVE_RESCALE_LIMIT {
            self.rescale()?;
        }
        Ok(())
    }

    /// Halve every count (with a `min_count` floor) to bound the total.
    fn rescale(&mut self) -> TokenizerResult<()> {
        let min_count = self.min_count;
        for count in self.counts.iter_mut() {
            *count = (*count / 2).max(min_count);
        }
        self.rebuild_prefix()
    }

    /// Copy the current counts back into `frequencies`, returning the total.
    ///
    /// Symbols that were dropped at construction time (zero count) are left
    /// untouched, so the caller's map keeps its original shape.
    pub(super) fn write_back(&self, frequencies: &mut HashMap<u32, u64>) -> u64 {
        for (symbol, count) in self.symbols.iter().zip(self.counts.iter()) {
            frequencies.insert(*symbol, *count);
        }
        self.total()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Range-coder frequency quantisation
// ─────────────────────────────────────────────────────────────────────────────

/// Quantise `counts` onto the `[0, scale]` grid as a contiguous cumulative
/// table.
///
/// The returned entries are `(symbol, scaled_low, scaled_high)` with
/// `scaled_low[0] == 0`, `scaled_low[i] == scaled_high[i - 1]`, and
/// `scaled_high[n - 1] == scale`. Every interval has width at least one, so no
/// two symbols can ever share a point of the grid.
///
/// This is what makes low-probability symbols safe. Naively scaling each
/// cumulative bound independently and clamping the width up to one leaves the
/// *following* symbol's lower bound unchanged, so the two intervals overlap
/// and a decoder resolving a value in the shared region silently returns the
/// wrong symbol. Here the widths are allocated first (one unit to every
/// symbol, the remainder distributed proportionally by largest remainder) and
/// the bounds are derived from them, which makes overlap unrepresentable.
///
/// # Errors
///
/// Returns [`TokenizerError::InvalidConfig`] when the alphabet is empty, when
/// the total count is zero, or when the alphabet has more symbols than the
/// grid has units — the last case cannot be coded at all and is reported
/// rather than silently corrupting the stream.
pub(super) fn scaled_cumulative_table(
    counts: &[(u32, u64)],
    scale: u64,
) -> TokenizerResult<Vec<(u32, u32, u32)>> {
    if counts.is_empty() {
        return Err(TokenizerError::InvalidConfig(
            "Range coder requires at least one symbol with a non-zero frequency".into(),
        ));
    }
    if counts.len() as u64 > scale {
        return Err(TokenizerError::InvalidConfig(format!(
            "Range coder alphabet size {} exceeds the coder's resolution {}; \
             merge symbols or use a wider coder",
            counts.len(),
            scale
        )));
    }

    let total: u64 = counts.iter().map(|&(_, count)| count).sum();
    if total == 0 {
        return Err(TokenizerError::InvalidConfig(
            "Range coder frequency table has a total count of zero".into(),
        ));
    }

    // Every symbol gets one unit up front; the rest is shared proportionally.
    let remaining = scale - counts.len() as u64;
    let mut widths: Vec<u64> = Vec::with_capacity(counts.len());
    let mut remainders: Vec<(u64, usize)> = Vec::with_capacity(counts.len());
    let mut assigned: u64 = 0;

    for (idx, &(_, count)) in counts.iter().enumerate() {
        let numerator = u128::from(count) * u128::from(remaining);
        let share = (numerator / u128::from(total)) as u64;
        let remainder = (numerator % u128::from(total)) as u64;
        let width = 1 + share;
        assigned += width;
        widths.push(width);
        remainders.push((remainder, idx));
    }

    // `sum(floor(count_i * remaining / total)) > remaining - n`, so the
    // leftover is strictly smaller than the alphabet size.
    let leftover = scale.saturating_sub(assigned);
    if leftover > 0 {
        // Largest remainder first; ties broken by position so encoder and
        // decoder always produce byte-identical tables.
        remainders.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for &(_, idx) in remainders.iter().take(leftover as usize) {
            if let Some(width) = widths.get_mut(idx) {
                *width += 1;
            }
        }
    }

    let mut table = Vec::with_capacity(counts.len());
    let mut acc: u64 = 0;
    for (&(symbol, _), &width) in counts.iter().zip(widths.iter()) {
        let low = acc;
        acc += width;
        table.push((symbol, low as u32, acc as u32));
    }

    if acc != scale {
        return Err(TokenizerError::InternalError(format!(
            "Range coder scaling produced a total of {} instead of {}",
            acc, scale
        )));
    }

    Ok(table)
}

/// Build the range coder's quantised cumulative table from a frequency map.
///
/// Shared by [`super::RangeEncoder`] and [`super::RangeDecoder`] so both sides
/// derive byte-identical tables from identical inputs.
pub(super) fn build_scaled_table(
    frequencies: &HashMap<u32, u64>,
) -> TokenizerResult<Vec<(u32, u32, u32)>> {
    let mut counts: Vec<(u32, u64)> = frequencies
        .iter()
        .filter(|(_, &freq)| freq > 0)
        .map(|(&symbol, &freq)| (symbol, freq))
        .collect();
    counts.sort_unstable_by_key(|&(symbol, _)| symbol);
    scaled_cumulative_table(&counts, RANGE_SCALE)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit I/O
// ─────────────────────────────────────────────────────────────────────────────

/// MSB-first bit packer used by the arithmetic encoder.
pub(super) struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    filled: u32,
}

impl BitWriter {
    /// Create an empty writer.
    pub(super) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            filled: 0,
        }
    }

    /// Append a single bit.
    pub(super) fn write_bit(&mut self, bit: bool) {
        self.current = (self.current << 1) | u8::from(bit);
        self.filled += 1;
        if self.filled == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.filled = 0;
        }
    }

    /// Append `bit`, then `*pending` copies of its complement, then reset
    /// `*pending`.
    ///
    /// This is the E3 (underflow) resolution step: the deferred bits are only
    /// known once the interval escapes the middle half.
    pub(super) fn write_bit_with_pending(&mut self, bit: bool, pending: &mut u64) {
        self.write_bit(bit);
        let complement = !bit;
        for _ in 0..*pending {
            self.write_bit(complement);
        }
        *pending = 0;
    }

    /// Flush the partial byte (zero padded) and return the packed bytes.
    pub(super) fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            // `filled` is in 1..=7 here, so the shift is always well defined.
            self.current <<= 8 - self.filled;
            self.bytes.push(self.current);
            self.current = 0;
            self.filled = 0;
        }
        std::mem::take(&mut self.bytes)
    }
}

/// MSB-first bit reader that zero-fills past the end of the stream.
///
/// The zero fill is required, not merely convenient: the encoder's terminating
/// sequence disambiguates the final interval only when the decoder treats the
/// bits after the payload as zeros.
pub(super) struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Create a reader over `data`.
    pub(super) fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    /// Read the next bit, returning `false` once the stream is exhausted.
    pub(super) fn read_bit(&mut self) -> bool {
        let byte_idx = self.bit_pos / 8;
        let bit_idx = self.bit_pos % 8;
        self.bit_pos += 1;
        match self.data.get(byte_idx) {
            Some(&byte) => (byte >> (7 - bit_idx)) & 1 == 1,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_table_is_contiguous_and_covers_the_grid() {
        let counts = vec![(0u32, 1u64), (1u32, 20_000u64)];
        let table = scaled_cumulative_table(&counts, RANGE_SCALE).unwrap();

        assert_eq!(table.len(), 2);
        assert_eq!(table[0].1, 0);
        assert_eq!(table[table.len() - 1].2 as u64, RANGE_SCALE);
        for window in table.windows(2) {
            assert_eq!(
                window[0].2, window[1].1,
                "intervals must be contiguous, got {:?}",
                window
            );
        }
        for &(symbol, low, high) in &table {
            assert!(high > low, "symbol {} has zero width", symbol);
        }
    }

    #[test]
    fn test_scaled_table_rejects_oversized_alphabet() {
        let counts: Vec<(u32, u64)> = (0..(RANGE_SCALE as u32 + 1)).map(|s| (s, 1u64)).collect();
        let result = scaled_cumulative_table(&counts, RANGE_SCALE);
        assert!(
            matches!(result, Err(TokenizerError::InvalidConfig(_))),
            "alphabet larger than the grid must be rejected"
        );
    }

    #[test]
    fn test_scaled_table_full_grid_alphabet() {
        let counts: Vec<(u32, u64)> = (0..RANGE_SCALE as u32).map(|s| (s, 1u64)).collect();
        let table = scaled_cumulative_table(&counts, RANGE_SCALE).unwrap();
        assert_eq!(table.len(), RANGE_SCALE as usize);
        for &(_, low, high) in &table {
            assert_eq!(high - low, 1);
        }
    }

    #[test]
    fn test_bit_writer_reader_round_trip() {
        let bits = [true, false, true, true, false, false, false, true, true];
        let mut writer = BitWriter::new();
        for &bit in &bits {
            writer.write_bit(bit);
        }
        let bytes = writer.finish();
        assert_eq!(bytes.len(), 2);

        let mut reader = BitReader::new(&bytes);
        for &bit in &bits {
            assert_eq!(reader.read_bit(), bit);
        }
        // Past the end of the stream the reader yields zeros forever.
        for _ in 0..64 {
            assert!(!reader.read_bit());
        }
    }

    #[test]
    fn test_freq_model_lookup_and_update() {
        let mut frequencies = HashMap::new();
        frequencies.insert(3u32, 2u64);
        frequencies.insert(7u32, 5u64);
        frequencies.insert(9u32, 0u64); // dropped: zero width is not codeable

        let mut model = FreqModel::from_frequencies(&frequencies, DEFAULT_MIN_COUNT).unwrap();
        assert_eq!(model.len(), 2);
        assert_eq!(model.total(), 7);
        assert_eq!(model.index_of(9), None);

        let idx = model.index_of(7).unwrap();
        assert_eq!(model.cumulative(idx), Some((2, 7)));
        assert_eq!(model.find_by_cumulative(0), model.index_of(3));
        assert_eq!(model.find_by_cumulative(6), Some(idx));
        assert_eq!(model.find_by_cumulative(7), None);

        model.update(idx).unwrap();
        assert_eq!(model.total(), 8);
        assert_eq!(model.cumulative(idx), Some((2, 8)));
        assert_eq!(model.symbol_at(idx), Some(7));
    }
}
