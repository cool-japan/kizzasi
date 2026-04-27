//! K-quant dequantization for GGUF format.
//!
//! Implements the five K-quantization schemes used in llama.cpp and GGUF files:
//! Q2_K, Q3_K, Q4_K, Q5_K, and Q8_K. Each function converts a raw byte buffer
//! representing one or more quantized blocks into a `Vec<f32>` of decoded weights.
//!
//! # Block layouts
//!
//! All K-quant types use 256-element super-blocks with internal sub-blocks of 16 or
//! 32 elements, each carrying its own scale (and optional minimum) encoded in a compact
//! bit-packed format.
//!
//! The scale packing for Q2_K through Q5_K uses 6-bit (or 4-bit for Q2_K) fields
//! stored in 12 bytes at a fixed offset inside each super-block. See the individual
//! function docs for the exact field layout.

use crate::error::{ModelError, ModelResult};

// ─────────────────────────────────────────────────────────────────────────────
// Shared utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a raw IEEE-754 f16 bit-pattern to f32.
///
/// Used instead of `half::f16` where we already have the raw `u16` bits from a
/// little-endian byte pair and want a plain `f32` without pulling in the full
/// `half` API. The `half` crate is already a workspace dependency and is used
/// elsewhere; this function is a thin, zero-overhead wrapper over the same
/// conversion that `half::f16::to_f32()` performs.
#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Read a little-endian f16 (2 bytes) from `data` at `offset` and return as f32.
#[inline]
fn read_f16_le(data: &[u8], offset: usize) -> ModelResult<f32> {
    if offset + 2 > data.len() {
        return Err(ModelError::simple_load_error(format!(
            "read_f16_le: offset {} + 2 exceeds buffer length {}",
            offset,
            data.len()
        )));
    }
    let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
    Ok(f16_bits_to_f32(bits))
}

/// Read a little-endian f32 (4 bytes) from `data` at `offset`.
#[inline]
fn read_f32_le(data: &[u8], offset: usize) -> ModelResult<f32> {
    if offset + 4 > data.len() {
        return Err(ModelError::simple_load_error(format!(
            "read_f32_le: offset {} + 4 exceeds buffer length {}",
            offset,
            data.len()
        )));
    }
    Ok(f32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

// ─────────────────────────────────────────────────────────────────────────────
// Q2_K — 84 bytes / 256 elements
// ─────────────────────────────────────────────────────────────────────────────
//
// Super-block layout (84 bytes total):
//   [0..2]   d      — f16 super-block scale
//   [2..4]   dmin   — f16 super-block minimum scale
//   [4..16]  scales — 12 bytes, 4-bit packed: 16 nibbles encode 8 sub-scales
//                     and 8 sub-mins interleaved (scale0, min0, scale1, min1 …)
//   [16..80] qs     — 64 bytes of 2-bit packed quants (4 quants per byte → 256)
//
// Each super-block is divided into 16 sub-blocks of 16 elements.
// For sub-block `s` (0..16):
//   scale_s = d    * (scales nibble 2s)
//   min_s   = dmin * (scales nibble 2s+1)
//   output[s*16 + i] = scale_s * q[s*16 + i] - min_s
//
// The 2-bit quants are packed 4-per-byte, LSB-first, in the qs region.

/// Decode a Q2_K quantized tensor.
///
/// `data` must contain exactly `(n / 256) * 84` bytes.
/// `n` must be a non-zero multiple of 256.
pub(crate) fn dequant_q2_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_ELEMS: usize = 256;
    const BLOCK_BYTES: usize = 84;
    if n == 0 || !n.is_multiple_of(BLOCK_ELEMS) {
        return Err(ModelError::simple_load_error(format!(
            "Q2_K: n_elements {} must be a non-zero multiple of {}",
            n, BLOCK_ELEMS
        )));
    }
    let n_blocks = n / BLOCK_ELEMS;
    let required = n_blocks * BLOCK_BYTES;
    if data.len() < required {
        return Err(ModelError::simple_load_error(format!(
            "Q2_K: buffer too small: need {} bytes, got {}",
            required,
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(n);

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;

        // Super-block scale/min
        let d = read_f16_le(data, base)?;
        let dmin = read_f16_le(data, base + 2)?;

        // 12 bytes of 4-bit packed sub-block scales and mins.
        // 12 bytes = 24 nibbles; nibble 2s → scale_s, nibble 2s+1 → min_s  (s = 0..16 = NUM_SUB_BLOCKS)
        // But 24 nibbles only give 12 pairs, so NUM_SUB_BLOCKS = 16 needs 16 pairs = 32 nibbles.
        // llama.cpp Q2_K uses only 8 distinct scales and 8 distinct mins (one per 32-element group),
        // each 4-bit, stored in 12 bytes (= 8 * 4-bit + 8 * 4-bit bytes, packed as nibble pairs).
        // The 256-element block thus has 8 "groups" of 32 elements.
        const NUM_GROUPS: usize = 8; // 256 / 32
        const GROUP_SIZE: usize = 32;

        // Read 16 nibbles from scales[4..16]: first 8 are scales, last 8 are mins,
        // but they are interleaved in nibble pairs: scale0 in low nibble of byte 4,
        // min0 in high nibble of byte 4, scale1 in low nibble of byte 5, etc.
        let scales_base = base + 4;
        let mut sub_scales = [0u8; NUM_GROUPS];
        let mut sub_mins = [0u8; NUM_GROUPS];
        for g in 0..NUM_GROUPS {
            let byte = data[scales_base + g]; // one byte per group (low = scale, high = min)
            sub_scales[g] = byte & 0x0F;
            sub_mins[g] = (byte >> 4) & 0x0F;
        }

        // 2-bit quants: 64 bytes starting at base+16, 4 quants per byte
        let qs_base = base + 16;

        for g in 0..NUM_GROUPS {
            let scale = d * sub_scales[g] as f32;
            let min = dmin * sub_mins[g] as f32;

            // Each group covers GROUP_SIZE elements; each element is 2 bits.
            // The quants for group g occupy bytes [g*8 .. g*8+8] in qs (since 32 quants / 4 per byte = 8 bytes).
            let qs_group_base = qs_base + g * (GROUP_SIZE / 4);
            for byte_idx in 0..(GROUP_SIZE / 4) {
                let byte = data[qs_group_base + byte_idx];
                // 4 quants per byte, LSB first
                for shift in [0u8, 2, 4, 6] {
                    let q = ((byte >> shift) & 0x03) as f32;
                    out.push(scale * q - min);
                }
            }
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Q3_K — 110 bytes / 256 elements
// ─────────────────────────────────────────────────────────────────────────────
//
// Super-block layout (110 bytes total — matches llama.cpp `block_q3_K`):
//   [0..32]    hmask  — 32 bytes; 1 high-bit per quant (256 bits)
//   [32..96]   qs     — 64 bytes; low 2 bits of each quant, 4-per-byte LSB first (256 quants)
//   [96..108]  scales — 12 bytes; 6-bit packed sub-block scales (16 sub-blocks × 6 bits)
//   [108..110] d      — f16 super-block scale delta
//
// Reconstruction:
//   q3  = (low2bits) | ((hmask_bit) << 2)  → range 0..7
//   val = d * scale_s * (q3 - 4)
//
// Scale packing (12 bytes = 96 bits, 16 × 6-bit values):
//   scales[i] (6-bit) for sub-block i (0..16), stored LSB-first across the 12 bytes.

/// Decode a Q3_K quantized tensor.
///
/// `data` must contain exactly `(n / 256) * 110` bytes.
/// `n` must be a non-zero multiple of 256.
pub(crate) fn dequant_q3_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_ELEMS: usize = 256;
    const BLOCK_BYTES: usize = 110;
    const SUB_BLOCK_SIZE: usize = 16;
    const NUM_SUB_BLOCKS: usize = BLOCK_ELEMS / SUB_BLOCK_SIZE; // 16

    if n == 0 || !n.is_multiple_of(BLOCK_ELEMS) {
        return Err(ModelError::simple_load_error(format!(
            "Q3_K: n_elements {} must be a non-zero multiple of {}",
            n, BLOCK_ELEMS
        )));
    }
    let n_blocks = n / BLOCK_ELEMS;
    let required = n_blocks * BLOCK_BYTES;
    if data.len() < required {
        return Err(ModelError::simple_load_error(format!(
            "Q3_K: buffer too small: need {} bytes, got {}",
            required,
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(n);

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;

        let hmask = &data[base..base + 32];
        let qs = &data[base + 32..base + 96]; // 64 bytes: low 2 bits of all 256 quants
        let scales_raw = &data[base + 96..base + 108]; // 12 bytes = 96 bits for 16 × 6-bit scales
        let d = read_f16_le(data, base + 108)?;

        // Decode 16 sub-block scales from 12 bytes (16 × 6-bit, LSB first packed)
        let sub_scales = decode_6bit_scales_16(scales_raw)?;

        // Decode 256 quants
        // qs[i] holds the low 2 bits of quants i*4 .. i*4+3 (4 per byte, LSB first)
        // hmask[i] holds the high bit of quants i*8 .. i*8+7 (one bit per quant)
        for (s, &raw_scale_u8) in sub_scales.iter().enumerate().take(NUM_SUB_BLOCKS) {
            // The scale for this sub-block. The 6-bit scale is sign-extended from the
            // middle; llama.cpp stores them as signed 6-bit values with bias.
            // Raw packed value is 0..63; subtract 32 to get signed range -32..31.
            let raw_scale = raw_scale_u8 as i32 - 32;
            let scale = d * raw_scale as f32;

            let elem_base = s * SUB_BLOCK_SIZE;
            for i in 0..SUB_BLOCK_SIZE {
                let elem = elem_base + i;
                // Low 2 bits: qs byte index = elem / 4, bit shift = (elem % 4) * 2
                let qs_byte = qs[elem / 4];
                let low2 = (qs_byte >> ((elem % 4) * 2)) & 0x03;
                // High bit: hmask byte index = elem / 8, bit index = elem % 8
                let hm_byte = hmask[elem / 8];
                let high1 = (hm_byte >> (elem % 8)) & 0x01;
                let q3 = (low2 | (high1 << 2)) as i32;
                out.push(scale * (q3 - 4) as f32);
            }
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Q4_K — 144 bytes / 256 elements
// ─────────────────────────────────────────────────────────────────────────────
//
// Super-block layout (144 bytes total):
//   [0..2]    d      — f16 super-block scale
//   [2..4]    dmin   — f16 super-block minimum scale
//   [4..16]   scales — 12 bytes; 6-bit packed: 8 sub-scales + 8 sub-mins (see below)
//   [16..144] qs     — 128 bytes; 4-bit nibbles, 2 quants per byte, 256 quants total
//
// Scale packing: 12 bytes encode 8 (scale, min) pairs, each 6 bits wide.
//   Bits [0..47] = scales[0..7], bits [48..95] = mins[0..7]
//   (both in LSB-first bit order across the 12 bytes)
//
// Each super-block has 8 sub-blocks of 32 elements. For sub-block s:
//   val = d * scale_s * q - dmin * min_s
// where q is the 4-bit value (range 0..15).

/// Decode a Q4_K quantized tensor.
///
/// `data` must contain exactly `(n / 256) * 144` bytes.
/// `n` must be a non-zero multiple of 256.
pub(crate) fn dequant_q4_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_ELEMS: usize = 256;
    const BLOCK_BYTES: usize = 144;
    const NUM_SUB_BLOCKS: usize = 8;
    const SUB_BLOCK_SIZE: usize = BLOCK_ELEMS / NUM_SUB_BLOCKS; // 32

    if n == 0 || !n.is_multiple_of(BLOCK_ELEMS) {
        return Err(ModelError::simple_load_error(format!(
            "Q4_K: n_elements {} must be a non-zero multiple of {}",
            n, BLOCK_ELEMS
        )));
    }
    let n_blocks = n / BLOCK_ELEMS;
    let required = n_blocks * BLOCK_BYTES;
    if data.len() < required {
        return Err(ModelError::simple_load_error(format!(
            "Q4_K: buffer too small: need {} bytes, got {}",
            required,
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(n);

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;

        let d = read_f16_le(data, base)?;
        let dmin = read_f16_le(data, base + 2)?;
        let scales_raw = &data[base + 4..base + 16]; // 12 bytes
        let qs = &data[base + 16..base + 144]; // 128 bytes

        // Decode 8 sub-scales and 8 sub-mins from 12 bytes.
        // Layout: bits [0..47] → 8 × 6-bit scales, bits [48..95] → 8 × 6-bit mins.
        let (sub_scales, sub_mins) = decode_6bit_scales_and_mins_8(scales_raw)?;

        for s in 0..NUM_SUB_BLOCKS {
            let scale = d * sub_scales[s] as f32;
            let min = dmin * sub_mins[s] as f32;

            // Each sub-block has SUB_BLOCK_SIZE quants packed as nibbles.
            // 32 quants / 2 per byte = 16 bytes per sub-block.
            let qs_base = s * (SUB_BLOCK_SIZE / 2); // 16 bytes per sub-block

            for byte_idx in 0..(SUB_BLOCK_SIZE / 2) {
                let byte = qs[qs_base + byte_idx];
                let lo = (byte & 0x0F) as f32;
                let hi = ((byte >> 4) & 0x0F) as f32;
                out.push(scale * lo - min);
                out.push(scale * hi - min);
            }
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Q5_K — 176 bytes / 256 elements
// ─────────────────────────────────────────────────────────────────────────────
//
// Super-block layout (176 bytes total):
//   [0..2]    d      — f16 super-block scale
//   [2..4]    dmin   — f16 super-block minimum scale
//   [4..16]   scales — 12 bytes; 6-bit packed (same layout as Q4_K)
//   [16..48]  qh     — 32 bytes; high bit of each quant (1 bit per element, 256 bits)
//   [48..176] qs     — 128 bytes; low 4 bits of each quant (2 nibbles per byte)
//
// Reconstruction:
//   q5 = (low4bits) | (highbit << 4)   → range 0..31
//   val = d * scale_s * q5 - dmin * min_s

/// Decode a Q5_K quantized tensor.
///
/// `data` must contain exactly `(n / 256) * 176` bytes.
/// `n` must be a non-zero multiple of 256.
pub(crate) fn dequant_q5_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_ELEMS: usize = 256;
    const BLOCK_BYTES: usize = 176;
    const NUM_SUB_BLOCKS: usize = 8;
    const SUB_BLOCK_SIZE: usize = BLOCK_ELEMS / NUM_SUB_BLOCKS; // 32

    if n == 0 || !n.is_multiple_of(BLOCK_ELEMS) {
        return Err(ModelError::simple_load_error(format!(
            "Q5_K: n_elements {} must be a non-zero multiple of {}",
            n, BLOCK_ELEMS
        )));
    }
    let n_blocks = n / BLOCK_ELEMS;
    let required = n_blocks * BLOCK_BYTES;
    if data.len() < required {
        return Err(ModelError::simple_load_error(format!(
            "Q5_K: buffer too small: need {} bytes, got {}",
            required,
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(n);

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;

        let d = read_f16_le(data, base)?;
        let dmin = read_f16_le(data, base + 2)?;
        let scales_raw = &data[base + 4..base + 16]; // 12 bytes
        let qh = &data[base + 16..base + 48]; // 32 bytes (256 high bits)
        let qs = &data[base + 48..base + 176]; // 128 bytes (256 low nibbles)

        let (sub_scales, sub_mins) = decode_6bit_scales_and_mins_8(scales_raw)?;

        for s in 0..NUM_SUB_BLOCKS {
            let scale = d * sub_scales[s] as f32;
            let min = dmin * sub_mins[s] as f32;

            // Each sub-block has 32 quants.
            // Low nibbles: 16 bytes in qs starting at s*16
            // High bits: 1 bit per quant in qh; element (s*32 + i) → qh byte = (s*32+i)/8, bit = (s*32+i)%8
            let elem_base = s * SUB_BLOCK_SIZE;
            let qs_base = s * (SUB_BLOCK_SIZE / 2); // 16 bytes

            for byte_idx in 0..(SUB_BLOCK_SIZE / 2) {
                let byte = qs[qs_base + byte_idx];
                let lo4 = (byte & 0x0F) as u32;
                let hi4 = ((byte >> 4) & 0x0F) as u32;

                let elem_lo = elem_base + byte_idx * 2;
                let elem_hi = elem_base + byte_idx * 2 + 1;

                let hbit_lo = ((qh[elem_lo / 8] >> (elem_lo % 8)) & 0x01) as u32;
                let hbit_hi = ((qh[elem_hi / 8] >> (elem_hi % 8)) & 0x01) as u32;

                let q_lo = (lo4 | (hbit_lo << 4)) as f32;
                let q_hi = (hi4 | (hbit_hi << 4)) as f32;

                out.push(scale * q_lo - min);
                out.push(scale * q_hi - min);
            }
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Q8_K — 292 bytes / 256 elements
// ─────────────────────────────────────────────────────────────────────────────
//
// Super-block layout (292 bytes total):
//   [0..4]    d      — f32 super-block scale
//   [4..260]  qs     — 256 bytes; i8 quantized values
//   [260..276] bsums — 8 × i16 partial sums (unused for inference dequant)
//   [276..292] padding
//
// Reconstruction:
//   val = d * qs[i]

/// Decode a Q8_K quantized tensor.
///
/// `data` must contain exactly `(n / 256) * 292` bytes.
/// `n` must be a non-zero multiple of 256.
pub(crate) fn dequant_q8_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_ELEMS: usize = 256;
    const BLOCK_BYTES: usize = 292;

    if n == 0 || !n.is_multiple_of(BLOCK_ELEMS) {
        return Err(ModelError::simple_load_error(format!(
            "Q8_K: n_elements {} must be a non-zero multiple of {}",
            n, BLOCK_ELEMS
        )));
    }
    let n_blocks = n / BLOCK_ELEMS;
    let required = n_blocks * BLOCK_BYTES;
    if data.len() < required {
        return Err(ModelError::simple_load_error(format!(
            "Q8_K: buffer too small: need {} bytes, got {}",
            required,
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(n);

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let d = read_f32_le(data, base)?;
        for i in 0..BLOCK_ELEMS {
            let q = data[base + 4 + i] as i8;
            out.push(d * q as f32);
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Scale-decoding helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Decode 16 × 6-bit sub-block scales from 12 bytes.
///
/// The 12 bytes (= 96 bits) are read LSB-first; successive groups of 6 bits
/// give the raw unsigned 6-bit value for each of the 16 sub-blocks.
///
/// Used by Q3_K.
fn decode_6bit_scales_16(raw: &[u8]) -> ModelResult<[u8; 16]> {
    if raw.len() < 12 {
        return Err(ModelError::simple_load_error(
            "decode_6bit_scales_16: need 12 bytes",
        ));
    }

    // Concatenate the 12 bytes into a 96-bit little-endian integer and extract
    // 16 × 6-bit fields.
    let mut result = [0u8; 16];

    // We'll track our bit cursor manually.
    let mut bit_buf: u32 = 0;
    let mut bits_available: u32 = 0;
    let mut byte_idx = 0usize;

    for slot in &mut result {
        // Refill bit buffer as needed
        while bits_available < 6 {
            if byte_idx >= 12 {
                return Err(ModelError::simple_load_error(
                    "decode_6bit_scales_16: unexpectedly exhausted input bytes",
                ));
            }
            bit_buf |= (raw[byte_idx] as u32) << bits_available;
            bits_available += 8;
            byte_idx += 1;
        }
        *slot = (bit_buf & 0x3F) as u8;
        bit_buf >>= 6;
        bits_available -= 6;
    }

    Ok(result)
}

/// Decode 8 × 6-bit sub-scales and 8 × 6-bit sub-mins from 12 bytes.
///
/// Layout: bits [0..47] → 8 sub-scales, bits [48..95] → 8 sub-mins.
/// Each value is an unsigned 6-bit integer (0..63).
///
/// Used by Q4_K and Q5_K.
fn decode_6bit_scales_and_mins_8(raw: &[u8]) -> ModelResult<([u8; 8], [u8; 8])> {
    if raw.len() < 12 {
        return Err(ModelError::simple_load_error(
            "decode_6bit_scales_and_mins_8: need 12 bytes",
        ));
    }

    // 12 bytes = 96 bits. First 48 bits = 8 × 6-bit scales, next 48 bits = 8 × 6-bit mins.
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    let mut bit_buf: u64 = 0;
    let mut bits_available: u32 = 0;
    let mut byte_idx = 0usize;

    // Helper closure — returns the next 6-bit value from the stream.
    // (We use a manual loop here instead of a closure to avoid borrow issues.)
    let next_6bits =
        |bit_buf: &mut u64, bits_available: &mut u32, byte_idx: &mut usize| -> ModelResult<u8> {
            while *bits_available < 6 {
                if *byte_idx >= 12 {
                    return Err(ModelError::simple_load_error(
                        "decode_6bit_scales_and_mins_8: unexpected byte exhaustion",
                    ));
                }
                *bit_buf |= (raw[*byte_idx] as u64) << *bits_available;
                *bits_available += 8;
                *byte_idx += 1;
            }
            let val = (*bit_buf & 0x3F) as u8;
            *bit_buf >>= 6;
            *bits_available -= 6;
            Ok(val)
        };

    for slot in &mut scales {
        *slot = next_6bits(&mut bit_buf, &mut bits_available, &mut byte_idx)?;
    }
    for slot in &mut mins {
        *slot = next_6bits(&mut bit_buf, &mut bits_available, &mut byte_idx)?;
    }

    Ok((scales, mins))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: build a minimal Q2_K block ───────────────────────────────────

    /// Build a single Q2_K block (84 bytes) with controlled values.
    ///
    /// `d_val` and `dmin_val` are written as f16. `scale_nibble` and `min_nibble`
    /// (each 0..15) are broadcast to all 8 sub-block entries. `q` (0..3) is the
    /// 2-bit quant used for every element.
    fn build_q2k_block(
        d_val: f32,
        dmin_val: f32,
        scale_nibble: u8,
        min_nibble: u8,
        q: u8,
    ) -> Vec<u8> {
        let mut block = vec![0u8; 84];

        // d (f16)
        let d_bits = half::f16::from_f32(d_val).to_bits();
        block[0] = (d_bits & 0xFF) as u8;
        block[1] = (d_bits >> 8) as u8;

        // dmin (f16)
        let dmin_bits = half::f16::from_f32(dmin_val).to_bits();
        block[2] = (dmin_bits & 0xFF) as u8;
        block[3] = (dmin_bits >> 8) as u8;

        // scales[4..12]: 8 bytes, each byte = (min_nibble << 4) | scale_nibble
        let scale_byte = ((min_nibble & 0x0F) << 4) | (scale_nibble & 0x0F);
        for i in 0..8 {
            block[4 + i] = scale_byte;
        }

        // qs[16..80]: 64 bytes, each byte holds 4 copies of the 2-bit quant
        let q_byte = (q & 0x03) | ((q & 0x03) << 2) | ((q & 0x03) << 4) | ((q & 0x03) << 6);
        for i in 0..64 {
            block[16 + i] = q_byte;
        }

        block
    }

    #[test]
    fn test_dequant_q2_k_round_trip() {
        // Scale = 2.0, min = 1.0, scale_nibble = 2, min_nibble = 1, q = 3
        // Expected per element: d * scale_nibble * q - dmin * min_nibble
        //   = 2.0 * 2 * 3 - 1.0 * 1 = 12 - 1 = 11
        let d = 2.0f32;
        let dmin = 1.0f32;
        let scale_nibble = 2u8;
        let min_nibble = 1u8;
        let q = 3u8;

        let block = build_q2k_block(d, dmin, scale_nibble, min_nibble, q);
        let result = dequant_q2_k(&block, 256).expect("dequant_q2_k failed");

        assert_eq!(result.len(), 256, "Output must have 256 elements");

        let expected = d * scale_nibble as f32 * q as f32 - dmin * min_nibble as f32;
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected).abs() < 0.05,
                "Element {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn test_dequant_q2_k_zero_quants() {
        // With q=0 and d=1.0, dmin=0.5, min_nibble=0 → expected = 0
        let block = build_q2k_block(1.0, 0.5, 4, 0, 0);
        let result = dequant_q2_k(&block, 256).expect("dequant_q2_k failed");
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert!(v.abs() < 0.05, "Expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q2_k_error_alignment() {
        let block = build_q2k_block(1.0, 0.0, 1, 0, 1);
        // n=128 is not a multiple of 256 → error
        assert!(dequant_q2_k(&block, 128).is_err());
    }

    // ── Q3_K ─────────────────────────────────────────────────────────────────

    /// Build a single Q3_K block (110 bytes).
    ///
    /// Layout (matching llama.cpp `block_q3_K`):
    ///   [0..32]    hmask  — 1 high-bit per quant
    ///   [32..96]   qs     — low 2 bits of each quant, 4-per-byte
    ///   [96..108]  scales — 16 × 6-bit values packed into 12 bytes
    ///   [108..110] d      — f16
    ///
    /// All high bits set to `high_bit` (0 or 1), all low-2-bit quants set to `low2`,
    /// all scales set to `scale_raw` (raw 6-bit value, 0..63).
    fn build_q3k_block(d_val: f32, high_bit: u8, low2: u8, scale_raw: u8) -> Vec<u8> {
        let mut block = vec![0u8; 110];

        // hmask [0..32]: each byte represents 8 quants' high bits
        let hm_byte = if high_bit != 0 { 0xFFu8 } else { 0x00u8 };
        block[..32].fill(hm_byte);

        // qs [32..96]: 64 bytes, 4 quants per byte (2 bits each), all = low2
        let qs_byte =
            (low2 & 0x03) | ((low2 & 0x03) << 2) | ((low2 & 0x03) << 4) | ((low2 & 0x03) << 6);
        for i in 0..64 {
            block[32 + i] = qs_byte;
        }

        // scales [96..108]: 12 bytes, 16 × 6-bit values all = scale_raw
        // Pack 16 × 6-bit values LSB-first into 12 bytes.
        let mut bit_buf: u32 = 0;
        let mut bits: u32 = 0;
        let mut out_idx = 96usize;
        for _ in 0..16 {
            bit_buf |= (scale_raw as u32 & 0x3F) << bits;
            bits += 6;
            while bits >= 8 {
                block[out_idx] = (bit_buf & 0xFF) as u8;
                bit_buf >>= 8;
                bits -= 8;
                out_idx += 1;
            }
        }
        if bits > 0 && out_idx < 108 {
            block[out_idx] = (bit_buf & 0xFF) as u8;
        }

        // d [108..110]
        let d_bits = half::f16::from_f32(d_val).to_bits();
        block[108] = (d_bits & 0xFF) as u8;
        block[109] = (d_bits >> 8) as u8;

        block
    }

    #[test]
    fn test_dequant_q3_k_block_layout() {
        // d=1.0, high_bit=0, low2=2, scale_raw=36 (= 4 after subtracting 32)
        // q3 = (low2=2) | (high_bit=0 << 2) = 2, q3 - 4 = -2
        // val = 1.0 * (36 - 32) * (2 - 4) = 4 * (-2) = -8
        let d = 1.0f32;
        let scale_raw = 36u8; // 36 - 32 = 4
        let low2 = 2u8;
        let high_bit = 0u8;
        let block = build_q3k_block(d, high_bit, low2, scale_raw);
        let result = dequant_q3_k(&block, 256).expect("dequant_q3_k failed");
        assert_eq!(result.len(), 256);
        let expected = d * (scale_raw as f32 - 32.0) * (low2 as f32 - 4.0);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected).abs() < 0.1,
                "Element {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn test_dequant_q3_k_high_bit_set() {
        // high_bit=1, low2=3 → q3 = 3 | (1 << 2) = 7, q3 - 4 = 3
        // scale_raw = 35 → scale = 35 - 32 = 3
        // val = d * 3 * 3 = 9
        let d = 1.0f32;
        let scale_raw = 35u8;
        let low2 = 3u8;
        let high_bit = 1u8;
        let block = build_q3k_block(d, high_bit, low2, scale_raw);
        let result = dequant_q3_k(&block, 256).expect("dequant_q3_k failed");
        let expected = d * (scale_raw as f32 - 32.0) * (7.0 - 4.0);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected).abs() < 0.1,
                "Element {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    // ── Q4_K ─────────────────────────────────────────────────────────────────

    /// Build a single Q4_K block (144 bytes).
    fn build_q4k_block(
        d_val: f32,
        dmin_val: f32,
        scale_raw: u8,
        min_raw: u8,
        nibble: u8,
    ) -> Vec<u8> {
        let mut block = vec![0u8; 144];

        // d [0..2]
        let d_bits = half::f16::from_f32(d_val).to_bits();
        block[0] = (d_bits & 0xFF) as u8;
        block[1] = (d_bits >> 8) as u8;

        // dmin [2..4]
        let dm_bits = half::f16::from_f32(dmin_val).to_bits();
        block[2] = (dm_bits & 0xFF) as u8;
        block[3] = (dm_bits >> 8) as u8;

        // scales [4..16]: 12 bytes for 8 × 6-bit scales + 8 × 6-bit mins
        encode_6bit_scales_and_mins_8(&mut block[4..16], scale_raw, min_raw);

        // qs [16..144]: 128 bytes, nibble repeated
        let nibble_byte = (nibble & 0x0F) | ((nibble & 0x0F) << 4);
        for i in 0..128 {
            block[16 + i] = nibble_byte;
        }

        block
    }

    /// Pack 8 × scale_raw and 8 × min_raw (each 6-bit) into 12 bytes.
    fn encode_6bit_scales_and_mins_8(out: &mut [u8], scale_raw: u8, min_raw: u8) {
        let mut bit_buf: u64 = 0u64;
        let mut bits: u32 = 0;
        let mut out_idx = 0usize;

        let mut push6 = |val: u8| {
            bit_buf |= (val as u64 & 0x3F) << bits;
            bits += 6;
            while bits >= 8 {
                if out_idx < out.len() {
                    out[out_idx] = (bit_buf & 0xFF) as u8;
                    out_idx += 1;
                }
                bit_buf >>= 8;
                bits -= 8;
            }
        };

        for _ in 0..8 {
            push6(scale_raw);
        }
        for _ in 0..8 {
            push6(min_raw);
        }

        if bits > 0 && out_idx < out.len() {
            out[out_idx] = (bit_buf & 0xFF) as u8;
        }
    }

    #[test]
    fn test_dequant_q4_k_nibble_packing() {
        // d=2.0, dmin=1.0, scale_raw=10, min_raw=5, nibble=7
        // val = d * scale * nibble - dmin * min
        //     = 2.0 * 10 * 7 - 1.0 * 5 = 140 - 5 = 135
        let d = 2.0f32;
        let dmin = 1.0f32;
        let scale_raw = 10u8;
        let min_raw = 5u8;
        let nibble = 7u8;
        let block = build_q4k_block(d, dmin, scale_raw, min_raw, nibble);
        let result = dequant_q4_k(&block, 256).expect("dequant_q4_k failed");
        assert_eq!(result.len(), 256);
        let expected = d * scale_raw as f32 * nibble as f32 - dmin * min_raw as f32;
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected).abs() < 0.5,
                "Element {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn test_dequant_q4_k_zero_min() {
        // With min_raw=0, val = d * scale * nibble
        let d = 1.0f32;
        let dmin = 1.0f32;
        let scale_raw = 8u8;
        let min_raw = 0u8;
        let nibble = 3u8;
        let block = build_q4k_block(d, dmin, scale_raw, min_raw, nibble);
        let result = dequant_q4_k(&block, 256).expect("dequant_q4_k failed");
        let expected = d * scale_raw as f32 * nibble as f32;
        for &v in &result {
            assert!(
                (v - expected).abs() < 0.3,
                "Expected {}, got {}",
                expected,
                v
            );
        }
    }

    // ── Q5_K ─────────────────────────────────────────────────────────────────

    /// Build a single Q5_K block (176 bytes).
    fn build_q5k_block(
        d_val: f32,
        dmin_val: f32,
        scale_raw: u8,
        min_raw: u8,
        low4: u8,
        high_bit: u8,
    ) -> Vec<u8> {
        let mut block = vec![0u8; 176];

        // d [0..2], dmin [2..4]
        let d_bits = half::f16::from_f32(d_val).to_bits();
        block[0] = (d_bits & 0xFF) as u8;
        block[1] = (d_bits >> 8) as u8;
        let dm_bits = half::f16::from_f32(dmin_val).to_bits();
        block[2] = (dm_bits & 0xFF) as u8;
        block[3] = (dm_bits >> 8) as u8;

        // scales [4..16]
        encode_6bit_scales_and_mins_8(&mut block[4..16], scale_raw, min_raw);

        // qh [16..48]: 32 bytes — each bit is the high bit of one quant
        let qh_byte = if high_bit != 0 { 0xFFu8 } else { 0x00u8 };
        for i in 0..32 {
            block[16 + i] = qh_byte;
        }

        // qs [48..176]: 128 bytes — low nibbles repeated
        let nibble_byte = (low4 & 0x0F) | ((low4 & 0x0F) << 4);
        for i in 0..128 {
            block[48 + i] = nibble_byte;
        }

        block
    }

    #[test]
    fn test_dequant_q5_k_high_bit_merge() {
        // d=1.0, dmin=0.0, scale_raw=4, min_raw=0
        // low4=5, high_bit=1 → q5 = 5 | (1 << 4) = 21
        // val = 1.0 * 4 * 21 - 0 = 84
        let d = 1.0f32;
        let dmin = 0.0f32;
        let scale_raw = 4u8;
        let min_raw = 0u8;
        let low4 = 5u8;
        let high_bit = 1u8;
        let block = build_q5k_block(d, dmin, scale_raw, min_raw, low4, high_bit);
        let result = dequant_q5_k(&block, 256).expect("dequant_q5_k failed");
        assert_eq!(result.len(), 256);
        let q5 = (low4 as f32) + (high_bit as f32) * 16.0;
        let expected = d * scale_raw as f32 * q5 - dmin * min_raw as f32;
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected).abs() < 0.5,
                "Element {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn test_dequant_q5_k_no_high_bit() {
        // high_bit=0 → q5 = low4 only
        let d = 1.0f32;
        let dmin = 1.0f32;
        let scale_raw = 3u8;
        let min_raw = 2u8;
        let low4 = 9u8;
        let high_bit = 0u8;
        let block = build_q5k_block(d, dmin, scale_raw, min_raw, low4, high_bit);
        let result = dequant_q5_k(&block, 256).expect("dequant_q5_k failed");
        let expected = d * scale_raw as f32 * low4 as f32 - dmin * min_raw as f32;
        for &v in &result {
            assert!(
                (v - expected).abs() < 0.5,
                "Expected {}, got {}",
                expected,
                v
            );
        }
    }

    // ── Q8_K ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_dequant_q8_k_identity() {
        // d = 1.0 as f32, qs[i] = (i % 128) as i8 mapped to [-128..127],
        // verify output[i] ≈ qs[i] * 1.0
        let mut block = vec![0u8; 292];

        // d = 1.0f32
        let d_bytes = 1.0f32.to_le_bytes();
        block[0..4].copy_from_slice(&d_bytes);

        // qs[i] = i as i8 for i in 0..256 (wrapping)
        for i in 0..256usize {
            block[4 + i] = (i as u8).wrapping_add(0); // 0..255 cast to i8
        }

        let result = dequant_q8_k(&block, 256).expect("dequant_q8_k failed");
        assert_eq!(result.len(), 256);

        for i in 0..256usize {
            let expected = (block[4 + i] as i8) as f32;
            assert!(
                (result[i] - expected).abs() < 1e-5,
                "Element {}: expected {}, got {}",
                i,
                expected,
                result[i]
            );
        }
    }

    #[test]
    fn test_dequant_q8_k_scale_multiply() {
        // d = 0.5, all qs = 100 → all output should be 50.0
        let mut block = vec![0u8; 292];
        let d_bytes = 0.5f32.to_le_bytes();
        block[0..4].copy_from_slice(&d_bytes);
        for i in 0..256 {
            block[4 + i] = 100u8;
        }

        let result = dequant_q8_k(&block, 256).expect("dequant_q8_k failed");
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 50.0).abs() < 1e-4,
                "Element {}: expected 50.0, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_dequant_q8_k_negative_quants() {
        // d = 2.0, qs = -10 (0xF6) → all output = -20.0
        let mut block = vec![0u8; 292];
        let d_bytes = 2.0f32.to_le_bytes();
        block[0..4].copy_from_slice(&d_bytes);
        let q_byte = (-10i8) as u8;
        for i in 0..256 {
            block[4 + i] = q_byte;
        }

        let result = dequant_q8_k(&block, 256).expect("dequant_q8_k failed");
        for &v in &result {
            assert!((v - (-20.0)).abs() < 1e-4, "Expected -20.0, got {}", v);
        }
    }

    // ── Error / boundary checks ───────────────────────────────────────────────

    #[test]
    fn test_dequant_error_on_zero_elements() {
        let data = vec![0u8; 84];
        assert!(dequant_q2_k(&data, 0).is_err(), "n=0 should error");
        assert!(dequant_q3_k(&data, 0).is_err(), "n=0 should error");
        assert!(dequant_q4_k(&data, 0).is_err(), "n=0 should error");
        assert!(dequant_q5_k(&data, 0).is_err(), "n=0 should error");
        assert!(dequant_q8_k(&data, 0).is_err(), "n=0 should error");
    }

    #[test]
    fn test_dequant_error_on_short_buffer() {
        // Each type needs more than 10 bytes for even 1 block, so 10-byte buffer → error
        let tiny = vec![0u8; 10];
        assert!(dequant_q2_k(&tiny, 256).is_err());
        assert!(dequant_q3_k(&tiny, 256).is_err());
        assert!(dequant_q4_k(&tiny, 256).is_err());
        assert!(dequant_q5_k(&tiny, 256).is_err());
        assert!(dequant_q8_k(&tiny, 256).is_err());
    }

    // ── Multi-block test ──────────────────────────────────────────────────────

    #[test]
    fn test_dequant_q8_k_multi_block() {
        // Two Q8_K blocks, each with d=3.0 and qs=1
        let mut data = vec![0u8; 292 * 2];
        let d_bytes = 3.0f32.to_le_bytes();
        for blk in 0..2 {
            let base = blk * 292;
            data[base..base + 4].copy_from_slice(&d_bytes);
            for i in 0..256 {
                data[base + 4 + i] = 1u8;
            } // i8 = 1 → output = 3.0
        }
        let result = dequant_q8_k(&data, 512).expect("dequant_q8_k failed");
        assert_eq!(result.len(), 512);
        for &v in &result {
            assert!((v - 3.0).abs() < 1e-4, "Expected 3.0, got {}", v);
        }
    }
}
