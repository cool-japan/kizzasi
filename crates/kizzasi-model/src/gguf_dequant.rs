//! K-quant dequantization for GGUF format.
//!
//! Implements the five K-quantization schemes used in llama.cpp and GGUF files:
//! Q2_K, Q3_K, Q4_K, Q5_K, and Q8_K. Each function converts a raw byte buffer
//! representing one or more quantized blocks into a `Vec<f32>` of decoded weights.
//!
//! # Block layouts
//!
//! All K-quant types use 256-element super-blocks (`QK_K = 256`). The struct
//! layouts below are the `#[repr(C)]` `block_q*_K` definitions from
//! `ggml-common.h`; field *order* matters because GGUF stores the raw structs:
//!
//! ```text
//! block_q2_K (84 B)  : scales[16]  qs[64]                  d(f16) dmin(f16)
//! block_q3_K (110 B) : hmask[32]   qs[64]   scales[12]     d(f16)
//! block_q4_K (144 B) : d(f16) dmin(f16) scales[12] qs[128]
//! block_q5_K (176 B) : d(f16) dmin(f16) scales[12] qh[32] qs[128]
//! block_q8_K (292 B) : d(f32) qs[256] bsums[16 × i16]
//! ```
//!
//! # Element order
//!
//! The decoders reproduce the element order of `dequantize_row_q*_K` exactly.
//! For Q2_K and Q3_K the traversal is *shift-major*: within each 128-element
//! group the 2-bit field selector (`shift` = 0, 2, 4, 6) is the outer loop and
//! the 32 payload bytes are scanned in two 16-element halves for each shift.
//! For Q4_K and Q5_K, sub-blocks are paired over the *same* 32 payload bytes —
//! the low nibbles form sub-block `2k` and the high nibbles sub-block `2k + 1`.
//!
//! Reading these buffers sequentially "4 quants per byte, LSB first" is the
//! intuitive interpretation and it is wrong: it yields a permutation of the
//! correct weights.

use crate::error::{ModelError, ModelResult};

#[cfg(test)]
mod tests;

/// Elements per K-quant super-block.
const QK_K: usize = 256;
/// Bytes used by the packed 6-bit scale/min table of Q3_K, Q4_K and Q5_K.
const K_SCALE_SIZE: usize = 12;

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

/// Validate `n` / buffer size for a K-quant tensor and return the block count.
fn super_block_count(label: &str, data: &[u8], n: usize, block_bytes: usize) -> ModelResult<usize> {
    if n == 0 || !n.is_multiple_of(QK_K) {
        return Err(ModelError::simple_load_error(format!(
            "{}: n_elements {} must be a non-zero multiple of {}",
            label, n, QK_K
        )));
    }
    let n_blocks = n / QK_K;
    let required = n_blocks * block_bytes;
    if data.len() < required {
        return Err(ModelError::simple_load_error(format!(
            "{}: buffer too small: need {} bytes, got {}",
            label,
            required,
            data.len()
        )));
    }
    Ok(n_blocks)
}

// ─────────────────────────────────────────────────────────────────────────────
// Q2_K — 84 bytes / 256 elements
// ─────────────────────────────────────────────────────────────────────────────
//
// Super-block layout (`block_q2_K`, 84 bytes):
//   [0..16]  scales — 16 bytes; low nibble = 4-bit sub-scale, high nibble = 4-bit sub-min
//   [16..80] qs     — 64 bytes; 2-bit quants
//   [80..82] d      — f16 super-block scale
//   [82..84] dmin   — f16 super-block minimum scale
//
// Each super-block holds 16 sub-blocks of 16 elements. `dequantize_row_q2_K`
// walks two 128-element groups; inside a group the shift (0, 2, 4, 6) is the
// outer loop and each shift consumes two scale entries (bytes 0..16 and 16..32
// of the group's 32 payload bytes).
//
//   value = d * (scales[s] & 0xF) * q - dmin * (scales[s] >> 4)

/// Decode a Q2_K quantized tensor.
///
/// `data` must contain at least `(n / 256) * 84` bytes.
/// `n` must be a non-zero multiple of 256.
pub(crate) fn dequant_q2_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_BYTES: usize = 84;
    const GROUP_ELEMS: usize = 128;
    const GROUP_BYTES: usize = 32;
    let n_blocks = super_block_count("Q2_K", data, n, BLOCK_BYTES)?;

    let mut out = vec![0.0f32; n];

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let out_base = b * QK_K;

        let scales = &data[base..base + 16];
        let qs = &data[base + 16..base + 80];
        let d = read_f16_le(data, base + 80)?;
        let dmin = read_f16_le(data, base + 82)?;

        let mut is = 0usize;
        for group in 0..(QK_K / GROUP_ELEMS) {
            let qs_group = &qs[group * GROUP_BYTES..(group + 1) * GROUP_BYTES];
            let y_base = out_base + group * GROUP_ELEMS;
            let mut y_index = 0usize;

            for shift_idx in 0..4usize {
                let shift = 2 * shift_idx as u32;
                for half in 0..2usize {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0x0F) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    for q in &qs_group[half * 16..half * 16 + 16] {
                        out[y_base + y_index] = dl * ((q >> shift) & 3) as f32 - ml;
                        y_index += 1;
                    }
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
// Super-block layout (`block_q3_K`, 110 bytes):
//   [0..32]    hmask  — 32 bytes; the 3rd (high) bit of each quant, *inverted*
//   [32..96]   qs     — 64 bytes; low 2 bits of each quant
//   [96..108]  scales — 12 bytes; 16 × 6-bit signed-biased sub-block scales
//   [108..110] d      — f16 super-block scale
//
// Reconstruction (`dequantize_row_q3_K`):
//   q3  = ((qs[l] >> shift) & 3) - (hmask[l] & m == 0 ? 4 : 0)
//   val = d * (scale[s] - 32) * q3
//
// `hmask` is indexed by the *payload byte* index within the 32-byte group and
// the bit selector `m` advances once per 32-element chunk (8 chunks per block),
// so it is **not** `hmask[elem / 8] >> (elem % 8)`.

/// Decode a Q3_K quantized tensor.
///
/// `data` must contain at least `(n / 256) * 110` bytes.
/// `n` must be a non-zero multiple of 256.
pub(crate) fn dequant_q3_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_BYTES: usize = 110;
    const GROUP_ELEMS: usize = 128;
    const GROUP_BYTES: usize = 32;
    let n_blocks = super_block_count("Q3_K", data, n, BLOCK_BYTES)?;

    let mut out = vec![0.0f32; n];

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let out_base = b * QK_K;

        let hmask = &data[base..base + 32];
        let qs = &data[base + 32..base + 96];
        let scales_raw = &data[base + 96..base + 96 + K_SCALE_SIZE];
        let d = read_f16_le(data, base + 108)?;

        let sub_scales = unpack_q3_k_scales(scales_raw)?;

        let mut is = 0usize;
        for group in 0..(QK_K / GROUP_ELEMS) {
            let qs_group = &qs[group * GROUP_BYTES..(group + 1) * GROUP_BYTES];
            let y_base = out_base + group * GROUP_ELEMS;

            for shift_idx in 0..4usize {
                let shift = 2 * shift_idx as u32;
                // One high-bit selector per 32-element chunk; 8 chunks per block.
                let mask = 1u8 << (4 * group + shift_idx);
                for half in 0..2usize {
                    let dl = d * (sub_scales[is] as f32 - 32.0);
                    is += 1;
                    for i in 0..16usize {
                        let q_index = half * 16 + i;
                        let low2 = ((qs_group[q_index] >> shift) & 3) as i32;
                        let high = if hmask[q_index] & mask == 0 { 4 } else { 0 };
                        let y_index = shift_idx * 32 + half * 16 + i;
                        out[y_base + y_index] = dl * (low2 - high) as f32;
                    }
                }
            }
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Q4_K — 144 bytes / 256 elements
// ─────────────────────────────────────────────────────────────────────────────
//
// Super-block layout (`block_q4_K`, 144 bytes):
//   [0..2]    d      — f16 super-block scale
//   [2..4]    dmin   — f16 super-block minimum scale
//   [4..16]   scales — 12 bytes; 8 (scale, min) pairs via `get_scale_min_k4`
//   [16..144] qs     — 128 bytes; 4-bit nibbles
//
// Eight sub-blocks of 32 elements. Sub-blocks are paired: for each 32-byte
// payload chunk the low nibbles are sub-block `2k` and the high nibbles are
// sub-block `2k + 1`.
//
//   value = d * scale_s * q - dmin * min_s

/// Decode a Q4_K quantized tensor.
///
/// `data` must contain at least `(n / 256) * 144` bytes.
/// `n` must be a non-zero multiple of 256.
pub(crate) fn dequant_q4_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_BYTES: usize = 144;
    let n_blocks = super_block_count("Q4_K", data, n, BLOCK_BYTES)?;

    let mut out = vec![0.0f32; n];

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let out_base = b * QK_K;

        let d = read_f16_le(data, base)?;
        let dmin = read_f16_le(data, base + 2)?;
        let scales_raw = &data[base + 4..base + 4 + K_SCALE_SIZE];
        let qs = &data[base + 16..base + BLOCK_BYTES];

        let mut y_index = 0usize;
        for pair in 0..4usize {
            let is = pair * 2;
            let chunk = &qs[pair * 32..pair * 32 + 32];

            let (sc, m) = get_scale_min_k4(is, scales_raw);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales_raw);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;

            for q in chunk {
                out[out_base + y_index] = d1 * (q & 0x0F) as f32 - m1;
                y_index += 1;
            }
            for q in chunk {
                out[out_base + y_index] = d2 * (q >> 4) as f32 - m2;
                y_index += 1;
            }
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Q5_K — 176 bytes / 256 elements
// ─────────────────────────────────────────────────────────────────────────────
//
// Super-block layout (`block_q5_K`, 176 bytes):
//   [0..2]    d      — f16 super-block scale
//   [2..4]    dmin   — f16 super-block minimum scale
//   [4..16]   scales — 12 bytes; same packing as Q4_K
//   [16..48]  qh     — 32 bytes; the 5th bit of every quant
//   [48..176] qs     — 128 bytes; low 4 bits of every quant
//
// As in Q4_K, sub-blocks are paired over the same 32 payload bytes. The `qh`
// byte index stays `0..32` for every pair while the bit selectors advance by
// two bits per pair (`u1` = 1, 4, 16, 64 and `u2` = 2, 8, 32, 128).
//
//   value = d * scale_s * (low4 + 16·high_bit) - dmin * min_s

/// Decode a Q5_K quantized tensor.
///
/// `data` must contain at least `(n / 256) * 176` bytes.
/// `n` must be a non-zero multiple of 256.
pub(crate) fn dequant_q5_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_BYTES: usize = 176;
    let n_blocks = super_block_count("Q5_K", data, n, BLOCK_BYTES)?;

    let mut out = vec![0.0f32; n];

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let out_base = b * QK_K;

        let d = read_f16_le(data, base)?;
        let dmin = read_f16_le(data, base + 2)?;
        let scales_raw = &data[base + 4..base + 4 + K_SCALE_SIZE];
        let qh = &data[base + 16..base + 48];
        let ql = &data[base + 48..base + BLOCK_BYTES];

        let mut y_index = 0usize;
        for pair in 0..4usize {
            let is = pair * 2;
            let chunk = &ql[pair * 32..pair * 32 + 32];
            let u1 = 1u8 << (2 * pair);
            let u2 = 2u8 << (2 * pair);

            let (sc, m) = get_scale_min_k4(is, scales_raw);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales_raw);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;

            for (q, h) in chunk.iter().zip(qh.iter()) {
                let to_add = if h & u1 != 0 { 16.0 } else { 0.0 };
                out[out_base + y_index] = d1 * ((q & 0x0F) as f32 + to_add) - m1;
                y_index += 1;
            }
            for (q, h) in chunk.iter().zip(qh.iter()) {
                let to_add = if h & u2 != 0 { 16.0 } else { 0.0 };
                out[out_base + y_index] = d2 * ((q >> 4) as f32 + to_add) - m2;
                y_index += 1;
            }
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Q8_K — 292 bytes / 256 elements
// ─────────────────────────────────────────────────────────────────────────────
//
// Super-block layout (`block_q8_K`, 292 bytes):
//   [0..4]     d     — f32 super-block scale
//   [4..260]   qs    — 256 bytes; i8 quantized values
//   [260..292] bsums — 16 × i16 partial sums (used by `vec_dot`, not by dequant)
//
// Reconstruction:
//   val = d * qs[i]

/// Decode a Q8_K quantized tensor.
///
/// `data` must contain at least `(n / 256) * 292` bytes.
/// `n` must be a non-zero multiple of 256.
pub(crate) fn dequant_q8_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_BYTES: usize = 292;
    let n_blocks = super_block_count("Q8_K", data, n, BLOCK_BYTES)?;

    let mut out = Vec::with_capacity(n);

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let d = read_f32_le(data, base)?;
        for i in 0..QK_K {
            let q = data[base + 4 + i] as i8;
            out.push(d * q as f32);
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Scale-decoding helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the 6-bit `(scale, min)` pair for sub-block `j` of a Q4_K / Q5_K
/// super-block — the `get_scale_min_k4` routine from `ggml`.
///
/// The 12 scale bytes hold eight 6-bit scales and eight 6-bit mins. The first
/// four of each are stored plainly in the low six bits of bytes 0..8; the upper
/// four pairs take their low nibble from bytes 8..12 and their top two bits
/// from the high bits of bytes 0..8.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Unpack the 16 six-bit sub-block scales of a Q3_K super-block.
///
/// `ggml` performs this with three little-endian `u32` loads and the
/// `kmask1 = 0x03030303` / `kmask2 = 0x0f0f0f0f` masks. Written per byte
/// (which is endian-independent) the same transform is, for `k` in `0..4`:
///
/// ```text
/// s[k]      = (b[k]     & 0x0F) | (((b[8 + k] >> 0) & 3) << 4)
/// s[4 + k]  = (b[4 + k] & 0x0F) | (((b[8 + k] >> 2) & 3) << 4)
/// s[8 + k]  = (b[k]     >> 4)   | (((b[8 + k] >> 4) & 3) << 4)
/// s[12 + k] = (b[4 + k] >> 4)   | (((b[8 + k] >> 6) & 3) << 4)
/// ```
///
/// The resulting values are biased by 32 (`scale = raw - 32`).
fn unpack_q3_k_scales(raw: &[u8]) -> ModelResult<[u8; 16]> {
    if raw.len() < K_SCALE_SIZE {
        return Err(ModelError::simple_load_error(format!(
            "unpack_q3_k_scales: need {} bytes, got {}",
            K_SCALE_SIZE,
            raw.len()
        )));
    }

    let mut scales = [0u8; 16];
    for k in 0..4usize {
        let top = raw[8 + k];
        scales[k] = (raw[k] & 0x0F) | ((top & 3) << 4);
        scales[4 + k] = (raw[4 + k] & 0x0F) | (((top >> 2) & 3) << 4);
        scales[8 + k] = (raw[k] >> 4) | (((top >> 4) & 3) << 4);
        scales[12 + k] = (raw[4 + k] >> 4) | (((top >> 6) & 3) << 4);
    }
    Ok(scales)
}

/// Inverse of [`unpack_q3_k_scales`]: pack 16 six-bit values into 12 bytes.
///
/// Only used by tests and by callers that need to synthesise Q3_K blocks, but
/// kept next to the unpacking routine so the two stay in sync.
#[cfg(test)]
fn pack_q3_k_scales(scales: &[u8; 16]) -> [u8; K_SCALE_SIZE] {
    let mut raw = [0u8; K_SCALE_SIZE];
    for k in 0..4usize {
        raw[k] = (scales[k] & 0x0F) | ((scales[8 + k] & 0x0F) << 4);
        raw[4 + k] = (scales[4 + k] & 0x0F) | ((scales[12 + k] & 0x0F) << 4);
        raw[8 + k] = ((scales[k] >> 4) & 3)
            | (((scales[4 + k] >> 4) & 3) << 2)
            | (((scales[8 + k] >> 4) & 3) << 4)
            | (((scales[12 + k] >> 4) & 3) << 6);
    }
    raw
}

/// Inverse of [`get_scale_min_k4`]: pack eight `(scale, min)` 6-bit pairs into
/// the 12-byte Q4_K / Q5_K scale table.
#[cfg(test)]
fn pack_scales_min_k4(scales: &[u8; 8], mins: &[u8; 8]) -> [u8; K_SCALE_SIZE] {
    let mut q = [0u8; K_SCALE_SIZE];
    for j in 0..4usize {
        q[j] = scales[j] & 63;
        q[j + 4] = mins[j] & 63;
    }
    for j in 4..8usize {
        q[j + 4] = (scales[j] & 0x0F) | ((mins[j] & 0x0F) << 4);
        q[j - 4] |= ((scales[j] >> 4) & 3) << 6;
        q[j] |= ((mins[j] >> 4) & 3) << 6;
    }
    q
}
