//! GGUF tensor dequantization routines.
//!
//! Provides f32 dequantization for each supported [`GgufQuantType`] variant.
//! The K-quant family (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q8_K`) delegates to
//! the shared [`crate::gguf_dequant`] implementations.
//!
//! # Element ordering
//!
//! Every decoder reproduces the exact element order of the corresponding
//! `dequantize_row_*` routine in `ggml`. For the legacy 32-element block types
//! this means the 16 *low* nibbles of a block map to elements `0..16` and the 16
//! *high* nibbles map to elements `16..32` — **not** the interleaved
//! `(lo0, hi0, lo1, hi1, …)` sequence a naive reading of the packing suggests.
//! The same applies to the 5th bit stored in `qh`: bit `j` belongs to element
//! `j` and bit `j + 16` to element `j + 16`.

use super::GgufQuantType;
use crate::error::{ModelError, ModelResult};
use crate::gguf_dequant as kquant;

/// Number of elements in a legacy (non-K) quantization block.
const LEGACY_BLOCK_ELEMS: usize = 32;
/// Number of packed bytes holding the quant payload of a legacy block.
const LEGACY_QUANT_BYTES: usize = LEGACY_BLOCK_ELEMS / 2;

/// Dequantize `data` bytes to `n_elements` f32 values according to `quant_type`.
pub fn dequantize(
    data: &[u8],
    quant_type: &GgufQuantType,
    n_elements: usize,
) -> ModelResult<Vec<f32>> {
    match quant_type {
        GgufQuantType::F32 => dequant_f32(data, n_elements),
        GgufQuantType::F16 => dequant_f16(data, n_elements),
        GgufQuantType::BF16 => dequant_bf16(data, n_elements),
        GgufQuantType::Q4_0 => dequant_q4_0(data, n_elements),
        GgufQuantType::Q4_1 => dequant_q4_1(data, n_elements),
        GgufQuantType::Q5_0 => dequant_q5_0(data, n_elements),
        GgufQuantType::Q5_1 => dequant_q5_1(data, n_elements),
        GgufQuantType::Q8_0 => dequant_q8_0(data, n_elements),
        GgufQuantType::Q6K => dequant_q6_k(data, n_elements),
        GgufQuantType::Q2K => kquant::dequant_q2_k(data, n_elements),
        GgufQuantType::Q3K => kquant::dequant_q3_k(data, n_elements),
        GgufQuantType::Q4K => kquant::dequant_q4_k(data, n_elements),
        GgufQuantType::Q5K => kquant::dequant_q5_k(data, n_elements),
        GgufQuantType::Q8K => kquant::dequant_q8_k(data, n_elements),
        qt => Err(ModelError::simple_load_error(format!(
            "Unsupported quant type for dequantization: {:?}",
            qt
        ))),
    }
}

/// Validate the element count / buffer length for a block-quantized tensor and
/// return the number of blocks.
fn legacy_block_count(
    label: &str,
    data: &[u8],
    n: usize,
    block_bytes: usize,
    block_elems: usize,
) -> ModelResult<usize> {
    if !n.is_multiple_of(block_elems) {
        return Err(ModelError::simple_load_error(format!(
            "{}: n_elements {} not divisible by {}",
            label, n, block_elems
        )));
    }
    let n_blocks = n / block_elems;
    let required = n_blocks * block_bytes;
    if data.len() < required {
        return Err(ModelError::simple_load_error(format!(
            "{} data buffer too small: need {} bytes, got {}",
            label,
            required,
            data.len()
        )));
    }
    Ok(n_blocks)
}

/// Read a little-endian f16 at `offset` and widen to f32.
///
/// The caller must have validated that `offset + 2 <= data.len()`.
#[inline]
fn f16_at(data: &[u8], offset: usize) -> f32 {
    let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
    half::f16::from_bits(bits).to_f32()
}

fn dequant_f32(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    if data.len() < n * 4 {
        return Err(ModelError::simple_load_error(format!(
            "F32 tensor needs {} bytes, got {}",
            n * 4,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * 4;
        let v = f32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]]);
        out.push(v);
    }
    Ok(out)
}

pub(super) fn dequant_f16(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    if data.len() < n * 2 {
        return Err(ModelError::simple_load_error(format!(
            "F16 tensor needs {} bytes, got {}",
            n * 2,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f16_at(data, i * 2));
    }
    Ok(out)
}

pub(super) fn dequant_bf16(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    if data.len() < n * 2 {
        return Err(ModelError::simple_load_error(format!(
            "BF16 tensor needs {} bytes, got {}",
            n * 2,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * 2;
        let bits = u16::from_le_bytes([data[base], data[base + 1]]);
        // BF16 → f32: sign+exp+7 mantissa bits occupy the upper 16 bits of f32
        out.push(f32::from_bits((bits as u32) << 16));
    }
    Ok(out)
}

/// Q4_0 block: 2 bytes delta (f16) + 16 bytes quantized nibbles → 32 f32.
///
/// Mirrors `dequantize_row_q4_0`: for `j` in `0..16`, the low nibble of byte `j`
/// becomes element `j` and the high nibble becomes element `j + 16`. Each nibble
/// `q` represents `(q - 8) * delta`.
pub(super) fn dequant_q4_0(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_BYTES: usize = 18; // 2 delta + 16 nibbles
    let n_blocks = legacy_block_count("Q4_0", data, n, BLOCK_BYTES, LEGACY_BLOCK_ELEMS)?;

    let mut out = vec![0.0f32; n];
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let out_base = b * LEGACY_BLOCK_ELEMS;
        let delta = f16_at(data, base);
        for j in 0..LEGACY_QUANT_BYTES {
            let byte = data[base + 2 + j];
            let x0 = (byte & 0x0F) as i32 - 8;
            let x1 = (byte >> 4) as i32 - 8;
            out[out_base + j] = x0 as f32 * delta;
            out[out_base + j + LEGACY_QUANT_BYTES] = x1 as f32 * delta;
        }
    }
    Ok(out)
}

/// Q4_1 block: 2 bytes delta (f16) + 2 bytes min (f16) + 16 bytes nibbles → 32 f32.
///
/// Mirrors `dequantize_row_q4_1`; nibble `q` represents `q * delta + min` and the
/// low/high nibble halves land in elements `0..16` / `16..32` respectively.
pub(super) fn dequant_q4_1(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_BYTES: usize = 20;
    let n_blocks = legacy_block_count("Q4_1", data, n, BLOCK_BYTES, LEGACY_BLOCK_ELEMS)?;

    let mut out = vec![0.0f32; n];
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let out_base = b * LEGACY_BLOCK_ELEMS;
        let delta = f16_at(data, base);
        let min = f16_at(data, base + 2);
        for j in 0..LEGACY_QUANT_BYTES {
            let byte = data[base + 4 + j];
            let x0 = (byte & 0x0F) as f32;
            let x1 = (byte >> 4) as f32;
            out[out_base + j] = x0 * delta + min;
            out[out_base + j + LEGACY_QUANT_BYTES] = x1 * delta + min;
        }
    }
    Ok(out)
}

/// Q5_0 block: 2 bytes delta (f16) + 4 bytes high bits (u32) + 16 bytes low nibbles → 32 f32.
///
/// Mirrors `dequantize_row_q5_0`. Bit `j` of `qh` is the 5th bit of element `j`
/// and bit `j + 16` is the 5th bit of element `j + 16`. Each 5-bit value `q`
/// (range 0–31) is dequantized as `(q - 16) * delta`.
pub(super) fn dequant_q5_0(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_BYTES: usize = 22;
    let n_blocks = legacy_block_count("Q5_0", data, n, BLOCK_BYTES, LEGACY_BLOCK_ELEMS)?;

    let mut out = vec![0.0f32; n];
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let out_base = b * LEGACY_BLOCK_ELEMS;
        let delta = f16_at(data, base);
        let qh = u32::from_le_bytes([
            data[base + 2],
            data[base + 3],
            data[base + 4],
            data[base + 5],
        ]);
        for j in 0..LEGACY_QUANT_BYTES {
            let byte = data[base + 6 + j];
            // 5th bit of element j    → qh bit j        (moved into bit 4)
            // 5th bit of element j+16 → qh bit (j + 16) (moved into bit 4)
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((byte & 0x0F) as u32 | xh_0) as i32 - 16;
            let x1 = ((byte >> 4) as u32 | xh_1) as i32 - 16;
            out[out_base + j] = x0 as f32 * delta;
            out[out_base + j + LEGACY_QUANT_BYTES] = x1 as f32 * delta;
        }
    }
    Ok(out)
}

/// Q5_1 block: 2 bytes delta (f16) + 2 bytes min (f16) + 4 bytes high bits + 16 bytes → 32 f32.
///
/// Mirrors `dequantize_row_q5_1`; the 5-bit value `q` is dequantized as
/// `q * delta + min` with the same `qh` bit assignment as Q5_0.
pub(super) fn dequant_q5_1(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_BYTES: usize = 24;
    let n_blocks = legacy_block_count("Q5_1", data, n, BLOCK_BYTES, LEGACY_BLOCK_ELEMS)?;

    let mut out = vec![0.0f32; n];
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let out_base = b * LEGACY_BLOCK_ELEMS;
        let delta = f16_at(data, base);
        let min = f16_at(data, base + 2);
        let qh = u32::from_le_bytes([
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 7],
        ]);
        for j in 0..LEGACY_QUANT_BYTES {
            let byte = data[base + 8 + j];
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((byte & 0x0F) as u32 | xh_0) as f32;
            let x1 = ((byte >> 4) as u32 | xh_1) as f32;
            out[out_base + j] = x0 * delta + min;
            out[out_base + j + LEGACY_QUANT_BYTES] = x1 * delta + min;
        }
    }
    Ok(out)
}

/// Q8_0 block: 2 bytes delta (f16) + 32 bytes i8 values → 32 f32.
///
/// Each i8 value `q` is scaled: `q * delta`. Element order is already
/// sequential in `ggml`, so no reordering is required.
pub(super) fn dequant_q8_0(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_BYTES: usize = 34;
    let n_blocks = legacy_block_count("Q8_0", data, n, BLOCK_BYTES, LEGACY_BLOCK_ELEMS)?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let delta = f16_at(data, base);
        for i in 0..LEGACY_BLOCK_ELEMS {
            let q = data[base + 2 + i] as i8;
            out.push(q as f32 * delta);
        }
    }
    Ok(out)
}

/// Q6_K block: 210 bytes → 256 f32 elements.
///
/// Block layout (`block_q6_K`):
/// - `[0..128]`   `ql`     — low 4 bits of each 6-bit value, packed as nibbles
/// - `[128..192]` `qh`     — high 2 bits, 4 values per byte
/// - `[192..208]` `scales` — 16 sub-block scales (i8, one per 16 elements)
/// - `[208..210]` `d`      — super-block scale (f16)
///
/// Decoding follows `dequantize_row_q6_K`: the 256 elements are processed as two
/// 128-element groups. Within a group, for `l` in `0..32`:
///
/// ```text
/// y[l     ] = d * sc[l/16 + 0] * ((ql[l]      & 0xF) | ((qh[l] >> 0) & 3) << 4 - 32)
/// y[l + 32] = d * sc[l/16 + 2] * ((ql[l + 32] & 0xF) | ((qh[l] >> 2) & 3) << 4 - 32)
/// y[l + 64] = d * sc[l/16 + 4] * ((ql[l]      >> 4)  | ((qh[l] >> 4) & 3) << 4 - 32)
/// y[l + 96] = d * sc[l/16 + 6] * ((ql[l + 32] >> 4)  | ((qh[l] >> 6) & 3) << 4 - 32)
/// ```
pub(super) fn dequant_q6_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_ELEMS: usize = 256;
    const BLOCK_BYTES: usize = 210;
    const GROUP_ELEMS: usize = 128;
    let n_blocks = legacy_block_count("Q6K", data, n, BLOCK_BYTES, BLOCK_ELEMS)?;

    let mut out = vec![0.0f32; n];
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let out_base = b * BLOCK_ELEMS;
        let ql = &data[base..base + 128];
        let qh = &data[base + 128..base + 192];
        let scales = &data[base + 192..base + 208];
        let d = f16_at(data, base + 208);

        for group in 0..(BLOCK_ELEMS / GROUP_ELEMS) {
            let ql_off = 64 * group;
            let qh_off = 32 * group;
            let sc_off = 8 * group;
            let y_off = out_base + GROUP_ELEMS * group;

            for l in 0..32usize {
                let is = l / 16;
                let ql_lo = ql[ql_off + l];
                let ql_hi = ql[ql_off + l + 32];
                let h = qh[qh_off + l];

                let q1 = ((ql_lo & 0x0F) as i32 | (((h) & 3) as i32) << 4) - 32;
                let q2 = ((ql_hi & 0x0F) as i32 | (((h >> 2) & 3) as i32) << 4) - 32;
                let q3 = ((ql_lo >> 4) as i32 | (((h >> 4) & 3) as i32) << 4) - 32;
                let q4 = ((ql_hi >> 4) as i32 | (((h >> 6) & 3) as i32) << 4) - 32;

                out[y_off + l] = d * (scales[sc_off + is] as i8) as f32 * q1 as f32;
                out[y_off + l + 32] = d * (scales[sc_off + is + 2] as i8) as f32 * q2 as f32;
                out[y_off + l + 64] = d * (scales[sc_off + is + 4] as i8) as f32 * q3 as f32;
                out[y_off + l + 96] = d * (scales[sc_off + is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_le(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }

    /// Indices whose value differs from `0.0`, together with the value.
    fn nonzero(out: &[f32]) -> Vec<(usize, f32)> {
        out.iter()
            .enumerate()
            .filter(|(_, v)| **v != 0.0)
            .map(|(i, v)| (i, *v))
            .collect()
    }

    fn sorted(values: &[f32]) -> Vec<f32> {
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    // ── Q4_0 ────────────────────────────────────────────────────────────────

    /// Build one Q4_0 block from 32 nibbles in *element* order.
    fn build_q4_0(delta: f32, nibbles: &[u8; 32]) -> Vec<u8> {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16_le(delta));
        for j in 0..16 {
            block[2 + j] = (nibbles[j] & 0x0F) | ((nibbles[j + 16] & 0x0F) << 4);
        }
        block
    }

    #[test]
    fn q4_0_low_nibble_maps_to_first_half() {
        // Only element 3 (a low nibble) is non-neutral; everything else quantizes to 8 → 0.
        let mut nibbles = [8u8; 32];
        nibbles[3] = 15;
        let block = build_q4_0(1.0, &nibbles);
        let out = dequant_q4_0(&block, 32).expect("q4_0");
        assert_eq!(nonzero(&out), vec![(3usize, 7.0f32)]);
    }

    #[test]
    fn q4_0_high_nibble_maps_to_second_half() {
        // Byte 0's *high* nibble is element 16, not element 1.
        let mut nibbles = [8u8; 32];
        nibbles[16] = 0;
        let block = build_q4_0(1.0, &nibbles);
        let out = dequant_q4_0(&block, 32).expect("q4_0");
        assert_eq!(nonzero(&out), vec![(16usize, -8.0f32)]);
    }

    #[test]
    fn q4_0_multiset_matches_reference_values() {
        let nibbles: [u8; 32] = std::array::from_fn(|i| (i % 16) as u8);
        let block = build_q4_0(0.5, &nibbles);
        let out = dequant_q4_0(&block, 32).expect("q4_0");
        let expected: Vec<f32> = nibbles.iter().map(|q| (*q as f32 - 8.0) * 0.5).collect();
        assert_eq!(sorted(&out), sorted(&expected));
        // Exact positional equality as well.
        for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "element {i}: {got} != {want}");
        }
    }

    // ── Q4_1 ────────────────────────────────────────────────────────────────

    fn build_q4_1(delta: f32, min: f32, nibbles: &[u8; 32]) -> Vec<u8> {
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&f16_le(delta));
        block[2..4].copy_from_slice(&f16_le(min));
        for j in 0..16 {
            block[4 + j] = (nibbles[j] & 0x0F) | ((nibbles[j + 16] & 0x0F) << 4);
        }
        block
    }

    #[test]
    fn q4_1_element_order_and_affine() {
        let nibbles: [u8; 32] = std::array::from_fn(|i| ((i * 7) % 16) as u8);
        let block = build_q4_1(0.25, -1.0, &nibbles);
        let out = dequant_q4_1(&block, 32).expect("q4_1");
        for (i, &q) in nibbles.iter().enumerate() {
            let want = q as f32 * 0.25 - 1.0;
            assert!(
                (out[i] - want).abs() < 1e-5,
                "element {i}: {} != {want}",
                out[i]
            );
        }
    }

    // ── Q5_0 / Q5_1 ─────────────────────────────────────────────────────────

    fn build_q5_0(delta: f32, quants: &[u8; 32]) -> Vec<u8> {
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&f16_le(delta));
        let mut qh = 0u32;
        for (j, &q) in quants.iter().enumerate() {
            if q & 0x10 != 0 {
                qh |= 1u32 << j;
            }
        }
        block[2..6].copy_from_slice(&qh.to_le_bytes());
        for j in 0..16 {
            block[6 + j] = (quants[j] & 0x0F) | ((quants[j + 16] & 0x0F) << 4);
        }
        block
    }

    #[test]
    fn q5_0_high_bit_follows_element_index() {
        // Element 20 is the only one with the 5th bit set; it lives in the high
        // nibble of byte 4 and uses qh bit 20.
        let mut quants = [16u8; 32]; // 16 → (16 - 16) * d == 0
        quants[20] = 31;
        let block = build_q5_0(1.0, &quants);
        let out = dequant_q5_0(&block, 32).expect("q5_0");
        assert_eq!(nonzero(&out), vec![(20usize, 15.0f32)]);
    }

    #[test]
    fn q5_0_full_range_round_trip() {
        let quants: [u8; 32] = std::array::from_fn(|i| i as u8);
        let block = build_q5_0(0.125, &quants);
        let out = dequant_q5_0(&block, 32).expect("q5_0");
        for (i, &q) in quants.iter().enumerate() {
            let want = (q as f32 - 16.0) * 0.125;
            assert!(
                (out[i] - want).abs() < 1e-6,
                "element {i}: {} != {want}",
                out[i]
            );
        }
    }

    fn build_q5_1(delta: f32, min: f32, quants: &[u8; 32]) -> Vec<u8> {
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&f16_le(delta));
        block[2..4].copy_from_slice(&f16_le(min));
        let mut qh = 0u32;
        for (j, &q) in quants.iter().enumerate() {
            if q & 0x10 != 0 {
                qh |= 1u32 << j;
            }
        }
        block[4..8].copy_from_slice(&qh.to_le_bytes());
        for j in 0..16 {
            block[8 + j] = (quants[j] & 0x0F) | ((quants[j + 16] & 0x0F) << 4);
        }
        block
    }

    #[test]
    fn q5_1_full_range_round_trip() {
        let quants: [u8; 32] = std::array::from_fn(|i| (31 - i) as u8);
        let block = build_q5_1(0.5, 2.0, &quants);
        let out = dequant_q5_1(&block, 32).expect("q5_1");
        for (i, &q) in quants.iter().enumerate() {
            let want = q as f32 * 0.5 + 2.0;
            assert!(
                (out[i] - want).abs() < 1e-5,
                "element {i}: {} != {want}",
                out[i]
            );
        }
    }

    // ── Q8_0 ────────────────────────────────────────────────────────────────

    #[test]
    fn q8_0_sequential_order() {
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&f16_le(0.5));
        for i in 0..32usize {
            block[2 + i] = (i as i32 - 16) as i8 as u8;
        }
        let out = dequant_q8_0(&block, 32).expect("q8_0");
        assert_eq!(out.len(), 32, "q8_0 must dequantize all 32 elements");
        for (i, &got) in out.iter().enumerate() {
            let want = (i as f32 - 16.0) * 0.5;
            assert!((got - want).abs() < 1e-6, "element {i}");
        }
    }

    // ── Q6_K ────────────────────────────────────────────────────────────────

    /// Reference encoder for one Q6_K block, written directly from the GGML
    /// element mapping so that it is independent of the decoder's loop shape.
    fn build_q6_k(d: f32, scales: &[i8; 16], quants: &[u8; 256]) -> Vec<u8> {
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for group in 0..2usize {
            let ql_off = 64 * group;
            let qh_off = 32 * group;
            let y_off = 128 * group;
            for l in 0..32usize {
                let q1 = quants[y_off + l] & 0x3F;
                let q2 = quants[y_off + l + 32] & 0x3F;
                let q3 = quants[y_off + l + 64] & 0x3F;
                let q4 = quants[y_off + l + 96] & 0x3F;
                ql[ql_off + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
                ql[ql_off + l + 32] = (q2 & 0x0F) | ((q4 & 0x0F) << 4);
                qh[qh_off + l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
            }
        }
        let mut block = vec![0u8; 210];
        block[0..128].copy_from_slice(&ql);
        block[128..192].copy_from_slice(&qh);
        for (i, s) in scales.iter().enumerate() {
            block[192 + i] = *s as u8;
        }
        block[208..210].copy_from_slice(&f16_le(d));
        block
    }

    #[test]
    fn q6_k_round_trips_reference_encoding() {
        let scales: [i8; 16] = std::array::from_fn(|i| (i as i8) - 8);
        let quants: [u8; 256] = std::array::from_fn(|i| ((i * 13) % 64) as u8);
        let block = build_q6_k(1.0, &scales, &quants);
        let out = dequant_q6_k(&block, 256).expect("q6_k");
        for i in 0..256usize {
            // Scale index: 16 elements per sub-block.
            let want = (scales[i / 16] as f32) * (quants[i] as f32 - 32.0);
            assert!(
                (out[i] - want).abs() < 1e-4,
                "element {i}: {} != {want}",
                out[i]
            );
        }
    }

    #[test]
    fn q6_k_isolates_single_element() {
        // All quants neutral (32 → 0 after the -32 bias) except element 100.
        let scales = [1i8; 16];
        let mut quants = [32u8; 256];
        quants[100] = 63;
        let block = build_q6_k(2.0, &scales, &quants);
        let out = dequant_q6_k(&block, 256).expect("q6_k");
        assert_eq!(nonzero(&out), vec![(100usize, 62.0f32)]);
    }

    #[test]
    fn q6_k_uses_all_sixteen_scales() {
        // A single quant per 16-element sub-block; each must pick up its own scale.
        let scales: [i8; 16] = std::array::from_fn(|i| (i as i8) + 1);
        let mut quants = [32u8; 256];
        for s in 0..16usize {
            quants[s * 16] = 33;
        }
        let block = build_q6_k(1.0, &scales, &quants);
        let out = dequant_q6_k(&block, 256).expect("q6_k");
        let got = nonzero(&out);
        assert_eq!(got.len(), 16);
        for (s, (idx, val)) in got.iter().enumerate() {
            assert_eq!(*idx, s * 16);
            assert!((val - (s as f32 + 1.0)).abs() < 1e-5, "sub-block {s}");
        }
    }

    // ── Error handling ──────────────────────────────────────────────────────

    #[test]
    fn legacy_decoders_reject_misaligned_lengths() {
        let data = vec![0u8; 1024];
        assert!(dequant_q4_0(&data, 31).is_err());
        assert!(dequant_q4_1(&data, 31).is_err());
        assert!(dequant_q5_0(&data, 31).is_err());
        assert!(dequant_q5_1(&data, 31).is_err());
        assert!(dequant_q8_0(&data, 31).is_err());
        assert!(dequant_q6_k(&data, 255).is_err());
    }

    #[test]
    fn legacy_decoders_reject_short_buffers() {
        let data = vec![0u8; 4];
        assert!(dequant_q4_0(&data, 32).is_err());
        assert!(dequant_q4_1(&data, 32).is_err());
        assert!(dequant_q5_0(&data, 32).is_err());
        assert!(dequant_q5_1(&data, 32).is_err());
        assert!(dequant_q8_0(&data, 32).is_err());
        assert!(dequant_q6_k(&data, 256).is_err());
    }
}
