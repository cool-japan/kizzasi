//! GGUF tensor dequantization routines.
//!
//! Provides f32 dequantization for each supported [`GgufQuantType`] variant.
//! The K-quant family (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q8_K`) delegates to
//! the shared [`crate::gguf_dequant`] implementations.

use super::GgufQuantType;
use crate::error::{ModelError, ModelResult};
use crate::gguf_dequant as kquant;

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
        let base = i * 2;
        let bits = u16::from_le_bytes([data[base], data[base + 1]]);
        out.push(half::f16::from_bits(bits).to_f32());
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

/// Q4_0 block: 2 bytes delta (f16) + 16 bytes quantized nibbles → 32 f32
///
/// Each nibble `q` represents `(q - 8) * delta`.
pub(super) fn dequant_q4_0(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18; // 2 delta + 16 nibbles
    if !n.is_multiple_of(BLOCK_ELEMS) {
        return Err(ModelError::simple_load_error(format!(
            "Q4_0: n_elements {} not divisible by {}",
            n, BLOCK_ELEMS
        )));
    }
    let n_blocks = n / BLOCK_ELEMS;
    if data.len() < n_blocks * BLOCK_BYTES {
        return Err(ModelError::simple_load_error("Q4_0 data buffer too small"));
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let delta_bits = u16::from_le_bytes([data[base], data[base + 1]]);
        let delta = half::f16::from_bits(delta_bits).to_f32();
        for byte_idx in 0..16usize {
            let byte = data[base + 2 + byte_idx];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out.push(lo as f32 * delta);
            out.push(hi as f32 * delta);
        }
    }
    Ok(out)
}

/// Q4_1 block: 2 bytes delta (f16) + 2 bytes min (f16) + 16 bytes nibbles → 32 f32
///
/// Each nibble `q` represents `q * delta + min`.
pub(super) fn dequant_q4_1(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 20;
    if !n.is_multiple_of(BLOCK_ELEMS) {
        return Err(ModelError::simple_load_error(format!(
            "Q4_1: n_elements {} not divisible by {}",
            n, BLOCK_ELEMS
        )));
    }
    let n_blocks = n / BLOCK_ELEMS;
    if data.len() < n_blocks * BLOCK_BYTES {
        return Err(ModelError::simple_load_error("Q4_1 data buffer too small"));
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let delta_bits = u16::from_le_bytes([data[base], data[base + 1]]);
        let delta = half::f16::from_bits(delta_bits).to_f32();
        let min_bits = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let min = half::f16::from_bits(min_bits).to_f32();
        for byte_idx in 0..16usize {
            let byte = data[base + 4 + byte_idx];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            out.push(lo * delta + min);
            out.push(hi * delta + min);
        }
    }
    Ok(out)
}

/// Q5_0 block: 2 bytes delta (f16) + 4 bytes high bits (u32) + 16 bytes low nibbles → 32 f32
///
/// Each 5-bit value `q` (range 0–31, then subtract 16) scaled by delta.
pub(super) fn dequant_q5_0(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 22;
    if !n.is_multiple_of(BLOCK_ELEMS) {
        return Err(ModelError::simple_load_error(format!(
            "Q5_0: n_elements {} not divisible by {}",
            n, BLOCK_ELEMS
        )));
    }
    let n_blocks = n / BLOCK_ELEMS;
    if data.len() < n_blocks * BLOCK_BYTES {
        return Err(ModelError::simple_load_error("Q5_0 data buffer too small"));
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let delta_bits = u16::from_le_bytes([data[base], data[base + 1]]);
        let delta = half::f16::from_bits(delta_bits).to_f32();
        // High bits: bit i of qh → 5th bit of element i
        let qh = u32::from_le_bytes([
            data[base + 2],
            data[base + 3],
            data[base + 4],
            data[base + 5],
        ]);
        for byte_idx in 0..16usize {
            let byte = data[base + 6 + byte_idx];
            let lo4 = (byte & 0x0F) as u32;
            let hi4 = ((byte >> 4) & 0x0F) as u32;
            let elem_lo = byte_idx * 2;
            let elem_hi = byte_idx * 2 + 1;
            let hi_lo = (qh >> elem_lo) & 1;
            let hi_hi = (qh >> elem_hi) & 1;
            let q_lo = (lo4 | (hi_lo << 4)) as i32 - 16;
            let q_hi = (hi4 | (hi_hi << 4)) as i32 - 16;
            out.push(q_lo as f32 * delta);
            out.push(q_hi as f32 * delta);
        }
    }
    Ok(out)
}

/// Q5_1 block: 2 bytes delta (f16) + 2 bytes min (f16) + 4 bytes high bits + 16 bytes → 32 f32
pub(super) fn dequant_q5_1(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 24;
    if !n.is_multiple_of(BLOCK_ELEMS) {
        return Err(ModelError::simple_load_error(format!(
            "Q5_1: n_elements {} not divisible by {}",
            n, BLOCK_ELEMS
        )));
    }
    let n_blocks = n / BLOCK_ELEMS;
    if data.len() < n_blocks * BLOCK_BYTES {
        return Err(ModelError::simple_load_error("Q5_1 data buffer too small"));
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let delta_bits = u16::from_le_bytes([data[base], data[base + 1]]);
        let delta = half::f16::from_bits(delta_bits).to_f32();
        let min_bits = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let min = half::f16::from_bits(min_bits).to_f32();
        let qh = u32::from_le_bytes([
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 7],
        ]);
        for byte_idx in 0..16usize {
            let byte = data[base + 8 + byte_idx];
            let lo4 = (byte & 0x0F) as u32;
            let hi4 = ((byte >> 4) & 0x0F) as u32;
            let elem_lo = byte_idx * 2;
            let elem_hi = byte_idx * 2 + 1;
            let hi_lo = (qh >> elem_lo) & 1;
            let hi_hi = (qh >> elem_hi) & 1;
            let q_lo = (lo4 | (hi_lo << 4)) as f32;
            let q_hi = (hi4 | (hi_hi << 4)) as f32;
            out.push(q_lo * delta + min);
            out.push(q_hi * delta + min);
        }
    }
    Ok(out)
}

/// Q8_0 block: 2 bytes delta (f16) + 32 bytes i8 values → 32 f32
///
/// Each i8 value `q` is scaled: `q * delta`.
pub(super) fn dequant_q8_0(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;
    if !n.is_multiple_of(BLOCK_ELEMS) {
        return Err(ModelError::simple_load_error(format!(
            "Q8_0: n_elements {} not divisible by {}",
            n, BLOCK_ELEMS
        )));
    }
    let n_blocks = n / BLOCK_ELEMS;
    if data.len() < n_blocks * BLOCK_BYTES {
        return Err(ModelError::simple_load_error("Q8_0 data buffer too small"));
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let delta_bits = u16::from_le_bytes([data[base], data[base + 1]]);
        let delta = half::f16::from_bits(delta_bits).to_f32();
        for i in 0..BLOCK_ELEMS {
            let q = data[base + 2 + i] as i8;
            out.push(q as f32 * delta);
        }
    }
    Ok(out)
}

/// Q6_K block: 210 bytes → 256 f32 elements.
///
/// Block layout:
/// - 128 bytes: low 4 bits of each 6-bit value, packed as nibbles (ql)
/// - 64 bytes:  high 2 bits for groups of 4, packed 4-per-byte (qh)
/// - 16 bytes:  sub-block scales (i8, one per 16 elements)
/// - 2 bytes:   block scale delta (f16)
pub(super) fn dequant_q6_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
    const BLOCK_ELEMS: usize = 256;
    const BLOCK_BYTES: usize = 210;
    if !n.is_multiple_of(BLOCK_ELEMS) {
        return Err(ModelError::simple_load_error(format!(
            "Q6K: n_elements {} not divisible by {}",
            n, BLOCK_ELEMS
        )));
    }
    let n_blocks = n / BLOCK_ELEMS;
    if data.len() < n_blocks * BLOCK_BYTES {
        return Err(ModelError::simple_load_error("Q6K data buffer too small"));
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        // ql: 128 bytes (low 4 bits, packed nibbles)
        let ql = &data[base..base + 128];
        // qh: 64 bytes (high 2 bits, 4 per byte)
        let qh = &data[base + 128..base + 192];
        // scales: 16 i8 values
        let scales_raw = &data[base + 192..base + 208];
        // delta: f16
        let delta_bits = u16::from_le_bytes([data[base + 208], data[base + 209]]);
        let delta = half::f16::from_bits(delta_bits).to_f32();

        // Reconstruct 256 6-bit values
        // ql[i] holds nibbles for element i and i+128 (lower 4 bits, upper 4 bits)
        // qh[i] holds high bits for 4 consecutive pairs
        for i in 0..128usize {
            // high bits byte index and bit positions
            let qh_byte = qh[i / 2];
            let shift_lo = (i % 2) * 4; // bits [shift_lo+1 : shift_lo] for even element
            let shift_hi = (i % 2) * 4 + 2; // bits [shift_hi+1 : shift_hi] for odd element (128+i)

            let q_lo_low4 = ql[i] & 0x0F;
            let q_hi_low4 = (ql[i] >> 4) & 0x0F;

            let q_lo_high2 = (qh_byte >> shift_lo) & 0x03;
            let q_hi_high2 = (qh_byte >> shift_hi) & 0x03;

            let q_lo = ((q_lo_high2 << 4) | q_lo_low4) as i32 - 32;
            let q_hi = ((q_hi_high2 << 4) | q_hi_low4) as i32 - 32;

            // Scale: one i8 per 16 elements → 16 sub-blocks of 16 elements each
            let scale_idx_lo = (i * 2) / 16; // element i*2 / 16
            let scale_idx_hi = (i * 2 + 1) / 16;

            if scale_idx_lo >= 16 || scale_idx_hi >= 16 {
                return Err(ModelError::simple_load_error(
                    "Q6K scale index out of range",
                ));
            }
            let scale_lo = scales_raw[scale_idx_lo] as i8 as f32;
            let scale_hi = scales_raw[scale_idx_hi] as i8 as f32;

            out.push(delta * scale_lo * q_lo as f32);
            out.push(delta * scale_hi * q_hi as f32);
        }
    }
    Ok(out)
}
