//! Byte-level tests for the K-quant decoders.
//!
//! The reference encoders below are written directly from the GGML block
//! layouts as *closed-form index formulas* (element index → payload byte, bit
//! position and sub-block), deliberately not sharing the decoders' loop shape.
//!
//! Every test uses **non-uniform** block contents. A block filled with a single
//! repeated quant value is invariant under any permutation of the output, which
//! is exactly why a wrong element order can survive a naive round-trip test.
//! The suite therefore combines:
//!
//! * one-hot tests — a single non-neutral quant, asserting *which* output index
//!   moves (catches permutation errors);
//! * per-sub-block scale tests — every sub-block gets a distinct scale/min
//!   (catches sub-block boundary errors);
//! * full positional round-trips over distinct quants (catches everything else).

use super::*;

/// Sub-block index that owns element `e` (16 elements per sub-block).
fn sub16(e: usize) -> usize {
    e / 16
}

/// Sub-block index that owns element `e` (32 elements per sub-block).
fn sub32(e: usize) -> usize {
    e / 32
}

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

// ─────────────────────────────────────────────────────────────────────────────
// Q2_K
// ─────────────────────────────────────────────────────────────────────────────

/// Build one `block_q2_K` (84 bytes).
///
/// `scale_nibbles[s]` / `min_nibbles[s]` are the 4-bit scale and min of
/// sub-block `s` (16 sub-blocks of 16 elements); `quants[e]` is the 2-bit quant
/// of element `e` in *logical element order*.
fn build_q2_k(
    d: f32,
    dmin: f32,
    scale_nibbles: &[u8; 16],
    min_nibbles: &[u8; 16],
    quants: &[u8; 256],
) -> Vec<u8> {
    let mut block = vec![0u8; 84];
    for s in 0..16usize {
        block[s] = (scale_nibbles[s] & 0x0F) | ((min_nibbles[s] & 0x0F) << 4);
    }
    for (e, q) in quants.iter().enumerate() {
        let group = e / 128;
        let r = e % 128;
        let byte = group * 32 + (r % 32);
        let shift = 2 * (r / 32);
        block[16 + byte] |= (q & 3) << shift;
    }
    block[80..82].copy_from_slice(&f16_le(d));
    block[82..84].copy_from_slice(&f16_le(dmin));
    block
}

#[test]
fn q2_k_positional_round_trip_with_distinct_scales() {
    let scale_nibbles: [u8; 16] = std::array::from_fn(|s| (s as u8) % 16);
    let min_nibbles: [u8; 16] = std::array::from_fn(|s| ((15 - s) as u8) % 16);
    let quants: [u8; 256] = std::array::from_fn(|e| ((e * 7) % 4) as u8);

    let block = build_q2_k(1.0, 0.5, &scale_nibbles, &min_nibbles, &quants);
    let out = dequant_q2_k(&block, 256).expect("q2_k");

    for e in 0..256usize {
        let s = sub16(e);
        let want = 1.0 * scale_nibbles[s] as f32 * quants[e] as f32 - 0.5 * min_nibbles[s] as f32;
        assert!(
            (out[e] - want).abs() < 1e-4,
            "element {e} (sub-block {s}): {} != {want}",
            out[e]
        );
    }
}

#[test]
fn q2_k_single_quant_lands_on_its_own_element() {
    // dmin = 0 so untouched elements decode to exactly 0.0.
    let scale_nibbles = [1u8; 16];
    let min_nibbles = [0u8; 16];
    let mut quants = [0u8; 256];
    quants[131] = 3;

    let block = build_q2_k(1.0, 0.0, &scale_nibbles, &min_nibbles, &quants);
    let out = dequant_q2_k(&block, 256).expect("q2_k");
    assert_eq!(nonzero(&out), vec![(131usize, 3.0f32)]);
}

#[test]
fn q2_k_reads_all_sixteen_sub_block_scales() {
    // One quant per sub-block; each must pick up its own scale. Q2_K scales are
    // 4-bit, so the distinct values cycle through 1..=15.
    let scale_nibbles: [u8; 16] = std::array::from_fn(|s| (s as u8 % 15) + 1);
    let min_nibbles = [0u8; 16];
    let mut quants = [0u8; 256];
    for s in 0..16usize {
        quants[s * 16 + 5] = 1;
    }

    let block = build_q2_k(1.0, 0.0, &scale_nibbles, &min_nibbles, &quants);
    let out = dequant_q2_k(&block, 256).expect("q2_k");
    let got = nonzero(&out);
    assert_eq!(got.len(), 16, "expected exactly one non-zero per sub-block");
    for (s, (idx, val)) in got.iter().enumerate() {
        assert_eq!(*idx, s * 16 + 5);
        assert!(
            (val - scale_nibbles[s] as f32).abs() < 1e-5,
            "sub-block {s}: {val} != {}",
            scale_nibbles[s]
        );
    }
}

#[test]
fn q2_k_multiset_matches_reference() {
    let scale_nibbles = [3u8; 16];
    let min_nibbles = [2u8; 16];
    let quants: [u8; 256] = std::array::from_fn(|e| (e % 4) as u8);
    let block = build_q2_k(0.5, 0.25, &scale_nibbles, &min_nibbles, &quants);
    let out = dequant_q2_k(&block, 256).expect("q2_k");
    let expected: Vec<f32> = (0..256)
        .map(|e| 0.5 * 3.0 * quants[e] as f32 - 0.25 * 2.0)
        .collect();
    assert_eq!(sorted(&out), sorted(&expected));
}

#[test]
fn q2_k_consumes_the_whole_super_block() {
    // Mutating any byte of the 84-byte block must change the decoded output.
    let scale_nibbles: [u8; 16] = std::array::from_fn(|s| (s as u8 % 15) + 1);
    let min_nibbles = [1u8; 16];
    let quants: [u8; 256] = std::array::from_fn(|e| ((e / 3) % 4) as u8);
    let block = build_q2_k(1.0, 0.5, &scale_nibbles, &min_nibbles, &quants);
    let base = dequant_q2_k(&block, 256).expect("q2_k");

    for byte in 0..84usize {
        let mut mutated = block.clone();
        mutated[byte] ^= 0x11;
        let out = dequant_q2_k(&mutated, 256).expect("q2_k");
        assert_ne!(out, base, "byte {byte} of the Q2_K block is never read");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Q3_K
// ─────────────────────────────────────────────────────────────────────────────

/// Build one `block_q3_K` (110 bytes).
///
/// `raw_scales[s]` is the *biased* 6-bit sub-block scale (`scale = raw - 32`);
/// `quants[e]` is the 3-bit value of element `e` (`value = quant - 4`).
fn build_q3_k(d: f32, raw_scales: &[u8; 16], quants: &[u8; 256]) -> Vec<u8> {
    let mut block = vec![0u8; 110];
    for (e, q) in quants.iter().enumerate() {
        let group = e / 128;
        let r = e % 128;
        let payload = r % 32;
        let shift = 2 * (r / 32);
        block[32 + group * 32 + payload] |= (q & 3) << shift;
        if (q >> 2) & 1 == 1 {
            block[payload] |= 1u8 << (4 * group + r / 32);
        }
    }
    let packed = pack_q3_k_scales(raw_scales);
    block[96..108].copy_from_slice(&packed);
    block[108..110].copy_from_slice(&f16_le(d));
    block
}

#[test]
fn q3_k_scale_packing_round_trips() {
    let raw: [u8; 16] = std::array::from_fn(|s| ((s * 5 + 1) % 64) as u8);
    let packed = pack_q3_k_scales(&raw);
    let unpacked = unpack_q3_k_scales(&packed).expect("unpack");
    assert_eq!(unpacked, raw);
}

#[test]
fn q3_k_positional_round_trip_with_distinct_scales() {
    let raw_scales: [u8; 16] = std::array::from_fn(|s| (32 + s) as u8);
    let quants: [u8; 256] = std::array::from_fn(|e| ((e * 3) % 8) as u8);

    let block = build_q3_k(1.0, &raw_scales, &quants);
    let out = dequant_q3_k(&block, 256).expect("q3_k");

    for e in 0..256usize {
        let s = sub16(e);
        let want = (raw_scales[s] as f32 - 32.0) * (quants[e] as f32 - 4.0);
        assert!(
            (out[e] - want).abs() < 1e-4,
            "element {e} (sub-block {s}): {} != {want}",
            out[e]
        );
    }
}

#[test]
fn q3_k_high_bit_is_indexed_by_payload_byte_not_element() {
    // Every element neutral (quant 4 → value 0) except element 200.
    let raw_scales = [33u8; 16]; // scale = 1
    let mut quants = [4u8; 256];
    quants[200] = 7; // value = 3
    let block = build_q3_k(1.0, &raw_scales, &quants);
    let out = dequant_q3_k(&block, 256).expect("q3_k");
    assert_eq!(nonzero(&out), vec![(200usize, 3.0f32)]);
}

#[test]
fn q3_k_reads_all_sixteen_sub_block_scales() {
    let raw_scales: [u8; 16] = std::array::from_fn(|s| (33 + s) as u8);
    let mut quants = [4u8; 256];
    for s in 0..16usize {
        quants[s * 16 + 9] = 5; // value = 1
    }
    let block = build_q3_k(1.0, &raw_scales, &quants);
    let out = dequant_q3_k(&block, 256).expect("q3_k");
    let got = nonzero(&out);
    assert_eq!(got.len(), 16);
    for (s, (idx, val)) in got.iter().enumerate() {
        assert_eq!(*idx, s * 16 + 9);
        assert!((val - (s as f32 + 1.0)).abs() < 1e-5, "sub-block {s}");
    }
}

#[test]
fn q3_k_consumes_the_whole_super_block() {
    let raw_scales: [u8; 16] = std::array::from_fn(|s| (35 + s) as u8);
    let quants: [u8; 256] = std::array::from_fn(|e| ((e / 5) % 8) as u8);
    let block = build_q3_k(1.0, &raw_scales, &quants);
    let base = dequant_q3_k(&block, 256).expect("q3_k");
    for byte in 0..110usize {
        let mut mutated = block.clone();
        mutated[byte] ^= 0x11;
        let out = dequant_q3_k(&mutated, 256).expect("q3_k");
        assert_ne!(out, base, "byte {byte} of the Q3_K block is never read");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Q4_K
// ─────────────────────────────────────────────────────────────────────────────

/// Build one `block_q4_K` (144 bytes) from per-sub-block 6-bit scales/mins and
/// 4-bit quants in logical element order.
fn build_q4_k(d: f32, dmin: f32, scales: &[u8; 8], mins: &[u8; 8], quants: &[u8; 256]) -> Vec<u8> {
    let mut block = vec![0u8; 144];
    block[0..2].copy_from_slice(&f16_le(d));
    block[2..4].copy_from_slice(&f16_le(dmin));
    block[4..16].copy_from_slice(&pack_scales_min_k4(scales, mins));
    for (e, q) in quants.iter().enumerate() {
        let pair = e / 64;
        let within = e % 64;
        let half = within / 32;
        let k = within % 32;
        let byte = 16 + pair * 32 + k;
        block[byte] |= (q & 0x0F) << (4 * half);
    }
    block
}

#[test]
fn q4_k_scale_packing_round_trips() {
    let scales: [u8; 8] = std::array::from_fn(|s| ((s * 9 + 1) % 64) as u8);
    let mins: [u8; 8] = std::array::from_fn(|s| ((s * 7 + 3) % 64) as u8);
    let packed = pack_scales_min_k4(&scales, &mins);
    for j in 0..8usize {
        let (sc, m) = get_scale_min_k4(j, &packed);
        assert_eq!(sc, scales[j], "scale {j}");
        assert_eq!(m, mins[j], "min {j}");
    }
}

#[test]
fn q4_k_positional_round_trip_with_distinct_scales() {
    let scales: [u8; 8] = std::array::from_fn(|s| (s + 1) as u8);
    let mins: [u8; 8] = std::array::from_fn(|s| (8 - s) as u8);
    let quants: [u8; 256] = std::array::from_fn(|e| ((e * 5) % 16) as u8);

    let block = build_q4_k(0.5, 0.25, &scales, &mins, &quants);
    let out = dequant_q4_k(&block, 256).expect("q4_k");

    for e in 0..256usize {
        let s = sub32(e);
        let want = 0.5 * scales[s] as f32 * quants[e] as f32 - 0.25 * mins[s] as f32;
        assert!(
            (out[e] - want).abs() < 1e-3,
            "element {e} (sub-block {s}): {} != {want}",
            out[e]
        );
    }
}

#[test]
fn q4_k_low_and_high_nibbles_belong_to_paired_sub_blocks() {
    // Elements 0..32 come from the low nibbles of bytes 0..32 and elements
    // 32..64 from the *high* nibbles of the same bytes.
    let scales = [1u8; 8];
    let mins = [0u8; 8];
    let mut quants = [0u8; 256];
    quants[40] = 9;
    let block = build_q4_k(1.0, 0.0, &scales, &mins, &quants);
    let out = dequant_q4_k(&block, 256).expect("q4_k");
    assert_eq!(nonzero(&out), vec![(40usize, 9.0f32)]);
}

#[test]
fn q4_k_reads_all_eight_sub_block_scales() {
    let scales: [u8; 8] = std::array::from_fn(|s| (s + 1) as u8);
    let mins = [0u8; 8];
    let mut quants = [0u8; 256];
    for s in 0..8usize {
        quants[s * 32 + 11] = 1;
    }
    let block = build_q4_k(1.0, 0.0, &scales, &mins, &quants);
    let out = dequant_q4_k(&block, 256).expect("q4_k");
    let got = nonzero(&out);
    assert_eq!(got.len(), 8);
    for (s, (idx, val)) in got.iter().enumerate() {
        assert_eq!(*idx, s * 32 + 11);
        assert!((val - (s as f32 + 1.0)).abs() < 1e-5, "sub-block {s}");
    }
}

#[test]
fn q4_k_consumes_the_whole_super_block() {
    let scales: [u8; 8] = std::array::from_fn(|s| (s + 3) as u8);
    let mins: [u8; 8] = std::array::from_fn(|s| (s + 1) as u8);
    let quants: [u8; 256] = std::array::from_fn(|e| ((e / 2) % 16) as u8);
    let block = build_q4_k(1.0, 0.5, &scales, &mins, &quants);
    let base = dequant_q4_k(&block, 256).expect("q4_k");
    for byte in 0..144usize {
        let mut mutated = block.clone();
        mutated[byte] ^= 0x11;
        let out = dequant_q4_k(&mutated, 256).expect("q4_k");
        assert_ne!(out, base, "byte {byte} of the Q4_K block is never read");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Q5_K
// ─────────────────────────────────────────────────────────────────────────────

/// Build one `block_q5_K` (176 bytes); `quants[e]` is a full 5-bit value.
fn build_q5_k(d: f32, dmin: f32, scales: &[u8; 8], mins: &[u8; 8], quants: &[u8; 256]) -> Vec<u8> {
    let mut block = vec![0u8; 176];
    block[0..2].copy_from_slice(&f16_le(d));
    block[2..4].copy_from_slice(&f16_le(dmin));
    block[4..16].copy_from_slice(&pack_scales_min_k4(scales, mins));
    for (e, q) in quants.iter().enumerate() {
        let pair = e / 64;
        let within = e % 64;
        let half = within / 32;
        let k = within % 32;
        block[48 + pair * 32 + k] |= (q & 0x0F) << (4 * half);
        if (q >> 4) & 1 == 1 {
            block[16 + k] |= 1u8 << (2 * pair + half);
        }
    }
    block
}

#[test]
fn q5_k_positional_round_trip_with_distinct_scales() {
    let scales: [u8; 8] = std::array::from_fn(|s| (s + 1) as u8);
    let mins: [u8; 8] = std::array::from_fn(|s| (s % 4) as u8);
    let quants: [u8; 256] = std::array::from_fn(|e| ((e * 11) % 32) as u8);

    let block = build_q5_k(0.5, 0.5, &scales, &mins, &quants);
    let out = dequant_q5_k(&block, 256).expect("q5_k");

    for e in 0..256usize {
        let s = sub32(e);
        let want = 0.5 * scales[s] as f32 * quants[e] as f32 - 0.5 * mins[s] as f32;
        assert!(
            (out[e] - want).abs() < 1e-3,
            "element {e} (sub-block {s}): {} != {want}",
            out[e]
        );
    }
}

#[test]
fn q5_k_high_bit_uses_a_shared_qh_byte_per_pair() {
    // The qh byte index is the payload index k (0..32) for *every* pair; only
    // the bit selector advances. Element 200 lives in pair 3, half 0, k = 8.
    let scales = [1u8; 8];
    let mins = [0u8; 8];
    let mut quants = [0u8; 256];
    quants[200] = 17; // low nibble 1, high bit set
    let block = build_q5_k(1.0, 0.0, &scales, &mins, &quants);
    let out = dequant_q5_k(&block, 256).expect("q5_k");
    assert_eq!(nonzero(&out), vec![(200usize, 17.0f32)]);
}

#[test]
fn q5_k_reads_all_eight_sub_block_scales() {
    let scales: [u8; 8] = std::array::from_fn(|s| (s + 1) as u8);
    let mins = [0u8; 8];
    let mut quants = [0u8; 256];
    for s in 0..8usize {
        quants[s * 32 + 3] = 16; // pure high bit → value 16
    }
    let block = build_q5_k(1.0, 0.0, &scales, &mins, &quants);
    let out = dequant_q5_k(&block, 256).expect("q5_k");
    let got = nonzero(&out);
    assert_eq!(got.len(), 8);
    for (s, (idx, val)) in got.iter().enumerate() {
        assert_eq!(*idx, s * 32 + 3);
        assert!(
            (val - 16.0 * (s as f32 + 1.0)).abs() < 1e-4,
            "sub-block {s}"
        );
    }
}

#[test]
fn q5_k_consumes_the_whole_super_block() {
    let scales: [u8; 8] = std::array::from_fn(|s| (s + 2) as u8);
    let mins: [u8; 8] = std::array::from_fn(|s| (s + 1) as u8);
    let quants: [u8; 256] = std::array::from_fn(|e| ((e / 2) % 32) as u8);
    let block = build_q5_k(1.0, 0.5, &scales, &mins, &quants);
    let base = dequant_q5_k(&block, 256).expect("q5_k");
    for byte in 0..176usize {
        let mut mutated = block.clone();
        mutated[byte] ^= 0x11;
        let out = dequant_q5_k(&mutated, 256).expect("q5_k");
        assert_ne!(out, base, "byte {byte} of the Q5_K block is never read");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Q8_K
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn q8_k_identity() {
    let mut block = vec![0u8; 292];
    block[0..4].copy_from_slice(&1.0f32.to_le_bytes());
    for i in 0..256usize {
        block[4 + i] = i as u8;
    }

    let result = dequant_q8_k(&block, 256).expect("q8_k");
    assert_eq!(result.len(), 256);
    for i in 0..256usize {
        let expected = (block[4 + i] as i8) as f32;
        assert!((result[i] - expected).abs() < 1e-5, "element {i}");
    }
}

#[test]
fn q8_k_scale_multiply() {
    let mut block = vec![0u8; 292];
    block[0..4].copy_from_slice(&0.5f32.to_le_bytes());
    for i in 0..256usize {
        block[4 + i] = 100u8;
    }
    let result = dequant_q8_k(&block, 256).expect("q8_k");
    for (i, &v) in result.iter().enumerate() {
        assert!((v - 50.0).abs() < 1e-4, "element {i}: {v}");
    }
}

#[test]
fn q8_k_negative_quants() {
    let mut block = vec![0u8; 292];
    block[0..4].copy_from_slice(&2.0f32.to_le_bytes());
    for i in 0..256usize {
        block[4 + i] = (-10i8) as u8;
    }
    let result = dequant_q8_k(&block, 256).expect("q8_k");
    for &v in &result {
        assert!((v - (-20.0)).abs() < 1e-4, "expected -20.0, got {v}");
    }
}

#[test]
fn q8_k_multi_block() {
    let mut data = vec![0u8; 292 * 2];
    for blk in 0..2usize {
        let base = blk * 292;
        data[base..base + 4].copy_from_slice(&3.0f32.to_le_bytes());
        for i in 0..256usize {
            data[base + 4 + i] = 1u8;
        }
    }
    let result = dequant_q8_k(&data, 512).expect("q8_k");
    assert_eq!(result.len(), 512);
    for &v in &result {
        assert!((v - 3.0).abs() < 1e-4, "expected 3.0, got {v}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-block and error handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn k_quants_decode_two_blocks_independently() {
    let scales: [u8; 8] = std::array::from_fn(|s| (s + 1) as u8);
    let mins = [0u8; 8];
    let quants_a: [u8; 256] = std::array::from_fn(|e| (e % 16) as u8);
    let quants_b: [u8; 256] = std::array::from_fn(|e| (15 - e % 16) as u8);

    let mut data = build_q4_k(1.0, 0.0, &scales, &mins, &quants_a);
    data.extend_from_slice(&build_q4_k(1.0, 0.0, &scales, &mins, &quants_b));

    let out = dequant_q4_k(&data, 512).expect("q4_k");
    assert_eq!(out.len(), 512);
    for e in 0..256usize {
        let s = sub32(e);
        let want_a = scales[s] as f32 * quants_a[e] as f32;
        let want_b = scales[s] as f32 * quants_b[e] as f32;
        assert!((out[e] - want_a).abs() < 1e-3, "block 0 element {e}");
        assert!((out[256 + e] - want_b).abs() < 1e-3, "block 1 element {e}");
    }
}

#[test]
fn zero_elements_is_an_error() {
    let data = vec![0u8; 292];
    assert!(dequant_q2_k(&data, 0).is_err());
    assert!(dequant_q3_k(&data, 0).is_err());
    assert!(dequant_q4_k(&data, 0).is_err());
    assert!(dequant_q5_k(&data, 0).is_err());
    assert!(dequant_q8_k(&data, 0).is_err());
}

#[test]
fn misaligned_element_count_is_an_error() {
    let data = vec![0u8; 1024];
    assert!(dequant_q2_k(&data, 128).is_err());
    assert!(dequant_q3_k(&data, 255).is_err());
    assert!(dequant_q4_k(&data, 257).is_err());
    assert!(dequant_q5_k(&data, 1).is_err());
    assert!(dequant_q8_k(&data, 511).is_err());
}

#[test]
fn short_buffer_is_an_error() {
    let tiny = vec![0u8; 10];
    assert!(dequant_q2_k(&tiny, 256).is_err());
    assert!(dequant_q3_k(&tiny, 256).is_err());
    assert!(dequant_q4_k(&tiny, 256).is_err());
    assert!(dequant_q5_k(&tiny, 256).is_err());
    assert!(dequant_q8_k(&tiny, 256).is_err());
}
