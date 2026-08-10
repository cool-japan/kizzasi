//! Benchmark: range coder encode/decode throughput.
//!
//! Drives [`RangeEncoder::encode`] and [`RangeDecoder::decode`] on synthetic
//! symbol streams. Two axes are swept:
//!
//! * `n` — number of symbols in the input stream.
//! * `skew` — geometric-distribution skew that controls how unbalanced the
//!   symbol probabilities are. `skew = 0.0` is a uniform distribution
//!   (worst-case for entropy coders), `skew = 0.9` is heavily skewed (best
//!   case — the coder can compress aggressively).
//!
//! Throughput is reported as **bytes per second**, where one symbol is
//! treated as 4 bytes (the size of `u32`) on the input side. This is the
//! conventional "uncompressed bytes / sec" metric used for entropy coders.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_tokenizer::{RangeDecoder, RangeEncoder};
use std::collections::HashMap;
use std::hint::black_box;

/// Symbol stream lengths to sweep.
const STREAM_SIZES: &[usize] = &[100, 1_000, 10_000];

/// Geometric distribution skew parameters.
///
/// * `0.0` — uniform symbol probabilities.
/// * `0.5` — moderate skew (each subsequent symbol half as likely).
/// * `0.9` — very heavy skew.
const SKEWS: &[f32] = &[0.0, 0.5, 0.9];

/// Number of distinct symbols in the alphabet.
const NUM_SYMBOLS: u32 = 16;

/// Build a frequency table over `[0, num_symbols)` with a geometric-style
/// skew.
///
/// `skew` is interpreted as a per-symbol decay ratio. `skew = 0.0` produces
/// the uniform distribution. Each symbol receives at least one count so the
/// cumulative-frequency table is well-defined.
fn build_freqs(num_symbols: u32, skew: f32) -> HashMap<u32, u64> {
    // Decay ratio per symbol index. Clamp to (0, 1] so we always get a
    // strictly positive, finite frequency for every symbol.
    let ratio = (1.0_f32 - skew).clamp(1e-3, 1.0);
    let mut freqs = HashMap::with_capacity(num_symbols as usize);
    for s in 0..num_symbols {
        let f = (ratio.powi(s as i32) * 1000.0).max(0.0) as u64 + 1;
        freqs.insert(s, f);
    }
    freqs
}

/// Build a deterministic symbol stream of length `n` whose distribution
/// roughly matches a `(num_symbols, skew)` geometric distribution.
///
/// Concretely we precompute the cumulative probability table from
/// `build_freqs` and map a deterministic LCG sequence to symbols by inverse
/// CDF lookup. The stream is reproducible across runs so benchmarks compare
/// like-for-like.
fn build_symbols(n: usize, num_symbols: u32, skew: f32) -> Vec<u32> {
    let freqs = build_freqs(num_symbols, skew);
    let total: u64 = freqs.values().sum();

    // Build sorted (symbol, cumulative) so we can binary-search.
    let mut entries: Vec<(u32, u64)> = (0..num_symbols)
        .map(|s| (s, *freqs.get(&s).expect("symbol must exist in freqs")))
        .collect();
    entries.sort_by_key(|e| e.0);

    let mut cumulative: Vec<(u64, u32)> = Vec::with_capacity(entries.len());
    let mut acc = 0u64;
    for (sym, freq) in entries {
        acc += freq;
        cumulative.push((acc, sym));
    }

    // Tiny LCG (Numerical Recipes constants) for a deterministic, fast
    // pseudo-random sequence — sufficient to feed the coder.
    let mut state: u64 = 0xC0FFEE_u64.wrapping_add(n as u64);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let draw = state % total;
        let sym = cumulative
            .iter()
            .find(|(c, _)| draw < *c)
            .map(|(_, s)| *s)
            .unwrap_or(num_symbols - 1);
        out.push(sym);
    }
    out
}

/// Pre-built bench input. Owned so the benchmark closure can simply borrow.
struct CodecInput {
    n: usize,
    skew: f32,
    encoder: RangeEncoder,
    decoder: RangeDecoder,
    symbols: Vec<u32>,
    encoded: Vec<u8>,
}

/// Build all inputs once so the bench timing does not include setup cost.
fn build_inputs() -> Vec<CodecInput> {
    let mut inputs = Vec::new();
    for &n in STREAM_SIZES {
        for &skew in SKEWS {
            let freqs = build_freqs(NUM_SYMBOLS, skew);
            let encoder = RangeEncoder::from_frequencies(freqs.clone())
                .expect("RangeEncoder::from_frequencies failed");
            let decoder = RangeDecoder::from_frequencies(freqs)
                .expect("RangeDecoder::from_frequencies failed");
            let symbols = build_symbols(n, NUM_SYMBOLS, skew);
            let encoded = encoder
                .encode(&symbols)
                .expect("RangeEncoder::encode failed in setup");
            inputs.push(CodecInput {
                n,
                skew,
                encoder,
                decoder,
                symbols,
                encoded,
            });
        }
    }
    inputs
}

/// Format `skew` into a stable string for the benchmark id (`0_5` etc.).
fn skew_tag(skew: f32) -> String {
    // Multiply by 10 and truncate to avoid floats in benchmark ids.
    format!("{}", (skew * 10.0) as u32)
}

/// Benchmark `RangeEncoder::encode` across (n, skew) combinations.
fn bench_range_encode(c: &mut Criterion) {
    let inputs = build_inputs();
    let mut group = c.benchmark_group("range_coder/encode");
    for input in &inputs {
        // Throughput in bytes/sec of *input* (4 bytes per symbol).
        group.throughput(Throughput::Bytes((input.n * 4) as u64));
        let id = BenchmarkId::new(
            format!("n{}_skew{}", input.n, skew_tag(input.skew)),
            input.n,
        );
        group.bench_with_input(id, input, |b, input| {
            b.iter(|| {
                input
                    .encoder
                    .encode(black_box(&input.symbols))
                    .expect("range_encoder.encode failed")
            });
        });
    }
    group.finish();
}

/// Benchmark `RangeDecoder::decode` across (n, skew) combinations.
fn bench_range_decode(c: &mut Criterion) {
    let inputs = build_inputs();
    let mut group = c.benchmark_group("range_coder/decode");
    for input in &inputs {
        group.throughput(Throughput::Bytes((input.n * 4) as u64));
        let id = BenchmarkId::new(
            format!("n{}_skew{}", input.n, skew_tag(input.skew)),
            input.n,
        );
        group.bench_with_input(id, input, |b, input| {
            b.iter(|| {
                input
                    .decoder
                    .decode(black_box(&input.encoded))
                    .expect("range_decoder.decode failed")
            });
        });
    }
    group.finish();
}

/// Benchmark a complete encode + decode round-trip.
///
/// Useful for end-to-end characterisation of streaming pipelines that pay
/// both costs per chunk.
fn bench_range_roundtrip(c: &mut Criterion) {
    let inputs = build_inputs();
    let mut group = c.benchmark_group("range_coder/roundtrip");
    for input in &inputs {
        group.throughput(Throughput::Bytes((input.n * 4) as u64));
        let id = BenchmarkId::new(
            format!("n{}_skew{}", input.n, skew_tag(input.skew)),
            input.n,
        );
        group.bench_with_input(id, input, |b, input| {
            b.iter(|| {
                let bytes = input
                    .encoder
                    .encode(black_box(&input.symbols))
                    .expect("encode failed in roundtrip");
                input
                    .decoder
                    .decode(black_box(&bytes))
                    .expect("decode failed in roundtrip")
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_range_encode,
    bench_range_decode,
    bench_range_roundtrip,
);
criterion_main!(benches);
