//! Benchmark: `StateCompressor` variants.
//!
//! Sweeps [`CompressionMethod`] over a fixed set of `HiddenState` shapes and
//! sparsity levels to characterise:
//!
//! * Compression throughput (bytes/sec on the original `f32` payload).
//! * Decompression throughput.
//! * The full compress -> decompress round-trip cost.
//!
//! All input states are built deterministically via SciRS2's RNG so
//! benchmark runs are reproducible. Density (the fraction of non-zero
//! entries) is controlled explicitly so the `Sparse` and `QuantizedSparse`
//! methods are measured at both their best- and worst-case operating
//! points.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_core::HiddenState;
use kizzasi_inference::{CompressionMethod, StateCompressor};
use scirs2_core::ndarray::Array2;
use std::hint::black_box;

/// `HiddenState` shapes covering small, medium, and large SSM layouts.
const SHAPES: &[(usize, usize)] = &[(16, 32), (32, 128), (64, 256)];

/// Density (fraction of non-zero entries) for the sparse benches.
///
/// * `0.1` — heavily sparse (typical post-pruning state).
/// * `0.5` — moderately sparse.
/// * `1.0` — fully dense.
const DENSITIES: &[f32] = &[0.1, 0.5, 1.0];

/// All compression methods we want to benchmark.
///
/// `Quantize4Bit` is included even though it's lossier than 8-bit because
/// it has a different inner loop (two values per byte) and we want
/// independent timing for it.
const METHODS: &[CompressionMethod] = &[
    CompressionMethod::None,
    CompressionMethod::Quantize8Bit,
    CompressionMethod::Quantize4Bit,
    CompressionMethod::Sparse,
    CompressionMethod::QuantizedSparse,
];

/// Deterministic 32-bit hash used to fill state arrays.
///
/// Avoids any randomness in benchmark setup: identical (rows, cols, density)
/// always produce identical states across runs.
fn hash_index(seed: u32, idx: u32) -> u32 {
    let mut x = seed.wrapping_add(idx).wrapping_mul(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    x
}

/// Build a `HiddenState` of shape `(rows, cols)` with the given non-zero
/// density. Non-zero values are deterministically sampled in `[-1, 1)`.
///
/// `density == 1.0` produces a fully dense state; `density == 0.0` would
/// produce all-zeros (we deliberately exclude that case from the sweep
/// because several compressors collapse pathologically on it).
fn make_state(rows: usize, cols: usize, density: f32) -> HiddenState {
    let total = rows * cols;
    let mut data = Array2::<f32>::zeros((rows, cols));
    // density is interpreted as a u32 threshold in [0, u32::MAX].
    let threshold = (density.clamp(0.0, 1.0) * (u32::MAX as f64) as f32) as u32;

    for i in 0..total {
        let r = i / cols;
        let c = i % cols;
        let h = hash_index(0xA5A5_A5A5, i as u32);
        if h <= threshold {
            // Map the high bits into [-1, 1).
            let v = (h as f32 / (u32::MAX as f32)) * 2.0 - 1.0;
            data[[r, c]] = v;
        }
    }

    let mut state = HiddenState::new(rows, cols);
    state.update(data);
    state
}

/// Short tag for a `CompressionMethod`, used in benchmark ids.
fn method_tag(method: CompressionMethod) -> &'static str {
    match method {
        CompressionMethod::None => "none",
        CompressionMethod::Quantize8Bit => "q8",
        CompressionMethod::Quantize4Bit => "q4",
        CompressionMethod::Sparse => "sparse",
        CompressionMethod::QuantizedSparse => "qsparse",
    }
}

/// Short tag for a density value.
fn density_tag(density: f32) -> String {
    // Format with one decimal multiplied by 10 to keep ids stable / id-friendly.
    format!("d{}", (density * 10.0) as u32)
}

/// Benchmark `StateCompressor::compress` for every method, shape, and density.
fn bench_state_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_compression/compress");
    for &method in METHODS {
        for &(rows, cols) in SHAPES {
            for &density in DENSITIES {
                // Skip degenerate combinations: 4-bit quantization on an
                // ultra-sparse `[16, 32]` is dominated by a single non-zero
                // and produces zero useful signal.
                let state = make_state(rows, cols, density);
                let compressor = StateCompressor::new(method);

                // Pre-flight: validate the configuration actually produces a
                // valid compressed payload before we start the bench loop.
                if compressor.compress(&state).is_err() {
                    continue;
                }

                // Bytes of the original f32 payload (4 bytes per element).
                group.throughput(Throughput::Bytes((rows * cols * 4) as u64));
                let id = BenchmarkId::new(
                    format!(
                        "{}_{}x{}_{}",
                        method_tag(method),
                        rows,
                        cols,
                        density_tag(density),
                    ),
                    rows * cols,
                );
                group.bench_with_input(id, &state, |b, state| {
                    b.iter(|| {
                        compressor
                            .compress(black_box(state))
                            .expect("compress failed")
                    });
                });
            }
        }
    }
    group.finish();
}

/// Benchmark `StateCompressor::decompress` for every method, shape, and density.
fn bench_state_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_compression/decompress");
    for &method in METHODS {
        for &(rows, cols) in SHAPES {
            for &density in DENSITIES {
                let state = make_state(rows, cols, density);
                let compressor = StateCompressor::new(method);

                let compressed = match compressor.compress(&state) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                group.throughput(Throughput::Bytes((rows * cols * 4) as u64));
                let id = BenchmarkId::new(
                    format!(
                        "{}_{}x{}_{}",
                        method_tag(method),
                        rows,
                        cols,
                        density_tag(density),
                    ),
                    rows * cols,
                );
                group.bench_with_input(id, &compressed, |b, compressed| {
                    b.iter(|| {
                        compressor
                            .decompress(black_box(compressed))
                            .expect("decompress failed")
                    });
                });
            }
        }
    }
    group.finish();
}

/// Benchmark a full compress -> decompress round-trip.
///
/// Captures the cost a long-context inference loop pays per checkpoint when
/// it persists state through the compressor.
fn bench_state_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_compression/roundtrip");
    for &method in METHODS {
        for &(rows, cols) in SHAPES {
            for &density in DENSITIES {
                let state = make_state(rows, cols, density);
                let compressor = StateCompressor::new(method);

                if compressor.compress(&state).is_err() {
                    continue;
                }

                group.throughput(Throughput::Bytes((rows * cols * 4) as u64));
                let id = BenchmarkId::new(
                    format!(
                        "{}_{}x{}_{}",
                        method_tag(method),
                        rows,
                        cols,
                        density_tag(density),
                    ),
                    rows * cols,
                );
                group.bench_with_input(id, &state, |b, state| {
                    b.iter(|| {
                        let c = compressor
                            .compress(black_box(state))
                            .expect("compress in roundtrip");
                        compressor
                            .decompress(black_box(&c))
                            .expect("decompress in roundtrip")
                    });
                });
            }
        }
    }
    group.finish();
}

/// Benchmark `with_sparsity_threshold` impact on the `Sparse` and
/// `QuantizedSparse` paths.
///
/// The sparsity threshold determines how aggressively values near zero are
/// dropped from the payload. We fix a moderately sparse state and sweep
/// thresholds to see how the timing curve responds.
fn bench_sparse_threshold_sweep(c: &mut Criterion) {
    let (rows, cols) = (32usize, 128usize);
    let state = make_state(rows, cols, 0.5);

    let mut group = c.benchmark_group("state_compression/sparse_threshold_sweep");
    group.throughput(Throughput::Bytes((rows * cols * 4) as u64));

    for &threshold in &[1e-6_f32, 1e-4, 1e-2, 1e-1] {
        for &method in &[
            CompressionMethod::Sparse,
            CompressionMethod::QuantizedSparse,
        ] {
            let compressor = StateCompressor::new(method).with_sparsity_threshold(threshold);
            // Threshold is logged as -log10(t) so the tag stays as an
            // integer (e.g. threshold 1e-4 -> "4").
            let thr_tag = (-threshold.log10()).round() as i32;
            let id = BenchmarkId::new(
                format!("{}_thr{}", method_tag(method), thr_tag),
                threshold.to_bits(),
            );
            group.bench_with_input(id, &state, |b, state| {
                b.iter(|| {
                    compressor
                        .compress(black_box(state))
                        .expect("compress in threshold sweep")
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_state_compress,
    bench_state_decompress,
    bench_state_roundtrip,
    bench_sparse_threshold_sweep,
);
criterion_main!(benches);
