//! Benchmark: sequential vs parallel scan throughput.
//!
//! Measures the speedup of [`parallel_scan`] (delegating to
//! `scirs2_core::distributed::parallel_scan::parallel_scan` Blelloch
//! three-phase algorithm) versus its sequential fallback at array sizes that
//! span the transition point from "too small to parallelize" (the local
//! `parallel_scan` only enters the parallel path for `data.len() >= 64`) to
//! "fully parallel" (large enough to cross `scirs2-core`'s sequential
//! threshold).
//!
//! Two operator families are exercised:
//!
//! * `AddOp` — scalar `f32` addition. Has an identity (`0.0`), so the
//!   parallel path actually reaches `scirs2-core`'s Blelloch implementation.
//! * `SSMScanOp` — the production SSM associative operator. It now carries a
//!   real identity element (`(ones(state_dim), zeros(state_dim))`, keyed by
//!   the state dimension the operator is constructed with), so `parallel_scan`
//!   also reaches the Blelloch implementation for this operator family, not
//!   just for `AddOp` -- this group measures actual SSM parallel-scan
//!   speedup, not merely dispatch overhead on top of a forced sequential
//!   fallback.
//!
//! Each benchmark reports throughput as elements per second.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_core::scan::{parallel_scan, AssociativeOp, SSMElement, SSMScanOp};
use scirs2_core::ndarray::Array1;
use std::hint::black_box;

/// Sizes spanning the relevant parallel-scan transition points.
///
/// * `32`, `48` — below `parallel_scan`'s internal `len >= 64` cutoff; the
///   "parallel" path collapses to sequential.
/// * `64`, `128`, `256` — above the local cutoff but typically below
///   `scirs2-core`'s `SEQUENTIAL_THRESHOLD`.
/// * `1024`, `4096`, `16384` — large enough to actually exercise the
///   work-efficient Blelloch parallel scan.
const SIZES: &[usize] = &[32, 64, 128, 256, 1024, 4096, 16384];

/// Reduced set of sizes for the SSM operator. Running 16384-element SSM
/// scans bloats wall time without adding signal beyond what 1024 already
/// shows.
const SSM_SIZES: &[usize] = &[64, 256, 1024];

/// SSM state dimension used for the per-element `SSMElement` payload.
const SSM_STATE_DIM: usize = 16;

/// Simple `f32`-addition associative operator with an identity (`0.0`).
///
/// Implements [`AssociativeOp`] so this bench can drive the production
/// `parallel_scan` API without touching internal helpers.
struct AddOp;

impl AssociativeOp<f32> for AddOp {
    fn combine(&self, a: &f32, b: &f32) -> f32 {
        a + b
    }

    fn identity(&self) -> Option<f32> {
        Some(0.0)
    }
}

/// Build a deterministic `Vec<f32>` of the requested length.
///
/// Using integer values (mod 128) keeps every partial sum representable
/// exactly in `f32`, so different scan paths are bit-identical and the
/// benchmark numbers are not contaminated by NaN/Inf edge cases.
fn make_f32_data(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 128) as f32).collect()
}

/// Build a deterministic `Vec<SSMElement>` of the requested length.
///
/// Each element is a small `state_dim`-vector with values near `0.9` for
/// `a_bar` (numerically stable across long sequences) and small `b_bar`
/// values, mirroring what an actual SSM emits.
fn make_ssm_data(n: usize, state_dim: usize) -> Vec<SSMElement> {
    (0..n)
        .map(|t| {
            let a_bar =
                Array1::from_shape_fn(state_dim, |d| 0.90 + (((t * 7 + d) % 19) as f32) * 0.001);
            let b_bar =
                Array1::from_shape_fn(state_dim, |d| 0.01 + (((t * 3 + d) % 17) as f32) * 0.0005);
            SSMElement { a_bar, b_bar }
        })
        .collect()
}

/// Reference inclusive sequential scan over `f32` (sum).
///
/// Used to benchmark the pure-sequential reduction independently of the
/// dispatch logic inside `parallel_scan`. Equivalent in shape to the private
/// `scan::sequential_scan`, but written here so the bench does not depend on
/// an internal item.
fn sequential_f32_sum(data: &[f32]) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    let mut acc = data[0];
    out.push(acc);
    for &v in &data[1..] {
        acc += v;
        out.push(acc);
    }
    out
}

/// Reference inclusive sequential scan over `SSMElement`.
fn sequential_ssm_scan(data: &[SSMElement]) -> Vec<SSMElement> {
    if data.is_empty() {
        return Vec::new();
    }
    let op = SSMScanOp::new(SSM_STATE_DIM);
    let mut out = Vec::with_capacity(data.len());
    out.push(data[0].clone());
    for i in 1..data.len() {
        let combined = op.combine(&out[i - 1], &data[i]);
        out.push(combined);
    }
    out
}

/// Pure-sequential reduction baseline.
///
/// Calls `sequential_f32_sum` directly so the measurement excludes the
/// `parallel_scan` dispatch overhead.
fn bench_sequential_scan_addop(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan/sequential_addop");
    for &n in SIZES {
        let data = make_f32_data(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| sequential_f32_sum(black_box(&data)));
        });
    }
    group.finish();
}

/// `parallel_scan` called with `parallel = false` — i.e. forced into its
/// internal sequential path. Measures the dispatch overhead on top of the
/// raw sequential scan.
fn bench_parallel_scan_addop_force_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan/parallel_addop_force_sequential");
    for &n in SIZES {
        let data = make_f32_data(n);
        let op = AddOp;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| parallel_scan(black_box(&data), &op, false));
        });
    }
    group.finish();
}

/// `parallel_scan` with `parallel = true`. Drops into `scirs2-core`'s
/// Blelloch implementation for inputs `>= 64`.
fn bench_parallel_scan_addop(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan/parallel_addop");
    for &n in SIZES {
        let data = make_f32_data(n);
        let op = AddOp;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| parallel_scan(black_box(&data), &op, true));
        });
    }
    group.finish();
}

/// Sequential SSM scan over the production `SSMScanOp` operator.
fn bench_sequential_scan_ssm(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan/sequential_ssm");
    for &n in SSM_SIZES {
        let data = make_ssm_data(n, SSM_STATE_DIM);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| sequential_ssm_scan(black_box(&data)));
        });
    }
    group.finish();
}

/// `parallel_scan` invoked with `SSMScanOp` (now carrying a real identity for
/// `state_dim = SSM_STATE_DIM`). Reaches `scirs2-core`'s Blelloch
/// implementation inside `parallel_scan_impl` for `n >= 64`, exercising the
/// actual parallel path rather than a forced sequential fallback.
fn bench_parallel_scan_ssm(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan/parallel_ssm");
    for &n in SSM_SIZES {
        let data = make_ssm_data(n, SSM_STATE_DIM);
        let op = SSMScanOp::new(SSM_STATE_DIM);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| parallel_scan(black_box(&data), &op, true));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sequential_scan_addop,
    bench_parallel_scan_addop_force_sequential,
    bench_parallel_scan_addop,
    bench_sequential_scan_ssm,
    bench_parallel_scan_ssm,
);
criterion_main!(benches);
