//! Benchmarks for optimization techniques
//!
//! Measures the performance impact of:
//! - Discretization caching
//! - Workspace pooling
//! - ILP operations
//! - Cache-aligned data structures

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kizzasi_core::optimizations::{ilp, CacheAligned, DiscretizationCache, WorkspaceGuard};
use kizzasi_core::{KizzasiConfig, SelectiveSSM, SignalPredictor};
use scirs2_core::ndarray::{arr1, Array1, Array2};

fn bench_discretization_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("discretization_cache");

    for &(name, hidden_dim, state_dim) in
        &[("small", 64, 8), ("medium", 256, 16), ("large", 512, 32)]
    {
        // Benchmark uncached
        group.bench_with_input(
            BenchmarkId::new("uncached", name),
            &(hidden_dim, state_dim),
            |bencher, &(h, s)| {
                bencher.iter(|| {
                    let a_mat = Array2::ones((h, s));
                    let b_mat = Array2::ones((h, s));
                    let delta = 0.1f32;
                    let _a_bar = a_mat.mapv(|x: f32| (delta * x).exp());
                    let _b_bar = b_mat.mapv(|x: f32| delta * x);
                });
            },
        );

        // Benchmark cached
        group.bench_with_input(
            BenchmarkId::new("cached", name),
            &(hidden_dim, state_dim),
            |bencher, &(h, s)| {
                let mut cache = DiscretizationCache::new(1, h, s);
                let a_mat = Array2::ones((h, s));
                let b_mat = Array2::ones((h, s));
                let delta = 0.1f32;
                let a_bar = a_mat.mapv(|x: f32| (delta * x).exp());
                let b_bar = b_mat.mapv(|x: f32| delta * x);
                cache.update(0, delta, a_bar, b_bar);

                bencher.iter(|| {
                    let _cached = cache.get(0, delta).expect("cache should hit");
                });
            },
        );
    }

    group.finish();
}

fn bench_workspace_pooling(c: &mut Criterion) {
    let mut group = c.benchmark_group("workspace_pooling");

    for &size in &[64, 256, 512] {
        // Benchmark with allocation
        group.bench_with_input(
            BenchmarkId::new("with_alloc", size),
            &size,
            |bencher, &s| {
                bencher.iter(|| {
                    let _temp = Array1::<f32>::zeros(s);
                    let _temp2 = Array2::<f32>::zeros((s, 8));
                });
            },
        );

        // Benchmark with pooling
        group.bench_with_input(BenchmarkId::new("with_pool", size), &size, |bencher, &s| {
            bencher.iter(|| {
                let _workspace = WorkspaceGuard::new(s, 8);
            });
        });
    }

    group.finish();
}

fn bench_ilp_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("ilp_operations");

    for &size in &[256, 1024, 4096] {
        let arr_a = arr1(&vec![1.0f32; size]);
        let arr_b = arr1(&vec![2.0f32; size]);

        // Standard dot product
        group.bench_with_input(
            BenchmarkId::new("standard_dot", size),
            &(&arr_a, &arr_b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    black_box((*a).dot(*b));
                });
            },
        );

        // ILP dot product
        group.bench_with_input(
            BenchmarkId::new("ilp_dot", size),
            &(&arr_a, &arr_b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    black_box(ilp::dot_unrolled(a.view(), b.view()));
                });
            },
        );
    }

    group.finish();
}

fn bench_cache_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_alignment");

    for &size in &[256, 1024, 4096] {
        let regular_data = vec![1.0f32; size];
        let aligned_data = CacheAligned::new(vec![1.0f32; size]);

        // Regular sum
        group.bench_with_input(
            BenchmarkId::new("regular", size),
            &regular_data,
            |bencher, data| {
                bencher.iter(|| {
                    black_box(data.iter().sum::<f32>());
                });
            },
        );

        // Aligned sum
        group.bench_with_input(
            BenchmarkId::new("aligned", size),
            &aligned_data,
            |bencher, data| {
                bencher.iter(|| {
                    black_box(data.get().iter().sum::<f32>());
                });
            },
        );
    }

    group.finish();
}

fn bench_ssm_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssm_full_optimization");

    for &(name, hidden_dim) in &[("small", 64), ("medium", 256), ("large", 512)] {
        let config = KizzasiConfig::new()
            .input_dim(8)
            .output_dim(8)
            .hidden_dim(hidden_dim)
            .state_dim(16)
            .num_layers(4);

        // Standard SSM step
        group.bench_with_input(
            BenchmarkId::new("standard", name),
            &config,
            |bencher, cfg| {
                let mut ssm = SelectiveSSM::new(cfg.clone()).expect("SSM creation failed");
                let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

                bencher.iter(|| {
                    black_box(ssm.step(&input).expect("step should succeed"));
                });
            },
        );

        // Optimized SSM step with workspace pooling
        group.bench_with_input(
            BenchmarkId::new("optimized", name),
            &config,
            |bencher, cfg| {
                let mut ssm = SelectiveSSM::new(cfg.clone()).expect("SSM creation failed");
                let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

                bencher.iter(|| {
                    let _workspace = WorkspaceGuard::new(hidden_dim, 16);
                    black_box(ssm.step(&input).expect("step should succeed"));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_discretization_cache,
    bench_workspace_pooling,
    bench_ilp_operations,
    bench_cache_alignment,
    bench_ssm_optimized,
);
criterion_main!(benches);
