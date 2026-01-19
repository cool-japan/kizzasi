//! Benchmarks for Session 9 Features
//!
//! - Kernel Fusion Optimizations
//! - ARM NEON SIMD Operations
//! - Fixed-Point Arithmetic

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_core::*;

/// Benchmark fused vs unfused LayerNorm + GELU
fn bench_fused_layernorm_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_layernorm_gelu");

    for size in [128, 256, 512, 1024].iter() {
        let x = vec![0.5f32; *size];
        let gamma = vec![1.0f32; *size];
        let beta = vec![0.0f32; *size];
        let eps = 1e-5;

        group.throughput(Throughput::Elements(*size as u64));

        // Fused version
        group.bench_with_input(BenchmarkId::new("fused", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    kernel_fusion::fused_layernorm_gelu(
                        black_box(&x),
                        black_box(&gamma),
                        black_box(&beta),
                        black_box(eps),
                    )
                    .unwrap(),
                );
            });
        });

        // Unfused version (separate operations)
        group.bench_with_input(BenchmarkId::new("unfused", size), size, |b, _| {
            b.iter(|| {
                // Compute mean
                let sum: f32 = x.iter().sum();
                let mean = sum / *size as f32;

                // Compute variance
                let var_sum: f32 = x.iter().map(|&xi| (xi - mean).powi(2)).sum();
                let variance = var_sum / *size as f32;
                let std_inv = 1.0 / (variance + eps).sqrt();

                // Normalize
                let normalized: Vec<f32> = x
                    .iter()
                    .zip(&gamma)
                    .zip(&beta)
                    .map(|((&xi, &g), &b)| {
                        let norm = (xi - mean) * std_inv;
                        norm * g + b
                    })
                    .collect();

                // GELU
                let output: Vec<f32> = normalized
                    .iter()
                    .map(|&x| {
                        let x3 = x * x * x;
                        let inner = 0.797_884_6_f32 * (x + 0.044715 * x3);
                        0.5 * x * (1.0 + inner.tanh())
                    })
                    .collect();

                black_box(output);
            });
        });
    }
    group.finish();
}

/// Benchmark fused QKV projection
fn bench_fused_qkv(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_qkv");

    for d_model in [64, 128, 256].iter() {
        let x = vec![0.5f32; *d_model];
        let w_qkv = vec![0.1f32; 3 * d_model * d_model];

        group.throughput(Throughput::Elements((3 * d_model * d_model) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(d_model), d_model, |b, _| {
            b.iter(|| {
                black_box(
                    kernel_fusion::fused_qkv_projection(
                        black_box(&x),
                        black_box(&w_qkv),
                        black_box(*d_model),
                    )
                    .unwrap(),
                );
            });
        });
    }
    group.finish();
}

/// Benchmark fused FFN
fn bench_fused_ffn(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_ffn");

    for d_model in [128, 256, 512].iter() {
        let d_ff = d_model * 4;
        let x = vec![0.5f32; *d_model];
        let w1 = vec![0.1f32; d_ff * d_model];
        let b1 = vec![0.0f32; d_ff];
        let w2 = vec![0.1f32; d_model * d_ff];
        let b2 = vec![0.0f32; *d_model];

        group.throughput(Throughput::Elements(
            (d_ff * d_model + d_model * d_ff) as u64,
        ));

        group.bench_with_input(BenchmarkId::from_parameter(d_model), d_model, |b, _| {
            b.iter(|| {
                black_box(
                    kernel_fusion::fused_ffn_gelu(
                        black_box(&x),
                        black_box(&w1),
                        black_box(&b1),
                        black_box(&w2),
                        black_box(&b2),
                        black_box(*d_model),
                        black_box(d_ff),
                    )
                    .unwrap(),
                );
            });
        });
    }
    group.finish();
}

/// Benchmark ARM NEON dot product vs standard
fn bench_neon_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_dot_product");

    for size in [128, 256, 512, 1024, 2048].iter() {
        let a = vec![0.5f32; *size];
        let b = vec![0.3f32; *size];

        group.throughput(Throughput::Elements(*size as u64));

        // NEON version
        group.bench_with_input(BenchmarkId::new("neon", size), size, |bench, _| {
            bench.iter(|| {
                black_box(simd_neon::neon_dot_product(black_box(&a), black_box(&b)));
            });
        });

        // Standard version
        group.bench_with_input(BenchmarkId::new("standard", size), size, |bench, _| {
            bench.iter(|| {
                black_box(simd::dot_product(black_box(&a), black_box(&b)));
            });
        });
    }
    group.finish();
}

/// Benchmark ARM NEON vector operations
fn bench_neon_vec_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_vec_ops");

    for size in [256, 512, 1024, 2048].iter() {
        let a = vec![0.5f32; *size];
        let b = vec![0.3f32; *size];
        let mut c = vec![0.0f32; *size];

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("neon_add", size), size, |bench, _| {
            bench.iter(|| {
                simd_neon::neon_vec_add(black_box(&a), black_box(&b), black_box(&mut c)).unwrap();
                black_box(&c);
            });
        });

        group.bench_with_input(BenchmarkId::new("neon_mul", size), size, |bench, _| {
            bench.iter(|| {
                simd_neon::neon_vec_mul(black_box(&a), black_box(&b), black_box(&mut c)).unwrap();
                black_box(&c);
            });
        });

        group.bench_with_input(BenchmarkId::new("neon_fma", size), size, |bench, _| {
            bench.iter(|| {
                simd_neon::neon_vec_fma(black_box(&a), black_box(&b), black_box(&mut c)).unwrap();
                black_box(&c);
            });
        });
    }
    group.finish();
}

/// Benchmark fixed-point vs floating-point arithmetic
fn bench_fixed_point_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_point_arithmetic");

    let iterations = 1000;

    // Basic operations
    group.throughput(Throughput::Elements(iterations));

    group.bench_function("fixed_point_add", |bencher| {
        use fixed_point::Q15_16;
        let a = Q15_16::from_f32(3.5);
        let b = Q15_16::from_f32(2.0);

        bencher.iter(|| {
            let mut result = a;
            for _ in 0..iterations {
                result = black_box(result + b);
            }
            black_box(result);
        });
    });

    group.bench_function("float_add", |bencher| {
        let a = 3.5f32;
        let b_val = 2.0f32;

        bencher.iter(|| {
            let mut result = a;
            for _ in 0..iterations {
                result = black_box(result + b_val);
            }
            black_box(result);
        });
    });

    group.bench_function("fixed_point_mul", |bencher| {
        use fixed_point::Q15_16;
        let a = Q15_16::from_f32(3.5);
        let b = Q15_16::from_f32(2.0);

        bencher.iter(|| {
            let mut result = a;
            for _ in 0..iterations {
                result = black_box(result * b);
            }
            black_box(result);
        });
    });

    group.bench_function("float_mul", |bencher| {
        let a = 3.5f32;
        let b_val = 2.0f32;

        bencher.iter(|| {
            let mut result = a;
            for _ in 0..iterations {
                result = black_box(result * b_val);
            }
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark fixed-point dot product
fn bench_fixed_point_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_point_dot_product");

    for size in [64, 128, 256, 512].iter() {
        use fixed_point::{vec_ops, Q15_16};

        let a: Vec<Q15_16> = (0..*size)
            .map(|i| Q15_16::from_f32((i as f32) * 0.01))
            .collect();
        let b: Vec<Q15_16> = (0..*size)
            .map(|i| Q15_16::from_f32((i as f32) * 0.02))
            .collect();

        let a_f32: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.01).collect();
        let b_f32: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.02).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("fixed_point", size), size, |bench, _| {
            bench.iter(|| {
                black_box(vec_ops::dot_product_q15_16(black_box(&a), black_box(&b)));
            });
        });

        group.bench_with_input(
            BenchmarkId::new("floating_point", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(simd::dot_product(black_box(&a_f32), black_box(&b_f32)));
                });
            },
        );
    }
    group.finish();
}

/// Benchmark fixed-point vector operations
fn bench_fixed_point_vec_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_point_vec_ops");

    for size in [128, 256, 512].iter() {
        use fixed_point::{vec_ops, Q15_16};

        let x: Vec<Q15_16> = (0..*size)
            .map(|i| Q15_16::from_f32((i as f32) * 0.01))
            .collect();
        let mut y = vec![Q15_16::ZERO; *size];

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("relu", size), size, |bench, _| {
            bench.iter(|| {
                vec_ops::relu_q15_16(black_box(&x), black_box(&mut y));
                black_box(&y);
            });
        });

        group.bench_with_input(BenchmarkId::new("layer_norm", size), size, |bench, _| {
            let eps = Q15_16::from_f32(1e-5);
            bench.iter(|| {
                vec_ops::layer_norm_q15_16(black_box(&x), black_box(&mut y), black_box(eps))
                    .unwrap();
                black_box(&y);
            });
        });
    }
    group.finish();
}

criterion_group!(
    session9_benches,
    bench_fused_layernorm_gelu,
    bench_fused_qkv,
    bench_fused_ffn,
    bench_neon_dot_product,
    bench_neon_vec_ops,
    bench_fixed_point_arithmetic,
    bench_fixed_point_dot_product,
    bench_fixed_point_vec_ops,
);
criterion_main!(session9_benches);
