//! Comprehensive benchmarks for advanced features in kizzasi-inference
//!
//! Covers:
//! - Precision conversions (FP32 to FP16/BF16)
//! - Temporal logic constraint evaluation
//! - Streaming engine async performance

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_inference::{
    streaming::{StreamConfig, StreamingEngine},
    temporal::{LTLFormula, STLFormula, TemporalBound, TemporalConstraintEnforcer},
    PrecisionConfig, PrecisionConverter,
};
use scirs2_core::ndarray::{Array1, Array2};
use std::hint::black_box;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Benchmark precision conversion operations
fn bench_precision_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_conversion");
    let config = PrecisionConfig::default();
    let converter = PrecisionConverter::new(config);

    // 1D arrays
    for size in [32, 64, 128, 256, 512, 1024].iter() {
        let data = Array1::from_elem(*size, 0.5f32);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("fp32_to_fp16_1d", size), size, |b, _| {
            b.iter(|| converter.to_fp16_1d(black_box(&data)))
        });

        group.bench_with_input(BenchmarkId::new("fp32_to_bf16_1d", size), size, |b, _| {
            b.iter(|| converter.to_bf16_1d(black_box(&data)))
        });
    }

    // 2D arrays
    for dim in [8, 16, 32, 64, 128].iter() {
        let data = Array2::from_elem((*dim, *dim), 0.5f32);
        group.throughput(Throughput::Elements((dim * dim) as u64));

        group.bench_with_input(BenchmarkId::new("fp32_to_fp16_2d", dim), dim, |b, _| {
            b.iter(|| converter.to_fp16_2d(black_box(&data)))
        });

        group.bench_with_input(BenchmarkId::new("fp32_to_bf16_2d", dim), dim, |b, _| {
            b.iter(|| converter.to_bf16_2d(black_box(&data)))
        });
    }

    group.finish();
}

/// Benchmark temporal logic constraint evaluation
fn bench_temporal_constraints(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_constraints");

    // LTL evaluation
    let ltl_always = LTLFormula::Always(Box::new(LTLFormula::Atomic(|x| x[0] > 0.0)));
    let ltl_eventually = LTLFormula::Eventually(Box::new(LTLFormula::Atomic(|x| x[0] > 0.5)));

    for trace_len in [10, 50, 100, 200, 500].iter() {
        let trace: Vec<Array1<f32>> = (0..*trace_len)
            .map(|i| Array1::from_vec(vec![0.3 + (i as f32) * 0.001]))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("ltl_always", trace_len),
            trace_len,
            |b, _| b.iter(|| ltl_always.check(black_box(&trace), 0)),
        );

        group.bench_with_input(
            BenchmarkId::new("ltl_eventually", trace_len),
            trace_len,
            |b, _| b.iter(|| ltl_eventually.check(black_box(&trace), 0)),
        );
    }

    // STL robustness evaluation
    let stl_predicate = STLFormula::Predicate(|x| x[0] - 0.5);
    let stl_always = STLFormula::Always {
        formula: Box::new(STLFormula::Predicate(|x| x[0] - 0.3)),
        bound: TemporalBound::new(0.0, 10.0),
    };

    for trace_len in [10, 50, 100, 200].iter() {
        let trace: Vec<Array1<f32>> = (0..*trace_len)
            .map(|i| Array1::from_vec(vec![0.4 + (i as f32) * 0.001]))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("stl_predicate", trace_len),
            trace_len,
            |b, _| b.iter(|| stl_predicate.robustness(black_box(&trace), 0.0)),
        );

        group.bench_with_input(
            BenchmarkId::new("stl_always", trace_len),
            trace_len,
            |b, _| b.iter(|| stl_always.robustness(black_box(&trace), 0.0)),
        );
    }

    // Temporal enforcer with multiple constraints
    group.bench_function("enforcer_check_all", |b| {
        let mut enforcer = TemporalConstraintEnforcer::new(100);

        // Add LTL constraints
        enforcer.add_ltl(LTLFormula::Atomic(|x| x[0] > 0.0));
        enforcer.add_ltl(LTLFormula::Atomic(|x| x[0] > 0.1));
        enforcer.add_ltl(LTLFormula::Atomic(|x| x[0] > 0.2));
        enforcer.add_ltl(LTLFormula::Atomic(|x| x[0] > 0.3));
        enforcer.add_ltl(LTLFormula::Atomic(|x| x[0] > 0.4));

        // Add some trace data
        for i in 0..50 {
            enforcer.update(Array1::from_vec(vec![0.5 + (i as f32) * 0.01]));
        }

        b.iter(|| black_box(enforcer.check_all()))
    });

    group.finish();
}

/// Benchmark streaming engine async operations
fn bench_streaming_async(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("streaming_async");

    // Benchmark single async step
    group.bench_function("single_step_async", |b| {
        b.iter(|| {
            rt.block_on(async {
                let stream_config = StreamConfig::default();
                let engine = StreamingEngine::new(stream_config).unwrap();
                let input = Array1::from_elem(32, 0.5);
                black_box(engine.step_async(input).await.unwrap())
            })
        })
    });

    // Benchmark async rollout with different lengths
    for steps in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("rollout_async", steps),
            steps,
            |b, &steps| {
                b.iter(|| {
                    rt.block_on(async {
                        let stream_config = StreamConfig::default();
                        let engine = StreamingEngine::new(stream_config).unwrap();
                        let input = Array1::from_elem(32, 0.5);
                        black_box(engine.rollout_async(input, steps).await.unwrap())
                    })
                })
            },
        );
    }

    // Benchmark concurrent async operations
    for concurrency in [2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_async", concurrency),
            concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    rt.block_on(async {
                        let stream_config = StreamConfig::default();
                        let engine = Arc::new(StreamingEngine::new(stream_config).unwrap());

                        let mut handles = vec![];
                        for _ in 0..concurrency {
                            let engine_clone = Arc::clone(&engine);
                            let handle = tokio::spawn(async move {
                                let input = Array1::from_elem(32, 0.5);
                                engine_clone.step_async(input).await.unwrap()
                            });
                            handles.push(handle);
                        }

                        for handle in handles {
                            black_box(handle.await.unwrap());
                        }
                    })
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    advanced_benches,
    bench_precision_conversion,
    bench_temporal_constraints,
    bench_streaming_async,
);
criterion_main!(advanced_benches);
