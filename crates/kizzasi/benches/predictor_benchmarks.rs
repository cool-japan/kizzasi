//! Performance benchmarks for Kizzasi predictor
//!
//! Run with:
//! ```bash
//! cargo bench --bench predictor_benchmarks
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi::prelude::*;
use std::hint::black_box;

// Benchmark single-step prediction with different model sizes
fn bench_step_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_prediction");

    let configs = vec![
        ("tiny", 2, 2, 16, 4, 1),
        ("small", 3, 3, 32, 8, 2),
        ("medium", 4, 4, 64, 16, 3),
        ("large", 8, 8, 128, 32, 4),
    ];

    for (name, input_dim, output_dim, hidden_dim, state_dim, num_layers) in configs {
        let config = KizzasiConfig::new()
            .input_dim(input_dim)
            .output_dim(output_dim)
            .hidden_dim(hidden_dim)
            .state_dim(state_dim)
            .num_layers(num_layers);

        let mut predictor = Kizzasi::new(config).unwrap();
        let input = Array1::from_vec(vec![0.5; input_dim]);

        group.bench_with_input(BenchmarkId::from_parameter(name), name, |b, _| {
            b.iter(|| predictor.step(black_box(&input)))
        });
    }

    group.finish();
}

// Benchmark multi-step prediction
fn bench_predict_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("predict_n");
    group.sample_size(50);

    let config = KizzasiConfig::new()
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(64)
        .state_dim(16)
        .num_layers(2);

    let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);

    for n_steps in [10, 50, 100, 200].iter() {
        let mut predictor = Kizzasi::new(config.clone()).unwrap();

        group.throughput(Throughput::Elements(*n_steps as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_steps), n_steps, |b, &n| {
            b.iter(|| {
                predictor.reset();
                predictor.predict_n(black_box(&input), black_box(n))
            })
        });
    }

    group.finish();
}

// Benchmark batch prediction
fn bench_predict_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("predict_batch");

    let config = KizzasiConfig::new()
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(32)
        .state_dim(8)
        .num_layers(2);

    for batch_size in [1, 10, 50, 100].iter() {
        let mut predictor = Kizzasi::new(config.clone()).unwrap();
        let inputs: Vec<Array1<f32>> = (0..*batch_size)
            .map(|i| Array1::from_vec(vec![i as f32 * 0.1, 0.2, 0.3]))
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    predictor.reset();
                    predictor.predict_batch(black_box(&inputs))
                })
            },
        );
    }

    group.finish();
}

// Benchmark different presets
fn bench_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("presets");

    // Audio preset
    {
        let mut predictor = KizzasiBuilder::audio_preset().build().unwrap();
        let input_dim = predictor.config().get_input_dim();
        let input = Array1::from_vec(vec![0.5; input_dim]);
        group.bench_function("audio", |b| b.iter(|| predictor.step(black_box(&input))));
    }

    // Robotics 3-DOF preset
    {
        let mut predictor = KizzasiBuilder::robotics_preset(3).build().unwrap();
        let input_dim = predictor.config().get_input_dim();
        let input = Array1::from_vec(vec![0.5; input_dim]);
        group.bench_function("robotics_3dof", |b| {
            b.iter(|| predictor.step(black_box(&input)))
        });
    }

    // Robotics 6-DOF preset
    {
        let mut predictor = KizzasiBuilder::robotics_preset(6).build().unwrap();
        let input_dim = predictor.config().get_input_dim();
        let input = Array1::from_vec(vec![0.5; input_dim]);
        group.bench_function("robotics_6dof", |b| {
            b.iter(|| predictor.step(black_box(&input)))
        });
    }

    // Sensor preset
    {
        let mut predictor = KizzasiBuilder::sensor_preset(10).build().unwrap();
        let input_dim = predictor.config().get_input_dim();
        let input = Array1::from_vec(vec![0.5; input_dim]);
        group.bench_function("sensor_10", |b| {
            b.iter(|| predictor.step(black_box(&input)))
        });
    }

    // Lightweight preset
    {
        let mut predictor = KizzasiBuilder::lightweight_preset(2, 2).build().unwrap();
        let input_dim = predictor.config().get_input_dim();
        let input = Array1::from_vec(vec![0.5; input_dim]);
        group.bench_function("lightweight", |b| {
            b.iter(|| predictor.step(black_box(&input)))
        });
    }

    // Control preset
    {
        let mut predictor = KizzasiBuilder::control_preset(8, 4).build().unwrap();
        let input_dim = predictor.config().get_input_dim();
        let input = Array1::from_vec(vec![0.5; input_dim]);
        group.bench_function("control", |b| b.iter(|| predictor.step(black_box(&input))));
    }

    group.finish();
}

// Benchmark reset operation
fn bench_reset(c: &mut Criterion) {
    let config = KizzasiConfig::new()
        .input_dim(4)
        .output_dim(4)
        .hidden_dim(128)
        .state_dim(32)
        .num_layers(4);

    let mut predictor = Kizzasi::new(config).unwrap();
    let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

    // Run some predictions to establish state
    for _ in 0..10 {
        let _ = predictor.step(&input);
    }

    c.bench_function("reset", |b| b.iter(|| predictor.reset()));
}

// Benchmark fork operation
fn bench_fork(c: &mut Criterion) {
    let config = KizzasiConfig::new()
        .input_dim(4)
        .output_dim(4)
        .hidden_dim(64)
        .state_dim(16)
        .num_layers(3);

    let predictor = Kizzasi::new(config).unwrap();

    c.bench_function("fork", |b| b.iter(|| black_box(predictor.fork().unwrap())));
}

// Benchmark with guardrails
#[cfg(feature = "logic")]
fn bench_with_guardrails(c: &mut Criterion) {
    use kizzasi::{ConstraintBuilder, Guardrail, GuardrailSet};

    let mut group = c.benchmark_group("guardrails");

    let config = KizzasiConfig::new()
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(32)
        .state_dim(8)
        .num_layers(2);

    let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);

    // Without guardrails
    let mut predictor_no_guards = Kizzasi::new(config.clone()).unwrap();
    group.bench_function("without_guardrails", |b| {
        b.iter(|| predictor_no_guards.step(black_box(&input)))
    });

    // With guardrails
    let mut predictor_with_guards = Kizzasi::new(config).unwrap();
    let mut guardrails = GuardrailSet::new();
    let constraint = ConstraintBuilder::new()
        .name("bounds")
        .greater_eq(-1.0)
        .less_eq(1.0)
        .build()
        .unwrap();
    guardrails.add_global(Guardrail::new(constraint, false));
    predictor_with_guards.set_guardrails(guardrails);

    group.bench_function("with_guardrails", |b| {
        b.iter(|| predictor_with_guards.step(black_box(&input)))
    });

    group.finish();
}

// Benchmark SignalInput conversions
fn bench_signal_input_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_input_conversions");

    let vec_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
    let array_data = [0.1f32, 0.2, 0.3, 0.4, 0.5];
    let slice_data: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5];

    group.bench_function("from_vec", |b| {
        b.iter(|| {
            let _: SignalInput = black_box(vec_data.clone()).into();
        })
    });

    group.bench_function("from_array", |b| {
        b.iter(|| {
            let _: SignalInput = black_box(array_data).into();
        })
    });

    group.bench_function("from_slice", |b| {
        b.iter(|| {
            let _: SignalInput = black_box(slice_data).into();
        })
    });

    group.bench_function("from_f32", |b| {
        b.iter(|| {
            let _: SignalInput = black_box(0.5f32).into();
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_step_prediction,
    bench_predict_n,
    bench_predict_batch,
    bench_presets,
    bench_reset,
    bench_fork,
    bench_signal_input_conversions,
);

#[cfg(feature = "logic")]
criterion_group!(guardrail_benches, bench_with_guardrails);

#[cfg(feature = "logic")]
criterion_main!(benches, guardrail_benches);

#[cfg(not(feature = "logic"))]
criterion_main!(benches);
