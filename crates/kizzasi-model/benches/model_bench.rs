//! Benchmarks for kizzasi-model architectures
//!
//! Run with: cargo bench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kizzasi_core::SignalPredictor;
use kizzasi_model::{
    mamba::{Mamba, MambaConfig},
    mamba2::{Mamba2, Mamba2Config},
    rwkv::{Rwkv, RwkvConfig},
    s4::{S4Config, S4D},
    transformer::{Transformer, TransformerConfig},
};
use scirs2_core::ndarray::Array1;
use std::hint::black_box;

fn create_test_signal(length: usize) -> Vec<f32> {
    (0..length).map(|t| (t as f32 * 0.1).sin()).collect()
}

fn bench_mamba_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("mamba_forward");

    for size in [32, 64, 128, 256] {
        let config = MambaConfig::new()
            .input_dim(1)
            .hidden_dim(size)
            .state_dim(16)
            .num_layers(4);

        let mut model = Mamba::new(config).unwrap();
        let signal = create_test_signal(100);

        group.bench_with_input(BenchmarkId::new("hidden_dim", size), &size, |b, _| {
            b.iter(|| {
                for &value in &signal {
                    let input = Array1::from_vec(vec![value]);
                    let _ = black_box(model.step(&input).unwrap());
                }
                model.reset();
            });
        });
    }

    group.finish();
}

fn bench_mamba2_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("mamba2_forward");

    for size in [32, 64, 128, 256] {
        let config = Mamba2Config::new()
            .input_dim(1)
            .hidden_dim(size)
            .num_heads(8)
            .num_layers(4);

        let mut model = Mamba2::new(config).unwrap();
        let signal = create_test_signal(100);

        group.bench_with_input(BenchmarkId::new("hidden_dim", size), &size, |b, _| {
            b.iter(|| {
                for &value in &signal {
                    let input = Array1::from_vec(vec![value]);
                    let _ = black_box(model.step(&input).unwrap());
                }
                model.reset();
            });
        });
    }

    group.finish();
}

fn bench_rwkv_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("rwkv_forward");

    for size in [32, 64, 128, 256] {
        let config = RwkvConfig::new()
            .input_dim(1)
            .hidden_dim(size)
            .num_heads(8)
            .num_layers(4);

        let mut model = Rwkv::new(config).unwrap();
        let signal = create_test_signal(100);

        group.bench_with_input(BenchmarkId::new("hidden_dim", size), &size, |b, _| {
            b.iter(|| {
                for &value in &signal {
                    let input = Array1::from_vec(vec![value]);
                    let _ = black_box(model.step(&input).unwrap());
                }
                model.reset();
            });
        });
    }

    group.finish();
}

fn bench_s4d_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("s4d_forward");

    for size in [32, 64, 128, 256] {
        let config = S4Config::new()
            .input_dim(1)
            .hidden_dim(size)
            .state_dim(16)
            .num_layers(4);

        let mut model = S4D::new(config).unwrap();
        let signal = create_test_signal(100);

        group.bench_with_input(BenchmarkId::new("hidden_dim", size), &size, |b, _| {
            b.iter(|| {
                for &value in &signal {
                    let input = Array1::from_vec(vec![value]);
                    let _ = black_box(model.step(&input).unwrap());
                }
                model.reset();
            });
        });
    }

    group.finish();
}

fn bench_transformer_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformer_forward");

    for size in [32, 64, 128] {
        let config = TransformerConfig::new()
            .input_dim(1)
            .hidden_dim(size)
            .num_heads(8)
            .num_layers(4)
            .max_seq_len(512);

        let mut model = Transformer::new(config).unwrap();
        let signal = create_test_signal(100);

        group.bench_with_input(BenchmarkId::new("hidden_dim", size), &size, |b, _| {
            b.iter(|| {
                for &value in &signal {
                    let input = Array1::from_vec(vec![value]);
                    let _ = black_box(model.step(&input).unwrap());
                }
                model.reset();
            });
        });
    }

    group.finish();
}

fn bench_model_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_comparison");
    let signal = create_test_signal(100);

    // Mamba
    let mamba_config = MambaConfig::new()
        .input_dim(1)
        .hidden_dim(128)
        .state_dim(16)
        .num_layers(4);
    let mut mamba = Mamba::new(mamba_config).unwrap();

    group.bench_function("Mamba", |b| {
        b.iter(|| {
            for &value in &signal {
                let input = Array1::from_vec(vec![value]);
                let _ = black_box(mamba.step(&input).unwrap());
            }
            mamba.reset();
        });
    });

    // Mamba2
    let mamba2_config = Mamba2Config::new()
        .input_dim(1)
        .hidden_dim(128)
        .num_heads(8)
        .num_layers(4);
    let mut mamba2 = Mamba2::new(mamba2_config).unwrap();

    group.bench_function("Mamba2", |b| {
        b.iter(|| {
            for &value in &signal {
                let input = Array1::from_vec(vec![value]);
                let _ = black_box(mamba2.step(&input).unwrap());
            }
            mamba2.reset();
        });
    });

    // RWKV
    let rwkv_config = RwkvConfig::new()
        .input_dim(1)
        .hidden_dim(128)
        .num_heads(8)
        .num_layers(4);
    let mut rwkv = Rwkv::new(rwkv_config).unwrap();

    group.bench_function("RWKV", |b| {
        b.iter(|| {
            for &value in &signal {
                let input = Array1::from_vec(vec![value]);
                let _ = black_box(rwkv.step(&input).unwrap());
            }
            rwkv.reset();
        });
    });

    // S4D
    let s4_config = S4Config::new()
        .input_dim(1)
        .hidden_dim(128)
        .state_dim(16)
        .num_layers(4);
    let mut s4 = S4D::new(s4_config).unwrap();

    group.bench_function("S4D", |b| {
        b.iter(|| {
            for &value in &signal {
                let input = Array1::from_vec(vec![value]);
                let _ = black_box(s4.step(&input).unwrap());
            }
            s4.reset();
        });
    });

    // Transformer
    let transformer_config = TransformerConfig::new()
        .input_dim(1)
        .hidden_dim(128)
        .num_heads(8)
        .num_layers(4)
        .max_seq_len(512);
    let mut transformer = Transformer::new(transformer_config).unwrap();

    group.bench_function("Transformer", |b| {
        b.iter(|| {
            for &value in &signal {
                let input = Array1::from_vec(vec![value]);
                let _ = black_box(transformer.step(&input).unwrap());
            }
            transformer.reset();
        });
    });

    group.finish();
}

fn bench_single_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_step");

    let input = Array1::from_vec(vec![0.5]);

    // Mamba
    let mamba_config = MambaConfig::new()
        .input_dim(1)
        .hidden_dim(128)
        .state_dim(16)
        .num_layers(4);
    let mut mamba = Mamba::new(mamba_config).unwrap();

    group.bench_function("Mamba", |b| {
        b.iter(|| {
            let _ = black_box(mamba.step(&input).unwrap());
        });
    });

    // Mamba2
    let mamba2_config = Mamba2Config::new()
        .input_dim(1)
        .hidden_dim(128)
        .num_heads(8)
        .num_layers(4);
    let mut mamba2 = Mamba2::new(mamba2_config).unwrap();

    group.bench_function("Mamba2", |b| {
        b.iter(|| {
            let _ = black_box(mamba2.step(&input).unwrap());
        });
    });

    // RWKV
    let rwkv_config = RwkvConfig::new()
        .input_dim(1)
        .hidden_dim(128)
        .num_heads(8)
        .num_layers(4);
    let mut rwkv = Rwkv::new(rwkv_config).unwrap();

    group.bench_function("RWKV", |b| {
        b.iter(|| {
            let _ = black_box(rwkv.step(&input).unwrap());
        });
    });

    // S4D
    let s4_config = S4Config::new()
        .input_dim(1)
        .hidden_dim(128)
        .state_dim(16)
        .num_layers(4);
    let mut s4 = S4D::new(s4_config).unwrap();

    group.bench_function("S4D", |b| {
        b.iter(|| {
            let _ = black_box(s4.step(&input).unwrap());
        });
    });

    // Transformer
    let transformer_config = TransformerConfig::new()
        .input_dim(1)
        .hidden_dim(128)
        .num_heads(8)
        .num_layers(4)
        .max_seq_len(512);
    let mut transformer = Transformer::new(transformer_config).unwrap();

    group.bench_function("Transformer", |b| {
        b.iter(|| {
            let _ = black_box(transformer.step(&input).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_mamba_forward,
    bench_mamba2_forward,
    bench_rwkv_forward,
    bench_s4d_forward,
    bench_transformer_forward,
    bench_model_comparison,
    bench_single_step
);
criterion_main!(benches);
