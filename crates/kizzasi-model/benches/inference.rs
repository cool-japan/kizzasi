//! Criterion benchmark suite for kizzasi-model inference
//!
//! Benchmark groups:
//! 1. per_step_latency  - single step inference for each model type
//! 2. batch_throughput  - processing multiple steps in sequence
//! 3. model_size_scaling - hidden dim scaling
//! 4. sequence_length_scaling - accumulated state over varying sequence lengths

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kizzasi_core::SignalPredictor;
use scirs2_core::ndarray::Array1;
use std::hint::black_box;

#[cfg(feature = "mamba")]
use kizzasi_model::mamba::{Mamba, MambaConfig};

#[cfg(feature = "mamba")]
use kizzasi_model::mamba2::{Mamba2, Mamba2Config};

use kizzasi_model::h3::{H3Config, H3};
use kizzasi_model::rwkv::{Rwkv, RwkvConfig};
use kizzasi_model::rwkv7::{Rwkv7Config, Rwkv7Model};
use kizzasi_model::s4::{S4Config, S4D};
use kizzasi_model::s5::{S5Config, S5};
use kizzasi_model::transformer::{Transformer, TransformerConfig};

// ---------------------------------------------------------------------------
// Group 1: per_step_latency
// ---------------------------------------------------------------------------

fn bench_per_step_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("per_step_latency");
    group.sample_size(50);

    let input = Array1::from_vec(vec![0.5]);

    // Mamba
    #[cfg(feature = "mamba")]
    {
        let config = MambaConfig::new()
            .input_dim(1)
            .hidden_dim(256)
            .state_dim(16)
            .num_layers(4);
        if let Ok(mut model) = Mamba::new(config) {
            group.bench_function("Mamba", |b| {
                b.iter(|| {
                    let _ = black_box(model.step(&input));
                });
            });
        }
    }

    // Mamba2
    #[cfg(feature = "mamba")]
    {
        let config = Mamba2Config::new()
            .input_dim(1)
            .hidden_dim(256)
            .num_heads(8)
            .num_layers(4);
        if let Ok(mut model) = Mamba2::new(config) {
            group.bench_function("Mamba2", |b| {
                b.iter(|| {
                    let _ = black_box(model.step(&input));
                });
            });
        }
    }

    // RWKV v6
    {
        let config = RwkvConfig::new()
            .input_dim(1)
            .hidden_dim(256)
            .num_heads(8)
            .num_layers(4);
        if let Ok(mut model) = Rwkv::new(config) {
            group.bench_function("RWKV_v6", |b| {
                b.iter(|| {
                    let _ = black_box(model.step(&input));
                });
            });
        }
    }

    // RWKV v7
    {
        let config = Rwkv7Config {
            input_dim: 1,
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            head_dim: 64,
            expand_factor: 2.0,
            context_length: 1024,
            time_decay_init: -5.0,
        };
        if let Ok(mut model) = Rwkv7Model::new(config) {
            group.bench_function("RWKV_v7", |b| {
                b.iter(|| {
                    let _ = black_box(model.step(&input));
                });
            });
        }
    }

    // S4D
    {
        let config = S4Config::new()
            .input_dim(1)
            .hidden_dim(256)
            .state_dim(16)
            .num_layers(4);
        if let Ok(mut model) = S4D::new(config) {
            group.bench_function("S4D", |b| {
                b.iter(|| {
                    let _ = black_box(model.step(&input));
                });
            });
        }
    }

    // S5
    {
        let config = S5Config::new(1, 256, 4);
        if let Ok(mut model) = S5::new(config) {
            group.bench_function("S5", |b| {
                b.iter(|| {
                    let _ = black_box(model.step(&input));
                });
            });
        }
    }

    // H3
    {
        let config = H3Config::new(1, 256, 4);
        if let Ok(mut model) = H3::new(config) {
            group.bench_function("H3", |b| {
                b.iter(|| {
                    let _ = black_box(model.step(&input));
                });
            });
        }
    }

    // Transformer
    {
        let config = TransformerConfig::new()
            .input_dim(1)
            .hidden_dim(256)
            .num_heads(8)
            .num_layers(4);
        if let Ok(mut model) = Transformer::new(config) {
            group.bench_function("Transformer", |b| {
                b.iter(|| {
                    let _ = black_box(model.step(&input));
                });
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: batch_throughput
// ---------------------------------------------------------------------------

fn bench_batch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_throughput");
    group.sample_size(20);

    for &batch_size in &[1usize, 4, 16, 64] {
        let signal: Vec<f32> = (0..batch_size).map(|t| (t as f32 * 0.1).sin()).collect();

        // RWKV v7
        {
            let config = Rwkv7Config {
                input_dim: 1,
                hidden_dim: 128,
                num_layers: 2,
                num_heads: 4,
                head_dim: 32,
                expand_factor: 2.0,
                context_length: 512,
                time_decay_init: -5.0,
            };
            if let Ok(mut model) = Rwkv7Model::new(config) {
                group.bench_with_input(
                    BenchmarkId::new("RWKV_v7", batch_size),
                    &batch_size,
                    |b, _| {
                        b.iter(|| {
                            for &val in &signal {
                                let input = Array1::from_vec(vec![val]);
                                let _ = black_box(model.step(&input));
                            }
                            model.reset();
                        });
                    },
                );
            }
        }

        // RWKV v6
        {
            let config = RwkvConfig::new()
                .input_dim(1)
                .hidden_dim(128)
                .num_heads(4)
                .num_layers(2);
            if let Ok(mut model) = Rwkv::new(config) {
                group.bench_with_input(
                    BenchmarkId::new("RWKV_v6", batch_size),
                    &batch_size,
                    |b, _| {
                        b.iter(|| {
                            for &val in &signal {
                                let input = Array1::from_vec(vec![val]);
                                let _ = black_box(model.step(&input));
                            }
                            model.reset();
                        });
                    },
                );
            }
        }

        // Mamba
        #[cfg(feature = "mamba")]
        {
            let config = MambaConfig::new()
                .input_dim(1)
                .hidden_dim(128)
                .state_dim(16)
                .num_layers(2);
            if let Ok(mut model) = Mamba::new(config) {
                group.bench_with_input(
                    BenchmarkId::new("Mamba", batch_size),
                    &batch_size,
                    |b, _| {
                        b.iter(|| {
                            for &val in &signal {
                                let input = Array1::from_vec(vec![val]);
                                let _ = black_box(model.step(&input));
                            }
                            model.reset();
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: model_size_scaling
// ---------------------------------------------------------------------------

fn bench_model_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_size_scaling");
    group.sample_size(20);

    let input = Array1::from_vec(vec![0.5]);

    for &hidden in &[64usize, 128, 256] {
        // RWKV v7 scaling
        {
            let num_heads = (hidden / 32).max(2);
            let head_dim = hidden / num_heads;
            let config = Rwkv7Config {
                input_dim: 1,
                hidden_dim: hidden,
                num_layers: 2,
                num_heads,
                head_dim,
                expand_factor: 2.0,
                context_length: 512,
                time_decay_init: -5.0,
            };
            if let Ok(mut model) = Rwkv7Model::new(config) {
                group.bench_with_input(BenchmarkId::new("RWKV_v7", hidden), &hidden, |b, _| {
                    b.iter(|| {
                        let _ = black_box(model.step(&input));
                    });
                });
            }
        }

        // Mamba scaling
        #[cfg(feature = "mamba")]
        {
            let config = MambaConfig::new()
                .input_dim(1)
                .hidden_dim(hidden)
                .state_dim(16)
                .num_layers(2);
            if let Ok(mut model) = Mamba::new(config) {
                group.bench_with_input(BenchmarkId::new("Mamba", hidden), &hidden, |b, _| {
                    b.iter(|| {
                        let _ = black_box(model.step(&input));
                    });
                });
            }
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: sequence_length_scaling
// ---------------------------------------------------------------------------

fn bench_sequence_length_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequence_length_scaling");
    group.sample_size(10);

    for &seq_len in &[10usize, 100, 1000] {
        let signal: Vec<f32> = (0..seq_len).map(|t| (t as f32 * 0.05).sin()).collect();

        // RWKV v7
        {
            let config = Rwkv7Config {
                input_dim: 1,
                hidden_dim: 64,
                num_layers: 2,
                num_heads: 2,
                head_dim: 32,
                expand_factor: 2.0,
                context_length: 2048,
                time_decay_init: -5.0,
            };
            if let Ok(mut model) = Rwkv7Model::new(config) {
                group.bench_with_input(BenchmarkId::new("RWKV_v7", seq_len), &seq_len, |b, _| {
                    b.iter(|| {
                        for &val in &signal {
                            let input = Array1::from_vec(vec![val]);
                            let _ = black_box(model.step(&input));
                        }
                        model.reset();
                    });
                });
            }
        }

        // RWKV v6
        {
            let config = RwkvConfig::new()
                .input_dim(1)
                .hidden_dim(64)
                .num_heads(4)
                .num_layers(2);
            if let Ok(mut model) = Rwkv::new(config) {
                group.bench_with_input(BenchmarkId::new("RWKV_v6", seq_len), &seq_len, |b, _| {
                    b.iter(|| {
                        for &val in &signal {
                            let input = Array1::from_vec(vec![val]);
                            let _ = black_box(model.step(&input));
                        }
                        model.reset();
                    });
                });
            }
        }

        // S4D
        {
            let config = S4Config::new()
                .input_dim(1)
                .hidden_dim(64)
                .state_dim(16)
                .num_layers(2);
            if let Ok(mut model) = S4D::new(config) {
                group.bench_with_input(BenchmarkId::new("S4D", seq_len), &seq_len, |b, _| {
                    b.iter(|| {
                        for &val in &signal {
                            let input = Array1::from_vec(vec![val]);
                            let _ = black_box(model.step(&input));
                        }
                        model.reset();
                    });
                });
            }
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_per_step_latency,
    bench_batch_throughput,
    bench_model_size_scaling,
    bench_sequence_length_scaling
);
criterion_main!(benches);
