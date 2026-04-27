//! Architecture comparison benchmarks for kizzasi-model
//!
//! Compares all supported model architectures across:
//! 1. Single-step latency at various hidden dimensions
//! 2. Hundred-step sequential processing
//! 3. Memory footprint (struct size)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kizzasi_core::SignalPredictor;
use scirs2_core::ndarray::Array1;
use std::hint::black_box;

#[cfg(feature = "mamba")]
use kizzasi_model::mamba::{Mamba, MambaConfig};
#[cfg(feature = "mamba")]
use kizzasi_model::mamba2::{Mamba2, Mamba2Config};
use kizzasi_model::rwkv::{Rwkv, RwkvConfig};
use kizzasi_model::s4::{S4Config, S4D};
use kizzasi_model::transformer::{Transformer, TransformerConfig};

const INPUT_DIM: usize = 4;
const NUM_LAYERS: usize = 4;
const HIDDEN_DIMS: [usize; 4] = [64, 128, 256, 512];

// ---------------------------------------------------------------------------
// Group 1: single_step_d{dim}
// ---------------------------------------------------------------------------

fn bench_single_step(c: &mut Criterion) {
    let input = Array1::from_vec(vec![0.5; INPUT_DIM]);

    for &hidden in &HIDDEN_DIMS {
        let group_name = format!("single_step_d{}", hidden);
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(20);

        // Mamba
        #[cfg(feature = "mamba")]
        {
            let config = MambaConfig::new()
                .input_dim(INPUT_DIM)
                .hidden_dim(hidden)
                .state_dim(16)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = Mamba::new(config) {
                group.bench_with_input(BenchmarkId::new("Mamba", hidden), &hidden, |b, _| {
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
                .input_dim(INPUT_DIM)
                .hidden_dim(hidden)
                .num_heads(8)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = Mamba2::new(config) {
                group.bench_with_input(BenchmarkId::new("Mamba2", hidden), &hidden, |b, _| {
                    b.iter(|| {
                        let _ = black_box(model.step(&input));
                    });
                });
            }
        }

        // RWKV v6
        {
            let config = RwkvConfig::new()
                .input_dim(INPUT_DIM)
                .hidden_dim(hidden)
                .num_heads(8)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = Rwkv::new(config) {
                group.bench_with_input(BenchmarkId::new("RWKV_v6", hidden), &hidden, |b, _| {
                    b.iter(|| {
                        let _ = black_box(model.step(&input));
                    });
                });
            }
        }

        // S4D
        {
            let config = S4Config::new()
                .input_dim(INPUT_DIM)
                .hidden_dim(hidden)
                .state_dim(16)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = S4D::new(config) {
                group.bench_with_input(BenchmarkId::new("S4D", hidden), &hidden, |b, _| {
                    b.iter(|| {
                        let _ = black_box(model.step(&input));
                    });
                });
            }
        }

        // Transformer
        {
            let config = TransformerConfig::new()
                .input_dim(INPUT_DIM)
                .hidden_dim(hidden)
                .num_heads(8)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = Transformer::new(config) {
                group.bench_with_input(BenchmarkId::new("Transformer", hidden), &hidden, |b, _| {
                    b.iter(|| {
                        let _ = black_box(model.step(&input));
                    });
                });
            }
        }

        group.finish();
    }
}

// ---------------------------------------------------------------------------
// Group 2: hundred_steps_d{dim}
// ---------------------------------------------------------------------------

fn bench_hundred_steps(c: &mut Criterion) {
    let input = Array1::from_vec(vec![0.5; INPUT_DIM]);

    for &hidden in &HIDDEN_DIMS {
        let group_name = format!("hundred_steps_d{}", hidden);
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(20);

        // Mamba
        #[cfg(feature = "mamba")]
        {
            let config = MambaConfig::new()
                .input_dim(INPUT_DIM)
                .hidden_dim(hidden)
                .state_dim(16)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = Mamba::new(config) {
                group.bench_with_input(BenchmarkId::new("Mamba", hidden), &hidden, |b, _| {
                    b.iter(|| {
                        for _ in 0..100 {
                            let _ = black_box(model.step(&input));
                        }
                        model.reset();
                    });
                });
            }
        }

        // Mamba2
        #[cfg(feature = "mamba")]
        {
            let config = Mamba2Config::new()
                .input_dim(INPUT_DIM)
                .hidden_dim(hidden)
                .num_heads(8)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = Mamba2::new(config) {
                group.bench_with_input(BenchmarkId::new("Mamba2", hidden), &hidden, |b, _| {
                    b.iter(|| {
                        for _ in 0..100 {
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
                .input_dim(INPUT_DIM)
                .hidden_dim(hidden)
                .num_heads(8)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = Rwkv::new(config) {
                group.bench_with_input(BenchmarkId::new("RWKV_v6", hidden), &hidden, |b, _| {
                    b.iter(|| {
                        for _ in 0..100 {
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
                .input_dim(INPUT_DIM)
                .hidden_dim(hidden)
                .state_dim(16)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = S4D::new(config) {
                group.bench_with_input(BenchmarkId::new("S4D", hidden), &hidden, |b, _| {
                    b.iter(|| {
                        for _ in 0..100 {
                            let _ = black_box(model.step(&input));
                        }
                        model.reset();
                    });
                });
            }
        }

        // Transformer
        {
            let config = TransformerConfig::new()
                .input_dim(INPUT_DIM)
                .hidden_dim(hidden)
                .num_heads(8)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = Transformer::new(config) {
                group.bench_with_input(BenchmarkId::new("Transformer", hidden), &hidden, |b, _| {
                    b.iter(|| {
                        for _ in 0..100 {
                            let _ = black_box(model.step(&input));
                        }
                        model.reset();
                    });
                });
            }
        }

        group.finish();
    }
}

// ---------------------------------------------------------------------------
// Group 3: memory_footprint
// ---------------------------------------------------------------------------

fn bench_memory_footprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_footprint");
    group.sample_size(10);

    let hidden: usize = 256;

    // Mamba
    #[cfg(feature = "mamba")]
    {
        let config = MambaConfig::new()
            .input_dim(INPUT_DIM)
            .hidden_dim(hidden)
            .state_dim(16)
            .num_layers(NUM_LAYERS);
        if let Ok(model) = Mamba::new(config) {
            group.bench_function("Mamba", |b| {
                b.iter(|| black_box(std::mem::size_of_val(&model)));
            });
        }
    }

    // Mamba2
    #[cfg(feature = "mamba")]
    {
        let config = Mamba2Config::new()
            .input_dim(INPUT_DIM)
            .hidden_dim(hidden)
            .num_heads(8)
            .num_layers(NUM_LAYERS);
        if let Ok(model) = Mamba2::new(config) {
            group.bench_function("Mamba2", |b| {
                b.iter(|| black_box(std::mem::size_of_val(&model)));
            });
        }
    }

    // RWKV v6
    {
        let config = RwkvConfig::new()
            .input_dim(INPUT_DIM)
            .hidden_dim(hidden)
            .num_heads(8)
            .num_layers(NUM_LAYERS);
        if let Ok(model) = Rwkv::new(config) {
            group.bench_function("RWKV_v6", |b| {
                b.iter(|| black_box(std::mem::size_of_val(&model)));
            });
        }
    }

    // S4D
    {
        let config = S4Config::new()
            .input_dim(INPUT_DIM)
            .hidden_dim(hidden)
            .state_dim(16)
            .num_layers(NUM_LAYERS);
        if let Ok(model) = S4D::new(config) {
            group.bench_function("S4D", |b| {
                b.iter(|| black_box(std::mem::size_of_val(&model)));
            });
        }
    }

    // Transformer
    {
        let config = TransformerConfig::new()
            .input_dim(INPUT_DIM)
            .hidden_dim(hidden)
            .num_heads(8)
            .num_layers(NUM_LAYERS);
        if let Ok(model) = Transformer::new(config) {
            group.bench_function("Transformer", |b| {
                b.iter(|| black_box(std::mem::size_of_val(&model)));
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_step,
    bench_hundred_steps,
    bench_memory_footprint
);
criterion_main!(benches);
