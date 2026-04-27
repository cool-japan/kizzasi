//! State-dimension sweep benchmarks for kizzasi-model.
//!
//! Sweeps state_dim ∈ {8, 16, 32, 64, 128} at fixed hidden_dim=128 and
//! input_dim=4, measuring single-step inference latency for Mamba, Mamba2,
//! S4D, and S5.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_core::SignalPredictor;
use scirs2_core::ndarray::Array1;
use std::hint::black_box;

#[cfg(feature = "mamba")]
use kizzasi_model::mamba::{Mamba, MambaConfig};
#[cfg(feature = "mamba")]
use kizzasi_model::mamba2::{Mamba2, Mamba2Config};
use kizzasi_model::s4::{S4Config, S4D};
use kizzasi_model::s5::{S5Config, S5};

const INPUT_DIM: usize = 4;
const HIDDEN_DIM: usize = 128;
const NUM_LAYERS: usize = 4;
const STATE_DIMS: [usize; 5] = [8, 16, 32, 64, 128];

// ---------------------------------------------------------------------------
// Sweep group
// ---------------------------------------------------------------------------

fn bench_state_dim_sweep(c: &mut Criterion) {
    let input = Array1::from_vec(vec![0.5f32; INPUT_DIM]);
    let mut group = c.benchmark_group("state_dim_sweep");
    group.sample_size(20);

    for &state_dim in &STATE_DIMS {
        group.throughput(Throughput::Elements(state_dim as u64));

        // Mamba
        #[cfg(feature = "mamba")]
        {
            let config = MambaConfig::new()
                .input_dim(INPUT_DIM)
                .hidden_dim(HIDDEN_DIM)
                .state_dim(state_dim)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = Mamba::new(config) {
                group.bench_with_input(
                    BenchmarkId::new("Mamba", state_dim),
                    &state_dim,
                    |b, &_sd| {
                        b.iter(|| {
                            let _ = black_box(model.step(&input));
                        });
                    },
                );
            }
        }

        // Mamba2 — num_heads must divide hidden_dim; use 8 (128/8=16)
        #[cfg(feature = "mamba")]
        {
            let config = Mamba2Config::new()
                .input_dim(INPUT_DIM)
                .hidden_dim(HIDDEN_DIM)
                .state_dim(state_dim)
                .num_heads(8)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = Mamba2::new(config) {
                group.bench_with_input(
                    BenchmarkId::new("Mamba2", state_dim),
                    &state_dim,
                    |b, &_sd| {
                        b.iter(|| {
                            let _ = black_box(model.step(&input));
                        });
                    },
                );
            }
        }

        // S4D
        {
            let config = S4Config::new()
                .input_dim(INPUT_DIM)
                .hidden_dim(HIDDEN_DIM)
                .state_dim(state_dim)
                .num_layers(NUM_LAYERS);
            if let Ok(mut model) = S4D::new(config) {
                group.bench_with_input(
                    BenchmarkId::new("S4D", state_dim),
                    &state_dim,
                    |b, &_sd| {
                        b.iter(|| {
                            let _ = black_box(model.step(&input));
                        });
                    },
                );
            }
        }

        // S5 — uses positional constructor; state_dim set via field
        {
            let mut config = S5Config::new(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS);
            config.state_dim = state_dim;
            if let Ok(mut model) = S5::new(config) {
                group.bench_with_input(BenchmarkId::new("S5", state_dim), &state_dim, |b, &_sd| {
                    b.iter(|| {
                        let _ = black_box(model.step(&input));
                    });
                });
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_state_dim_sweep);
criterion_main!(benches);

// ---------------------------------------------------------------------------
// Smoke tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(dead_code, unused_imports)]
mod tests {
    use kizzasi_core::SignalPredictor;
    use scirs2_core::ndarray::Array1;

    #[cfg(feature = "mamba")]
    use kizzasi_model::mamba::{Mamba, MambaConfig};
    #[cfg(feature = "mamba")]
    use kizzasi_model::mamba2::{Mamba2, Mamba2Config};
    use kizzasi_model::s4::{S4Config, S4D};
    use kizzasi_model::s5::{S5Config, S5};

    const INPUT_DIM: usize = 4;
    const HIDDEN_DIM: usize = 128;
    const NUM_LAYERS: usize = 4;
    const SMOKE_STATE_DIM: usize = 16;

    #[cfg(feature = "mamba")]
    #[test]
    fn smoke_mamba_state_dim_step() {
        let input = Array1::from_vec(vec![0.5f32; INPUT_DIM]);
        let config = MambaConfig::new()
            .input_dim(INPUT_DIM)
            .hidden_dim(HIDDEN_DIM)
            .state_dim(SMOKE_STATE_DIM)
            .num_layers(NUM_LAYERS);
        let mut model = Mamba::new(config).expect("Mamba config should be valid");
        let out = model.step(&input).expect("Mamba step should succeed");
        assert_eq!(out.len(), INPUT_DIM, "Mamba output dim mismatch");
    }

    #[cfg(feature = "mamba")]
    #[test]
    fn smoke_mamba2_state_dim_step() {
        let input = Array1::from_vec(vec![0.5f32; INPUT_DIM]);
        let config = Mamba2Config::new()
            .input_dim(INPUT_DIM)
            .hidden_dim(HIDDEN_DIM)
            .state_dim(SMOKE_STATE_DIM)
            .num_heads(8)
            .num_layers(NUM_LAYERS);
        let mut model = Mamba2::new(config).expect("Mamba2 config should be valid");
        let out = model.step(&input).expect("Mamba2 step should succeed");
        assert_eq!(out.len(), INPUT_DIM, "Mamba2 output dim mismatch");
    }

    #[test]
    fn smoke_s4d_state_dim_step() {
        let input = Array1::from_vec(vec![0.5f32; INPUT_DIM]);
        let config = S4Config::new()
            .input_dim(INPUT_DIM)
            .hidden_dim(HIDDEN_DIM)
            .state_dim(SMOKE_STATE_DIM)
            .num_layers(NUM_LAYERS);
        let mut model = S4D::new(config).expect("S4D config should be valid");
        let out = model.step(&input).expect("S4D step should succeed");
        assert_eq!(out.len(), INPUT_DIM, "S4D output dim mismatch");
    }

    #[test]
    fn smoke_s5_state_dim_step() {
        let input = Array1::from_vec(vec![0.5f32; INPUT_DIM]);
        let mut config = S5Config::new(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS);
        config.state_dim = SMOKE_STATE_DIM;
        let mut model = S5::new(config).expect("S5 config should be valid");
        let out = model.step(&input).expect("S5 step should succeed");
        assert_eq!(out.len(), INPUT_DIM, "S5 output dim mismatch");
    }
}
