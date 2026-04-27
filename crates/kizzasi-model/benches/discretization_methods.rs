//! Discretization method comparison benchmarks for kizzasi-model.
//!
//! Benchmarks ZOH (diagonal), Bilinear (Tustin), and Forward Euler
//! discretization across state_dim ∈ {16, 64, 256} with fixed dt=0.01.
//! Correctness check (L∞ error vs exact matrix-exponential ground truth
//! for diagonal A) is printed once at registration time, outside the timed
//! section.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_core::numerics::{
    bilinear_discretize, forward_euler_discretize, zoh_discretize_diagonal,
};
use scirs2_core::ndarray::Array2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate negative-real diagonal elements using a LCG for reproducibility.
fn make_diagonal_a(state_dim: usize, seed: u64) -> scirs2_core::ndarray::Array1<f32> {
    let mut rng_state = seed;
    scirs2_core::ndarray::Array1::from_iter((0..state_dim).map(|_| {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let x = ((rng_state >> 33) as f32) / (u32::MAX as f32); // [0, 1)
        -(0.1 + x * 1.9) // negative, in [-2.0, -0.1]
    }))
}

/// Generate a B matrix using a LCG for reproducibility.
fn make_b(state_dim: usize, in_dim: usize, seed: u64) -> Array2<f32> {
    let mut rng_state = seed.wrapping_add(99);
    Array2::from_shape_fn((state_dim, in_dim), |_| {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((rng_state >> 33) as f32) / (u32::MAX as f32) - 0.5
    })
}

// ---------------------------------------------------------------------------
// Benchmark group
// ---------------------------------------------------------------------------

fn bench_discretization_methods(c: &mut Criterion) {
    const STATE_DIMS: [usize; 3] = [16, 64, 256];
    const DT: f32 = 0.01;

    let mut group = c.benchmark_group("discretization_methods");

    for &state_dim in &STATE_DIMS {
        let a = make_diagonal_a(state_dim, 42);
        let b = make_b(state_dim, 4, 42);

        // Correctness check (not timed): L∞ error vs exact expm ground truth.
        // For diagonal A, expm(dt*A) = diag(exp(dt*a_i)) — this is what ZOH gives exactly.
        let (a_zoh, _) = zoh_discretize_diagonal(&a, &b, DT);
        let (a_bil, _) = bilinear_discretize(&a, &b, DT);
        let (a_fe, _) = forward_euler_discretize(&a, &b, DT);

        let linf_zoh: f32 = a_zoh
            .iter()
            .zip(a.iter())
            .map(|(az, ai)| (az - (DT * ai).exp()).abs())
            .fold(0.0f32, f32::max);
        let linf_bil: f32 = a_bil
            .iter()
            .zip(a.iter())
            .map(|(ab, ai)| (ab - (DT * ai).exp()).abs())
            .fold(0.0f32, f32::max);
        let linf_fe: f32 = a_fe
            .iter()
            .zip(a.iter())
            .map(|(af, ai)| (af - (DT * ai).exp()).abs())
            .fold(0.0f32, f32::max);

        eprintln!(
            "state_dim={state_dim}: L\u{221e}_err zoh={linf_zoh:.2e} bilinear={linf_bil:.2e} fe={linf_fe:.2e}"
        );

        group.throughput(Throughput::Elements(state_dim as u64));

        group.bench_with_input(
            BenchmarkId::new("ZOH", state_dim),
            &state_dim,
            |bench, &_sd| {
                let a = a.clone();
                let b = b.clone();
                bench.iter(|| std::hint::black_box(zoh_discretize_diagonal(&a, &b, DT)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Bilinear", state_dim),
            &state_dim,
            |bench, &_sd| {
                let a = a.clone();
                let b = b.clone();
                bench.iter(|| std::hint::black_box(bilinear_discretize(&a, &b, DT)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ForwardEuler", state_dim),
            &state_dim,
            |bench, &_sd| {
                let a = a.clone();
                let b = b.clone();
                bench.iter(|| std::hint::black_box(forward_euler_discretize(&a, &b, DT)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_discretization_methods);
criterion_main!(benches);

// ---------------------------------------------------------------------------
// Unit tests (included in bench binary via #[cfg(test)])
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(dead_code, unused_imports)]
mod tests {
    use kizzasi_core::numerics::{
        bilinear_discretize, discretize, forward_euler_discretize, zoh_discretize_diagonal,
        DiscretizationMethod,
    };
    use scirs2_core::ndarray::Array2;

    fn make_diagonal_a(state_dim: usize, seed: u64) -> scirs2_core::ndarray::Array1<f32> {
        let mut rng_state = seed;
        scirs2_core::ndarray::Array1::from_iter((0..state_dim).map(|_| {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let x = ((rng_state >> 33) as f32) / (u32::MAX as f32);
            -(0.1 + x * 1.9)
        }))
    }

    fn make_b(state_dim: usize, in_dim: usize, seed: u64) -> Array2<f32> {
        let mut rng_state = seed.wrapping_add(99);
        Array2::from_shape_fn((state_dim, in_dim), |_| {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((rng_state >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
    }

    #[test]
    fn discretization_methods_numerical_parity() {
        let state_dim = 16;
        let dt = 0.01f32;
        let a = make_diagonal_a(state_dim, 123);
        let b = make_b(state_dim, 4, 123);

        // ZOH via dispatch must match direct call
        let (a_zoh, b_zoh) = zoh_discretize_diagonal(&a, &b, dt);
        let (a_zoh2, b_zoh2) = discretize(DiscretizationMethod::Zoh, &a, &b, dt);

        for (x, y) in a_zoh.iter().zip(a_zoh2.iter()) {
            assert!((x - y).abs() < 1e-7, "ZOH A mismatch: {x} vs {y}");
        }
        for (x, y) in b_zoh.iter().zip(b_zoh2.iter()) {
            assert!((x - y).abs() < 1e-7, "ZOH B mismatch: {x} vs {y}");
        }

        // Bilinear stability: all |a_bar| < 1 for negative-real a
        let (a_bil, _) = bilinear_discretize(&a, &b, dt);
        for &x in a_bil.iter() {
            assert!(x.abs() < 1.0, "bilinear stability violated: {x}");
        }

        // Forward Euler O(dt^2) approximation to ZOH-diagonal at small dt
        let (a_fe, _) = forward_euler_discretize(&a, &b, dt);
        let max_err: f32 = a_fe
            .iter()
            .zip(a_zoh.iter())
            .map(|(f, z)| (f - z).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-3,
            "FE vs ZOH error {max_err} too large at dt={dt}"
        );
    }
}
