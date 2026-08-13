//! Benchmarks for projection algorithms
//!
//! Run with: cargo bench -p kizzasi-logic

use kizzasi_logic::{
    ConstraintBuilder, DykstraProjection, GeometricSet, GradientProjection, LinearConstraint,
    NonlinearConstraint, SetMembershipConstraint,
};
use scirs2_core::ndarray::Array1;

/// Benchmark simple projection
pub fn bench_simple_projection() {
    let constraint = ConstraintBuilder::new()
        .name("bound")
        .in_range(-10.0, 10.0)
        .build()
        .unwrap();

    let input = [15.0, -20.0, 5.0];

    // Warm up
    for _ in 0..100 {
        let _ = constraint.project(input[0]);
    }

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..10000 {
        let _ = constraint.project(input[0]);
    }
    let duration = start.elapsed();

    println!("Simple projection: {:?} per iteration", duration / 10000);
}

/// Benchmark Dykstra's algorithm
pub fn bench_dykstra_projection() {
    let c1 = LinearConstraint::less_eq(vec![1.0, 0.0], 5.0);
    let c2 = LinearConstraint::greater_eq(vec![1.0, 0.0], -5.0);
    let c3 = LinearConstraint::less_eq(vec![0.0, 1.0], 5.0);

    let dykstra = DykstraProjection::new(vec![c1, c2, c3])
        .with_tolerance(1e-6)
        .with_max_iterations(100);

    let input = Array1::from_vec(vec![10.0, 10.0]);

    // Warm up
    for _ in 0..10 {
        let _ = dykstra.project(&input);
    }

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = dykstra.project(&input);
    }
    let duration = start.elapsed();

    println!(
        "Dykstra projection (3 constraints): {:?} per iteration",
        duration / 1000
    );
}

/// Benchmark gradient-based projection
pub fn bench_gradient_projection() {
    let constraint =
        NonlinearConstraint::inequality("circle", |x: &[f32]| x[0] * x[0] + x[1] * x[1] - 1.0)
            .with_gradient(|x: &[f32]| vec![2.0 * x[0], 2.0 * x[1]]);

    let proj = GradientProjection::new()
        .with_max_iterations(100)
        .with_step_size(0.1);

    let input = Array1::from_vec(vec![2.0, 2.0]);

    // Warm up
    for _ in 0..10 {
        let _ = proj.project(&input, std::slice::from_ref(&constraint));
    }

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = proj.project(&input, std::slice::from_ref(&constraint));
    }
    let duration = start.elapsed();

    println!(
        "Gradient projection (nonlinear): {:?} per iteration",
        duration / 1000
    );
}

/// Benchmark set membership projection
pub fn bench_set_projection() {
    let ball = GeometricSet::ball(vec![0.0, 0.0], 1.0).expect("valid ball");
    let constraint = SetMembershipConstraint::new("unit_ball", ball);

    let input = vec![2.0, 2.0];

    // Warm up
    for _ in 0..100 {
        let _ = constraint.project(&input);
    }

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..10000 {
        let _ = constraint.project(&input);
    }
    let duration = start.elapsed();

    println!(
        "Set projection (ball): {:?} per iteration",
        duration / 10000
    );
}

/// Benchmark batch constraint checking
pub fn bench_batch_checking() {
    use kizzasi_logic::BatchConstraintChecker;
    use scirs2_core::ndarray::Array2;

    let c1 = ConstraintBuilder::new()
        .name("lower")
        .greater_eq(-10.0)
        .build()
        .unwrap();

    let c2 = ConstraintBuilder::new()
        .name("upper")
        .less_eq(10.0)
        .build()
        .unwrap();

    let mut checker = BatchConstraintChecker::new(vec![c1, c2]);

    // Create batch of 1000 points
    let mut points_vec = Vec::with_capacity(1000);
    for i in 0..1000 {
        points_vec.push((i as f32 - 500.0) / 50.0);
    }
    let points = Array2::from_shape_vec((1000, 1), points_vec).unwrap();

    // Warm up
    for _ in 0..10 {
        let _ = checker.check_batch(&points);
    }

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = checker.check_batch(&points);
    }
    let duration = start.elapsed();

    println!(
        "Batch checking (1000 points, 2 constraints): {:?} per batch",
        duration / 100
    );
}

/// Run all benchmarks
fn main() {
    println!("Running projection algorithm benchmarks...\n");

    bench_simple_projection();
    bench_dykstra_projection();
    bench_gradient_projection();
    bench_set_projection();
    bench_batch_checking();

    println!("\nBenchmarks complete!");
}
