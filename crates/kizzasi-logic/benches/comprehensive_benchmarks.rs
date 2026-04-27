//! Comprehensive benchmarks for kizzasi-logic
//!
//! Benchmarks covering:
//! - Constraint checking and projection
//! - Penalty function computation
//! - Decomposition algorithms (ADMM, BCD)
//! - Guardrail validation
//! - Advanced constraints (chance, robust, CVaR)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_logic::*;
use scirs2_core::ndarray::Array1;
use std::hint::black_box;

// ============================================
// Basic Constraint Operations
// ============================================

fn bench_constraint_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraint_check");

    let constraint = ConstraintBuilder::new()
        .name("test")
        .less_than(10.0)
        .build()
        .unwrap();

    group.bench_function("single_value", |b| {
        b.iter(|| {
            let value = black_box(5.0f32);
            black_box(constraint.check(value))
        })
    });

    group.bench_function("violated_value", |b| {
        b.iter(|| {
            let value = black_box(15.0f32);
            black_box(constraint.check(value))
        })
    });

    group.finish();
}

fn bench_constraint_violation(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraint_violation");

    let constraint = ConstraintBuilder::new()
        .name("test")
        .less_than(10.0)
        .build()
        .unwrap();

    for size in [1, 10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let values: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
            b.iter(|| {
                for &val in &values {
                    black_box(constraint.violation(black_box(val)));
                }
            })
        });
    }

    group.finish();
}

fn bench_constraint_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraint_projection");

    let constraint = ConstraintBuilder::new()
        .name("test")
        .in_range(-10.0, 10.0)
        .build()
        .unwrap();

    group.bench_function("project_in_range", |b| {
        b.iter(|| {
            let value = black_box(5.0f32);
            black_box(constraint.project(value))
        })
    });

    group.bench_function("project_out_of_range", |b| {
        b.iter(|| {
            let value = black_box(20.0f32);
            black_box(constraint.project(value))
        })
    });

    group.finish();
}

// ============================================
// Penalty Functions
// ============================================

fn bench_penalty_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("penalty_functions");

    let violation = 2.5f32;
    let weight = 1.0f32;

    group.bench_function("l1", |b| {
        b.iter(|| {
            let v = black_box(violation);
            let w = black_box(weight);
            black_box(PenaltyFunction::L1.compute(v, w))
        })
    });

    group.bench_function("l2", |b| {
        b.iter(|| {
            let v = black_box(violation);
            let w = black_box(weight);
            black_box(PenaltyFunction::L2.compute(v, w))
        })
    });

    group.bench_function("huber", |b| {
        let penalty = PenaltyFunction::Huber { delta: 1.0 };
        b.iter(|| {
            let v = black_box(violation);
            let w = black_box(weight);
            black_box(penalty.compute(v, w))
        })
    });

    group.finish();
}

// ============================================
// Constraint-Aware Loss
// ============================================

fn bench_constraint_aware_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraint_aware_loss");

    for num_constraints in [1, 5, 10, 20].iter() {
        let constraints: Vec<_> = (0..*num_constraints)
            .map(|i| {
                ConstraintBuilder::new()
                    .name(&format!("c{}", i))
                    .less_than((i + 1) as f32 * 10.0)
                    .build()
                    .unwrap()
            })
            .collect();

        let loss_fn = ConstraintAwareLoss::new(constraints, PenaltyFunction::L2, 1.0);

        group.throughput(Throughput::Elements(*num_constraints as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_constraints),
            num_constraints,
            |b, _| {
                b.iter(|| {
                    let predictions = vec![5.0f32; *num_constraints];
                    let task_loss = 1.0f32;
                    black_box(loss_fn.compute_loss(&predictions, task_loss))
                })
            },
        );
    }

    group.finish();
}

// ============================================
// Guardrail System
// ============================================

fn bench_guardrail_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("guardrail_validation");

    for num_guards in [1, 5, 10, 20].iter() {
        let mut guardrail_set = GuardrailSet::new();
        for i in 0..*num_guards {
            let constraint = ConstraintBuilder::new()
                .name(&format!("guard{}", i))
                .less_than((i + 1) as f32 * 10.0)
                .build()
                .unwrap();
            guardrail_set.add_global(Guardrail::new(constraint, false));
        }

        group.throughput(Throughput::Elements(*num_guards as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_guards),
            num_guards,
            |b, _| {
                b.iter(|| {
                    let input = Array1::from_vec(vec![5.0f32; *num_guards]);
                    black_box(guardrail_set.validate(&input))
                })
            },
        );
    }

    group.finish();
}

// ============================================
// Sliding Window Constraints
// ============================================

fn bench_sliding_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("sliding_window");

    for window_size in [5, 10, 20, 50].iter() {
        let constraint = SlidingWindowConstraint::new(
            "test",
            *window_size,
            SlidingWindowFn::MeanInRange { lo: 0.0, hi: 10.0 },
        );

        let mut checker = SlidingWindowChecker::new();
        checker.add(constraint);

        group.throughput(Throughput::Elements(*window_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    for i in 0..*window_size {
                        black_box(checker.push_and_check(i as f32));
                    }
                })
            },
        );
    }

    group.finish();
}

// ============================================
// Differentiable Projection
// ============================================

fn bench_differentiable_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("differentiable_projection");

    for temp in [0.1, 1.0, 10.0].iter() {
        let proj = DifferentiableProjection::new(*temp);

        group.bench_with_input(BenchmarkId::from_parameter(temp), temp, |b, _| {
            b.iter(|| {
                let x = black_box(15.0f32);
                black_box(proj.soft_project_box(x, 0.0, 10.0))
            })
        });
    }

    group.finish();
}

// ============================================
// Consensus ADMM
// ============================================

fn bench_consensus_admm(c: &mut Criterion) {
    let mut group = c.benchmark_group("consensus_admm");

    for num_blocks in [2, 4, 8, 16].iter() {
        let config = ADMMConfig::default();
        let dimension = 10;
        let mut admm = ConsensusADMM::new(*num_blocks, dimension, config);

        let x0 = Array1::from_vec(vec![1.0; dimension]);
        admm.initialize(&x0);

        group.throughput(Throughput::Elements(*num_blocks as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_blocks),
            num_blocks,
            |b, _| {
                b.iter(|| {
                    // Simple quadratic local update
                    admm.iterate(|_block_id, _x_old, z_minus_u, rho| {
                        let target = Array1::from_elem(dimension, 2.0);
                        (&target + &(z_minus_u * rho)) / (1.0 + rho)
                    })
                })
            },
        );
    }

    group.finish();
}

// ============================================
// Block Coordinate Descent
// ============================================

fn bench_block_coordinate_descent(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_coordinate_descent");

    for block_size in [2, 5, 10, 20].iter() {
        let dimension = 100;
        let blocks = block_utils::uniform_blocks(dimension, *block_size);
        let mut bcd = BlockCoordinateDescent::new(blocks, dimension).unwrap();

        let x0 = Array1::from_vec(vec![1.0; dimension]);
        bcd.initialize(&x0);

        group.throughput(Throughput::Elements(*block_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(block_size),
            block_size,
            |b, _| {
                b.iter(|| {
                    bcd.update_block(0, |_full_x, indices| {
                        Array1::from_vec(vec![0.0; indices.len()])
                    })
                })
            },
        );
    }

    group.finish();
}

// ============================================
// Advanced Constraints
// ============================================

fn bench_chance_constraint(c: &mut Criterion) {
    let mut group = c.benchmark_group("chance_constraint");

    let chance = ChanceConstraint::gaussian("test", 0.95, 25.0, 3.0);

    group.bench_function("get_tightened_bound", |b| {
        b.iter(|| black_box(chance.get_tightened_bound()))
    });

    group.finish();
}

fn bench_cvar_constraint(c: &mut Criterion) {
    let mut group = c.benchmark_group("cvar_constraint");

    for num_samples in [100, 500, 1000, 5000].iter() {
        let cvar = CVaRConstraint::new("test", 0.05, 100.0, *num_samples);

        let losses: Vec<f32> = (0..*num_samples).map(|i| i as f32 * 0.1).collect();

        group.throughput(Throughput::Elements(*num_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_samples),
            num_samples,
            |b, _| b.iter(|| black_box(cvar.compute_cvar(&losses))),
        );
    }

    group.finish();
}

fn bench_robust_constraint(c: &mut Criterion) {
    let mut group = c.benchmark_group("robust_constraint");

    let robust = RobustConstraint::box_uncertain("test", vec![-5.0, -3.0], vec![5.0, 3.0]);

    group.bench_function("worst_case_scenario", |b| {
        b.iter(|| {
            let nominal = vec![10.0f32, 20.0];
            black_box(robust.worst_case_scenario(&nominal))
        })
    });

    group.finish();
}

// ============================================
// Composed Constraints
// ============================================

fn bench_composed_constraints(c: &mut Criterion) {
    let mut group = c.benchmark_group("composed_constraints");

    let c1 = ConstraintBuilder::new()
        .name("c1")
        .greater_than(0.0)
        .build()
        .unwrap();

    let c2 = ConstraintBuilder::new()
        .name("c2")
        .less_than(100.0)
        .build()
        .unwrap();

    let c3 = ConstraintBuilder::new()
        .name("c3")
        .less_than(50.0)
        .build()
        .unwrap();

    let composed_2 =
        ComposedConstraint::single(c1.clone()).and(ComposedConstraint::single(c2.clone()));

    let composed_3 = ComposedConstraint::single(c1.clone())
        .and(ComposedConstraint::single(c2.clone()))
        .and(ComposedConstraint::single(c3.clone()));

    group.bench_function("two_constraints", |b| {
        b.iter(|| {
            let value = black_box(25.0f32);
            black_box(composed_2.check(value))
        })
    });

    group.bench_function("three_constraints", |b| {
        b.iter(|| {
            let value = black_box(25.0f32);
            black_box(composed_3.check(value))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_constraint_check,
    bench_constraint_violation,
    bench_constraint_projection,
    bench_penalty_functions,
    bench_constraint_aware_loss,
    bench_guardrail_validation,
    bench_sliding_window,
    bench_differentiable_projection,
    bench_consensus_admm,
    bench_block_coordinate_descent,
    bench_chance_constraint,
    bench_cvar_constraint,
    bench_robust_constraint,
    bench_composed_constraints,
);

criterion_main!(benches);
