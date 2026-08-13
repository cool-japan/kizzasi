# kizzasi-logic

Constraint enforcement and safety guardrails for Kizzasi AGSP.

**Status**: Stable — 1053 `pub` items, 398/398 tests passing.

## Overview

TensorLogic bridge providing constraint satisfaction, optimization, and safety guarantees for signal prediction. Ensures predictions satisfy physical laws, safety bounds, and domain constraints.

## Features

- **Constraint Types**: Linear, quadratic, nonlinear, temporal, geometric
- **Temporal Logic**: LTL and STL for time-series properties
- **Constraint Repair**: Infeasibility diagnosis (IIS), conflict resolution, minimal relaxation
- **Projection Methods**: Gradient-based, Dykstra's alternating, and a pure-Rust two-phase simplex LP solver (`lp_simplex`, always available); the OSQP quadratic-program backend is behind the opt-in `qp-solver` feature, which is **not pure Rust** — the `osqp` crate builds the OSQP C sources
- **Training Integration**: Differentiable projections, Lagrangian relaxation
- **Optimization**: MPC (hard-projected control constraints), Benders decomposition (real two-phase-simplex subproblem and cut generation; the master is solved as a continuous LP relaxation over its box bounds — integrality would need branch-and-bound), multi-objective, distributed/consensus ADMM
- **Online Learning**: Streaming constraint discovery, active boundary learning, feedback-driven tuning
- **Batch Checking & Projection**: `GPUConstraintChecker`, `GPUProjector`, `GPUGradientComputer` — batch-oriented APIs named for a future GPU backend; no such backend is implemented yet, `is_gpu_available()` always returns `false`, and every batch call always runs on CPU
- **Incremental Solving**: Real-time constraint updates
- **TensorLogic-IR integration**: compile symbolic `TLExpr` expressions into executable constraints for use with `kizzasi-model`'s constraint bridge

## Quick Start

```rust
use kizzasi_logic::{Array1, ConstrainedInference, ConstraintBuilder, Guardrail, GuardrailSet};

// Define a safety bound: every value must stay in [-10.0, 10.0]
let position_limit = ConstraintBuilder::new()
    .name("position_limit")
    .in_range(-10.0, 10.0)
    .build()?;

let mut guardrails = GuardrailSet::new();
guardrails.add_global(Guardrail::new(position_limit, false)); // false = soft-project on violation

// Check and project predictions
let prediction = Array1::from_vec(vec![12.0, 3.0]); // dim 0 violates the bound
let safe_prediction = guardrails.constrain(&prediction)?; // -> [10.0, 3.0]
```

`kizzasi-logic` also compiles symbolic `TLExpr` expressions (from `tensorlogic-ir`) directly
into executable constraints via `TlExprCompiler`:

```rust
use kizzasi_logic::{Array1, TLExpr, TlExprCompiler};

// Build the constraint x[0] <= 1.0 from a symbolic TLExpr
let expr = TLExpr::Lte(
    Box::new(TLExpr::Pred { name: "dim_0".into(), args: vec![] }),
    Box::new(TLExpr::Pred { name: "1.0".into(), args: vec![] }),
);
let compiled = TlExprCompiler::new().compile(&expr, "x_le_1", 1)?;

assert!(compiled.evaluate(&Array1::from_vec(vec![0.5_f32]))?);
assert!(!compiled.evaluate(&Array1::from_vec(vec![2.0_f32]))?);
```

## Constraint Solving

- Projection algorithms: <10μs for simple constraints
- Batch processing: 1M constraint checks/sec
- 398 comprehensive tests, all passing

## Documentation

- [API Documentation](https://docs.rs/kizzasi-logic)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
