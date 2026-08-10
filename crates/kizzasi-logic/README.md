# kizzasi-logic

Constraint enforcement and safety guardrails for Kizzasi AGSP.

## Overview

TensorLogic bridge providing constraint satisfaction, optimization, and safety guarantees for signal prediction. Ensures predictions satisfy physical laws, safety bounds, and domain constraints.

## Features

- **Constraint Types**: Linear, quadratic, nonlinear, temporal, geometric
- **Temporal Logic**: LTL and STL for time-series properties
- **Projection Methods**: Gradient-based, Dykstra's alternating, QP solvers
- **Training Integration**: Differentiable projections, Lagrangian relaxation
- **Optimization**: MPC, Benders decomposition, multi-objective
- **GPU Acceleration**: Parallel constraint checking
- **Incremental Solving**: Real-time constraint updates
- **TensorLogic-IR integration**: compile symbolic `TLExpr` expressions into executable constraints for use with `kizzasi-model`'s constraint bridge

## Quick Start

```rust
use kizzasi_logic::{RangeConstraint, GuardrailSet, ConstraintChecker};

// Define safety bounds
let position_limit = RangeConstraint::new(0, -10.0, 10.0)?; // dim, min, max
let velocity_limit = RangeConstraint::new(1, -5.0, 5.0)?;

let mut guardrails = GuardrailSet::new();
guardrails.add(Box::new(position_limit));
guardrails.add(Box::new(velocity_limit));

// Check and project predictions
let prediction = Array1::from_vec(vec![12.0, 3.0]); // Violates position limit
let safe_prediction = guardrails.project(&prediction)?; // Projects to [10.0, 3.0]
```

## Constraint Solving

- Projection algorithms: <10μs for simple constraints
- Batch processing: 1M constraint checks/sec
- GPU acceleration: 10x speedup for large batches
- 307 comprehensive tests, all passing

## Documentation

- [API Documentation](https://docs.rs/kizzasi-logic)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
