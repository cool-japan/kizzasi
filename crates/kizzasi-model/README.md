# kizzasi-model

Model architectures for Kizzasi AGSP - Mamba, RWKV, S4, Transformer.

## Overview

Production-ready implementations of state-of-the-art sequence models with unified interfaces. All models support O(1) recurrent inference for streaming applications.

## Features

- **Mamba & Mamba2**: Selective state space models with SSD
- **RWKV v5/v6/v7**: Receptance Weighted Key Value architecture
- **S4/S4D/S5**: Structured state space models with HiPPO initialization
- **H3**: Hungry Hungry Hippos with shift SSMs
- **Transformer**: KV-cache optimized attention
- **Hybrid**: Combined Mamba + Attention architectures
- **MoE**: Mixture of Experts layer with routing strategies

## Quick Start

```rust
use kizzasi_model::{Mamba, MambaConfig, AutoregressiveModel};

// Create Mamba model
let config = MambaConfig::base(32, 64); // input_dim, hidden_dim
let mut model = Mamba::new(config)?;

// Single-step inference
let input = Array1::zeros(32);
let output = model.forward(&input)?;

// Or use presets
let tiny_model = Mamba::tiny(32, 32);  // For edge devices
let large_model = Mamba::large(64, 1024); // High accuracy
```

## Supported Models

| Model | Complexity | Memory | Best For |
|-------|------------|--------|----------|
| Mamba2 | O(1) | Low | Real-time streaming |
| RWKV | O(1) | Very Low | Long sequences |
| S4D | O(1) | Low | Continuous signals |
| Transformer | O(n²) | High | Short contexts |
| Hybrid | O(n) | Medium | Balanced performance |

## TensorLogic-IR Integration

v0.2.2 adds symbolic constraint compilation via TensorLogic-IR expressions. Constraints built with `kizzasi-logic`'s `TLExpr` can be compiled to executable `CompiledConstraint` objects for model output validation.

```rust
use kizzasi_model::tensorlogic_bridge::{constraint_from_tl_expr, compile_constraints};
use kizzasi_logic::TLExpr;

// Compile a single constraint from a TLExpr
let expr = TLExpr::bounded("logits", 0.0, 1.0); // values in [0, 1]
let constraint = constraint_from_tl_expr("output_range", &expr, 2)?;

// Evaluate the compiled constraint against model output
let output = model.forward(&input)?;
constraint.evaluate(&output)?;

// Batch-compile multiple constraints at once
let specs = vec![
    ("output_range",  TLExpr::bounded("logits", 0.0, 1.0), 2),
    ("seq_len_bound", TLExpr::max_dim("seq", 512),          1),
];
let constraints = compile_constraints(&specs)?;
for c in &constraints {
    c.evaluate(&output)?;
}
```

`constraint_from_tl_expr(name, expr, num_dims)` compiles a single named expression into a `CompiledConstraint`. `compile_constraints` accepts a slice of `(name, TLExpr, num_dims)` tuples and returns all compiled constraints in one call, which is more efficient for validating several invariants simultaneously.

## Mamba SSM Forward Pass

```mermaid
flowchart LR
    X["x_t (input)"] --> PROJ["Linear Projection"]
    PROJ --> DELTA["Δ (timescale)"]
    PROJ --> B["B (input gate)"]
    PROJ --> C["C (output gate)"]
    DELTA --> DISC["ZOH Discretization\nA_bar, B_bar"]
    H_PREV["h_{t-1} (state)"] --> SSM
    DISC --> SSM["SSM Recurrence\nh_t = A_bar·h + B_bar·x"]
    B --> SSM
    SSM --> H_NEXT["h_t (new state)"]
    C --> OUT["Output\ny_t = C·h_t"]
    SSM --> OUT
```

## Documentation

- [API Documentation](https://docs.rs/kizzasi-model)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
