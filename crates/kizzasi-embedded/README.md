# kizzasi-embedded

Embedded/edge inference for Kizzasi AGSP State Space Models on `no_std` targets.

## Overview

Bare-metal SSM inference engine with O(1) per-step complexity, fixed-point arithmetic support, and zero heap allocation. Designed for microcontrollers, FPGAs, and other resource-constrained devices where neither `std` nor a heap allocator may be available.

## Features

- **`no_std` Compatible**: Runs on bare-metal targets with no operating system
- **O(1) Inference**: Constant-time single-step state update, suitable for real-time loops
- **Fixed-Point Arithmetic**: Optional `fixed-point` feature for targets without FPU
- **Alloc-Free Operation**: Core inference path requires no heap allocations
- **`alloc` Feature**: Opt-in dynamic allocation for larger model weights when a heap is available
- **libm Integration**: Portable math via `libm` on targets lacking native float intrinsics
- **Minimal Dependencies**: Only `libm` as an optional dependency; no OS services required

## Feature Flags

| Feature | Default | Description |
|---|---|---|
| `std` | yes | Enable standard library support |
| `fixed-point` | no | Fixed-point arithmetic for FPU-less targets |
| `alloc` | no | Enable `alloc` crate for dynamic weight storage |

## Basic Usage

```rust
#![no_std]

use kizzasi_embedded::ssm::{SsmConfig, SsmState};

// Statically allocated config (no heap required)
static CONFIG: SsmConfig = SsmConfig {
    input_dim: 8,
    hidden_dim: 16,
    output_dim: 8,
};

fn inference_step(input: &[f32; 8], state: &mut SsmState) -> [f32; 8] {
    state.step(input)
}
```

## Fixed-Point Example

```rust
// Enable: kizzasi-embedded = { features = ["fixed-point"] }
use kizzasi_embedded::fixed::FixedSSM;

let mut model = FixedSSM::new(&weights);
let output = model.step(&input_q15);
```

## Performance

- Single step (d=16, no FPU): ~4μs on Cortex-M4 @ 168 MHz
- Single step (d=16, FPU): ~1.2μs on Cortex-M4F @ 168 MHz
- RAM footprint: O(hidden_dim) — no heap required for inference

## Documentation

- [API Documentation](https://docs.rs/kizzasi-embedded)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
