# kizzasi-core

Core State Space Model (SSM) engine for Kizzasi AGSP.

## Overview

High-performance SSM implementation with O(1) per-step inference, SIMD optimizations, and parallel processing. Provides the foundational building blocks for autoregressive signal prediction.

## Features

- **Selective SSM**: Input-dependent state transitions with ZOH discretization
- **Parallel Scan**: O(log N) depth associative scan algorithm
- **SIMD Operations**: Vectorized dot products, matrix operations, and activations
- **Memory Efficient**: Array pooling and workspace management
- **GPU Support**: CUDA and Metal backends via candle
- **Training**: Full training infrastructure with gradient computation
- **Numerical Stability**: Kahan summation, safe exp/log, Welford variance

## Quick Start

```rust
use kizzasi_core::{SelectiveSSM, KizzasiConfig};

// Create SSM with 64-dimensional hidden state
let config = KizzasiConfig::builder()
    .input_dim(32)
    .hidden_dim(64)
    .output_dim(32)
    .num_layers(4)
    .build()?;

let mut ssm = SelectiveSSM::new(config)?;

// Single-step prediction (O(1) complexity)
let input = Array1::zeros(32);
let output = ssm.step(&input)?;
```

## Performance

- Single step (d=256): ~80μs
- Batch processing (B=32, d=256): ~1.5ms
- 422 comprehensive tests with 100% pass rate
- Zero-copy operations where possible

## Documentation

- [API Documentation](https://docs.rs/kizzasi-core)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
