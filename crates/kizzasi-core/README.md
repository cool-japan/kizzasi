# kizzasi-core

Core State Space Model (SSM) engine for Kizzasi AGSP.

## Overview

High-performance SSM implementation with O(1) per-step inference, SIMD optimizations, and parallel processing. Provides the foundational building blocks for autoregressive signal prediction.

## Features

- **Selective SSM**: Input-dependent state transitions with ZOH discretization
- **Parallel Scan**: O(log N) depth associative scan algorithm
- **SIMD Operations**: Vectorized dot products, matrix operations, and activations — multi-SIMD backends (aarch64, AVX512, NEON)
- **Memory Efficient**: Array pooling and workspace management; memory profiling (`MemoryProfiler`, `ProfilingSession`)
- **GPU Support**: CUDA and Metal backends via candle; GPU-accelerated SSM prefix scan (added in v0.2.2)
- **Training**: Full training infrastructure with gradient computation; LoRA adapters (parameter-efficient fine-tuning)
- **Numerical Stability**: Kahan summation, safe exp/log, Welford variance
- **Attention**: Flash Attention and Efficient Attention variants
- **Quantization**: Dynamic quantization with INT8/FP16
- **Pruning**: Gradient and structured pruning
- **Checkpoint Compatibility**: PyTorch checkpoint loading (`PyTorchCheckpoint`, `PyTorchConverter`)

## Architectures

- **Mamba2**: Structured state-space duality with SSD kernel
- **S4D**: Diagonal S4 with DPLR parameterization
- **RWKV v7**: TimeMixing/ChannelMixing variant
- **RetNet**: Multi-Scale Retention
- **H3**: Hungry Hungry Hippos with Diagonal/Shift SSMs
- **S5**: 5th generation SSM

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
- 435 comprehensive tests with 100% pass rate
- Zero-copy operations where possible

## Documentation

- [API Documentation](https://docs.rs/kizzasi-core)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
