# kizzasi-core

Core State Space Model (SSM) engine for Kizzasi AGSP.

![status](https://img.shields.io/badge/status-stable-brightgreen)

## Overview

High-performance SSM implementation with O(1) per-step inference, SIMD optimizations, and parallel processing. Provides the foundational building blocks for autoregressive signal prediction.

## Features

- **Selective SSM**: Input-dependent state transitions with ZOH discretization; per-channel Δ from a rank-`dt_rank` projection, one recurrent state per layer
- **Parallel Scan**: O(log N) depth associative scan algorithm
- **SIMD Operations**: Vectorized dot products, matrix operations, and activations — `simd` dispatches at runtime to the AVX-512/AVX2 or NEON backend, with a portable unrolled fallback
- **Memory Efficient**: Array pooling and workspace management; memory profiling (`MemoryProfiler`, `ProfilingSession`)
- **GPU Device Selection API**: `DeviceConfig`/`DeviceType` and a pluggable `SsmBackend` trait (`CpuSsmBackend` ships here). The off-by-default `metal` feature is a real backend switch: it forwards `candle-core/metal` + `candle-nn/metal`, so `DeviceType::Metal` creates a live GPU device and `is_metal_available()`/`get_best_device()` report it (Apple platforms only — candle's Metal kernels do not build elsewhere). There is deliberately **no** `cuda` feature and no `DeviceType::Cuda`: candle's CUDA backend requires an NVIDIA toolkit at build time, which no Cargo feature can be made conditional on, so offering the flag would break `--all-features` on every non-CUDA machine (see `Cargo.toml` for the target-scoped workaround that `cargo metadata` itself rejects). Portable GPU-accelerated scan is provided by the companion `kizzasi-webgpu` crate, wired in through the `kizzasi` facade's `webgpu` feature (`kizzasi::ssm_backend::select_ssm_backend()`).
- **Training**: Full training infrastructure with gradient computation, LR schedules applied to the optimizer, mixed-precision loss scaling, segment-wise gradient checkpointing, chronological validation splits, and checkpoints that persist AdamW moment state; LoRA adapters (parameter-efficient fine-tuning)
- **Numerical Stability**: Kahan summation, safe exp/log, Welford variance
- **Attention**: Flash Attention and Efficient Attention variants
- **Quantization**: Dynamic INT8 quantization (per-tensor and per-channel) and bit-packed INT4; `QuantizationType::FP16` is explicitly rejected by `DynamicQuantizer` (real half-precision storage is not implemented — see `quantization.rs`)
- **Pruning**: Gradient and structured pruning
- **Checkpoint Compatibility**: PyTorch checkpoint loading (`PyTorchCheckpoint`, `PyTorchConverter`)
- **Causal Convolutions**: `CausalConv1d`, `DepthwiseCausalConv1d`, `DilatedCausalConv1d`, `ShortConv`, WaveNet-style `DilatedStack`
- **Kernel Fusion**: Fused LayerNorm+activation, QKV projection, FFN, and SSM step kernels for reduced memory traffic
- **Embedded Building Blocks**: `FixedPool`/`BumpAllocator`/`StackAllocator` allocators (`embedded_alloc`) and Q15.16/Q7.8 fixed-point arithmetic (`fixed_point`) are self-contained, dependency-free modules usable standalone. The crate as a whole is **not** `no_std`: `candle-core`, `safetensors`, `serde_json` and `chrono` are unconditional std dependencies, and most modules use `std::` directly.

## Architectures

- **Mamba2**: Structured state-space duality with SSD kernel
- **S4D**: Diagonal S4 with DPLR parameterization
- **RWKV v7**: TimeMixing/ChannelMixing variant
- **RetNet**: Multi-Scale Retention
- **H3**: Hungry Hungry Hippos with Diagonal/Shift SSMs
- **S5**: 5th generation SSM

## Quick Start

```rust
use kizzasi_core::{Array1, KizzasiConfig, SelectiveSSM, SignalPredictor};

// Create SSM with 64-dimensional hidden state
let config = KizzasiConfig::new()
    .input_dim(32)
    .hidden_dim(64)
    .output_dim(32)
    .num_layers(4);

let mut ssm = SelectiveSSM::new(config)?;

// Single-step prediction (O(1) complexity)
let input = Array1::<f32>::zeros(32);
let output = ssm.step(&input)?;
```

## Performance

- Single step (d=256): ~80μs
- Batch processing (B=32, d=256): ~1.5ms
- 516 tests passing, 4 platform-gated tests skipped, 0 failures (`cargo nextest run --all-features`)
- Zero-copy operations where possible

## Documentation

- [API Documentation](https://docs.rs/kizzasi-core)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
