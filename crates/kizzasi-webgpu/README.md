# kizzasi-webgpu

GPU-accelerated WebGPU compute kernels for SSM inference in the Kizzasi ecosystem.

![version](https://img.shields.io/badge/version-0.2.2-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

## Overview

`kizzasi-webgpu` provides a `wgpu`-backed GPU compute layer for Kizzasi signal
processing pipelines. It exposes four WGSL shader kernels — SSM prefix scan,
matrix-vector multiply, SiLU activation, and RMS normalization — all embedded
at compile time with no external shader files required. By default the crate
compiles to 100% pure Rust with no GPU dependencies; activate the `webgpu`
feature to unlock the actual GPU operations. Sequences longer than 256 elements
fall back to the CPU path automatically, keeping correctness guarantees
unconditional.

## Features

- **SSM prefix scan** (`ssm_scan_gpu`) — Blelloch work-efficient parallel scan
  over `(a, bu)` pairs using the associative operator `(a₂·a₁, a₂·bu₁ + bu₂)`;
  up to 256 elements per workgroup
- **Matrix-vector multiply** (`matvec_gpu`) — row-major `output = matrix × vector`
  dispatched entirely on the GPU
- **SiLU activation** (`silu_gpu`) — element-wise `x / (1 + exp(-x))`
- **RMS normalization** (`rms_norm_gpu`) — `output[i] = (input[i] / rms(input)) × weight[i]`

## Usage

```toml
[dependencies]
kizzasi-webgpu = { version = "0.2", features = ["webgpu"] }
```

```rust
use kizzasi_webgpu::{WebGpuBackend, ssm_scan_gpu, WebGpuError};

#[tokio::main]
async fn main() -> Result<(), WebGpuError> {
    let backend = WebGpuBackend::new().await?;

    let a_vals:  Vec<f32> = vec![0.9; 128];
    let bu_vals: Vec<f32> = vec![0.1; 128];

    // GPU prefix scan — falls back to CPU for sequences > 256 elements
    let result = ssm_scan_gpu(&backend, &a_vals, &bu_vals).await?;

    println!("scan output length: {}", result.len());
    Ok(())
}
```

Without the `webgpu` feature the call returns `Err(WebGpuError::BackendUnavailable)`
immediately, so the crate is always safe to use as a compile-time dependency in
feature-gated codepaths.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `webgpu` | off | Enables GPU operations via `wgpu`. Without this flag all entry points return `WebGpuError::BackendUnavailable`. |

## Supported Backends

| Backend | Platform |
|---------|----------|
| Metal | macOS / iOS |
| Vulkan | Linux / Windows |
| DirectX 12 | Windows |

Backend selection is handled automatically by `wgpu`; `WebGpuBackend::new()`
requests a high-performance adapter and surfaces `WebGpuError::AdapterRequest`
if no suitable GPU is found.

## License

Licensed under the Apache License, Version 2.0.
