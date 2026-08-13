# kizzasi-webgpu

GPU-accelerated WebGPU compute kernels for SSM inference in the Kizzasi ecosystem.

![version](https://img.shields.io/badge/version-0.2.4-blue)
![status](https://img.shields.io/badge/status-alpha-orange)
![tests](https://img.shields.io/badge/tests-53%20passing-brightgreen)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

## Overview

`kizzasi-webgpu` provides a `wgpu`-backed GPU compute layer for Kizzasi signal
processing pipelines. It exposes four WGSL shader kernels — SSM prefix scan,
matrix-vector multiply, SiLU activation, and RMS normalization — all embedded
at compile time with no external shader files required. By default the crate
compiles to 100% pure Rust with no GPU dependencies; activate the `webgpu`
feature to unlock the actual GPU operations.

Each kernel comes in two flavours: a `&[f32]` → `Vec<f32>` form that uploads and
downloads for you, and a `*_gpu_buf` form that takes and returns a `GpuBuffer`
so intermediates stay resident on the device across a chain of operations.

## Features

- **SSM prefix scan** (`ssm_scan_gpu`) — Blelloch work-efficient parallel scan
  over `(a, bu)` pairs using the associative operator `(a₂·a₁, a₂·bu₁ + bu₂)`.
  Sequences of **any length** are scanned exactly: each work-group scans a
  256-element block, the block aggregates are scanned by the same kernel
  (recursively when needed), and a final pass folds the block prefixes back in
- **Matrix-vector multiply** (`matvec_gpu`) — row-major `output = matrix × vector`
  dispatched entirely on the GPU
- **SiLU activation** (`silu_gpu`) — element-wise `x / (1 + exp(-x))`
- **RMS normalization** (`rms_norm_gpu`) — `output[i] = (input[i] / rms(input)) × weight[i]`,
  single work-group, up to `MAX_RMS_NORM_LEN` (256) elements; longer inputs
  return a typed error rather than a truncated result
- **Device residency** (`silu_gpu_buf`, `rms_norm_gpu_buf`, `matvec_gpu_buf`,
  `ssm_scan_gpu_buf`) — chain kernels without host round trips
- **Hybrid SSM backend** (`WebGpuSsmBackend`) — implements `kizzasi-core`'s
  `SsmBackend` trait. Sequences at or above a configurable crossover length
  (`DEFAULT_GPU_THRESHOLD`, 1024) run on the GPU; shorter ones run on the CPU,
  where dispatch and readback would otherwise dominate. If the GPU path fails,
  it logs a warning and completes the scan on the CPU, and `backend_name()`
  reports which path the last scan actually took

## Error handling

Every GPU entry point validates its request against the device limits and runs
inside a `wgpu` error scope, so validation failures, allocation failures and
over-sized requests return a `WebGpuError` instead of reaching `wgpu`'s
panicking default error handler. `WebGpuBackend::new()` requests the adapter's
own limits, so capable hardware is not pinned to downlevel defaults.

`download_f32` blocks the calling thread until the GPU mapping completes
(bounded by an internal timeout); call it inside `spawn_blocking` from an async
context.

## Usage

```toml
[dependencies]
kizzasi-webgpu = { version = "0.2", features = ["webgpu"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

```rust
use kizzasi_webgpu::{WebGpuBackend, ssm_scan_gpu, WebGpuError};

#[tokio::main]
async fn main() -> Result<(), WebGpuError> {
    let backend = WebGpuBackend::new().await?;

    // (a, bu) pairs for the SSM associative scan — any length is supported;
    // the kernel switches to the multi-block path beyond 256 elements.
    let elements: Vec<(f32, f32)> = vec![(0.9, 0.1); 4096];
    let result = ssm_scan_gpu(&backend, &elements)?;

    println!("scan output length: {}", result.len());
    Ok(())
}
```

Without the `webgpu` feature the call returns `Err(WebGpuError::BackendUnavailable)`
immediately, so the crate is always safe to use as a compile-time dependency in
feature-gated codepaths. To let a single code path pick the cheaper backend per
input length — and to survive a GPU fault — wrap the backend in
`WebGpuSsmBackend` (implements `kizzasi-core`'s `SsmBackend` trait) instead of
calling `ssm_scan_gpu` directly.

Keeping intermediates on the device:

```rust,no_run
# use kizzasi_webgpu::{WebGpuBackend, matvec_gpu_buf, silu_gpu_buf, WebGpuError};
# async fn chain(backend: &WebGpuBackend, matrix: &[f32], vector: &[f32]) -> Result<(), WebGpuError> {
let matrix_buf = backend.upload_f32(matrix, "matrix")?;
let vector_buf = backend.upload_f32(vector, "vector")?;

let product = matvec_gpu_buf(backend, &matrix_buf, 128, 64, &vector_buf)?;
let activated = silu_gpu_buf(backend, &product)?;   // no host round trip

let output = backend.download_f32(&activated)?;
# let _ = output;
# Ok(())
# }
```

A larger runnable demo covering `matvec_gpu`, `silu_gpu`, and `rms_norm_gpu`
is available at
[`examples/webgpu_kernels_demo.rs`](examples/webgpu_kernels_demo.rs):

```sh
cargo run --example webgpu_kernels_demo --features webgpu -p kizzasi-webgpu
```

## Testing

```sh
cargo nextest run -p kizzasi-webgpu --all-features
```

GPU tests skip themselves when the host has no adapter. Set
`KIZZASI_REQUIRE_GPU=1` to turn a missing adapter into a test failure instead,
so a machine that is supposed to have a GPU cannot report a vacuous pass.

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
