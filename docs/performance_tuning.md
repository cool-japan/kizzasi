# Performance Tuning Guide

This guide covers the key levers for squeezing maximum throughput and minimum
latency out of Kizzasi in production.

---

## Overview

Kizzasi inference cost is determined by:

1. **Model size** — hidden dimension, state dimension, number of layers
2. **Batch size** — amortise fixed overhead; tune for your latency SLO
3. **SIMD paths** — ARM64 NEON or x86-64 AVX2/AVX-512, selected at compile time
4. **State management** — reset vs stateful streaming, context warming
5. **Quantisation** — INT8 / FP16 for smaller models and lower memory bandwidth
6. **Hardware** — CPU (default), Metal (macOS, `metal`), WebGPU (portable, `webgpu`)

The `examples/performance_tuning_demo.rs` file in this repository contains
runnable benchmarks for each of these areas.

---

## Model Selection

| Architecture | Best for | Latency | Quality |
|---|---|---|---|
| **Mamba** | General-purpose real-time signals | ★★★★★ | ★★★★ |
| **Mamba2** | Long-context streaming, audio | ★★★★ | ★★★★★ |
| **RWKV** | Efficient edge inference | ★★★★★ | ★★★ |
| **S4D** | Periodic / oscillatory signals | ★★★★ | ★★★★ |
| **Transformer** | Short-context, accuracy-critical | ★★ | ★★★★★ |

### Sizing guidance

```
hidden_dim  state_dim  layers  ~params   recommended use
    64          8         2     ~100 K   embedded, < 1 ms target
   128         16         4     ~800 K   desktop / edge server
   256         32         6    ~6.5 M   GPU-accelerated workloads
   512         64        12    ~52 M    large-scale training
```

Double `hidden_dim` → roughly 4× parameter growth and 2–4× latency increase.

### RWKV vs Mamba

- **RWKV** has lower per-step FLOP count but wider channel-mix operations
  that bottleneck on memory bandwidth for small batch sizes.
- **Mamba** is generally faster for single-sequence streaming on modern CPUs
  because its SSM scan maps better to SIMD dot products.

---

## Batch Processing

### BatchScheduler

For server deployments, use `kizzasi_inference::BatchScheduler` to collect
individual requests into micro-batches before inference:

```rust
use kizzasi_inference::{BatchScheduler, BatchConfig};

let config = BatchConfig {
    max_batch_size: 16,
    max_wait_ms: 5,      // collect requests for up to 5 ms
    ..Default::default()
};
let scheduler = BatchScheduler::new(config);
```

### Tuning batch size

| Batch size | Per-sample cost | When to use |
|---|---|---|
| 1 | Highest | Hard latency SLO (< 1 ms) |
| 4–8 | ~30–50% lower | Balanced throughput/latency |
| 16–32 | ~60–70% lower | Throughput-maximising server |
| 64+ | ~75–80% lower | Offline processing / training |

The crossover point depends on `hidden_dim`: larger models amortise overhead
more aggressively and benefit more from large batches.

### `predict_batch` vs looping `step`

```rust
// Slower: N independent step() calls, each initialises a fresh state
for x in &inputs {
    predictor.reset();
    let _ = predictor.step(x)?;
}

// Faster: single predict_batch call, one state init, parallel output computation
let outputs = predictor.predict_batch(&inputs)?;
```

---

## SIMD Optimisation

Kizzasi's hot paths live in `kizzasi-core` and are implemented via
`scirs2-core` array operations.  The SIMD tier is chosen **at compile time**
from `RUSTFLAGS`.

### Enabling platform-optimal SIMD

```bash
# ARM64 — enables NEON (FMLA, LD4, etc.)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# x86-64 — enables AVX2 and, where available, AVX-512
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Operations that benefit from SIMD

| Operation | NEON | AVX2 |
|---|---|---|
| Dot product (input projection) | `fmla` lanes | `_mm256_fmadd_ps` |
| Layer normalisation | parallel `fsub`/`fdiv` | `_mm256_div_ps` |
| Softmax (attention) | NEON `vmaxvq_f32` + `vexpq_f32` | AVX-512 `_mm512_exp_ps` |
| SiLU / GELU | fused multiply-add | `_mm256_fmadd_ps` |
| Parallel scan (SSM) | block-parallel `vmla` | `_mm256_add_ps` trees |

Operations that do **not** significantly benefit: small matrix multiplies
(< 16×16) and scalar branches in token-level logic.

### Cross-compilation

For ARM64 embedded targets:

```toml
# .cargo/config.toml
[target.aarch64-unknown-linux-gnu]
rustflags = ["-C", "target-feature=+neon"]
```

---

## Memory Management

### State reuse

Hidden states are heap-allocated once per predictor and reused across steps.
Avoid calling `reset()` inside a tight inference loop unless you explicitly
need a fresh context window.

```rust
// Good — state is accumulated across the entire signal
for sample in audio_stream {
    let pred = predictor.step(&sample)?;
}

// Wasteful — discards learned context on every sample
for sample in audio_stream {
    predictor.reset();      // re-zeroes all hidden states
    let pred = predictor.step(&sample)?;
}
```

### KV-cache limits (Transformer backend)

When `ModelType::Transformer` is selected, a KV-cache is maintained up to
`context_window` entries (default 4096).  For long-running streams:

- Reduce `context_window` to cap memory usage.
- Use SSM backends (Mamba / RWKV) to eliminate KV-cache entirely — their
  O(1) inference step requires only a fixed-size hidden state.

### Fork for branching

`fork()` clones the hidden state without allocating a new model.  This is
cheaper than reconstructing a predictor from scratch:

```rust
// Cheap: clone state, share model weights
let mut branch = predictor.fork()?;

// Expensive: allocate new model + re-encode context
let mut fresh = KizzasiBuilder::new()...build()?;
for x in &context_history { let _ = fresh.step(x)?; }
```

---

## Profiling

### `std::time::Instant` (application-level)

```rust
use std::time::Instant;

let t0 = Instant::now();
let _ = predictor.step(&input)?;
let us = t0.elapsed().as_nanos() as f64 / 1_000.0;
println!("step latency: {:.2} µs", us);
```

### `ProfilingRegistry` (model-level, kizzasi-model)

`kizzasi_model::profiling::ProfilingRegistry` accumulates per-operation timings
without heap allocation when disabled:

```rust
use kizzasi_model::profiling::{ProfilingRegistry, TimingGuard};

let mut reg = ProfilingRegistry::new();
reg.enable();

{
    let _g = TimingGuard::new(&mut reg, "ssm_step");
    let _ = model.step(&input)?;
}  // elapsed recorded on drop

for (name, acc) in reg.summary() {
    println!("{name}: {:.1} µs avg  (n={})",
        acc.mean().unwrap().as_micros(), acc.count);
}
```

### Tracing spans (distributed tracing)

With `tracing` enabled, add spans around inference calls:

```rust
use tracing::{info_span, Instrument};

async {
    let _ = predictor.step(&input)
        .instrument(info_span!("inference", model = "mamba2"))
        .await;
}.await;
```

Export to Jaeger or Zipkin via `tracing-opentelemetry`.

### `ComprehensiveProfiler` (multi-model comparison)

```rust
use kizzasi_model::profiling::ComprehensiveProfiler;

let report = ComprehensiveProfiler::new()
    .num_steps(500)
    .profile_all_models()?;

println!("{}", report.format_report());
```

---

## Quantisation

### INT8 inference

```rust
use kizzasi_model::quantization::QuantizationConfig;

let config = QuantizationConfig::int8();
// apply to model weights before inference
```

**Tradeoffs:**
- ~2–4× memory reduction vs FP32
- ~1.5–2× throughput increase on CPU (reduced memory bandwidth)
- Small accuracy degradation (~0.1–0.5 dB SNR for audio, model-dependent)

### FP16 / BF16

Enabled automatically when `candle` targets a Metal device (`--features metal`).  On CPU:

```rust
use kizzasi_model::mixed_precision::MixedPrecisionConfig;

let config = MixedPrecisionConfig::fp16_compute_fp32_accumulate();
```

BF16 is generally preferred over FP16 for training because of its larger
dynamic range; FP16 is better for inference-only deployments where the extra
precision of BF16 is not needed.

---

## Hardware-Specific Notes

### CPU (default)

- Compile with `RUSTFLAGS="-C target-cpu=native"` for SIMD.
- Use Rayon parallel iterators for batch processing on multi-core machines.
- `kizzasi-core` uses `rayon` internally for `predict_batch`.

### CUDA

Not available. kizzasi ships no `cuda` feature: candle's CUDA backend needs an
NVIDIA toolkit at *build* time (its build scripts abort without one) and Cargo
cannot make a feature conditional on the host toolchain, so the flag would
break every build on a machine without CUDA. Use `webgpu` below for portable
GPU acceleration, or depend on candle directly if you need its CUDA kernels.

### Metal (macOS / Apple Silicon)

```toml
[dependencies]
kizzasi = { version = "0.2", features = ["metal"] }
```

```rust
use kizzasi_core::device::{DeviceConfig, DeviceType};

let device = DeviceConfig::new()
    .with_device_type(DeviceType::Metal)
    .create_device()?;
```

`DeviceType::Metal` only exists with the `metal` feature, which forwards
candle's own Metal backend — a build without it cannot name a GPU device, let
alone silently fall back to one. Metal inference leverages the unified memory
architecture of Apple Silicon, eliminating CPU↔GPU data transfer latency for
most operations.

The backend itself is Apple-only, but the feature resolves on every platform:
it goes through the `kizzasi-metal` crate, which target-scopes the candle
dependency so `--all-features` still builds on Linux and Windows. On those
targets the feature is inert and says so — `is_metal_available()` returns
`false`, `get_best_device()` stays on CPU, and `create_device()` for
`DeviceType::Metal` returns a `DeviceError` naming the target.

### WebGPU (portable: Metal / Vulkan / DX12)

```toml
[dependencies]
kizzasi = { version = "0.2", features = ["webgpu"] }
```

```rust
// Returns the GPU-backed scan backend when an adapter is reachable, and the
// CPU backend (with a logged reason) when it is not — it never fails.
let backend = kizzasi::ssm_backend::select_ssm_backend().await;
let states = backend.ssm_scan(&elements)?;
tracing::info!(backend = backend.backend_name(), "scan backend");
```

The GPU path pays off on long sequences; `WebGpuSsmBackend` runs sequences
shorter than its threshold on the CPU rather than paying dispatch and readback
costs.

---

## Summary Checklist

- [ ] Compile with `RUSTFLAGS="-C target-cpu=native"` for SIMD.
- [ ] Choose the smallest model config that meets your quality target.
- [ ] Use `predict_batch` instead of looping `step` for throughput.
- [ ] Avoid `reset()` in tight loops — let state accumulate for streaming.
- [ ] Use `fork()` instead of re-constructing predictors for branching.
- [ ] Profile with `ProfilingRegistry` before optimising further.
- [ ] Consider INT8 quantisation for memory-bound workloads.
- [ ] Enable the `webgpu` feature (any GPU) or `metal` (Apple) for GPU-backed work.
