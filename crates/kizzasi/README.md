# kizzasi

![Status](https://img.shields.io/badge/status-stable-brightgreen)

**Autoregressive General-Purpose Signal Predictor (AGSP)**

Neuro-symbolic architecture for continuous signal streams with state space models and constraint enforcement.

## Overview

Kizzasi (兆し - "sign/omen") is a Rust-native system for predicting continuous signal streams using state space models. Treats all modalities (audio, sensors, video, control signals) as equivalent signal streams.

## Key Features

- **Unified Interface**: Single API for all model architectures
- **O(1) Inference**: Constant-time per-step prediction for streaming
- **Constraint Enforcement**: Safety guardrails via TensorLogic integration
- **Real-Time I/O**: MQTT, audio, sensors, video streams
- **Production Ready**: 163 tests, zero warnings, full documentation
- **Async/Streaming**: Tokio-based async prediction pipelines
- **Model Versioning**: A/B testing, hot-swapping, canary deployments

## Quick Start

```rust
use kizzasi::prelude::*;

// Create predictor with Mamba2 model
let mut predictor = KizzasiBuilder::new()
    .model_type(ModelType::Mamba2)
    .input_dim(32)
    .output_dim(32)
    .hidden_dim(64)
    .build()?;

// Single-step prediction
let input = Array1::from_vec(vec![0.1; 32]);
let output = predictor.step(&input)?;

// Multi-step rollout (100 steps ahead)
let predictions = predictor.predict_n(&input, 100)?;

// With safety guardrails
let mut guardrails = GuardrailSet::new();
let bound = ConstraintBuilder::new()
    .name("output_bounds")
    .greater_eq(-1.0)
    .less_eq(1.0)
    .build()?;
guardrails.add_global(Guardrail::new(bound, false));
predictor.set_guardrails(guardrails);
let constrained = predictor.step(&input)?;
```

## Architecture

```
kizzasi/
├── kizzasi-core      # SSM engine, SIMD, parallel scan
├── kizzasi-model     # Mamba, RWKV, S4, Transformer
├── kizzasi-tokenizer # VQ-VAE, quantization, compression
├── kizzasi-inference # Pipeline, sampling, streaming
├── kizzasi-logic     # Constraints, optimization, safety
├── kizzasi-io        # MQTT, audio, sensors, video
└── kizzasi           # Unified facade (this crate)
```

### What this crate pulls in

`kizzasi` depends on `kizzasi-core` and `kizzasi-model` (both unconditional),
plus `kizzasi-logic` and `kizzasi-io` behind the `logic` / `io` features, and
`kizzasi-webgpu` behind `webgpu`. It does **not** re-export
`kizzasi-tokenizer` or `kizzasi-inference` — add those as separate
dependencies if you need tokenization or the inference pipeline/servers.

### Features

`default = ["std", "full"]`, and `full = ["io", "logic", "mqtt", "async",
"config-files", "macros"]` — everything in `full` is pure Rust, so the default
build links no C library of kizzasi's choosing.

Three features are outside `full` and off by default:

| Feature | What it adds | Cost |
|---|---|---|
| `audio` | Live audio device I/O (`kizzasi-io/audio`) | **Not pure Rust** — cpal links libasound (`alsa-sys`) on Linux, CoreAudio on Apple, WASAPI on Windows |
| `webgpu` | GPU SSM scan through `kizzasi-webgpu` | Pure Rust (wgpu → Metal/Vulkan/DX12 system drivers) |
| `metal` | candle's Metal backend in `kizzasi-core` | Apple platforms only |

```rust
// GPU when one is reachable, CPU otherwise — never fails.
let backend = kizzasi::ssm_backend::select_ssm_backend().await;
let states = backend.ssm_scan(&elements)?;
```

There is no `cuda` feature; see `kizzasi-core`'s `Cargo.toml` for why a CUDA
flag cannot be offered without breaking every build that lacks an NVIDIA
toolkit.

### Model selection

`ModelType` really does select the architecture that runs; the four variants
compute different functions:

| `ModelType` | Engine | Notes |
|---|---|---|
| `Mamba2` (default) | `kizzasi_core::SelectiveSSM` | Selective scan with an input-dependent per-channel Δ. **Not** the chunked SSD/multi-head formulation of the Mamba-2 paper — use `kizzasi_model::mamba2::Mamba2` directly for that. The only engine supporting `fork()` and full-state checkpoints. |
| `Mamba` | `kizzasi_model::mamba::Mamba` | Gated block with a causal convolution. |
| `S4` | `kizzasi_model::s4::S4D` | Diagonal structured SSM. |
| `Rwkv` | `kizzasi_model::rwkv::Rwkv` | Multi-head linear attention. |

`kizzasi-model` also ships `Transformer`, `S5`, `Rwkv7`, `H3` and others.
`ModelType` has no variant for them (adding one would break every exhaustive
`match` outside this crate), so construct those from `kizzasi-model` directly.

Capabilities that are limited to the `Mamba2` engine, because the
`kizzasi-model` architectures are neither `Clone` nor able to export their
weights in memory:

- `Kizzasi::fork()` — returns a typed error for the other three.
- `FullStateCheckpoint` / `save_full_checkpoint` — same.

For those architectures, persist with `Kizzasi::save_weights()` and rebuild
with `KizzasiBuilder::weights_path()`.

### Weights

`KizzasiBuilder::weights_path(path)` loads the file while the predictor is
being built. A missing file, a shape mismatch, or a checkpoint that matches no
parameter of the model is a hard error — the builder never falls back to
random initialisation once a path is given. Omitting `weights_path` yields a
randomly-initialised model.

```rust
use kizzasi::prelude::*;

let trained = KizzasiBuilder::new()
    .input_dim(3)
    .output_dim(3)
    .hidden_dim(128)
    .build()?;
trained.save_weights("model.json")?;

let restored = KizzasiBuilder::new()
    .input_dim(3)
    .output_dim(3)
    .hidden_dim(128)
    .weights_path("model.json")
    .build()?;
```

`.safetensors` checkpoints (using the `kizzasi-model` tensor naming
convention) are accepted for the `Mamba` / `S4` / `Rwkv` backends.

## Use Cases

- **Robotics Control**: Real-time motor control with safety constraints
- **Anomaly Detection**: Learn normal patterns, detect deviations
- **Audio Processing**: Next-sample prediction, streaming synthesis
- **Sensor Fusion**: Multi-modal signal integration
- **Video Prediction**: Frame-to-frame prediction with temporal coherence

## Presets

```rust
use kizzasi::prelude::*;

// Audio processing (44.1kHz, mono)
let audio_predictor = KizzasiBuilder::audio_preset().build()?;

// Robotics control (6-DOF arm)
let robot = KizzasiBuilder::robotics_preset(6).build()?;

// Sensor streams (IoT, 16 sensors)
let sensor = KizzasiBuilder::sensor_preset(16).build()?;

// Lightweight (edge devices)
let edge = KizzasiBuilder::lightweight_preset(32, 32).build()?;

// Video frame prediction (large context window)
let video = KizzasiBuilder::video_preset(256).build()?;

// Real-time control loop (state_dim, action_dim)
let control = KizzasiBuilder::control_preset(12, 6).build()?;
```

## Performance

- Mamba2 latency: <100μs per step
- Throughput: 320K predictions/sec (distributed)
- Memory: <50MB for typical models

Note on allocations: `step_slice` and `step_inplace` are ergonomic wrappers,
not allocation-free paths — the model's step signature takes an owned
`Array1<f32>`, so the input is copied once per call. `step_inplace` does save
the caller from allocating a `Vec` for each output. See their rustdoc for the
exact behaviour.

## Documentation

- [API Documentation](https://docs.rs/kizzasi)
- [GitHub Repository](https://github.com/cool-japan/kizzasi)
- [Architecture Guide](https://github.com/cool-japan/kizzasi/blob/master/ARCHITECTURE.md)

## Citation

If you use Kizzasi in research, please cite:

```bibtex
@software{kizzasi2024,
  title = {Kizzasi: Autoregressive General-Purpose Signal Predictor},
  author = {COOLJAPAN Team},
  year = {2024},
  url = {https://github.com/cool-japan/kizzasi}
}
```

## License

Licensed under the Apache License, Version 2.0.

## Contributing

See [CONTRIBUTING.md](https://github.com/cool-japan/kizzasi/blob/master/CONTRIBUTING.md)
