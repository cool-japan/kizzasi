# Kizzasi (兆し)

**Autoregressive General-Purpose Signal Predictor (AGSP)**

[![Crates.io](https://img.shields.io/crates/v/kizzasi.svg)](https://crates.io/crates/kizzasi)
[![Documentation](https://docs.rs/kizzasi/badge.svg)](https://docs.rs/kizzasi)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

*"Predicting the flux of the world with the precision of logic."*

---

## Overview

**Kizzasi** (Japanese: 兆し, meaning "sign/omen/premonition") is a Rust-native autoregressive predictor designed for **continuous signal streams**—audio waveforms, sensor data, robotics control signals, and video frames. Unlike traditional Large Language Models (LLMs) that operate on discrete text tokens, Kizzasi is purpose-built for the continuous domain.

### The AGSP Paradigm

The term "Language Model" is a misnomer—what we actually have are **General-Purpose Signal Predictors**. Kizzasi embraces this insight:

- **Text tokens** are just one type of signal (discrete vocabulary indices)
- **Audio samples** are continuous signals (44.1kHz waveforms)
- **Sensor readings** are multivariate time series
- **Video frames** are high-dimensional spatial-temporal signals

All these modalities can be processed by the same autoregressive architecture: *predict the next value(s) based on history*.

### Core Innovation: Neuro-Symbolic Architecture

Kizzasi combines the **learning capability** of State Space Models (Mamba/RWKV/S4) with the **strict reliability** of TensorLogic constraints. This ensures predicted signals:

1. Follow statistical likelihoods (learned from data)
2. Adhere to physical laws (conservation, causality)
3. Respect safety constraints (bounds, rate limits)
4. Satisfy logical rules (domain-specific requirements)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                    kizzasi                                       │
│                              (Unified Facade API)                                │
├─────────────┬──────────────┬──────────────┬──────────────┬──────────────────────┤
│ kizzasi-    │  kizzasi-    │  kizzasi-    │  kizzasi-    │     kizzasi-io       │
│   core      │    model     │  tokenizer   │  inference   │  (World Connectors)  │
│  (Engine)   │   (Archs)    │  (Encoding)  │  (Pipeline)  │                      │
├─────────────┴──────────────┴──────────────┴──────────────┤                      │
│                       kizzasi-logic                       │                      │
│                  (Constraint Enforcement)                 │                      │
├───────────────────────────────────────────────────────────┴──────────────────────┤
│                            COOLJAPAN Ecosystem                                   │
│            scirs2-core  |  scirs2-signal  |  tensorlogic  |  candle              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Crate Structure

| Crate | Description | SLoC |
|-------|-------------|------|
| [`kizzasi`](crates/kizzasi) | Unified facade with prelude and ergonomic API | ~500 |
| [`kizzasi-core`](crates/kizzasi-core) | SSM engine, embeddings, SIMD optimizations, parallel scan | ~4,500 |
| [`kizzasi-model`](crates/kizzasi-model) | Mamba/Mamba2, RWKV, S4/S4D, Transformer architectures | ~4,000 |
| [`kizzasi-tokenizer`](crates/kizzasi-tokenizer) | VQ-VAE, μ-law, quantizers, multi-scale tokenization | ~3,000 |
| [`kizzasi-inference`](crates/kizzasi-inference) | Pipeline orchestration, sampling, batching, streaming | ~4,500 |
| [`kizzasi-logic`](crates/kizzasi-logic) | Constraints, guardrails, projections, training losses | ~5,000 |
| [`kizzasi-io`](crates/kizzasi-io) | MQTT, Audio, WebSocket, Serial, File, DSP | ~4,500 |

**Total: ~25,000 lines of Rust code**

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
kizzasi = "0.1"
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|:-------:|
| `std` | Standard library support | ✓ |
| `full` | Enable all features | ✓ |
| `io` | Physical world connectors | ✓ |
| `logic` | TensorLogic constraints | ✓ |
| `mqtt` | MQTT client (rumqttc) | ✓ |
| `audio` | Audio I/O (cpal) | ✓ |
| `async` | Async/streaming support | ○ |
| `mamba` | Mamba/Mamba2 models | ○ |

Minimal installation:
```toml
kizzasi = { version = "0.1", default-features = false, features = ["std"] }
```

---

## Quick Start

### Basic Prediction

```rust
use kizzasi::prelude::*;

fn main() -> KizzasiResult<()> {
    // Configure predictor with Mamba2 backend
    let config = KizzasiConfig::new()
        .model_type(ModelType::Mamba2)
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(256)
        .state_dim(16)
        .num_layers(4)
        .context_window(8192);

    let mut predictor = Kizzasi::new(config)?;

    // Single step prediction (O(1) complexity)
    let input = array![0.1, 0.2, 0.3];
    let output = predictor.step(&input)?;

    println!("Predicted: {:?}", output);
    Ok(())
}
```

### With Safety Constraints

```rust
use kizzasi::prelude::*;

fn main() -> KizzasiResult<()> {
    let config = KizzasiConfig::new()
        .model_type(ModelType::Rwkv)
        .input_dim(3)
        .output_dim(3);

    let mut predictor = Kizzasi::new(config)?;

    // Define safety constraints
    let guardrails = GuardrailSet::new()
        .add(Guardrail::new(
            ConstraintBuilder::new()
                .name("velocity_limit")
                .bound(0, BoundType::Range(-1.0, 1.0))  // Clamp to [-1, 1]
                .bound(1, BoundType::LessThan(100.0))   // Max value < 100
                .build()?
        ))
        .add(Guardrail::new(
            ConstraintBuilder::new()
                .name("rate_limit")
                .rate_limit(0.1)  // Max change per step
                .build()?
        ));

    predictor.set_guardrails(guardrails);

    // Predictions automatically satisfy constraints
    let input = array![0.5, 0.5, 0.5];
    let safe_output = predictor.step(&input)?;

    Ok(())
}
```

### Real-Time Audio Processing

```rust
use kizzasi::prelude::*;

fn main() -> KizzasiResult<()> {
    // Use audio preset for optimized configuration
    let config = KizzasiConfig::audio_preset()
        .sample_rate(44100.0);

    let mut predictor = Kizzasi::new(config)?;

    // Stream from microphone
    let audio_config = AudioConfig::new()
        .sample_rate(44100)
        .channels(1)
        .buffer_size(1024);

    let mut audio = AudioInput::new(audio_config)?;
    audio.start()?;

    loop {
        let buffer = audio.read()?;
        for sample in buffer.iter() {
            let prediction = predictor.step(&array![*sample])?;
            // Use prediction for audio effect, anomaly detection, etc.
        }
    }
}
```

### Multi-Step Prediction

```rust
use kizzasi::prelude::*;

fn main() -> KizzasiResult<()> {
    let config = KizzasiConfig::new()
        .model_type(ModelType::S4D)
        .input_dim(6)
        .output_dim(6);

    let mut predictor = Kizzasi::new(config)?;

    // Predict N steps into the future
    let initial = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let trajectory = predictor.predict_n(&initial, 100)?;

    println!("Predicted {} future states", trajectory.len());
    Ok(())
}
```

---

## Model Architectures

Kizzasi supports multiple state-of-the-art sequence modeling architectures:

| Model | Per-Step Complexity | State Size | Best For |
|-------|:-------------------:|:----------:|----------|
| **Mamba2** | O(1) | O(d·N) | Default choice, balanced |
| **RWKV** | O(1) | O(d) | Lightweight, fast |
| **S4D** | O(1) | O(d·N) | Smooth dynamics |
| **Transformer** | O(L) | O(L·d) | Baseline comparison |

### Architecture Selection Guide

```rust
// High-performance, long context
ModelType::Mamba2  // Selective SSM with SSD

// Lightweight, embedded systems
ModelType::Rwkv    // Linear attention, minimal state

// Smooth signal dynamics
ModelType::S4D     // HiPPO initialization, diagonal SSM

// Comparison/research
ModelType::Transformer  // Standard attention (O(N) per step)
```

---

## Constraint System

### Constraint Types

| Type | Description | Example |
|------|-------------|---------|
| `Range(min, max)` | Value in [min, max] | Joint angles |
| `LessThan(max)` | Upper bound | Velocity limits |
| `GreaterThan(min)` | Lower bound | Minimum pressure |
| `RateLimit(delta)` | Max change per step | Smooth motion |
| `Linear(a, b)` | a·x ≤ b | Conservation laws |
| `Quadratic(Q, c, b)` | x'Qx + c'x ≤ b | Energy bounds |
| `Temporal(LTL)` | Always/Eventually/Until | Safety properties |

### Training Integration

Constraints can be enforced during training as differentiable losses:

```rust
use kizzasi_logic::{ConstraintAwareLoss, LagrangianRelaxation};

// Combine task loss with constraint violation penalty
let loss = ConstraintAwareLoss::new()
    .task_loss(mse_loss)
    .constraint_loss(guardrails.violation_loss(&prediction))
    .weight(0.1);

// Or use Lagrangian relaxation for adaptive weighting
let relaxation = LagrangianRelaxation::new(guardrails)
    .learning_rate(0.01);
```

---

## Tokenization Strategies

Kizzasi provides multiple signal-to-token conversion strategies:

| Tokenizer | Type | Vocab Size | Best For |
|-----------|:----:|:----------:|----------|
| `ContinuousTokenizer` | Continuous | ∞ | Default, floating-point signals |
| `VQVAETokenizer` | Discrete | Configurable | Learned codebooks |
| `MuLawCodec` | Discrete | 256/65536 | Audio compression |
| `LinearQuantizer` | Discrete | 2^bits | Simple quantization |
| `MultiScaleTokenizer` | Hierarchical | Variable | Multi-resolution |
| `PyramidTokenizer` | Residual | Variable | Progressive refinement |

```rust
use kizzasi_tokenizer::{VQVAETokenizer, VQConfig};

// Create VQ-VAE tokenizer with 1024 codebook entries
let config = VQConfig::new()
    .codebook_size(1024)
    .embed_dim(256)
    .ema_decay(0.99);

let tokenizer = VQVAETokenizer::new(config)?;

let tokens = tokenizer.encode(&signal)?;
let reconstructed = tokenizer.decode(&tokens)?;
```

---

## Use Cases

### 1. Robotics Control

Real-time motor control with safety bounds:

```rust
let config = KizzasiConfig::robotics_preset()
    .input_dim(12)   // 6 joint positions + 6 velocities
    .output_dim(6);  // 6 torque commands

let guardrails = GuardrailSet::new()
    .add(joint_limits())      // Physical joint ranges
    .add(velocity_limits())   // Maximum angular velocities
    .add(torque_limits());    // Actuator saturation
```

### 2. Industrial Anomaly Detection

Predictive maintenance for IoT sensors:

```rust
let config = KizzasiConfig::sensor_preset()
    .input_dim(32);  // 32 sensor channels

// Train on "normal" operation data
// At runtime, large prediction errors indicate anomalies
let prediction = predictor.step(&sensor_reading)?;
let anomaly_score = (prediction - actual).mapv(|x| x.abs()).sum();
```

### 3. Audio Synthesis

Next-sample prediction for audio effects:

```rust
let config = KizzasiConfig::audio_preset()
    .model_type(ModelType::Rwkv);  // Fast, lightweight

// WaveNet-style sample-by-sample generation
let mut predictor = Kizzasi::new(config)?;
for sample in input_audio.iter() {
    let next_sample = predictor.step(&array![*sample])?;
    output_audio.push(next_sample[0]);
}
```

### 4. Video Frame Prediction

Anime in-betweening and frame interpolation:

```rust
let config = KizzasiConfig::new()
    .model_type(ModelType::Mamba2)
    .input_dim(1024)   // Frame embedding dimension
    .output_dim(1024);

// Enforce skeleton/pose constraints
let guardrails = GuardrailSet::new()
    .add(bone_length_constraints())
    .add(joint_angle_limits());
```

---

## Performance

### Inference Characteristics

| Metric | Mamba2 | RWKV | S4D | Transformer |
|--------|:------:|:----:|:---:|:-----------:|
| Per-step latency | ~100μs | ~50μs | ~80μs | ~500μs |
| Memory (state) | O(d·N) | O(d) | O(d·N) | O(L·d) |
| Context length | Unlimited | Unlimited | Unlimited | Fixed L |
| Training parallel | ✓ | ✓ | ✓ | ✓ |

### Optimization Features

- **SIMD Vectorization**: Optimized dot products, layer norms, softmax
- **Parallel Scan**: O(log N) depth parallel SSM scan
- **Memory Pooling**: Reusable array allocations
- **Batch Processing**: Efficient multi-sequence inference
- **Continuous Batching**: Dynamic batch formation for streaming

---

## COOLJAPAN Ecosystem

Kizzasi is part of the COOLJAPAN scientific computing ecosystem:

| Crate | Purpose |
|-------|---------|
| [scirs2-core](https://crates.io/crates/scirs2-core) | Array operations, random, SIMD |
| [scirs2-signal](https://crates.io/crates/scirs2-signal) | Signal processing algorithms |
| [scirs2-fft](https://crates.io/crates/scirs2-fft) | Fast Fourier Transform |
| [tensorlogic](https://crates.io/crates/tensorlogic) | Neuro-symbolic constraints |
| [candle-core](https://crates.io/crates/candle-core) | ML backend (GPU acceleration) |

See [KIZZASI_POLICY.md](KIZZASI_POLICY.md) for dependency guidelines.

---

## Project Statistics

```
Language            Files        Lines         Code     Comments       Blanks
───────────────────────────────────────────────────────────────────────────────
Rust                   89       31,735       25,009        1,457        5,269
TOML                    8          370          268           47           55
Markdown               18        2,449            0        1,872          577
───────────────────────────────────────────────────────────────────────────────
Total                 115       34,554       25,277        3,376        5,901
```

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/cool-japan/kizzasi
cd kizzasi
cargo build --all-features
cargo test --all-features
```

### Code Quality

```bash
# Format
cargo fmt

# Lint
cargo clippy --all-features

# Benchmarks
cargo bench

# Documentation
cargo doc --all-features --no-deps
```

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

## Acknowledgments

- The COOLJAPAN ecosystem contributors
- Mamba/S4 research teams at CMU and Princeton
- RWKV community
- The Rust ML ecosystem (candle, burn)

---

*Kizzasi: Sensing the future, one prediction at a time.*
