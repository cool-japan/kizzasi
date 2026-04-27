# kizzasi

**Autoregressive General-Purpose Signal Predictor (AGSP)**

Neuro-symbolic architecture for continuous signal streams with state space models and constraint enforcement.

## Overview

Kizzasi (兆し - "sign/omen") is a Rust-native system for predicting continuous signal streams using state space models. Treats all modalities (audio, sensors, video, control signals) as equivalent signal streams.

## Key Features

- **Unified Interface**: Single API for all model architectures
- **O(1) Inference**: Constant-time per-step prediction for streaming
- **Constraint Enforcement**: Safety guardrails via TensorLogic integration
- **Real-Time I/O**: MQTT, audio, sensors, video streams
- **Production Ready**: 158 tests, zero warnings, full documentation
- **Async/Streaming**: Tokio-based async prediction pipelines
- **Model Versioning**: A/B testing, hot-swapping, canary deployments

## Quick Start

```rust
use kizzasi::{Kizzasi, ModelType};

// Create predictor with Mamba2 model
let predictor = Kizzasi::builder()
    .model_type(ModelType::Mamba2)
    .input_dim(32)
    .output_dim(32)
    .hidden_dim(64)
    .build()?;

// Single-step prediction
let input = vec![0.1, 0.2, 0.3, /* ... */];
let output = predictor.step(input)?;

// Multi-step rollout
let predictions = predictor.predict_n(input, 100)?;

// With safety guardrails
use kizzasi::prelude::*;
let constrained = predictor
    .with_guardrails(my_constraints)
    .step(input)?;
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

## Use Cases

- **Robotics Control**: Real-time motor control with safety constraints
- **Anomaly Detection**: Learn normal patterns, detect deviations
- **Audio Processing**: Next-sample prediction, streaming synthesis
- **Sensor Fusion**: Multi-modal signal integration
- **Video Prediction**: Frame-to-frame prediction with temporal coherence

## Presets

```rust
// Audio processing (16kHz, mel features)
let audio_predictor = Kizzasi::audio_preset(80)?;

// Robotics control (6-DOF arm)
let robot = Kizzasi::robotics_preset(6)?;

// Sensor streams (IoT)
let sensor = Kizzasi::sensor_preset(16)?;

// Lightweight (edge devices)
let edge = Kizzasi::lightweight_preset(32)?;
```

## Performance

- Mamba2 latency: <100μs per step
- Throughput: 320K predictions/sec (distributed)
- Memory: <50MB for typical models
- Zero-copy operations where possible

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
