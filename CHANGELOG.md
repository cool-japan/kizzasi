# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-04-27

### Changed
- Bump version to 0.2.1 for dependency compatibility (oxirs-core reqwest TLS feature resolution)

## [0.2.0] - 2026-04-26 (Partially released)

### Added

#### New Architectures & Models
- **RWKV v5** and **RWKV v7** with data-dependent time decay
- **Neural ODE** continuous-time models
- **Spiking neural network** (neuromorphic SSM)
- **Flash Linear Attention** kernel
- **Speculative decoding** for faster inference
- **Multi-modal fusion** (audio + vision + control)

#### Training & Optimization
- **Full backpropagation** through SSM recurrence (`backprop_ssm.rs`)
- **Gradient checkpointing** for memory-efficient training
- **LoRA adapters** for efficient fine-tuning
- **Curriculum learning** with progressive difficulty
- **Architecture search** (NAS) for model selection
- **Model pruning** and **ONNX export**

#### Deployment & Integration
- **Python bindings** via PyO3/maturin (`kizzasi-python`)
- **no_std embedded** support (`kizzasi-embedded`)
- **WASM compilation** with browser demo
- **Docker** and **Kubernetes** deployment manifests
- **gRPC** and **REST API** inference servers
- **HuggingFace Hub** API client for model download
- **GGUF format** loader with full dequantization
- **Distributed prediction** with load balancing

#### Signal Processing
- **Cepstral analysis** and pitch detection
- **Time-frequency analysis** (Gabor, S-transform, Wigner-Ville)
- **Machine learning** signal denoising and anomaly detection
- **Advanced resampling** (Farrow, time-varying, arbitrary SRC)

#### Documentation & Benchmarks
- Mathematical formulations for all SSM architectures
- Architecture comparison benchmark suite (5 models x 4 dims)
- Fine-tuning workflow example
- Performance tuning guide

### Changed
- **Version bump**: 0.1.0 -> 0.2.0
- JSON weight I/O for all model types (save/load_weights_json)
- NameRemapper for HuggingFace key translation
- Factory injection wired for all model types
- File splits to keep all files under 2000 lines

### Technical Details
- **122,000+** lines of Rust code across 351 source files
- **2,235** tests passing (up from 397)
- Zero clippy warnings
- Pure Rust (COOLJAPAN policy compliant)

## [0.1.0] - 2026-01-18

### Added

#### Core Features
- **Kizzasi Core Engine** (`kizzasi-core`)
  - Selective State Space Model (SSM) implementation with parallel scan
  - Discretization caching and workspace pooling for optimization
  - ILP operations and cache-aligned data structures
  - SIMD-optimized embeddings and signal processing
  - Hardware acceleration support (CUDA/Metal feature-gated)

- **Model Architectures** (`kizzasi-model`)
  - Mamba and Mamba2 state space models
  - RWKV architecture implementation
  - S4/S4D diagonal state space models
  - Transformer architecture support
  - Unified model factory for easy configuration
  - HuggingFace-compatible weight loading

- **Signal Tokenization** (`kizzasi-tokenizer`)
  - VQ-VAE (Vector Quantized Variational Autoencoder)
  - Residual VQ-VAE for hierarchical encoding
  - μ-law compression/expansion codec
  - Linear, adaptive, and deadzone quantizers
  - Multi-scale temporal tokenization
  - Domain-specific tokenizers (music, environmental audio)

- **Inference Pipeline** (`kizzasi-inference`)
  - Streaming inference with configurable sampling
  - Temperature, top-k, and top-p sampling strategies
  - Batch processing with dynamic batching
  - Memory-efficient state management
  - Multi-modal input support

- **Constraint Enforcement** (`kizzasi-logic`)
  - Linear and nonlinear constraint projection
  - Gradient projection methods
  - ADMM (Alternating Direction Method of Multipliers)
  - Lagrangian relaxation for soft constraints
  - LTL (Linear Temporal Logic) formula support
  - Sliding window constraint checkers

- **Physical World I/O** (`kizzasi-io`)
  - MQTT client for IoT integration
  - Real-time audio I/O via CPAL
  - WebSocket streaming support
  - Serial port communication
  - File I/O (WAV, CSV, HDF5)
  - Advanced DSP: FFT, filtering, resampling
  - Beamforming and DOA estimation
  - Hilbert-Huang Transform (EMD/EEMD)
  - Quality metrics (PESQ, STOI, POLQA)
  - Source separation (FastICA, NMF, PCA)

- **Unified Facade** (`kizzasi`)
  - Ergonomic prelude module
  - Simple API for common use cases
  - Re-exports all sub-crates

### Technical Details
- Pure Rust implementation (COOLJAPAN policy compliant)
- No `unwrap()` calls in production code
- Platform-specific ROS2 support (Linux only)
- Comprehensive test suite with property-based testing
- ~91,000 lines of Rust code across 292 source files

### Dependencies
- Built on COOLJAPAN ecosystem: `scirs2-core`, `scirs2-signal`, `scirs2-fft`
- Uses `tensorlogic` for constraint verification
- Candle backend for tensor operations

[0.1.0]: https://github.com/cool-japan/kizzasi/releases/tag/v0.1.0
