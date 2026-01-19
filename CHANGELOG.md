# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
