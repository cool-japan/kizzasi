# Kizzasi Development Roadmap

## Project Overview

**Kizzasi** (兆し) - Autoregressive General-Purpose Signal Predictor (AGSP)

A Rust-native system for predicting continuous signal streams (audio, sensors, video, control signals) using State Space Models with neuro-symbolic constraint enforcement.

---

## Current Status (v0.2.1)

### Codebase Metrics

| Metric | Value |
|--------|-------|
| Total Lines | ~123,000 Rust ⬆️ |
| Crates | 10 |
| Test Count | 2,277 ✅ |
| Coverage | Core paths + comprehensive |
| Last Updated | 2026-04-27 |

### Implementation Status by Crate

| Crate | Status | Completion |
|-------|:------:|:----------:|
| kizzasi-core | Production | 90% |
| kizzasi-model | Production | 90% ⬆️ |
| kizzasi-tokenizer | Production | 85% |
| kizzasi-inference | Production | 85% ⬆️ |
| kizzasi-logic | Production | 90% |
| kizzasi-io | Production | 80% |
| kizzasi | Production | 85% |

---

## Architecture Overview

```
kizzasi/
├── crates/
│   ├── kizzasi/              # Main facade (prelude, unified API)
│   ├── kizzasi-core/         # SSM engine, embeddings, SIMD, parallel scan
│   ├── kizzasi-model/        # Mamba, Mamba2, RWKV, S4D, Transformer
│   ├── kizzasi-tokenizer/    # VQ-VAE, μ-law, quantizers, multi-scale
│   ├── kizzasi-inference/    # Pipeline, batch, sampling, streaming
│   ├── kizzasi-logic/        # Constraints, guardrails, projections
│   └── kizzasi-io/           # MQTT, Audio, WebSocket, Serial, File
└── KIZZASI_POLICY.md         # Ecosystem guidelines
```

---

## Priority Matrix

### P0: Critical (Blocking Release)

- [x] **Weight Loading**: Load pre-trained Mamba weights from safetensors ✅
- [x] **GPU Acceleration**: CUDA/Metal backend via candle ✅
- [x] **Training Loop**: Training infrastructure with GPU support (TrainingLoop, TrainingConfig, TrainingResult, GradientSync) ✅

### P1: High Priority (v0.2)

- [x] **Checkpoint Compatibility**: JSON weight format done (save/load_weights_json for all models, NameRemapper for HF key translation, factory injection wired); GGUF loading exists; full PyTorch .pth done via `candle_core::pickle::read_all`; HuggingFace Hub API client complete (hf_hub.rs — blocking HTTP client with caching, auth, SafeTensors shard download, `load_from_hub` convenience fn; feature-gated `hf-hub`) (completed 2026-04-18)
  - **Goal:** Loading a real HuggingFace Mamba `.pth` (single-file and multi-shard) through `PyTorchConverter` returns an `ArrayD<f32>` map with correct shapes and HF→internal name translation. All three pre-existing stub sites reconciled.
  - **Design:** `candle_core::pickle::read_all(path)` core; `tensor_to_ndarray` helper; backward-compat `load_checkpoint` (Array2) + new `load_checkpoint_raw` (ArrayD) + `load_pth_sharded` + `load_from_huggingface_pth`; `PthIndex` from `pytorch_model.bin.index.json`; `split_x_proj` helper; NameRemapper extended with `backbone.embeddings.weight`, `backbone.norm_f.weight`, `lm_head.weight` tied-weights handling; kizzasi-core stubs deleted.
  - **Files:** `crates/kizzasi-model/src/pytorch_compat.rs`, `crates/kizzasi-model/src/loader.rs`, `crates/kizzasi-core/src/pytorch_compat.rs`, `crates/kizzasi-core/src/weights.rs`, `crates/kizzasi-model/tests/pytorch_pth_roundtrip.rs`
  - **Tests:** tensor_to_ndarray rank/dtype coverage; PthIndex JSON parsing; split_x_proj roundtrip; integration load+rename fixture; negative path/rank-3 errors.
- [x] **Quantization**: INT8/FP16 inference support
- [x] **Distributed Inference**: Multi-GPU support (in-process data-parallel)
- [x] **Python Bindings**: PyO3 wrapper for kizzasi (PyPI CI via maturin) ✅

### P2: Medium Priority (v0.3+)

- [x] **no_std Support**: Embedded systems (ARM Cortex-M)
- [x] **WASM Compilation**: Browser inference
- [x] **LoRA Adapters**: Efficient fine-tuning
- [ ] **Pre-trained Models**: "Kizzasi-Takumi" model zoo

### P3: Future Research

- [x] **Multi-Modal Fusion**: Audio + Vision + Control
- [x] **Neuromorphic SSMs**: Spiking neural network integration
- [x] **Continuous-Time Models**: ODE-based dynamics

---

## Detailed Roadmap

### Phase 1: Foundation (COMPLETED)

#### kizzasi-core
- [x] HiddenState management with O(1) update
- [x] ContinuousEmbedding layer
- [x] KizzasiConfig builder pattern
- [x] SignalPredictor trait
- [x] SelectiveSSM base implementation
- [x] SIMD optimizations (dot product, layer norm, softmax, fast exp)
- [x] Array pooling for memory efficiency
- [x] Parallel batch processing
- [x] Layer normalization (LayerNorm, RMSNorm)
- [x] Gating mechanisms (SiLU, GELU, GLU/SwiGLU/GeGLU)
- [x] Causal convolutions (CausalConv1d, DepthwiseCausalConv1d, DilatedStack)
- [x] Numerical stability (Kahan sum, Welford variance, safe exp/log)
- [x] Parallel scan (associative scan, O(log N) depth)
- [x] RetNet (Multi-Scale Retention)
- [x] S4D (Diagonal SSM with HiPPO)
- [x] Griffin (Gated Linear Attention)
- [x] GPU device abstraction (DeviceConfig, DeviceType) ✅
- [x] GPU memory management (TensorTransfer, MemoryStats, GPUMemoryPool) ✅
- [x] Training infrastructure with GPU support (TrainableSSM) ✅

#### kizzasi-model
- [x] ModelType enum (Mamba, Mamba2, RWKV, S4, S4D, Transformer)
- [x] AutoregressiveModel trait
- [x] Mamba implementation with selective SSM ✅
- [x] Input-dependent B, C parameters for Mamba ✅
- [x] Optimized ZOH discretization with Taylor approximation ✅
- [x] Mamba2 with SSD (State Space Duality)
- [x] RWKV v6 implementation
- [x] S4D implementation
- [x] Transformer baseline
- [x] SafeTensors loader with weight loading methods ✅

#### kizzasi-tokenizer
- [x] SignalTokenizer trait
- [x] ContinuousTokenizer
- [x] MuLawCodec (8-bit/16-bit)
- [x] LinearQuantizer
- [x] VQ-VAE with EMA updates
- [x] Residual VQ (RVQ)
- [x] Multi-scale tokenizer
- [x] Pyramid tokenizer with residual encoding
- [x] Advanced quantizers (Adaptive, DeadZone, NonUniform, Lloyd-Max)
- [x] Batch processing (BatchTokenizer, StreamingTokenizer)
- [x] Serialization (JSON, Bincode)

#### kizzasi-inference
- [x] InferenceContext (history + state management)
- [x] InferenceEngine (single-step prediction)
- [x] Pipeline builder pattern
- [x] Sampling strategies (Greedy, Temperature, Top-k, Top-p)
- [x] Beam search with constraints
- [x] Batch processing (BatchScheduler, continuous batching)
- [x] Checkpoint management
- [x] Metrics and profiling
- [x] Model registry
- [x] kizzasi-logic constraint integration ✅

#### kizzasi-logic
- [x] Constraint types (Range, LessThan, GreaterThan, Equals)
- [x] ConstraintBuilder
- [x] Guardrail enforcement
- [x] ConstrainedProjection
- [x] TemporalConstraint (rate-of-change limits)
- [x] ComposedConstraint (AND, OR, NOT, Implies)
- [x] LinearConstraint (Ax <= b)
- [x] QuadraticConstraint (x'Qx + c'x <= b)
- [x] SlidingWindowConstraint (mean, variance, trend)
- [x] LTL operators (Always, Eventually, Until, Release)
- [x] Soft/Hard constraint distinction
- [x] Penalty functions (L1, L2, Huber, LogBarrier)
- [x] Differentiable projection
- [x] Lagrangian relaxation
- [x] Dykstra's alternating projection
- [x] Batch constraint checking with caching

#### kizzasi-io
- [x] SignalStream trait
- [x] StreamConfig
- [x] MqttClient with TLS, QoS, reconnection
- [x] AudioInput/AudioOutput via cpal
- [x] Signal generators (sine, noise, chirp, etc.)
- [x] SignalProcessor (FFT, filtering)
- [x] IIR/FIR filters
- [x] Spectrogram computation
- [x] MFCC extraction
- [x] Wavelet transforms (DWT, SWT)
- [x] Ring buffer for real-time
- [x] Lock-free queues
- [x] Health monitoring
- [x] WebSocket stream
- [x] Serial port support
- [x] File I/O (WAV, CSV, HDF5)
- [x] OSC protocol
- [x] TCP/UDP sockets

---

### Phase 2: Production Readiness (IN PROGRESS)

#### Weight Loading & Model Compatibility
- [x] SafeTensors infrastructure ✅
- [x] Mamba weight loading from SafeTensors ✅
- [x] RWKV weight loading from SafeTensors ✅
- [x] Transformer weight loading from SafeTensors ✅
- [x] Mamba2 weight loading from SafeTensors ✅
- [x] S4D weight loading from SafeTensors ✅
- [x] 3D tensor loading for convolution weights ✅
- [x] Mamba2 convolution weight loading ✅
- [x] S4D convolution weight loading ✅
- [x] Comprehensive weight format documentation ✅
- [x] Weight inspection utilities (print_summary, search_tensors, get_size_stats) ✅
- [x] HuggingFace compatibility documentation ✅
- [x] Load Mamba weights from HuggingFace via NameRemapper (HF→internal key translation) ✅
- [x] Load RWKV weights from official releases (load_weights_json / JSON round-trip) ✅
- [x] Convert PyTorch checkpoints (pytorch_compat.rs name mapping + JSON I/O) ✅
- [x] Support GGUF format ✅
- [x] Incremental weight loading for large models

#### GPU Acceleration
- [x] CUDA backend via candle ✅
- [x] Metal backend for macOS ✅
- [x] Automatic device selection ✅
- [x] Mixed precision (FP16/BF16) ✅
- [x] DeviceConfig for CPU/CUDA/Metal ✅
- [x] GPU memory management utilities ✅
- [x] Tensor transfer utilities ✅
- [x] Memory pooling and tracking ✅
- [x] Flash-linear-attention kernel
- [x] Multi-GPU data parallel support

#### Training Infrastructure
- [x] DataLoader for time-series ✅
- [x] Training loop with constraint loss ✅
- [x] Checkpoint save/load ✅
- [x] Learning rate schedulers (7 types) ✅
- [x] Gradient clipping ✅
- [x] Metrics tracking and early stopping ✅
- [x] Curriculum learning

#### Performance Optimization
- [x] Profile and optimize hot paths (tracing spans + ProfilingRegistry)
- [x] Benchmark against PyTorch Mamba
- [x] Memory-efficient gradient checkpointing
- [x] Speculative decoding
- [x] Multi-modal input fusion

---

### Phase 3: Ecosystem Integration

#### Python Bindings
- [x] PyO3 wrapper for kizzasi ✅
- [x] NumPy array interop (scirs2-numpy) ✅
- [x] pip installable package (kizzasi on PyPI via maturin CI) ✅
- [x] Jupyter notebook examples

#### ROS2 Integration
- [x] ROS2 subscriber/publisher bridge
- [x] Sensor message conversion
- [x] Real-time control loop

#### Cloud Deployment
- [x] gRPC server for inference
- [x] REST API wrapper
- [x] Docker container
- [x] Kubernetes operator

---

### Phase 4: Edge Deployment

#### Embedded Support
- [x] no_std compilation
- [x] ARM64 optimization (NEON via aarch64 intrinsics)
- [x] Fixed-point quantization (INT8)
- [x] Model pruning
- [x] TensorRT/ONNX export

#### WebAssembly
- [x] WASM compilation target
- [x] Browser inference demo
- [ ] WebGPU acceleration

---

## Testing & Quality

### Current Coverage
- [x] Unit tests for core modules
- [x] Integration tests for pipeline
- [x] Benchmark suite (criterion)
- [x] Numerical stability tests

### Planned
- [x] Property-based tests (proptest) ✅
- [x] Fuzzing for input validation
- [x] CI/CD pipeline (GitHub Actions)
- [x] Performance regression tests
- [x] Cross-platform testing (CI matrix: Linux, macOS, Windows)

---

## Documentation

### Completed
- [x] README.md for all crates
- [x] TODO.md for all crates
- [x] KIZZASI_POLICY.md
- [x] API documentation (rustdoc)

### Planned
- [x] Architecture diagrams (Mermaid)
- [x] Tutorial: Getting Started
- [x] Tutorial: Building a Robotics Controller
- [x] Tutorial: Audio Processing Pipeline
- [x] Performance Tuning Guide
- [x] Migration Guide (from PyTorch)

---

## Use Cases & Applications

### Robotics Control
- Real-time motor control (100Hz+)
- Joint angle/velocity prediction
- Collision constraint enforcement
- Safety envelope guarantees

### Industrial Anomaly Detection
- Learn "normal" sensor patterns
- Real-time deviation detection
- Rate-of-change guardrails
- Predictive maintenance alerts

### Audio Processing
- Next-sample prediction (WaveNet-style)
- Real-time audio effects
- μ-law quantization
- Streaming synthesis

### Video Prediction
- Frame-to-frame prediction
- Anime in-betweening
- Skeleton constraint enforcement
- Cross-modal audio-video sync

---

## Notes

- **Naming**: "Kizzasi" (兆し) means "sign/omen/premonition" in Japanese
- **Philosophy**: AGSP treats all modalities as equivalent signal streams
- **Key Insight**: "Language Model" is a misnomer—these are "General-Purpose Signal Predictors"
- **Dependencies**: Following KIZZASI_POLICY.md (scirs2-core, tensorlogic, candle)

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| v0.1.0 | 2024-12 | Initial release, core SSM engine |
| v0.2.0 | 2026-03 | JSON weight I/O (save/load_weights_json all models), NameRemapper, factory injection, file splits (vqvae, training) |
| v0.3.0 | TBD | Training infrastructure |
| v1.0.0 | TBD | Production-ready, stable API |

---

*Last Updated: 2026-03-16*

## Proposed follow-ups

### Root TODO.md
- **`Pre-trained Models` (vague):** Needs concrete decisions: (1) Which architectures? Mamba-130M / Mamba-370M / RWKV-430M? (2) Training corpus — The Pile subset or custom multimodal? (3) Hosting: HF Hub mirror vs. self-hosted R2 bucket? (4) Naming under "Kizzasi-Takumi" — semver scheme?
- **`WebGPU acceleration` (oversized ~3000 LoC):** Proposed split into 4 future `/ultra` items:
  1. `kizzasi-webgpu` crate skeleton + `wgpu 22` + `WebGpuBackend::new()` + buffer upload/download (~500 LoC)
  2. `SsmBackend` trait in `kizzasi-core` + CPU default impl + `WebGpu` variant gated (~200 LoC)
  3. First WGSL kernel — SSM scan (Blelloch work-group=256) + dispatcher (~400 LoC Rust + ~150 LoC WGSL)
  4. Matvec + elementwise (silu, rms_norm) kernels + browser demo (~600 LoC Rust + ~200 LoC WGSL + HTML)
