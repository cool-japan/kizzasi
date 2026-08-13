# Kizzasi Development Roadmap

## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check)

- [x] `kizzasi-model`: `crates/kizzasi-model/src/gguf/mod.rs:384` — replace placeholder data_offset (0, patched below comment) with correct computation before the patching step (planned 2026-06-20)
  - **Goal:** GGUF tensor data offsets correct for any spec-valid `general.alignment` value; no `GgufTensorInfo` ever holds `data_offset: 0`.
  - **Design:** (1) `GGUF_DEFAULT_ALIGNMENT: usize = 32` const; (2) `GgufMetaValue::as_u64()` accessor (tolerant across unsigned/non-negative integer types); (3) resolve alignment from `metadata.get("general.alignment")` with power-of-two validation; (4) two-phase parse — raw descriptors collected first, cursor aligned by resolved alignment, then `GgufTensorInfo` built with `data_offset = data_section_start + offset` directly — no placeholder, no patch loop; (5) update doc comments.
  - **Files:** `crates/kizzasi-model/src/gguf/mod.rs`
  - **Tests:** `test_custom_alignment_64`, `test_invalid_alignment_rejected`, `test_alignment_metadata_as_uint32`; keep existing GGUF tests green.
  - **Risk:** Existing tests that hard-code 32-byte padding must remain green; the restructure must not regress default-alignment parsing.

## Production-Readiness Sweep (2026-08-10, /loop ultracode audit)

A 16-agent exhaustive audit (11 per-crate + 5 cross-cutting lenses) surfaced 344 deduplicated findings
(39 critical / 115 high / 158 medium / 32 low), triaged into 20 conflict-free work packages.
Full machine-readable findings: session scratchpad `findings_deduped.json` / `packages/*.json`.

### Critical findings (all scheduled for implementation)

**kizzasi**
- [x] `kizzasi/src/distributed.rs:263` — DistributedPredictor::set_guardrails is a silent no-op — safety constraints are never applied to any worker
- [x] `kizzasi/src/predictor.rs:137` — KizzasiConfig.weights_path is accepted everywhere but never loads any weights — models always run on random init
- [x] `kizzasi/src/predictor.rs:95` — ModelType selection is a complete no-op: every model type produces the identical SelectiveSSM, and the facade cannot reach kizzasi-model at all
- [x] `kizzasi/src/predictor.rs:302` — KizzasiConfig::model_type is stored but never dispatched — facade always runs SelectiveSSM
- [x] `kizzasi/src/predictor.rs:628` — Kizzasi::fork() re-randomizes all model weights instead of cloning them
- [x] `kizzasi/src/predictor.rs:302` — ModelType is inert: Mamba/Mamba2/S4/RWKV all build the identical SelectiveSSM
**kizzasi-core**
- [x] `kizzasi-core/src/config.rs:139` — weights_path is accepted by the builder and config file but never loaded — models silently run on random weights
- [x] `kizzasi-core/src/config.rs:101` — weights_path / load_weights is stored and never read — trained weights silently never load
- [x] `kizzasi-core/src/scan.rs:151` — Advertised O(log N) parallel SSM scan always falls back to the sequential scan because identity() returns None
- [x] `kizzasi-core/src/simd.rs:53` — dot_view silently returns 0.0 for non-contiguous views instead of erroring
- [x] `kizzasi-core/src/training_loop.rs:307` — LR scheduler is computed and logged but never applied to the optimizer — all schedulers are no-ops
**kizzasi-embedded**
- [x] `kizzasi-embedded/src/ssm.rs:155` — MambaStep treats `a_log` as raw A instead of log(-A): real Mamba weights diverge silently
**kizzasi-inference**
- [x] `kizzasi-inference/src/adapters/rest.rs:388` — REST /infer endpoint is a hardcoded mock: it never runs the inference engine
- [x] `kizzasi-inference/src/batch.rs:173` — BatchScheduler can never load a model — the entire continuous-batching module always errors
- [x] `kizzasi-inference/src/pool.rs:255` — TensorPool transmutes Vec<T> based on a caller-supplied dtype string — safe API causes UB / heap corruption
- [x] `kizzasi-inference/src/streaming.rs:191` — Batched streaming path busy-spins a core forever and its output stream never terminates
**kizzasi-io**
- [x] `kizzasi-io/src/compression.rs:402` — decompress_dpcm slices out of bounds on attacker/corruption-controlled predictor_order
- [x] `kizzasi-io/src/mqtt.rs:415` — MQTT never re-subscribes after a reconnect, so the client silently receives nothing forever
- [x] `kizzasi-io/src/mqtt.rs:277` — MQTT silently connects in plaintext when use_tls is true but no CA certificate path is set
- [x] `kizzasi-io/src/ros2.rs:199` — Ros2Stream drops every subscription immediately and read() always returns zeros
- [x] `kizzasi-io/src/ros2.rs:288` — Ros2Stream silently returns all-zero arrays forever: subscriptions are dropped immediately and the buffer is never filled
- [x] `kizzasi-io/src/signal/wavelets.rs:249` — WaveletAnalyzer::idwt does not invert dwt for any wavelet except Haar — IDWT/denoise output is garbage
- [x] `kizzasi-io/src/video.rs:1205` — VideoReader is a stub that reports success: never opens the source, always returns end-of-stream, fabricates metadata
- [x] `kizzasi-io/src/zeromq.rs:232` — ZmqStream::connect creates no socket; the default SUB pattern always errors; PULL/PUB build and destroy a socket per message; 7 config fields never read
**kizzasi-logic**
- [x] `kizzasi-logic/src/constraint/basic.rs:205` — Constraint::project() produces points that Constraint::check() rejects (strict bounds)
- [x] `kizzasi-logic/src/constraint_repair.rs:112` — ConstraintRepairer returns the unmodified input as the 'repaired point' and evaluates every constraint at point[0]
- [x] `kizzasi-logic/src/decomposition.rs:767` — BendersDecomposition is a fake solver: subproblem returns hardcoded constants and iterate() always reports converged with a zero solution
- [x] `kizzasi-logic/src/mpc.rs:457` — MPCController::project_control ignores the actual control constraints and clamps to a hardcoded [-10, 10]
**kizzasi-model**
- [x] `kizzasi-model/src/gguf/dequant.rs:113` — Legacy GGUF quant types emit elements in the wrong order (Q4_0/Q4_1/Q5_0/Q5_1/Q6_K)
- [x] `kizzasi-model/src/gguf_dequant.rs:89` — GGUF K-quant dequantization does not match the GGML block layout (Q2_K/Q3_K/Q4_K/Q5_K produce garbage)
- [x] `kizzasi-model/src/mamba.rs:404` — Mamba selective-SSM recomputes the B and C projections batch_size times (loop-invariant inner sum)
- [x] `kizzasi-model/src/rwkv.rs:309` — RWKV WKV recurrence is numerically wrong: scalar normalizer shared across channels, decay applied without the exp(-exp(w)) transform
**kizzasi-python**
- [x] `kizzasi-python/python/kizzasi/__init__.py:19` — Python package __init__.py exports only 2 of the 12 registered classes — 10 documented classes raise AttributeError
- [x] `kizzasi-python/src/lib.rs:70` — pymodule init symbol is PyInit_kizzasi but maturin installs the artifact as kizzasi._kizzasi, which CPython loads via PyInit__kizzasi
- [x] `kizzasi-python/src/predictor.rs:83` — Guardrails silently discard min_val whenever both min_val and max_val are given — lower bound is never enforced
**kizzasi-tokenizer**
- [x] `kizzasi-tokenizer/src/entropy/encoder.rs:354` — ArithmeticEncoder emits a fixed 12-byte payload with no renormalization — silently wrong decode plus u64 underflow panic
- [x] `kizzasi-tokenizer/src/entropy/encoder.rs:492` — RangeEncoder/RangeDecoder silently corrupt streams when any symbol probability is below 2^-14
**kizzasi-webgpu**
- [x] `kizzasi-webgpu/src/backend.rs:198` — All wgpu validation errors abort the process; the documented WebGpuError::Other path is unreachable
- [x] `kizzasi-webgpu/src/ssm_scan.rs:199` — ssm_scan_gpu silently returns wrong results for sequences > 256 elements

### Results (completed 2026-08-10, same day)

All 20 packages executed (17 parallel per-crate + 3 serial cross-crate) and the verification gate passed:

- **Outcomes:** 291 implemented, 18 honest-docs (API made truthful where real implementation is infeasible in pure Rust), 1 deferred (facade doctest wiring, below), ~34 absorbed by sibling packages in the same crate chain (re-verified individually).
- **All 39 critical findings resolved** (checkboxes above).
- **Tests:** 2,744 → 3,624 passed (+880; 619 new regression tests reported by implementers), 0 failed, 24 skipped (pre-existing #[ignore] only — verified no new #[ignore] was introduced).
- **Gates:** cargo build/clippy(--all-targets)/nextest/doc — all zero warnings; `cargo deny check bans` passes against the new workspace `deny.toml`.
- [ ] Deferred: facade doctest wiring (`#![doc = include_str!("../README.md")]` + converting `rust,ignore` fences) — docs-only, tracked for a later pass.
- [ ] Known-honest remaining gap: PEAQ NN weights are LCG-seeded placeholders behind `WEIGHTS_ARE_TRAINED = false` (ITU-R BS.1387-1 Annex 2 tables are paid); output documented as relative-only.

### Work packages (high/medium/low handled per package)

| Package | Model | Findings | Scope |
|---------|-------|---------:|-------|
| core-A | opus | 11 | kizzasi-core: training loop, SSM state, backend trait |
| core-B | sonnet | 35 | kizzasi-core: scan/parallel/SIMD/S5/misc |
| emb | opus | 22 | kizzasi-embedded: a_log, fixed-point |
| inf-A | opus | 18 | kizzasi-inference: engine/batch/streaming/pool/speculative |
| inf-B | sonnet | 21 | kizzasi-inference: REST, hotswap, LoRA, samplers |
| io-A | opus | 11 | kizzasi-io: zeromq/ros2/video adapters |
| io-B | sonnet | 22 | kizzasi-io: mqtt, compression, misc |
| io-C | opus | 4 | kizzasi-io: wavelets, STOI/PESQ |
| logic-A | opus | 7 | kizzasi-logic: Benders, MPC, constraint repair |
| logic-B | sonnet | 27 | kizzasi-logic: gpu_acceleration honesty, misc |
| macros | sonnet | 16 | kizzasi-macros: misc |
| model-A | opus | 7 | kizzasi-model: GGUF dequant, RWKV recurrence |
| model-B | sonnet | 26 | kizzasi-model: mamba perf, weight I/O, distributed, features |
| py | sonnet | 19 | kizzasi-python: exports, GIL, guardrails |
| tok-A | opus | 4 | kizzasi-tokenizer: entropy coders |
| tok-B | sonnet | 25 | kizzasi-tokenizer: misc |
| w2-crosscut | sonnet | 4 | cross-crate leftovers |
| w2-facade | opus | 41 | facade: ModelType dispatch, weights_path, OptimizationConfig wiring |
| w2-features | opus | 6 | workspace: cuda/metal/webgpu features, purity, deny.toml |
| webgpu | opus | 18 | kizzasi-webgpu: scan correctness, errors, thresholds |

### Wave 1 — kizzasi-io video-pure feature (2026-08-11)

Follow-up to the `io-A` package above (`kizzasi-io/src/video.rs:1205`, the stub `VideoReader`):
a from-scratch pure-Rust video backend, built in six packages (B1-B6) on top of the FFmpeg fix.

- [x] **B1 — dispatch facade**: `src/video.rs` (2152 lines) split into
  `src/video/{mod, types, processing, backend_ffmpeg, backend_pure, backend_pure_camera}.rs`;
  `VideoBackend` enum (`Auto`/`Ffmpeg`/`Pure`) added to `VideoConfig::backend`; `resolve_backend`/
  `resolve_auto` route per source (Y4M file / other file / camera / network) between the two
  backend implementations. Public `kizzasi_io::Video*` surface unchanged.
- [x] **B2 — Y4M pure path**: `backend_pure::PureReader` decodes Y4M end to end via
  `oximedia-container`'s `Y4mDemuxer` + `oximedia-core` conversion + `oximedia-cv` rescale, held to
  the same observable contract as `FfmpegReader` (frame indices, timestamps, decimation, buffering,
  `max_frames`) and enforced by `reader_tests` running its suite once per compiled-in backend.
- [x] **B3 — camera pure path**: `backend_pure_camera` opens a live device via `oximedia-capture`
  on Linux (V4L2) / macOS (AVFoundation) / Windows (Media Foundation), negotiating NV12 → YUYV/UYVY
  → planar 4:2:0/RGB24 → MJPEG in ascending conversion cost; `CameraDevice::list_devices()` real
  enumeration on every platform (Linux now `VIDIOC_QUERYCAP`-filtered, replacing the old
  `video`-only path's unconditional `/dev/video*` scan); `VideoConfig::from_camera`'s
  `camera_format` default-platform bug fixed (was hardcoded `"video4linux2"` on every OS).
- [x] **B4 — mock e2e tests**: camera-path tests drive `oximedia_capture::mock`, a scripted,
  deterministic capture backend, so the full suite (decimation, `max_frames`, tiny buffers, format
  conversion, seek refusal, a scripted fatal capture error) runs without real hardware.
- [x] **B5 — test-matrix hardening**: four-config audit (`video`; `video-pure` alone,
  `--no-default-features --features std,video-pure`; both; neither) — **291 / 335 / 344 / 267**
  tests passed respectively, 0 failed, 0 skipped, confirmed via `scripts/check-video-matrix.sh`
  (new; also runs `bench --no-run` on all four configs, `clippy --all-targets -D warnings` on
  `video,video-pure` and `video-pure` alone, `cargo doc`/`test --doc` with
  `RUSTDOCFLAGS="-D warnings"`, and `cargo deny check bans`). Added
  `reader_tests::test_auto_routes_y4m_to_pure_observably`: opens the same 4:2:0 Y4M file once via
  `Auto` and once forced onto `Pure` and asserts byte-identical frames -- chosen over the `Cmono`
  parity fixture because Pure/FFmpeg YUV→RGB conversion is *not* guaranteed to agree there, which
  is what gives the comparison discriminating power without a backend-introspection API. Confirmed
  all five `tests/integration_tests.rs` video cfg sites are already `any(feature = "video", feature
  = "video-pure")` (no fix needed). Confirmed the bench file's `#[cfg(feature = "video")]` /
  `#[cfg(not(feature = "video"))]` `criterion_main!` arms compile under all four configs as-is; left
  untouched (not broken) -- `bench_optical_flow` itself stays gated on `video` only, so it does not
  run under a `video-pure`-only build, a coverage gap rather than a compile break.
- [x] **B6 — documentation**: `src/video/mod.rs` module doc gained an explicit per-source routing
  table, a "what the pure backend does not do" list (OxiMedia's Red List: H.264/H.265/H.266/AAC
  excluded permanently on patent grounds, not "not yet"), and a verification-status paragraph.
  `crates/kizzasi-io/README.md` and root `README.md` coherence pass (see CHANGELOG `[0.2.3]` for
  the itemised changes); new "Testing the video backends" section in the crate README.

**Verification claim, precisely scoped:** Y4M decode and camera capture are exercised on every
test run via synthetic Y4M fixtures and `oximedia_capture::mock` (scripted, deterministic, no real
hardware or OS permission prompt touched). Real hardware is a narrower claim: while preparing B6,
`CameraDevice::list_devices()` was run against the real platform API on this workspace's macOS
development machine and correctly reached AVFoundation and surfaced a genuine TCC permission error
(`IoError::Connection`, message naming TCC) rather than fabricating a device list or panicking --
evidence the integration is wired to the real capture API, not evidence a real camera was
enumerated (this machine has not granted the process Camera access). Linux and Windows capture
have not been exercised against real hardware from this repository at all.

**Deferred (not attempted this wave -- upstream or out of scope, not silently dropped):**
- **Network-stream pure video.** `VideoSource::Network` stays FFmpeg-only. `oximedia-net` has a
  real RTSP 1.0 client/server and RTP packet parsing (`RtpPacket::parse`, RFC 3550), but
  `kizzasi-io` does not depend on `oximedia-net` at all today, and RTP packet parsing is not the
  same thing as the per-codec RTP depacketizers a decode pipeline needs on top of it (`oximedia-net`
  itself documents that depacketization "lives in the codec depacketizers", a separate piece).
  Wiring this up is a multi-step upstream dependency, not a kizzasi-io-local fix -- and even once
  wired, it would only ever cover patent-free codecs the Red List allows, which most legacy RTSP
  cameras do not emit.
- **O(1) Y4M seek.** `PureReader::seek` reopens the file and drains frames sequentially back to the
  target (`backend_pure.rs`'s `reopen` + drain, see its `debug!` log message) rather than computing
  a byte offset directly. Y4M's fixed per-frame size after the header makes an O(1) seek
  computable in principle; `oximedia-container`'s `Y4mDemuxer` does not expose that today, so doing
  it properly is upstream work, not a kizzasi-io-local one.
- **NV12 cross-path byte-equality.** The Y4M/container path's YUV→RGB (`oximedia-core`'s
  `convert::pixel::yuv420_to_rgb`, BT.601 fixed-point scaled ×1024) and the camera/capture path's
  NV12→RGB (`convert::simd_pixel::nv12_to_rgb24`, a separate `YuvCoeffs`-based fixed-point
  implementation) are two independent converters in `oximedia-core`. Unifying them so the same
  conceptual YUV 4:2:0 sample produces byte-identical RGB regardless of which path decoded it is
  upstream `oximedia-core` work; `reader_tests` already documents (rather than hides) the resulting
  cross-backend colour tolerance for 4:2:0 content.
- **Real-device smoke testing.** Left to whoever runs this on hardware with a camera attached and
  permission granted. `oximedia-capture` itself already has this convention one layer down --
  `OXIMEDIA_CAPTURE_DEVICE`-gated `#[ignore]`d tests in its own `tests/live_capture.rs` -- but
  `kizzasi-io` has no equivalent test of its own yet; adding one (or documenting running
  `oximedia-capture`'s directly) is future work, not done this wave.

## Project Overview

**Kizzasi** (兆し) - Autoregressive General-Purpose Signal Predictor (AGSP)

A Rust-native system for predicting continuous signal streams (audio, sensors, video, control signals) using State Space Models with neuro-symbolic constraint enforcement.

---

## Current Status (v0.2.4)

### Codebase Metrics

| Metric | Value |
|--------|-------|
| Total Lines | ~170,200 Rust |
| Crates | 12 |
| Test Count | 3,688 passed, 24 skipped (workspace, all-features) |
| Coverage | Core paths + comprehensive |
| Last Updated | 2026-08-12 |

### Implementation Status by Crate

| Crate | Status | Completion |
|-------|:------:|:----------:|
| kizzasi-core | Stable | 90% |
| kizzasi-model | Stable | 90% |
| kizzasi-tokenizer | Stable | 85% |
| kizzasi-inference | Stable | 85% |
| kizzasi-logic | Stable | 90% |
| kizzasi-io | Stable | 80% |
| kizzasi | Stable | 85% |
| kizzasi-webgpu | Alpha | 80% |
| kizzasi-embedded | Stable | 75% |
| kizzasi-macros | Stable | 100% ✅ |
| kizzasi-metal | Stable | 100% ✅ |
| kizzasi-python | Alpha | 80% |

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
- [x] Multi-speaker tokenization (k-means++ codebook, EMA updates, encode_blind, re_target)
- [x] Perceptual quantization (24-band Bark + Terhardt ATH, Hann-windowed STFT/ISTFT via oxifft)
- [x] PEAQ Basic Model (ITU-R BS.1387-1): 109-band ear model, 11 MOVs, 11→3→1 MLP, ODG in [-4, 0]

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

#### kizzasi-webgpu
- [x] WebGpuBackend with wgpu 29 (pure-Rust WGSL kernels)
- [x] Blelloch work-efficient SSM prefix scan (WGSL, work-group 256)
- [x] Matvec, SiLU, RMS-norm WGSL kernels
- [x] GpuBuffer (storage + staging), buffer upload/download
- [x] SsmBackend trait in kizzasi-core + CpuSsmBackend default impl
- [x] Feature-gated: `webgpu = ["dep:wgpu"]`; default features 100% Pure Rust

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
- [x] WebGPU acceleration (kizzasi-webgpu crate, wgpu 29, see WebGPU Acceleration section below)

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
| v0.2.2 | 2026-08-09 | WebGPU acceleration (kizzasi-webgpu), real tensorlogic-ir constraint pipeline, PEAQ perceptual audio quality evaluation, workspace-wide numerical/logic correctness sweep |
| v0.2.3 | 2026-08-12 | 100% C-free default build (candle onig→fancy-regex patch), pure-Rust `video-pure` feature (OxiMedia Y4M decode + camera capture), pure `mqtt-tls` (rustls + RustCrypto provider), new `kizzasi-metal` crate for cross-platform-safe Apple Metal activation |
| v0.3.0 | TBD | Training infrastructure |
| v1.0.0 | TBD | Production-ready, stable API |

---

*Last Updated: 2026-08-12*

### v0.2.x iteration follow-up (2026-05-17)

Inline-TODO cleanup pass. Four independent tracks merged via parallel sub-agents; workspace `cargo clippy -D warnings` and `nextest run --workspace --exclude kizzasi-io` both green.

- **Track A — Range coder** (`kizzasi-tokenizer/src/entropy.rs`): rewrote `RangeEncoder` and `RangeDecoder` with LZMA-style carry-propagation (cache byte + pending-`0xFF` counter). Two `#[ignore]`d tests reactivated; +6 new tests including 10k-symbol uniform-256 alphabet, 99/1 skew, and deterministic random-table proptest. File 1394 → 1594 lines.
- **Track B — INT8 quantization** (`kizzasi-core/src/weights.rs`): replaced stub `quantize_tensor` with full affine quantizer tracking `(scale, zero_point)`; added `QuantizedTensor`, `PerChannelQuantizedTensor`, and `dequantize_*` companions. +7 round-trip / per-channel-accuracy tests. File 438 → 932 lines.
- **Track C — Compression tests** (`kizzasi-inference/src/compression.rs`): added 10 unit tests for `StateCompressor` covering dense/sparse/8-bit roundtrips, multi-step drift, threshold boundary, and edge cases. File 397 → 694 lines.
- **Track D — Training loop** (`kizzasi-core/src/training_loop.rs`): real `compute_grad_norm` via `candle_core::backprop::GradStore` iteration over `VarMap::all_vars`; real `fit()` batch iteration via `TimeSeriesDataLoader::iter_batches`. +5 tests including convergence on AR(1) synthetic series. File 1005 → 1383 lines.

**Deferred (rationale documented in source):** perceptual.rs PEAQ weight verification (blocked on ITU-R BS.1387-1 Annex 2 procurement).

### v0.2.x iteration follow-up #2 (2026-05-17)

Discovered the `scirs2-core` parallel API was already mature in `0.4.4` (which kizzasi pins), unblocking the three "future-blocked" TODOs. Three more parallel tracks merged + one pre-existing flaky test repaired.

- **Track E — Parallel SSM scan + attention** (`kizzasi-core/src/scan.rs`, `efficient_attention.rs`): replaced sequential fallbacks with `scirs2_core::parallel_ops::IntoParallelIterator` for `parallel_ssm_batch` and `forward_parallel`; documented limitation that `parallel_scan_impl`'s Blelloch path only activates when `AssociativeOp::identity()` is `Some` (SSM op returns `None` because the identity is shape-dependent — falls back to sequential). +3 bit-exact / 1e-6 tolerance reference tests.
- **Track F — FilterTransformer** (`kizzasi-inference/src/streaming.rs`): added `FilterTransformer<I, F: Fn(&I) -> bool>` with `Arc<F>` cloning; wraps `tokio_stream::StreamExt::filter` with synchronous `futures::future::ready` so no `&I` is held across `await`. +4 tests (basic / passes-all / passes-none / preserves-order). File 621 → 707 lines.
- **Track G — kizzasi-embedded expansion** (`crates/kizzasi-embedded/`): new `src/presets.rs` (192 lines) with `stm32h7()`, `rp2040()`, `esp32c3()` factories tuned to flash/RAM budgets; 3 examples (`no_std_basic`, `fixed_point_cortex_m`, `stm32h7_preset`); new `tests/integration.rs` (224 lines) with 7 integration tests including INT8 round-trip on `SsmState` and f32-vs-Q16 fixed-point agreement; +16 tests total (25 → 41). Crate maturity 75% → ~95%.
- **Test fix** (`kizzasi-model/tests/model_comparison.rs:104`): `test_state_persistence` was non-deterministic on master (pre-existing) — saved state AFTER step then asserted re-applying step produced identical output. Repaired with snapshot-then-reset-then-double-restore pattern (same as the sibling `comprehensive_tests::test_state_persistence`) that validates determinism without depending on lossy `get_states`/`set_states` round-trip fidelity. Logged a hint about the underlying state-round-trip fidelity gap in a code comment.

**Workspace status post-iteration:** `cargo check --workspace --exclude kizzasi-io --all-features` clean; `cargo clippy --workspace --exclude kizzasi-io --all-features --no-deps --tests --examples -- -D warnings` clean; `cargo nextest run --workspace --exclude kizzasi-io --all-features` → **2255 passed, 0 failed, 24 skipped**.

**Still deferred:** perceptual.rs PEAQ weight verification (procurement).

### v0.2.x iteration follow-up #3 (2026-05-17)

Three more independent tracks merged via parallel sub-agents.

- **Track H — Mamba state round-trip fidelity** (`kizzasi-core/src/conv.rs`, `kizzasi-model/src/mamba.rs`, `kizzasi-model/tests/model_comparison.rs`): root cause of the lossy round-trip was `CausalConv1d::get_history` truncating to `kernel_size - 1` frames while `forward_step` momentarily kept `kernel_size` frames between push and trim. Fixed by returning the full buffer; `set_history` now validates length ≤ `kernel_size` and per-frame channel dim, returning `CoreResult<()>` (previously panicked). Same fix applied to `DepthwiseCausalConv1d`. `Mamba::set_states` now propagates errors via `ModelError::load_error` instead of panicking. +5 tests including the strict `test_state_roundtrip_fidelity` (`<1e-6` agreement across 5+ steps + advance + restore).
- **Track I — kizzasi-embedded strict no_std build** (`crates/kizzasi-embedded/`): wired the dormant `libm` optional dep into the feature matrix. Added a `core_math` shim module (`std` → `f32::*`, `libm` → `libm::*`) used by `math.rs` (`sqrt`) and `quantize.rs` (`round`). Made `std = ["alloc"]` so `extern crate alloc` is gated cleanly; brought `vec!` into scope under `not(feature = "std")`. Crate-level `compile_error!` guards the unsupported `no_std + no_libm` configuration with a clear error message. Build matrix now: `default (std)` ✓, `--no-default-features --features alloc,libm` ✓, `--no-default-features --features alloc,libm,fixed-point` ✓.
- **Track J — Rustdoc warnings cleanup** (5 files): cleared all 20 rustdoc warnings (12 in `kizzasi-core/numerics.rs` array-index literal links, 1 in `kizzasi-core/conv.rs` (`Self::set_history`), 1 in `kizzasi-model/pytorch_compat.rs`, 1 in `kizzasi-embedded/math.rs`, 2 in `kizzasi-inference/adapters/mqtt.rs`). `cargo doc --workspace --exclude kizzasi-io --all-features --no-deps` now emits **zero warnings**.

**Workspace status post-iteration:** `cargo check --workspace --exclude kizzasi-io --all-features` clean · `cargo clippy --workspace --exclude kizzasi-io --all-features --no-deps --tests --examples -- -D warnings` clean · `cargo doc --workspace --exclude kizzasi-io --all-features --no-deps` clean (0 warnings) · `cargo nextest run --workspace --exclude kizzasi-io --all-features` → **2260 passed, 0 failed, 24 skipped** · `cargo build -p kizzasi-embedded --no-default-features --features alloc,libm` clean.

**Still deferred:** perceptual.rs PEAQ weight verification (procurement); LoRA/Sampling Python wrappers (incremental — deferred to next Python track).

### v0.2.x iteration #4 (2026-05-17)

Four parallel enhancement tracks: Python expansion, benchmark suite, e2e test, refactor.

- **Track K — Python bindings expansion** (`kizzasi-python/`): split `lib.rs` (706 → 69 lines) into modular `config.rs`, `predictor.rs`, plus new `ensemble.rs` (`PyEnsemblePredictor` wrapping `kizzasi::ensemble::EnsemblePredictor` with average/weighted/median voting), and `optimized.rs` (`PyOptimizedPredictor` wrapping `kizzasi::optimization::OptimizedPredictor` with cache TTL + stats). 3 Python example files (`basic.py`, `ensemble.py`, `optimized.py`). +21 net Python tests (41 total in kizzasi-python — config 10, predictor 10, ensemble 11, optimized 10).
- **Track L — Benchmark suite** (4 new bench files): `kizzasi-core/benches/parallel_scan_bench.rs` (208 lines) sequential vs parallel scan + SSM scan; `kizzasi-core/benches/int8_quantization_bench.rs` (249 lines) per-tensor + per-channel quantize/dequantize at 64²/256²/1024²; `kizzasi-tokenizer/benches/range_coder_bench.rs` (216 lines) encode/decode bytes/sec across n × skew; `kizzasi-inference/benches/state_compression_bench.rs` (271 lines) all 5 CompressionMethod variants × density. Plus repaired pre-existing `kizzasi-tokenizer/benches/comprehensive_benchmarks.rs` (SIMD type renames from struct→fn). `cargo bench --workspace --no-run` now clean.
- **Track M — End-to-end pipeline test** (`crates/kizzasi/tests/end_to_end_pipeline.rs`, 160 lines): 3 tests — signal pipeline (sine → Mamba inference), constrained inference (`ConstraintBuilder::new().in_range(-1, 1).build()` projection), streaming with `FilterTransformer`. All passing in ~0.9s.
- **Track N — Refactor near-limit files**: `kizzasi-tokenizer/src/entropy.rs` (1594 lines) split into `entropy/{mod.rs (725), encoder.rs (548), decoder.rs (373)}`. `kizzasi-model/src/gguf.rs` (1596 → 1473 lines) with extracted `binary_io.rs` (231 lines, 4 new tests, also reused in gguf alignment logic). Public APIs unchanged.

**Workspace status post-iteration #4:** `cargo check --workspace --exclude kizzasi-io --all-features` clean · `cargo clippy --workspace --exclude kizzasi-io --all-features --no-deps --tests --examples --benches -- -D warnings` clean · `cargo doc --workspace --exclude kizzasi-io --all-features --no-deps` zero warnings · `cargo nextest run --workspace --exclude kizzasi-io --all-features` → **2291 passed, 0 failed, 24 skipped** · `cargo bench --workspace --exclude kizzasi-io --no-run` clean · `cargo build -p kizzasi-embedded --no-default-features --features alloc,libm` clean.

**Still deferred:** PEAQ weight procurement; documentation guides (architecture_overview, cross_modal_integration, deployment_and_serving — defer to a docs iter).

### v0.2.x iteration #5 (2026-05-17)

Four parallel tracks: Python LoRA+Sampling, gguf refactor, hot-path tracing, crate examples.

- **Track O — LoRA + Sampling Python wrappers** (`kizzasi-python/src/{lora,sampling}.rs` + 2 examples): `PyLoRAAdapter` (316L) exposes `LoRAAdapter::{add_layer, merge_all, unmerge_all, total_parameters, avg_parameter_ratio}`; `PySamplingConfig` + `PySampler` (458L) covers Greedy/Temperature/TopK/TopP (BeamSearch/Custom/Adaptive deferred). +19 tests (kizzasi-python: 41 → 60).
- **Track P — gguf.rs refactor** (`crates/kizzasi-model/src/gguf/`): split into `mod.rs` (1134L, top-level types + parser + tests) and `dequant.rs` (343L, block-wise dequantization). Old `gguf.rs` removed. Public API unchanged. `incremental_loader.rs`'s `crate::gguf::dequant` import path preserved.
- **Track Q — Hot-path tracing instrumentation** (`crates/kizzasi-core/src/{scan,efficient_attention,simd}.rs`): 12 functions instrumented. DEBUG level for top-level paths (`parallel_scan_impl`, `parallel_ssm_scan`, `parallel_ssm_batch`, `EfficientMultiHeadAttention::forward`, `FusedAttentionKernel::{forward,forward_parallel}`); TRACE level for SIMD inner-loop kernels (`matvec`, `ssm_state_update`, `layer_norm`, `softmax`, `online_softmax`, `fused_softmax_attend`). +1 test (`test_parallel_scan_with_tracing`). No new deps.
- **Track R — Crate examples** (3 new): `kizzasi-tokenizer/examples/vqvae_roundtrip.rs` (123L, codebook-size sweep showing monotonic MSE improvement), `kizzasi-tokenizer/examples/perceptual_quant.rs` (108L, Bark-band psychoacoustic quantizer demo), `kizzasi-macros/examples/kizzasi_config_derive.rs` (90L, derive macro happy + error paths). All build, run, produce expected output.

**Workspace status post-iteration #5:** `cargo check` ✓ · `cargo clippy --workspace --exclude kizzasi-io --all-features --no-deps --tests --examples --benches -- -D warnings` ✓ · `cargo doc --workspace --exclude kizzasi-io --all-features --no-deps` zero warnings ✓ · `cargo nextest run --workspace --exclude kizzasi-io --all-features` → **2311 passed, 0 failed, 24 skipped** ✓ · `cargo bench --workspace --exclude kizzasi-io --no-run` clean ✓ · `cargo build -p kizzasi-embedded --no-default-features --features alloc,libm` clean ✓.

**Workspace status post-iteration #6 (2026-05-17):** Same checks all pass · `cargo nextest run` → **2323 passed, 0 failed, 24 skipped** ✓.

**Iteration #6 completed tracks:**
- **Track S** — Python beam search wrappers: `PyBeamSearch`, `PyConstrainedBeamSearch`, `PyRejectionSampler` added to `kizzasi-python/src/beam_search.rs` (696L) with Python-callable constraint functions via `Python::attach`; 12 new tests; registered in `lib.rs`.
- **Track T** — `rwkv7.rs` (1431L) split into `rwkv7/mod.rs` (1102L) + `rwkv7/time_mixing.rs` (217L) + `rwkv7/channel_mixing.rs` (132L); public API unchanged via re-exports.
- **Track U** — Architecture documentation: `docs/architecture_overview.md` (566L) and `docs/cross_modal_integration.md` (488L) added.

**Future iteration candidates:** ~~`deployment_and_serving.md` doc guide~~ ✅; ~~multimodal_fusion integration test~~ ✅; ~~HF weight roundtrip test~~ ✅; ~~attribute parsing for `#[derive(KizzasiConfig)]` (kizzasi-macros)~~ ✅; pre-trained model zoo design decisions (blocked on human decisions: architecture, corpus, hosting).

### v0.2.x iteration #7 (2026-05-30)

Four parallel tracks — all four "future iteration candidates" from iteration #6 completed.

- **Track 1 — kizzasi-macros real attribute parsing** (`crates/kizzasi-macros/src/`): replaced the three stub derive macros (which ignored their attributes and `panic!`-ed on errors) with a fully modular implementation: `attrs.rs` (shared `parse_config_field_attrs` / `parse_preset_attr` helpers via `parse_nested_meta`), `config.rs` (builder codegen with `#[config(default=EXPR)]` / `#[config(validate="fn")]` / `#[config(skip)]`), `preset.rs` (per-name `{name}_preset()` constructors fixing the duplicate-method bug, with unknown-field validation and duplicate-name detection), `instrumented.rs` (`#[metrics]` field discovery with `collector`-name fallback, generics support). Tests: `tests/config_derive.rs` (9 tests) + `tests/preset_derive.rs` (4 tests). `Instrumented` tested from `crates/kizzasi/tests/instrumented_derive.rs` (2 tests, avoids circular dep). Example updated to exercise all three `#[config(...)]` features.
- **Track 2 — Multimodal fusion integration test** (`crates/kizzasi-inference/tests/multimodal_integration.rs`, 10 tests): first `tests/`-dir integration test for multimodal fusion; covers EarlyFusion (equal+mixed dims), WeightedFusion, MaxPooling, CrossAttention, multi-step state maintenance, reset, dimension-mismatch error, unknown-modality error, preprocessor. All span `kizzasi-inference` + `kizzasi-model` (S4D).
- **Track 3 — HF weight roundtrip test** (`crates/kizzasi-model/tests/hf_weight_roundtrip.rs`, 15 tests): `NameRemapper` key-translation tests (9, including all known HF→internal rules), Mamba JSON save/load cycle (key count=18, finite step output), `fill_lm_head_from_embedding` complementary tests (3). No network/hf-hub/pth required.
- **Track 4 — Documentation** (`docs/deployment_and_serving.md`, 330L): REST (`RestAdapter`/`RestServer`, 3 endpoints), gRPC (`GrpcAdapter`/`GrpcServer`/`InferenceService`), WebSocket, MQTT, Docker (2-stage Dockerfile, ports 8080+50051), docker-compose (kizzasi + mosquitto). Honest note that there is no default server binary. Cross-linked from `architecture_overview.md` §7 and §10.

**Workspace status post-iteration #7:** `cargo check --workspace --exclude kizzasi-io --all-features` clean · `cargo clippy --workspace --exclude kizzasi-io --all-features --no-deps --tests --examples --benches -- -D warnings` clean · `cargo doc --workspace --exclude kizzasi-io --all-features --no-deps` zero warnings · `cargo nextest run --workspace --exclude kizzasi-io --all-features` → **2363 passed, 0 failed, 24 skipped** (+40 new tests: 9 config_derive + 4 preset_derive + 2 instrumented + 10 multimodal + 15 HF-roundtrip).

**Still blocked (no action possible without human decisions):** PEAQ weight verification (ITU-R BS.1387-1 Annex 2 procurement); pre-trained model zoo (architecture/corpus/hosting decisions); reference-impl latency comparison (real PyTorch fixture needed).

### v0.2.x iteration #8 (2026-05-30)

Five parallel tracks: three correctness fixes + two integration test files + training documentation.

- **Track 1 — kizzasi-core: correct dense matrix exponential** (`crates/kizzasi-core/src/numerics.rs`): replaced the broken `pade_coefficients` (wrong values), `solve_linear` (fixed 10-iteration Richardson step), and `matrix_exp_pade` (first-order fallback) with mathematically correct implementations. `pade_coefficients` now returns Higham 2005 Table 10.4 Padé\[13,13\] coefficients; `solve_linear` implements LU decomposition with partial pivoting + forward/back substitution (total function, near-zero pivot regularized by 1e-8); `matrix_exp_pade` correctly assembles U (odd powers) / V (even powers) and solves `(V-U)X=(V+U)`. `zoh_discretize` (dense) rewritten to use exact ZOH formula `B_d = A⁻¹(A_d-I)B` via the corrected `solve_linear`, with near-zero fallback. +8 tests: identity, diagonal, nilpotent, skew-symmetric (rotation), small-norm Taylor agreement, scaling-squaring, LU solve, exact scalar ZOH. Existing `test_zoh_discretize` updated to exact values.

- **Track 2 — kizzasi-model: Transformer KV-cache state round-trip fidelity** (`crates/kizzasi-model/src/transformer.rs`): `get_states` previously saved only K cache; `set_states` restored only K cache (comment: "simplified - in practice would restore both K and V"). Fixed: `get_states` now packs both K and V into a `[2·cache_len, hidden_dim]` array; `set_states` splits at `nrows/2`, restoring both caches faithfully. Empty-cache sentinel uses `step_count==0`. Odd row-count returns `ModelError::load_error` (no panic). Added `tests/transformer_state_roundtrip.rs` with 4 tests: empty-cache roundtrip, KV-cache fidelity (<1e-5), state count mismatch error, immediate restore preserves output.

- **Track 3 — kizzasi-inference: real LoRA adapter save/load** (`crates/kizzasi-inference/src/lora.rs`): `LoraAdapterLoader::load` previously returned `Array2::zeros((rank,128))` placeholders (hardcoded 128-dim). Fixed: added `save()` method + private `MatrixJson` serialization type (`{"shape":[rows,cols],"data":[[…]…]}`); rewrote `load()` to read real `lora_a.json`/`lora_b.json` with proper shape validation. Doc-comment updated to document the JSON format. +4 tests: full save/load roundtrip, matrix element equality, missing adapter → Err, config fields preserved.

- **Track 4 — kizzasi-inference: cross-crate integration tests for SpeculativeDecoder and ModelEnsemble** (`crates/kizzasi-inference/tests/`): Created `speculative_integration.rs` (9 tests with real S4D models, input_dim=1: token count, finite values, acceptance rate, identical-model greedy acceptance=1.0, reset_stats, different draft token counts, config builder chain, empty/single generation) and `ensemble_integration.rs` (11 tests: single/multi-model Average/Weighted/Voting/ProductOfExperts strategies, num_models, empty-models error, weight count mismatch, weights-not-summing-to-1 error, EnsembleBuilder API). Both use `kizzasi_model::s4::{S4Config, S4D}` for real cross-crate coverage.

- **Track 5 — Documentation: `docs/training_guide.md`** (836 lines, the largest previously undocumented workflow): 11 sections verified against source — TrainingLoop/TrainingConfig/TrainingResult/callbacks, LossFunction variants, Adam/SGD optimizers with gradient clipping, 7-type LR scheduler comparison table, CurriculumScheduler/CurriculumStrategy/CurriculumDataProvider, checkpointing, distributed GradientSync/DataParallelModel, constraint-aware loss integration, end-to-end example from `train_ssm.rs`. Cross-linked from `architecture_overview.md §10`.

**Workspace status post-iteration #8:** `cargo check --workspace --exclude kizzasi-io --all-features` clean · `cargo clippy --workspace --exclude kizzasi-io --all-features --no-deps --tests --examples --benches -- -D warnings` clean · `cargo doc --workspace --exclude kizzasi-io --all-features --no-deps` zero warnings · `cargo nextest run --workspace --exclude kizzasi-io --all-features` → **2401 passed, 0 failed, 24 skipped** (+38 new tests: 8 matrix-exp + 4 Transformer-roundtrip + 4 LoRA + 9 speculative + 11 ensemble + 2 existing-zoh-tests-updated).

**Still blocked (no action possible without human decisions):** PEAQ weight verification (ITU-R BS.1387-1 Annex 2 procurement); pre-trained model zoo (architecture/corpus/hosting decisions); reference-impl latency comparison (real PyTorch fixture needed).

### v0.2.x iteration #9 (2026-05-31)

Five parallel tracks: two correctness bugs fixed + two logic stubs implemented + two cross-crate integration test files.

- **Track 1 — kizzasi-core: FlashAttention multi-head + batch correctness** (`crates/kizzasi-core/src/flash_attention.rs`): Fixed two interlocked bugs masked by weak tests. `reshape_qkv` was a no-op returning `x.clone()` unchanged; `flash_attention_forward` was therefore interpreting `[batch, seq, d_model]` as `[seq_len, num_heads, head_dim]` — multi-head splitting never happened and all heads computed the same degenerate attention over the wrong axes. The `if batch_size==1 { … } else { … }` branches were textually identical, silently ignoring all batches past index 0. Fixed: `forward` now loops over the batch axis, slices each `[seq, d_model]` element, reshapes via `into_shape_with_order` to the correct `[seq, num_heads, head_dim]` layout, runs the (unchanged, correct) inner tiling kernel, then reshapes back and writes the result. Replaced `reshape_qkv` no-op with a real `slice_to_heads` helper. +4 tests using `FusedAttentionKernel::forward` as the reference oracle: single-head oracle agreement, multi-head oracle agreement (was failing before fix), batch independence, causal multi-head oracle agreement. All tolerances <1e-4.

- **Track 2 — kizzasi-model: real MoE experts + router** (`crates/kizzasi-model/src/moe.rs`): `Expert` was zero-initialized (`weights`, `input_proj`, `output_proj` all `Array2::zeros`) producing a zero vector for every input. `Router::new` was also zero-initialized despite its *"random initialization"* doc, making routing logits all-zero and routing degenerate. No weight-loading path exists, confirming these were pure stubs. Fixed: `Expert` replaced with a two-layer FFN (`w1: [input_dim, hidden_dim]`, SiLU activation, `w2: [hidden_dim, output_dim]`) using Glorot uniform init via `scirs2_core::random::{rng, RngExt}` (matching the `FeedForward::new` pattern in `transformer.rs`). `Router::new` likewise uses Glorot init. +4 tests: non-zero output per expert, two experts produce different outputs, router is input-dependent, MoE end-to-end non-zero output.

- **Track 3 — kizzasi-logic: two genuine stubs** (`crates/kizzasi-logic/src/constraint_repair.rs`, `online_learning.rs`, `constraint/basic.rs`): Fixed `ConflictResolver::might_conflict` (was `return false` unconditionally → `find_conflicts` always returned `[]`) with real feasible-interval disjointness: each `BoundType` maps to a `[lo, hi]` interval; two same-dimension constraints conflict iff their intervals are disjoint. Added `pub fn bound(&self) -> &BoundType` accessor to `Constraint` to expose the interval data. Fixed `ActiveConstraintBoundaryLearner::refine` (was a complete no-op) with a 5-epoch annealed-learning-rate perceptron update using the existing `shift_rhs` + `update_coefficients_towards` API on `LinearConstraint`. +7 tests: disjoint constraints detected as conflicting, overlapping constraints not flagged, different-dimension constraints not flagged, non-overlapping InRange pair detected, refine tightens when infeasible samples labeled, refine loosens when feasible samples labeled, refine no-ops with <2 labeled samples.

- **Track 4 — kizzasi-logic: MPC + constraint repair + online learning integration tests** (NEW `crates/kizzasi-logic/tests/control_and_learning_integration.rs`, 17 tests): First `tests/`-dir coverage for `MPCController`/`QuadraticCost`/`LinearDynamics`, `ConstraintRepairer`/`IISFinder`/`ConflictResolver`, and all five `OnlineLearning*` types. MPC: 1-D integrator solve, receding-horizon trajectory, warm_start/reset, predicted states, control constraint. Constraint repair: all four `RepairStrategy` variants, priority repair, IIS finder, conflict resolver resolve, no-violation fast path. Online learning: convergence, anomaly detection, feedback accumulation, end-to-end system, toggle flags, initial confidence validity.

- **Track 5 — kizzasi-model: Neural ODE cross-crate integration tests** (NEW `crates/kizzasi-model/tests/neural_ode_integration.rs`, 14 tests): First `tests/`-dir coverage for `NeuralOdeModel`, `AugmentedNeuralOde`, and `OdeIntegrator` accuracy beyond the inline scalar `dx/dt=-x` tests. Integrator: 2-D harmonic oscillator completes one period within 1% (<1e-3), RK4 < Euler accuracy comparison, RK4 convergence order (halving dt reduces error by ≥4×), AdaptiveRk45 accuracy <0.01. NeuralOdeModel: step finiteness, 10-step multi-step, state round-trip (<1e-3), reset restores fresh behavior, model_type/hidden_dim. AugmentedNeuralOde: dimension arithmetic, zero-augment error path, step finiteness, multi-step time advance.

**Workspace status post-iteration #9:** `cargo check --workspace --exclude kizzasi-io --all-features` clean · `cargo clippy --workspace --exclude kizzasi-io --all-features --no-deps --tests --examples --benches -- -D warnings` clean · `cargo doc --workspace --exclude kizzasi-io --all-features --no-deps` zero warnings · `cargo nextest run --workspace --exclude kizzasi-io --all-features` → **2447 passed, 0 failed, 24 skipped** (+46 new tests: 4 flash-attention oracle + 4 MoE non-zero + 7 logic-stubs + 17 control/learning integration + 14 neural-ode integration).

**Still blocked (no action possible without human decisions):** PEAQ weight verification (ITU-R BS.1387-1 Annex 2 procurement); pre-trained model zoo (architecture/corpus/hosting decisions); reference-impl latency comparison (real PyTorch fixture needed).

### v0.2.x iteration #10 (2026-05-31)

Five parallel tracks — two online-learning stubs made real, AVX-512 exp vectorised, A-weighted SNR implemented, two registry factory arms wired, high-order derivative finite differences completed.

- **Track 1 — kizzasi-logic: two `OnlineConstraintLearner` / `FeedbackConstraintTuner` stubs** (`crates/kizzasi-logic/src/online_learning.rs`): `OnlineConstraintLearner::refine_constraint` computed `update_scale` but discarded it with `let _ = update_scale;`. Fixed by calling `self.constraint.shift_rhs(update_scale)` and `update_coefficients_towards(sample_slice, update_scale.signum() * lr * 0.1)`. `FeedbackConstraintTuner::tune` computed `satisfaction_gap` but discarded it with `let _ = satisfaction_gap;`. Fixed by calling `self.constraint.shift_rhs(satisfaction_gap * self.adaptation_rate)` when gap > 0.1. Removed `#[allow(dead_code)]` from `adaptation_rate` field. +5 tests (refine loosens, refine tightens, no-update when correct, tuner loosens below target, tuner no-op below threshold).

- **Track 2 — kizzasi-core: AVX-512 `fast_exp_avx512_impl` real vectorised exp** (`crates/kizzasi-core/src/simd_avx512.rs`): `fast_exp_avx512_impl` was marked `#[target_feature(enable = "avx512f")]` but fell back to a scalar loop calling `x[i].exp()` per element. Replaced with a real Cody-Waite range-reduction + 5th-order Horner polynomial + `_mm512_scalef_ps` implementation: computes `n = round(x * log2e)` via `_mm512_roundscale_ps::<8>`, `r = x - n*ln2` via `_mm512_fnmadd_ps`, evaluates `e^r ≈ 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r/120))))` via nested `_mm512_fmadd_ps`, then scales by `2^n` via `_mm512_scalef_ps`. Processes 16 f32 values per SIMD iteration with scalar tail for remainders. +3 tests.

- **Track 3 — kizzasi-tokenizer: real A-weighted SNR** (`crates/kizzasi-tokenizer/src/metrics.rs`): `SnrMetrics::compute` copied `weighted_snr_db = segmental_snr_db` (placeholder). Replaced with real IEC 61672-1 A-weighting: added `fn a_weighting(f_hz: f32) -> f32` (standard four-pole formula with 20.6/107.7/737.9/12200 Hz poles) and `fn spectral_weighted_snr(orig, recon, segment_len)` that computes per-segment DFT, applies A-weighting at `f = k*44100/N` Hz, accumulates weighted signal/noise powers, and converts to dB. Falls back to unweighted when all A-weights are zero. +4 tests (DC returns 0.0, midrange peaks, weighted diverges from segmental, finite result).

- **Track 4 — kizzasi-inference: NeuralOde + MultiModal registry factory** (`crates/kizzasi-inference/src/registry.rs`): both `ModelType::NeuralOde` and `ModelType::MultiModal` previously returned `Err(PipelineConfig(...))`. Added real factory arms: NeuralOde builds `NeuralOdeConfig` with Rk4 solver, dt=0.01; MultiModal builds a single `Modality::Sensor` encoder with `FusionStrategy::Addition`. Both propagate `ModelError` through `InferenceError::ModelError`. +4 tests (registry creates NeuralOde, registry creates MultiModal, NeuralOde step finite, MultiModal step finite).

- **Track 5 — kizzasi-logic: `DerivativeConstraint` general n-th order finite differences** (`crates/kizzasi-logic/src/differential_constraints.rs`): `compute_derivative` returned `None` for `DerivativeOrder::Custom(n)` where n ≥ 4. Implemented general backward finite difference `∇ⁿ f(t) = sum_{k=0}^{n} (-1)^k C(n,k) f(t-k·dt)` with binomial coefficients computed via multiplicative recurrence, effective_n clamped to 8 for numeric stability. +3 tests (4th-order of quadratic ≈ 0, insufficient history returns None, impulse produces violation > 0).

**Workspace status post-iteration #10:** `cargo check --workspace --exclude kizzasi-io --all-features` clean · `cargo clippy --workspace --exclude kizzasi-io --all-features --no-deps --tests --examples --benches -- -D warnings` clean · `cargo doc --workspace --exclude kizzasi-io --all-features --no-deps` zero warnings · `cargo nextest run --workspace --exclude kizzasi-io --all-features` → **2463 passed, 0 failed, 24 skipped** (+16 new tests: 5 online-learning + 3 AVX-512 exp + 4 A-weighted SNR + 4 registry factory + 3 high-order derivatives).

**Still blocked (no action possible without human decisions):** PEAQ weight verification (ITU-R BS.1387-1 Annex 2 procurement); pre-trained model zoo (architecture/corpus/hosting decisions); reference-impl latency comparison (real PyTorch fixture needed).

### v0.2.x iteration #11 (2026-05-31)

Five parallel tracks fixing genuine correctness/functional gaps:

- **Track 1** (`crates/kizzasi-logic/src/violation_explanation.rs`): `MinimalViolatingSubsetFinder::find_mvs` was a placeholder returning all violated constraints. Replaced with gradient-based greedy shrink: computes central-finite-difference violation gradients per constraint, sorts by ascending violation magnitude, removes any constraint whose gradient is dominated (cosine similarity > 0.9) by another in the running MVS. Zero-gradient constraints are unconditionally kept. +5 tests.
- **Track 2** (`crates/kizzasi-core/src/ssm.rs`): `selective_scan_step` used hardcoded `delta = 0.1`. Added `dt_proj_vectors: Vec<Array1<f32>>` (one per layer) to `SelectiveSSM`; delta now computed as `softplus(dt_proj · x)` making it input-dependent (Mamba selectivity). Added `softplus` helper with large-x fast path. +4 tests.
- **Track 3** (`crates/kizzasi-tokenizer/src/specialized.rs`): `FourierTokenizer::fft()` and `ifft()` used O(n²) naive DFT. Replaced with Cooley-Tukey radix-2 DIT FFT (in-place, bit-reversal + butterfly, zero-pad to next power of 2). Added `fft_radix2`, `bit_reverse`, `next_power_of_2` helpers. +4 tests.
- **Track 4** (`crates/kizzasi-logic/src/decomposition.rs`): `BendersMasterProblem::solve_master` used `x = x.mapv(|v| v + 0.1)` for feasibility cut satisfaction — wrong (constant shift ignores cut gradient). Replaced with minimum-norm projected gradient step: `x += α·c / ||c||²` where `α = (-cut_value + 1e-6) / ||c||²`. +3 tests.
- **Track 5** (`crates/kizzasi-model/src/hybrid.rs`): `AttentionBlock::forward` was "simplified single-head version" — computed attention over full `[hidden_dim]` vectors ignoring `num_heads`/`head_dim`. Replaced with real multi-head attention: per-head Q/K/V slicing `[h*head_dim..(h+1)*head_dim]`, per-head scaled dot-product with KV cache, concatenated head outputs. Removed stale `#[allow(dead_code)]`. +4 tests.

**Workspace status post-iteration #11:** `cargo check` clean · `cargo clippy -D warnings` clean · `cargo doc` zero warnings · `cargo nextest run` → **2483 passed, 0 failed, 24 skipped** (+20 new tests: 5+4+4+3+4).

**Still blocked (no action possible without human decisions):** PEAQ weight verification (ITU-R BS.1387-1 Annex 2 procurement); pre-trained model zoo (architecture/corpus/hosting decisions); reference-impl latency comparison (real PyTorch fixture needed).

### v0.2.x iteration #12 (2026-05-31)

Five parallel tracks fixing genuine correctness bugs (each verified against an in-repo oracle or published standard):

- **Track 1** (`crates/kizzasi-model/src/s5.rs`): `S5Block::new` initialized `log_a[i] = -ln(i+1)` causing `a_bar = exp(dt/(i+1)) > 1.0` → diverging SSM dynamics. Replaced with HiPPO-LegS diagonal magnitudes `log_a[n] = ln((2n+1)/2)`, negated ZOH eigenvalue (`a_i = -exp(log_a[i])`), and proper `B̄` scale `B·(1−ā)/(−a)` — mirroring the working `s4.rs` oracle. Also fixed the duplicate buggy formula in `load_weights_json`. +4 tests.
- **Track 2** (`crates/kizzasi-core/src/s5.rs`): `init_a_matrix` set `a[i,i] = exp(log_lambda) > 0` → positive diagonal → `a_bar > 1` via ZOH → diverging. Negated diagonal (`-exp(log_lambda)`). Replaced the element-wise loop in `discretize_block` (which gave first-order off-diagonal approximation) with `numerics::zoh_discretize` (Padé[13,13] matrix-exp + LU-solved B̄). +4 tests.
- **Track 3** (`crates/kizzasi-logic/src/constraint_propagation.rs`): `BacktrackingSearch::backtrack` advertised forward checking (public builder `with_forward_checking(true)` is the default) but had an empty `if self.use_forward_checking { }` block. Wired the already-tested in-file `ForwardChecker::prune` into the backtrack loop: on domain wipeout skip branch; on success recurse. +3 tests asserting sound/complete solutions and non-inert flag.
- **Track 4** (`crates/kizzasi-tokenizer/src/advanced_quant.rs`): `NonUniformQuantizer::lloyd_max_gaussian` had wrong-sign inverse-CDF (non-monotonic bin edges: `z=+0.134` at `p=0.125` vs true quantile `−1.15`). Replaced with a real Lloyd-Max iteration: midpoint boundaries + truncated-Gaussian conditional centroids using `φ`/`Φ` (Abramowitz-Stegun 7.1.26 approximation). Converges in <200 iterations. Verified against Max-1960 N=8 table (boundaries ±0.501, ±1.050, ±1.748). +4 tests.
- **Track 5** (`crates/kizzasi-tokenizer/src/specialized.rs` + new `tests/transform_roundtrip.rs`): `WaveletTokenizer` and `DCTTokenizer` discarded `max_val` in `encode` but hardcoded `1.0` in `decode`, breaking round-trip for any signal with coefficient magnitude ≠ 1.0. Fixed by prepending `max_val` as `tokens[0]` (self-describing stream header); `decode` reads it back. Updated `embed_dim()` (+1) and all affected length assertions in `proptest_suite.rs`/`integration_tests.rs`. Added new `tests/transform_roundtrip.rs` with 8 round-trip and scale-equivariance tests.

**Workspace status post-iteration #12:** `cargo check` clean · `cargo clippy -D warnings` clean · `cargo doc` zero warnings · `cargo nextest run` → **2500 passed, 0 failed, 24 skipped** (+17 new tests: 4+4+3+4+8 – note Track 5 new file adds 8, others add inline tests).

**Still blocked (no action possible without human decisions):** PEAQ weight verification (ITU-R BS.1387-1 Annex 2 procurement); pre-trained model zoo (architecture/corpus/hosting decisions); reference-impl latency comparison (real PyTorch fixture needed).

### v0.2.x iteration #13 (2026-05-31)

Four parallel tracks fixing verified correctness bugs, each with an in-repo oracle or published standard:

- **Track 1** (`crates/kizzasi-model/src/mamba2.rs`): `ssd_step` state update used `b_x[j] * 0.01` — a hardcoded constant that dropped the per-head input entirely (`x[head_start..head_end]` never appeared on the RHS), making the state rank-degenerate and input-magnitude-insensitive. Replaced with the correct outer-product recurrence `new_h[i,j] = a_diag[j]*h[i,j] + x[head_start+i]*b_x[j]` (oracle: `mamba.rs:471-481`). Removed dead `else { 0.99 }` branch. +3 tests.
- **Track 2** (`crates/kizzasi-inference/src/sampling.rs`): `SamplingConfig.seed` (doc: *"Random seed for reproducibility"*) was completely ignored — `sample_categorical` always called `rng()` creating a fresh global RNG, so seeded samplers were non-reproducible. Added `rng: StdRng` field to `Sampler`; initialized via `seeded_rng(seed)` when seed is set. Also added `temperature ≤ 1e-6 → greedy` guard to prevent NaN from `logits/0`. +3 tests (reproducibility, different-seeds-differ, temperature-0=greedy).
- **Track 3** (`crates/kizzasi-inference/src/temporal.rs`): `STLFormula::robustness` Until arm computed the φ₁ infimum over `(start..i)` where `start = time + bound.lower`, but canonical quantitative STL (Donzé & Maler 2010) requires `inf_{t''∈[t,t']}` — starting at evaluation time `t`, not `t+a`. The fix: `(time as usize..i)`. Added 4 Until robustness tests (zero existed before), including a prefix-violation regression test.
- **Track 4** (`crates/kizzasi-logic/src/mpc.rs`): `project_control` used `clamp(-10.0, 10.0)` ignoring constraint geometry — a constraint `u ≤ 0.5` was never actually enforced. Replaced with a finite-difference projected-gradient step using only `ViolationComputable::violation()`: `x -= (viol / ||grad||²) * grad` (central differences, up to 20 iterations, early exit on satisfaction). Recovers `LinearConstraint::project` exactly for linear constraints. +1 test.

**Workspace status post-iteration #13:** `cargo check` clean · `cargo clippy -D warnings` clean · `cargo doc` zero warnings · `cargo nextest run` → **2511 passed, 0 failed, 24 skipped** (+11 new tests: 3+3+4+1).

**Still blocked (no action possible without human decisions):** PEAQ weight verification (ITU-R BS.1387-1 Annex 2 procurement); pre-trained model zoo (architecture/corpus/hosting decisions); reference-impl latency comparison (real PyTorch fixture needed).

### v0.2.x iteration #14 (2026-06-01)

Four parallel tracks fixing confirmed correctness bugs, each verified against a published standard or in-repo oracle:

- **Track 1** (`crates/kizzasi-model/src/h3.rs`): `ShiftSSM::forward` iterated the history `VecDeque` with `iter().enumerate()`, giving index 0 = oldest input. Standard causal FIR: `y[t] = sum_k w[k]*x[t-k]` binds `w[0]` to the current sample. The mapping was reversed (oldest got `row(0)`) and unstable during warm-up (the lag bound to `row(0)` slid as the buffer filled). Fix: `self.history.iter().rev().enumerate()` — index 0 = newest. Oracle: H3 paper (Fu et al. 2022); standard causal convolution. +3 tests.
- **Track 2** (`crates/kizzasi-model/src/spiking.rs`): `LifLayer::step` refractory branch wrote `state.voltages[i] * leak_factor` — this double-leaks (re-applying decay to an already-decayed voltage) and drops all synaptic input. Standard LIF model (Gerstner & Kistler §2.1; Dayan & Abbott Ch.1): during absolute refractory, membrane is **clamped at the reset value** from the spike. Fix: `updated_voltages[i] = state.voltages[i]` (hold the previously-stored reset value). +3 tests.
- **Track 3** (`crates/kizzasi-logic/src/incremental_solver.rs`): `repair_solution` finite-difference gradient loop perturbed `perturbed[i]` but evaluated `constraint.violation(perturbed[0])` for every `i`. `gradient[i] = 0` for all `i > 0` — only dimension 0 was ever repaired. `recompute_violations` had the same dimension-0 blindspot. Fix: use `constraint.dimension().unwrap_or(0)` to index the correct component in both methods. Also fixed `add_constraint`/`modify_constraint` which had the same bug. Oracle: `Constraint::dimension()` accessor in `constraint/basic.rs:220`. +3 tests.
- **Track 4** (`crates/kizzasi-logic/src/constraint_repair.rs`): `generate_suggestions_gradient` (= `compute_repaired_point`) had the identical dimension-0 gradient bug as Track 3. Additionally, the activation condition `violation > slack + self.tolerance` was never true since `slack` was initialized from the same violation, suppressing the descent entirely. Fix: dimension-aware index + changed condition to `violation > self.tolerance`. Oracle: same `Constraint::dimension()`. +2 tests.

**Workspace status post-iteration #14:** `cargo check` clean · `cargo clippy -D warnings` clean · `cargo doc` zero warnings · `cargo nextest run` → **2518 passed, 0 failed, 24 skipped** (+7 new tests: 3+3+3+2 — Track 3 agent also fixed the same bug in `add_constraint`/`modify_constraint`).

### v0.2.x iteration #15 (2026-06-01)

Three parallel tracks fixing confirmed correctness bugs, each verified against a published standard or in-repo oracle:

- **Track 1** (`crates/kizzasi-model/src/mamba.rs`): `SelectiveSsm` B̄ ZOH discretization — when `|a_n| < 1e-8` (near-zero eigenvalue, reachable from loaded checkpoint weights), the code substituted `safe_a_n = -1.0`, giving `(a_bar-1)/(-1)·B ≈ 0` (input coupling vanishes). The correct ZOH limit as `a_n → 0` is `Δ·B` (L'Hôpital: `(e^{Δa}-1)/a → Δ`). Fix: nested branch uses `delta[i] * b_vec[[i,n]]` when `|a_n| < 1e-8`; the bogus `-1.0` substitution is removed. Also fixed a pre-existing unused-Result clippy warning at line 1098 (`let _ = layer.conv.set_history(...)`). Oracle: ZOH discretization; pattern already in `crates/kizzasi-core/src/s4d.rs:197-200`. +2 tests.
- **Track 2** (`crates/kizzasi-tokenizer/src/mulaw.rs`): `MuLawCodec::quantize(1.0)` returned 256 — one past `vocab_size()=256`. The formula `((encoded+1)*half_levels).round()` yields 256 at the encoded=+1 boundary. Fix: clamp to `[0, levels-1]` (not rescale — that would break `dequantize`). Also updated the existing test that wrongly asserted 256 → now asserts 255. Oracle: token index must be `< vocab_size()`; standard symmetric quantization maps `[−1,1]` to `[0, levels-1]`. +1 test (1 existing test updated).
- **Track 3** (`crates/kizzasi-core/src/mamba2.rs`): `Mamba2Layer::forward` computed `a = self.a_log.mapv(|v| (-v.exp()).abs())`. The `.abs()` negated the negation: `|-exp(v)| = exp(v) > 0`. With default `a_log ∈ (−0.693, 0]`, this gave `a ∈ [0.5, 1.0)`, then `a_bar = exp(a·dt) ∈ (1.0, 1.105] > 1` — **unstable** (state diverges). Fix: remove `.abs()` → `a = -exp(a_log) < 0`, giving stable `a_bar ∈ (0, 1)`. Oracle: Mamba/SSM requires negative-definite A; every sibling (`mamba.rs:442`, `s5.rs:144`) uses `-log_a.exp()`. Existing tests masked the bug (too few steps with weak is_finite checks). +1 stability test (200-step bounded-output loop).

**Workspace status post-iteration #15:** `cargo check` clean · `cargo clippy -D warnings` clean · `cargo doc` zero warnings · `cargo nextest run` → **2522 passed, 0 failed, 24 skipped** (+4 new tests: 2+1+1).

**Still blocked (no action possible without human decisions):** PEAQ weight verification (ITU-R BS.1387-1 Annex 2 procurement); pre-trained model zoo (architecture/corpus/hosting decisions); reference-impl latency comparison (real PyTorch fixture needed).

### v0.2.x iteration #16 (2026-06-02)

Two parallel tracks fixing confirmed correctness bugs, each verified against an in-repo oracle:

- **Track 1** (`crates/kizzasi-model/src/backprop.rs`): `SsmBackward::backward` computed `dx[t,d]` as `(Σb_bar/N_state) · (Σdh_t/N_state)` — a product of two independent means — instead of the correct dot product `(b_bar.row(t)·dh_t)/input_dim`. Oracle (in-file sibling): lines 654-662 establish `x_scalar = x.row(t).mean()` and `db[t,n] = dh_t[n]*x_scalar`, confirming `∂h/∂x[t,d] = b_bar[t,n]/input_dim` and therefore `dx[t,d] = b_bar.row(t)·dh_t / input_dim`. Existing tests only checked shapes and `da_norm`, masking the bug. Fix: `let dx_scalar = b_bar.row(t).dot(&dh_t) / input_dim`. +2 tests (input_dim=1 case pins dot vs product; input_dim=2 case pins the `/input_dim` pooling factor).
- **Track 2** (`crates/kizzasi-logic/src/mpc.rs`): `MPCSolution::is_feasible()` returned `total_cost < INFINITY` — tautologically true for any finite cost, since constraint violations are folded in as finite penalties (`violation * 100.0`). Fix: add `constraint_violation: f32` field to `MPCSolution`, computed in `solve()` via a new `compute_raw_violation()` helper (same loop as `constraint_violation_cost` but without the ×100 factor); rewrite `is_feasible()` to `total_cost.is_finite() && constraint_violation <= 1e-4` (1e-4 matches codebase tolerance and is above `project_control`'s 1e-6 floor). Oracle: constraint_violation_cost's loop. Updated the in-file test constructor and all 7 `is_feasible()` call-sites in tests pass unchanged. +2 tests (direct construction: 0.0 vs 5.0 violation; nonfinite total_cost guard).

**Workspace status post-iteration #16:** `cargo check` clean · `cargo clippy -D warnings` clean · `cargo doc` zero warnings · `cargo nextest run` → **2525 passed, 0 failed, 24 skipped** (+3 new tests).

**Still blocked (no action possible without human decisions):** PEAQ weight verification (ITU-R BS.1387-1 Annex 2 procurement); pre-trained model zoo (architecture/corpus/hosting decisions); reference-impl latency comparison (real PyTorch fixture needed).

### v0.2.x iteration #17 (2026-06-02)

Two parallel tracks fixing confirmed correctness bugs, each verified against an in-repo oracle:

- **Track 1** (`crates/kizzasi-logic/src/distributed_admm.rs`): `QuadraticSubproblem::solve()` used `lb[0]`/`ub[0]` (first element only) in the partial-bound arms `(Some(lb), None)` and `(None, Some(ub))`, silently clipping every element to the first element's bound. Oracle: `box_clip` in the same file (lines 194–200) correctly uses element-wise zip iteration. Fix: replace `x.mapv(|v| v.max(lb[0]))` with `x.iter().zip(lb.iter()).map(|(xi, li)| xi.max(*li)).collect()` (and symmetric for upper bound). Also added `with_lower_bound` and `with_upper_bound` builder methods for the partial-bound case. +2 tests (lb=[1.0,5.0] — correct[1]=5.0, bug gives 1.0; ub=[3.0,6.0] — correct[1]=6.0, bug gives 3.0).
- **Track 2** (`crates/kizzasi-model/src/rwkv.rs`): `TimeMixing::forward` used raw `k[idx]` instead of `k[idx].exp()` in both the WKV state update (`new_wkv = w * wkv + k*v`) and the WKV normalizer update (`wkv_norm = w * wkv_norm + k`). Module-level doc (lines 41–47) explicitly states the recurrence uses `e^{k_t}`. With negative k, `wkv_norm` goes negative → clamped to 1e-8 → output explodes ~1e8. Oracle: `w = time_decay.exp()` on line 308 is the in-code sibling showing the same `.exp()` pattern. Fix: `let exp_k = k[idx].exp()` and substitute in both lines. +2 tests (`test_rwkv_wkv_exp_k_bounded_output`: key_proj.fill(-1.0) forces k≈-4 → pre-fix output ~1e8, post-fix bounded; `test_rwkv_wkv_positive_denominator`: public API finiteness guard).

**Workspace status post-iteration #17:** `cargo check` clean · `cargo clippy -D warnings` clean · `cargo doc` zero warnings · `cargo nextest run` → **2529 passed, 0 failed, 24 skipped** (+4 new tests).

**Still blocked (no action possible without human decisions):** PEAQ weight verification (ITU-R BS.1387-1 Annex 2 procurement); pre-trained model zoo (architecture/corpus/hosting decisions); reference-impl latency comparison (real PyTorch fixture needed).

### v0.2.x iteration #18 (2026-06-03)

Four parallel tracks fixing confirmed correctness bugs in three crates, each verified against a published or in-repo oracle:

- **Track 1** (`crates/kizzasi-tokenizer/src/continuous.rs`): `TrainableContinuousTokenizer::new()` called `.affine(0.0, enc_scale)` which computes `randn * 0 + enc_scale` — a constant tensor, defeating Xavier initialization entirely. Oracle: candle documentation — `.affine(mul, add)` computes `x * mul + add`; correct Xavier init is `.affine(enc_scale, 0.0)`. Fix: swap arguments on both encoder and decoder init lines. +2 tests (`test_xavier_init_encoder_not_constant`: encode one-hot input, assert output dims vary; `test_xavier_init_two_instances_differ`: two independent tokenizers must produce different encodings).
- **Track 2** (`crates/kizzasi-logic/src/decomposition.rs`): `ConsensusADMM::iterate()` accumulated `dual_residual` inside the `for i in 0..num_blocks` loop where `z_diff` is loop-invariant, giving `dual_res = rho * sqrt(num_blocks) * ‖z_diff‖` instead of the correct `rho * ‖z_diff‖`. Oracle: Boyd et al. (2010), "Distributed Optimization via ADMM", §3.3 eq (3.13). Fix: compute `dual_sq = z_diff.iter().map(|x| x*x).sum()` once outside the loop. +2 tests (`test_admm_dual_residual_single_block_baseline`: num_blocks=1, z_diff=[3,4], expected 5.0; `test_admm_dual_residual_multi_block_not_scaled`: num_blocks=3, rho=2, expected 2.0, bug gives 2*sqrt(3)≈3.46).
- **Track 3** (`crates/kizzasi-logic/src/visualization.rs`): `Colormap::Viridis` used an ad-hoc polynomial giving RGB (0,0,229) at t=0 vs reference (68,1,84) and (255,0,0) at t=1 vs reference (253,231,37) — completely wrong. Oracle: matplotlib Viridis 9-point reference anchors (published perceptually-uniform colormap). Fix: replaced polynomial with `viridis_lerp()` — piecewise linear interpolation over 9 anchors at t=0,0.125,...,1.0. +2 tests (endpoints ≤2 units from reference, midpoint t=0.5 ≤3 units from (33,145,140)).
- **Track 4** (`crates/kizzasi-model/src/training.rs`): `Optimizer::step()` Adam L2 path used `wd * param.data` (missing `lr` factor), giving per-step decay `= wd = 0.01` instead of correct `lr * wd = 0.00001` for typical lr=0.001, wd=0.01 (1000x too large). Also had an `return Ok(())` early exit that skipped the gradient update application. Oracle: AdamW path in the same function (lines 416–419) correctly uses `(1 - lr * wd)`. Fix: unified both paths to `param.data *= (1 - lr * wd)`, removed early return. +2 tests (`test_adam_l2_weight_decay_scale`: zero-grad step asserts residual ≈ 0.99999, not 0.99; `test_adam_vs_adamw_decay_match`: Adam L2 and AdamW must agree within 1e-7).

**Workspace status post-iteration #18:** `cargo check` clean · `cargo clippy -D warnings` clean · `cargo doc` zero warnings · `cargo nextest run` → **2537 passed, 0 failed, 24 skipped** (+8 new tests).

**Still blocked (no action possible without human decisions):** PEAQ weight verification (ITU-R BS.1387-1 Annex 2 procurement); pre-trained model zoo (architecture/corpus/hosting decisions); reference-impl latency comparison (real PyTorch fixture needed).

## Planned: tensorlogic-ir Integration (v0.2.x)

### Background
`kizzasi-logic` currently contains a hand-rolled `SymbolicExpr` (6 variants, 459 LoC) and a
custom bytecode stack-VM (`ConstraintExpr` → `CompiledConstraint`, 783 LoC). Both partially
duplicate functionality already present in `tensorlogic-ir`.

### Phase 1 – Replace `SymbolicExpr` with `TLExpr` (kizzasi-logic)
- [x] Add `tensorlogic-ir = "0.1.0"` to workspace + `kizzasi-logic` dependencies (completed 2026-04-27)
  - **Goal:** `tensorlogic-ir 0.1.0` resolvable as a workspace dep and consumed by `kizzasi-logic` via `tensorlogic-ir.workspace = true`.
  - **Design:** WIP already done. Workspace `Cargo.toml` uncomments `tensorlogic-ir = { version = "0.1.0" }`; `crates/kizzasi-logic/Cargo.toml` adds `tensorlogic-ir.workspace = true`.
  - **Files:** `Cargo.toml` (verified, +1/-1), `crates/kizzasi-logic/Cargo.toml` (verified, +1/-0).
  - **Prerequisites:** none.
  - **Tests:** `cargo check -p kizzasi-logic --all-features` must succeed.
  - **Risk:** dependency resolver may pull a transitive conflicting with `scirs2-core 0.4.2`.
- [x] **Rewrite `tensorlogic_integration.rs`** with `TLExprEvaluator` and Gödel logic (completed 2026-04-27)
  - **Goal:** Replace 6-variant local `SymbolicExpr` with full `tensorlogic_ir::TLExpr` evaluation. Soft (Gödel) semantics: And→min, Or→max, Not→1−x clamped, Imply→max(1−a,b) clamped.
  - **Design:** WIP already done (615 lines). Includes `TLExprEvaluator`, `tl_var()`, `tl_const()`, `ConstraintLearner.learn_box_constraints_as_expr()` returning `TLExpr`, and `ConstraintSynthesizer.synthesize_from_template()` returning `TLExpr`. Re-exports `TLExpr`, `Term as TlTerm`, `TypeAnnotation` from `tensorlogic_ir`.
  - **Files:** `crates/kizzasi-logic/src/tensorlogic_integration.rs` (verified, +383/-227).
  - **Prerequisites:** Item 1.
  - **Tests:** 8 unit tests in `tensorlogic_integration::tests` covering arithmetic, comparison, soft-And, evaluate_bool, learner box, learner-as-expr, linear separator, synthesis-box.
  - **Risk:** `LogicError::InvalidConstraint` arm uses `std::mem::discriminant` for unsupported variants — loses variant name, matches existing error style.
- [x] Update `lib.rs` public exports (remove `SymbolicExpr`, add `TLExprEvaluator`) (completed 2026-04-27)
  - **Goal:** Public API of `kizzasi-logic` reflects the TL-based surface; downstream callers use `kizzasi_logic::{TLExpr, TLExprEvaluator, TlExprCompiler, tl_const, tl_var}`.
  - **Design:** WIP already done (lib.rs). Drops `SymbolicExpr`; adds `tl_const`, `tl_var`, `TLExpr`, `TLExprEvaluator`, `TlTerm`, `TypeAnnotation`, and `TlExprCompiler` (re-exported from `compiler` module).
  - **Files:** `crates/kizzasi-logic/src/lib.rs` (verified, +3/-1).
  - **Prerequisites:** Items 1 and 2.
  - **Tests:** `cargo build -p kizzasi-logic --all-features` succeeds; downstream crates still compile.
  - **Risk:** external consumers of `kizzasi_logic::SymbolicExpr` will break. Audit: `rg "SymbolicExpr" crates/` must return 0 hits outside deleted definition.

### Phase 2 – `TlExprCompiler`: TLExpr → CompiledConstraint
- [x] **Add `TlExprCompiler` to `compiler.rs`** (TLExpr → ConstraintExpr → CompiledConstraint) (completed 2026-04-27)
  - **Goal:** Lower a `tensorlogic_ir::TLExpr` to the existing `ConstraintExpr` AST and through the stack-VM to `CompiledConstraint`. Supports `compile_optimized` with TL-level `algebraic_simplify` + `constant_fold` followed by stack-VM `.optimize()`.
  - **Design:** WIP already done (`compiler.rs` lines 785–1102, ~437 added LoC). `TlExprCompiler::new()`, `with_var(name, dim) -> Self`, `lower(&TLExpr) -> LogicResult<ConstraintExpr>`, `compile(...)`, `compile_optimized(...)`. Min/Max lowered via `(a±b ± |a−b|)/2`. `Pow` for exponents {0.5→Sqrt, 1.0→identity, 2.0→Mul(x,x)}. `Imply(a,b)→Or(Not(a),b)`. `Eq(a,b)→And(Le,Ge)`. `Lt`/`Gt` collapse to `Le`/`Ge`.
  - **Files:** `crates/kizzasi-logic/src/compiler.rs` (verified, +437/-0). ~1220 lines total — under 2000-line threshold.
  - **Prerequisites:** Item 1.
  - **Tests:** 8 unit tests in `compiler::tl_compiler_tests`: dim shorthand, const from pred name, var from dim_map, Lte bound, And box constraint, optimized constant fold, named-var map, Sqrt(L2-ball).
  - **Risk:** `Lt`/`Gt` → `Le`/`Ge` collapse loses strict-inequality semantics at boundary. Documented in source as "boundary-equivalent for continuous signals".
- [x] Comprehensive tests: TLExprEvaluator, TlExprCompiler, end-to-end constraint from TLExpr (completed 2026-04-27)
  - **Goal:** Cover full pipeline `ConstraintSynthesizer → TLExpr → TlExprCompiler → CompiledConstraint → evaluate(Array1<f32>)`. Verifies symbolic and numeric layers agree on constraint satisfaction.
  - **Design:** Add `#[cfg(test)] mod end_to_end_tests` in `tensorlogic_integration.rs`. Three tests: `test_synthesizer_to_compiled_box`, `test_learner_to_compiled_box`, `test_evaluator_compiler_agreement`.
  - **Files:** `crates/kizzasi-logic/src/tensorlogic_integration.rs` (append second `end_to_end_tests` module; projected ~700 LoC total).
  - **Prerequisites:** Items 2, 3, 4.
  - **Tests:** the three new end-to-end tests must pass under `cargo nextest run -p kizzasi-logic`.
  - **Risk:** evaluator/compiler may diverge at boundaries where `Lt`/`Gt` collapse to `Le`/`Ge`. Test 3 picks samples away from boundaries to side-step the edge.

### Rationale
| Old | New |
|---|---|
| `SymbolicExpr` (6 variants, f32 eval only) | `TLExpr` (50+ variants, symbolic + fuzzy + temporal) |
| Manual simplify rules in `simplify()` | `algebraic_simplify` + `constant_fold` from tensorlogic-ir |
| No symbolic serialization | TLExpr fully serializable (JSON/binary via `tensorlogic_ir::serialization`) |

---

## Proposed follow-ups

### Root TODO.md
- **`Pre-trained Models` (vague):** Needs concrete decisions: (1) Which architectures? Mamba-130M / Mamba-370M / RWKV-430M? (2) Training corpus — The Pile subset or custom multimodal? (3) Hosting: HF Hub mirror vs. self-hosted R2 bucket? (4) Naming under "Kizzasi-Takumi" — semver scheme?
- **`WebGPU acceleration` (oversized ~3000 LoC):** Proposed split into 4 future `/ultra` items:
  1. `kizzasi-webgpu` crate skeleton + `wgpu 22` + `WebGpuBackend::new()` + buffer upload/download (~500 LoC)
  2. `SsmBackend` trait in `kizzasi-core` + CPU default impl + `WebGpu` variant gated (~200 LoC)
  3. First WGSL kernel — SSM scan (Blelloch work-group=256) + dispatcher (~400 LoC Rust + ~150 LoC WGSL)
  4. Matvec + elementwise (silu, rms_norm) kernels + browser demo (~600 LoC Rust + ~200 LoC WGSL + HTML)

---

## WebGPU Acceleration (0.2.x)

- [x] **Item 1** — `kizzasi-webgpu` crate skeleton + `wgpu 29` + `WebGpuBackend::new()` + buffer upload/download (673 LoC, completed 2026-04-27)
  - **Crate:** `crates/kizzasi-webgpu/`
  - **Feature gate:** `webgpu = ["dep:wgpu"]`; default features are 100% Pure Rust (no C/Fortran).
  - **API:** `WebGpuBackend::new()` (async, requests high-perf adapter), `upload_f32()`, `download_f32()`, `submit_noop()`, `adapter_info()`.
  - **Buffers:** `GpuBuffer` (storage: `STORAGE|COPY_SRC|COPY_DST`; staging: `MAP_READ|COPY_DST`).
  - **Tests:** 4 unit tests (always compile/pass) + 3 GPU integration tests (`--features webgpu`, graceful skip when no adapter).
  - **wgpu version:** 29.0.1 (latest at time of implementation; task spec listed 22 which was stale).
- [x] **Item 2** — `SsmBackend` trait in `kizzasi-core` + CPU default + WebGpu variant (~200 LoC) (completed 2026-04-27)
  - **Crate:** `crates/kizzasi-core/src/ssm_backend.rs` (147 LoC)
  - **Trait:** `SsmBackend` (Send + Sync) with `ssm_scan(&[(f32,f32)]) -> CoreResult<Vec<(f32,f32)>>` and `backend_name()`.
  - **CPU impl:** `CpuSsmBackend` — sequential inclusive prefix scan with zero heap allocation overhead.
  - **Factory:** `default_backend() -> Box<dyn SsmBackend>` returns `CpuSsmBackend`; WebGPU variant lives in `kizzasi-webgpu` and can be substituted at the call site.
  - **Tests:** 7 unit tests (empty, single, two-element, identity, longer sequence, backend name, Send/Sync thread test). All 429 kizzasi-core tests pass; 0 clippy warnings.
- [x] **Item 3** — First WGSL kernel: SSM scan (Blelloch, work-group=256) + dispatcher (~550 LoC) (completed 2026-04-27)
  - **WGSL kernel:** `crates/kizzasi-webgpu/src/shaders/ssm_scan.wgsl` (105 LoC) — Blelloch work-efficient exclusive prefix scan converted to inclusive, work-group size 256.
  - **Dispatcher:** `crates/kizzasi-webgpu/src/ssm_scan.rs` (415 LoC) — `ssm_scan_gpu()` builds pipeline, uploads input, dispatches kernel, reads back results. Feature-gated; returns `BackendUnavailable` without `--features webgpu`.
  - **`WebGpuSsmBackend`:** `crates/kizzasi-webgpu/src/ssm_backend.rs` (59 LoC) — implements `SsmBackend` trait, falls back to `CpuSsmBackend` for sequences > 256 elements.
  - **Tests:** 4 tests without feature (all pass), 10 tests with feature including GPU round-trips vs CPU reference, identity element, two-element correctness. All pass.
  - **Clippy:** 0 warnings in both feature modes. Files all under 2000 LoC.
- [x] **Item 4** — Matvec + elementwise kernels (silu, rms_norm) + browser demo (~800 LoC Rust + ~130 LoC WGSL + HTML) (completed 2026-04-27)
  - **WGSL kernels:** `matvec.wgsl` (35 LoC), `silu.wgsl` (31 LoC), `rms_norm.wgsl` (63 LoC — single-pass tree-reduction, n ≤ 256).
  - **Dispatchers:** `matvec.rs` (511 LoC — `matvec_gpu()`, 4-binding layout, one thread per row, dispatch ceil(rows/64) workgroups); `elementwise.rs` (719 LoC — `silu_gpu()` + `rms_norm_gpu()` + `MAX_RMS_NORM_LEN`, separate shader modules and bind group layouts).
  - **Example:** `examples/webgpu_kernels_demo.rs` (127 LoC) demonstrating all three kernels with CPU verification.
  - **Browser demo:** `examples/webgpu_demo.html` — native JS WebGPU SiLU demo, no WASM.
  - **Tests:** 12 tests without feature (all pass), 21 tests with feature including all GPU round-trips vs CPU reference. 0 clippy warnings in both modes.
  - **Exports:** `lib.rs` re-exports `matvec_gpu`, `silu_gpu`, `rms_norm_gpu`, `MAX_RMS_NORM_LEN`.
