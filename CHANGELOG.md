# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-08-09

This release spans roughly three months of work across every crate in the workspace: a new
WebGPU acceleration backend, a real `tensorlogic-ir` constraint pipeline, PEAQ perceptual audio
quality evaluation, and — the bulk of the diff — a wide sweep of numerical and logic correctness
fixes in the model, core, and constraint layers, several of which mean previously-shipped
functionality (training, some architectures' state-space recurrences, weight serialization) was
silently non-functional prior to this release. See below for the full breakdown.

### Added

**kizzasi-webgpu** (new workspace crate)
- WebGPU (`wgpu 30`) acceleration backend for kizzasi-core's SSM/signal kernels — pure Rust and
  GPU-free by default, unlocked via the `webgpu` feature flag.
- `WebGpuBackend` for adapter/device management, `GpuBuffer` (storage + staging), buffer
  upload/download.
- WGSL kernels: elementwise SiLU activation, RMS Norm, matrix-vector multiplication, and a
  Blelloch work-efficient parallel prefix scan for the SSM recurrence.
- `WebGpuSsmBackend`, an `SsmBackend` implementation that runs short sequences on the GPU and
  falls back to the CPU backend for longer ones; `webgpu_kernels_demo` example.

**kizzasi-core**
- `SsmBackend` trait, `CpuSsmBackend`, `default_backend()` in a new `ssm_backend` module — the
  pluggable backend abstraction that `WebGpuSsmBackend` above implements.
- INT8 quantization API on `WeightLoader`: `quantize_tensor`/`dequantize_tensor` (per-tensor
  affine) and `quantize_per_channel`/`dequantize_per_channel` (per-channel affine), returning new
  `QuantizedTensor`/`PerChannelQuantizedTensor` types.
- Head-wise (`PruningGranularity::Head`) and block-wise (`PruningGranularity::Block`) structured
  pruning on `StructuredPruner`, configured via `PruningConfig::head_size`/`block_size` and
  `with_head_size()`/`with_block_size()` builders.
- Tracing span instrumentation across attention, SSM scan, and SIMD kernels for latency profiling.
- `int8_quantization_bench` and `parallel_scan_bench` Criterion benchmarks.

**kizzasi-embedded**
- `libm` feature: `f32` math shims (`sqrt`, `round`) for `no_std` builds without `std`.
- Platform presets `stm32h7()`, `rp2040()`, `esp32c3()` in a new `presets` module.
- `no_std_basic`, `fixed_point_cortex_m`, `stm32h7_preset` examples.

**kizzasi-model**
- `tensorlogic_bridge` module bridging kizzasi-logic: `constraint_from_tl_expr()` and
  `compile_constraints()` compile symbolic `TLExpr` constraints into executable
  `CompiledConstraint`s.
- `binary_io` module: shared little-endian binary-read primitives, factored out of the GGUF
  parser for reuse by future binary loaders.
- GGUF: support the `general.alignment` metadata key for tensor data-section alignment
  (previously hardcoded to 32 bytes); `GgufMetaValue::as_u64()`.
- RWKV7 (split from a single `rwkv7.rs` into `rwkv7/{mod,channel_mixing,time_mixing}.rs`):
  `save_weights_json`/`load_weights_json` for both the time-mixing and channel-mixing blocks.
- `ModelFactory::create_rwkv7`/`create_s5` now inject supplied weights into the constructed model
  instead of discarding them with a warning.
- New tests: `hf_weight_roundtrip`, `neural_ode_integration`, `transformer_state_roundtrip`.

**kizzasi-tokenizer**
- PEAQ (ITU-R BS.1387-1 Basic Model) perceptual audio quality evaluator (`peaq` module):
  `PeaqEvaluator` computes all 11 Basic Model Output Variables through a Bark-band `EarModel`
  (outer/middle-ear weighting, masking spreading, IIR time-smoothing) and maps them to a
  Distortion Index and Objective Difference Grade (`OdgGrade`, range `[-4, 0]`) via an 11→3→1
  neural network. The network weights are LCG-seeded placeholders pending the paid ITU BS.1387-1
  Annex 2 tables (`peaq::nn::WEIGHTS_VERIFIED == false`), so ODG/DI values are qualitative trend
  indicators, not certified scores.
- Perceptual/Bark-scale psychoacoustic quantization (`perceptual` module): `PerceptualQuantizer`
  allocates bits per critical band proportional to exceedance over the Terhardt absolute
  threshold of hearing, using Hann-windowed STFT/ISTFT via the new `oxifft` dependency;
  `BarkBands`, `frequency_to_bark`, `absolute_threshold_db` exposed as standalone functions.
- Multi-speaker tokenization (`multi_speaker` module): `MultiSpeakerTokenizer` with a
  k-means++-initialized, EMA-updated `SpeakerCodebook`, blind speaker inference
  (`encode_blind`), explicit speaker assignment (`encode_with_speaker`), and speaker re-tagging
  (`re_target`).
- `range_coder_bench` benchmark; `perceptual_quant`, `vqvae_roundtrip`, `inference_integration`
  examples.
- `SpeechTokenizer` now implements `Debug`.
- New regression, edge-case (NaN/Inf/subnormal/empty/boundary), and transform-roundtrip test
  suites.

**kizzasi-logic**
- Real `tensorlogic-ir` (`0.1.1`) backing for symbolic constraints, replacing the previous
  commented-out placeholder: `TlExprCompiler` lowers `TLExpr` to the stack-VM
  `CompiledConstraint` (with `compile_optimized()` applying algebraic simplification and constant
  folding first); `TLExprEvaluator` is a direct f32 evaluator using Gödel soft-logic semantics
  (`And`=min, `Or`=max, `Not`=1−a, `Imply`=max(1−a,b)); `tl_var()`/`tl_const()` build `TLExpr`
  leaf nodes; `TLExpr`/`TlTerm`/`TypeAnnotation` re-exported from `kizzasi_logic`.
- `ConstraintLearner::learn_box_constraints_as_expr()`, `Constraint::bound()`,
  `LinearConstraint::shift_rhs()`/`update_coefficients_towards()`,
  `QuadraticSubproblem::with_lower_bound()`/`with_upper_bound()`.
- Custom n-th order (up to 8th) backward finite-difference derivatives on
  `DerivativeConstraint`/`DerivativeOrder::Custom` (previously unimplemented above 3rd order).
- `tests/control_and_learning_integration.rs`: end-to-end MPC, constraint repair, and online
  learning coverage.

**kizzasi-inference**
- `InferenceError::Timeout` variant for request/response timeouts.
- `LoraAdapterLoader::save()` to persist a `LoraAdapter` and its `LoraConfig` to disk.
- `FilterTransformer` stream transformer (filters items via a synchronous predicate).
- `state_compression_bench` benchmark.
- `ensemble_integration`, `multimodal_integration`, `speculative_integration` integration tests
  against live S4D/Mamba models.

**kizzasi-io**
- `WebSocketStream::check_ping_signal()` to drain ping-interval ticks and send WebSocket ping
  frames.

**kizzasi-python**
- `BeamSearch`, `ConstrainedBeamSearch` (Python-callable hard/soft constraints), and
  `RejectionSampler` for autoregressive decoding.
- `EnsemblePredictor`: multi-model ensemble prediction (average, weighted, weighted-average,
  median, confidence, majority-vote strategies).
- `OptimizedPredictor`: workspace pooling, SIMD kernels, and an optional TTL-based LRU result
  cache, with `cache_stats()`/`optimization_stats()` introspection.
- `LoRAAdapter`: register per-module layers from NumPy base weights, run `forward()`, and
  merge/unmerge corrections into the base weights in place.
- `SamplingConfig`/`Sampler`: greedy, temperature, top-k, and top-p sampling, single-vector and
  batched.

**kizzasi-macros**
- `#[config(default = EXPR)]`, `#[config(validate = "path")]`, `#[config(skip)]` field attributes
  in `#[derive(KizzasiConfig)]` are now actually honored (previously accepted but silently
  ignored).
- A named `<name>_preset()` constructor per `#[preset(name = "...", field = value, ...)]` struct
  attribute in `#[derive(Preset)]`, supporting multiple presets per struct.
- `#[metrics]`-annotated field (or a field literally named `collector`) and generic struct
  support in `#[derive(Instrumented)]`.

### Changed

**Dependencies**
- `scirs2-core`/`-linalg`/`-signal`/`-fft`/`-series`: 0.4.2 → 0.6.5.
- `candle-core`/`candle-nn`: 0.10.2 → 0.11.0.
- `oxicode`: 0.2 → 0.2.6.
- `oxifft`: 0.3 → 0.4.2 (now also a direct dependency of kizzasi-tokenizer, used by the new
  perceptual quantizer).
- `oxirs-core`/`oxirs-gql`: 0.2.4 → 0.4.1.
- `pyo3`: 0.28 → 0.29; `scirs2-numpy`: 0.4.2 → 0.6.5.
- `safetensors`: 0.7 → 0.8.
- `cpal`: 0.17 → 0.18 (kizzasi-io adapted to the new `build_input_stream`/`build_output_stream`
  signatures).
- `tower-http`: 0.6 → 0.7; `tokio-serial`: 5.4 → 5.5.
- `wasm-bindgen`: 0.2.118 → 0.2.126; `js-sys`: 0.3.95 → 0.3.103.
- New workspace member `crates/kizzasi-webgpu`.

**Breaking changes**
- kizzasi-embedded: `std` now implies `alloc`; `no_std` builds must additionally enable the new
  `libm` feature for `f32` math, or they fail to compile with a directed `compile_error!`.
- kizzasi-core: `CausalConv1d::set_history`/`DepthwiseCausalConv1d::set_history` now return
  `CoreResult<()>` instead of panicking on a mismatched history length.
- kizzasi-core: `PruningConfig` gained `head_size`/`block_size` fields — construct via
  `Default`/builders rather than a struct literal.
- kizzasi-tokenizer: the `entropy` range coder (now split into `entropy/{mod,encoder,decoder}.rs`)
  emits an LZMA-style carry-propagating byte stream; data encoded with 0.2.1 cannot be decoded by
  0.2.2 and vice versa (type/function names unchanged).
- kizzasi-tokenizer: `DCTTokenizer`/`WaveletTokenizer::encode` now prepend a `max_val` header
  token, so `embed_dim()` and encoded token-array lengths grow by one.
- kizzasi-logic: `MPCSolution` gained a `constraint_violation: f32` field.
- kizzasi-logic: `ConstraintSynthesizer::synthesize_from_template()` and its template methods now
  return `TLExpr` instead of the removed `SymbolicExpr`.
- kizzasi-inference: new `InferenceError::Timeout` variant (exhaustive `match`es need a new arm).
- kizzasi-io: `WebSocketStream::start_ping_task` now returns
  `(JoinHandle<()>, mpsc::Receiver<()>)` instead of `JoinHandle<()>`.
- kizzasi-macros: `#[derive(Preset)]` now generates one `<name>_preset()` method per declared
  preset instead of a single fixed `preset()`; existing `.preset()` call sites no longer compile.

**Other**
- kizzasi-core: `FusedAttentionKernel::forward_parallel` now runs a genuine rayon-parallel
  implementation, and `parallel_scan`/`parallel_ssm_batch` dispatch to scirs2-core's Blelloch
  parallel scan / rayon batch iteration, instead of silently falling back to sequential execution.
- kizzasi-core: `activations::fast_exp_avx512` is now a true AVX-512-vectorized
  Cody-Waite/polynomial implementation instead of a scalar `exp()` fallback; results may differ
  slightly in the last bits.
- kizzasi-logic: `ConstraintRepairer`'s `ElasticProgramming` strategy now descends until violation
  is within tolerance instead of stopping at the slack budget, changing repaired-point output.
- kizzasi-python: `lib.rs` split into `config`/`predictor`/`ensemble`/`optimized`/`lora`/
  `sampling`/`beam_search` modules (existing `Predictor`/`Config`/`ModelType`/`ConstraintSpec`
  classes unaffected); `KizzasiConfig`/`Preset`/`Instrumented` derive-macro errors are now
  spanned compile errors instead of raw panics; `LazyKizzasi` recovers from a poisoned mutex
  instead of panicking.
- kizzasi-inference/kizzasi-io: benches/adapters updated for cpal 0.18; the `advanced_features`
  bench now requires the `streaming` feature (previously unconditional).
- Replaced `unwrap()` with descriptive `expect()`/proper error returns across kizzasi-core,
  kizzasi-logic, and kizzasi-io per COOLJAPAN no-unwrap policy.

### Fixed

**kizzasi-core**
- `FlashAttention::forward` misinterpreted its `[batch, seq_len, d_model]` input as
  already-split `[seq_len, num_heads, head_dim]` (the old `reshape_qkv` was a no-op), and for
  `batch_size > 1` computed and copied only the first batch element into every output batch slot.
- `Mamba2Layer::forward`: the discretized `A` matrix was computed as `(-a_log.exp()).abs()`
  (always positive) instead of `-a_log.exp()` (negative), letting the recurrent state diverge
  over long sequences.
- `S5Layer`: the per-block continuous-time `A` diagonal was initialized as `exp(log_lambda)`
  (always positive) instead of `-exp(log_lambda)`, and `discretize_block` used an ad hoc
  first-order approximation instead of exact ZOH.
- `matrix_exp_pade`/`zoh_discretize`: replaced a truncated low-order Padé approximation and a
  first-order `B_d` estimate with an exact Padé[13,13] matrix exponential (Higham 2005) and
  closed-form zero-order-hold discretization via an LU-based linear solve.
- `SelectiveSSM::step`: the discretization time-step `Δ` was hardcoded to `0.1`, giving the model
  no input-dependent selectivity; `Δ` is now `softplus(dt_proj · x)`, restoring the Mamba
  selective-scan mechanism.
- `TrainableSSM::ssm_layer` ignored the `A`, `B`, `C` matrices entirely (only computing a `D * x`
  skip connection), so state-space parameters never received gradients during training; it now
  runs the full S4D-style selective-scan recurrence.
- **`Trainer::fit()` built training and validation batches as a hardcoded empty `Vec::new()`
  every epoch, so no training or validation ever actually ran.** Batches are now built from the
  `TimeSeriesDataLoader` via a new `collect_epoch_batches` helper.
- `TrainingConfig::grad_clip` was silently inert — `train_epoch` consumed the unclipped gradients
  before the (no-op) `clip_gradients` ever ran; gradients are now clipped by global L2 norm before
  the optimizer step.
- `Trainer::compute_grad_norm` always returned the hardcoded placeholder `1.0` instead of the
  true global L2 gradient norm.
- **`WeightLoader::save_safetensors` wrote an empty, invalid buffer**; it now serializes real
  `TensorView`s via the `safetensors` crate.
- `CausalConv1d`/`DepthwiseCausalConv1d` `get_history()`/`set_history()` could drop the most
  recently buffered frame when snapshotting mid-`forward_step`, corrupting output after a
  restore.
- Potential panics on `NaN` in `StructuredPruner`/`GradientPruner` threshold and importance sorts.

**kizzasi-model**
- Mixture-of-Experts `Router` and `Expert` weights were zero-initialized (routing was
  input-independent and every expert's output was identically zero); both now use
  Glorot/He-uniform initialization, and `Expert` is a real two-layer SiLU feed-forward network.
- `Mamba2Layer`'s SSD state update ignored the actual per-head input value (multiplied the
  B-projection by a hardcoded `0.01` constant instead of the input).
- `S5`'s `S5Block` ZOH discretization produced `a_bar > 1` (unbounded, unstable state growth)
  from an inverted sign convention on `log_a`.
- RWKV (v4) `TimeMixing`'s WKV recurrence failed to exponentiate the key term (`exp(k)`) in the
  denominator, which could go negative, get clamped to `1e-8`, and inflate the output toward
  ~1e8.
- `Transformer::get_states`/`set_states` silently dropped the value (V) cache and restored keys
  only, corrupting attention after any state restore; both K and V now round-trip (the packed
  `HiddenState` layout is incompatible with snapshots taken before this fix).
- The standard (L2, non-decoupled) Adam optimizer applied `weight_decay` unscaled by the learning
  rate (up to ~1000x too strong versus the intended per-step decay); it now applies the same
  `(1 - lr * weight_decay)` factor as AdamW.
- `Mamba`'s `SelectiveSSM` ZOH discretization: the near-zero-decay fallback substituted
  `a_n = -1.0`, collapsing `b_bar` to ~0 and suppressing state updates; it now uses the correct
  L'Hôpital limit `delta * B`.
- `hybrid::AttentionBlock` computed attention scores over the full concatenated Q/K/V vectors
  instead of per head; it now slices each head's `head_dim` span before scoring.
- `LifLayer`'s absolute refractory period continued applying `leak_factor` decay to the membrane
  voltage instead of holding it at the reset value, for `SoftReset` and other non-hard reset
  modes.
- `h3::ShiftSSM` iterated its input history oldest-first, multiplying `shift_weights` row 0 by
  the oldest buffered input instead of the current one.
- HuggingFace Mamba checkpoint loading assumed a hardcoded `state_size = 16` when inferring
  `dt_rank`; it now probes common state sizes (16/32/64/128) and picks the one yielding a valid
  power-of-two `dt_rank`.
- `SsmBackward`'s `dx` gradient multiplied two independent means instead of the per-timestep dot
  product scaled correctly, producing incorrect input gradients.
- `pytorch_compat::convert::chw_to_hwc`/`transpose_if_needed` were no-op placeholders
  (`tensor.clone()`) despite their documented purpose; they now perform the actual transpose.
- **`save_weights` on `Mamba`, `Mamba2`, `Rwkv`, `S4D`, and `Transformer` silently wrote the JSON
  `save_weights_json` format while named/documented as SafeTensors**, producing files that
  `ModelLoader`/`load_weights` and any real SafeTensors reader could not parse; all five now
  serialize genuine, round-trippable SafeTensors binaries.

**kizzasi-tokenizer**
- `TrainableContinuousTokenizer`'s Xavier initialization called `Tensor::affine` with `(mul, add)`
  swapped, so encoder/decoder weights were all initialized to the same constant instead of scaled
  random values.
- `DCTTokenizer::decode`/`WaveletTokenizer::decode` dequantized using a hardcoded `max_val = 1.0`,
  silently mis-scaling reconstruction for any signal whose peak coefficient magnitude differed
  from 1.0; the true peak magnitude is now carried through as a header token.
- `RangeEncoder`/`RangeDecoder` precision bugs on long or skewed symbol sequences (previously
  masked by `#[ignore]`d tests); the rewritten carry-propagating coder now round-trips correctly.
- `NonUniformQuantizer::lloyd_max_gaussian` now runs the actual Lloyd-Max boundary/centroid
  iteration to convergence instead of a crude percentile approximation.
- `PerceptualMetrics::weighted_snr_db` was a placeholder that just copied `segmental_snr_db`; it
  now computes a true IEC 61672-1 A-weighted frequency-domain SNR.
- `MuLawCodec::quantize` returned an out-of-range level (256) for input `1.0` on an 8-bit codec
  whose valid levels are `0..=255`; the result is now clamped to `levels - 1`.

**kizzasi-logic**
- `ConsensusADMM` dual residual accumulated per block instead of once per iteration,
  over-reporting it by a factor of √(num_blocks).
- `BendersDecomposition::solve_master` nudged a violated feasibility cut by an arbitrary fixed
  `+0.1` instead of restoring feasibility; it now takes a minimum-norm projected-gradient step.
- `QuadraticSubproblem::solve` clipped every element to `lb[0]`/`ub[0]` when only one-sided
  bounds were set; bounds now apply element-wise.
- `BacktrackingSearch::with_forward_checking(true)` was a no-op; it now prunes domains via
  `ForwardChecker` before recursing.
- `ConstraintRepairer` and `IncrementalSolver` (`add_constraint`, `modify_constraint`, its repair
  loop, `recompute_violations`) always read and perturbed dimension 0 regardless of a
  constraint's declared dimension, so constraints on other dimensions never converged.
- `ConflictResolver::might_conflict` always returned `false`; it now detects real structural
  conflicts (disjoint bounds, strict-boundary contradictions, incompatible equalities/ranges).
- `MPCSolution::is_feasible()` was tautologically true (`total_cost < f32::INFINITY`); it now
  also checks tracked `constraint_violation` against tolerance.
- `OnlineConstraintLearner::observe()`, `ActiveConstraintBoundaryLearner::refine()`, and
  `FeedbackConstraintTuner::add_feedback()` each computed an update and discarded it; all three
  now apply their computed adjustments.
- `ConstraintSynthesizer::synthesize_box()` discarded the upper bound (synthesized constraints
  were unbounded above); `synthesize_quadratic()` returned a mislabeled linear bound instead of
  an actual quadratic form.
- `ConstraintLearner::learn_box_constraints()` returned an inverted `f32::MAX`/`f32::MIN`-based
  range when no example had data for the requested dimension; it now returns an error.
- Hardened `ConstraintLearner::learn_linear_separator()` and `ConstraintSynthesizer` against
  out-of-bounds panics on empty or ragged example/variable inputs.
- `MinimalViolatingSubsetFinder::find_mvs()` always returned every violated constraint; it now
  performs gradient-based greedy redundancy removal to return an actual minimal subset.
- `Colormap::Viridis::map()` used an inaccurate ad hoc polynomial (off by over 100 RGB units at
  the midpoint); it now interpolates over 9 published Viridis reference anchors.

**kizzasi-inference**
- `MqttAdapter::request()` previously always failed with `NotImplemented`; it now correlates
  requests and responses via an internal oneshot channel and returns the real response, or a
  `Timeout` error after 30 seconds.
- `LoraAdapterLoader::load()` previously returned a zero-filled placeholder adapter regardless of
  what was on disk; it now reads and reconstructs the actual matrices written by `save()`
  (returns `Err` if weight files are missing, instead of silently succeeding with zeros).
- `ModelRegistry` now constructs `NeuralOde` and `MultiModal` models instead of rejecting them
  with a "not yet supported" error.
- `Sampler` now honors `SamplingConfig::seed` for reproducible categorical sampling (previously
  always drew from OS entropy regardless of the configured seed), and `temperature_sample` no
  longer produces `NaN` at or near zero temperature (short-circuits to greedy sampling).
- `STLFormula::Until` now evaluates `phi1` over the entire prefix `[t, t']` per standard
  quantitative STL semantics, rather than only from the window's lower bound.

**kizzasi-io**
- `StreamRecorder` JSON output now terminates the header with a newline so it no longer merges
  with the first frame and fails to parse.
- `StreamPlayer` now supports JSON and CSV playback and frame seeking for binary recordings —
  both previously failed immediately with "not yet implemented" errors.
- `WebSocketStream`'s ping-keepalive task now actually sends WebSocket ping frames on each
  interval tick instead of only logging a debug message, and skips a spurious immediate first
  tick.

**kizzasi-python**
- `EnsemblePredictor::stats()`'s `avg_variance` was hardcoded to `0.0`; it now reports the actual
  inter-model prediction variance.

### Removed
- kizzasi-logic: the `SymbolicExpr` enum, its inherent methods (`var`, `constant`, `add`, `sub`,
  `mul`, `le`, `ge`, `evaluate`, `simplify`), and its `Display` impl — superseded by
  `tensorlogic_ir::TLExpr` and the new `TLExprEvaluator`.

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

[0.2.2]: https://github.com/cool-japan/kizzasi/releases/tag/v0.2.2
[0.1.0]: https://github.com/cool-japan/kizzasi/releases/tag/v0.1.0
