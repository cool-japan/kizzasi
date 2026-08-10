# Kizzasi Architecture Overview

**Version 0.2.2** | Autoregressive General-Purpose Signal Predictor (AGSP)

---

## 1. System Overview

Kizzasi is a Rust-native framework for predicting continuous multi-modal signal streams. Where large language models predict the next discrete text token, Kizzasi predicts the next continuous signal vector — the next audio sample, the next sensor reading, the next joint angle, or the next frame embedding.

### Core Philosophy

**Infinite context through State Space Models.** SSMs such as Mamba, RWKV, and S4D maintain a fixed-size hidden state that is updated at every step in O(1) time. Regardless of how long a sequence has been running, memory usage and per-step latency remain constant. This makes SSMs the primary backends for Kizzasi, while the standard Transformer is provided for comparison and hybrid architectures.

**Signal modality agnosticism.** Kizzasi treats audio waveforms, video frame embeddings, IMU readings, and robot joint angles as equivalent tokenized sequences. A single model can be conditioned on any combination of these modalities, and prediction heads can emit signals in any domain.

**Compositional safety.** Predictions pass through a constraint layer (`kizzasi-logic`) that projects outputs onto a user-defined feasible region. Constraints can express physical limits (joint angles, voltage rails), temporal logic properties (LTL/STL formulas), and cross-modal coupling rules.

---

## 2. Crate Dependency Graph

The workspace contains eleven crates. The arrows below indicate compile-time dependencies (A → B means A depends on B).

```
                        ┌─────────────────────────────────────────┐
                        │           kizzasi  (facade)             │
                        │  re-exports all public APIs             │
                        └────────┬────────────┬───────────────────┘
                                 │            │
               ┌─────────────────▼──┐    ┌───▼────────────────────┐
               │  kizzasi-inference │    │   kizzasi-python        │
               │  Pipeline, Engine  │    │   PyO3 bindings         │
               └──┬──────┬──────┬──┘    └────────────────────────┘
                  │      │      │
         ┌────────▼─┐  ┌─▼──────▼────────┐  ┌─────────────────┐
         │kizzasi-  │  │  kizzasi-model   │  │ kizzasi-logic   │
         │tokenizer │  │  Mamba/RWKV/S4D  │  │ Constraints     │
         └────┬─────┘  └────────┬─────────┘  └────────┬────────┘
              │                 │                      │
              └─────────────────▼──────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │     kizzasi-core        │
                    │  SSM primitives, SIMD,  │
                    │  HiddenState, traits    │
                    └────────────────────────┘

  Independent / platform-specific crates:
  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐
  │  kizzasi-io     │  │ kizzasi-embedded │  │kizzasi-webgpu │
  │  cpal/hound/    │  │ no_std SSM for   │  │ wgpu-backed   │
  │  MQTT/gRPC      │  │ STM32H7/RP2040/  │  │ WGSL kernels  │
  └─────────────────┘  │ ESP32C3          │  └───────────────┘
                        └──────────────────┘

  ┌─────────────────┐
  │  kizzasi-macros │
  │  #[derive(      │
  │  KizzasiConfig)]│
  └─────────────────┘
```

**Dependency notes:**
- `kizzasi-core` is the only crate that all other crates depend on; it defines `SignalPredictor`, `HiddenState`, and SIMD primitives.
- `kizzasi-model` re-exports `kizzasi-core` types and adds `AutoregressiveModel`.
- `kizzasi-inference` is the integration layer: it imports model, tokenizer, and logic crates and wires them into a `Pipeline`.
- `kizzasi-embedded` and `kizzasi-webgpu` are independent specializations; they do not depend on `kizzasi-inference`.
- `kizzasi-python` depends on `kizzasi` (the facade) and wraps it with PyO3.

---

## 3. Core Abstractions

### 3.1 `SignalPredictor` (kizzasi-core)

The foundational trait. Every model, regardless of architecture, must implement it.

```rust
pub trait SignalPredictor {
    /// Perform one autoregressive step.
    /// Updates internal hidden state and returns the predicted next signal.
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>>;

    /// Reset hidden state to all-zeros.
    fn reset(&mut self);

    /// Returns usize::MAX for SSMs (unbounded context) or a finite value
    /// for Transformer-based models.
    fn context_window(&self) -> usize;
}
```

The key invariant: calling `step` is O(1) in sequence length for SSM backends. This makes streaming inference over arbitrarily long signals practical.

### 3.2 `AutoregressiveModel` (kizzasi-model)

Extends `SignalPredictor` with architecture-specific metadata and state serialization.

```rust
pub trait AutoregressiveModel: SignalPredictor + Send {
    fn hidden_dim(&self) -> usize;
    fn state_dim(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn model_type(&self) -> ModelType;

    /// Retrieve per-layer hidden states (for checkpointing or hot-swap).
    fn get_states(&self) -> Vec<HiddenState>;

    /// Restore per-layer hidden states.
    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()>;

    /// Load weights from a JSON weight map (optional override).
    fn load_weights_json(&mut self, path: &Path) -> ModelResult<()>;

    /// Save weights to a JSON weight map (optional override).
    fn save_weights_json(&self, path: &Path) -> ModelResult<()>;
}
```

### 3.3 `SignalTokenizer` (kizzasi-tokenizer)

Maps between continuous `Array1<f32>` signals and latent embeddings. The tokenizer sits at the entry and exit of the pipeline.

```rust
pub trait SignalTokenizer {
    /// Project a raw signal into the latent embedding space.
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>>;

    /// Reconstruct a raw signal from latent embeddings.
    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>>;

    /// Latent embedding dimension.
    fn embed_dim(&self) -> usize;

    /// Vocabulary size (0 for continuous tokenizers).
    fn vocab_size(&self) -> usize;
}
```

Note that `encode` and `decode` both return `Array1<f32>`, not integer token indices. Kizzasi tokenizers operate in the continuous embedding domain by default; discretization is opt-in through `MuLawCodec` or `VQVAETokenizer`.

### 3.4 `ConstrainedInference` (kizzasi-logic)

Applied after every model step to guarantee predictions stay within a valid region.

```rust
pub trait ConstrainedInference {
    /// Project a prediction onto the constraint-satisfying manifold.
    fn constrain(&self, prediction: &Array1<f32>) -> LogicResult<Array1<f32>>;

    /// True iff the prediction satisfies all constraints.
    fn validate(&self, prediction: &Array1<f32>) -> bool;

    /// Soft violation penalty for use as a training loss term.
    fn violation_loss(&self, prediction: &Array1<f32>) -> f32;
}
```

---

## 4. Neural Backends

| Architecture | Inference Complexity | Context | Memory per Step | Primary Use Case |
|---|---|---|---|---|
| **Mamba / Mamba2** | O(1) | Infinite | Fixed state | Audio, sensor streams |
| **RWKV v5/v6/v7** | O(1) | Infinite | Fixed state | Long-horizon time series |
| **S4 / S4D** | O(1) | Infinite | Fixed state | DSP-adjacent signals, spectral |
| **S5** | O(1) | Infinite | Fixed state | Structured SSM with SISO blocks |
| **H3** | O(1) | Infinite | Fixed state | Hybrid conv + SSM layers |
| **Neural ODE** | O(steps) | Continuous | ODE solver buffer | Physical simulation |
| **Transformer** | O(N) per step | Finite (KV cache) | Grows with length | Short-context comparison |

**Choosing a backend:**

- For real-time streaming where latency must be constant: Mamba or RWKV.
- For audio signals where spectral structure matters: S4D or Mamba with a CausalConv1d stem.
- For resource-constrained edge devices: `kizzasi-embedded` provides standalone Mamba and S4 steps without the full model crate.
- For research comparison: Transformer with KV-cache is available via `ModelType::Transformer`.

All backends support the same `step` / `reset` / `get_states` / `set_states` interface, so they are interchangeable within a `Pipeline`.

---

## 5. Inference Pipeline

The `Pipeline` type in `kizzasi-inference` wires together tokenization, the model forward pass, constraint enforcement, and optional pre- and post-processing hooks.

### 5.1 Data Flow

```
Raw signal (Array1<f32>)
  │
  ├─ [PreprocessHook, ...]      optional user callbacks
  │
  ├─ SignalTokenizer::encode()  kizzasi-tokenizer
  │    Continuous embedding or discrete codebook lookup
  │
  ├─ InferenceEngine::step()    kizzasi-model
  │    SSM / Transformer forward pass
  │    Hidden state updated in place
  │
  ├─ GuardrailSet::constrain()  kizzasi-logic
  │    Range, monotonicity, temporal constraints
  │
  ├─ SignalTokenizer::decode()  kizzasi-tokenizer
  │    Reconstruct signal from embeddings
  │
  └─ [PostprocessHook, ...]     optional user callbacks
       │
       ▼
Output signal (Array1<f32>)
```

### 5.2 Building a Pipeline

```rust
use kizzasi_inference::{PipelineBuilder, EngineConfig};
use kizzasi_model::s4::{S4Config, S4D};
use kizzasi_logic::{GuardrailSet, ConstraintBuilder, Guardrail};

let model = S4D::new(
    S4Config::new()
        .input_dim(1)
        .hidden_dim(256)
        .state_dim(16)
        .num_layers(4)
        .diagonal(true),
)?;

let mut guardrails = GuardrailSet::new();
let constraint = ConstraintBuilder::new()
    .name("audio_range")
    .greater_eq(-1.0)
    .less_eq(1.0)
    .build()?;
guardrails.add_global(Guardrail::new(constraint, false));

let engine_config = EngineConfig::new(1, 1);

let mut pipeline = PipelineBuilder::new()
    .engine_config(engine_config)
    .model(Box::new(model))
    .guardrails(guardrails)
    .build()?;
```

### 5.3 Sampling Strategies

When the model output represents a distribution over signal values, a sampler selects the actual output:

| Strategy | Description |
|---|---|
| `Greedy` | Always pick the maximum (deterministic) |
| `Temperature` | Scale logits before sampling |
| `TopK` | Sample uniformly from top-K candidates |
| `TopP` | Nucleus sampling: smallest set with cumulative prob >= p |
| `BeamSearch` | Maintain multiple hypotheses across steps |

```rust
use kizzasi_inference::sampling::{SamplingConfig, SamplingStrategy};

let sampling = SamplingConfig::new()
    .strategy(SamplingStrategy::TopP)
    .top_p(0.9)
    .temperature(0.8);
```

### 5.4 Streaming and Adapters

The `streaming` feature (enabled with `features = ["streaming"]`) provides `FilterTransformer` for real-time signal transformation. Protocol adapters are available for:

- WebSocket (`features = ["websocket"]`)
- MQTT (`features = ["mqtt"]`)
- gRPC (`features = ["grpc"]`)
- REST (`features = ["rest"]`)

Each adapter wraps a `Pipeline` and handles framing, backpressure, and reconnection.

### 5.5 Speculative Decoding

For latency-sensitive workloads, `SpeculativeDecoder` pairs a small draft model with the main model. The draft generates K candidate steps; the main model verifies all K in a single batch call. Matches are accepted; the first mismatch triggers a rewind. This typically achieves 2-3x throughput improvement.

```rust
use kizzasi_inference::{SpeculativeConfig, SpeculativeDecoder};

let config = SpeculativeConfig::new()
    .num_draft_tokens(4)
    .greedy_verification(true);
```

---

## 6. State Management

### 6.1 HiddenState Structure

Each SSM layer maintains a `HiddenState` struct in `kizzasi-core`:

```rust
pub struct HiddenState {
    /// 2-D recurrent state: shape [hidden_dim, state_dim]
    state: Array2<f32>,
    /// Number of time steps processed since last reset
    step_count: usize,
    /// Optional causal convolution history buffer
    conv_history: Option<Vec<Vec<f32>>>,
}
```

A model with `L` layers returns `Vec<HiddenState>` from `get_states()`, one per layer. For Mamba with `hidden_dim = 256`, `state_dim = 16`, and 4 layers, the full state occupies `4 × 256 × 16 × 4 bytes = 65 536 bytes` (64 KB), regardless of sequence length.

### 6.2 Checkpointing

`kizzasi-inference` provides `CheckpointManager` for saving and restoring state:

```rust
use kizzasi_inference::CheckpointManager;

// Save current state
let manager = CheckpointManager::new("/tmp/checkpoints");
let checkpoint = manager.save(&model)?;

// Later: restore from checkpoint
manager.restore(&mut model, &checkpoint)?;
```

Checkpoints serialize `HiddenState` via `serde` (MessagePack by default, JSON optional). The `serde(default)` attribute on `conv_history` ensures backward compatibility as the schema evolves.

### 6.3 State Compression

For memory-constrained deployments, `StateCompressor` reduces checkpoint size:

| Method | Description | Typical Ratio |
|---|---|---|
| `None` | Full f32 precision | 1x |
| `Quantize8Bit` | INT8 quantization | ~4x |
| `Quantize4Bit` | INT4 quantization | ~8x |
| `Sparse` | Store only nonzero values | variable |
| `QuantizedSparse` | INT8 + sparsity mask | variable |

### 6.4 Hot-Swapping

When the `async` feature is enabled, `HotSwapManager` allows replacing a live model without service interruption:

```rust
use kizzasi_inference::HotSwapManager;

// Background: load new model weights
manager.prepare_swap("v2.0.0", new_model_path).await?;

// Atomic switch with no dropped requests
manager.activate("v2.0.0", SwapStrategy::Graceful).await?;

// If issues appear, roll back instantly
manager.rollback().await?;
```

Swap strategies: `Immediate` (atomic swap), `Graceful` (drain in-flight requests first), and `Gradual { percentage }` (canary rollout by traffic fraction).

### 6.5 Ensembling

`ModelEnsemble` combines predictions from multiple independently initialized models:

| Strategy | Behavior |
|---|---|
| `Average` | Element-wise mean of all outputs |
| `Weighted` | Weighted mean by per-model confidence |
| `Voting` | Majority vote on argmax |
| `ProductOfExperts` | Multiply probability densities |

Ensembling typically improves robustness for out-of-distribution inputs at the cost of proportionally higher compute.

---

## 7. Deployment Options

### 7.1 Native Rust

The default deployment. Add `kizzasi` to your `Cargo.toml`:

```toml
[dependencies]
kizzasi = { version = "0.2", features = ["full"] }
```

The `full` feature enables `io`, `logic`, `async`, `config-files`, and `macros`.

### 7.2 Python Bindings (PyO3)

`kizzasi-python` provides a zero-copy PyO3 interface. Build the wheel with `maturin`:

```bash
cd crates/kizzasi-python
maturin develop --release
```

Usage:

```python
import kizzasi
import numpy as np

config = kizzasi.Config.audio(44100)
predictor = kizzasi.Predictor(config)

sample = np.array([0.5], dtype=np.float32)
output = predictor.step(sample)

# Multi-step autoregressive generation
rollout = predictor.predict_n(sample, n_steps=100)
predictor.reset()

# Ensemble of 3 models
ensemble = kizzasi.EnsemblePredictor(config, n_models=3, voting="average")

# LoRA fine-tuning adapter
adapter = kizzasi.LoRAAdapter(predictor, rank=8)
```

Exported classes: `Predictor`, `EnsemblePredictor`, `OptimizedPredictor`, `LoRAAdapter`, `Sampler`, `SamplingConfig`, `Config`, `ModelType`, `ConstraintSpec`.

### 7.3 Embedded / no_std

`kizzasi-embedded` compiles without the standard library, targeting ARM Cortex-M, RISC-V, and other microcontrollers.

```toml
[dependencies]
kizzasi-embedded = { version = "0.2", default-features = false, features = ["alloc", "libm"] }
```

Platform presets handle RAM budgets automatically:

| Preset | Target | d_model | d_state | State RAM |
|---|---|---|---|---|
| `stm32h7()` | Cortex-M7 | 64 | 16 | ~576 B |
| `rp2040()` | Cortex-M0+ | 32 | 8 | ~288 B |
| `esp32c3()` | RV32IMC | 48 | 12 | ~432 B |

The `fixed-point` feature substitutes Q16.16 arithmetic for `f32`, essential for FPU-less cores.

```rust
use kizzasi_embedded::{stm32h7, SsmState};

let cfg = stm32h7();
let mut state = SsmState::new(&cfg);
// step() updates state in place with no heap allocation
```

### 7.4 WebGPU

`kizzasi-webgpu` provides a `wgpu`-backed GPU compute layer. The crate always compiles to pure Rust; activate actual GPU operations with the `webgpu` feature.

```toml
[dependencies]
kizzasi-webgpu = { version = "0.2", features = ["webgpu"] }
```

Available GPU kernels (implemented as WGSL shaders):

- `ssm_scan_gpu` — parallel associative scan for SSM recurrence
- `matvec_gpu` — batched matrix-vector multiply
- `rms_norm_gpu` / `silu_gpu` — fused elementwise kernels

Without a physical GPU the backend returns `WebGpuError::BackendUnavailable`, allowing the same binary to fall back to CPU transparently.

### 7.5 Network Serving (REST, gRPC, WebSocket, MQTT)

`kizzasi-inference` provides feature-gated server adapters for exposing the inference pipeline over standard network protocols. See [Deployment and Serving](deployment_and_serving.md) for full documentation including configuration, endpoint reference, Docker, and docker-compose.

---

## 8. Quick Start: End-to-End Example

The following example builds a Mamba2-backed predictor, applies mu-law tokenization, runs a 10-step rollout, and enforces range constraints.

```rust
use kizzasi::prelude::*;
use kizzasi_tokenizer::{MuLawCodec, SignalTokenizer};
use kizzasi_logic::{ConstraintBuilder, Guardrail, GuardrailSet};
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build a Mamba2 predictor (1-dimensional audio signal)
    let mut predictor = KizzasiBuilder::new()
        .model_type(ModelType::Mamba2)
        .input_dim(1)
        .output_dim(1)
        .hidden_dim(64)
        .state_dim(16)
        .num_layers(4)
        .build()?;

    // 2. Attach a mu-law tokenizer for audio (256 companded levels)
    let codec = MuLawCodec::new(8); // 8-bit mu-law

    // 3. Define safety constraints: output must lie in [-1.0, 1.0]
    let mut guardrails = GuardrailSet::new();
    let range_constraint = ConstraintBuilder::new()
        .name("audio_range")
        .greater_eq(-1.0)
        .less_eq(1.0)
        .build()?;
    guardrails.add_global(Guardrail::new(range_constraint, false));
    predictor.set_guardrails(guardrails);

    // 4. Encode an initial sample
    let raw_input = Array1::from_vec(vec![0.3f32]);
    let embedding = codec.encode(&raw_input)?;

    // 5. Run a 10-step autoregressive rollout
    predictor.reset();
    let trajectory = predictor.predict_n(&embedding, 10)?;

    for (i, row) in trajectory.outer_iter().enumerate() {
        // Decode back to waveform domain
        let decoded = codec.decode(&row.to_owned())?;
        println!("step {:2}: sample = {:.4}", i + 1, decoded[0]);
    }

    // 6. Save state for later resumption
    let states = predictor.get_states();
    println!("Saved {} layer states", states.len());

    // 7. Restore and continue from the saved state
    predictor.reset();
    predictor.set_states(states)?;
    let continued = predictor.step(&embedding)?;
    println!("Continued step: {:.4}", continued[0]);

    Ok(())
}
```

**Key points from this example:**

- `KizzasiBuilder` provides a fluent API; `audio_preset()` and `robotics_preset(dof)` offer opinionated defaults.
- `predict_n` runs the autoregressive loop internally and returns an `Array2<f32>` of shape `[steps, output_dim]`.
- Constraints are applied inside `predict_n` at every step; no manual enforcement is needed.
- `get_states` / `set_states` enable pause-resume and state transfer between processes.

---

## 9. Error Handling

Each crate defines its own error type:

| Crate | Error type | Result alias |
|---|---|---|
| `kizzasi-core` | `CoreError` | `CoreResult<T>` |
| `kizzasi-model` | `ModelError` | `ModelResult<T>` |
| `kizzasi-tokenizer` | `TokenizerError` | `TokenizerResult<T>` |
| `kizzasi-inference` | `InferenceError` | `InferenceResult<T>` |
| `kizzasi-logic` | `LogicError` | `LogicResult<T>` |
| `kizzasi-embedded` | `EmbeddedError` | `EmbeddedResult<T>` |
| `kizzasi-webgpu` | `WebGpuError` | `WebGpuResult<T>` |
| `kizzasi` | `KizzasiError` | `KizzasiResult<T>` |

All error types implement `std::error::Error` and use `thiserror` for ergonomic variant definitions. The facade crate re-exports `KizzasiError` as the primary user-facing error type.

---

## 10. Further Reading

- [Training Guide](training_guide.md) — training loop, optimizers, LR schedulers, curriculum learning
- [Deployment and Serving](deployment_and_serving.md) — REST, gRPC, WebSocket, MQTT adapters, Docker
- `docs/cross_modal_integration.md` — multi-modal pipelines and modality fusion
- `docs/performance_tuning.md` — SIMD acceleration, kernel fusion, batch tuning
- `docs/pytorch_migration.md` — porting models from PyTorch checkpoints (GGUF, SafeTensors)
- `crates/kizzasi/examples/getting_started.rs` — runnable end-to-end example
- `crates/kizzasi-inference/examples/multimodal_with_constraints.rs` — multi-modal inference
- `crates/kizzasi/examples/robotics_control.rs` — 6-DOF joint angle prediction with safety guardrails
