# Cross-Modal Integration Guide

**Version 0.2.2** | Multi-modal signal prediction with Kizzasi AGSP

---

## 1. Multi-Modal Signals

Kizzasi treats every continuous signal source as an `Array1<f32>` and routes it through the same autoregressive prediction loop. The distinction between modalities lives entirely in how that array is produced (the preprocessor), how dimensions are combined (the fusion strategy), and which constraints apply to the output.

The four built-in modality types are:

| `ModalityType` | Typical Signal | Representative Dimensions |
|---|---|---|
| `Audio` | Raw waveform samples or mel-spectrogram frames | 1 (mono) to 128 (mel bins) |
| `Video` | CNN / ViT embedding of a video frame | 256 – 2048 (depends on backbone) |
| `Sensor` | IMU, pressure, temperature, or current readings | 3 – 64 |
| `Text` | Language model embeddings | 256 – 1024 |
| `Custom(&str)` | Any user-defined modality | user-defined |

The `Custom` variant accepts a static string identifier, which is used as the deterministic sort key during fusion and as the codebook key in `CrossModalTokenizer`.

---

## 2. ModalityFusion via `MultiModalPipeline`

`kizzasi_inference::MultiModalPipeline` is the primary integration point for combining multiple signal streams. It wraps an `InferenceEngine` and applies a chosen `FusionStrategy` before each model step.

### 2.1 Builder Pattern

```rust
use kizzasi_inference::{
    EngineConfig, FusionStrategy, ModalityConfig, ModalityType,
    MultiModalPipeline,
};

let engine_config = EngineConfig::new(
    48,  // total fused input_dim (3 modalities × 16 each)
    16,  // model output_dim
);

let mut pipeline = MultiModalPipeline::builder()
    .engine_config(engine_config)
    .modality(ModalityType::Audio, 16)
    .modality(ModalityType::Video, 16)
    .modality(ModalityType::Sensor, 16)
    .fusion_strategy(FusionStrategy::EarlyFusion)
    .build()?;
```

`ModalityConfig::new` accepts the modality type and its expected input dimension. Feeding an array of the wrong length produces `InferenceError::DimensionMismatch` before fusion runs.

### 2.2 Forward Pass

```rust
use scirs2_core::ndarray::Array1;

let audio  = Array1::from_vec(vec![/* 16 audio features */]);
let video  = Array1::from_vec(vec![/* 16 frame embeddings */]);
let sensor = Array1::from_vec(vec![/* 16 IMU readings */]);

let output = pipeline.forward(&[
    (ModalityType::Audio,  audio),
    (ModalityType::Video,  video),
    (ModalityType::Sensor, sensor),
])?;
```

The `forward` call validates dimensions, applies modality-specific preprocessors, fuses all inputs, and calls the underlying engine.

### 2.3 Resetting State

Because the underlying `InferenceEngine` maintains SSM hidden state across calls, call `pipeline.reset()` when starting a new independent sequence:

```rust
pipeline.reset();
```

---

## 3. Temporal Alignment of Streams

Different sources typically arrive at different sample rates. Kizzasi does not impose a single global clock; instead, alignment is handled in the modality preprocessors before fusion.

### 3.1 Preprocessing with a Custom Resampler

```rust
use kizzasi_inference::{ModalityConfig, ModalityType, ModalityPreprocessor};
use scirs2_core::ndarray::Array1;
use std::sync::Arc;

// Video arrives at 30 Hz; audio at 44 100 Hz.
// The pipeline runs at 100 Hz. The video preprocessor must
// interpolate the last known frame embedding to each 10 ms window.
let video_at_100hz: ModalityPreprocessor = Arc::new(|embedding| {
    // embedding is already the frame-level feature vector; no further
    // resampling needed if the caller pre-computes the 100 Hz grid.
    Ok(embedding.clone())
});

// Audio is block-averaged into 10 ms frames (441 samples → 16 mel bins).
let audio_at_100hz: ModalityPreprocessor = Arc::new(|frame| {
    // Assume frame is a 16-bin mel vector produced upstream.
    Ok(frame.mapv(|x| x.clamp(-1.0, 1.0)))
});

let audio_config = ModalityConfig::new(ModalityType::Audio, 16)
    .preprocessor(audio_at_100hz);

let video_config = ModalityConfig::new(ModalityType::Video, 16)
    .preprocessor(video_at_100hz);
```

The preprocessor is typed as:
```rust
type ModalityPreprocessor =
    Arc<dyn Fn(&Array1<f32>) -> InferenceResult<Array1<f32>> + Send + Sync>;
```

It runs before fusion on every call to `forward`. For complex resampling (Sinc interpolation, mel filterbank), compute the resampled representation upstream and pass the result in.

### 3.2 Missing Modality Handling

If a modality is unavailable for a given step, supply a zero vector or the last known value. For safety-critical pipelines, the constraint layer (`GuardrailSet`) can reject outputs when a required modality has been absent for too many consecutive steps.

---

## 4. Fusion Strategies

`FusionStrategy` controls how per-modality arrays are combined into a single vector that the model receives.

### 4.1 Available Strategies

| Strategy | Behavior | Output Dimension |
|---|---|---|
| `EarlyFusion` | Concatenate all modalities (sorted by name) | sum of all `input_dim`s |
| `LateFusion` | Average modalities of equal dimension, then concatenate groups | same as EarlyFusion |
| `WeightedFusion` | Weighted concatenation, divided by total weight | same as EarlyFusion |
| `MaxPooling` | Element-wise max across modalities of equal dimension, then concatenate | same as EarlyFusion |
| `CrossAttention` | Dot-product attention-weighted combination, then concatenate | same as EarlyFusion |
| `Hierarchical` | Blend of EarlyFusion and WeightedFusion outputs | same as EarlyFusion |

### 4.2 Choosing a Strategy

**EarlyFusion** is the simplest and most commonly used. It concatenates all modality vectors in a deterministic order (alphabetically by modality name) and feeds the combined vector directly to the model. The model learns cross-modal interactions through its own layers.

**WeightedFusion** is appropriate when one modality is more reliable or informative than the others. Audio is typically given higher weight than sensor data in audio-guided synthesis tasks.

**CrossAttention** computes pairwise dot-product attention scores between modalities sharing the same dimension. This allows the fusion itself to down-weight a noisy modality without any learned parameters — the fusion is a function of the current input values only.

**Hierarchical** blends EarlyFusion and WeightedFusion by averaging their outputs element-wise. It provides a middle ground between full concatenation and weighted combination.

### 4.3 Setting Fusion Weights

```rust
let audio_config = ModalityConfig::new(ModalityType::Audio, 16)
    .fusion_weight(2.0);  // Audio is twice as important

let sensor_config = ModalityConfig::new(ModalityType::Sensor, 16)
    .fusion_weight(1.0);  // Sensor baseline weight
```

Weights are used only with `WeightedFusion`; other strategies ignore them.

---

## 5. Cross-Modal Tokenization

For pipelines that project modalities into a shared embedding space before the model, `kizzasi-tokenizer` provides `CrossModalTokenizer`.

```rust
use kizzasi_tokenizer::{
    CrossModalTokenizer, ModalityKind, ModalityTokenizerConfig,
};

let mut tokenizer = CrossModalTokenizer::new(64); // shared_dim = 64

// Register each modality with its raw input dimension
tokenizer.add_modality(
    ModalityKind::Audio,
    ModalityTokenizerConfig { input_dim: 16, codebook_size: 512 },
)?;
tokenizer.add_modality(
    ModalityKind::Video,
    ModalityTokenizerConfig { input_dim: 256, codebook_size: 1024 },
)?;
tokenizer.add_modality(
    ModalityKind::Sensor,
    ModalityTokenizerConfig { input_dim: 8, codebook_size: 256 },
)?;
```

`CrossModalTokenizer` applies a per-modality linear encoder, a discrete codebook lookup, an alignment projection, and a learned modality-type embedding. All three modalities emerge in the same 64-dimensional space, ready for concatenation or attention.

`ModalityKind::Control` is also available for robot joint angles and action commands, and `ModalityKind::Custom(String)` for user-defined streams.

---

## 6. Practical Examples

### 6.1 Audio + Sensor: EQ Adjustment Guided by Sensor Input

An equalizer that adapts filter coefficients in real time based on environmental measurements (e.g., room temperature, humidity affecting speaker response).

```rust
use kizzasi_inference::{
    EngineConfig, FusionStrategy, ModalityConfig, ModalityPreprocessor,
    ModalityType, MultiModalPipeline, SamplingConfig, SamplingStrategy,
};
use kizzasi_logic::{ConstraintBuilder, Guardrail, GuardrailSet};
use scirs2_core::ndarray::Array1;
use std::sync::Arc;

// Audio: 32-bin mel spectrogram frame
// Sensor: temperature (1) + humidity (1) + pressure (1) = 3 values
// Output: 32 EQ gain coefficients (in dB)

let audio_normalize: ModalityPreprocessor = Arc::new(|frame| {
    let max = frame.iter().cloned().fold(0.0f32, f32::max);
    Ok(if max > 0.0 { frame.mapv(|x| x / max) } else { frame.clone() })
});

let sensor_normalize: ModalityPreprocessor = Arc::new(|readings| {
    // Clamp sensor readings to known physical bounds
    Ok(readings.mapv(|x| x.clamp(-5.0, 5.0)))
});

let audio_config = ModalityConfig::new(ModalityType::Audio, 32)
    .preprocessor(audio_normalize)
    .fusion_weight(3.0);

let sensor_config = ModalityConfig::new(ModalityType::Sensor, 3)
    .preprocessor(sensor_normalize)
    .fusion_weight(1.0);

// Constraint: EQ gains must stay in [-12 dB, +12 dB]
let mut guardrails = GuardrailSet::new();
let eq_bounds = ConstraintBuilder::new()
    .name("eq_gain_db")
    .greater_eq(-12.0)
    .less_eq(12.0)
    .build()?;
guardrails.add_global(Guardrail::new(eq_bounds, false));

let engine_config = EngineConfig::new(35, 32) // 32 + 3 = 35 input dims
    .sampling(
        SamplingConfig::new()
            .strategy(SamplingStrategy::Temperature)
            .temperature(0.5), // low temperature → stable EQ
    );

let mut pipeline = MultiModalPipeline::builder()
    .engine_config(engine_config)
    .add_modality(audio_config)
    .add_modality(sensor_config)
    .fusion_strategy(FusionStrategy::WeightedFusion)
    .build()?;

// Per-frame processing loop
loop {
    let mel_frame: Array1<f32> = /* compute from audio buffer */;
    let sensor_readings: Array1<f32> = /* read from hardware */;

    let eq_gains = pipeline.forward(&[
        (ModalityType::Audio, mel_frame),
        (ModalityType::Sensor, sensor_readings),
    ])?;

    apply_eq_filter(&eq_gains); // user code
}
```

### 6.2 Video + Audio: Synchronized Prediction

A world model that jointly predicts the next video frame embedding and the corresponding audio frame, maintaining audio-visual synchrony.

```rust
use kizzasi_inference::{
    EngineConfig, FusionStrategy, ModalityConfig, ModalityType,
    MultiModalPipeline,
};
use scirs2_core::ndarray::Array1;

// Video: 128-dim ViT patch embeddings
// Audio: 64-dim mel spectrogram
// Output: 192 = 128 video + 64 audio (split by the caller)

let engine_config = EngineConfig::new(192, 192);

let mut pipeline = MultiModalPipeline::builder()
    .engine_config(engine_config)
    .modality(ModalityType::Video, 128)
    .modality(ModalityType::Audio, 64)
    .fusion_strategy(FusionStrategy::EarlyFusion)
    .build()?;

// Autoregressive generation loop
let mut video_embed: Array1<f32> = initial_video_frame_embedding();
let mut audio_embed: Array1<f32> = initial_audio_frame_embedding();

for frame_idx in 0..total_frames {
    let output = pipeline.forward(&[
        (ModalityType::Video, video_embed.clone()),
        (ModalityType::Audio, audio_embed.clone()),
    ])?;

    // Split output: first 128 elements → next video frame,
    //               next 64 elements → next audio frame.
    // Modalities are sorted alphabetically by name: audio < video,
    // so audio comes first in EarlyFusion output.
    let (next_audio_slice, next_video_slice) = output.as_slice()
        .unwrap()
        .split_at(64);

    audio_embed = Array1::from_vec(next_audio_slice.to_vec());
    video_embed = Array1::from_vec(next_video_slice.to_vec());

    decode_and_emit(frame_idx, &video_embed, &audio_embed);
}
```

**EarlyFusion sort order note:** Modalities are concatenated in alphabetical order of their `ModalityType::name()` string: `"audio"` < `"sensor"` < `"text"` < `"video"`. When splitting the output, account for this ordering.

---

## 7. Configuration Patterns

### 7.1 Setting `input_dim` Correctly

The `engine_config` `input_dim` must equal the total number of elements that the fused vector presents to the model. For `EarlyFusion`, this is the sum of all per-modality `input_dim`s:

```rust
// Three modalities: audio (32), video (128), sensor (8)
// EarlyFusion input_dim = 32 + 128 + 8 = 168

let engine_config = EngineConfig::new(168, output_dim);

let pipeline = MultiModalPipeline::builder()
    .engine_config(engine_config)
    .modality(ModalityType::Audio, 32)
    .modality(ModalityType::Video, 128)
    .modality(ModalityType::Sensor, 8)
    .fusion_strategy(FusionStrategy::EarlyFusion)
    .build()?;
```

For `LateFusion` and `MaxPooling`, arrays of different dimensions are grouped by dimension and concatenated after pooling; the resulting size is still the sum of all `input_dim`s (one representative per group). Set `input_dim` to the same value as EarlyFusion for safety.

For `WeightedFusion` the output has the same total dimension as the concatenation of all modalities, normalized by total weight. The `input_dim` is again the sum of all per-modality dimensions.

### 7.2 Using CrossModalTokenizer as a Pre-Stage

When modality dimensions differ significantly, consider projecting each into a common dimension with `CrossModalTokenizer` before `MultiModalPipeline`:

```rust
// Encode each modality into a shared 64-dimensional space
let audio_tok = tokenizer.encode_modality(ModalityKind::Audio, &raw_audio)?;
let video_tok = tokenizer.encode_modality(ModalityKind::Video, &raw_video)?;

// Now all modalities are 64-dim; pass to a pipeline with input_dim = 128
let output = pipeline.forward(&[
    (ModalityType::Audio, audio_tok),
    (ModalityType::Video, video_tok),
])?;
```

This reduces model complexity because the SSM sees a uniform-dimensional sequence rather than a mixed-width concatenation.

---

## 8. Constraints Across Modalities

`kizzasi-logic` constraints apply to the output vector after fusion and model forward pass. For multi-modal outputs, you can define constraints that couple values from different modalities.

### 8.1 Per-Modality Range Constraints

```rust
use kizzasi_logic::{ConstraintBuilder, Guardrail, GuardrailSet};

let mut guardrails = GuardrailSet::new();

// Audio output indices [0, 31] must be in [-1, 1]
for i in 0..32 {
    let c = ConstraintBuilder::new()
        .name(&format!("audio_range_{i}"))
        .dimension(i)
        .greater_eq(-1.0)
        .less_eq(1.0)
        .build()?;
    guardrails.add_global(Guardrail::new(c, false));
}

// Sensor output indices [32, 34] must be physically non-negative
for i in 32..35 {
    let c = ConstraintBuilder::new()
        .name(&format!("sensor_nonneg_{i}"))
        .dimension(i)
        .greater_eq(0.0)
        .build()?;
    guardrails.add_global(Guardrail::new(c, false));
}
```

### 8.2 Temporal Logic Constraints

For properties that span multiple steps (e.g., "sensor reading must not decrease for more than three consecutive steps"), use the `TemporalConstraintEnforcer` from `kizzasi-inference`:

```rust
use kizzasi_inference::temporal::{LTLFormula, TemporalConstraintEnforcer};

// "Always: sensor output must remain above 0"
let always_positive = LTLFormula::Always(
    Box::new(LTLFormula::Atomic(|x| x[32] > 0.0)), // index 32 = first sensor dim
);

let enforcer = TemporalConstraintEnforcer::new(always_positive, history_len);
```

Signal Temporal Logic (`STLFormula`) supports continuous robustness semantics and time-bounded operators such as `Always[t1, t2]` and `Eventually[t1, t2]`.

### 8.3 Cross-Modal Coupling Constraints

A coupling constraint captures a relationship between values from different modalities in the output vector, for example "the predicted audio energy must not exceed the video motion magnitude":

```rust
use kizzasi_logic::NonlinearConstraint;

let coupling = NonlinearConstraint::new(|output: &Array1<f32>| {
    let audio_energy: f32 = output.slice(s![0..32]).iter().map(|x| x * x).sum();
    let video_motion: f32 = output.slice(s![32..160]).iter().map(|x| x.abs()).sum();
    // Constraint: audio_energy <= video_motion + slack
    audio_energy - video_motion - 0.5
}, NonlinearConstraintType::LessEq);
```

---

## 9. Performance Tips

### 9.1 Fuse Preprocessing and Tokenization

Each `ModalityPreprocessor` closure runs inline before fusion. Avoid heavy allocations inside the closure; prefer views and in-place operations where possible. If the preprocessing involves an FFT or filterbank, compute it in a worker thread and pass the result as a pre-computed embedding.

### 9.2 Batch Multiple Time Steps

When you have a buffer of N time steps available (e.g., 100 ms of audio at 10 ms frame rate = 10 frames), call `pipeline.forward` 10 times in a loop rather than spinning a background task per frame. The SSM hidden state update is sequential within a session, but multiple independent sessions can run in parallel on separate threads since `MultiModalPipeline` is not `Sync` by default (wrap in a `Mutex` or use per-thread instances).

### 9.3 Memory Management

Modality arrays are cloned inside `forward` during the preprocessing step. For high-throughput workloads:

- Pre-allocate `Array1<f32>` buffers outside the loop and reuse them.
- Use the `TensorPool` from `kizzasi-inference` to avoid per-frame allocations.
- For the `CrossModalTokenizer`, the encoder weight matrices are shared via `Arc`; constructing multiple `CrossModalTokenizer` instances that share the same codebook is inexpensive.

### 9.4 Choosing the Right Model Size

For multi-modal pipelines the effective `input_dim` grows with the number of modalities. A model of `hidden_dim = 256` processes a 3-modality EarlyFusion input of 168 dimensions without trouble, but the hidden state size and weight parameters scale with `hidden_dim`. As a rule of thumb:

- 1–2 modalities: `hidden_dim = 64–128`
- 3–4 modalities: `hidden_dim = 128–256`
- 5+ modalities or large per-modality dims: `hidden_dim >= 256`, consider `CrossModalTokenizer` to normalize input size first.

### 9.5 Leveraging CrossAttention for Noise Robustness

`FusionStrategy::CrossAttention` computes pairwise dot-product similarity between modality vectors. A noisy modality whose embedding is nearly orthogonal to the others receives a lower attention weight automatically, without any learned parameters. This is computationally free relative to model complexity and can improve robustness in sensor-degraded conditions.

### 9.6 GPU Acceleration

For pipelines with many modalities and large per-modality dimensions, offload the SSM scan to `kizzasi-webgpu`:

```toml
[dependencies]
kizzasi-webgpu = { version = "0.2", features = ["webgpu"] }
```

The `WebGpuSsmBackend` replaces the CPU parallel scan while the fusion and preprocessing remain on CPU. This is most beneficial when `hidden_dim` exceeds 512 and the sequence is processed in batches of 32 or more frames.

---

## 10. Further Reading

- `docs/architecture_overview.md` — crate dependency graph, trait definitions, deployment options
- `crates/kizzasi-inference/examples/multimodal_with_constraints.rs` — complete multi-modal example with all fusion strategies
- `crates/kizzasi-inference/examples/ensemble_streaming.rs` — ensemble with streaming adapter
- `crates/kizzasi-tokenizer/examples/audio_pipeline.rs` — mu-law and VQ-VAE audio tokenization
- `crates/kizzasi-tokenizer/src/cross_modal.rs` — `CrossModalTokenizer` implementation and `ModalityKind` variants
- `crates/kizzasi-inference/src/multimodal.rs` — `MultiModalPipeline`, `FusionStrategy`, and `ModalityConfig` source
