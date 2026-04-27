# PyTorch Migration Guide

This guide explains how to migrate PyTorch / HuggingFace workflows to Kizzasi.

---

## Overview: Philosophy Differences

| Aspect | PyTorch / HuggingFace | Kizzasi |
|---|---|---|
| **Paradigm** | Language Model (discrete tokens) | AGSP (continuous signal prediction) |
| **Input** | Integer token IDs | Continuous float vectors |
| **State** | External KV-cache or `hidden_states` arg | Internal `HiddenState`, managed implicitly |
| **Inference API** | `output = model(input_ids, past_key_values=…)` | `output = predictor.step(&input)?` |
| **Constraints** | None (post-hoc filtering) | Built-in `GuardrailSet` / `ConstraintBuilder` |
| **Safety** | Pickle-based `.pth` | SafeTensors (zero-copy, no arbitrary code exec) |

Kizzasi is not a drop-in replacement for `transformers.AutoModel`.  It is a
**signal predictor**: audio samples, sensor readings, robot joint angles, and
video pixels are all equivalent continuous streams.  Text can be tokenised to
continuous representations (via VQ-VAE) but this is not the primary use case.

---

## Weight Formats

| Format | Extension | Producer | Kizzasi support |
|---|---|---|---|
| **SafeTensors** | `.safetensors` | HuggingFace | Full (`ModelLoader`) |
| **GGUF** | `.gguf` | llama.cpp / Ollama | Full (`GgufFile`) |
| **JSON weight map** | `.json` | Kizzasi native | Full (`save/load_weights_json`) |
| **PyTorch checkpoint** | `.pth` / `.pt` | PyTorch | Partial (`PyTorchConverter`) |

### Format recommendation

- Use **SafeTensors** for interoperability with HuggingFace Hub.
- Use **JSON** for Kizzasi-native models; it requires no extra dependencies at runtime.
- Use **GGUF** for deployment on constrained hardware with quantised weights.
- Avoid `.pth` / `.pt` in new code — pickle execution is a security risk.

---

## Loading HuggingFace Models

### `HfHubClient` and `load_from_hub`

Add the `hf-hub` feature to your `Cargo.toml`:

```toml
[dependencies]
kizzasi-model = { version = "0.1", features = ["hf-hub"] }
```

Then:

```rust
use kizzasi_model::hf_hub::{HfHubClient, HfHubConfig, load_from_hub};
use kizzasi_model::loader::NameRemapper;

// Reads $HF_TOKEN from the environment automatically
let cfg = HfHubConfig::default();

// Downloads and caches the file under ~/.cache/kizzasi/hub
let raw_weights = load_from_hub(
    "state-spaces/mamba-130m",   // repo_id
    "model.safetensors",          // filename
    &cfg,
)?;

// Translate HF key names to Kizzasi names
let weights = NameRemapper::new().remap_map(raw_weights);

// Load into your model
mamba_model.load_weights_from_dict(&weights)?;
```

### Authentication for private repos

```rust
let cfg = HfHubConfig {
    token: Some("hf_...".to_string()),
    ..Default::default()
};
```

Or set the `HF_TOKEN` environment variable — `HfHubConfig::default()` reads it
automatically.

### Cache behaviour

Files are stored under `HfHubConfig::cache_dir` (default:
`~/.cache/kizzasi/hub`).  Subsequent calls with the same `(repo_id, filename,
revision)` tuple reuse the cached file without re-downloading.

---

## Key Name Translation

HuggingFace Mamba checkpoints use a different naming convention from Kizzasi's
internal names.  `NameRemapper` handles the common cases automatically.

### Complete mapping table

| HuggingFace key pattern | Kizzasi key |
|---|---|
| `embedding.weight` | `input_proj` |
| `lm_head.weight` | `output_proj` |
| `layers.{n}.mixer.in_proj.weight` | `layers.{n}.input_proj` |
| `layers.{n}.mixer.out_proj.weight` | `layers.{n}.output_proj` |
| `layers.{n}.attn.q_proj.weight` | `layers.{n}.attention.q` |
| `layers.{n}.attn.k_proj.weight` | `layers.{n}.attention.k` |
| `layers.{n}.attn.v_proj.weight` | `layers.{n}.attention.v` |
| `layers.{n}.attn.o_proj.weight` | `layers.{n}.attention.out` |
| `layers.{n}.mlp.gate_proj.weight` | `layers.{n}.ff.gate` |
| `layers.{n}.mlp.up_proj.weight` | `layers.{n}.ff.up` |
| `layers.{n}.mlp.down_proj.weight` | `layers.{n}.ff.down` |
| *(any other key)* | passed through unchanged |

### Custom mappings

For non-standard checkpoints, provide a `HashMap<String, String>` to
`WeightLoader::with_name_mapping`:

```rust
use std::collections::HashMap;
use kizzasi_model::loader::{ModelLoader, WeightLoader};

let custom_map: HashMap<String, String> = [
    ("backbone.layers.0.mixer.in_proj.weight", "layers.0.in_proj"),
    ("backbone.layers.0.mixer.A_log",          "layers.0.ssm.log_a"),
].iter()
 .map(|(k, v)| (k.to_string(), v.to_string()))
 .collect();

let loader = ModelLoader::new("custom_model.safetensors")?;
let wl = WeightLoader::new(loader).with_name_mapping(custom_map);
```

---

## Mamba Architecture Mapping

### PyTorch Mamba → Kizzasi Mamba

| PyTorch (`mamba_ssm.Mamba`) | Kizzasi (`kizzasi_model::mamba`) |
|---|---|
| `d_model` | `hidden_dim` |
| `d_state` | `state_dim` |
| `d_conv` | (fixed at 4 in current impl) |
| `expand` | (expand factor implicit in `inner_dim`) |
| `MambaBlock.in_proj` | `layers[i].in_proj` |
| `MambaBlock.conv1d` | `layers[i].conv` |
| `MambaBlock.x_proj` | split → `ssm.delta_proj`, `ssm.b_proj`, `ssm.c_proj` |
| `MambaBlock.dt_proj` | `layers[i].ssm.delta_proj` |
| `MambaBlock.A_log` | `layers[i].ssm.log_a` |
| `MambaBlock.D` | `layers[i].ssm.d_skip` |
| `MambaBlock.out_proj` | `layers[i].out_proj` |

**Important**: HuggingFace `x_proj` is a combined matrix
`[time_step_rank + state_size*2, intermediate_size]` that must be split into
three separate projections (`delta`, `B`, `C`).  This splitting is handled by
`kizzasi_model::pytorch_compat::PyTorchConverter`.

### Mamba2 differences

Mamba2 uses the **State Space Duality (SSD)** framework.  Key differences:

- Uses multi-head formulation (`num_heads` parameter)
- Replaces `A_log` with per-head learnable `A` matrices
- Adds `dt_limit` and `chunk_size` parameters
- `x_proj` is replaced by separate `z`, `x`, `B`, `C`, `dt` projections

---

## RWKV Architecture Mapping

### RWKV v6 → Kizzasi RWKV

| PyTorch RWKV | Kizzasi RWKV |
|---|---|
| `time_mix.time_decay` | `layers[i].time_mix.w_time_decay` |
| `time_mix.time_first` | `layers[i].time_mix.w_time_first` |
| `time_mix.key.weight` | `layers[i].time_mix.w_k` |
| `time_mix.value.weight` | `layers[i].time_mix.w_v` |
| `time_mix.receptance.weight` | `layers[i].time_mix.w_r` |
| `time_mix.output.weight` | `layers[i].time_mix.w_o` |
| `channel_mix.key.weight` | `layers[i].channel_mix.w_k` |
| `channel_mix.value.weight` | `layers[i].channel_mix.w_v` |
| `channel_mix.receptance.weight` | `layers[i].channel_mix.w_r` |

### RWKV v7

RWKV v7 (`Rwkv7Model`) introduces time-varying WKV with learnable `w_a` and
`w_b` matrices per head.  See `kizzasi_model::rwkv7` for details.

---

## Inference API

### Python PyTorch

```python
model = Mamba(d_model=64, d_state=16)
state = None
outputs = []
for x in input_stream:
    x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
    y, state = model(x_tensor, inference_params=state)
    outputs.append(y.squeeze().tolist())
```

### Rust Kizzasi

```rust
let mut predictor = KizzasiBuilder::new()
    .model_type(ModelType::Mamba)
    .hidden_dim(64)
    .state_dim(16)
    .build()?;

let mut outputs = Vec::new();
for x in &input_stream {
    let y = predictor.step(x)?;   // state managed internally
    outputs.push(y);
}
```

### Key differences

| Aspect | PyTorch | Kizzasi |
|---|---|---|
| State threading | Explicit `state` argument | Implicit (call `reset()` to zero) |
| Multi-step | Python `for` loop | `predictor.predict_n(&x, n)?` |
| Batch | `model(batch_tensor)` | `predictor.predict_batch(&inputs)?` |
| Branching | `state.clone()` | `predictor.fork()?` |

---

## Training

Kizzasi includes a training loop (`kizzasi_model::training_loop::TrainingLoop`)
that mirrors the PyTorch `Trainer` API.

```rust
use kizzasi_model::training_loop::{TrainingConfig, TrainingLoop, AdamOptimizer};

let config = TrainingConfig {
    learning_rate: 1e-4,
    num_epochs: 10,
    batch_size: 32,
    ..Default::default()
};

let mut trainer = TrainingLoop::new(config);
let result = trainer.train(&mut model, &mut data_provider, &mut optimizer)?;
println!("Final loss: {:.4}", result.final_loss);
```

### Loss functions

Kizzasi uses mean-squared error (MSE) for continuous signal prediction.
Custom loss functions can be injected via the `TrainingCallback` trait.

### Constraint-aware training

Add `kizzasi-logic` constraints to the training objective via
`ConstrainedInference`:

```rust
use kizzasi_logic::ConstrainedInference;

let constrained = ConstrainedInference::new(model, guardrails);
// constrainted.step() enforces guardrails at inference time
```

---

## Troubleshooting

### "Tensor not found: layers.0.in_proj"

**Cause**: Weight keys in the checkpoint use a different naming convention.

**Fix**: Run `ModelLoader::new(path)?.print_summary()` to list all keys, then
build a custom `HashMap<String, String>` mapping and pass it to
`WeightLoader::with_name_mapping(map)`.

### Dimension mismatch on load

**Cause**: Model was built with different `hidden_dim` / `state_dim` than the
checkpoint.

**Fix**: Use `ModelLoader::tensor_info(name)` to inspect shapes, then
reconstruct the model with matching dimensions before loading.

### Very high latency on first step

**Cause**: Lazy initialisation of SIMD buffers and OS page faults on cold memory.

**Fix**: Run 10–20 warmup steps with `predictor.step(&dummy_input)?` before
measuring or serving production traffic.

### NaN / Inf outputs

**Cause**: Input values outside the expected range, or uninitialised weights
(e.g. loading partially-matching checkpoint).

**Fix**:
1. Clamp inputs to `[-1, 1]` (or the model's training distribution).
2. Load with `WeightLoader::strict(false)` and inspect which weights are missing.
3. Check for `log_a` values that are too large — clamp to `[-10, 0]`.

### "missing feature `hf-hub`"

**Fix**: Add `features = ["hf-hub"]` to the `kizzasi-model` dependency:

```toml
kizzasi-model = { version = "0.1", features = ["hf-hub"] }
```
