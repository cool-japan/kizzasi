# kizzasi

**Kizzasi** (兆し) — Autoregressive General-Purpose Signal Predictor (AGSP)

Python bindings for the [Kizzasi](https://github.com/cool-japan/kizzasi) Rust framework, providing high-performance signal prediction using State Space Models (Mamba, RWKV, S4, Spiking NNs, Neural ODE) with neuro-symbolic constraint enforcement.

![status](https://img.shields.io/badge/status-alpha-orange)
![version](https://img.shields.io/badge/version-0.2.4-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

All classes below are fully implemented — 0 stub markers. Covered by both the crate's Rust-side unit test suite (`cargo nextest run -p kizzasi-python --all-features`, 100+ tests, including `Python::attach`-based tests that call the real `#[pymethods]` through actual `PyArray` conversions rather than the wrapped Rust types directly) and a Python-level `pytest` suite (`tests/test_smoke.py`) that imports the built wheel and exercises every registered class through the public Python API.

## Features

- **`Config` / `ModelType`** — builder-style configuration with `audio`/`robotics`/`sensor`/`lightweight` presets over four backbones (Mamba, Mamba2, S4, RWKV); `model_type` accepts either a `str` or a `ModelType` instance
- **`Predictor`** — O(1)-per-step autoregressive prediction (`step`/`predict_n`/`step_list`) with per-dimension `ConstraintSpec` guardrails (clamp or hard-reject); compute runs with the GIL released
- **`EnsemblePredictor`** — multi-model ensembles with six voting strategies (average, weighted, weighted-average, median, confidence, majority)
- **`OptimizedPredictor`** — a TTL-based LRU result cache (real, measurable via `cache_stats()`) plus `enable_simd`/`workspace_pool_size` constructor knobs that are accepted for forward compatibility but currently have **no effect on computation** — see [Optimized Predictor](#optimized-predictor)
- **`LoRAAdapter`** — register per-module Low-Rank Adaptation layers over NumPy base weight matrices; run, merge, and unmerge corrections in place
- **`SamplingConfig` / `Sampler`** — greedy, temperature, top-k, and top-p sampling, single-vector and batched
- **`BeamSearch` / `ConstrainedBeamSearch` / `RejectionSampler`** — autoregressive decoding utilities, with Python-callable hard/soft constraints (a raising constraint callable propagates as a real exception, not a silent "violated")
- **`MuLawCodec`** — mu-law (ITU-T G.711) audio companding: scalar and vectorized quantize/dequantize, matching the same codec's WASM binding surface in `kizzasi-tokenizer`

## Module Layout

The Rust source (`src/`) is split by responsibility; each module maps 1:1 to a section below and to the classes registered in the `kizzasi` extension module:

| Module | Exposes |
|---|---|
| `config` | `Config`, `ModelType` |
| `predictor` | `Predictor`, `ConstraintSpec` |
| `ensemble` | `EnsemblePredictor` |
| `optimized` | `OptimizedPredictor` |
| `lora` | `LoRAAdapter` |
| `sampling` | `SamplingConfig`, `Sampler` |
| `beam_search` | `BeamSearch`, `ConstrainedBeamSearch`, `RejectionSampler` |
| `mulaw` | `MuLawCodec` |

All classes are re-exported from the single `kizzasi` extension module — the split is internal and does not change any import path. `python/kizzasi/__init__.py` imports and re-exports every one of them by name (checked by `tests/test_smoke.py::test_all_classes_are_exported`, which asserts every name in `__all__` actually resolves).

## Installation

```bash
pip install kizzasi          # latest release
pip install kizzasi==0.2.4   # pin to the current release
```

## Quick Start

```python
import numpy as np
import kizzasi

# Configure a signal predictor
cfg = kizzasi.Config(
    input_dim=8,
    output_dim=8,
    hidden_dim=64,
    num_layers=2,
    model_type="s4",  # valid values: "mamba", "mamba2", "s4", "rwkv"
)

predictor = kizzasi.Predictor(cfg)

# Single-step prediction (O(1) per step for SSMs)
x = np.random.randn(8).astype(np.float32)
y = predictor.step(x)

# Multi-step prediction
ys = predictor.predict_n(x, n_steps=100)  # shape: (100, 8)

# Reset state
predictor.reset()
```

## Presets

```python
# Audio prediction (44.1 kHz by default; context_window scales with
# sample_rate so the *value* stays proportional to the 44.1kHz default
# across rates -- see the Config section below: context_window is metadata
# only today, so this scaling does not translate into an enforced
# real-time horizon)
cfg = kizzasi.Config.audio(sample_rate=16000)

# Robot control (6-DOF joint angles → actions)
cfg = kizzasi.Config.robotics(state_dim=6, action_dim=6)

# IoT sensor streams
cfg = kizzasi.Config.sensor(num_sensors=9)

# Lightweight configuration for resource-constrained environments
cfg = kizzasi.Config.lightweight(input_dim=4, output_dim=4)
```

## Config

```python
kizzasi.Config(
    input_dim,
    output_dim,
    hidden_dim=256,
    num_layers=4,
    state_dim=16,
    context_window=8192,   # accepted, validated (> 0), readable back -- no effect on
                           # memory or computation today, see note below
    model_type="mamba2",   # "mamba", "mamba2", "s4", "rwkv" -- or a ModelType instance
)
```

`Config(...)` and its property setters never validate or allocate — they are
plain data holders. Validation (every dimension `> 0`, and a total
parameter-count cap) happens when a `Predictor` / `EnsemblePredictor` /
`OptimizedPredictor` is actually constructed from the config, raising
`ValueError` instead of letting an oversized `hidden_dim` abort the process
when the allocator gives up.

> **Note:** `context_window` is accepted, range-checked (`> 0`), and
> readable back via `Config.context_window` / `Predictor.context_window` —
> but it does not bound memory or computation. The underlying `SelectiveSSM`
> engine (used by the `mamba2` backbone) keeps a fixed-size per-layer hidden
> state that never grows with the number of steps taken, so there is no
> history buffer for this value to size or truncate: two configs that differ
> only in `context_window` build byte-identical predictors and produce
> identical predictions for the same input sequence. `Config.audio()`'s
> `sample_rate`-based scaling of `context_window` (see [Presets](#presets))
> is real arithmetic, but the resulting number is, likewise, not enforced
> anywhere downstream yet.

### ModelType enum

```python
kizzasi.ModelType.MAMBA
kizzasi.ModelType.MAMBA2
kizzasi.ModelType.S4
kizzasi.ModelType.RWKV
```

`model_type` accepts a `ModelType` instance anywhere it accepts a `str`:

```python
cfg = kizzasi.Config(4, 4, model_type=kizzasi.ModelType.RWKV)
cfg.model_type = kizzasi.ModelType.S4   # property setter also accepts either
assert cfg.model_type == "s4"           # always reads back as the canonical str
```

## Predictor

> **Note:** `Predictor` is pinned to the thread that created it (create one
> instance per thread; do not share a single instance across threads).
> `step`/`predict_n`/`step_list` release the GIL for the duration of their
> computation, so separate per-thread instances run their SSM math in true
> parallel rather than serializing on the GIL.

```python
predictor = kizzasi.Predictor(cfg)

predictor.step(input)           # single-step prediction → np.ndarray
predictor.predict_n(input, n)   # multi-step prediction  → np.ndarray (n, output_dim)
predictor.step_list(input)      # single step, returns Python list

predictor.reset()               # reset internal SSM state
```

## Guardrails

Guardrails enforce per-dimension constraints on predictions and can hard-reject out-of-range outputs.

```python
import kizzasi

spec_angle   = kizzasi.ConstraintSpec("joint_angle", min_val=-3.14, max_val=3.14)
spec_torque  = kizzasi.ConstraintSpec("torque",      min_val=-100.0, max_val=100.0,
                                      dimension=1, hard_reject=True)

predictor.set_guardrails([spec_angle, spec_torque])

# Check / remove
if predictor.has_guardrails():
    predictor.clear_guardrails()
```

### ConstraintSpec

```python
kizzasi.ConstraintSpec(
    name,                # str  — human-readable label
    min_val=None,        # float | None — lower bound (inclusive)
    max_val=None,        # float | None — upper bound (inclusive)
    dimension=None,      # int  | None — output dimension index to constrain (None = all)
    hard_reject=False,   # bool — raise an error instead of clamping when violated
)
```

## Sampling

`SamplingConfig` is a builder-style configuration for four core sampling strategies; `Sampler` draws values from raw logit vectors according to that configuration.

```python
import numpy as np
import kizzasi

config = kizzasi.SamplingConfig()   # default: strategy="greedy", temperature=1.0
config.strategy("top_k")            # "greedy" | "temperature" | "top_k" | "top_p"
config.top_k(3)                     # k >= 1; also flips strategy to "top_k"
config.temperature(1.5)             # must be finite and > 0
config.seed(42)

sampler = kizzasi.Sampler(config)   # config is cloned; later mutation of `config` is not reflected

logits = np.array([1.0, 3.0, 0.5, 2.5, 1.8], dtype=np.float32)
sampler.sample(logits)              # -> float, single sampled value

batch_logits = np.random.randn(8, 5).astype(np.float32)
sampler.sample_batch(batch_logits)  # -> np.ndarray shape (8,)

sampler.strategy_name               # -> "top_k"

config.get_temperature              # -> 1.5  (read-back getters; note the `get_` prefix --
config.get_top_k                    # -> 3     `temperature`/`top_k`/`top_p`/`seed` are already
config.get_top_p                    # -> None  taken by the builder-style setters above)
config.get_seed                     # -> 42
```

> **Note:** `Sampler` is stateful — it reuses one RNG across calls, so a fixed seed produces a deterministic *sequence* of samples, not a repeated single value. `top_p(p)` takes `p` in `(0, 1]` and also flips the strategy to `"top_p"`; strategy names are case-insensitive and accept aliases (`"temp"`, `"topk"`/`"top-k"`, `"topp"`/`"top-p"`/`"nucleus"`).

> **Note:** `Sampler(config)` raises `ValueError` if `config`'s strategy is `"top_k"`/`"top_p"` but the matching `top_k(k)`/`top_p(p)` was never called — previously this silently sampled with the engine's internal default (`k=10` or `p=0.9`) with no indication anywhere that the value had never been set.

## Beam Search & Rejection Sampling

Three decoding utilities operate on logits matrices supplied one step at a time. `BeamSearch` is unconstrained; `ConstrainedBeamSearch` and `RejectionSampler` accept arbitrary Python callables as hard or soft constraints on the candidate sequence.

> **Note:** if a constraint callable raises (a typo, an `IndexError` from indexing the initial empty beam, a non-`bool` return, ...), that exception propagates out of `expand()`/`sample()` — wrapped in a `RuntimeError` with the original chained as `__cause__` (`except RuntimeError as e: e.__cause__`), or re-raised unchanged for `KeyboardInterrupt`/`SystemExit`. It is never silently treated as "constraint violated".

```python
import numpy as np
import kizzasi

bs = kizzasi.BeamSearch(beam_width=3)

# First call: exactly 1 active beam -> logits shape (1, vocab_size)
logits = np.random.randn(1, 8).astype(np.float32)
bs.expand(logits)

# Later calls: beam_width active beams -> logits shape (beam_width, vocab_size)
logits3 = np.random.randn(3, 8).astype(np.float32)
bs.expand(logits3)

bs.best_sequence()   # -> np.ndarray shape (n,) float32, or None if there are no beams
bs.best_log_prob()   # -> float, or None
bs.num_beams()       # -> int
bs.all_beams()       # -> list[dict], each {"sequence": np.ndarray, "log_prob": float}
```

### ConstrainedBeamSearch

```python
cbs = kizzasi.ConstrainedBeamSearch(beam_width=4)

# constraint_fn: Python callable (sequence: list[float]) -> bool
cbs.add_constraint(lambda seq: len(seq) == 0 or seq[-1] < 5.0)

# Optional: soft constraints subtract a log-prob penalty instead of discarding the beam
cbs.enable_soft_constraints(0.5)

logits = np.random.randn(1, 10).astype(np.float32)
cbs.expand(logits)

cbs.best_sequence()     # -> np.ndarray or None
cbs.num_beams()         # -> int
cbs.num_constraints()   # -> int
cbs.all_beams()         # -> list[dict], same shape as BeamSearch.all_beams()
```

### RejectionSampler

```python
config = kizzasi.SamplingConfig()
config.strategy("temperature")
config.temperature(1.0)
config.seed(42)

rs = kizzasi.RejectionSampler(config)         # config is cloned at construction
rs.add_constraint(lambda seq: seq[-1] < 3.0)  # receives context + [candidate]
rs.set_max_attempts(50)
rs.set_fallback_strategy("best_candidate")    # "best_candidate"/"best" | "greedy" | "error"

logits = np.array([2.0, 2.5, 1.8, 0.1, 0.05], dtype=np.float32)
value = rs.sample(logits, context=[])   # -> float
rs.num_constraints()                    # -> int
```

## Ensemble Prediction

`EnsemblePredictor` builds `n_models` independent predictors from one shared `Config` and combines their step-by-step outputs with a configurable voting strategy.

```python
import numpy as np
import kizzasi

cfg = kizzasi.Config(input_dim=4, output_dim=4, hidden_dim=32, num_layers=2)

ensemble = kizzasi.EnsemblePredictor(
    cfg, n_models=3, voting="weighted_average", weights=[1.0, 0.7, 0.3],
)
# voting: "average" | "weighted" | "weighted_average" | "median" | "confidence" | "majority"
# weights: optional, must match n_models in length and be non-negative; defaults to all 1.0

x = np.random.randn(4).astype(np.float32)
y = ensemble.step(x)                    # -> np.ndarray shape (output_dim,)
ys = ensemble.predict_n(x, n_steps=50)  # -> np.ndarray shape (50, output_dim); requires input_dim == output_dim

ensemble.set_weight(1, 0.9)   # re-weight model index 1 (0-based)

stats = ensemble.stats()
# {"num_models", "total_predictions", "avg_variance", "voting_strategy", "model_weights"}

ensemble.num_models        # -> int
ensemble.voting_strategy   # -> str
ensemble.reset()           # reset every ensemble member's internal state
```

## Optimized Predictor

`OptimizedPredictor` wraps a predictor with an optional TTL-based LRU result cache. `enable_simd` and `workspace_pool_size` are accepted and reported back for forward compatibility, but **currently have no effect on computation** — see the note below before relying on them for performance.

```python
import numpy as np
import kizzasi

cfg = kizzasi.Config(input_dim=4, output_dim=4, hidden_dim=32, num_layers=2)

opt = kizzasi.OptimizedPredictor(
    cfg,
    cache_ttl_ms=1000,          # 0 disables the result cache; default 1000
    enable_simd=True,           # accepted, reported via .simd_enabled -- no computational effect today
    workspace_pool_size=16,     # accepted -- no computational effect today
    result_cache_size=1000,     # default 1000
)

x = np.random.randn(4).astype(np.float32)
y = opt.step(x)                    # -> np.ndarray shape (output_dim,)
ys = opt.predict_n(x, n_steps=20)  # -> np.ndarray shape (20, output_dim); bypasses the result cache

opt.cache_stats()          # {"size", "capacity", "hits", "misses", "hit_rate", "enabled"} -- real
opt.optimization_stats()   # {"total_predictions", "cached_predictions", "cache_time_saved_us",
                            #  "avg_prediction_time_us",
                            #  "workspace_pool_hits", "workspace_allocations"}  -- last two always 0

opt.clear_cache()          # alias for reset(): clears the cache *and* predictor state together
```

> **Note:** `OptimizedPredictor` does not expose a cache-only reset — `reset()` and `clear_cache()` both clear the result cache and the underlying predictor's hidden state.

> **Note:** The result cache (`cache_ttl_ms`/`result_cache_size`, `cache_stats()`) is real and measurable. `enable_simd` and `workspace_pool_size` are stored and reported back (`.simd_enabled`, and `optimization_stats()`'s dict) but the underlying `kizzasi` engine does not yet implement SIMD kernel dispatch or workspace-pool accounting — `step`/`predict_n` run identical code regardless of `enable_simd`'s value, and `workspace_pool_hits`/`workspace_allocations` are always `0`. Wiring real implementations for these requires changes to the `kizzasi` crate's optimization engine, tracked separately from this binding crate.

## LoRA Adapters

`LoRAAdapter` applies a Low-Rank Adaptation correction (`y = W x + alpha/rank * B(A x)`) on top of NumPy base weight matrices, registered one module at a time.

```python
import numpy as np
import kizzasi

adapter = kizzasi.LoRAAdapter("my_adapter", rank=8, alpha=16.0, dropout=0.0)  # dropout defaults to 0.0

base = np.random.randn(64, 128).astype(np.float32)   # (out_features, in_features)
adapter.add_layer("layer_1", base)

x = np.random.randn(128).astype(np.float32)
y = adapter.forward("layer_1", x)   # -> np.ndarray shape (out_features,) = (64,)

adapter.merge_all()      # fold every LoRA correction into its base weight, in place
adapter.unmerge_all()    # reverse merge_all()

adapter.total_parameters()      # -> int: sum of rank * (in_features + out_features) over all modules
adapter.avg_parameter_ratio()   # -> float: avg LoRA-params / base-params ratio (0.0 with no layers)
adapter.module_names()          # -> list[str] (insertion order not preserved)
len(adapter)                    # -> int, number of registered modules

adapter.name          # -> "my_adapter"
adapter.rank          # -> 8
adapter.alpha         # -> 16.0
adapter.dropout       # -> 0.0
adapter.num_layers    # -> 1 (after add_layer above)
adapter.is_training   # -> False (adapters start in evaluation mode)
```

> **Note:** `dropout` is real (inverted dropout applied to the LoRA input path: each element independently zeroed with probability `dropout`, survivors rescaled by `1 / (1 - dropout)`), but it is inert until `adapter.train()` is called. A freshly constructed `LoRAAdapter` -- and every adapter until `train()` is called -- is in *evaluation* mode, where `forward` ignores `dropout` entirely and is fully deterministic no matter what it is set to:
>
> ```python
> adapter.is_training      # -> False
> adapter.forward("layer_1", x)   # deterministic, even with dropout > 0
>
> adapter.train()          # switch to training mode (also applies to layers added afterwards)
> adapter.is_training      # -> True
> adapter.forward("layer_1", x)   # now stochastic if dropout > 0 -- repeated calls can differ
>
> adapter.eval()           # switch back (the default mode)
> ```
>
> Note also that `add_layer`'s freshly initialised `B` matrix is all zeros (so the initial effective weight equals `base`, per LoRA's standard init) — the LoRA path, and therefore any dropout applied to it, has no observable effect on `forward`'s output until `B` has actually been trained away from zero.
>
> `merge_all()`/`unmerge_all()` fold the *weights* (`B(A) * scaling`) into the base matrix; that is a static transform of the trained parameters, not a forward pass, so it is never subject to dropout regardless of training mode — and once merged, `forward` has no separate LoRA path left to apply dropout to. `adapter.train(); adapter.merge_all(); adapter.forward(...)` is therefore deterministic; that is correct behavior, not dropout silently failing again.

## MuLawCodec

`MuLawCodec` applies mu-law (ITU-T G.711) companding — a logarithmic quantization scheme that preserves dynamic range for quiet sounds better than linear quantization, the standard codec for telephony and WaveNet-style audio models. It mirrors the same codec's WASM binding (`WasmMuLawCodec` in `kizzasi-tokenizer`), so Python and JavaScript users get equivalent functionality.

```python
import numpy as np
import kizzasi

codec = kizzasi.MuLawCodec(bits=8)          # mu = 255; bits must be in 1..=16
# codec = kizzasi.MuLawCodec.with_mu(100.0, bits=8)  # explicit mu instead

signal = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

# Scalar
level = codec.quantize(0.0)      # -> int, 128 (midpoint of 256 levels for bits=8)
sample = codec.dequantize(level) # -> float, ~0.0

# Vectorized, discrete int32 levels in [0, vocab_size)
levels = codec.quantize_array(signal)      # -> np.ndarray[int32]
reconstructed = codec.dequantize_array(levels)  # -> np.ndarray[float32]

# Vectorized via the SignalTokenizer trait convention (same values as
# quantize_array, as float32 token ids instead of int32)
tokens = codec.encode(signal)
reconstructed = codec.decode(tokens)

codec.bits         # -> 8
codec.mu           # -> 255.0
codec.vocab_size   # -> 256
```

## License

Apache-2.0 © COOLJAPAN OU (Team Kitasan)
