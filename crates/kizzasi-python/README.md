# kizzasi

**Kizzasi** (兆し) — Autoregressive General-Purpose Signal Predictor (AGSP)

Python bindings for the [Kizzasi](https://github.com/cool-japan/kizzasi) Rust framework, providing high-performance signal prediction using State Space Models (Mamba, RWKV, S4, Spiking NNs, Neural ODE) with neuro-symbolic constraint enforcement.

## Installation

```bash
pip install kizzasi
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
# Audio prediction (44.1 kHz by default)
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
    context_window=8192,
    model_type="mamba2",   # "mamba", "mamba2", "s4", "rwkv"
)
```

### ModelType enum

```python
kizzasi.ModelType.MAMBA
kizzasi.ModelType.MAMBA2
kizzasi.ModelType.S4
kizzasi.ModelType.RWKV
```

## Predictor

> **Note:** `Predictor` is not thread-safe. Use separate `Predictor` instances per thread.

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
```

> **Note:** `Sampler` is stateful — it reuses one RNG across calls, so a fixed seed produces a deterministic *sequence* of samples, not a repeated single value. `top_p(p)` takes `p` in `(0, 1]` and also flips the strategy to `"top_p"`; strategy names are case-insensitive and accept aliases (`"temp"`, `"topk"`/`"top-k"`, `"topp"`/`"top-p"`/`"nucleus"`).

## Beam Search & Rejection Sampling

Three decoding utilities operate on logits matrices supplied one step at a time. `BeamSearch` is unconstrained; `ConstrainedBeamSearch` and `RejectionSampler` accept arbitrary Python callables as hard or soft constraints on the candidate sequence.

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

`OptimizedPredictor` wraps a predictor with workspace pooling, optional SIMD kernels, and an optional TTL-based LRU result cache.

```python
import numpy as np
import kizzasi

cfg = kizzasi.Config(input_dim=4, output_dim=4, hidden_dim=32, num_layers=2)

opt = kizzasi.OptimizedPredictor(
    cfg,
    cache_ttl_ms=1000,          # 0 disables the result cache; default 1000
    enable_simd=True,           # default True
    workspace_pool_size=16,     # default 16
    result_cache_size=1000,     # default 1000
)

x = np.random.randn(4).astype(np.float32)
y = opt.step(x)                    # -> np.ndarray shape (output_dim,)
ys = opt.predict_n(x, n_steps=20)  # -> np.ndarray shape (20, output_dim); bypasses the result cache

opt.cache_stats()          # {"size", "capacity", "hits", "misses", "hit_rate", "enabled"}
opt.optimization_stats()   # {"total_predictions", "cached_predictions", "cache_time_saved_us",
                            #  "avg_prediction_time_us", "workspace_pool_hits", "workspace_allocations"}

opt.clear_cache()          # alias for reset(): clears the cache *and* predictor state together
```

> **Note:** `OptimizedPredictor` does not expose a cache-only reset — `reset()` and `clear_cache()` both clear the result cache and the underlying predictor's hidden state.

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

adapter.name        # -> "my_adapter"
adapter.rank        # -> 8
adapter.alpha       # -> 16.0
adapter.dropout     # -> 0.0
adapter.num_layers  # -> 1 (after add_layer above)
```

## License

Apache-2.0 © COOLJAPAN OU (Team Kitasan)
