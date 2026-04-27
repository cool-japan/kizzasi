# kizzasi

**Kizzasi** (兆し) — Autoregressive General-Purpose Signal Predictor (AGSP)

Python bindings for the [Kizzasi](https://github.com/cool-japan/kizzasi) Rust framework, providing high-performance signal prediction using State Space Models (Mamba, RWKV, S4D, Spiking NNs, Neural ODE) with neuro-symbolic constraint enforcement.

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
    model_type="s4d",  # or "rwkv", "transformer"
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
# Audio prediction (16 kHz, 80-dim features)
cfg = kizzasi.Config.audio(sample_rate=16000)

# Robot control (6-DOF joint angles → actions)
cfg = kizzasi.Config.robotics(state_dim=6, action_dim=6)

# IoT sensor streams
cfg = kizzasi.Config.sensor(num_sensors=9)
```

## License

Apache-2.0 © COOLJAPAN OU (Team Kitasan)
