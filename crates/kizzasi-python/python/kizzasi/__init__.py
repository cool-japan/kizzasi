"""
Kizzasi — Autoregressive General-Purpose Signal Predictor

Python bindings for the Kizzasi AGSP framework, providing high-performance
signal prediction using State Space Models (Mamba, RWKV, S4D, etc.) with
neuro-symbolic constraint enforcement.

Example
-------
>>> import kizzasi
>>> cfg = kizzasi.Config(input_dim=8, output_dim=8, hidden_dim=64, num_layers=2)
>>> predictor = kizzasi.Predictor(cfg)
>>> import numpy as np
>>> x = np.zeros(8, dtype=np.float32)
>>> y = predictor.step(x)

"""

from ._kizzasi import (
    Config,
    Predictor,
    __version__,
)

__all__ = [
    "Config",
    "Predictor",
    "__version__",
]
