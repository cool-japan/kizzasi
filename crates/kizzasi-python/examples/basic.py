"""Basic Kizzasi inference example.

Demonstrates the smallest possible workflow:

  1. Build a Config (via the ``audio`` preset).
  2. Construct a Predictor.
  3. Feed a single float32 ndarray through ``step``.
  4. Run a short autoregressive rollout with ``predict_n``.
  5. Reset internal hidden state.

Run with::

    python examples/basic.py
"""

from __future__ import annotations

import numpy as np

import kizzasi


def main() -> None:
    # 1. Audio preset: input_dim=1, output_dim=1, Mamba2 backbone.
    config = kizzasi.Config.audio(sample_rate=44100)
    print(f"Using config: {config}")

    # 2. Build the predictor.
    predictor = kizzasi.Predictor(config)
    print(f"Predictor: {predictor}")
    print(
        f"  input_dim={predictor.input_dim}, "
        f"output_dim={predictor.output_dim}, "
        f"model_type='{predictor.model_type}'"
    )

    # 3. Single-step prediction.
    sample = np.array([0.5], dtype=np.float32)
    output = predictor.step(sample)
    print(f"step({sample.tolist()}) -> {output.tolist()}")

    # 4. Multi-step autoregressive rollout (10 future samples).
    rollout = predictor.predict_n(sample, n_steps=10)
    print(f"predict_n(..., n_steps=10) -> shape={rollout.shape}")
    print(f"  first sample: {rollout[0].tolist()}")
    print(f"  last sample:  {rollout[-1].tolist()}")

    # 5. Reset hidden state for the next session.
    predictor.reset()
    print("predictor.reset() done.")


if __name__ == "__main__":
    main()
