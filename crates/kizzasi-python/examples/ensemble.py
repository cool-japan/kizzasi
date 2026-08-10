"""Multi-model ensemble example.

Builds three independent predictors from a shared config and combines their
outputs with one of the supported voting strategies (``"average"``,
``"weighted"``, ``"weighted_average"``, ``"median"``, ``"confidence"``,
``"majority"``).

Run with::

    python examples/ensemble.py
"""

from __future__ import annotations

import numpy as np

import kizzasi


def main() -> None:
    # Robotics preset: 6-DOF joint state -> 4 motor commands.
    config = kizzasi.Config.robotics(state_dim=6, action_dim=4)
    print(f"Config: {config}")

    # ---- average voting --------------------------------------------------
    ensemble = kizzasi.EnsemblePredictor(
        config,
        n_models=3,
        voting="average",
    )
    print(f"Ensemble (average): {ensemble}")

    state = np.array([0.1, 0.0, -0.2, 0.05, 0.0, 0.0], dtype=np.float32)
    action = ensemble.step(state)
    print(f"step(state) -> shape={action.shape}, value={action.tolist()}")

    stats = ensemble.stats()
    print(
        f"  stats: num_models={stats['num_models']}, "
        f"total_predictions={stats['total_predictions']}, "
        f"avg_variance={stats['avg_variance']:.6f}"
    )

    # ---- weighted voting -------------------------------------------------
    weighted = kizzasi.EnsemblePredictor(
        config,
        n_models=3,
        voting="weighted_average",
        weights=[0.6, 0.3, 0.1],
    )
    print(f"\nEnsemble (weighted_average, weights=[0.6, 0.3, 0.1]): {weighted}")
    action = weighted.step(state)
    print(f"step(state) -> {action.tolist()}")

    # Update a single model's weight on the fly.
    weighted.set_weight(0, 0.9)
    print(f"After set_weight(0, 0.9): weights={weighted.stats()['model_weights']}")

    # ---- median voting (no weights required) ----------------------------
    median = kizzasi.EnsemblePredictor(config, n_models=5, voting="median")
    print(f"\nEnsemble (median, n=5): {median}")
    median.reset()
    for i in range(3):
        out = median.step(state)
        print(f"  step {i}: {out.tolist()}")


if __name__ == "__main__":
    main()
