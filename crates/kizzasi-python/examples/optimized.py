"""Optimized predictor example.

Wraps a Kizzasi predictor with workspace pooling, optional SIMD acceleration,
and an LRU result cache. Caching is keyed on the input vector and bounded by
``cache_ttl_ms`` and ``result_cache_size``.

Run with::

    python examples/optimized.py
"""

from __future__ import annotations

import numpy as np

import kizzasi


def main() -> None:
    # Sensor preset: 8 sensor channels in/out.
    config = kizzasi.Config.sensor(num_sensors=8)
    print(f"Config: {config}")

    # Cache TTL of 500ms with SIMD enabled.
    opt = kizzasi.OptimizedPredictor(
        config,
        cache_ttl_ms=500,
        enable_simd=True,
        workspace_pool_size=16,
        result_cache_size=1000,
    )
    print(f"OptimizedPredictor: {opt}")
    print(
        f"  simd_enabled={opt.simd_enabled}, "
        f"cache_enabled={opt.cache_enabled}, "
        f"cache_ttl_ms={opt.cache_ttl_ms}"
    )

    # Feed two distinct inputs, then repeat the first to demonstrate caching.
    sample_a = np.linspace(0.0, 1.0, num=8, dtype=np.float32)
    sample_b = np.linspace(1.0, 0.0, num=8, dtype=np.float32)

    out_a = opt.step(sample_a)
    out_b = opt.step(sample_b)
    out_a_again = opt.step(sample_a)  # should hit the cache

    print(f"step(sample_a) shape={out_a.shape}")
    print(f"step(sample_b) shape={out_b.shape}")
    print(f"step(sample_a) again -> identical? {np.allclose(out_a, out_a_again)}")

    print("\ncache stats:", opt.cache_stats())
    print("opt stats:   ", opt.optimization_stats())

    # Manually flush the cache (also resets predictor state).
    opt.clear_cache()
    print("\nAfter clear_cache():", opt.cache_stats())


if __name__ == "__main__":
    main()
