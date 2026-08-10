"""Sampling strategy demonstration.

Shows the four core strategies exposed by ``kizzasi.SamplingConfig`` and
``kizzasi.Sampler``:

  - greedy (deterministic — always picks the max-logit index)
  - temperature (scaled softmax)
  - top_k (sample only among the k largest logits)
  - top_p (nucleus — sample only inside the smallest set with cumulative
           probability >= p)

Run with::

    python examples/sampling.py
"""

from __future__ import annotations

import numpy as np

import kizzasi

LOGITS = np.array([1.0, 3.0, 0.5, 2.5, 1.8], dtype=np.float32)


def show(label: str, sampler: kizzasi.Sampler, n: int = 5) -> None:
    """Run ``n`` independent samples through ``sampler`` and print them."""
    draws = [float(sampler.sample(LOGITS)) for _ in range(n)]
    print(f"{label:>12s}: {draws}")


def main() -> None:
    print(f"logits = {LOGITS.tolist()}")
    print(f"argmax = {int(np.argmax(LOGITS))}")

    # 1. Greedy (deterministic; should always pick index 1, the largest logit).
    cfg = kizzasi.SamplingConfig()
    cfg.strategy("greedy")
    show("greedy", kizzasi.Sampler(cfg))

    # 2. Temperature sampling with a fixed seed (so the example is reproducible).
    cfg = kizzasi.SamplingConfig()
    cfg.strategy("temperature")
    cfg.temperature(1.5)
    cfg.seed(42)
    show("temperature", kizzasi.Sampler(cfg))

    # 3. Top-k sampling (k=3 — only the 3 largest logits are candidates).
    cfg = kizzasi.SamplingConfig()
    cfg.top_k(3)
    cfg.seed(42)
    show("top_k(3)", kizzasi.Sampler(cfg))

    # 4. Top-p (nucleus) sampling (p=0.9).
    cfg = kizzasi.SamplingConfig()
    cfg.top_p(0.9)
    cfg.seed(42)
    show("top_p(0.9)", kizzasi.Sampler(cfg))

    # 5. Batched sampling: each row of `batch` is an independent logit vector.
    batch = np.stack([LOGITS, LOGITS[::-1].copy(), LOGITS * 2.0]).astype(np.float32)
    cfg = kizzasi.SamplingConfig()
    cfg.strategy("greedy")
    sampler = kizzasi.Sampler(cfg)
    out = sampler.sample_batch(batch)
    print(f"\nbatch greedy ({batch.shape}) -> {out.tolist()}")


if __name__ == "__main__":
    main()
