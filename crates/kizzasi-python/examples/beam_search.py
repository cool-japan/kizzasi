"""Beam search and rejection sampling demonstration.

Shows three classes exposed by the ``kizzasi`` Python extension:

  - ``BeamSearch``          — plain beam search over a vocabulary
  - ``ConstrainedBeamSearch`` — beam search with Python callable hard/soft
                               constraints
  - ``RejectionSampler``    — rejection sampler with Python callable
                               constraints and configurable fallback strategies

Run with::

    python examples/beam_search.py
"""

from __future__ import annotations

import numpy as np

import kizzasi

VOCAB_SIZE = 10
RNG = np.random.default_rng(seed=42)


# ============================================================================
# 1. Plain beam search
# ============================================================================


def demo_beam_search() -> None:
    """Run a few expansion steps with a plain BeamSearch."""
    print("=== BeamSearch ===")

    bs = kizzasi.BeamSearch(beam_width=3)
    print(f"Initial: {bs}")  # uses __repr__
    print(f"  num_beams={bs.num_beams()}")  # 1 empty beam at start

    # Step 1: one beam -> logits shape (1, VOCAB_SIZE)
    logits_step1 = RNG.standard_normal((1, VOCAB_SIZE)).astype(np.float32)
    bs.expand(logits_step1)
    print(f"After step 1: num_beams={bs.num_beams()}")

    # Step 2: beam_width beams -> logits shape (beam_width, VOCAB_SIZE)
    n_beams = bs.num_beams()
    logits_step2 = RNG.standard_normal((n_beams, VOCAB_SIZE)).astype(np.float32)
    bs.expand(logits_step2)

    best_seq = bs.best_sequence()
    best_lp = bs.best_log_prob()
    print(f"After step 2: best_sequence={best_seq}, best_log_prob={best_lp:.4f}")

    all_b = bs.all_beams()
    print(f"  all beams ({len(all_b)} total):")
    for i, beam_dict in enumerate(all_b):
        print(
            f"    beam[{i}] log_prob={beam_dict['log_prob']:.4f}  "
            f"len(sequence)={len(beam_dict['sequence'])}"
        )
    print()


# ============================================================================
# 2. Constrained beam search
# ============================================================================


def demo_constrained_beam_search() -> None:
    """Beam search that keeps only even-indexed tokens."""
    print("=== ConstrainedBeamSearch (hard constraints) ===")

    cbs = kizzasi.ConstrainedBeamSearch(beam_width=4)
    print(f"Initial: {cbs}")

    # Hard constraint: every value in the sequence must be an even index.
    cbs.add_constraint(lambda seq: all(v % 2 == 0 for v in seq))
    print(f"  num_constraints={cbs.num_constraints()}")

    logits = RNG.standard_normal((1, VOCAB_SIZE)).astype(np.float32)
    cbs.expand(logits)

    n_beams = cbs.num_beams()
    logits2 = RNG.standard_normal((n_beams, VOCAB_SIZE)).astype(np.float32)
    cbs.expand(logits2)

    best = cbs.best_sequence()
    print(f"  best_sequence (even indices only): {best}")
    print()

    print("=== ConstrainedBeamSearch (soft constraints) ===")

    scbs = kizzasi.ConstrainedBeamSearch(beam_width=3)
    # Soft constraint: prefer sequences whose last token is small
    scbs.add_constraint(lambda seq: len(seq) == 0 or seq[-1] < 5.0)
    scbs.enable_soft_constraints(penalty=2.0)

    logits_s = RNG.standard_normal((1, VOCAB_SIZE)).astype(np.float32)
    scbs.expand(logits_s)

    n_s = scbs.num_beams()
    logits_s2 = RNG.standard_normal((n_s, VOCAB_SIZE)).astype(np.float32)
    scbs.expand(logits_s2)

    print(f"  soft-constrained best_sequence: {scbs.best_sequence()}")
    print()


# ============================================================================
# 3. Rejection sampler
# ============================================================================


def demo_rejection_sampler() -> None:
    """Sample tokens whose index is in the lower half of the vocabulary."""
    print("=== RejectionSampler ===")

    cfg = kizzasi.SamplingConfig()
    cfg.strategy("temperature")
    cfg.temperature(1.0)
    cfg.seed(7)

    rs = kizzasi.RejectionSampler(cfg)
    print(f"Initial: {rs}")

    # Only accept tokens from the lower half of the vocabulary.
    rs.add_constraint(lambda seq: seq[-1] < VOCAB_SIZE / 2)
    rs.set_max_attempts(200)
    rs.set_fallback_strategy("best_candidate")

    print(f"  num_constraints={rs.num_constraints()}")

    logits = RNG.standard_normal(VOCAB_SIZE).astype(np.float32)
    context: list[float] = []

    samples = []
    for _ in range(5):
        val = rs.sample(logits, context=context)
        context.append(val)
        samples.append(val)

    print(f"  sampled (should be < {VOCAB_SIZE // 2}): {[int(v) for v in samples]}")
    print()


# ============================================================================
# Entry point
# ============================================================================


def main() -> None:
    demo_beam_search()
    demo_constrained_beam_search()
    demo_rejection_sampler()


if __name__ == "__main__":
    main()
