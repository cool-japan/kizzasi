"""LoRA (Low-Rank Adaptation) adapter demonstration.

Shows the smallest end-to-end workflow for ``kizzasi.LoRAAdapter``:

  1. Construct an adapter with a chosen rank / alpha.
  2. Register one or more LoRA layers from base weight matrices.
  3. Run a forward pass through a named layer.
  4. Merge / unmerge the LoRA contribution into the base weights.

Run with::

    python examples/lora.py
"""

from __future__ import annotations

import numpy as np

import kizzasi


def main() -> None:
    # 1. Build the adapter. rank=8 and alpha=16 are the LoRA defaults
    #    recommended in the original paper.
    adapter = kizzasi.LoRAAdapter(name="my_adapter", rank=8, alpha=16.0)
    print(f"adapter: {adapter}")
    print(f"  name={adapter.name}, rank={adapter.rank}, alpha={adapter.alpha}")
    print(f"  initial layers={adapter.num_layers}")

    # 2. Register a layer from its base weight (out_features=64, in_features=128).
    base_weight = np.random.randn(64, 128).astype(np.float32)
    adapter.add_layer("layer_1", base_weight)

    # A second layer with different dimensions.
    base_weight_2 = np.random.randn(32, 64).astype(np.float32)
    adapter.add_layer("layer_2", base_weight_2)

    print(
        f"after add: layers={adapter.num_layers}, "
        f"total LoRA params={adapter.total_parameters()}",
    )
    print(f"  avg parameter ratio vs full: {adapter.avg_parameter_ratio():.4f}")
    print(f"  module names: {sorted(adapter.module_names())}")

    # 3. Forward pass through layer_1: input must match in_features=128.
    inp = np.random.randn(128).astype(np.float32)
    out = adapter.forward("layer_1", inp)
    print(f"forward('layer_1', shape={inp.shape}) -> shape={out.shape}")

    # 4. Merge the LoRA correction into the base weights for fast inference,
    #    then unmerge to restore the original layout.
    adapter.merge_all()
    print("merge_all() done; further forward calls skip the LoRA correction")
    out_merged = adapter.forward("layer_1", inp)
    # After merge, the effective weight is captured in base_weight, so the
    # output should be (nearly) identical to the pre-merge forward call.
    print(f"  max(|out - out_merged|) = {float(np.abs(out - out_merged).max()):.3e}")

    adapter.unmerge_all()
    print("unmerge_all() done; ready for resumed LoRA training")


if __name__ == "__main__":
    main()
