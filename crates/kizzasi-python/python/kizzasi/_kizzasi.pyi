"""Type stubs for kizzasi._kizzasi (Rust extension module).

Regenerated from the crate's actual `#[pymethods]` surface (all 13
registered classes) rather than hand-maintained prose. The previous version
of this file documented `model_type` values ("rwkv", "s4d", "transformer",
"s4") that `Config`'s parser rejects outright (valid values are "mamba",
"mamba2", "s4", "rwkv"; the stub's own documented default, "s4d", was one of
the rejected ones), wrong defaults for every dimension, and covered only 2
of the 13 classes -- `py.typed` marks this package PEP 561 type-checked, so
a checker trusted every one of those inaccuracies.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

__version__: str

class ModelType:
    """Selector for the underlying SSM architecture.

    Accepted wherever a model type is expected (`Config(model_type=...)`,
    `cfg.model_type = ...`) as an alternative to the equivalent lowercase
    `str` (e.g. `ModelType.MAMBA2` <-> `"mamba2"`).
    """

    MAMBA: ModelType
    MAMBA2: ModelType
    S4: ModelType
    RWKV: ModelType

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Config:
    """Configuration for a Kizzasi AGSP predictor.

    A `Config` is a plain data holder: constructing one, or mutating its
    fields, never validates dimensions or allocates anything. Validation
    (dimension > 0, and a total-parameter-count cap) happens when a
    `Predictor` / `EnsemblePredictor` / `OptimizedPredictor` is actually
    built from it -- the point where the engine would otherwise allocate.

    Parameters
    ----------
    input_dim : int
        Dimension of the input signal.
    output_dim : int
        Dimension of the output prediction.
    hidden_dim : int, optional
        Hidden state dimension (model capacity). Default 256.
    num_layers : int, optional
        Number of model layers. Default 4.
    state_dim : int, optional
        SSM state dimension. Default 16.
    context_window : int, optional
        Context window length in steps. Default 8192.
    model_type : str | ModelType, optional
        Architecture: `"mamba"`, `"mamba2"`, `"s4"`, `"rwkv"`
        (case-insensitive), or the equivalent `ModelType` classattr.
        Default `"mamba2"`.
    """

    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int
    state_dim: int
    context_window: int
    model_type: str

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        state_dim: int = 16,
        context_window: int = 8192,
        model_type: str | ModelType = "mamba2",
    ) -> None: ...

    @staticmethod
    def audio(sample_rate: int = 44100) -> Config:
        """Audio signal prediction preset (`input_dim=output_dim=1`).

        `context_window` scales with `sample_rate` so every rate covers the
        same real-time horizon as the 44.1 kHz reference default (8192
        samples ~= 185.8 ms), rounded up to the next power of two.

        Raises
        ------
        ValueError
            If `sample_rate` is 0.
        """
        ...

    @staticmethod
    def robotics(state_dim: int, action_dim: int) -> Config:
        """Robot control / control-loop preset."""
        ...

    @staticmethod
    def sensor(num_sensors: int) -> Config:
        """Multi-sensor fusion preset (`input_dim=output_dim=num_sensors`)."""
        ...

    @staticmethod
    def lightweight(input_dim: int, output_dim: int) -> Config:
        """Lightweight embedded / edge preset (`model_type="mamba"`)."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ConstraintSpec:
    """Scalar value-range constraint for a `Predictor`'s guardrails.

    Parameters
    ----------
    name : str
        Human-readable label.
    min_val : float | None, optional
        Lower bound (inclusive). Must be finite. Default `None`.
    max_val : float | None, optional
        Upper bound (inclusive). Must be finite and, if `min_val` is also
        given, `>= min_val`. Default `None`.
    dimension : int | None, optional
        Output-dimension index to constrain; `None` applies to every
        dimension. If given, must be `< output_dim` of the `Predictor` this
        spec is later passed to (checked at `set_guardrails` time, not
        here). Default `None`.
    hard_reject : bool, optional
        Raise instead of clamping when violated. Default `False`.

    Raises
    ------
    ValueError
        If neither `min_val` nor `max_val` is given.
    """

    name: str
    min_val: float | None
    max_val: float | None
    dimension: int | None
    hard_reject: bool

    def __init__(
        self,
        name: str,
        min_val: float | None = None,
        max_val: float | None = None,
        dimension: int | None = None,
        hard_reject: bool = False,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Predictor:
    """Autoregressive signal predictor.

    `Predictor` is pinned to the thread that created it (mirrors PyO3's
    `unsendable`): create one instance per thread rather than sharing a
    single instance across threads. `step`/`predict_n`/`step_list` release
    the GIL for the duration of their computation, so separate
    per-thread instances run their SSM math in true parallel.

    Parameters
    ----------
    config : Config
        Model configuration. Raises `ValueError` if any dimension is 0 or
        the total parameter count exceeds the supported cap.
    """

    def __init__(self, config: Config) -> None: ...

    def step(self, input: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict one step ahead.

        Parameters
        ----------
        input : ndarray of shape (input_dim,), dtype float32

        Returns
        -------
        ndarray of shape (output_dim,), dtype float32
        """
        ...

    def predict_n(
        self,
        input: NDArray[np.float32],
        n_steps: int,
    ) -> NDArray[np.float32]:
        """Predict `n_steps` ahead autoregressively.

        Returns
        -------
        ndarray of shape (n_steps, output_dim), dtype float32
        """
        ...

    def step_list(self, input: list[float]) -> list[float]:
        """Single step accepting/returning plain Python lists instead of ndarrays."""
        ...

    def reset(self) -> None:
        """Reset internal hidden state to zero."""
        ...

    def set_guardrails(self, specs: list[ConstraintSpec]) -> None:
        """Apply value-range guardrails to clip / reject out-of-bound predictions.

        Raises
        ------
        ValueError
            If any spec's `dimension` is out of range for `output_dim`, or
            any bound is non-finite or `min_val > max_val`.
        """
        ...

    def clear_guardrails(self) -> None:
        """Remove all currently active guardrails."""
        ...

    def has_guardrails(self) -> bool: ...
    @property
    def input_dim(self) -> int: ...
    @property
    def output_dim(self) -> int: ...
    @property
    def hidden_dim(self) -> int: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def state_dim(self) -> int: ...
    @property
    def context_window(self) -> int: ...
    @property
    def model_type(self) -> str: ...
    @property
    def config(self) -> Config:
        """A clone of the `Config` this predictor was constructed from."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class EnsemblePredictor:
    """Multi-model ensemble: `n_models` independent predictors from one
    shared `Config`, combined per-step by a voting strategy.

    Parameters
    ----------
    config : Config
    n_models : int
        Must be `>= 1`.
    voting : str, optional
        `"average"`, `"weighted"`, `"weighted_average"`, `"median"`,
        `"confidence"`, or `"majority"` (case-insensitive, with a few
        aliases). Default `"average"`.
    weights : list[float] | None, optional
        Per-model weight, length must equal `n_models`, each finite and
        `>= 0`. Defaults to all `1.0`.
    """

    def __init__(
        self,
        config: Config,
        n_models: int,
        voting: str = "average",
        weights: list[float] | None = None,
    ) -> None: ...

    def step(self, input: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def predict_n(
        self,
        input: NDArray[np.float32],
        n_steps: int,
    ) -> NDArray[np.float32]:
        """Requires `input_dim == output_dim`."""
        ...

    def reset(self) -> None:
        """Reset every ensemble member's internal state."""
        ...

    def set_weight(self, index: int, weight: float) -> None:
        """Update model `index`'s weight (0-based). `weight` must be finite and `>= 0`."""
        ...

    def stats(self) -> dict[str, object]:
        """`num_models`, `total_predictions`, `avg_variance`,
        `voting_strategy`, `model_weights` (list[float])."""
        ...

    @property
    def num_models(self) -> int: ...
    @property
    def voting_strategy(self) -> str: ...
    @property
    def input_dim(self) -> int: ...
    @property
    def output_dim(self) -> int: ...
    @property
    def config(self) -> Config: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class OptimizedPredictor:
    """Predictor wrapper adding an optional TTL-based LRU result cache.

    The cache serves `predict_stateless` only; `step` and `predict_n` always
    recompute.

    `enable_simd` and `workspace_pool_size` are accepted and reported back
    (`simd_enabled`, `optimization_stats()`'s `workspace_pool_hits`/
    `workspace_allocations`) for forward compatibility, but currently have
    **no effect on computation** -- the underlying engine does not yet
    implement SIMD dispatch or workspace-pool accounting.

    Parameters
    ----------
    config : Config
    cache_ttl_ms : int, optional
        Result cache TTL in milliseconds; `0` disables the cache. Default 1000.
    enable_simd : bool, optional
        Currently inert -- see above. Default `True`.
    workspace_pool_size : int, optional
        Currently inert -- see above. Default 16.
    result_cache_size : int, optional
        Max cached entries. Default 1000.
    """

    def __init__(
        self,
        config: Config,
        cache_ttl_ms: int = 1000,
        enable_simd: bool = True,
        workspace_pool_size: int = 16,
        result_cache_size: int = 1000,
    ) -> None: ...

    def step(self, input: NDArray[np.float32]) -> NDArray[np.float32]:
        """Advance the recurrence by one step.

        Never served from the result cache: a cached answer would be computed
        under a different hidden state and would skip the state update. Use
        `predict_stateless` for the memoisable evaluation.
        """
        ...

    def predict_stateless(
        self,
        input: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Evaluate the model as a pure function of `input`, from a reset
        state, restoring the live stream state afterwards.

        This is the call the result cache serves -- repeated identical inputs
        within the TTL are answered from the cache and appear as `hits` in
        `cache_stats()`.
        """
        ...

    def predict_n(
        self,
        input: NDArray[np.float32],
        n_steps: int,
    ) -> NDArray[np.float32]:
        """Bypasses the result cache. `n_steps` must be `>= 1`."""
        ...

    def reset(self) -> None:
        """Reset predictor state and clear the result cache."""
        ...

    def clear_cache(self) -> None:
        """Alias for `reset()` -- there is no cache-only reset."""
        ...

    def cache_stats(self) -> dict[str, object]:
        """`size`, `capacity`, `hits`, `misses`, `hit_rate`, `enabled`."""
        ...

    def optimization_stats(self) -> dict[str, object]:
        """`total_predictions`, `cached_predictions`, `cache_time_saved_us`,
        `avg_prediction_time_us`, and the always-`0` placeholders
        `workspace_pool_hits`, `workspace_allocations` (see class docstring)."""
        ...

    @property
    def input_dim(self) -> int: ...
    @property
    def output_dim(self) -> int: ...
    @property
    def simd_enabled(self) -> bool:
        """Reflects the `enable_simd` constructor argument only; has no
        effect on computation in the current engine."""
        ...

    @property
    def cache_enabled(self) -> bool: ...
    @property
    def cache_ttl_ms(self) -> int: ...
    @property
    def config(self) -> Config: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class LoRAAdapter:
    """Low-Rank Adaptation adapter: `y = W x + (alpha / rank) * B(A x)`.

    Parameters
    ----------
    name : str
        Opaque identifier (used in `__repr__` and error messages).
    rank : int
        Low-rank dimension; must be `> 0`.
    alpha : float
        Scaling factor; must be `> 0`.
    dropout : float, optional
        Per-layer dropout probability in `[0, 1)`. Default `0.0`.

    Raises
    ------
    RuntimeError
        If `rank`, `alpha`, or `dropout` is out of range (validation
        errors from the underlying `LoRAConfig::validate()` are mapped to
        `RuntimeError`, not `ValueError`).
    """

    def __init__(self, name: str, rank: int, alpha: float, dropout: float = 0.0) -> None: ...

    def add_layer(self, module_name: str, base_weight: NDArray[np.float32]) -> None:
        """`base_weight` has shape `(out_features, in_features)`."""
        ...

    def forward(self, module: str, input: NDArray[np.float32]) -> NDArray[np.float32]:
        """Returns an ndarray of shape `(out_features,)`.

        Raises
        ------
        ValueError
            If `module` was never registered via `add_layer`.
        """
        ...

    def merge_all(self) -> None:
        """Fold every LoRA correction into the base weights, in place."""
        ...

    def unmerge_all(self) -> None:
        """Reverse `merge_all()`."""
        ...

    def total_parameters(self) -> int: ...
    def avg_parameter_ratio(self) -> float: ...
    def module_names(self) -> list[str]:
        """Insertion order is *not* preserved (backed by a `HashMap`)."""
        ...

    @property
    def name(self) -> str: ...
    @property
    def rank(self) -> int: ...
    @property
    def alpha(self) -> float: ...
    @property
    def dropout(self) -> float: ...
    @property
    def num_layers(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...

class SamplingConfig:
    """Builder-style sampling configuration.

    Every mutator (`strategy`, `temperature`, `top_k`, `top_p`, `seed`)
    returns `None` and modifies `self` in place; validation happens at the
    setter, not at construction, so a partial configuration can be built
    incrementally before being passed to `Sampler(...)`.

    Note the `get_*` properties are named that way deliberately (not the
    bare `temperature`/`top_k`/`top_p`/`seed`, which are the setter
    methods above) -- Python attribute names cannot be shared between a
    method and a property.
    """

    def __init__(self) -> None: ...

    def strategy(self, name: str) -> None:
        """`"greedy"`, `"temperature"` (alias `"temp"`), `"top_k"`
        (aliases `"topk"`, `"top-k"`), or `"top_p"` (aliases `"topp"`,
        `"top-p"`, `"nucleus"`). Case-insensitive."""
        ...

    def temperature(self, t: float) -> None:
        """Must be finite and `> 0`."""
        ...

    def top_k(self, k: int) -> None:
        """Must be `>= 1`. Also flips `strategy` to `"top_k"`."""
        ...

    def top_p(self, p: float) -> None:
        """Must be in `(0, 1]`. Also flips `strategy` to `"top_p"`."""
        ...

    def seed(self, s: int) -> None: ...
    @property
    def strategy_name(self) -> str: ...
    @property
    def get_temperature(self) -> float: ...
    @property
    def get_top_k(self) -> int | None: ...
    @property
    def get_top_p(self) -> float | None: ...
    @property
    def get_seed(self) -> int | None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Sampler:
    """Draws values from logit vectors per a `SamplingConfig`.

    Stateful: the underlying RNG is reused across calls, so a fixed seed
    produces a deterministic *sequence*, not a repeated single value.

    Parameters
    ----------
    config : SamplingConfig
        Cloned at construction; later mutation of `config` is not reflected.

    Raises
    ------
    ValueError
        If `config.strategy_name` is `"top_k"`/`"top_p"` but the matching
        `top_k(k)`/`top_p(p)` was never called.
    """

    def __init__(self, config: SamplingConfig) -> None: ...

    def sample(self, logits: NDArray[np.float32]) -> float:
        """Sample a single value from a 1-D logits vector."""
        ...

    def sample_batch(self, logits: NDArray[np.float32]) -> NDArray[np.float32]:
        """`logits` has shape `(batch, vocab)`; returns shape `(batch,)`."""
        ...

    @property
    def strategy_name(self) -> str: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class BeamSearch:
    """Plain (unconstrained) beam search.

    On construction there is exactly one empty beam; after the first
    `expand()` call there are `beam_width` beams (or fewer if the
    vocabulary is smaller than `beam_width`).

    Parameters
    ----------
    beam_width : int
        Must be `>= 1`.
    """

    def __init__(self, beam_width: int) -> None: ...

    def expand(self, logits: NDArray[np.float32]) -> None:
        """`logits` has shape `(current_beams, vocab_size)`; the row count
        must equal `num_beams()`."""
        ...

    def best_sequence(self) -> NDArray[np.float32] | None: ...
    def best_log_prob(self) -> float | None: ...
    def num_beams(self) -> int: ...
    def all_beams(self) -> list[dict[str, object]]:
        """Each dict: `{"sequence": ndarray, "log_prob": float}`."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ConstrainedBeamSearch:
    """`BeamSearch` extended with Python-callable constraints.

    Constraints are hard (violating beams filtered out) by default, or
    soft (log-probability penalty) via `enable_soft_constraints`.

    Parameters
    ----------
    beam_width : int
        Must be `>= 1`.
    """

    def __init__(self, beam_width: int) -> None: ...

    def add_constraint(self, constraint_fn: Callable[[list[float]], bool]) -> None:
        """`constraint_fn` receives the current candidate sequence and must
        return `bool`; `True` means satisfied.

        If `constraint_fn` raises during a later `expand()` call, that
        exception (wrapped in `RuntimeError` with the original chained as
        `__cause__`, or re-raised unchanged for `KeyboardInterrupt`/
        `SystemExit`) is raised from `expand()` -- it is not silently
        treated as "violated".
        """
        ...

    def enable_soft_constraints(self, penalty: float) -> None:
        """`penalty` (non-negative) is subtracted from a violating beam's
        log-probability instead of discarding it."""
        ...

    def expand(self, logits: NDArray[np.float32]) -> None: ...
    def best_sequence(self) -> NDArray[np.float32] | None: ...
    def num_beams(self) -> int: ...
    def num_constraints(self) -> int: ...
    def all_beams(self) -> list[dict[str, object]]: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class RejectionSampler:
    """Rejection sampler: draws up to `max_attempts` candidates from a
    base sampler and returns the first one satisfying all constraints,
    falling back per `set_fallback_strategy` if none do.

    Parameters
    ----------
    config : SamplingConfig
        Cloned at construction; later mutation of `config` is not reflected.
    """

    def __init__(self, config: SamplingConfig) -> None: ...

    def add_constraint(self, constraint_fn: Callable[[list[float]], bool]) -> None:
        """Receives `context + [candidate]`. See `ConstrainedBeamSearch.add_constraint`
        for exception-propagation behaviour -- identical here."""
        ...

    def set_max_attempts(self, n: int) -> None: ...
    def set_fallback_strategy(self, name: str) -> None:
        """`"best_candidate"` (alias `"best"`), `"greedy"`, or `"error"`.
        Case-insensitive; hyphens/underscores interchangeable."""
        ...

    def sample(self, logits: NDArray[np.float32], context: list[float]) -> float: ...
    def num_constraints(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class MuLawCodec:
    """mu-law (ITU-T G.711) companding codec for audio signal quantization.

    Parameters
    ----------
    bits : int, optional
        Quantization bit depth, `1..=16`. Default `8` (mu = 255).

    Raises
    ------
    RuntimeError
        If `bits` is out of `1..=16`.
    """

    def __init__(self, bits: int = 8) -> None: ...

    @staticmethod
    def with_mu(mu: float, bits: int = 8) -> MuLawCodec:
        """Explicit `mu` instead of the standard `(2**bits) - 1`.

        Raises
        ------
        RuntimeError
            If `mu` is non-finite/non-positive, or `bits` is out of `1..=16`.
        """
        ...

    def quantize(self, x: float) -> int:
        """`x` in `[-1.0, 1.0]` (clamped otherwise) -> level in `[0, vocab_size)`."""
        ...

    def dequantize(self, level: int) -> float: ...
    def quantize_array(self, signal: NDArray[np.float32]) -> NDArray[np.int32]:
        """Vectorized `quantize`."""
        ...

    def dequantize_array(self, levels: NDArray[np.int32]) -> NDArray[np.float32]:
        """Vectorized `dequantize`."""
        ...

    def encode(self, signal: NDArray[np.float32]) -> NDArray[np.float32]:
        """Discrete levels as `float32` token ids (same values as
        `quantize_array`, different dtype) -- the `SignalTokenizer` trait
        convention shared by every tokenizer in `kizzasi-tokenizer`."""
        ...

    def decode(self, tokens: NDArray[np.float32]) -> NDArray[np.float32]:
        """Inverse of `encode`."""
        ...

    @property
    def bits(self) -> int: ...
    @property
    def mu(self) -> float: ...
    @property
    def vocab_size(self) -> int:
        """`2 ** bits`."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
