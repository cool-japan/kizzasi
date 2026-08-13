"""Python-level smoke test suite for the `kizzasi` extension module.

Before this file existed, `crates/kizzasi-python` had zero Python-level
tests: `pyproject.toml` pointed `[tool.pytest.ini_options] testpaths` at a
`tests/` directory that did not exist, and the crate's 72 (now much larger)
Rust `#[test]` suite exercised the wrapped Rust structs directly (via
`.inner` or free functions), never the actual `#[pymethods]` ABI — numpy
array conversion, exception mapping to Python exception types, module
import, or attribute exposure. That gap is exactly how two critical bugs
(the `__init__.py` export list covering 2 of 12 registered classes, and a
`#[pymodule]` name that didn't match the installed `_kizzasi` extension
filename) could have reached PyPI undetected: both are static-analysis-only
findings that a wheel-install-and-import step would catch immediately.

This suite constructs every registered class through the real Python API
(not the Rust internals) and exercises the fixes made across the crate's
findings. Run with `pytest` from `crates/kizzasi-python/` against an
installed/built `kizzasi` wheel (e.g. `maturin develop` then `pytest`, or
`pip install .` in a venv then `pytest`).
"""

from __future__ import annotations

import numpy as np
import pytest

kizzasi = pytest.importorskip("kizzasi")


# ---------------------------------------------------------------------------
# Module-level: every registered pyclass must actually be importable.
# ---------------------------------------------------------------------------


def test_all_classes_are_exported():
    """Regression for the critical `__init__.py` export-gap bug: only
    `Config`/`Predictor`/`__version__` were re-exported from `._kizzasi`
    even though 12 classes (13 after `MuLawCodec`) are registered in the
    extension module. Every name in `__all__` must resolve to a real
    attribute, and nothing in `__all__` may be a leftover stale name.

    Checks *both* directions on purpose: a subset check alone (every name
    in `__all__` resolves) cannot detect the original bug, because the
    original broken `__all__ = ["Config", "Predictor", "__version__"]` was
    itself a subset that resolved perfectly fine -- the bug was that
    `__all__` was missing names, not that it contained bad ones. Comparing
    against `dir(kizzasi._kizzasi)` (the actual compiled-extension surface)
    is what catches a newly `#[pymodule] m.add_class::<...>()`-registered
    class that a future change forgets to add to `__init__.py`.
    """
    from kizzasi import _kizzasi as _ext

    assert kizzasi.__all__, "kizzasi.__all__ must not be empty"
    for name in kizzasi.__all__:
        assert hasattr(kizzasi, name), f"kizzasi.{name} is in __all__ but missing"

    registered = {n for n in dir(_ext) if not n.startswith("_")}
    exported = set(kizzasi.__all__)
    missing = registered - exported
    assert not missing, f"registered in _kizzasi but not re-exported: {sorted(missing)}"


def test_version_is_nonempty_string():
    assert isinstance(kizzasi.__version__, str)
    assert kizzasi.__version__


# ---------------------------------------------------------------------------
# Config / ModelType
# ---------------------------------------------------------------------------


def test_config_basic_construction():
    cfg = kizzasi.Config(input_dim=4, output_dim=4, hidden_dim=32, num_layers=2)
    assert cfg.input_dim == 4
    assert cfg.output_dim == 4
    assert cfg.model_type == "mamba2"


def test_config_presets():
    assert kizzasi.Config.robotics(state_dim=6, action_dim=6).input_dim == 6
    assert kizzasi.Config.sensor(num_sensors=9).input_dim == 9
    assert kizzasi.Config.lightweight(input_dim=4, output_dim=4).model_type == "mamba"


def test_config_audio_sample_rate_scales_context_window():
    """Regression: `Config.audio(sample_rate)` used to discard its argument
    (`let _ = sample_rate;`) and always return the same fixed config."""
    reference = kizzasi.Config.audio(44100)
    lower = kizzasi.Config.audio(16000)
    higher = kizzasi.Config.audio(96000)
    assert reference.context_window == 8192  # unchanged default behaviour
    assert lower.context_window != reference.context_window
    assert lower.context_window < reference.context_window < higher.context_window


def test_config_audio_rejects_zero_sample_rate():
    with pytest.raises(ValueError):
        kizzasi.Config.audio(0)


def test_model_type_enum_accepted_by_config():
    """Regression: `ModelType` was registered and documented but no API
    accepted it -- `Config(model_type=ModelType.X)` raised `TypeError`."""
    cfg = kizzasi.Config(4, 4, model_type=kizzasi.ModelType.RWKV)
    assert cfg.model_type == "rwkv"
    cfg.model_type = kizzasi.ModelType.S4
    assert cfg.model_type == "s4"
    # str still works too.
    cfg.model_type = "mamba2"
    assert cfg.model_type == "mamba2"


def test_model_type_repr():
    assert repr(kizzasi.ModelType.MAMBA2) == "ModelType.MAMBA2"


def test_predictor_rejects_huge_hidden_dim():
    """Regression: an unvalidated `hidden_dim` (e.g. 10**9) reached
    `SelectiveSSM::new`'s allocations directly and could abort the whole
    process on allocation failure instead of raising a catchable error.
    `Config(...)` itself never validates (plain data holder); `Predictor()`
    is the actual allocation choke point and must reject it."""
    huge_cfg = kizzasi.Config(1, 1, hidden_dim=10**9)
    with pytest.raises(ValueError):
        kizzasi.Predictor(huge_cfg)


def test_predictor_rejects_zero_dimensions():
    zero_cfg = kizzasi.Config(0, 1, hidden_dim=8, num_layers=1, state_dim=4, context_window=16)
    with pytest.raises(ValueError):
        kizzasi.Predictor(zero_cfg)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config() -> "kizzasi.Config":
    return kizzasi.Config(input_dim=4, output_dim=4, hidden_dim=32, num_layers=2)


def test_predictor_step_and_predict_n(small_config):
    predictor = kizzasi.Predictor(small_config)
    x = np.zeros(4, dtype=np.float32)
    y = predictor.step(x)
    assert y.shape == (4,)
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))

    ys = predictor.predict_n(x, n_steps=5)
    assert ys.shape == (5, 4)
    assert np.all(np.isfinite(ys))

    out_list = predictor.step_list([0.1, 0.2, 0.3, 0.4])
    assert len(out_list) == 4

    predictor.reset()


def test_predictor_step_dimension_mismatch(small_config):
    predictor = kizzasi.Predictor(small_config)
    with pytest.raises(ValueError):
        predictor.step(np.zeros(3, dtype=np.float32))


def test_predictor_guardrails_two_sided_bound_enforced(small_config):
    """Regression for the critical guardrail bug: a two-sided
    `ConstraintSpec(min_val=..., max_val=...)` only enforced `max_val`
    because `build_guardrails` overwrote `min_val` with `max_val` via a
    single-bound `ConstraintBuilder`. README's own robotics example
    (`ConstraintSpec("joint_angle", min_val=-3.14, max_val=3.14)`) is this
    exact shape.
    """
    cfg = kizzasi.Config(1, 1, hidden_dim=16, num_layers=1, state_dim=4, context_window=64)
    predictor = kizzasi.Predictor(cfg)
    predictor.set_guardrails(
        [kizzasi.ConstraintSpec("bounds", min_val=-1.0, max_val=1.0)]
    )
    assert predictor.has_guardrails()
    for _ in range(30):
        out = predictor.step(np.array([5.0], dtype=np.float32))
        assert -1.0 - 1e-4 <= out[0] <= 1.0 + 1e-4, out[0]
    predictor.clear_guardrails()
    assert not predictor.has_guardrails()


def test_predictor_guardrails_reject_out_of_range_dimension(small_config):
    """Regression: a dimensional `ConstraintSpec` whose `dimension` was out
    of range for `output_dim` was silently accepted and silently enforced
    nothing (`GuardrailSet::constrain` skips out-of-range dims)."""
    predictor = kizzasi.Predictor(small_config)
    with pytest.raises(ValueError):
        predictor.set_guardrails(
            [kizzasi.ConstraintSpec("bad", min_val=-1.0, max_val=1.0, dimension=99)]
        )
    assert not predictor.has_guardrails()


def test_constraint_spec_requires_a_bound():
    with pytest.raises(ValueError):
        kizzasi.ConstraintSpec("bad")


# ---------------------------------------------------------------------------
# EnsemblePredictor
# ---------------------------------------------------------------------------


def test_ensemble_predictor_basic(small_config):
    ensemble = kizzasi.EnsemblePredictor(small_config, n_models=3, voting="average")
    x = np.zeros(4, dtype=np.float32)
    out = ensemble.step(x)
    assert out.shape == (4,)
    outs = ensemble.predict_n(x, n_steps=3)
    assert outs.shape == (3, 4)
    stats = ensemble.stats()
    assert stats["num_models"] == 3


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), -1.0])
def test_ensemble_predictor_rejects_bad_weights(small_config, bad_weight):
    """Regression: NaN/infinite/negative weights passed the core's
    `weight < 0.0` check (false for NaN) and silently poisoned every
    weighted prediction with NaN, with no error raised anywhere."""
    with pytest.raises(ValueError):
        kizzasi.EnsemblePredictor(
            small_config, n_models=2, voting="weighted", weights=[1.0, bad_weight]
        )


def test_ensemble_set_weight_rejects_nan(small_config):
    ensemble = kizzasi.EnsemblePredictor(
        small_config, n_models=2, voting="weighted_average", weights=[1.0, 1.0]
    )
    with pytest.raises(ValueError):
        ensemble.set_weight(0, float("nan"))


# ---------------------------------------------------------------------------
# OptimizedPredictor
# ---------------------------------------------------------------------------


def test_optimized_predictor_basic(small_config):
    opt = kizzasi.OptimizedPredictor(small_config, cache_ttl_ms=500)
    x = np.zeros(4, dtype=np.float32)
    out = opt.step(x)
    assert out.shape == (4,)
    stats = opt.cache_stats()
    assert "hit_rate" in stats


def test_optimized_predictor_workspace_stats_are_documented_placeholders(small_config):
    """Regression (honest-docs): `workspace_pool_hits`/`workspace_allocations`
    are always 0 in the current engine -- verify the *behavior* stays
    consistent with the documentation rather than silently starting to lie
    in one direction (reporting nonzero) without anyone noticing."""
    opt = kizzasi.OptimizedPredictor(small_config, cache_ttl_ms=500)
    x = np.zeros(4, dtype=np.float32)
    for _ in range(5):
        opt.step(x)
    stats = opt.optimization_stats()
    assert stats["workspace_pool_hits"] == 0
    assert stats["workspace_allocations"] == 0


def test_optimized_predictor_predict_n_rejects_zero_steps(small_config):
    opt = kizzasi.OptimizedPredictor(small_config)
    with pytest.raises(ValueError):
        opt.predict_n(np.zeros(4, dtype=np.float32), n_steps=0)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sampler_rejects_top_k_strategy_without_value():
    """Regression: `strategy("top_k")` without a following `top_k(k)` call
    silently sampled with the engine's internal default (k=10), with no
    error and no documented default anywhere."""
    config = kizzasi.SamplingConfig()
    config.strategy("top_k")
    with pytest.raises(ValueError):
        kizzasi.Sampler(config)


def test_sampler_rejects_top_p_strategy_without_value():
    config = kizzasi.SamplingConfig()
    config.strategy("top_p")
    with pytest.raises(ValueError):
        kizzasi.Sampler(config)


def test_sampler_greedy_and_batch():
    config = kizzasi.SamplingConfig()
    config.strategy("greedy")
    sampler = kizzasi.Sampler(config)
    logits = np.array([1.0, 3.0, 0.5, 2.5, 1.8], dtype=np.float32)
    value = sampler.sample(logits)
    assert value == pytest.approx(1.0)  # index of max value

    batch_logits = np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32)
    out = sampler.sample_batch(batch_logits)
    assert out.shape == (4,)


def test_sampler_top_k_with_value_accepted():
    config = kizzasi.SamplingConfig()
    config.top_k(3)  # also flips strategy to "top_k"
    sampler = kizzasi.Sampler(config)
    assert sampler.strategy_name == "top_k"


def test_sampling_config_getters_are_readable_and_dont_collide_with_setters():
    """Regression (found while writing this suite, not one of the audited
    findings): `#[getter] fn get_temperature` etc. would, without an
    explicit `#[getter(name)]`, be exposed to Python as the bare property
    `temperature` -- colliding with, and silently shadowed by, the
    already-present builder-style *method* also named `temperature`. The
    getters must be readable under their own `get_*` name, and the
    builder-style methods must remain callable.
    """
    config = kizzasi.SamplingConfig()
    assert callable(config.temperature)
    assert callable(config.top_k)
    assert callable(config.seed)

    config.temperature(0.7)
    config.top_k(5)
    config.seed(42)

    assert config.get_temperature == pytest.approx(0.7, abs=1e-5)
    assert config.get_top_k == 5
    assert config.get_seed == 42
    assert config.get_top_p is None


# ---------------------------------------------------------------------------
# Beam search / constrained beam search / rejection sampling
# ---------------------------------------------------------------------------


def test_beam_search_basic():
    bs = kizzasi.BeamSearch(beam_width=3)
    assert bs.num_beams() == 1
    bs.expand(np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32))
    assert bs.num_beams() >= 1
    assert bs.best_sequence() is not None


def test_beam_search_rejects_zero_width():
    with pytest.raises(ValueError):
        kizzasi.BeamSearch(beam_width=0)


def test_constrained_beam_search_constraint_exception_propagates():
    """Regression: a constraint callback that raised an exception (typo,
    `IndexError` on the initial empty beam, ...) was silently swallowed into
    "constraint violated" (`.unwrap_or(false)`), including
    `KeyboardInterrupt`/`SystemExit`. It must now surface as a real Python
    exception with the original error chained as `__cause__`.
    """
    cbs = kizzasi.ConstrainedBeamSearch(beam_width=2)

    def raises(seq):
        raise RuntimeError("constraint bug")

    cbs.add_constraint(raises)
    with pytest.raises(RuntimeError) as exc_info:
        cbs.expand(np.random.default_rng(0).standard_normal((1, 6)).astype(np.float32))
    assert exc_info.value.__cause__ is not None
    assert "constraint bug" in str(exc_info.value.__cause__)


def test_constrained_beam_search_working_constraint():
    cbs = kizzasi.ConstrainedBeamSearch(beam_width=2)
    cbs.add_constraint(lambda seq: True)
    cbs.expand(np.random.default_rng(0).standard_normal((1, 6)).astype(np.float32))
    assert cbs.num_constraints() == 1


def test_rejection_sampler_basic():
    config = kizzasi.SamplingConfig()
    config.strategy("temperature")
    config.temperature(1.0)
    config.seed(42)
    rs = kizzasi.RejectionSampler(config)
    rs.add_constraint(lambda seq: True)
    value = rs.sample(np.array([2.0, 2.5, 1.8, 0.1, 0.05], dtype=np.float32), context=[])
    assert isinstance(value, float)


# ---------------------------------------------------------------------------
# LoRAAdapter
# ---------------------------------------------------------------------------


def test_lora_adapter_forward():
    adapter = kizzasi.LoRAAdapter("test_adapter", rank=4, alpha=8.0)
    base = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)
    adapter.add_layer("layer1", base)
    assert adapter.num_layers == 1
    out = adapter.forward("layer1", np.zeros(16, dtype=np.float32))
    assert out.shape == (8,)
    adapter.merge_all()
    adapter.unmerge_all()


def test_lora_adapter_rejects_invalid_rank():
    # LoRAConfig::validate()'s error is mapped through the crate-wide
    # `to_py_err` helper, which wraps every domain error as RuntimeError
    # (not ValueError) -- consistent with every other `.validate()`-style
    # check in this crate that predates this test file.
    with pytest.raises(RuntimeError):
        kizzasi.LoRAAdapter("bad", rank=0, alpha=8.0)


# ---------------------------------------------------------------------------
# MuLawCodec
# ---------------------------------------------------------------------------


def test_mulaw_codec_scalar_roundtrip():
    codec = kizzasi.MuLawCodec(bits=8)
    assert codec.vocab_size == 256
    assert codec.quantize(0.0) == 128
    assert codec.quantize(-1.0) == 0
    assert codec.quantize(1.0) == 255
    for x in (-1.0, -0.5, 0.0, 0.5, 1.0):
        level = codec.quantize(x)
        back = codec.dequantize(level)
        assert abs(back - x) < 0.05


def test_mulaw_codec_array_roundtrip():
    codec = kizzasi.MuLawCodec(bits=8)
    signal = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

    levels = codec.quantize_array(signal)
    assert levels.dtype == np.int32
    back = codec.dequantize_array(levels)
    assert np.allclose(back, signal, atol=0.05)

    tokens = codec.encode(signal)
    reconstructed = codec.decode(tokens)
    assert np.allclose(reconstructed, signal, atol=0.05)


def test_mulaw_codec_rejects_invalid_bits():
    # MuLawCodec::try_new's error is mapped through `to_py_err`, same
    # RuntimeError convention as test_lora_adapter_rejects_invalid_rank.
    with pytest.raises(RuntimeError):
        kizzasi.MuLawCodec(bits=0)
    with pytest.raises(RuntimeError):
        kizzasi.MuLawCodec(bits=17)
