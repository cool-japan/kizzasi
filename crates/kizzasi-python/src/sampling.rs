//! Sampling-strategy Python wrappers.
//!
//! Exposes:
//! - [`PySamplingConfig`] — a builder-style mirror of
//!   [`kizzasi_inference::sampling::SamplingConfig`] limited to the four core
//!   strategies (`greedy`, `temperature`, `top_k`, `top_p`).
//! - [`PySampler`] — a wrapper around
//!   [`kizzasi_inference::sampling::Sampler`] that runs single-vector and
//!   batched sampling on `f32` NumPy arrays.
//!
//! Beam search, custom samplers, and adaptive-rejection samplers are
//! intentionally *not* exposed here — they require either Python GIL-bound
//! callbacks or multi-step state that is better left to a higher-level
//! wrapper.
//!
//! ## Example
//!
//! ```python
//! import numpy as np
//! import kizzasi
//!
//! config = kizzasi.SamplingConfig()
//! config.strategy("top_k")
//! config.top_k(3)
//! config.temperature(1.5)
//! config.seed(42)
//!
//! sampler = kizzasi.Sampler(config)
//! logits = np.array([1.0, 3.0, 0.5, 2.5, 1.8], dtype=np.float32)
//! print(sampler.sample(logits))
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scirs2_numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};

use kizzasi_inference::{Sampler, SamplingConfig, SamplingStrategy};
use scirs2_core::ndarray::{Array1, Array2};

use crate::predictor::to_py_err;

/// Map a strategy name to a [`SamplingStrategy`]. Case-insensitive.
fn parse_strategy(name: &str) -> PyResult<SamplingStrategy> {
    match name.to_lowercase().as_str() {
        "greedy" => Ok(SamplingStrategy::Greedy),
        "temperature" | "temp" => Ok(SamplingStrategy::Temperature),
        "top_k" | "topk" | "top-k" => Ok(SamplingStrategy::TopK),
        "top_p" | "topp" | "top-p" | "nucleus" => Ok(SamplingStrategy::TopP),
        other => Err(PyValueError::new_err(format!(
            "Unsupported sampling strategy '{}'. Valid options: greedy, \
             temperature, top_k, top_p",
            other
        ))),
    }
}

/// Canonical lowercase name of a [`SamplingStrategy`]. Strategies that this
/// wrapper does not expose (`BeamSearch`, `Custom`) round-trip through their
/// debug-style name for diagnostic purposes.
fn strategy_name(strategy: SamplingStrategy) -> &'static str {
    match strategy {
        SamplingStrategy::Greedy => "greedy",
        SamplingStrategy::Temperature => "temperature",
        SamplingStrategy::TopK => "top_k",
        SamplingStrategy::TopP => "top_p",
        SamplingStrategy::BeamSearch => "beam_search",
        SamplingStrategy::Custom => "custom",
    }
}

/// Builder-style sampling configuration.
///
/// Each mutator returns `()` and modifies `self` in place so the wrapper is
/// natural to use from Python. Validation happens at the setter site, *not*
/// at construction, so partial configurations can be built incrementally
/// before being passed to [`PySampler::new`].
#[pyclass(name = "SamplingConfig", from_py_object)]
#[derive(Clone)]
pub struct PySamplingConfig {
    pub(crate) inner: SamplingConfig,
}

#[pymethods]
impl PySamplingConfig {
    /// Build a default configuration: greedy strategy, temperature `1.0`,
    /// no top-k / top-p, no seed.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SamplingConfig::new(),
        }
    }

    /// Select the sampling strategy by name.
    ///
    /// Accepts: `"greedy"`, `"temperature"` (alias `"temp"`), `"top_k"`
    /// (alias `"topk"`, `"top-k"`), `"top_p"` (alias `"topp"`, `"top-p"`,
    /// `"nucleus"`). Case-insensitive.
    pub fn strategy(&mut self, name: &str) -> PyResult<()> {
        self.inner.strategy = parse_strategy(name)?;
        Ok(())
    }

    /// Set the softmax temperature. Must be strictly positive.
    pub fn temperature(&mut self, t: f32) -> PyResult<()> {
        if !(t.is_finite() && t > 0.0) {
            return Err(PyValueError::new_err(
                "temperature must be a finite value > 0",
            ));
        }
        self.inner.temperature = t;
        Ok(())
    }

    /// Set the top-k candidate count. Must be `>= 1`. Also flips the strategy
    /// to `TopK` (matching the underlying builder semantics).
    pub fn top_k(&mut self, k: usize) -> PyResult<()> {
        if k == 0 {
            return Err(PyValueError::new_err("top_k must be >= 1"));
        }
        self.inner.strategy = SamplingStrategy::TopK;
        self.inner.top_k = Some(k);
        Ok(())
    }

    /// Set the top-p (nucleus) cumulative-probability threshold. Must be in
    /// `(0, 1]`. Also flips the strategy to `TopP`.
    pub fn top_p(&mut self, p: f32) -> PyResult<()> {
        if !(p.is_finite() && p > 0.0 && p <= 1.0) {
            return Err(PyValueError::new_err("top_p must be in (0, 1]"));
        }
        self.inner.strategy = SamplingStrategy::TopP;
        self.inner.top_p = Some(p);
        Ok(())
    }

    /// Set the random seed for reproducibility.
    pub fn seed(&mut self, s: u64) {
        self.inner.seed = Some(s);
    }

    /// Currently configured strategy name.
    #[getter]
    pub fn strategy_name(&self) -> String {
        strategy_name(self.inner.strategy).to_string()
    }

    /// Configured softmax temperature.
    ///
    /// Named `get_temperature` (not the bare `temperature`, which is
    /// already the builder-style setter method above) — without an
    /// explicit `#[getter(name)]` override, PyO3 auto-strips the `get_`
    /// prefix from a `#[getter]` method's Rust name, which would silently
    /// collide with and be shadowed by that setter (Python classes cannot
    /// have a plain method and a property share one name; whichever
    /// `#[pymethods]` registers last wins, leaving the other completely
    /// unreachable). Every getter below has the same explicit-name fix for
    /// the same reason.
    #[getter(get_temperature)]
    pub fn get_temperature(&self) -> f32 {
        self.inner.temperature
    }

    /// Configured top-k value (or `None` if unset).
    #[getter(get_top_k)]
    pub fn get_top_k(&self) -> Option<usize> {
        self.inner.top_k
    }

    /// Configured top-p value (or `None` if unset).
    #[getter(get_top_p)]
    pub fn get_top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    /// Configured random seed (or `None` if unset).
    #[getter(get_seed)]
    pub fn get_seed(&self) -> Option<u64> {
        self.inner.seed
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplingConfig(strategy='{}', temperature={}, top_k={:?}, top_p={:?}, seed={:?})",
            strategy_name(self.inner.strategy),
            self.inner.temperature,
            self.inner.top_k,
            self.inner.top_p,
            self.inner.seed,
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl Default for PySamplingConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Sampler that draws values from logit vectors according to a
/// [`PySamplingConfig`].
///
/// The sampler is stateful — the underlying [`Sampler`] re-uses a single RNG
/// across calls, so repeated `sample` invocations with the same logits and
/// the same seed will produce a deterministic *sequence* (but each call
/// returns a different value).
#[pyclass(name = "Sampler")]
pub struct PySampler {
    inner: Sampler,
}

#[pymethods]
impl PySampler {
    /// Construct a sampler from a [`PySamplingConfig`]. The config is cloned
    /// into the sampler, so later mutations to the Python `SamplingConfig`
    /// object will *not* be reflected here.
    ///
    /// Rejects a config whose `strategy` is `"top_k"`/`"top_p"` but whose
    /// matching value was never set: previously `config.strategy("top_k")`
    /// without a following `config.top_k(k)` silently sampled with the
    /// engine's internal default (`k=10`, or `p=0.9` for `"top_p"|`), with
    /// no indication anywhere that the value was never configured.
    #[new]
    pub fn new(config: &PySamplingConfig) -> PyResult<Self> {
        match config.inner.strategy {
            SamplingStrategy::TopK if config.inner.top_k.is_none() => {
                return Err(PyValueError::new_err(
                    "SamplingConfig strategy is 'top_k' but top_k(k) was never called; \
                     call config.top_k(k) before constructing a Sampler",
                ));
            }
            SamplingStrategy::TopP if config.inner.top_p.is_none() => {
                return Err(PyValueError::new_err(
                    "SamplingConfig strategy is 'top_p' but top_p(p) was never called; \
                     call config.top_p(p) before constructing a Sampler",
                ));
            }
            _ => {}
        }
        Ok(Self {
            inner: Sampler::new(config.inner.clone()),
        })
    }

    /// Sample a single value from a 1-D logits vector.
    ///
    /// Returns a single `f32` — the sampled value. For greedy / top-k /
    /// top-p sampling this is the index of the chosen logit cast to `f32`;
    /// the actual numeric meaning depends on the inference pipeline.
    ///
    /// Runs with the GIL released (`Python::detach`).
    pub fn sample(&mut self, py: Python<'_>, logits: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
        let arr: Array1<f32> = logits.as_array().to_owned();
        let inner = &mut self.inner;
        py.detach(|| inner.sample(&arr)).map_err(to_py_err)
    }

    /// Sample a batch of values from a `[batch, vocab]` logits matrix.
    ///
    /// Returns a 1-D float32 NumPy array of length `batch`. Runs with the
    /// GIL released (`Python::detach`).
    pub fn sample_batch<'py>(
        &mut self,
        py: Python<'py>,
        logits: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let arr: Array2<f32> = logits.as_array().to_owned();
        let inner = &mut self.inner;
        let out = py.detach(|| inner.sample_batch(&arr)).map_err(to_py_err)?;
        Ok(out.into_pyarray(py))
    }

    /// Canonical name of the active strategy.
    #[getter]
    pub fn strategy_name(&self) -> String {
        strategy_name(self.inner.config().strategy).to_string()
    }

    fn __repr__(&self) -> String {
        let cfg = self.inner.config();
        format!(
            "Sampler(strategy='{}', temperature={}, top_k={:?}, top_p={:?})",
            strategy_name(cfg.strategy),
            cfg.temperature,
            cfg.top_k,
            cfg.top_p,
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_config_basic() {
        let mut cfg = PySamplingConfig::new();
        assert_eq!(cfg.strategy_name(), "greedy");
        assert!((cfg.get_temperature() - 1.0).abs() < 1e-6);
        assert_eq!(cfg.get_top_k(), None);
        assert_eq!(cfg.get_top_p(), None);
        assert_eq!(cfg.get_seed(), None);

        cfg.strategy("temperature").expect("strategy");
        assert_eq!(cfg.strategy_name(), "temperature");

        cfg.temperature(0.7).expect("temperature");
        assert!((cfg.get_temperature() - 0.7).abs() < 1e-6);

        cfg.top_k(5).expect("top_k");
        assert_eq!(cfg.get_top_k(), Some(5));
        // top_k() also flips strategy to TopK
        assert_eq!(cfg.strategy_name(), "top_k");

        cfg.top_p(0.9).expect("top_p");
        assert_eq!(cfg.get_top_p(), Some(0.9));
        assert_eq!(cfg.strategy_name(), "top_p");

        cfg.seed(42);
        assert_eq!(cfg.get_seed(), Some(42));
    }

    #[test]
    fn test_sampling_config_invalid() {
        let mut cfg = PySamplingConfig::new();

        // bogus strategy
        assert!(cfg.strategy("nonexistent").is_err());
        assert!(cfg.strategy("beam_search").is_err());
        assert!(cfg.strategy("custom").is_err());

        // temperature <= 0
        assert!(cfg.temperature(0.0).is_err());
        assert!(cfg.temperature(-0.1).is_err());
        assert!(cfg.temperature(f32::NAN).is_err());
        assert!(cfg.temperature(f32::INFINITY).is_err());

        // k == 0
        assert!(cfg.top_k(0).is_err());

        // p out of range
        assert!(cfg.top_p(0.0).is_err());
        assert!(cfg.top_p(-0.1).is_err());
        assert!(cfg.top_p(1.5).is_err());
        assert!(cfg.top_p(f32::NAN).is_err());
    }

    #[test]
    fn test_sampling_config_strategy_aliases() {
        let mut cfg = PySamplingConfig::new();
        for name in &["greedy", "GREEDY", "Greedy"] {
            assert!(cfg.strategy(name).is_ok(), "alias '{}' should parse", name);
            assert_eq!(cfg.strategy_name(), "greedy");
        }
        for name in &["top_k", "TopK", "top-k"] {
            assert!(cfg.strategy(name).is_ok(), "alias '{}' should parse", name);
            assert_eq!(cfg.strategy_name(), "top_k");
        }
        for name in &["top_p", "topp", "nucleus"] {
            assert!(cfg.strategy(name).is_ok(), "alias '{}' should parse", name);
            assert_eq!(cfg.strategy_name(), "top_p");
        }
    }

    #[test]
    fn test_sampler_greedy() {
        let mut cfg = PySamplingConfig::new();
        cfg.strategy("greedy").expect("strategy");
        let mut sampler = PySampler::new(&cfg).expect("sampler");

        // Sample directly via the inner Sampler (avoids needing the GIL).
        let logits = Array1::from_vec(vec![1.0_f32, 3.0, 2.0]);
        let v = sampler.inner.sample(&logits).expect("greedy sample");
        // Greedy returns the *index* of the max value cast to f32 — here, 1.
        assert!((v - 1.0).abs() < 1e-6, "expected index 1, got {}", v);
        assert_eq!(sampler.strategy_name(), "greedy");
    }

    #[test]
    fn test_sampler_temperature() {
        let mut cfg = PySamplingConfig::new();
        cfg.strategy("temperature").expect("strategy");
        cfg.temperature(1.0).expect("temperature");
        cfg.seed(42);
        let mut sampler = PySampler::new(&cfg).expect("sampler");

        let logits = Array1::from_vec(vec![0.1_f32, 0.5, 0.3, 0.8, 0.2]);
        let v = sampler.inner.sample(&logits).expect("temp sample");
        // Result is an index in [0, 5), so 0 <= v < 5 and finite.
        assert!(v.is_finite(), "sampled value not finite: {}", v);
        assert!((0.0..5.0).contains(&v), "sampled index out of range: {}", v,);
    }

    #[test]
    fn test_sampler_top_k() {
        let mut cfg = PySamplingConfig::new();
        cfg.top_k(2).expect("top_k");
        cfg.seed(7);
        let mut sampler = PySampler::new(&cfg).expect("sampler");

        let logits = Array1::from_vec(vec![0.1_f32, 0.5, 0.3, 0.8, 0.2]);
        let v = sampler.inner.sample(&logits).expect("top_k sample");
        assert!(v.is_finite());
        // top-2 of the input picks indices 3 (0.8) and 1 (0.5), so the
        // sampled index must be one of those.
        let idx = v as usize;
        assert!(
            idx == 3 || idx == 1,
            "top_k=2 should pick index 3 or 1, got {}",
            idx,
        );
    }

    #[test]
    fn test_sampler_top_p() {
        let mut cfg = PySamplingConfig::new();
        cfg.top_p(0.9).expect("top_p");
        cfg.seed(13);
        let mut sampler = PySampler::new(&cfg).expect("sampler");

        let logits = Array1::from_vec(vec![0.1_f32, 0.5, 0.3, 0.8, 0.2]);
        let v = sampler.inner.sample(&logits).expect("top_p sample");
        assert!(v.is_finite());
        assert!((0.0..5.0).contains(&v));
    }

    // Regression: without an explicit `#[getter(get_temperature)]` name,
    // PyO3 auto-strips the `get_` prefix, so `get_temperature` would be
    // exposed to Python as the property `temperature` — colliding with,
    // and being silently shadowed by, the builder-style method also named
    // `temperature` (a Python class cannot have a plain method and a
    // property share one attribute name). This must be visible as a real,
    // independently-named Python attribute, not just callable from Rust.
    #[test]
    fn test_sampling_config_getters_do_not_collide_with_setters_in_python() {
        Python::initialize();
        Python::attach(|py| {
            let mut cfg = PySamplingConfig::new();
            cfg.temperature(0.7).expect("temperature");
            cfg.top_k(5).expect("top_k");
            cfg.seed(42);

            let py_cfg = Py::new(py, cfg).expect("py object").into_bound(py);
            // `temperature` must remain the *callable* setter, not a float.
            let temperature_attr = py_cfg.getattr("temperature").expect("temperature attr");
            assert!(
                temperature_attr.is_callable(),
                "`temperature` must stay the builder-style setter method"
            );
            // `get_temperature` must be present and be a plain float value,
            // not a method.
            let get_temperature_attr = py_cfg
                .getattr("get_temperature")
                .expect("get_temperature attr must exist");
            assert!(
                !get_temperature_attr.is_callable(),
                "`get_temperature` must be a property value, not a method"
            );
            let value: f32 = get_temperature_attr.extract().expect("extract f32");
            assert!((value - 0.7).abs() < 1e-5);

            let get_top_k: Option<usize> = py_cfg
                .getattr("get_top_k")
                .expect("get_top_k attr")
                .extract()
                .expect("extract");
            assert_eq!(get_top_k, Some(5));

            let get_seed: Option<u64> = py_cfg
                .getattr("get_seed")
                .expect("get_seed attr")
                .extract()
                .expect("extract");
            assert_eq!(get_seed, Some(42));
        });
    }

    #[test]
    fn test_sampler_batch_greedy() {
        let mut cfg = PySamplingConfig::new();
        cfg.strategy("greedy").expect("strategy");
        let mut sampler = PySampler::new(&cfg).expect("sampler");

        let logits = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.1_f32, 0.5, 0.3, 0.2, // row 0: max at 1
                0.8, 0.2, 0.1, 0.3, // row 1: max at 0
                0.2, 0.3, 0.9, 0.1, // row 2: max at 2
            ],
        )
        .expect("logits shape");

        let out = sampler.inner.sample_batch(&logits).expect("batch");
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
        assert!((out[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sampling_config_clone() {
        let mut a = PySamplingConfig::new();
        a.strategy("top_k").expect("strategy");
        a.top_k(7).expect("k");
        a.temperature(0.5).expect("t");

        let b = a.clone();
        assert_eq!(b.strategy_name(), "top_k");
        assert_eq!(b.get_top_k(), Some(7));
        assert!((b.get_temperature() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sampler_repr() {
        let mut cfg = PySamplingConfig::new();
        cfg.strategy("greedy").expect("strategy");
        let sampler = PySampler::new(&cfg).expect("sampler");
        let r = sampler.__repr__();
        assert!(r.contains("Sampler("));
        assert!(r.contains("greedy"));
    }

    #[test]
    fn test_sampling_config_repr() {
        let mut cfg = PySamplingConfig::new();
        cfg.strategy("top_p").expect("strategy");
        cfg.top_p(0.95).expect("p");
        let r = cfg.__repr__();
        assert!(r.contains("SamplingConfig("));
        assert!(r.contains("top_p"));
    }

    // Regression for the medium bug where `strategy("top_k")` without a
    // following `top_k(k)` call silently sampled with the engine's internal
    // default (k=10) — same for `strategy("top_p")` without `top_p(p)`
    // (default p=0.9) — with no error and no documented default anywhere.
    #[test]
    fn test_sampler_new_rejects_top_k_without_value() {
        let mut cfg = PySamplingConfig::new();
        cfg.strategy("top_k").expect("strategy");
        // top_k(k) deliberately not called.
        assert!(PySampler::new(&cfg).is_err());
    }

    #[test]
    fn test_sampler_new_rejects_top_p_without_value() {
        let mut cfg = PySamplingConfig::new();
        cfg.strategy("top_p").expect("strategy");
        // top_p(p) deliberately not called.
        assert!(PySampler::new(&cfg).is_err());
    }

    #[test]
    fn test_sampler_new_accepts_top_k_with_value() {
        let mut cfg = PySamplingConfig::new();
        cfg.top_k(3).expect("top_k"); // also flips strategy to TopK
        assert!(PySampler::new(&cfg).is_ok());
    }

    #[test]
    fn test_sampler_sample_crosses_pyo3_boundary() {
        use scirs2_numpy::{PyArray1, PyArrayMethods};
        Python::initialize();
        Python::attach(|py| {
            let mut cfg = PySamplingConfig::new();
            cfg.strategy("greedy").expect("strategy");
            let mut sampler = PySampler::new(&cfg).expect("sampler");
            let py_arr = PyArray1::from_vec(py, vec![1.0_f32, 3.0, 2.0]);
            let v = sampler
                .sample(py, py_arr.readonly())
                .expect("sample via pymethod");
            assert!((v - 1.0).abs() < 1e-6, "expected index 1, got {}", v);
        });
    }

    #[test]
    fn test_sampler_sample_batch_crosses_pyo3_boundary() {
        use scirs2_numpy::{PyArray2, PyArrayMethods};
        Python::initialize();
        Python::attach(|py| {
            let mut cfg = PySamplingConfig::new();
            cfg.strategy("greedy").expect("strategy");
            let mut sampler = PySampler::new(&cfg).expect("sampler");
            let py_arr = PyArray2::from_vec2(
                py,
                &[
                    vec![0.1_f32, 0.5, 0.3, 0.2],
                    vec![0.8, 0.2, 0.1, 0.3],
                    vec![0.2, 0.3, 0.9, 0.1],
                ],
            )
            .expect("logits shape");
            let out = sampler
                .sample_batch(py, py_arr.readonly())
                .expect("sample_batch via pymethod");
            let out_ro = out.readonly();
            let view = out_ro.as_array();
            assert_eq!(view.len(), 3);
            assert!((view[0] - 1.0).abs() < 1e-6);
            assert!((view[1] - 0.0).abs() < 1e-6);
            assert!((view[2] - 2.0).abs() < 1e-6);
        });
    }

    /// `Sampler` must stay `Send` for `Python::detach` (used by `sample`/
    /// `sample_batch`) to compile, and for `unsendable` to be safely
    /// removable from `#[pyclass(name = "Sampler")]` — unlike `Kizzasi`,
    /// `Sampler` holds no `PluginManager`, so it is expected to be both
    /// `Send` and `Sync`.
    #[test]
    fn test_sampler_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Sampler>();
    }
}
