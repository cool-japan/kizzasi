//! Optimized predictor Python wrapper.
//!
//! Exposes [`PyOptimizedPredictor`] — a wrapper around
//! [`kizzasi::optimization::OptimizedPredictor`] that adds:
//! * an optional LRU result cache (TTL configurable in milliseconds) — this
//!   part is real and measurable via [`PyOptimizedPredictor::cache_stats`].
//!   It serves [`PyOptimizedPredictor::predict_stateless`] and nothing else:
//!   `step` drives a stateful recurrence, so memoising it would answer under
//!   the wrong hidden state *and* skip the state update (see
//!   `kizzasi::optimization::OptimizationConfig::enable_result_cache`). Before
//!   `predict_stateless` was exposed here, the cache was unreachable from
//!   Python and `cache_stats()["hits"]` could never leave zero,
//! * `enable_simd` / `workspace_pool_size` constructor parameters that are
//!   accepted, stored, and reported back, but **currently have no effect on
//!   computation**: `kizzasi::optimization::OptimizedPredictor::step` /
//!   `predict_n` never read `OptimizationConfig::enable_simd` or
//!   `enable_workspace_pooling`, and the `workspace_pool_hits` /
//!   `workspace_allocations` counters in
//!   [`PyOptimizedPredictor::optimization_stats`] are always `0` because
//!   nothing in the engine increments them. Wiring real SIMD dispatch or
//!   workspace-pool accounting requires changes to `kizzasi::optimization`
//!   in the `kizzasi` crate, outside this binding crate's scope — until
//!   that lands, treat `enable_simd`/`workspace_pool_size` as forward-
//!   compatible no-ops, not as a performance lever.
//! * cache and optimization statistics exposed as Python dicts.
//!
//! `OptimizedPredictor` does not expose a separate "clear cache only" call;
//! [`PyOptimizedPredictor::reset`] resets both predictor state *and* the
//! result cache when caching is enabled.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};

use ::kizzasi::optimization::{OptimizationConfig, OptimizedPredictor};
use ::kizzasi::Kizzasi;
use scirs2_core::ndarray::Array1;

use crate::config::PyKizzasiConfig;
use crate::predictor::to_py_err;

/// Optimized predictor with an optional TTL-based LRU result cache.
///
/// `enable_simd` and `workspace_pool_size` are accepted for forward
/// compatibility but are currently inert — see the module-level docs for
/// why (the wiring lives in the `kizzasi` crate, outside this binding
/// crate's scope) and [`Self::simd_enabled`] / [`Self::optimization_stats`]
/// for exactly what that means at each call site.
#[pyclass(name = "OptimizedPredictor", unsendable)]
pub struct PyOptimizedPredictor {
    inner: OptimizedPredictor,
    config_snapshot: PyKizzasiConfig,
    input_dim: usize,
    output_dim: usize,
    cache_ttl_ms: u64,
    cache_enabled: bool,
    simd_enabled: bool,
}

#[pymethods]
impl PyOptimizedPredictor {
    /// Build an optimized predictor from a Config.
    ///
    /// Parameters:
    /// - `config`: shared [`Config`] for the underlying predictor.
    /// - `cache_ttl_ms`: TTL for cached results (milliseconds). `0` disables
    ///   the result cache entirely.
    /// - `enable_simd`: stored and returned via [`Self::simd_enabled`], but
    ///   **currently has no effect on computation** — see the module docs.
    /// - `workspace_pool_size`: accepted for forward compatibility;
    ///   **currently has no effect** — see the module docs.
    /// - `result_cache_size`: max cached entries (default 1000).
    #[new]
    #[pyo3(signature = (
        config,
        cache_ttl_ms = 1000,
        enable_simd = true,
        workspace_pool_size = 16,
        result_cache_size = 1000,
    ))]
    pub fn new(
        config: &PyKizzasiConfig,
        cache_ttl_ms: u64,
        enable_simd: bool,
        workspace_pool_size: usize,
        result_cache_size: usize,
    ) -> PyResult<Self> {
        let core_config = config.to_core_config()?;
        let predictor = Kizzasi::new(core_config).map_err(to_py_err)?;

        let cache_enabled = cache_ttl_ms > 0 && result_cache_size > 0;
        let opt_config = OptimizationConfig::default()
            .with_workspace_pooling(true)
            .with_simd(enable_simd)
            .with_workspace_pool_size(workspace_pool_size)
            .with_result_cache(cache_enabled)
            .with_result_cache_size(result_cache_size)
            .with_cache_ttl(cache_ttl_ms.max(1));

        let inner = OptimizedPredictor::new(predictor, opt_config);

        Ok(Self {
            inner,
            config_snapshot: config.clone(),
            input_dim: config.input_dim,
            output_dim: config.output_dim,
            cache_ttl_ms,
            cache_enabled,
            simd_enabled: enable_simd,
        })
    }

    /// Single autoregressive prediction step. Never served from the result
    /// cache — this advances the recurrence, so a cached answer would be
    /// computed under a different hidden state and would skip the state
    /// update. Use [`Self::predict_stateless`] for the memoisable evaluation.
    /// Runs with the GIL released (`Python::detach`).
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let arr = input.as_array();
        if arr.len() != self.input_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Input length {} does not match predictor input_dim {}",
                arr.len(),
                self.input_dim
            )));
        }
        let input_arr: Array1<f32> = arr.to_owned();
        let inner = &mut self.inner;
        let output = py.detach(|| inner.step(&input_arr)).map_err(to_py_err)?;
        Ok(output.into_pyarray(py))
    }

    /// N-step autoregressive prediction. Bypasses the result cache.
    ///
    /// Returns a `(n_steps, output_dim)` float32 ndarray. Runs with the GIL
    /// released (`Python::detach`).
    pub fn predict_n<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<'_, f32>,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if n_steps == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_steps must be >= 1",
            ));
        }
        let arr = input.as_array();
        if arr.len() != self.input_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Input length {} does not match predictor input_dim {}",
                arr.len(),
                self.input_dim
            )));
        }
        let input_arr: Array1<f32> = arr.to_owned();
        let inner = &mut self.inner;
        let predictions = py
            .detach(|| inner.predict_n(&input_arr, n_steps))
            .map_err(to_py_err)?;
        Ok(predictions.into_pyarray(py))
    }

    /// Evaluate the model as a pure function of `input`, from a reset state,
    /// restoring the live stream state afterwards.
    ///
    /// This is the call the result cache serves: identical inputs within the
    /// configured TTL are answered from the cache and show up as `hits` in
    /// [`Self::cache_stats`]. Unlike [`Self::step`] it does not advance the
    /// stream. Runs with the GIL released (`Python::detach`).
    pub fn predict_stateless<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let arr = input.as_array();
        if arr.len() != self.input_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Input length {} does not match predictor input_dim {}",
                arr.len(),
                self.input_dim
            )));
        }
        let input_arr: Array1<f32> = arr.to_owned();
        let inner = &mut self.inner;
        let output = py
            .detach(|| inner.predict_stateless(&input_arr))
            .map_err(to_py_err)?;
        Ok(output.into_pyarray(py))
    }

    /// Reset predictor state and clear caches.
    pub fn reset(&mut self) -> PyResult<()> {
        self.inner.reset().map_err(to_py_err)
    }

    /// Clear cached results (and reset predictor state — the underlying
    /// optimizer does not expose a cache-only reset).
    pub fn clear_cache(&mut self) -> PyResult<()> {
        self.reset()
    }

    /// Return cache statistics: `size`, `capacity`, `hits`, `misses`, `hit_rate`.
    pub fn cache_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.cache_stats().map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("size", stats.size)?;
        dict.set_item("capacity", stats.capacity)?;
        dict.set_item("hits", stats.hits)?;
        dict.set_item("misses", stats.misses)?;
        dict.set_item("hit_rate", stats.hit_rate)?;
        dict.set_item("enabled", self.cache_enabled)?;
        Ok(dict)
    }

    /// Return optimization stats: `total_predictions`, `cached_predictions`,
    /// `cache_time_saved_us`, `avg_prediction_time_us`,
    /// `workspace_pool_hits`, `workspace_allocations`.
    ///
    /// `workspace_pool_hits` and `workspace_allocations` are **always `0`**:
    /// nothing in the current `kizzasi::optimization` engine increments
    /// them, regardless of `workspace_pool_size` or how many predictions
    /// have run. Treat them as reserved for when workspace-pool accounting
    /// is implemented upstream, not as real telemetry today.
    pub fn optimization_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.optimization_stats().map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("total_predictions", s.total_predictions)?;
        dict.set_item("cached_predictions", s.cached_predictions)?;
        dict.set_item("cache_time_saved_us", s.cache_time_saved_us)?;
        dict.set_item("avg_prediction_time_us", s.avg_prediction_time_us)?;
        dict.set_item("workspace_pool_hits", s.workspace_pool_hits)?;
        dict.set_item("workspace_allocations", s.workspace_allocations)?;
        Ok(dict)
    }

    /// Per-step input dimension.
    #[getter]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Per-step output dimension.
    #[getter]
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Whether SIMD acceleration was *requested* at construction.
    ///
    /// This reflects the `enable_simd` argument passed to `__init__` only —
    /// it does **not** mean SIMD kernels are actually used. The current
    /// `kizzasi::optimization` engine never reads this flag, so `step`/
    /// `predict_n` run identical code whether this is `True` or `False`.
    #[getter]
    pub fn simd_enabled(&self) -> bool {
        self.simd_enabled
    }

    /// Whether the result cache is currently active.
    #[getter]
    pub fn cache_enabled(&self) -> bool {
        self.cache_enabled
    }

    /// Configured cache TTL in milliseconds (0 means disabled).
    #[getter]
    pub fn cache_ttl_ms(&self) -> u64 {
        self.cache_ttl_ms
    }

    /// Shared configuration used to construct the underlying predictor.
    #[getter]
    pub fn config(&self) -> PyKizzasiConfig {
        self.config_snapshot.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "OptimizedPredictor(input_dim={}, output_dim={}, simd={}, \
             cache_enabled={}, cache_ttl_ms={})",
            self.input_dim,
            self.output_dim,
            self.simd_enabled,
            self.cache_enabled,
            self.cache_ttl_ms,
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> PyKizzasiConfig {
        PyKizzasiConfig::new(2, 2, 32, 1, 4, 256, "mamba2".to_string())
    }

    #[test]
    fn test_optimized_basic_creation() {
        let cfg = small_cfg();
        let opt = PyOptimizedPredictor::new(&cfg, 1000, true, 16, 1000);
        assert!(opt.is_ok(), "creation failed: {:?}", opt.err());
        let o = opt.expect("optimizer");
        assert_eq!(o.input_dim(), 2);
        assert_eq!(o.output_dim(), 2);
        assert!(o.simd_enabled());
        assert!(o.cache_enabled());
        assert_eq!(o.cache_ttl_ms(), 1000);
    }

    #[test]
    fn test_optimized_step_inner() {
        let cfg = small_cfg();
        let mut o = PyOptimizedPredictor::new(&cfg, 1000, true, 16, 1000).expect("optimizer");
        let input = Array1::from_vec(vec![0.1_f32, 0.2]);
        let out = o.inner.step(&input).expect("step");
        assert_eq!(out.len(), 2);
        for v in out.iter() {
            assert!(v.is_finite(), "output value {} not finite", v);
        }
    }

    #[test]
    fn test_optimized_cache_disabled_when_ttl_zero() {
        let cfg = small_cfg();
        let o = PyOptimizedPredictor::new(&cfg, 0, true, 16, 1000).expect("optimizer");
        assert!(!o.cache_enabled());
    }

    #[test]
    fn test_optimized_cache_disabled_when_size_zero() {
        let cfg = small_cfg();
        let o = PyOptimizedPredictor::new(&cfg, 1000, true, 16, 0).expect("optimizer");
        assert!(!o.cache_enabled());
    }

    #[test]
    fn test_optimized_simd_off() {
        let cfg = small_cfg();
        let o = PyOptimizedPredictor::new(&cfg, 1000, false, 16, 1000).expect("optimizer");
        assert!(!o.simd_enabled());
    }

    #[test]
    fn test_optimized_reset_clears_cache_when_enabled() {
        let cfg = small_cfg();
        let mut o = PyOptimizedPredictor::new(&cfg, 1000, true, 16, 1000).expect("optimizer");
        let input = Array1::from_vec(vec![0.1_f32, 0.2]);
        // Populate the cache through the *memoisable* entry point. `step` is
        // deliberately never cached (it advances the recurrence), so driving
        // this through `step` -- as this test used to -- could never produce
        // a hit and only asserted that the cache stayed empty.
        let _ = o
            .inner
            .predict_stateless(&input)
            .expect("stateless predict 1");
        let _ = o
            .inner
            .predict_stateless(&input)
            .expect("stateless predict 2");
        let stats_before = o.inner.cache_stats().expect("cache stats");
        assert!(
            stats_before.hits >= 1,
            "a repeated stateless prediction must hit the cache, got {stats_before:?}"
        );
        // Reset should clear cache
        o.reset().expect("reset");
        let stats_after = o.inner.cache_stats().expect("cache stats after");
        assert_eq!(stats_after.hits, 0);
        assert_eq!(stats_after.size, 0);
    }

    #[test]
    fn test_optimized_repr() {
        let cfg = small_cfg();
        let o = PyOptimizedPredictor::new(&cfg, 500, false, 8, 100).expect("optimizer");
        let r = o.__repr__();
        assert!(r.contains("OptimizedPredictor("));
        assert!(r.contains("input_dim=2"));
        assert!(r.contains("simd=false"));
    }

    // Regression for the test-gap finding: this previously never passed
    // `n_steps=0` at all — it called `predict_n(&input, 1)` and asserted
    // `is_ok()`, leaving the `n_steps == 0` guard in the real `#[pymethods]`
    // `predict_n` completely uncovered. Now it drives the actual guard
    // through `Python::attach` + a real `PyArray1`.
    #[test]
    fn test_optimized_predict_n_zero_steps_rejected() {
        use scirs2_numpy::{PyArray1, PyArrayMethods};
        Python::initialize();
        Python::attach(|py| {
            let cfg = small_cfg();
            let mut o = PyOptimizedPredictor::new(&cfg, 1000, true, 16, 1000).expect("optimizer");
            let py_arr = PyArray1::from_vec(py, vec![0.1_f32, 0.2]);
            let res = o.predict_n(py, py_arr.readonly(), 0);
            assert!(res.is_err(), "n_steps=0 must be rejected");
        });
    }

    #[test]
    fn test_optimized_predict_n_crosses_pyo3_boundary() {
        use scirs2_numpy::{PyArray1, PyArrayMethods};
        Python::initialize();
        Python::attach(|py| {
            let cfg = small_cfg();
            let mut o = PyOptimizedPredictor::new(&cfg, 1000, true, 16, 1000).expect("optimizer");
            let py_arr = PyArray1::from_vec(py, vec![0.1_f32, 0.2]);
            let out = o
                .predict_n(py, py_arr.readonly(), 3)
                .expect("predict_n via pymethod");
            let out_ro = out.readonly();
            let view = out_ro.as_array();
            assert_eq!(view.shape(), &[3, 2]);
        });
    }

    #[test]
    fn test_optimized_step_crosses_pyo3_boundary() {
        use scirs2_numpy::{PyArray1, PyArrayMethods};
        Python::initialize();
        Python::attach(|py| {
            let cfg = small_cfg();
            let mut o = PyOptimizedPredictor::new(&cfg, 1000, true, 16, 1000).expect("optimizer");
            let py_arr = PyArray1::from_vec(py, vec![0.1_f32, 0.2]);
            let out = o.step(py, py_arr.readonly()).expect("step via pymethod");
            let out_ro = out.readonly();
            assert_eq!(out_ro.as_array().len(), 2);
        });
    }

    #[test]
    fn test_optimized_clear_cache_alias() {
        let cfg = small_cfg();
        let mut o = PyOptimizedPredictor::new(&cfg, 1000, true, 16, 1000).expect("optimizer");
        let input = Array1::from_vec(vec![0.1_f32, 0.2]);
        let _ = o.inner.step(&input).expect("step1");
        o.clear_cache().expect("clear_cache");
        let stats = o.inner.cache_stats().expect("cache stats");
        assert_eq!(stats.size, 0);
    }

    #[test]
    fn test_optimized_input_dim_mismatch_in_predict_n() {
        let cfg = small_cfg();
        let mut o = PyOptimizedPredictor::new(&cfg, 1000, true, 16, 1000).expect("optimizer");
        // predict_n with mismatched input dimension should be caught by inner predictor.
        let bad_input = Array1::from_vec(vec![0.1_f32, 0.2, 0.3]);
        let res = o.inner.predict_n(&bad_input, 2);
        assert!(res.is_err());
    }

    // Regression for the honest-docs fix: `workspace_pool_hits` and
    // `workspace_allocations` must stay documented (and observed) as
    // always-zero placeholders, not silently-wrong "real" telemetry.
    #[test]
    fn test_optimization_stats_workspace_fields_are_always_zero() {
        let cfg = small_cfg();
        let mut o = PyOptimizedPredictor::new(&cfg, 1000, true, 16, 1000).expect("optimizer");
        let input = Array1::from_vec(vec![0.1_f32, 0.2]);
        for _ in 0..5 {
            let _ = o.inner.step(&input).expect("step");
        }
        let stats = o.inner.optimization_stats().expect("optimization stats");
        assert_eq!(stats.workspace_pool_hits, 0);
        assert_eq!(stats.workspace_allocations, 0);
    }

    /// `OptimizedPredictor` must stay `Send` for `Python::detach` (used by
    /// `step`/`predict_n`) to compile.
    #[test]
    fn test_optimized_predictor_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<OptimizedPredictor>();
    }
}
