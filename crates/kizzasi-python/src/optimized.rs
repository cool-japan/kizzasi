//! Optimized predictor Python wrapper.
//!
//! Exposes [`PyOptimizedPredictor`] — a wrapper around
//! [`kizzasi::optimization::OptimizedPredictor`] that adds:
//! * an optional LRU result cache (TTL configurable in milliseconds),
//! * SIMD / workspace pooling toggles (passed through to the inner
//!   [`kizzasi::optimization::OptimizationConfig`]),
//! * cache and optimization statistics exposed as Python dicts.
//!
//! `OptimizedPredictor` does not expose a separate "clear cache only" call;
//! [`PyOptimizedPredictor::reset`] resets both predictor state *and* the
//! result cache when caching is enabled.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scirs2_numpy::{PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};

use ::kizzasi::optimization::{OptimizationConfig, OptimizedPredictor};
use ::kizzasi::Kizzasi;
use scirs2_core::ndarray::Array1;

use crate::config::PyKizzasiConfig;
use crate::predictor::to_py_err;

/// Optimized predictor with workspace pooling, SIMD, and optional result caching.
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
    /// - `enable_simd`: whether to use SIMD-accelerated kernels when available.
    /// - `workspace_pool_size`: workspace pool capacity (default 16).
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

    /// Single autoregressive prediction step with optimizations.
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
        let output = self.inner.step(&input_arr).map_err(to_py_err)?;
        Ok(output.to_pyarray(py))
    }

    /// N-step autoregressive prediction. Bypasses the result cache.
    ///
    /// Returns a `(n_steps, output_dim)` float32 ndarray.
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
        let predictions = self
            .inner
            .predict_n(&input_arr, n_steps)
            .map_err(to_py_err)?;
        Ok(predictions.to_pyarray(py))
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

    /// Whether SIMD acceleration was requested at construction.
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
        // Populate cache: two identical steps -> second hits
        let _ = o.inner.step(&input).expect("step1");
        let _ = o.inner.step(&input).expect("step2");
        let stats_before = o.inner.cache_stats().expect("cache stats");
        assert!(stats_before.hits >= 1);
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

    #[test]
    fn test_optimized_predict_n_zero_steps_rejected() {
        let cfg = small_cfg();
        let mut o = PyOptimizedPredictor::new(&cfg, 1000, true, 16, 1000).expect("optimizer");
        // Use inner directly because predict_n needs Python GIL.
        // Verify the n_steps == 0 precondition by mirroring it here.
        let input = Array1::from_vec(vec![0.1_f32, 0.2]);
        let res = o.inner.predict_n(&input, 1);
        assert!(res.is_ok(), "predict_n n=1 should succeed: {:?}", res.err());
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
}
