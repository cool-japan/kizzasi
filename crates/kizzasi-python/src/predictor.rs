//! Single-predictor Python wrapper.
//!
//! Exposes:
//! - [`PyConstraintSpec`] — scalar value-range constraint description used by
//!   the predictor's guardrails.
//! - [`PyPredictor`] — wraps a single [`kizzasi::Kizzasi`] instance and offers
//!   `step`, `predict_n`, `step_list`, `reset`, and guardrail management.

use pyo3::prelude::*;
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};

use ::kizzasi::Kizzasi;
use scirs2_core::ndarray::Array1;

use crate::config::PyKizzasiConfig;

/// Convert any Display-able error into a Python RuntimeError.
pub(crate) fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Scalar value-range constraint for use with the predictor's guardrails.
#[pyclass(name = "ConstraintSpec", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyConstraintSpec {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub min_val: Option<f32>,
    #[pyo3(get, set)]
    pub max_val: Option<f32>,
    #[pyo3(get, set)]
    pub dimension: Option<usize>,
    #[pyo3(get, set)]
    pub hard_reject: bool,
}

#[pymethods]
impl PyConstraintSpec {
    #[new]
    #[pyo3(signature = (name, min_val = None, max_val = None, dimension = None, hard_reject = false))]
    pub fn new(
        name: String,
        min_val: Option<f32>,
        max_val: Option<f32>,
        dimension: Option<usize>,
        hard_reject: bool,
    ) -> PyResult<Self> {
        if min_val.is_none() && max_val.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "At least one of min_val or max_val must be provided",
            ));
        }
        Ok(Self {
            name,
            min_val,
            max_val,
            dimension,
            hard_reject,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ConstraintSpec(name='{}', min_val={:?}, max_val={:?}, dimension={:?}, hard_reject={})",
            self.name, self.min_val, self.max_val, self.dimension, self.hard_reject
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Translate a slice of [`PyConstraintSpec`] into a `GuardrailSet`.
///
/// Both bounds are validated *before* any `ConstraintBuilder` call: a
/// two-sided spec routes to `in_range(lo, hi)` (previously it silently
/// dropped `min_val` — `less_eq` after `greater_eq` overwrote the single
/// `bound` field the builder holds, so only the upper bound was ever
/// enforced). `in_range`'s enforcement is `value.clamp(lo, hi)`, which
/// panics if `lo > hi` or either bound is NaN, so both are rejected here
/// with a `PyValueError` rather than being allowed to reach `clamp`. A NaN
/// single-sided bound is rejected too: `f32::max`/`f32::min` silently
/// ignore a NaN operand, which previously made `check()` always report a
/// violation while `project()` silently passed the value through
/// unchanged — a guardrail that looks active but enforces nothing.
pub(crate) fn build_guardrails(
    specs: &[PyConstraintSpec],
) -> PyResult<kizzasi_logic::GuardrailSet> {
    use kizzasi_logic::{ConstraintBuilder, Guardrail, GuardrailSet};

    let mut guardrail_set = GuardrailSet::new();
    for spec in specs {
        if let Some(lo) = spec.min_val {
            if !lo.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "ConstraintSpec '{}': min_val must be finite, got {}",
                    spec.name, lo
                )));
            }
        }
        if let Some(hi) = spec.max_val {
            if !hi.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "ConstraintSpec '{}': max_val must be finite, got {}",
                    spec.name, hi
                )));
            }
        }
        if let (Some(lo), Some(hi)) = (spec.min_val, spec.max_val) {
            if lo > hi {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "ConstraintSpec '{}': min_val ({}) must be <= max_val ({})",
                    spec.name, lo, hi
                )));
            }
        }

        let builder = ConstraintBuilder::new().name(&spec.name);
        let builder = match (spec.min_val, spec.max_val) {
            (Some(lo), Some(hi)) => builder.in_range(lo, hi),
            (Some(lo), None) => builder.greater_eq(lo),
            (None, Some(hi)) => builder.less_eq(hi),
            // Reachable via `spec.min_val = None` / `spec.max_val = None`
            // mutation after construction (both are `#[pyo3(set)]`), even
            // though `PyConstraintSpec::new` rejects it up front. Leaving
            // no bound set makes `ConstraintBuilder::build()` return its
            // existing "bound is required" error below instead of a panic.
            (None, None) => builder,
        };
        let constraint = builder.build().map_err(to_py_err)?;
        let guardrail = Guardrail::new(constraint, spec.hard_reject);
        match spec.dimension {
            Some(dim) => guardrail_set.add_dimensional(dim, guardrail),
            None => guardrail_set.add_global(guardrail),
        }
    }
    Ok(guardrail_set)
}

/// The main Kizzasi AGSP predictor.
///
/// `step`/`predict_n`/`step_list` release the GIL for the duration of their
/// computation (`Python::detach`), so separate `Predictor` instances —
/// **one created and used per thread** — run their SSM math in true
/// parallel instead of serializing on the GIL.
///
/// Predictor instances are still `unsendable` (pinned to the thread that
/// created them): [`Kizzasi`] is `Send` (verified below) but not `Sync` —
/// it embeds a `kizzasi::plugin::PluginManager`, whose `Vec<Box<dyn
/// Plugin>>` is only `Send` because `Plugin: Send` (not `Plugin: Send +
/// Sync`) — and PyO3 requires a non-`unsendable` pyclass to be both. That
/// trait bound lives in the `kizzasi` crate, outside this binding crate's
/// scope. In practice this only matters if you try to *share* one
/// `Predictor` object across threads (e.g. via a queue); creating a fresh
/// instance per worker thread, the documented and recommended pattern, is
/// unaffected and now genuinely parallel during computation.
#[pyclass(name = "Predictor", unsendable)]
pub struct PyPredictor {
    inner: Kizzasi,
    config_snapshot: PyKizzasiConfig,
}

#[pymethods]
impl PyPredictor {
    /// Construct a Predictor from a Config.
    #[new]
    pub fn new(config: &PyKizzasiConfig) -> PyResult<Self> {
        let core_config = config.to_core_config()?;
        let inner = Kizzasi::new(core_config).map_err(to_py_err)?;
        Ok(Self {
            inner,
            config_snapshot: config.clone(),
        })
    }

    /// Single autoregressive prediction step.
    ///
    /// Parameters: input ndarray shape (input_dim,) float32
    /// Returns: ndarray shape (output_dim,) float32
    ///
    /// The SSM step itself runs with the GIL released (`Python::detach`), so
    /// other Python threads keep running while this instance computes.
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let arr = input.as_array();
        let expected = self.inner.input_dim();
        let actual = arr.len();
        if actual != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Input length {} does not match predictor input_dim {}",
                actual, expected
            )));
        }
        let input_arr: Array1<f32> = arr.to_owned();
        let inner = &mut self.inner;
        let output = py.detach(|| inner.step(&input_arr)).map_err(to_py_err)?;
        Ok(output.into_pyarray(py))
    }

    /// N-step autoregressive prediction.
    ///
    /// Parameters: input ndarray (input_dim,) float32, n_steps int
    /// Returns: ndarray (n_steps, output_dim) float32
    ///
    /// The autoregressive loop runs with the GIL released (`Python::detach`).
    pub fn predict_n<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<'_, f32>,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let arr = input.as_array();
        let expected = self.inner.input_dim();
        let actual = arr.len();
        if actual != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Input length {} does not match predictor input_dim {}",
                actual, expected
            )));
        }
        let input_arr: Array1<f32> = arr.to_owned();
        let inner = &mut self.inner;
        let predictions = py
            .detach(|| inner.predict_n(&input_arr, n_steps))
            .map_err(to_py_err)?;
        Ok(predictions.into_pyarray(py))
    }

    /// Single prediction step accepting a Python list.
    ///
    /// Runs with the GIL released (`Python::detach`) like [`Self::step`].
    pub fn step_list(&mut self, py: Python<'_>, input: Vec<f32>) -> PyResult<Vec<f32>> {
        let expected = self.inner.input_dim();
        let actual = input.len();
        if actual != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Input length {} does not match predictor input_dim {}",
                actual, expected
            )));
        }
        let inner = &mut self.inner;
        let output = py
            .detach(move || {
                let input_arr = Array1::from_vec(input);
                inner.step(&input_arr)
            })
            .map_err(to_py_err)?;
        Ok(output.to_vec())
    }

    /// Reset the predictor's internal hidden state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Apply value-range guardrails to clip / reject out-of-bound predictions.
    ///
    /// Any spec whose `dimension` is out of range for this predictor's
    /// `output_dim` is rejected up front: `GuardrailSet::constrain` silently
    /// skips out-of-range dimensional guardrails (`if *dim < result.len()`),
    /// so without this check a mistyped `dimension` (e.g. an off-by-one on a
    /// `hard_reject=True` safety limit) would be accepted, report
    /// `has_guardrails() == True`, and enforce nothing.
    pub fn set_guardrails(&mut self, specs: Vec<PyConstraintSpec>) -> PyResult<()> {
        let output_dim = self.inner.output_dim();
        for spec in &specs {
            if let Some(dim) = spec.dimension {
                if dim >= output_dim {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "ConstraintSpec '{}': dimension {} is out of range for \
                         output_dim {} (valid range: 0..{})",
                        spec.name, dim, output_dim, output_dim
                    )));
                }
            }
        }
        let guardrails = build_guardrails(&specs)?;
        self.inner.set_guardrails(guardrails);
        Ok(())
    }

    /// Remove all currently active guardrails.
    pub fn clear_guardrails(&mut self) {
        self.inner.clear_guardrails();
    }

    /// Whether any guardrails are currently active.
    pub fn has_guardrails(&self) -> bool {
        self.inner.has_guardrails()
    }

    #[getter]
    pub fn input_dim(&self) -> usize {
        self.inner.input_dim()
    }

    #[getter]
    pub fn output_dim(&self) -> usize {
        self.inner.output_dim()
    }

    #[getter]
    pub fn hidden_dim(&self) -> usize {
        self.inner.hidden_dim()
    }

    #[getter]
    pub fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    #[getter]
    pub fn state_dim(&self) -> usize {
        self.inner.state_dim()
    }

    #[getter]
    pub fn context_window(&self) -> usize {
        self.inner.context_window()
    }

    #[getter]
    pub fn model_type(&self) -> String {
        self.config_snapshot.model_type.clone()
    }

    #[getter]
    pub fn config(&self) -> PyKizzasiConfig {
        self.config_snapshot.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Predictor(input_dim={}, output_dim={}, hidden_dim={}, model_type='{}')",
            self.inner.input_dim(),
            self.inner.output_dim(),
            self.inner.hidden_dim(),
            self.config_snapshot.model_type,
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pilot test proving the `Python::attach` + real `PyArray` pattern used
    // throughout this module's expanded test suite actually crosses the
    // PyO3 boundary end-to-end (numpy alloc -> pymethod call -> numpy read
    // back), rather than calling through `.inner` and skipping PyO3 entirely.
    #[test]
    fn test_predictor_step_crosses_pyo3_boundary_pilot() {
        use scirs2_numpy::{PyArray1, PyArrayMethods};
        Python::initialize();
        Python::attach(|py| {
            let cfg = PyKizzasiConfig::new(2, 2, 16, 1, 4, 64, "mamba2".to_string());
            let mut pred = PyPredictor::new(&cfg).expect("predictor");
            let py_arr = PyArray1::from_vec(py, vec![0.1_f32, 0.2]);
            let readonly = py_arr.readonly();
            let out = pred.step(py, readonly).expect("step via pymethod");
            let out_ro = out.readonly();
            let view = out_ro.as_array();
            assert_eq!(view.len(), 2);
            for v in view.iter() {
                assert!(v.is_finite());
            }
        });
    }

    #[test]
    fn test_predictor_creation_mamba2() {
        let cfg = PyKizzasiConfig::new(3, 3, 64, 2, 8, 1024, "mamba2".to_string());
        let pred = PyPredictor::new(&cfg);
        assert!(pred.is_ok(), "Predictor::new failed: {:?}", pred.err());
        let p = pred.expect("predictor");
        assert_eq!(p.input_dim(), 3);
        assert_eq!(p.output_dim(), 3);
        assert_eq!(p.hidden_dim(), 64);
        assert_eq!(p.model_type(), "mamba2");
    }

    #[test]
    fn test_predictor_creation_all_model_types() {
        for model_type in &["mamba", "mamba2", "s4", "rwkv"] {
            let cfg = PyKizzasiConfig::new(2, 2, 32, 1, 4, 256, model_type.to_string());
            let pred = PyPredictor::new(&cfg);
            assert!(
                pred.is_ok(),
                "Failed to create predictor with model_type='{}': {:?}",
                model_type,
                pred.err()
            );
        }
    }

    #[test]
    fn test_predictor_reset() {
        let cfg = PyKizzasiConfig::new(2, 2, 32, 1, 4, 256, "mamba2".to_string());
        let mut pred = PyPredictor::new(&cfg).expect("predictor");
        pred.reset();
        pred.reset();
    }

    // `step_list` now takes an (implicit-to-Python, injected-by-pyo3) `py:
    // Python<'_>` parameter so it can release the GIL via `Python::detach`
    // internally, so these two Rust-level tests need a live GIL token —
    // same signature-driven update as the other `Python::attach`-based
    // tests in this module.
    #[test]
    fn test_predictor_step_list() {
        Python::initialize();
        Python::attach(|py| {
            let cfg = PyKizzasiConfig::new(3, 3, 32, 1, 4, 256, "mamba2".to_string());
            let mut pred = PyPredictor::new(&cfg).expect("predictor");
            let output = pred.step_list(py, vec![0.1, 0.2, 0.3]);
            assert!(output.is_ok(), "step_list failed: {:?}", output.err());
            let out = output.expect("output");
            assert_eq!(out.len(), 3);
            for v in &out {
                assert!(v.is_finite(), "output value {} is not finite", v);
            }
        });
    }

    #[test]
    fn test_predictor_step_list_dimension_mismatch() {
        Python::initialize();
        Python::attach(|py| {
            let cfg = PyKizzasiConfig::new(3, 3, 32, 1, 4, 256, "mamba2".to_string());
            let mut pred = PyPredictor::new(&cfg).expect("predictor");
            let result = pred.step_list(py, vec![0.1, 0.2]);
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_version_not_empty() {
        let version = env!("CARGO_PKG_VERSION");
        assert!(!version.is_empty());
    }

    #[test]
    fn test_constraint_spec_valid() {
        let spec = PyConstraintSpec::new(
            "joint_limit".to_string(),
            Some(-std::f32::consts::PI),
            Some(std::f32::consts::PI),
            None,
            false,
        );
        assert!(spec.is_ok());
        let s = spec.expect("spec");
        assert_eq!(s.name, "joint_limit");
        assert!((s.min_val.expect("min") + std::f32::consts::PI).abs() < 1e-6);
        assert!((s.max_val.expect("max") - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn test_constraint_spec_requires_at_least_one_bound() {
        let spec = PyConstraintSpec::new("bad".to_string(), None, None, None, false);
        assert!(spec.is_err());
    }

    #[test]
    fn test_predictor_repr() {
        let cfg = PyKizzasiConfig::new(4, 2, 64, 2, 8, 512, "mamba".to_string());
        let pred = PyPredictor::new(&cfg).expect("predictor");
        let r = pred.__repr__();
        assert!(r.contains("Predictor("));
        assert!(r.contains("input_dim=4"));
        assert!(r.contains("mamba"));
    }

    #[test]
    fn test_predictor_guardrail_round_trip() {
        let cfg = PyKizzasiConfig::new(2, 2, 32, 1, 4, 256, "mamba2".to_string());
        let mut pred = PyPredictor::new(&cfg).expect("predictor");
        assert!(!pred.has_guardrails());

        let spec = PyConstraintSpec::new("bounds".to_string(), Some(-1.0), Some(1.0), None, false)
            .expect("spec");
        pred.set_guardrails(vec![spec]).expect("set guardrails");
        assert!(pred.has_guardrails());

        pred.clear_guardrails();
        assert!(!pred.has_guardrails());
    }

    /// `Kizzasi` must stay `Send` for `Python::detach` (used by `step` /
    /// `predict_n` / `step_list`) to compile at all, and for `unsendable`
    /// to be safely removable from `#[pyclass(name = "Predictor", ...)]`.
    /// If a future change to `kizzasi::Kizzasi` (or a field it embeds) makes
    /// it `!Send`, this fails with a clear, localized error instead of an
    /// opaque one from the pyo3 macro expansion.
    #[test]
    fn test_kizzasi_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Kizzasi>();
    }

    // Regression for the critical bug where `build_guardrails` overwrote
    // `min_val` with `max_val` (both went through the same single-bound
    // `ConstraintBuilder::bound` field), so a two-sided
    // `ConstraintSpec(min_val=..., max_val=...)` only ever enforced the
    // upper bound. README.md's own robotics example
    // (`ConstraintSpec("joint_angle", min_val=-3.14, max_val=3.14)`) is
    // exactly this shape.
    #[test]
    fn test_guardrail_two_sided_bound_clamps_below_min() {
        let guardrails = build_guardrails(&[PyConstraintSpec::new(
            "joint_angle".to_string(),
            Some(-1.0),
            Some(1.0),
            None,
            false,
        )
        .expect("spec")])
        .expect("guardrails");
        use kizzasi_logic::ConstrainedInference;
        let below_min = scirs2_core::ndarray::Array1::from_vec(vec![-5.0_f32]);
        let clamped = guardrails.constrain(&below_min).expect("constrain");
        assert!(
            (clamped[0] - (-1.0)).abs() < 1e-6,
            "expected lower bound -1.0 to be enforced, got {}",
            clamped[0]
        );
        let above_max = scirs2_core::ndarray::Array1::from_vec(vec![5.0_f32]);
        let clamped_hi = guardrails.constrain(&above_max).expect("constrain");
        assert!(
            (clamped_hi[0] - 1.0).abs() < 1e-6,
            "expected upper bound 1.0 to still be enforced, got {}",
            clamped_hi[0]
        );
    }

    #[test]
    fn test_guardrail_rejects_nan_bound() {
        let spec = PyConstraintSpec::new("x".to_string(), Some(f32::NAN), None, None, false)
            .expect("spec construction itself does not validate finiteness");
        let result = build_guardrails(&[spec]);
        assert!(result.is_err(), "NaN min_val must be rejected, not panic");
    }

    #[test]
    fn test_guardrail_rejects_inverted_bounds() {
        // min_val > max_val would panic inside f32::clamp if routed to
        // in_range unchecked.
        let spec = PyConstraintSpec::new("x".to_string(), Some(5.0), Some(-5.0), None, false)
            .expect("spec construction itself does not validate ordering");
        let result = build_guardrails(&[spec]);
        assert!(
            result.is_err(),
            "min_val > max_val must be rejected, not panic"
        );
    }

    // Regression for the medium bug where a dimensional `ConstraintSpec`
    // with an out-of-range `dimension` was silently accepted by
    // `set_guardrails` and then silently skipped by
    // `GuardrailSet::constrain` (`if *dim < result.len()`), so
    // `has_guardrails()` reported `True` while nothing was enforced.
    #[test]
    fn test_set_guardrails_rejects_out_of_range_dimension() {
        let cfg = PyKizzasiConfig::new(4, 4, 16, 1, 4, 64, "mamba2".to_string());
        let mut pred = PyPredictor::new(&cfg).expect("predictor");
        let spec = PyConstraintSpec::new(
            "torque".to_string(),
            Some(-100.0),
            Some(100.0),
            Some(7), // output_dim is 4, so valid dims are 0..4
            true,
        )
        .expect("spec");
        let result = pred.set_guardrails(vec![spec]);
        assert!(result.is_err(), "out-of-range dimension must be rejected");
        assert!(!pred.has_guardrails());
    }

    #[test]
    fn test_predict_n_crosses_pyo3_boundary() {
        use scirs2_numpy::{PyArray1, PyArrayMethods};
        Python::initialize();
        Python::attach(|py| {
            let cfg = PyKizzasiConfig::new(2, 2, 16, 1, 4, 64, "mamba2".to_string());
            let mut pred = PyPredictor::new(&cfg).expect("predictor");
            let py_arr = PyArray1::from_vec(py, vec![0.1_f32, 0.2]);
            let readonly = py_arr.readonly();
            let out = pred
                .predict_n(py, readonly, 5)
                .expect("predict_n via pymethod");
            let out_ro = out.readonly();
            let view = out_ro.as_array();
            assert_eq!(view.shape(), &[5, 2]);
            for v in view.iter() {
                assert!(v.is_finite());
            }
        });
    }

    #[test]
    fn test_step_list_crosses_pyo3_boundary() {
        Python::initialize();
        Python::attach(|py| {
            let cfg = PyKizzasiConfig::new(3, 3, 16, 1, 4, 64, "mamba2".to_string());
            let mut pred = PyPredictor::new(&cfg).expect("predictor");
            let out = pred
                .step_list(py, vec![0.1, 0.2, 0.3])
                .expect("step_list via pymethod");
            assert_eq!(out.len(), 3);
            for v in &out {
                assert!(v.is_finite());
            }
        });
    }
}
