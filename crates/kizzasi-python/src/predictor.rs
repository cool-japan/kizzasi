//! Single-predictor Python wrapper.
//!
//! Exposes:
//! - [`PyConstraintSpec`] — scalar value-range constraint description used by
//!   the predictor's guardrails.
//! - [`PyPredictor`] — wraps a single [`kizzasi::Kizzasi`] instance and offers
//!   `step`, `predict_n`, `step_list`, `reset`, and guardrail management.

use pyo3::prelude::*;
use scirs2_numpy::{PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};

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
pub(crate) fn build_guardrails(
    specs: &[PyConstraintSpec],
) -> PyResult<kizzasi_logic::GuardrailSet> {
    use kizzasi_logic::{ConstraintBuilder, Guardrail, GuardrailSet};

    let mut guardrail_set = GuardrailSet::new();
    for spec in specs {
        let mut builder = ConstraintBuilder::new().name(&spec.name);
        if let Some(lo) = spec.min_val {
            builder = builder.greater_eq(lo);
        }
        if let Some(hi) = spec.max_val {
            builder = builder.less_eq(hi);
        }
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
/// Predictor instances are NOT thread-safe (unsendable). Use separate
/// instances per thread or protect shared access with a lock in Python.
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
        let output = self.inner.step(&input_arr).map_err(to_py_err)?;
        Ok(output.to_pyarray(py))
    }

    /// N-step autoregressive prediction.
    ///
    /// Parameters: input ndarray (input_dim,) float32, n_steps int
    /// Returns: ndarray (n_steps, output_dim) float32
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
        let predictions = self
            .inner
            .predict_n(&input_arr, n_steps)
            .map_err(to_py_err)?;
        Ok(predictions.to_pyarray(py))
    }

    /// Single prediction step accepting a Python list.
    pub fn step_list(&mut self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        let expected = self.inner.input_dim();
        let actual = input.len();
        if actual != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Input length {} does not match predictor input_dim {}",
                actual, expected
            )));
        }
        let input_arr = Array1::from_vec(input);
        let output = self.inner.step(&input_arr).map_err(to_py_err)?;
        Ok(output.to_vec())
    }

    /// Reset the predictor's internal hidden state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Apply value-range guardrails to clip / reject out-of-bound predictions.
    pub fn set_guardrails(&mut self, specs: Vec<PyConstraintSpec>) -> PyResult<()> {
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

    #[test]
    fn test_predictor_step_list() {
        let cfg = PyKizzasiConfig::new(3, 3, 32, 1, 4, 256, "mamba2".to_string());
        let mut pred = PyPredictor::new(&cfg).expect("predictor");
        let output = pred.step_list(vec![0.1, 0.2, 0.3]);
        assert!(output.is_ok(), "step_list failed: {:?}", output.err());
        let out = output.expect("output");
        assert_eq!(out.len(), 3);
        for v in &out {
            assert!(v.is_finite(), "output value {} is not finite", v);
        }
    }

    #[test]
    fn test_predictor_step_list_dimension_mismatch() {
        let cfg = PyKizzasiConfig::new(3, 3, 32, 1, 4, 256, "mamba2".to_string());
        let mut pred = PyPredictor::new(&cfg).expect("predictor");
        let result = pred.step_list(vec![0.1, 0.2]);
        assert!(result.is_err());
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
}
