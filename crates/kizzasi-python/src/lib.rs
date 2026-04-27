//! # kizzasi-python
//!
//! PyO3-based Python bindings for the Kizzasi AGSP (Autoregressive General-Purpose
//! Signal Predictor) ecosystem.
//!
//! ## Usage from Python
//!
//! ```python
//! import kizzasi
//!
//! # Create a predictor with the audio preset
//! config = kizzasi.Config.audio(44100)
//! predictor = kizzasi.Predictor(config)
//!
//! # Single-step prediction
//! import numpy as np
//! inp = np.array([0.5], dtype=np.float32)
//! out = predictor.step(inp)
//!
//! # Multi-step prediction (autoregressive)
//! steps = predictor.predict_n(inp, n_steps=10)
//!
//! # Reset internal state
//! predictor.reset()
//! ```

#![deny(warnings)]
#![deny(clippy::all)]

use pyo3::prelude::*;
use scirs2_numpy::{PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};

use ::kizzasi::Kizzasi;
use kizzasi_core::{KizzasiConfig, ModelType};
use scirs2_core::ndarray::Array1;

// ============================================================================
// Error conversion helper
// ============================================================================

/// Convert any Display-able error into a Python RuntimeError.
fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

// ============================================================================
// PyModelType
// ============================================================================

/// Selector for the underlying SSM architecture.
#[pyclass(name = "ModelType", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyModelType {
    inner: ModelType,
}

impl PyModelType {
    fn from_inner(inner: ModelType) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyModelType {
    #[classattr]
    #[allow(non_snake_case)]
    fn MAMBA() -> Self {
        Self::from_inner(ModelType::Mamba)
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn MAMBA2() -> Self {
        Self::from_inner(ModelType::Mamba2)
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn S4() -> Self {
        Self::from_inner(ModelType::S4)
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn RWKV() -> Self {
        Self::from_inner(ModelType::Rwkv)
    }

    fn __repr__(&self) -> String {
        match self.inner {
            ModelType::Mamba => "ModelType.MAMBA".to_string(),
            ModelType::Mamba2 => "ModelType.MAMBA2".to_string(),
            ModelType::S4 => "ModelType.S4".to_string(),
            ModelType::Rwkv => "ModelType.RWKV".to_string(),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// PyKizzasiConfig
// ============================================================================

/// Configuration for a Kizzasi AGSP predictor.
#[pyclass(name = "Config", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyKizzasiConfig {
    #[pyo3(get, set)]
    pub input_dim: usize,
    #[pyo3(get, set)]
    pub output_dim: usize,
    #[pyo3(get, set)]
    pub hidden_dim: usize,
    #[pyo3(get, set)]
    pub num_layers: usize,
    #[pyo3(get, set)]
    pub state_dim: usize,
    #[pyo3(get, set)]
    pub context_window: usize,
    /// SSM architecture name: "mamba", "mamba2", "s4", "rwkv".
    #[pyo3(get, set)]
    pub model_type: String,
}

impl PyKizzasiConfig {
    fn parse_model_type(s: &str) -> Result<ModelType, PyErr> {
        match s.to_lowercase().as_str() {
            "mamba" => Ok(ModelType::Mamba),
            "mamba2" => Ok(ModelType::Mamba2),
            "s4" => Ok(ModelType::S4),
            "rwkv" => Ok(ModelType::Rwkv),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown model_type '{}'. Valid options: mamba, mamba2, s4, rwkv",
                other
            ))),
        }
    }

    fn to_core_config(&self) -> Result<KizzasiConfig, PyErr> {
        let model_type = Self::parse_model_type(&self.model_type)?;
        Ok(KizzasiConfig::new()
            .model_type(model_type)
            .input_dim(self.input_dim)
            .output_dim(self.output_dim)
            .hidden_dim(self.hidden_dim)
            .num_layers(self.num_layers)
            .state_dim(self.state_dim)
            .context_window(self.context_window))
    }
}

#[pymethods]
impl PyKizzasiConfig {
    #[new]
    #[pyo3(signature = (
        input_dim,
        output_dim,
        hidden_dim = 256,
        num_layers = 4,
        state_dim = 16,
        context_window = 8192,
        model_type = "mamba2".to_string()
    ))]
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        state_dim: usize,
        context_window: usize,
        model_type: String,
    ) -> Self {
        Self {
            input_dim,
            output_dim,
            hidden_dim,
            num_layers,
            state_dim,
            context_window,
            model_type,
        }
    }

    /// Audio signal prediction preset.
    #[staticmethod]
    #[pyo3(signature = (sample_rate = 44100))]
    pub fn audio(sample_rate: u32) -> Self {
        let _ = sample_rate;
        Self {
            input_dim: 1,
            output_dim: 1,
            hidden_dim: 256,
            num_layers: 4,
            state_dim: 16,
            context_window: 8192,
            model_type: "mamba2".to_string(),
        }
    }

    /// Robotics / control-loop preset.
    #[staticmethod]
    pub fn robotics(state_dim: usize, action_dim: usize) -> Self {
        Self {
            input_dim: state_dim,
            output_dim: action_dim,
            hidden_dim: 128,
            num_layers: 3,
            state_dim: 8,
            context_window: 1024,
            model_type: "mamba2".to_string(),
        }
    }

    /// Multi-sensor fusion preset.
    #[staticmethod]
    pub fn sensor(num_sensors: usize) -> Self {
        Self {
            input_dim: num_sensors,
            output_dim: num_sensors,
            hidden_dim: 64,
            num_layers: 2,
            state_dim: 8,
            context_window: 2048,
            model_type: "mamba2".to_string(),
        }
    }

    /// Lightweight embedded / edge preset.
    #[staticmethod]
    pub fn lightweight(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            hidden_dim: 32,
            num_layers: 1,
            state_dim: 4,
            context_window: 512,
            model_type: "mamba".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Config(input_dim={}, output_dim={}, hidden_dim={}, num_layers={}, \
             state_dim={}, context_window={}, model_type='{}')",
            self.input_dim,
            self.output_dim,
            self.hidden_dim,
            self.num_layers,
            self.state_dim,
            self.context_window,
            self.model_type
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// PyConstraintSpec
// ============================================================================

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

// ============================================================================
// build_guardrails helper
// ============================================================================

fn build_guardrails(specs: &[PyConstraintSpec]) -> PyResult<kizzasi_logic::GuardrailSet> {
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

// ============================================================================
// PyPredictor
// ============================================================================

/// The main Kizzasi AGSP predictor.
///
/// Predictor instances are NOT thread-safe (unsendable). Use separate instances
/// per thread or protect shared access with a lock in Python.
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

// ============================================================================
// Module registration
// ============================================================================

#[pymodule]
#[pyo3(name = "kizzasi")]
fn kizzasi_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKizzasiConfig>()?;
    m.add_class::<PyPredictor>()?;
    m.add_class::<PyConstraintSpec>()?;
    m.add_class::<PyModelType>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "Kizzasi AGSP — PyO3 Python bindings")?;
    Ok(())
}

// ============================================================================
// Tests (pure Rust — no Python interpreter required)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation_defaults() {
        let cfg = PyKizzasiConfig::new(4, 4, 256, 4, 16, 8192, "mamba2".to_string());
        assert_eq!(cfg.input_dim, 4);
        assert_eq!(cfg.output_dim, 4);
        assert_eq!(cfg.hidden_dim, 256);
        assert_eq!(cfg.num_layers, 4);
        assert_eq!(cfg.state_dim, 16);
        assert_eq!(cfg.context_window, 8192);
        assert_eq!(cfg.model_type, "mamba2");
    }

    #[test]
    fn test_audio_preset_dims() {
        let cfg = PyKizzasiConfig::audio(44100);
        assert_eq!(cfg.input_dim, 1);
        assert_eq!(cfg.output_dim, 1);
        assert_eq!(cfg.hidden_dim, 256);
        assert_eq!(cfg.num_layers, 4);
    }

    #[test]
    fn test_robotics_preset_dims() {
        let cfg = PyKizzasiConfig::robotics(6, 4);
        assert_eq!(cfg.input_dim, 6);
        assert_eq!(cfg.output_dim, 4);
        assert_eq!(cfg.hidden_dim, 128);
        assert_eq!(cfg.num_layers, 3);
    }

    #[test]
    fn test_sensor_preset_dims() {
        let cfg = PyKizzasiConfig::sensor(10);
        assert_eq!(cfg.input_dim, 10);
        assert_eq!(cfg.output_dim, 10);
        assert_eq!(cfg.hidden_dim, 64);
        assert_eq!(cfg.num_layers, 2);
    }

    #[test]
    fn test_lightweight_preset_dims() {
        let cfg = PyKizzasiConfig::lightweight(2, 3);
        assert_eq!(cfg.input_dim, 2);
        assert_eq!(cfg.output_dim, 3);
        assert_eq!(cfg.hidden_dim, 32);
        assert_eq!(cfg.num_layers, 1);
        assert_eq!(cfg.model_type, "mamba");
    }

    #[test]
    fn test_config_repr() {
        let cfg = PyKizzasiConfig::audio(44100);
        let r = cfg.__repr__();
        assert!(r.contains("Config("));
        assert!(r.contains("input_dim=1"));
        assert!(r.contains("mamba2"));
    }

    #[test]
    fn test_core_config_conversion_valid() {
        let cfg = PyKizzasiConfig::new(3, 3, 64, 2, 8, 512, "mamba2".to_string());
        let result = cfg.to_core_config();
        assert!(result.is_ok(), "Expected Ok, got {:?}", result.err());
        let core = result.expect("core config");
        assert_eq!(core.get_input_dim(), 3);
        assert_eq!(core.get_output_dim(), 3);
    }

    #[test]
    fn test_core_config_conversion_invalid_model_type() {
        let cfg = PyKizzasiConfig::new(1, 1, 32, 1, 4, 512, "transformer".to_string());
        assert!(cfg.to_core_config().is_err());
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
}
