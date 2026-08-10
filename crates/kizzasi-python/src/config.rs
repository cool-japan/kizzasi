//! Configuration and model-type Python wrappers.
//!
//! This module exposes:
//! - [`PyModelType`] — selector for the underlying SSM architecture.
//! - [`PyKizzasiConfig`] — configuration object with builder-style presets
//!   (`audio`, `robotics`, `sensor`, `lightweight`).
//!
//! These are pure data types — they hold no Rust handles and are cheap to
//! clone, which lets them be shared across multiple predictors (ensembles,
//! optimized predictors, …).

use pyo3::prelude::*;

use kizzasi_core::{KizzasiConfig, ModelType};

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
    /// Parse the textual model-type identifier. Case-insensitive.
    pub(crate) fn parse_model_type(s: &str) -> Result<ModelType, PyErr> {
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

    /// Convert this Python config into the internal [`KizzasiConfig`].
    pub(crate) fn to_core_config(&self) -> Result<KizzasiConfig, PyErr> {
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
    fn test_model_type_repr_round_trip() {
        // Each classattr returns a PyModelType; check repr matches.
        assert_eq!(PyModelType::MAMBA().__repr__(), "ModelType.MAMBA");
        assert_eq!(PyModelType::MAMBA2().__repr__(), "ModelType.MAMBA2");
        assert_eq!(PyModelType::S4().__repr__(), "ModelType.S4");
        assert_eq!(PyModelType::RWKV().__repr__(), "ModelType.RWKV");
    }

    #[test]
    fn test_parse_model_type_case_insensitive() {
        assert_eq!(
            PyKizzasiConfig::parse_model_type("MAMBA").expect("parse"),
            ModelType::Mamba
        );
        assert_eq!(
            PyKizzasiConfig::parse_model_type("Mamba2").expect("parse"),
            ModelType::Mamba2
        );
        assert!(PyKizzasiConfig::parse_model_type("nonexistent").is_err());
    }
}
