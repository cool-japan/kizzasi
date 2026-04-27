//! Model registry for loading and managing different architectures
//!
//! This module provides a centralized registry for loading various model
//! architectures supported by kizzasi-model:
//! - Mamba/Mamba2: Selective State Space Models
//! - RWKV: Linear attention models
//! - S4/S4D: Structured State Space Models
//! - Transformer: Standard attention models

use crate::error::{InferenceError, InferenceResult};
use kizzasi_model::{AutoregressiveModel, ModelType};
use std::path::Path;

/// Configuration for model loading
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    /// Type of model architecture
    pub model_type: ModelType,
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// State dimension (for SSMs)
    pub state_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Optional path to pretrained weights
    pub weights_path: Option<String>,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(model_type: ModelType) -> Self {
        Self {
            model_type,
            input_dim: 1,
            hidden_dim: 256,
            num_layers: 4,
            state_dim: 16,
            output_dim: 1,
            weights_path: None,
        }
    }

    /// Set input dimension
    pub fn input_dim(mut self, dim: usize) -> Self {
        self.input_dim = dim;
        self
    }

    /// Set hidden dimension
    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self
    }

    /// Set number of layers
    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Set state dimension
    pub fn state_dim(mut self, dim: usize) -> Self {
        self.state_dim = dim;
        self
    }

    /// Set output dimension
    pub fn output_dim(mut self, dim: usize) -> Self {
        self.output_dim = dim;
        self
    }

    /// Set weights path
    pub fn weights_path(mut self, path: impl Into<String>) -> Self {
        self.weights_path = Some(path.into());
        self
    }
}

/// Model registry for creating and managing model instances
pub struct ModelRegistry {
    /// Available model configurations
    configs: std::collections::HashMap<String, ModelConfig>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            configs: std::collections::HashMap::new(),
        }
    }

    /// Register a model configuration with a name
    pub fn register(&mut self, name: impl Into<String>, config: ModelConfig) {
        self.configs.insert(name.into(), config);
    }

    /// Get a registered configuration
    pub fn get_config(&self, name: &str) -> Option<&ModelConfig> {
        self.configs.get(name)
    }

    /// List all registered model names
    pub fn list_models(&self) -> Vec<String> {
        self.configs.keys().cloned().collect()
    }

    /// Create a model wrapper from a configuration
    pub fn create_model(&self, name: &str) -> InferenceResult<Box<dyn AutoregressiveModel>> {
        let config = self
            .get_config(name)
            .ok_or_else(|| InferenceError::PipelineConfig(format!("Model '{}' not found", name)))?;

        self.create_from_config(config)
    }

    /// Create a model from configuration
    fn create_from_config(
        &self,
        config: &ModelConfig,
    ) -> InferenceResult<Box<dyn AutoregressiveModel>> {
        match config.model_type {
            ModelType::Mamba2 => {
                #[cfg(feature = "mamba")]
                {
                    use kizzasi_model::mamba2::{Mamba2, Mamba2Config};
                    let num_heads = (config.hidden_dim / 64).max(1);
                    let model_config = Mamba2Config {
                        input_dim: config.input_dim,
                        hidden_dim: config.hidden_dim,
                        state_dim: config.state_dim,
                        num_heads,
                        head_dim: config.hidden_dim / num_heads,
                        expand_factor: 2,
                        conv_kernel_size: 4,
                        num_layers: config.num_layers,
                        dropout: 0.0,
                        use_rms_norm: true,
                        chunk_size: 256,
                    };
                    let model = Mamba2::new(model_config).map_err(InferenceError::ModelError)?;
                    Ok(Box::new(model))
                }
                #[cfg(not(feature = "mamba"))]
                Err(InferenceError::PipelineConfig(
                    "Mamba2 requires the 'mamba' feature to be enabled".into(),
                ))
            }
            ModelType::Rwkv => {
                use kizzasi_model::rwkv::{Rwkv, RwkvConfig};
                let num_heads = (config.hidden_dim / 64).max(1);
                let model_config = RwkvConfig {
                    input_dim: config.input_dim,
                    hidden_dim: config.hidden_dim,
                    intermediate_dim: config.hidden_dim * 4,
                    num_layers: config.num_layers,
                    num_heads,
                    head_dim: config.hidden_dim / num_heads,
                    dropout: 0.0,
                    time_decay_init: -5.0,
                    use_rms_norm: true,
                };
                let model = Rwkv::new(model_config).map_err(InferenceError::ModelError)?;
                Ok(Box::new(model))
            }
            ModelType::S4 | ModelType::S4D => {
                use kizzasi_model::s4::{S4Config, S4D};
                let model_config = S4Config {
                    input_dim: config.input_dim,
                    hidden_dim: config.hidden_dim,
                    state_dim: config.state_dim,
                    num_layers: config.num_layers,
                    dropout: 0.0,
                    dt_min: 0.001,
                    dt_max: 0.1,
                    use_diagonal: config.model_type == ModelType::S4D,
                    use_rms_norm: true,
                };
                let model = S4D::new(model_config).map_err(InferenceError::ModelError)?;
                Ok(Box::new(model))
            }
            ModelType::Transformer => {
                use kizzasi_model::transformer::{Transformer, TransformerConfig};
                let num_heads = (config.hidden_dim / 64).max(1);
                let model_config = TransformerConfig {
                    input_dim: config.input_dim,
                    hidden_dim: config.hidden_dim,
                    num_heads,
                    head_dim: config.hidden_dim / num_heads,
                    ff_dim: config.hidden_dim * 4,
                    num_layers: config.num_layers,
                    max_seq_len: 8192,
                    dropout: 0.1,
                    use_rms_norm: true,
                    causal: true,
                };
                let model = Transformer::new(model_config).map_err(InferenceError::ModelError)?;
                Ok(Box::new(model))
            }
            ModelType::Mamba => {
                #[cfg(feature = "mamba")]
                {
                    use kizzasi_model::mamba::{Mamba, MambaConfig};
                    let model_config = MambaConfig {
                        input_dim: config.input_dim,
                        hidden_dim: config.hidden_dim,
                        state_dim: config.state_dim,
                        expand_factor: 2,
                        conv_kernel_size: 4,
                        num_layers: config.num_layers,
                        dropout: 0.0,
                        use_mamba2: false,
                    };
                    let model = Mamba::new(model_config).map_err(InferenceError::ModelError)?;
                    Ok(Box::new(model))
                }
                #[cfg(not(feature = "mamba"))]
                Err(InferenceError::PipelineConfig(
                    "Mamba requires the 'mamba' feature to be enabled".into(),
                ))
            }
            ModelType::Rwkv5 => {
                use kizzasi_model::rwkv5::{Rwkv5Config, Rwkv5Model};
                let num_heads = (config.hidden_dim / 64).max(1);
                let model_config = Rwkv5Config {
                    input_dim: config.input_dim,
                    hidden_dim: config.hidden_dim,
                    num_layers: config.num_layers,
                    num_heads,
                    head_dim: config.hidden_dim / num_heads,
                    intermediate_dim: config.hidden_dim * 4,
                    context_length: 8192,
                    use_rms_norm: true,
                };
                let model = Rwkv5Model::new(model_config).map_err(InferenceError::ModelError)?;
                Ok(Box::new(model))
            }
            ModelType::NeuralOde => Err(InferenceError::PipelineConfig(
                "NeuralOde not yet supported in registry - use NeuralOdeModel directly".into(),
            )),
            ModelType::MultiModal => Err(InferenceError::PipelineConfig(
                "MultiModal not yet supported in registry - use MultiModalModel directly".into(),
            )),
            ModelType::Snn => {
                use kizzasi_model::spiking::{SpikingConfig, SpikingNeuralNetwork};
                let model_config = SpikingConfig::new(
                    config.input_dim,
                    config.hidden_dim,
                    config.output_dim,
                    config.num_layers,
                );
                let model =
                    SpikingNeuralNetwork::new(model_config).map_err(InferenceError::ModelError)?;
                Ok(Box::new(model))
            }
            ModelType::MultiScale => {
                use kizzasi_model::temporal_multiscale::{
                    MultiScaleConfig, MultiScaleModel, ScaleFusion,
                };
                // Build default scale factors: [1, 2, 4, ...] up to num_layers
                let scale_factors: Vec<usize> =
                    (0..config.num_layers).map(|i| 1_usize << i).collect();
                let model_config = MultiScaleConfig {
                    input_dim: config.input_dim,
                    hidden_dim: config.hidden_dim,
                    output_dim: config.output_dim,
                    num_scales: config.num_layers,
                    scale_factors,
                    fusion: ScaleFusion::Weighted,
                    context_length: 2048,
                };
                let model =
                    MultiScaleModel::new(model_config).map_err(InferenceError::ModelError)?;
                Ok(Box::new(model))
            }
        }
    }

    /// Load weights from a JSON file into a model.
    ///
    /// The file must contain a JSON object of the form `{ "key": [f32, ...], ... }`,
    /// which is the format produced by each model's `save_weights_json` method.
    ///
    /// This delegates to the model's `AutoregressiveModel::load_weights_json` override.
    /// Models that do not implement the override will return an appropriate error.
    pub fn load_weights(
        &self,
        model: &mut dyn AutoregressiveModel,
        path: impl AsRef<Path>,
    ) -> InferenceResult<()> {
        let path = path.as_ref();

        // Validate the file exists and is readable before passing to model.
        let file = std::fs::File::open(path).map_err(InferenceError::IoError)?;

        // Deserialise to verify the JSON is well-formed; the model method
        // will re-read and apply the data.
        let _weights_check: std::collections::HashMap<String, Vec<f32>> =
            serde_json::from_reader(file).map_err(|e| {
                InferenceError::SerializationError(format!(
                    "Failed to parse weight file '{}': {}",
                    path.display(),
                    e
                ))
            })?;

        // Delegate actual loading to the model.
        model
            .load_weights_json(path)
            .map_err(InferenceError::ModelError)?;

        tracing::info!(
            "Weights loaded from '{}' into {} model",
            path.display(),
            model.model_type()
        );

        Ok(())
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for common model configurations
pub struct ModelBuilder {
    config: ModelConfig,
}

impl ModelBuilder {
    /// Start building a Mamba model
    pub fn mamba() -> Self {
        Self {
            config: ModelConfig::new(ModelType::Mamba),
        }
    }

    /// Start building a Mamba2 model
    pub fn mamba2() -> Self {
        Self {
            config: ModelConfig::new(ModelType::Mamba2),
        }
    }

    /// Start building an RWKV model
    pub fn rwkv() -> Self {
        Self {
            config: ModelConfig::new(ModelType::Rwkv),
        }
    }

    /// Start building an S4 model
    pub fn s4() -> Self {
        Self {
            config: ModelConfig::new(ModelType::S4),
        }
    }

    /// Start building an S4D model
    pub fn s4d() -> Self {
        Self {
            config: ModelConfig::new(ModelType::S4D),
        }
    }

    /// Start building a Transformer model
    pub fn transformer() -> Self {
        Self {
            config: ModelConfig::new(ModelType::Transformer),
        }
    }

    /// Set dimensions
    pub fn dims(mut self, input: usize, hidden: usize, output: usize) -> Self {
        self.config.input_dim = input;
        self.config.hidden_dim = hidden;
        self.config.output_dim = output;
        self
    }

    /// Set number of layers
    pub fn layers(mut self, n: usize) -> Self {
        self.config.num_layers = n;
        self
    }

    /// Set state dimension
    pub fn state_dim(mut self, dim: usize) -> Self {
        self.config.state_dim = dim;
        self
    }

    /// Build the configuration
    pub fn build(self) -> ModelConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::new(ModelType::Mamba2)
            .input_dim(3)
            .hidden_dim(128)
            .num_layers(2)
            .output_dim(3);

        assert_eq!(config.model_type, ModelType::Mamba2);
        assert_eq!(config.input_dim, 3);
        assert_eq!(config.hidden_dim, 128);
        assert_eq!(config.num_layers, 2);
    }

    #[test]
    fn test_registry_register() {
        let mut registry = ModelRegistry::new();
        let config = ModelConfig::new(ModelType::Rwkv);

        registry.register("test_model", config);

        assert!(registry.get_config("test_model").is_some());
        assert_eq!(registry.list_models().len(), 1);
    }

    #[test]
    fn test_model_builder() {
        let config = ModelBuilder::s4d()
            .dims(1, 256, 1)
            .layers(4)
            .state_dim(16)
            .build();

        assert_eq!(config.model_type, ModelType::S4D);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.num_layers, 4);
    }

    #[test]
    fn test_create_rwkv_model() {
        let mut registry = ModelRegistry::new();
        let config = ModelBuilder::rwkv().dims(1, 64, 10).layers(2).build();

        registry.register("rwkv_test", config);

        let result = registry.create_model("rwkv_test");
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.model_type(), ModelType::Rwkv);
        assert_eq!(model.hidden_dim(), 64);
    }

    #[test]
    fn test_create_s4_model() {
        let mut registry = ModelRegistry::new();
        let config = ModelBuilder::s4d().dims(1, 128, 10).layers(3).build();

        registry.register("s4_test", config);

        let result = registry.create_model("s4_test");
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.model_type(), ModelType::S4D);
    }

    #[test]
    fn test_create_transformer_model() {
        let mut registry = ModelRegistry::new();
        let config = ModelBuilder::transformer()
            .dims(1, 128, 10)
            .layers(2)
            .build();

        registry.register("transformer_test", config);

        let result = registry.create_model("transformer_test");
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.model_type(), ModelType::Transformer);
    }

    #[cfg(feature = "mamba")]
    #[test]
    fn test_mamba_registry_create() {
        let mut registry = ModelRegistry::new();
        // Use hidden_dim=64 (divisible by 1 or 2 heads, small for speed)
        let config = ModelBuilder::mamba()
            .dims(1, 64, 1)
            .layers(2)
            .state_dim(8)
            .build();

        registry.register("mamba_test", config);

        let result = registry.create_model("mamba_test");
        assert!(
            result.is_ok(),
            "Mamba registry creation failed: {:?}",
            result.err()
        );

        let model = result.expect("model should be created");
        assert_eq!(model.model_type(), ModelType::Mamba);
        assert_eq!(model.hidden_dim(), 64);
    }

    #[cfg(feature = "mamba")]
    #[test]
    fn test_mamba2_registry_create() {
        let mut registry = ModelRegistry::new();
        // hidden_dim=64 with num_heads = (64/64).max(1) = 1
        let config = ModelBuilder::mamba2()
            .dims(1, 64, 1)
            .layers(2)
            .state_dim(8)
            .build();

        registry.register("mamba2_test", config);

        let result = registry.create_model("mamba2_test");
        assert!(
            result.is_ok(),
            "Mamba2 registry creation failed: {:?}",
            result.err()
        );

        let model = result.expect("model should be created");
        assert_eq!(model.model_type(), ModelType::Mamba2);
        assert_eq!(model.hidden_dim(), 64);
    }

    // -----------------------------------------------------------------
    // WS-B: InferenceRegistry::load_weights tests
    // -----------------------------------------------------------------

    /// Test that load_weights correctly reads a JSON weight file and calls the
    /// model's load_weights_json method.
    #[test]
    fn test_registry_load_weights_from_file() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static REGISTRY_LOAD_COUNTER: AtomicU64 = AtomicU64::new(0);
        let uid = REGISTRY_LOAD_COUNTER.fetch_add(1, Ordering::Relaxed);

        let registry = ModelRegistry::new();

        // Build a small Transformer (always available, no feature gate needed)
        let config = ModelBuilder::transformer().dims(1, 64, 1).layers(2).build();

        let mut model = registry.create_from_config(&config).expect("create model");

        // First save model weights to a temp file, then call load_weights.
        let mut save_path = std::env::temp_dir();
        save_path.push(format!("kizzasi_registry_load_weights_test_{}.json", uid));

        model
            .save_weights_json(&save_path)
            .expect("save_weights_json via trait");

        let result = registry.load_weights(model.as_mut(), &save_path);
        let _ = std::fs::remove_file(&save_path);

        assert!(
            result.is_ok(),
            "load_weights should succeed: {:?}",
            result.err()
        );
    }

    /// Test that load_weights returns an error for a non-existent file.
    #[test]
    fn test_registry_load_weights_missing_file() {
        let registry = ModelRegistry::new();

        let config = ModelBuilder::transformer().dims(1, 64, 1).layers(1).build();

        let mut model = registry.create_from_config(&config).expect("create model");

        let missing = std::path::Path::new("/tmp/__kizzasi_nonexistent_weight_file__.json");
        let result = registry.load_weights(model.as_mut(), missing);
        assert!(result.is_err(), "missing file should produce error");
    }

    /// Test that load_weights returns an error for malformed JSON.
    #[test]
    fn test_registry_load_weights_bad_json() {
        let registry = ModelRegistry::new();

        let config = ModelBuilder::transformer().dims(1, 64, 1).layers(1).build();

        let mut model = registry.create_from_config(&config).expect("create model");

        use std::sync::atomic::{AtomicU64, Ordering};
        static REGISTRY_BAD_JSON_COUNTER: AtomicU64 = AtomicU64::new(0);
        let bad_uid = REGISTRY_BAD_JSON_COUNTER.fetch_add(1, Ordering::Relaxed);

        let mut bad_path = std::env::temp_dir();
        bad_path.push(format!("kizzasi_registry_bad_json_test_{}.json", bad_uid));

        // Write invalid JSON
        std::fs::write(&bad_path, b"not valid json").expect("write bad json");
        let result = registry.load_weights(model.as_mut(), &bad_path);
        let _ = std::fs::remove_file(&bad_path);

        assert!(result.is_err(), "bad JSON should produce error");
    }
}
