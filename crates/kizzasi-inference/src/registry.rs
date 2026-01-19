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
                // TODO: Fix Mamba2 model exports in kizzasi-model
                Err(InferenceError::PipelineConfig(
                    "Mamba2 not yet supported - models not exported".into(),
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
                // TODO: Fix Mamba model exports in kizzasi-model
                Err(InferenceError::PipelineConfig(
                    "Mamba not yet supported - models not exported".into(),
                ))
            }
        }
    }

    /// Load weights from a file
    pub fn load_weights(
        &self,
        _model: &mut dyn AutoregressiveModel,
        path: impl AsRef<Path>,
    ) -> InferenceResult<()> {
        // Placeholder for weight loading
        // Will integrate with kizzasi_model::loader once implemented
        let _path = path.as_ref();
        tracing::info!("Weight loading not yet implemented");
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
}
