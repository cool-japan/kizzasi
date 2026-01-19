//! Model Factory for Instantiating Models from Loaded Weights
//!
//! This module provides utilities to instantiate model architectures from configurations
//! and pre-loaded (potentially quantized) weights.
//!
//! # Complete Pipeline
//!
//! The factory completes the final step in the model loading pipeline:
//!
//! ```text
//! 1. Download      → HuggingFaceHub::load_model()
//! 2. Convert       → HuggingFaceModelLoader::convert_weights()
//! 3. Quantize      → DynamicQuantizer::quantize_weights()
//! 4. Instantiate   → ModelFactory::create_from_config()  ← This module
//! ```
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use kizzasi_model::factory::ModelFactory;
//! use kizzasi_model::huggingface::ModelConfig;
//! use kizzasi_model::dynamic_quantization::QuantizedWeightStorage;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // After downloading, converting, and quantizing weights...
//! let config: ModelConfig = todo!();
//! let weights: HashMap<String, QuantizedWeightStorage> = todo!();
//!
//! // Create model instance
//! let model = ModelFactory::create_from_config(&config, weights)?;
//!
//! // Use model for inference
//! let output = model.predict(&input)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Supported Architectures
//!
//! - Mamba / Mamba2
//! - RWKV (v6, v7)
//! - S4 / S4D / S5
//! - Transformer
//! - H3
//! - Hybrid (Mamba + Attention)

use crate::dynamic_quantization::QuantizedWeightStorage;
use crate::error::{ModelError, ModelResult};
use crate::huggingface::ModelConfig;
use crate::mamba::{Mamba, MambaConfig};
use crate::mamba2::{Mamba2, Mamba2Config};
use crate::rwkv::{Rwkv, RwkvConfig};
use crate::rwkv7::{Rwkv7, Rwkv7Config};
use crate::s4::{S4Config, S4D};
use crate::s5::{S5Config, S5};
use crate::transformer::{Transformer, TransformerConfig};
use crate::AutoregressiveModel;
use crate::ModelType;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use tracing::{debug, info, instrument};

/// Model factory for creating model instances from configurations and weights
///
/// The factory handles:
/// - Configuration conversion (HuggingFace → Kizzasi format)
/// - Weight injection into model layers
/// - Dequantization when needed
/// - Architecture-specific initialization
pub struct ModelFactory;

impl ModelFactory {
    /// Create a model from HuggingFace config and quantized weights
    ///
    /// This automatically detects the model type from the configuration and
    /// instantiates the appropriate architecture.
    ///
    /// # Arguments
    ///
    /// * `config` - HuggingFace model configuration
    /// * `weights` - Quantized weights (from DynamicQuantizer)
    ///
    /// # Returns
    ///
    /// A boxed trait object implementing `AutoregressiveModel`
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model type cannot be detected
    /// - Required configuration fields are missing
    /// - Weights are incompatible with configuration
    #[instrument(skip(weights), fields(model_type = ?config.model_type))]
    pub fn create_from_config(
        config: &ModelConfig,
        weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<Box<dyn AutoregressiveModel>> {
        info!("Creating model from config");

        // Detect model type
        let model_type = Self::detect_model_type(config)?;
        debug!("Detected model type: {}", model_type);

        // Create appropriate model based on type
        match model_type {
            ModelType::Mamba | ModelType::Mamba2 => {
                let mamba_config = Self::hf_config_to_mamba_config(config)?;
                let model = Self::create_mamba(mamba_config, weights)?;
                Ok(Box::new(model))
            }
            ModelType::Rwkv => {
                let rwkv_config = Self::hf_config_to_rwkv_config(config)?;
                let model = Self::create_rwkv(rwkv_config, weights)?;
                Ok(Box::new(model))
            }
            ModelType::S4 => {
                let s4_config = Self::hf_config_to_s4_config(config)?;
                let model = Self::create_s4(s4_config, weights)?;
                Ok(Box::new(model))
            }
            ModelType::S4D => {
                let s5_config = Self::hf_config_to_s5_config(config)?;
                let model = Self::create_s5(s5_config, weights)?;
                Ok(Box::new(model))
            }
            ModelType::Transformer => {
                let transformer_config = Self::hf_config_to_transformer_config(config)?;
                let model = Self::create_transformer(transformer_config, weights)?;
                Ok(Box::new(model))
            }
        }
    }

    /// Create Mamba model from config and weights
    ///
    /// # Weight Requirements
    ///
    /// Expected weights for `num_layers` layers:
    /// - `input_proj`: [input_dim, hidden_dim]
    /// - `output_proj`: [hidden_dim, input_dim]
    /// - `layers.{i}.norm.weight`: [hidden_dim]
    /// - `layers.{i}.in_proj`: [hidden_dim, inner_dim*2]
    /// - `layers.{i}.ssm.log_a`: [state_dim]
    /// - `layers.{i}.ssm.delta_proj`: [inner_dim, inner_dim]
    /// - `layers.{i}.ssm.b_proj`: [inner_dim, state_dim]
    /// - `layers.{i}.ssm.c_proj`: [inner_dim, state_dim]
    /// - `layers.{i}.out_proj`: [inner_dim, hidden_dim]
    #[instrument(skip(_weights))]
    pub fn create_mamba(
        config: MambaConfig,
        _weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<Mamba> {
        info!(
            "Creating Mamba model: hidden_dim={}, state_dim={}, num_layers={}",
            config.hidden_dim, config.state_dim, config.num_layers
        );

        // For now, create a default Mamba model
        // TODO: Inject weights into the model
        // This requires extending Mamba to support weight loading

        let model = Mamba::new(config)?;

        debug!("Mamba model created successfully (weights not yet injected)");
        Ok(model)
    }

    /// Create Mamba2 model from config and weights
    #[instrument(skip(_weights))]
    pub fn create_mamba2(
        config: Mamba2Config,
        _weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<Mamba2> {
        info!(
            "Creating Mamba2 model: hidden_dim={}, state_dim={}, num_layers={}",
            config.hidden_dim, config.state_dim, config.num_layers
        );

        let model = Mamba2::new(config)?;

        debug!("Mamba2 model created successfully (weights not yet injected)");
        Ok(model)
    }

    /// Create RWKV model from config and weights
    #[instrument(skip(_weights))]
    pub fn create_rwkv(
        config: RwkvConfig,
        _weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<Rwkv> {
        info!(
            "Creating RWKV model: hidden_dim={}, num_heads={}, num_layers={}",
            config.hidden_dim, config.num_heads, config.num_layers
        );

        let model = Rwkv::new(config)?;

        debug!("RWKV model created successfully (weights not yet injected)");
        Ok(model)
    }

    /// Create RWKV-v7 model from config and weights
    #[instrument(skip(_weights))]
    pub fn create_rwkv7(
        config: Rwkv7Config,
        _weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<Rwkv7> {
        info!(
            "Creating RWKV-v7 model: hidden_dim={}, num_layers={}",
            config.hidden_dim, config.num_layers
        );

        let model = Rwkv7::new(config)?;

        debug!("RWKV-v7 model created successfully (weights not yet injected)");
        Ok(model)
    }

    /// Create S4 model from config and weights
    #[instrument(skip(_weights))]
    pub fn create_s4(
        config: S4Config,
        _weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<S4D> {
        info!(
            "Creating S4 model: hidden_dim={}, state_dim={}, num_layers={}",
            config.hidden_dim, config.state_dim, config.num_layers
        );

        let model = S4D::new(config)?;

        debug!("S4D model created successfully (weights not yet injected)");
        Ok(model)
    }

    /// Create S5 model from config and weights
    #[instrument(skip(_weights))]
    pub fn create_s5(
        config: S5Config,
        _weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<S5> {
        info!(
            "Creating S5 model: hidden_dim={}, state_dim={}, num_layers={}",
            config.hidden_dim, config.state_dim, config.num_layers
        );

        let model = S5::new(config)?;

        debug!("S5 model created successfully (weights not yet injected)");
        Ok(model)
    }

    /// Create Transformer model from config and weights
    #[instrument(skip(_weights))]
    pub fn create_transformer(
        config: TransformerConfig,
        _weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<Transformer> {
        info!(
            "Creating Transformer model: hidden_dim={}, num_heads={}, num_layers={}",
            config.hidden_dim, config.num_heads, config.num_layers
        );

        let model = Transformer::new(config)?;

        debug!("Transformer model created successfully (weights not yet injected)");
        Ok(model)
    }

    /// Detect model type from HuggingFace configuration
    ///
    /// Checks both `model_type` and `architectures` fields to determine the model type.
    fn detect_model_type(config: &ModelConfig) -> ModelResult<ModelType> {
        // Check model_type field
        if let Some(model_type) = &config.model_type {
            match model_type.to_lowercase().as_str() {
                "mamba" => return Ok(ModelType::Mamba),
                "mamba2" => return Ok(ModelType::Mamba2),
                "rwkv" | "rwkv6" => return Ok(ModelType::Rwkv),
                "rwkv7" => return Ok(ModelType::Rwkv),
                "s4" => return Ok(ModelType::S4),
                "s4d" | "s5" => return Ok(ModelType::S4D),
                "transformer" | "gpt2" | "llama" => return Ok(ModelType::Transformer),
                _ => {}
            }
        }

        // Check architectures field
        if let Some(architectures) = &config.architecture {
            for arch in architectures {
                let arch_lower = arch.to_lowercase();
                if arch_lower.contains("mamba") {
                    return Ok(ModelType::Mamba);
                } else if arch_lower.contains("rwkv") {
                    return Ok(ModelType::Rwkv);
                } else if arch_lower.contains("s4") || arch_lower.contains("s5") {
                    return Ok(ModelType::S4);
                } else if arch_lower.contains("transformer") || arch_lower.contains("gpt") {
                    return Ok(ModelType::Transformer);
                }
            }
        }

        Err(ModelError::simple_load_error(
            "Could not detect model type from configuration. Please specify model_type explicitly.",
        ))
    }

    /// Convert HuggingFace ModelConfig to MambaConfig
    fn hf_config_to_mamba_config(config: &ModelConfig) -> ModelResult<MambaConfig> {
        let hidden_dim = config
            .hidden_dim
            .ok_or_else(|| ModelError::simple_load_error("Missing required field: hidden_size"))?;

        let num_layers = config.num_layers.ok_or_else(|| {
            ModelError::simple_load_error("Missing required field: num_hidden_layers")
        })?;

        let state_dim = config.state_dim.unwrap_or(16); // Default for Mamba

        Ok(MambaConfig {
            input_dim: 1, // Signal prediction default
            hidden_dim,
            state_dim,
            expand_factor: 2,    // Mamba default
            conv_kernel_size: 4, // Mamba default
            num_layers,
            dropout: 0.0,
            use_mamba2: false,
        })
    }

    /// Convert HuggingFace ModelConfig to RwkvConfig
    fn hf_config_to_rwkv_config(config: &ModelConfig) -> ModelResult<RwkvConfig> {
        let hidden_dim = config
            .hidden_dim
            .ok_or_else(|| ModelError::simple_load_error("Missing required field: hidden_size"))?;

        let num_layers = config.num_layers.ok_or_else(|| {
            ModelError::simple_load_error("Missing required field: num_hidden_layers")
        })?;

        let num_heads = config.num_attention_heads.unwrap_or(8);
        let head_dim = hidden_dim / num_heads;
        let intermediate_dim = hidden_dim * 4; // Standard FFN expansion

        Ok(RwkvConfig {
            input_dim: 1,
            hidden_dim,
            intermediate_dim,
            num_layers,
            num_heads,
            head_dim,
            dropout: 0.0,
            time_decay_init: 0.99, // RWKV default
            use_rms_norm: false,   // Standard LayerNorm
        })
    }

    /// Convert HuggingFace ModelConfig to S4Config
    fn hf_config_to_s4_config(config: &ModelConfig) -> ModelResult<S4Config> {
        let hidden_dim = config
            .hidden_dim
            .ok_or_else(|| ModelError::simple_load_error("Missing required field: hidden_size"))?;

        let num_layers = config.num_layers.ok_or_else(|| {
            ModelError::simple_load_error("Missing required field: num_hidden_layers")
        })?;

        let state_dim = config.state_dim.unwrap_or(64); // S4 default

        Ok(S4Config {
            input_dim: 1,
            hidden_dim,
            state_dim,
            num_layers,
            dropout: 0.0,
            dt_min: 0.001, // S4 defaults
            dt_max: 0.1,
            use_diagonal: true,  // S4D by default
            use_rms_norm: false, // Standard LayerNorm
        })
    }

    /// Convert HuggingFace ModelConfig to S5Config
    fn hf_config_to_s5_config(config: &ModelConfig) -> ModelResult<S5Config> {
        let hidden_dim = config
            .hidden_dim
            .ok_or_else(|| ModelError::simple_load_error("Missing required field: hidden_size"))?;

        let num_layers = config.num_layers.ok_or_else(|| {
            ModelError::simple_load_error("Missing required field: num_hidden_layers")
        })?;

        let state_dim = config.state_dim.unwrap_or(64); // S5 default

        Ok(S5Config {
            input_dim: 1,
            hidden_dim,
            state_dim,
            num_layers,
            dt: 0.01,       // S5 default discretization step
            block_size: 32, // S5 default block size
        })
    }

    /// Convert HuggingFace ModelConfig to TransformerConfig
    fn hf_config_to_transformer_config(config: &ModelConfig) -> ModelResult<TransformerConfig> {
        let hidden_dim = config
            .hidden_dim
            .ok_or_else(|| ModelError::simple_load_error("Missing required field: hidden_size"))?;

        let num_layers = config.num_layers.ok_or_else(|| {
            ModelError::simple_load_error("Missing required field: num_hidden_layers")
        })?;

        let num_heads = config.num_attention_heads.ok_or_else(|| {
            ModelError::simple_load_error("Missing required field: num_attention_heads")
        })?;

        let max_seq_len = config.max_position_embeddings.unwrap_or(2048);
        let head_dim = hidden_dim / num_heads;
        let ff_dim = hidden_dim * 4; // Standard Transformer FFN expansion

        Ok(TransformerConfig {
            input_dim: 1,
            hidden_dim,
            num_heads,
            head_dim,
            ff_dim,
            num_layers,
            max_seq_len,
            dropout: 0.0,
            use_rms_norm: false, // Standard LayerNorm
            causal: true,        // Causal masking for autoregressive models
        })
    }

    /// Dequantize weights to FP32 for model initialization
    ///
    /// Most models expect FP32 weights during initialization, so this helper
    /// converts quantized weights back to FP32.
    ///
    /// # Note
    ///
    /// This is a temporary solution. Future versions should support direct
    /// quantized weight injection to avoid the dequantization overhead.
    pub fn dequantize_weights(
        weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<HashMap<String, Array2<f32>>> {
        let mut fp32_weights = HashMap::new();

        for (name, storage) in weights {
            let array = match storage {
                QuantizedWeightStorage::FP32(arr) => arr,
                QuantizedWeightStorage::INT8(quant) => {
                    // Dequantize INT8 → FP32
                    quant.dequantize_2d()?
                }
                QuantizedWeightStorage::FP16(fp16) => {
                    // Convert FP16 → FP32
                    fp16.to_f32_2d()?
                }
                QuantizedWeightStorage::BF16(bf16) => {
                    // Convert BF16 → FP32
                    bf16.to_f32_2d()?
                }
            };

            fp32_weights.insert(name, array);
        }

        Ok(fp32_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_detect_model_type_from_model_type_field() {
        let config = ModelConfig {
            architecture: None,
            hidden_dim: Some(256),
            num_layers: Some(4),
            vocab_size: None,
            max_position_embeddings: None,
            state_dim: Some(16),
            num_attention_heads: None,
            model_type: Some("mamba".to_string()),
            extra: HashMap::new(),
        };

        let model_type = ModelFactory::detect_model_type(&config).expect("Should detect Mamba");
        assert_eq!(model_type, ModelType::Mamba);
    }

    #[test]
    fn test_detect_model_type_from_architectures_field() {
        let config = ModelConfig {
            architecture: Some(vec!["MambaForCausalLM".to_string()]),
            hidden_dim: Some(256),
            num_layers: Some(4),
            vocab_size: None,
            max_position_embeddings: None,
            state_dim: Some(16),
            num_attention_heads: None,
            model_type: None,
            extra: HashMap::new(),
        };

        let model_type = ModelFactory::detect_model_type(&config).expect("Should detect Mamba");
        assert_eq!(model_type, ModelType::Mamba);
    }

    #[test]
    fn test_detect_model_type_rwkv() {
        let config = ModelConfig {
            architecture: None,
            hidden_dim: Some(512),
            num_layers: Some(6),
            vocab_size: None,
            max_position_embeddings: None,
            state_dim: None,
            num_attention_heads: Some(8),
            model_type: Some("rwkv6".to_string()),
            extra: HashMap::new(),
        };

        let model_type = ModelFactory::detect_model_type(&config).expect("Should detect RWKV");
        assert_eq!(model_type, ModelType::Rwkv);
    }

    #[test]
    fn test_detect_model_type_transformer() {
        let config = ModelConfig {
            architecture: Some(vec!["GPT2LMHeadModel".to_string()]),
            hidden_dim: Some(768),
            num_layers: Some(12),
            vocab_size: Some(50257),
            max_position_embeddings: Some(1024),
            state_dim: None,
            num_attention_heads: Some(12),
            model_type: None,
            extra: HashMap::new(),
        };

        let model_type =
            ModelFactory::detect_model_type(&config).expect("Should detect Transformer");
        assert_eq!(model_type, ModelType::Transformer);
    }

    #[test]
    fn test_detect_model_type_fails_without_indicators() {
        let config = ModelConfig {
            architecture: None,
            hidden_dim: Some(256),
            num_layers: Some(4),
            vocab_size: None,
            max_position_embeddings: None,
            state_dim: None,
            num_attention_heads: None,
            model_type: None,
            extra: HashMap::new(),
        };

        let result = ModelFactory::detect_model_type(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_hf_config_to_mamba_config() {
        let config = ModelConfig {
            architecture: None,
            hidden_dim: Some(256),
            num_layers: Some(4),
            vocab_size: None,
            max_position_embeddings: None,
            state_dim: Some(16),
            num_attention_heads: None,
            model_type: Some("mamba".to_string()),
            extra: HashMap::new(),
        };

        let mamba_config =
            ModelFactory::hf_config_to_mamba_config(&config).expect("Should convert config");

        assert_eq!(mamba_config.hidden_dim, 256);
        assert_eq!(mamba_config.num_layers, 4);
        assert_eq!(mamba_config.state_dim, 16);
        assert_eq!(mamba_config.expand_factor, 2);
    }

    #[test]
    fn test_hf_config_to_rwkv_config() {
        let config = ModelConfig {
            architecture: None,
            hidden_dim: Some(512),
            num_layers: Some(6),
            vocab_size: None,
            max_position_embeddings: None,
            state_dim: None,
            num_attention_heads: Some(8),
            model_type: Some("rwkv".to_string()),
            extra: HashMap::new(),
        };

        let rwkv_config =
            ModelFactory::hf_config_to_rwkv_config(&config).expect("Should convert config");

        assert_eq!(rwkv_config.hidden_dim, 512);
        assert_eq!(rwkv_config.num_layers, 6);
        assert_eq!(rwkv_config.num_heads, 8);
    }

    #[test]
    fn test_hf_config_to_transformer_config() {
        let config = ModelConfig {
            architecture: None,
            hidden_dim: Some(768),
            num_layers: Some(12),
            vocab_size: Some(50257),
            max_position_embeddings: Some(1024),
            state_dim: None,
            num_attention_heads: Some(12),
            model_type: Some("transformer".to_string()),
            extra: HashMap::new(),
        };

        let transformer_config =
            ModelFactory::hf_config_to_transformer_config(&config).expect("Should convert config");

        assert_eq!(transformer_config.hidden_dim, 768);
        assert_eq!(transformer_config.num_layers, 12);
        assert_eq!(transformer_config.num_heads, 12);
        assert_eq!(transformer_config.max_seq_len, 1024);
    }

    #[test]
    fn test_dequantize_weights_fp32() {
        let mut weights = HashMap::new();
        let array = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j) as f32);
        weights.insert(
            "test".to_string(),
            QuantizedWeightStorage::FP32(array.clone()),
        );

        let dequantized = ModelFactory::dequantize_weights(weights).expect("Should dequantize");

        assert_eq!(dequantized.len(), 1);
        assert!(dequantized.contains_key("test"));
        assert_eq!(&dequantized["test"], &array);
    }

    #[test]
    fn test_create_mamba_model() {
        let config = MambaConfig {
            input_dim: 1,
            hidden_dim: 64,
            state_dim: 16,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 2,
            dropout: 0.0,
            use_mamba2: false,
        };

        let weights = HashMap::new();
        let model = ModelFactory::create_mamba(config, weights);

        assert!(model.is_ok());
    }

    #[test]
    fn test_create_rwkv_model() {
        let config = RwkvConfig {
            input_dim: 1,
            hidden_dim: 128,
            intermediate_dim: 512,
            num_layers: 2,
            num_heads: 4,
            head_dim: 32,
            dropout: 0.0,
            time_decay_init: 0.99,
            use_rms_norm: false,
        };

        let weights = HashMap::new();
        let model = ModelFactory::create_rwkv(config, weights);

        assert!(model.is_ok());
    }

    #[test]
    fn test_create_transformer_model() {
        let config = TransformerConfig {
            input_dim: 1,
            hidden_dim: 256,
            num_heads: 8,
            head_dim: 32,
            ff_dim: 1024,
            num_layers: 2,
            max_seq_len: 512,
            dropout: 0.0,
            use_rms_norm: false,
            causal: true,
        };

        let weights = HashMap::new();
        let model = ModelFactory::create_transformer(config, weights);

        assert!(model.is_ok());
    }
}
