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
// Architecture imports follow the same feature gates as the modules they come
// from (see `[features]` in Cargo.toml): a build without `rwkv` has no
// `crate::rwkv`, so importing it unconditionally would not compile.
#[cfg(feature = "mamba")]
use crate::mamba::{Mamba, MambaConfig};
#[cfg(feature = "mamba")]
use crate::mamba2::{Mamba2, Mamba2Config};
#[cfg(feature = "rwkv")]
use crate::rwkv::{Rwkv, RwkvConfig};
#[cfg(feature = "rwkv")]
use crate::rwkv7::{Rwkv7, Rwkv7Config};
#[cfg(feature = "s4")]
use crate::s4::{S4Config, S4D};
#[cfg(feature = "s4")]
use crate::s5::{S5Config, S5};
#[cfg(feature = "transformer")]
use crate::transformer::{Transformer, TransformerConfig};
use crate::AutoregressiveModel;
use crate::ModelType;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use tracing::{debug, info, instrument, warn};

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
    // With every architecture feature off, all match arms below are the
    // "not compiled in" error arms, which do not consume `weights`.
    #[cfg_attr(
        not(any(
            feature = "mamba",
            feature = "rwkv",
            feature = "s4",
            feature = "transformer"
        )),
        allow(unused_variables)
    )]
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
            #[cfg(feature = "mamba")]
            ModelType::Mamba => {
                let mamba_config = Self::hf_config_to_mamba_config(config)?;
                let model = Self::create_mamba(mamba_config, weights)?;
                Ok(Box::new(model))
            }
            #[cfg(not(feature = "mamba"))]
            ModelType::Mamba => Err(ModelError::unsupported_operation(
                "from_config",
                "Mamba (rebuild kizzasi-model with the `mamba` feature)",
            )),
            #[cfg(feature = "mamba")]
            ModelType::Mamba2 => {
                let mamba2_config = Self::hf_config_to_mamba2_config(config)?;
                let model = Self::create_mamba2(mamba2_config, weights)?;
                Ok(Box::new(model))
            }
            #[cfg(not(feature = "mamba"))]
            ModelType::Mamba2 => Err(ModelError::unsupported_operation(
                "from_config",
                "Mamba2 (rebuild kizzasi-model with the `mamba` feature)",
            )),
            #[cfg(not(feature = "rwkv"))]
            ModelType::Rwkv => Err(ModelError::unsupported_operation(
                "from_config",
                "RWKV (rebuild kizzasi-model with the `rwkv` feature)",
            )),
            #[cfg(feature = "rwkv")]
            ModelType::Rwkv => {
                // `detect_model_type` maps both "rwkv"/"rwkv6" and "rwkv7"
                // to this same coarse `ModelType::Rwkv` — there is no
                // `ModelType::Rwkv7` variant, because `ModelType` is also
                // matched exhaustively by `kizzasi-inference::registry`
                // (a different crate this fix cannot touch), so adding a
                // variant there would be a breaking change out of scope
                // here. Disambiguate from the raw config string instead, so
                // an RWKV-7 checkpoint still builds a real `Rwkv7` model
                // instead of silently mis-loading into an RWKV-v6 `Rwkv`.
                if Self::config_names(config, "rwkv7") {
                    let rwkv7_config = Self::hf_config_to_rwkv7_config(config)?;
                    let model = Self::create_rwkv7(rwkv7_config, weights)?;
                    Ok(Box::new(model))
                } else {
                    let rwkv_config = Self::hf_config_to_rwkv_config(config)?;
                    let model = Self::create_rwkv(rwkv_config, weights)?;
                    Ok(Box::new(model))
                }
            }
            #[cfg(feature = "s4")]
            ModelType::S4 => {
                let s4_config = Self::hf_config_to_s4_config(config)?;
                let model = Self::create_s4(s4_config, weights)?;
                Ok(Box::new(model))
            }
            #[cfg(not(feature = "s4"))]
            ModelType::S4 => Err(ModelError::unsupported_operation(
                "from_config",
                "S4 (rebuild kizzasi-model with the `s4` feature)",
            )),
            #[cfg(not(feature = "s4"))]
            ModelType::S4D => Err(ModelError::unsupported_operation(
                "from_config",
                "S4D/S5 (rebuild kizzasi-model with the `s4` feature)",
            )),
            #[cfg(feature = "s4")]
            ModelType::S4D => {
                // Same situation as RWKV above: `detect_model_type` maps
                // both "s4d" and "s5" to this same `ModelType::S4D` (no
                // `ModelType::S5` variant, for the same cross-crate reason).
                // Disambiguate from the raw config string: "s5" builds the
                // real S5 architecture (parallel-scan SSM); "s4d" builds the
                // diagonal S4 kernel via `create_s4`, which is what the
                // `S4D` struct actually is (see `S4Config::use_diagonal`).
                if Self::config_names(config, "s5") {
                    let s5_config = Self::hf_config_to_s5_config(config)?;
                    let model = Self::create_s5(s5_config, weights)?;
                    Ok(Box::new(model))
                } else {
                    let s4_config = Self::hf_config_to_s4_config(config)?;
                    let model = Self::create_s4(s4_config, weights)?;
                    Ok(Box::new(model))
                }
            }
            #[cfg(feature = "transformer")]
            ModelType::Transformer => {
                let transformer_config = Self::hf_config_to_transformer_config(config)?;
                let model = Self::create_transformer(transformer_config, weights)?;
                Ok(Box::new(model))
            }
            #[cfg(not(feature = "transformer"))]
            ModelType::Transformer => Err(ModelError::unsupported_operation(
                "from_config",
                "Transformer (rebuild kizzasi-model with the `transformer` feature)",
            )),
            ModelType::Rwkv5 => {
                // RWKV v5 models are created directly, not from HF configs
                Err(ModelError::unsupported_operation(
                    "from_config",
                    "RWKV5 (create directly via Rwkv5Model::new)",
                ))
            }
            ModelType::NeuralOde => {
                // Neural ODE models are created directly, not from HF configs
                Err(ModelError::unsupported_operation(
                    "from_config",
                    "NeuralODE (create directly via NeuralOdeModel::new)",
                ))
            }
            ModelType::MultiModal => {
                // Multi-modal models are created directly, not from HF configs
                Err(ModelError::unsupported_operation(
                    "from_config",
                    "MultiModal (create directly via MultiModalModel::new)",
                ))
            }
            ModelType::Snn => {
                // SNN models are created directly, not from HF configs
                Err(ModelError::unsupported_operation(
                    "from_config",
                    "SNN (create directly via SpikingNeuralNetwork::new)",
                ))
            }
            ModelType::MultiScale => {
                // Multi-scale models are created directly, not from HF configs
                Err(ModelError::unsupported_operation(
                    "from_config",
                    "MultiScale (create directly via MultiScaleModel::new)",
                ))
            }
        }
    }

    /// Convert quantized weights to a `HashMap<String, Vec<f32>>` parameter map.
    ///
    /// Each `QuantizedWeightStorage` value is dequantised to FP32 and then flattened
    /// into a contiguous `Vec<f32>`.  The resulting map is handed straight to a
    /// model's `load_weights_map` method — it never touches the filesystem.
    // Weight-application helpers used by every `create_*` constructor; with
    // no architecture compiled in there is no constructor to use them.
    #[cfg(any(
        feature = "mamba",
        feature = "rwkv",
        feature = "s4",
        feature = "transformer"
    ))]
    fn quantized_to_f32_vecs(
        weights: &HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<HashMap<String, Vec<f32>>> {
        let mut out = HashMap::with_capacity(weights.len());
        for (name, storage) in weights {
            let array: Array2<f32> = storage.to_fp32()?;
            // Use iter().copied().collect() to guarantee we only collect the
            // logical elements of the array (not raw backing-store bytes which
            // may include extra capacity from slicing / non-contiguous layouts).
            let flat: Vec<f32> = array.iter().copied().collect();
            out.insert(name.clone(), flat);
        }
        Ok(out)
    }

    /// Reject a weight map whose names match nothing in the target model.
    ///
    /// Weight loading is deliberately partial — a caller may supply a subset of
    /// tensors. But a map that matches *zero* parameters means the names came
    /// from a different naming scheme entirely; returning `Ok` there would hand
    /// back a randomly-initialised model while reporting a successful load.
    // Weight-application helpers used by every `create_*` constructor; with
    // no architecture compiled in there is no constructor to use them.
    #[cfg(any(
        feature = "mamba",
        feature = "rwkv",
        feature = "s4",
        feature = "transformer"
    ))]
    fn require_weights_applied(model: &str, applied: usize, supplied: usize) -> ModelResult<()> {
        if supplied > 0 && applied == 0 {
            return Err(ModelError::load_error(
                format!("{} weight injection", model),
                format!(
                    "none of the {} supplied tensors matched this model's parameter names; \
                     the model would have kept its random initialisation",
                    supplied
                ),
            ));
        }
        Ok(())
    }

    /// Create Mamba model from config and weights
    ///
    /// # Weight Requirements
    ///
    /// Expected weights for `num_layers` layers:
    /// - `input_proj`: `[input_dim, hidden_dim]`
    /// - `output_proj`: `[hidden_dim, input_dim]`
    /// - `layers.{i}.norm.weight`: `[hidden_dim]`
    /// - `layers.{i}.in_proj`: `[hidden_dim, inner_dim*2]`
    /// - `layers.{i}.ssm.log_a`: `[state_dim]`
    /// - `layers.{i}.ssm.delta_proj`: `[inner_dim, inner_dim]`
    /// - `layers.{i}.ssm.b_proj`: `[inner_dim, state_dim]`
    /// - `layers.{i}.ssm.c_proj`: `[inner_dim, state_dim]`
    /// - `layers.{i}.out_proj`: `[inner_dim, hidden_dim]`
    #[instrument(skip(weights))]
    #[cfg(feature = "mamba")]
    pub fn create_mamba(
        config: MambaConfig,
        weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<Mamba> {
        info!(
            "Creating Mamba model: hidden_dim={}, state_dim={}, num_layers={}",
            config.hidden_dim, config.state_dim, config.num_layers
        );

        let mut model = Mamba::new(config)?;

        if !weights.is_empty() {
            let f32_weights = Self::quantized_to_f32_vecs(&weights)?;
            let applied = model.load_weights_map(&f32_weights)?;
            Self::require_weights_applied("Mamba", applied, f32_weights.len())?;
            debug!(
                "Mamba model weights injected successfully ({} tensors)",
                applied
            );
        } else {
            warn!("Mamba model created without weights (empty weights map)");
        }

        debug!("Mamba model created successfully");
        Ok(model)
    }

    /// Create Mamba2 model from config and weights
    #[instrument(skip(weights))]
    #[cfg(feature = "mamba")]
    pub fn create_mamba2(
        config: Mamba2Config,
        weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<Mamba2> {
        info!(
            "Creating Mamba2 model: hidden_dim={}, state_dim={}, num_layers={}",
            config.hidden_dim, config.state_dim, config.num_layers
        );

        let mut model = Mamba2::new(config)?;

        if !weights.is_empty() {
            let f32_weights = Self::quantized_to_f32_vecs(&weights)?;
            let applied = model.load_weights_map(&f32_weights)?;
            Self::require_weights_applied("Mamba2", applied, f32_weights.len())?;
            debug!(
                "Mamba2 model weights injected successfully ({} tensors)",
                applied
            );
        } else {
            warn!("Mamba2 model created without weights (empty weights map)");
        }

        debug!("Mamba2 model created successfully");
        Ok(model)
    }

    /// Create RWKV model from config and weights
    #[instrument(skip(weights))]
    #[cfg(feature = "rwkv")]
    pub fn create_rwkv(
        config: RwkvConfig,
        weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<Rwkv> {
        info!(
            "Creating RWKV model: hidden_dim={}, num_heads={}, num_layers={}",
            config.hidden_dim, config.num_heads, config.num_layers
        );

        let mut model = Rwkv::new(config)?;

        if !weights.is_empty() {
            let f32_weights = Self::quantized_to_f32_vecs(&weights)?;
            let applied = model.load_weights_map(&f32_weights)?;
            Self::require_weights_applied("RWKV", applied, f32_weights.len())?;
            debug!(
                "RWKV model weights injected successfully ({} tensors)",
                applied
            );
        } else {
            warn!("RWKV model created without weights (empty weights map)");
        }

        debug!("RWKV model created successfully");
        Ok(model)
    }

    /// Create RWKV-v7 model from config and weights
    #[instrument(skip(weights))]
    #[cfg(feature = "rwkv")]
    pub fn create_rwkv7(
        config: Rwkv7Config,
        weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<Rwkv7> {
        info!(
            "Creating RWKV-v7 model: hidden_dim={}, num_layers={}",
            config.hidden_dim, config.num_layers
        );

        let mut model = Rwkv7::new(config)?;

        if !weights.is_empty() {
            let f32_weights = Self::quantized_to_f32_vecs(&weights)?;
            let applied = model.load_weights_map(&f32_weights)?;
            Self::require_weights_applied("RWKV-v7", applied, f32_weights.len())?;
            debug!(
                "RWKV-v7 model weights injected successfully ({} tensors)",
                applied
            );
        } else {
            warn!("RWKV-v7 model created without weights (empty weights map)");
        }

        debug!("RWKV-v7 model created successfully");
        Ok(model)
    }

    /// Create S4 model from config and weights
    #[instrument(skip(weights))]
    #[cfg(feature = "s4")]
    pub fn create_s4(
        config: S4Config,
        weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<S4D> {
        info!(
            "Creating S4 model: hidden_dim={}, state_dim={}, num_layers={}",
            config.hidden_dim, config.state_dim, config.num_layers
        );

        let mut model = S4D::new(config)?;

        if !weights.is_empty() {
            let f32_weights = Self::quantized_to_f32_vecs(&weights)?;
            let applied = model.load_weights_map(&f32_weights)?;
            Self::require_weights_applied("S4D", applied, f32_weights.len())?;
            debug!(
                "S4D model weights injected successfully ({} tensors)",
                applied
            );
        } else {
            warn!("S4D model created without weights (empty weights map)");
        }

        debug!("S4D model created successfully");
        Ok(model)
    }

    /// Create S5 model from config and weights
    #[instrument(skip(weights))]
    #[cfg(feature = "s4")]
    pub fn create_s5(
        config: S5Config,
        weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<S5> {
        info!(
            "Creating S5 model: hidden_dim={}, state_dim={}, num_layers={}",
            config.hidden_dim, config.state_dim, config.num_layers
        );

        let mut model = S5::new(config)?;

        if !weights.is_empty() {
            let f32_weights = Self::quantized_to_f32_vecs(&weights)?;
            let applied = model.load_weights_map(&f32_weights)?;
            Self::require_weights_applied("S5", applied, f32_weights.len())?;
            debug!(
                "S5 model weights injected successfully ({} tensors)",
                applied
            );
        } else {
            warn!("S5 model created without weights (empty weights map)");
        }

        debug!("S5 model created successfully");
        Ok(model)
    }

    /// Create Transformer model from config and weights
    #[instrument(skip(weights))]
    #[cfg(feature = "transformer")]
    pub fn create_transformer(
        config: TransformerConfig,
        weights: HashMap<String, QuantizedWeightStorage>,
    ) -> ModelResult<Transformer> {
        info!(
            "Creating Transformer model: hidden_dim={}, num_heads={}, num_layers={}",
            config.hidden_dim, config.num_heads, config.num_layers
        );

        let mut model = Transformer::new(config)?;

        if !weights.is_empty() {
            let f32_weights = Self::quantized_to_f32_vecs(&weights)?;
            let applied = model.load_weights_map(&f32_weights)?;
            Self::require_weights_applied("Transformer", applied, f32_weights.len())?;
            debug!(
                "Transformer model weights injected successfully ({} tensors)",
                applied
            );
        } else {
            warn!("Transformer model created without weights (empty weights map)");
        }

        debug!("Transformer model created successfully");
        Ok(model)
    }

    /// Case-insensitively check whether `config.model_type` equals `needle`,
    /// or any entry of `config.architecture` contains `needle`.
    ///
    /// Used by [`Self::create_from_config`] to disambiguate architectures
    /// that share one coarse [`ModelType`] variant (RWKV-6 vs RWKV-7, S4D vs
    /// S5) from the original config string, since `ModelType` itself cannot
    /// grow new variants for this without breaking `kizzasi-inference`'s
    /// exhaustive match over it.
    ///
    /// Only the `rwkv` and `s4` arms need this disambiguation, so a build
    /// without either feature has no caller for it.
    #[cfg(any(feature = "rwkv", feature = "s4"))]
    fn config_names(config: &ModelConfig, needle: &str) -> bool {
        let matches_type = config
            .model_type
            .as_deref()
            .is_some_and(|s| s.eq_ignore_ascii_case(needle));
        let matches_arch = config.architecture.as_ref().is_some_and(|archs| {
            archs
                .iter()
                .any(|a| a.to_lowercase().contains(&needle.to_lowercase()))
        });
        matches_type || matches_arch
    }

    /// Detect model type from HuggingFace configuration
    ///
    /// Checks both `model_type` and `architectures` fields to determine the model type.
    ///
    /// Note: `"rwkv7"` and `"s5"` intentionally collapse to the same coarse
    /// [`ModelType::Rwkv`] / [`ModelType::S4D`] as their v6/S4D siblings —
    /// see [`Self::create_from_config`], which re-inspects the raw config
    /// string via [`Self::config_names`] to route to the correct concrete
    /// architecture despite the shared `ModelType`.
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
                "neural_ode" | "neuralode" => return Ok(ModelType::NeuralOde),
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
    #[cfg(feature = "mamba")]
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

    /// Convert HuggingFace ModelConfig to Mamba2Config
    #[cfg(feature = "mamba")]
    fn hf_config_to_mamba2_config(config: &ModelConfig) -> ModelResult<Mamba2Config> {
        let hidden_dim = config
            .hidden_dim
            .ok_or_else(|| ModelError::simple_load_error("Missing required field: hidden_size"))?;

        let num_layers = config.num_layers.ok_or_else(|| {
            ModelError::simple_load_error("Missing required field: num_hidden_layers")
        })?;

        let state_dim = config.state_dim.unwrap_or(64); // Mamba2 default (typically 64-128)
        let num_heads = config
            .num_attention_heads
            .unwrap_or_else(|| (hidden_dim / 64).max(1))
            .max(1);
        let head_dim = hidden_dim / num_heads;

        Ok(Mamba2Config {
            input_dim: 1,
            hidden_dim,
            state_dim,
            num_heads,
            head_dim,
            expand_factor: 2,    // Mamba2 default
            conv_kernel_size: 4, // Mamba2 default
            num_layers,
            dropout: 0.0,
            use_rms_norm: true,
            chunk_size: 256, // Mamba2 SSD default
        })
    }

    /// Convert HuggingFace ModelConfig to RwkvConfig
    #[cfg(feature = "rwkv")]
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
            // `time_decay_init` lives in RWKV's log-log space: the per-step
            // decay is `exp(-exp(w))`. The reference implementations initialise
            // it around -5.0 (decay ≈ 0.993). A value of 0.99 would mean a decay
            // of exp(-exp(0.99)) ≈ 0.068 — effectively a one-step memory.
            time_decay_init: -5.0,
            use_rms_norm: false, // Standard LayerNorm
        })
    }

    /// Convert HuggingFace ModelConfig to Rwkv7Config
    #[cfg(feature = "rwkv")]
    fn hf_config_to_rwkv7_config(config: &ModelConfig) -> ModelResult<Rwkv7Config> {
        let hidden_dim = config
            .hidden_dim
            .ok_or_else(|| ModelError::simple_load_error("Missing required field: hidden_size"))?;

        let num_layers = config.num_layers.ok_or_else(|| {
            ModelError::simple_load_error("Missing required field: num_hidden_layers")
        })?;

        let num_heads = config
            .num_attention_heads
            .unwrap_or_else(|| (hidden_dim / 64).max(1))
            .max(1);
        let head_dim = hidden_dim / num_heads;
        let context_length = config.max_position_embeddings.unwrap_or(16384);

        Ok(Rwkv7Config {
            input_dim: 1,
            hidden_dim,
            num_layers,
            num_heads,
            head_dim,
            expand_factor: 3.5, // RWKV-7 default FFN expansion
            context_length,
            // See the matching comment on `hf_config_to_rwkv_config` above:
            // same log-log decay space, same reference-implementation init.
            time_decay_init: -5.0,
        })
    }

    /// Convert HuggingFace ModelConfig to S4Config
    #[cfg(feature = "s4")]
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
    #[cfg(feature = "s4")]
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
    #[cfg(feature = "transformer")]
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
    fn test_create_from_config_mamba2_routes_to_mamba2_not_mamba() {
        // Regression test for id102: "mamba2" used to route through
        // `hf_config_to_mamba_config` (which hardcodes `use_mamba2: false`)
        // and `create_mamba`, silently building a Mamba-v1 model.
        let config = ModelConfig {
            architecture: None,
            hidden_dim: Some(256),
            num_layers: Some(2),
            vocab_size: None,
            max_position_embeddings: None,
            state_dim: None, // omitted: Mamba defaults to 16, Mamba2 to 64
            num_attention_heads: None,
            model_type: Some("mamba2".to_string()),
            extra: HashMap::new(),
        };

        let model = ModelFactory::create_from_config(&config, HashMap::new())
            .expect("mamba2 config should build successfully");

        assert_eq!(model.model_type(), ModelType::Mamba2);
        assert_eq!(
            model.state_dim(),
            64,
            "state_dim=64 is Mamba2Config's default; Mamba's default is 16, \
             so this also proves hf_config_to_mamba2_config (not \
             hf_config_to_mamba_config) was used"
        );
    }

    #[test]
    fn test_create_from_config_rwkv7_routes_to_rwkv7_not_rwkv6() {
        // Regression test for id102: "rwkv7" used to detect as the same
        // coarse `ModelType::Rwkv` as v6 and route through `create_rwkv`
        // (RWKV-v6), silently building the wrong architecture.
        //
        // Discriminated via weight-key schema: `w_a` ("bonus/attention
        // gate") is an RWKV-7-only concept (see `Rwkv7TimeMixing`); RWKV-v6
        // has no such key. If routing regressed to RWKV-v6, this weight map
        // would match nothing and `create_rwkv` would return an error via
        // `require_weights_applied`.
        let hidden_dim = 8usize;
        let config = ModelConfig {
            architecture: None,
            hidden_dim: Some(hidden_dim),
            num_layers: Some(1),
            vocab_size: None,
            max_position_embeddings: None,
            state_dim: None,
            num_attention_heads: Some(2),
            model_type: Some("rwkv7".to_string()),
            extra: HashMap::new(),
        };

        let mut weights = HashMap::new();
        let arr = Array2::<f32>::zeros((hidden_dim, hidden_dim));
        weights.insert(
            "layers.0.time_mixing.w_a".to_string(),
            QuantizedWeightStorage::FP32(arr),
        );

        let model = ModelFactory::create_from_config(&config, weights)
            .expect("rwkv7 config with a w_a weight should route to Rwkv7 and apply it");
        assert_eq!(model.hidden_dim(), hidden_dim);
    }

    #[test]
    fn test_create_from_config_s5_and_s4d_route_to_different_structs() {
        // Regression test for id102: "s4d" and "s5" both detected as the
        // same coarse `ModelType::S4D` and both routed through `create_s5`,
        // so an S4D checkpoint silently built an S5 model. `S5::model_type`
        // returns `ModelType::S4` (it has no dedicated variant) while
        // `S4D::model_type` returns `ModelType::S4D` — different concrete
        // structs, distinguishable via this pre-existing asymmetry.
        let base = ModelConfig {
            architecture: None,
            hidden_dim: Some(16),
            num_layers: Some(1),
            vocab_size: None,
            max_position_embeddings: None,
            state_dim: Some(8),
            num_attention_heads: None,
            model_type: None,
            extra: HashMap::new(),
        };

        let s4d_config = ModelConfig {
            model_type: Some("s4d".to_string()),
            ..base.clone()
        };
        let s4d_model = ModelFactory::create_from_config(&s4d_config, HashMap::new())
            .expect("s4d config should build successfully");
        assert_eq!(s4d_model.model_type(), ModelType::S4D);

        let s5_config = ModelConfig {
            model_type: Some("s5".to_string()),
            ..base
        };
        let s5_model = ModelFactory::create_from_config(&s5_config, HashMap::new())
            .expect("s5 config should build successfully");
        assert_eq!(
            s5_model.model_type(),
            ModelType::S4,
            "S5::model_type() == ModelType::S4 is what proves the S5 struct \
             (not S4D) was built for an 's5' model_type"
        );
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
            time_decay_init: -5.0,
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

    // -----------------------------------------------------------------
    // WS-B: factory weight injection tests
    // -----------------------------------------------------------------

    /// Shared mutex to serialise the factory weight-injection tests.
    ///
    /// The factory itself no longer touches the filesystem, but these tests
    /// still stage reference weights through temporary JSON files, and
    /// [`test_factory_injection_writes_no_temp_files`] scans the shared temp
    /// directory. Holding this mutex keeps those file-system observations from
    /// interleaving across the concurrently-running unit tests.
    fn factory_injection_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::{Mutex, OnceLock};
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Verify that creating a Mamba model with empty weights does not error.
    #[cfg(feature = "mamba")]
    #[test]
    fn test_factory_weight_injection_empty() {
        let config = crate::mamba::MambaConfig {
            input_dim: 1,
            hidden_dim: 32,
            state_dim: 8,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 1,
            dropout: 0.0,
            use_mamba2: false,
        };

        let empty_weights: HashMap<String, QuantizedWeightStorage> = HashMap::new();
        let result = ModelFactory::create_mamba(config, empty_weights);
        assert!(
            result.is_ok(),
            "create_mamba with empty weights should succeed: {:?}",
            result.err()
        );
    }

    /// Verify that a Mamba model can be created and its weights round-tripped
    /// through the factory's JSON injection pathway.
    #[cfg(feature = "mamba")]
    #[test]
    fn test_factory_weight_injection_mamba() {
        let _guard = factory_injection_lock();
        use crate::dynamic_quantization::QuantizedWeightStorage;
        use crate::mamba::{Mamba, MambaConfig};

        // Create a small reference model, save its weights, then inject them
        // into a factory-created model via FP32 QuantizedWeightStorage.
        let config = MambaConfig {
            input_dim: 1,
            hidden_dim: 32,
            state_dim: 8,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 1,
            dropout: 0.0,
            use_mamba2: false,
        };

        let reference = Mamba::new(config.clone()).expect("reference model");

        // Save reference weights to temp file (unique per invocation to avoid
        // races when multiple test threads call this function simultaneously).
        use std::sync::atomic::{AtomicU64, Ordering};
        static WS_B_COUNTER: AtomicU64 = AtomicU64::new(0);
        let uid = WS_B_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut tmp_path = std::env::temp_dir();
        tmp_path.push(format!("kizzasi_factory_ws_b_test_mamba_{}.json", uid));
        reference
            .save_weights_json(&tmp_path)
            .expect("save_weights_json");

        // Read back the JSON and build QuantizedWeightStorage::FP32 entries
        let file = std::fs::File::open(&tmp_path).expect("open temp file");
        let f32_map: HashMap<String, Vec<f32>> =
            serde_json::from_reader(file).expect("deserialise");
        let _ = std::fs::remove_file(&tmp_path);

        let mut quant_weights: HashMap<String, QuantizedWeightStorage> = HashMap::new();
        for (k, v) in f32_map {
            let len = v.len();
            // Reshape to a 1×N array (flat representation matches load_weights_json contract)
            let arr = Array2::from_shape_vec((1, len), v).expect("reshape to Array2");
            quant_weights.insert(k, QuantizedWeightStorage::FP32(arr));
        }

        // Create via factory — weight injection must not error
        let result = ModelFactory::create_mamba(config, quant_weights);
        assert!(
            result.is_ok(),
            "create_mamba with weights should succeed: {:?}",
            result.err()
        );
    }

    /// Verify the full roundtrip: save → factory-inject → verify model exists.
    #[cfg(feature = "mamba")]
    #[test]
    fn test_roundtrip_factory_save_load() {
        let _guard = factory_injection_lock();
        use crate::dynamic_quantization::QuantizedWeightStorage;
        use crate::mamba::{Mamba, MambaConfig};

        let config = MambaConfig {
            input_dim: 1,
            hidden_dim: 32,
            state_dim: 8,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 1,
            dropout: 0.0,
            use_mamba2: false,
        };

        // Step 1: save reference weights (unique path to avoid concurrent test races)
        use std::sync::atomic::{AtomicU64, Ordering};
        static ROUNDTRIP_COUNTER: AtomicU64 = AtomicU64::new(0);
        let uid = ROUNDTRIP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let reference = Mamba::new(config.clone()).expect("reference");
        let mut save_path = std::env::temp_dir();
        save_path.push(format!("kizzasi_factory_roundtrip_test_{}.json", uid));
        reference.save_weights_json(&save_path).expect("save");

        // Step 2: read back
        let file = std::fs::File::open(&save_path).expect("open");
        let f32_map: HashMap<String, Vec<f32>> = serde_json::from_reader(file).expect("deser");
        let _ = std::fs::remove_file(&save_path);

        let key_count = f32_map.len();
        assert!(key_count > 0, "saved weights must be non-empty");

        // Step 3: wrap in QuantizedWeightStorage and inject via factory
        let mut quant_weights: HashMap<String, QuantizedWeightStorage> = HashMap::new();
        for (k, v) in f32_map {
            let len = v.len();
            let arr = Array2::from_shape_vec((1, len), v).expect("reshape");
            quant_weights.insert(k, QuantizedWeightStorage::FP32(arr));
        }

        let model = ModelFactory::create_mamba(config, quant_weights)
            .expect("factory round-trip must succeed");

        // Step 4: verify model is functional — hidden_dim matches
        assert_eq!(model.hidden_dim(), 32);
    }

    /// Count the temp-directory entries the old on-disk injection path used to
    /// leave behind.
    fn factory_temp_weight_files() -> usize {
        let dir = std::env::temp_dir();
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return 0;
        };
        entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("kizzasi_factory_weights_")
            })
            .count()
    }

    /// The factory must inject weights in-process.
    ///
    /// It used to serialise the whole parameter set to a temporary JSON file
    /// and read it straight back — an f32 → decimal-text → f32 round-trip that
    /// needed a multiple of the model size in free temp space and leaked the
    /// file if the process died mid-load.
    #[cfg(feature = "mamba")]
    #[test]
    fn test_factory_injection_writes_no_temp_files() {
        let _guard = factory_injection_lock();
        use crate::dynamic_quantization::QuantizedWeightStorage;
        use crate::mamba::MambaConfig;

        let config = MambaConfig {
            input_dim: 2,
            hidden_dim: 16,
            state_dim: 4,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 1,
            dropout: 0.0,
            use_mamba2: false,
        };

        let mut quant_weights: HashMap<String, QuantizedWeightStorage> = HashMap::new();
        let values: Vec<f32> = (0..(config.input_dim * config.hidden_dim))
            .map(|i| i as f32 * 0.25)
            .collect();
        let arr = Array2::from_shape_vec((1, values.len()), values).expect("reshape");
        quant_weights.insert("input_proj".to_string(), QuantizedWeightStorage::FP32(arr));

        let before = factory_temp_weight_files();
        let model = ModelFactory::create_mamba(config, quant_weights).expect("create_mamba");
        let after = factory_temp_weight_files();

        assert_eq!(
            before, after,
            "factory must not stage weights through a temp file"
        );
        assert_eq!(model.hidden_dim(), 16);
    }

    /// A malformed weight entry must surface as a typed error, proving the map
    /// really reaches the model's `load_weights_map` rather than being dropped.
    #[cfg(feature = "mamba")]
    #[test]
    fn test_factory_injection_reports_shape_mismatch() {
        use crate::dynamic_quantization::QuantizedWeightStorage;
        use crate::mamba::MambaConfig;

        let config = MambaConfig {
            input_dim: 2,
            hidden_dim: 16,
            state_dim: 4,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 1,
            dropout: 0.0,
            use_mamba2: false,
        };

        let mut quant_weights: HashMap<String, QuantizedWeightStorage> = HashMap::new();
        // input_proj must be 2×16 = 32 values; give it 5.
        let arr = Array2::from_shape_vec((1, 5), vec![0.0f32; 5]).expect("reshape");
        quant_weights.insert("input_proj".to_string(), QuantizedWeightStorage::FP32(arr));

        let msg = match ModelFactory::create_mamba(config, quant_weights) {
            Ok(_) => panic!("a wrong-shaped weight must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("input_proj"),
            "error should name the offending tensor, got: {msg}"
        );
    }

    /// A weight map whose names match nothing must be an error, not a silent
    /// "injected successfully" over a randomly-initialised model.
    #[cfg(feature = "mamba")]
    #[test]
    fn test_factory_rejects_weight_map_that_matches_nothing() {
        use crate::dynamic_quantization::QuantizedWeightStorage;
        use crate::mamba::MambaConfig;

        let config = MambaConfig {
            input_dim: 2,
            hidden_dim: 16,
            state_dim: 4,
            expand_factor: 2,
            conv_kernel_size: 4,
            num_layers: 1,
            dropout: 0.0,
            use_mamba2: false,
        };

        let mut quant_weights: HashMap<String, QuantizedWeightStorage> = HashMap::new();
        let arr = Array2::from_shape_vec((1, 4), vec![0.0f32; 4]).expect("reshape");
        quant_weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            QuantizedWeightStorage::FP32(arr),
        );

        let msg = match ModelFactory::create_mamba(config, quant_weights) {
            Ok(_) => panic!("a weight map matching no parameter must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("matched"),
            "error should explain that nothing matched, got: {msg}"
        );
    }

    /// The same guard must protect S5, whose parameter names differ entirely
    /// from the HuggingFace conventions callers usually start from.
    #[test]
    fn test_factory_rejects_unmatched_weights_for_s5() {
        use crate::dynamic_quantization::QuantizedWeightStorage;

        let config = S5Config::new(2, 8, 1);
        let mut quant_weights: HashMap<String, QuantizedWeightStorage> = HashMap::new();
        let arr = Array2::from_shape_vec((1, 4), vec![1.0f32; 4]).expect("reshape");
        quant_weights.insert(
            "backbone.layers.0.mixer.A_log".to_string(),
            QuantizedWeightStorage::FP32(arr),
        );

        assert!(
            ModelFactory::create_s5(config, quant_weights).is_err(),
            "S5 must not report success for a weight map it ignored entirely"
        );
    }

    /// Test that PyTorchConverter weight name conversion works in pytorch_compat.
    #[test]
    fn test_pytorch_compat_weight_conversion() {
        use crate::pytorch_compat::PyTorchConverter;

        let converter = PyTorchConverter::new();

        // Standard Mamba-style name
        let mapped = converter.map_name("mixer.in_proj.weight");
        // After applying default mappings: "mixer.in_proj" → "in_proj"
        // then "." → "_"
        assert!(
            mapped.contains("in_proj"),
            "mapped name should contain 'in_proj', got: {}",
            mapped
        );
    }
}
