//! HuggingFace Model Loading and Weight Conversion
//!
//! This module provides utilities to load models from HuggingFace Hub and automatically
//! convert weight names and shapes to match Kizzasi's internal format.
//!
//! # Architecture Differences
//!
//! HuggingFace models use different naming conventions and weight structures than Kizzasi:
//!
//! ## Mamba Models
//!
//! ```text
//! HuggingFace                          Kizzasi
//! ─────────────────────────────────────────────────────────
//! backbone.embeddings                  → input_proj
//! backbone.layers.{i}.norm            → layers.{i}.norm
//! backbone.layers.{i}.mixer.in_proj   → layers.{i}.in_proj
//! backbone.layers.{i}.mixer.conv1d    → layers.{i}.conv
//! backbone.layers.{i}.mixer.x_proj    → split into:
//!                                        - layers.{i}.ssm.delta_proj (first chunk)
//!                                        - layers.{i}.ssm.b_proj (middle chunk)
//!                                        - layers.{i}.ssm.c_proj (last chunk)
//! backbone.layers.{i}.mixer.dt_proj   → layers.{i}.ssm.delta_proj
//! backbone.layers.{i}.mixer.A_log     → layers.{i}.ssm.log_a
//! backbone.layers.{i}.mixer.D         → layers.{i}.ssm.d_skip
//! backbone.layers.{i}.mixer.out_proj  → layers.{i}.out_proj
//! lm_head.weight                       → output_proj
//! ```
//!
//! ## RWKV Models
//!
//! ```text
//! HuggingFace                          Kizzasi
//! ─────────────────────────────────────────────────────────
//! emb.weight                           → input_proj
//! blocks.{i}.ln1.weight               → layers.{i}.norm.weight
//! blocks.{i}.att.time_decay           → layers.{i}.time_mix.decay
//! blocks.{i}.att.time_first           → layers.{i}.time_mix.first
//! blocks.{i}.att.key.weight           → layers.{i}.time_mix.key_weight
//! blocks.{i}.att.value.weight         → layers.{i}.time_mix.value_weight
//! blocks.{i}.att.receptance.weight    → layers.{i}.time_mix.receptance_weight
//! blocks.{i}.att.output.weight        → layers.{i}.time_mix.output_weight
//! blocks.{i}.ffn.key.weight           → layers.{i}.channel_mix.key_weight
//! blocks.{i}.ffn.value.weight         → layers.{i}.channel_mix.value_weight
//! blocks.{i}.ffn.receptance.weight    → layers.{i}.channel_mix.receptance_weight
//! head.weight                          → output_proj
//! ```

use crate::dynamic_quantization::{DynamicQuantizer, QuantStrategy, QuantizedWeightStorage};
use crate::error::{ModelError, ModelResult};
use crate::huggingface::{HuggingFaceHub, ModelConfig};
use crate::loader::ModelLoader;
use scirs2_core::ndarray::{s, Array2};
use std::collections::HashMap;

/// Weight conversion strategy for different model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionStrategy {
    /// HuggingFace Mamba format → Kizzasi Mamba
    MambaHF,
    /// HuggingFace RWKV format → Kizzasi RWKV
    RwkvHF,
    /// Direct mapping (no conversion needed)
    Direct,
}

/// HuggingFace model loader with automatic weight conversion
pub struct HuggingFaceModelLoader {
    /// HuggingFace Hub client
    hub: HuggingFaceHub,
    /// Conversion strategy
    strategy: ConversionStrategy,
}

impl HuggingFaceModelLoader {
    /// Create a new HuggingFace model loader
    pub fn new() -> ModelResult<Self> {
        Ok(Self {
            hub: HuggingFaceHub::new()?,
            strategy: ConversionStrategy::Direct,
        })
    }

    /// Set the conversion strategy
    pub fn with_strategy(mut self, strategy: ConversionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set authentication token
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.hub = self.hub.with_token(token);
        self
    }

    /// Load model configuration and automatically detect conversion strategy
    pub async fn detect_and_load(
        &mut self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> ModelResult<(ModelConfig, HashMap<String, Array2<f32>>)> {
        // Load configuration
        let config = self.hub.load_config(repo_id, revision).await?;

        // Detect architecture
        if let Some(model_type) = &config.model_type {
            self.strategy = match model_type.to_lowercase().as_str() {
                "mamba" | "mamba2" => ConversionStrategy::MambaHF,
                "rwkv" | "rwkv6" | "rwkv7" => ConversionStrategy::RwkvHF,
                _ => ConversionStrategy::Direct,
            };
        } else if let Some(arch) = &config.architecture {
            // Check first architecture string
            if let Some(arch_name) = arch.first() {
                self.strategy = match arch_name.to_lowercase().as_str() {
                    s if s.contains("mamba") => ConversionStrategy::MambaHF,
                    s if s.contains("rwkv") => ConversionStrategy::RwkvHF,
                    _ => ConversionStrategy::Direct,
                };
            }
        }

        tracing::info!(
            "Detected model type: {:?}, using strategy: {:?}",
            config.model_type,
            self.strategy
        );

        // Load and convert weights
        let weights = self.load_and_convert_weights(repo_id, revision).await?;

        Ok((config, weights))
    }

    /// Load weights from HuggingFace and convert to Kizzasi format
    pub async fn load_and_convert_weights(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> ModelResult<HashMap<String, Array2<f32>>> {
        // Download model using HuggingFaceHub
        let loader = self.hub.load_model_loader(repo_id, revision).await?;

        // Convert weights based on strategy
        match self.strategy {
            ConversionStrategy::MambaHF => self.convert_mamba_weights(&loader),
            ConversionStrategy::RwkvHF => self.convert_rwkv_weights(&loader),
            ConversionStrategy::Direct => {
                // Load all 2D tensors directly
                let mut weights = HashMap::new();
                for name in loader.list_tensors() {
                    if let Ok(tensor) = loader.load_array2(&name) {
                        weights.insert(name, tensor);
                    }
                }
                Ok(weights)
            }
        }
    }

    /// Convert HuggingFace Mamba weights to Kizzasi format
    fn convert_mamba_weights(
        &self,
        loader: &ModelLoader,
    ) -> ModelResult<HashMap<String, Array2<f32>>> {
        let mut kizzasi_weights = HashMap::new();
        let tensor_names = loader.list_tensors();

        tracing::info!(
            "Converting {} HuggingFace Mamba tensors to Kizzasi format",
            tensor_names.len()
        );

        for hf_name in tensor_names {
            // Convert name from HuggingFace to Kizzasi format
            let kizzasi_name = self.convert_mamba_name(&hf_name);

            // Special handling for tensors that need reshaping
            if hf_name.contains("mixer.x_proj") && hf_name.ends_with(".weight") {
                // HuggingFace x_proj needs to be split into dt, B, C projections
                // x_proj shape: [intermediate_size, dt_rank + 2*state_size]
                // Split into:
                //   - delta_proj: [intermediate_size, dt_rank]
                //   - b_proj: [intermediate_size, state_size]
                //   - c_proj: [intermediate_size, state_size]

                if let Ok(x_proj) = loader.load_array2(&hf_name) {
                    let (_intermediate_size, combined_dim) = x_proj.dim();

                    // Try to infer dimensions (this is heuristic, may need config)
                    // Common Mamba configs: dt_rank = 64, state_size = 16
                    // So combined_dim = 64 + 16*2 = 96
                    let state_size = 16; // TODO: Get from config
                    let dt_rank = combined_dim - 2 * state_size;

                    if dt_rank > 0 && dt_rank + 2 * state_size == combined_dim {
                        // Extract delta_proj
                        let delta_proj = x_proj.slice(s![.., ..dt_rank]).to_owned();
                        let delta_name = kizzasi_name.replace("x_proj", "ssm.delta_proj");
                        kizzasi_weights.insert(delta_name, delta_proj);

                        // Extract b_proj
                        let b_proj = x_proj
                            .slice(s![.., dt_rank..dt_rank + state_size])
                            .to_owned();
                        let b_name = kizzasi_name.replace("x_proj", "ssm.b_proj");
                        kizzasi_weights.insert(b_name, b_proj);

                        // Extract c_proj
                        let c_proj = x_proj.slice(s![.., dt_rank + state_size..]).to_owned();
                        let c_name = kizzasi_name.replace("x_proj", "ssm.c_proj");
                        kizzasi_weights.insert(c_name, c_proj);

                        tracing::debug!("Split x_proj {} into delta/b/c projections (dt_rank={}, state_size={})",
                                       hf_name, dt_rank, state_size);
                        continue;
                    } else {
                        tracing::warn!("Could not infer dimensions for x_proj splitting: combined_dim={}, inferred dt_rank={}, state_size={}",
                                      combined_dim, dt_rank, state_size);
                    }
                }
            }

            // Standard weight loading
            if let Ok(tensor) = loader.load_array2(&hf_name) {
                kizzasi_weights.insert(kizzasi_name, tensor);
            } else if let Ok(tensor) = loader.load_array1(&hf_name) {
                // Convert 1D to 2D for compatibility
                let len = tensor.len();
                let tensor_2d = tensor
                    .to_shape((len, 1))
                    .map_err(|e| {
                        ModelError::simple_load_error(format!("Failed to reshape tensor: {}", e))
                    })?
                    .to_owned();
                kizzasi_weights.insert(kizzasi_name, tensor_2d);
            }
        }

        tracing::info!("Converted to {} Kizzasi tensors", kizzasi_weights.len());
        Ok(kizzasi_weights)
    }

    /// Convert HuggingFace Mamba tensor name to Kizzasi format
    fn convert_mamba_name(&self, hf_name: &str) -> String {
        let mut name = hf_name.to_string();

        // Backbone prefix removal
        name = name.replace("backbone.", "");

        // Embedding layer
        if name.starts_with("embeddings") || name == "embedding.weight" {
            return "input_proj".to_string();
        }

        // Output head
        if name.starts_with("lm_head") {
            return name.replace("lm_head", "output_proj");
        }

        // Layer-level conversions
        name = name.replace(".mixer.", ".");
        name = name.replace("conv1d.", "conv.");

        // SSM-specific conversions
        name = name.replace(".A_log", ".ssm.log_a");
        name = name.replace(".D.", ".ssm.d_skip.");
        name = name.replace(".D", ".ssm.d_skip");
        name = name.replace("dt_proj", "ssm.dt_proj");

        // x_proj handled separately due to splitting
        // Just rename for now, splitting happens in convert_mamba_weights
        name = name.replace("x_proj", "ssm.x_proj");

        name
    }

    /// Convert HuggingFace RWKV weights to Kizzasi format
    fn convert_rwkv_weights(
        &self,
        loader: &ModelLoader,
    ) -> ModelResult<HashMap<String, Array2<f32>>> {
        let mut kizzasi_weights = HashMap::new();
        let tensor_names = loader.list_tensors();

        tracing::info!(
            "Converting {} HuggingFace RWKV tensors to Kizzasi format",
            tensor_names.len()
        );

        for hf_name in tensor_names {
            let kizzasi_name = self.convert_rwkv_name(&hf_name);

            // Load tensor (try 2D first, then 1D)
            if let Ok(tensor) = loader.load_array2(&hf_name) {
                kizzasi_weights.insert(kizzasi_name, tensor);
            } else if let Ok(tensor) = loader.load_array1(&hf_name) {
                // Convert 1D to 2D for compatibility
                let len = tensor.len();
                let tensor_2d = tensor
                    .to_shape((len, 1))
                    .map_err(|e| {
                        ModelError::simple_load_error(format!("Failed to reshape tensor: {}", e))
                    })?
                    .to_owned();
                kizzasi_weights.insert(kizzasi_name, tensor_2d);
            }
        }

        tracing::info!("Converted to {} Kizzasi tensors", kizzasi_weights.len());
        Ok(kizzasi_weights)
    }

    /// Convert HuggingFace RWKV tensor name to Kizzasi format
    fn convert_rwkv_name(&self, hf_name: &str) -> String {
        let mut name = hf_name.to_string();

        // Embedding layer
        if name.starts_with("emb.weight") || name == "emb" {
            return "input_proj".to_string();
        }

        // Output head
        if name.starts_with("head.weight") || name.starts_with("head.") {
            return name.replace("head", "output_proj");
        }

        // Block → Layer renaming
        name = name.replace("blocks.", "layers.");

        // Layer normalization
        name = name.replace("ln1.", "norm.");
        name = name.replace("ln2.", "norm2.");

        // Attention (time mixing) conversions
        name = name.replace(".att.", ".time_mix.");
        name = name.replace("time_decay", "decay");
        name = name.replace("time_first", "first");

        // FFN (channel mixing) conversions
        name = name.replace(".ffn.", ".channel_mix.");

        name
    }

    /// Get the underlying HuggingFace Hub client
    pub fn hub(&self) -> &HuggingFaceHub {
        &self.hub
    }

    /// Get the current conversion strategy
    pub fn strategy(&self) -> ConversionStrategy {
        self.strategy
    }

    /// Load and quantize model weights from HuggingFace
    ///
    /// # Arguments
    ///
    /// * `repo_id` - Repository ID (e.g., "state-spaces/mamba-130m")
    /// * `revision` - Optional git revision (defaults to "main")
    /// * `quant_strategy` - Quantization strategy to apply
    ///
    /// # Returns
    ///
    /// Tuple of (model config, quantized weights, quantization stats)
    pub async fn load_and_quantize(
        &mut self,
        repo_id: &str,
        revision: Option<&str>,
        quant_strategy: QuantStrategy,
    ) -> ModelResult<(
        ModelConfig,
        HashMap<String, QuantizedWeightStorage>,
        crate::dynamic_quantization::QuantizationStats,
    )> {
        // Load config and weights using detect_and_load for automatic strategy detection
        let (config, weights) = self.detect_and_load(repo_id, revision).await?;

        // Create quantizer
        let quantizer = DynamicQuantizer::new().with_strategy(quant_strategy);

        // Quantize weights
        let quantized_weights = quantizer.quantize_weights(&weights)?;

        // Calculate statistics
        let stats = quantizer.calculate_memory_savings(&weights, &quantized_weights);

        tracing::info!(
            "Quantized {} weights using {:?}: {:.2}x compression ({} → {})",
            quantized_weights.len(),
            quant_strategy,
            stats.compression_ratio,
            crate::dynamic_quantization::QuantizationStats::format_size(stats.original_size_bytes),
            crate::dynamic_quantization::QuantizationStats::format_size(stats.quantized_size_bytes)
        );

        Ok((config, quantized_weights, stats))
    }

    /// Load model with automatic quantization detection
    ///
    /// Applies mixed-precision quantization based on layer sensitivity
    pub async fn load_with_auto_quantization(
        &mut self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> ModelResult<(
        ModelConfig,
        HashMap<String, QuantizedWeightStorage>,
        crate::dynamic_quantization::QuantizationStats,
    )> {
        // First detect architecture
        let (_config, _) = self.detect_and_load(repo_id, revision).await?;

        // Use mixed precision by default for best accuracy/size tradeoff
        self.load_and_quantize(repo_id, revision, QuantStrategy::MixedPrecision)
            .await
    }

    /// Complete end-to-end pipeline: download → convert → quantize → instantiate
    ///
    /// This is the recommended way to load models from HuggingFace.
    ///
    /// # Arguments
    ///
    /// * `repo_id` - HuggingFace model repository ID
    /// * `revision` - Optional revision/branch (defaults to "main")
    /// * `quant_strategy` - Quantization strategy (or use `load_model_auto` for defaults)
    ///
    /// # Returns
    ///
    /// A boxed trait object implementing `AutoregressiveModel`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use kizzasi_model::huggingface_loader::HuggingFaceModelLoader;
    /// use kizzasi_model::dynamic_quantization::QuantStrategy;
    /// use kizzasi_core::SignalPredictor;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut loader = HuggingFaceModelLoader::new()?;
    /// let model = loader.load_model(
    ///     "state-spaces/mamba-130m",
    ///     None,
    ///     QuantStrategy::FP16
    /// ).await?;
    ///
    /// // Use model for inference
    /// let output = model.predict(&input)?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load_model(
        &mut self,
        repo_id: &str,
        revision: Option<&str>,
        quant_strategy: QuantStrategy,
    ) -> ModelResult<Box<dyn crate::AutoregressiveModel>> {
        use crate::factory::ModelFactory;

        tracing::info!("Loading model '{}' end-to-end", repo_id);

        // Load, convert, and quantize
        let (config, quantized_weights, stats) = self
            .load_and_quantize(repo_id, revision, quant_strategy)
            .await?;

        tracing::info!(
            "Creating model instance: compression={:.1}x, memory_saved={:.2} bytes",
            stats.compression_ratio,
            stats.memory_saved_bytes
        );

        // Instantiate model
        let model = ModelFactory::create_from_config(&config, quantized_weights)?;

        tracing::info!("Model loaded successfully: {}", model.model_type());

        Ok(model)
    }

    /// Load model with automatic defaults (recommended)
    ///
    /// Uses `MixedPrecision` quantization for optimal accuracy/size tradeoff.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use kizzasi_model::huggingface_loader::HuggingFaceModelLoader;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut loader = HuggingFaceModelLoader::new()?;
    /// let model = loader.load_model_auto("state-spaces/mamba-130m", None).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load_model_auto(
        &mut self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> ModelResult<Box<dyn crate::AutoregressiveModel>> {
        self.load_model(repo_id, revision, QuantStrategy::MixedPrecision)
            .await
    }
}

impl Default for HuggingFaceModelLoader {
    fn default() -> Self {
        Self::new().expect("Failed to create default HuggingFaceModelLoader")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_strategy() {
        let loader = HuggingFaceModelLoader::new().unwrap();
        assert_eq!(loader.strategy(), ConversionStrategy::Direct);

        let loader = loader.with_strategy(ConversionStrategy::MambaHF);
        assert_eq!(loader.strategy(), ConversionStrategy::MambaHF);
    }

    #[test]
    fn test_mamba_name_conversion() {
        let loader = HuggingFaceModelLoader::new().unwrap();

        // Test embeddings
        assert_eq!(
            loader.convert_mamba_name("backbone.embeddings.weight"),
            "input_proj"
        );

        // Test layer norm
        assert_eq!(
            loader.convert_mamba_name("backbone.layers.0.norm.weight"),
            "layers.0.norm.weight"
        );

        // Test mixer components
        assert_eq!(
            loader.convert_mamba_name("backbone.layers.0.mixer.in_proj.weight"),
            "layers.0.in_proj.weight"
        );

        assert_eq!(
            loader.convert_mamba_name("backbone.layers.0.mixer.conv1d.weight"),
            "layers.0.conv.weight"
        );

        // Test SSM parameters
        assert_eq!(
            loader.convert_mamba_name("backbone.layers.0.mixer.A_log"),
            "layers.0.ssm.log_a"
        );

        assert_eq!(
            loader.convert_mamba_name("backbone.layers.0.mixer.D"),
            "layers.0.ssm.d_skip"
        );

        // Test output head
        assert_eq!(
            loader.convert_mamba_name("lm_head.weight"),
            "output_proj.weight"
        );
    }

    #[test]
    fn test_rwkv_name_conversion() {
        let loader = HuggingFaceModelLoader::new().unwrap();

        // Test embeddings
        assert_eq!(loader.convert_rwkv_name("emb.weight"), "input_proj");

        // Test layer components
        assert_eq!(
            loader.convert_rwkv_name("blocks.0.ln1.weight"),
            "layers.0.norm.weight"
        );

        // Test attention (time mixing)
        assert_eq!(
            loader.convert_rwkv_name("blocks.0.att.time_decay"),
            "layers.0.time_mix.decay"
        );

        assert_eq!(
            loader.convert_rwkv_name("blocks.0.att.key.weight"),
            "layers.0.time_mix.key.weight"
        );

        // Test FFN (channel mixing)
        assert_eq!(
            loader.convert_rwkv_name("blocks.0.ffn.key.weight"),
            "layers.0.channel_mix.key.weight"
        );

        // Test output head
        assert_eq!(
            loader.convert_rwkv_name("head.weight"),
            "output_proj.weight"
        );
    }

    #[test]
    fn test_with_token() {
        let loader = HuggingFaceModelLoader::new()
            .unwrap()
            .with_token("test_token");

        assert_eq!(loader.hub().token.as_deref(), Some("test_token"));
    }
}
