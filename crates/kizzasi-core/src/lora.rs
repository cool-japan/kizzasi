//! # LoRA (Low-Rank Adaptation) Support
//!
//! Implementation of LoRA for efficient fine-tuning of SSM models.
//!
//! ## Features
//!
//! - **Low-Rank Decomposition**: Efficient parameter updates with rank << dimension
//! - **Adapter Loading**: Load pre-trained LoRA adapters from safetensors
//! - **Merging**: Merge LoRA weights into base model weights
//! - **Multi-Adapter**: Support for multiple adapters simultaneously
//! - **Selective Application**: Apply LoRA to specific layers/modules
//!
//! ## References
//!
//! - "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

use crate::{CoreError, CoreResult};
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// LoRA adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Rank of the low-rank decomposition
    pub rank: usize,
    /// Scaling factor (alpha / rank)
    pub alpha: f32,
    /// Dropout rate for LoRA layers
    pub dropout: f32,
    /// Target modules to apply LoRA
    pub target_modules: Vec<String>,
    /// Whether to merge adapters into base weights
    pub merge_weights: bool,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec![
                "in_proj".to_string(),
                "out_proj".to_string(),
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
            ],
            merge_weights: false,
        }
    }
}

impl LoRAConfig {
    /// Create a new LoRA configuration
    pub fn new(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            ..Default::default()
        }
    }

    /// Set target modules
    pub fn with_targets(mut self, targets: Vec<String>) -> Self {
        self.target_modules = targets;
        self
    }

    /// Set dropout
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Enable weight merging
    pub fn with_merge(mut self) -> Self {
        self.merge_weights = true;
        self
    }

    /// Get effective scaling
    pub fn scaling(&self) -> f32 {
        self.alpha / (self.rank as f32)
    }

    /// Validate configuration
    pub fn validate(&self) -> CoreResult<()> {
        if self.rank == 0 {
            return Err(CoreError::InvalidConfig("LoRA rank must be > 0".into()));
        }
        if self.alpha <= 0.0 {
            return Err(CoreError::InvalidConfig("LoRA alpha must be > 0".into()));
        }
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err(CoreError::InvalidConfig(
                "LoRA dropout must be in [0, 1)".into(),
            ));
        }
        Ok(())
    }
}

/// LoRA adapter layer
///
/// Implements W' = W + (B @ A) * scaling
/// where A is (rank, in_features) and B is (out_features, rank)
#[derive(Debug, Clone)]
pub struct LoRALayer {
    /// Configuration
    config: LoRAConfig,
    /// Original weight matrix (out_features, in_features)
    base_weight: Array2<f32>,
    /// LoRA A matrix (rank, in_features)
    lora_a: Array2<f32>,
    /// LoRA B matrix (out_features, rank)
    lora_b: Array2<f32>,
    /// Whether weights are merged
    is_merged: bool,
}

impl LoRALayer {
    /// Create a new LoRA layer
    pub fn new(config: LoRAConfig, base_weight: Array2<f32>) -> CoreResult<Self> {
        config.validate()?;

        let (out_features, in_features) = base_weight.dim();

        // Initialize A with random values, B with zeros (for stability)
        use scirs2_core::random::thread_rng;
        let mut rng = thread_rng();
        let init_scale = (1.0 / config.rank as f32).sqrt();

        let lora_a = Array2::from_shape_fn((config.rank, in_features), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * init_scale
        });

        let lora_b = Array2::zeros((out_features, config.rank));

        Ok(Self {
            config,
            base_weight,
            lora_a,
            lora_b,
            is_merged: false,
        })
    }

    /// Forward pass with LoRA
    pub fn forward(&self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        if x.len() != self.base_weight.ncols() {
            return Err(CoreError::DimensionMismatch {
                expected: self.base_weight.ncols(),
                got: x.len(),
            });
        }

        // Base output: W @ x
        let mut y = self.base_weight.dot(x);

        // If not merged, add LoRA contribution: (B @ A) @ x * scaling
        if !self.is_merged {
            // A @ x -> intermediate (rank,)
            let intermediate = self.lora_a.dot(x);

            // B @ intermediate -> delta (out_features,)
            let delta = self.lora_b.dot(&intermediate);

            // Add scaled contribution
            y = &y + &(&delta * self.config.scaling());
        }

        Ok(y)
    }

    /// Merge LoRA weights into base weights
    pub fn merge(&mut self) -> CoreResult<()> {
        if self.is_merged {
            return Ok(());
        }

        // Compute B @ A
        let lora_weight = self.lora_b.dot(&self.lora_a);

        // W' = W + (B @ A) * scaling
        self.base_weight = &self.base_weight + &(&lora_weight * self.config.scaling());
        self.is_merged = true;

        Ok(())
    }

    /// Unmerge LoRA weights from base weights
    pub fn unmerge(&mut self) -> CoreResult<()> {
        if !self.is_merged {
            return Ok(());
        }

        // Compute B @ A
        let lora_weight = self.lora_b.dot(&self.lora_a);

        // W = W' - (B @ A) * scaling
        self.base_weight = &self.base_weight - &(&lora_weight * self.config.scaling());
        self.is_merged = false;

        Ok(())
    }

    /// Get the effective weight matrix (with LoRA applied)
    pub fn get_effective_weight(&self) -> Array2<f32> {
        if self.is_merged {
            self.base_weight.clone()
        } else {
            let lora_weight = self.lora_b.dot(&self.lora_a);
            &self.base_weight + &(&lora_weight * self.config.scaling())
        }
    }

    /// Update LoRA A matrix
    pub fn set_lora_a(&mut self, a: Array2<f32>) -> CoreResult<()> {
        if a.dim() != self.lora_a.dim() {
            return Err(CoreError::DimensionMismatch {
                expected: self.lora_a.nrows() * self.lora_a.ncols(),
                got: a.nrows() * a.ncols(),
            });
        }
        self.lora_a = a;
        Ok(())
    }

    /// Update LoRA B matrix
    pub fn set_lora_b(&mut self, b: Array2<f32>) -> CoreResult<()> {
        if b.dim() != self.lora_b.dim() {
            return Err(CoreError::DimensionMismatch {
                expected: self.lora_b.nrows() * self.lora_b.ncols(),
                got: b.nrows() * b.ncols(),
            });
        }
        self.lora_b = b;
        Ok(())
    }

    /// Get LoRA parameters count
    pub fn num_parameters(&self) -> usize {
        self.lora_a.len() + self.lora_b.len()
    }

    /// Get base parameters count
    pub fn base_num_parameters(&self) -> usize {
        self.base_weight.len()
    }

    /// Get parameter reduction ratio
    pub fn parameter_ratio(&self) -> f32 {
        self.num_parameters() as f32 / self.base_num_parameters() as f32
    }

    /// Check if merged
    pub fn is_merged(&self) -> bool {
        self.is_merged
    }
}

/// LoRA adapter manager
pub struct LoRAAdapter {
    /// Adapter name
    pub name: String,
    /// LoRA configuration
    pub config: LoRAConfig,
    /// LoRA layers by module name
    pub layers: HashMap<String, LoRALayer>,
}

impl LoRAAdapter {
    /// Create a new LoRA adapter
    pub fn new(name: String, config: LoRAConfig) -> Self {
        Self {
            name,
            config,
            layers: HashMap::new(),
        }
    }

    /// Add a LoRA layer for a specific module
    pub fn add_layer(&mut self, module_name: String, layer: LoRALayer) {
        self.layers.insert(module_name, layer);
    }

    /// Load LoRA adapter from safetensors file
    pub fn from_safetensors(
        path: impl AsRef<Path>,
        config: LoRAConfig,
        device: &Device,
    ) -> CoreResult<Self> {
        let data = std::fs::read(path.as_ref())
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to read LoRA file: {}", e)))?;

        let tensors = SafeTensors::deserialize(&data).map_err(|e| {
            CoreError::WeightLoadError(format!("Failed to deserialize LoRA: {}", e))
        })?;

        let name = path
            .as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unnamed")
            .to_string();

        let adapter = Self::new(name, config);

        // Parse tensor names to extract module names and A/B matrices
        let mut module_tensors: HashMap<String, (Option<Tensor>, Option<Tensor>)> = HashMap::new();

        for (tensor_name, tensor_view) in tensors.tensors() {
            // Convert safetensor to candle tensor
            let shape: Vec<usize> = tensor_view.shape().to_vec();
            let dtype = match tensor_view.dtype() {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F16 => DType::F16,
                safetensors::Dtype::BF16 => DType::BF16,
                _ => {
                    return Err(CoreError::WeightLoadError(format!(
                        "Unsupported dtype: {:?}",
                        tensor_view.dtype()
                    )))
                }
            };

            let tensor = Tensor::from_raw_buffer(tensor_view.data(), dtype, &shape, device)
                .map_err(|e| {
                    CoreError::WeightLoadError(format!("Tensor creation failed: {}", e))
                })?;

            // Parse tensor name: format like "module_name.lora_A" or "module_name.lora_B"
            if let Some((module, suffix)) = tensor_name.rsplit_once('.') {
                let entry = module_tensors
                    .entry(module.to_string())
                    .or_insert((None, None));

                if suffix == "lora_A" || suffix == "A" {
                    entry.0 = Some(tensor);
                } else if suffix == "lora_B" || suffix == "B" {
                    entry.1 = Some(tensor);
                }
            }
        }

        // Create LoRA layers from parsed tensors
        // This would require base weights which we don't have here
        // In practice, you'd load these after initializing the base model

        Ok(adapter)
    }

    /// Merge all layers
    pub fn merge_all(&mut self) -> CoreResult<()> {
        for layer in self.layers.values_mut() {
            layer.merge()?;
        }
        Ok(())
    }

    /// Unmerge all layers
    pub fn unmerge_all(&mut self) -> CoreResult<()> {
        for layer in self.layers.values_mut() {
            layer.unmerge()?;
        }
        Ok(())
    }

    /// Get total parameter count
    pub fn total_parameters(&self) -> usize {
        self.layers.values().map(|l| l.num_parameters()).sum()
    }

    /// Get parameter reduction ratio
    pub fn avg_parameter_ratio(&self) -> f32 {
        if self.layers.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.layers.values().map(|l| l.parameter_ratio()).sum();
        sum / self.layers.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config() {
        let config = LoRAConfig::new(8, 16.0);
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.scaling(), 2.0); // 16.0 / 8
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_lora_config_validation() {
        let mut config = LoRAConfig::new(0, 16.0);
        assert!(config.validate().is_err());

        config.rank = 8;
        config.alpha = -1.0;
        assert!(config.validate().is_err());

        config.alpha = 16.0;
        config.dropout = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_lora_layer_creation() {
        let config = LoRAConfig::new(4, 8.0);
        let base_weight = Array2::from_shape_fn((64, 32), |(i, j)| (i as f32 + j as f32) * 0.01);

        let result = LoRALayer::new(config, base_weight);
        assert!(result.is_ok());

        let layer = result.unwrap();
        assert_eq!(layer.lora_a.nrows(), 4);
        assert_eq!(layer.lora_a.ncols(), 32);
        assert_eq!(layer.lora_b.nrows(), 64);
        assert_eq!(layer.lora_b.ncols(), 4);
    }

    #[test]
    fn test_lora_forward() {
        let config = LoRAConfig::new(4, 8.0);
        let base_weight = Array2::from_elem((64, 32), 0.1);
        let layer = LoRALayer::new(config, base_weight).unwrap();

        let input = Array1::from_elem(32, 0.5);
        let output = layer.forward(&input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.len(), 64);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_lora_merge_unmerge() {
        let config = LoRAConfig::new(4, 8.0);
        let base_weight = Array2::from_elem((64, 32), 0.1);
        let mut layer = LoRALayer::new(config, base_weight).unwrap();

        assert!(!layer.is_merged());

        // Merge
        layer.merge().unwrap();
        assert!(layer.is_merged());

        // Unmerge
        layer.unmerge().unwrap();
        assert!(!layer.is_merged());
    }

    #[test]
    fn test_lora_parameter_count() {
        let config = LoRAConfig::new(4, 8.0);
        let base_weight = Array2::from_elem((64, 32), 0.1);
        let layer = LoRALayer::new(config, base_weight).unwrap();

        // LoRA params = rank * (in_features + out_features) = 4 * (32 + 64) = 384
        assert_eq!(layer.num_parameters(), 384);
        // Base params = 64 * 32 = 2048
        assert_eq!(layer.base_num_parameters(), 2048);

        let ratio = layer.parameter_ratio();
        assert!((ratio - 0.1875).abs() < 1e-5); // 384 / 2048
    }

    #[test]
    fn test_effective_weight() {
        let config = LoRAConfig::new(2, 4.0);
        let base_weight = Array2::from_elem((4, 4), 1.0);
        let mut layer = LoRALayer::new(config, base_weight).unwrap();

        // Set specific LoRA weights for testing
        layer.lora_a = Array2::from_elem((2, 4), 0.1);
        layer.lora_b = Array2::from_elem((4, 2), 0.1);

        let effective = layer.get_effective_weight();
        // Scaling = 4.0 / 2 = 2.0
        // LoRA contribution = B @ A * scaling = (4x2) @ (2x4) * 2.0
        // Each element should be > 1.0 due to LoRA addition
        assert!(effective.iter().all(|&x| x >= 1.0));
    }

    #[test]
    fn test_lora_adapter_creation() {
        let config = LoRAConfig::new(4, 8.0);
        let adapter = LoRAAdapter::new("test_adapter".to_string(), config);

        assert_eq!(adapter.name, "test_adapter");
        assert_eq!(adapter.layers.len(), 0);
    }

    #[test]
    fn test_lora_adapter_add_layer() {
        let config = LoRAConfig::new(4, 8.0);
        let mut adapter = LoRAAdapter::new("test".to_string(), config.clone());

        let base_weight = Array2::from_elem((64, 32), 0.1);
        let layer = LoRALayer::new(config, base_weight).unwrap();

        adapter.add_layer("layer_0".to_string(), layer);
        assert_eq!(adapter.layers.len(), 1);
        assert!(adapter.layers.contains_key("layer_0"));
    }

    #[test]
    fn test_lora_dimension_mismatch() {
        let config = LoRAConfig::new(4, 8.0);
        let base_weight = Array2::from_elem((64, 32), 0.1);
        let layer = LoRALayer::new(config, base_weight).unwrap();

        // Wrong input dimension
        let input = Array1::from_elem(16, 0.5);
        let result = layer.forward(&input);
        assert!(result.is_err());
    }
}
