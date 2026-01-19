//! LoRA (Low-Rank Adaptation) adapter loading for inference
//!
//! This module provides support for loading and applying LoRA adapters
//! at inference time, allowing efficient model fine-tuning and adaptation.

use crate::error::{InferenceError, InferenceResult};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// LoRA adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of the low-rank matrices
    pub rank: usize,
    /// Scaling factor (alpha / rank)
    pub alpha: f32,
    /// Dropout rate for LoRA layers
    pub dropout: f32,
    /// Target modules to apply LoRA to
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        }
    }
}

impl LoraConfig {
    /// Create a new LoRA configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the rank
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    /// Set alpha (scaling factor)
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set dropout rate
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Add target module
    pub fn add_target_module(mut self, module: impl Into<String>) -> Self {
        self.target_modules.push(module.into());
        self
    }

    /// Get the effective scaling factor
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

/// A LoRA adapter consisting of two low-rank matrices
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Low-rank matrix A (rank × in_features)
    pub lora_a: Array2<f32>,
    /// Low-rank matrix B (out_features × rank)
    pub lora_b: Array2<f32>,
    /// Scaling factor
    pub scaling: f32,
    /// Adapter name/identifier
    pub name: String,
}

impl LoraAdapter {
    /// Create a new LoRA adapter
    pub fn new(
        lora_a: Array2<f32>,
        lora_b: Array2<f32>,
        scaling: f32,
        name: impl Into<String>,
    ) -> InferenceResult<Self> {
        // Validate dimensions: A is (rank, in_features), B is (out_features, rank)
        let rank_a = lora_a.nrows();
        let rank_b = lora_b.ncols();

        if rank_a != rank_b {
            return Err(InferenceError::DimensionMismatch {
                expected: rank_a,
                got: rank_b,
            });
        }

        Ok(Self {
            lora_a,
            lora_b,
            scaling,
            name: name.into(),
        })
    }

    /// Get the rank of this adapter
    pub fn rank(&self) -> usize {
        self.lora_a.nrows()
    }

    /// Get input features dimension
    pub fn in_features(&self) -> usize {
        self.lora_a.ncols()
    }

    /// Get output features dimension
    pub fn out_features(&self) -> usize {
        self.lora_b.nrows()
    }

    /// Apply the LoRA adapter to an input
    ///
    /// Computes: output = input + scaling * (input @ A^T @ B^T)
    pub fn apply(&self, input: &Array1<f32>) -> InferenceResult<Array1<f32>> {
        if input.len() != self.in_features() {
            return Err(InferenceError::DimensionMismatch {
                expected: self.in_features(),
                got: input.len(),
            });
        }

        // Compute input @ A^T
        let mut hidden = Array1::zeros(self.rank());
        for i in 0..self.rank() {
            hidden[i] = input.dot(&self.lora_a.row(i));
        }

        // Compute hidden @ B^T
        let mut output = Array1::zeros(self.out_features());
        for i in 0..self.out_features() {
            output[i] = hidden.dot(&self.lora_b.row(i));
        }

        // Scale and add to original input (identity residual)
        // For dimension matching, we assume output has same dim as input for residual
        // In practice, output dimension might differ - this is simplified
        if output.len() == input.len() {
            output = &output * self.scaling + input;
        } else {
            output = &output * self.scaling;
        }

        Ok(output)
    }

    /// Apply the LoRA adapter to a batch of inputs
    pub fn apply_batch(&self, inputs: &Array2<f32>) -> InferenceResult<Array2<f32>> {
        let batch_size = inputs.nrows();
        let mut outputs = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let input_row = inputs.row(i).to_owned();
            let output_row = self.apply(&input_row)?;
            outputs.push(output_row);
        }

        // Stack outputs into a 2D array
        let out_dim = outputs[0].len();
        let flat: Vec<f32> = outputs.into_iter().flat_map(|x| x.to_vec()).collect();

        Array2::from_shape_vec((batch_size, out_dim), flat).map_err(|e| {
            InferenceError::ForwardError(format!("Failed to stack LoRA outputs: {}", e))
        })
    }
}

/// Manager for multiple LoRA adapters
pub struct LoraAdapterManager {
    /// Map of adapter names to adapters
    adapters: HashMap<String, LoraAdapter>,
    /// Active adapter name (if any)
    active_adapter: Option<String>,
    /// Configuration
    config: LoraConfig,
}

impl LoraAdapterManager {
    /// Create a new adapter manager
    pub fn new(config: LoraConfig) -> Self {
        Self {
            adapters: HashMap::new(),
            active_adapter: None,
            config,
        }
    }

    /// Register a new adapter
    pub fn register_adapter(&mut self, adapter: LoraAdapter) {
        let name = adapter.name.clone();
        self.adapters.insert(name, adapter);
    }

    /// Activate an adapter by name
    pub fn activate(&mut self, name: impl AsRef<str>) -> InferenceResult<()> {
        let name_ref = name.as_ref();
        if !self.adapters.contains_key(name_ref) {
            return Err(InferenceError::ForwardError(format!(
                "Adapter '{}' not found",
                name_ref
            )));
        }
        self.active_adapter = Some(name_ref.to_string());
        Ok(())
    }

    /// Deactivate the current adapter
    pub fn deactivate(&mut self) {
        self.active_adapter = None;
    }

    /// Get the active adapter
    pub fn active_adapter(&self) -> Option<&LoraAdapter> {
        self.active_adapter
            .as_ref()
            .and_then(|name| self.adapters.get(name))
    }

    /// Apply the active adapter (if any) to input
    pub fn apply(&self, input: &Array1<f32>) -> InferenceResult<Array1<f32>> {
        if let Some(adapter) = self.active_adapter() {
            adapter.apply(input)
        } else {
            // No active adapter, return input unchanged
            Ok(input.clone())
        }
    }

    /// Apply the active adapter to a batch
    pub fn apply_batch(&self, inputs: &Array2<f32>) -> InferenceResult<Array2<f32>> {
        if let Some(adapter) = self.active_adapter() {
            adapter.apply_batch(inputs)
        } else {
            Ok(inputs.clone())
        }
    }

    /// List all registered adapters
    pub fn list_adapters(&self) -> Vec<&String> {
        self.adapters.keys().collect()
    }

    /// Get adapter by name
    pub fn get_adapter(&self, name: impl AsRef<str>) -> Option<&LoraAdapter> {
        self.adapters.get(name.as_ref())
    }

    /// Remove an adapter
    pub fn remove_adapter(&mut self, name: impl AsRef<str>) -> Option<LoraAdapter> {
        let name_ref = name.as_ref();
        // Deactivate if it's the active one
        if self.active_adapter.as_deref() == Some(name_ref) {
            self.deactivate();
        }
        self.adapters.remove(name_ref)
    }

    /// Get configuration
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }
}

/// Builder for creating LoRA adapters from components
pub struct LoraAdapterBuilder {
    lora_a: Option<Array2<f32>>,
    lora_b: Option<Array2<f32>>,
    scaling: f32,
    name: String,
}

impl LoraAdapterBuilder {
    /// Create a new builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            lora_a: None,
            lora_b: None,
            scaling: 1.0,
            name: name.into(),
        }
    }

    /// Set matrix A
    pub fn lora_a(mut self, matrix: Array2<f32>) -> Self {
        self.lora_a = Some(matrix);
        self
    }

    /// Set matrix B
    pub fn lora_b(mut self, matrix: Array2<f32>) -> Self {
        self.lora_b = Some(matrix);
        self
    }

    /// Set scaling factor
    pub fn scaling(mut self, scaling: f32) -> Self {
        self.scaling = scaling;
        self
    }

    /// Set scaling from config
    pub fn scaling_from_config(mut self, config: &LoraConfig) -> Self {
        self.scaling = config.scaling();
        self
    }

    /// Build the adapter
    pub fn build(self) -> InferenceResult<LoraAdapter> {
        let lora_a = self.lora_a.ok_or_else(|| {
            InferenceError::ForwardError("LoRA matrix A not provided".to_string())
        })?;
        let lora_b = self.lora_b.ok_or_else(|| {
            InferenceError::ForwardError("LoRA matrix B not provided".to_string())
        })?;

        LoraAdapter::new(lora_a, lora_b, self.scaling, self.name)
    }
}

/// LoRA adapter loader for reading from disk
pub struct LoraAdapterLoader {
    /// Base path for adapter files
    base_path: PathBuf,
}

impl LoraAdapterLoader {
    /// Create a new loader with base path
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// Load an adapter from directory
    ///
    /// Expected structure:
    /// - adapter_name/
    ///   - config.json
    ///   - lora_a.safetensors (or .npy)
    ///   - lora_b.safetensors (or .npy)
    pub fn load(
        &self,
        adapter_name: impl AsRef<str>,
    ) -> InferenceResult<(LoraAdapter, LoraConfig)> {
        let adapter_path = self.base_path.join(adapter_name.as_ref());

        // Load config
        let config_path = adapter_path.join("config.json");
        let config: LoraConfig = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
                InferenceError::ForwardError(format!("Failed to read config: {}", e))
            })?;
            serde_json::from_str(&config_str).map_err(|e| {
                InferenceError::ForwardError(format!("Failed to parse config: {}", e))
            })?
        } else {
            LoraConfig::default()
        };

        // For now, return a placeholder adapter since we don't have actual file loading
        // In a real implementation, you'd load from safetensors or numpy files
        let rank = config.rank;
        let lora_a = Array2::zeros((rank, 128)); // Placeholder dimensions
        let lora_b = Array2::zeros((128, rank));
        let scaling = config.scaling();

        let adapter = LoraAdapter::new(lora_a, lora_b, scaling, adapter_name.as_ref())?;
        Ok((adapter, config))
    }

    /// List available adapters in the base path
    pub fn list_available(&self) -> InferenceResult<Vec<String>> {
        let mut adapters = Vec::new();

        let entries = std::fs::read_dir(&self.base_path).map_err(|e| {
            InferenceError::ForwardError(format!("Failed to read adapter directory: {}", e))
        })?;

        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    adapters.push(name.to_string());
                }
            }
        }

        Ok(adapters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config() {
        let config = LoraConfig::new().rank(16).alpha(32.0);

        assert_eq!(config.rank, 16);
        assert_eq!(config.alpha, 32.0);
        assert_eq!(config.scaling(), 2.0); // alpha / rank = 32 / 16
    }

    #[test]
    fn test_lora_adapter_creation() {
        let lora_a = Array2::from_shape_vec((4, 8), vec![1.0; 32]).unwrap();
        let lora_b = Array2::from_shape_vec((8, 4), vec![0.5; 32]).unwrap();

        let adapter = LoraAdapter::new(lora_a, lora_b, 0.5, "test").unwrap();

        assert_eq!(adapter.rank(), 4);
        assert_eq!(adapter.in_features(), 8);
        assert_eq!(adapter.out_features(), 8);
    }

    #[test]
    fn test_lora_adapter_dimension_mismatch() {
        let lora_a = Array2::from_shape_vec((4, 8), vec![1.0; 32]).unwrap();
        let lora_b = Array2::from_shape_vec((8, 5), vec![0.5; 40]).unwrap(); // Rank mismatch

        let result = LoraAdapter::new(lora_a, lora_b, 0.5, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_adapter_apply() {
        let rank = 2;
        let in_features = 4;
        let out_features = 4;

        let lora_a = Array2::from_shape_vec((rank, in_features), vec![0.1; 8]).unwrap();
        let lora_b = Array2::from_shape_vec((out_features, rank), vec![0.2; 8]).unwrap();

        let adapter = LoraAdapter::new(lora_a, lora_b, 1.0, "test").unwrap();

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = adapter.apply(&input).unwrap();

        assert_eq!(output.len(), out_features);
        // Output should be input + LoRA modification
    }

    #[test]
    fn test_lora_manager() {
        let config = LoraConfig::new();
        let mut manager = LoraAdapterManager::new(config);

        let lora_a = Array2::from_shape_vec((2, 4), vec![0.1; 8]).unwrap();
        let lora_b = Array2::from_shape_vec((4, 2), vec![0.2; 8]).unwrap();
        let adapter = LoraAdapter::new(lora_a, lora_b, 1.0, "adapter1").unwrap();

        manager.register_adapter(adapter);
        assert_eq!(manager.list_adapters().len(), 1);

        manager.activate("adapter1").unwrap();
        assert!(manager.active_adapter().is_some());

        manager.deactivate();
        assert!(manager.active_adapter().is_none());
    }

    #[test]
    fn test_lora_manager_apply_without_adapter() {
        let config = LoraConfig::new();
        let manager = LoraAdapterManager::new(config);

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = manager.apply(&input).unwrap();

        // Without adapter, output should equal input
        assert_eq!(output, input);
    }

    #[test]
    fn test_lora_builder() {
        let lora_a = Array2::from_shape_vec((2, 4), vec![0.1; 8]).unwrap();
        let lora_b = Array2::from_shape_vec((4, 2), vec![0.2; 8]).unwrap();

        let adapter = LoraAdapterBuilder::new("test")
            .lora_a(lora_a)
            .lora_b(lora_b)
            .scaling(0.5)
            .build()
            .unwrap();

        assert_eq!(adapter.name, "test");
        assert_eq!(adapter.scaling, 0.5);
    }

    #[test]
    fn test_lora_builder_missing_matrix() {
        let lora_a = Array2::from_shape_vec((2, 4), vec![0.1; 8]).unwrap();

        let result = LoraAdapterBuilder::new("test")
            .lora_a(lora_a)
            // Missing lora_b
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_lora_adapter_batch() {
        let lora_a = Array2::from_shape_vec((2, 4), vec![0.1; 8]).unwrap();
        let lora_b = Array2::from_shape_vec((4, 2), vec![0.2; 8]).unwrap();
        let adapter = LoraAdapter::new(lora_a, lora_b, 1.0, "test").unwrap();

        let inputs = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, // Sample 1
                5.0, 6.0, 7.0, 8.0, // Sample 2
                9.0, 10.0, 11.0, 12.0, // Sample 3
            ],
        )
        .unwrap();

        let outputs = adapter.apply_batch(&inputs).unwrap();
        assert_eq!(outputs.nrows(), 3);
    }
}
