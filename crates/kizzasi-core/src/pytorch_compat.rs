//! # PyTorch Compatibility Layer
//!
//! Utilities for loading and converting PyTorch checkpoints to Kizzasi format.
//!
//! ## Features
//!
//! - **Checkpoint Loading**: Load PyTorch `.pth` or `.pt` files
//! - **Tensor Conversion**: Convert PyTorch tensors to ndarray/candle format
//! - **Weight Mapping**: Map PyTorch layer names to Kizzasi layer names
//! - **Architecture Detection**: Automatically detect model architecture
//! - **Validation**: Verify checkpoint compatibility
//!
//! ## Supported Formats
//!
//! - PyTorch state_dict (via safetensors)
//! - HuggingFace checkpoints
//! - Custom Mamba/SSM checkpoints

use crate::{CoreError, CoreResult};
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// PyTorch checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchCheckpoint {
    /// Model architecture name
    pub architecture: String,
    /// Number of layers
    pub num_layers: Option<usize>,
    /// Hidden dimension
    pub hidden_dim: Option<usize>,
    /// Model dimension
    pub d_model: Option<usize>,
    /// State dimension
    pub d_state: Option<usize>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Weight mapping configuration for different architectures
#[derive(Debug, Clone)]
pub struct WeightMapping {
    /// Source layer name pattern (PyTorch)
    pub source_pattern: String,
    /// Target layer name (Kizzasi)
    pub target_name: String,
    /// Whether to transpose the weight matrix
    pub transpose: bool,
}

/// PyTorch checkpoint converter
pub struct PyTorchConverter {
    /// Device for tensor operations
    device: Device,
    /// Weight mappings
    mappings: Vec<WeightMapping>,
}

impl PyTorchConverter {
    /// Create a new PyTorch converter
    pub fn new(device: Device) -> Self {
        Self {
            device,
            mappings: Vec::new(),
        }
    }

    /// Create converter with CPU device
    pub fn new_cpu() -> Self {
        Self::new(Device::Cpu)
    }

    /// Add a weight mapping
    pub fn add_mapping(&mut self, source: &str, target: &str, transpose: bool) {
        self.mappings.push(WeightMapping {
            source_pattern: source.to_string(),
            target_name: target.to_string(),
            transpose,
        });
    }

    /// Load checkpoint from safetensors file
    pub fn load_safetensors(&self, path: impl AsRef<Path>) -> CoreResult<HashMap<String, Tensor>> {
        let data = std::fs::read(path.as_ref())
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to read file: {}", e)))?;

        let tensors = SafeTensors::deserialize(&data).map_err(|e| {
            CoreError::WeightLoadError(format!("Failed to deserialize safetensors: {}", e))
        })?;

        let mut weights = HashMap::new();

        for (name, tensor_view) in tensors.tensors() {
            let tensor = self.safetensor_to_candle(&tensor_view)?;
            weights.insert(name.to_string(), tensor);
        }

        Ok(weights)
    }

    /// Convert safetensor view to candle tensor
    fn safetensor_to_candle(&self, view: &safetensors::tensor::TensorView) -> CoreResult<Tensor> {
        let shape: Vec<usize> = view.shape().to_vec();
        let dtype = match view.dtype() {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::F16 => DType::F16,
            safetensors::Dtype::BF16 => DType::BF16,
            safetensors::Dtype::F64 => DType::F64,
            safetensors::Dtype::I64 => DType::I64,
            safetensors::Dtype::U8 => DType::U8,
            _ => {
                return Err(CoreError::WeightLoadError(format!(
                    "Unsupported dtype: {:?}",
                    view.dtype()
                )))
            }
        };

        let data = view.data();
        Tensor::from_raw_buffer(data, dtype, &shape, &self.device)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to create tensor: {}", e)))
    }

    /// Convert candle tensor to ndarray Array2
    pub fn tensor_to_array2(&self, tensor: &Tensor) -> CoreResult<Array2<f32>> {
        if tensor.rank() != 2 {
            return Err(CoreError::WeightLoadError(format!(
                "Expected 2D tensor, got rank {}",
                tensor.rank()
            )));
        }

        let shape = tensor.shape();
        let rows = shape.dims()[0];
        let cols = shape.dims()[1];

        // Convert to f32 if needed
        let tensor_f32 = if tensor.dtype() != DType::F32 {
            tensor.to_dtype(DType::F32).map_err(|e| {
                CoreError::WeightLoadError(format!("Failed to convert dtype: {}", e))
            })?
        } else {
            tensor.clone()
        };

        // Get data as Vec<f32>
        let data: Vec<f32> = tensor_f32
            .to_vec2()
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to convert to vec: {}", e)))?
            .into_iter()
            .flatten()
            .collect();

        Array2::from_shape_vec((rows, cols), data).map_err(CoreError::ShapeError)
    }

    /// Convert candle tensor to ndarray Array1
    pub fn tensor_to_array1(&self, tensor: &Tensor) -> CoreResult<Array1<f32>> {
        if tensor.rank() != 1 {
            return Err(CoreError::WeightLoadError(format!(
                "Expected 1D tensor, got rank {}",
                tensor.rank()
            )));
        }

        // Convert to f32 if needed
        let tensor_f32 = if tensor.dtype() != DType::F32 {
            tensor.to_dtype(DType::F32).map_err(|e| {
                CoreError::WeightLoadError(format!("Failed to convert dtype: {}", e))
            })?
        } else {
            tensor.clone()
        };

        // Get data as Vec<f32>
        let data: Vec<f32> = tensor_f32
            .to_vec1()
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to convert to vec: {}", e)))?;

        Ok(Array1::from_vec(data))
    }

    /// Apply weight mappings to convert PyTorch names to Kizzasi names
    pub fn apply_mappings(
        &self,
        weights: HashMap<String, Tensor>,
    ) -> CoreResult<HashMap<String, Tensor>> {
        let mut mapped_weights = HashMap::new();

        for (source_name, tensor) in weights {
            // Try to find a matching mapping
            let mut mapped = false;
            for mapping in &self.mappings {
                if source_name.contains(&mapping.source_pattern) {
                    let mut target_tensor = tensor.clone();

                    // Transpose if needed
                    if mapping.transpose && target_tensor.rank() == 2 {
                        target_tensor = target_tensor
                            .t()
                            .map_err(|e| {
                                CoreError::WeightLoadError(format!("Failed to transpose: {}", e))
                            })?
                            .contiguous()
                            .map_err(|e| {
                                CoreError::WeightLoadError(format!(
                                    "Failed to make contiguous: {}",
                                    e
                                ))
                            })?;
                    }

                    mapped_weights.insert(mapping.target_name.clone(), target_tensor);
                    mapped = true;
                    break;
                }
            }

            // If no mapping found, keep original name
            if !mapped {
                mapped_weights.insert(source_name, tensor);
            }
        }

        Ok(mapped_weights)
    }

    /// Detect architecture from checkpoint
    pub fn detect_architecture(
        &self,
        weights: &HashMap<String, Tensor>,
    ) -> CoreResult<PyTorchCheckpoint> {
        let mut metadata = HashMap::new();
        let mut architecture = "unknown".to_string();
        let mut num_layers = None;
        let mut hidden_dim = None;
        let mut d_model = None;
        let mut d_state = None;

        // Analyze weight names to detect architecture
        for (name, tensor) in weights {
            // Detect Mamba
            if name.contains("mixer") || name.contains("ssm") {
                architecture = "mamba".to_string();
            }
            // Detect Mamba-2
            else if name.contains("ssd") || name.contains("mamba2") {
                architecture = "mamba2".to_string();
            }
            // Detect S4/S4D
            else if name.contains("s4") {
                architecture = "s4d".to_string();
            }
            // Detect S5
            else if name.contains("s5") || name.contains("block_diagonal") {
                architecture = "s5".to_string();
            }
            // Detect RetNet
            else if name.contains("retention") {
                architecture = "retnet".to_string();
            }

            // Extract dimensions
            if name.contains("layers.") {
                // Count layers
                if let Some(layer_str) = name.split("layers.").nth(1) {
                    if let Some(layer_num_str) = layer_str.split('.').next() {
                        if let Ok(layer_num) = layer_num_str.parse::<usize>() {
                            num_layers = Some(num_layers.unwrap_or(0).max(layer_num + 1));
                        }
                    }
                }
            }

            // Detect dimensions from tensor shapes
            let shape = tensor.shape();
            if (name.contains("in_proj") || name.contains("embedding")) && shape.rank() == 2 {
                d_model = Some(shape.dims()[0]);
            }
            if (name.contains("dt_proj") || name.contains("ssm")) && shape.rank() == 2 {
                hidden_dim = Some(shape.dims()[0]);
            }
            if (name.contains("a_log") || name.contains("lambda")) && shape.rank() >= 1 {
                d_state = Some(shape.dims()[shape.rank() - 1]);
            }
        }

        metadata.insert("num_weights".to_string(), weights.len().to_string());

        Ok(PyTorchCheckpoint {
            architecture,
            num_layers,
            hidden_dim,
            d_model,
            d_state,
            metadata,
        })
    }

    /// Create default mappings for Mamba architecture
    pub fn create_mamba_mappings(&mut self) {
        // Input embeddings
        self.add_mapping("embedding.weight", "embedding_w", false);

        // Layer mappings (example for layer 0)
        for i in 0..32 {
            // Adjust layer count as needed
            let prefix = format!("layers.{}", i);
            let target_prefix = format!("layer_{}", i);

            self.add_mapping(
                &format!("{}.mixer.in_proj", prefix),
                &format!("{}.in_proj_w", target_prefix),
                true,
            );
            self.add_mapping(
                &format!("{}.mixer.out_proj", prefix),
                &format!("{}.out_proj_w", target_prefix),
                true,
            );
            self.add_mapping(
                &format!("{}.mixer.conv1d.weight", prefix),
                &format!("{}.conv1d_w", target_prefix),
                false,
            );
            self.add_mapping(
                &format!("{}.mixer.conv1d.bias", prefix),
                &format!("{}.conv1d_b", target_prefix),
                false,
            );
            self.add_mapping(
                &format!("{}.mixer.dt_proj", prefix),
                &format!("{}.dt_proj_w", target_prefix),
                true,
            );
            self.add_mapping(
                &format!("{}.mixer.A_log", prefix),
                &format!("{}.a_log", target_prefix),
                false,
            );
            self.add_mapping(
                &format!("{}.mixer.D", prefix),
                &format!("{}.d_param", target_prefix),
                false,
            );
            self.add_mapping(
                &format!("{}.norm.weight", prefix),
                &format!("{}.norm_w", target_prefix),
                false,
            );
            self.add_mapping(
                &format!("{}.norm.bias", prefix),
                &format!("{}.norm_b", target_prefix),
                false,
            );
        }

        // Output head
        self.add_mapping("lm_head.weight", "output_w", true);
    }

    /// Create default mappings for S4D architecture
    pub fn create_s4d_mappings(&mut self) {
        for i in 0..32 {
            let prefix = format!("layers.{}", i);
            let target_prefix = format!("layer_{}", i);

            self.add_mapping(
                &format!("{}.input_proj", prefix),
                &format!("{}.input_proj", target_prefix),
                true,
            );
            self.add_mapping(
                &format!("{}.output_proj", prefix),
                &format!("{}.output_proj", target_prefix),
                true,
            );
            self.add_mapping(
                &format!("{}.lambda", prefix),
                &format!("{}.lambda", target_prefix),
                false,
            );
            self.add_mapping(
                &format!("{}.B", prefix),
                &format!("{}.b", target_prefix),
                false,
            );
            self.add_mapping(
                &format!("{}.C", prefix),
                &format!("{}.c", target_prefix),
                false,
            );
            self.add_mapping(
                &format!("{}.D", prefix),
                &format!("{}.d", target_prefix),
                false,
            );
        }
    }
}

/// Helper function to detect architecture from checkpoint file
pub fn detect_checkpoint_architecture(path: impl AsRef<Path>) -> CoreResult<PyTorchCheckpoint> {
    let converter = PyTorchConverter::new_cpu();
    let weights = converter.load_safetensors(path)?;
    converter.detect_architecture(&weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = PyTorchConverter::new_cpu();
        assert_eq!(converter.mappings.len(), 0);
    }

    #[test]
    fn test_add_mapping() {
        let mut converter = PyTorchConverter::new_cpu();
        converter.add_mapping("layers.0.weight", "layer_0_w", true);
        assert_eq!(converter.mappings.len(), 1);
        assert_eq!(converter.mappings[0].source_pattern, "layers.0.weight");
        assert_eq!(converter.mappings[0].target_name, "layer_0_w");
        assert!(converter.mappings[0].transpose);
    }

    #[test]
    fn test_mamba_mappings() {
        let mut converter = PyTorchConverter::new_cpu();
        converter.create_mamba_mappings();
        assert!(!converter.mappings.is_empty());
        // Should have mappings for multiple layers
        let has_layer_0 = converter
            .mappings
            .iter()
            .any(|m| m.target_name.contains("layer_0"));
        assert!(has_layer_0);
    }

    #[test]
    fn test_s4d_mappings() {
        let mut converter = PyTorchConverter::new_cpu();
        converter.create_s4d_mappings();
        assert!(!converter.mappings.is_empty());
    }

    #[test]
    fn test_tensor_conversion() {
        let converter = PyTorchConverter::new_cpu();

        // Create a test tensor
        let data = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let tensor = Tensor::new(data, &Device::Cpu).unwrap();

        let array = converter.tensor_to_array2(&tensor).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array[[0, 0]], 1.0);
        assert_eq!(array[[1, 2]], 6.0);
    }

    #[test]
    fn test_tensor_1d_conversion() {
        let converter = PyTorchConverter::new_cpu();

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, &Device::Cpu).unwrap();

        let array = converter.tensor_to_array1(&tensor).unwrap();
        assert_eq!(array.len(), 4);
        assert_eq!(array[0], 1.0);
        assert_eq!(array[3], 4.0);
    }

    #[test]
    fn test_architecture_detection() {
        let converter = PyTorchConverter::new_cpu();
        let mut weights = HashMap::new();

        // Create dummy tensors that look like Mamba weights
        let tensor = Tensor::zeros((256, 128), DType::F32, &Device::Cpu).unwrap();
        weights.insert("layers.0.mixer.in_proj.weight".to_string(), tensor.clone());
        weights.insert("layers.0.mixer.A_log".to_string(), tensor.clone());

        let checkpoint = converter.detect_architecture(&weights).unwrap();
        assert_eq!(checkpoint.architecture, "mamba");
        assert_eq!(checkpoint.num_layers, Some(1));
    }

    #[test]
    fn test_checkpoint_metadata() {
        let checkpoint = PyTorchCheckpoint {
            architecture: "mamba2".to_string(),
            num_layers: Some(24),
            hidden_dim: Some(768),
            d_model: Some(768),
            d_state: Some(16),
            metadata: HashMap::new(),
        };

        assert_eq!(checkpoint.architecture, "mamba2");
        assert_eq!(checkpoint.num_layers, Some(24));
    }
}
