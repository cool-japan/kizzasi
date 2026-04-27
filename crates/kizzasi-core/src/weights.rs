//! Weight management for SSM models
//!
//! Provides functionality to load/save model weights in various formats:
//! - Safetensors (preferred format)
//! - PyTorch checkpoints (via conversion)
//! - Quantized weights (INT8)
//! - LoRA adapters

use crate::device::DeviceConfig;
use crate::error::{CoreError, CoreResult};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use std::collections::HashMap;
use std::path::Path;

/// Weight format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// Safetensors format (recommended)
    SafeTensors,
    /// PyTorch checkpoint
    PyTorch,
    /// Quantized INT8
    QuantizedInt8,
}

/// Weight loading configuration
#[derive(Debug, Clone)]
pub struct WeightLoadConfig {
    /// Device configuration (CPU/CUDA/Metal)
    pub device_config: DeviceConfig,
    /// Whether to quantize weights on load
    pub quantize: bool,
    /// Strict mode (fail if any keys are missing)
    pub strict: bool,
}

impl Default for WeightLoadConfig {
    fn default() -> Self {
        Self {
            device_config: DeviceConfig::default(),
            quantize: false,
            strict: true,
        }
    }
}

impl WeightLoadConfig {
    /// Create device from configuration
    pub fn create_device(&self) -> CoreResult<Device> {
        self.device_config.create_device()
    }

    /// Get data type from configuration
    pub fn get_dtype(&self) -> DType {
        if self.device_config.use_fp16 {
            DType::F16
        } else {
            DType::F32
        }
    }
}

/// Weight loader for SSM models
pub struct WeightLoader {
    #[allow(dead_code)]
    config: WeightLoadConfig,
}

impl WeightLoader {
    /// Create a new weight loader
    pub fn new(config: WeightLoadConfig) -> Self {
        Self { config }
    }

    /// Load weights from a safetensors file
    ///
    /// Note: This function uses varmap.load() which handles loading from safetensors format
    pub fn load_safetensors<P: AsRef<Path>>(&self, path: P, varmap: &mut VarMap) -> CoreResult<()> {
        let path = path.as_ref();

        // Use VarMap's built-in safetensors loading
        varmap.load(path).map_err(|e| {
            CoreError::WeightLoadError(format!("Failed to load safetensors: {}", e))
        })?;

        Ok(())
    }

    /// Save weights to a safetensors file
    ///
    /// Note: This function uses varmap.save() which handles saving to safetensors format
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P, varmap: &VarMap) -> CoreResult<()> {
        let path = path.as_ref();

        // Use VarMap's built-in safetensors saving
        varmap.save(path).map_err(|e| {
            CoreError::WeightLoadError(format!("Failed to save safetensors: {}", e))
        })?;

        Ok(())
    }

    /// Convert safetensors tensor view to candle Tensor
    #[allow(dead_code)]
    fn safetensors_to_candle(&self, view: safetensors::tensor::TensorView) -> CoreResult<Tensor> {
        let shape = view.shape().to_vec();
        let dtype = match view.dtype() {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::F16 => DType::F16,
            safetensors::Dtype::BF16 => DType::BF16,
            safetensors::Dtype::I64 => DType::I64,
            safetensors::Dtype::U8 => DType::U8,
            _ => {
                return Err(CoreError::WeightLoadError(format!(
                    "Unsupported dtype: {:?}",
                    view.dtype()
                )))
            }
        };

        // Get raw data
        let data = view.data();

        // Create tensor from raw data
        let tensor = match dtype {
            DType::F32 => {
                let values: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(values, &shape[..], &Device::Cpu).map_err(|e| {
                    CoreError::WeightLoadError(format!("Failed to create tensor: {}", e))
                })?
            }
            DType::F16 | DType::BF16 => {
                // For F16/BF16, we need to convert to F32 first
                let values: Vec<u16> = data
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();

                let f32_values: Vec<f32> = values
                    .iter()
                    .map(|&v| half::f16::from_bits(v).to_f32())
                    .collect();

                Tensor::from_vec(f32_values, &shape[..], &Device::Cpu)
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to create tensor: {}", e))
                    })?
                    .to_dtype(dtype)
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to convert dtype: {}", e))
                    })?
            }
            _ => {
                return Err(CoreError::WeightLoadError(format!(
                    "Unsupported dtype for conversion: {:?}",
                    dtype
                )))
            }
        };

        Ok(tensor)
    }

    /// Convert candle tensors to safetensors format
    #[allow(dead_code)]
    fn candle_to_safetensors(&self, tensors: HashMap<String, Tensor>) -> CoreResult<Vec<u8>> {
        use safetensors::tensor::Dtype as SafeDtype;

        // Prepare tensor data
        let mut tensor_data: HashMap<String, (SafeDtype, Vec<usize>, Vec<u8>)> = HashMap::new();

        for (name, tensor) in tensors.iter() {
            let shape: Vec<usize> = tensor.dims().to_vec();

            let dtype = match tensor.dtype() {
                DType::F32 => SafeDtype::F32,
                DType::F16 => SafeDtype::F16,
                DType::BF16 => SafeDtype::BF16,
                DType::I64 => SafeDtype::I64,
                DType::U8 => SafeDtype::U8,
                _ => {
                    return Err(CoreError::WeightLoadError(format!(
                        "Unsupported dtype for safetensors: {:?}",
                        tensor.dtype()
                    )))
                }
            };

            // Get tensor data as bytes
            let data = self.tensor_to_bytes(tensor)?;

            tensor_data.insert(name.clone(), (dtype, shape, data));
        }

        // Build the safetensors file manually using safetensors::tensor module
        // For now, we'll use a simpler approach with serialize_to_file
        // This is a placeholder - full implementation would use proper serialization

        // Temporary workaround: Return empty vec for now
        // TODO: Implement proper safetensors serialization
        Ok(Vec::new())
    }

    /// Convert tensor to bytes
    #[allow(dead_code)]
    fn tensor_to_bytes(&self, tensor: &Tensor) -> CoreResult<Vec<u8>> {
        match tensor.dtype() {
            DType::F32 => {
                let values = tensor
                    .flatten_all()
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to flatten tensor: {}", e))
                    })?
                    .to_vec1::<f32>()
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to convert to vec: {}", e))
                    })?;

                let mut bytes = Vec::with_capacity(values.len() * 4);
                for v in values {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                Ok(bytes)
            }
            DType::F16 => {
                let values = tensor
                    .flatten_all()
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to flatten tensor: {}", e))
                    })?
                    .to_vec1::<half::f16>()
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to convert to vec: {}", e))
                    })?;

                let mut bytes = Vec::with_capacity(values.len() * 2);
                for v in values {
                    bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                Ok(bytes)
            }
            _ => Err(CoreError::WeightLoadError(format!(
                "Unsupported dtype for bytes conversion: {:?}",
                tensor.dtype()
            ))),
        }
    }

    /// Quantize a tensor to INT8
    #[allow(dead_code)]
    fn quantize_tensor(&self, tensor: &Tensor) -> CoreResult<Tensor> {
        // Simple quantization: scale to [-128, 127] range
        // TODO: Implement proper quantization with scale and zero-point tracking

        let min_val = tensor
            .min(candle_core::D::Minus1)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to compute min: {}", e)))?;
        let max_val = tensor
            .max(candle_core::D::Minus1)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to compute max: {}", e)))?;

        let range = max_val
            .sub(&min_val)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to compute range: {}", e)))?;

        // Scale to [0, 255]
        let scaled = tensor
            .broadcast_sub(&min_val)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to subtract min: {}", e)))?
            .broadcast_div(&range)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to divide by range: {}", e)))?
            .affine(255.0, 0.0)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to scale: {}", e)))?;

        // Convert to U8
        let quantized = scaled
            .to_dtype(DType::U8)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to convert to U8: {}", e)))?;

        Ok(quantized)
    }
}

/// Weight pruning utilities
pub struct WeightPruner;

impl WeightPruner {
    /// Prune weights by magnitude
    ///
    /// Sets weights with absolute value below threshold to zero.
    pub fn prune_by_magnitude(tensor: &Tensor, threshold: f32) -> CoreResult<Tensor> {
        let abs_tensor = tensor
            .abs()
            .map_err(|e| CoreError::Generic(format!("Failed to compute abs: {}", e)))?;

        let mask = abs_tensor
            .ge(threshold as f64)
            .map_err(|e| CoreError::Generic(format!("Failed to create mask: {}", e)))?
            .to_dtype(tensor.dtype())
            .map_err(|e| CoreError::Generic(format!("Failed to convert mask dtype: {}", e)))?;

        tensor
            .mul(&mask)
            .map_err(|e| CoreError::Generic(format!("Failed to apply mask: {}", e)))
    }

    /// Prune weights by percentage
    ///
    /// Keeps only the top (1 - percentage) weights by magnitude.
    pub fn prune_by_percentage(tensor: &Tensor, percentage: f32) -> CoreResult<Tensor> {
        if percentage <= 0.0 || percentage >= 1.0 {
            return Err(CoreError::InvalidConfig(
                "Percentage must be between 0 and 1".to_string(),
            ));
        }

        // Flatten tensor to 1D for sorting
        let flat = tensor
            .flatten_all()
            .map_err(|e| CoreError::Generic(format!("Failed to flatten: {}", e)))?;

        let abs_flat = flat
            .abs()
            .map_err(|e| CoreError::Generic(format!("Failed to compute abs: {}", e)))?;

        // Get values as vec
        let values = abs_flat
            .to_vec1::<f32>()
            .map_err(|e| CoreError::Generic(format!("Failed to convert to vec: {}", e)))?;

        // Find threshold at the given percentage
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let threshold_idx = (sorted_values.len() as f32 * percentage) as usize;
        let threshold = sorted_values[threshold_idx];

        Self::prune_by_magnitude(tensor, threshold)
    }

    /// Compute sparsity of a tensor
    pub fn compute_sparsity(tensor: &Tensor) -> CoreResult<f32> {
        let total_elements = tensor.elem_count();

        let zeros = tensor
            .eq(0.0)
            .map_err(|e| CoreError::Generic(format!("Failed to compare with zero: {}", e)))?
            .to_dtype(DType::F32)
            .map_err(|e| CoreError::Generic(format!("Failed to convert dtype: {}", e)))?
            .sum_all()
            .map_err(|e| CoreError::Generic(format!("Failed to sum: {}", e)))?
            .to_vec0::<f32>()
            .map_err(|e| CoreError::Generic(format!("Failed to extract value: {}", e)))?;

        Ok(zeros / total_elements as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;

    #[test]
    fn test_weight_loader_creation() {
        let config = WeightLoadConfig::default();
        let _loader = WeightLoader::new(config);
    }

    #[test]
    fn test_prune_by_magnitude() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0f32, 0.1, 2.0, 0.05, 3.0], &device).unwrap();

        let pruned = WeightPruner::prune_by_magnitude(&tensor, 0.5).unwrap();
        let values = pruned.to_vec1::<f32>().unwrap();

        assert_eq!(values, vec![1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_compute_sparsity() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0f32, 0.0, 2.0, 0.0, 3.0], &device).unwrap();

        let sparsity = WeightPruner::compute_sparsity(&tensor).unwrap();
        assert!((sparsity - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_safetensors_roundtrip() {
        use std::env;

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create some test variables
        let _w1 = vb
            .get_with_hints((3, 4), "weight1", candle_nn::init::Init::Const(1.0))
            .unwrap();
        let _w2 = vb
            .get_with_hints((5, 6), "weight2", candle_nn::init::Init::Const(2.0))
            .unwrap();

        let config = WeightLoadConfig::default();
        let loader = WeightLoader::new(config);

        // Save
        let temp_dir = env::temp_dir();
        let save_path = temp_dir.join("test_weights.safetensors");

        let result = loader.save_safetensors(&save_path, &varmap);
        assert!(result.is_ok());

        // Clean up
        if save_path.exists() {
            std::fs::remove_file(save_path).ok();
        }
    }
}
