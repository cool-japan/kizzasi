//! Weight loading from safetensors format
//!
//! This module provides functionality to load pre-trained model weights
//! from the safetensors format, which is safer and faster than PyTorch
//! pickle files.
//!
//! # Safetensors Format
//!
//! Safetensors is a simple format for storing tensors safely (as opposed to pickle)
//! and that is still fast (zero-copy). It's used by Hugging Face and other ML frameworks.
//!
//! # Weight Naming Conventions
//!
//! Kizzasi models expect specific weight naming patterns. Each model architecture
//! has its own convention documented in the respective model module.
//!
//! ## Mamba Weight Format
//!
//! Mamba models expect the following weight structure:
//!
//! ```text
//! input_proj                      [input_dim, hidden_dim]
//! output_proj                     [hidden_dim, input_dim]
//! layers.{i}.norm.weight          [hidden_dim]
//! layers.{i}.norm.bias            [hidden_dim] (optional)
//! layers.{i}.in_proj              [hidden_dim, inner_dim*2]
//! layers.{i}.conv.weight          [out_channels, in_channels, kernel_size]
//! layers.{i}.conv.bias            [out_channels] (optional)
//! layers.{i}.ssm.log_a            [state_dim]
//! layers.{i}.ssm.delta_proj       [inner_dim, inner_dim]
//! layers.{i}.ssm.delta_bias       [inner_dim]
//! layers.{i}.ssm.b_proj           [inner_dim, state_dim]
//! layers.{i}.ssm.c_proj           [inner_dim, state_dim]
//! layers.{i}.ssm.d_skip           [inner_dim]
//! layers.{i}.out_proj             [inner_dim, hidden_dim]
//! ```
//!
//! ## RWKV Weight Format
//!
//! RWKV v6 models expect:
//!
//! ```text
//! input_proj                      [input_dim, hidden_dim]
//! output_proj                     [hidden_dim, input_dim]
//! layers.{i}.norm.weight          [hidden_dim]
//! layers.{i}.time_mix.w_r         [num_heads, head_dim]
//! layers.{i}.time_mix.w_k         [num_heads, head_dim]
//! layers.{i}.time_mix.w_v         [num_heads, head_dim]
//! layers.{i}.time_mix.w_g         [num_heads, head_dim]
//! layers.{i}.time_mix.w_a         [num_heads, head_dim]
//! layers.{i}.time_mix.w_b         [num_heads, head_dim]
//! layers.{i}.channel_mix.w_r      [hidden_dim]
//! layers.{i}.channel_mix.w_k      [hidden_dim]
//! layers.{i}.channel_mix.w_v      [hidden_dim]
//! ```
//!
//! ## HuggingFace Compatibility
//!
//! HuggingFace Mamba models use a different architecture and naming:
//!
//! ```text
//! HuggingFace:                    Kizzasi:
//! backbone.embeddings          →  input_proj
//! backbone.layers.{i}.norm     →  layers.{i}.norm
//! backbone.layers.{i}.mixer.in_proj → layers.{i}.in_proj
//! backbone.layers.{i}.mixer.conv1d → layers.{i}.conv
//! backbone.layers.{i}.mixer.x_proj → (needs splitting)
//! backbone.layers.{i}.mixer.dt_proj → layers.{i}.ssm.delta_proj
//! backbone.layers.{i}.mixer.A_log → layers.{i}.ssm.log_a
//! backbone.layers.{i}.mixer.D → layers.{i}.ssm.d_skip
//! backbone.layers.{i}.mixer.out_proj → layers.{i}.out_proj
//! lm_head                      →  output_proj
//! ```
//!
//! **Important**: HuggingFace's `x_proj` combines time_step, B, and C projections
//! into a single matrix that must be split during conversion:
//!
//! ```text
//! x_proj [intermediate_size, time_step_rank + state_size*2]
//!   ↓ split ↓
//! dt [time_step_rank], B [state_size], C [state_size]
//! ```
//!
//! # Conversion Utilities
//!
//! Use `WeightLoader` for advanced loading with validation and name mapping:
//!
//! ```ignore
//! use kizzasi_model::loader::{ModelLoader, WeightLoader};
//! use kizzasi_model::ModelType;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let loader = ModelLoader::new("mamba.safetensors")?;
//! let weight_loader = WeightLoader::new(loader)
//!     .model_type(ModelType::Mamba)
//!     .strict(false);  // Allow missing optional weights
//!
//! // Inspect checkpoint structure
//! weight_loader.print_weights();
//!
//! // Get suggested mappings for HuggingFace format
//! let mappings = weight_loader.suggest_huggingface_mapping();
//! # Ok(())
//! # }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use kizzasi_model::loader::ModelLoader;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let loader = ModelLoader::new("model.safetensors")?;
//! let tensor_names = loader.list_tensors();
//! // Load specific tensors as needed
//! # Ok(())
//! # }
//! ```

use crate::error::{ModelError, ModelResult};
use crate::ModelType;
use safetensors::tensor::SafeTensors;
use scirs2_core::ndarray::{Array1, Array2, ArrayD};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Weight loader for safetensors format
pub struct ModelLoader {
    /// Loaded safetensors data
    tensors: SafeTensors<'static>,
    /// Raw file data (kept alive for tensors)
    _data: Vec<u8>,
}

impl ModelLoader {
    /// Load a safetensors file from disk
    pub fn new<P: AsRef<Path>>(path: P) -> ModelResult<Self> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| ModelError::simple_load_error(format!("Failed to open file: {}", e)))?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| ModelError::simple_load_error(format!("Failed to read file: {}", e)))?;

        // Leak the data to get a 'static lifetime
        // This is safe because we keep the Vec alive in the struct
        let data_static = Box::leak(data.clone().into_boxed_slice());

        let tensors = SafeTensors::deserialize(data_static).map_err(|e| {
            ModelError::simple_load_error(format!("Failed to parse safetensors: {}", e))
        })?;

        Ok(Self {
            tensors,
            _data: data,
        })
    }

    /// Load a safetensors from bytes
    pub fn from_bytes(data: Vec<u8>) -> ModelResult<Self> {
        let data_static = Box::leak(data.clone().into_boxed_slice());

        let tensors = SafeTensors::deserialize(data_static).map_err(|e| {
            ModelError::simple_load_error(format!("Failed to parse safetensors: {}", e))
        })?;

        Ok(Self {
            tensors,
            _data: data,
        })
    }

    /// List all available tensor names in the file
    pub fn list_tensors(&self) -> Vec<String> {
        self.tensors.names().iter().map(|s| s.to_string()).collect()
    }

    /// Get metadata about a specific tensor
    pub fn tensor_info(&self, name: &str) -> Option<TensorInfo> {
        self.tensors.tensor(name).ok().map(|view| TensorInfo {
            name: name.to_string(),
            shape: view.shape().to_vec(),
            dtype: format!("{:?}", view.dtype()),
        })
    }

    /// Load a 1D tensor (Array1<f32>)
    pub fn load_array1(&self, name: &str) -> ModelResult<Array1<f32>> {
        let view = self.tensors.tensor(name).map_err(|e| {
            ModelError::simple_load_error(format!("Tensor '{}' not found: {}", name, e))
        })?;

        let shape = view.shape();
        if shape.len() != 1 {
            return Err(ModelError::simple_load_error(format!(
                "Expected 1D tensor for '{}', got shape {:?}",
                name, shape
            )));
        }

        let data = view.data();
        let float_data = match view.dtype() {
            safetensors::Dtype::F32 => {
                // Convert bytes to f32
                data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect::<Vec<_>>()
            }
            safetensors::Dtype::F64 => {
                // Convert f64 to f32
                data.chunks_exact(8)
                    .map(|chunk| {
                        let bytes = [
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ];
                        f64::from_le_bytes(bytes) as f32
                    })
                    .collect::<Vec<_>>()
            }
            dtype => {
                return Err(ModelError::simple_load_error(format!(
                    "Unsupported dtype for '{}': {:?}",
                    name, dtype
                )));
            }
        };

        Ok(Array1::from_vec(float_data))
    }

    /// Load a 2D tensor (Array2<f32>)
    pub fn load_array2(&self, name: &str) -> ModelResult<Array2<f32>> {
        let view = self.tensors.tensor(name).map_err(|e| {
            ModelError::simple_load_error(format!("Tensor '{}' not found: {}", name, e))
        })?;

        let shape = view.shape();
        if shape.len() != 2 {
            return Err(ModelError::simple_load_error(format!(
                "Expected 2D tensor for '{}', got shape {:?}",
                name, shape
            )));
        }

        let data = view.data();
        let float_data = match view.dtype() {
            safetensors::Dtype::F32 => data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>(),
            safetensors::Dtype::F64 => data
                .chunks_exact(8)
                .map(|chunk| {
                    let bytes = [
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ];
                    f64::from_le_bytes(bytes) as f32
                })
                .collect::<Vec<_>>(),
            dtype => {
                return Err(ModelError::simple_load_error(format!(
                    "Unsupported dtype for '{}': {:?}",
                    name, dtype
                )));
            }
        };

        Array2::from_shape_vec((shape[0], shape[1]), float_data)
            .map_err(|e| ModelError::simple_load_error(format!("Failed to create Array2: {}", e)))
    }

    /// Load a tensor of arbitrary dimension
    pub fn load_array(&self, name: &str) -> ModelResult<ArrayD<f32>> {
        let view = self.tensors.tensor(name).map_err(|e| {
            ModelError::simple_load_error(format!("Tensor '{}' not found: {}", name, e))
        })?;

        let shape = view.shape();
        let data = view.data();

        let float_data = match view.dtype() {
            safetensors::Dtype::F32 => data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>(),
            safetensors::Dtype::F64 => data
                .chunks_exact(8)
                .map(|chunk| {
                    let bytes = [
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ];
                    f64::from_le_bytes(bytes) as f32
                })
                .collect::<Vec<_>>(),
            safetensors::Dtype::F16 => {
                // For F16, we need to convert to f32
                // Note: This is a simplified conversion
                data.chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect::<Vec<_>>()
            }
            dtype => {
                return Err(ModelError::simple_load_error(format!(
                    "Unsupported dtype for '{}': {:?}",
                    name, dtype
                )));
            }
        };

        ArrayD::from_shape_vec(shape, float_data)
            .map_err(|e| ModelError::simple_load_error(format!("Failed to create ArrayD: {}", e)))
    }

    /// Load a 3D tensor as Vec<Vec<Vec<f32>>>
    ///
    /// This is useful for convolution weights [out_channels, in_channels, kernel_size]
    pub fn load_array3(&self, name: &str) -> ModelResult<Vec<Vec<Vec<f32>>>> {
        let array_d = self.load_array(name)?;

        if array_d.ndim() != 3 {
            return Err(ModelError::simple_load_error(format!(
                "Expected 3D tensor for '{}', got {}D tensor",
                name,
                array_d.ndim()
            )));
        }

        let shape = array_d.shape();
        let dim0 = shape[0];
        let dim1 = shape[1];
        let dim2 = shape[2];

        // Convert ArrayD to nested Vec structure
        let mut result = Vec::with_capacity(dim0);
        for i in 0..dim0 {
            let mut dim1_vec = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut dim2_vec = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    dim2_vec.push(array_d[[i, j, k]]);
                }
                dim1_vec.push(dim2_vec);
            }
            result.push(dim1_vec);
        }

        Ok(result)
    }

    /// Check if a tensor exists
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.tensor(name).is_ok()
    }

    /// Load all tensors into a HashMap
    pub fn load_all(&self) -> ModelResult<HashMap<String, ArrayD<f32>>> {
        let mut result = HashMap::new();
        for name in self.list_tensors() {
            let array = self.load_array(&name)?;
            result.insert(name, array);
        }
        Ok(result)
    }

    /// Print a summary of all tensors in the file
    ///
    /// This is useful for inspecting checkpoint files and understanding their structure
    pub fn print_summary(&self) {
        println!("SafeTensors Weight Summary");
        println!("==========================");
        println!("Total tensors: {}", self.list_tensors().len());
        println!();

        // Group by prefix
        let mut prefixes: HashMap<String, Vec<String>> = HashMap::new();
        for name in self.list_tensors() {
            let parts: Vec<&str> = name.split('.').collect();
            let prefix = if parts.len() > 1 {
                parts[0..parts.len() - 1].join(".")
            } else {
                "root".to_string()
            };
            prefixes.entry(prefix).or_default().push(name);
        }

        for (prefix, tensors) in prefixes.iter() {
            println!("\n[{}]", prefix);
            for name in tensors {
                if let Some(info) = self.tensor_info(name) {
                    println!(
                        "  {} - shape: {:?}, dtype: {}",
                        name, info.shape, info.dtype
                    );
                }
            }
        }
    }

    /// Get statistics about tensor sizes
    pub fn get_size_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        let mut total_params = 0usize;

        for name in self.list_tensors() {
            if let Some(info) = self.tensor_info(&name) {
                let size: usize = info.shape.iter().product();
                stats.insert(name.clone(), size);
                total_params += size;
            }
        }

        stats.insert("__total_parameters".to_string(), total_params);
        stats
    }

    /// Search for tensors matching a pattern
    ///
    /// # Example
    /// ```ignore
    /// // Find all conv weights
    /// let conv_tensors = loader.search_tensors("conv.weight");
    /// ```
    pub fn search_tensors(&self, pattern: &str) -> Vec<String> {
        self.list_tensors()
            .into_iter()
            .filter(|name| name.contains(pattern))
            .collect()
    }
}

/// Information about a tensor in the safetensors file
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data type as string
    pub dtype: String,
}

/// Builder for loading model weights with validation
pub struct WeightLoader {
    loader: ModelLoader,
    model_type: Option<ModelType>,
    strict: bool,
}

impl WeightLoader {
    /// Create a new weight loader
    pub fn new(loader: ModelLoader) -> Self {
        Self {
            loader,
            model_type: None,
            strict: true,
        }
    }

    /// Set the expected model type
    pub fn model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = Some(model_type);
        self
    }

    /// Set whether to enforce strict loading (all weights must be present)
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Validate that all required weights are present
    pub fn validate_weights(&self, required: &[&str]) -> ModelResult<()> {
        if !self.strict {
            return Ok(());
        }

        let missing: Vec<_> = required
            .iter()
            .filter(|&&name| !self.loader.has_tensor(name))
            .copied()
            .collect();

        if !missing.is_empty() {
            return Err(ModelError::simple_load_error(format!(
                "Missing required weights: {:?}",
                missing
            )));
        }

        Ok(())
    }

    /// Get the underlying loader
    pub fn loader(&self) -> &ModelLoader {
        &self.loader
    }

    /// Create a name mapping from source format to target format
    ///
    /// # Example
    /// ```ignore
    /// let mapping = HashMap::from([
    ///     ("backbone.layers.0.mixer.in_proj.weight", "layers.0.in_proj"),
    ///     ("backbone.layers.0.mixer.A_log", "layers.0.ssm.log_a"),
    /// ]);
    /// let mapped_loader = WeightLoader::new(loader).with_name_mapping(mapping);
    /// ```
    pub fn with_name_mapping(self, _mapping: HashMap<String, String>) -> Self {
        // TODO: Implement name remapping
        // This requires storing the mapping and using it during tensor lookups
        self
    }

    /// Print available weights and their shapes
    ///
    /// This is useful for understanding what weights are available in the checkpoint
    pub fn print_weights(&self) {
        self.loader.print_summary();
    }

    /// Get suggested weight mappings for HuggingFace format
    ///
    /// Returns a list of (hf_name, kizzasi_name) pairs that can be used
    /// to convert HuggingFace checkpoints to Kizzasi format
    pub fn suggest_huggingface_mapping(&self) -> Vec<(String, String)> {
        let mut mappings = Vec::new();
        let tensors = self.loader.list_tensors();

        // Check if this looks like a HuggingFace checkpoint
        if tensors.iter().any(|t| t.contains("backbone.layers")) {
            for tensor in &tensors {
                if let Some(kizzasi_name) = self.hf_to_kizzasi_name(tensor) {
                    mappings.push((tensor.clone(), kizzasi_name));
                }
            }
        }

        mappings
    }

    /// Convert HuggingFace weight name to Kizzasi format
    ///
    /// # HuggingFace → Kizzasi Mapping
    ///
    /// - `backbone.embeddings` → `input_proj`
    /// - `backbone.layers.{i}.norm.weight` → `layers.{i}.norm.weight`
    /// - `backbone.layers.{i}.mixer.in_proj` → `layers.{i}.in_proj`
    /// - `backbone.layers.{i}.mixer.conv1d` → `layers.{i}.conv`
    /// - `backbone.layers.{i}.mixer.A_log` → `layers.{i}.ssm.log_a`
    /// - `backbone.layers.{i}.mixer.D` → `layers.{i}.ssm.d_skip`
    /// - `backbone.layers.{i}.mixer.out_proj` → `layers.{i}.out_proj`
    /// - `lm_head` → `output_proj`
    ///
    /// Note: HuggingFace uses `x_proj` + `dt_proj` for selective parameters,
    /// while Kizzasi uses separate `delta_proj`, `b_proj`, `c_proj`.
    /// This requires splitting/combining weights during conversion.
    fn hf_to_kizzasi_name(&self, hf_name: &str) -> Option<String> {
        // Simple prefix replacement
        let name = hf_name
            .replace("backbone.", "")
            .replace(".mixer.", ".")
            .replace("conv1d", "conv")
            .replace("A_log", "ssm.log_a")
            .replace(".D", ".ssm.d_skip");

        if name.is_empty() {
            None
        } else {
            Some(name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_info() {
        let info = TensorInfo {
            name: "test".to_string(),
            shape: vec![2, 3],
            dtype: "F32".to_string(),
        };
        assert_eq!(info.name, "test");
        assert_eq!(info.shape, vec![2, 3]);
    }
}
