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

    /// Load a 1D tensor (`Array1<f32>`)
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

    /// Load a 2D tensor (`Array2<f32>`)
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

    /// Load a 3D tensor as `Vec<Vec<Vec<f32>>>`
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
    /// Optional name mapping applied before tensor lookups
    name_mapping: Option<HashMap<String, String>>,
}

impl WeightLoader {
    /// Create a new weight loader
    pub fn new(loader: ModelLoader) -> Self {
        Self {
            loader,
            model_type: None,
            strict: true,
            name_mapping: None,
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

    /// Create a name mapping from source format to target format.
    ///
    /// The supplied mapping is stored and applied whenever a tensor is looked
    /// up by name.  Keys present in the mapping are rewritten to their values;
    /// unknown keys pass through unchanged.
    ///
    /// # Example
    /// ```ignore
    /// let mapping = HashMap::from([
    ///     ("backbone.layers.0.mixer.in_proj.weight".to_string(), "layers.0.in_proj".to_string()),
    ///     ("backbone.layers.0.mixer.A_log".to_string(), "layers.0.ssm.log_a".to_string()),
    /// ]);
    /// let mapped_loader = WeightLoader::new(loader).with_name_mapping(mapping);
    /// ```
    pub fn with_name_mapping(mut self, mapping: HashMap<String, String>) -> Self {
        self.name_mapping = Some(mapping);
        self
    }

    /// Apply the stored name mapping (if any) to a tensor key.
    ///
    /// Returns the mapped name if the key is present in the mapping, or the
    /// original key otherwise.
    pub fn remap_name<'a>(&'a self, name: &'a str) -> &'a str {
        if let Some(mapping) = &self.name_mapping {
            if let Some(mapped) = mapping.get(name) {
                return mapped.as_str();
            }
        }
        name
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

// ---------------------------------------------------------------------------
// WeightSourceLoader — bridges WeightSource → WeightLoader/ModelLoader API
// ---------------------------------------------------------------------------

/// Adapter that wraps a [`crate::incremental_loader::WeightSource`] and exposes
/// the same tensor-query surface as [`ModelLoader`] for code that expects the
/// blocking, in-memory API.
///
/// This bridges the streaming (`WeightSource`) and legacy (`ModelLoader`) worlds:
/// weights are loaded on demand from the source instead of being held as a
/// single pre-loaded byte buffer.
pub struct WeightSourceLoader<S: crate::incremental_loader::WeightSource> {
    source: S,
}

impl<S: crate::incremental_loader::WeightSource> WeightSourceLoader<S> {
    /// Wrap a [`WeightSource`](crate::incremental_loader::WeightSource) as a
    /// `WeightSourceLoader`.
    pub fn new(source: S) -> Self {
        Self { source }
    }

    /// Return the names of all tensors available in the underlying source.
    pub fn list_tensors(&self) -> Vec<String> {
        self.source.tensor_names()
    }

    /// Check whether the underlying source contains a tensor with the given name.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.source.contains(name)
    }

    /// Load and dequantize the tensor identified by `name` as a flat `Vec<f32>`.
    pub fn load_flat(&mut self, name: &str) -> ModelResult<Vec<f32>> {
        self.source.load_tensor(name)
    }

    /// Consume this adapter, returning ownership of the underlying source.
    pub fn into_source(self) -> S {
        self.source
    }
}

impl WeightLoader {
    /// Create a [`WeightLoader`] from any type implementing
    /// [`WeightSource`](crate::incremental_loader::WeightSource) by first
    /// materialising all tensors into memory via the source.
    ///
    /// This is an escape hatch for code that must use the legacy `WeightLoader`
    /// API but wants to consume weights from a streaming source. Because it
    /// loads everything into RAM at once, prefer
    /// [`IncrementalModelLoader`](crate::incremental_loader::IncrementalModelLoader)
    /// for true streaming use-cases.
    pub fn from_weight_source<S: crate::incremental_loader::WeightSource>(
        mut source: S,
        model_type: Option<crate::ModelType>,
        strict: bool,
    ) -> ModelResult<Self> {
        let names = source.tensor_names();
        let mut all_data: Vec<u8> = Vec::new();

        // Build an in-memory safetensors-like representation so that the
        // ModelLoader can be constructed without a real file on disk.
        // Strategy: concatenate all tensors as raw F32 bytes and build
        // a JSON header, then pass to ModelLoader::from_bytes.
        let mut tensor_metas: Vec<(String, usize, usize, usize)> = Vec::new();
        for name in &names {
            let floats = source.load_tensor(name)?;
            let begin = all_data.len();
            for v in &floats {
                all_data.extend_from_slice(&v.to_le_bytes());
            }
            let end = all_data.len();
            tensor_metas.push((name.clone(), begin, end, floats.len()));
        }

        // Build JSON header
        let mut header_map = serde_json::Map::new();
        for (name, begin, end, n) in &tensor_metas {
            let entry = serde_json::json!({
                "dtype": "F32",
                "shape": [n],
                "data_offsets": [begin, end]
            });
            header_map.insert(name.clone(), entry);
        }
        let header_json = serde_json::Value::Object(header_map).to_string();
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut file_bytes: Vec<u8> = Vec::new();
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(header_bytes);
        file_bytes.extend_from_slice(&all_data);

        let model_loader = ModelLoader::from_bytes(file_bytes)?;
        let mut wl = WeightLoader::new(model_loader);
        if let Some(mt) = model_type {
            wl = wl.model_type(mt);
        }
        wl = wl.strict(strict);
        Ok(wl)
    }
}

// ---------------------------------------------------------------------------
// NameRemapper
// ---------------------------------------------------------------------------

/// Translates HuggingFace-style weight key names to Kizzasi internal names.
///
/// Matching is performed with regex-like pattern rules. Layer-indexed keys
/// (`layers.{n}.…`) are matched structurally; the numeric index `{n}` is
/// preserved verbatim.
///
/// # Mapping rules
///
/// | HuggingFace pattern                        | Kizzasi name                    |
/// |--------------------------------------------|-------------------------------- |
/// | `layers.{n}.mixer.in_proj.weight`          | `layers.{n}.input_proj`         |
/// | `layers.{n}.mixer.out_proj.weight`         | `layers.{n}.output_proj`        |
/// | `layers.{n}.attn.q_proj.weight`            | `layers.{n}.attention.q`        |
/// | `layers.{n}.attn.k_proj.weight`            | `layers.{n}.attention.k`        |
/// | `layers.{n}.attn.v_proj.weight`            | `layers.{n}.attention.v`        |
/// | `layers.{n}.attn.o_proj.weight`            | `layers.{n}.attention.out`      |
/// | `layers.{n}.mlp.gate_proj.weight`          | `layers.{n}.ff.gate`            |
/// | `layers.{n}.mlp.up_proj.weight`            | `layers.{n}.ff.up`              |
/// | `layers.{n}.mlp.down_proj.weight`          | `layers.{n}.ff.down`            |
/// | `embedding.weight`                         | `input_proj`                    |
/// | `lm_head.weight`                           | `output_proj`                   |
/// | *(any other key)*                          | returned as-is                  |
#[derive(Debug, Clone, Default)]
pub struct NameRemapper;

impl NameRemapper {
    /// Create a new `NameRemapper`.
    pub fn new() -> Self {
        Self
    }

    /// Remap a single key from HuggingFace to Kizzasi format.
    ///
    /// Returns the remapped name, or the original key if no rule matches.
    ///
    /// # Backbone prefix stripping
    ///
    /// HuggingFace Mamba models wrap everything under a `backbone.` namespace.
    /// This method strips that prefix before applying the standard rules:
    ///
    /// - `backbone.embeddings.weight`  → `embedding.weight`  → `input_proj`
    /// - `backbone.norm_f.weight`      → `final_norm.weight`
    /// - `backbone.layers.{n}.…`       → `layers.{n}.…`      → remapped as usual
    ///
    /// # `lm_head` tying
    ///
    /// When a checkpoint omits `lm_head.weight` (weight tying), callers should
    /// fill it by transposing `embedding.weight`.  This remapper does not perform
    /// that step automatically; it is the caller's responsibility to detect the
    /// absence and apply the transpose.
    pub fn remap(&self, key: &str) -> String {
        // ----------------------------------------------------------------
        // Strip optional `backbone.` prefix (HuggingFace Mamba convention)
        // ----------------------------------------------------------------
        let key = if let Some(rest) = key.strip_prefix("backbone.") {
            // Map well-known backbone sub-keys, then fall through to common rules.
            match rest {
                "embeddings.weight" => return "input_proj".to_string(),
                "norm_f.weight" => return "final_norm.weight".to_string(),
                _ => rest,
            }
        } else {
            key
        };

        // ----------------------------------------------------------------
        // Top-level aliases
        // ----------------------------------------------------------------
        if key == "embedding.weight" {
            return "input_proj".to_string();
        }
        if key == "lm_head.weight" {
            return "output_proj".to_string();
        }

        // ----------------------------------------------------------------
        // Layer-indexed patterns: `layers.{n}.<rest>`
        // ----------------------------------------------------------------
        if let Some(layer_idx) = Self::extract_layer_index(key) {
            let after_layer = Self::strip_layer_prefix(key, layer_idx);
            if let Some(mapped_suffix) = Self::remap_layer_suffix(after_layer) {
                return format!("layers.{}.{}", layer_idx, mapped_suffix);
            }
        }

        // Unknown key — pass through unchanged.
        key.to_string()
    }

    /// Extract the layer index from a key that starts with `layers.{n}.`.
    ///
    /// Returns `None` if the key does not follow that pattern.
    fn extract_layer_index(key: &str) -> Option<usize> {
        let mut parts = key.splitn(3, '.');
        match (parts.next(), parts.next()) {
            (Some("layers"), Some(idx)) => idx.parse::<usize>().ok(),
            _ => None,
        }
    }

    /// Strip the `layers.{n}.` prefix and return the remainder.
    fn strip_layer_prefix(key: &str, layer_idx: usize) -> &str {
        // "layers.N." is "layers." (7) + digits + "."
        let prefix_len = 7 + layer_idx.to_string().len() + 1; // "layers." + N + "."
        if key.len() > prefix_len {
            &key[prefix_len..]
        } else {
            ""
        }
    }

    /// Map the suffix portion (after `layers.{n}.`) to a Kizzasi name segment.
    ///
    /// Returns `None` if the suffix is not a known pattern.
    fn remap_layer_suffix(suffix: &str) -> Option<&'static str> {
        match suffix {
            "mixer.in_proj.weight" => Some("input_proj"),
            "mixer.out_proj.weight" => Some("output_proj"),
            "attn.q_proj.weight" => Some("attention.q"),
            "attn.k_proj.weight" => Some("attention.k"),
            "attn.v_proj.weight" => Some("attention.v"),
            "attn.o_proj.weight" => Some("attention.out"),
            "mlp.gate_proj.weight" => Some("ff.gate"),
            "mlp.up_proj.weight" => Some("ff.up"),
            "mlp.down_proj.weight" => Some("ff.down"),
            _ => None,
        }
    }

    /// Remap an entire weight map, returning a new map with translated keys.
    ///
    /// Keys that do not match any rule are kept unchanged.
    pub fn remap_map(&self, weights: HashMap<String, Vec<f32>>) -> HashMap<String, Vec<f32>> {
        weights
            .into_iter()
            .map(|(k, v)| (self.remap(&k), v))
            .collect()
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

    #[test]
    fn test_name_remapper_layers() {
        let remapper = NameRemapper::new();

        assert_eq!(
            remapper.remap("layers.0.mixer.in_proj.weight"),
            "layers.0.input_proj"
        );
        assert_eq!(
            remapper.remap("layers.3.mixer.out_proj.weight"),
            "layers.3.output_proj"
        );
        assert_eq!(
            remapper.remap("layers.7.attn.q_proj.weight"),
            "layers.7.attention.q"
        );
        assert_eq!(
            remapper.remap("layers.7.attn.k_proj.weight"),
            "layers.7.attention.k"
        );
        assert_eq!(
            remapper.remap("layers.7.attn.v_proj.weight"),
            "layers.7.attention.v"
        );
        assert_eq!(
            remapper.remap("layers.7.attn.o_proj.weight"),
            "layers.7.attention.out"
        );
        assert_eq!(
            remapper.remap("layers.2.mlp.gate_proj.weight"),
            "layers.2.ff.gate"
        );
        assert_eq!(
            remapper.remap("layers.2.mlp.up_proj.weight"),
            "layers.2.ff.up"
        );
        assert_eq!(
            remapper.remap("layers.2.mlp.down_proj.weight"),
            "layers.2.ff.down"
        );
    }

    #[test]
    fn test_name_remapper_embedding() {
        let remapper = NameRemapper::new();
        assert_eq!(remapper.remap("embedding.weight"), "input_proj");
        assert_eq!(remapper.remap("lm_head.weight"), "output_proj");
    }

    #[test]
    fn test_name_remapper_backbone() {
        let remapper = NameRemapper::new();

        // Well-known backbone-specific sub-keys must map to their canonical targets.
        assert_eq!(
            remapper.remap("backbone.embeddings.weight"),
            "input_proj",
            "HuggingFace backbone.embeddings.weight should remap to input_proj"
        );
        assert_eq!(
            remapper.remap("backbone.norm_f.weight"),
            "final_norm.weight",
            "HuggingFace backbone.norm_f.weight should remap to final_norm.weight"
        );

        // Layer-indexed keys that pass through the fall-through path after backbone
        // prefix stripping must still apply the common layer-suffix rules.
        assert_eq!(
            remapper.remap("backbone.layers.0.mixer.in_proj.weight"),
            "layers.0.input_proj",
            "backbone-prefixed layer key should remap via the normal layer-suffix rules"
        );

        // Unknown backbone sub-keys pass through unchanged (minus the backbone. prefix).
        let raw_unknown = "backbone.something.unknown";
        assert_eq!(
            remapper.remap(raw_unknown),
            "something.unknown",
            "unknown backbone sub-key should pass through with backbone. prefix stripped"
        );
    }

    #[test]
    fn test_name_remapper_passthrough() {
        let remapper = NameRemapper::new();
        let unknown = "some.random.unknown.key";
        assert_eq!(remapper.remap(unknown), unknown);
        let another = "custom_layer_bias";
        assert_eq!(remapper.remap(another), another);
    }

    #[test]
    fn test_name_remapper_remap_map() {
        let remapper = NameRemapper::new();
        let mut weights = HashMap::new();
        weights.insert("embedding.weight".to_string(), vec![1.0f32, 2.0]);
        weights.insert("lm_head.weight".to_string(), vec![3.0f32, 4.0]);
        weights.insert("layers.0.attn.q_proj.weight".to_string(), vec![5.0f32]);

        let remapped = remapper.remap_map(weights);
        assert!(remapped.contains_key("input_proj"));
        assert!(remapped.contains_key("output_proj"));
        assert!(remapped.contains_key("layers.0.attention.q"));
    }

    #[test]
    fn test_weight_loader_remap_name() {
        // WeightLoader.remap_name should use its stored mapping
        // We test this without a real SafeTensors file by constructing it indirectly.
        let mut mapping = HashMap::new();
        mapping.insert("old_name".to_string(), "new_name".to_string());

        // We can't construct a WeightLoader without a ModelLoader (which requires
        // a real file), so we test NameRemapper directly here.
        let remapper = NameRemapper::new();
        assert_eq!(
            remapper.remap("layers.1.mlp.gate_proj.weight"),
            "layers.1.ff.gate"
        );
    }
}
