//! PyTorch Checkpoint Compatibility
//!
//! Utilities for loading weights from PyTorch checkpoints and HuggingFace models.
//!
//! # Features
//!
//! - **PyTorch .pth/.pt loading**: Load weights from PyTorch checkpoint files
//! - **HuggingFace Hub integration**: Load models from HuggingFace repositories
//! - **Weight name mapping**: Automatic mapping between PyTorch and Rust naming conventions
//! - **Format conversion**: Convert PyTorch tensor formats to ndarray
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::pytorch_compat::PyTorchConverter;
//!
//! let converter = PyTorchConverter::new();
//! let weights = converter.load_checkpoint_raw("model.pth")?;
//! model.load_weights_from_dict(&weights)?;
//! ```

use crate::error::{ModelError, ModelResult};
use crate::gguf::GgufFile;
#[cfg(feature = "hf-hub")]
use crate::hf_hub::load_from_hub;
use scirs2_core::ndarray::{Array1, Array2, ArrayD, Ix2, IxDyn};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Convert a `candle_core::Tensor` to a dynamic-rank `ArrayD<f32>`.
///
/// The tensor is first converted to `f32` dtype (a no-op if already `f32`),
/// then flattened to a `Vec<f32>`, and finally reshaped into an `ArrayD`.
fn tensor_to_ndarray(t: &candle_core::Tensor) -> ModelResult<ArrayD<f32>> {
    use candle_core::DType;
    let t = t.to_dtype(DType::F32)?;
    let shape: Vec<usize> = t.shape().dims().to_vec();
    let data: Vec<f32> = t.flatten_all()?.to_vec1::<f32>()?;
    ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| ModelError::load_error("tensor_to_ndarray", format!("shape mismatch: {e}")))
}

// ---------------------------------------------------------------------------
// PthIndex — pytorch_model.bin.index.json parser
// ---------------------------------------------------------------------------

/// Parsed content of `pytorch_model.bin.index.json` for sharded HuggingFace
/// checkpoints.
#[derive(Debug)]
struct PthIndex {
    weight_map: std::collections::HashMap<String, String>,
}

impl PthIndex {
    /// Parse the index JSON string.
    fn from_json(json_str: &str) -> ModelResult<Self> {
        let v: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
            ModelError::load_error("PthIndex::from_json", format!("JSON parse error: {e}"))
        })?;
        let weight_map_val = v.get("weight_map").ok_or_else(|| {
            ModelError::load_error(
                "PthIndex::from_json",
                "missing 'weight_map' field in index JSON",
            )
        })?;
        let weight_map: std::collections::HashMap<String, String> =
            serde_json::from_value(weight_map_val.clone()).map_err(|e| {
                ModelError::load_error(
                    "PthIndex::from_json",
                    format!("weight_map parse error: {e}"),
                )
            })?;
        Ok(Self { weight_map })
    }

    /// Group tensor names by the shard file that contains them.
    ///
    /// Returns a `BTreeMap` so shards are iterated in deterministic order.
    fn shards_grouped(&self) -> std::collections::BTreeMap<String, Vec<String>> {
        let mut map: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();
        for (tensor_name, shard_file) in &self.weight_map {
            map.entry(shard_file.clone())
                .or_default()
                .push(tensor_name.clone());
        }
        map
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// PyTorch weight name mapping rules
#[derive(Debug, Clone)]
pub struct NameMapping {
    /// Source pattern (PyTorch naming)
    pub source: String,
    /// Target pattern (Rust naming)
    pub target: String,
}

impl NameMapping {
    /// Create a new name mapping
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
        }
    }
}

/// PyTorch checkpoint converter
#[derive(Debug)]
pub struct PyTorchConverter {
    /// Name mappings for weight conversion
    pub mappings: Vec<NameMapping>,
}

impl PyTorchConverter {
    /// Create a new PyTorch converter with default mappings
    pub fn new() -> Self {
        Self {
            mappings: Self::default_mappings(),
        }
    }

    /// Default name mappings for common architectures
    fn default_mappings() -> Vec<NameMapping> {
        vec![
            // Mamba/SSM mappings
            NameMapping::new("mixer.in_proj", "in_proj"),
            NameMapping::new("mixer.x_proj", "x_proj"),
            NameMapping::new("mixer.dt_proj", "dt_proj"),
            NameMapping::new("mixer.A_log", "log_a"),
            NameMapping::new("mixer.D", "d_skip"),
            NameMapping::new("mixer.out_proj", "out_proj"),
            NameMapping::new("mixer.conv1d", "conv"),
            // RWKV mappings
            NameMapping::new("time_mixing.time_decay", "time_decay"),
            NameMapping::new("time_mixing.time_first", "time_first"),
            NameMapping::new("time_mixing.key", "key_proj"),
            NameMapping::new("time_mixing.value", "value_proj"),
            NameMapping::new("time_mixing.receptance", "receptance_proj"),
            NameMapping::new("time_mixing.output", "output_proj"),
            // Channel mixing
            NameMapping::new("channel_mixing.key", "channel_key"),
            NameMapping::new("channel_mixing.value", "channel_value"),
            NameMapping::new("channel_mixing.receptance", "channel_receptance"),
            // Transformer mappings
            NameMapping::new("self_attn.q_proj", "q_proj"),
            NameMapping::new("self_attn.k_proj", "k_proj"),
            NameMapping::new("self_attn.v_proj", "v_proj"),
            NameMapping::new("self_attn.out_proj", "out_proj"),
            NameMapping::new("mlp.fc1", "fc1"),
            NameMapping::new("mlp.fc2", "fc2"),
            // Layer normalization
            NameMapping::new("layer_norm", "ln"),
            NameMapping::new("norm", "ln"),
            // Generic patterns
            NameMapping::new("weight", "weight"),
            NameMapping::new("bias", "bias"),
        ]
    }

    /// Add a custom name mapping
    pub fn add_mapping(&mut self, source: impl Into<String>, target: impl Into<String>) {
        self.mappings.push(NameMapping::new(source, target));
    }

    /// Map a PyTorch weight name to Rust naming convention
    pub fn map_name(&self, pytorch_name: &str) -> String {
        let mut result = pytorch_name.to_string();

        for mapping in &self.mappings {
            result = result.replace(&mapping.source, &mapping.target);
        }

        // Additional transformations
        result = result.replace("layers.", "layer_");
        result = result.replace("blocks.", "block_");
        result = result.replace(".", "_");

        result
    }

    /// Load a PyTorch `.pth` checkpoint and return all tensors as shape-preserving
    /// `ArrayD<f32>` arrays.
    ///
    /// Uses `candle_core::pickle::read_all` to parse the ZIP-embedded pickle
    /// format used by `torch.save`.  Every tensor is converted to `f32` before
    /// being handed to the caller.
    ///
    /// # Errors
    ///
    /// - File not found / cannot be opened
    /// - Invalid or unsupported pickle format
    /// - Dtype conversion failure
    /// - Shape mismatch between tensor metadata and raw data
    pub fn load_checkpoint_raw<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> ModelResult<HashMap<String, ArrayD<f32>>> {
        let path = path.as_ref();
        let pairs = candle_core::pickle::read_all(path).map_err(|e| {
            ModelError::load_error(
                "PyTorchConverter::load_checkpoint_raw",
                format!("failed to read '{}': {e}", path.display()),
            )
        })?;

        let mut result = HashMap::with_capacity(pairs.len());
        for (name, tensor) in pairs {
            let arr = tensor_to_ndarray(&tensor)?;
            result.insert(name, arr);
        }
        Ok(result)
    }

    /// Load checkpoint from PyTorch `.pth` file.
    ///
    /// Returns all tensors reshaped into `Array2<f32>`.  Scalars (rank 0) and
    /// 1-D tensors are wrapped into `(1, N)` matrices.  Tensors with rank ≥ 3
    /// return an error; use [`Self::load_checkpoint_raw`] for shape-preserving
    /// loading of higher-dimensional tensors.
    pub fn load_checkpoint<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> ModelResult<HashMap<String, Array2<f32>>> {
        let raw = self.load_checkpoint_raw(path)?;
        let mut out = HashMap::with_capacity(raw.len());
        for (name, arr) in raw {
            let shape = arr.shape().to_vec();
            let arr2 = match shape.len() {
                0 => {
                    let v = arr.into_raw_vec_and_offset().0;
                    Array2::from_shape_vec((1, 1), v).map_err(|e| {
                        ModelError::load_error(
                            "PyTorchConverter::load_checkpoint",
                            format!("scalar reshape for '{name}': {e}"),
                        )
                    })?
                }
                1 => {
                    let n = shape[0];
                    let v = arr.into_raw_vec_and_offset().0;
                    Array2::from_shape_vec((1, n), v).map_err(|e| {
                        ModelError::load_error(
                            "PyTorchConverter::load_checkpoint",
                            format!("1-D reshape for '{name}': {e}"),
                        )
                    })?
                }
                2 => arr.into_dimensionality::<Ix2>().map_err(|e| {
                    ModelError::load_error(
                        "PyTorchConverter::load_checkpoint",
                        format!("2-D cast for '{name}': {e}"),
                    )
                })?,
                n => {
                    return Err(ModelError::load_error(
                        "PyTorchConverter::load_checkpoint",
                        format!(
                            "tensor '{name}' has rank {n} — use load_checkpoint_raw \
                             for shape-preserving loading"
                        ),
                    ))
                }
            };
            out.insert(name, arr2);
        }
        Ok(out)
    }

    /// Load a sharded HuggingFace `.pth` checkpoint from a directory.
    ///
    /// Reads `pytorch_model.bin.index.json` to discover the shard files, then
    /// loads each shard in lexicographic order and merges all tensors into a
    /// single flat map.
    ///
    /// # Errors
    ///
    /// - Index file missing or unparseable
    /// - Any shard file missing or invalid
    pub fn load_pth_sharded<P: AsRef<Path>>(
        &self,
        dir: P,
    ) -> ModelResult<HashMap<String, ArrayD<f32>>> {
        let dir = dir.as_ref();
        let index_path = dir.join("pytorch_model.bin.index.json");
        let json_str = std::fs::read_to_string(&index_path).map_err(|e| {
            ModelError::load_error(
                "PyTorchConverter::load_pth_sharded",
                format!("cannot read index '{}': {e}", index_path.display()),
            )
        })?;
        let index = PthIndex::from_json(&json_str)?;
        let mut result = HashMap::new();
        for (shard_file, _tensor_names) in index.shards_grouped() {
            let shard_path = dir.join(&shard_file);
            let shard_tensors = self.load_checkpoint_raw(&shard_path)?;
            result.extend(shard_tensors);
        }
        Ok(result)
    }

    /// Split HuggingFace Mamba's fused `x_proj` weight into the three constituent
    /// sub-projections: `dt_proj`, `b_proj`, and `c_proj`.
    ///
    /// HuggingFace stores these concatenated along axis 0:
    /// ```text
    /// fused shape: [dt_rank + 2*d_state, inner_dim]
    ///              ├── dt_proj : [dt_rank,  inner_dim]
    ///              ├── b_proj  : [d_state,  inner_dim]
    ///              └── c_proj  : [d_state,  inner_dim]
    /// ```
    ///
    /// # Errors
    ///
    /// - Input is not a 2-D tensor
    /// - First dimension does not equal `dt_rank + 2 * d_state`
    pub fn split_x_proj(
        &self,
        fused: ArrayD<f32>,
        dt_rank: usize,
        d_state: usize,
    ) -> ModelResult<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        if fused.ndim() != 2 {
            return Err(ModelError::load_error(
                "PyTorchConverter::split_x_proj",
                format!("expected 2-D tensor, got {}D", fused.ndim()),
            ));
        }
        let total_rows = dt_rank + 2 * d_state;
        if fused.shape()[0] != total_rows {
            return Err(ModelError::load_error(
                "PyTorchConverter::split_x_proj",
                format!(
                    "expected {} rows (dt_rank={} + 2*d_state={}), got {}",
                    total_rows,
                    dt_rank,
                    d_state,
                    fused.shape()[0]
                ),
            ));
        }
        let fused_2d = fused.into_dimensionality::<Ix2>().map_err(|e| {
            ModelError::load_error(
                "PyTorchConverter::split_x_proj",
                format!("dimensionality cast: {e}"),
            )
        })?;
        use scirs2_core::ndarray::s;
        let dt_proj = fused_2d.slice(s![..dt_rank, ..]).to_owned();
        let b_proj = fused_2d
            .slice(s![dt_rank..dt_rank + d_state, ..])
            .to_owned();
        let c_proj = fused_2d.slice(s![dt_rank + d_state.., ..]).to_owned();
        Ok((dt_proj, b_proj, c_proj))
    }

    /// Load HuggingFace `.pth` checkpoint from the Hub.
    ///
    /// First tries to download `pytorch_model.bin.index.json`.  If present the
    /// model is sharded: all shard files are downloaded and merged.  If absent,
    /// `pytorch_model.bin` is downloaded directly.
    ///
    /// # Feature flag
    ///
    /// Requires the `hf-hub` Cargo feature.
    #[cfg(feature = "hf-hub")]
    pub fn load_from_huggingface_pth(
        &self,
        model_id: &str,
    ) -> ModelResult<HashMap<String, ArrayD<f32>>> {
        use crate::hf_hub::{HfHubClient, HfHubConfig};

        let config = HfHubConfig::default();
        let client = HfHubClient::new(config)?;

        // Try to download the shard index first.
        let index_result = client.download_file(model_id, "pytorch_model.bin.index.json", "main");

        if let Ok(index_path) = index_result {
            let dir = index_path.parent().ok_or_else(|| {
                ModelError::load_error(
                    "PyTorchConverter::load_from_huggingface_pth",
                    "downloaded index path has no parent directory",
                )
            })?;
            return self.load_pth_sharded(dir);
        }

        // Fall back to single-file download.
        let bin_path = client.download_file(model_id, "pytorch_model.bin", "main")?;
        self.load_checkpoint_raw(&bin_path)
    }

    /// Load weights from a HuggingFace Hub repository.
    ///
    /// Downloads all `.safetensors` shards for the given `model_id` from the
    /// HuggingFace Hub (using the `"main"` revision), converts every tensor from
    /// its native dtype to `f32`, reshapes each flat `Vec<f32>` into a 1×N
    /// `Array2<f32>`, and applies the converter's weight-name mappings before
    /// returning the result.
    ///
    /// Files are cached under `~/.cache/kizzasi/hub` (or a system temp fallback),
    /// so subsequent calls for the same model are served from disk without
    /// re-downloading.
    ///
    /// Authentication is read automatically from the `HF_TOKEN` environment
    /// variable when present.
    ///
    /// # Feature flag
    ///
    /// This method requires the `hf-hub` Cargo feature.  Without it the call
    /// always returns an informative `ModelError::LoadError`.
    ///
    /// # Errors
    ///
    /// - Network or HTTP failure during download
    /// - SafeTensors deserialisation failure
    /// - Shape error when reshaping a tensor (should not occur in practice)
    pub fn load_from_huggingface(
        &self,
        model_id: &str,
    ) -> ModelResult<HashMap<String, Array2<f32>>> {
        #[cfg(feature = "hf-hub")]
        {
            // Download all .safetensors shards and merge them into a flat
            // HashMap<tensor_name, Vec<f32>>.  The "main" revision is used by
            // default; callers that need a specific commit or tag should construct
            // an HfHubClient directly.
            let raw_weights = load_from_hub(model_id, "main", None)?;

            let mut result: HashMap<String, Array2<f32>> =
                HashMap::with_capacity(raw_weights.len());

            for (key, values) in raw_weights {
                let n = values.len();
                // Reshape the flat Vec<f32> into a 1×N Array2.  from_shape_vec
                // can only fail when the shape is inconsistent with the data
                // length, which cannot happen here because we derived n from
                // values.len().
                let array =
                    Array2::from_shape_vec((1, n), values).map_err(|e| ModelError::LoadError {
                        context: format!(
                            "PyTorchConverter::load_from_huggingface \
                             – reshape tensor \"{key}\""
                        ),
                        message: e.to_string(),
                    })?;

                // Apply the weight-name mapping so the resulting keys match the
                // Rust/internal naming conventions used in this crate.
                let mapped_key = self.map_name(&key);
                result.insert(mapped_key, array);
            }

            Ok(result)
        }

        #[cfg(not(feature = "hf-hub"))]
        {
            let _ = model_id; // suppress unused-variable warning
            Err(ModelError::LoadError {
                context: "PyTorchConverter::load_from_huggingface".to_string(),
                message:
                    "HuggingFace Hub integration requires the `hf-hub` Cargo feature. \
                          Enable it in Cargo.toml with: kizzasi-model = { features = [\"hf-hub\"] }"
                        .to_string(),
            })
        }
    }

    /// Convert PyTorch tensor shape to ndarray shape
    pub fn convert_shape(&self, pytorch_shape: &[i64]) -> Vec<usize> {
        pytorch_shape.iter().map(|&d| d as usize).collect()
    }

    /// Extract model configuration from a local checkpoint path.
    ///
    /// Attempts to read a `config.json` file located at `checkpoint` (when it is
    /// a directory) or alongside the checkpoint file (when it is a plain file,
    /// i.e. `checkpoint.parent()/config.json`).  Each top-level JSON string
    /// value is inserted as-is; numeric and boolean values are converted to
    /// their string representation.
    ///
    /// For HuggingFace Hub models where the checkpoint has not been downloaded
    /// yet, use `HfHubClient::download_file` (available with the `hf-hub`
    /// feature, see [`crate::hf_hub`]) to fetch `"config.json"` first:
    ///
    /// ```rust,ignore
    /// use kizzasi_model::hf_hub::{HfHubClient, HfHubConfig};
    /// let client = HfHubClient::default_client()?;
    /// let cfg_path = client.download_file("org/model", "config.json", "main")?;
    /// let converter = PyTorchConverter::new();
    /// let config = converter.extract_config(&cfg_path)?;
    /// ```
    pub fn extract_config(&self, checkpoint: &Path) -> ModelResult<HashMap<String, String>> {
        // Resolve the config.json location: if checkpoint is a directory look
        // inside it; otherwise look in the same directory as the file.
        let config_path = if checkpoint.is_dir() {
            checkpoint.join("config.json")
        } else {
            checkpoint
                .parent()
                .ok_or_else(|| ModelError::LoadError {
                    context: "PyTorchConverter::extract_config".to_string(),
                    message: format!(
                        "checkpoint path '{}' has no parent directory",
                        checkpoint.display()
                    ),
                })?
                .join("config.json")
        };

        if !config_path.exists() {
            return Err(ModelError::LoadError {
                context: "PyTorchConverter::extract_config".to_string(),
                message: format!(
                    "config.json not found at '{}'. \
                     For HuggingFace Hub models use \
                     `HfHubClient::download_file(repo_id, \"config.json\", \"main\")` \
                     to fetch it first.",
                    config_path.display()
                ),
            });
        }

        let raw = std::fs::read_to_string(&config_path).map_err(|e| ModelError::LoadError {
            context: "PyTorchConverter::extract_config – read config.json".to_string(),
            message: e.to_string(),
        })?;

        let json: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| ModelError::LoadError {
                context: "PyTorchConverter::extract_config – parse JSON".to_string(),
                message: e.to_string(),
            })?;

        let obj = json.as_object().ok_or_else(|| ModelError::LoadError {
            context: "PyTorchConverter::extract_config".to_string(),
            message: "config.json is not a JSON object at the top level".to_string(),
        })?;

        let mut config: HashMap<String, String> = HashMap::with_capacity(obj.len());
        for (k, v) in obj {
            let str_val = match v {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::Bool(b) => b.to_string(),
                serde_json::Value::Null => "null".to_string(),
                // Arrays and nested objects are serialised to compact JSON so
                // that no information is silently discarded.
                other => other.to_string(),
            };
            config.insert(k.clone(), str_val);
        }

        Ok(config)
    }
}

impl Default for PyTorchConverter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Weight-tying helper
// ---------------------------------------------------------------------------

/// Fill `"lm_head.weight"` (or its remapped form `"output_proj"`) from
/// `"embedding.weight"` when the checkpoint omits the output projection due to
/// weight-tying.
///
/// HuggingFace Mamba and many transformer checkpoints share the token-embedding
/// matrix between the input projection and the (transposed) output projection.
/// When this function is called on a raw weight map:
///
/// - If both `"lm_head.weight"` **and** `"output_proj"` are already present,
///   nothing is done (returns `Ok(())`).
/// - If either `"embedding.weight"` or `"input_proj"` is present, the
///   corresponding 2-D matrix is transposed and inserted under the key
///   `"output_proj"`.
/// - If no embedding weight is found at all, an error is returned.
///
/// # Key conventions
///
/// The function checks both the raw HuggingFace key names (`"embedding.weight"`,
/// `"lm_head.weight"`) and the remapped Kizzasi names (`"input_proj"`,
/// `"output_proj"`) so that it works regardless of whether `NameRemapper::remap`
/// has already been applied to the map.
///
/// # Errors
///
/// Returns `ModelError::LoadError` when neither `embedding.weight` nor
/// `input_proj` can be found in `weights`.
pub fn fill_lm_head_from_embedding(weights: &mut HashMap<String, Array2<f32>>) -> ModelResult<()> {
    // 1. Check if lm_head is already present under either key name.
    if weights.contains_key("lm_head.weight") || weights.contains_key("output_proj") {
        return Ok(());
    }

    // 2. Locate the embedding weight (may be raw or already remapped).
    let embedding = if let Some(e) = weights.get("embedding.weight") {
        e.clone()
    } else if let Some(e) = weights.get("input_proj") {
        e.clone()
    } else {
        return Err(ModelError::load_error(
            "fill_lm_head_from_embedding",
            "neither 'embedding.weight' nor 'input_proj' found in weight map; \
             cannot synthesize tied lm_head weight",
        ));
    };

    // 3. Transpose: embedding is [vocab_size, hidden_dim]; lm_head is
    //    [hidden_dim, vocab_size] in PyTorch convention (acts as a linear
    //    projection from hidden to logits). We store the transposed view.
    let transposed = embedding.t().to_owned();

    weights.insert("output_proj".to_string(), transposed);
    Ok(())
}

/// GGUF format support for quantized model loading
#[derive(Debug)]
pub struct GGUFLoader {
    /// File path
    pub path: String,
}

impl GGUFLoader {
    /// Create a new GGUF loader
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    /// Load quantized weights from GGUF file.
    ///
    /// Delegates to [`GgufFile`] for full GGUF parsing and dequantization.
    pub fn load_weights(&self) -> ModelResult<HashMap<String, Array2<f32>>> {
        let path = Path::new(&self.path);
        let gguf = GgufFile::open(path)?;
        gguf.load_all_tensors_f32()
    }

    /// Load quantized weights from an explicit path.
    ///
    /// Convenience method for cases where the path differs from `self.path`.
    pub fn load_weights_from_path(&self, path: &Path) -> ModelResult<HashMap<String, Array2<f32>>> {
        let gguf = GgufFile::open(path)?;
        gguf.load_all_tensors_f32()
    }

    /// Get quantization type(s) used in the GGUF file.
    ///
    /// Returns a comma-separated list of unique quantization type names present
    /// across all tensors in the file.
    pub fn get_quantization_type(&self) -> ModelResult<String> {
        let path = Path::new(&self.path);
        let gguf = GgufFile::open(path)?;
        let inspection = gguf.inspect();
        if inspection.quant_types_used.is_empty() {
            return Err(ModelError::simple_load_error(
                "No tensors found in GGUF file".to_string(),
            ));
        }
        Ok(inspection.quant_types_used.join(", "))
    }
}

/// Checkpoint format detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFormat {
    /// PyTorch .pth/.pt format
    PyTorch,
    /// SafeTensors format
    SafeTensors,
    /// GGUF quantized format
    GGUF,
    /// HuggingFace model directory
    HuggingFace,
    /// Unknown format
    Unknown,
}

impl CheckpointFormat {
    /// Detect checkpoint format from file extension
    pub fn detect(path: &Path) -> Self {
        if let Some(ext) = path.extension() {
            match ext.to_str() {
                Some("pth") | Some("pt") => CheckpointFormat::PyTorch,
                Some("safetensors") => CheckpointFormat::SafeTensors,
                Some("gguf") => CheckpointFormat::GGUF,
                _ => CheckpointFormat::Unknown,
            }
        } else if path.is_dir() {
            // Check if directory contains HuggingFace model files
            if path.join("config.json").exists() || path.join("pytorch_model.bin").exists() {
                CheckpointFormat::HuggingFace
            } else {
                CheckpointFormat::Unknown
            }
        } else {
            CheckpointFormat::Unknown
        }
    }
}

/// Weight conversion utilities
pub mod convert {
    use super::*;

    /// Convert PyTorch CHW format to HWC (for convolutions).
    ///
    /// For a 2-D weight matrix the "channel" axis is axis-0 and the spatial
    /// axis is axis-1.  Transposing gives the HWC (row-major contiguous) view
    /// that many inference back-ends expect.  The result is always a
    /// fresh owned allocation (`.to_owned()` after `.t()`).
    pub fn chw_to_hwc(tensor: &Array2<f32>) -> Array2<f32> {
        tensor.t().to_owned()
    }

    /// Convert PyTorch row-major to column-major if needed.
    ///
    /// When `needs_transpose` is `true` the matrix axes are swapped and the
    /// result is returned as a contiguous owned allocation.  Otherwise the
    /// input is cloned unchanged.
    pub fn transpose_if_needed(tensor: &Array2<f32>, needs_transpose: bool) -> Array2<f32> {
        if needs_transpose {
            tensor.t().to_owned()
        } else {
            tensor.clone()
        }
    }

    /// Dequantize INT8 weights to FP32
    pub fn dequantize_int8(
        quantized: &[i8],
        scale: f32,
        zero_point: i8,
    ) -> ModelResult<Array1<f32>> {
        let dequantized: Vec<f32> = quantized
            .iter()
            .map(|&q| ((q as i32 - zero_point as i32) as f32) * scale)
            .collect();

        Ok(Array1::from_vec(dequantized))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_mapping() {
        let converter = PyTorchConverter::new();

        let pytorch_name = "mixer.in_proj.weight";
        let mapped = converter.map_name(pytorch_name);

        assert!(mapped.contains("in_proj"));
        assert!(mapped.contains("weight"));
    }

    #[test]
    fn test_checkpoint_format_detection() {
        let pth_path = Path::new("model.pth");
        assert_eq!(
            CheckpointFormat::detect(pth_path),
            CheckpointFormat::PyTorch
        );

        let st_path = Path::new("model.safetensors");
        assert_eq!(
            CheckpointFormat::detect(st_path),
            CheckpointFormat::SafeTensors
        );

        let gguf_path = Path::new("model.gguf");
        assert_eq!(CheckpointFormat::detect(gguf_path), CheckpointFormat::GGUF);
    }

    #[test]
    fn test_shape_conversion() {
        let converter = PyTorchConverter::new();
        let pytorch_shape = vec![2i64, 3i64, 4i64];
        let rust_shape = converter.convert_shape(&pytorch_shape);

        assert_eq!(rust_shape, vec![2usize, 3usize, 4usize]);
    }

    #[test]
    fn test_dequantize_int8() {
        let quantized = vec![0i8, 10i8, -10i8, 127i8, -128i8];
        let scale = 0.1;
        let zero_point = 0i8;

        let dequantized = convert::dequantize_int8(&quantized, scale, zero_point)
            .expect("Failed to dequantize INT8");

        assert!((dequantized[0] - 0.0).abs() < 1e-5);
        assert!((dequantized[1] - 1.0).abs() < 1e-5);
        assert!((dequantized[2] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_add_custom_mapping() {
        let mut converter = PyTorchConverter::new();
        converter.add_mapping("custom.source", "custom_target");

        let mapped = converter.map_name("custom.source.weight");
        assert!(mapped.contains("custom_target"));
    }

    // -----------------------------------------------------------------------
    // fill_lm_head_from_embedding tests
    // -----------------------------------------------------------------------

    #[test]
    fn fill_lm_head_noop_when_lm_head_present() {
        let mut weights: HashMap<String, Array2<f32>> = HashMap::new();
        let emb = Array2::from_shape_vec((4, 3), vec![1.0f32; 12]).unwrap();
        let lm_head = Array2::from_shape_vec((3, 4), vec![2.0f32; 12]).unwrap();
        weights.insert("embedding.weight".to_string(), emb);
        weights.insert("lm_head.weight".to_string(), lm_head);

        fill_lm_head_from_embedding(&mut weights).expect("should succeed");
        // lm_head.weight must not have been overwritten
        assert_eq!(
            weights["lm_head.weight"][[0, 0]],
            2.0f32,
            "lm_head.weight should not be overwritten when already present"
        );
        // output_proj must NOT have been inserted (we only insert output_proj, not lm_head)
        assert!(
            !weights.contains_key("output_proj"),
            "output_proj should not be synthesised when lm_head.weight is present"
        );
    }

    #[test]
    fn fill_lm_head_noop_when_output_proj_present() {
        let mut weights: HashMap<String, Array2<f32>> = HashMap::new();
        let output = Array2::from_shape_vec((3, 4), vec![9.0f32; 12]).unwrap();
        weights.insert("output_proj".to_string(), output);

        fill_lm_head_from_embedding(&mut weights).expect("should succeed");
        // Nothing should have changed
        assert_eq!(weights.len(), 1);
        assert_eq!(
            weights["output_proj"][[0, 0]],
            9.0f32,
            "output_proj should not be overwritten"
        );
    }

    #[test]
    fn fill_lm_head_from_embedding_raw_key() {
        // Uses the HF raw key "embedding.weight" — output_proj must be synthesised.
        let vocab_size = 8;
        let hidden_dim = 4;
        let data: Vec<f32> = (0..(vocab_size * hidden_dim)).map(|i| i as f32).collect();
        let emb = Array2::from_shape_vec((vocab_size, hidden_dim), data).expect("valid shape");

        let mut weights: HashMap<String, Array2<f32>> = HashMap::new();
        weights.insert("embedding.weight".to_string(), emb.clone());

        fill_lm_head_from_embedding(&mut weights).expect("should succeed");

        assert!(
            weights.contains_key("output_proj"),
            "output_proj should be inserted"
        );
        let out = &weights["output_proj"];
        assert_eq!(
            out.shape(),
            &[hidden_dim, vocab_size],
            "output_proj should be the transpose of embedding"
        );
        // Spot-check: emb[[r, c]] == out[[c, r]]
        assert!((out[[0, 0]] - emb[[0, 0]]).abs() < 1e-6);
        assert!((out[[1, 0]] - emb[[0, 1]]).abs() < 1e-6);
        assert!((out[[0, 3]] - emb[[3, 0]]).abs() < 1e-6);
    }

    #[test]
    fn fill_lm_head_from_input_proj_remapped_key() {
        // Uses the remapped key "input_proj" (after NameRemapper applied).
        let vocab_size = 6;
        let hidden_dim = 3;
        let data: Vec<f32> = (0..(vocab_size * hidden_dim)).map(|i| i as f32).collect();
        let emb = Array2::from_shape_vec((vocab_size, hidden_dim), data).expect("valid shape");

        let mut weights: HashMap<String, Array2<f32>> = HashMap::new();
        weights.insert("input_proj".to_string(), emb.clone());

        fill_lm_head_from_embedding(&mut weights).expect("should succeed");

        let out = &weights["output_proj"];
        assert_eq!(out.shape(), &[hidden_dim, vocab_size]);
        // Spot-check transpose correctness.
        assert!((out[[2, 5]] - emb[[5, 2]]).abs() < 1e-6);
    }

    #[test]
    fn fill_lm_head_error_when_no_embedding() {
        let mut weights: HashMap<String, Array2<f32>> = HashMap::new();
        weights.insert("some.other.key".to_string(), Array2::zeros((2, 2)));
        let result = fill_lm_head_from_embedding(&mut weights);
        assert!(
            result.is_err(),
            "should return an error when no embedding weight is present"
        );
    }
}
