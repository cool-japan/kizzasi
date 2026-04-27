//! Incremental / streaming weight loading for large model (7B+) support.
//!
//! This module provides the [`WeightSource`] trait and two concrete implementations:
//!
//! - [`GgufFileSource`]: Streams weights from a GGUF file via seek-based lazy loading,
//!   dequantizing quantized types to f32 on the fly.
//! - [`SafeTensorsSource`]: Streams weights from a `.safetensors` file, converting
//!   BF16/F16/F32 data to f32 as each tensor is requested.
//!
//! The [`IncrementalModelLoader`] wraps any `WeightSource` and provides layer-by-layer
//! streaming via [`IncrementalModelLoader::load_all_streaming`], which is the primary
//! API for loading 7B+ models without holding all weights in RAM simultaneously.
//!
//! # Layer Prefix Extraction
//!
//! Tensor names following the `"layers.N.<rest>"` convention are grouped into
//! per-layer buckets identified by the prefix `"layers.N."`. Tensors that do not
//! match this pattern (e.g. `"embed"`, `"lm_head"`) are grouped under a synthetic
//! `"_misc."` prefix, which is always presented last in streaming order.
//!
//! # COOLJAPAN Policy Compliance
//!
//! - Pure Rust — no C or Fortran dependencies.
//! - No `unwrap()` anywhere; all error paths use `?` or `ok_or_else`.
//! - `serde_json` (workspace dep) for SafeTensors header parsing.
//! - `half` (workspace dep) for BF16/F16 → F32 conversion.

use crate::error::{ModelError, ModelResult};
use crate::gguf::{GgufFile, GgufQuantType, GgufTensorInfo};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// WeightSource trait
// ─────────────────────────────────────────────────────────────────────────────

/// A streaming/incremental source of model weight tensors.
///
/// Implementations must be `Send + Sync` so that they can be passed across
/// thread boundaries when used with multi-threaded inference runtimes.
///
/// The primary operations are:
/// - [`tensor_names`](WeightSource::tensor_names): enumerate all available tensor keys.
/// - [`load_tensor`](WeightSource::load_tensor): load and dequantize one tensor to `Vec<f32>`.
/// - [`contains`](WeightSource::contains): membership check without loading.
/// - [`total_bytes_estimate`](WeightSource::total_bytes_estimate): rough file-size hint for
///   progress reporting.
pub trait WeightSource: Send + Sync {
    /// Return the names of all tensors available in this source.
    fn tensor_names(&self) -> Vec<String>;

    /// Load and dequantize the tensor identified by `name`, returning a flat `Vec<f32>`.
    ///
    /// # Errors
    /// Returns [`ModelError`] if the tensor is not found, the file cannot be read,
    /// or dequantization fails.
    fn load_tensor(&mut self, name: &str) -> ModelResult<Vec<f32>>;

    /// Return `true` if the source contains a tensor with the given `name`.
    fn contains(&self, name: &str) -> bool;

    /// Return a rough estimate of the total number of bytes occupied by all
    /// tensor data in the underlying file (used for progress reporting).
    fn total_bytes_estimate(&self) -> u64;
}

// ─────────────────────────────────────────────────────────────────────────────
// GgufTensorMeta — lightweight per-tensor metadata stored by GgufFileSource
// ─────────────────────────────────────────────────────────────────────────────

/// Compact metadata record stored by [`GgufFileSource`] for each tensor.
#[derive(Debug, Clone)]
struct GgufTensorMeta {
    /// Absolute byte offset within the file where this tensor's data begins.
    data_offset: u64,
    /// Quantization type (determines dequantization path and byte size).
    quant_type: GgufQuantType,
    /// Total number of scalar elements in the tensor.
    n_elements: usize,
    /// Raw byte length of the tensor's on-disk data.
    byte_len: usize,
}

impl GgufTensorMeta {
    /// Build from a parsed [`GgufTensorInfo`] descriptor.
    fn from_info(info: &GgufTensorInfo) -> ModelResult<Self> {
        let n_elements = info.n_elements() as usize;
        let byte_len = compute_gguf_byte_len(&info.quant_type, n_elements, &info.name)?;
        Ok(Self {
            data_offset: info.data_offset,
            quant_type: info.quant_type,
            n_elements,
            byte_len,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GgufFileSource
// ─────────────────────────────────────────────────────────────────────────────

/// [`WeightSource`] backed by a GGUF file, using seek-based lazy tensor loading.
///
/// Each call to [`load_tensor`](WeightSource::load_tensor) seeks to the tensor's
/// data region, reads only the required bytes, and dequantizes them. This means
/// only one tensor's worth of raw data is in memory at a time.
pub struct GgufFileSource {
    /// Open file handle used for all seeks and reads.
    file: std::fs::File,
    /// Tensor metadata indexed by tensor name.
    tensor_infos: HashMap<String, GgufTensorMeta>,
    /// Total size of the underlying file in bytes.
    file_size: u64,
}

impl GgufFileSource {
    /// Open a GGUF file and parse its header + tensor index.
    ///
    /// The file is kept open after construction for subsequent lazy reads.
    pub fn open(path: &Path) -> ModelResult<Self> {
        // Parse header using GgufFile (reads entire file into memory temporarily
        // to parse the header; after this call the header data is dropped).
        let gguf = GgufFile::open(path)?;

        let file_size = std::fs::metadata(path)
            .map_err(|e| {
                ModelError::simple_load_error(format!("Failed to stat GGUF file {:?}: {}", path, e))
            })?
            .len();

        // Build metadata index
        let mut tensor_infos = HashMap::with_capacity(gguf.tensors.len());
        for info in &gguf.tensors {
            let meta = GgufTensorMeta::from_info(info)?;
            tensor_infos.insert(info.name.clone(), meta);
        }

        // Open file handle for subsequent reads
        let file = std::fs::File::open(path).map_err(|e| {
            ModelError::simple_load_error(format!("Failed to open GGUF file {:?}: {}", path, e))
        })?;

        Ok(Self {
            file,
            tensor_infos,
            file_size,
        })
    }
}

impl WeightSource for GgufFileSource {
    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensor_infos.keys().cloned().collect();
        names.sort();
        names
    }

    fn load_tensor(&mut self, name: &str) -> ModelResult<Vec<f32>> {
        let meta = self.tensor_infos.get(name).ok_or_else(|| {
            ModelError::simple_load_error(format!("GgufFileSource: tensor '{}' not found", name))
        })?;

        // Copy fields to avoid borrow conflicts with self.file
        let data_offset = meta.data_offset;
        let quant_type = meta.quant_type;
        let n_elements = meta.n_elements;
        let byte_len = meta.byte_len;

        self.file.seek(SeekFrom::Start(data_offset)).map_err(|e| {
            ModelError::simple_load_error(format!(
                "GgufFileSource: seek to tensor '{}' at offset {} failed: {}",
                name, data_offset, e
            ))
        })?;

        let mut raw = vec![0u8; byte_len];
        self.file.read_exact(&mut raw).map_err(|e| {
            ModelError::simple_load_error(format!(
                "GgufFileSource: read {} bytes for tensor '{}' failed: {}",
                byte_len, name, e
            ))
        })?;

        dequantize_gguf(&raw, &quant_type, n_elements, name)
    }

    fn contains(&self, name: &str) -> bool {
        self.tensor_infos.contains_key(name)
    }

    fn total_bytes_estimate(&self) -> u64 {
        self.file_size
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SafeTensorsSource
// ─────────────────────────────────────────────────────────────────────────────

/// Data type tag parsed from a SafeTensors header.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SafeTensorDtype {
    F32,
    F16,
    Bf16,
    F64,
}

impl SafeTensorDtype {
    /// Parse dtype string as it appears in the SafeTensors JSON header.
    fn from_str(s: &str) -> ModelResult<Self> {
        match s {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::Bf16),
            "F64" => Ok(Self::F64),
            other => Err(ModelError::simple_load_error(format!(
                "SafeTensorsSource: unsupported dtype '{}'",
                other
            ))),
        }
    }

    /// Number of bytes per scalar element.
    fn bytes_per_element(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::F64 => 8,
        }
    }
}

/// Per-tensor metadata as parsed from the SafeTensors JSON header.
#[derive(Debug, Clone)]
struct SafeTensorInfo {
    /// Parsed dtype.
    dtype: SafeTensorDtype,
    /// Shape dimensions (outermost first, standard row-major order).
    shape: Vec<usize>,
    /// `[begin, end)` byte offsets relative to the start of the data region.
    data_offsets: (u64, u64),
}

/// [`WeightSource`] backed by a `.safetensors` file.
///
/// The JSON header is parsed once at construction time; subsequent calls to
/// [`load_tensor`](WeightSource::load_tensor) seek directly to each tensor's
/// data region and convert it to f32.
pub struct SafeTensorsSource {
    /// Open file handle.
    file: std::fs::File,
    /// Per-tensor metadata.
    header: HashMap<String, SafeTensorInfo>,
    /// Absolute file offset where the raw data region begins (immediately after
    /// the JSON header, i.e. at byte `8 + header_size`).
    data_start_offset: u64,
    /// Total file size in bytes.
    file_size: u64,
}

impl SafeTensorsSource {
    /// Open a `.safetensors` file and parse its JSON header.
    ///
    /// The binary layout is:
    /// ```text
    /// [0..8]           header_size   — u64 LE: length of the JSON string
    /// [8..8+header_size] JSON header  — tensor metadata
    /// [8+header_size..]  raw data     — tensor bytes, BF16/F16/F32
    /// ```
    pub fn open(path: &Path) -> ModelResult<Self> {
        let mut file = std::fs::File::open(path).map_err(|e| {
            ModelError::simple_load_error(format!(
                "SafeTensorsSource: cannot open {:?}: {}",
                path, e
            ))
        })?;

        let file_size = file
            .seek(SeekFrom::End(0))
            .map_err(|e| ModelError::simple_load_error(format!("seek to end failed: {}", e)))?;

        // Rewind to start
        file.seek(SeekFrom::Start(0))
            .map_err(|e| ModelError::simple_load_error(format!("seek to start failed: {}", e)))?;

        // Read 8-byte header size
        let mut size_buf = [0u8; 8];
        file.read_exact(&mut size_buf).map_err(|e| {
            ModelError::simple_load_error(format!(
                "SafeTensorsSource: failed to read header size: {}",
                e
            ))
        })?;
        let header_size = u64::from_le_bytes(size_buf);

        // Read JSON header
        let mut json_buf = vec![0u8; header_size as usize];
        file.read_exact(&mut json_buf).map_err(|e| {
            ModelError::simple_load_error(format!(
                "SafeTensorsSource: failed to read {} bytes of JSON header: {}",
                header_size, e
            ))
        })?;

        let data_start_offset = 8 + header_size;

        // Parse JSON
        let json_str = std::str::from_utf8(&json_buf).map_err(|e| {
            ModelError::simple_load_error(format!(
                "SafeTensorsSource: JSON header is not valid UTF-8: {}",
                e
            ))
        })?;

        let root: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
            ModelError::simple_load_error(format!(
                "SafeTensorsSource: failed to parse JSON header: {}",
                e
            ))
        })?;

        let obj = root.as_object().ok_or_else(|| {
            ModelError::simple_load_error("SafeTensorsSource: JSON root is not an object")
        })?;

        let mut header = HashMap::with_capacity(obj.len());
        for (key, val) in obj {
            // Skip the special `__metadata__` key
            if key == "__metadata__" {
                continue;
            }

            let dtype_str = val.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| {
                ModelError::simple_load_error(format!(
                    "SafeTensorsSource: tensor '{}' missing 'dtype'",
                    key
                ))
            })?;

            let dtype = SafeTensorDtype::from_str(dtype_str)?;

            let shape_arr = val.get("shape").and_then(|v| v.as_array()).ok_or_else(|| {
                ModelError::simple_load_error(format!(
                    "SafeTensorsSource: tensor '{}' missing 'shape'",
                    key
                ))
            })?;

            let shape = shape_arr
                .iter()
                .map(|v| {
                    v.as_u64().ok_or_else(|| {
                        ModelError::simple_load_error(format!(
                            "SafeTensorsSource: tensor '{}' shape element is not a u64",
                            key
                        ))
                    })
                })
                .collect::<ModelResult<Vec<u64>>>()?
                .into_iter()
                .map(|d| d as usize)
                .collect();

            let offsets_arr = val
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    ModelError::simple_load_error(format!(
                        "SafeTensorsSource: tensor '{}' missing 'data_offsets'",
                        key
                    ))
                })?;

            if offsets_arr.len() != 2 {
                return Err(ModelError::simple_load_error(format!(
                    "SafeTensorsSource: tensor '{}' data_offsets must have 2 elements, got {}",
                    key,
                    offsets_arr.len()
                )));
            }

            let begin = offsets_arr[0].as_u64().ok_or_else(|| {
                ModelError::simple_load_error(format!(
                    "SafeTensorsSource: tensor '{}' data_offsets[0] is not a u64",
                    key
                ))
            })?;

            let end = offsets_arr[1].as_u64().ok_or_else(|| {
                ModelError::simple_load_error(format!(
                    "SafeTensorsSource: tensor '{}' data_offsets[1] is not a u64",
                    key
                ))
            })?;

            header.insert(
                key.clone(),
                SafeTensorInfo {
                    dtype,
                    shape,
                    data_offsets: (begin, end),
                },
            );
        }

        Ok(Self {
            file,
            header,
            data_start_offset,
            file_size,
        })
    }
}

impl WeightSource for SafeTensorsSource {
    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.header.keys().cloned().collect();
        names.sort();
        names
    }

    fn load_tensor(&mut self, name: &str) -> ModelResult<Vec<f32>> {
        let info = self.header.get(name).ok_or_else(|| {
            ModelError::simple_load_error(format!("SafeTensorsSource: tensor '{}' not found", name))
        })?;

        let (begin, end) = info.data_offsets;
        let byte_len = (end - begin) as usize;
        let dtype = info.dtype.clone();
        let n_elements: usize = if info.shape.is_empty() {
            1
        } else {
            info.shape.iter().product()
        };

        // Validate
        let expected_bytes = n_elements * dtype.bytes_per_element();
        if byte_len != expected_bytes {
            return Err(ModelError::simple_load_error(format!(
                "SafeTensorsSource: tensor '{}' byte range [{}, {}) has {} bytes, expected {} (shape={:?}, dtype={:?})",
                name, begin, end, byte_len, expected_bytes, info.shape, dtype
            )));
        }

        let abs_offset = self.data_start_offset + begin;
        self.file.seek(SeekFrom::Start(abs_offset)).map_err(|e| {
            ModelError::simple_load_error(format!(
                "SafeTensorsSource: seek to tensor '{}' at {} failed: {}",
                name, abs_offset, e
            ))
        })?;

        let mut raw = vec![0u8; byte_len];
        self.file.read_exact(&mut raw).map_err(|e| {
            ModelError::simple_load_error(format!(
                "SafeTensorsSource: read {} bytes for tensor '{}' failed: {}",
                byte_len, name, e
            ))
        })?;

        convert_safetensors_bytes_to_f32(&raw, &dtype, n_elements, name)
    }

    fn contains(&self, name: &str) -> bool {
        self.header.contains_key(name)
    }

    fn total_bytes_estimate(&self) -> u64 {
        self.file_size
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IncrementalModelLoader
// ─────────────────────────────────────────────────────────────────────────────

/// Layer prefix extracted from a tensor name.
///
/// Tensor names matching `"layers.<N>.<rest>"` get prefix `"layers.<N>."`.
/// All other tensor names are grouped under the synthetic `"_misc."` prefix.
const MISC_PREFIX: &str = "_misc.";

/// Wraps a [`WeightSource`] and provides layer-by-layer streaming iteration.
///
/// On construction, all tensor names are scanned to extract unique `"layers.N."`
/// prefixes. The [`load_all_streaming`](Self::load_all_streaming) method then
/// iterates over these prefixes in sorted order, loading one layer at a time
/// and invoking a user-supplied callback. Miscellaneous tensors (not matching
/// the `layers.N.` pattern) are presented last, grouped under `"_misc."`.
pub struct IncrementalModelLoader<S: WeightSource> {
    source: S,
    /// Sorted unique layer prefixes (e.g. `["layers.0.", "layers.1.", …, "_misc."]`).
    layer_prefixes: Vec<String>,
}

impl<S: WeightSource> IncrementalModelLoader<S> {
    /// Construct a new loader from a weight source.
    ///
    /// Tensor names are scanned once to build the list of unique layer prefixes.
    pub fn new(source: S) -> Self {
        let names = source.tensor_names();
        let mut prefixes: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut has_misc = false;

        for name in &names {
            if let Some(prefix) = extract_layer_prefix(name) {
                prefixes.insert(prefix);
            } else {
                has_misc = true;
            }
        }

        let mut layer_prefixes: Vec<String> = prefixes.into_iter().collect();
        if has_misc {
            layer_prefixes.push(MISC_PREFIX.to_string());
        }

        Self {
            source,
            layer_prefixes,
        }
    }

    /// Load all tensors whose names start with `prefix`.
    ///
    /// The special prefix `"_misc."` loads all tensors that do **not** match
    /// the `"layers.N."` pattern.
    ///
    /// Returns a `HashMap<tensor_name, Vec<f32>>` for the group.
    pub fn load_layer(&mut self, prefix: &str) -> ModelResult<HashMap<String, Vec<f32>>> {
        let names: Vec<String> = if prefix == MISC_PREFIX {
            // Collect tensors that do not belong to any regular layer prefix
            self.source
                .tensor_names()
                .into_iter()
                .filter(|n| extract_layer_prefix(n).is_none())
                .collect()
        } else {
            self.source
                .tensor_names()
                .into_iter()
                .filter(|n| n.starts_with(prefix))
                .collect()
        };

        let mut result = HashMap::with_capacity(names.len());
        for name in names {
            let tensor = self.source.load_tensor(&name)?;
            result.insert(name, tensor);
        }
        Ok(result)
    }

    /// Stream through all layers in order, invoking `callback` once per layer prefix.
    ///
    /// The callback receives:
    /// - `prefix`: the layer prefix string (e.g. `"layers.0."` or `"_misc."`)
    /// - `tensors`: a `HashMap<tensor_name, Vec<f32>>` for that layer
    ///
    /// If `callback` returns an `Err`, iteration stops immediately and the error
    /// is propagated.
    pub fn load_all_streaming<F>(&mut self, mut callback: F) -> ModelResult<()>
    where
        F: FnMut(&str, HashMap<String, Vec<f32>>) -> ModelResult<()>,
    {
        let prefixes = self.layer_prefixes.clone();
        for prefix in &prefixes {
            let tensors = self.load_layer(prefix)?;
            callback(prefix, tensors)?;
        }
        Ok(())
    }

    /// Return the list of unique layer prefixes discovered in the weight source.
    ///
    /// The list is sorted lexicographically; the `"_misc."` bucket (if present)
    /// always appears last.
    pub fn layer_prefixes(&self) -> &[String] {
        &self.layer_prefixes
    }

    /// Return a shared reference to the underlying weight source.
    pub fn source(&self) -> &S {
        &self.source
    }

    /// Consume the loader and return ownership of the underlying weight source.
    pub fn into_source(self) -> S {
        self.source
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the `"layers.N."` prefix from a tensor name, or return `None`.
///
/// Matches names of the form `"layers.<decimal>.<anything>"`.
fn extract_layer_prefix(name: &str) -> Option<String> {
    // Fast path: must start with "layers."
    let rest = name.strip_prefix("layers.")?;

    // Find the next dot after the layer index digits
    let dot_pos = rest.find('.')?;
    let idx_str = &rest[..dot_pos];

    // Validate that the index is all decimal digits
    if idx_str.is_empty() || !idx_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    Some(format!("layers.{}.", idx_str))
}

/// Dequantize raw GGUF bytes to f32 using the appropriate scheme.
fn dequantize_gguf(
    raw: &[u8],
    quant_type: &GgufQuantType,
    n_elements: usize,
    tensor_name: &str,
) -> ModelResult<Vec<f32>> {
    use crate::gguf::dequant;
    dequant::dequantize(raw, quant_type, n_elements).map_err(|e| {
        ModelError::simple_load_error(format!(
            "GgufFileSource: dequantize failed for tensor '{}': {}",
            tensor_name, e
        ))
    })
}

/// Convert raw SafeTensors bytes to a flat `Vec<f32>`.
fn convert_safetensors_bytes_to_f32(
    raw: &[u8],
    dtype: &SafeTensorDtype,
    n_elements: usize,
    tensor_name: &str,
) -> ModelResult<Vec<f32>> {
    match dtype {
        SafeTensorDtype::F32 => {
            if raw.len() != n_elements * 4 {
                return Err(ModelError::simple_load_error(format!(
                    "SafeTensorsSource: F32 tensor '{}' has {} bytes, expected {}",
                    tensor_name,
                    raw.len(),
                    n_elements * 4
                )));
            }
            Ok(raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        SafeTensorDtype::F16 => {
            if raw.len() != n_elements * 2 {
                return Err(ModelError::simple_load_error(format!(
                    "SafeTensorsSource: F16 tensor '{}' has {} bytes, expected {}",
                    tensor_name,
                    raw.len(),
                    n_elements * 2
                )));
            }
            Ok(raw
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect())
        }
        SafeTensorDtype::Bf16 => {
            if raw.len() != n_elements * 2 {
                return Err(ModelError::simple_load_error(format!(
                    "SafeTensorsSource: BF16 tensor '{}' has {} bytes, expected {}",
                    tensor_name,
                    raw.len(),
                    n_elements * 2
                )));
            }
            Ok(raw
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect())
        }
        SafeTensorDtype::F64 => {
            if raw.len() != n_elements * 8 {
                return Err(ModelError::simple_load_error(format!(
                    "SafeTensorsSource: F64 tensor '{}' has {} bytes, expected {}",
                    tensor_name,
                    raw.len(),
                    n_elements * 8
                )));
            }
            Ok(raw
                .chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect())
        }
    }
}

/// Compute the number of raw bytes occupied by a GGUF tensor on disk.
fn compute_gguf_byte_len(
    quant_type: &GgufQuantType,
    n_elements: usize,
    tensor_name: &str,
) -> ModelResult<usize> {
    // Helper for block-aligned types
    let block_check = |block_elems: usize, block_bytes: usize| -> ModelResult<usize> {
        if n_elements == 0 || !n_elements.is_multiple_of(block_elems) {
            return Err(ModelError::simple_load_error(format!(
                "GgufFileSource: tensor '{}' has {} elements, not a multiple of {}",
                tensor_name, n_elements, block_elems
            )));
        }
        Ok((n_elements / block_elems) * block_bytes)
    };

    match quant_type {
        GgufQuantType::F32 => Ok(n_elements * 4),
        GgufQuantType::F16 | GgufQuantType::BF16 => Ok(n_elements * 2),
        GgufQuantType::Q4_0 => block_check(32, 18),
        GgufQuantType::Q4_1 => block_check(32, 20),
        GgufQuantType::Q5_0 => block_check(32, 22),
        GgufQuantType::Q5_1 => block_check(32, 24),
        GgufQuantType::Q8_0 => block_check(32, 34),
        GgufQuantType::Q8_1 => block_check(32, 36),
        GgufQuantType::Q2K => block_check(256, 84),
        GgufQuantType::Q3K => block_check(256, 110),
        GgufQuantType::Q4K => block_check(256, 144),
        GgufQuantType::Q5K => block_check(256, 176),
        GgufQuantType::Q6K => block_check(256, 210),
        GgufQuantType::Q8K => block_check(256, 292),
        qt => Err(ModelError::simple_load_error(format!(
            "GgufFileSource: cannot compute byte size for unsupported quant type {:?} (tensor '{}')",
            qt, tensor_name
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SafeTensors binary builder ────────────────────────────────────────────

    /// Build a valid `.safetensors` binary from a slice of `(name, f32_data)` pairs.
    ///
    /// All tensors are stored as 1-D F32 arrays. The resulting bytes conform to
    /// the SafeTensors format specification:
    /// - 8 bytes: JSON header length (u64 LE)
    /// - `header_len` bytes: JSON header
    /// - remaining bytes: raw tensor data (F32 little-endian)
    fn make_synthetic_safetensors(tensors: &[(&str, Vec<f32>)]) -> Vec<u8> {
        // Build per-tensor data and accumulate byte offsets
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut tensor_metas: Vec<(&str, usize, usize, usize)> = Vec::new(); // name, begin, end, len

        for (name, vals) in tensors {
            let begin = data_bytes.len();
            for v in vals.iter() {
                data_bytes.extend_from_slice(&v.to_le_bytes());
            }
            let end = data_bytes.len();
            tensor_metas.push((name, begin, end, vals.len()));
        }

        // Build JSON header using serde_json
        let mut header_map = serde_json::Map::new();
        for (name, begin, end, n) in &tensor_metas {
            let entry = serde_json::json!({
                "dtype": "F32",
                "shape": [n],
                "data_offsets": [begin, end]
            });
            header_map.insert((*name).to_string(), entry);
        }
        let header_json = serde_json::Value::Object(header_map).to_string();
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        // Assemble file
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(header_bytes);
        out.extend_from_slice(&data_bytes);
        out
    }

    // ── Minimal GGUF binary builder ───────────────────────────────────────────

    /// Build a minimal GGUF binary containing a single F32 tensor.
    ///
    /// The file has the minimal valid structure:
    /// - 4 magic bytes `b"GGUF"`
    /// - u32 version = 2
    /// - u64 tensor_count = 1
    /// - u64 kv_count = 0
    /// - tensor info: name (u64 len + bytes), shape (u32 ndims=1, u64 dim), quant=0 (F32), offset=0
    /// - 32-byte alignment padding
    /// - tensor data: n * 4 bytes of f32 LE
    fn make_synthetic_gguf_f32(tensor_name: &str, values: &[f32]) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        // Magic
        buf.extend_from_slice(b"GGUF");
        // Version = 2 (u32 LE)
        buf.extend_from_slice(&2u32.to_le_bytes());
        // tensor_count (u64)
        buf.extend_from_slice(&1u64.to_le_bytes());
        // kv_count (u64)
        buf.extend_from_slice(&0u64.to_le_bytes());

        // No KV metadata entries

        // Tensor info
        let name_bytes = tensor_name.as_bytes();
        // name length (u64)
        buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
        // name bytes
        buf.extend_from_slice(name_bytes);
        // n_dims (u32)
        buf.extend_from_slice(&1u32.to_le_bytes());
        // dim[0] (u64)
        buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
        // quant type = 0 (F32, u32)
        buf.extend_from_slice(&0u32.to_le_bytes());
        // offset within data section = 0 (u64)
        buf.extend_from_slice(&0u64.to_le_bytes());

        // Pad header to 32-byte alignment
        let current_len = buf.len();
        let aligned = (current_len + 31) & !31;
        let pad = aligned - current_len;
        buf.extend(std::iter::repeat_n(0u8, pad));

        // Data section: raw f32 LE
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        buf
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_safetensors_source_single_tensor() {
        let tensors = &[("weight", vec![1.0f32, 2.0, 3.0, 4.0])];
        let data = make_synthetic_safetensors(tensors);
        let path = std::env::temp_dir().join("kizzasi_test_safetensors_single.safetensors");
        std::fs::write(&path, &data).expect("write test file");

        let mut src = SafeTensorsSource::open(&path).expect("open SafeTensorsSource");
        assert!(src.contains("weight"), "tensor 'weight' should be present");
        let loaded = src.load_tensor("weight").expect("load_tensor weight");
        assert_eq!(loaded, vec![1.0f32, 2.0, 3.0, 4.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_weight_source_contains() {
        let tensors = &[("alpha", vec![0.5f32, 1.5]), ("beta", vec![2.0f32, 3.0])];
        let data = make_synthetic_safetensors(tensors);
        let path = std::env::temp_dir().join("kizzasi_test_safetensors_contains.safetensors");
        std::fs::write(&path, &data).expect("write test file");

        let src = SafeTensorsSource::open(&path).expect("open");
        assert!(src.contains("alpha"));
        assert!(src.contains("beta"));
        assert!(
            !src.contains("gamma"),
            "should not contain non-existent tensor"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_incremental_loader_layer_prefixes() {
        let tensors = &[
            ("layers.0.weight", vec![1.0f32, 2.0]),
            ("layers.0.bias", vec![0.1f32]),
            ("layers.1.weight", vec![3.0f32, 4.0]),
            ("embed", vec![0.5f32]),
        ];
        let data = make_synthetic_safetensors(tensors);
        let path = std::env::temp_dir().join("kizzasi_test_safetensors_layer_prefixes.safetensors");
        std::fs::write(&path, &data).expect("write test file");

        let src = SafeTensorsSource::open(&path).expect("open");
        let loader = IncrementalModelLoader::new(src);

        let prefixes = loader.layer_prefixes();
        assert!(
            prefixes.contains(&"layers.0.".to_string()),
            "expected 'layers.0.' in prefixes, got {:?}",
            prefixes
        );
        assert!(
            prefixes.contains(&"layers.1.".to_string()),
            "expected 'layers.1.' in prefixes, got {:?}",
            prefixes
        );
        assert!(
            prefixes.contains(&MISC_PREFIX.to_string()),
            "expected '{}' in prefixes for 'embed', got {:?}",
            MISC_PREFIX,
            prefixes
        );
        // _misc. should be last
        assert_eq!(
            prefixes.last().map(String::as_str),
            Some(MISC_PREFIX),
            "_misc. prefix should be last"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_incremental_loader_streaming_callback() {
        let tensors = &[
            ("layers.0.weight", vec![1.0f32]),
            ("layers.0.bias", vec![0.0f32]),
            ("layers.1.weight", vec![2.0f32]),
            ("lm_head", vec![3.0f32]),
        ];
        let data = make_synthetic_safetensors(tensors);
        let path = std::env::temp_dir().join("kizzasi_test_safetensors_streaming.safetensors");
        std::fs::write(&path, &data).expect("write test file");

        let src = SafeTensorsSource::open(&path).expect("open");
        let mut loader = IncrementalModelLoader::new(src);

        let mut invocation_count = 0usize;
        loader
            .load_all_streaming(|_prefix, _tensors| {
                invocation_count += 1;
                Ok(())
            })
            .expect("streaming failed");

        // Expect 3 callbacks: layers.0., layers.1., _misc.
        assert_eq!(
            invocation_count, 3,
            "expected 3 callbacks (layers.0., layers.1., _misc.), got {}",
            invocation_count
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_gguf_file_source_lazy_load() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let data = make_synthetic_gguf_f32("test_tensor", &values);
        let path = std::env::temp_dir().join("kizzasi_test_gguf_source.gguf");
        std::fs::write(&path, &data).expect("write test gguf file");

        let mut src = GgufFileSource::open(&path).expect("open GgufFileSource");
        assert!(src.contains("test_tensor"), "tensor should be present");

        let loaded = src.load_tensor("test_tensor").expect("load_tensor");
        assert_eq!(loaded.len(), values.len(), "element count mismatch");
        for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "element {}: expected {}, got {}",
                i,
                expected,
                got
            );
        }
        assert!(
            !src.contains("nonexistent"),
            "nonexistent tensor should not be present"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_safetensors_source_multiple_tensors_values() {
        let tensors = &[("a", vec![10.0f32, 20.0, 30.0]), ("b", vec![-1.0f32, -2.0])];
        let data = make_synthetic_safetensors(tensors);
        let path = std::env::temp_dir().join("kizzasi_test_safetensors_multi.safetensors");
        std::fs::write(&path, &data).expect("write test file");

        let mut src = SafeTensorsSource::open(&path).expect("open");

        let a = src.load_tensor("a").expect("load a");
        assert_eq!(a, vec![10.0f32, 20.0, 30.0]);

        let b = src.load_tensor("b").expect("load b");
        assert_eq!(b, vec![-1.0f32, -2.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_extract_layer_prefix_valid() {
        assert_eq!(
            extract_layer_prefix("layers.0.weight"),
            Some("layers.0.".to_string())
        );
        assert_eq!(
            extract_layer_prefix("layers.123.bias"),
            Some("layers.123.".to_string())
        );
    }

    #[test]
    fn test_extract_layer_prefix_invalid() {
        assert_eq!(extract_layer_prefix("embed"), None);
        assert_eq!(extract_layer_prefix("lm_head.weight"), None);
        assert_eq!(extract_layer_prefix("layers_bad.0.weight"), None);
        assert_eq!(extract_layer_prefix("layers.abc.weight"), None);
    }

    #[test]
    fn test_weight_source_total_bytes_estimate() {
        let tensors = &[("x", vec![1.0f32, 2.0])];
        let data = make_synthetic_safetensors(tensors);
        let expected_size = data.len() as u64;
        let path = std::env::temp_dir().join("kizzasi_test_safetensors_bytes_estimate.safetensors");
        std::fs::write(&path, &data).expect("write");

        let src = SafeTensorsSource::open(&path).expect("open");
        assert_eq!(src.total_bytes_estimate(), expected_size);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_safetensors_source_missing_tensor_error() {
        let tensors = &[("existing", vec![1.0f32])];
        let data = make_synthetic_safetensors(tensors);
        let path = std::env::temp_dir().join("kizzasi_test_safetensors_missing.safetensors");
        std::fs::write(&path, &data).expect("write");

        let mut src = SafeTensorsSource::open(&path).expect("open");
        assert!(src.load_tensor("nonexistent").is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_incremental_loader_load_layer() {
        let tensors = &[
            ("layers.0.weight", vec![5.0f32, 6.0]),
            ("layers.0.bias", vec![0.5f32]),
            ("layers.1.weight", vec![7.0f32]),
        ];
        let data = make_synthetic_safetensors(tensors);
        let path = std::env::temp_dir().join("kizzasi_test_safetensors_load_layer.safetensors");
        std::fs::write(&path, &data).expect("write");

        let src = SafeTensorsSource::open(&path).expect("open");
        let mut loader = IncrementalModelLoader::new(src);

        let layer0 = loader
            .load_layer("layers.0.")
            .expect("load_layer layers.0.");
        assert!(layer0.contains_key("layers.0.weight"));
        assert!(layer0.contains_key("layers.0.bias"));
        assert!(!layer0.contains_key("layers.1.weight"));

        let _ = std::fs::remove_file(&path);
    }
}
