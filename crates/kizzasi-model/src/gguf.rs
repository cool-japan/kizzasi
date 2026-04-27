//! GGUF Format Support
//!
//! Implements parsing and tensor loading for the GGUF (Georgi Gerganov Unified Format)
//! binary format used by llama.cpp and related tools for storing quantized LLM weights.
//!
//! # Format Overview
//!
//! A GGUF file consists of:
//! - Magic bytes `b"GGUF"` followed by a u32 version number
//! - Tensor count and metadata KV count (u64 for v2+, u32 for v1)
//! - Metadata key-value pairs (typed)
//! - Tensor info array (name, shape, quant type, offset into data section)
//! - Data section (32-byte aligned after header)
//!
//! # Quantization Types
//!
//! Supports F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q6_K, Q2_K, Q3_K, Q4_K,
//! Q5_K, and Q8_K dequantization. IQ* types return an unsupported error.

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Return type of [`parse_gguf_buffer`]: `(version, metadata, tensors, data_offset)`.
type GgufBufferParsed = (
    u32,
    HashMap<String, GgufMetaValue>,
    Vec<GgufTensorInfo>,
    u64,
);

// ──────────────────────────────────────────────────────────────────────────────
// Binary parsing primitives
// ──────────────────────────────────────────────────────────────────────────────

fn read_u8(buf: &[u8], pos: &mut usize) -> ModelResult<u8> {
    if *pos >= buf.len() {
        return Err(ModelError::simple_load_error(format!(
            "Buffer underflow reading u8 at position {}",
            pos
        )));
    }
    let v = buf[*pos];
    *pos += 1;
    Ok(v)
}

fn read_u16_le(buf: &[u8], pos: &mut usize) -> ModelResult<u16> {
    let end = *pos + 2;
    if end > buf.len() {
        return Err(ModelError::simple_load_error(format!(
            "Buffer underflow reading u16 at position {}",
            pos
        )));
    }
    let v = u16::from_le_bytes([buf[*pos], buf[*pos + 1]]);
    *pos = end;
    Ok(v)
}

fn read_u32_le(buf: &[u8], pos: &mut usize) -> ModelResult<u32> {
    let end = *pos + 4;
    if end > buf.len() {
        return Err(ModelError::simple_load_error(format!(
            "Buffer underflow reading u32 at position {}",
            pos
        )));
    }
    let v = u32::from_le_bytes([buf[*pos], buf[*pos + 1], buf[*pos + 2], buf[*pos + 3]]);
    *pos = end;
    Ok(v)
}

fn read_u64_le(buf: &[u8], pos: &mut usize) -> ModelResult<u64> {
    let end = *pos + 8;
    if end > buf.len() {
        return Err(ModelError::simple_load_error(format!(
            "Buffer underflow reading u64 at position {}",
            pos
        )));
    }
    let v = u64::from_le_bytes([
        buf[*pos],
        buf[*pos + 1],
        buf[*pos + 2],
        buf[*pos + 3],
        buf[*pos + 4],
        buf[*pos + 5],
        buf[*pos + 6],
        buf[*pos + 7],
    ]);
    *pos = end;
    Ok(v)
}

fn read_i8(buf: &[u8], pos: &mut usize) -> ModelResult<i8> {
    read_u8(buf, pos).map(|v| v as i8)
}

fn read_i16_le(buf: &[u8], pos: &mut usize) -> ModelResult<i16> {
    read_u16_le(buf, pos).map(|v| v as i16)
}

fn read_i32_le(buf: &[u8], pos: &mut usize) -> ModelResult<i32> {
    read_u32_le(buf, pos).map(|v| v as i32)
}

fn read_i64_le(buf: &[u8], pos: &mut usize) -> ModelResult<i64> {
    read_u64_le(buf, pos).map(|v| v as i64)
}

fn read_f32_le(buf: &[u8], pos: &mut usize) -> ModelResult<f32> {
    read_u32_le(buf, pos).map(f32::from_bits)
}

fn read_f64_le(buf: &[u8], pos: &mut usize) -> ModelResult<f64> {
    read_u64_le(buf, pos).map(f64::from_bits)
}

fn read_bool(buf: &[u8], pos: &mut usize) -> ModelResult<bool> {
    read_u8(buf, pos).map(|v| v != 0)
}

/// Read a GGUF string using v2+ encoding (u64 length prefix).
fn read_string_v2(buf: &[u8], pos: &mut usize) -> ModelResult<String> {
    let len = read_u64_le(buf, pos)? as usize;
    let end = *pos + len;
    if end > buf.len() {
        return Err(ModelError::simple_load_error(format!(
            "Buffer underflow reading string of length {} at position {}",
            len, pos
        )));
    }
    let s = std::str::from_utf8(&buf[*pos..end]).map_err(|e| {
        ModelError::simple_load_error(format!("Invalid UTF-8 in GGUF string: {}", e))
    })?;
    let owned = s.to_owned();
    *pos = end;
    Ok(owned)
}

/// Read a GGUF string using v1 encoding (u32 length prefix).
fn read_string_v1(buf: &[u8], pos: &mut usize) -> ModelResult<String> {
    let len = read_u32_le(buf, pos)? as usize;
    let end = *pos + len;
    if end > buf.len() {
        return Err(ModelError::simple_load_error(format!(
            "Buffer underflow reading v1 string of length {} at position {}",
            len, pos
        )));
    }
    let s = std::str::from_utf8(&buf[*pos..end]).map_err(|e| {
        ModelError::simple_load_error(format!("Invalid UTF-8 in GGUF v1 string: {}", e))
    })?;
    let owned = s.to_owned();
    *pos = end;
    Ok(owned)
}

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// GGUF metadata value type discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> ModelResult<Self> {
        match v {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            other => Err(ModelError::simple_load_error(format!(
                "Unknown GGUF value type: {}",
                other
            ))),
        }
    }
}

/// Typed GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufMetaValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Array(Vec<GgufMetaValue>),
}

/// Quantization type for a GGUF tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufQuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    BF16 = 30,
}

impl GgufQuantType {
    fn from_u32(v: u32) -> ModelResult<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::IQ2XXS),
            17 => Ok(Self::IQ2XS),
            18 => Ok(Self::IQ3XXS),
            19 => Ok(Self::IQ1S),
            20 => Ok(Self::IQ4NL),
            21 => Ok(Self::IQ3S),
            22 => Ok(Self::IQ2S),
            23 => Ok(Self::IQ4XS),
            30 => Ok(Self::BF16),
            other => Err(ModelError::simple_load_error(format!(
                "Unknown GGUF quantization type: {}",
                other
            ))),
        }
    }

    /// Human-readable name for inspection output.
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2K",
            Self::Q3K => "Q3K",
            Self::Q4K => "Q4K",
            Self::Q5K => "Q5K",
            Self::Q6K => "Q6K",
            Self::Q8K => "Q8K",
            Self::IQ2XXS => "IQ2XXS",
            Self::IQ2XS => "IQ2XS",
            Self::IQ3XXS => "IQ3XXS",
            Self::IQ1S => "IQ1S",
            Self::IQ4NL => "IQ4NL",
            Self::IQ3S => "IQ3S",
            Self::IQ2S => "IQ2S",
            Self::IQ4XS => "IQ4XS",
            Self::BF16 => "BF16",
        }
    }
}

/// Information about a single tensor stored in a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Name of the tensor (e.g. `"token_embd.weight"`).
    pub name: String,
    /// Shape dimensions in GGUF order (innermost first).
    pub shape: Vec<u64>,
    /// Quantization / data type of the stored tensor.
    pub quant_type: GgufQuantType,
    /// Byte offset of this tensor's data relative to the start of the data section.
    pub offset: u64,
    /// Absolute byte offset of this tensor's data within the file.
    /// Populated as `data_section_start + offset` after parsing.
    pub data_offset: u64,
}

impl GgufTensorInfo {
    /// Total number of scalar elements in this tensor.
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// A parsed GGUF file ready for tensor loading.
#[derive(Debug)]
pub struct GgufFile {
    /// GGUF format version (1, 2, or 3).
    pub version: u32,
    /// All metadata key-value pairs.
    pub metadata: HashMap<String, GgufMetaValue>,
    /// Ordered list of tensor descriptors.
    pub tensors: Vec<GgufTensorInfo>,
    /// Byte offset in the file where the data section begins.
    data_offset: u64,
    /// Path to the source file (used when reading tensor data).
    file_path: PathBuf,
}

/// Lightweight summary of a GGUF file for quick inspection.
#[derive(Debug, Clone)]
pub struct GgufInspection {
    /// GGUF format version.
    pub version: u32,
    /// Number of tensors.
    pub tensor_count: usize,
    /// Number of metadata entries.
    pub metadata_count: usize,
    /// Model architecture string from `general.architecture`, if present.
    pub architecture: Option<String>,
    /// All tensor names in file order.
    pub tensor_names: Vec<String>,
    /// Total number of scalar parameters across all tensors.
    pub total_param_count: u64,
    /// Unique quantization type names used across all tensors.
    pub quant_types_used: Vec<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Parsing helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Read a single typed metadata value from `buf` at `*pos`.
/// `version` controls whether strings use u32 or u64 length prefixes.
fn read_meta_value(
    buf: &[u8],
    pos: &mut usize,
    vtype: GgufValueType,
    version: u32,
) -> ModelResult<GgufMetaValue> {
    match vtype {
        GgufValueType::Uint8 => Ok(GgufMetaValue::Uint8(read_u8(buf, pos)?)),
        GgufValueType::Int8 => Ok(GgufMetaValue::Int8(read_i8(buf, pos)?)),
        GgufValueType::Uint16 => Ok(GgufMetaValue::Uint16(read_u16_le(buf, pos)?)),
        GgufValueType::Int16 => Ok(GgufMetaValue::Int16(read_i16_le(buf, pos)?)),
        GgufValueType::Uint32 => Ok(GgufMetaValue::Uint32(read_u32_le(buf, pos)?)),
        GgufValueType::Int32 => Ok(GgufMetaValue::Int32(read_i32_le(buf, pos)?)),
        GgufValueType::Float32 => Ok(GgufMetaValue::Float32(read_f32_le(buf, pos)?)),
        GgufValueType::Bool => Ok(GgufMetaValue::Bool(read_bool(buf, pos)?)),
        GgufValueType::String => {
            let s = if version >= 2 {
                read_string_v2(buf, pos)?
            } else {
                read_string_v1(buf, pos)?
            };
            Ok(GgufMetaValue::String(s))
        }
        GgufValueType::Uint64 => Ok(GgufMetaValue::Uint64(read_u64_le(buf, pos)?)),
        GgufValueType::Int64 => Ok(GgufMetaValue::Int64(read_i64_le(buf, pos)?)),
        GgufValueType::Float64 => Ok(GgufMetaValue::Float64(read_f64_le(buf, pos)?)),
        GgufValueType::Array => {
            let elem_type_raw = read_u32_le(buf, pos)?;
            let elem_type = GgufValueType::from_u32(elem_type_raw)?;
            let count = if version >= 2 {
                read_u64_le(buf, pos)? as usize
            } else {
                read_u32_le(buf, pos)? as usize
            };
            let mut elements = Vec::with_capacity(count);
            for _ in 0..count {
                elements.push(read_meta_value(buf, pos, elem_type, version)?);
            }
            Ok(GgufMetaValue::Array(elements))
        }
    }
}

/// Parse the header and tensor info sections of the buffer.
/// Returns `(version, metadata, tensors, data_offset)`.
fn parse_gguf_buffer(buf: &[u8], file_path: &Path) -> ModelResult<GgufBufferParsed> {
    // Magic
    if buf.len() < 4 {
        return Err(ModelError::simple_load_error("File too small to be GGUF"));
    }
    if &buf[0..4] != b"GGUF" {
        return Err(ModelError::simple_load_error(format!(
            "Invalid GGUF magic in {:?}",
            file_path
        )));
    }
    let mut pos = 4usize;

    // Version
    let version = read_u32_le(buf, &mut pos)?;
    if version == 0 || version > 3 {
        return Err(ModelError::simple_load_error(format!(
            "Unsupported GGUF version: {}",
            version
        )));
    }

    // Counts differ between v1 and v2+
    let (tensor_count, kv_count) = if version >= 2 {
        let tc = read_u64_le(buf, &mut pos)? as usize;
        let kv = read_u64_le(buf, &mut pos)? as usize;
        (tc, kv)
    } else {
        let tc = read_u32_le(buf, &mut pos)? as usize;
        let kv = read_u32_le(buf, &mut pos)? as usize;
        (tc, kv)
    };

    // Metadata KV pairs
    let mut metadata = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = if version >= 2 {
            read_string_v2(buf, &mut pos)?
        } else {
            read_string_v1(buf, &mut pos)?
        };
        let vtype_raw = read_u32_le(buf, &mut pos)?;
        let vtype = GgufValueType::from_u32(vtype_raw)?;
        let value = read_meta_value(buf, &mut pos, vtype, version)?;
        metadata.insert(key, value);
    }

    // Tensor info
    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = if version >= 2 {
            read_string_v2(buf, &mut pos)?
        } else {
            read_string_v1(buf, &mut pos)?
        };

        let n_dims = read_u32_le(buf, &mut pos)? as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            if version >= 2 {
                shape.push(read_u64_le(buf, &mut pos)?);
            } else {
                shape.push(read_u32_le(buf, &mut pos)? as u64);
            }
        }

        let quant_raw = read_u32_le(buf, &mut pos)?;
        let quant_type = GgufQuantType::from_u32(quant_raw)?;
        let offset = read_u64_le(buf, &mut pos)?;

        // data_offset will be patched after we know the data section start
        tensors.push(GgufTensorInfo {
            name,
            shape,
            quant_type,
            offset,
            data_offset: 0, // placeholder; patched below
        });
    }

    // Data section begins at the next 32-byte aligned boundary after the header.
    let aligned_offset = (pos as u64 + 31) & !31u64;

    // Patch data_offset for each tensor to be the absolute file offset.
    for t in &mut tensors {
        t.data_offset = aligned_offset + t.offset;
    }

    Ok((version, metadata, tensors, aligned_offset))
}

// ──────────────────────────────────────────────────────────────────────────────
// GgufFile implementation
// ──────────────────────────────────────────────────────────────────────────────

impl GgufFile {
    /// Parse a GGUF file from the given path.
    ///
    /// The entire file is read into memory; tensor data is then read on demand
    /// via [`load_tensor_f32`](Self::load_tensor_f32).
    pub fn open(path: &Path) -> ModelResult<Self> {
        let buf = std::fs::read(path).map_err(|e| {
            ModelError::simple_load_error(format!("Failed to read GGUF file {:?}: {}", path, e))
        })?;
        let (version, metadata, tensors, data_offset) = parse_gguf_buffer(&buf, path)?;
        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
            file_path: path.to_path_buf(),
        })
    }

    /// Load a single tensor by name, dequantizing to f32, and reshape to 2D.
    ///
    /// If the tensor has more than 2 dimensions, all dimensions except the last
    /// are folded into the first axis (i.e. `[d0, d1, …, dn] → [d0*…*d(n-1), dn]`).
    /// A 1-D tensor of length `n` becomes `[1, n]`.
    pub fn load_tensor_f32(&self, name: &str) -> ModelResult<Array2<f32>> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| {
                ModelError::simple_load_error(format!(
                    "Tensor '{}' not found in GGUF file {:?}",
                    name, self.file_path
                ))
            })?;

        let n_elements = info.n_elements() as usize;

        // Read the raw bytes for this tensor from the file.
        let byte_offset = self.data_offset + info.offset;
        let byte_len = tensor_byte_size(info)?;

        let file_buf = std::fs::read(&self.file_path).map_err(|e| {
            ModelError::simple_load_error(format!(
                "Failed to re-read GGUF file {:?}: {}",
                self.file_path, e
            ))
        })?;

        let start = byte_offset as usize;
        let end = start + byte_len;
        if end > file_buf.len() {
            return Err(ModelError::simple_load_error(format!(
                "Tensor '{}' data region [{}, {}) exceeds file size {}",
                name,
                start,
                end,
                file_buf.len()
            )));
        }
        let raw = &file_buf[start..end];

        let floats = dequant::dequantize(raw, &info.quant_type, n_elements)?;

        // Reshape to 2D
        let (rows, cols) = shape_to_2d(&info.shape);
        Array2::from_shape_vec((rows, cols), floats).map_err(|e| {
            ModelError::simple_load_error(format!(
                "Failed to reshape tensor '{}' to ({}, {}): {}",
                name, rows, cols, e
            ))
        })
    }

    /// Load all tensors in the file into a `HashMap<name, Array2<f32>>`.
    pub fn load_all_tensors_f32(&self) -> ModelResult<HashMap<String, Array2<f32>>> {
        // Read the file once and reuse it for all tensors.
        let file_buf = std::fs::read(&self.file_path).map_err(|e| {
            ModelError::simple_load_error(format!(
                "Failed to read GGUF file {:?}: {}",
                self.file_path, e
            ))
        })?;

        let mut result = HashMap::with_capacity(self.tensors.len());
        for info in &self.tensors {
            let n_elements = info.n_elements() as usize;
            let byte_offset = (self.data_offset + info.offset) as usize;
            let byte_len = tensor_byte_size(info)?;
            let end = byte_offset + byte_len;

            if end > file_buf.len() {
                return Err(ModelError::simple_load_error(format!(
                    "Tensor '{}' data region [{}, {}) exceeds file size {}",
                    info.name,
                    byte_offset,
                    end,
                    file_buf.len()
                )));
            }
            let raw = &file_buf[byte_offset..end];
            let floats = dequant::dequantize(raw, &info.quant_type, n_elements)?;

            let (rows, cols) = shape_to_2d(&info.shape);
            let array = Array2::from_shape_vec((rows, cols), floats).map_err(|e| {
                ModelError::simple_load_error(format!(
                    "Failed to reshape tensor '{}' to ({}, {}): {}",
                    info.name, rows, cols, e
                ))
            })?;
            result.insert(info.name.clone(), array);
        }
        Ok(result)
    }

    /// Return the model architecture string from the `general.architecture` metadata key.
    pub fn architecture(&self) -> Option<&str> {
        match self.metadata.get("general.architecture") {
            Some(GgufMetaValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Build a lightweight inspection summary of this file.
    pub fn inspect(&self) -> GgufInspection {
        let tensor_names: Vec<String> = self.tensors.iter().map(|t| t.name.clone()).collect();
        let total_param_count: u64 = self.tensors.iter().map(|t| t.n_elements()).sum();

        // Collect unique quant type names
        let mut quant_set: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for t in &self.tensors {
            quant_set.insert(t.quant_type.name());
        }
        let mut quant_types_used: Vec<String> =
            quant_set.into_iter().map(|s| s.to_owned()).collect();
        quant_types_used.sort();

        GgufInspection {
            version: self.version,
            tensor_count: self.tensors.len(),
            metadata_count: self.metadata.len(),
            architecture: self.architecture().map(|s| s.to_owned()),
            tensor_names,
            total_param_count,
            quant_types_used,
        }
    }

    /// Return a slice of all tensor names in file order.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// Load a single tensor lazily by name using an open `Read + Seek` handle.
    ///
    /// This avoids re-reading the whole file; instead it seeks directly to the
    /// tensor data region and reads only the required bytes. The result is a
    /// flat `Vec<f32>` of dequantized values.
    ///
    /// # Parameters
    /// - `reader`: An open handle to the GGUF file (must support both `Read` and `Seek`).
    /// - `info`:   The [`GgufTensorInfo`] descriptor for the tensor to load.
    pub fn load_tensor_lazy<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        info: &GgufTensorInfo,
    ) -> ModelResult<Vec<f32>> {
        use std::io::SeekFrom;

        let n_elements = info.n_elements() as usize;
        let byte_len = tensor_byte_size(info)?;

        // Seek to the absolute file offset of this tensor's data.
        reader
            .seek(SeekFrom::Start(info.data_offset))
            .map_err(|e| {
                ModelError::simple_load_error(format!(
                    "Failed to seek to tensor '{}' at offset {}: {}",
                    info.name, info.data_offset, e
                ))
            })?;

        let mut raw = vec![0u8; byte_len];
        reader.read_exact(&mut raw).map_err(|e| {
            ModelError::simple_load_error(format!(
                "Failed to read {} bytes for tensor '{}': {}",
                byte_len, info.name, e
            ))
        })?;

        dequant::dequantize(&raw, &info.quant_type, n_elements)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Shape utilities
// ──────────────────────────────────────────────────────────────────────────────

/// Convert an n-dimensional shape to `(rows, cols)` for Array2.
/// - 0-D → (1, 1)
/// - 1-D `[n]` → (1, n)
/// - 2-D `[r, c]` → (r, c)
/// - N-D `[d0, …, d(n-1), dn]` → (d0*…*d(n-2), d(n-1))  [i.e. flatten all but last]
fn shape_to_2d(shape: &[u64]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0] as usize),
        2 => (shape[0] as usize, shape[1] as usize),
        n => {
            let cols = shape[n - 1] as usize;
            let rows: u64 = shape[..n - 1].iter().product();
            (rows as usize, cols)
        }
    }
}

/// Compute the number of bytes occupied by a tensor's data on disk.
fn tensor_byte_size(info: &GgufTensorInfo) -> ModelResult<usize> {
    let n = info.n_elements() as usize;
    match info.quant_type {
        GgufQuantType::F32 => Ok(n * 4),
        GgufQuantType::F16 | GgufQuantType::BF16 => Ok(n * 2),
        GgufQuantType::Q4_0 => {
            // 32 elements per block, block = 18 bytes
            if !n.is_multiple_of(32) {
                return Err(ModelError::simple_load_error(format!(
                    "Q4_0 tensor '{}' has {} elements, not divisible by 32",
                    info.name, n
                )));
            }
            Ok((n / 32) * 18)
        }
        GgufQuantType::Q4_1 => {
            if !n.is_multiple_of(32) {
                return Err(ModelError::simple_load_error(format!(
                    "Q4_1 tensor '{}' has {} elements, not divisible by 32",
                    info.name, n
                )));
            }
            Ok((n / 32) * 20)
        }
        GgufQuantType::Q5_0 => {
            if !n.is_multiple_of(32) {
                return Err(ModelError::simple_load_error(format!(
                    "Q5_0 tensor '{}' has {} elements, not divisible by 32",
                    info.name, n
                )));
            }
            Ok((n / 32) * 22)
        }
        GgufQuantType::Q5_1 => {
            if !n.is_multiple_of(32) {
                return Err(ModelError::simple_load_error(format!(
                    "Q5_1 tensor '{}' has {} elements, not divisible by 32",
                    info.name, n
                )));
            }
            Ok((n / 32) * 24)
        }
        GgufQuantType::Q8_0 => {
            if !n.is_multiple_of(32) {
                return Err(ModelError::simple_load_error(format!(
                    "Q8_0 tensor '{}' has {} elements, not divisible by 32",
                    info.name, n
                )));
            }
            Ok((n / 32) * 34)
        }
        GgufQuantType::Q8_1 => {
            if !n.is_multiple_of(32) {
                return Err(ModelError::simple_load_error(format!(
                    "Q8_1 tensor '{}' has {} elements, not divisible by 32",
                    info.name, n
                )));
            }
            Ok((n / 32) * 36)
        }
        GgufQuantType::Q6K => {
            // 256 elements per block, block = 210 bytes
            if !n.is_multiple_of(256) {
                return Err(ModelError::simple_load_error(format!(
                    "Q6K tensor '{}' has {} elements, not divisible by 256",
                    info.name, n
                )));
            }
            Ok((n / 256) * 210)
        }
        GgufQuantType::Q2K => {
            if !n.is_multiple_of(256) {
                return Err(ModelError::simple_load_error(format!(
                    "Q2K tensor has {} elements, not divisible by 256",
                    n
                )));
            }
            Ok((n / 256) * 84)
        }
        GgufQuantType::Q3K => {
            if !n.is_multiple_of(256) {
                return Err(ModelError::simple_load_error(format!(
                    "Q3K tensor has {} elements, not divisible by 256",
                    n
                )));
            }
            Ok((n / 256) * 110)
        }
        GgufQuantType::Q4K => {
            if !n.is_multiple_of(256) {
                return Err(ModelError::simple_load_error(format!(
                    "Q4K tensor has {} elements, not divisible by 256",
                    n
                )));
            }
            Ok((n / 256) * 144)
        }
        GgufQuantType::Q5K => {
            if !n.is_multiple_of(256) {
                return Err(ModelError::simple_load_error(format!(
                    "Q5K tensor has {} elements, not divisible by 256",
                    n
                )));
            }
            Ok((n / 256) * 176)
        }
        GgufQuantType::Q8K => {
            if !n.is_multiple_of(256) {
                return Err(ModelError::simple_load_error(format!(
                    "Q8K tensor has {} elements, not divisible by 256",
                    n
                )));
            }
            Ok((n / 256) * 292)
        }
        // IQ types – variable block sizes; we return an error
        qt => Err(ModelError::simple_load_error(format!(
            "Cannot compute byte size for unsupported quant type {:?}",
            qt
        ))),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Dequantization
// ──────────────────────────────────────────────────────────────────────────────

pub(crate) mod dequant {
    use super::GgufQuantType;
    use crate::error::{ModelError, ModelResult};
    use crate::gguf_dequant as kquant;

    /// Dequantize `data` bytes to `n_elements` f32 values according to `quant_type`.
    pub fn dequantize(
        data: &[u8],
        quant_type: &GgufQuantType,
        n_elements: usize,
    ) -> ModelResult<Vec<f32>> {
        match quant_type {
            GgufQuantType::F32 => dequant_f32(data, n_elements),
            GgufQuantType::F16 => dequant_f16(data, n_elements),
            GgufQuantType::BF16 => dequant_bf16(data, n_elements),
            GgufQuantType::Q4_0 => dequant_q4_0(data, n_elements),
            GgufQuantType::Q4_1 => dequant_q4_1(data, n_elements),
            GgufQuantType::Q5_0 => dequant_q5_0(data, n_elements),
            GgufQuantType::Q5_1 => dequant_q5_1(data, n_elements),
            GgufQuantType::Q8_0 => dequant_q8_0(data, n_elements),
            GgufQuantType::Q6K => dequant_q6_k(data, n_elements),
            GgufQuantType::Q2K => kquant::dequant_q2_k(data, n_elements),
            GgufQuantType::Q3K => kquant::dequant_q3_k(data, n_elements),
            GgufQuantType::Q4K => kquant::dequant_q4_k(data, n_elements),
            GgufQuantType::Q5K => kquant::dequant_q5_k(data, n_elements),
            GgufQuantType::Q8K => kquant::dequant_q8_k(data, n_elements),
            qt => Err(ModelError::simple_load_error(format!(
                "Unsupported quant type for dequantization: {:?}",
                qt
            ))),
        }
    }

    fn dequant_f32(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
        if data.len() < n * 4 {
            return Err(ModelError::simple_load_error(format!(
                "F32 tensor needs {} bytes, got {}",
                n * 4,
                data.len()
            )));
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * 4;
            let v =
                f32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]]);
            out.push(v);
        }
        Ok(out)
    }

    pub(super) fn dequant_f16(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
        if data.len() < n * 2 {
            return Err(ModelError::simple_load_error(format!(
                "F16 tensor needs {} bytes, got {}",
                n * 2,
                data.len()
            )));
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * 2;
            let bits = u16::from_le_bytes([data[base], data[base + 1]]);
            out.push(half::f16::from_bits(bits).to_f32());
        }
        Ok(out)
    }

    pub(super) fn dequant_bf16(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
        if data.len() < n * 2 {
            return Err(ModelError::simple_load_error(format!(
                "BF16 tensor needs {} bytes, got {}",
                n * 2,
                data.len()
            )));
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * 2;
            let bits = u16::from_le_bytes([data[base], data[base + 1]]);
            // BF16 → f32: sign+exp+7 mantissa bits occupy the upper 16 bits of f32
            out.push(f32::from_bits((bits as u32) << 16));
        }
        Ok(out)
    }

    /// Q4_0 block: 2 bytes delta (f16) + 16 bytes quantized nibbles → 32 f32
    ///
    /// Each nibble `q` represents `(q - 8) * delta`.
    pub(super) fn dequant_q4_0(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
        const BLOCK_ELEMS: usize = 32;
        const BLOCK_BYTES: usize = 18; // 2 delta + 16 nibbles
        if !n.is_multiple_of(BLOCK_ELEMS) {
            return Err(ModelError::simple_load_error(format!(
                "Q4_0: n_elements {} not divisible by {}",
                n, BLOCK_ELEMS
            )));
        }
        let n_blocks = n / BLOCK_ELEMS;
        if data.len() < n_blocks * BLOCK_BYTES {
            return Err(ModelError::simple_load_error("Q4_0 data buffer too small"));
        }
        let mut out = Vec::with_capacity(n);
        for b in 0..n_blocks {
            let base = b * BLOCK_BYTES;
            let delta_bits = u16::from_le_bytes([data[base], data[base + 1]]);
            let delta = half::f16::from_bits(delta_bits).to_f32();
            for byte_idx in 0..16usize {
                let byte = data[base + 2 + byte_idx];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                out.push(lo as f32 * delta);
                out.push(hi as f32 * delta);
            }
        }
        Ok(out)
    }

    /// Q4_1 block: 2 bytes delta (f16) + 2 bytes min (f16) + 16 bytes nibbles → 32 f32
    ///
    /// Each nibble `q` represents `q * delta + min`.
    pub(super) fn dequant_q4_1(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
        const BLOCK_ELEMS: usize = 32;
        const BLOCK_BYTES: usize = 20;
        if !n.is_multiple_of(BLOCK_ELEMS) {
            return Err(ModelError::simple_load_error(format!(
                "Q4_1: n_elements {} not divisible by {}",
                n, BLOCK_ELEMS
            )));
        }
        let n_blocks = n / BLOCK_ELEMS;
        if data.len() < n_blocks * BLOCK_BYTES {
            return Err(ModelError::simple_load_error("Q4_1 data buffer too small"));
        }
        let mut out = Vec::with_capacity(n);
        for b in 0..n_blocks {
            let base = b * BLOCK_BYTES;
            let delta_bits = u16::from_le_bytes([data[base], data[base + 1]]);
            let delta = half::f16::from_bits(delta_bits).to_f32();
            let min_bits = u16::from_le_bytes([data[base + 2], data[base + 3]]);
            let min = half::f16::from_bits(min_bits).to_f32();
            for byte_idx in 0..16usize {
                let byte = data[base + 4 + byte_idx];
                let lo = (byte & 0x0F) as f32;
                let hi = ((byte >> 4) & 0x0F) as f32;
                out.push(lo * delta + min);
                out.push(hi * delta + min);
            }
        }
        Ok(out)
    }

    /// Q5_0 block: 2 bytes delta (f16) + 4 bytes high bits (u32) + 16 bytes low nibbles → 32 f32
    ///
    /// Each 5-bit value `q` (range 0–31, then subtract 16) scaled by delta.
    pub(super) fn dequant_q5_0(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
        const BLOCK_ELEMS: usize = 32;
        const BLOCK_BYTES: usize = 22;
        if !n.is_multiple_of(BLOCK_ELEMS) {
            return Err(ModelError::simple_load_error(format!(
                "Q5_0: n_elements {} not divisible by {}",
                n, BLOCK_ELEMS
            )));
        }
        let n_blocks = n / BLOCK_ELEMS;
        if data.len() < n_blocks * BLOCK_BYTES {
            return Err(ModelError::simple_load_error("Q5_0 data buffer too small"));
        }
        let mut out = Vec::with_capacity(n);
        for b in 0..n_blocks {
            let base = b * BLOCK_BYTES;
            let delta_bits = u16::from_le_bytes([data[base], data[base + 1]]);
            let delta = half::f16::from_bits(delta_bits).to_f32();
            // High bits: bit i of qh → 5th bit of element i
            let qh = u32::from_le_bytes([
                data[base + 2],
                data[base + 3],
                data[base + 4],
                data[base + 5],
            ]);
            for byte_idx in 0..16usize {
                let byte = data[base + 6 + byte_idx];
                let lo4 = (byte & 0x0F) as u32;
                let hi4 = ((byte >> 4) & 0x0F) as u32;
                let elem_lo = byte_idx * 2;
                let elem_hi = byte_idx * 2 + 1;
                let hi_lo = (qh >> elem_lo) & 1;
                let hi_hi = (qh >> elem_hi) & 1;
                let q_lo = (lo4 | (hi_lo << 4)) as i32 - 16;
                let q_hi = (hi4 | (hi_hi << 4)) as i32 - 16;
                out.push(q_lo as f32 * delta);
                out.push(q_hi as f32 * delta);
            }
        }
        Ok(out)
    }

    /// Q5_1 block: 2 bytes delta (f16) + 2 bytes min (f16) + 4 bytes high bits + 16 bytes → 32 f32
    pub(super) fn dequant_q5_1(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
        const BLOCK_ELEMS: usize = 32;
        const BLOCK_BYTES: usize = 24;
        if !n.is_multiple_of(BLOCK_ELEMS) {
            return Err(ModelError::simple_load_error(format!(
                "Q5_1: n_elements {} not divisible by {}",
                n, BLOCK_ELEMS
            )));
        }
        let n_blocks = n / BLOCK_ELEMS;
        if data.len() < n_blocks * BLOCK_BYTES {
            return Err(ModelError::simple_load_error("Q5_1 data buffer too small"));
        }
        let mut out = Vec::with_capacity(n);
        for b in 0..n_blocks {
            let base = b * BLOCK_BYTES;
            let delta_bits = u16::from_le_bytes([data[base], data[base + 1]]);
            let delta = half::f16::from_bits(delta_bits).to_f32();
            let min_bits = u16::from_le_bytes([data[base + 2], data[base + 3]]);
            let min = half::f16::from_bits(min_bits).to_f32();
            let qh = u32::from_le_bytes([
                data[base + 4],
                data[base + 5],
                data[base + 6],
                data[base + 7],
            ]);
            for byte_idx in 0..16usize {
                let byte = data[base + 8 + byte_idx];
                let lo4 = (byte & 0x0F) as u32;
                let hi4 = ((byte >> 4) & 0x0F) as u32;
                let elem_lo = byte_idx * 2;
                let elem_hi = byte_idx * 2 + 1;
                let hi_lo = (qh >> elem_lo) & 1;
                let hi_hi = (qh >> elem_hi) & 1;
                let q_lo = (lo4 | (hi_lo << 4)) as f32;
                let q_hi = (hi4 | (hi_hi << 4)) as f32;
                out.push(q_lo * delta + min);
                out.push(q_hi * delta + min);
            }
        }
        Ok(out)
    }

    /// Q8_0 block: 2 bytes delta (f16) + 32 bytes i8 values → 32 f32
    ///
    /// Each i8 value `q` is scaled: `q * delta`.
    pub(super) fn dequant_q8_0(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
        const BLOCK_ELEMS: usize = 32;
        const BLOCK_BYTES: usize = 34;
        if !n.is_multiple_of(BLOCK_ELEMS) {
            return Err(ModelError::simple_load_error(format!(
                "Q8_0: n_elements {} not divisible by {}",
                n, BLOCK_ELEMS
            )));
        }
        let n_blocks = n / BLOCK_ELEMS;
        if data.len() < n_blocks * BLOCK_BYTES {
            return Err(ModelError::simple_load_error("Q8_0 data buffer too small"));
        }
        let mut out = Vec::with_capacity(n);
        for b in 0..n_blocks {
            let base = b * BLOCK_BYTES;
            let delta_bits = u16::from_le_bytes([data[base], data[base + 1]]);
            let delta = half::f16::from_bits(delta_bits).to_f32();
            for i in 0..BLOCK_ELEMS {
                let q = data[base + 2 + i] as i8;
                out.push(q as f32 * delta);
            }
        }
        Ok(out)
    }

    /// Q6_K block: 210 bytes → 256 f32 elements.
    ///
    /// Block layout:
    /// - 128 bytes: low 4 bits of each 6-bit value, packed as nibbles (ql)
    /// - 64 bytes:  high 2 bits for groups of 4, packed 4-per-byte (qh)
    /// - 16 bytes:  sub-block scales (i8, one per 16 elements)
    /// - 2 bytes:   block scale delta (f16)
    pub(super) fn dequant_q6_k(data: &[u8], n: usize) -> ModelResult<Vec<f32>> {
        const BLOCK_ELEMS: usize = 256;
        const BLOCK_BYTES: usize = 210;
        if !n.is_multiple_of(BLOCK_ELEMS) {
            return Err(ModelError::simple_load_error(format!(
                "Q6K: n_elements {} not divisible by {}",
                n, BLOCK_ELEMS
            )));
        }
        let n_blocks = n / BLOCK_ELEMS;
        if data.len() < n_blocks * BLOCK_BYTES {
            return Err(ModelError::simple_load_error("Q6K data buffer too small"));
        }
        let mut out = Vec::with_capacity(n);
        for b in 0..n_blocks {
            let base = b * BLOCK_BYTES;
            // ql: 128 bytes (low 4 bits, packed nibbles)
            let ql = &data[base..base + 128];
            // qh: 64 bytes (high 2 bits, 4 per byte)
            let qh = &data[base + 128..base + 192];
            // scales: 16 i8 values
            let scales_raw = &data[base + 192..base + 208];
            // delta: f16
            let delta_bits = u16::from_le_bytes([data[base + 208], data[base + 209]]);
            let delta = half::f16::from_bits(delta_bits).to_f32();

            // Reconstruct 256 6-bit values
            // ql[i] holds nibbles for element i and i+128 (lower 4 bits, upper 4 bits)
            // qh[i] holds high bits for 4 consecutive pairs
            for i in 0..128usize {
                // high bits byte index and bit positions
                let qh_byte = qh[i / 2];
                let shift_lo = (i % 2) * 4; // bits [shift_lo+1 : shift_lo] for even element
                let shift_hi = (i % 2) * 4 + 2; // bits [shift_hi+1 : shift_hi] for odd element (128+i)

                let q_lo_low4 = ql[i] & 0x0F;
                let q_hi_low4 = (ql[i] >> 4) & 0x0F;

                let q_lo_high2 = (qh_byte >> shift_lo) & 0x03;
                let q_hi_high2 = (qh_byte >> shift_hi) & 0x03;

                let q_lo = ((q_lo_high2 << 4) | q_lo_low4) as i32 - 32;
                let q_hi = ((q_hi_high2 << 4) | q_hi_low4) as i32 - 32;

                // Scale: one i8 per 16 elements → 16 sub-blocks of 16 elements each
                let scale_idx_lo = (i * 2) / 16; // element i*2 / 16
                let scale_idx_hi = (i * 2 + 1) / 16;

                if scale_idx_lo >= 16 || scale_idx_hi >= 16 {
                    return Err(ModelError::simple_load_error(
                        "Q6K scale index out of range",
                    ));
                }
                let scale_lo = scales_raw[scale_idx_lo] as i8 as f32;
                let scale_hi = scales_raw[scale_idx_hi] as i8 as f32;

                out.push(delta * scale_lo * q_lo as f32);
                out.push(delta * scale_hi * q_hi as f32);
            }
        }
        Ok(out)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: write a minimal GGUF header to `buf` ──────────────────────────

    /// Write GGUF magic, version, tensor_count, and kv_count to `buf`.
    fn write_gguf_header(buf: &mut Vec<u8>, version: u32, tensor_count: u64, kv_count: u64) {
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&version.to_le_bytes());
        if version >= 2 {
            buf.extend_from_slice(&tensor_count.to_le_bytes());
            buf.extend_from_slice(&kv_count.to_le_bytes());
        } else {
            buf.extend_from_slice(&(tensor_count as u32).to_le_bytes());
            buf.extend_from_slice(&(kv_count as u32).to_le_bytes());
        }
    }

    /// Write a v2+ GGUF string (u64 length prefix).
    fn write_str_v2(buf: &mut Vec<u8>, s: &str) {
        let bytes = s.as_bytes();
        buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(bytes);
    }

    /// Write a v1 GGUF string (u32 length prefix).
    #[allow(dead_code)]
    fn write_str_v1(buf: &mut Vec<u8>, s: &str) {
        let bytes = s.as_bytes();
        buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(bytes);
    }

    /// Pad `buf` to 32-byte alignment.
    fn pad_to_32(buf: &mut Vec<u8>) {
        let rem = buf.len() % 32;
        if rem != 0 {
            let pad = 32 - rem;
            buf.extend(std::iter::repeat_n(0u8, pad));
        }
    }

    // ── Test 1 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_magic_validation() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_bad_magic.bin");
        let mut buf = Vec::new();
        buf.extend_from_slice(b"XXXX");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, &buf).unwrap();
        let result = GgufFile::open(&path);
        assert!(result.is_err(), "Expected error for bad magic");
        let _ = std::fs::remove_file(&path);
    }

    // ── Test 2 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_version_1_parse() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_v1_empty.bin");
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 1, 0, 0);
        pad_to_32(&mut buf);
        std::fs::write(&path, &buf).unwrap();
        let file = GgufFile::open(&path).expect("v1 parse failed");
        assert_eq!(file.version, 1);
        assert!(file.tensors.is_empty());
        assert!(file.metadata.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    // ── Test 3 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_version_2_parse() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_v2_empty.bin");
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 2, 0, 0);
        pad_to_32(&mut buf);
        std::fs::write(&path, &buf).unwrap();
        let file = GgufFile::open(&path).expect("v2 parse failed");
        assert_eq!(file.version, 2);
        assert!(file.tensors.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    // ── Test 4 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_version_3_parse() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_v3_empty.bin");
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 3, 0, 0);
        pad_to_32(&mut buf);
        std::fs::write(&path, &buf).unwrap();
        let file = GgufFile::open(&path).expect("v3 parse failed");
        assert_eq!(file.version, 3);
        assert!(file.tensors.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    // ── Test 5 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_metadata_uint32() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_meta_uint32.bin");
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 2, 0, 1);
        // key
        write_str_v2(&mut buf, "my.key");
        // value type = Uint32 = 4
        buf.extend_from_slice(&4u32.to_le_bytes());
        // value
        buf.extend_from_slice(&42u32.to_le_bytes());
        pad_to_32(&mut buf);
        std::fs::write(&path, &buf).unwrap();
        let file = GgufFile::open(&path).expect("parse failed");
        match file.metadata.get("my.key") {
            Some(GgufMetaValue::Uint32(v)) => assert_eq!(*v, 42u32),
            other => panic!("Expected Uint32(42), got {:?}", other),
        }
        let _ = std::fs::remove_file(&path);
    }

    // ── Test 6 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_metadata_string() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_meta_string.bin");
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 2, 0, 1);
        write_str_v2(&mut buf, "general.name");
        buf.extend_from_slice(&8u32.to_le_bytes()); // String type = 8
        write_str_v2(&mut buf, "test-model");
        pad_to_32(&mut buf);
        std::fs::write(&path, &buf).unwrap();
        let file = GgufFile::open(&path).expect("parse failed");
        match file.metadata.get("general.name") {
            Some(GgufMetaValue::String(s)) => assert_eq!(s, "test-model"),
            other => panic!("Expected String, got {:?}", other),
        }
        let _ = std::fs::remove_file(&path);
    }

    // ── Test 7 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_metadata_float32() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_meta_float32.bin");
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 2, 0, 1);
        write_str_v2(&mut buf, "param.scale");
        buf.extend_from_slice(&6u32.to_le_bytes()); // Float32 = 6
        buf.extend_from_slice(&std::f32::consts::PI.to_le_bytes());
        pad_to_32(&mut buf);
        std::fs::write(&path, &buf).unwrap();
        let file = GgufFile::open(&path).expect("parse failed");
        match file.metadata.get("param.scale") {
            Some(GgufMetaValue::Float32(v)) => assert!((v - std::f32::consts::PI).abs() < 1e-5),
            other => panic!("Expected Float32, got {:?}", other),
        }
        let _ = std::fs::remove_file(&path);
    }

    // ── Test 8 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_metadata_array_uint32() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_meta_array_uint32.bin");
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 2, 0, 1);
        write_str_v2(&mut buf, "layer.sizes");
        buf.extend_from_slice(&9u32.to_le_bytes()); // Array = 9
        buf.extend_from_slice(&4u32.to_le_bytes()); // elem type = Uint32
        buf.extend_from_slice(&3u64.to_le_bytes()); // count = 3
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&20u32.to_le_bytes());
        buf.extend_from_slice(&30u32.to_le_bytes());
        pad_to_32(&mut buf);
        std::fs::write(&path, &buf).unwrap();
        let file = GgufFile::open(&path).expect("parse failed");
        match file.metadata.get("layer.sizes") {
            Some(GgufMetaValue::Array(arr)) => {
                assert_eq!(arr.len(), 3);
                match (&arr[0], &arr[1], &arr[2]) {
                    (
                        GgufMetaValue::Uint32(a),
                        GgufMetaValue::Uint32(b),
                        GgufMetaValue::Uint32(c),
                    ) => {
                        assert_eq!(*a, 10);
                        assert_eq!(*b, 20);
                        assert_eq!(*c, 30);
                    }
                    _ => panic!("Unexpected array element types"),
                }
            }
            other => panic!("Expected Array, got {:?}", other),
        }
        let _ = std::fs::remove_file(&path);
    }

    // ── Test 9 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_f32_tensor_load() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_f32_tensor.bin");
        // 2x3 f32 tensor
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 2, 1, 0);
        // Tensor info
        write_str_v2(&mut buf, "my_tensor");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
        buf.extend_from_slice(&3u64.to_le_bytes()); // dim0 = 3 (inner)
        buf.extend_from_slice(&2u64.to_le_bytes()); // dim1 = 2 (outer)
        buf.extend_from_slice(&0u32.to_le_bytes()); // quant type F32 = 0
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset = 0
        pad_to_32(&mut buf);
        // Data section
        for v in &values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &buf).unwrap();
        let file = GgufFile::open(&path).expect("parse failed");
        let arr = file.load_tensor_f32("my_tensor").expect("load failed");
        // GGUF dims are inner-first, so dim0=3, dim1=2 → shape_to_2d gives (dim0=3 inner, dim1=2 outer) → (2, 3) after shape_to_2d
        // shape = [3, 2] → shape_to_2d returns (3, 2) since shape[0]=3, shape[1]=2
        assert_eq!(arr.nrows() * arr.ncols(), 6);
        // Verify all values are present
        let flat: Vec<f32> = arr.iter().cloned().collect();
        for v in &values {
            assert!(flat.contains(v), "Value {} not found", v);
        }
        let _ = std::fs::remove_file(&path);
    }

    // ── Test 10 ───────────────────────────────────────────────────────────────

    #[test]
    fn test_f16_tensor_load() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_f16_tensor.bin");
        // 4x4 f16 tensor = 16 elements
        let n: usize = 16;
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 2, 1, 0);
        write_str_v2(&mut buf, "f16_tensor");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim0 = 4
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim1 = 4
        buf.extend_from_slice(&1u32.to_le_bytes()); // quant F16 = 1
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset = 0
        pad_to_32(&mut buf);
        // Data: 16 f16 values
        for i in 0..n {
            let val = half::f16::from_f32(i as f32 * 0.5);
            buf.extend_from_slice(&val.to_bits().to_le_bytes());
        }
        std::fs::write(&path, &buf).unwrap();
        let file = GgufFile::open(&path).expect("parse failed");
        let arr = file.load_tensor_f32("f16_tensor").expect("load failed");
        assert_eq!(arr.nrows(), 4);
        assert_eq!(arr.ncols(), 4);
        let _ = std::fs::remove_file(&path);
    }

    // ── Test 11 ───────────────────────────────────────────────────────────────

    #[test]
    fn test_q4_0_tensor_load() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_q4_0_tensor.bin");
        // 32 elements = 1 Q4_0 block (18 bytes)
        // delta = 1.0 (as f16), all nibbles = 8 (maps to 0 after -8)
        let n: usize = 32;
        let delta_f16 = half::f16::from_f32(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&delta_f16.to_bits().to_le_bytes()); // 2 bytes delta
        block.extend(std::iter::repeat_n(0x88u8, 16)); // nibbles = 8 (lo=8,hi=8)

        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 2, 1, 0);
        write_str_v2(&mut buf, "q4_tensor");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        buf.extend_from_slice(&(n as u64).to_le_bytes()); // dim0 = 32
        buf.extend_from_slice(&2u32.to_le_bytes()); // Q4_0 = 2
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset = 0
        pad_to_32(&mut buf);
        buf.extend_from_slice(&block);
        std::fs::write(&path, &buf).unwrap();
        let file = GgufFile::open(&path).expect("parse failed");
        let arr = file.load_tensor_f32("q4_tensor").expect("load failed");
        assert_eq!(arr.nrows() * arr.ncols(), n);
        // nibble 8 - 8 = 0, so all output values should be 0.0
        for v in arr.iter() {
            assert_eq!(*v, 0.0f32, "Expected 0.0, got {}", v);
        }
        let _ = std::fs::remove_file(&path);
    }

    // ── Test 12 ───────────────────────────────────────────────────────────────

    #[test]
    fn test_architecture_extraction() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_arch.bin");
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 2, 0, 1);
        write_str_v2(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // String
        write_str_v2(&mut buf, "llama");
        pad_to_32(&mut buf);
        std::fs::write(&path, &buf).unwrap();
        let file = GgufFile::open(&path).expect("parse failed");
        assert_eq!(file.architecture(), Some("llama"));
        let _ = std::fs::remove_file(&path);
    }

    // ── Test 13 ───────────────────────────────────────────────────────────────

    #[test]
    fn test_multi_tensor_loading() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_multi_tensor.bin");
        // Two 1D tensors of 4 f32 elements each
        let vals_a: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let vals_b: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let bytes_per_tensor: u64 = 4 * 4; // 4 elements × 4 bytes

        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 2, 2, 0);

        // Tensor A info
        write_str_v2(&mut buf, "tensor_a");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim0 = 4
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32 = 0
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

        // Tensor B info
        write_str_v2(&mut buf, "tensor_b");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim0 = 4
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32 = 0
        buf.extend_from_slice(&bytes_per_tensor.to_le_bytes()); // offset after first tensor

        pad_to_32(&mut buf);
        // Data: tensor A then tensor B
        for v in &vals_a {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in &vals_b {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &buf).unwrap();

        let file = GgufFile::open(&path).expect("parse failed");
        let all = file.load_all_tensors_f32().expect("load all failed");
        assert!(all.contains_key("tensor_a"), "tensor_a missing");
        assert!(all.contains_key("tensor_b"), "tensor_b missing");

        let a = &all["tensor_a"];
        let b = &all["tensor_b"];
        assert_eq!(a.nrows() * a.ncols(), 4);
        assert_eq!(b.nrows() * b.ncols(), 4);

        let flat_a: Vec<f32> = a.iter().cloned().collect();
        let flat_b: Vec<f32> = b.iter().cloned().collect();
        assert_eq!(flat_a, vals_a);
        assert_eq!(flat_b, vals_b);

        let _ = std::fs::remove_file(&path);
    }
}
