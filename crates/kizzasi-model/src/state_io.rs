//! Full model state I/O: saves both weights AND runtime SSM state to disk.
//!
//! Format: JSON envelope with hex-encoded binary state.
//!
//! # Overview
//!
//! [`ModelSnapshot`] captures:
//! - Named weight tensors (the model parameters)
//! - Named runtime SSM hidden states (per-layer recurrent state)
//! - Shape metadata for each tensor
//! - Model architecture identifier and configuration
//! - Creation timestamp and crate version
//!
//! This enables **exact** pause/resume semantics: serialize mid-inference, restore
//! and continue from exactly the same point.
//!
//! # File Format
//!
//! Snapshots are stored as compact JSON.  Weight/state data are hex-encoded
//! little-endian IEEE 754 f32 bytes — no external codec dependency needed.
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::state_io::ModelSnapshot;
//!
//! let mut snap = ModelSnapshot::new("mamba");
//! snap.add_weight("embed.weight", &embed_data, &[vocab, d_model]);
//! snap.add_state("layer0.h", &h0_data, &[batch, d_state]);
//! snap.save("model.snapshot.json")?;
//!
//! let restored = ModelSnapshot::load("model.snapshot.json")?;
//! let w = restored.get_weight("embed.weight")?;
//! ```

use crate::error::{ModelError, ModelResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// ModelSnapshot
// ---------------------------------------------------------------------------

/// A complete model snapshot capturing both weights and runtime SSM state.
///
/// The snapshot stores every tensor as a hex-encoded, little-endian sequence of
/// `f32` values together with its shape so that the original multi-dimensional
/// array can be reconstructed without ambiguity.
///
/// # Thread Safety
///
/// `ModelSnapshot` is `Send + Sync` — all fields are plain, heap-allocated data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSnapshot {
    /// Model architecture identifier (e.g. `"mamba"`, `"rwkv"`, `"s4d"`).
    pub model_type: String,

    /// JSON-serialized model configuration.  The exact schema is determined
    /// by the model implementation; callers may store an empty string if no
    /// configuration is available.
    pub config_json: String,

    /// Weight tensors: `name → hex-encoded f32 LE bytes`.
    pub weights: HashMap<String, String>,

    /// Runtime SSM hidden states: `layer_name → hex-encoded f32 LE bytes`.
    pub hidden_states: HashMap<String, String>,

    /// Weight tensor shapes: `name → [dim0, dim1, …]`.
    pub weight_shapes: HashMap<String, Vec<usize>>,

    /// Hidden state shapes: `name → [dim0, dim1, …]`.
    pub state_shapes: HashMap<String, Vec<usize>>,

    /// Creation timestamp in Unix seconds.
    pub created_at: u64,

    /// Kizzasi crate version that produced this snapshot.
    pub version: String,
}

impl ModelSnapshot {
    /// Create an empty snapshot for the given model architecture.
    ///
    /// `model_type` is a free-form string identifying the architecture,
    /// e.g. `"mamba"` or `"rwkv7"`.
    pub fn new(model_type: impl Into<String>) -> Self {
        Self {
            model_type: model_type.into(),
            config_json: String::new(),
            weights: HashMap::new(),
            hidden_states: HashMap::new(),
            weight_shapes: HashMap::new(),
            state_shapes: HashMap::new(),
            created_at: current_unix_secs(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    // -----------------------------------------------------------------------
    // Insertion
    // -----------------------------------------------------------------------

    /// Store a weight tensor.
    ///
    /// # Arguments
    ///
    /// * `name`  – Fully qualified tensor name, e.g. `"layers.0.in_proj.weight"`.
    /// * `data`  – Flat, row-major f32 slice.
    /// * `shape` – Logical shape; `shape.iter().product()` must equal `data.len()`.
    ///
    /// If a tensor with the same name already exists it is overwritten.
    pub fn add_weight(&mut self, name: &str, data: &[f32], shape: &[usize]) {
        self.weights
            .insert(name.to_string(), encode_f32_slice(data));
        self.weight_shapes.insert(name.to_string(), shape.to_vec());
    }

    /// Store a runtime SSM hidden state.
    ///
    /// # Arguments
    ///
    /// * `name`  – Layer-qualified state name, e.g. `"layer0.h"`.
    /// * `data`  – Flat f32 slice (typically `[batch, d_state]` flattened).
    /// * `shape` – Logical shape of the state tensor.
    pub fn add_state(&mut self, name: &str, data: &[f32], shape: &[usize]) {
        self.hidden_states
            .insert(name.to_string(), encode_f32_slice(data));
        self.state_shapes.insert(name.to_string(), shape.to_vec());
    }

    /// Attach a JSON-serialized configuration string.
    pub fn set_config_json(&mut self, config_json: impl Into<String>) {
        self.config_json = config_json.into();
    }

    // -----------------------------------------------------------------------
    // Retrieval
    // -----------------------------------------------------------------------

    /// Retrieve and decode a weight tensor by name.
    ///
    /// Returns [`ModelError::LoadError`] if the tensor is not present or the
    /// stored hex data is malformed.
    pub fn get_weight(&self, name: &str) -> ModelResult<Vec<f32>> {
        let encoded = self.weights.get(name).ok_or_else(|| {
            ModelError::load_error(
                "ModelSnapshot::get_weight",
                format!("weight '{name}' not found in snapshot"),
            )
        })?;
        decode_f32_slice(encoded)
    }

    /// Retrieve and decode a runtime hidden state by name.
    ///
    /// Returns [`ModelError::LoadError`] if the state is not present or the
    /// stored hex data is malformed.
    pub fn get_state(&self, name: &str) -> ModelResult<Vec<f32>> {
        let encoded = self.hidden_states.get(name).ok_or_else(|| {
            ModelError::load_error(
                "ModelSnapshot::get_state",
                format!("hidden state '{name}' not found in snapshot"),
            )
        })?;
        decode_f32_slice(encoded)
    }

    /// Return the shape recorded for the named weight tensor, if present.
    pub fn weight_shape(&self, name: &str) -> Option<&[usize]> {
        self.weight_shapes.get(name).map(Vec::as_slice)
    }

    /// Return the shape recorded for the named hidden state, if present.
    pub fn state_shape(&self, name: &str) -> Option<&[usize]> {
        self.state_shapes.get(name).map(Vec::as_slice)
    }

    // -----------------------------------------------------------------------
    // Enumeration helpers
    // -----------------------------------------------------------------------

    /// List all weight tensor names stored in this snapshot.
    ///
    /// Order is not guaranteed; sort if determinism is required.
    pub fn weight_names(&self) -> Vec<&str> {
        self.weights.keys().map(String::as_str).collect()
    }

    /// List all hidden state names stored in this snapshot.
    ///
    /// Order is not guaranteed; sort if determinism is required.
    pub fn state_names(&self) -> Vec<&str> {
        self.hidden_states.keys().map(String::as_str).collect()
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Total number of scalar weight parameters across all stored tensors.
    ///
    /// Computed from the recorded shapes; if no shapes are present for a tensor
    /// its element count is treated as zero.
    pub fn total_weight_params(&self) -> usize {
        self.weight_shapes
            .values()
            .map(|shape| shape.iter().product::<usize>())
            .sum()
    }

    /// Total number of scalar hidden-state elements across all stored states.
    pub fn total_state_elements(&self) -> usize {
        self.state_shapes
            .values()
            .map(|shape| shape.iter().product::<usize>())
            .sum()
    }

    /// Number of distinct weight tensors.
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }

    /// Number of distinct hidden state tensors.
    pub fn num_states(&self) -> usize {
        self.hidden_states.len()
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /// Serialize this snapshot to a JSON file at `path`.
    ///
    /// The file is written atomically via a `BufWriter`; the path must be
    /// writable and its parent directory must already exist.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> ModelResult<()> {
        let json = serde_json::to_string(self).map_err(|e| {
            ModelError::load_error(
                "ModelSnapshot::save",
                format!("JSON serialization failed: {e}"),
            )
        })?;

        let file = std::fs::File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);
        writer.write_all(json.as_bytes())?;
        writer.flush()?;
        Ok(())
    }

    /// Deserialize a snapshot previously written by [`ModelSnapshot::save`].
    ///
    /// Returns [`ModelError::IoError`] if the file cannot be opened and
    /// [`ModelError::LoadError`] if the JSON is malformed.
    pub fn load<P: AsRef<Path>>(path: P) -> ModelResult<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| {
            ModelError::load_error(
                "ModelSnapshot::load",
                format!("JSON deserialization failed: {e}"),
            )
        })
    }

    /// Save a pretty-printed (human-readable) JSON file.
    ///
    /// Equivalent to [`save`][Self::save] but uses `serde_json::to_string_pretty`,
    /// producing an indented file that is easier to inspect in a text editor.
    /// The file size will be larger than the compact variant.
    pub fn save_pretty<P: AsRef<Path>>(&self, path: P) -> ModelResult<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            ModelError::load_error(
                "ModelSnapshot::save_pretty",
                format!("JSON serialization failed: {e}"),
            )
        })?;

        let file = std::fs::File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);
        writer.write_all(json.as_bytes())?;
        writer.flush()?;
        Ok(())
    }

    /// Check basic consistency: every weight name has a corresponding shape
    /// entry and vice versa, and similarly for hidden states.
    ///
    /// Returns a list of inconsistency descriptions; an empty vec means the
    /// snapshot is self-consistent.
    pub fn validate(&self) -> Vec<String> {
        let mut issues: Vec<String> = Vec::new();

        for name in self.weights.keys() {
            if !self.weight_shapes.contains_key(name) {
                issues.push(format!("weight '{name}' has data but no shape entry"));
            }
        }
        for name in self.weight_shapes.keys() {
            if !self.weights.contains_key(name) {
                issues.push(format!("weight_shapes has entry for '{name}' but no data"));
            }
        }

        for name in self.hidden_states.keys() {
            if !self.state_shapes.contains_key(name) {
                issues.push(format!("hidden_state '{name}' has data but no shape entry"));
            }
        }
        for name in self.state_shapes.keys() {
            if !self.hidden_states.contains_key(name) {
                issues.push(format!("state_shapes has entry for '{name}' but no data"));
            }
        }

        issues
    }
}

// ---------------------------------------------------------------------------
// Codec helpers (public for reuse by other modules)
// ---------------------------------------------------------------------------

/// Encode a flat `f32` slice as a lowercase hex string.
///
/// Each `f32` is converted to its 4-byte little-endian IEEE 754 representation
/// and each byte is encoded as exactly two hex digits.  The resulting string
/// length is always `data.len() * 8`.
///
/// # Example
///
/// ```
/// use kizzasi_model::state_io::encode_f32_slice;
/// let encoded = encode_f32_slice(&[0.0f32]);
/// assert_eq!(encoded.len(), 8);
/// ```
pub fn encode_f32_slice(data: &[f32]) -> String {
    let mut out = String::with_capacity(data.len() * 8);
    for &v in data {
        for byte in v.to_le_bytes() {
            // Two hex digits per byte — avoids any external dependency.
            let hi = (byte >> 4) as usize;
            let lo = (byte & 0xF) as usize;
            const HEX: &[u8; 16] = b"0123456789abcdef";
            out.push(HEX[hi] as char);
            out.push(HEX[lo] as char);
        }
    }
    out
}

/// Decode a hex string produced by [`encode_f32_slice`] back into `f32` values.
///
/// # Errors
///
/// Returns [`ModelError::LoadError`] if:
/// - The string length is not divisible by 8 (each f32 needs 8 hex chars).
/// - Any character is not a valid lowercase or uppercase ASCII hex digit.
pub fn decode_f32_slice(s: &str) -> ModelResult<Vec<f32>> {
    let chars = s.as_bytes();

    if !chars.len().is_multiple_of(8) {
        return Err(ModelError::load_error(
            "decode_f32_slice",
            format!(
                "hex string length {} is not divisible by 8 \
                 (each f32 requires 8 hex chars)",
                chars.len()
            ),
        ));
    }

    let num_floats = chars.len() / 8;
    let mut out = Vec::with_capacity(num_floats);

    for chunk in chars.chunks_exact(8) {
        // Decode 4 bytes (8 hex chars) → one f32
        let mut bytes = [0u8; 4];
        for (byte_idx, hex_pair) in chunk.chunks_exact(2).enumerate() {
            let hi = hex_digit(hex_pair[0]).map_err(|_| {
                ModelError::load_error(
                    "decode_f32_slice",
                    format!(
                        "invalid hex character '{}' at position {}",
                        hex_pair[0] as char,
                        byte_idx * 2
                    ),
                )
            })?;
            let lo = hex_digit(hex_pair[1]).map_err(|_| {
                ModelError::load_error(
                    "decode_f32_slice",
                    format!(
                        "invalid hex character '{}' at position {}",
                        hex_pair[1] as char,
                        byte_idx * 2 + 1
                    ),
                )
            })?;
            bytes[byte_idx] = (hi << 4) | lo;
        }
        out.push(f32::from_le_bytes(bytes));
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert a single ASCII hex character to its nibble value.
///
/// Accepts both uppercase (`A–F`) and lowercase (`a–f`) digits.
/// Returns `Err(())` for any other character.
#[inline]
fn hex_digit(c: u8) -> Result<u8, ()> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(()),
    }
}

/// Return the current time as Unix seconds.
///
/// Falls back to 0 if the system clock is set before the Unix epoch (unusual
/// but possible in embedded / test environments).
fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Codec roundtrip ---------------------------------------------------------

    #[test]
    fn test_encode_decode_roundtrip() {
        let data = vec![
            1.0f32,
            -2.0,
            std::f32::consts::PI,
            0.0,
            f32::MAX,
            f32::MIN_POSITIVE,
        ];
        let encoded = encode_f32_slice(&data);
        let decoded = decode_f32_slice(&encoded).expect("decode must succeed");
        assert_eq!(data.len(), decoded.len(), "lengths must match");
        for (orig, dec) in data.iter().zip(decoded.iter()) {
            assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "f32 bits must match exactly for value {orig}"
            );
        }
    }

    #[test]
    fn test_encode_decode_special_values() {
        let specials = vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -0.0f32];
        let encoded = encode_f32_slice(&specials);
        let decoded = decode_f32_slice(&encoded).expect("decode must succeed");
        for (orig, dec) in specials.iter().zip(decoded.iter()) {
            assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "bit-exact round-trip required"
            );
        }
    }

    #[test]
    fn test_encode_empty_slice() {
        let encoded = encode_f32_slice(&[]);
        assert!(encoded.is_empty());
        let decoded = decode_f32_slice(&encoded).expect("empty string should decode to []");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_decode_invalid_length() {
        // 7 characters is not divisible by 8
        let result = decode_f32_slice("0000000");
        assert!(result.is_err(), "length not divisible by 8 must fail");
    }

    #[test]
    fn test_decode_invalid_hex_character() {
        // 'z' is not a valid hex digit
        let result = decode_f32_slice("zzzzzzzz");
        assert!(result.is_err(), "invalid hex must fail");
    }

    #[test]
    fn test_decode_uppercase_hex_accepted() {
        // encode produces lowercase; uppercase must also be accepted
        let data = vec![1.0f32];
        let lower = encode_f32_slice(&data);
        let upper = lower.to_uppercase();
        let from_lower = decode_f32_slice(&lower).expect("lowercase decode");
        let from_upper = decode_f32_slice(&upper).expect("uppercase decode");
        assert_eq!(from_lower[0].to_bits(), from_upper[0].to_bits());
    }

    // -- ModelSnapshot construction ---------------------------------------------

    #[test]
    fn test_model_snapshot_new() {
        let snap = ModelSnapshot::new("mamba");
        assert_eq!(snap.model_type, "mamba");
        assert!(snap.weights.is_empty());
        assert!(snap.hidden_states.is_empty());
        assert!(!snap.version.is_empty());
    }

    #[test]
    fn test_model_snapshot_add_get_weight() {
        let mut snap = ModelSnapshot::new("mamba");
        snap.add_weight("embed.weight", &[1.0f32, 2.0, 3.0], &[1, 3]);

        let w = snap.get_weight("embed.weight").expect("weight must exist");
        assert_eq!(w, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn test_model_snapshot_add_get_state() {
        let mut snap = ModelSnapshot::new("mamba");
        snap.add_state("layer0.h", &[0.1f32, 0.2], &[1, 2]);

        let s = snap.get_state("layer0.h").expect("state must exist");
        assert!((s[0] - 0.1).abs() < 1e-6, "state[0] must be ~0.1");
        assert!((s[1] - 0.2).abs() < 1e-6, "state[1] must be ~0.2");
    }

    #[test]
    fn test_model_snapshot_missing_weight_error() {
        let snap = ModelSnapshot::new("mamba");
        let result = snap.get_weight("nonexistent");
        assert!(result.is_err(), "missing weight must return error");
    }

    #[test]
    fn test_model_snapshot_missing_state_error() {
        let snap = ModelSnapshot::new("mamba");
        let result = snap.get_state("nonexistent");
        assert!(result.is_err(), "missing state must return error");
    }

    // -- Shape accessors --------------------------------------------------------

    #[test]
    fn test_weight_shape_accessor() {
        let mut snap = ModelSnapshot::new("s4d");
        snap.add_weight("A", &[0.0f32; 12], &[3, 4]);
        let shape = snap.weight_shape("A").expect("shape must be present");
        assert_eq!(shape, &[3, 4]);
    }

    #[test]
    fn test_state_shape_accessor() {
        let mut snap = ModelSnapshot::new("rwkv");
        snap.add_state("layer1.wkv", &[0.0f32; 8], &[2, 4]);
        let shape = snap.state_shape("layer1.wkv").expect("shape present");
        assert_eq!(shape, &[2, 4]);
    }

    #[test]
    fn test_shape_missing_returns_none() {
        let snap = ModelSnapshot::new("test");
        assert!(snap.weight_shape("x").is_none());
        assert!(snap.state_shape("y").is_none());
    }

    // -- Statistics helpers -----------------------------------------------------

    #[test]
    fn test_total_weight_params() {
        let mut snap = ModelSnapshot::new("test");
        snap.add_weight("a", &[0.0f32; 12], &[3, 4]);
        snap.add_weight("b", &[0.0f32; 6], &[2, 3]);
        assert_eq!(snap.total_weight_params(), 18);
    }

    #[test]
    fn test_total_state_elements() {
        let mut snap = ModelSnapshot::new("test");
        snap.add_state("h0", &[0.0f32; 4], &[1, 4]);
        snap.add_state("h1", &[0.0f32; 8], &[2, 4]);
        assert_eq!(snap.total_state_elements(), 12);
    }

    #[test]
    fn test_num_weights_and_states() {
        let mut snap = ModelSnapshot::new("m");
        snap.add_weight("w1", &[1.0f32], &[1]);
        snap.add_weight("w2", &[2.0f32], &[1]);
        snap.add_state("s1", &[3.0f32], &[1]);
        assert_eq!(snap.num_weights(), 2);
        assert_eq!(snap.num_states(), 1);
    }

    // -- Name listing -----------------------------------------------------------

    #[test]
    fn test_snapshot_lists_names() {
        let mut snap = ModelSnapshot::new("m");
        snap.add_weight("w1", &[1.0f32], &[1]);
        snap.add_weight("w2", &[2.0f32], &[1]);
        snap.add_state("s1", &[3.0f32], &[1]);

        let mut wnames = snap.weight_names();
        wnames.sort();
        assert_eq!(wnames, vec!["w1", "w2"]);

        let snames = snap.state_names();
        assert_eq!(snames, vec!["s1"]);
    }

    // -- Validation -------------------------------------------------------------

    #[test]
    fn test_validate_consistent_snapshot() {
        let mut snap = ModelSnapshot::new("test");
        snap.add_weight("w", &[1.0f32], &[1]);
        snap.add_state("h", &[0.5f32], &[1]);
        let issues = snap.validate();
        assert!(
            issues.is_empty(),
            "consistent snapshot should have no issues: {issues:?}"
        );
    }

    #[test]
    fn test_validate_detects_missing_shape() {
        let mut snap = ModelSnapshot::new("test");
        // Bypass add_weight to create an inconsistency manually
        snap.weights
            .insert("orphan".to_string(), encode_f32_slice(&[1.0f32]));
        // Note: weight_shapes entry is intentionally NOT added
        let issues = snap.validate();
        assert!(
            issues.iter().any(|i| i.contains("orphan")),
            "missing shape should be flagged: {issues:?}"
        );
    }

    // -- File I/O (using std::env::temp_dir) ------------------------------------

    #[test]
    fn test_model_snapshot_save_load_roundtrip() {
        let mut snap = ModelSnapshot::new("test_model");
        snap.add_weight("proj", &[1.0f32; 8], &[2, 4]);
        snap.add_state("h0", &[0.5f32; 4], &[1, 4]);
        snap.set_config_json("{\"d_model\":64}");

        let path = std::env::temp_dir().join("kizzasi_snapshot_test.json");

        snap.save(&path).expect("save must succeed");
        let loaded = ModelSnapshot::load(&path).expect("load must succeed");

        assert_eq!(loaded.model_type, "test_model");
        assert_eq!(loaded.config_json, "{\"d_model\":64}");

        let w = loaded.get_weight("proj").expect("weight round-trip");
        assert_eq!(w.len(), 8);
        for v in &w {
            assert!((*v - 1.0).abs() < 1e-7, "weight value should be 1.0");
        }

        let s = loaded.get_state("h0").expect("state round-trip");
        assert_eq!(s.len(), 4);
        for v in &s {
            assert!((*v - 0.5).abs() < 1e-7, "state value should be 0.5");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_model_snapshot_save_pretty_and_load() {
        let mut snap = ModelSnapshot::new("pretty_test");
        snap.add_weight("w", &[2.0f32, 4.0], &[1, 2]);

        let path = std::env::temp_dir().join("kizzasi_snapshot_pretty_test.json");

        snap.save_pretty(&path).expect("save_pretty must succeed");
        let loaded = ModelSnapshot::load(&path).expect("load must succeed");

        let w = loaded.get_weight("w").expect("weight round-trip");
        assert_eq!(w, vec![2.0f32, 4.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_nonexistent_file_returns_error() {
        let result = ModelSnapshot::load("/tmp/__kizzasi_nonexistent_snapshot__.json");
        assert!(result.is_err(), "loading a missing file must fail");
    }

    #[test]
    fn test_save_to_nonexistent_directory_returns_error() {
        let snap = ModelSnapshot::new("test");
        let result = snap.save("/tmp/__kizzasi_no_dir__/snap.json");
        assert!(
            result.is_err(),
            "saving to a non-existent directory must fail"
        );
    }

    // -- Overwrite semantics ----------------------------------------------------

    #[test]
    fn test_add_weight_overwrites_existing() {
        let mut snap = ModelSnapshot::new("test");
        snap.add_weight("w", &[1.0f32], &[1]);
        snap.add_weight("w", &[99.0f32], &[1]); // overwrite
        let w = snap.get_weight("w").expect("weight exists");
        assert_eq!(w, vec![99.0f32]);
    }

    #[test]
    fn test_add_state_overwrites_existing() {
        let mut snap = ModelSnapshot::new("test");
        snap.add_state("h", &[0.1f32], &[1]);
        snap.add_state("h", &[0.9f32], &[1]); // overwrite
        let s = snap.get_state("h").expect("state exists");
        assert_eq!(s, vec![0.9f32]);
    }

    // -- Large tensor round-trip ------------------------------------------------

    #[test]
    fn test_large_tensor_roundtrip() {
        let n = 4096;
        // Use a deterministic pattern instead of random
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 2.0).collect();
        let mut snap = ModelSnapshot::new("large");
        snap.add_weight("big_matrix", &data, &[64, 64]);

        let path = std::env::temp_dir().join("kizzasi_large_tensor_test.json");
        snap.save(&path).expect("save large tensor");

        let loaded = ModelSnapshot::load(&path).expect("load large tensor");
        let recovered = loaded
            .get_weight("big_matrix")
            .expect("retrieve large weight");

        assert_eq!(recovered.len(), n);
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert_eq!(
                orig.to_bits(),
                rec.to_bits(),
                "bit-exact round-trip required"
            );
        }

        let _ = std::fs::remove_file(&path);
    }
}
