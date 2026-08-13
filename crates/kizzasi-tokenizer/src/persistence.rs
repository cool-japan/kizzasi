//! Model persistence and checkpoint management
//!
//! This module provides functionality for saving and loading trained tokenizer
//! models, including weights, configurations, and training state.
//!
//! # Features
//!
//! - **Safetensors**: Efficient weight serialization using safetensors format
//! - **Checkpoints**: Save/load training checkpoints with metadata
//! - **Versioning**: Track model versions and compatibility
//! - **Configuration**: Export/import model configurations as JSON/TOML
//!
//! # Example
//!
//! ```ignore
//! use kizzasi_tokenizer::persistence::{ModelCheckpoint, save_checkpoint};
//! use kizzasi_tokenizer::TrainableContinuousTokenizer;
//!
//! let tokenizer = TrainableContinuousTokenizer::new(8, 16)?;
//! // ... train the model ...
//!
//! let checkpoint = ModelCheckpoint::from_trainable_tokenizer(&tokenizer, "v1.0")?;
//! save_checkpoint(&checkpoint, "model_checkpoint.safetensors")?;
//! ```

use crate::error::{TokenizerError, TokenizerResult};
use crate::{ReconstructionMetrics, TrainingConfig};
use chrono::{DateTime, Utc};
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Convert a `usize` length/dimension to `u32` for the on-disk binary
/// checkpoint format, failing loudly instead of silently wrapping.
///
/// The format's length fields are `u32`; casting with `as` instead of this
/// helper truncates any value `> u32::MAX` (reachable for real model
/// weights, e.g. tensors exceeding 4 GiB) to its low 32 bits and returns
/// `Ok(())` — the save "succeeds" but writes a file whose stored length no
/// longer matches the data that follows it, silently corrupting every
/// subsequent tensor a reader parses from that point on.
fn len_as_u32(len: usize, what: &str) -> TokenizerResult<u32> {
    u32::try_from(len).map_err(|_| {
        TokenizerError::InvalidConfig(format!(
            "{what} ({len}) exceeds u32::MAX ({}) and cannot be represented in this checkpoint \
             format; the save was rejected rather than silently truncated",
            u32::MAX
        ))
    })
}

/// Model version for compatibility tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelVersion {
    /// Major version (breaking changes)
    pub major: u32,
    /// Minor version (new features)
    pub minor: u32,
    /// Patch version (bug fixes)
    pub patch: u32,
}

impl ModelVersion {
    /// Create a new model version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse version from string (e.g., "1.2.3")
    pub fn parse(s: &str) -> TokenizerResult<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(TokenizerError::InvalidConfig(format!(
                "Invalid version string: {}",
                s
            )));
        }

        let major = parts[0]
            .parse()
            .map_err(|_| TokenizerError::InvalidConfig("Invalid major version".into()))?;
        let minor = parts[1]
            .parse()
            .map_err(|_| TokenizerError::InvalidConfig("Invalid minor version".into()))?;
        let patch = parts[2]
            .parse()
            .map_err(|_| TokenizerError::InvalidConfig("Invalid patch version".into()))?;

        Ok(Self::new(major, minor, patch))
    }

    /// Check if this version is compatible with another version
    pub fn is_compatible_with(&self, other: &ModelVersion) -> bool {
        // Compatible if major versions match
        self.major == other.major
    }
}

impl fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Model metadata for checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model version
    pub version: ModelVersion,
    /// Model type (e.g., "TrainableContinuousTokenizer")
    pub model_type: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
    /// Input dimension
    pub input_dim: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Training configuration (if available)
    pub training_config: Option<TrainingConfig>,
    /// Training metrics (if available)
    pub metrics: Option<ReconstructionMetrics>,
    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

impl ModelMetadata {
    /// Create new metadata
    pub fn new(
        version: ModelVersion,
        model_type: String,
        input_dim: usize,
        embed_dim: usize,
    ) -> Self {
        let now = Utc::now();
        Self {
            version,
            model_type,
            created_at: now,
            modified_at: now,
            input_dim,
            embed_dim,
            training_config: None,
            metrics: None,
            custom: HashMap::new(),
        }
    }

    /// Update the modified timestamp
    pub fn touch(&mut self) {
        self.modified_at = Utc::now();
    }

    /// Add custom metadata
    pub fn add_custom(&mut self, key: String, value: String) {
        self.custom.insert(key, value);
    }
}

/// Model checkpoint containing weights and metadata
#[derive(Debug)]
pub struct ModelCheckpoint {
    /// Checkpoint metadata
    pub metadata: ModelMetadata,
    /// Model weights as tensor data
    pub weights: HashMap<String, Vec<f32>>,
    /// Weight shapes for reconstruction
    pub shapes: HashMap<String, Vec<usize>>,
}

impl ModelCheckpoint {
    /// Create a new checkpoint
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            metadata,
            weights: HashMap::new(),
            shapes: HashMap::new(),
        }
    }

    /// Add a weight tensor to the checkpoint
    pub fn add_weight(&mut self, name: String, data: Vec<f32>, shape: Vec<usize>) {
        self.weights.insert(name.clone(), data);
        self.shapes.insert(name, shape);
    }

    /// Add a 2D array weight
    pub fn add_array2(&mut self, name: String, array: &Array2<f32>) {
        let shape = array.shape();
        let data: Vec<f32> = array.iter().copied().collect();
        self.add_weight(name, data, vec![shape[0], shape[1]]);
    }

    /// Get a weight tensor
    pub fn get_weight(&self, name: &str) -> Option<(&[f32], &[usize])> {
        self.weights
            .get(name)
            .and_then(|w| self.shapes.get(name).map(|s| (w.as_slice(), s.as_slice())))
    }

    /// Get a weight as Array2
    pub fn get_array2(&self, name: &str) -> TokenizerResult<Array2<f32>> {
        let (data, shape) = self
            .get_weight(name)
            .ok_or_else(|| TokenizerError::InvalidConfig(format!("Weight '{}' not found", name)))?;

        if shape.len() != 2 {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected 2D array for '{}', got {}D",
                name,
                shape.len()
            )));
        }

        let mut array = Array2::zeros((shape[0], shape[1]));
        for (i, &val) in data.iter().enumerate() {
            let row = i / shape[1];
            let col = i % shape[1];
            array[[row, col]] = val;
        }

        Ok(array)
    }

    /// Save checkpoint to safetensors format
    ///
    /// File-I/O failures propagate as [`TokenizerError::IoError`] via `?`
    /// and `std::io::Error`'s `#[from]` conversion (rather than a flattened
    /// `InternalError(String)`), so callers can inspect
    /// `std::io::Error::kind()` -- e.g. to distinguish a permissions failure
    /// (`ErrorKind::PermissionDenied`) from a missing parent directory
    /// (`ErrorKind::NotFound`) or a full disk (`ErrorKind::WriteZero`) --
    /// instead of matching on a formatted message string.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> TokenizerResult<()> {
        let path = path.as_ref();

        // Convert weights to safetensors format
        let mut tensors = Vec::new();
        for (name, data) in &self.weights {
            let shape = self.shapes.get(name).ok_or_else(|| {
                TokenizerError::InternalError(format!("Missing shape for weight '{}'", name))
            })?;

            // Convert f32 to bytes
            let data_bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

            tensors.push((name.clone(), shape.clone(), data_bytes));
        }

        // Serialize metadata to JSON
        let metadata_json = serde_json::to_string(&self.metadata)
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        // Create safetensors data
        let mut data_map: HashMap<String, (Vec<usize>, Vec<u8>)> = HashMap::new();
        for (name, shape, data) in tensors {
            data_map.insert(name, (shape, data));
        }

        // Write to file
        let mut file = File::create(path)?;

        // Write metadata length (u32) + metadata + tensors
        let metadata_bytes = metadata_json.as_bytes();
        let metadata_len = len_as_u32(metadata_bytes.len(), "metadata length")?;

        file.write_all(&metadata_len.to_le_bytes())?;
        file.write_all(metadata_bytes)?;

        // Write tensor data
        for (name, (shape, data)) in data_map {
            // Write: name_len (u32) + name + shape_len (u32) + shape + data_len (u32) + data
            let name_bytes = name.as_bytes();
            let name_len = len_as_u32(name_bytes.len(), &format!("tensor '{name}' name length"))?;
            file.write_all(&name_len.to_le_bytes())?;
            file.write_all(name_bytes)?;

            let shape_len = len_as_u32(shape.len(), &format!("tensor '{name}' shape length"))?;
            file.write_all(&shape_len.to_le_bytes())?;
            for &dim in &shape {
                let dim_u32 = len_as_u32(dim, &format!("tensor '{name}' dimension"))?;
                file.write_all(&dim_u32.to_le_bytes())?;
            }

            let data_len = len_as_u32(data.len(), &format!("tensor '{name}' data length"))?;
            file.write_all(&data_len.to_le_bytes())?;
            file.write_all(&data)?;
        }

        Ok(())
    }

    /// Load checkpoint from safetensors format
    ///
    /// File-I/O failures propagate as [`TokenizerError::IoError`] -- see
    /// [`Self::save`]'s doc comment.
    pub fn load<P: AsRef<Path>>(path: P) -> TokenizerResult<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;

        // Read metadata length
        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)?;
        let metadata_len = u32::from_le_bytes(len_buf) as usize;

        // Read metadata
        let mut metadata_buf = vec![0u8; metadata_len];
        file.read_exact(&mut metadata_buf)?;
        let metadata: ModelMetadata = serde_json::from_slice(&metadata_buf).map_err(|e| {
            TokenizerError::InternalError(format!("Failed to parse metadata: {}", e))
        })?;

        let mut checkpoint = ModelCheckpoint::new(metadata);

        // Read tensors
        loop {
            // Try to read name length
            let mut name_len_buf = [0u8; 4];
            match file.read_exact(&mut name_len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                // Preserve the real `std::io::Error` (and its `ErrorKind`)
                // via `#[from]` instead of flattening it into a message string.
                Err(e) => return Err(e.into()),
            }
            let name_len = u32::from_le_bytes(name_len_buf) as usize;

            // Read name
            let mut name_buf = vec![0u8; name_len];
            file.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)
                .map_err(|e| TokenizerError::InternalError(format!("Invalid UTF-8: {}", e)))?;

            // Read shape
            let mut shape_len_buf = [0u8; 4];
            file.read_exact(&mut shape_len_buf)?;
            let shape_len = u32::from_le_bytes(shape_len_buf) as usize;

            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                let mut dim_buf = [0u8; 4];
                file.read_exact(&mut dim_buf)?;
                shape.push(u32::from_le_bytes(dim_buf) as usize);
            }

            // Read data
            let mut data_len_buf = [0u8; 4];
            file.read_exact(&mut data_len_buf)?;
            let data_len = u32::from_le_bytes(data_len_buf) as usize;

            let mut data_bytes = vec![0u8; data_len];
            file.read_exact(&mut data_bytes)?;

            // Convert bytes to f32
            let data: Vec<f32> = data_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            checkpoint.add_weight(name, data, shape);
        }

        Ok(checkpoint)
    }
}

/// Save a training configuration to JSON
///
/// File-I/O failures propagate as [`TokenizerError::IoError`] -- see
/// [`ModelCheckpoint::save`]'s doc comment.
pub fn save_config<P: AsRef<Path>>(config: &TrainingConfig, path: P) -> TokenizerResult<()> {
    let json = serde_json::to_string_pretty(config)
        .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

    std::fs::write(path, json)?;

    Ok(())
}

/// Load a training configuration from JSON
///
/// File-I/O failures propagate as [`TokenizerError::IoError`] -- see
/// [`ModelCheckpoint::save`]'s doc comment.
pub fn load_config<P: AsRef<Path>>(path: P) -> TokenizerResult<TrainingConfig> {
    let json = std::fs::read_to_string(path)?;

    let config = serde_json::from_str(&json)
        .map_err(|e| TokenizerError::InternalError(format!("Failed to parse config: {}", e)))?;

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    /// Regression: `len_as_u32` must reject lengths that don't fit in
    /// `u32`, rather than silently truncating them (as the previous `as
    /// u32` casts at every write site did) into a small number while the
    /// full-size data still gets written after it — desynchronizing every
    /// tensor a reader parses from that point on.
    #[test]
    fn test_len_as_u32_rejects_overflow() {
        let too_big = u32::MAX as usize + 1;
        assert!(len_as_u32(too_big, "test length").is_err());
        assert!(len_as_u32(usize::MAX, "test length").is_err());
    }

    #[test]
    fn test_len_as_u32_accepts_in_range_values() {
        assert_eq!(len_as_u32(0, "test length").unwrap(), 0);
        assert_eq!(len_as_u32(42, "test length").unwrap(), 42);
        assert_eq!(
            len_as_u32(u32::MAX as usize, "test length").unwrap(),
            u32::MAX
        );
    }

    #[test]
    fn test_model_version() {
        let v1 = ModelVersion::new(1, 2, 3);
        assert_eq!(v1.to_string(), "1.2.3");

        let v2 = ModelVersion::parse("1.2.3").unwrap();
        assert_eq!(v1, v2);

        assert!(v1.is_compatible_with(&v2));

        let v3 = ModelVersion::new(2, 0, 0);
        assert!(!v1.is_compatible_with(&v3));
    }

    #[test]
    fn test_model_metadata() {
        let version = ModelVersion::new(1, 0, 0);
        let mut metadata = ModelMetadata::new(version, "TestModel".to_string(), 8, 16);

        metadata.add_custom("author".to_string(), "Test User".to_string());
        assert_eq!(metadata.custom.get("author").unwrap(), "Test User");

        let before = metadata.modified_at;
        std::thread::sleep(std::time::Duration::from_millis(10));
        metadata.touch();
        assert!(metadata.modified_at > before);
    }

    #[test]
    fn test_checkpoint_creation() {
        let version = ModelVersion::new(1, 0, 0);
        let metadata = ModelMetadata::new(version, "TestModel".to_string(), 8, 16);
        let mut checkpoint = ModelCheckpoint::new(metadata);

        // Add some weights
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        checkpoint.add_weight("test_weight".to_string(), data.clone(), shape.clone());

        // Retrieve weights
        let (retrieved_data, retrieved_shape) = checkpoint.get_weight("test_weight").unwrap();
        assert_eq!(retrieved_data, &data[..]);
        assert_eq!(retrieved_shape, &shape[..]);
    }

    #[test]
    fn test_checkpoint_array2() {
        let version = ModelVersion::new(1, 0, 0);
        let metadata = ModelMetadata::new(version, "TestModel".to_string(), 8, 16);
        let mut checkpoint = ModelCheckpoint::new(metadata);

        // Create a 2x3 array
        let mut array = Array2::zeros((2, 3));
        array[[0, 0]] = 1.0;
        array[[0, 1]] = 2.0;
        array[[0, 2]] = 3.0;
        array[[1, 0]] = 4.0;
        array[[1, 1]] = 5.0;
        array[[1, 2]] = 6.0;

        checkpoint.add_array2("matrix".to_string(), &array);

        // Retrieve and verify
        let retrieved = checkpoint.get_array2("matrix").unwrap();
        assert_eq!(retrieved.shape(), &[2, 3]);
        assert_eq!(retrieved[[0, 0]], 1.0);
        assert_eq!(retrieved[[1, 2]], 6.0);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let temp_dir = env::temp_dir();
        let checkpoint_path = temp_dir.join("test_checkpoint.safetensors");

        // Create and save checkpoint
        let version = ModelVersion::new(1, 0, 0);
        let mut metadata = ModelMetadata::new(version, "TestModel".to_string(), 4, 8);
        metadata.add_custom("test".to_string(), "value".to_string());

        let mut checkpoint = ModelCheckpoint::new(metadata);

        let mut encoder = Array2::zeros((4, 8));
        for i in 0..4 {
            for j in 0..8 {
                encoder[[i, j]] = (i * 8 + j) as f32;
            }
        }
        checkpoint.add_array2("encoder".to_string(), &encoder);

        checkpoint.save(&checkpoint_path).unwrap();

        // Load and verify
        let loaded = ModelCheckpoint::load(&checkpoint_path).unwrap();
        assert_eq!(loaded.metadata.model_type, "TestModel");
        assert_eq!(loaded.metadata.input_dim, 4);
        assert_eq!(loaded.metadata.embed_dim, 8);
        assert_eq!(loaded.metadata.custom.get("test").unwrap(), "value");

        let loaded_encoder = loaded.get_array2("encoder").unwrap();
        assert_eq!(loaded_encoder.shape(), &[4, 8]);
        assert_eq!(loaded_encoder[[0, 0]], 0.0);
        assert_eq!(loaded_encoder[[3, 7]], 31.0);

        // Cleanup
        std::fs::remove_file(&checkpoint_path).ok();
    }

    #[test]
    fn test_save_load_config() {
        let temp_dir = env::temp_dir();
        let config_path = temp_dir.join("test_config.json");

        let config = TrainingConfig {
            learning_rate: 0.001,
            num_epochs: 50,
            batch_size: 16,
            ..Default::default()
        };

        save_config(&config, &config_path).unwrap();
        let loaded_config = load_config(&config_path).unwrap();

        assert_eq!(loaded_config.learning_rate, 0.001);
        assert_eq!(loaded_config.num_epochs, 50);
        assert_eq!(loaded_config.batch_size, 16);

        // Cleanup
        std::fs::remove_file(&config_path).ok();
    }

    // Regression: `save`/`load`/`save_config`/`load_config` used to
    // flatten every `std::io::Error` into `TokenizerError::InternalError`'s
    // formatted string via `map_err`, discarding `ErrorKind` (a caller
    // could no longer tell `NotFound` apart from `PermissionDenied` or any
    // other I/O failure without parsing the message text). They now
    // propagate the real `std::io::Error` via `TokenizerError::IoError`'s
    // `#[from]` conversion (plain `?`), so `ErrorKind` survives end to end.
    #[test]
    fn test_load_missing_file_preserves_io_error_kind() {
        let temp_dir = env::temp_dir();
        let missing_path = temp_dir
            .join("kizzasi_tokenizer_persistence_regression_missing_checkpoint.safetensors");
        // Make sure it really doesn't exist before asserting on the error.
        std::fs::remove_file(&missing_path).ok();

        let err = ModelCheckpoint::load(&missing_path).expect_err("missing file must error");
        match err {
            TokenizerError::IoError(io_err) => {
                assert_eq!(io_err.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected TokenizerError::IoError(NotFound), got {other:?}"),
        }
    }

    #[test]
    fn test_load_config_missing_file_preserves_io_error_kind() {
        let temp_dir = env::temp_dir();
        let missing_path =
            temp_dir.join("kizzasi_tokenizer_persistence_regression_missing_config.json");
        std::fs::remove_file(&missing_path).ok();

        let err = load_config(&missing_path).expect_err("missing file must error");
        match err {
            TokenizerError::IoError(io_err) => {
                assert_eq!(io_err.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected TokenizerError::IoError(NotFound), got {other:?}"),
        }
    }

    #[test]
    fn test_save_to_missing_directory_preserves_io_error_kind() {
        // Writing under a parent directory that does not exist must
        // surface the real `NotFound` `std::io::Error`, not an opaque
        // `InternalError` string.
        let temp_dir = env::temp_dir();
        let bogus_path = temp_dir
            .join("kizzasi_tokenizer_persistence_regression_nonexistent_dir")
            .join("checkpoint.safetensors");

        let version = ModelVersion::new(1, 0, 0);
        let metadata = ModelMetadata::new(version, "TestModel".to_string(), 4, 8);
        let checkpoint = ModelCheckpoint::new(metadata);

        let err = checkpoint
            .save(&bogus_path)
            .expect_err("missing dir must error");
        match err {
            TokenizerError::IoError(io_err) => {
                assert_eq!(io_err.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected TokenizerError::IoError(NotFound), got {other:?}"),
        }
    }
}
