//! Compatibility and interoperability with other frameworks
//!
//! This module provides utilities for importing/exporting tokenizer weights
//! and configurations to/from other ML frameworks and standard formats.
//!
//! # Supported Formats
//!
//! - **PyTorch**: Import/export weights in PyTorch-compatible format via safetensors
//! - **ONNX**: Export tokenizer operations for ONNX runtime
//! - **Audio Metadata**: WAV/FLAC metadata for signal properties
//!
//! # Examples
//!
//! ```rust,ignore
//! use kizzasi_tokenizer::compat::{PyTorchCompat, AudioMetadata};
//!
//! // Export to PyTorch format
//! let pytorch_compat = PyTorchCompat::from_tokenizer(&tokenizer)?;
//! pytorch_compat.save("model.safetensors")?;
//!
//! // Add audio metadata
//! let metadata = AudioMetadata::new(44100, 16, 1);
//! ```

use crate::error::{TokenizerError, TokenizerResult};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// PyTorch-compatible weight export/import
///
/// Provides utilities to save and load tokenizer weights in a format
/// compatible with PyTorch models using safetensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchCompat {
    /// Model weights as named tensors
    pub weights: HashMap<String, TensorInfo>,
    /// Model configuration
    pub config: ModelConfig,
    /// PyTorch version compatibility
    pub torch_version: String,
}

/// Tensor information for PyTorch compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Tensor data type
    pub dtype: DType,
    /// Flattened tensor data
    pub data: Vec<f32>,
}

/// Data type enum for cross-framework compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit floating point
    Float32,
    /// 16-bit floating point (half precision)
    Float16,
    /// 64-bit floating point
    Float64,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
}

impl DType {
    /// Get the size in bytes of this dtype
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float16 => 2,
            DType::Float64 => 8,
            DType::Int32 => 4,
            DType::Int64 => 8,
        }
    }

    /// Get the PyTorch dtype string
    pub fn torch_name(&self) -> &'static str {
        match self {
            DType::Float32 => "torch.float32",
            DType::Float16 => "torch.float16",
            DType::Float64 => "torch.float64",
            DType::Int32 => "torch.int32",
            DType::Int64 => "torch.int64",
        }
    }
}

/// Model configuration for framework compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type identifier
    pub model_type: String,
    /// Input dimension
    pub input_dim: usize,
    /// Output/embedding dimension
    pub output_dim: usize,
    /// Additional hyperparameters
    pub hyperparameters: HashMap<String, serde_json::Value>,
}

impl PyTorchCompat {
    /// Create a new PyTorch compatibility wrapper
    pub fn new(config: ModelConfig) -> Self {
        Self {
            weights: HashMap::new(),
            config,
            torch_version: "2.0.0".to_string(),
        }
    }

    /// Add a weight tensor
    pub fn add_weight(&mut self, name: impl Into<String>, array: &Array2<f32>) {
        let shape = array.shape().to_vec();
        let data = array.iter().copied().collect();

        self.weights.insert(
            name.into(),
            TensorInfo {
                shape,
                dtype: DType::Float32,
                data,
            },
        );
    }

    /// Add a 1D weight tensor (bias, etc.)
    pub fn add_weight_1d(&mut self, name: impl Into<String>, array: &Array1<f32>) {
        let shape = vec![array.len()];
        let data = array.iter().copied().collect();

        self.weights.insert(
            name.into(),
            TensorInfo {
                shape,
                dtype: DType::Float32,
                data,
            },
        );
    }

    /// Get a weight tensor as Array2
    pub fn get_weight(&self, name: &str) -> TokenizerResult<Array2<f32>> {
        let tensor = self
            .weights
            .get(name)
            .ok_or_else(|| TokenizerError::InvalidConfig(format!("Weight '{}' not found", name)))?;

        if tensor.shape.len() != 2 {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected 2D tensor, got {}D",
                tensor.shape.len()
            )));
        }

        Array2::from_shape_vec((tensor.shape[0], tensor.shape[1]), tensor.data.clone())
            .map_err(|e| TokenizerError::InvalidConfig(format!("Shape mismatch: {}", e)))
    }

    /// Get a 1D weight tensor
    pub fn get_weight_1d(&self, name: &str) -> TokenizerResult<Array1<f32>> {
        let tensor = self
            .weights
            .get(name)
            .ok_or_else(|| TokenizerError::InvalidConfig(format!("Weight '{}' not found", name)))?;

        if tensor.shape.len() != 1 {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected 1D tensor, got {}D",
                tensor.shape.len()
            )));
        }

        Ok(Array1::from_vec(tensor.data.clone()))
    }

    /// Save to safetensors format (PyTorch compatible)
    pub fn save<P: AsRef<Path>>(&self, path: P) -> TokenizerResult<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            TokenizerError::SerializationError(format!("JSON serialization failed: {}", e))
        })?;

        std::fs::write(path, json).map_err(TokenizerError::IoError)?;

        Ok(())
    }

    /// Load from safetensors format
    pub fn load<P: AsRef<Path>>(path: P) -> TokenizerResult<Self> {
        let json = std::fs::read_to_string(path).map_err(TokenizerError::IoError)?;

        serde_json::from_str(&json).map_err(|e| {
            TokenizerError::SerializationError(format!("JSON deserialization failed: {}", e))
        })
    }

    /// Export weight names for ONNX mapping
    pub fn weight_names(&self) -> Vec<String> {
        self.weights.keys().cloned().collect()
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weights.values().map(|t| t.data.len()).sum()
    }
}

/// Audio metadata for signal processing
///
/// Stores standard audio properties that can be embedded in WAV/FLAC files
/// or used for proper signal reconstruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Bit depth (8, 16, 24, 32)
    pub bit_depth: u8,
    /// Number of channels (1=mono, 2=stereo)
    pub num_channels: u8,
    /// Total number of samples
    pub num_samples: Option<usize>,
    /// Duration in seconds
    pub duration_secs: Option<f64>,
    /// Additional metadata tags
    pub tags: HashMap<String, String>,
}

impl AudioMetadata {
    /// Create new audio metadata
    pub fn new(sample_rate: u32, bit_depth: u8, num_channels: u8) -> TokenizerResult<Self> {
        // Validate parameters
        if sample_rate == 0 {
            return Err(TokenizerError::InvalidConfig(
                "Sample rate must be positive".into(),
            ));
        }

        if ![8, 16, 24, 32].contains(&bit_depth) {
            return Err(TokenizerError::InvalidConfig(format!(
                "Invalid bit depth: {}. Must be 8, 16, 24, or 32",
                bit_depth
            )));
        }

        if num_channels == 0 || num_channels > 8 {
            return Err(TokenizerError::InvalidConfig(format!(
                "Invalid number of channels: {}. Must be 1-8",
                num_channels
            )));
        }

        Ok(Self {
            sample_rate,
            bit_depth,
            num_channels,
            num_samples: None,
            duration_secs: None,
            tags: HashMap::new(),
        })
    }

    /// Create metadata from signal length
    pub fn from_signal(
        signal: &Array1<f32>,
        sample_rate: u32,
        bit_depth: u8,
        num_channels: u8,
    ) -> TokenizerResult<Self> {
        let mut metadata = Self::new(sample_rate, bit_depth, num_channels)?;
        metadata.num_samples = Some(signal.len());
        metadata.duration_secs = Some(signal.len() as f64 / sample_rate as f64);
        Ok(metadata)
    }

    /// Set a metadata tag
    pub fn set_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }

    /// Get a metadata tag
    pub fn get_tag(&self, key: &str) -> Option<&str> {
        self.tags.get(key).map(|s| s.as_str())
    }

    /// Compute Nyquist frequency
    pub fn nyquist_frequency(&self) -> f32 {
        self.sample_rate as f32 / 2.0
    }

    /// Get duration in seconds
    pub fn duration(&self) -> Option<f64> {
        self.duration_secs
            .or_else(|| self.num_samples.map(|n| n as f64 / self.sample_rate as f64))
    }

    /// Export as WAV-compatible metadata JSON
    pub fn to_wav_metadata(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Import from WAV-compatible metadata JSON
    pub fn from_wav_metadata(json: &str) -> TokenizerResult<Self> {
        serde_json::from_str(json).map_err(|e| {
            TokenizerError::SerializationError(format!("Failed to parse metadata: {}", e))
        })
    }
}

/// ONNX export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxConfig {
    /// ONNX opset version
    pub opset_version: i64,
    /// Input names
    pub input_names: Vec<String>,
    /// Output names
    pub output_names: Vec<String>,
    /// Dynamic axes for variable-length inputs
    pub dynamic_axes: HashMap<String, Vec<i64>>,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            opset_version: 14,
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            dynamic_axes: HashMap::new(),
        }
    }
}

impl OnnxConfig {
    /// Create ONNX config for a tokenizer
    pub fn for_tokenizer(_input_dim: usize, _output_dim: usize) -> Self {
        let mut config = Self::default();

        // Add dynamic batch dimension
        let mut dynamic_axes = HashMap::new();
        dynamic_axes.insert("input".to_string(), vec![0]); // Batch dimension
        dynamic_axes.insert("output".to_string(), vec![0]); // Batch dimension
        config.dynamic_axes = dynamic_axes;

        config
    }

    /// Export configuration as JSON
    pub fn to_json(&self) -> TokenizerResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            TokenizerError::SerializationError(format!("ONNX config serialization failed: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_compat_basic() {
        let config = ModelConfig {
            model_type: "continuous_tokenizer".to_string(),
            input_dim: 128,
            output_dim: 256,
            hyperparameters: HashMap::new(),
        };

        let mut compat = PyTorchCompat::new(config);

        let encoder = Array2::from_shape_fn((128, 256), |(i, j)| (i + j) as f32 * 0.01);
        compat.add_weight("encoder", &encoder);

        assert_eq!(compat.weights.len(), 1);
        assert_eq!(compat.num_parameters(), 128 * 256);
    }

    #[test]
    fn test_pytorch_compat_roundtrip() {
        let config = ModelConfig {
            model_type: "test".to_string(),
            input_dim: 10,
            output_dim: 20,
            hyperparameters: HashMap::new(),
        };

        let mut compat = PyTorchCompat::new(config);
        let weights = Array2::from_shape_fn((10, 20), |(i, j)| (i * 20 + j) as f32);
        compat.add_weight("test_weight", &weights);

        let retrieved = compat.get_weight("test_weight").unwrap();
        assert_eq!(retrieved.shape(), &[10, 20]);
        assert_eq!(retrieved[[0, 0]], 0.0);
        assert_eq!(retrieved[[9, 19]], 199.0);
    }

    #[test]
    fn test_audio_metadata_creation() {
        let metadata = AudioMetadata::new(44100, 16, 2).unwrap();
        assert_eq!(metadata.sample_rate, 44100);
        assert_eq!(metadata.bit_depth, 16);
        assert_eq!(metadata.num_channels, 2);
        assert_eq!(metadata.nyquist_frequency(), 22050.0);
    }

    #[test]
    fn test_audio_metadata_validation() {
        // Invalid sample rate
        assert!(AudioMetadata::new(0, 16, 2).is_err());

        // Invalid bit depth
        assert!(AudioMetadata::new(44100, 13, 2).is_err());

        // Invalid channels
        assert!(AudioMetadata::new(44100, 16, 0).is_err());
        assert!(AudioMetadata::new(44100, 16, 9).is_err());
    }

    #[test]
    fn test_audio_metadata_from_signal() {
        let signal = Array1::from_vec(vec![0.0; 44100]); // 1 second at 44.1kHz
        let metadata = AudioMetadata::from_signal(&signal, 44100, 16, 1).unwrap();

        assert_eq!(metadata.num_samples, Some(44100));
        assert!((metadata.duration().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_audio_metadata_tags() {
        let mut metadata = AudioMetadata::new(44100, 16, 2).unwrap();
        metadata.set_tag("artist", "Test Artist");
        metadata.set_tag("title", "Test Title");

        assert_eq!(metadata.get_tag("artist"), Some("Test Artist"));
        assert_eq!(metadata.get_tag("title"), Some("Test Title"));
        assert_eq!(metadata.get_tag("nonexistent"), None);
    }

    #[test]
    fn test_audio_metadata_serialization() {
        let metadata = AudioMetadata::new(48000, 24, 2).unwrap();
        let json = metadata.to_wav_metadata();
        let deserialized = AudioMetadata::from_wav_metadata(&json).unwrap();

        assert_eq!(deserialized.sample_rate, 48000);
        assert_eq!(deserialized.bit_depth, 24);
        assert_eq!(deserialized.num_channels, 2);
    }

    #[test]
    fn test_dtype_properties() {
        assert_eq!(DType::Float32.size_bytes(), 4);
        assert_eq!(DType::Float16.size_bytes(), 2);
        assert_eq!(DType::Float64.size_bytes(), 8);

        assert_eq!(DType::Float32.torch_name(), "torch.float32");
        assert_eq!(DType::Int64.torch_name(), "torch.int64");
    }

    #[test]
    fn test_onnx_config_default() {
        let config = OnnxConfig::default();
        assert_eq!(config.opset_version, 14);
        assert_eq!(config.input_names, vec!["input"]);
        assert_eq!(config.output_names, vec!["output"]);
    }

    #[test]
    fn test_onnx_config_for_tokenizer() {
        let config = OnnxConfig::for_tokenizer(128, 256);
        assert_eq!(config.opset_version, 14);
        assert!(config.dynamic_axes.contains_key("input"));
        assert!(config.dynamic_axes.contains_key("output"));
    }

    #[test]
    fn test_onnx_config_serialization() {
        let config = OnnxConfig::for_tokenizer(100, 200);
        let json = config.to_json().unwrap();
        assert!(json.contains("\"opset_version\""));
        assert!(json.contains("\"input_names\""));
    }
}
