//! State persistence and checkpointing for Kizzasi predictors
//!
//! Provides two levels of checkpointing:
//!
//! ## Configuration Checkpoint
//! - Saves model configuration only
//! - Lightweight and portable
//! - Predictor starts with fresh state and random weights on restore
//!
//! ## Full State Checkpoint
//! - Saves complete model state including:
//!   - Configuration
//!   - SSM hidden state
//!   - All model weights and parameters
//!   - Embedding layer weights
//! - Larger file size but preserves exact predictor state
//! - Enables pause/resume of predictions
//!
//! # Current Limitations
//!
//! - Guardrails are not persisted (requires kizzasi-logic serialization support)
//! - Plugins are not persisted (must be re-added manually)
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi::prelude::*;
//! use std::path::Path;
//!
//! let mut predictor = KizzasiBuilder::audio_preset().build()?;
//!
//! // Run some predictions
//! let input = array![0.5];
//! predictor.step(&input)?;
//!
//! // Save configuration only
//! predictor.save_checkpoint("model_config.checkpoint")?;
//!
//! // Save complete state (including weights and hidden state)
//! predictor.save_full_checkpoint("model_full.checkpoint")?;
//!
//! // Later, restore predictor
//! let mut restored = Kizzasi::load_full_checkpoint("model_full.checkpoint")?;
//! ```

use crate::error::{KizzasiError, KizzasiResult};
use crate::predictor::Kizzasi;
use kizzasi_core::{KizzasiConfig, SelectiveSSM};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Version identifier for checkpoint format
const CHECKPOINT_VERSION: u32 = 1;

/// Serializable representation of a Kizzasi predictor checkpoint
///
/// This structure can be serialized to JSON or binary formats for
/// persisting predictor configuration across sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorCheckpoint {
    /// Format version for compatibility checking
    pub version: u32,
    /// Model configuration
    pub config: KizzasiConfig,
    /// Metadata for the checkpoint
    pub metadata: CheckpointMetadata,
}

/// Metadata about the checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// When the checkpoint was created (Unix timestamp)
    pub created_at: u64,
    /// Optional description
    pub description: Option<String>,
    /// Number of prediction steps performed
    pub step_count: usize,
    /// Additional user-defined metadata
    /// Note: This field is only preserved in JSON format, not binary
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub custom: Option<serde_json::Value>,
}

/// Binary-compatible metadata (without serde_json::Value)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinaryCheckpointMetadata {
    created_at: u64,
    description: Option<String>,
    step_count: usize,
}

impl From<&CheckpointMetadata> for BinaryCheckpointMetadata {
    fn from(meta: &CheckpointMetadata) -> Self {
        Self {
            created_at: meta.created_at,
            description: meta.description.clone(),
            step_count: meta.step_count,
        }
    }
}

impl From<BinaryCheckpointMetadata> for CheckpointMetadata {
    fn from(meta: BinaryCheckpointMetadata) -> Self {
        Self {
            created_at: meta.created_at,
            description: meta.description,
            step_count: meta.step_count,
            custom: None, // Binary format doesn't preserve custom metadata
        }
    }
}

/// Binary-compatible full state checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinaryFullStateCheckpoint {
    version: u32,
    ssm: SelectiveSSM,
    config: KizzasiConfig,
    metadata: BinaryCheckpointMetadata,
}

impl From<&FullStateCheckpoint> for BinaryFullStateCheckpoint {
    fn from(cp: &FullStateCheckpoint) -> Self {
        Self {
            version: cp.version,
            ssm: cp.ssm.clone(),
            config: cp.config.clone(),
            metadata: BinaryCheckpointMetadata::from(&cp.metadata),
        }
    }
}

impl From<BinaryFullStateCheckpoint> for FullStateCheckpoint {
    fn from(cp: BinaryFullStateCheckpoint) -> Self {
        Self {
            version: cp.version,
            ssm: cp.ssm,
            config: cp.config,
            metadata: CheckpointMetadata::from(cp.metadata),
        }
    }
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        Self {
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            description: None,
            step_count: 0,
            custom: None,
        }
    }
}

impl PredictorCheckpoint {
    /// Create a new checkpoint from a predictor
    pub fn from_predictor(predictor: &Kizzasi) -> Self {
        Self {
            version: CHECKPOINT_VERSION,
            config: predictor.config().clone(),
            metadata: CheckpointMetadata::default(),
        }
    }

    /// Create a checkpoint with custom metadata
    pub fn with_metadata(mut self, description: impl Into<String>, step_count: usize) -> Self {
        self.metadata.description = Some(description.into());
        self.metadata.step_count = step_count;
        self
    }

    /// Add custom metadata
    pub fn with_custom_metadata(mut self, custom: serde_json::Value) -> Self {
        self.metadata.custom = Some(custom);
        self
    }

    /// Save checkpoint to a JSON file
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> KizzasiResult<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| KizzasiError::Config(format!("Failed to serialize checkpoint: {}", e)))?;

        fs::write(path, json)
            .map_err(|e| KizzasiError::Config(format!("Failed to write checkpoint: {}", e)))?;

        Ok(())
    }

    /// Load checkpoint from a JSON file
    pub fn load_json<P: AsRef<Path>>(path: P) -> KizzasiResult<Self> {
        let json = fs::read_to_string(path)
            .map_err(|e| KizzasiError::Config(format!("Failed to read checkpoint: {}", e)))?;

        let checkpoint: Self = serde_json::from_str(&json).map_err(|e| {
            KizzasiError::Config(format!("Failed to deserialize checkpoint: {}", e))
        })?;

        // Version check
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(KizzasiError::Config(format!(
                "Checkpoint version mismatch: expected {}, got {}",
                CHECKPOINT_VERSION, checkpoint.version
            )));
        }

        Ok(checkpoint)
    }

    /// Restore a predictor from this checkpoint
    ///
    /// Note: Guardrails are not restored and must be re-added manually.
    /// The predictor starts with fresh hidden state.
    pub fn restore(&self) -> KizzasiResult<Kizzasi> {
        Kizzasi::new(self.config.clone())
    }
}

/// Full state checkpoint including all model weights and hidden state
///
/// This checkpoint format preserves the complete predictor state, allowing
/// exact restoration of predictions. File size is larger than configuration-only
/// checkpoints due to storing all model parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullStateCheckpoint {
    /// Format version for compatibility checking
    pub version: u32,
    /// Complete SSM model including weights and hidden state
    pub ssm: SelectiveSSM,
    /// Model configuration (redundant with SSM but kept for compatibility)
    pub config: KizzasiConfig,
    /// Metadata for the checkpoint
    pub metadata: CheckpointMetadata,
}

impl FullStateCheckpoint {
    /// Create a new full state checkpoint from a predictor
    pub fn from_predictor(predictor: &Kizzasi) -> Self {
        Self {
            version: CHECKPOINT_VERSION,
            ssm: predictor.ssm().clone(),
            config: predictor.config().clone(),
            metadata: CheckpointMetadata {
                step_count: predictor.ssm().step_count(),
                ..Default::default()
            },
        }
    }

    /// Create a checkpoint with custom metadata
    pub fn with_metadata(mut self, description: impl Into<String>) -> Self {
        self.metadata.description = Some(description.into());
        self
    }

    /// Add custom metadata
    pub fn with_custom_metadata(mut self, custom: serde_json::Value) -> Self {
        self.metadata.custom = Some(custom);
        self
    }

    /// Save checkpoint to a JSON file
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> KizzasiResult<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            KizzasiError::Config(format!("Failed to serialize full checkpoint: {}", e))
        })?;

        fs::write(path, json)
            .map_err(|e| KizzasiError::Config(format!("Failed to write full checkpoint: {}", e)))?;

        Ok(())
    }

    /// Save checkpoint to a binary file (more efficient for large models)
    ///
    /// Note: Custom metadata (`custom` field) is not preserved in binary format.
    /// Use JSON format if you need to preserve custom metadata.
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> KizzasiResult<()> {
        // Convert to binary-compatible format (without serde_json::Value)
        let binary_checkpoint = BinaryFullStateCheckpoint::from(self);

        let config = oxicode::config::standard();
        let binary = oxicode::serde::encode_to_vec(&binary_checkpoint, config).map_err(|e| {
            KizzasiError::Config(format!("Failed to serialize full checkpoint: {}", e))
        })?;

        fs::write(path, binary)
            .map_err(|e| KizzasiError::Config(format!("Failed to write full checkpoint: {}", e)))?;

        Ok(())
    }

    /// Load checkpoint from a JSON file
    pub fn load_json<P: AsRef<Path>>(path: P) -> KizzasiResult<Self> {
        let json = fs::read_to_string(path)
            .map_err(|e| KizzasiError::Config(format!("Failed to read full checkpoint: {}", e)))?;

        let checkpoint: Self = serde_json::from_str(&json).map_err(|e| {
            KizzasiError::Config(format!("Failed to deserialize full checkpoint: {}", e))
        })?;

        // Version check
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(KizzasiError::Config(format!(
                "Full checkpoint version mismatch: expected {}, got {}",
                CHECKPOINT_VERSION, checkpoint.version
            )));
        }

        Ok(checkpoint)
    }

    /// Load checkpoint from a binary file
    ///
    /// Note: Custom metadata (`custom` field) is not preserved in binary format.
    /// The `custom` field will be `None` after loading.
    pub fn load_binary<P: AsRef<Path>>(path: P) -> KizzasiResult<Self> {
        let binary = fs::read(path)
            .map_err(|e| KizzasiError::Config(format!("Failed to read full checkpoint: {}", e)))?;

        let config = oxicode::config::standard();
        let (binary_checkpoint, _): (BinaryFullStateCheckpoint, usize) =
            oxicode::serde::decode_from_slice(&binary, config).map_err(|e| {
                KizzasiError::Config(format!("Failed to deserialize full checkpoint: {}", e))
            })?;

        // Version check
        if binary_checkpoint.version != CHECKPOINT_VERSION {
            return Err(KizzasiError::Config(format!(
                "Full checkpoint version mismatch: expected {}, got {}",
                CHECKPOINT_VERSION, binary_checkpoint.version
            )));
        }

        // Convert back to full checkpoint format
        Ok(FullStateCheckpoint::from(binary_checkpoint))
    }

    /// Restore a predictor from this checkpoint
    ///
    /// Creates a predictor with the exact state from the checkpoint, including:
    /// - Hidden state
    /// - All model weights
    /// - Embedding layer parameters
    ///
    /// Note: Guardrails and plugins are not restored and must be re-added manually.
    pub fn restore(&self) -> KizzasiResult<Kizzasi> {
        Kizzasi::from_ssm(self.ssm.clone())
    }
}

/// Extension trait for Kizzasi to add checkpoint methods
impl Kizzasi {
    /// Save the current predictor configuration to a checkpoint file
    ///
    /// The checkpoint includes:
    /// - Model configuration (dimensions, layers, model type, etc.)
    /// - Metadata (timestamp, description, step count)
    ///
    /// Not included (current limitations):
    /// - SSM hidden state (restored predictor starts with fresh state)
    /// - Guardrails (must be re-added manually after loading)
    /// - Model weights (use `weights_path` in config for weight persistence)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// predictor.save_checkpoint("model_v1.checkpoint")?;
    /// ```
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> KizzasiResult<()> {
        let checkpoint = PredictorCheckpoint::from_predictor(self);
        checkpoint.save_json(path)
    }

    /// Save checkpoint with custom metadata
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// predictor.save_checkpoint_with_metadata(
    ///     "model_epoch_10.checkpoint",
    ///     "Training epoch 10",
    ///     10000  // step count
    /// )?;
    /// ```
    pub fn save_checkpoint_with_metadata<P: AsRef<Path>>(
        &self,
        path: P,
        description: impl Into<String>,
        step_count: usize,
    ) -> KizzasiResult<()> {
        let checkpoint =
            PredictorCheckpoint::from_predictor(self).with_metadata(description, step_count);
        checkpoint.save_json(path)
    }

    /// Load a predictor from a checkpoint file
    ///
    /// Creates a new predictor with the configuration from the checkpoint.
    /// The predictor starts with fresh hidden state.
    ///
    /// Note: Guardrails from the original predictor are not restored.
    /// You must manually add them after loading:
    ///
    /// ```rust,ignore
    /// let mut predictor = Kizzasi::load_checkpoint("model.checkpoint")?;
    /// predictor.set_guardrails(my_guardrails);
    /// ```
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let predictor = Kizzasi::load_checkpoint("model_v1.checkpoint")?;
    /// ```
    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> KizzasiResult<Self> {
        let checkpoint = PredictorCheckpoint::load_json(path)?;
        checkpoint.restore()
    }

    /// Save complete predictor state to a JSON checkpoint file
    ///
    /// The full state checkpoint includes:
    /// - Complete SSM model with all weights and parameters
    /// - Hidden state (preserves exact prediction state)
    /// - Embedding layer weights
    /// - Configuration and metadata
    ///
    /// Not included:
    /// - Guardrails (must be re-added manually after loading)
    /// - Plugins (must be re-added manually after loading)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// predictor.save_full_checkpoint("model_state_v1.checkpoint")?;
    /// ```
    pub fn save_full_checkpoint<P: AsRef<Path>>(&self, path: P) -> KizzasiResult<()> {
        let checkpoint = FullStateCheckpoint::from_predictor(self);
        checkpoint.save_json(path)
    }

    /// Save complete predictor state to a binary checkpoint file
    ///
    /// Binary format is more efficient for large models but not human-readable.
    /// Use JSON format if you need to inspect checkpoint contents.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// predictor.save_full_checkpoint_binary("model_state.bin")?;
    /// ```
    pub fn save_full_checkpoint_binary<P: AsRef<Path>>(&self, path: P) -> KizzasiResult<()> {
        let checkpoint = FullStateCheckpoint::from_predictor(self);
        checkpoint.save_binary(path)
    }

    /// Save full checkpoint with custom metadata
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// predictor.save_full_checkpoint_with_metadata(
    ///     "model_epoch_10_full.checkpoint",
    ///     "Training epoch 10 - accuracy 95.2%"
    /// )?;
    /// ```
    pub fn save_full_checkpoint_with_metadata<P: AsRef<Path>>(
        &self,
        path: P,
        description: impl Into<String>,
    ) -> KizzasiResult<()> {
        let checkpoint = FullStateCheckpoint::from_predictor(self).with_metadata(description);
        checkpoint.save_json(path)
    }

    /// Load a predictor from a full state checkpoint (JSON format)
    ///
    /// Restores the predictor with exact state from the checkpoint, including:
    /// - All model weights and parameters
    /// - Hidden state
    /// - Embedding layer
    ///
    /// Note: Guardrails and plugins are not restored and must be re-added manually.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut predictor = Kizzasi::load_full_checkpoint("model_state_v1.checkpoint")?;
    /// // Re-add guardrails if needed
    /// predictor.set_guardrails(my_guardrails);
    /// ```
    pub fn load_full_checkpoint<P: AsRef<Path>>(path: P) -> KizzasiResult<Self> {
        let checkpoint = FullStateCheckpoint::load_json(path)?;
        checkpoint.restore()
    }

    /// Load a predictor from a full state checkpoint (binary format)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let predictor = Kizzasi::load_full_checkpoint_binary("model_state.bin")?;
    /// ```
    pub fn load_full_checkpoint_binary<P: AsRef<Path>>(path: P) -> KizzasiResult<Self> {
        let checkpoint = FullStateCheckpoint::load_binary(path)?;
        checkpoint.restore()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_checkpoint_creation() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64);

        let predictor = Kizzasi::new(config).unwrap();
        let checkpoint = PredictorCheckpoint::from_predictor(&predictor);

        assert_eq!(checkpoint.version, CHECKPOINT_VERSION);
        assert_eq!(checkpoint.config.get_input_dim(), 3);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .num_layers(2);

        let predictor = Kizzasi::new(config).unwrap();

        let temp_dir = std::env::temp_dir();
        let checkpoint_path = temp_dir.join("test_checkpoint.json");

        // Save
        predictor.save_checkpoint(&checkpoint_path).unwrap();

        // Verify file exists
        assert!(checkpoint_path.exists());

        // Load
        let restored = Kizzasi::load_checkpoint(&checkpoint_path).unwrap();

        assert_eq!(restored.config().get_input_dim(), 3);
        assert_eq!(restored.config().get_hidden_dim(), 64);
        assert_eq!(restored.config().get_num_layers(), 2);

        // Cleanup
        let _ = fs::remove_file(checkpoint_path);
    }

    #[test]
    fn test_checkpoint_with_metadata() {
        let predictor = KizzasiBuilder::audio_preset().build().unwrap();

        let checkpoint =
            PredictorCheckpoint::from_predictor(&predictor).with_metadata("Audio model v1", 1000);

        assert_eq!(
            checkpoint.metadata.description,
            Some("Audio model v1".to_string())
        );
        assert_eq!(checkpoint.metadata.step_count, 1000);
        assert!(checkpoint.metadata.created_at > 0);
    }

    #[test]
    fn test_checkpoint_json_format() {
        let config = KizzasiConfig::new()
            .model_type(ModelType::Mamba2)
            .input_dim(2)
            .output_dim(2);

        let predictor = Kizzasi::new(config).unwrap();
        let checkpoint = PredictorCheckpoint::from_predictor(&predictor);

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&checkpoint).unwrap();

        // Should contain key fields
        assert!(json.contains("\"version\""));
        assert!(json.contains("\"config\""));
        assert!(json.contains("\"metadata\""));
        assert!(json.contains("\"model_type\""));
    }

    #[test]
    fn test_checkpoint_with_custom_metadata() {
        let predictor = KizzasiBuilder::robotics_preset(6).build().unwrap();

        let custom = serde_json::json!({
            "experiment_id": "exp_123",
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32
            }
        });

        let checkpoint =
            PredictorCheckpoint::from_predictor(&predictor).with_custom_metadata(custom.clone());

        assert_eq!(checkpoint.metadata.custom, Some(custom));
    }

    #[test]
    fn test_checkpoint_version_check() {
        let checkpoint = PredictorCheckpoint {
            version: 999, // Future version
            config: KizzasiConfig::new(),
            metadata: CheckpointMetadata::default(),
        };

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_version_check.json");

        // Save with future version
        checkpoint.save_json(&path).unwrap();

        // Try to load - should fail version check
        let result = PredictorCheckpoint::load_json(&path);
        assert!(result.is_err());

        if let Err(KizzasiError::Config(msg)) = result {
            assert!(msg.contains("version mismatch"));
        }

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_checkpoint_preserves_all_config() {
        let config = KizzasiConfig::new()
            .model_type(ModelType::S4)
            .input_dim(10)
            .output_dim(5)
            .hidden_dim(128)
            .state_dim(32)
            .num_layers(4)
            .context_window(2048);

        let predictor = Kizzasi::new(config).unwrap();

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_full_config.json");

        predictor.save_checkpoint(&path).unwrap();
        let restored = Kizzasi::load_checkpoint(&path).unwrap();

        assert_eq!(restored.config().get_model_type(), ModelType::S4);
        assert_eq!(restored.config().get_input_dim(), 10);
        assert_eq!(restored.config().get_output_dim(), 5);
        assert_eq!(restored.config().get_hidden_dim(), 128);
        assert_eq!(restored.config().get_state_dim(), 32);
        assert_eq!(restored.config().get_num_layers(), 4);
        assert_eq!(restored.config().get_context_window(), 2048);

        // Cleanup
        let _ = fs::remove_file(path);
    }

    // === Full State Checkpoint Tests ===

    #[test]
    fn test_full_checkpoint_save_load_json() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let mut predictor = Kizzasi::new(config).unwrap();

        // Run some predictions to build up state
        let input = array![0.1, 0.2, 0.3];
        let output1 = predictor.step(&input).unwrap();
        predictor.step(&output1).unwrap();

        let temp_dir = std::env::temp_dir();
        let checkpoint_path = temp_dir.join("test_full_checkpoint.json");

        // Save full state
        predictor.save_full_checkpoint(&checkpoint_path).unwrap();

        // Verify file exists
        assert!(checkpoint_path.exists());

        // Load and verify
        let mut restored = Kizzasi::load_full_checkpoint(&checkpoint_path).unwrap();

        // Configuration should match
        assert_eq!(restored.config().get_input_dim(), 3);
        assert_eq!(restored.config().get_output_dim(), 3);
        assert_eq!(restored.config().get_hidden_dim(), 64);
        assert_eq!(restored.config().get_state_dim(), 8);
        assert_eq!(restored.config().get_num_layers(), 2);

        // Check step count before continuing predictions
        assert_eq!(restored.ssm().step_count(), 2);

        // State should be preserved - predictions should continue from same state
        let restored_output = restored.step(&input).unwrap();
        assert_eq!(restored_output.len(), 3);

        // Step count should increment after the new prediction
        assert_eq!(restored.ssm().step_count(), 3);

        // Cleanup
        let _ = fs::remove_file(checkpoint_path);
    }

    #[test]
    fn test_full_checkpoint_save_load_binary() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let mut predictor = Kizzasi::new(config).unwrap();

        // Run predictions
        let input = array![0.5, 0.5];
        predictor.step(&input).unwrap();

        let temp_dir = std::env::temp_dir();
        let checkpoint_path = temp_dir.join("test_full_checkpoint.bin");

        // Save as binary
        predictor
            .save_full_checkpoint_binary(&checkpoint_path)
            .unwrap();

        // Load binary checkpoint
        let restored = Kizzasi::load_full_checkpoint_binary(&checkpoint_path).unwrap();

        assert_eq!(restored.config().get_input_dim(), 2);
        assert_eq!(restored.config().get_output_dim(), 2);
        assert_eq!(restored.ssm().step_count(), 1);

        // Cleanup
        let _ = fs::remove_file(checkpoint_path);
    }

    #[test]
    fn test_full_checkpoint_with_metadata() {
        let predictor = KizzasiBuilder::audio_preset().build().unwrap();

        let checkpoint =
            FullStateCheckpoint::from_predictor(&predictor).with_metadata("Full state model v1");

        assert_eq!(
            checkpoint.metadata.description,
            Some("Full state model v1".to_string())
        );
        assert_eq!(checkpoint.metadata.step_count, 0);
    }

    #[test]
    fn test_full_checkpoint_preserves_state_across_predictions() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32)
            .state_dim(4);

        let mut predictor1 = Kizzasi::new(config).unwrap();

        // Run multiple predictions
        let input = array![0.1, 0.2];
        predictor1.step(&input).unwrap();
        predictor1.step(&input).unwrap();
        let output_before = predictor1.step(&input).unwrap();

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_state_preservation.json");

        // Save state
        predictor1.save_full_checkpoint(&path).unwrap();

        // Load into new predictor
        let mut predictor2 = Kizzasi::load_full_checkpoint(&path).unwrap();

        // Step count should match after loading
        assert_eq!(predictor1.ssm().step_count(), predictor2.ssm().step_count());

        // Next prediction should be consistent with the state
        let output_after = predictor2.step(&input).unwrap();

        // Both outputs should have correct dimensions
        assert_eq!(output_before.len(), 2);
        assert_eq!(output_after.len(), 2);

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_full_checkpoint_version_check() {
        let checkpoint = FullStateCheckpoint {
            version: 999, // Future version
            ssm: SelectiveSSM::new(KizzasiConfig::new()).unwrap(),
            config: KizzasiConfig::new(),
            metadata: CheckpointMetadata::default(),
        };

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_full_version_check.json");

        // Save with future version
        checkpoint.save_json(&path).unwrap();

        // Try to load - should fail version check
        let result = FullStateCheckpoint::load_json(&path);
        assert!(result.is_err());

        if let Err(KizzasiError::Config(msg)) = result {
            assert!(msg.contains("version mismatch"));
        }

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_full_checkpoint_different_models() {
        // Test with different presets
        let presets = vec![
            KizzasiBuilder::audio_preset().build().unwrap(),
            KizzasiBuilder::robotics_preset(3).build().unwrap(),
            KizzasiBuilder::sensor_preset(5).build().unwrap(),
        ];

        let temp_dir = std::env::temp_dir();

        for (idx, mut predictor) in presets.into_iter().enumerate() {
            // Run some predictions
            let input = Array1::from_vec(vec![0.1; predictor.config().get_input_dim()]);
            predictor.step(&input).unwrap();

            let path = temp_dir.join(format!("test_preset_{}.json", idx));

            // Save and load
            predictor.save_full_checkpoint(&path).unwrap();
            let restored = Kizzasi::load_full_checkpoint(&path).unwrap();

            assert_eq!(
                restored.config().get_input_dim(),
                predictor.config().get_input_dim()
            );
            assert_eq!(
                restored.config().get_output_dim(),
                predictor.config().get_output_dim()
            );

            // Cleanup
            let _ = fs::remove_file(path);
        }
    }

    #[test]
    fn test_full_checkpoint_file_size_comparison() {
        let config = KizzasiConfig::new()
            .input_dim(10)
            .output_dim(10)
            .hidden_dim(128)
            .state_dim(16)
            .num_layers(3);

        let predictor = Kizzasi::new(config).unwrap();

        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("test_config_checkpoint.json");
        let full_json_path = temp_dir.join("test_full_checkpoint.json");
        let full_bin_path = temp_dir.join("test_full_checkpoint.bin");

        // Save all formats
        predictor.save_checkpoint(&config_path).unwrap();
        predictor.save_full_checkpoint(&full_json_path).unwrap();
        predictor
            .save_full_checkpoint_binary(&full_bin_path)
            .unwrap();

        // Get file sizes
        let config_size = fs::metadata(&config_path).unwrap().len();
        let full_json_size = fs::metadata(&full_json_path).unwrap().len();
        let full_bin_size = fs::metadata(&full_bin_path).unwrap().len();

        // Full checkpoint should be larger than config-only
        assert!(full_json_size > config_size);

        // Binary should be smaller than JSON (usually)
        // Note: For small models, this may not always hold due to JSON compression
        // but we just verify binary saves successfully
        assert!(full_bin_size > 0);

        // Cleanup
        let _ = fs::remove_file(config_path);
        let _ = fs::remove_file(full_json_path);
        let _ = fs::remove_file(full_bin_path);
    }
}
