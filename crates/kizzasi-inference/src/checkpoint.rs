//! State checkpointing and serialization
//!
//! This module provides functionality for saving and loading inference state,
//! enabling:
//! - Long-running inference sessions with state persistence
//! - State migration between processes
//! - Distributed inference with state sharding
//! - Rollback to previous states for constraint violations

use crate::context::{ContextConfig, InferenceContext};
use crate::error::{InferenceError, InferenceResult};
use kizzasi_core::HiddenState;
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Version for checkpoint format (for backward compatibility)
const CHECKPOINT_VERSION: u32 = 1;

/// Serializable representation of a HiddenState
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableHiddenState {
    /// Hidden dimension
    hidden_dim: usize,
    /// State dimension
    state_dim: usize,
    /// Flattened state matrix (row-major)
    state_data: Vec<f32>,
    /// Whether state has been updated
    updated: bool,
}

impl SerializableHiddenState {
    /// Convert from HiddenState
    fn from_hidden_state(hs: &HiddenState) -> Self {
        let state = hs.state();
        let shape = state.shape();
        let state_data: Vec<f32> = state.iter().copied().collect();

        Self {
            hidden_dim: shape[0],
            state_dim: shape[1],
            state_data,
            updated: true, // Assume updated if we're serializing
        }
    }

    /// Convert to HiddenState
    fn to_hidden_state(&self) -> InferenceResult<HiddenState> {
        if self.state_data.len() != self.hidden_dim * self.state_dim {
            return Err(InferenceError::DimensionMismatch {
                expected: self.hidden_dim * self.state_dim,
                got: self.state_data.len(),
            });
        }

        let mut hs = HiddenState::new(self.hidden_dim, self.state_dim);

        // Reconstruct the state matrix
        let state_array = scirs2_core::ndarray::Array2::from_shape_vec(
            (self.hidden_dim, self.state_dim),
            self.state_data.clone(),
        )
        .map_err(|e| InferenceError::SerializationError(e.to_string()))?;

        hs.update(state_array);
        Ok(hs)
    }
}

/// Checkpoint containing full inference state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Checkpoint format version
    version: u32,
    /// Context configuration
    config: ContextConfig,
    /// Serialized hidden states for each layer
    states: Vec<SerializableHiddenState>,
    /// History of past inputs (flattened)
    history: Vec<Vec<f32>>,
    /// Number of steps processed
    step_count: usize,
    /// Optional metadata
    metadata: CheckpointMetadata,
}

/// Metadata for checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Timestamp when checkpoint was created
    pub timestamp: u64,
    /// Optional description
    pub description: String,
    /// Model identifier
    pub model_id: String,
    /// Custom tags
    pub tags: Vec<String>,
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0); // Fallback to 0 if system time is before UNIX_EPOCH

        Self {
            timestamp,
            description: String::new(),
            model_id: String::from("unknown"),
            tags: Vec::new(),
        }
    }
}

impl Checkpoint {
    /// Create a checkpoint from an InferenceContext
    pub fn from_context(context: &InferenceContext) -> Self {
        let states: Vec<SerializableHiddenState> = context
            .states()
            .iter()
            .map(SerializableHiddenState::from_hidden_state)
            .collect();

        let history: Vec<Vec<f32>> = context
            .recent_history(context.history_len())
            .into_iter()
            .rev() // Reverse back to chronological order
            .map(|arr| arr.iter().copied().collect())
            .collect();

        Self {
            version: CHECKPOINT_VERSION,
            config: context.config().clone(),
            states,
            history,
            step_count: context.step_count(),
            metadata: CheckpointMetadata::default(),
        }
    }

    /// Restore an InferenceContext from this checkpoint
    pub fn to_context(&self) -> InferenceResult<InferenceContext> {
        if self.version != CHECKPOINT_VERSION {
            return Err(InferenceError::SerializationError(format!(
                "Incompatible checkpoint version: expected {}, got {}",
                CHECKPOINT_VERSION, self.version
            )));
        }

        let mut context = InferenceContext::new(self.config.clone());

        // Restore states
        for (i, serialized_state) in self.states.iter().enumerate() {
            let state = serialized_state.to_hidden_state()?;
            context.update_state(i, state)?;
        }

        // Restore history
        for hist_vec in &self.history {
            let arr = Array1::from_vec(hist_vec.clone());
            context.push(arr);
        }

        // Restore step count (push increments it, so we need to adjust)
        // Actually, InferenceContext doesn't expose step_count setter, so this is handled by push

        Ok(context)
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: CheckpointMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set description
    pub fn with_description(mut self, description: String) -> Self {
        self.metadata.description = description;
        self
    }

    /// Set model ID
    pub fn with_model_id(mut self, model_id: String) -> Self {
        self.metadata.model_id = model_id;
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: String) -> Self {
        self.metadata.tags.push(tag);
        self
    }

    /// Get metadata
    pub fn metadata(&self) -> &CheckpointMetadata {
        &self.metadata
    }

    /// Get checkpoint version
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Get step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Save checkpoint to file (JSON format)
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> InferenceResult<()> {
        let file =
            File::create(path).map_err(|e| InferenceError::SerializationError(e.to_string()))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| InferenceError::SerializationError(e.to_string()))?;
        Ok(())
    }

    /// Load checkpoint from JSON file
    pub fn load_json<P: AsRef<Path>>(path: P) -> InferenceResult<Self> {
        let file =
            File::open(path).map_err(|e| InferenceError::SerializationError(e.to_string()))?;
        let reader = BufReader::new(file);
        let checkpoint = serde_json::from_reader(reader)
            .map_err(|e| InferenceError::SerializationError(e.to_string()))?;
        Ok(checkpoint)
    }

    /// Save checkpoint to file (MessagePack format - more compact)
    #[cfg(feature = "msgpack")]
    pub fn save_msgpack<P: AsRef<Path>>(&self, path: P) -> InferenceResult<()> {
        let file =
            File::create(path).map_err(|e| InferenceError::SerializationError(e.to_string()))?;
        let mut writer = BufWriter::new(file);
        rmp_serde::encode::write(&mut writer, self)
            .map_err(|e| InferenceError::SerializationError(e.to_string()))?;
        Ok(())
    }

    /// Load checkpoint from MessagePack file
    #[cfg(feature = "msgpack")]
    pub fn load_msgpack<P: AsRef<Path>>(path: P) -> InferenceResult<Self> {
        let file =
            File::open(path).map_err(|e| InferenceError::SerializationError(e.to_string()))?;
        let reader = BufReader::new(file);
        let checkpoint = rmp_serde::from_read(reader)
            .map_err(|e| InferenceError::SerializationError(e.to_string()))?;
        Ok(checkpoint)
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> InferenceResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| InferenceError::SerializationError(e.to_string()))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> InferenceResult<Self> {
        serde_json::from_slice(bytes).map_err(|e| InferenceError::SerializationError(e.to_string()))
    }
}

/// Manager for checkpoint snapshots (for rollback)
#[derive(Debug)]
pub struct CheckpointManager {
    /// Maximum number of checkpoints to keep
    max_checkpoints: usize,
    /// Stack of checkpoints (most recent first)
    checkpoints: VecDeque<Checkpoint>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(max_checkpoints: usize) -> Self {
        Self {
            max_checkpoints,
            checkpoints: VecDeque::new(),
        }
    }

    /// Save a checkpoint
    pub fn save(&mut self, checkpoint: Checkpoint) {
        if self.checkpoints.len() >= self.max_checkpoints {
            self.checkpoints.pop_back();
        }
        self.checkpoints.push_front(checkpoint);
    }

    /// Get the most recent checkpoint
    pub fn latest(&self) -> Option<&Checkpoint> {
        self.checkpoints.front()
    }

    /// Rollback to previous checkpoint
    pub fn rollback(&mut self) -> Option<Checkpoint> {
        self.checkpoints.pop_front()
    }

    /// Get checkpoint at index (0 = most recent)
    pub fn get(&self, index: usize) -> Option<&Checkpoint> {
        self.checkpoints.get(index)
    }

    /// Number of stored checkpoints
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Check if manager is empty
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }

    /// Clear all checkpoints
    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation() {
        let config = ContextConfig::new().num_layers(2).store_history(true);
        let mut context = InferenceContext::new(config);

        context.push(Array1::from_vec(vec![1.0, 2.0]));
        context.push(Array1::from_vec(vec![3.0, 4.0]));

        let checkpoint = Checkpoint::from_context(&context);
        assert_eq!(checkpoint.version(), CHECKPOINT_VERSION);
        assert_eq!(checkpoint.states.len(), 2);
        assert_eq!(checkpoint.history.len(), 2);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut config = ContextConfig::new();
        config.num_layers = 2;
        config.hidden_dim = 4;
        config.store_history = true;
        let mut context = InferenceContext::new(config);

        context.push(Array1::from_vec(vec![1.0, 2.0]));
        context.push(Array1::from_vec(vec![3.0, 4.0]));

        let checkpoint = Checkpoint::from_context(&context);
        let restored = checkpoint.to_context().unwrap();

        assert_eq!(restored.step_count(), context.step_count());
        assert_eq!(restored.states().len(), context.states().len());
    }

    #[test]
    fn test_checkpoint_serialization() {
        let config = ContextConfig::new().num_layers(2).store_history(true);
        let mut context = InferenceContext::new(config);

        context.push(Array1::from_vec(vec![1.0]));

        let checkpoint = Checkpoint::from_context(&context);
        let bytes = checkpoint.to_bytes().unwrap();
        let restored = Checkpoint::from_bytes(&bytes).unwrap();

        assert_eq!(restored.version(), checkpoint.version());
        assert_eq!(restored.states.len(), checkpoint.states.len());
    }

    #[test]
    fn test_checkpoint_metadata() {
        let config = ContextConfig::new().num_layers(1);
        let context = InferenceContext::new(config);

        let checkpoint = Checkpoint::from_context(&context)
            .with_description("Test checkpoint".to_string())
            .with_model_id("test-model".to_string())
            .with_tag("v1".to_string());

        assert_eq!(checkpoint.metadata().description, "Test checkpoint");
        assert_eq!(checkpoint.metadata().model_id, "test-model");
        assert_eq!(checkpoint.metadata().tags, vec!["v1"]);
    }

    #[test]
    fn test_checkpoint_manager() {
        let mut manager = CheckpointManager::new(3);
        let config = ContextConfig::new().num_layers(1).store_history(true);

        // Save 5 checkpoints
        for i in 0..5 {
            let mut context = InferenceContext::new(config.clone());
            context.push(Array1::from_vec(vec![i as f32]));
            let checkpoint = Checkpoint::from_context(&context);
            manager.save(checkpoint);
        }

        // Should only keep last 3
        assert_eq!(manager.len(), 3);

        // Most recent should be step 4
        let latest = manager.latest().unwrap();
        assert_eq!(latest.history[0][0], 4.0);
    }

    #[test]
    fn test_checkpoint_rollback() {
        let mut manager = CheckpointManager::new(5);
        let config = ContextConfig::new().num_layers(1).store_history(true);

        for i in 0..3 {
            let mut context = InferenceContext::new(config.clone());
            context.push(Array1::from_vec(vec![i as f32]));
            manager.save(Checkpoint::from_context(&context));
        }

        assert_eq!(manager.len(), 3);

        let rolled_back = manager.rollback().unwrap();
        assert_eq!(rolled_back.history[0][0], 2.0);
        assert_eq!(manager.len(), 2);
    }

    #[test]
    fn test_checkpoint_file_io() {
        use std::env;

        let config = ContextConfig::new().num_layers(2).store_history(true);
        let mut context = InferenceContext::new(config);

        context.push(Array1::from_vec(vec![1.0, 2.0]));
        context.push(Array1::from_vec(vec![3.0, 4.0]));

        let checkpoint =
            Checkpoint::from_context(&context).with_description("Test save/load".to_string());

        // Save to temporary file
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("test_checkpoint.json");

        checkpoint.save_json(&path).unwrap();

        // Load back
        let loaded = Checkpoint::load_json(&path).unwrap();
        assert_eq!(loaded.metadata().description, "Test save/load");
        assert_eq!(loaded.states.len(), 2);
        assert_eq!(loaded.history.len(), 2);

        // Cleanup
        std::fs::remove_file(path).ok();
    }
}
