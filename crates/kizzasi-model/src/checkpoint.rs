//! Checkpointing and Training Utilities
//!
//! Provides utilities for:
//! - Model checkpointing and restoration
//! - Early stopping
//! - Training validation
//! - Gradient checkpointing for memory efficiency
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::checkpoint::{CheckpointManager, EarlyStopping};
//!
//! let mut manager = CheckpointManager::new("checkpoints/model");
//! let mut early_stopping = EarlyStopping::new(patience=5, min_delta=0.001);
//!
//! for epoch in 0..num_epochs {
//!     let val_loss = train_epoch(&mut model);
//!
//!     manager.save_checkpoint(epoch, &model, val_loss)?;
//!     if early_stopping.should_stop(val_loss) {
//!         break;
//!     }
//! }
//! ```

use crate::error::{ModelError, ModelResult};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Epoch number
    pub epoch: usize,
    /// Global step
    pub step: usize,
    /// Validation loss
    pub val_loss: Option<f32>,
    /// Training loss
    pub train_loss: Option<f32>,
    /// Best validation loss so far
    pub best_val_loss: Option<f32>,
    /// Timestamp
    pub timestamp: String,
    /// Model type
    pub model_type: String,
    /// Additional metrics
    pub metrics: std::collections::HashMap<String, f32>,
}

impl CheckpointMetadata {
    /// Create new checkpoint metadata
    pub fn new(epoch: usize, step: usize) -> Self {
        Self {
            epoch,
            step,
            val_loss: None,
            train_loss: None,
            best_val_loss: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
            model_type: "unknown".to_string(),
            metrics: std::collections::HashMap::new(),
        }
    }

    /// Set validation loss
    pub fn with_val_loss(mut self, loss: f32) -> Self {
        self.val_loss = Some(loss);
        self
    }

    /// Set training loss
    pub fn with_train_loss(mut self, loss: f32) -> Self {
        self.train_loss = Some(loss);
        self
    }

    /// Add custom metric
    pub fn with_metric(mut self, name: String, value: f32) -> Self {
        self.metrics.insert(name, value);
        self
    }
}

/// Checkpoint manager
#[derive(Debug, Clone)]
pub struct CheckpointManager {
    /// Base directory for checkpoints
    pub checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    /// Keep best N checkpoints
    pub keep_best: usize,
    /// Best checkpoint metadata
    pub best_checkpoint: Option<CheckpointMetadata>,
    /// Recent checkpoints
    recent_checkpoints: VecDeque<PathBuf>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.as_ref().to_path_buf(),
            max_checkpoints: 5,
            keep_best: 3,
            best_checkpoint: None,
            recent_checkpoints: VecDeque::new(),
        }
    }

    /// Set maximum number of checkpoints
    pub fn max_checkpoints(mut self, max: usize) -> Self {
        self.max_checkpoints = max;
        self
    }

    /// Set number of best checkpoints to keep
    pub fn keep_best(mut self, n: usize) -> Self {
        self.keep_best = n;
        self
    }

    /// Generate checkpoint path
    pub fn checkpoint_path(&self, epoch: usize) -> PathBuf {
        self.checkpoint_dir
            .join(format!("checkpoint_epoch_{}.bin", epoch))
    }

    /// Get best checkpoint path
    pub fn best_checkpoint_path(&self) -> PathBuf {
        self.checkpoint_dir.join("best_checkpoint.bin")
    }

    /// Save checkpoint metadata
    pub fn save_metadata(&self, metadata: &CheckpointMetadata) -> ModelResult<()> {
        let path = self
            .checkpoint_dir
            .join(format!("metadata_epoch_{}.json", metadata.epoch));

        std::fs::create_dir_all(&self.checkpoint_dir).map_err(|e| {
            ModelError::load_error(
                "checkpoint save",
                format!("Failed to create checkpoint directory: {}", e),
            )
        })?;

        let json = serde_json::to_string_pretty(metadata).map_err(|e| {
            ModelError::load_error(
                "checkpoint save",
                format!("Failed to serialize metadata: {}", e),
            )
        })?;

        std::fs::write(&path, json).map_err(|e| {
            ModelError::load_error(
                "checkpoint save",
                format!("Failed to write metadata: {}", e),
            )
        })?;

        tracing::info!("Saved checkpoint metadata to {:?}", path);
        Ok(())
    }

    /// Load checkpoint metadata
    pub fn load_metadata(&self, epoch: usize) -> ModelResult<CheckpointMetadata> {
        let path = self
            .checkpoint_dir
            .join(format!("metadata_epoch_{}.json", epoch));

        let json = std::fs::read_to_string(&path).map_err(|e| {
            ModelError::load_error("checkpoint load", format!("Failed to read metadata: {}", e))
        })?;

        let metadata: CheckpointMetadata = serde_json::from_str(&json).map_err(|e| {
            ModelError::load_error(
                "checkpoint load",
                format!("Failed to deserialize metadata: {}", e),
            )
        })?;

        Ok(metadata)
    }

    /// Check if should save based on validation loss
    pub fn is_best(&self, val_loss: f32) -> bool {
        if let Some(ref best) = self.best_checkpoint {
            if let Some(best_loss) = best.val_loss {
                return val_loss < best_loss;
            }
        }
        true
    }

    /// Update best checkpoint
    pub fn update_best(&mut self, metadata: CheckpointMetadata) {
        if let Some(val_loss) = metadata.val_loss {
            if self.is_best(val_loss) {
                self.best_checkpoint = Some(metadata);
                tracing::info!("New best checkpoint with val_loss: {}", val_loss);
            }
        }
    }

    /// Cleanup old checkpoints
    pub fn cleanup_old_checkpoints(&mut self) -> ModelResult<()> {
        while self.recent_checkpoints.len() > self.max_checkpoints {
            if let Some(old_checkpoint) = self.recent_checkpoints.pop_front() {
                if old_checkpoint.exists() {
                    std::fs::remove_file(&old_checkpoint).map_err(|e| {
                        ModelError::load_error(
                            "checkpoint cleanup",
                            format!("Failed to remove old checkpoint: {}", e),
                        )
                    })?;
                    tracing::debug!("Removed old checkpoint: {:?}", old_checkpoint);
                }
            }
        }
        Ok(())
    }
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStopping {
    /// Number of epochs to wait for improvement
    pub patience: usize,
    /// Minimum change to qualify as improvement
    pub min_delta: f32,
    /// Current patience counter
    counter: usize,
    /// Best loss seen so far
    best_loss: Option<f32>,
    /// Whether to stop training
    stopped: bool,
}

impl EarlyStopping {
    /// Create new early stopping
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            counter: 0,
            best_loss: None,
            stopped: false,
        }
    }

    /// Check if training should stop
    pub fn should_stop(&mut self, current_loss: f32) -> bool {
        if self.stopped {
            return true;
        }

        if let Some(best_loss) = self.best_loss {
            if current_loss < best_loss - self.min_delta {
                // Improved
                self.best_loss = Some(current_loss);
                self.counter = 0;
                tracing::info!(
                    "Validation improved: {:.6} -> {:.6}",
                    best_loss,
                    current_loss
                );
            } else {
                // No improvement
                self.counter += 1;
                tracing::debug!(
                    "No improvement for {} epochs (best: {:.6}, current: {:.6})",
                    self.counter,
                    best_loss,
                    current_loss
                );

                if self.counter >= self.patience {
                    self.stopped = true;
                    tracing::info!(
                        "Early stopping triggered after {} epochs without improvement",
                        self.patience
                    );
                    return true;
                }
            }
        } else {
            // First validation
            self.best_loss = Some(current_loss);
            tracing::info!("Initial validation loss: {:.6}", current_loss);
        }

        false
    }

    /// Reset early stopping
    pub fn reset(&mut self) {
        self.counter = 0;
        self.best_loss = None;
        self.stopped = false;
    }

    /// Check if stopped
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Get best loss
    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }
}

/// Validation metrics tracker
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    /// Metric history
    history: std::collections::HashMap<String, Vec<f32>>,
    /// Window size for moving average
    window_size: usize,
}

impl ValidationMetrics {
    /// Create new validation metrics tracker
    pub fn new() -> Self {
        Self {
            history: std::collections::HashMap::new(),
            window_size: 10,
        }
    }

    /// Set window size for moving average
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Record a metric
    pub fn record(&mut self, name: &str, value: f32) {
        self.history
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    /// Get metric history
    pub fn get_history(&self, name: &str) -> Option<&Vec<f32>> {
        self.history.get(name)
    }

    /// Get latest value
    pub fn get_latest(&self, name: &str) -> Option<f32> {
        self.history.get(name).and_then(|h| h.last().copied())
    }

    /// Get moving average
    pub fn get_moving_avg(&self, name: &str) -> Option<f32> {
        self.history.get(name).and_then(|history| {
            if history.is_empty() {
                return None;
            }

            let start = history.len().saturating_sub(self.window_size);
            let window = &history[start..];
            let sum: f32 = window.iter().sum();
            Some(sum / window.len() as f32)
        })
    }

    /// Check if metric is improving
    pub fn is_improving(&self, name: &str, min_delta: f32) -> bool {
        if let Some(history) = self.history.get(name) {
            if history.len() < 2 {
                return true;
            }

            let current = history[history.len() - 1];
            let previous = history[history.len() - 2];
            current < previous - min_delta
        } else {
            false
        }
    }

    /// Get best value
    pub fn get_best(&self, name: &str) -> Option<f32> {
        self.history.get(name).and_then(|h| {
            h.iter()
                .copied()
                .fold(None, |min, x| Some(min.map_or(x, |m: f32| m.min(x))))
        })
    }

    /// Print summary
    pub fn print_summary(&self) {
        tracing::info!("=== Validation Metrics Summary ===");
        for (name, history) in &self.history {
            if let (Some(latest), Some(best)) = (history.last(), self.get_best(name)) {
                let avg = self.get_moving_avg(name).unwrap_or(0.0);
                tracing::info!(
                    "{}: latest={:.6}, best={:.6}, avg={:.6}",
                    name,
                    latest,
                    best,
                    avg
                );
            }
        }
    }
}

impl Default for ValidationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Gradient checkpointing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientCheckpointConfig {
    /// Enable gradient checkpointing
    pub enabled: bool,
    /// Number of checkpoint segments
    pub num_segments: usize,
    /// Use CPU offloading
    pub cpu_offload: bool,
}

impl Default for GradientCheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_segments: 4,
            cpu_offload: false,
        }
    }
}

impl GradientCheckpointConfig {
    /// Create new gradient checkpoint config
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable gradient checkpointing
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Set number of segments
    pub fn segments(mut self, n: usize) -> Self {
        self.num_segments = n;
        self
    }

    /// Enable CPU offloading
    pub fn cpu_offload(mut self, enable: bool) -> Self {
        self.cpu_offload = enable;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_metadata() {
        let meta = CheckpointMetadata::new(5, 1000)
            .with_val_loss(0.5)
            .with_train_loss(0.6)
            .with_metric("accuracy".to_string(), 0.95);

        assert_eq!(meta.epoch, 5);
        assert_eq!(meta.step, 1000);
        assert_eq!(meta.val_loss, Some(0.5));
        assert_eq!(meta.train_loss, Some(0.6));
        assert_eq!(meta.metrics.get("accuracy"), Some(&0.95));
    }

    #[test]
    fn test_early_stopping() {
        let mut early_stop = EarlyStopping::new(3, 0.01);

        assert!(!early_stop.should_stop(1.0)); // First validation
        assert!(!early_stop.should_stop(0.9)); // Improvement
        assert!(!early_stop.should_stop(0.85)); // Improvement
        assert!(!early_stop.should_stop(0.851)); // No improvement (counter=1)
        assert!(!early_stop.should_stop(0.852)); // No improvement (counter=2)
        assert!(early_stop.should_stop(0.853)); // Trigger early stopping (counter=3)
    }

    #[test]
    fn test_early_stopping_reset() {
        let mut early_stop = EarlyStopping::new(2, 0.01);

        early_stop.should_stop(1.0);
        early_stop.should_stop(1.0);
        early_stop.should_stop(1.0);

        assert!(early_stop.is_stopped());

        early_stop.reset();
        assert!(!early_stop.is_stopped());
        assert_eq!(early_stop.best_loss(), None);
    }

    #[test]
    fn test_validation_metrics() {
        let mut metrics = ValidationMetrics::new().window_size(3);

        metrics.record("loss", 1.0);
        metrics.record("loss", 0.9);
        metrics.record("loss", 0.8);
        metrics.record("loss", 0.7);

        assert_eq!(metrics.get_latest("loss"), Some(0.7));
        assert_eq!(metrics.get_best("loss"), Some(0.7));

        let avg = metrics
            .get_moving_avg("loss")
            .expect("Failed to get moving avg");
        assert!((avg - 0.8).abs() < 1e-6); // (0.9 + 0.8 + 0.7) / 3
    }

    #[test]
    fn test_validation_metrics_improving() {
        let mut metrics = ValidationMetrics::new();

        metrics.record("loss", 1.0);
        metrics.record("loss", 0.8);

        assert!(metrics.is_improving("loss", 0.1));

        metrics.record("loss", 0.79);
        assert!(!metrics.is_improving("loss", 0.1)); // Change < min_delta
    }

    #[test]
    fn test_checkpoint_manager_is_best() {
        let mut manager = CheckpointManager::new("/tmp/checkpoints");

        assert!(manager.is_best(1.0));

        let meta = CheckpointMetadata::new(1, 100).with_val_loss(1.0);
        manager.update_best(meta);

        assert!(!manager.is_best(1.1));
        assert!(manager.is_best(0.9));
    }

    #[test]
    fn test_gradient_checkpoint_config() {
        let config = GradientCheckpointConfig::new()
            .enable()
            .segments(8)
            .cpu_offload(true);

        assert!(config.enabled);
        assert_eq!(config.num_segments, 8);
        assert!(config.cpu_offload);
    }
}
