//! Checkpoint persistence for the training loop
//!
//! Split out of [`training_loop`](super::training_loop) to keep both modules
//! well under the 2000-line ceiling.
//!
//! - [`CheckpointMetadata`] — serialisable training state written next to the weights
//! - `impl Trainer` — [`Trainer::save_checkpoint`] / [`Trainer::load_checkpoint`]
//!   and the auto/best convenience wrappers

use crate::config::KizzasiConfig;
use crate::error::{CoreError, CoreResult};
use crate::metrics::{MetricsLogger, TrainingMetrics};
use crate::training_core::{TrainableSSM, TrainingConfig};
use crate::training_loop::{Trainer, DEFAULT_STEPS_PER_EPOCH};
use serde::{Deserialize, Serialize};

impl Trainer {
    /// Filename of the optimizer-state sidecar for a checkpoint `name`
    fn optimizer_state_filename(name: &str) -> String {
        format!("{}.optim.safetensors", name)
    }

    /// Save checkpoint to disk
    ///
    /// Writes three files:
    ///
    /// | File | Contents |
    /// |------|----------|
    /// | `<name>.safetensors` | model weights |
    /// | `<name>.optim.safetensors` | AdamW first/second moments and step counter |
    /// | `<name>.json` | [`CheckpointMetadata`]: config, metrics, step/epoch |
    ///
    /// Persisting the optimizer moments is what lets [`Trainer::load_checkpoint`]
    /// resume without the loss spike that a zero-initialised Adam state causes.
    ///
    /// # Arguments
    /// * `path` - Directory to save checkpoint files
    /// * `name` - Checkpoint name (without extension)
    ///
    /// # Example
    /// ```rust,ignore
    /// trainer.save_checkpoint("checkpoints", "epoch_10")?;
    /// ```
    pub fn save_checkpoint<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        name: &str,
    ) -> CoreResult<()> {
        use std::fs;
        use std::path::PathBuf;

        let checkpoint_dir = path.as_ref();
        fs::create_dir_all(checkpoint_dir).map_err(|e| {
            CoreError::Generic(format!("Failed to create checkpoint directory: {}", e))
        })?;

        // Save model weights to safetensors
        let weights_path: PathBuf = checkpoint_dir.join(format!("{}.safetensors", name));
        self.model
            .save_weights(&weights_path)
            .map_err(|e| CoreError::Generic(format!("Failed to save model weights: {}", e)))?;

        // Save optimizer moments so a resume continues Adam rather than
        // restarting it.
        let optimizer_path: PathBuf = checkpoint_dir.join(Self::optimizer_state_filename(name));
        self.optimizer
            .save_state(&optimizer_path)
            .map_err(|e| CoreError::Generic(format!("Failed to save optimizer state: {}", e)))?;

        // Create checkpoint metadata
        let metadata = CheckpointMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            current_step: self.current_step,
            current_epoch: self.metrics.summary().total_epochs,
            config: self.config.clone(),
            metrics: self.metrics.clone(),
            optimizer_state_saved: true,
        };

        // Save metadata to JSON
        let metadata_path: PathBuf = checkpoint_dir.join(format!("{}.json", name));
        let metadata_json = serde_json::to_string_pretty(&metadata).map_err(|e| {
            CoreError::Generic(format!("Failed to serialize checkpoint metadata: {}", e))
        })?;

        fs::write(&metadata_path, metadata_json).map_err(|e| {
            CoreError::Generic(format!("Failed to write checkpoint metadata: {}", e))
        })?;

        tracing::info!(
            "Checkpoint saved: weights={}, optimizer={}, metadata={}",
            weights_path.display(),
            optimizer_path.display(),
            metadata_path.display()
        );

        Ok(())
    }

    /// Load checkpoint and resume training
    ///
    /// Creates a new Trainer from a saved checkpoint, restoring model weights,
    /// **optimizer moments**, configuration, and training state.
    ///
    /// A checkpoint written before optimizer state was persisted
    /// (`optimizer_state_saved == false`) loads with a fresh Adam state and
    /// logs a warning about the resulting update discontinuity. If the metadata
    /// claims optimizer state was saved but the sidecar is unreadable, the load
    /// fails rather than resuming from a silently zeroed optimizer.
    ///
    /// # Arguments
    /// * `path` - Directory containing checkpoint files
    /// * `name` - Checkpoint name (without extension)
    /// * `model_config` - Model configuration (must match saved model)
    ///
    /// # Example
    /// ```rust,ignore
    /// let trainer = Trainer::load_checkpoint("checkpoints", "epoch_10", model_config)?;
    /// // Continue training from epoch 10
    /// ```
    pub fn load_checkpoint<P: AsRef<std::path::Path>>(
        path: P,
        name: &str,
        model_config: KizzasiConfig,
    ) -> CoreResult<Self> {
        use std::fs;
        use std::path::PathBuf;

        let checkpoint_dir = path.as_ref();

        // Load metadata from JSON
        let metadata_path: PathBuf = checkpoint_dir.join(format!("{}.json", name));
        let metadata_json = fs::read_to_string(&metadata_path).map_err(|e| {
            CoreError::Generic(format!("Failed to read checkpoint metadata: {}", e))
        })?;

        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json).map_err(|e| {
            CoreError::Generic(format!("Failed to parse checkpoint metadata: {}", e))
        })?;

        // Load model weights
        let weights_path: PathBuf = checkpoint_dir.join(format!("{}.safetensors", name));
        let mut model = TrainableSSM::new(model_config, metadata.config.clone())?;
        model
            .load_weights(&weights_path)
            .map_err(|e| CoreError::Generic(format!("Failed to load model weights: {}", e)))?;

        // Create trainer with loaded state
        let mut optimizer = model.create_optimizer()?;

        if metadata.optimizer_state_saved {
            let optimizer_path: PathBuf = checkpoint_dir.join(Self::optimizer_state_filename(name));
            optimizer.load_state(&optimizer_path).map_err(|e| {
                CoreError::Generic(format!(
                    "Checkpoint metadata declares saved optimizer state but {} could not be \
                     restored: {}",
                    optimizer_path.display(),
                    e
                ))
            })?;
        } else {
            tracing::warn!(
                "Checkpoint '{}' predates optimizer-state persistence; AdamW moment estimates \
                 restart from zero, so expect a transient loss spike after resuming",
                name
            );
        }

        let total_steps = (metadata.config.epochs * DEFAULT_STEPS_PER_EPOCH).max(1);
        let scheduler = Self::create_scheduler(&metadata.config, total_steps);

        let logger = MetricsLogger::new()
            .with_verbose(metadata.config.track_metrics)
            .with_log_interval(metadata.config.log_interval);

        tracing::info!(
            "Checkpoint loaded: version={}, step={}, epoch={}",
            metadata.version,
            metadata.current_step,
            metadata.current_epoch
        );

        Ok(Self {
            model,
            optimizer,
            config: metadata.config,
            scheduler,
            metrics: metadata.metrics,
            logger,
            current_step: metadata.current_step,
            total_steps,
        })
    }

    /// Save checkpoint with automatic naming (epoch-based)
    ///
    /// Convenience method that automatically names checkpoints based on current epoch.
    ///
    /// # Example
    /// ```rust,ignore
    /// trainer.save_checkpoint_auto("checkpoints")?;
    /// // Creates: checkpoints/checkpoint_epoch_5.safetensors, etc.
    /// ```
    pub fn save_checkpoint_auto<P: AsRef<std::path::Path>>(&self, path: P) -> CoreResult<()> {
        let current_epoch = self.metrics.summary().total_epochs;
        let name = format!("checkpoint_epoch_{}", current_epoch);
        self.save_checkpoint(path, &name)
    }

    /// Save checkpoint if this is the best epoch (lowest validation loss)
    ///
    /// Automatically saves a "best" checkpoint when validation loss improves.
    ///
    /// # Example
    /// ```rust,ignore
    /// // After each validation epoch
    /// trainer.save_best_checkpoint("checkpoints")?;
    /// ```
    pub fn save_best_checkpoint<P: AsRef<std::path::Path>>(&self, path: P) -> CoreResult<()> {
        let summary = self.metrics.summary();

        // Only save if this is the best epoch
        // Note: total_epochs is 1-indexed (count), best_epoch is 0-indexed (epoch number)
        if let (Some(best_epoch), Some(_best_loss)) = (summary.best_epoch, summary.best_val_loss) {
            // Current epoch is total_epochs - 1 (convert from count to 0-indexed)
            let current_epoch = summary.total_epochs.saturating_sub(1);
            if current_epoch == best_epoch {
                tracing::info!("New best validation loss! Saving best checkpoint");
                return self.save_checkpoint(path, "best");
            }
        }

        Ok(())
    }
}

/// Checkpoint metadata for training state persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Package version when checkpoint was created
    pub version: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Current training step
    pub current_step: usize,
    /// Current epoch number
    pub current_epoch: usize,
    /// Training configuration
    pub config: TrainingConfig,
    /// Training metrics history
    pub metrics: TrainingMetrics,
    /// Whether a `<name>.optim.safetensors` sidecar accompanies this checkpoint.
    ///
    /// `#[serde(default)]` (i.e. `false`) for checkpoints written before
    /// optimizer state was persisted, which is exactly the signal
    /// [`Trainer::load_checkpoint`] uses to warn about the resume discontinuity
    /// instead of failing on the missing file.
    #[serde(default)]
    pub optimizer_state_saved: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resume_continues_optimizer_state() {
        // Regression: `save_checkpoint` documented saving optimizer state but
        // only wrote weights, so resuming restarted AdamW's moments from zero
        // and produced the classic post-resume update discontinuity.
        use crate::config::KizzasiConfig;
        use crate::training_core::TrainableSSM;
        use crate::training_loop::Loss;
        use candle_core::Tensor;
        use std::env;
        use std::fs;

        let temp_dir = env::temp_dir().join("kizzasi_checkpoint_optimizer_resume_test");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let model_config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(1);

        let training_config = TrainingConfig {
            learning_rate: 1e-2,
            track_metrics: false,
            grad_clip: None,
            early_stopping_patience: None,
            ..Default::default()
        };

        let model = TrainableSSM::new(model_config.clone(), training_config.clone()).unwrap();
        let device = model.device().clone();
        let mut trainer = Trainer::new(model, training_config).unwrap();

        let inputs = Tensor::new(&[[[0.1f32, 0.2], [0.3, 0.4], [0.5, 0.6]]], &device).unwrap();
        let targets = Tensor::new(&[[[1.0f32, -1.0], [0.5, -0.5], [-0.2, 0.8]]], &device).unwrap();
        let batches = vec![(inputs, targets)];

        // Two steps, then checkpoint.
        trainer.train_epoch(&batches, Loss::mse).unwrap();
        trainer.train_epoch(&batches, Loss::mse).unwrap();
        trainer.save_checkpoint(&temp_dir, "resume").unwrap();
        assert!(temp_dir.join("resume.optim.safetensors").exists());

        // Uninterrupted third step.
        trainer.train_epoch(&batches, Loss::mse).unwrap();
        let uninterrupted = snapshot_named(&trainer);

        // Resumed third step.
        let mut resumed = Trainer::load_checkpoint(&temp_dir, "resume", model_config).unwrap();
        assert_eq!(resumed.current_step(), 2);
        resumed.train_epoch(&batches, Loss::mse).unwrap();
        let after_resume = snapshot_named(&resumed);

        assert_eq!(uninterrupted.len(), after_resume.len());
        assert!(!uninterrupted.is_empty());
        let mut compared = 0usize;
        for ((name_a, vals_a), (name_b, vals_b)) in uninterrupted.iter().zip(after_resume.iter()) {
            assert_eq!(name_a, name_b);
            for (i, (a, b)) in vals_a.iter().zip(vals_b.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "{name_a}[{i}]: resuming diverged from an uninterrupted run ({a} vs {b})"
                );
                compared += 1;
            }
        }
        assert!(compared > 0);

        fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_legacy_checkpoint_without_optimizer_state_still_loads() {
        // A checkpoint whose metadata lacks `optimizer_state_saved` must load
        // (with a warning) rather than fail on the missing sidecar.
        use crate::config::KizzasiConfig;
        use crate::training_core::TrainableSSM;
        use std::env;
        use std::fs;

        let temp_dir = env::temp_dir().join("kizzasi_checkpoint_legacy_optimizer_test");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let model_config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(1);

        let training_config = TrainingConfig::default();
        let model = TrainableSSM::new(model_config.clone(), training_config.clone()).unwrap();
        let trainer = Trainer::new(model, training_config).unwrap();
        trainer.save_checkpoint(&temp_dir, "legacy").unwrap();

        // Strip the newer metadata field and the sidecar to emulate an old checkpoint.
        let metadata_path = temp_dir.join("legacy.json");
        let raw = fs::read_to_string(&metadata_path).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&raw).unwrap();
        if let Some(map) = value.as_object_mut() {
            map.remove("optimizer_state_saved");
        }
        fs::write(
            &metadata_path,
            serde_json::to_string_pretty(&value).unwrap(),
        )
        .unwrap();
        fs::remove_file(temp_dir.join("legacy.optim.safetensors")).unwrap();

        let loaded = Trainer::load_checkpoint(&temp_dir, "legacy", model_config).unwrap();
        assert_eq!(loaded.current_step(), 0);

        fs::remove_dir_all(&temp_dir).unwrap();
    }

    /// Model parameters keyed by name and sorted, so two runs are comparable
    /// despite `VarMap`'s hash iteration order.
    fn snapshot_named(trainer: &Trainer) -> Vec<(String, Vec<f32>)> {
        let data = trainer
            .model()
            .varmap()
            .data()
            .lock()
            .expect("varmap mutex poisoned");
        let mut out: Vec<(String, Vec<f32>)> = data
            .iter()
            .map(|(name, var)| {
                (
                    name.clone(),
                    var.as_tensor()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                )
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    #[test]
    fn test_checkpoint_save_load() {
        use crate::config::KizzasiConfig;
        use crate::training_core::TrainableSSM;
        use std::env;
        use std::fs;

        let temp_dir = env::temp_dir().join("kizzasi_checkpoint_test");
        fs::create_dir_all(&temp_dir).unwrap();

        // Create a model
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let training_config = TrainingConfig {
            epochs: 5,
            learning_rate: 1e-3,
            ..Default::default()
        };

        let model = TrainableSSM::new(config.clone(), training_config.clone()).unwrap();
        let trainer = Trainer::new(model, training_config).unwrap();

        // Save checkpoint
        trainer
            .save_checkpoint(&temp_dir, "test_checkpoint")
            .unwrap();

        // Verify files exist
        assert!(temp_dir.join("test_checkpoint.safetensors").exists());
        assert!(temp_dir.join("test_checkpoint.json").exists());

        // Load checkpoint
        let loaded_trainer =
            Trainer::load_checkpoint(&temp_dir, "test_checkpoint", config).unwrap();

        // Verify loaded config matches
        assert_eq!(loaded_trainer.config.epochs, 5);
        assert_eq!(loaded_trainer.config.learning_rate, 1e-3);
        assert_eq!(loaded_trainer.current_step, 0);

        // Clean up
        fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_checkpoint_auto_save() {
        use crate::config::KizzasiConfig;
        use crate::training_core::TrainableSSM;
        use std::env;
        use std::fs;

        let temp_dir = env::temp_dir().join("kizzasi_checkpoint_auto_test");
        fs::create_dir_all(&temp_dir).unwrap();

        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let training_config = TrainingConfig::default();
        let model = TrainableSSM::new(config, training_config.clone()).unwrap();
        let mut trainer = Trainer::new(model, training_config).unwrap();

        // Record some metrics to simulate training
        trainer.metrics.record_train_loss(0, 0.5);

        // Save checkpoint with auto naming
        trainer.save_checkpoint_auto(&temp_dir).unwrap();

        // Verify file exists with auto-generated name
        assert!(temp_dir.join("checkpoint_epoch_1.safetensors").exists());
        assert!(temp_dir.join("checkpoint_epoch_1.json").exists());

        // Clean up
        fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_checkpoint_best_save() {
        use crate::config::KizzasiConfig;
        use crate::training_core::TrainableSSM;
        use std::env;
        use std::fs;

        let temp_dir = env::temp_dir().join("kizzasi_checkpoint_best_test");
        fs::create_dir_all(&temp_dir).unwrap();

        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let training_config = TrainingConfig::default();
        let model = TrainableSSM::new(config, training_config.clone()).unwrap();
        let mut trainer = Trainer::new(model, training_config).unwrap();

        // Simulate training epoch 0 (not best yet)
        trainer.metrics.record_train_loss(0, 1.2);
        trainer.metrics.record_val_loss(0, 1.0);
        trainer.save_best_checkpoint(&temp_dir).unwrap();

        // Epoch 0 is the best so far, so checkpoint should be saved
        assert!(temp_dir.join("best.safetensors").exists());
        assert!(temp_dir.join("best.json").exists());

        // Simulate training epoch 1 with worse loss (should not overwrite)
        trainer.metrics.record_train_loss(1, 0.9);
        trainer.metrics.record_val_loss(1, 1.2);

        // Remove old best to test that it doesn't get overwritten
        fs::remove_file(temp_dir.join("best.safetensors")).unwrap();
        fs::remove_file(temp_dir.join("best.json")).unwrap();

        trainer.save_best_checkpoint(&temp_dir).unwrap();
        // Should not save because epoch 1 is not the best
        assert!(!temp_dir.join("best.safetensors").exists());

        // Clean up
        fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_checkpoint_metadata() {
        use crate::config::KizzasiConfig;
        use crate::training_core::TrainableSSM;
        use std::env;
        use std::fs;

        let temp_dir = env::temp_dir().join("kizzasi_checkpoint_metadata_test");
        fs::create_dir_all(&temp_dir).unwrap();

        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let training_config = TrainingConfig::default();
        let model = TrainableSSM::new(config, training_config.clone()).unwrap();
        let mut trainer = Trainer::new(model, training_config).unwrap();

        // Add some metrics
        trainer.metrics.record_train_loss(0, 0.5);
        trainer.metrics.record_val_loss(0, 0.45);

        // Save checkpoint
        trainer.save_checkpoint(&temp_dir, "metadata_test").unwrap();

        // Load and verify metadata
        let metadata_path = temp_dir.join("metadata_test.json");
        let metadata_json = fs::read_to_string(&metadata_path).unwrap();
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json).unwrap();

        assert_eq!(metadata.version, env!("CARGO_PKG_VERSION"));
        assert!(!metadata.timestamp.is_empty());
        assert_eq!(metadata.current_step, 0);
        assert!(metadata.metrics.val_loss(0).is_some());
        assert_eq!(metadata.metrics.val_loss(0).unwrap(), 0.45);

        // Clean up
        fs::remove_dir_all(&temp_dir).unwrap();
    }
}
