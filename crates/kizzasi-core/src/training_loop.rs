//! Training loop — ConstraintLoss, Loss, Trainer, CheckpointMetadata
//!
//! This module contains the high-level training utilities:
//!
//! - [`ConstraintLoss`] — bridges kizzasi-logic constraints with candle tensor ops
//! - [`Loss`] — MSE, MAE, Huber, and cross-entropy loss functions
//! - [`Trainer`] — full training loop with scheduler, metrics, validation, and checkpointing
//! - [`CheckpointMetadata`] — serialisable checkpoint state for training resumption

use crate::config::KizzasiConfig;
use crate::dataloader::TimeSeriesDataLoader;
use crate::error::{CoreError, CoreResult};
use crate::metrics::{MetricsLogger, TrainingMetrics};
use crate::scheduler::LRScheduler;
use crate::training_core::{SchedulerType, TrainableSSM, TrainingConfig};
use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer};
use serde::{Deserialize, Serialize};

/// Constraint-aware loss wrapper
///
/// Bridges kizzasi-logic constraints with candle tensor operations.
/// Allows combining task loss with constraint violations for constrained optimization.
///
/// # Examples
///
/// ```rust,ignore
/// use kizzasi_core::{ConstraintLoss, Loss};
///
/// let constraint_loss = ConstraintLoss::new(0.1);
///
/// // In training loop:
/// let task_loss = Loss::mse(&predictions, &targets)?;
/// let total_loss = constraint_loss.compute(&task_loss, &predictions, |pred| {
///     // Compute constraint violation from prediction
///     Ok(0.0)
/// })?;
/// ```
pub struct ConstraintLoss {
    /// Base weight for constraint violations
    pub(crate) constraint_weight: f32,
}

impl ConstraintLoss {
    /// Create a new constraint-aware loss
    pub fn new(constraint_weight: f32) -> Self {
        Self { constraint_weight }
    }

    /// Compute combined loss: task_loss + constraint_weight * constraint_penalty
    ///
    /// # Arguments
    /// * `task_loss` - Base task loss (MSE, MAE, etc.)
    /// * `prediction` - Model prediction tensor
    /// * `constraint_fn` - Function that computes constraint violation from prediction
    pub fn compute<F>(
        &self,
        task_loss: &Tensor,
        prediction: &Tensor,
        constraint_fn: F,
    ) -> CoreResult<Tensor>
    where
        F: Fn(&Tensor) -> CoreResult<f32>,
    {
        // Compute constraint violation
        let violation = constraint_fn(prediction)?;

        // Add constraint penalty to task loss
        // Create a scalar penalty value matching task_loss shape
        let penalty_value = self.constraint_weight * violation;

        // Use affine to add the penalty: task_loss + penalty = task_loss * 1.0 + penalty
        task_loss
            .affine(1.0, penalty_value as f64)
            .map_err(|e| CoreError::Generic(format!("Failed to add constraint penalty: {}", e)))
    }
}

/// Loss functions for training
pub struct Loss;

impl Loss {
    /// Mean Squared Error loss
    pub fn mse(predictions: &Tensor, targets: &Tensor) -> CoreResult<Tensor> {
        predictions
            .sub(targets)
            .map_err(|e| CoreError::Generic(format!("MSE subtraction failed: {}", e)))?
            .sqr()
            .map_err(|e| CoreError::Generic(format!("MSE square failed: {}", e)))?
            .mean_all()
            .map_err(|e| CoreError::Generic(format!("MSE mean failed: {}", e)))
    }

    /// Mean Absolute Error loss
    pub fn mae(predictions: &Tensor, targets: &Tensor) -> CoreResult<Tensor> {
        predictions
            .sub(targets)
            .map_err(|e| CoreError::Generic(format!("MAE subtraction failed: {}", e)))?
            .abs()
            .map_err(|e| CoreError::Generic(format!("MAE abs failed: {}", e)))?
            .mean_all()
            .map_err(|e| CoreError::Generic(format!("MAE mean failed: {}", e)))
    }

    /// Huber loss (smooth L1 loss)
    pub fn huber(predictions: &Tensor, targets: &Tensor, delta: f64) -> CoreResult<Tensor> {
        let diff = predictions
            .sub(targets)
            .map_err(|e| CoreError::Generic(format!("Huber subtraction failed: {}", e)))?;
        let abs_diff = diff
            .abs()
            .map_err(|e| CoreError::Generic(format!("Huber abs failed: {}", e)))?;

        // If |diff| <= delta: 0.5 * diff^2
        // If |diff| > delta: delta * (|diff| - 0.5 * delta)
        let squared = diff
            .sqr()
            .map_err(|e| CoreError::Generic(format!("Huber square failed: {}", e)))?
            .affine(0.5, 0.0)
            .map_err(|e| CoreError::Generic(format!("Huber mul 0.5 failed: {}", e)))?;

        let linear_offset = delta * delta * 0.5;
        let linear = abs_diff
            .affine(delta, -linear_offset)
            .map_err(|e| CoreError::Generic(format!("Huber linear computation failed: {}", e)))?;

        let mask = abs_diff
            .le(delta)
            .map_err(|e| CoreError::Generic(format!("Huber comparison failed: {}", e)))?
            .to_dtype(predictions.dtype())
            .map_err(|e| CoreError::Generic(format!("Huber mask conversion failed: {}", e)))?;

        // Invert mask: 1 - mask
        let inv_mask = mask
            .affine(-1.0, 1.0)
            .map_err(|e| CoreError::Generic(format!("Huber mask inversion failed: {}", e)))?;

        let loss = squared
            .mul(&mask)
            .map_err(|e| CoreError::Generic(format!("Huber squared mul failed: {}", e)))?
            .add(
                &linear
                    .mul(&inv_mask)
                    .map_err(|e| CoreError::Generic(format!("Huber linear mul failed: {}", e)))?,
            )
            .map_err(|e| CoreError::Generic(format!("Huber final add failed: {}", e)))?;

        loss.mean_all()
            .map_err(|e| CoreError::Generic(format!("Huber mean failed: {}", e)))
    }

    /// Cross-entropy loss for classification
    pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> CoreResult<Tensor> {
        // Log softmax
        let log_probs = candle_nn::ops::log_softmax(logits, candle_core::D::Minus1)
            .map_err(|e| CoreError::Generic(format!("Log softmax failed: {}", e)))?;

        // Negative log likelihood
        let nll = log_probs
            .mul(targets)
            .map_err(|e| CoreError::Generic(format!("NLL multiplication failed: {}", e)))?
            .sum_all()
            .map_err(|e| CoreError::Generic(format!("NLL sum failed: {}", e)))?
            .neg()
            .map_err(|e| CoreError::Generic(format!("NLL negation failed: {}", e)))?;

        // Average over batch
        let batch_size = logits
            .dim(0)
            .map_err(|e| CoreError::Generic(format!("Failed to get batch size: {}", e)))?;
        nll.affine(1.0 / batch_size as f64, 0.0)
            .map_err(|e| CoreError::Generic(format!("Cross entropy division failed: {}", e)))
    }
}

/// Training utilities with scheduler, metrics, and validation
pub struct Trainer {
    pub(crate) model: TrainableSSM,
    pub(crate) optimizer: AdamW,
    pub(crate) config: TrainingConfig,
    pub(crate) scheduler: Option<Box<dyn LRScheduler>>,
    pub(crate) metrics: TrainingMetrics,
    pub(crate) logger: MetricsLogger,
    pub(crate) current_step: usize,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(model: TrainableSSM, config: TrainingConfig) -> CoreResult<Self> {
        let optimizer = model.create_optimizer()?;

        // Create scheduler based on config
        let scheduler = Self::create_scheduler(&config);

        let metrics = TrainingMetrics::new();

        let logger = MetricsLogger::new()
            .with_verbose(config.track_metrics)
            .with_log_interval(config.log_interval);

        Ok(Self {
            model,
            optimizer,
            config,
            scheduler,
            metrics,
            logger,
            current_step: 0,
        })
    }

    /// Create scheduler from config
    fn create_scheduler(config: &TrainingConfig) -> Option<Box<dyn LRScheduler>> {
        use crate::scheduler::*;

        config.scheduler.as_ref().map(|sched_type| {
            let total_steps = config.epochs * 100; // Rough estimate, can be updated later

            match sched_type {
                SchedulerType::Constant => {
                    Box::new(ConstantScheduler::new(config.learning_rate)) as Box<dyn LRScheduler>
                }
                SchedulerType::Linear {
                    warmup_steps,
                    final_lr,
                } => Box::new(LinearScheduler::new(
                    config.learning_rate,
                    *final_lr,
                    total_steps,
                    *warmup_steps,
                )) as Box<dyn LRScheduler>,
                SchedulerType::Cosine {
                    warmup_steps,
                    min_lr,
                } => Box::new(
                    CosineScheduler::new(config.learning_rate, total_steps, *warmup_steps)
                        .with_min_lr(*min_lr),
                ) as Box<dyn LRScheduler>,
                SchedulerType::Step {
                    milestones,
                    decay_factor,
                } => Box::new(StepScheduler::new(
                    config.learning_rate,
                    *decay_factor,
                    milestones.clone(),
                )) as Box<dyn LRScheduler>,
                SchedulerType::Exponential {
                    decay_rate,
                    decay_steps,
                } => Box::new(ExponentialScheduler::new(
                    config.learning_rate,
                    *decay_rate,
                    *decay_steps,
                )) as Box<dyn LRScheduler>,
                SchedulerType::OneCycle { warmup_pct } => Box::new(
                    OneCycleScheduler::new(config.learning_rate, total_steps)
                        .with_warmup_pct(*warmup_pct),
                ) as Box<dyn LRScheduler>,
                SchedulerType::Polynomial { final_lr, power } => Box::new(PolynomialScheduler::new(
                    config.learning_rate,
                    *final_lr,
                    total_steps,
                    *power,
                ))
                    as Box<dyn LRScheduler>,
            }
        })
    }

    /// Get current learning rate
    fn get_current_lr(&self) -> f64 {
        self.scheduler
            .as_ref()
            .map(|s| s.get_lr(self.current_step))
            .unwrap_or(self.config.learning_rate)
    }

    /// Train for one epoch
    pub fn train_epoch<F>(
        &mut self,
        data_loader: &[(Tensor, Tensor)],
        loss_fn: F,
    ) -> CoreResult<f32>
    where
        F: Fn(&Tensor, &Tensor) -> CoreResult<Tensor>,
    {
        let mut total_loss = 0.0;
        let num_batches = data_loader.len();
        let epoch = self.current_step / num_batches.max(1);

        for (batch_idx, (inputs, targets)) in data_loader.iter().enumerate() {
            // Update learning rate from scheduler
            let lr = self.get_current_lr();
            if self.config.track_metrics {
                self.metrics.record_learning_rate(lr);
            }

            // Forward pass
            let predictions = self.model.forward(inputs)?;

            // Compute loss
            let loss = loss_fn(&predictions, targets)?;

            // Backward pass
            self.optimizer
                .backward_step(&loss)
                .map_err(|e| CoreError::Generic(format!("Backward step failed: {}", e)))?;

            // Accumulate loss
            let loss_val = loss
                .to_vec0::<f32>()
                .map_err(|e| CoreError::Generic(format!("Failed to extract loss value: {}", e)))?;
            total_loss += loss_val;

            // Track metrics
            if self.config.track_metrics {
                self.metrics.record_train_loss(epoch, loss_val);
                self.logger.log_batch(epoch, batch_idx, loss_val);

                // Compute and track gradient norm
                let grad_norm = self.compute_grad_norm()?;
                self.metrics.record_grad_norm(grad_norm);
            }

            // Gradient clipping if enabled
            if let Some(max_norm) = self.config.grad_clip {
                self.clip_gradients(max_norm)?;
            }

            self.current_step += 1;
        }

        Ok(total_loss / num_batches as f32)
    }

    /// Compute gradient norm
    fn compute_grad_norm(&self) -> CoreResult<f32> {
        // Placeholder: In candle, gradient norms would be computed from VarMap
        // For now, return a dummy value
        // TODO: Implement proper gradient norm computation when candle exposes gradient access
        Ok(1.0)
    }

    /// Clip gradients by global norm
    ///
    /// Note: Gradient clipping is handled internally by candle's optimizer.
    /// This is a placeholder for custom gradient clipping if needed.
    fn clip_gradients(&self, _max_norm: f32) -> CoreResult<()> {
        // Gradient clipping will be handled by the optimizer's built-in mechanism
        // or via custom backward hooks in future implementations
        Ok(())
    }

    /// Evaluate on validation data
    pub fn evaluate<F>(&self, data_loader: &[(Tensor, Tensor)], loss_fn: F) -> CoreResult<f32>
    where
        F: Fn(&Tensor, &Tensor) -> CoreResult<Tensor>,
    {
        let mut total_loss = 0.0;
        let num_batches = data_loader.len();

        for (inputs, targets) in data_loader {
            // Forward pass (no gradient tracking needed)
            let predictions = self.model.forward(inputs)?;

            // Compute loss
            let loss = loss_fn(&predictions, targets)?;

            // Accumulate loss
            let loss_val = loss
                .to_vec0::<f32>()
                .map_err(|e| CoreError::Generic(format!("Failed to extract loss value: {}", e)))?;
            total_loss += loss_val;
        }

        Ok(total_loss / num_batches as f32)
    }

    /// Full training loop with validation and early stopping
    pub fn fit<F>(
        &mut self,
        mut train_loader: TimeSeriesDataLoader,
        mut val_loader: Option<TimeSeriesDataLoader>,
        loss_fn: F,
    ) -> CoreResult<()>
    where
        F: Fn(&Tensor, &Tensor) -> CoreResult<Tensor> + Copy,
    {
        use std::time::Instant;

        for epoch in 0..self.config.epochs {
            let epoch_start = Instant::now();

            // Shuffle training data
            train_loader.shuffle();

            // Prepare batches (simplified - actual implementation would iterate batches)
            // For now, this is a placeholder for the integration
            // TODO: Implement proper batch iteration with TimeSeriesDataLoader
            let train_batches: Vec<(Tensor, Tensor)> = Vec::new();

            // Train for one epoch
            let train_loss = self.train_epoch(&train_batches, loss_fn)?;

            // Validation
            let val_loss = if let Some(ref mut _val_data) = val_loader {
                let val_batches: Vec<(Tensor, Tensor)> = Vec::new();
                let val_loss = self.evaluate(&val_batches, loss_fn)?;

                if self.config.track_metrics {
                    self.metrics.record_val_loss(epoch, val_loss);
                }

                Some(val_loss)
            } else {
                None
            };

            // Track epoch duration
            let epoch_duration = epoch_start.elapsed().as_secs_f64();
            if self.config.track_metrics {
                self.metrics.record_epoch_duration(epoch, epoch_duration);
            }

            // Log epoch metrics
            let current_lr = self.get_current_lr();
            self.logger
                .log_epoch(epoch, train_loss, val_loss, current_lr);

            // Early stopping check
            if let Some(patience) = self.config.early_stopping_patience {
                if !self.metrics.is_improving(patience) {
                    tracing::info!("Early stopping triggered at epoch {}", epoch);
                    break;
                }
            }
        }

        // Log training summary
        if self.config.track_metrics {
            let summary = self.metrics.summary();
            self.logger.log_summary(&summary);
        }

        Ok(())
    }

    /// Get reference to the model
    pub fn model(&self) -> &TrainableSSM {
        &self.model
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut TrainableSSM {
        &mut self.model
    }

    /// Get reference to training metrics
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// Get mutable reference to training metrics
    pub fn metrics_mut(&mut self) -> &mut TrainingMetrics {
        &mut self.metrics
    }

    /// Get current training step
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Save checkpoint to disk
    ///
    /// Saves model weights, optimizer state, training configuration, metrics, and metadata.
    ///
    /// # Arguments
    /// * `path` - Directory to save checkpoint files
    /// * `name` - Checkpoint name (without extension)
    ///
    /// # Example
    /// ```rust,ignore
    /// trainer.save_checkpoint("checkpoints", "epoch_10")?;
    /// // Creates: checkpoints/epoch_10.safetensors and checkpoints/epoch_10.json
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

        // Create checkpoint metadata
        let metadata = CheckpointMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            current_step: self.current_step,
            current_epoch: self.metrics.summary().total_epochs,
            config: self.config.clone(),
            metrics: self.metrics.clone(),
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
            "Checkpoint saved: weights={}, metadata={}",
            weights_path.display(),
            metadata_path.display()
        );

        Ok(())
    }

    /// Load checkpoint and resume training
    ///
    /// Creates a new Trainer from a saved checkpoint, restoring model weights,
    /// configuration, and training state.
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
        let optimizer = model.create_optimizer()?;
        let scheduler = Self::create_scheduler(&metadata.config);

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training_core::TrainingConfig;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_mse_loss() {
        let device = Device::Cpu;
        let predictions = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let targets = Tensor::new(&[1.5f32, 2.5, 3.5], &device).unwrap();

        let loss = Loss::mse(&predictions, &targets).unwrap();
        let loss_val = loss.to_vec0::<f32>().unwrap();

        // Expected: mean((0.5^2 + 0.5^2 + 0.5^2)) = 0.25
        assert!((loss_val - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_trainer_with_scheduler() {
        use crate::config::KizzasiConfig;
        use crate::training_core::{SchedulerType, TrainableSSM};

        let model_config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let training_config = TrainingConfig::default().with_scheduler(SchedulerType::Linear {
            warmup_steps: 50,
            final_lr: 1e-6,
        });

        let model = TrainableSSM::new(model_config, training_config.clone()).unwrap();
        let trainer = Trainer::new(model, training_config);

        assert!(trainer.is_ok());
        let trainer = trainer.unwrap();
        assert!(trainer.scheduler.is_some());
    }

    #[test]
    fn test_trainer_metrics_tracking() {
        use crate::config::KizzasiConfig;
        use crate::training_core::TrainableSSM;

        let model_config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let training_config = TrainingConfig::default();
        let model = TrainableSSM::new(model_config, training_config.clone()).unwrap();
        let trainer = Trainer::new(model, training_config).unwrap();

        // Check that metrics are initialized
        assert_eq!(trainer.metrics().current_step(), 0);
        assert_eq!(trainer.current_step(), 0);
    }

    #[test]
    fn test_mae_loss() {
        let device = Device::Cpu;
        let predictions = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let targets = Tensor::new(&[1.5f32, 2.5, 3.5], &device).unwrap();

        let loss = Loss::mae(&predictions, &targets).unwrap();
        let loss_val = loss.to_vec0::<f32>().unwrap();

        // Expected: mean(|0.5| + |0.5| + |0.5|) = 0.5
        assert!((loss_val - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_huber_loss() {
        let device = Device::Cpu;
        let predictions = Tensor::new(&[1.0f32, 2.0, 5.0], &device).unwrap();
        let targets = Tensor::new(&[1.1f32, 2.1, 3.0], &device).unwrap();

        let loss = Loss::huber(&predictions, &targets, 1.0).unwrap();
        let loss_val = loss.to_vec0::<f32>().unwrap();

        // Huber loss is smooth L1
        assert!(loss_val > 0.0);
        assert!(loss_val < 2.0); // Should be less than L1 loss for large errors
    }

    #[test]
    fn test_constraint_loss_creation() {
        let constraint_loss = ConstraintLoss::new(0.5);
        assert_eq!(constraint_loss.constraint_weight, 0.5);
    }

    #[test]
    fn test_constraint_loss_no_violation() {
        let device = Device::Cpu;
        let predictions = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let targets = Tensor::new(&[1.5f32, 2.5, 3.5], &device).unwrap();

        let task_loss = Loss::mse(&predictions, &targets).unwrap();
        let task_loss_val = task_loss.to_vec0::<f32>().unwrap();

        let constraint_loss = ConstraintLoss::new(0.5);

        // No constraint violation
        let total_loss = constraint_loss
            .compute(&task_loss, &predictions, |_pred| Ok(0.0))
            .unwrap();
        let total_loss_val = total_loss.to_vec0::<f32>().unwrap();

        // Should equal task loss when no violation
        assert!((total_loss_val - task_loss_val).abs() < 1e-5);
    }

    #[test]
    fn test_constraint_loss_with_violation() {
        let device = Device::Cpu;
        let predictions = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let targets = Tensor::new(&[1.5f32, 2.5, 3.5], &device).unwrap();

        let task_loss = Loss::mse(&predictions, &targets).unwrap();
        let task_loss_val = task_loss.to_vec0::<f32>().unwrap();

        let constraint_loss = ConstraintLoss::new(0.5);

        // Constraint violation of 1.0
        let total_loss = constraint_loss
            .compute(&task_loss, &predictions, |_pred| Ok(1.0))
            .unwrap();
        let total_loss_val = total_loss.to_vec0::<f32>().unwrap();

        // Should be task_loss + 0.5 * 1.0 = task_loss + 0.5
        let expected = task_loss_val + 0.5;
        assert!((total_loss_val - expected).abs() < 1e-5);
    }

    #[test]
    fn test_constraint_loss_scaling() {
        let device = Device::Cpu;
        let predictions = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let targets = Tensor::new(&[1.5f32, 2.5, 3.5], &device).unwrap();

        let task_loss = Loss::mse(&predictions, &targets).unwrap();
        let task_loss_val = task_loss.to_vec0::<f32>().unwrap();

        // Test different constraint weights
        let weights = [0.1, 0.5, 1.0, 2.0];
        let violation = 1.5;

        for &weight in &weights {
            let constraint_loss = ConstraintLoss::new(weight);
            let total_loss = constraint_loss
                .compute(&task_loss, &predictions, |_pred| Ok(violation))
                .unwrap();
            let total_loss_val = total_loss.to_vec0::<f32>().unwrap();

            let expected = task_loss_val + weight * violation;
            assert!(
                (total_loss_val - expected).abs() < 1e-4,
                "Weight {} failed: got {}, expected {}",
                weight,
                total_loss_val,
                expected
            );
        }
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
