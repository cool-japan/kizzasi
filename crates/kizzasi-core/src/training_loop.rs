//! Training loop — ConstraintLoss, Loss, Trainer, CheckpointMetadata
//!
//! This module contains the high-level training utilities:
//!
//! - [`ConstraintLoss`] — bridges kizzasi-logic constraints with candle tensor ops
//! - [`Loss`] — MSE, MAE, Huber, and cross-entropy loss functions
//! - [`Trainer`] — full training loop with scheduler, metrics, validation, and checkpointing
//!
//! Checkpoint persistence (`save_checkpoint` / `load_checkpoint` and
//! `CheckpointMetadata`) lives in [`training_checkpoint`](super::training_checkpoint).

use crate::dataloader::TimeSeriesDataLoader;
use crate::error::{CoreError, CoreResult};
use crate::metrics::{MetricsLogger, TrainingMetrics};
use crate::optimizer::KizzasiAdamW;
use crate::scheduler::LRScheduler;
use crate::training_core::{SchedulerType, TrainableSSM, TrainingConfig};
use candle_core::backprop::GradStore;
use candle_core::Tensor;

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
    pub(crate) optimizer: KizzasiAdamW,
    pub(crate) config: TrainingConfig,
    pub(crate) scheduler: Option<Box<dyn LRScheduler>>,
    pub(crate) metrics: TrainingMetrics,
    pub(crate) logger: MetricsLogger,
    pub(crate) current_step: usize,
    /// Horizon handed to step-based schedulers.
    ///
    /// Seeded with a placeholder (`epochs * DEFAULT_STEPS_PER_EPOCH`) because
    /// the batch count is unknown until a data loader shows up, then replaced
    /// with the true `num_batches * epochs` by [`Trainer::set_total_steps`],
    /// which [`Trainer::fit`] calls before the first epoch.
    pub(crate) total_steps: usize,
}

/// Steps-per-epoch assumed before a data loader reveals the real batch count.
pub(crate) const DEFAULT_STEPS_PER_EPOCH: usize = 100;

impl Trainer {
    /// Create a new trainer
    pub fn new(model: TrainableSSM, config: TrainingConfig) -> CoreResult<Self> {
        let optimizer = model.create_optimizer()?;

        // Create scheduler based on config. The horizon is provisional until
        // `fit` (or an explicit `set_total_steps`) supplies the real one.
        let total_steps = (config.epochs * DEFAULT_STEPS_PER_EPOCH).max(1);
        let scheduler = Self::create_scheduler(&config, total_steps);

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
            total_steps,
        })
    }

    /// Reparameterise the scheduler for a known training horizon.
    ///
    /// Step-based schedules (linear/cosine/one-cycle/polynomial) are defined
    /// over `total_steps`; feeding them a guess makes the decay end early or
    /// never finish. [`Trainer::fit`] calls this with
    /// `train_loader.num_batches() * epochs` before the first epoch, and
    /// callers driving [`Trainer::train_epoch`] directly should call it too.
    ///
    /// `total_steps` is clamped to at least 1.
    pub fn set_total_steps(&mut self, total_steps: usize) {
        let total_steps = total_steps.max(1);
        if total_steps == self.total_steps {
            return;
        }
        self.total_steps = total_steps;
        self.scheduler = Self::create_scheduler(&self.config, total_steps);
    }

    /// The horizon currently used by step-based schedulers
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Create scheduler from config for a given training horizon
    pub(crate) fn create_scheduler(
        config: &TrainingConfig,
        total_steps: usize,
    ) -> Option<Box<dyn LRScheduler>> {
        use crate::scheduler::*;

        config.scheduler.as_ref().map(|sched_type| {
            let total_steps = total_steps.max(1);

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

    /// Get the learning rate the scheduler prescribes for the current step
    pub fn get_current_lr(&self) -> f64 {
        self.scheduler
            .as_ref()
            .map(|s| s.get_lr(self.current_step))
            .unwrap_or(self.config.learning_rate)
    }

    /// The learning rate the optimizer will actually apply on its next step
    pub fn optimizer_learning_rate(&self) -> f64 {
        self.optimizer.learning_rate()
    }

    /// Extract a scalar tensor as `f32` regardless of its dtype.
    ///
    /// Mixed-precision runs produce an F16/BF16 loss; `to_vec0::<f32>` would
    /// reject those outright, so cast first.
    fn scalar_to_f32(value: &Tensor) -> CoreResult<f32> {
        value
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| CoreError::Generic(format!("Failed to cast scalar to f32: {}", e)))?
            .to_vec0::<f32>()
            .map_err(|e| CoreError::Generic(format!("Failed to extract scalar value: {}", e)))
    }

    /// Effective loss-scaling factor.
    ///
    /// A non-finite or non-positive configured value is treated as "no
    /// scaling" instead of poisoning every gradient with NaN/Inf.
    fn effective_loss_scale(&self) -> f32 {
        let scale = self.config.loss_scale;
        if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        }
    }

    /// Train for one epoch
    ///
    /// For each batch the training loop performs:
    ///
    /// 1. Forward pass through the model
    /// 2. Loss evaluation
    /// 3. Loss scaling by `config.loss_scale` (mixed-precision underflow guard)
    /// 4. Explicit backward pass to materialise a [`GradStore`]
    /// 5. Gradient unscaling — undone *before* clipping so `config.grad_clip`
    ///    keeps its configured meaning
    /// 6. Non-finite gradient check (the batch is skipped rather than stepped)
    /// 7. Optional global-norm gradient clipping (`config.grad_clip`)
    /// 8. Gradient-norm telemetry (when `config.track_metrics` is set)
    /// 9. Optimizer step against the (possibly clipped) gradients
    ///
    /// The scheduled learning rate is pushed into the optimizer at the top of
    /// every batch, independently of `config.track_metrics` — the schedule
    /// drives training, the metric is only telemetry.
    ///
    /// Performing the backward pass explicitly (rather than via
    /// [`candle_nn::Optimizer::backward_step`]) lets us inspect, unscale and clip the
    /// gradients before they are consumed by the optimizer.
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

        let loss_scale = self.effective_loss_scale();
        let scaling_enabled = loss_scale != 1.0;

        for (batch_idx, (inputs, targets)) in data_loader.iter().enumerate() {
            // Apply the scheduled learning rate to the optimizer.
            let lr = self.get_current_lr();
            self.optimizer.set_learning_rate(lr);
            if self.config.track_metrics {
                self.metrics.record_learning_rate(lr);
            }

            // Forward + backward. With gradient checkpointing the model drives
            // both halves so it can recompute activations segment by segment;
            // otherwise the graph from the plain forward pass is reused.
            let (loss_val, mut grads) = if self.config.use_gradient_checkpointing {
                self.model
                    .forward_backward_checkpointed(inputs, targets, &loss_fn, loss_scale)?
            } else {
                let predictions = self.model.forward(inputs)?;

                // Compute loss (unscaled — this is what telemetry reports)
                let loss = loss_fn(&predictions, targets)?;
                let loss_val = Self::scalar_to_f32(&loss)?;

                // Scale the loss before the backward pass so small
                // mixed-precision gradients do not flush to zero.
                let backward_root = if scaling_enabled {
                    loss.affine(loss_scale as f64, 0.0)
                        .map_err(|e| CoreError::Generic(format!("Loss scaling failed: {}", e)))?
                } else {
                    loss
                };

                // Backward pass — retain the gradient store so we can inspect,
                // unscale and clip the gradients before stepping the optimizer.
                let mut grads = backward_root
                    .backward()
                    .map_err(|e| CoreError::Generic(format!("Backward pass failed: {}", e)))?;

                // Undo the loss scaling before anything reads the gradient
                // magnitudes.
                if scaling_enabled {
                    self.scale_gradients(&mut grads, 1.0 / loss_scale as f64)?;
                }

                (loss_val, grads)
            };

            let mut grad_norm = self.compute_grad_norm(&grads)?;

            // A non-finite gradient would corrupt the optimizer's moment
            // estimates permanently; drop the batch instead.
            if !grad_norm.is_finite() {
                tracing::warn!(
                    "Skipping optimizer step for epoch {} batch {}: non-finite gradient norm ({})",
                    epoch,
                    batch_idx,
                    grad_norm
                );
                total_loss += loss_val;
                if self.config.track_metrics {
                    self.metrics.record_train_loss(epoch, loss_val);
                    self.logger.log_batch(epoch, batch_idx, loss_val);
                }
                self.current_step += 1;
                continue;
            }

            // Gradient clipping (global-norm) before metrics so that the
            // recorded value matches what is actually applied.
            if let Some(max_norm) = self.config.grad_clip {
                if max_norm.is_finite() && max_norm > 0.0 && grad_norm > max_norm {
                    self.scale_gradients(&mut grads, (max_norm / grad_norm) as f64)?;
                    grad_norm = max_norm;
                }
            }

            // Track gradient norm metric (post-clip, matches optimizer input).
            if self.config.track_metrics {
                self.metrics.record_grad_norm(grad_norm);
            }

            // Optimizer step using the (possibly clipped) gradients.
            self.optimizer
                .step(&grads)
                .map_err(|e| CoreError::Generic(format!("Optimizer step failed: {}", e)))?;

            // Accumulate loss
            total_loss += loss_val;

            // Loss / batch telemetry
            if self.config.track_metrics {
                self.metrics.record_train_loss(epoch, loss_val);
                self.logger.log_batch(epoch, batch_idx, loss_val);
            }

            self.current_step += 1;
        }

        Ok(total_loss / num_batches as f32)
    }

    /// Compute the global L2 norm of all parameter gradients.
    ///
    /// Iterates the model's [`candle_nn::VarMap`] and accumulates `||g||_2^2` for every
    /// variable that has a corresponding gradient in `grads`, then returns
    /// the square root.
    ///
    /// Variables without a gradient (e.g. detached parameters, or parameters
    /// the current loss does not depend on) are skipped — they contribute 0
    /// to the norm.
    ///
    /// Exposed for callers driving their own training loop instead of
    /// [`Trainer::train_epoch`].
    pub fn compute_grad_norm(&self, grads: &GradStore) -> CoreResult<f32> {
        let mut sum_sq: f64 = 0.0;

        for var in self.model.varmap().all_vars() {
            let grad = match grads.get(&var) {
                Some(g) => g,
                None => continue,
            };

            // Cast to f32 so the norm calculation is stable regardless of the
            // parameter dtype (e.g. F16/BF16 mixed precision).
            let grad_f32 = grad
                .to_dtype(candle_core::DType::F32)
                .map_err(|e| CoreError::Generic(format!("grad to_dtype failed: {}", e)))?;
            let local_sq = grad_f32
                .sqr()
                .map_err(|e| CoreError::Generic(format!("grad sqr failed: {}", e)))?
                .sum_all()
                .map_err(|e| CoreError::Generic(format!("grad sum_all failed: {}", e)))?
                .to_scalar::<f32>()
                .map_err(|e| CoreError::Generic(format!("grad to_scalar failed: {}", e)))?;
            sum_sq += local_sq as f64;
        }

        Ok(sum_sq.sqrt() as f32)
    }

    /// Clip gradients by their global L2 norm.
    ///
    /// If the global norm exceeds `max_norm`, every gradient tensor is scaled
    /// by `max_norm / global_norm` (matching the semantics of PyTorch's
    /// `torch.nn.utils.clip_grad_norm_`). The clipping is performed in-place
    /// on the [`GradStore`] by reinserting the scaled gradient tensors.
    ///
    /// `max_norm` is interpreted as a finite positive threshold; non-positive
    /// or non-finite values are treated as "no clipping" so that misconfigured
    /// hyperparameters do not silently zero out the gradients.
    ///
    /// [`Trainer::train_epoch`] performs the same clipping inline (reusing the
    /// norm it has already computed); this entry point is for callers driving
    /// their own loop.
    pub fn clip_gradients(&self, grads: &mut GradStore, max_norm: f32) -> CoreResult<()> {
        if !max_norm.is_finite() || max_norm <= 0.0 {
            return Ok(());
        }

        let total_norm = self.compute_grad_norm(grads)?;
        if !total_norm.is_finite() || total_norm <= max_norm {
            return Ok(());
        }

        self.scale_gradients(grads, (max_norm / total_norm) as f64)
    }

    /// Multiply every parameter gradient in `grads` by `factor` in place.
    ///
    /// Used for loss-scale removal and for global-norm clipping. Variables
    /// without a gradient are skipped.
    fn scale_gradients(&self, grads: &mut GradStore, factor: f64) -> CoreResult<()> {
        // Collect the vars first so we are not holding immutable borrows on
        // `grads` while mutating it.
        let vars = self.model.varmap().all_vars();
        for var in vars.iter() {
            let scaled = match grads.get(var) {
                Some(g) => g
                    .affine(factor, 0.0)
                    .map_err(|e| CoreError::Generic(format!("grad scale failed: {}", e)))?,
                None => continue,
            };
            grads.insert(var, scaled);
        }

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
            total_loss += Self::scalar_to_f32(&loss)?;
        }

        Ok(total_loss / num_batches as f32)
    }

    /// Materialise one epoch's worth of `(input, target)` tensors from a
    /// [`TimeSeriesDataLoader`].
    ///
    /// `iter_batches` already handles per-epoch shuffling (after the first
    /// epoch), so the caller does not need to invoke `shuffle()` manually.
    /// Each yielded ndarray batch is converted to candle tensors on the
    /// trainer's device.
    fn collect_epoch_batches(
        loader: &mut TimeSeriesDataLoader,
        device: &candle_core::Device,
    ) -> CoreResult<Vec<(Tensor, Tensor)>> {
        // Snapshot the batch count up-front; `iter_batches` would also yield
        // this many items but we pre-allocate to avoid rehashing.
        let cap = loader.num_batches();
        let mut batches: Vec<(Tensor, Tensor)> = Vec::with_capacity(cap);

        // `iter_batches` borrows `loader` mutably and shuffles internally on
        // epochs past the first. We collect the ndarray pairs first because
        // `to_tensors` only needs an immutable borrow but the iterator holds a
        // mutable one for the duration of the for-loop.
        let mut raw_batches: Vec<(
            scirs2_core::ndarray::Array2<f32>,
            scirs2_core::ndarray::Array2<f32>,
        )> = Vec::with_capacity(cap);

        for batch in loader.iter_batches() {
            let (inputs, targets) = batch?;
            raw_batches.push((inputs, targets));
        }

        for (inputs, targets) in raw_batches.into_iter() {
            let (x, y) = loader.to_tensors(&inputs, &targets, device)?;
            batches.push((x, y));
        }

        Ok(batches)
    }

    /// Full training loop with validation and early stopping.
    ///
    /// Iterates over the supplied data loaders, materialising one epoch's
    /// worth of `(input, target)` tensors per iteration. Validation batches
    /// are extracted with the same machinery but reuse the loader's
    /// configured shuffle behaviour (validation loaders typically have
    /// `shuffle == false`).
    ///
    /// # Validation split
    ///
    /// When `val_loader` is `None` and `config.validation_split` lies in
    /// `(0, 1)`, the trailing fraction of `train_loader`'s series is split off
    /// chronologically as the validation set (see
    /// [`TimeSeriesDataLoader::split_chronological`]) using
    /// `config.batch_size`. Early stopping then runs on *validation* loss. If
    /// the series is too short to split, the run continues without validation
    /// and logs a warning — it is never silently ignored.
    ///
    /// # Scheduler horizon
    ///
    /// Before the first epoch the scheduler is reparameterised for the true
    /// horizon, `train_loader.num_batches() * config.epochs`, so step-based
    /// schedules complete exactly at the end of training.
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

        // Derive a validation set from `validation_split` when the caller did
        // not hand one over.
        let split = self.config.validation_split;
        if val_loader.is_none() && split.is_finite() && split > 0.0 && split < 1.0 {
            match train_loader.split_chronological(split, self.config.batch_size.max(1)) {
                Ok((train_part, val_part)) => {
                    tracing::info!(
                        "Derived validation set from validation_split={}: {} train / {} val batches",
                        split,
                        train_part.num_batches(),
                        val_part.num_batches()
                    );
                    train_loader = train_part;
                    val_loader = Some(val_part);
                }
                Err(e) => {
                    tracing::warn!(
                        "validation_split={} could not be honoured ({}); \
                         training without validation and early stopping on training loss",
                        split,
                        e
                    );
                }
            }
        }

        // Now that the true batch count is known, rebuild the scheduler for
        // the real horizon instead of the placeholder estimate.
        self.set_total_steps(train_loader.num_batches().max(1) * self.config.epochs.max(1));

        // Snapshot device on the model's home device; `train_epoch` and
        // `evaluate` both move tensors to this device implicitly via the
        // forward pass, but the tensors must live there to begin with.
        let device = self.model.device().clone();

        for epoch in 0..self.config.epochs {
            let epoch_start = Instant::now();

            // Materialise one epoch's worth of training batches as candle
            // tensors. `iter_batches` handles per-epoch shuffling internally.
            let train_batches = Self::collect_epoch_batches(&mut train_loader, &device)?;

            // Train for one epoch
            let train_loss = self.train_epoch(&train_batches, loss_fn)?;

            // Validation
            let val_loss = if let Some(ref mut val_data) = val_loader {
                let val_batches = Self::collect_epoch_batches(val_data, &device)?;
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

    /// Build a trainer plus a two-batch dataset for scheduler/loss-scale tests.
    fn build_step_test_trainer(
        training_config: TrainingConfig,
    ) -> (Trainer, Vec<(Tensor, Tensor)>) {
        use crate::config::KizzasiConfig;
        use crate::training_core::TrainableSSM;

        let model_config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(1);

        let model = TrainableSSM::new(model_config, training_config.clone()).unwrap();
        let device = model.device().clone();
        let trainer = Trainer::new(model, training_config).unwrap();

        let inputs = Tensor::new(&[[[0.1f32, 0.2], [0.3, 0.4], [0.5, 0.6]]], &device).unwrap();
        let targets = Tensor::new(&[[[1.0f32, -1.0], [0.5, -0.5], [-0.2, 0.8]]], &device).unwrap();

        (trainer, vec![(inputs, targets)])
    }

    #[test]
    fn test_scheduler_lr_is_applied_to_optimizer() {
        // Regression: the scheduled LR used to be computed and logged but never
        // pushed into the optimizer, so every schedule was a no-op.
        let training_config = TrainingConfig {
            learning_rate: 1e-2,
            track_metrics: false,
            grad_clip: None,
            early_stopping_patience: None,
            ..Default::default()
        }
        .with_scheduler(SchedulerType::Linear {
            warmup_steps: 2,
            final_lr: 1e-6,
        });

        let (mut trainer, batches) = build_step_test_trainer(training_config);
        trainer.set_total_steps(10);

        // Step 0 sits at the very start of the linear warmup.
        let lr_before = trainer.optimizer_learning_rate();
        trainer.train_epoch(&batches, Loss::mse).unwrap();
        let lr_step0 = trainer.optimizer_learning_rate();
        assert_eq!(
            lr_step0, 0.0,
            "linear warmup starts at 0; optimizer LR was {lr_step0}"
        );
        assert!(
            lr_before != lr_step0,
            "optimizer LR never changed (was {lr_before})"
        );

        // Two more steps take us past the warmup into the decay phase.
        trainer.train_epoch(&batches, Loss::mse).unwrap();
        trainer.train_epoch(&batches, Loss::mse).unwrap();

        let lr_step2 = trainer.optimizer_learning_rate();
        assert!(
            lr_step2 > 0.0,
            "optimizer LR should be positive after warmup, got {lr_step2}"
        );

        // The optimizer holds the LR of the last *applied* step, i.e. the step
        // just before `current_step`.
        let applied_step = trainer.current_step() - 1;
        let expected = trainer
            .scheduler
            .as_ref()
            .expect("scheduler configured")
            .get_lr(applied_step);
        assert!(
            (lr_step2 - expected).abs() < 1e-12,
            "optimizer LR {lr_step2} does not match the schedule at step {applied_step} ({expected})"
        );
    }

    /// Snapshot every model parameter, keyed by name and sorted, so two runs
    /// are comparable. `VarMap::all_vars()` iterates a `HashMap` and therefore
    /// yields a different order in every run.
    fn named_parameter_snapshot(trainer: &Trainer) -> Vec<(String, Vec<f32>)> {
        let data = trainer
            .model
            .varmap()
            .data()
            .lock()
            .expect("varmap mutex poisoned");
        let mut snapshot: Vec<(String, Vec<f32>)> = data
            .iter()
            .map(|(name, var)| {
                let values = var
                    .as_tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                (name.clone(), values)
            })
            .collect();
        snapshot.sort_by(|a, b| a.0.cmp(&b.0));
        snapshot
    }

    #[test]
    fn test_no_scheduler_keeps_configured_learning_rate() {
        let training_config = TrainingConfig {
            learning_rate: 3e-3,
            track_metrics: false,
            early_stopping_patience: None,
            ..Default::default()
        };

        let (mut trainer, batches) = build_step_test_trainer(training_config);
        trainer.train_epoch(&batches, Loss::mse).unwrap();

        assert!(
            (trainer.optimizer_learning_rate() - 3e-3).abs() < 1e-12,
            "LR drifted without a scheduler: {}",
            trainer.optimizer_learning_rate()
        );
    }

    #[test]
    fn test_set_total_steps_reparameterises_scheduler() {
        // Regression: the horizon used to be hard-coded to `epochs * 100`, so a
        // cosine schedule never reached its terminal LR for real datasets.
        let training_config = TrainingConfig {
            learning_rate: 1e-2,
            epochs: 2,
            track_metrics: false,
            early_stopping_patience: None,
            ..Default::default()
        }
        .with_scheduler(SchedulerType::Cosine {
            warmup_steps: 0,
            min_lr: 1e-6,
        });

        let (mut trainer, _batches) = build_step_test_trainer(training_config);

        // Placeholder horizon: epochs * DEFAULT_STEPS_PER_EPOCH.
        assert_eq!(trainer.total_steps(), 2 * DEFAULT_STEPS_PER_EPOCH);

        trainer.set_total_steps(20);
        assert_eq!(trainer.total_steps(), 20);

        trainer.current_step = 20;
        let terminal_lr = trainer.get_current_lr();
        assert!(
            (terminal_lr - 1e-6).abs() < 1e-9,
            "cosine schedule should land on min_lr at the final step, got {terminal_lr}"
        );

        // With the old placeholder horizon the same step is nowhere near the end.
        trainer.set_total_steps(200);
        let mid_lr = trainer.get_current_lr();
        assert!(
            mid_lr > terminal_lr * 100.0,
            "a longer horizon must leave far more LR to decay, got {mid_lr}"
        );
    }

    #[test]
    fn test_fit_uses_true_batch_count_for_scheduler_horizon() {
        use crate::config::KizzasiConfig;
        use crate::dataloader::{DataLoaderConfig, TimeSeriesDataLoader};
        use crate::training_core::TrainableSSM;
        use scirs2_core::ndarray::Array2;

        let n_steps = 160usize;
        let raw: Vec<f32> = (0..n_steps).map(|t| ((t as f32) * 0.1).sin()).collect();
        let data = Array2::from_shape_vec((n_steps, 1), raw).unwrap();

        let dl_config = DataLoaderConfig::default()
            .with_window_size(8)
            .with_horizon(8)
            .with_batch_size(4)
            .with_shuffle(false);
        let loader = TimeSeriesDataLoader::new(data, dl_config).unwrap();

        let model_config = KizzasiConfig::new()
            .input_dim(1)
            .output_dim(1)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(1);

        let training_config = TrainingConfig {
            learning_rate: 1e-3,
            epochs: 2,
            batch_size: 4,
            track_metrics: true,
            early_stopping_patience: None,
            validation_split: 0.0,
            ..Default::default()
        }
        .with_scheduler(SchedulerType::Cosine {
            warmup_steps: 0,
            min_lr: 1e-6,
        });

        let expected_steps = loader.num_batches() * 2;
        let model = TrainableSSM::new(model_config, training_config.clone()).unwrap();
        let mut trainer = Trainer::new(model, training_config).unwrap();

        trainer.fit(loader, None, Loss::mse).unwrap();

        assert_eq!(
            trainer.total_steps(),
            expected_steps,
            "fit must derive the horizon from num_batches * epochs"
        );
        assert_eq!(trainer.current_step(), expected_steps);
        // The schedule must actually be exhausted by the end of training.
        let final_lr = trainer.get_current_lr();
        assert!(
            (final_lr - 1e-6).abs() < 1e-9,
            "cosine schedule did not finish: final LR {final_lr}"
        );
    }

    #[test]
    fn test_loss_scaling_is_unscaled_before_the_step() {
        // Regression: `with_fp16()` set `loss_scale = 128.0` and the trainer
        // ignored it. Now that scaling is applied, the *unscaling* must be exact
        // — a scaled run and an unscaled run must produce the same parameter
        // update from the same initial weights.
        fn run(loss_scale: f32) -> Vec<(String, Vec<f32>)> {
            let training_config = TrainingConfig {
                learning_rate: 1e-2,
                track_metrics: false,
                grad_clip: None,
                early_stopping_patience: None,
                loss_scale,
                ..Default::default()
            };

            let (mut trainer, batches) = build_step_test_trainer(training_config);

            // Pin the initial weights so both runs start from the same point.
            for var in trainer.model.varmap().all_vars() {
                let ones = Tensor::ones(var.shape(), var.dtype(), var.device()).unwrap();
                let seeded = ones.affine(0.05, 0.01).unwrap();
                var.set(&seeded).unwrap();
            }

            trainer.train_epoch(&batches, Loss::mse).unwrap();
            named_parameter_snapshot(&trainer)
        }

        let unscaled = run(1.0);
        let scaled = run(128.0);

        assert_eq!(unscaled.len(), scaled.len());
        assert!(!unscaled.is_empty());
        for ((u_name, u_vals), (s_name, s_vals)) in unscaled.iter().zip(scaled.iter()) {
            assert_eq!(u_name, s_name, "parameter sets differ between runs");
            assert_eq!(u_vals.len(), s_vals.len());
            for (i, (u, s)) in u_vals.iter().zip(s_vals.iter()).enumerate() {
                assert!(
                    (u - s).abs() < 1e-6,
                    "{u_name}[{i}] diverged between loss_scale=1 and loss_scale=128: {u} vs {s}"
                );
            }
        }
    }

    #[test]
    fn test_non_finite_gradients_skip_the_optimizer_step() {
        // A NaN loss must not be allowed to poison AdamW's moment estimates.
        let training_config = TrainingConfig {
            learning_rate: 1e-2,
            track_metrics: false,
            grad_clip: None,
            early_stopping_patience: None,
            ..Default::default()
        };

        let (mut trainer, batches) = build_step_test_trainer(training_config);

        let before: Vec<f32> = trainer
            .model
            .varmap()
            .all_vars()
            .iter()
            .flat_map(|v| {
                v.as_tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect();

        let nan_loss = |p: &Tensor, t: &Tensor| -> CoreResult<Tensor> {
            let base = Loss::mse(p, t)?;
            base.affine(f64::NAN, 0.0)
                .map_err(|e| CoreError::Generic(format!("{e}")))
        };

        trainer.train_epoch(&batches, nan_loss).unwrap();

        let after: Vec<f32> = trainer
            .model
            .varmap()
            .all_vars()
            .iter()
            .flat_map(|v| {
                v.as_tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect();

        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(after.iter()) {
            assert_eq!(b, a, "parameters must be untouched when gradients are NaN");
        }
        // The step counter still advances so the schedule stays aligned.
        assert_eq!(trainer.current_step(), 1);
    }

    #[test]
    fn test_fit_derives_validation_split_when_no_loader_supplied() {
        use crate::config::KizzasiConfig;
        use crate::dataloader::{DataLoaderConfig, TimeSeriesDataLoader};
        use crate::training_core::TrainableSSM;
        use scirs2_core::ndarray::Array2;

        let n_steps = 240usize;
        let raw: Vec<f32> = (0..n_steps).map(|t| ((t as f32) * 0.07).cos()).collect();
        let data = Array2::from_shape_vec((n_steps, 1), raw).unwrap();

        let dl_config = DataLoaderConfig::default()
            .with_window_size(8)
            .with_horizon(8)
            .with_batch_size(4)
            .with_shuffle(false);
        let loader = TimeSeriesDataLoader::new(data, dl_config).unwrap();

        let model_config = KizzasiConfig::new()
            .input_dim(1)
            .output_dim(1)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(1);

        let training_config = TrainingConfig {
            learning_rate: 1e-3,
            epochs: 2,
            batch_size: 2,
            track_metrics: true,
            early_stopping_patience: None,
            validation_split: 0.25,
            ..Default::default()
        };

        let model = TrainableSSM::new(model_config, training_config.clone()).unwrap();
        let mut trainer = Trainer::new(model, training_config).unwrap();

        trainer.fit(loader, None, Loss::mse).unwrap();

        // Regression: `validation_split` used to be dead, so no validation loss
        // was ever recorded when the caller passed `None`.
        assert!(
            trainer.metrics().val_loss(0).is_some(),
            "validation_split did not produce a validation set"
        );
        assert!(trainer.metrics().val_loss(1).is_some());
    }

    #[test]
    fn test_derived_validation_split_does_not_trip_early_stopping_immediately() {
        // `validation_split` and `early_stopping_patience` are both non-zero by
        // default, so activating the split also activates early stopping. Guard
        // against it firing on the very first epochs and truncating a run that
        // previously always went the distance.
        use crate::config::KizzasiConfig;
        use crate::dataloader::{DataLoaderConfig, TimeSeriesDataLoader};
        use crate::training_core::TrainableSSM;
        use scirs2_core::ndarray::Array2;

        let n_steps = 240usize;
        let raw: Vec<f32> = (0..n_steps).map(|t| ((t as f32) * 0.05).sin()).collect();
        let data = Array2::from_shape_vec((n_steps, 1), raw).unwrap();

        let dl_config = DataLoaderConfig::default()
            .with_window_size(8)
            .with_horizon(8)
            .with_batch_size(4)
            .with_shuffle(false);
        let loader = TimeSeriesDataLoader::new(data, dl_config).unwrap();

        let model_config = KizzasiConfig::new()
            .input_dim(1)
            .output_dim(1)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(1);

        // Defaults for validation_split (0.2) and early_stopping_patience (5).
        let training_config = TrainingConfig {
            learning_rate: 1e-3,
            epochs: 3,
            batch_size: 4,
            track_metrics: true,
            ..Default::default()
        };
        assert_eq!(training_config.validation_split, 0.2);
        assert_eq!(training_config.early_stopping_patience, Some(5));

        let expected_batches = loader
            .split_chronological(0.2, 4)
            .expect("series is long enough to split")
            .0
            .num_batches();

        let model = TrainableSSM::new(model_config, training_config.clone()).unwrap();
        let mut trainer = Trainer::new(model, training_config).unwrap();
        trainer.fit(loader, None, Loss::mse).unwrap();

        for epoch in 0..3 {
            assert!(
                trainer.metrics().val_loss(epoch).is_some(),
                "epoch {epoch} did not run; early stopping fired too eagerly"
            );
        }
        assert_eq!(trainer.current_step(), expected_batches * 3);
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

    /// Build a minimal trainer suitable for gradient-norm and convergence
    /// tests. Returns the trainer plus a dummy `(input, target)` batch with a
    /// non-trivial residual so the loss gradient is well-defined and non-zero.
    fn build_grad_test_trainer(grad_clip: Option<f32>) -> (Trainer, Tensor, Tensor) {
        use crate::config::KizzasiConfig;
        use crate::training_core::TrainableSSM;

        let model_config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(1);

        let training_config = TrainingConfig {
            learning_rate: 1e-3,
            track_metrics: false,
            grad_clip,
            early_stopping_patience: None,
            ..Default::default()
        };

        let model = TrainableSSM::new(model_config, training_config.clone()).unwrap();
        let device = model.device().clone();
        let trainer = Trainer::new(model, training_config).unwrap();

        // Deterministic non-trivial input/target so the residual is non-zero
        // and the gradient is well-defined. Shape: [batch=1, seq=3, dim=2].
        let inputs = Tensor::new(&[[[0.1f32, 0.2], [0.3, 0.4], [0.5, 0.6]]], &device).unwrap();
        let targets = Tensor::new(&[[[1.0f32, -1.0], [0.5, -0.5], [-0.2, 0.8]]], &device).unwrap();

        (trainer, inputs, targets)
    }

    #[test]
    fn test_compute_grad_norm_nonzero() {
        // After a real backward pass, the gradient norm must be finite and
        // strictly positive — not the historical hard-coded `1.0`.
        let (trainer, inputs, targets) = build_grad_test_trainer(None);

        let predictions = trainer.model.forward(&inputs).unwrap();
        let loss = Loss::mse(&predictions, &targets).unwrap();
        let grads = loss.backward().unwrap();

        let norm = trainer.compute_grad_norm(&grads).unwrap();
        assert!(
            norm.is_finite(),
            "gradient norm should be finite, got {}",
            norm
        );
        assert!(norm > 0.0, "gradient norm should be > 0, got {}", norm);
        // It should definitely not be the placeholder 1.0 by accident — the
        // model has dozens of parameters so the L2 norm is essentially never
        // exactly 1.
        assert!(
            (norm - 1.0).abs() > 1e-6,
            "gradient norm equals the historical placeholder value {}",
            norm
        );
    }

    #[test]
    fn test_compute_grad_norm_scales_with_loss() {
        // Analytical check: for MSE loss `(1/n) * sum((p - t)^2)` the upstream
        // gradient is `(2/n) * (p - t)`. By the chain rule the gradient w.r.t.
        // every parameter is linear in the residual `(p - t)`, so scaling the
        // residual by `k` scales the global gradient L2 norm by `|k|`.
        //
        // To get a clean factor of 2 the new target tensor must be *constant*
        // w.r.t. the parameters (otherwise backprop also flows through the
        // target). We materialise the prediction values, then build a fresh
        // constant target `t' = 2t - p` so that `p - t' = 2(p - t)`.
        let (trainer, inputs, targets) = build_grad_test_trainer(None);
        let device = trainer.model.device().clone();

        let predictions = trainer.model.forward(&inputs).unwrap();

        let pred_shape = predictions.dims().to_vec();
        let pred_vals: Vec<f32> = predictions.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let target_vals: Vec<f32> = targets.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(pred_vals.len(), target_vals.len());

        // Build a fresh, detached `targets_a` tensor with the same values as
        // the original targets so the comparison is apples-to-apples.
        let targets_a = Tensor::from_vec(target_vals.clone(), pred_shape.clone(), &device).unwrap();
        let loss_a = Loss::mse(&predictions, &targets_a).unwrap();
        let grads_a = loss_a.backward().unwrap();
        let norm_a = trainer.compute_grad_norm(&grads_a).unwrap();

        // `t' = 2t - p` → `p - t' = 2*(p - t)`.
        let scaled_target_vals: Vec<f32> = target_vals
            .iter()
            .zip(pred_vals.iter())
            .map(|(t, p)| 2.0 * t - p)
            .collect();
        let targets_b = Tensor::from_vec(scaled_target_vals, pred_shape, &device).unwrap();

        let loss_b = Loss::mse(&predictions, &targets_b).unwrap();
        let grads_b = loss_b.backward().unwrap();
        let norm_b = trainer.compute_grad_norm(&grads_b).unwrap();

        assert!(norm_a > 0.0 && norm_b > 0.0);
        let ratio = norm_b / norm_a;
        // Within 10% tolerance: the relation is exact in theory but small
        // numerical effects (mean, summation) introduce sub-percent error.
        assert!(
            (ratio - 2.0).abs() < 0.2,
            "expected gradient-norm ratio ~2.0 when residual doubles, got {} (norm_a={}, norm_b={})",
            ratio,
            norm_a,
            norm_b
        );
    }

    #[test]
    fn test_clip_gradients_caps_global_norm() {
        // Build a trainer with a very tight clip threshold and verify that
        // post-clipping the gradient norm is at most `max_norm` (within a
        // small tolerance). When the pre-clip norm is below the threshold the
        // gradients should be left untouched.
        let (trainer, inputs, targets) = build_grad_test_trainer(Some(0.01));

        let predictions = trainer.model.forward(&inputs).unwrap();
        let loss = Loss::mse(&predictions, &targets).unwrap();
        let mut grads = loss.backward().unwrap();

        let pre_norm = trainer.compute_grad_norm(&grads).unwrap();
        trainer.clip_gradients(&mut grads, 0.01).unwrap();
        let post_norm = trainer.compute_grad_norm(&grads).unwrap();

        if pre_norm > 0.01 {
            assert!(
                post_norm <= 0.01 + 1e-4,
                "clipped grad norm {} should be <= 0.01 (pre={})",
                post_norm,
                pre_norm
            );
        } else {
            // No clipping was needed; norm should be unchanged.
            assert!((post_norm - pre_norm).abs() < 1e-5);
        }
    }

    #[test]
    fn test_clip_gradients_noop_when_below_threshold() {
        // A generous threshold should leave the gradients untouched.
        let (trainer, inputs, targets) = build_grad_test_trainer(Some(1e6));

        let predictions = trainer.model.forward(&inputs).unwrap();
        let loss = Loss::mse(&predictions, &targets).unwrap();
        let mut grads = loss.backward().unwrap();

        let pre_norm = trainer.compute_grad_norm(&grads).unwrap();
        trainer.clip_gradients(&mut grads, 1e6).unwrap();
        let post_norm = trainer.compute_grad_norm(&grads).unwrap();

        assert!(
            (post_norm - pre_norm).abs() < 1e-4,
            "no-op clip should not change norm: pre={}, post={}",
            pre_norm,
            post_norm
        );
    }

    #[test]
    fn test_fit_converges_on_synthetic_series() {
        use crate::config::KizzasiConfig;
        use crate::dataloader::{DataLoaderConfig, TimeSeriesDataLoader};
        use crate::training_core::TrainableSSM;
        use scirs2_core::ndarray::Array2;

        // Build a synthetic AR(1) time series: y[t] = 0.7 * y[t-1] + noise.
        // The SSM should be able to fit at least some signal during a short
        // training run, so the final epoch loss should be strictly less than
        // the initial epoch loss.
        let n_steps: usize = 200;
        let n_features: usize = 1;
        let mut raw = vec![0.0f32; n_steps * n_features];
        raw[0] = 0.5;
        // Deterministic "noise" via a small periodic perturbation so the test
        // is reproducible without pulling in a PRNG dependency.
        for t in 1..n_steps {
            let phase = (t as f32) * 0.13;
            let noise = 0.05 * phase.sin();
            raw[t] = 0.7 * raw[t - 1] + noise;
        }
        let data = Array2::from_shape_vec((n_steps, n_features), raw).unwrap();

        let dl_config = DataLoaderConfig::default()
            .with_window_size(16)
            .with_batch_size(4)
            .with_horizon(16) // Match window so input/target seq lens match
            .with_shuffle(false);
        let loader = TimeSeriesDataLoader::new(data, dl_config).unwrap();

        let model_config = KizzasiConfig::new()
            .input_dim(n_features)
            .output_dim(n_features)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(1);

        let training_config = TrainingConfig {
            learning_rate: 1e-2,
            epochs: 5,
            batch_size: 4,
            track_metrics: true,
            early_stopping_patience: None,
            grad_clip: Some(1.0),
            ..Default::default()
        };

        let model = TrainableSSM::new(model_config, training_config.clone()).unwrap();
        let mut trainer = Trainer::new(model, training_config).unwrap();

        trainer.fit(loader, None, Loss::mse).unwrap();

        // Verify that we recorded losses for each epoch and that the trend is
        // downward.
        let initial_loss = trainer.metrics.average_train_loss(0);
        let final_loss = trainer.metrics.average_train_loss(4);

        assert!(initial_loss.is_some(), "no initial epoch loss recorded");
        assert!(final_loss.is_some(), "no final epoch loss recorded");

        let initial = initial_loss.unwrap();
        let final_l = final_loss.unwrap();

        assert!(
            initial.is_finite() && final_l.is_finite(),
            "losses must be finite: initial={}, final={}",
            initial,
            final_l
        );
        assert!(
            final_l < initial,
            "fit() did not reduce loss: initial={}, final={}",
            initial,
            final_l
        );

        // The trainer must have stepped at least once per epoch.
        assert!(trainer.current_step() > 0, "no training steps performed");
    }
}
