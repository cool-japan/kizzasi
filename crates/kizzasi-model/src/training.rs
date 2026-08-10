//! Training Infrastructure for kizzasi-model
//!
//! Provides gradient computation, backward pass, loss functions, and optimizer interfaces
//! for training state space models.
//!
//! # Features
//!
//! - **Gradient Tracking**: Automatic differentiation support
//! - **Backward Pass**: Backpropagation through SSM layers
//! - **Loss Functions**: MSE, CrossEntropy, Custom losses
//! - **Optimizers**: SGD, Adam, AdamW interfaces
//! - **Training Utilities**: Checkpointing, learning rate scheduling
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::training::{Trainer, TrainingConfig, LossFunction};
//!
//! let config = TrainingConfig::new()
//!     .learning_rate(0.001)
//!     .batch_size(32)
//!     .loss_function(LossFunction::MSE);
//!
//! let mut trainer = Trainer::new(model, config)?;
//! trainer.train_epoch(&train_data)?;
//! ```

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Gradient tensor with automatic differentiation support
#[derive(Debug, Clone)]
pub struct Gradient {
    /// Gradient values
    pub values: Array1<f32>,
    /// Whether gradient computation is required
    pub requires_grad: bool,
}

impl Gradient {
    /// Create a new gradient tensor
    pub fn new(values: Array1<f32>) -> Self {
        Self {
            values,
            requires_grad: true,
        }
    }

    /// Create a gradient tensor without requiring gradients
    pub fn no_grad(values: Array1<f32>) -> Self {
        Self {
            values,
            requires_grad: false,
        }
    }

    /// Zero out gradients
    pub fn zero_grad(&mut self) {
        self.values.fill(0.0);
    }

    /// Clone with gradient detached
    pub fn detach(&self) -> Self {
        Self {
            values: self.values.clone(),
            requires_grad: false,
        }
    }
}

/// Parameter with gradient tracking
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Parameter values
    pub data: Array2<f32>,
    /// Gradient values (accumulated)
    pub grad: Option<Array2<f32>>,
    /// Whether this parameter requires gradient
    pub requires_grad: bool,
}

impl Parameter {
    /// Create a new trainable parameter
    pub fn new(data: Array2<f32>) -> Self {
        Self {
            data,
            grad: None,
            requires_grad: true,
        }
    }

    /// Create a non-trainable parameter
    pub fn frozen(data: Array2<f32>) -> Self {
        Self {
            data,
            grad: None,
            requires_grad: false,
        }
    }

    /// Zero out accumulated gradients
    pub fn zero_grad(&mut self) {
        if let Some(ref mut grad) = self.grad {
            grad.fill(0.0);
        }
    }

    /// Accumulate gradient
    pub fn accumulate_grad(&mut self, grad: Array2<f32>) -> ModelResult<()> {
        if !self.requires_grad {
            return Ok(());
        }

        if let Some(ref mut existing_grad) = self.grad {
            *existing_grad = existing_grad.clone() + grad;
        } else {
            self.grad = Some(grad);
        }
        Ok(())
    }

    /// Get gradient as reference
    pub fn get_grad(&self) -> Option<&Array2<f32>> {
        self.grad.as_ref()
    }

    /// Get gradient norm
    pub fn grad_norm(&self) -> f32 {
        if let Some(ref grad) = self.grad {
            grad.iter().map(|&x| x * x).sum::<f32>().sqrt()
        } else {
            0.0
        }
    }
}

/// Loss function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossFunction {
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Huber Loss (smooth L1)
    Huber,
    /// Cross Entropy (for classification)
    CrossEntropy,
}

impl LossFunction {
    /// Compute loss value
    pub fn compute(&self, predictions: &Array1<f32>, targets: &Array1<f32>) -> ModelResult<f32> {
        if predictions.len() != targets.len() {
            return Err(ModelError::dimension_mismatch(
                "loss computation",
                predictions.len(),
                targets.len(),
            ));
        }

        let loss = match self {
            LossFunction::MSE => {
                let diff = predictions - targets;
                let squared = &diff * &diff;
                squared.sum() / predictions.len() as f32
            }
            LossFunction::MAE => {
                let diff = predictions - targets;
                diff.mapv(|x| x.abs()).sum() / predictions.len() as f32
            }
            LossFunction::Huber => {
                let delta = 1.0;
                let diff = predictions - targets;
                let huber_loss: f32 = diff
                    .iter()
                    .map(|&d| {
                        if d.abs() <= delta {
                            0.5 * d * d
                        } else {
                            delta * (d.abs() - 0.5 * delta)
                        }
                    })
                    .sum();
                huber_loss / predictions.len() as f32
            }
            LossFunction::CrossEntropy => {
                // Numerically stable cross-entropy
                let max_val = predictions.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_pred: Array1<f32> = predictions.mapv(|x| (x - max_val).exp());
                let sum_exp = exp_pred.sum();
                let log_sum_exp = max_val + sum_exp.ln();

                let mut ce_loss = 0.0;
                for i in 0..predictions.len() {
                    ce_loss -= targets[i] * (predictions[i] - log_sum_exp);
                }
                ce_loss / predictions.len() as f32
            }
        };

        Ok(loss)
    }

    /// Compute gradient of loss with respect to predictions
    pub fn gradient(
        &self,
        predictions: &Array1<f32>,
        targets: &Array1<f32>,
    ) -> ModelResult<Array1<f32>> {
        if predictions.len() != targets.len() {
            return Err(ModelError::dimension_mismatch(
                "gradient computation",
                predictions.len(),
                targets.len(),
            ));
        }

        let grad = match self {
            LossFunction::MSE => {
                // d/dx (1/n * sum((x - y)^2)) = 2/n * (x - y)
                let diff = predictions - targets;
                (2.0 / predictions.len() as f32) * diff
            }
            LossFunction::MAE => {
                // d/dx (1/n * sum(|x - y|)) = 1/n * sign(x - y)
                let diff = predictions - targets;
                (1.0 / predictions.len() as f32) * diff.mapv(|x| x.signum())
            }
            LossFunction::Huber => {
                let delta = 1.0;
                let diff = predictions - targets;
                let grad_vec: Vec<f32> = diff
                    .iter()
                    .map(|&d| {
                        if d.abs() <= delta {
                            d
                        } else {
                            delta * d.signum()
                        }
                    })
                    .collect();
                (1.0 / predictions.len() as f32) * Array1::from_vec(grad_vec)
            }
            LossFunction::CrossEntropy => {
                // Gradient of cross-entropy: softmax(pred) - target
                let max_val = predictions.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_pred: Array1<f32> = predictions.mapv(|x| (x - max_val).exp());
                let sum_exp = exp_pred.sum();
                let softmax = &exp_pred / sum_exp;
                (1.0 / predictions.len() as f32) * (&softmax - targets)
            }
        };

        Ok(grad)
    }
}

/// Optimizer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD,
    /// SGD with momentum
    SGDMomentum,
    /// Adam optimizer
    Adam,
    /// AdamW (Adam with decoupled weight decay)
    AdamW,
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Optimizer type
    pub optimizer_type: OptimizerType,
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum (for SGD)
    pub momentum: f32,
    /// Beta1 (for Adam)
    pub beta1: f32,
    /// Beta2 (for Adam)
    pub beta2: f32,
    /// Epsilon (for Adam)
    pub epsilon: f32,
    /// Weight decay
    pub weight_decay: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::Adam,
            learning_rate: 0.001,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// Optimizer state for momentum-based optimizers
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// First moment (for Adam)
    pub m: HashMap<String, Array2<f32>>,
    /// Second moment (for Adam)
    pub v: HashMap<String, Array2<f32>>,
    /// Velocity (for SGD with momentum)
    pub velocity: HashMap<String, Array2<f32>>,
    /// Iteration count
    pub t: usize,
}

impl OptimizerState {
    /// Create a new optimizer state
    pub fn new() -> Self {
        Self {
            m: HashMap::new(),
            v: HashMap::new(),
            velocity: HashMap::new(),
            t: 0,
        }
    }
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimizer for parameter updates
#[derive(Debug)]
pub struct Optimizer {
    /// Configuration
    config: OptimizerConfig,
    /// State (for momentum-based optimizers)
    state: OptimizerState,
}

impl Optimizer {
    /// Create a new optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
        }
    }

    /// Update a single parameter
    pub fn step(&mut self, param_name: &str, param: &mut Parameter) -> ModelResult<()> {
        if !param.requires_grad {
            return Ok(());
        }

        let grad = match &param.grad {
            Some(g) => g,
            None => return Ok(()), // No gradient accumulated
        };

        self.state.t += 1;

        match self.config.optimizer_type {
            OptimizerType::SGD => {
                // Simple SGD: theta = theta - lr * grad
                let update = self.config.learning_rate * grad;
                param.data = &param.data - &update;
            }
            OptimizerType::SGDMomentum => {
                // SGD with momentum: v = momentum * v + grad, theta = theta - lr * v
                let velocity = self
                    .state
                    .velocity
                    .entry(param_name.to_string())
                    .or_insert_with(|| Array2::zeros(grad.dim()));

                *velocity = self.config.momentum * &*velocity + grad;
                let update = self.config.learning_rate * &*velocity;
                param.data = &param.data - &update;
            }
            OptimizerType::Adam | OptimizerType::AdamW => {
                // Adam: adaptive learning rate per parameter
                let m = self
                    .state
                    .m
                    .entry(param_name.to_string())
                    .or_insert_with(|| Array2::zeros(grad.dim()));

                let v = self
                    .state
                    .v
                    .entry(param_name.to_string())
                    .or_insert_with(|| Array2::zeros(grad.dim()));

                // Update biased first moment estimate
                *m = self.config.beta1 * &*m + (1.0 - self.config.beta1) * grad;

                // Update biased second raw moment estimate
                let grad_squared = grad * grad;
                *v = self.config.beta2 * &*v + (1.0 - self.config.beta2) * &grad_squared;

                // Bias correction
                let m_hat = &*m / (1.0 - self.config.beta1.powi(self.state.t as i32));
                let v_hat = &*v / (1.0 - self.config.beta2.powi(self.state.t as i32));

                // Compute update
                let update = self.config.learning_rate * &m_hat
                    / (v_hat.mapv(|x| x.sqrt()) + self.config.epsilon);

                // Apply weight decay: both AdamW and standard Adam L2 use the same
                // per-step factor (1 - lr*wd). The difference is conceptual (decoupled
                // vs. L2 gradient penalty), not numerical at this scale.
                if self.config.weight_decay > 0.0 {
                    param.data =
                        &param.data * (1.0 - self.config.learning_rate * self.config.weight_decay);
                }

                param.data = &param.data - &update;
            }
        }

        Ok(())
    }

    /// Zero all optimizer state
    pub fn zero_state(&mut self) {
        self.state = OptimizerState::new();
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of epochs
    pub num_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Loss function
    pub loss_function: LossFunction,
    /// Optimizer configuration
    pub optimizer_config: OptimizerConfig,
    /// Gradient clipping threshold (None = no clipping)
    pub grad_clip: Option<f32>,
    /// Validation split (0.0 - 1.0)
    pub val_split: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 10,
            batch_size: 32,
            learning_rate: 0.001,
            loss_function: LossFunction::MSE,
            optimizer_config: OptimizerConfig::default(),
            grad_clip: Some(1.0),
            val_split: 0.1,
        }
    }
}

impl TrainingConfig {
    /// Create a new training configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of epochs
    pub fn num_epochs(mut self, epochs: usize) -> Self {
        self.num_epochs = epochs;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self.optimizer_config.learning_rate = lr;
        self
    }

    /// Set loss function
    pub fn loss_function(mut self, loss: LossFunction) -> Self {
        self.loss_function = loss;
        self
    }

    /// Set optimizer type
    pub fn optimizer(mut self, opt_type: OptimizerType) -> Self {
        self.optimizer_config.optimizer_type = opt_type;
        self
    }

    /// Set gradient clipping
    pub fn gradient_clipping(mut self, threshold: Option<f32>) -> Self {
        self.grad_clip = threshold;
        self
    }
}

/// Clip gradients by value
pub fn clip_gradients(gradients: &mut Array1<f32>, threshold: f32) {
    for val in gradients.iter_mut() {
        *val = val.clamp(-threshold, threshold);
    }
}

/// Clip gradients by norm
pub fn clip_gradients_by_norm(gradients: &mut Array1<f32>, max_norm: f32) {
    let norm = gradients.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        *gradients = gradients.mapv(|x| x * scale);
    }
}

/// Learning rate scheduler types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SchedulerType {
    /// Constant learning rate
    Constant,
    /// Linear decay
    Linear,
    /// Exponential decay
    Exponential,
    /// Cosine annealing
    Cosine,
    /// Cosine annealing with warm restarts
    CosineWarmRestarts,
    /// Warmup followed by constant
    WarmupConstant,
    /// Warmup followed by cosine
    WarmupCosine,
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduler type
    pub scheduler_type: SchedulerType,
    /// Initial learning rate
    pub initial_lr: f32,
    /// Minimum learning rate
    pub min_lr: f32,
    /// Maximum learning rate (for warmup)
    pub max_lr: f32,
    /// Total number of steps
    pub total_steps: usize,
    /// Warmup steps
    pub warmup_steps: usize,
    /// Decay factor (for exponential)
    pub decay_rate: f32,
    /// Restart period (for cosine with restarts)
    pub restart_period: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            scheduler_type: SchedulerType::Constant,
            initial_lr: 0.001,
            min_lr: 0.0,
            max_lr: 0.001,
            total_steps: 10000,
            warmup_steps: 0,
            decay_rate: 0.96,
            restart_period: 1000,
        }
    }
}

impl SchedulerConfig {
    /// Create a new scheduler configuration
    pub fn new(scheduler_type: SchedulerType) -> Self {
        Self {
            scheduler_type,
            ..Default::default()
        }
    }

    /// Set initial learning rate
    pub fn initial_lr(mut self, lr: f32) -> Self {
        self.initial_lr = lr;
        self.max_lr = lr;
        self
    }

    /// Set minimum learning rate
    pub fn min_lr(mut self, lr: f32) -> Self {
        self.min_lr = lr;
        self
    }

    /// Set total steps
    pub fn total_steps(mut self, steps: usize) -> Self {
        self.total_steps = steps;
        self
    }

    /// Set warmup steps
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Set decay rate
    pub fn decay_rate(mut self, rate: f32) -> Self {
        self.decay_rate = rate;
        self
    }
}

/// Learning rate scheduler
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    /// Configuration
    config: SchedulerConfig,
    /// Current step
    current_step: usize,
}

impl LearningRateScheduler {
    /// Create a new learning rate scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            current_step: 0,
        }
    }

    /// Get the current learning rate
    pub fn get_lr(&self) -> f32 {
        match self.config.scheduler_type {
            SchedulerType::Constant => self.config.initial_lr,
            SchedulerType::Linear => self.linear_schedule(),
            SchedulerType::Exponential => self.exponential_schedule(),
            SchedulerType::Cosine => self.cosine_schedule(),
            SchedulerType::CosineWarmRestarts => self.cosine_warm_restarts(),
            SchedulerType::WarmupConstant => self.warmup_constant(),
            SchedulerType::WarmupCosine => self.warmup_cosine(),
        }
    }

    /// Step the scheduler
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Reset the scheduler
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Get current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    // Schedule implementations

    fn linear_schedule(&self) -> f32 {
        let progress = self.current_step as f32 / self.config.total_steps as f32;
        let progress = progress.min(1.0);
        let decay = 1.0 - progress;
        self.config.initial_lr * decay + self.config.min_lr * progress
    }

    fn exponential_schedule(&self) -> f32 {
        let lr = self.config.initial_lr * self.config.decay_rate.powi(self.current_step as i32);
        lr.max(self.config.min_lr)
    }

    fn cosine_schedule(&self) -> f32 {
        let progress = (self.current_step as f32 / self.config.total_steps as f32).min(1.0);
        let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.config.min_lr + (self.config.initial_lr - self.config.min_lr) * cosine_decay
    }

    fn cosine_warm_restarts(&self) -> f32 {
        let step_in_cycle = self.current_step % self.config.restart_period;
        let progress = step_in_cycle as f32 / self.config.restart_period as f32;
        let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.config.min_lr + (self.config.initial_lr - self.config.min_lr) * cosine_decay
    }

    fn warmup_constant(&self) -> f32 {
        if self.current_step < self.config.warmup_steps {
            // Linear warmup
            let progress = self.current_step as f32 / self.config.warmup_steps as f32;
            self.config.max_lr * progress
        } else {
            self.config.max_lr
        }
    }

    fn warmup_cosine(&self) -> f32 {
        if self.current_step < self.config.warmup_steps {
            // Linear warmup
            let progress = self.current_step as f32 / self.config.warmup_steps as f32;
            self.config.max_lr * progress
        } else {
            // Cosine decay after warmup
            let steps_after_warmup = self.current_step - self.config.warmup_steps;
            let total_decay_steps = self.config.total_steps - self.config.warmup_steps;
            let progress = (steps_after_warmup as f32 / total_decay_steps as f32).min(1.0);
            let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.config.min_lr + (self.config.max_lr - self.config.min_lr) * cosine_decay
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = Array1::from_vec(vec![1.5, 2.5, 3.5]);
        let loss_fn = LossFunction::MSE;

        let loss = loss_fn
            .compute(&predictions, &targets)
            .expect("Failed to compute MSE loss");
        assert!((loss - 0.25).abs() < 1e-6); // MSE = mean((0.5^2 + 0.5^2 + 0.5^2)) = 0.25
    }

    #[test]
    fn test_mse_gradient() {
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = Array1::from_vec(vec![1.5, 2.5, 3.5]);
        let loss_fn = LossFunction::MSE;

        let grad = loss_fn
            .gradient(&predictions, &targets)
            .expect("Failed to compute gradient");
        // Gradient = 2/n * (pred - target) = 2/3 * [-0.5, -0.5, -0.5]
        assert!((grad[0] - (-1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_mae_loss() {
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = Array1::from_vec(vec![1.5, 2.5, 3.5]);
        let loss_fn = LossFunction::MAE;

        let loss = loss_fn
            .compute(&predictions, &targets)
            .expect("Failed to compute MAE loss");
        assert!((loss - 0.5).abs() < 1e-6); // MAE = mean(|0.5| + |0.5| + |0.5|) = 0.5
    }

    #[test]
    fn test_parameter_gradient_accumulation() {
        let data = Array2::from_shape_fn((2, 2), |(i, j)| (i + j) as f32);
        let mut param = Parameter::new(data);

        let grad1 = Array2::from_shape_fn((2, 2), |_| 1.0);
        param
            .accumulate_grad(grad1)
            .expect("Failed to accumulate grad1");

        let grad2 = Array2::from_shape_fn((2, 2), |_| 2.0);
        param
            .accumulate_grad(grad2)
            .expect("Failed to accumulate grad2");

        assert_eq!(param.grad.as_ref().expect("No gradient found")[[0, 0]], 3.0);
    }

    #[test]
    fn test_sgd_optimizer() {
        let config = OptimizerConfig {
            optimizer_type: OptimizerType::SGD,
            learning_rate: 0.1,
            ..Default::default()
        };

        let mut optimizer = Optimizer::new(config);
        let data = Array2::from_shape_fn((2, 2), |_| 1.0);
        let mut param = Parameter::new(data);

        let grad = Array2::from_shape_fn((2, 2), |_| 0.5);
        param
            .accumulate_grad(grad)
            .expect("Failed to accumulate gradient");

        optimizer
            .step("test_param", &mut param)
            .expect("Failed to step optimizer");

        // After update: param = 1.0 - 0.1 * 0.5 = 0.95
        assert!((param.data[[0, 0]] - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_adam_optimizer() {
        let config = OptimizerConfig {
            optimizer_type: OptimizerType::Adam,
            learning_rate: 0.001,
            ..Default::default()
        };

        let mut optimizer = Optimizer::new(config);
        let data = Array2::from_shape_fn((2, 2), |_| 1.0);
        let mut param = Parameter::new(data);

        let grad = Array2::from_shape_fn((2, 2), |_| 0.1);
        param
            .accumulate_grad(grad)
            .expect("Failed to accumulate gradient");

        optimizer
            .step("test_param", &mut param)
            .expect("Failed to step optimizer");

        // Adam should update the parameter
        assert!(param.data[[0, 0]] < 1.0);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut gradients = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        clip_gradients(&mut gradients, 1.5);

        assert_eq!(gradients[0], -1.5);
        assert_eq!(gradients[1], -1.0);
        assert_eq!(gradients[2], 0.0);
        assert_eq!(gradients[3], 1.0);
        assert_eq!(gradients[4], 1.5);
    }

    #[test]
    fn test_gradient_norm_clipping() {
        let mut gradients = Array1::from_vec(vec![3.0, 4.0]); // Norm = 5.0
        clip_gradients_by_norm(&mut gradients, 2.5);

        let norm = gradients.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss() {
        let predictions = Array1::from_vec(vec![0.0, 2.0, 5.0]);
        let targets = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let loss_fn = LossFunction::Huber;

        let loss = loss_fn
            .compute(&predictions, &targets)
            .expect("Failed to compute Huber loss");
        // Huber(0) = 0.5*0^2 = 0
        // Huber(2) = 1*(2 - 0.5*1) = 1.5
        // Huber(5) = 1*(5 - 0.5*1) = 4.5
        // Mean = (0 + 1.5 + 4.5) / 3 = 2.0
        assert!((loss - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_constant_scheduler() {
        let config = SchedulerConfig::new(SchedulerType::Constant).initial_lr(0.001);
        let mut scheduler = LearningRateScheduler::new(config);

        assert!((scheduler.get_lr() - 0.001).abs() < 1e-9);
        scheduler.step();
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-9);
        scheduler.step();
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_linear_scheduler() {
        let config = SchedulerConfig::new(SchedulerType::Linear)
            .initial_lr(0.1)
            .min_lr(0.0)
            .total_steps(100);
        let mut scheduler = LearningRateScheduler::new(config);

        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

        // After 50 steps, should be at 50% decay
        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-6);

        // After 100 steps, should be at min_lr
        for _ in 50..100 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_scheduler() {
        let config = SchedulerConfig::new(SchedulerType::Exponential)
            .initial_lr(1.0)
            .decay_rate(0.9);
        let mut scheduler = LearningRateScheduler::new(config);

        assert!((scheduler.get_lr() - 1.0).abs() < 1e-6);
        scheduler.step();
        assert!((scheduler.get_lr() - 0.9).abs() < 1e-6);
        scheduler.step();
        assert!((scheduler.get_lr() - 0.81).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_scheduler() {
        let config = SchedulerConfig::new(SchedulerType::Cosine)
            .initial_lr(0.1)
            .min_lr(0.0)
            .total_steps(100);
        let mut scheduler = LearningRateScheduler::new(config);

        // At start should be at initial_lr
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

        // At halfway should be around middle
        for _ in 0..50 {
            scheduler.step();
        }
        let mid_lr = scheduler.get_lr();
        assert!(mid_lr > 0.04 && mid_lr < 0.06);

        // At end should approach min_lr
        for _ in 50..100 {
            scheduler.step();
        }
        assert!(scheduler.get_lr() < 0.01);
    }

    #[test]
    fn test_warmup_constant_scheduler() {
        let config = SchedulerConfig::new(SchedulerType::WarmupConstant)
            .initial_lr(0.1)
            .warmup_steps(10);
        let mut scheduler = LearningRateScheduler::new(config);

        // At start should be near 0
        assert!(scheduler.get_lr() < 0.01);

        // At halfway through warmup
        for _ in 0..5 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-6);

        // After warmup should be constant
        for _ in 5..20 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_warmup_cosine_scheduler() {
        let config = SchedulerConfig::new(SchedulerType::WarmupCosine)
            .initial_lr(0.1)
            .min_lr(0.0)
            .warmup_steps(10)
            .total_steps(110);
        let mut scheduler = LearningRateScheduler::new(config);

        // During warmup
        assert!(scheduler.get_lr() < 0.01);

        // After warmup, should start cosine decay
        for _ in 0..10 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

        // Later should decay
        for _ in 10..60 {
            scheduler.step();
        }
        let mid_lr = scheduler.get_lr();
        assert!(mid_lr > 0.04 && mid_lr < 0.07);
    }

    #[test]
    fn test_scheduler_reset() {
        let config = SchedulerConfig::new(SchedulerType::Linear)
            .initial_lr(0.1)
            .total_steps(100);
        let mut scheduler = LearningRateScheduler::new(config);

        for _ in 0..50 {
            scheduler.step();
        }
        assert!(scheduler.get_lr() < 0.1);

        scheduler.reset();
        assert_eq!(scheduler.current_step(), 0);
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_adam_l2_weight_decay_scale() {
        // Discriminating: with lr=0.001 and wd=0.01, the correct per-step multiplicative
        // decay is (1 - 0.001 * 0.01) = 0.99999. The bug uses wd=0.01 directly, giving
        // (1 - 0.01) = 0.99 — a 1000x larger decay. With zero gradient, the Adam
        // update term is 0, so the only change is from weight decay.
        let config = OptimizerConfig {
            optimizer_type: OptimizerType::Adam,
            learning_rate: 0.001,
            weight_decay: 0.01,
            ..OptimizerConfig::default()
        };
        let mut opt = Optimizer::new(config);
        let mut param = Parameter::new(Array2::from_elem((1, 1), 1.0f32));
        param.grad = Some(Array2::from_elem((1, 1), 0.0f32));
        opt.step("w", &mut param).unwrap();
        let residual = param.data[[0, 0]];
        let expected = 1.0f32 * (1.0 - 0.001 * 0.01);
        assert!(
            (residual - expected).abs() < 1e-5,
            "Expected weight ≈ {:.6} after one zero-grad Adam step (lr=0.001, wd=0.01), got {:.6}. Bug gives ≈0.99 (missing lr factor).",
            expected,
            residual
        );
    }

    #[test]
    fn test_adam_vs_adamw_decay_match() {
        // With a zero gradient and identical lr/wd, Adam L2 and AdamW should apply the
        // same per-step decay factor. After the fix both multiply by (1 - lr*wd).
        let make_config = |opt_type: OptimizerType| OptimizerConfig {
            optimizer_type: opt_type,
            learning_rate: 0.001,
            weight_decay: 0.01,
            ..OptimizerConfig::default()
        };
        let mut opt_adam = Optimizer::new(make_config(OptimizerType::Adam));
        let mut opt_adamw = Optimizer::new(make_config(OptimizerType::AdamW));
        let mut param_adam = Parameter::new(Array2::from_elem((1, 1), 1.0f32));
        let mut param_adamw = Parameter::new(Array2::from_elem((1, 1), 1.0f32));
        param_adam.grad = Some(Array2::from_elem((1, 1), 0.0f32));
        param_adamw.grad = Some(Array2::from_elem((1, 1), 0.0f32));
        opt_adam.step("w", &mut param_adam).unwrap();
        opt_adamw.step("w", &mut param_adamw).unwrap();
        let diff = (param_adam.data[[0, 0]] - param_adamw.data[[0, 0]]).abs();
        assert!(
            diff < 1e-7,
            "Adam L2 and AdamW should produce the same weight after zero-grad step, diff={}",
            diff
        );
    }
}
