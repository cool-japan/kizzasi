//! Training infrastructure for SSM models
//!
//! Provides trainable versions of SSM components with automatic differentiation,
//! loss functions, and optimization utilities using candle-core.
//!
//! # Features
//!
//! - **TrainableSSM**: Differentiable SSM model with automatic gradient tracking
//! - **Trainer**: Full training loop with scheduler, metrics, and validation
//! - **Loss Functions**: MSE, MAE, Huber, Cross-Entropy
//! - **LR Scheduling**: Integrated support for all scheduler types
//! - **Metrics Tracking**: Automatic loss, LR, and gradient monitoring
//! - **Early Stopping**: Validation-based early stopping with patience
//!
//! ## Module Layout
//!
//! The implementation is split across:
//! - [`training_core`](super::training_core): SchedulerType, MixedPrecision,
//!   TrainingConfig, TrainableSSM
//! - [`training_loop`](super::training_loop): ConstraintLoss, Loss, Trainer
//! - [`training_checkpoint`](super::training_checkpoint): CheckpointMetadata and
//!   the `Trainer` checkpoint save/load methods

pub use crate::training_checkpoint::CheckpointMetadata;
pub use crate::training_core::{MixedPrecision, SchedulerType, TrainableSSM, TrainingConfig};
pub use crate::training_loop::{ConstraintLoss, Loss, Trainer};
