//! Training metrics and logging utilities
//!
//! Provides comprehensive metrics tracking for model training, including:
//! - Loss tracking (training/validation)
//! - Learning rate monitoring
//! - Gradient statistics
//! - Performance metrics (throughput, memory usage)
//!
//! # Examples
//!
//! ```rust
//! use kizzasi_core::metrics::{TrainingMetrics, MetricsLogger};
//!
//! let mut metrics = TrainingMetrics::new();
//!
//! // Track metrics during training
//! metrics.record_train_loss(0, 0.5);
//! metrics.record_val_loss(0, 0.45);
//! metrics.record_learning_rate(1e-3);
//!
//! // Get statistics
//! let avg_loss = metrics.average_train_loss(0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Training metrics tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss history [epoch -> [batch_losses]]
    train_losses: HashMap<usize, Vec<f32>>,
    /// Validation loss history [epoch -> loss]
    val_losses: HashMap<usize, f32>,
    /// Learning rate history [step -> lr]
    learning_rates: Vec<f64>,
    /// Gradient norms [step -> norm]
    grad_norms: Vec<f32>,
    /// Epoch durations [epoch -> seconds]
    epoch_durations: HashMap<usize, f64>,
    /// Current step
    current_step: usize,
    /// Best validation loss seen
    best_val_loss: Option<f32>,
    /// Best epoch
    best_epoch: Option<usize>,
}

impl TrainingMetrics {
    /// Create a new metrics tracker
    pub fn new() -> Self {
        Self {
            train_losses: HashMap::new(),
            val_losses: HashMap::new(),
            learning_rates: Vec::new(),
            grad_norms: Vec::new(),
            epoch_durations: HashMap::new(),
            current_step: 0,
            best_val_loss: None,
            best_epoch: None,
        }
    }

    /// Record training loss for a batch
    pub fn record_train_loss(&mut self, epoch: usize, loss: f32) {
        self.train_losses.entry(epoch).or_default().push(loss);
    }

    /// Record validation loss for an epoch
    pub fn record_val_loss(&mut self, epoch: usize, loss: f32) {
        self.val_losses.insert(epoch, loss);

        // Update best validation loss
        match self.best_val_loss {
            None => {
                self.best_val_loss = Some(loss);
                self.best_epoch = Some(epoch);
            }
            Some(best) if loss < best => {
                self.best_val_loss = Some(loss);
                self.best_epoch = Some(epoch);
            }
            _ => {}
        }
    }

    /// Record learning rate
    pub fn record_learning_rate(&mut self, lr: f64) {
        self.learning_rates.push(lr);
        self.current_step += 1;
    }

    /// Record gradient norm
    pub fn record_grad_norm(&mut self, norm: f32) {
        self.grad_norms.push(norm);
    }

    /// Record epoch duration
    pub fn record_epoch_duration(&mut self, epoch: usize, duration_secs: f64) {
        self.epoch_durations.insert(epoch, duration_secs);
    }

    /// Get average training loss for an epoch
    pub fn average_train_loss(&self, epoch: usize) -> Option<f32> {
        self.train_losses.get(&epoch).map(|losses| {
            let sum: f32 = losses.iter().sum();
            sum / losses.len() as f32
        })
    }

    /// Get validation loss for an epoch
    pub fn val_loss(&self, epoch: usize) -> Option<f32> {
        self.val_losses.get(&epoch).copied()
    }

    /// Get best validation loss
    pub fn best_val_loss(&self) -> Option<f32> {
        self.best_val_loss
    }

    /// Get epoch with best validation loss
    pub fn best_epoch(&self) -> Option<usize> {
        self.best_epoch
    }

    /// Get current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get learning rate at step
    pub fn lr_at_step(&self, step: usize) -> Option<f64> {
        self.learning_rates.get(step).copied()
    }

    /// Get last learning rate
    pub fn last_lr(&self) -> Option<f64> {
        self.learning_rates.last().copied()
    }

    /// Get average gradient norm over last N steps
    pub fn average_grad_norm(&self, last_n: usize) -> Option<f32> {
        if self.grad_norms.is_empty() {
            return None;
        }

        let start = self.grad_norms.len().saturating_sub(last_n);
        let norms = &self.grad_norms[start..];
        let sum: f32 = norms.iter().sum();
        Some(sum / norms.len() as f32)
    }

    /// Check if validation loss is improving
    pub fn is_improving(&self, patience: usize) -> bool {
        if let Some(best_epoch) = self.best_epoch {
            let latest_epoch = self.val_losses.keys().max().copied().unwrap_or(0);
            latest_epoch - best_epoch <= patience
        } else {
            true
        }
    }

    /// Get total training time
    pub fn total_training_time(&self) -> f64 {
        self.epoch_durations.values().sum()
    }

    /// Get summary statistics
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_epochs: self.train_losses.len(),
            best_val_loss: self.best_val_loss,
            best_epoch: self.best_epoch,
            total_training_time: self.total_training_time(),
            final_train_loss: self
                .train_losses
                .keys()
                .max()
                .and_then(|&e| self.average_train_loss(e)),
            final_val_loss: self.val_losses.keys().max().and_then(|&e| self.val_loss(e)),
        }
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_epochs: usize,
    pub best_val_loss: Option<f32>,
    pub best_epoch: Option<usize>,
    pub total_training_time: f64,
    pub final_train_loss: Option<f32>,
    pub final_val_loss: Option<f32>,
}

/// Metrics logger for console output
pub struct MetricsLogger {
    verbose: bool,
    log_interval: usize,
}

impl MetricsLogger {
    pub fn new() -> Self {
        Self {
            verbose: true,
            log_interval: 10,
        }
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_log_interval(mut self, interval: usize) -> Self {
        self.log_interval = interval;
        self
    }

    /// Log epoch metrics
    pub fn log_epoch(&self, epoch: usize, train_loss: f32, val_loss: Option<f32>, lr: f64) {
        if !self.verbose {
            return;
        }

        if let Some(val) = val_loss {
            tracing::info!(
                "Epoch {}: train_loss={:.6}, val_loss={:.6}, lr={:.2e}",
                epoch,
                train_loss,
                val,
                lr
            );
        } else {
            tracing::info!(
                "Epoch {}: train_loss={:.6}, lr={:.2e}",
                epoch,
                train_loss,
                lr
            );
        }
    }

    /// Log batch metrics
    pub fn log_batch(&self, epoch: usize, batch: usize, loss: f32) {
        if !self.verbose || !batch.is_multiple_of(self.log_interval) {
            return;
        }

        tracing::debug!("Epoch {} | Batch {}: loss={:.6}", epoch, batch, loss);
    }

    /// Log training summary
    pub fn log_summary(&self, summary: &MetricsSummary) {
        if !self.verbose {
            return;
        }

        tracing::info!("=== Training Summary ===");
        tracing::info!("Total epochs: {}", summary.total_epochs);
        if let Some(best_loss) = summary.best_val_loss {
            tracing::info!(
                "Best val loss: {:.6} (epoch {})",
                best_loss,
                summary.best_epoch.unwrap()
            );
        }
        if let Some(final_loss) = summary.final_train_loss {
            tracing::info!("Final train loss: {:.6}", final_loss);
        }
        if let Some(final_val) = summary.final_val_loss {
            tracing::info!("Final val loss: {:.6}", final_val);
        }
        tracing::info!("Total training time: {:.2}s", summary.total_training_time);
    }
}

impl Default for MetricsLogger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = TrainingMetrics::new();
        assert_eq!(metrics.current_step(), 0);
        assert_eq!(metrics.best_val_loss(), None);
    }

    #[test]
    fn test_record_train_loss() {
        let mut metrics = TrainingMetrics::new();
        metrics.record_train_loss(0, 1.0);
        metrics.record_train_loss(0, 0.9);
        metrics.record_train_loss(0, 0.8);

        let avg = metrics.average_train_loss(0).unwrap();
        assert!((avg - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_record_val_loss() {
        let mut metrics = TrainingMetrics::new();
        metrics.record_val_loss(0, 1.0);
        metrics.record_val_loss(1, 0.8);
        metrics.record_val_loss(2, 0.9);

        assert_eq!(metrics.val_loss(1), Some(0.8));
        assert_eq!(metrics.best_val_loss(), Some(0.8));
        assert_eq!(metrics.best_epoch(), Some(1));
    }

    #[test]
    fn test_learning_rate_tracking() {
        let mut metrics = TrainingMetrics::new();
        metrics.record_learning_rate(1e-3);
        metrics.record_learning_rate(9e-4);
        metrics.record_learning_rate(8e-4);

        assert_eq!(metrics.current_step(), 3);
        assert_eq!(metrics.last_lr(), Some(8e-4));
        assert_eq!(metrics.lr_at_step(0), Some(1e-3));
    }

    #[test]
    fn test_gradient_norm_tracking() {
        let mut metrics = TrainingMetrics::new();
        metrics.record_grad_norm(1.0);
        metrics.record_grad_norm(2.0);
        metrics.record_grad_norm(3.0);
        metrics.record_grad_norm(4.0);

        let avg = metrics.average_grad_norm(2).unwrap();
        assert!((avg - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_is_improving() {
        let mut metrics = TrainingMetrics::new();
        metrics.record_val_loss(0, 1.0);
        metrics.record_val_loss(1, 0.8);
        metrics.record_val_loss(2, 0.85);

        assert!(metrics.is_improving(5)); // Within patience
        assert!(metrics.is_improving(1)); // Still within 1 epoch of best
        assert!(!metrics.is_improving(0)); // No patience
    }

    #[test]
    fn test_epoch_duration() {
        let mut metrics = TrainingMetrics::new();
        metrics.record_epoch_duration(0, 10.5);
        metrics.record_epoch_duration(1, 9.8);

        assert!((metrics.total_training_time() - 20.3).abs() < 1e-6);
    }

    #[test]
    fn test_summary() {
        let mut metrics = TrainingMetrics::new();
        metrics.record_train_loss(0, 1.0);
        metrics.record_train_loss(0, 0.9);
        metrics.record_val_loss(0, 0.85);
        metrics.record_epoch_duration(0, 10.0);

        let summary = metrics.summary();
        assert_eq!(summary.total_epochs, 1);
        assert_eq!(summary.best_val_loss, Some(0.85));
        assert!((summary.total_training_time - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_metrics_logger() {
        let logger = MetricsLogger::new()
            .with_verbose(false)
            .with_log_interval(5);

        // Just test that it doesn't panic
        logger.log_epoch(0, 0.5, Some(0.45), 1e-3);
        logger.log_batch(0, 5, 0.6);

        let summary = MetricsSummary {
            total_epochs: 10,
            best_val_loss: Some(0.1),
            best_epoch: Some(5),
            total_training_time: 100.0,
            final_train_loss: Some(0.2),
            final_val_loss: Some(0.15),
        };
        logger.log_summary(&summary);
    }
}
