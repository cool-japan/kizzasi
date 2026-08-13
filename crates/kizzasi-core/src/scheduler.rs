//! Learning rate schedulers for training
//!
//! Provides various learning rate scheduling strategies for optimizing
//! model training convergence.
//!
//! # Strategies
//!
//! - **Constant**: Fixed learning rate
//! - **Linear**: Linear warmup and decay
//! - **Cosine**: Cosine annealing with warmup
//! - **Step**: Step-wise decay at milestones
//! - **Exponential**: Exponential decay
//! - **OneCycle**: One-cycle learning rate policy
//!
//! # Examples
//!
//! ```rust
//! use kizzasi_core::scheduler::{LRScheduler, CosineScheduler};
//!
//! let scheduler = CosineScheduler::new(1e-3, 1000, 100);
//!
//! for step in 0..1000 {
//!     let lr = scheduler.get_lr(step);
//!     // Update optimizer with new learning rate
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Learning rate scheduler trait
pub trait LRScheduler {
    /// Get learning rate for a given step
    fn get_lr(&self, step: usize) -> f64;

    /// Get the last learning rate
    fn last_lr(&self) -> f64;
}

/// Constant learning rate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantScheduler {
    lr: f64,
}

impl ConstantScheduler {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantScheduler {
    fn get_lr(&self, _step: usize) -> f64 {
        self.lr
    }

    fn last_lr(&self) -> f64 {
        self.lr
    }
}

/// Linear warmup and decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearScheduler {
    /// Initial learning rate
    initial_lr: f64,
    /// Final learning rate
    final_lr: f64,
    /// Number of warmup steps
    warmup_steps: usize,
    /// Total number of training steps
    total_steps: usize,
    /// Last computed learning rate
    last_lr: f64,
}

impl LinearScheduler {
    pub fn new(initial_lr: f64, final_lr: f64, total_steps: usize, warmup_steps: usize) -> Self {
        Self {
            initial_lr,
            final_lr,
            warmup_steps,
            total_steps,
            last_lr: initial_lr,
        }
    }
}

impl LRScheduler for LinearScheduler {
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.initial_lr * (step as f64 / self.warmup_steps as f64)
        } else {
            // Linear decay. `total_steps` may be smaller than `warmup_steps`
            // when the horizon is derived from a short dataset, so the decay
            // window is saturating and floored at one step.
            let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1) as f64;
            let progress = ((step - self.warmup_steps) as f64 / decay_steps).clamp(0.0, 1.0);
            self.initial_lr + (self.final_lr - self.initial_lr) * progress
        }
    }

    fn last_lr(&self) -> f64 {
        self.last_lr
    }
}

/// Cosine annealing with warmup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosineScheduler {
    /// Maximum learning rate
    max_lr: f64,
    /// Minimum learning rate
    min_lr: f64,
    /// Total number of steps
    total_steps: usize,
    /// Number of warmup steps
    warmup_steps: usize,
    /// Last computed learning rate
    last_lr: f64,
}

impl CosineScheduler {
    pub fn new(max_lr: f64, total_steps: usize, warmup_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr: 0.0,
            total_steps,
            warmup_steps,
            last_lr: max_lr,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LRScheduler for CosineScheduler {
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.max_lr * (step as f64 / self.warmup_steps as f64)
        } else {
            // Cosine annealing. Guard against a horizon shorter than the
            // warmup (possible once `total_steps` is derived from the real
            // batch count) which would underflow `usize` and divide by zero.
            let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1) as f64;
            let progress = ((step - self.warmup_steps) as f64 / decay_steps).clamp(0.0, 1.0);
            let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
            self.min_lr + (self.max_lr - self.min_lr) * cosine
        }
    }

    fn last_lr(&self) -> f64 {
        self.last_lr
    }
}

/// Step-wise decay at milestones
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepScheduler {
    /// Initial learning rate
    initial_lr: f64,
    /// Decay factor (multiply LR by this at each milestone)
    decay_factor: f64,
    /// Steps at which to decay the learning rate
    milestones: Vec<usize>,
    /// Last computed learning rate
    last_lr: f64,
}

impl StepScheduler {
    pub fn new(initial_lr: f64, decay_factor: f64, milestones: Vec<usize>) -> Self {
        Self {
            initial_lr,
            decay_factor,
            milestones,
            last_lr: initial_lr,
        }
    }
}

impl LRScheduler for StepScheduler {
    fn get_lr(&self, step: usize) -> f64 {
        let num_decays = self.milestones.iter().filter(|&&m| step >= m).count();
        self.initial_lr * self.decay_factor.powi(num_decays as i32)
    }

    fn last_lr(&self) -> f64 {
        self.last_lr
    }
}

/// Exponential decay scheduler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialScheduler {
    /// Initial learning rate
    initial_lr: f64,
    /// Decay rate per step
    decay_rate: f64,
    /// Decay every N steps
    decay_steps: usize,
    /// Last computed learning rate
    last_lr: f64,
}

impl ExponentialScheduler {
    pub fn new(initial_lr: f64, decay_rate: f64, decay_steps: usize) -> Self {
        Self {
            initial_lr,
            decay_rate,
            decay_steps,
            last_lr: initial_lr,
        }
    }
}

impl LRScheduler for ExponentialScheduler {
    fn get_lr(&self, step: usize) -> f64 {
        // A zero decay period would divide by zero; treat it as "decay every
        // step" rather than panicking.
        let num_decays = step / self.decay_steps.max(1);
        self.initial_lr * self.decay_rate.powi(num_decays as i32)
    }

    fn last_lr(&self) -> f64 {
        self.last_lr
    }
}

/// One-cycle learning rate policy
///
/// Increases LR from initial to max over warmup, then decreases to final over remaining steps.
/// Popular for super-convergence training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneCycleScheduler {
    /// Initial learning rate
    initial_lr: f64,
    /// Maximum learning rate
    max_lr: f64,
    /// Final learning rate
    final_lr: f64,
    /// Total number of steps
    total_steps: usize,
    /// Percentage of steps for warmup (0.0 to 1.0)
    warmup_pct: f64,
    /// Last computed learning rate
    last_lr: f64,
}

impl OneCycleScheduler {
    pub fn new(max_lr: f64, total_steps: usize) -> Self {
        Self {
            initial_lr: max_lr / 25.0,
            max_lr,
            final_lr: max_lr / 10000.0,
            total_steps,
            warmup_pct: 0.3,
            last_lr: max_lr / 25.0,
        }
    }

    pub fn with_warmup_pct(mut self, warmup_pct: f64) -> Self {
        self.warmup_pct = warmup_pct.clamp(0.0, 1.0);
        self
    }

    pub fn with_div_factor(mut self, div_factor: f64) -> Self {
        self.initial_lr = self.max_lr / div_factor;
        self
    }

    pub fn with_final_div_factor(mut self, final_div_factor: f64) -> Self {
        self.final_lr = self.max_lr / final_div_factor;
        self
    }
}

impl LRScheduler for OneCycleScheduler {
    fn get_lr(&self, step: usize) -> f64 {
        let warmup_steps = (self.total_steps as f64 * self.warmup_pct) as usize;

        if step < warmup_steps {
            // Increase from initial to max
            let progress = step as f64 / warmup_steps as f64;
            self.initial_lr + (self.max_lr - self.initial_lr) * progress
        } else {
            // Decrease from max to final. `warmup_pct == 1.0` makes the decay
            // window empty, so floor it at one step to avoid a 0/0 NaN.
            let decay_steps = self.total_steps.saturating_sub(warmup_steps).max(1) as f64;
            let progress = ((step - warmup_steps) as f64 / decay_steps).clamp(0.0, 1.0);
            let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
            self.final_lr + (self.max_lr - self.final_lr) * cosine
        }
    }

    fn last_lr(&self) -> f64 {
        self.last_lr
    }
}

/// Polynomial decay scheduler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialScheduler {
    /// Initial learning rate
    initial_lr: f64,
    /// Final learning rate
    final_lr: f64,
    /// Total number of steps
    total_steps: usize,
    /// Power of polynomial (2.0 for quadratic)
    power: f64,
    /// Last computed learning rate
    last_lr: f64,
}

impl PolynomialScheduler {
    pub fn new(initial_lr: f64, final_lr: f64, total_steps: usize, power: f64) -> Self {
        Self {
            initial_lr,
            final_lr,
            total_steps,
            power,
            last_lr: initial_lr,
        }
    }
}

impl LRScheduler for PolynomialScheduler {
    fn get_lr(&self, step: usize) -> f64 {
        if step >= self.total_steps {
            return self.final_lr;
        }

        let progress = step as f64 / self.total_steps as f64;
        let decay = (1.0 - progress).powf(self.power);
        self.final_lr + (self.initial_lr - self.final_lr) * decay
    }

    fn last_lr(&self) -> f64 {
        self.last_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let scheduler = ConstantScheduler::new(1e-3);
        assert_eq!(scheduler.get_lr(0), 1e-3);
        assert_eq!(scheduler.get_lr(100), 1e-3);
        assert_eq!(scheduler.get_lr(1000), 1e-3);
    }

    #[test]
    fn test_linear_scheduler() {
        let scheduler = LinearScheduler::new(1e-3, 1e-5, 1000, 100);

        // Warmup phase
        assert!(scheduler.get_lr(0) < scheduler.get_lr(50));
        assert!(scheduler.get_lr(50) < scheduler.get_lr(100));

        // Decay phase
        assert!(scheduler.get_lr(500) > scheduler.get_lr(750));
        assert!(scheduler.get_lr(750) > scheduler.get_lr(1000));
    }

    #[test]
    fn test_cosine_scheduler() {
        let scheduler = CosineScheduler::new(1e-3, 1000, 100).with_min_lr(1e-5);

        // Warmup phase
        let lr_0 = scheduler.get_lr(0);
        let lr_50 = scheduler.get_lr(50);
        let lr_100 = scheduler.get_lr(100);
        assert!(lr_0 < lr_50);
        assert!(lr_50 < lr_100);

        // Cosine decay
        let lr_500 = scheduler.get_lr(500);
        let lr_1000 = scheduler.get_lr(1000);
        assert!(lr_500 > lr_1000);

        // Should approach min_lr at the end
        assert!((lr_1000 - 1e-5).abs() < 1e-6);
    }

    #[test]
    fn test_step_scheduler() {
        let scheduler = StepScheduler::new(1.0, 0.1, vec![100, 200, 300]);

        assert!((scheduler.get_lr(0) - 1.0).abs() < 1e-10);
        assert!((scheduler.get_lr(99) - 1.0).abs() < 1e-10);
        assert!((scheduler.get_lr(100) - 0.1).abs() < 1e-10);
        assert!((scheduler.get_lr(199) - 0.1).abs() < 1e-10);
        assert!((scheduler.get_lr(200) - 0.01).abs() < 1e-10);
        assert!((scheduler.get_lr(300) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_scheduler() {
        let scheduler = ExponentialScheduler::new(1.0, 0.96, 100);

        let lr_0 = scheduler.get_lr(0);
        let lr_100 = scheduler.get_lr(100);
        let lr_200 = scheduler.get_lr(200);

        assert_eq!(lr_0, 1.0);
        assert!((lr_100 - 0.96).abs() < 1e-6);
        assert!((lr_200 - 0.96 * 0.96).abs() < 1e-6);
    }

    #[test]
    fn test_onecycle_scheduler() {
        let scheduler = OneCycleScheduler::new(1e-3, 1000).with_warmup_pct(0.3);

        let lr_0 = scheduler.get_lr(0);
        let lr_150 = scheduler.get_lr(150); // During warmup
        let lr_300 = scheduler.get_lr(300); // At peak
        let lr_650 = scheduler.get_lr(650); // During decay
        let lr_1000 = scheduler.get_lr(1000); // At end

        // Should increase during warmup
        assert!(lr_0 < lr_150);
        assert!(lr_150 < lr_300);

        // Should decrease after peak
        assert!(lr_300 > lr_650);
        assert!(lr_650 > lr_1000);

        // Peak should be close to max_lr
        assert!((lr_300 - 1e-3).abs() < 1e-4);
    }

    #[test]
    fn test_polynomial_scheduler() {
        let scheduler = PolynomialScheduler::new(1.0, 0.1, 1000, 2.0);

        let lr_0 = scheduler.get_lr(0);
        let lr_500 = scheduler.get_lr(500);
        let lr_1000 = scheduler.get_lr(1000);

        assert_eq!(lr_0, 1.0);
        assert!(lr_500 > lr_1000);
        assert!((lr_1000 - 0.1).abs() < 1e-6);

        // Should decay non-linearly
        let lr_250 = scheduler.get_lr(250);
        let lr_750 = scheduler.get_lr(750);
        // Due to quadratic decay, early steps should have higher LR difference
        let early_diff = lr_0 - lr_250;
        let late_diff = lr_500 - lr_750;
        assert!(early_diff > late_diff);
    }
}
