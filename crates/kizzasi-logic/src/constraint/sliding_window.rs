use serde::{Deserialize, Serialize};

// ============================================================================
// Sliding Window Constraints
// ============================================================================

/// Sliding window constraint for time-series data
///
/// Enforces constraints over a sliding window of recent values.
/// Useful for smoothness, bounded variation, and statistical constraints.
#[derive(Debug, Clone)]
pub struct SlidingWindowConstraint {
    name: String,
    window_size: usize,
    constraint_fn: SlidingWindowFn,
    buffer: Vec<f32>,
    weight: f32,
}

/// Types of sliding window constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlidingWindowFn {
    /// Mean of window must be in range [lo, hi]
    MeanInRange { lo: f32, hi: f32 },
    /// Variance of window must be <= max_var
    MaxVariance { max_var: f32 },
    /// Range (max - min) of window must be <= max_range
    MaxRange { max_range: f32 },
    /// Sum of absolute differences must be <= max_variation
    BoundedVariation { max_variation: f32 },
    /// All values in window must be in range [lo, hi]
    AllInRange { lo: f32, hi: f32 },
    /// At least one value in window must satisfy the range
    AnyInRange { lo: f32, hi: f32 },
    /// Trend constraint: regression slope must be in range
    TrendInRange { min_slope: f32, max_slope: f32 },
}

impl SlidingWindowConstraint {
    /// Create a new sliding window constraint
    pub fn new(name: &str, window_size: usize, constraint_fn: SlidingWindowFn) -> Self {
        Self {
            name: name.to_string(),
            window_size,
            constraint_fn,
            buffer: Vec::with_capacity(window_size),
            weight: 1.0,
        }
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Push a new value and check constraint
    ///
    /// Returns (satisfied, violation)
    pub fn push_and_check(&mut self, value: f32) -> (bool, f32) {
        // Add to buffer
        self.buffer.push(value);
        if self.buffer.len() > self.window_size {
            self.buffer.remove(0);
        }

        // Not enough data yet
        if self.buffer.len() < self.window_size {
            return (true, 0.0);
        }

        self.check_window()
    }

    /// Check constraint on current window
    pub(crate) fn check_window(&self) -> (bool, f32) {
        match &self.constraint_fn {
            SlidingWindowFn::MeanInRange { lo, hi } => {
                let mean: f32 = self.buffer.iter().sum::<f32>() / self.buffer.len() as f32;
                if mean >= *lo && mean <= *hi {
                    (true, 0.0)
                } else if mean < *lo {
                    (false, lo - mean)
                } else {
                    (false, mean - hi)
                }
            }
            SlidingWindowFn::MaxVariance { max_var } => {
                let n = self.buffer.len() as f32;
                let mean: f32 = self.buffer.iter().sum::<f32>() / n;
                let var: f32 = self.buffer.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
                if var <= *max_var {
                    (true, 0.0)
                } else {
                    (false, var - max_var)
                }
            }
            SlidingWindowFn::MaxRange { max_range } => {
                let min = self.buffer.iter().cloned().reduce(f32::min).unwrap_or(0.0);
                let max = self.buffer.iter().cloned().reduce(f32::max).unwrap_or(0.0);
                let range = max - min;
                if range <= *max_range {
                    (true, 0.0)
                } else {
                    (false, range - max_range)
                }
            }
            SlidingWindowFn::BoundedVariation { max_variation } => {
                let variation: f32 = self.buffer.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
                if variation <= *max_variation {
                    (true, 0.0)
                } else {
                    (false, variation - max_variation)
                }
            }
            SlidingWindowFn::AllInRange { lo, hi } => {
                let violation: f32 = self
                    .buffer
                    .iter()
                    .map(|&x| {
                        if x < *lo {
                            lo - x
                        } else if x > *hi {
                            x - hi
                        } else {
                            0.0
                        }
                    })
                    .sum();
                (violation == 0.0, violation)
            }
            SlidingWindowFn::AnyInRange { lo, hi } => {
                let any_in_range = self.buffer.iter().any(|&x| x >= *lo && x <= *hi);
                if any_in_range {
                    (true, 0.0)
                } else {
                    // Violation is minimum distance to range
                    let min_dist = self
                        .buffer
                        .iter()
                        .map(|&x| if x < *lo { lo - x } else { x - hi })
                        .reduce(f32::min)
                        .unwrap_or(0.0);
                    (false, min_dist)
                }
            }
            SlidingWindowFn::TrendInRange {
                min_slope,
                max_slope,
            } => {
                // Simple linear regression: slope = Cov(i, x) / Var(i)
                let n = self.buffer.len() as f32;
                let mean_i = (n - 1.0) / 2.0;
                let mean_x: f32 = self.buffer.iter().sum::<f32>() / n;

                let cov: f32 = self
                    .buffer
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| (i as f32 - mean_i) * (x - mean_x))
                    .sum();
                let var_i: f32 = (0..self.buffer.len())
                    .map(|i| (i as f32 - mean_i).powi(2))
                    .sum();

                let slope = if var_i > f32::EPSILON {
                    cov / var_i
                } else {
                    0.0
                };

                if slope >= *min_slope && slope <= *max_slope {
                    (true, 0.0)
                } else if slope < *min_slope {
                    (false, min_slope - slope)
                } else {
                    (false, slope - max_slope)
                }
            }
        }
    }

    /// Reset the buffer
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Get current window contents
    pub fn window(&self) -> &[f32] {
        &self.buffer
    }

    /// Get constraint name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get window size
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get weight
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Check if buffer is full
    pub fn is_ready(&self) -> bool {
        self.buffer.len() >= self.window_size
    }
}

/// Collection of sliding window constraints
#[derive(Debug, Default)]
pub struct SlidingWindowChecker {
    constraints: Vec<SlidingWindowConstraint>,
}

impl SlidingWindowChecker {
    /// Create a new checker
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a constraint
    pub fn add(&mut self, constraint: SlidingWindowConstraint) {
        self.constraints.push(constraint);
    }

    /// Push a value and check all constraints
    pub fn push_and_check(&mut self, value: f32) -> Vec<(String, bool, f32)> {
        self.constraints
            .iter_mut()
            .map(|c| {
                let (sat, viol) = c.push_and_check(value);
                (c.name().to_string(), sat, viol * c.weight())
            })
            .collect()
    }

    /// Total weighted violation
    pub fn total_violation(&self) -> f32 {
        self.constraints
            .iter()
            .filter(|c| c.is_ready())
            .map(|c| {
                let (_, viol) = c.check_window();
                viol * c.weight()
            })
            .sum()
    }

    /// Reset all constraints
    pub fn reset(&mut self) {
        for c in &mut self.constraints {
            c.reset();
        }
    }

    /// Check if all constraints are satisfied
    pub fn all_satisfied(&self) -> bool {
        self.constraints
            .iter()
            .filter(|c| c.is_ready())
            .all(|c| c.check_window().0)
    }
}
