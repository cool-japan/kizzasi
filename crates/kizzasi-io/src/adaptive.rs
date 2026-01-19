//! Adaptive buffering and rate control for streams
//!
//! This module provides sophisticated buffering strategies that dynamically
//! adapt to network conditions, latency, and throughput requirements.

use crate::error::{IoError, IoResult};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Strategy for adapting buffer size
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveStrategy {
    /// Minimize latency (smaller buffers)
    LowLatency,
    /// Maximize throughput (larger buffers)
    HighThroughput,
    /// Balance latency and throughput
    Balanced,
    /// Custom thresholds
    Custom,
}

/// Configuration for adaptive buffer
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Initial buffer size
    pub initial_size: usize,
    /// Minimum buffer size
    pub min_size: usize,
    /// Maximum buffer size
    pub max_size: usize,
    /// Adaptation strategy
    pub strategy: AdaptiveStrategy,
    /// Target latency (for low latency mode)
    pub target_latency: Duration,
    /// Underrun threshold (0.0 to 1.0)
    pub underrun_threshold: f32,
    /// Overrun threshold (0.0 to 1.0)
    pub overrun_threshold: f32,
    /// Adjustment step size (samples)
    pub adjustment_step: usize,
    /// Measurement window for statistics
    pub measurement_window: Duration,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            initial_size: 4096,
            min_size: 1024,
            max_size: 16384,
            strategy: AdaptiveStrategy::Balanced,
            target_latency: Duration::from_millis(50),
            underrun_threshold: 0.2,
            overrun_threshold: 0.8,
            adjustment_step: 512,
            measurement_window: Duration::from_secs(1),
        }
    }
}

impl AdaptiveConfig {
    /// Create low latency configuration
    pub fn low_latency() -> Self {
        Self {
            initial_size: 2048,
            min_size: 512,
            max_size: 8192,
            strategy: AdaptiveStrategy::LowLatency,
            target_latency: Duration::from_millis(20),
            underrun_threshold: 0.3,
            overrun_threshold: 0.7,
            adjustment_step: 256,
            ..Default::default()
        }
    }

    /// Create high throughput configuration
    pub fn high_throughput() -> Self {
        Self {
            initial_size: 8192,
            min_size: 4096,
            max_size: 32768,
            strategy: AdaptiveStrategy::HighThroughput,
            target_latency: Duration::from_millis(100),
            underrun_threshold: 0.1,
            overrun_threshold: 0.9,
            adjustment_step: 1024,
            ..Default::default()
        }
    }
}

/// Adaptive buffer that adjusts size based on usage patterns
pub struct AdaptiveBuffer<T> {
    buffer: VecDeque<T>,
    config: AdaptiveConfig,
    current_capacity: usize,
    stats: BufferStats,
}

/// Statistics for buffer performance
#[derive(Debug, Clone)]
struct BufferStats {
    total_writes: u64,
    total_reads: u64,
    underruns: u64,
    overruns: u64,
    last_adjustment: Instant,
    fill_levels: VecDeque<f32>,
    latencies: VecDeque<Duration>,
}

impl BufferStats {
    fn new() -> Self {
        Self {
            total_writes: 0,
            total_reads: 0,
            underruns: 0,
            overruns: 0,
            last_adjustment: Instant::now(),
            fill_levels: VecDeque::with_capacity(100),
            latencies: VecDeque::with_capacity(100),
        }
    }

    fn record_fill_level(&mut self, level: f32) {
        self.fill_levels.push_back(level);
        if self.fill_levels.len() > 100 {
            self.fill_levels.pop_front();
        }
    }

    fn average_fill_level(&self) -> f32 {
        if self.fill_levels.is_empty() {
            return 0.0;
        }
        self.fill_levels.iter().sum::<f32>() / self.fill_levels.len() as f32
    }

    #[allow(dead_code)]
    fn record_latency(&mut self, latency: Duration) {
        self.latencies.push_back(latency);
        if self.latencies.len() > 100 {
            self.latencies.pop_front();
        }
    }

    fn average_latency(&self) -> Duration {
        if self.latencies.is_empty() {
            return Duration::from_secs(0);
        }
        let total: Duration = self.latencies.iter().sum();
        total / self.latencies.len() as u32
    }
}

impl<T> AdaptiveBuffer<T> {
    /// Create new adaptive buffer with configuration
    pub fn new(config: AdaptiveConfig) -> Self {
        let capacity = config.initial_size;
        Self {
            buffer: VecDeque::with_capacity(capacity),
            config,
            current_capacity: capacity,
            stats: BufferStats::new(),
        }
    }

    /// Create with default configuration
    pub fn with_capacity(capacity: usize) -> Self {
        let config = AdaptiveConfig {
            initial_size: capacity,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Push item to buffer
    pub fn push(&mut self, item: T) -> IoResult<()> {
        let fill_level = self.fill_level();
        self.stats.record_fill_level(fill_level);
        self.stats.total_writes += 1;

        if self.buffer.len() >= self.current_capacity {
            self.stats.overruns += 1;
            if fill_level > self.config.overrun_threshold {
                self.try_grow()?;
            }
            return Err(IoError::BufferFull);
        }

        self.buffer.push_back(item);
        Ok(())
    }

    /// Pop item from buffer
    pub fn pop(&mut self) -> IoResult<T> {
        let fill_level = self.fill_level();
        self.stats.record_fill_level(fill_level);
        self.stats.total_reads += 1;

        match self.buffer.pop_front() {
            Some(item) => {
                if fill_level < self.config.underrun_threshold {
                    self.try_shrink();
                }
                Ok(item)
            }
            None => {
                self.stats.underruns += 1;
                Err(IoError::BufferEmpty)
            }
        }
    }

    /// Get current fill level (0.0 to 1.0)
    pub fn fill_level(&self) -> f32 {
        if self.current_capacity == 0 {
            return 0.0;
        }
        self.buffer.len() as f32 / self.current_capacity as f32
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get current capacity
    pub fn capacity(&self) -> usize {
        self.current_capacity
    }

    /// Get buffer statistics
    pub fn stats(&self) -> AdaptiveBufferStats {
        AdaptiveBufferStats {
            total_writes: self.stats.total_writes,
            total_reads: self.stats.total_reads,
            underruns: self.stats.underruns,
            overruns: self.stats.overruns,
            current_size: self.buffer.len(),
            current_capacity: self.current_capacity,
            average_fill_level: self.stats.average_fill_level(),
            average_latency: self.stats.average_latency(),
        }
    }

    /// Try to grow buffer capacity
    fn try_grow(&mut self) -> IoResult<()> {
        let elapsed = self.stats.last_adjustment.elapsed();
        if elapsed < self.config.measurement_window {
            return Ok(());
        }

        let new_capacity =
            (self.current_capacity + self.config.adjustment_step).min(self.config.max_size);

        if new_capacity > self.current_capacity {
            self.current_capacity = new_capacity;
            self.buffer.reserve(self.config.adjustment_step);
            self.stats.last_adjustment = Instant::now();
            tracing::debug!(
                "Grew adaptive buffer to {} (from {})",
                new_capacity,
                self.current_capacity
            );
        }

        Ok(())
    }

    /// Try to shrink buffer capacity
    fn try_shrink(&mut self) {
        let elapsed = self.stats.last_adjustment.elapsed();
        if elapsed < self.config.measurement_window {
            return;
        }

        let new_capacity = (self
            .current_capacity
            .saturating_sub(self.config.adjustment_step))
        .max(self.config.min_size);

        if new_capacity < self.current_capacity {
            self.current_capacity = new_capacity;
            self.stats.last_adjustment = Instant::now();
            tracing::debug!(
                "Shrunk adaptive buffer to {} (from {})",
                new_capacity,
                self.current_capacity
            );
        }
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Statistics about adaptive buffer performance
#[derive(Debug, Clone)]
pub struct AdaptiveBufferStats {
    pub total_writes: u64,
    pub total_reads: u64,
    pub underruns: u64,
    pub overruns: u64,
    pub current_size: usize,
    pub current_capacity: usize,
    pub average_fill_level: f32,
    pub average_latency: Duration,
}

/// Rate limiter for controlling data flow
pub struct RateLimiter {
    /// Target rate (samples per second)
    target_rate: f32,
    /// Current accumulated tokens
    tokens: f32,
    /// Maximum burst size
    max_burst: f32,
    /// Last update time
    last_update: Instant,
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new(samples_per_second: f32) -> Self {
        Self {
            target_rate: samples_per_second,
            tokens: 0.0,
            max_burst: samples_per_second,
            last_update: Instant::now(),
        }
    }

    /// Create with custom burst size
    pub fn with_burst(samples_per_second: f32, max_burst: f32) -> Self {
        Self {
            target_rate: samples_per_second,
            tokens: 0.0,
            max_burst,
            last_update: Instant::now(),
        }
    }

    /// Try to consume tokens for n samples
    /// Returns true if allowed, false if rate limited
    pub fn try_consume(&mut self, n_samples: usize) -> bool {
        self.update_tokens();

        if self.tokens >= n_samples as f32 {
            self.tokens -= n_samples as f32;
            true
        } else {
            false
        }
    }

    /// Wait until tokens are available (async)
    pub async fn consume(&mut self, n_samples: usize) {
        while !self.try_consume(n_samples) {
            let deficit = n_samples as f32 - self.tokens;
            let wait_time = Duration::from_secs_f32(deficit / self.target_rate);
            tokio::time::sleep(wait_time).await;
        }
    }

    /// Update token bucket based on elapsed time
    fn update_tokens(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f32();
        self.last_update = now;

        self.tokens = (self.tokens + elapsed * self.target_rate).min(self.max_burst);
    }

    /// Get current token count
    pub fn available_tokens(&mut self) -> f32 {
        self.update_tokens();
        self.tokens
    }

    /// Set new target rate
    pub fn set_rate(&mut self, samples_per_second: f32) {
        self.target_rate = samples_per_second;
    }
}

/// Adaptive rate controller that adjusts based on buffer levels
pub struct AdaptiveRateController {
    rate_limiter: RateLimiter,
    target_fill_level: f32,
    min_rate: f32,
    max_rate: f32,
    adjustment_factor: f32,
}

impl AdaptiveRateController {
    /// Create new adaptive rate controller
    pub fn new(initial_rate: f32, min_rate: f32, max_rate: f32) -> Self {
        Self {
            rate_limiter: RateLimiter::new(initial_rate),
            target_fill_level: 0.5,
            min_rate,
            max_rate,
            adjustment_factor: 0.1,
        }
    }

    /// Adjust rate based on buffer fill level
    pub fn adjust_rate(&mut self, fill_level: f32) {
        let error = fill_level - self.target_fill_level;
        let current_rate = self.rate_limiter.target_rate;

        // If buffer is too full, increase rate (send faster)
        // If buffer is too empty, decrease rate (send slower)
        let adjustment = -error * self.adjustment_factor * current_rate;
        let new_rate = (current_rate + adjustment).clamp(self.min_rate, self.max_rate);

        self.rate_limiter.set_rate(new_rate);
    }

    /// Try to consume tokens
    pub fn try_consume(&mut self, n_samples: usize) -> bool {
        self.rate_limiter.try_consume(n_samples)
    }

    /// Wait and consume tokens
    pub async fn consume(&mut self, n_samples: usize) {
        self.rate_limiter.consume(n_samples).await;
    }

    /// Get current rate
    pub fn current_rate(&self) -> f32 {
        self.rate_limiter.target_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_buffer_basic() {
        let mut buffer = AdaptiveBuffer::with_capacity(10);

        // Push items
        for i in 0..5 {
            buffer.push(i).unwrap();
        }

        assert_eq!(buffer.len(), 5);
        assert!(buffer.fill_level() > 0.0);

        // Pop items
        for i in 0..5 {
            assert_eq!(buffer.pop().unwrap(), i);
        }

        assert!(buffer.is_empty());
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(1000.0); // 1000 samples/sec

        // Should not be able to consume 2000 samples immediately
        assert!(!limiter.try_consume(2000));

        // Should be able to consume 500 samples
        std::thread::sleep(Duration::from_millis(500));
        assert!(limiter.try_consume(500));
    }

    #[test]
    fn test_adaptive_config() {
        let config = AdaptiveConfig::low_latency();
        assert_eq!(config.strategy, AdaptiveStrategy::LowLatency);
        assert!(config.min_size < config.initial_size);
        assert!(config.initial_size < config.max_size);
    }

    #[tokio::test]
    async fn test_rate_limiter_async() {
        let mut limiter = RateLimiter::new(1000.0); // 1000 samples/sec

        let start = Instant::now();
        limiter.consume(100).await; // 100 samples at 1000/sec = 100ms max
        let elapsed = start.elapsed();

        // Should complete within reasonable time
        // Allow some tolerance for test environment variability
        assert!(elapsed < Duration::from_millis(500));
    }
}
