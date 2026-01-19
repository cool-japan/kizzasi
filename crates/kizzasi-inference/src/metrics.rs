//! Performance metrics and profiling for inference
//!
//! This module provides tools for monitoring and profiling inference performance:
//! - Latency tracking
//! - Throughput measurement
//! - Resource utilization
//! - Bottleneck identification

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Performance metrics for inference operations
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Total number of inference steps
    pub total_steps: u64,
    /// Total time spent in inference (microseconds)
    pub total_time_us: u64,
    /// Recent latencies (for rolling statistics)
    recent_latencies: VecDeque<u64>,
    /// Maximum rolling window size
    window_size: usize,
    /// Peak latency observed (microseconds)
    pub peak_latency_us: u64,
    /// Minimum latency observed (microseconds)
    pub min_latency_us: u64,
    /// Start time
    start_time: Instant,
}

impl InferenceMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self::with_window_size(100)
    }

    /// Create with custom window size for rolling statistics
    pub fn with_window_size(window_size: usize) -> Self {
        Self {
            total_steps: 0,
            total_time_us: 0,
            recent_latencies: VecDeque::with_capacity(window_size),
            window_size,
            peak_latency_us: 0,
            min_latency_us: u64::MAX,
            start_time: Instant::now(),
        }
    }

    /// Record a new inference step
    pub fn record_step(&mut self, latency_us: u64) {
        self.total_steps += 1;
        self.total_time_us += latency_us;

        // Update peak and min
        self.peak_latency_us = self.peak_latency_us.max(latency_us);
        if latency_us > 0 {
            self.min_latency_us = self.min_latency_us.min(latency_us);
        }

        // Rolling window
        if self.recent_latencies.len() >= self.window_size {
            self.recent_latencies.pop_front();
        }
        self.recent_latencies.push_back(latency_us);
    }

    /// Get average latency (microseconds)
    pub fn avg_latency_us(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.total_time_us as f64 / self.total_steps as f64
        }
    }

    /// Get recent average latency (microseconds)
    pub fn recent_avg_latency_us(&self) -> f64 {
        if self.recent_latencies.is_empty() {
            0.0
        } else {
            let sum: u64 = self.recent_latencies.iter().sum();
            sum as f64 / self.recent_latencies.len() as f64
        }
    }

    /// Get throughput (steps per second)
    pub fn throughput(&self) -> f64 {
        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        if elapsed_secs == 0.0 {
            0.0
        } else {
            self.total_steps as f64 / elapsed_secs
        }
    }

    /// Get p50, p95, p99 latencies (microseconds)
    pub fn percentiles(&self) -> (u64, u64, u64) {
        if self.recent_latencies.is_empty() {
            return (0, 0, 0);
        }

        let mut sorted: Vec<u64> = self.recent_latencies.iter().copied().collect();
        sorted.sort_unstable();

        let p50_idx = (sorted.len() as f64 * 0.50) as usize;
        let p95_idx = (sorted.len() as f64 * 0.95) as usize;
        let p99_idx = (sorted.len() as f64 * 0.99) as usize;

        (
            sorted.get(p50_idx).copied().unwrap_or(0),
            sorted.get(p95_idx).copied().unwrap_or(0),
            sorted.get(p99_idx).copied().unwrap_or(0),
        )
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.total_steps = 0;
        self.total_time_us = 0;
        self.recent_latencies.clear();
        self.peak_latency_us = 0;
        self.min_latency_us = u64::MAX;
        self.start_time = Instant::now();
    }

    /// Get a summary report
    pub fn summary(&self) -> MetricsSummary {
        let (p50, p95, p99) = self.percentiles();

        MetricsSummary {
            total_steps: self.total_steps,
            avg_latency_us: self.avg_latency_us(),
            recent_avg_latency_us: self.recent_avg_latency_us(),
            peak_latency_us: self.peak_latency_us,
            min_latency_us: if self.min_latency_us == u64::MAX {
                0
            } else {
                self.min_latency_us
            },
            throughput_per_sec: self.throughput(),
            p50_latency_us: p50,
            p95_latency_us: p95,
            p99_latency_us: p99,
            uptime_secs: self.start_time.elapsed().as_secs_f64(),
        }
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of performance metrics
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub total_steps: u64,
    pub avg_latency_us: f64,
    pub recent_avg_latency_us: f64,
    pub peak_latency_us: u64,
    pub min_latency_us: u64,
    pub throughput_per_sec: f64,
    pub p50_latency_us: u64,
    pub p95_latency_us: u64,
    pub p99_latency_us: u64,
    pub uptime_secs: f64,
}

impl MetricsSummary {
    /// Print a formatted report
    pub fn print_report(&self) {
        println!("=== Inference Performance Report ===");
        println!("Total steps: {}", self.total_steps);
        println!("Uptime: {:.2}s", self.uptime_secs);
        println!("Throughput: {:.2} steps/sec", self.throughput_per_sec);
        println!("\nLatency (microseconds):");
        println!("  Average: {:.2} µs", self.avg_latency_us);
        println!("  Recent avg: {:.2} µs", self.recent_avg_latency_us);
        println!("  Min: {} µs", self.min_latency_us);
        println!("  Max: {} µs", self.peak_latency_us);
        println!("\nPercentiles:");
        println!("  P50: {} µs", self.p50_latency_us);
        println!("  P95: {} µs", self.p95_latency_us);
        println!("  P99: {} µs", self.p99_latency_us);
        println!("=====================================");
    }
}

/// Timer for measuring operation duration
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Get elapsed time as Duration
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Profiler for tracking different stages of inference
#[derive(Debug, Clone)]
pub struct InferenceProfiler {
    /// Time spent in tokenization (microseconds)
    pub tokenization_us: u64,
    /// Time spent in model forward pass (microseconds)
    pub forward_pass_us: u64,
    /// Time spent in sampling (microseconds)
    pub sampling_us: u64,
    /// Time spent in constraint enforcement (microseconds)
    pub constraints_us: u64,
    /// Time spent in detokenization (microseconds)
    pub detokenization_us: u64,
    /// Number of profiled steps
    pub step_count: u64,
}

impl InferenceProfiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            tokenization_us: 0,
            forward_pass_us: 0,
            sampling_us: 0,
            constraints_us: 0,
            detokenization_us: 0,
            step_count: 0,
        }
    }

    /// Record tokenization time
    pub fn record_tokenization(&mut self, duration_us: u64) {
        self.tokenization_us += duration_us;
    }

    /// Record forward pass time
    pub fn record_forward_pass(&mut self, duration_us: u64) {
        self.forward_pass_us += duration_us;
    }

    /// Record sampling time
    pub fn record_sampling(&mut self, duration_us: u64) {
        self.sampling_us += duration_us;
    }

    /// Record constraint enforcement time
    pub fn record_constraints(&mut self, duration_us: u64) {
        self.constraints_us += duration_us;
    }

    /// Record detokenization time
    pub fn record_detokenization(&mut self, duration_us: u64) {
        self.detokenization_us += duration_us;
    }

    /// Increment step count
    pub fn increment_step(&mut self) {
        self.step_count += 1;
    }

    /// Get total time across all stages
    pub fn total_time_us(&self) -> u64 {
        self.tokenization_us
            + self.forward_pass_us
            + self.sampling_us
            + self.constraints_us
            + self.detokenization_us
    }

    /// Get breakdown percentages
    pub fn breakdown(&self) -> ProfileBreakdown {
        let total = self.total_time_us() as f64;
        if total == 0.0 {
            return ProfileBreakdown::default();
        }

        ProfileBreakdown {
            tokenization_pct: (self.tokenization_us as f64 / total) * 100.0,
            forward_pass_pct: (self.forward_pass_us as f64 / total) * 100.0,
            sampling_pct: (self.sampling_us as f64 / total) * 100.0,
            constraints_pct: (self.constraints_us as f64 / total) * 100.0,
            detokenization_pct: (self.detokenization_us as f64 / total) * 100.0,
        }
    }

    /// Print profiling report
    pub fn print_report(&self) {
        let breakdown = self.breakdown();
        let total = self.total_time_us();

        println!("=== Inference Profiling Report ===");
        println!("Total steps: {}", self.step_count);
        println!("Total time: {} µs ({:.2} ms)", total, total as f64 / 1000.0);
        println!("\nTime breakdown:");
        println!(
            "  Tokenization:   {:>8} µs ({:>5.2}%)",
            self.tokenization_us, breakdown.tokenization_pct
        );
        println!(
            "  Forward pass:   {:>8} µs ({:>5.2}%)",
            self.forward_pass_us, breakdown.forward_pass_pct
        );
        println!(
            "  Sampling:       {:>8} µs ({:>5.2}%)",
            self.sampling_us, breakdown.sampling_pct
        );
        println!(
            "  Constraints:    {:>8} µs ({:>5.2}%)",
            self.constraints_us, breakdown.constraints_pct
        );
        println!(
            "  Detokenization: {:>8} µs ({:>5.2}%)",
            self.detokenization_us, breakdown.detokenization_pct
        );
        println!("===================================");
    }

    /// Reset the profiler
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for InferenceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Breakdown of time spent in different stages
#[derive(Debug, Clone, Default)]
pub struct ProfileBreakdown {
    pub tokenization_pct: f64,
    pub forward_pass_pct: f64,
    pub sampling_pct: f64,
    pub constraints_pct: f64,
    pub detokenization_pct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_metrics_creation() {
        let metrics = InferenceMetrics::new();
        assert_eq!(metrics.total_steps, 0);
        assert_eq!(metrics.total_time_us, 0);
    }

    #[test]
    fn test_record_step() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_step(1000);
        metrics.record_step(2000);

        assert_eq!(metrics.total_steps, 2);
        assert_eq!(metrics.total_time_us, 3000);
        assert_eq!(metrics.avg_latency_us(), 1500.0);
    }

    #[test]
    fn test_percentiles() {
        let mut metrics = InferenceMetrics::new();
        for i in 1..=100 {
            metrics.record_step(i * 100);
        }

        let (p50, p95, p99) = metrics.percentiles();
        assert!(p50 > 4000 && p50 < 6000);
        assert!(p95 > 9000);
        assert!(p99 > 9800);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        thread::sleep(Duration::from_micros(100));
        let elapsed = timer.elapsed_us();

        assert!(elapsed >= 100);
    }

    #[test]
    fn test_profiler() {
        let mut profiler = InferenceProfiler::new();

        profiler.record_tokenization(100);
        profiler.record_forward_pass(500);
        profiler.record_sampling(50);
        profiler.increment_step();

        assert_eq!(profiler.total_time_us(), 650);
        assert_eq!(profiler.step_count, 1);

        let breakdown = profiler.breakdown();
        assert!((breakdown.tokenization_pct - 15.38).abs() < 0.1);
        assert!((breakdown.forward_pass_pct - 76.92).abs() < 0.1);
    }

    #[test]
    fn test_metrics_summary() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_step(1000);
        metrics.record_step(2000);
        metrics.record_step(1500);

        let summary = metrics.summary();
        assert_eq!(summary.total_steps, 3);
        assert_eq!(summary.min_latency_us, 1000);
        assert_eq!(summary.peak_latency_us, 2000);
    }
}
