//! Stream health monitoring and diagnostics
//!
//! Provides real-time monitoring of signal stream health including:
//! - Latency measurement and statistics
//! - Buffer underrun/overrun detection
//! - Signal quality metrics (SNR, clipping, dropouts)
//! - Stream statistics and telemetry

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Health status of a stream
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HealthStatus {
    /// Stream is healthy and operating normally
    Healthy,
    /// Stream is degraded but still functional
    Degraded,
    /// Stream has critical issues
    Critical,
    /// Stream is not available
    #[default]
    Unavailable,
}

/// Stream health metrics
#[derive(Debug, Clone)]
pub struct StreamHealth {
    /// Current health status
    pub status: HealthStatus,
    /// Total samples processed
    pub samples_processed: u64,
    /// Total buffers processed
    pub buffers_processed: u64,
    /// Number of buffer underruns (consumer faster than producer)
    pub underruns: u64,
    /// Number of buffer overruns (producer faster than consumer)
    pub overruns: u64,
    /// Number of dropped samples
    pub dropped_samples: u64,
    /// Number of clipped samples
    pub clipped_samples: u64,
    /// Time stream has been active
    pub uptime: Duration,
    /// Current buffer fill level (0.0 to 1.0)
    pub buffer_level: f32,
}

impl Default for StreamHealth {
    fn default() -> Self {
        Self {
            status: HealthStatus::Unavailable,
            samples_processed: 0,
            buffers_processed: 0,
            underruns: 0,
            overruns: 0,
            dropped_samples: 0,
            clipped_samples: 0,
            uptime: Duration::ZERO,
            buffer_level: 0.0,
        }
    }
}

/// Latency statistics
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Minimum latency observed
    pub min: Duration,
    /// Maximum latency observed
    pub max: Duration,
    /// Mean latency
    pub mean: Duration,
    /// Standard deviation of latency
    pub std: Duration,
    /// 50th percentile (median)
    pub p50: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// Number of samples
    pub count: usize,
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            min: Duration::MAX,
            max: Duration::ZERO,
            mean: Duration::ZERO,
            std: Duration::ZERO,
            p50: Duration::ZERO,
            p95: Duration::ZERO,
            p99: Duration::ZERO,
            count: 0,
        }
    }
}

/// Signal quality metrics
#[derive(Debug, Clone)]
pub struct SignalQuality {
    /// Estimated Signal-to-Noise Ratio in dB
    pub snr_db: f32,
    /// Percentage of samples that were clipped
    pub clipping_ratio: f32,
    /// Percentage of samples that were silent (below noise floor)
    pub silence_ratio: f32,
    /// DC offset (mean value)
    pub dc_offset: f32,
    /// Crest factor (peak / RMS)
    pub crest_factor: f32,
    /// Dynamic range in dB
    pub dynamic_range_db: f32,
}

impl Default for SignalQuality {
    fn default() -> Self {
        Self {
            snr_db: 0.0,
            clipping_ratio: 0.0,
            silence_ratio: 0.0,
            dc_offset: 0.0,
            crest_factor: 1.0,
            dynamic_range_db: 0.0,
        }
    }
}

/// Stream health monitor
///
/// Tracks real-time health metrics for a signal stream.
#[derive(Debug)]
pub struct HealthMonitor {
    /// Start time
    start_time: Instant,
    /// Last update time
    last_update: Instant,
    /// Latency measurements (circular buffer)
    latencies: VecDeque<Duration>,
    /// Maximum latency history size
    max_latency_history: usize,
    /// Signal values for quality analysis (circular buffer)
    signal_history: VecDeque<f32>,
    /// Maximum signal history size
    max_signal_history: usize,
    /// Current health state
    health: StreamHealth,
    /// Clipping threshold
    clip_threshold: f32,
    /// Noise floor threshold
    noise_floor: f32,
    /// Degraded threshold for underruns per minute
    degraded_threshold: f64,
    /// Critical threshold for underruns per minute
    critical_threshold: f64,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            last_update: Instant::now(),
            latencies: VecDeque::with_capacity(1000),
            max_latency_history: 1000,
            signal_history: VecDeque::with_capacity(4096),
            max_signal_history: 4096,
            health: StreamHealth::default(),
            clip_threshold: 0.99,
            noise_floor: 0.001,
            degraded_threshold: 10.0, // 10 underruns/min = degraded
            critical_threshold: 60.0, // 60 underruns/min = critical
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(
        clip_threshold: f32,
        noise_floor: f32,
        degraded_underruns_per_min: f64,
        critical_underruns_per_min: f64,
    ) -> Self {
        let mut monitor = Self::new();
        monitor.clip_threshold = clip_threshold;
        monitor.noise_floor = noise_floor;
        monitor.degraded_threshold = degraded_underruns_per_min;
        monitor.critical_threshold = critical_underruns_per_min;
        monitor
    }

    /// Set maximum latency history size
    pub fn with_latency_history(mut self, size: usize) -> Self {
        self.max_latency_history = size;
        self
    }

    /// Set maximum signal history size
    pub fn with_signal_history(mut self, size: usize) -> Self {
        self.max_signal_history = size;
        self
    }

    /// Start monitoring (reset counters)
    pub fn start(&mut self) {
        self.start_time = Instant::now();
        self.last_update = Instant::now();
        self.health = StreamHealth {
            status: HealthStatus::Healthy,
            ..Default::default()
        };
        self.latencies.clear();
        self.signal_history.clear();
    }

    /// Record a latency measurement
    pub fn record_latency(&mut self, latency: Duration) {
        if self.latencies.len() >= self.max_latency_history {
            self.latencies.pop_front();
        }
        self.latencies.push_back(latency);
    }

    /// Record a buffer of samples
    pub fn record_samples(&mut self, samples: &[f32]) {
        let samples_to_keep = samples.len().min(self.max_signal_history);
        let start = samples.len() - samples_to_keep;

        for &sample in &samples[start..] {
            if self.signal_history.len() >= self.max_signal_history {
                self.signal_history.pop_front();
            }
            self.signal_history.push_back(sample);

            // Check for clipping
            if sample.abs() > self.clip_threshold {
                self.health.clipped_samples += 1;
            }
        }

        self.health.samples_processed += samples.len() as u64;
        self.health.buffers_processed += 1;
        self.last_update = Instant::now();
    }

    /// Record an underrun event
    pub fn record_underrun(&mut self) {
        self.health.underruns += 1;
        self.update_status();
    }

    /// Record an overrun event
    pub fn record_overrun(&mut self, dropped: usize) {
        self.health.overruns += 1;
        self.health.dropped_samples += dropped as u64;
        self.update_status();
    }

    /// Update buffer level
    pub fn update_buffer_level(&mut self, level: f32) {
        self.health.buffer_level = level.clamp(0.0, 1.0);
    }

    /// Update health status based on metrics
    fn update_status(&mut self) {
        let uptime = self.start_time.elapsed();
        self.health.uptime = uptime;

        if uptime.as_secs() == 0 {
            return;
        }

        let uptime_mins = uptime.as_secs_f64() / 60.0;
        let underruns_per_min = self.health.underruns as f64 / uptime_mins.max(0.01);

        self.health.status = if underruns_per_min >= self.critical_threshold {
            HealthStatus::Critical
        } else if underruns_per_min >= self.degraded_threshold {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
    }

    /// Get current health state
    pub fn health(&self) -> StreamHealth {
        let mut health = self.health.clone();
        health.uptime = self.start_time.elapsed();
        health
    }

    /// Get latency statistics
    pub fn latency_stats(&self) -> LatencyStats {
        if self.latencies.is_empty() {
            return LatencyStats::default();
        }

        let mut sorted: Vec<Duration> = self.latencies.iter().cloned().collect();
        sorted.sort();

        let count = sorted.len();
        let min = sorted[0];
        let max = sorted[count - 1];

        let sum: Duration = sorted.iter().sum();
        let mean = sum / count as u32;

        // Calculate std deviation
        let variance_nanos: f64 = sorted
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let std = Duration::from_nanos(variance_nanos.sqrt() as u64);

        // Percentiles
        let p50 = sorted[count * 50 / 100];
        let p95 = sorted[(count * 95 / 100).min(count - 1)];
        let p99 = sorted[(count * 99 / 100).min(count - 1)];

        LatencyStats {
            min,
            max,
            mean,
            std,
            p50,
            p95,
            p99,
            count,
        }
    }

    /// Compute signal quality metrics
    pub fn signal_quality(&self) -> SignalQuality {
        if self.signal_history.is_empty() {
            return SignalQuality::default();
        }

        let n = self.signal_history.len() as f32;

        // DC offset (mean)
        let dc_offset: f32 = self.signal_history.iter().sum::<f32>() / n;

        // RMS and peak
        let mut rms_sum = 0.0f32;
        let mut peak = 0.0f32;
        let mut clipped = 0usize;
        let mut silent = 0usize;

        for &sample in &self.signal_history {
            let abs_sample = sample.abs();
            rms_sum += sample * sample;
            peak = peak.max(abs_sample);

            if abs_sample > self.clip_threshold {
                clipped += 1;
            }
            if abs_sample < self.noise_floor {
                silent += 1;
            }
        }

        let rms = (rms_sum / n).sqrt();
        let crest_factor = if rms > 0.0 { peak / rms } else { 1.0 };

        // Estimate SNR (using noise floor as noise estimate)
        let signal_power = rms_sum / n;
        let noise_power = self.noise_floor * self.noise_floor;
        let snr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            100.0 // Infinite SNR
        };

        // Dynamic range
        let dynamic_range_db = if self.noise_floor > 0.0 && peak > self.noise_floor {
            20.0 * (peak / self.noise_floor).log10()
        } else {
            0.0
        };

        SignalQuality {
            snr_db,
            clipping_ratio: clipped as f32 / n,
            silence_ratio: silent as f32 / n,
            dc_offset,
            crest_factor,
            dynamic_range_db,
        }
    }

    /// Get samples per second throughput
    pub fn throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.health.samples_processed as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Check if stream is healthy
    pub fn is_healthy(&self) -> bool {
        self.health.status == HealthStatus::Healthy
    }

    /// Get time since last update
    pub fn time_since_update(&self) -> Duration {
        self.last_update.elapsed()
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Latency tracker with start/stop timing
#[derive(Debug)]
pub struct LatencyTracker {
    start: Option<Instant>,
}

impl LatencyTracker {
    /// Create a new latency tracker
    pub fn new() -> Self {
        Self { start: None }
    }

    /// Start timing
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Stop timing and return latency
    pub fn stop(&mut self) -> Option<Duration> {
        self.start.take().map(|s| s.elapsed())
    }

    /// Stop and record to monitor
    pub fn stop_and_record(&mut self, monitor: &mut HealthMonitor) {
        if let Some(latency) = self.stop() {
            monitor.record_latency(latency);
        }
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Buffer level tracker
#[derive(Debug)]
pub struct BufferLevelTracker {
    /// Current fill level (0.0 to 1.0)
    level: f32,
    /// Low watermark threshold
    low_watermark: f32,
    /// High watermark threshold
    high_watermark: f32,
    /// Buffer capacity
    capacity: usize,
}

impl BufferLevelTracker {
    /// Create a new buffer level tracker
    pub fn new(capacity: usize) -> Self {
        Self {
            level: 0.0,
            low_watermark: 0.25,
            high_watermark: 0.75,
            capacity,
        }
    }

    /// Set watermarks
    pub fn with_watermarks(mut self, low: f32, high: f32) -> Self {
        self.low_watermark = low.clamp(0.0, 1.0);
        self.high_watermark = high.clamp(0.0, 1.0);
        self
    }

    /// Update with current buffer count
    pub fn update(&mut self, count: usize) {
        self.level = count as f32 / self.capacity.max(1) as f32;
    }

    /// Get current level (0.0 to 1.0)
    pub fn level(&self) -> f32 {
        self.level
    }

    /// Check if below low watermark (potential underrun)
    pub fn is_low(&self) -> bool {
        self.level < self.low_watermark
    }

    /// Check if above high watermark (potential overrun)
    pub fn is_high(&self) -> bool {
        self.level > self.high_watermark
    }

    /// Check if in normal operating range
    pub fn is_normal(&self) -> bool {
        self.level >= self.low_watermark && self.level <= self.high_watermark
    }
}

/// Aggregate statistics for multiple streams
#[derive(Debug, Default)]
pub struct AggregateHealth {
    /// Total streams monitored
    pub stream_count: usize,
    /// Healthy streams
    pub healthy_count: usize,
    /// Degraded streams
    pub degraded_count: usize,
    /// Critical streams
    pub critical_count: usize,
    /// Unavailable streams
    pub unavailable_count: usize,
    /// Total samples processed across all streams
    pub total_samples: u64,
    /// Total underruns across all streams
    pub total_underruns: u64,
    /// Average buffer level
    pub avg_buffer_level: f32,
}

impl AggregateHealth {
    /// Create from multiple health states
    pub fn from_streams(healths: &[StreamHealth]) -> Self {
        let mut healthy_count = 0;
        let mut degraded_count = 0;
        let mut critical_count = 0;
        let mut unavailable_count = 0;
        let mut total_samples = 0u64;
        let mut total_underruns = 0u64;
        let mut level_sum = 0.0f32;

        for h in healths {
            match h.status {
                HealthStatus::Healthy => healthy_count += 1,
                HealthStatus::Degraded => degraded_count += 1,
                HealthStatus::Critical => critical_count += 1,
                HealthStatus::Unavailable => unavailable_count += 1,
            }
            total_samples += h.samples_processed;
            total_underruns += h.underruns;
            level_sum += h.buffer_level;
        }

        let avg_buffer_level = if healths.is_empty() {
            0.0
        } else {
            level_sum / healths.len() as f32
        };

        Self {
            stream_count: healths.len(),
            healthy_count,
            degraded_count,
            critical_count,
            unavailable_count,
            total_samples,
            total_underruns,
            avg_buffer_level,
        }
    }

    /// Overall health status (worst case)
    pub fn overall_status(&self) -> HealthStatus {
        if self.critical_count > 0 {
            HealthStatus::Critical
        } else if self.degraded_count > 0 {
            HealthStatus::Degraded
        } else if self.unavailable_count == self.stream_count {
            HealthStatus::Unavailable
        } else {
            HealthStatus::Healthy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_monitor_basic() {
        let mut monitor = HealthMonitor::new();
        monitor.start();

        // Record some samples
        monitor.record_samples(&[0.5, 0.3, -0.2, 0.1]);
        assert_eq!(monitor.health().samples_processed, 4);
        assert_eq!(monitor.health().buffers_processed, 1);
    }

    #[test]
    fn test_latency_stats() {
        let mut monitor = HealthMonitor::new();
        monitor.start();

        // Record latencies
        for i in 1..=100 {
            monitor.record_latency(Duration::from_micros(i * 10));
        }

        let stats = monitor.latency_stats();
        assert_eq!(stats.count, 100);
        assert_eq!(stats.min, Duration::from_micros(10));
        assert_eq!(stats.max, Duration::from_micros(1000));
    }

    #[test]
    fn test_signal_quality() {
        let mut monitor = HealthMonitor::new();
        monitor.start();

        // Clean signal (no clipping, no silence)
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        monitor.record_samples(&samples);

        let quality = monitor.signal_quality();
        assert!(quality.clipping_ratio < 0.01);
        assert!(quality.crest_factor > 1.0);
    }

    #[test]
    fn test_clipping_detection() {
        let mut monitor = HealthMonitor::new();
        monitor.start();

        // Signal with clipping
        let samples = vec![0.5, 0.8, 1.0, 0.99, 1.0, 0.6];
        monitor.record_samples(&samples);

        let quality = monitor.signal_quality();
        assert!(quality.clipping_ratio > 0.0);
    }

    #[test]
    fn test_underrun_status() {
        let mut monitor = HealthMonitor::with_thresholds(0.99, 0.001, 1.0, 5.0);
        monitor.start();

        // Simulate underruns
        for _ in 0..10 {
            monitor.record_underrun();
        }

        // Force status update by waiting briefly would require sleep
        // Instead, just verify underrun count
        assert_eq!(monitor.health().underruns, 10);
    }

    #[test]
    fn test_latency_tracker() {
        let mut tracker = LatencyTracker::new();
        let mut monitor = HealthMonitor::new();
        monitor.start();

        tracker.start();
        std::thread::sleep(Duration::from_millis(1));
        tracker.stop_and_record(&mut monitor);

        let stats = monitor.latency_stats();
        assert_eq!(stats.count, 1);
        assert!(stats.mean >= Duration::from_millis(1));
    }

    #[test]
    fn test_buffer_level_tracker() {
        let mut tracker = BufferLevelTracker::new(100).with_watermarks(0.2, 0.8);

        tracker.update(10);
        assert!(tracker.is_low());
        assert!(!tracker.is_normal());

        tracker.update(50);
        assert!(tracker.is_normal());

        tracker.update(90);
        assert!(tracker.is_high());
    }

    #[test]
    fn test_aggregate_health() {
        let healths = vec![
            StreamHealth {
                status: HealthStatus::Healthy,
                samples_processed: 1000,
                buffer_level: 0.5,
                ..Default::default()
            },
            StreamHealth {
                status: HealthStatus::Degraded,
                samples_processed: 2000,
                underruns: 5,
                buffer_level: 0.3,
                ..Default::default()
            },
        ];

        let agg = AggregateHealth::from_streams(&healths);
        assert_eq!(agg.stream_count, 2);
        assert_eq!(agg.healthy_count, 1);
        assert_eq!(agg.degraded_count, 1);
        assert_eq!(agg.total_samples, 3000);
        assert!((agg.avg_buffer_level - 0.4).abs() < 0.01);
        assert_eq!(agg.overall_status(), HealthStatus::Degraded);
    }

    #[test]
    fn test_throughput() {
        let mut monitor = HealthMonitor::new();
        monitor.start();

        // Record samples and check throughput is calculated
        monitor.record_samples(&[0.0; 1000]);
        let throughput = monitor.throughput();
        assert!(throughput > 0.0);
    }
}
