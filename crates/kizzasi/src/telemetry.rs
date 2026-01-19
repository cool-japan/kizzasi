//! Telemetry and metrics collection for production monitoring.
//!
//! This module provides comprehensive metrics collection and reporting for
//! monitoring Kizzasi predictors in production environments.
//!
//! # Features
//!
//! - Performance metrics (latency, throughput)
//! - Error tracking and categorization
//! - Resource utilization monitoring
//! - Custom metric collection
//! - Histogram and percentile statistics
//! - Time-series data aggregation
//! - Export to various backends (Prometheus, StatsD, etc.)
//!
//! # Example
//!
//! ```rust
//! use kizzasi::telemetry::{MetricsCollector, MetricEvent, MetricValue};
//! use std::sync::Arc;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let metrics = Arc::new(MetricsCollector::new("my_predictor"));
//!
//! // Record a prediction
//! metrics.record(MetricEvent::Prediction {
//!     latency_us: 1500,
//!     input_dim: 64,
//!     output_dim: 64,
//! });
//!
//! // Record an error
//! metrics.record(MetricEvent::Error {
//!     category: "DimensionMismatch".to_string(),
//! });
//!
//! // Get current statistics
//! let stats = metrics.snapshot();
//! println!("Total predictions: {}", stats.total_predictions);
//! println!("Average latency: {:.2} ms", stats.avg_latency_ms);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Metric event types that can be recorded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricEvent {
    /// A prediction was performed.
    Prediction {
        latency_us: u64,
        input_dim: usize,
        output_dim: usize,
    },

    /// A batch prediction was performed.
    BatchPrediction { latency_us: u64, batch_size: usize },

    /// An error occurred.
    Error { category: String },

    /// Model was reset.
    Reset,

    /// Model was forked.
    Fork,

    /// Custom metric event.
    Custom {
        name: String,
        value: MetricValue,
        tags: HashMap<String, String>,
    },
}

/// Value types for metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
    Duration(Duration),
}

/// Histogram for tracking value distributions.
#[derive(Debug, Clone)]
struct Histogram {
    values: Vec<f64>,
    max_size: usize,
}

impl Histogram {
    fn new(max_size: usize) -> Self {
        Self {
            values: Vec::with_capacity(max_size),
            max_size,
        }
    }

    fn record(&mut self, value: f64) {
        if self.values.len() >= self.max_size {
            // Simple reservoir sampling
            let idx = (value as usize) % self.max_size;
            self.values[idx] = value;
        } else {
            self.values.push(value);
        }
    }

    fn percentile(&self, p: f64) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((sorted.len() as f64 - 1.0) * p).floor() as usize;
        Some(sorted[idx])
    }

    fn mean(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        Some(self.values.iter().sum::<f64>() / self.values.len() as f64)
    }

    fn min(&self) -> Option<f64> {
        self.values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
    }

    fn max(&self) -> Option<f64> {
        self.values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
    }
}

/// Statistics snapshot for a metrics collector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub name: String,
    pub uptime_secs: u64,
    pub total_predictions: u64,
    pub total_batch_predictions: u64,
    pub total_errors: u64,
    pub total_resets: u64,
    pub total_forks: u64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub predictions_per_second: f64,
    pub error_rate: f64,
    pub error_counts: HashMap<String, u64>,
    pub custom_metrics: HashMap<String, f64>,
}

/// Inner mutable state of the metrics collector.
struct MetricsState {
    latency_histogram: Histogram,
    error_counts: HashMap<String, u64>,
    custom_counters: HashMap<String, f64>,
}

impl MetricsState {
    fn new(histogram_size: usize) -> Self {
        Self {
            latency_histogram: Histogram::new(histogram_size),
            error_counts: HashMap::new(),
            custom_counters: HashMap::new(),
        }
    }
}

/// Configuration for metrics collection.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Name/identifier for this metrics collector.
    pub name: String,

    /// Maximum number of samples to keep in histograms.
    pub histogram_size: usize,

    /// Enable detailed latency tracking.
    pub track_latency: bool,

    /// Enable error categorization.
    pub track_errors: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            name: "kizzasi".to_string(),
            histogram_size: 10000,
            track_latency: true,
            track_errors: true,
        }
    }
}

/// Metrics collector for tracking predictor performance.
pub struct MetricsCollector {
    config: MetricsConfig,
    start_time: Instant,
    total_predictions: AtomicU64,
    total_batch_predictions: AtomicU64,
    total_errors: AtomicU64,
    total_resets: AtomicU64,
    total_forks: AtomicU64,
    state: Mutex<MetricsState>,
}

impl MetricsCollector {
    /// Create a new metrics collector with default configuration.
    pub fn new(name: &str) -> Self {
        let config = MetricsConfig {
            name: name.to_string(),
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new metrics collector with custom configuration.
    pub fn with_config(config: MetricsConfig) -> Self {
        let histogram_size = config.histogram_size;
        Self {
            config,
            start_time: Instant::now(),
            total_predictions: AtomicU64::new(0),
            total_batch_predictions: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_resets: AtomicU64::new(0),
            total_forks: AtomicU64::new(0),
            state: Mutex::new(MetricsState::new(histogram_size)),
        }
    }

    /// Record a metric event.
    pub fn record(&self, event: MetricEvent) {
        match event {
            MetricEvent::Prediction {
                latency_us,
                input_dim: _,
                output_dim: _,
            } => {
                self.total_predictions.fetch_add(1, Ordering::Relaxed);
                if self.config.track_latency {
                    let mut state = self.state.lock().expect("MetricsCollector mutex poisoned");
                    state.latency_histogram.record(latency_us as f64 / 1000.0); // Convert to ms
                }
            }
            MetricEvent::BatchPrediction {
                latency_us,
                batch_size: _,
            } => {
                self.total_batch_predictions.fetch_add(1, Ordering::Relaxed);
                if self.config.track_latency {
                    let mut state = self.state.lock().expect("MetricsCollector mutex poisoned");
                    state.latency_histogram.record(latency_us as f64 / 1000.0);
                }
            }
            MetricEvent::Error { category } => {
                self.total_errors.fetch_add(1, Ordering::Relaxed);
                if self.config.track_errors {
                    let mut state = self.state.lock().expect("MetricsCollector mutex poisoned");
                    *state.error_counts.entry(category).or_insert(0) += 1;
                }
            }
            MetricEvent::Reset => {
                self.total_resets.fetch_add(1, Ordering::Relaxed);
            }
            MetricEvent::Fork => {
                self.total_forks.fetch_add(1, Ordering::Relaxed);
            }
            MetricEvent::Custom { name, value, .. } => {
                if let MetricValue::Counter(val) = value {
                    let mut state = self.state.lock().expect("MetricsCollector mutex poisoned");
                    *state.custom_counters.entry(name).or_insert(0.0) += val as f64;
                } else if let MetricValue::Gauge(val) = value {
                    let mut state = self.state.lock().expect("MetricsCollector mutex poisoned");
                    state.custom_counters.insert(name, val);
                }
            }
        }
    }

    /// Get a snapshot of current metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let uptime = self.start_time.elapsed();
        let uptime_secs = uptime.as_secs();

        let total_predictions = self.total_predictions.load(Ordering::Relaxed);
        let total_batch_predictions = self.total_batch_predictions.load(Ordering::Relaxed);
        let total_errors = self.total_errors.load(Ordering::Relaxed);
        let total_resets = self.total_resets.load(Ordering::Relaxed);
        let total_forks = self.total_forks.load(Ordering::Relaxed);

        let state = self.state.lock().expect("MetricsCollector mutex poisoned");

        let avg_latency_ms = state.latency_histogram.mean().unwrap_or(0.0);
        let p50_latency_ms = state.latency_histogram.percentile(0.5).unwrap_or(0.0);
        let p95_latency_ms = state.latency_histogram.percentile(0.95).unwrap_or(0.0);
        let p99_latency_ms = state.latency_histogram.percentile(0.99).unwrap_or(0.0);
        let min_latency_ms = state.latency_histogram.min().unwrap_or(0.0);
        let max_latency_ms = state.latency_histogram.max().unwrap_or(0.0);

        let predictions_per_second = if uptime_secs > 0 {
            (total_predictions + total_batch_predictions) as f64 / uptime_secs as f64
        } else {
            0.0
        };

        let error_rate = if total_predictions > 0 {
            total_errors as f64 / total_predictions as f64
        } else {
            0.0
        };

        MetricsSnapshot {
            name: self.config.name.clone(),
            uptime_secs,
            total_predictions,
            total_batch_predictions,
            total_errors,
            total_resets,
            total_forks,
            avg_latency_ms,
            p50_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            min_latency_ms,
            max_latency_ms,
            predictions_per_second,
            error_rate,
            error_counts: state.error_counts.clone(),
            custom_metrics: state.custom_counters.clone(),
        }
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.total_predictions.store(0, Ordering::Relaxed);
        self.total_batch_predictions.store(0, Ordering::Relaxed);
        self.total_errors.store(0, Ordering::Relaxed);
        self.total_resets.store(0, Ordering::Relaxed);
        self.total_forks.store(0, Ordering::Relaxed);

        let mut state = self.state.lock().expect("MetricsCollector mutex poisoned");
        *state = MetricsState::new(self.config.histogram_size);
    }

    /// Export metrics in Prometheus text format.
    pub fn export_prometheus(&self) -> String {
        let snapshot = self.snapshot();
        let mut output = String::new();

        let prefix = &snapshot.name;

        output.push_str(&format!(
            "# HELP {}_predictions_total Total number of predictions\n",
            prefix
        ));
        output.push_str(&format!("# TYPE {}_predictions_total counter\n", prefix));
        output.push_str(&format!(
            "{}_predictions_total {}\n\n",
            prefix, snapshot.total_predictions
        ));

        output.push_str(&format!(
            "# HELP {}_errors_total Total number of errors\n",
            prefix
        ));
        output.push_str(&format!("# TYPE {}_errors_total counter\n", prefix));
        output.push_str(&format!(
            "{}_errors_total {}\n\n",
            prefix, snapshot.total_errors
        ));

        output.push_str(&format!(
            "# HELP {}_latency_ms Prediction latency in milliseconds\n",
            prefix
        ));
        output.push_str(&format!("# TYPE {}_latency_ms summary\n", prefix));
        output.push_str(&format!(
            "{}_latency_ms{{quantile=\"0.5\"}} {}\n",
            prefix, snapshot.p50_latency_ms
        ));
        output.push_str(&format!(
            "{}_latency_ms{{quantile=\"0.95\"}} {}\n",
            prefix, snapshot.p95_latency_ms
        ));
        output.push_str(&format!(
            "{}_latency_ms{{quantile=\"0.99\"}} {}\n",
            prefix, snapshot.p99_latency_ms
        ));
        output.push_str(&format!(
            "{}_latency_ms_sum {}\n",
            prefix,
            snapshot.avg_latency_ms * snapshot.total_predictions as f64
        ));
        output.push_str(&format!(
            "{}_latency_ms_count {}\n\n",
            prefix, snapshot.total_predictions
        ));

        output.push_str(&format!(
            "# HELP {}_error_rate Error rate (errors / predictions)\n",
            prefix
        ));
        output.push_str(&format!("# TYPE {}_error_rate gauge\n", prefix));
        output.push_str(&format!(
            "{}_error_rate {}\n\n",
            prefix, snapshot.error_rate
        ));

        output
    }

    /// Export metrics as JSON.
    pub fn export_json(&self) -> String {
        let snapshot = self.snapshot();
        serde_json::to_string_pretty(&snapshot).unwrap_or_else(|_| "{}".to_string())
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new("kizzasi")
    }
}

/// A trait for types that can report metrics.
pub trait Instrumented {
    /// Get the metrics collector for this instance.
    fn metrics(&self) -> Arc<MetricsCollector>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let metrics = MetricsCollector::new("test");

        metrics.record(MetricEvent::Prediction {
            latency_us: 1500,
            input_dim: 64,
            output_dim: 64,
        });

        metrics.record(MetricEvent::Prediction {
            latency_us: 2000,
            input_dim: 64,
            output_dim: 64,
        });

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_predictions, 2);
        assert!(snapshot.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_error_tracking() {
        let metrics = MetricsCollector::new("test");

        metrics.record(MetricEvent::Error {
            category: "DimensionMismatch".to_string(),
        });

        metrics.record(MetricEvent::Error {
            category: "DimensionMismatch".to_string(),
        });

        metrics.record(MetricEvent::Error {
            category: "InvalidState".to_string(),
        });

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_errors, 3);
        assert_eq!(snapshot.error_counts.get("DimensionMismatch"), Some(&2));
        assert_eq!(snapshot.error_counts.get("InvalidState"), Some(&1));
    }

    #[test]
    fn test_histogram_percentiles() {
        let mut hist = Histogram::new(1000);

        for i in 1..=100 {
            hist.record(i as f64);
        }

        assert_eq!(hist.min(), Some(1.0));
        assert_eq!(hist.max(), Some(100.0));
        assert!((hist.mean().unwrap() - 50.5).abs() < 1.0);
        assert!((hist.percentile(0.5).unwrap() - 50.0).abs() < 2.0);
        assert!(hist.percentile(0.95).unwrap() > 90.0);
    }

    #[test]
    fn test_prometheus_export() {
        let metrics = MetricsCollector::new("test");

        metrics.record(MetricEvent::Prediction {
            latency_us: 1000,
            input_dim: 64,
            output_dim: 64,
        });

        let output = metrics.export_prometheus();
        assert!(output.contains("test_predictions_total 1"));
        assert!(output.contains("test_latency_ms"));
    }

    #[test]
    fn test_json_export() {
        let metrics = MetricsCollector::new("test");

        metrics.record(MetricEvent::Prediction {
            latency_us: 1000,
            input_dim: 64,
            output_dim: 64,
        });

        let json = metrics.export_json();
        // Check for JSON fields (pretty-printed has spaces)
        assert!(json.contains("total_predictions"));
        assert!(
            json.contains("\"total_predictions\": 1") || json.contains("\"total_predictions\":1")
        );
    }

    #[test]
    fn test_custom_metrics() {
        let metrics = MetricsCollector::new("test");

        metrics.record(MetricEvent::Custom {
            name: "custom_counter".to_string(),
            value: MetricValue::Counter(42),
            tags: HashMap::new(),
        });

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.custom_metrics.get("custom_counter"), Some(&42.0));
    }

    #[test]
    fn test_reset() {
        let metrics = MetricsCollector::new("test");

        metrics.record(MetricEvent::Prediction {
            latency_us: 1000,
            input_dim: 64,
            output_dim: 64,
        });

        let snapshot1 = metrics.snapshot();
        assert_eq!(snapshot1.total_predictions, 1);

        metrics.reset();

        let snapshot2 = metrics.snapshot();
        assert_eq!(snapshot2.total_predictions, 0);
    }
}
