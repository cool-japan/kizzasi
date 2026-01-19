//! Metrics Monitoring Example
//!
//! This example demonstrates comprehensive metrics collection and monitoring
//! for production ML systems using Kizzasi's telemetry capabilities.
//!
//! Run with: cargo run --example metrics_monitoring --features full

use kizzasi::prelude::*;
use kizzasi::telemetry::{MetricEvent, MetricValue, MetricsCollector, MetricsConfig};
use std::collections::HashMap;
use std::sync::Arc;

/// Monitored predictor with automatic metrics collection
struct MonitoredPredictor {
    predictor: Kizzasi,
    metrics: Arc<MetricsCollector>,
    #[allow(dead_code)]
    name: String,
}

impl MonitoredPredictor {
    fn new(name: &str, config: KizzasiConfig) -> Result<Self> {
        let metrics_config = MetricsConfig {
            name: name.to_string(),
            histogram_size: 10000,
            track_latency: true,
            track_errors: true,
        };

        Ok(Self {
            predictor: Kizzasi::new(config)?,
            metrics: Arc::new(MetricsCollector::with_config(metrics_config)),
            name: name.to_string(),
        })
    }

    fn predict(&mut self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let start = std::time::Instant::now();

        let result = self.predictor.step(input);

        let latency_us = start.elapsed().as_micros() as u64;

        match &result {
            Ok(output) => {
                self.metrics.record(MetricEvent::Prediction {
                    latency_us,
                    input_dim: input.len(),
                    output_dim: output.len(),
                });
            }
            Err(e) => {
                self.metrics.record(MetricEvent::Error {
                    category: format!("{:?}", e.category()),
                });
            }
        }

        result
    }

    fn predict_batch(&mut self, inputs: &[Array1<f32>]) -> Result<Vec<Array1<f32>>> {
        let start = std::time::Instant::now();

        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.predictor.step(input)?);
        }

        let latency_us = start.elapsed().as_micros() as u64;
        self.metrics.record(MetricEvent::BatchPrediction {
            latency_us,
            batch_size: inputs.len(),
        });

        Ok(outputs)
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.predictor.reset();
        self.metrics.record(MetricEvent::Reset);
    }

    fn get_metrics(&self) -> Arc<MetricsCollector> {
        self.metrics.clone()
    }
}

/// Metrics dashboard for monitoring
struct MetricsDashboard {
    collectors: Vec<Arc<MetricsCollector>>,
}

impl MetricsDashboard {
    fn new() -> Self {
        Self {
            collectors: Vec::new(),
        }
    }

    fn add_collector(&mut self, collector: Arc<MetricsCollector>) {
        self.collectors.push(collector);
    }

    fn display_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════╗");
        println!("║          METRICS DASHBOARD                    ║");
        println!("╚═══════════════════════════════════════════════╝\n");

        for collector in &self.collectors {
            let snapshot = collector.snapshot();

            println!("📊 {}", snapshot.name);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  Predictions:     {}", snapshot.total_predictions);
            println!("  Batch Ops:       {}", snapshot.total_batch_predictions);
            println!("  Errors:          {}", snapshot.total_errors);
            println!("  Resets:          {}", snapshot.total_resets);
            println!("  Uptime:          {}s", snapshot.uptime_secs);
            println!();
            println!("  Latency:");
            println!("    Average:       {:.3} ms", snapshot.avg_latency_ms);
            println!("    P50:           {:.3} ms", snapshot.p50_latency_ms);
            println!("    P95:           {:.3} ms", snapshot.p95_latency_ms);
            println!("    P99:           {:.3} ms", snapshot.p99_latency_ms);
            println!("    Min:           {:.3} ms", snapshot.min_latency_ms);
            println!("    Max:           {:.3} ms", snapshot.max_latency_ms);
            println!();
            println!("  Performance:");
            println!(
                "    Throughput:    {:.2} pred/s",
                snapshot.predictions_per_second
            );
            println!("    Error Rate:    {:.2}%", snapshot.error_rate * 100.0);
            println!();

            if !snapshot.error_counts.is_empty() {
                println!("  Error Breakdown:");
                for (category, count) in &snapshot.error_counts {
                    println!("    {}: {}", category, count);
                }
                println!();
            }

            if !snapshot.custom_metrics.is_empty() {
                println!("  Custom Metrics:");
                for (name, value) in &snapshot.custom_metrics {
                    println!("    {}: {:.2}", name, value);
                }
                println!();
            }
        }
    }

    fn export_all_prometheus(&self) -> String {
        let mut output = String::new();
        for collector in &self.collectors {
            output.push_str(&collector.export_prometheus());
            output.push('\n');
        }
        output
    }

    fn export_all_json(&self) -> String {
        let mut snapshots = Vec::new();
        for collector in &self.collectors {
            let snapshot = collector.snapshot();
            snapshots.push(snapshot);
        }
        serde_json::to_string_pretty(&snapshots).unwrap_or_else(|_| "[]".to_string())
    }
}

fn main() -> Result<()> {
    println!("\n╔════════════════════════════════════════╗");
    println!("║   Metrics Monitoring Example           ║");
    println!("╚════════════════════════════════════════╝\n");

    // Create dashboard
    let mut dashboard = MetricsDashboard::new();

    // Create multiple monitored predictors with different configurations
    println!("🔧 Creating monitored predictors...\n");

    let config_small = KizzasiConfig::new()
        .input_dim(16)
        .output_dim(16)
        .hidden_dim(64)
        .context_window(512);

    let config_medium = KizzasiConfig::new()
        .input_dim(64)
        .output_dim(64)
        .hidden_dim(256)
        .context_window(2048);

    let config_large = KizzasiConfig::new()
        .input_dim(128)
        .output_dim(128)
        .hidden_dim(512)
        .context_window(4096);

    let mut predictor_small = MonitoredPredictor::new("small_model", config_small)?;
    let mut predictor_medium = MonitoredPredictor::new("medium_model", config_medium)?;
    let mut predictor_large = MonitoredPredictor::new("large_model", config_large)?;

    dashboard.add_collector(predictor_small.get_metrics());
    dashboard.add_collector(predictor_medium.get_metrics());
    dashboard.add_collector(predictor_large.get_metrics());

    println!("✓ Created 3 monitored predictors\n");

    // Simulate workload
    println!("🔄 Running workload simulation...\n");

    // Small model: high frequency, low latency
    println!("  Small model: Processing 1000 rapid predictions...");
    for _ in 0..1000 {
        let input = Array1::from_vec(vec![0.1; 16]);
        let _ = predictor_small.predict(&input);
    }

    // Medium model: moderate frequency
    println!("  Medium model: Processing 500 predictions...");
    for _ in 0..500 {
        let input = Array1::from_vec(vec![0.2; 64]);
        let _ = predictor_medium.predict(&input);
    }

    // Large model: lower frequency, batch processing
    println!("  Large model: Processing 10 batches of 50...");
    for _ in 0..10 {
        let batch: Vec<_> = (0..50).map(|_| Array1::from_vec(vec![0.3; 128])).collect();
        let _ = predictor_large.predict_batch(&batch);
    }

    println!("\n✓ Workload completed\n");

    // Simulate some custom metrics
    predictor_small.get_metrics().record(MetricEvent::Custom {
        name: "cache_hit_rate".to_string(),
        value: MetricValue::Gauge(0.85),
        tags: HashMap::new(),
    });

    predictor_medium.get_metrics().record(MetricEvent::Custom {
        name: "memory_usage_mb".to_string(),
        value: MetricValue::Gauge(256.5),
        tags: HashMap::new(),
    });

    // Display dashboard
    dashboard.display_summary();

    // Export metrics
    println!("\n📤 Exporting Metrics\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Prometheus Format:\n");
    println!("{}", dashboard.export_all_prometheus());

    println!("\n📝 JSON Format (first 500 chars):\n");
    let json = dashboard.export_all_json();
    let preview = if json.len() > 500 {
        &json[..500]
    } else {
        &json
    };
    println!("{}...\n", preview);

    // Demonstrate alert conditions
    println!("\n⚠️  Alert Monitoring\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for collector in &dashboard.collectors {
        let snapshot = collector.snapshot();

        // Check latency SLA
        if snapshot.p99_latency_ms > 100.0 {
            println!(
                "🔴 ALERT: {} - P99 latency ({:.2}ms) exceeds SLA (100ms)",
                snapshot.name, snapshot.p99_latency_ms
            );
        } else {
            println!("✓ {}: P99 latency within SLA", snapshot.name);
        }

        // Check error rate
        if snapshot.error_rate > 0.01 {
            println!(
                "🔴 ALERT: {} - Error rate ({:.2}%) exceeds threshold (1%)",
                snapshot.name,
                snapshot.error_rate * 100.0
            );
        } else {
            println!("✓ {}: Error rate within threshold", snapshot.name);
        }

        // Check throughput
        if snapshot.predictions_per_second < 1.0 && snapshot.total_predictions > 0 {
            println!(
                "⚠️  WARNING: {} - Low throughput ({:.2} pred/s)",
                snapshot.name, snapshot.predictions_per_second
            );
        } else if snapshot.total_predictions > 0 {
            println!("✓ {}: Throughput healthy", snapshot.name);
        }

        println!();
    }

    println!("✅ Metrics monitoring example completed!\n");

    Ok(())
}
