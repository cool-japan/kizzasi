//! Anomaly detection example with sensor monitoring
//!
//! This example demonstrates using Kizzasi for real-time anomaly detection
//! in multi-sensor systems. The predictor learns normal patterns and flags
//! deviations that exceed thresholds.
//!
//! Run with:
//! ```bash
//! cargo run --example anomaly_detection
//! ```

use kizzasi::prelude::*;
use std::f32::consts::PI;

/// Anomaly detector configuration
struct AnomalyDetector {
    predictor: Kizzasi,
    prediction_threshold: f32,
    history_window: usize,
    prediction_errors: Vec<f32>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    fn new(num_sensors: usize, prediction_threshold: f32, history_window: usize) -> Result<Self> {
        let predictor = KizzasiBuilder::sensor_preset(num_sensors).build()?;

        Ok(Self {
            predictor,
            prediction_threshold,
            history_window,
            prediction_errors: Vec::with_capacity(history_window),
        })
    }

    /// Process a sensor reading and detect anomalies
    fn process(&mut self, reading: &Array1<f32>) -> Result<AnomalyReport> {
        // Get prediction based on historical patterns
        let prediction = self.predictor.step(reading)?;

        // Calculate prediction error (L2 norm)
        let error = reading
            .iter()
            .zip(prediction.iter())
            .map(|(actual, pred)| (actual - pred).powi(2))
            .sum::<f32>()
            .sqrt();

        // Update error history
        self.prediction_errors.push(error);
        if self.prediction_errors.len() > self.history_window {
            self.prediction_errors.remove(0);
        }

        // Calculate statistics
        let mean_error = if !self.prediction_errors.is_empty() {
            self.prediction_errors.iter().sum::<f32>() / self.prediction_errors.len() as f32
        } else {
            0.0
        };

        let std_error = if self.prediction_errors.len() > 1 {
            let variance = self
                .prediction_errors
                .iter()
                .map(|e| (e - mean_error).powi(2))
                .sum::<f32>()
                / (self.prediction_errors.len() - 1) as f32;
            variance.sqrt()
        } else {
            0.0
        };

        // Detect anomaly using multiple criteria
        let is_anomaly = error > self.prediction_threshold
            || (std_error > 0.0 && error > mean_error + 3.0 * std_error);

        let severity = if error > mean_error + 5.0 * std_error {
            AnomalySeverity::Critical
        } else if error > mean_error + 3.0 * std_error {
            AnomalySeverity::High
        } else if error > mean_error + 2.0 * std_error {
            AnomalySeverity::Medium
        } else if error > mean_error + std_error {
            AnomalySeverity::Low
        } else {
            AnomalySeverity::Normal
        };

        Ok(AnomalyReport {
            is_anomaly,
            severity,
            prediction_error: error,
            mean_error,
            std_error,
            actual: reading.clone(),
            predicted: prediction,
        })
    }

    /// Reset the detector (e.g., after maintenance or recalibration)
    fn reset(&mut self) {
        self.predictor.reset();
        self.prediction_errors.clear();
    }
}

/// Anomaly severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnomalySeverity {
    Normal,
    Low,
    Medium,
    High,
    Critical,
}

impl AnomalySeverity {
    fn symbol(&self) -> &str {
        match self {
            Self::Normal => "✓",
            Self::Low => "⚠",
            Self::Medium => "⚠⚠",
            Self::High => "⚠⚠⚠",
            Self::Critical => "❌",
        }
    }
}

/// Anomaly detection report
struct AnomalyReport {
    is_anomaly: bool,
    severity: AnomalySeverity,
    prediction_error: f32,
    mean_error: f32,
    std_error: f32,
    actual: Array1<f32>,
    predicted: Array1<f32>,
}

impl AnomalyReport {
    fn display(&self, sample_idx: usize) {
        let status = if self.is_anomaly {
            format!("{} ANOMALY", self.severity.symbol())
        } else {
            format!("{} Normal", self.severity.symbol())
        };

        println!(
            "  Sample {:3}: {} | Error: {:.4} (μ={:.4}, σ={:.4})",
            sample_idx, status, self.prediction_error, self.mean_error, self.std_error
        );

        if self.is_anomaly {
            println!(
                "    Actual:    {:?}",
                self.actual
                    .iter()
                    .map(|x| format!("{:.3}", x))
                    .collect::<Vec<_>>()
            );
            println!(
                "    Predicted: {:?}",
                self.predicted
                    .iter()
                    .map(|x| format!("{:.3}", x))
                    .collect::<Vec<_>>()
            );
        }
    }
}

/// Generate simulated sensor data with injected anomalies
fn generate_sensor_data(num_samples: usize, num_sensors: usize) -> Vec<Array1<f32>> {
    let mut data = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 * 0.1;

        let mut values = Vec::with_capacity(num_sensors);
        for sensor_idx in 0..num_sensors {
            // Normal pattern: sine wave with small noise
            let phase = sensor_idx as f32 * PI / 4.0;
            let mut value = (t + phase).sin() * 0.5 + 0.5;

            // Add small noise
            value += (t * 10.0 + sensor_idx as f32).sin() * 0.05;

            // Inject anomalies at specific points
            if matches!(i, 50..=52) {
                // Spike anomaly (all sensors)
                value += 0.8;
            } else if matches!(i, 100 | 101) && sensor_idx == 1 {
                // Single sensor failure (sensor 1 flatlines)
                value = 0.0;
            } else if (150..155).contains(&i) && sensor_idx == 2 {
                // Sensor drift (sensor 2 drifts high)
                value += 0.3 * (i - 150) as f32 * 0.1;
            }

            values.push(value);
        }

        data.push(Array1::from_vec(values));
    }

    data
}

fn main() -> Result<()> {
    println!("=== Kizzasi Anomaly Detection Example ===\n");

    // Configuration
    let num_sensors = 3;
    let prediction_threshold = 0.2;
    let history_window = 20;

    println!("Configuration:");
    println!("  - Number of sensors: {}", num_sensors);
    println!("  - Prediction threshold: {}", prediction_threshold);
    println!("  - History window: {}\n", history_window);

    // Create anomaly detector
    let mut detector = AnomalyDetector::new(num_sensors, prediction_threshold, history_window)?;
    println!("✓ Anomaly detector initialized\n");

    // Generate sensor data with injected anomalies
    let num_samples = 200;
    let sensor_data = generate_sensor_data(num_samples, num_sensors);

    println!("Processing {} sensor samples...\n", num_samples);

    // Process data and detect anomalies
    let mut anomaly_count = 0;
    let mut reports = Vec::new();

    for reading in sensor_data.iter() {
        let report = detector.process(reading)?;

        if report.is_anomaly {
            anomaly_count += 1;
        }

        reports.push(report);
    }

    // Display summary
    println!("\n=== Detection Summary ===");
    println!("  Total samples: {}", num_samples);
    println!("  Anomalies detected: {}", anomaly_count);
    println!(
        "  Anomaly rate: {:.2}%\n",
        (anomaly_count as f32 / num_samples as f32) * 100.0
    );

    // Display anomalies
    println!("Detected Anomalies:");
    for (i, report) in reports.iter().enumerate() {
        if report.is_anomaly {
            report.display(i);
        }
    }
    println!();

    // Display normal samples (for comparison)
    println!("Sample Normal Readings:");
    for i in [10, 30, 80, 120].iter() {
        if let Some(report) = reports.get(*i) {
            if !report.is_anomaly {
                report.display(*i);
            }
        }
    }
    println!();

    // Severity distribution
    let mut severity_counts = [0; 5];
    for report in &reports {
        let idx = match report.severity {
            AnomalySeverity::Normal => 0,
            AnomalySeverity::Low => 1,
            AnomalySeverity::Medium => 2,
            AnomalySeverity::High => 3,
            AnomalySeverity::Critical => 4,
        };
        severity_counts[idx] += 1;
    }

    println!("Severity Distribution:");
    println!("  Normal:   {} samples", severity_counts[0]);
    println!("  Low:      {} samples", severity_counts[1]);
    println!("  Medium:   {} samples", severity_counts[2]);
    println!("  High:     {} samples", severity_counts[3]);
    println!("  Critical: {} samples", severity_counts[4]);
    println!();

    // Demonstrate reset functionality
    println!("Testing detector reset...");
    detector.reset();
    let test_reading = Array1::from_vec(vec![0.5; num_sensors]);
    let report = detector.process(&test_reading)?;
    println!("  ✓ Detector reset successful");
    println!(
        "    First reading after reset: {} (as expected)\n",
        if report.is_anomaly {
            "may flag anomaly"
        } else {
            "normal"
        }
    );

    println!("=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  • Kizzasi learns normal signal patterns automatically");
    println!("  • Multi-criteria anomaly detection (threshold + statistical)");
    println!("  • Real-time monitoring with sliding window statistics");
    println!("  • Severity classification for alert prioritization");
    println!("  • Suitable for: IoT sensors, industrial monitoring, predictive maintenance");

    Ok(())
}
