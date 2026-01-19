//! Visualization and debugging tools for constraints
//!
//! This module provides utilities for:
//! - Inspecting constraint states and violations
//! - Debugging constraint composition and decomposition
//! - Analyzing constraint satisfaction over time
//! - Generating human-readable constraint reports

use crate::ViolationComputable;
use std::collections::HashMap;
use std::fmt;

/// Statistics about constraint violations over a sequence of values
#[derive(Debug, Clone)]
pub struct ViolationStats {
    /// Total number of samples
    pub num_samples: usize,
    /// Number of violations
    pub num_violations: usize,
    /// Mean violation magnitude
    pub mean_violation: f32,
    /// Maximum violation magnitude
    pub max_violation: f32,
    /// Minimum violation magnitude (for violated samples)
    pub min_violation: f32,
    /// Standard deviation of violations
    pub std_violation: f32,
}

impl ViolationStats {
    /// Create violation statistics from a sequence of values
    pub fn from_values<C: ViolationComputable>(constraint: &C, values: &[f32]) -> Self {
        let num_samples = values.len();
        let mut num_violations = 0;
        let mut violation_sum = 0.0f32;
        let mut max_violation = 0.0f32;
        let mut min_violation = f32::MAX;
        let mut violations = Vec::new();

        for &val in values {
            let viol = constraint.violation(&[val]);
            if viol > 1e-6 {
                num_violations += 1;
                violation_sum += viol;
                max_violation = max_violation.max(viol);
                min_violation = min_violation.min(viol);
                violations.push(viol);
            }
        }

        let mean_violation = if num_violations > 0 {
            violation_sum / num_violations as f32
        } else {
            0.0
        };

        let std_violation = if num_violations > 0 {
            let variance: f32 = violations
                .iter()
                .map(|&v| (v - mean_violation).powi(2))
                .sum::<f32>()
                / num_violations as f32;
            variance.sqrt()
        } else {
            0.0
        };

        if min_violation == f32::MAX {
            min_violation = 0.0;
        }

        Self {
            num_samples,
            num_violations,
            mean_violation,
            max_violation,
            min_violation,
            std_violation,
        }
    }

    /// Violation rate (fraction of samples that violate)
    pub fn violation_rate(&self) -> f32 {
        if self.num_samples == 0 {
            0.0
        } else {
            self.num_violations as f32 / self.num_samples as f32
        }
    }

    /// Satisfaction rate (fraction of samples that satisfy)
    pub fn satisfaction_rate(&self) -> f32 {
        1.0 - self.violation_rate()
    }
}

impl fmt::Display for ViolationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Violation Statistics:")?;
        writeln!(f, "  Samples: {}", self.num_samples)?;
        writeln!(
            f,
            "  Violations: {} ({:.1}%)",
            self.num_violations,
            self.violation_rate() * 100.0
        )?;
        writeln!(
            f,
            "  Satisfactions: {} ({:.1}%)",
            self.num_samples - self.num_violations,
            self.satisfaction_rate() * 100.0
        )?;
        writeln!(f, "  Mean violation: {:.4}", self.mean_violation)?;
        writeln!(f, "  Max violation: {:.4}", self.max_violation)?;
        writeln!(f, "  Min violation: {:.4}", self.min_violation)?;
        writeln!(f, "  Std violation: {:.4}", self.std_violation)?;
        Ok(())
    }
}

/// Time series analysis of constraint satisfaction
pub struct ConstraintTimeSeries {
    name: String,
    values: Vec<f32>,
    violations: Vec<f32>,
    timestamps: Vec<usize>,
}

impl ConstraintTimeSeries {
    /// Create a new time series tracker
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
            violations: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    /// Add a sample to the time series
    pub fn add_sample<C: ViolationComputable>(
        &mut self,
        constraint: &C,
        value: f32,
        timestamp: usize,
    ) {
        let violation = constraint.violation(&[value]);
        self.values.push(value);
        self.violations.push(violation);
        self.timestamps.push(timestamp);
    }

    /// Get statistics for the entire time series
    pub fn stats(&self) -> ViolationStats {
        let num_samples = self.values.len();
        let mut num_violations = 0;
        let mut violation_sum = 0.0f32;
        let mut max_violation = 0.0f32;
        let mut min_violation = f32::MAX;
        let mut active_violations = Vec::new();

        for &viol in &self.violations {
            if viol > 1e-6 {
                num_violations += 1;
                violation_sum += viol;
                max_violation = max_violation.max(viol);
                min_violation = min_violation.min(viol);
                active_violations.push(viol);
            }
        }

        let mean_violation = if num_violations > 0 {
            violation_sum / num_violations as f32
        } else {
            0.0
        };

        let std_violation = if num_violations > 0 {
            let variance: f32 = active_violations
                .iter()
                .map(|&v| (v - mean_violation).powi(2))
                .sum::<f32>()
                / num_violations as f32;
            variance.sqrt()
        } else {
            0.0
        };

        if min_violation == f32::MAX {
            min_violation = 0.0;
        }

        ViolationStats {
            num_samples,
            num_violations,
            mean_violation,
            max_violation,
            min_violation,
            std_violation,
        }
    }

    /// Get the name of this time series
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get violation history
    pub fn violations(&self) -> &[f32] {
        &self.violations
    }

    /// Get value history
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Get timestamps
    pub fn timestamps(&self) -> &[usize] {
        &self.timestamps
    }

    /// Find periods of continuous violation
    pub fn violation_periods(&self) -> Vec<(usize, usize, f32)> {
        let mut periods = Vec::new();
        let mut in_violation = false;
        let mut start = 0;
        let mut max_viol_in_period = 0.0f32;

        for (i, &viol) in self.violations.iter().enumerate() {
            if viol > 1e-6 {
                if !in_violation {
                    in_violation = true;
                    start = i;
                    max_viol_in_period = viol;
                } else {
                    max_viol_in_period = max_viol_in_period.max(viol);
                }
            } else if in_violation {
                periods.push((start, i - 1, max_viol_in_period));
                in_violation = false;
                max_viol_in_period = 0.0;
            }
        }

        // Handle case where violation continues to end
        if in_violation {
            periods.push((start, self.violations.len() - 1, max_viol_in_period));
        }

        periods
    }
}

/// Debugging report for multiple constraints
pub struct ConstraintReport {
    constraint_names: Vec<String>,
    stats: HashMap<String, ViolationStats>,
}

impl ConstraintReport {
    /// Create a new constraint report
    pub fn new() -> Self {
        Self {
            constraint_names: Vec::new(),
            stats: HashMap::new(),
        }
    }

    /// Add constraint statistics to the report
    pub fn add_constraint(&mut self, name: impl Into<String>, stats: ViolationStats) {
        let name = name.into();
        self.constraint_names.push(name.clone());
        self.stats.insert(name, stats);
    }

    /// Get statistics for a specific constraint
    pub fn get_stats(&self, name: &str) -> Option<&ViolationStats> {
        self.stats.get(name)
    }

    /// Get all constraint names
    pub fn constraint_names(&self) -> &[String] {
        &self.constraint_names
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let mut report = String::from("Constraint Report\n");
        report.push_str("=================\n\n");

        for name in &self.constraint_names {
            if let Some(stats) = self.stats.get(name) {
                report.push_str(&format!("Constraint: {}\n", name));
                report.push_str(&format!(
                    "  Violation rate: {:.1}%\n",
                    stats.violation_rate() * 100.0
                ));
                report.push_str(&format!("  Mean violation: {:.4}\n", stats.mean_violation));
                report.push_str(&format!("  Max violation: {:.4}\n", stats.max_violation));
                report.push('\n');
            }
        }

        report
    }

    /// Find the most frequently violated constraint
    pub fn most_violated(&self) -> Option<(&str, &ViolationStats)> {
        self.stats
            .iter()
            .max_by(|a, b| {
                a.1.violation_rate()
                    .partial_cmp(&b.1.violation_rate())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, stats)| (name.as_str(), stats))
    }

    /// Find the constraint with the largest violations
    pub fn largest_violations(&self) -> Option<(&str, &ViolationStats)> {
        self.stats
            .iter()
            .max_by(|a, b| {
                a.1.max_violation
                    .partial_cmp(&b.1.max_violation)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, stats)| (name.as_str(), stats))
    }
}

impl Default for ConstraintReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ConstraintReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Constraint inspector for interactive debugging
pub struct ConstraintInspector {
    samples: Vec<f32>,
}

impl ConstraintInspector {
    /// Create a new constraint inspector
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Add a sample value to inspect
    pub fn add_sample(&mut self, value: f32) {
        self.samples.push(value);
    }

    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Inspect a constraint against all samples
    pub fn inspect<C: ViolationComputable + ?Sized>(
        &self,
        constraint: &C,
        name: &str,
    ) -> InspectionResult {
        let mut satisfied = Vec::new();
        let mut violated = Vec::new();

        for (i, &val) in self.samples.iter().enumerate() {
            let viol = constraint.violation(&[val]);
            if viol <= 1e-6 {
                satisfied.push((i, val));
            } else {
                violated.push((i, val, viol));
            }
        }

        InspectionResult {
            constraint_name: name.to_string(),
            total_samples: self.samples.len(),
            satisfied,
            violated,
        }
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Get all samples
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
}

impl Default for ConstraintInspector {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of constraint inspection
pub struct InspectionResult {
    constraint_name: String,
    total_samples: usize,
    satisfied: Vec<(usize, f32)>,
    violated: Vec<(usize, f32, f32)>,
}

impl InspectionResult {
    /// Get constraint name
    pub fn constraint_name(&self) -> &str {
        &self.constraint_name
    }

    /// Get total number of samples
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get satisfied samples (index, value)
    pub fn satisfied(&self) -> &[(usize, f32)] {
        &self.satisfied
    }

    /// Get violated samples (index, value, violation)
    pub fn violated(&self) -> &[(usize, f32, f32)] {
        &self.violated
    }

    /// Get violation rate
    pub fn violation_rate(&self) -> f32 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.violated.len() as f32 / self.total_samples as f32
        }
    }

    /// Print a summary
    pub fn print_summary(&self) {
        println!("Inspection: {}", self.constraint_name);
        println!("  Total samples: {}", self.total_samples);
        println!(
            "  Satisfied: {} ({:.1}%)",
            self.satisfied.len(),
            (self.satisfied.len() as f32 / self.total_samples as f32) * 100.0
        );
        println!(
            "  Violated: {} ({:.1}%)",
            self.violated.len(),
            self.violation_rate() * 100.0
        );
    }

    /// Print detailed violation information
    pub fn print_violations(&self, max_items: usize) {
        println!("\nViolations for {}:", self.constraint_name);
        let n = self.violated.len().min(max_items);
        for (i, &(idx, val, viol)) in self.violated.iter().take(n).enumerate() {
            println!(
                "  [{}] Sample {}: value={:.4}, violation={:.4}",
                i + 1,
                idx,
                val,
                viol
            );
        }
        if self.violated.len() > max_items {
            println!(
                "  ... and {} more violations",
                self.violated.len() - max_items
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConstraintBuilder;

    #[test]
    fn test_violation_stats() {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .less_than(10.0)
            .build()
            .unwrap();

        let values = vec![5.0, 8.0, 12.0, 15.0, 9.0];
        let stats = ViolationStats::from_values(&constraint, &values);

        assert_eq!(stats.num_samples, 5);
        assert_eq!(stats.num_violations, 2); // 12.0 and 15.0 violate
        assert!(stats.violation_rate() > 0.39 && stats.violation_rate() < 0.41);
    }

    #[test]
    fn test_constraint_time_series() {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .less_than(10.0)
            .build()
            .unwrap();

        let mut ts = ConstraintTimeSeries::new("test_series");
        ts.add_sample(&constraint, 5.0, 0);
        ts.add_sample(&constraint, 12.0, 1);
        ts.add_sample(&constraint, 15.0, 2);
        ts.add_sample(&constraint, 8.0, 3);

        let stats = ts.stats();
        assert_eq!(stats.num_samples, 4);
        assert_eq!(stats.num_violations, 2);

        let periods = ts.violation_periods();
        assert_eq!(periods.len(), 1); // One continuous period
        assert_eq!(periods[0].0, 1); // Starts at index 1
        assert_eq!(periods[0].1, 2); // Ends at index 2
    }

    #[test]
    fn test_constraint_report() {
        let constraint1 = ConstraintBuilder::new()
            .name("c1")
            .less_than(10.0)
            .build()
            .unwrap();

        let constraint2 = ConstraintBuilder::new()
            .name("c2")
            .less_than(5.0)
            .build()
            .unwrap();

        let values = vec![3.0, 7.0, 12.0];

        let mut report = ConstraintReport::new();
        report.add_constraint("c1", ViolationStats::from_values(&constraint1, &values));
        report.add_constraint("c2", ViolationStats::from_values(&constraint2, &values));

        assert_eq!(report.constraint_names().len(), 2);

        let most_violated = report.most_violated();
        assert!(most_violated.is_some());
    }

    #[test]
    fn test_constraint_inspector() {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .less_than(10.0)
            .build()
            .unwrap();

        let mut inspector = ConstraintInspector::new();
        inspector.add_sample(5.0);
        inspector.add_sample(12.0);
        inspector.add_sample(8.0);

        let result = inspector.inspect(&constraint, "test");
        assert_eq!(result.total_samples(), 3);
        assert_eq!(result.satisfied().len(), 2);
        assert_eq!(result.violated().len(), 1);
    }
}
