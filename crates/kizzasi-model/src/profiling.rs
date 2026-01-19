//! Model Profiling and Benchmarking Utilities
//!
//! Provides tools for measuring model performance:
//! - **Latency measurement**: Per-step and batched inference timing
//! - **Memory profiling**: Track memory usage and allocations
//! - **Throughput analysis**: Tokens/sequences per second
//! - **Bottleneck identification**: Find performance-critical sections
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::profiling::ModelProfiler;
//! use kizzasi_model::mamba::Mamba;
//!
//! let model = Mamba::new(config)?;
//! let profiler = ModelProfiler::new(model);
//!
//! let results = profiler.profile_inference(num_steps)?;
//! println!("Average latency: {:.2}μs", results.avg_latency_us);
//! ```

use crate::error::ModelResult;
use crate::AutoregressiveModel;
use scirs2_core::ndarray::Array1;
use std::time::{Duration, Instant};

/// Profiling results for model inference
#[derive(Debug, Clone)]
pub struct ProfilingResults {
    /// Total number of steps profiled
    pub num_steps: usize,
    /// Total execution time
    pub total_duration: Duration,
    /// Average latency per step
    pub avg_latency_us: f64,
    /// Minimum latency observed
    pub min_latency_us: f64,
    /// Maximum latency observed
    pub max_latency_us: f64,
    /// Median latency
    pub median_latency_us: f64,
    /// 95th percentile latency
    pub p95_latency_us: f64,
    /// 99th percentile latency
    pub p99_latency_us: f64,
    /// Throughput (steps per second)
    pub throughput_steps_per_sec: f64,
    /// Standard deviation of latency
    pub std_dev_us: f64,
}

impl ProfilingResults {
    /// Create results from timing measurements
    pub fn from_timings(timings: &[Duration]) -> Self {
        let num_steps = timings.len();

        if num_steps == 0 {
            return Self::default();
        }

        // Convert to microseconds
        let mut latencies_us: Vec<f64> = timings
            .iter()
            .map(|d| d.as_secs_f64() * 1_000_000.0)
            .collect();

        latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let total_duration: Duration = timings.iter().sum();
        let total_us = total_duration.as_secs_f64() * 1_000_000.0;
        let avg_latency_us = total_us / num_steps as f64;

        let min_latency_us = latencies_us[0];
        let max_latency_us = latencies_us[num_steps - 1];

        let median_latency_us = if num_steps.is_multiple_of(2) {
            (latencies_us[num_steps / 2 - 1] + latencies_us[num_steps / 2]) / 2.0
        } else {
            latencies_us[num_steps / 2]
        };

        let p95_idx = ((num_steps as f64 * 0.95) as usize).min(num_steps - 1);
        let p95_latency_us = latencies_us[p95_idx];

        let p99_idx = ((num_steps as f64 * 0.99) as usize).min(num_steps - 1);
        let p99_latency_us = latencies_us[p99_idx];

        let throughput_steps_per_sec = num_steps as f64 / total_duration.as_secs_f64();

        // Calculate standard deviation
        let variance = latencies_us
            .iter()
            .map(|&x| (x - avg_latency_us).powi(2))
            .sum::<f64>()
            / num_steps as f64;
        let std_dev_us = variance.sqrt();

        Self {
            num_steps,
            total_duration,
            avg_latency_us,
            min_latency_us,
            max_latency_us,
            median_latency_us,
            p95_latency_us,
            p99_latency_us,
            throughput_steps_per_sec,
            std_dev_us,
        }
    }

    /// Format results as a human-readable string
    pub fn format_report(&self) -> String {
        format!(
            "Profiling Results:\n\
             ==================\n\
             Steps:              {}\n\
             Total Duration:     {:.2}ms\n\
             Average Latency:    {:.2}μs\n\
             Min Latency:        {:.2}μs\n\
             Max Latency:        {:.2}μs\n\
             Median Latency:     {:.2}μs\n\
             P95 Latency:        {:.2}μs\n\
             P99 Latency:        {:.2}μs\n\
             Std Dev:            {:.2}μs\n\
             Throughput:         {:.2} steps/sec\n",
            self.num_steps,
            self.total_duration.as_secs_f64() * 1000.0,
            self.avg_latency_us,
            self.min_latency_us,
            self.max_latency_us,
            self.median_latency_us,
            self.p95_latency_us,
            self.p99_latency_us,
            self.std_dev_us,
            self.throughput_steps_per_sec,
        )
    }
}

impl Default for ProfilingResults {
    fn default() -> Self {
        Self {
            num_steps: 0,
            total_duration: Duration::ZERO,
            avg_latency_us: 0.0,
            min_latency_us: 0.0,
            max_latency_us: 0.0,
            median_latency_us: 0.0,
            p95_latency_us: 0.0,
            p99_latency_us: 0.0,
            throughput_steps_per_sec: 0.0,
            std_dev_us: 0.0,
        }
    }
}

/// Model profiler for performance measurement
pub struct ModelProfiler<M: AutoregressiveModel> {
    model: M,
    warmup_steps: usize,
}

impl<M: AutoregressiveModel> ModelProfiler<M> {
    /// Create a new profiler
    pub fn new(model: M) -> Self {
        Self {
            model,
            warmup_steps: 10,
        }
    }

    /// Set number of warmup steps (default: 10)
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Profile single-step inference
    ///
    /// # Arguments
    ///
    /// * `num_steps` - Number of inference steps to measure
    /// * `input_dim` - Input dimension
    ///
    /// # Returns
    ///
    /// Profiling results with timing statistics
    pub fn profile_inference(
        &mut self,
        num_steps: usize,
        input_dim: usize,
    ) -> ModelResult<ProfilingResults> {
        // Warmup
        let warmup_input = Array1::zeros(input_dim);
        for _ in 0..self.warmup_steps {
            let _ = self.model.step(&warmup_input)?;
        }

        // Reset state for clean measurements
        self.model.reset();

        // Measure inference
        let mut timings = Vec::with_capacity(num_steps);
        let input = Array1::from_elem(input_dim, 1.0);

        for _ in 0..num_steps {
            let start = Instant::now();
            let _ = self.model.step(&input)?;
            timings.push(start.elapsed());
        }

        Ok(ProfilingResults::from_timings(&timings))
    }

    /// Profile with varying input sizes
    ///
    /// Useful for understanding how performance scales with input dimension
    pub fn profile_input_scaling(
        &mut self,
        input_dims: &[usize],
        steps_per_dim: usize,
    ) -> ModelResult<Vec<(usize, ProfilingResults)>> {
        let mut results = Vec::with_capacity(input_dims.len());

        for &dim in input_dims {
            self.model.reset();
            let profile = self.profile_inference(steps_per_dim, dim)?;
            results.push((dim, profile));
        }

        Ok(results)
    }

    /// Profile memory usage (estimates based on model dimensions)
    pub fn estimate_memory_usage(&self) -> MemoryProfile {
        let hidden_dim = self.model.hidden_dim();
        let state_dim = self.model.state_dim();
        let num_layers = self.model.num_layers();

        // Estimate per-layer state size (in bytes)
        // Typical SSM: hidden_dim * state_dim * 4 bytes (f32)
        let state_bytes_per_layer = hidden_dim * state_dim * 4;
        let total_state_bytes = state_bytes_per_layer * num_layers;

        // Estimate weight memory (rough approximation)
        // Typical: multiple weight matrices per layer
        let weight_estimate = hidden_dim * hidden_dim * 4 * num_layers * 5; // ~5 matrices per layer

        MemoryProfile {
            hidden_dim,
            state_dim,
            num_layers,
            state_memory_bytes: total_state_bytes,
            estimated_weight_memory_bytes: weight_estimate,
            total_estimated_bytes: total_state_bytes + weight_estimate,
        }
    }

    /// Get reference to underlying model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable reference to underlying model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Consume profiler and return model
    pub fn into_model(self) -> M {
        self.model
    }
}

/// Memory usage profile
#[derive(Debug, Clone)]
pub struct MemoryProfile {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// State dimension
    pub state_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Memory for hidden states (bytes)
    pub state_memory_bytes: usize,
    /// Estimated weight memory (bytes)
    pub estimated_weight_memory_bytes: usize,
    /// Total estimated memory (bytes)
    pub total_estimated_bytes: usize,
}

impl MemoryProfile {
    /// Format memory profile as human-readable string
    pub fn format_report(&self) -> String {
        format!(
            "Memory Profile:\n\
             ===============\n\
             Hidden Dim:         {}\n\
             State Dim:          {}\n\
             Num Layers:         {}\n\
             State Memory:       {:.2} MB\n\
             Weight Memory:      {:.2} MB (estimated)\n\
             Total Memory:       {:.2} MB (estimated)\n",
            self.hidden_dim,
            self.state_dim,
            self.num_layers,
            self.state_memory_bytes as f64 / 1_048_576.0,
            self.estimated_weight_memory_bytes as f64 / 1_048_576.0,
            self.total_estimated_bytes as f64 / 1_048_576.0,
        )
    }
}

/// Benchmark suite for comparing models
pub struct BenchmarkSuite {
    num_steps: usize,
    warmup_steps: usize,
    input_dim: usize,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            num_steps: 1000,
            warmup_steps: 10,
            input_dim: 1,
        }
    }

    /// Set number of benchmark steps
    pub fn num_steps(mut self, steps: usize) -> Self {
        self.num_steps = steps;
        self
    }

    /// Set number of warmup steps
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Set input dimension
    pub fn input_dim(mut self, dim: usize) -> Self {
        self.input_dim = dim;
        self
    }

    /// Run benchmark on a model
    pub fn benchmark<M: AutoregressiveModel>(&self, model: M) -> ModelResult<ProfilingResults> {
        let mut profiler = ModelProfiler::new(model).warmup_steps(self.warmup_steps);
        profiler.profile_inference(self.num_steps, self.input_dim)
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Bottleneck severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BottleneckSeverity {
    /// Low severity - minor optimization opportunity
    Low,
    /// Medium severity - noticeable performance impact
    Medium,
    /// High severity - significant performance bottleneck
    High,
    /// Critical severity - major performance issue
    Critical,
}

/// Information about a specific bottleneck
#[derive(Debug, Clone)]
pub struct BottleneckInfo {
    /// Name of the bottleneck (e.g., "Matrix multiplication")
    pub name: String,
    /// Description of the issue
    pub description: String,
    /// Estimated time spent (microseconds)
    pub estimated_time_us: f64,
    /// Percentage of total execution time
    pub percentage_of_total: f64,
    /// Severity level
    pub severity: BottleneckSeverity,
    /// Recommended optimizations
    pub recommendations: Vec<String>,
}

/// Bottleneck analysis for a single model
#[derive(Debug, Clone)]
pub struct ModelBottleneckAnalysis {
    /// Model type name
    pub model_name: String,
    /// Profiling results
    pub results: ProfilingResults,
    /// Memory profile
    pub memory: MemoryProfile,
    /// Identified bottlenecks
    pub bottlenecks: Vec<BottleneckInfo>,
    /// Overall performance rating (0-100, higher is better)
    pub performance_score: f64,
}

impl ModelBottleneckAnalysis {
    /// Analyze a model for bottlenecks
    pub fn analyze<M: AutoregressiveModel>(
        model: M,
        model_name: String,
        num_steps: usize,
    ) -> ModelResult<Self> {
        let mut profiler = ModelProfiler::new(model).warmup_steps(10);

        // Profile inference
        let results = profiler.profile_inference(num_steps, 1)?;

        // Get memory profile
        let memory = profiler.estimate_memory_usage();

        // Identify bottlenecks based on performance characteristics
        let bottlenecks = Self::identify_bottlenecks(&results, &memory, &model_name);

        // Calculate performance score (0-100)
        let performance_score = Self::calculate_performance_score(&results, &memory);

        Ok(Self {
            model_name,
            results,
            memory,
            bottlenecks,
            performance_score,
        })
    }

    /// Identify bottlenecks from profiling results
    fn identify_bottlenecks(
        results: &ProfilingResults,
        memory: &MemoryProfile,
        model_name: &str,
    ) -> Vec<BottleneckInfo> {
        let mut bottlenecks = Vec::new();

        // Check latency bottlenecks
        if results.avg_latency_us > 200.0 {
            let severity = if results.avg_latency_us > 1000.0 {
                BottleneckSeverity::Critical
            } else if results.avg_latency_us > 500.0 {
                BottleneckSeverity::High
            } else {
                BottleneckSeverity::Medium
            };

            bottlenecks.push(BottleneckInfo {
                name: "High average latency".to_string(),
                description: format!(
                    "Average latency of {:.2}μs exceeds target of 200μs",
                    results.avg_latency_us
                ),
                estimated_time_us: results.avg_latency_us,
                percentage_of_total: 100.0,
                severity,
                recommendations: vec![
                    "Consider using SIMD optimizations".to_string(),
                    "Enable parallel processing for multi-head operations".to_string(),
                    "Use cache-friendly memory layouts".to_string(),
                ],
            });
        }

        // Check latency variance (high variance indicates unstable performance)
        let cv = results.std_dev_us / results.avg_latency_us; // Coefficient of variation
        if cv > 0.5 {
            bottlenecks.push(BottleneckInfo {
                name: "High latency variance".to_string(),
                description: format!(
                    "Standard deviation {:.2}μs is {:.1}% of mean, indicating unstable performance",
                    results.std_dev_us,
                    cv * 100.0
                ),
                estimated_time_us: results.std_dev_us,
                percentage_of_total: cv * 100.0,
                severity: if cv > 1.0 {
                    BottleneckSeverity::High
                } else {
                    BottleneckSeverity::Medium
                },
                recommendations: vec![
                    "Investigate cache misses and memory allocation patterns".to_string(),
                    "Use memory pooling to reduce allocation variance".to_string(),
                ],
            });
        }

        // Check memory usage
        let memory_mb = memory.total_estimated_bytes as f64 / (1024.0 * 1024.0);
        if memory_mb > 100.0 {
            bottlenecks.push(BottleneckInfo {
                name: "High memory usage".to_string(),
                description: format!("Estimated memory usage of {:.2}MB is high", memory_mb),
                estimated_time_us: 0.0,
                percentage_of_total: 0.0,
                severity: if memory_mb > 500.0 {
                    BottleneckSeverity::High
                } else {
                    BottleneckSeverity::Medium
                },
                recommendations: vec![
                    "Consider quantization (INT8/FP16) to reduce memory footprint".to_string(),
                    "Use sparse representations where applicable".to_string(),
                ],
            });
        }

        // Model-specific bottleneck identification
        if model_name.contains("Transformer") {
            // Transformer-specific bottlenecks
            if results.avg_latency_us > 500.0 {
                bottlenecks.push(BottleneckInfo {
                    name: "Quadratic attention complexity".to_string(),
                    description: "Transformer attention has O(N²) complexity per step".to_string(),
                    estimated_time_us: results.avg_latency_us * 0.7, // Estimate 70% in attention
                    percentage_of_total: 70.0,
                    severity: BottleneckSeverity::High,
                    recommendations: vec![
                        "Consider using linear attention variants (e.g., Performers)".to_string(),
                        "Use Flash Attention for memory-efficient attention".to_string(),
                        "Switch to SSM-based models (Mamba, RWKV) for O(1) inference".to_string(),
                    ],
                });
            }
        }

        bottlenecks
    }

    /// Calculate overall performance score (0-100)
    fn calculate_performance_score(results: &ProfilingResults, memory: &MemoryProfile) -> f64 {
        // Target: <100μs latency, <50MB memory
        let latency_score = 100.0 * (100.0 / (results.avg_latency_us + 100.0));
        let memory_mb = memory.total_estimated_bytes as f64 / (1024.0 * 1024.0);
        let memory_score = 100.0 * (50.0 / (memory_mb + 50.0));

        // Variance penalty
        let cv = results.std_dev_us / results.avg_latency_us;
        let stability_score = 100.0 * (1.0 / (1.0 + cv));

        // Weighted average
        (latency_score * 0.5 + memory_score * 0.3 + stability_score * 0.2).min(100.0)
    }

    /// Format analysis as a detailed report
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("\n═══════════════════════════════════════\n");
        report.push_str(&format!("  {} Analysis Report\n", self.model_name));
        report.push_str("═══════════════════════════════════════\n\n");

        // Performance score
        report.push_str(&format!(
            "Performance Score: {:.1}/100\n\n",
            self.performance_score
        ));

        // Profiling results
        report.push_str(&self.results.format_report());
        report.push('\n');

        // Memory profile
        report.push_str(&self.memory.format_report());
        report.push('\n');

        // Bottlenecks
        if self.bottlenecks.is_empty() {
            report.push_str("✓ No significant bottlenecks identified!\n");
        } else {
            report.push_str(&format!(
                "⚠ {} Bottleneck(s) Identified:\n\n",
                self.bottlenecks.len()
            ));

            for (i, bottleneck) in self.bottlenecks.iter().enumerate() {
                let severity_icon = match bottleneck.severity {
                    BottleneckSeverity::Low => "ℹ",
                    BottleneckSeverity::Medium => "⚠",
                    BottleneckSeverity::High => "⚠⚠",
                    BottleneckSeverity::Critical => "🔥",
                };

                report.push_str(&format!(
                    "{}. {} {}\n",
                    i + 1,
                    severity_icon,
                    bottleneck.name
                ));
                report.push_str(&format!("   {}\n", bottleneck.description));

                if bottleneck.estimated_time_us > 0.0 {
                    report.push_str(&format!(
                        "   Time: {:.2}μs ({:.1}% of total)\n",
                        bottleneck.estimated_time_us, bottleneck.percentage_of_total
                    ));
                }

                report.push_str("   Recommendations:\n");
                for rec in &bottleneck.recommendations {
                    report.push_str(&format!("     • {}\n", rec));
                }
                report.push('\n');
            }
        }

        report
    }
}

/// Comprehensive comparison of all models
#[derive(Debug, Clone)]
pub struct ComprehensiveComparison {
    /// Analyses for each model
    pub analyses: Vec<ModelBottleneckAnalysis>,
    /// Best performing model by latency
    pub fastest_model: String,
    /// Most memory efficient model
    pub most_memory_efficient: String,
    /// Overall best model (by performance score)
    pub best_overall: String,
}

impl ComprehensiveComparison {
    /// Generate comparison report
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push('\n');
        report.push_str("╔═══════════════════════════════════════════════════════════════╗\n");
        report.push_str("║      COMPREHENSIVE MODEL PERFORMANCE COMPARISON               ║\n");
        report.push_str("╚═══════════════════════════════════════════════════════════════╝\n\n");

        // Summary table
        report.push_str("┌─────────────┬──────────────┬──────────────┬──────────────┬────────┐\n");
        report.push_str("│ Model       │ Avg Latency  │ Throughput   │ Memory (MB)  │ Score  │\n");
        report.push_str("├─────────────┼──────────────┼──────────────┼──────────────┼────────┤\n");

        for analysis in &self.analyses {
            let memory_mb = analysis.memory.total_estimated_bytes as f64 / (1024.0 * 1024.0);
            report.push_str(&format!(
                "│ {:11} │ {:9.2} μs │ {:9.1} /s │ {:11.2}  │ {:5.1}  │\n",
                analysis.model_name,
                analysis.results.avg_latency_us,
                analysis.results.throughput_steps_per_sec,
                memory_mb,
                analysis.performance_score
            ));
        }

        report
            .push_str("└─────────────┴──────────────┴──────────────┴──────────────┴────────┘\n\n");

        // Winners
        report.push_str(&format!(
            "🏆 Fastest Model:            {}\n",
            self.fastest_model
        ));
        report.push_str(&format!(
            "💾 Most Memory Efficient:    {}\n",
            self.most_memory_efficient
        ));
        report.push_str(&format!(
            "⭐ Best Overall:             {}\n\n",
            self.best_overall
        ));

        // Detailed analyses
        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("           DETAILED BOTTLENECK ANALYSES\n");
        report.push_str("═══════════════════════════════════════════════════════════════\n");

        for analysis in &self.analyses {
            report.push_str(&analysis.format_report());
        }

        report
    }
}

/// Comprehensive profiler for all models
pub struct ComprehensiveProfiler {
    num_steps: usize,
}

impl ComprehensiveProfiler {
    /// Create a new comprehensive profiler
    pub fn new() -> Self {
        Self { num_steps: 1000 }
    }

    /// Set number of steps for profiling
    pub fn num_steps(mut self, steps: usize) -> Self {
        self.num_steps = steps;
        self
    }

    /// Profile all available models and generate comprehensive comparison
    pub fn profile_all_models(&self) -> ModelResult<ComprehensiveComparison> {
        use crate::{mamba::*, mamba2::*, rwkv::*, s4::*, s5::*, transformer::*};

        let mut analyses = Vec::new();

        // Define common configuration
        let hidden_dim = 256;
        let num_layers = 4;
        let state_dim = 64;

        // Profile Mamba
        let mamba_config = MambaConfig::default()
            .hidden_dim(hidden_dim)
            .state_dim(state_dim)
            .num_layers(num_layers);

        let mamba = Mamba::new(mamba_config)?;
        let mamba_analysis =
            ModelBottleneckAnalysis::analyze(mamba, "Mamba".to_string(), self.num_steps)?;
        analyses.push(mamba_analysis);

        // Profile Mamba2
        let mamba2_config = Mamba2Config::default()
            .hidden_dim(hidden_dim)
            .state_dim(state_dim)
            .num_layers(num_layers)
            .num_heads(4);

        let mamba2 = Mamba2::new(mamba2_config)?;
        let mamba2_analysis =
            ModelBottleneckAnalysis::analyze(mamba2, "Mamba2".to_string(), self.num_steps)?;
        analyses.push(mamba2_analysis);

        // Profile RWKV
        let rwkv_config = RwkvConfig::default()
            .hidden_dim(hidden_dim)
            .num_layers(num_layers)
            .num_heads(4);

        let rwkv = Rwkv::new(rwkv_config)?;
        let rwkv_analysis =
            ModelBottleneckAnalysis::analyze(rwkv, "RWKV".to_string(), self.num_steps)?;
        analyses.push(rwkv_analysis);

        // Profile S4D
        let s4_config = S4Config::default()
            .hidden_dim(hidden_dim)
            .state_dim(state_dim)
            .num_layers(num_layers);

        let s4 = S4D::new(s4_config)?;
        let s4_analysis = ModelBottleneckAnalysis::analyze(s4, "S4D".to_string(), self.num_steps)?;
        analyses.push(s4_analysis);

        // Profile S5 (S5Config doesn't have fluent setters, so we use new with defaults)
        let s5_config = S5Config::new(1, hidden_dim, num_layers);

        let s5 = S5::new(s5_config)?;
        let s5_analysis = ModelBottleneckAnalysis::analyze(s5, "S5".to_string(), self.num_steps)?;
        analyses.push(s5_analysis);

        // Profile Transformer
        let transformer_config = TransformerConfig::default()
            .hidden_dim(hidden_dim)
            .num_heads(4)
            .num_layers(num_layers);

        let transformer = Transformer::new(transformer_config)?;
        let transformer_analysis = ModelBottleneckAnalysis::analyze(
            transformer,
            "Transformer".to_string(),
            self.num_steps,
        )?;
        analyses.push(transformer_analysis);

        // Determine winners
        let fastest_model = analyses
            .iter()
            .min_by(|a, b| {
                a.results
                    .avg_latency_us
                    .partial_cmp(&b.results.avg_latency_us)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|a| a.model_name.clone())
            .unwrap_or_default();

        let most_memory_efficient = analyses
            .iter()
            .min_by(|a, b| {
                a.memory
                    .total_estimated_bytes
                    .cmp(&b.memory.total_estimated_bytes)
            })
            .map(|a| a.model_name.clone())
            .unwrap_or_default();

        let best_overall = analyses
            .iter()
            .max_by(|a, b| {
                a.performance_score
                    .partial_cmp(&b.performance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|a| a.model_name.clone())
            .unwrap_or_default();

        Ok(ComprehensiveComparison {
            analyses,
            fastest_model,
            most_memory_efficient,
            best_overall,
        })
    }
}

impl Default for ComprehensiveProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba::{Mamba, MambaConfig};

    #[test]
    fn test_profiling_results() {
        let timings = vec![
            Duration::from_micros(100),
            Duration::from_micros(150),
            Duration::from_micros(120),
            Duration::from_micros(200),
            Duration::from_micros(110),
        ];

        let results = ProfilingResults::from_timings(&timings);

        assert_eq!(results.num_steps, 5);
        assert!(results.avg_latency_us > 0.0);
        assert!(results.min_latency_us <= results.avg_latency_us);
        assert!(results.avg_latency_us <= results.max_latency_us);
        assert!(results.throughput_steps_per_sec > 0.0);
    }

    #[test]
    #[ignore] // Slow test: ~39s due to comprehensive profiling (100 steps)
    fn test_model_profiler() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let model = Mamba::new(config).expect("Failed to create Mamba model");

        let mut profiler = ModelProfiler::new(model).warmup_steps(5);

        let results = profiler
            .profile_inference(100, 1)
            .expect("Failed to profile inference");

        assert_eq!(results.num_steps, 100);
        assert!(results.avg_latency_us > 0.0);
        assert!(results.throughput_steps_per_sec > 0.0);
    }

    #[test]
    #[ignore] // Slow test: ~104s due to multiple model creation and profiling runs
    fn test_input_scaling() {
        // Test with different hidden dimensions (not input dims, since input_dim is fixed by model config)
        let hidden_dims = vec![32, 64, 128];
        let mut results = Vec::new();

        for hidden_dim in hidden_dims {
            let config = MambaConfig::default().hidden_dim(hidden_dim).num_layers(2);
            let model = Mamba::new(config).expect("Failed to create Mamba model");

            let mut profiler = ModelProfiler::new(model).warmup_steps(5);
            let profile = profiler
                .profile_inference(50, 1)
                .expect("Failed to profile inference");

            results.push((hidden_dim, profile));
        }

        assert_eq!(results.len(), 3);

        // Verify all profiles are valid
        for (dim, result) in &results {
            assert_eq!(result.num_steps, 50);
            assert!(*dim > 0);
            assert!(result.avg_latency_us > 0.0);
        }
    }

    #[test]
    fn test_memory_profile() {
        let config = MambaConfig::default().hidden_dim(256).num_layers(4);
        let model = Mamba::new(config).expect("Failed to create Mamba model");

        let profiler = ModelProfiler::new(model);
        let memory = profiler.estimate_memory_usage();

        assert_eq!(memory.hidden_dim, 256);
        assert_eq!(memory.num_layers, 4);
        assert!(memory.total_estimated_bytes > 0);
    }

    #[test]
    #[ignore] // Slow test: ~32s due to benchmark suite execution (100 steps)
    fn test_benchmark_suite() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let model = Mamba::new(config).expect("Failed to create Mamba model");

        let suite = BenchmarkSuite::new().num_steps(100).warmup_steps(5);

        let results = suite.benchmark(model).expect("Failed to run benchmark");

        assert_eq!(results.num_steps, 100);
        assert!(results.avg_latency_us > 0.0);
    }

    #[test]
    fn test_format_report() {
        let timings = vec![Duration::from_micros(100); 10];
        let results = ProfilingResults::from_timings(&timings);

        let report = results.format_report();
        assert!(report.contains("Profiling Results"));
        assert!(report.contains("Average Latency"));
        assert!(report.contains("Throughput"));
    }
}
