//! Memory usage profiling and performance monitoring utilities.
//!
//! This module provides tools for profiling memory usage, tracking allocations,
//! and analyzing performance characteristics of tokenizers.
//!
//! # Features
//!
//! - Memory usage tracking per operation
//! - Peak memory detection
//! - Allocation statistics
//! - Memory leak detection
//! - Performance benchmarking
//! - Timeline analysis
//!
//! # Example
//!
//! ```
//! use kizzasi_tokenizer::profiling::{MemoryProfiler, ProfileScope};
//! use kizzasi_tokenizer::{LinearQuantizer, Quantizer};
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut profiler = MemoryProfiler::new();
//!
//! {
//!     let _scope = ProfileScope::new(&mut profiler, "quantization");
//!     let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).expect("Quantizer creation failed");
//!     let signal = Array1::linspace(-1.0, 1.0, 1000);
//!     // Quantize individual values
//!     let _encoded: Vec<i32> = signal.iter().map(|&x| quantizer.quantize(x)).collect();
//! }
//!
//! let report = profiler.report();
//! println!("{}", report);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Memory allocation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    /// Timestamp of the allocation
    pub timestamp: u64,
    /// Size in bytes
    pub size: usize,
    /// Operation name
    pub operation: String,
    /// Allocation or deallocation
    pub event_type: EventType,
}

/// Type of memory event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// Memory allocation
    Allocation,
    /// Memory deallocation
    Deallocation,
}

/// Statistics for a profiling scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeStats {
    /// Operation name
    pub name: String,
    /// Number of times executed
    pub count: usize,
    /// Total time spent
    pub total_duration: Duration,
    /// Minimum duration
    pub min_duration: Duration,
    /// Maximum duration
    pub max_duration: Duration,
    /// Peak memory used (bytes)
    pub peak_memory: usize,
    /// Total memory allocated (bytes)
    pub total_allocated: usize,
    /// Total memory deallocated (bytes)
    pub total_deallocated: usize,
}

impl ScopeStats {
    /// Create new scope statistics
    fn new(name: String) -> Self {
        Self {
            name,
            count: 0,
            total_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            peak_memory: 0,
            total_allocated: 0,
            total_deallocated: 0,
        }
    }

    /// Update with new measurement
    fn update(&mut self, duration: Duration, allocated: usize, deallocated: usize) {
        self.count += 1;
        self.total_duration += duration;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        self.total_allocated += allocated;
        self.total_deallocated += deallocated;

        let current_usage = self.total_allocated.saturating_sub(self.total_deallocated);
        self.peak_memory = self.peak_memory.max(current_usage);
    }

    /// Get average duration
    pub fn avg_duration(&self) -> Duration {
        if self.count > 0 {
            self.total_duration / self.count as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get net memory usage (allocated - deallocated)
    pub fn net_memory(&self) -> isize {
        self.total_allocated as isize - self.total_deallocated as isize
    }
}

/// Memory profiler for tracking allocations and performance
#[derive(Debug)]
pub struct MemoryProfiler {
    /// Start time
    start_time: Instant,
    /// All allocation events
    events: Vec<AllocationEvent>,
    /// Statistics per scope
    scopes: HashMap<String, ScopeStats>,
    /// Current memory usage
    current_memory: usize,
    /// Peak memory usage
    peak_memory: usize,
    /// Active scopes stack
    active_scopes: Vec<(String, Instant, usize)>,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            events: Vec::new(),
            scopes: HashMap::new(),
            current_memory: 0,
            peak_memory: 0,
            active_scopes: Vec::new(),
        }
    }

    /// Get elapsed time since profiler creation
    fn elapsed(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }

    /// Record an allocation
    pub fn record_allocation(&mut self, operation: &str, size: usize) {
        let event = AllocationEvent {
            timestamp: self.elapsed(),
            size,
            operation: operation.to_string(),
            event_type: EventType::Allocation,
        };

        self.current_memory += size;
        self.peak_memory = self.peak_memory.max(self.current_memory);
        self.events.push(event);
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, operation: &str, size: usize) {
        let event = AllocationEvent {
            timestamp: self.elapsed(),
            size,
            operation: operation.to_string(),
            event_type: EventType::Deallocation,
        };

        self.current_memory = self.current_memory.saturating_sub(size);
        self.events.push(event);
    }

    /// Start a profiling scope
    pub fn start_scope(&mut self, name: &str) {
        let start_time = Instant::now();
        let start_memory = self.current_memory;
        self.active_scopes
            .push((name.to_string(), start_time, start_memory));
    }

    /// End a profiling scope
    pub fn end_scope(&mut self, name: &str) {
        if let Some(pos) = self.active_scopes.iter().rposition(|(n, _, _)| n == name) {
            let (scope_name, start_time, start_memory) = self.active_scopes.remove(pos);
            let duration = start_time.elapsed();
            let end_memory = self.current_memory;

            let allocated = end_memory.saturating_sub(start_memory);
            let deallocated = start_memory.saturating_sub(end_memory);

            self.scopes
                .entry(scope_name.clone())
                .or_insert_with(|| ScopeStats::new(scope_name))
                .update(duration, allocated, deallocated);
        }
    }

    /// Get current memory usage
    pub fn current_memory(&self) -> usize {
        self.current_memory
    }

    /// Get peak memory usage
    pub fn peak_memory(&self) -> usize {
        self.peak_memory
    }

    /// Get total number of allocation events
    pub fn total_allocations(&self) -> usize {
        self.events
            .iter()
            .filter(|e| e.event_type == EventType::Allocation)
            .count()
    }

    /// Get total number of deallocation events
    pub fn total_deallocations(&self) -> usize {
        self.events
            .iter()
            .filter(|e| e.event_type == EventType::Deallocation)
            .count()
    }

    /// Get total bytes allocated
    pub fn total_bytes_allocated(&self) -> usize {
        self.events
            .iter()
            .filter(|e| e.event_type == EventType::Allocation)
            .map(|e| e.size)
            .sum()
    }

    /// Get total bytes deallocated
    pub fn total_bytes_deallocated(&self) -> usize {
        self.events
            .iter()
            .filter(|e| e.event_type == EventType::Deallocation)
            .map(|e| e.size)
            .sum()
    }

    /// Check for potential memory leaks
    pub fn check_leaks(&self) -> Vec<String> {
        let mut leaks = Vec::new();

        for (name, stats) in &self.scopes {
            let net = stats.net_memory();
            if net > 1024 {
                // More than 1KB leaked
                leaks.push(format!("{}: {} bytes potentially leaked", name, net));
            }
        }

        if self.current_memory > 1024 && self.total_deallocations() < self.total_allocations() {
            leaks.push(format!(
                "Global: {} bytes not deallocated ({} allocs, {} deallocs)",
                self.current_memory,
                self.total_allocations(),
                self.total_deallocations()
            ));
        }

        leaks
    }

    /// Get statistics for a specific scope
    pub fn scope_stats(&self, name: &str) -> Option<&ScopeStats> {
        self.scopes.get(name)
    }

    /// Get all scope statistics
    pub fn all_scopes(&self) -> &HashMap<String, ScopeStats> {
        &self.scopes
    }

    /// Get all events
    pub fn events(&self) -> &[AllocationEvent] {
        &self.events
    }

    /// Generate a profiling report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Memory Profiling Report ===\n\n");

        // Overall statistics
        report.push_str("Overall Statistics:\n");
        report.push_str(&format!(
            "  Peak memory: {} bytes ({:.2} MB)\n",
            self.peak_memory,
            self.peak_memory as f64 / 1024.0 / 1024.0
        ));
        report.push_str(&format!(
            "  Current memory: {} bytes ({:.2} MB)\n",
            self.current_memory,
            self.current_memory as f64 / 1024.0 / 1024.0
        ));
        report.push_str(&format!(
            "  Total allocations: {}\n",
            self.total_allocations()
        ));
        report.push_str(&format!(
            "  Total deallocations: {}\n",
            self.total_deallocations()
        ));
        report.push_str(&format!(
            "  Total allocated: {} bytes ({:.2} MB)\n",
            self.total_bytes_allocated(),
            self.total_bytes_allocated() as f64 / 1024.0 / 1024.0
        ));
        report.push_str(&format!(
            "  Total deallocated: {} bytes ({:.2} MB)\n\n",
            self.total_bytes_deallocated(),
            self.total_bytes_deallocated() as f64 / 1024.0 / 1024.0
        ));

        // Scope statistics
        if !self.scopes.is_empty() {
            report.push_str("Scope Statistics:\n");
            let mut scopes: Vec<_> = self.scopes.iter().collect();
            scopes.sort_by_key(|(_, stats)| std::cmp::Reverse(stats.peak_memory));

            for (name, stats) in scopes {
                report.push_str(&format!("\n  {}:\n", name));
                report.push_str(&format!("    Count: {}\n", stats.count));
                report.push_str(&format!("    Avg duration: {:?}\n", stats.avg_duration()));
                report.push_str(&format!("    Min duration: {:?}\n", stats.min_duration));
                report.push_str(&format!("    Max duration: {:?}\n", stats.max_duration));
                report.push_str(&format!("    Peak memory: {} bytes\n", stats.peak_memory));
                report.push_str(&format!(
                    "    Total allocated: {} bytes\n",
                    stats.total_allocated
                ));
                report.push_str(&format!(
                    "    Total deallocated: {} bytes\n",
                    stats.total_deallocated
                ));
                report.push_str(&format!("    Net memory: {} bytes\n", stats.net_memory()));
            }
            report.push('\n');
        }

        // Memory leaks
        let leaks = self.check_leaks();
        if !leaks.is_empty() {
            report.push_str("Potential Memory Leaks:\n");
            for leak in leaks {
                report.push_str(&format!("  - {}\n", leak));
            }
        } else {
            report.push_str("No memory leaks detected.\n");
        }

        report
    }

    /// Clear all profiling data
    pub fn clear(&mut self) {
        self.events.clear();
        self.scopes.clear();
        self.current_memory = 0;
        self.peak_memory = 0;
        self.active_scopes.clear();
        self.start_time = Instant::now();
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII scope for automatic profiling
pub struct ProfileScope<'a> {
    profiler: &'a mut MemoryProfiler,
    name: String,
}

impl<'a> ProfileScope<'a> {
    /// Create a new profile scope
    pub fn new(profiler: &'a mut MemoryProfiler, name: &str) -> Self {
        profiler.start_scope(name);
        Self {
            profiler,
            name: name.to_string(),
        }
    }
}

impl<'a> Drop for ProfileScope<'a> {
    fn drop(&mut self) {
        self.profiler.end_scope(&self.name);
    }
}

/// Memory snapshot for timeline analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Timestamp in microseconds
    pub timestamp: u64,
    /// Memory usage in bytes
    pub memory_used: usize,
    /// Active operation
    pub operation: Option<String>,
}

/// Timeline analyzer for memory usage over time
#[derive(Debug)]
pub struct TimelineAnalyzer {
    /// Memory snapshots
    snapshots: Vec<MemorySnapshot>,
    /// Start time
    start_time: Instant,
}

impl TimelineAnalyzer {
    /// Create a new timeline analyzer
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Record a snapshot
    pub fn snapshot(&mut self, memory_used: usize, operation: Option<&str>) {
        let timestamp = self.start_time.elapsed().as_micros() as u64;
        self.snapshots.push(MemorySnapshot {
            timestamp,
            memory_used,
            operation: operation.map(String::from),
        });
    }

    /// Get all snapshots
    pub fn snapshots(&self) -> &[MemorySnapshot] {
        &self.snapshots
    }

    /// Get peak memory usage
    pub fn peak_memory(&self) -> usize {
        self.snapshots
            .iter()
            .map(|s| s.memory_used)
            .max()
            .unwrap_or(0)
    }

    /// Get average memory usage
    pub fn avg_memory(&self) -> usize {
        if self.snapshots.is_empty() {
            0
        } else {
            self.snapshots.iter().map(|s| s.memory_used).sum::<usize>() / self.snapshots.len()
        }
    }

    /// Find operations with highest memory usage
    pub fn top_operations(&self, n: usize) -> Vec<(String, usize)> {
        let mut op_memory: HashMap<String, usize> = HashMap::new();

        for snapshot in &self.snapshots {
            if let Some(ref op) = snapshot.operation {
                let entry = op_memory.entry(op.clone()).or_insert(0);
                *entry = (*entry).max(snapshot.memory_used);
            }
        }

        let mut sorted: Vec<_> = op_memory.into_iter().collect();
        sorted.sort_by_key(|(_, mem)| std::cmp::Reverse(*mem));
        sorted.truncate(n);
        sorted
    }

    /// Generate CSV report
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("timestamp_us,memory_bytes,operation\n");
        for snapshot in &self.snapshots {
            csv.push_str(&format!(
                "{},{},{}\n",
                snapshot.timestamp,
                snapshot.memory_used,
                snapshot.operation.as_deref().unwrap_or("")
            ));
        }
        csv
    }

    /// Clear all snapshots
    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.start_time = Instant::now();
    }
}

impl Default for TimelineAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler_creation() {
        let profiler = MemoryProfiler::new();
        assert_eq!(profiler.current_memory(), 0);
        assert_eq!(profiler.peak_memory(), 0);
    }

    #[test]
    fn test_record_allocation() {
        let mut profiler = MemoryProfiler::new();
        profiler.record_allocation("test", 1024);
        assert_eq!(profiler.current_memory(), 1024);
        assert_eq!(profiler.peak_memory(), 1024);
        assert_eq!(profiler.total_allocations(), 1);
    }

    #[test]
    fn test_record_deallocation() {
        let mut profiler = MemoryProfiler::new();
        profiler.record_allocation("test", 2048);
        profiler.record_deallocation("test", 1024);
        assert_eq!(profiler.current_memory(), 1024);
        assert_eq!(profiler.peak_memory(), 2048);
    }

    #[test]
    fn test_scope_tracking() {
        let mut profiler = MemoryProfiler::new();

        profiler.start_scope("operation1");
        profiler.record_allocation("op1", 512);
        std::thread::sleep(Duration::from_millis(10));
        profiler.end_scope("operation1");

        let stats = profiler.scope_stats("operation1").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.total_duration >= Duration::from_millis(10));
        assert_eq!(stats.total_allocated, 512);
    }

    #[test]
    fn test_profile_scope_raii() {
        let mut profiler = MemoryProfiler::new();

        {
            let _scope = ProfileScope::new(&mut profiler, "scoped_op");
            std::thread::sleep(Duration::from_millis(5));
        }

        let stats = profiler.scope_stats("scoped_op").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.total_duration >= Duration::from_millis(5));
    }

    #[test]
    fn test_nested_scopes() {
        let mut profiler = MemoryProfiler::new();

        profiler.start_scope("outer");
        profiler.start_scope("inner");
        profiler.end_scope("inner");
        profiler.end_scope("outer");

        assert!(profiler.scope_stats("outer").is_some());
        assert!(profiler.scope_stats("inner").is_some());
    }

    #[test]
    fn test_leak_detection() {
        let mut profiler = MemoryProfiler::new();

        profiler.start_scope("leaky_op");
        profiler.record_allocation("leaky_op", 2048);
        // No deallocation!
        profiler.end_scope("leaky_op");

        let leaks = profiler.check_leaks();
        assert!(!leaks.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let mut profiler = MemoryProfiler::new();

        profiler.start_scope("test_op");
        profiler.record_allocation("test_op", 1024);
        profiler.record_deallocation("test_op", 1024);
        profiler.end_scope("test_op");

        let report = profiler.report();
        assert!(report.contains("Memory Profiling Report"));
        assert!(report.contains("test_op"));
    }

    #[test]
    fn test_clear() {
        let mut profiler = MemoryProfiler::new();
        profiler.record_allocation("test", 1024);
        profiler.clear();

        assert_eq!(profiler.current_memory(), 0);
        assert_eq!(profiler.peak_memory(), 0);
        assert_eq!(profiler.total_allocations(), 0);
    }

    #[test]
    fn test_timeline_analyzer() {
        let mut analyzer = TimelineAnalyzer::new();

        analyzer.snapshot(1024, Some("op1"));
        analyzer.snapshot(2048, Some("op2"));
        analyzer.snapshot(512, Some("op1"));

        assert_eq!(analyzer.snapshots().len(), 3);
        assert_eq!(analyzer.peak_memory(), 2048);
        assert_eq!(analyzer.avg_memory(), (1024 + 2048 + 512) / 3);
    }

    #[test]
    fn test_top_operations() {
        let mut analyzer = TimelineAnalyzer::new();

        analyzer.snapshot(1024, Some("op1"));
        analyzer.snapshot(2048, Some("op2"));
        analyzer.snapshot(1536, Some("op1"));
        analyzer.snapshot(512, Some("op3"));

        let top = analyzer.top_operations(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "op2");
        assert_eq!(top[0].1, 2048);
    }

    #[test]
    fn test_csv_export() {
        let mut analyzer = TimelineAnalyzer::new();
        analyzer.snapshot(1024, Some("test"));

        let csv = analyzer.to_csv();
        assert!(csv.contains("timestamp_us,memory_bytes,operation"));
        assert!(csv.contains("1024"));
        assert!(csv.contains("test"));
    }

    #[test]
    fn test_scope_stats_avg_duration() {
        let mut stats = ScopeStats::new("test".to_string());
        stats.update(Duration::from_millis(10), 0, 0);
        stats.update(Duration::from_millis(20), 0, 0);
        stats.update(Duration::from_millis(30), 0, 0);

        let avg = stats.avg_duration();
        assert_eq!(avg, Duration::from_millis(20));
    }

    #[test]
    fn test_scope_stats_peak_memory() {
        let mut stats = ScopeStats::new("test".to_string());
        stats.update(Duration::from_millis(1), 1024, 0);
        stats.update(Duration::from_millis(1), 2048, 512);
        stats.update(Duration::from_millis(1), 512, 1024);

        // Peak should track maximum net memory
        assert!(stats.peak_memory > 0);
    }
}
