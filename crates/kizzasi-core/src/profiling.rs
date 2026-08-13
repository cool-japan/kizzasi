//! Performance profiling utilities for kizzasi-core
//!
//! Provides tools for measuring and analyzing performance:
//! - CPU cycle counting
//! - Time measurement with high precision
//! - Memory profiling
//! - Operation counters
//! - Statistical analysis

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Performance counter for tracking operations
#[derive(Debug, Default)]
pub struct PerfCounter {
    count: AtomicUsize,
    total_time_ns: AtomicU64,
    min_time_ns: AtomicU64,
    max_time_ns: AtomicU64,
}

impl PerfCounter {
    /// Create a new performance counter
    pub const fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
            total_time_ns: AtomicU64::new(0),
            min_time_ns: AtomicU64::new(u64::MAX),
            max_time_ns: AtomicU64::new(0),
        }
    }

    /// Record a measurement
    pub fn record(&self, duration_ns: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_time_ns.fetch_add(duration_ns, Ordering::Relaxed);

        // Update min
        let mut current_min = self.min_time_ns.load(Ordering::Relaxed);
        while duration_ns < current_min {
            match self.min_time_ns.compare_exchange_weak(
                current_min,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }

        // Update max
        let mut current_max = self.max_time_ns.load(Ordering::Relaxed);
        while duration_ns > current_max {
            match self.max_time_ns.compare_exchange_weak(
                current_max,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }

    /// Get the number of measurements
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the total time in nanoseconds
    pub fn total_ns(&self) -> u64 {
        self.total_time_ns.load(Ordering::Relaxed)
    }

    /// Get the average time in nanoseconds
    pub fn average_ns(&self) -> u64 {
        let count = self.count();
        if count == 0 {
            return 0;
        }
        self.total_ns() / count as u64
    }

    /// Get the minimum time in nanoseconds
    pub fn min_ns(&self) -> u64 {
        let min = self.min_time_ns.load(Ordering::Relaxed);
        if min == u64::MAX {
            0
        } else {
            min
        }
    }

    /// Get the maximum time in nanoseconds
    pub fn max_ns(&self) -> u64 {
        self.max_time_ns.load(Ordering::Relaxed)
    }

    /// Reset the counter
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.total_time_ns.store(0, Ordering::Relaxed);
        self.min_time_ns.store(u64::MAX, Ordering::Relaxed);
        self.max_time_ns.store(0, Ordering::Relaxed);
    }

    /// Get statistics summary
    pub fn stats(&self) -> CounterStats {
        CounterStats {
            count: self.count(),
            total_ns: self.total_ns(),
            average_ns: self.average_ns(),
            min_ns: self.min_ns(),
            max_ns: self.max_ns(),
        }
    }
}

/// Statistics snapshot from a performance counter
#[derive(Debug, Clone, Copy)]
pub struct CounterStats {
    pub count: usize,
    pub total_ns: u64,
    pub average_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
}

impl CounterStats {
    /// Convert to microseconds
    pub fn average_us(&self) -> f64 {
        self.average_ns as f64 / 1000.0
    }

    /// Convert to milliseconds
    pub fn average_ms(&self) -> f64 {
        self.average_ns as f64 / 1_000_000.0
    }

    /// Get throughput (operations per second)
    pub fn throughput(&self) -> f64 {
        if self.average_ns == 0 {
            return 0.0;
        }
        1_000_000_000.0 / self.average_ns as f64
    }
}

/// Timer for measuring elapsed time
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

    /// Get elapsed time in nanoseconds
    pub fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    /// Get elapsed time as Duration
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Reset the timer
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

/// RAII scope timer that records to a counter on drop
pub struct ScopeTimer<'a> {
    counter: &'a PerfCounter,
    start: Instant,
}

impl<'a> ScopeTimer<'a> {
    /// Create a new scope timer
    pub fn new(counter: &'a PerfCounter) -> Self {
        Self {
            counter,
            start: Instant::now(),
        }
    }
}

impl<'a> Drop for ScopeTimer<'a> {
    fn drop(&mut self) {
        let elapsed_ns = self.start.elapsed().as_nanos() as u64;
        self.counter.record(elapsed_ns);
    }
}

/// Memory profiler for tracking allocations
#[derive(Debug, Default)]
pub struct MemoryProfiler {
    current_bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
    allocations: AtomicUsize,
    deallocations: AtomicUsize,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub const fn new() -> Self {
        Self {
            current_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            allocations: AtomicUsize::new(0),
            deallocations: AtomicUsize::new(0),
        }
    }

    /// Record an allocation
    pub fn allocate(&self, bytes: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let new_current = self.current_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;

        // Update peak
        let mut current_peak = self.peak_bytes.load(Ordering::Relaxed);
        while new_current > current_peak {
            match self.peak_bytes.compare_exchange_weak(
                current_peak,
                new_current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_peak = x,
            }
        }
    }

    /// Record a deallocation
    pub fn deallocate(&self, bytes: usize) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Get current memory usage
    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Get peak memory usage
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Get allocation count
    pub fn allocations(&self) -> usize {
        self.allocations.load(Ordering::Relaxed)
    }

    /// Get deallocation count
    pub fn deallocations(&self) -> usize {
        self.deallocations.load(Ordering::Relaxed)
    }

    /// Get net allocations (allocations - deallocations)
    pub fn net_allocations(&self) -> isize {
        self.allocations() as isize - self.deallocations() as isize
    }

    /// Reset the profiler
    pub fn reset(&self) {
        self.current_bytes.store(0, Ordering::Relaxed);
        self.peak_bytes.store(0, Ordering::Relaxed);
        self.allocations.store(0, Ordering::Relaxed);
        self.deallocations.store(0, Ordering::Relaxed);
    }

    /// Get memory statistics
    pub fn stats(&self) -> ProfilerMemoryStats {
        ProfilerMemoryStats {
            current_bytes: self.current_bytes(),
            peak_bytes: self.peak_bytes(),
            allocations: self.allocations(),
            deallocations: self.deallocations(),
        }
    }
}

/// Memory statistics snapshot from profiler
#[derive(Debug, Clone, Copy)]
pub struct ProfilerMemoryStats {
    pub current_bytes: usize,
    pub peak_bytes: usize,
    pub allocations: usize,
    pub deallocations: usize,
}

impl ProfilerMemoryStats {
    /// Get current memory in MB
    pub fn current_mb(&self) -> f64 {
        self.current_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory in MB
    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Profiling session that tracks multiple counters
#[derive(Debug)]
pub struct ProfilingSession {
    name: String,
    counters: Vec<(String, PerfCounter)>,
    memory: MemoryProfiler,
    start_time: Instant,
}

impl ProfilingSession {
    /// Create a new profiling session
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            counters: Vec::new(),
            memory: MemoryProfiler::new(),
            start_time: Instant::now(),
        }
    }

    /// Add a counter to track
    pub fn add_counter(&mut self, name: impl Into<String>) -> usize {
        let idx = self.counters.len();
        self.counters.push((name.into(), PerfCounter::new()));
        idx
    }

    /// Get a counter by index
    pub fn counter(&self, idx: usize) -> Option<&PerfCounter> {
        self.counters.get(idx).map(|(_, c)| c)
    }

    /// Get the memory profiler
    pub fn memory(&self) -> &MemoryProfiler {
        &self.memory
    }

    /// Get total elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Generate a report
    pub fn report(&self) -> String {
        let mut output = format!("Profiling Session: {}\n", self.name);
        output.push_str(&format!("Total Time: {:?}\n\n", self.elapsed()));

        output.push_str("Performance Counters:\n");
        for (name, counter) in &self.counters {
            let stats = counter.stats();
            output.push_str(&format!(
                "  {}: {} calls, avg: {:.2}μs, min: {:.2}μs, max: {:.2}μs\n",
                name,
                stats.count,
                stats.average_us(),
                stats.min_ns as f64 / 1000.0,
                stats.max_ns as f64 / 1000.0
            ));
        }

        let mem_stats = self.memory.stats();
        output.push_str("\nMemory:\n");
        output.push_str(&format!("  Current: {:.2} MB\n", mem_stats.current_mb()));
        output.push_str(&format!("  Peak: {:.2} MB\n", mem_stats.peak_mb()));
        output.push_str(&format!(
            "  Allocations: {} (Net: {})\n",
            mem_stats.allocations,
            mem_stats.allocations as isize - mem_stats.deallocations as isize
        ));

        output
    }
}

/// Macro to time a block of code
#[macro_export]
macro_rules! time_block {
    ($counter:expr, $block:expr) => {{
        let _timer = $crate::profiling::ScopeTimer::new($counter);
        $block
    }};
}

/// Macro to profile memory usage of a block
#[macro_export]
macro_rules! profile_memory {
    ($profiler:expr, $size:expr, $block:expr) => {{
        $profiler.allocate($size);
        let result = $block;
        $profiler.deallocate($size);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_perf_counter() {
        let counter = PerfCounter::new();

        counter.record(100);
        counter.record(200);
        counter.record(150);

        assert_eq!(counter.count(), 3);
        assert_eq!(counter.total_ns(), 450);
        assert_eq!(counter.average_ns(), 150);
        assert_eq!(counter.min_ns(), 100);
        assert_eq!(counter.max_ns(), 200);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        thread::sleep(Duration::from_millis(10));
        let elapsed = timer.elapsed_ns();

        // Should be at least 10ms
        assert!(elapsed >= 10_000_000);
    }

    #[test]
    fn test_scope_timer() {
        let counter = PerfCounter::new();

        {
            let _timer = ScopeTimer::new(&counter);
            thread::sleep(Duration::from_millis(10));
        }

        assert_eq!(counter.count(), 1);
        assert!(counter.total_ns() >= 10_000_000);
    }

    #[test]
    fn test_memory_profiler() {
        let profiler = MemoryProfiler::new();

        profiler.allocate(1024);
        assert_eq!(profiler.current_bytes(), 1024);
        assert_eq!(profiler.peak_bytes(), 1024);

        profiler.allocate(2048);
        assert_eq!(profiler.current_bytes(), 3072);
        assert_eq!(profiler.peak_bytes(), 3072);

        profiler.deallocate(1024);
        assert_eq!(profiler.current_bytes(), 2048);
        assert_eq!(profiler.peak_bytes(), 3072); // Peak stays

        assert_eq!(profiler.allocations(), 2);
        assert_eq!(profiler.deallocations(), 1);
    }

    #[test]
    fn test_profiling_session() {
        let mut session = ProfilingSession::new("test_session");

        let counter_idx = session.add_counter("test_op");
        let counter = session.counter(counter_idx).unwrap();

        counter.record(100);
        counter.record(200);

        session.memory().allocate(1024);

        let report = session.report();
        assert!(report.contains("test_session"));
        assert!(report.contains("test_op"));
        assert!(report.contains("2 calls"));
    }

    #[test]
    fn test_counter_stats() {
        let counter = PerfCounter::new();
        counter.record(1_000_000); // 1ms

        let stats = counter.stats();
        assert_eq!(stats.average_us(), 1000.0);
        assert_eq!(stats.average_ms(), 1.0);
        assert_eq!(stats.throughput(), 1000.0); // 1000 ops/sec
    }

    #[test]
    fn test_concurrent_counter() {
        use std::sync::Arc;

        let counter = Arc::new(PerfCounter::new());
        let mut handles = vec![];

        // Spawn multiple threads recording measurements
        for i in 0..10 {
            let counter_clone = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    counter_clone.record((i + 1) * 100);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.count(), 1000);
    }

    #[test]
    fn test_time_block_macro() {
        let counter = PerfCounter::new();

        time_block!(&counter, {
            thread::sleep(Duration::from_millis(10));
        });

        assert_eq!(counter.count(), 1);
        assert!(counter.total_ns() >= 10_000_000);
    }

    #[test]
    fn test_profile_memory_macro() {
        let profiler = MemoryProfiler::new();

        profile_memory!(&profiler, 1024, {
            // Do some work
            let _v = vec![0u8; 1024];
        });

        assert_eq!(profiler.allocations(), 1);
        assert_eq!(profiler.deallocations(), 1);
        assert_eq!(profiler.current_bytes(), 0);
    }
}
