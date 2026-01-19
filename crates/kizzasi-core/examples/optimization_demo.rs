//! Demonstration of performance optimizations in kizzasi-core
//!
//! This example shows the performance improvements from:
//! 1. Discretization caching
//! 2. Workspace pooling (allocation reduction)
//! 3. ILP-optimized operations
//! 4. Cache-aligned data structures

use kizzasi_core::optimizations::{ilp, CacheAligned, DiscretizationCache, WorkspaceGuard};
use kizzasi_core::profiling::{PerfCounter, Timer};
use scirs2_core::ndarray::{arr1, Array1, Array2};

fn main() {
    println!("=== Kizzasi Core Optimization Demonstration ===\n");

    demo_discretization_cache();
    demo_workspace_pooling();
    demo_ilp_operations();
    demo_cache_aligned();
}

fn demo_discretization_cache() {
    println!("1. Discretization Cache");
    println!("{}", "=".repeat(50));

    let num_layers = 4;
    let hidden_dim = 256;
    let state_dim = 16;

    let mut cache = DiscretizationCache::new(num_layers, hidden_dim, state_dim);
    let delta = 0.1f32;

    let counter_uncached = PerfCounter::new();
    let counter_cached = PerfCounter::new();

    // Warmup
    for layer in 0..num_layers {
        let a = Array2::ones((hidden_dim, state_dim));
        let b = Array2::ones((hidden_dim, state_dim));
        let a_bar = a.mapv(|x: f32| (delta * x).exp());
        let b_bar = b.mapv(|x: f32| delta * x);
        cache.update(layer, delta, a_bar, b_bar);
    }

    // Benchmark uncached
    for _ in 0..1000 {
        let timer = Timer::start();
        for _layer in 0..num_layers {
            let a = Array2::ones((hidden_dim, state_dim));
            let b = Array2::ones((hidden_dim, state_dim));
            let _a_bar = a.mapv(|x: f32| (delta * x).exp());
            let _b_bar = b.mapv(|x: f32| delta * x);
        }
        counter_uncached.record(timer.elapsed_ns());
    }

    // Benchmark cached
    for _ in 0..1000 {
        let timer = Timer::start();
        for layer in 0..num_layers {
            let _cached = cache.get(layer, delta).expect("cache should hit");
        }
        counter_cached.record(timer.elapsed_ns());
    }

    let uncached_stats = counter_uncached.stats();
    let cached_stats = counter_cached.stats();

    println!("Uncached: {:.2}μs avg", uncached_stats.average_us());
    println!("Cached:   {:.2}μs avg", cached_stats.average_us());
    println!(
        "Speedup:  {:.2}x\n",
        uncached_stats.average_us() / cached_stats.average_us()
    );
}

fn demo_workspace_pooling() {
    println!("2. Workspace Pooling (Allocation Reduction)");
    println!("{}", "=".repeat(50));

    let hidden_dim = 256;
    let state_dim = 16;
    let iterations = 10000;

    let counter_alloc = PerfCounter::new();
    let counter_pooled = PerfCounter::new();

    // Benchmark with allocations
    for _ in 0..iterations {
        let timer = Timer::start();
        let _temp = Array1::<f32>::zeros(hidden_dim);
        let _temp2 = Array2::<f32>::zeros((hidden_dim, state_dim));
        counter_alloc.record(timer.elapsed_ns());
    }

    // Benchmark with pooling
    for _ in 0..iterations {
        let timer = Timer::start();
        let mut _guard = WorkspaceGuard::new(hidden_dim, state_dim);
        counter_pooled.record(timer.elapsed_ns());
    }

    let alloc_stats = counter_alloc.stats();
    let pooled_stats = counter_pooled.stats();

    println!("With Allocation: {:.2}ns avg", alloc_stats.average_ns);
    println!("With Pooling:    {:.2}ns avg", pooled_stats.average_ns);
    println!(
        "Speedup:         {:.2}x\n",
        alloc_stats.average_ns as f64 / pooled_stats.average_ns as f64
    );
}

fn demo_ilp_operations() {
    println!("3. Instruction-Level Parallelism");
    println!("{}", "=".repeat(50));

    let size = 1024;
    let a = arr1(&vec![1.0f32; size]);
    let b = arr1(&vec![2.0f32; size]);

    let counter_standard = PerfCounter::new();
    let counter_ilp = PerfCounter::new();

    // Benchmark standard dot product
    for _ in 0..10000 {
        let timer = Timer::start();
        let _result = a.dot(&b);
        counter_standard.record(timer.elapsed_ns());
    }

    // Benchmark ILP dot product
    for _ in 0..10000 {
        let timer = Timer::start();
        let _result = ilp::dot_unrolled(a.view(), b.view());
        counter_ilp.record(timer.elapsed_ns());
    }

    let standard_stats = counter_standard.stats();
    let ilp_stats = counter_ilp.stats();

    println!("Standard Dot: {:.2}ns avg", standard_stats.average_ns);
    println!("ILP Dot:      {:.2}ns avg", ilp_stats.average_ns);
    println!(
        "Speedup:      {:.2}x\n",
        standard_stats.average_ns as f64 / ilp_stats.average_ns as f64
    );
}

fn demo_cache_aligned() {
    println!("4. Cache-Aligned Data Structures");
    println!("{}", "=".repeat(50));

    let size = 1024;
    let counter_regular = PerfCounter::new();
    let counter_aligned = PerfCounter::new();

    // Benchmark regular Vec
    let regular_data = vec![1.0f32; size];
    for _ in 0..10000 {
        let timer = Timer::start();
        let _sum: f32 = regular_data.iter().sum();
        counter_regular.record(timer.elapsed_ns());
    }

    // Benchmark cache-aligned Vec
    let aligned_data = CacheAligned::new(vec![1.0f32; size]);
    for _ in 0..10000 {
        let timer = Timer::start();
        let _sum: f32 = aligned_data.get().iter().sum();
        counter_aligned.record(timer.elapsed_ns());
    }

    let regular_stats = counter_regular.stats();
    let aligned_stats = counter_aligned.stats();

    println!("Regular:  {:.2}ns avg", regular_stats.average_ns);
    println!("Aligned:  {:.2}ns avg", aligned_stats.average_ns);
    println!(
        "Speedup:  {:.2}x\n",
        regular_stats.average_ns as f64 / aligned_stats.average_ns as f64
    );

    println!("=== Summary ===");
    println!("All optimizations combined provide significant performance improvements:");
    println!("- Cache: Reduces repeated computations");
    println!("- Pooling: Eliminates allocation overhead");
    println!("- ILP: Exploits CPU parallelism within instructions");
    println!("- Alignment: Improves cache line utilization");
}
