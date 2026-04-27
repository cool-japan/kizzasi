//! # Performance Tuning Demo for Kizzasi
//!
//! Demonstrates key performance optimization techniques for production deployments:
//!
//! 1. **Timing measurements** — Manual `std::time::Instant` profiling and where to hook in
//!    `kizzasi_model::profiling::ProfilingRegistry` for lower-level instrumentation.
//! 2. **Batch vs single-step inference** — Measuring throughput differences and when
//!    batch scheduling pays off.
//! 3. **Model size tradeoffs** — Comparing Mamba-Tiny vs larger hidden/state dimensions
//!    and how they affect latency vs quality.
//! 4. **State management** — Resetting vs reusing hidden state; warmup patterns.
//!
//! ## SIMD note
//!
//! On **ARM64** (Apple M-series, Ampere, Neoverse) the hot paths in
//! `kizzasi-core` use **NEON** intrinsics for dot products, layer norm, and
//! softmax.  On **x86-64** the equivalent code uses **AVX2/AVX-512**.  These
//! paths are selected at compile time through `scirs2-core`; no runtime flag is
//! needed.
//!
//! ## ProfilingRegistry
//!
//! For fine-grained per-operation timing inside the model itself, use
//! `kizzasi_model::profiling::ProfilingRegistry` (available when `kizzasi-model`
//! is a direct dependency of your crate):
//!
//! ```text
//! let mut reg = ProfilingRegistry::new();
//! reg.enable();
//! {
//!     let _g = TimingGuard::new(&mut reg, "ssm_step");
//!     model.step(&input)?;
//! }
//! for (name, acc) in reg.summary() {
//!     println!("{name}: {:.2}µs avg", acc.mean().unwrap().as_micros());
//! }
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example performance_tuning_demo -p kizzasi
//! ```

use kizzasi::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== Kizzasi Performance Tuning Demo ===\n");

    demo_timing_measurements()?;
    demo_batch_vs_single()?;
    demo_model_sizes()?;
    demo_state_management()?;

    println!("\n=== Performance Demo Complete ===");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Timing measurements
// ─────────────────────────────────────────────────────────────────────────────

/// Shows how to profile individual prediction steps with `std::time::Instant`.
///
/// For production workloads, hook in `kizzasi_model::profiling::ProfilingRegistry`
/// at the model-crate level to get per-layer breakdowns without changing call sites.
fn demo_timing_measurements() -> Result<()> {
    println!("--- 1. Timing Measurements ---");

    let mut predictor = KizzasiBuilder::new()
        .model_type(ModelType::Mamba2)
        .input_dim(8)
        .output_dim(8)
        .hidden_dim(64)
        .state_dim(16)
        .num_layers(2)
        .build()?;

    let input = array![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let warmup_steps = 10;
    let bench_steps = 100;

    // --- Warmup ---------------------------------------------------------
    // The first several steps initialise lazy state buffers and may touch cold
    // cache lines.  Always warm up before measuring.
    for _ in 0..warmup_steps {
        let _ = predictor.step(&input)?;
    }
    predictor.reset();

    // --- Benchmark ------------------------------------------------------
    let mut timings_us: Vec<f64> = Vec::with_capacity(bench_steps);
    for _ in 0..bench_steps {
        let t0 = Instant::now();
        let _ = predictor.step(&input)?;
        timings_us.push(t0.elapsed().as_nanos() as f64 / 1_000.0);
    }

    let total_us: f64 = timings_us.iter().sum();
    let avg_us = total_us / bench_steps as f64;
    let mut sorted = timings_us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_us = sorted.first().copied().unwrap_or(0.0);
    let max_us = sorted.last().copied().unwrap_or(0.0);
    let p95_us = sorted[(bench_steps as f64 * 0.95) as usize];
    let throughput = bench_steps as f64 / (total_us / 1_000_000.0);

    println!("  Model:         Mamba2 (hidden=64, state=16, layers=2)");
    println!(
        "  Steps:         {} (after {} warmup)",
        bench_steps, warmup_steps
    );
    println!("  Avg latency:   {:.2} µs", avg_us);
    println!("  Min latency:   {:.2} µs", min_us);
    println!("  Max latency:   {:.2} µs", max_us);
    println!("  P95 latency:   {:.2} µs", p95_us);
    println!("  Throughput:    {:.0} steps/sec", throughput);

    // SIMD hint: if avg_us is suspiciously high, confirm the binary was compiled
    // with `RUSTFLAGS="-C target-cpu=native"` to unlock AVX2 / NEON paths.
    println!("  [tip] Compile with RUSTFLAGS=\"-C target-cpu=native\" to enable");
    println!("        platform-optimal SIMD (NEON on ARM64, AVX2 on x86-64).\n");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Batch vs single-step inference
// ─────────────────────────────────────────────────────────────────────────────

/// Compares per-sample throughput when processing inputs one at a time versus
/// using `predict_batch`, which amortises fixed overhead across the batch.
fn demo_batch_vs_single() -> Result<()> {
    println!("--- 2. Batch vs Single-Step Inference ---");

    let mut predictor = KizzasiBuilder::new()
        .model_type(ModelType::Mamba2)
        .input_dim(4)
        .output_dim(4)
        .hidden_dim(128)
        .state_dim(16)
        .num_layers(3)
        .build()?;

    let batch_sizes = [1usize, 4, 8, 16, 32];
    let num_repeats = 20;

    println!(
        "  {:>10}  {:>16}  {:>16}  {:>14}",
        "batch_size", "total_time_ms", "per_sample_µs", "throughput/s"
    );
    println!("  {}", "-".repeat(60));

    for &bs in &batch_sizes {
        // Build a batch of independent inputs
        let inputs: Vec<Array1<f32>> = (0..bs)
            .map(|i| {
                let v = i as f32 / bs as f32;
                array![v, v * 0.5, v * 0.25, v * 0.125]
            })
            .collect();

        // Warmup
        for _ in 0..5 {
            predictor.reset();
            let _ = predictor.predict_batch(&inputs)?;
        }

        // Measure
        let t0 = Instant::now();
        for _ in 0..num_repeats {
            predictor.reset();
            let _ = predictor.predict_batch(&inputs)?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;

        let total_samples = bs * num_repeats;
        let per_sample_us = t0.elapsed().as_nanos() as f64 / (total_samples as f64 * 1_000.0);
        let throughput = total_samples as f64 / t0.elapsed().as_secs_f64();

        println!(
            "  {:>10}  {:>16.3}  {:>16.2}  {:>14.0}",
            bs, elapsed_ms, per_sample_us, throughput
        );
    }

    println!();
    println!("  [note] Larger batches amortise fixed overhead (state init, norm).");
    println!("         For latency-sensitive pipelines, use batch_size=1.");
    println!("         For throughput-oriented workloads, tune batch_size ≥ 8.\n");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Model size tradeoffs
// ─────────────────────────────────────────────────────────────────────────────

/// Compares three model configurations (Tiny / Small / Medium) by measuring
/// single-step latency and reporting the quality/speed tradeoff.
fn demo_model_sizes() -> Result<()> {
    println!("--- 3. Model Size vs Latency Tradeoff ---");

    struct Config {
        name: &'static str,
        hidden: usize,
        state: usize,
        layers: usize,
    }

    let configs = [
        Config {
            name: "Mamba-Tiny",
            hidden: 64,
            state: 8,
            layers: 2,
        },
        Config {
            name: "Mamba-Small",
            hidden: 128,
            state: 16,
            layers: 4,
        },
        Config {
            name: "Mamba-Medium",
            hidden: 256,
            state: 32,
            layers: 6,
        },
    ];

    let input_dim = 16usize;
    let input: Array1<f32> = Array1::from_vec((0..input_dim).map(|i| i as f32 / 16.0).collect());
    let steps = 50;

    println!(
        "  {:>14}  {:>8}  {:>7}  {:>7}  {:>14}  {:>13}",
        "config", "hidden", "state", "layers", "avg_latency_µs", "throughput/s"
    );
    println!("  {}", "-".repeat(70));

    for cfg in &configs {
        let mut predictor = KizzasiBuilder::new()
            .model_type(ModelType::Mamba)
            .input_dim(input_dim)
            .output_dim(input_dim)
            .hidden_dim(cfg.hidden)
            .state_dim(cfg.state)
            .num_layers(cfg.layers)
            .build()?;

        // Warmup
        for _ in 0..10 {
            let _ = predictor.step(&input)?;
        }
        predictor.reset();

        // Measure
        let t0 = Instant::now();
        for _ in 0..steps {
            let _ = predictor.step(&input)?;
        }
        let elapsed = t0.elapsed();
        let avg_us = elapsed.as_nanos() as f64 / (steps as f64 * 1_000.0);
        let throughput = steps as f64 / elapsed.as_secs_f64();

        let params_est = 2 * cfg.hidden * input_dim   // input/output proj
            + cfg.layers * (cfg.hidden * cfg.hidden   // in_proj
                + cfg.state * cfg.hidden              // ssm matrices
                + cfg.hidden); // norms/biases

        println!(
            "  {:>14}  {:>8}  {:>7}  {:>7}  {:>14.2}  {:>13.0}  (~{} params)",
            cfg.name, cfg.hidden, cfg.state, cfg.layers, avg_us, throughput, params_est
        );
    }

    println!();
    println!("  [guidance]");
    println!("    - Mamba-Tiny:   real-time embedded / edge (< 1 ms target)");
    println!("    - Mamba-Small:  desktop / server inference with moderate quality");
    println!("    - Mamba-Medium: high-quality signal prediction, GPU recommended");
    println!("    - Scale hidden_dim by 2× for roughly 4× parameter growth.\n");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. State management
// ─────────────────────────────────────────────────────────────────────────────

/// Demonstrates the difference between `reset()` (zero state) vs `fork()` for
/// branching, and how stateful inference amortises startup cost over long
/// sequences.
fn demo_state_management() -> Result<()> {
    println!("--- 4. State Management ---");

    let mut predictor = KizzasiBuilder::new()
        .model_type(ModelType::Mamba2)
        .input_dim(4)
        .output_dim(4)
        .hidden_dim(64)
        .state_dim(16)
        .num_layers(2)
        .build()?;

    let input = array![0.3f32, 0.5, 0.7, 0.9];
    let context_len = 20usize;
    let probe_steps = 50usize;

    // --- Strategy A: reset before every prediction sequence ----------------
    // Suitable when each input sequence is statistically independent (e.g.,
    // processing unrelated sensor bursts).
    let t_reset = Instant::now();
    for _ in 0..probe_steps {
        predictor.reset();
        let _ = predictor.step(&input)?;
    }
    let reset_us = t_reset.elapsed().as_nanos() as f64 / (probe_steps as f64 * 1_000.0);

    // --- Strategy B: warm the context, then keep stepping ------------------
    // Suitable for streaming inference on a continuous signal.  The model
    // accumulates context, producing better-calibrated outputs as the hidden
    // state converges.
    predictor.reset();
    for _ in 0..context_len {
        let _ = predictor.step(&input)?; // context warmup
    }

    let t_stateful = Instant::now();
    for _ in 0..probe_steps {
        let _ = predictor.step(&input)?;
    }
    let stateful_us = t_stateful.elapsed().as_nanos() as f64 / (probe_steps as f64 * 1_000.0);

    println!("  Context window:   {} steps", predictor.context_window());
    println!("  Warmup length:    {} steps", context_len);
    println!();
    println!(
        "  Strategy A (reset-per-query):   {:.2} µs / step",
        reset_us
    );
    println!(
        "  Strategy B (stateful streaming): {:.2} µs / step",
        stateful_us
    );
    println!();

    // --- Strategy C: fork for independent branches -------------------------
    // `fork()` clones the current hidden state, creating a sibling predictor
    // that continues from the same context — useful for tree-search, beam
    // search, or A/B hypothesis testing.
    predictor.reset();
    for _ in 0..context_len {
        let _ = predictor.step(&input)?;
    }

    let mut branch_a = predictor.fork()?;
    let mut branch_b = predictor.fork()?;

    let out_a = branch_a.step(&array![1.0f32, 0.0, 0.0, 0.0])?;
    let out_b = branch_b.step(&array![0.0f32, 0.0, 0.0, 1.0])?;

    println!("  Fork demo (both branches start from the same context):");
    println!("    Branch A (high first dim):  {:?}", out_a);
    println!("    Branch B (high last dim):   {:?}", out_b);
    println!();

    println!("  [recommendations]");
    println!("    - Use reset()  for stateless / request-level inference.");
    println!("    - Use stateful stepping for continuous signal streams.");
    println!("    - Use fork()   for hypothesis search without re-encoding context.");

    Ok(())
}
