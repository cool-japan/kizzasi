//! Model comparison example
//!
//! Compares different model architectures on the same task,
//! measuring performance and characteristics.

use kizzasi_core::SignalPredictor;
use kizzasi_model::{
    mamba::{Mamba, MambaConfig},
    mamba2::{Mamba2, Mamba2Config},
    rwkv::{Rwkv, RwkvConfig},
    s4::{S4Config, S4D},
    transformer::{Transformer, TransformerConfig},
    AutoregressiveModel,
};
use scirs2_core::ndarray::Array1;
use std::time::Instant;

struct BenchmarkResult {
    model_name: String,
    total_time_us: u128,
    avg_time_per_step_us: f64,
    predictions: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Model Architecture Comparison ===\n");

    let sequence_length = 100;
    let num_warmup = 10;

    // Generate test sequence
    let signal: Vec<f32> = (0..sequence_length)
        .map(|t| (t as f32 * 0.1).sin() + (t as f32 * 0.05).cos() * 0.5)
        .collect();

    println!("Test sequence length: {}", sequence_length);
    println!("Warmup steps: {}", num_warmup);
    println!();

    // Benchmark each model
    let mut results = Vec::new();

    println!("Benchmarking Mamba...");
    let mamba = Mamba::new(
        MambaConfig::new()
            .input_dim(1)
            .hidden_dim(128)
            .state_dim(16)
            .num_layers(4),
    )?;
    results.push(benchmark_model("Mamba", mamba, &signal, num_warmup)?);

    println!("Benchmarking Mamba2...");
    let mamba2 = Mamba2::new(
        Mamba2Config::new()
            .input_dim(1)
            .hidden_dim(128)
            .num_heads(8)
            .num_layers(4),
    )?;
    results.push(benchmark_model("Mamba2", mamba2, &signal, num_warmup)?);

    println!("Benchmarking RWKV...");
    let rwkv = Rwkv::new(
        RwkvConfig::new()
            .input_dim(1)
            .hidden_dim(128)
            .num_heads(8)
            .num_layers(4),
    )?;
    results.push(benchmark_model("RWKV", rwkv, &signal, num_warmup)?);

    println!("Benchmarking S4D...");
    let s4d = S4D::new(
        S4Config::new()
            .input_dim(1)
            .hidden_dim(128)
            .state_dim(32)
            .num_layers(4),
    )?;
    results.push(benchmark_model("S4D", s4d, &signal, num_warmup)?);

    println!("Benchmarking Transformer...");
    let transformer = Transformer::new(
        TransformerConfig::new()
            .input_dim(1)
            .hidden_dim(128)
            .num_heads(8)
            .num_layers(4)
            .max_seq_len(256),
    )?;
    results.push(benchmark_model(
        "Transformer",
        transformer,
        &signal,
        num_warmup,
    )?);

    // Display results
    println!("\n=== Benchmark Results ===\n");
    println!(
        "{:<15} {:>15} {:>20} {:>15}",
        "Model", "Total Time (ms)", "Avg Time/Step (μs)", "Throughput (steps/s)"
    );
    println!("{:-<70}", "");

    for result in &results {
        let total_ms = result.total_time_us as f64 / 1000.0;
        let throughput = 1_000_000.0 / result.avg_time_per_step_us;
        println!(
            "{:<15} {:>15.2} {:>20.2} {:>15.0}",
            result.model_name, total_ms, result.avg_time_per_step_us, throughput
        );
    }

    // Find fastest model
    let fastest = results
        .iter()
        .min_by(|a, b| {
            a.avg_time_per_step_us
                .partial_cmp(&b.avg_time_per_step_us)
                .unwrap()
        })
        .unwrap();

    println!(
        "\n✓ Fastest: {} ({:.2} μs per step)",
        fastest.model_name, fastest.avg_time_per_step_us
    );

    // Compute relative speedups
    println!("\n=== Relative Performance ===\n");
    for result in &results {
        let speedup = fastest.avg_time_per_step_us / result.avg_time_per_step_us;
        println!("{:<15} {:.2}x", result.model_name, speedup);
    }

    // Check prediction stability
    println!("\n=== Numerical Stability ===\n");
    for result in &results {
        let finite_count = result
            .predictions
            .iter()
            .filter(|&&x| x.is_finite())
            .count();
        let finite_ratio = finite_count as f64 / result.predictions.len() as f64;
        let status = if finite_ratio == 1.0 { "✓" } else { "✗" };
        println!(
            "{} {:<15} {}/{} finite ({:.1}%)",
            status,
            result.model_name,
            finite_count,
            result.predictions.len(),
            finite_ratio * 100.0
        );
    }

    Ok(())
}

fn benchmark_model<M: SignalPredictor + AutoregressiveModel>(
    name: &str,
    mut model: M,
    signal: &[f32],
    num_warmup: usize,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    // Warmup
    for &value in signal.iter().take(num_warmup) {
        let input = Array1::from_vec(vec![value]);
        let _ = model.step(&input)?;
    }

    // Reset and benchmark
    model.reset();

    let start = Instant::now();
    let mut predictions = Vec::new();

    for &value in signal.iter() {
        let input = Array1::from_vec(vec![value]);
        let output = model.step(&input)?;
        predictions.push(output[0]);
    }

    let elapsed = start.elapsed();
    let total_time_us = elapsed.as_micros();
    let avg_time_per_step_us = total_time_us as f64 / signal.len() as f64;

    Ok(BenchmarkResult {
        model_name: name.to_string(),
        total_time_us,
        avg_time_per_step_us,
        predictions,
    })
}
