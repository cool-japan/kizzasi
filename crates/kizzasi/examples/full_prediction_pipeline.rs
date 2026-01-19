//! Basic prediction example
//!
//! This example demonstrates the most basic usage of Kizzasi for signal prediction.
//!
//! Run with:
//! ```bash
//! cargo run --example basic_prediction
//! ```

use kizzasi::prelude::*;

fn main() -> Result<()> {
    println!("=== Kizzasi Basic Prediction Example ===\n");

    // Create a simple predictor using the builder pattern
    let mut predictor = KizzasiBuilder::new()
        .model_type(ModelType::Mamba2)
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(64)
        .state_dim(8)
        .num_layers(2)
        .build()?;

    println!("✓ Predictor created with:");
    println!("  - Model: Mamba2");
    println!("  - Input dimension: 3");
    println!("  - Output dimension: 3");
    println!("  - Hidden dimension: 64");
    println!("  - Context window: {}\n", predictor.context_window());

    // Single step prediction
    let input = array![0.1, 0.2, 0.3];
    println!("Single step prediction:");
    println!("  Input:  {:?}", input);

    let output = predictor.step(&input)?;
    println!("  Output: {:?}\n", output);

    // Multi-step prediction (autoregressive)
    let n_steps = 5;
    println!("Multi-step prediction ({} steps):", n_steps);
    println!("  Initial input: {:?}", input);

    predictor.reset(); // Reset state for clean prediction
    let predictions = predictor.predict_n(&input, n_steps)?;

    for (i, row) in predictions.outer_iter().enumerate() {
        println!("  Step {}: {:?}", i + 1, row);
    }
    println!();

    // Prediction until condition
    println!("Predict until condition (max 10 steps):");
    predictor.reset();

    let predictions = predictor.predict_until(&input, 10, |output, step| {
        // Stop when any value exceeds 0.5 or after 5 steps
        output.iter().any(|&x| x.abs() > 0.5) || step >= 4
    })?;

    println!("  Stopped after {} steps", predictions.len());
    for (i, pred) in predictions.iter().enumerate() {
        println!("  Step {}: {:?}", i + 1, pred);
    }
    println!();

    // Batch prediction
    println!("Batch prediction (3 inputs):");
    predictor.reset();

    let inputs = vec![
        array![0.1, 0.2, 0.3],
        array![0.4, 0.5, 0.6],
        array![0.7, 0.8, 0.9],
    ];

    let outputs = predictor.predict_batch(&inputs)?;
    for (i, (input, output)) in inputs.iter().zip(outputs.iter()).enumerate() {
        println!("  Input {}: {:?} -> {:?}", i + 1, input, output);
    }
    println!();

    // Forking for parallel predictions
    println!("Forking predictor for parallel scenarios:");
    let _forked = predictor.fork()?;
    println!("  ✓ Created fork with same configuration");
    println!("  - Both predictors can run independently");
    println!("  - Useful for exploring different prediction paths\n");

    println!("=== Example Complete ===");
    Ok(())
}
