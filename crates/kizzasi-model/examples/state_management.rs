//! State management example
//!
//! This example demonstrates how to save, restore, and manipulate
//! model states for checkpointing and continuation.

use kizzasi_core::SignalPredictor;
use kizzasi_model::{
    mamba::{Mamba, MambaConfig},
    AutoregressiveModel,
};
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== State Management Example ===\n");

    let config = MambaConfig::new()
        .input_dim(1)
        .hidden_dim(64)
        .state_dim(16)
        .num_layers(3);

    let mut model = Mamba::new(config)?;

    // Generate a sequence
    let sequence: Vec<f32> = (0..20).map(|t| (t as f32 * 0.3).sin()).collect();

    println!("Processing sequence of {} steps...", sequence.len());

    // Process first part of sequence
    let split_point = 10;
    let mut outputs_part1 = Vec::new();

    for &value in sequence.iter().take(split_point) {
        let input = Array1::from_vec(vec![value]);
        let output = model.step(&input)?;
        outputs_part1.push(output[0]);
    }

    println!("Processed first {} steps", split_point);
    println!(
        "Last output from part 1: {:.6}",
        outputs_part1.last().unwrap()
    );

    // Save the state at this point
    let saved_states = model.get_states();
    println!("\n✓ Saved model state ({} layers)", saved_states.len());

    // Continue processing
    let mut outputs_part2_original = Vec::new();
    for &value in sequence.iter().skip(split_point) {
        let input = Array1::from_vec(vec![value]);
        let output = model.step(&input)?;
        outputs_part2_original.push(output[0]);
    }

    println!("Completed full sequence");
    println!(
        "Final output: {:.6}",
        outputs_part2_original.last().unwrap()
    );

    // Now reset the model and restore state to split point
    println!("\n--- Restoring from checkpoint ---");
    model.reset();
    println!("✓ Model reset");

    // Process first part again
    for &value in sequence.iter().take(split_point) {
        let input = Array1::from_vec(vec![value]);
        let _ = model.step(&input)?;
    }

    // Restore the saved state
    model.set_states(saved_states)?;
    println!("✓ State restored to checkpoint");

    // Continue from checkpoint
    let mut outputs_part2_restored = Vec::new();
    for &value in sequence.iter().skip(split_point) {
        let input = Array1::from_vec(vec![value]);
        let output = model.step(&input)?;
        outputs_part2_restored.push(output[0]);
    }

    println!("Completed sequence from restored checkpoint");
    println!(
        "Final output: {:.6}",
        outputs_part2_restored.last().unwrap()
    );

    // Verify that restored path produces same results
    println!("\n--- Verification ---");
    let max_diff: f32 = outputs_part2_original
        .iter()
        .zip(outputs_part2_restored.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |acc, x| acc.max(x));

    println!(
        "Maximum difference between original and restored: {:.10}",
        max_diff
    );

    if max_diff < 1e-6 {
        println!("✓ State restoration verified! (diff < 1e-6)");
    } else {
        println!(
            "⚠ State restoration has differences (diff = {:.10})",
            max_diff
        );
    }

    // Demonstrate reset functionality
    println!("\n--- Testing Reset ---");
    model.reset();

    let input = Array1::from_vec(vec![sequence[split_point]]);
    let output_after_reset = model.step(&input)?;

    println!(
        "Output at split point after reset: {:.6}",
        output_after_reset[0]
    );
    println!(
        "Original output at split point: {:.6}",
        outputs_part1.last().unwrap()
    );

    let reset_diff = (output_after_reset[0] - outputs_part1.last().unwrap()).abs();
    println!("Difference: {:.10}", reset_diff);

    if reset_diff > 1e-6 {
        println!("✓ Reset verified! Output changed as expected");
    } else {
        println!("⚠ Reset may not be working correctly");
    }

    Ok(())
}
