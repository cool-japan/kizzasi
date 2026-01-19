//! Multi-step prediction example
//!
//! Demonstrates autoregressive generation where the model's
//! outputs are fed back as inputs for future predictions.

use kizzasi_core::SignalPredictor;
use kizzasi_model::{
    mamba::{Mamba, MambaConfig},
    AutoregressiveModel,
};
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Step Autoregressive Prediction ===\n");

    let config = MambaConfig::new()
        .input_dim(1)
        .hidden_dim(128)
        .state_dim(16)
        .num_layers(6);

    let mut model = Mamba::new(config)?;

    println!(
        "Model: {} layers, {} hidden dim, {} state dim",
        model.num_layers(),
        model.hidden_dim(),
        model.state_dim()
    );

    // Generate training sequence (sinusoidal)
    let train_length = 50;
    let train_sequence: Vec<f32> = (0..train_length).map(|t| (t as f32 * 0.2).sin()).collect();

    println!("\nTraining on {} time steps...", train_length);

    // "Train" the model by feeding it the sequence
    // (In reality this just builds up the recurrent state)
    for &value in &train_sequence {
        let input = Array1::from_vec(vec![value]);
        let _ = model.step(&input)?;
    }

    println!("✓ Model primed with training sequence");

    // Now perform autoregressive prediction
    let prediction_length = 30;
    println!(
        "\nGenerating {} autoregressive predictions...",
        prediction_length
    );

    let mut predictions = Vec::new();
    let mut current_input = *train_sequence.last().unwrap();

    for step in 0..prediction_length {
        let input = Array1::from_vec(vec![current_input]);
        let output = model.step(&input)?;
        let predicted_value = output[0];

        predictions.push(predicted_value);

        // Use prediction as next input (autoregressive)
        current_input = predicted_value;

        if step < 10 || step % 5 == 0 {
            println!(
                "Step {}: input={:.4}, predicted={:.4}",
                step, input[0], predicted_value
            );
        }
    }

    // Generate ground truth for comparison
    let ground_truth: Vec<f32> = (train_length..train_length + prediction_length)
        .map(|t| (t as f32 * 0.2).sin())
        .collect();

    // Compute prediction error
    println!("\n=== Prediction Quality ===\n");
    let mse: f32 = predictions
        .iter()
        .zip(ground_truth.iter())
        .map(|(p, g)| (p - g).powi(2))
        .sum::<f32>()
        / predictions.len() as f32;

    let rmse = mse.sqrt();
    println!("RMSE: {:.6}", rmse);

    // Compute correlation
    let pred_mean: f32 = predictions.iter().sum::<f32>() / predictions.len() as f32;
    let truth_mean: f32 = ground_truth.iter().sum::<f32>() / ground_truth.len() as f32;

    let numerator: f32 = predictions
        .iter()
        .zip(ground_truth.iter())
        .map(|(p, g)| (p - pred_mean) * (g - truth_mean))
        .sum();

    let denom_pred: f32 = predictions
        .iter()
        .map(|p| (p - pred_mean).powi(2))
        .sum::<f32>()
        .sqrt();
    let denom_truth: f32 = ground_truth
        .iter()
        .map(|g| (g - truth_mean).powi(2))
        .sum::<f32>()
        .sqrt();

    let correlation = numerator / (denom_pred * denom_truth);
    println!("Correlation: {:.6}", correlation);

    // Display first few predictions vs ground truth
    println!("\n=== Sample Predictions ===\n");
    println!(
        "{:<8} {:>12} {:>12} {:>12}",
        "Step", "Predicted", "Ground Truth", "Error"
    );
    println!("{:-<48}", "");

    for (i, (pred, truth)) in predictions
        .iter()
        .zip(ground_truth.iter())
        .enumerate()
        .take(15)
    {
        let error = (pred - truth).abs();
        println!("{:<8} {:>12.6} {:>12.6} {:>12.6}", i, pred, truth, error);
    }

    // Check numerical stability
    let all_finite = predictions.iter().all(|&x| x.is_finite());
    let diverged = predictions.iter().any(|&x| x.abs() > 10.0);

    println!("\n=== Numerical Stability ===\n");
    println!(
        "All predictions finite: {}",
        if all_finite { "✓" } else { "✗" }
    );
    println!("Predictions bounded: {}", if !diverged { "✓" } else { "✗" });

    if all_finite && !diverged {
        println!("\n✓ Autoregressive generation successful!");
    } else {
        println!("\n⚠ Numerical issues detected");
    }

    // Demonstrate prediction with different horizon lengths
    println!("\n=== Prediction Horizon Analysis ===\n");

    for &horizon in &[5, 10, 20, 30] {
        model.reset();

        // Prime with training data
        for &value in &train_sequence {
            let input = Array1::from_vec(vec![value]);
            let _ = model.step(&input)?;
        }

        // Predict with this horizon
        let mut horizon_predictions = Vec::new();
        let mut current = *train_sequence.last().unwrap();

        for _ in 0..horizon {
            let input = Array1::from_vec(vec![current]);
            let output = model.step(&input)?;
            horizon_predictions.push(output[0]);
            current = output[0];
        }

        let horizon_truth: Vec<f32> = (train_length..train_length + horizon)
            .map(|t| (t as f32 * 0.2).sin())
            .collect();

        let horizon_rmse = (horizon_predictions
            .iter()
            .zip(horizon_truth.iter())
            .map(|(p, g)| (p - g).powi(2))
            .sum::<f32>()
            / horizon as f32)
            .sqrt();

        println!("Horizon {:>2}: RMSE = {:.6}", horizon, horizon_rmse);
    }

    Ok(())
}
