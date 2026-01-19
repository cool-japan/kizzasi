//! Basic signal prediction example
//!
//! This example demonstrates how to use different model architectures
//! for simple signal prediction tasks.

use kizzasi_core::SignalPredictor;
use kizzasi_model::{
    mamba::{Mamba, MambaConfig},
    mamba2::{Mamba2, Mamba2Config},
    rwkv::{Rwkv, RwkvConfig},
    s4::{S4Config, S4D},
    AutoregressiveModel,
};
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Kizzasi Model - Basic Signal Prediction ===\n");

    // Create a simple sinusoidal signal to predict
    let time_steps = 20;
    let mut signal = Vec::new();
    for t in 0..time_steps {
        let value = (t as f32 * 0.5).sin();
        signal.push(value);
    }

    println!("Input signal (first 10 values): {:?}\n", &signal[..10]);

    // Test Mamba
    println!("--- Mamba Model ---");
    test_model_prediction("Mamba", create_mamba()?, &signal)?;

    // Test Mamba2
    println!("\n--- Mamba2 Model ---");
    test_model_prediction("Mamba2", create_mamba2()?, &signal)?;

    // Test RWKV
    println!("\n--- RWKV Model ---");
    test_model_prediction("RWKV", create_rwkv()?, &signal)?;

    // Test S4D
    println!("\n--- S4D Model ---");
    test_model_prediction("S4D", create_s4d()?, &signal)?;

    Ok(())
}

fn create_mamba() -> Result<Mamba, Box<dyn std::error::Error>> {
    let config = MambaConfig::new()
        .input_dim(1)
        .hidden_dim(64)
        .state_dim(16)
        .num_layers(4);
    Ok(Mamba::new(config)?)
}

fn create_mamba2() -> Result<Mamba2, Box<dyn std::error::Error>> {
    let config = Mamba2Config::new()
        .input_dim(1)
        .hidden_dim(64)
        .num_heads(4)
        .num_layers(4);
    Ok(Mamba2::new(config)?)
}

fn create_rwkv() -> Result<Rwkv, Box<dyn std::error::Error>> {
    let config = RwkvConfig::new()
        .input_dim(1)
        .hidden_dim(64)
        .num_heads(4)
        .num_layers(4);
    Ok(Rwkv::new(config)?)
}

fn create_s4d() -> Result<S4D, Box<dyn std::error::Error>> {
    let config = S4Config::new()
        .input_dim(1)
        .hidden_dim(64)
        .state_dim(16)
        .num_layers(4);
    Ok(S4D::new(config)?)
}

fn test_model_prediction<M: SignalPredictor + AutoregressiveModel>(
    name: &str,
    mut model: M,
    signal: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Model: {}", name);
    println!("Architecture: {}", model.model_type());
    println!("Hidden dim: {}", model.hidden_dim());
    println!("State dim: {}", model.state_dim());
    println!("Layers: {}", model.num_layers());
    println!(
        "Context window: {}",
        if model.context_window() == usize::MAX {
            "∞".to_string()
        } else {
            model.context_window().to_string()
        }
    );

    // Process the signal
    let mut predictions = Vec::new();
    for &value in signal.iter().take(10) {
        let input = Array1::from_vec(vec![value]);
        let output = model.step(&input)?;
        predictions.push(output[0]);
    }

    println!("Predictions (first 10): {:?}", predictions);

    // Check numerical stability
    let all_finite = predictions.iter().all(|&x| x.is_finite());
    println!("All predictions finite: {}", all_finite);

    Ok(())
}
