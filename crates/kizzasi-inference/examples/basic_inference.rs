//! Basic inference example
//!
//! This example demonstrates how to:
//! 1. Create an inference engine
//! 2. Load a model
//! 3. Perform single-step prediction
//! 4. Perform multi-step rollout
//!
//! Run with: cargo run --example basic_inference

use kizzasi_inference::{EngineConfig, InferenceEngine, ModelBuilder, ModelRegistry};
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Kizzasi Inference: Basic Example ===\n");

    // 1. Create a model registry
    let mut registry = ModelRegistry::new();

    // 2. Register an S4D model configuration
    let model_config = ModelBuilder::s4d()
        .dims(1, 128, 10) // input_dim, hidden_dim, output_dim
        .layers(4)
        .state_dim(16)
        .build();

    registry.register("signal_predictor", model_config);

    // 3. Create the model
    println!("Loading S4D model...");
    let model = registry.create_model("signal_predictor")?;

    // 4. Create engine configuration
    let engine_config = EngineConfig::new(1, 10);

    // 5. Create inference engine with model
    let mut engine = InferenceEngine::with_model(engine_config, model);

    println!("Model loaded successfully!");
    println!("Model info: {:?}\n", engine.model_info());

    // 6. Single-step prediction
    println!("=== Single-Step Prediction ===");
    let input = Array1::from_vec(vec![0.5]);
    println!("Input: {:?}", input);

    let output = engine.step(&input)?;
    println!("Output: {:?}", output);
    println!("Step count: {}\n", engine.step_count());

    // 7. Multi-step rollout
    println!("=== Multi-Step Rollout (10 steps) ===");
    engine.reset();

    let initial = Array1::from_vec(vec![0.3]);
    let outputs = engine.rollout(&initial, 10)?;

    println!("Generated {} predictions", outputs.len());
    for (i, output) in outputs.iter().enumerate() {
        println!("Step {}: {:?}", i + 1, output);
    }

    println!("\n=== Done ===");
    Ok(())
}
