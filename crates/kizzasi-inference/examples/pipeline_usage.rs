//! Pipeline usage example
//!
//! This example demonstrates the complete inference pipeline:
//! - Pipeline construction with builder pattern
//! - Model integration
//! - Rollout predictions
//!
//! Run with: cargo run --example pipeline_usage

use kizzasi_inference::{
    EngineConfig, ModelBuilder, ModelRegistry, PipelineBuilder, SamplingConfig, SamplingStrategy,
};
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Pipeline Usage Example ===\n");

    // 1. Create and configure model
    let mut registry = ModelRegistry::new();
    let model_config = ModelBuilder::transformer()
        .dims(1, 128, 10)
        .layers(3)
        .build();

    registry.register("transformer", model_config);
    let model = registry.create_model("transformer")?;

    // 2. Configure sampling
    let sampling = SamplingConfig::new()
        .strategy(SamplingStrategy::TopK)
        .top_k(5)
        .temperature(0.8);

    // 3. Build pipeline
    let engine_config = EngineConfig::new(1, 10)
        .sampling(sampling)
        .use_embeddings(true);

    let mut pipeline = PipelineBuilder::new()
        .engine_config(engine_config)
        .model(model)
        .with_constraints() // Enable constraint enforcement (placeholder)
        .build()?;

    println!("Pipeline created successfully!");
    println!("Has constraints: {}", pipeline.has_constraints());
    println!("Has tokenizer: {}\n", pipeline.has_tokenizer());

    // 4. Single prediction
    println!("=== Single Prediction ===");
    let input = Array1::from_vec(vec![0.5]);
    let output = pipeline.forward(&input)?;
    println!("Input: {:?}", input);
    println!("Output: {:?}\n", output);

    // 5. Multi-step rollout
    println!("=== Multi-Step Rollout (7 steps) ===");
    pipeline.reset();

    let initial = Array1::from_vec(vec![0.3]);
    let predictions = pipeline.rollout(&initial, 7)?;

    for (i, pred) in predictions.iter().enumerate() {
        println!("Step {}: {:?}", i + 1, pred);
    }

    // 6. Engine info
    println!("\n=== Engine Info ===");
    println!("Steps processed: {}", pipeline.engine().step_count());
    if let Some(info) = pipeline.engine().model_info() {
        println!("Model type: {:?}", info.model_type);
        println!("Hidden dim: {}", info.hidden_dim);
        println!("Layers: {}", info.num_layers);
    }

    println!("\n=== Done ===");
    Ok(())
}
