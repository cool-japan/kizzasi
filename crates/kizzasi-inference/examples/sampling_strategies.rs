//! Sampling strategies example
//!
//! This example demonstrates different sampling strategies:
//! - Greedy sampling
//! - Temperature scaling
//! - Top-k sampling
//! - Top-p (nucleus) sampling
//!
//! Run with: cargo run --example sampling_strategies

use kizzasi_inference::{
    EngineConfig, InferenceEngine, ModelBuilder, ModelRegistry, SamplingConfig, SamplingStrategy,
};
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sampling Strategies Example ===\n");

    // Create model registry
    let mut registry = ModelRegistry::new();
    let model_config = ModelBuilder::rwkv().dims(1, 64, 1).layers(2).build(); // RWKV has no output-projection layer: output_dim == input_dim
    registry.register("rwkv_model", model_config);

    let input = Array1::from_vec(vec![0.5]);

    // 1. Greedy Sampling
    println!("=== Greedy Sampling (deterministic) ===");
    let sampling = SamplingConfig::new().strategy(SamplingStrategy::Greedy);
    let config = EngineConfig::new(1, 1)
        .sampling(sampling)
        .use_embeddings(true);
    let mut engine =
        InferenceEngine::with_model(config.clone(), registry.create_model("rwkv_model")?);

    let output = engine.step(&input)?;
    println!("Output: {:?}\n", output);

    // 2. Temperature Sampling
    println!("=== Temperature Sampling ===");
    for temp in &[0.5, 1.0, 1.5] {
        let sampling = SamplingConfig::new()
            .strategy(SamplingStrategy::Temperature)
            .temperature(*temp)
            .seed(42);

        let config = EngineConfig::new(1, 1)
            .sampling(sampling)
            .use_embeddings(true);
        let mut engine = InferenceEngine::with_model(config, registry.create_model("rwkv_model")?);

        let output = engine.step(&input)?;
        println!("Temperature {}: {:?}", temp, output);
    }
    println!();

    // 3. Top-k Sampling
    println!("=== Top-k Sampling ===");
    for k in &[1, 3, 5] {
        let sampling = SamplingConfig::new().top_k(*k).seed(42);

        let config = EngineConfig::new(1, 1)
            .sampling(sampling)
            .use_embeddings(true);
        let mut engine = InferenceEngine::with_model(config, registry.create_model("rwkv_model")?);

        let output = engine.step(&input)?;
        println!("Top-{}: {:?}", k, output);
    }
    println!();

    // 4. Top-p Sampling
    println!("=== Top-p (Nucleus) Sampling ===");
    for p in &[0.7, 0.9, 0.95] {
        let sampling = SamplingConfig::new().top_p(*p).seed(42);

        let config = EngineConfig::new(1, 1)
            .sampling(sampling)
            .use_embeddings(true);
        let mut engine = InferenceEngine::with_model(config, registry.create_model("rwkv_model")?);

        let output = engine.step(&input)?;
        println!("Top-p {}: {:?}", p, output);
    }

    println!("\n=== Done ===");
    Ok(())
}
