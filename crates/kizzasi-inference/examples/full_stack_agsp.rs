//! Full-stack AGSP example with all kizzasi components
//!
//! This example demonstrates a complete Adaptive Generative Signal Predictor using:
//! - kizzasi-tokenizer: Signal tokenization
//! - kizzasi-model: RWKV model architecture
//! - kizzasi-logic: Constraint enforcement with guardrails
//! - kizzasi-inference: Complete inference pipeline
//!
//! Scenario: Time-series prediction with safety constraints

use kizzasi_inference::{
    EngineConfig, InferenceResult, PipelineBuilder, SamplingConfig, SamplingStrategy,
};
use kizzasi_logic::{ConstraintBuilder, Guardrail, GuardrailSet};
use kizzasi_model::rwkv::{Rwkv, RwkvConfig};
use kizzasi_tokenizer::ContinuousTokenizer;
use scirs2_core::ndarray::Array1;

fn main() -> InferenceResult<()> {
    println!("=== Full-Stack AGSP Example ===\n");
    println!("Scenario: Predicting temperature time-series with safety bounds\n");

    // 1. Create tokenizer (Continuous for signal tokenization)
    println!("Step 1: Setting up signal tokenizer...");
    let tokenizer = ContinuousTokenizer::new(
        16, // input dim
        64, // embedding dim
    );
    println!("  Tokenizer configured with 64-dim embeddings\n");

    // 2. Create RWKV model
    println!("Step 2: Initializing RWKV model...");
    let model_config = RwkvConfig::new()
        .input_dim(64) // matches tokenizer embedding dim
        .hidden_dim(256)
        .intermediate_dim(512)
        .num_layers(4);

    let model = Rwkv::new(model_config)?;
    println!("  Model: 4-layer RWKV with 256 hidden dim\n");

    // 3. Configure sampling strategy
    println!("Step 3: Configuring sampling...");
    let sampling = SamplingConfig::new()
        .strategy(SamplingStrategy::TopP)
        .top_p(0.9)
        .temperature(0.7);
    println!("  Using nucleus sampling (p=0.9, temp=0.7)\n");

    // 4. Build inference pipeline
    println!("Step 4: Building inference pipeline...");
    let engine_config = EngineConfig::new(64, 64)
        .sampling(sampling)
        .use_embeddings(true);

    let mut pipeline = PipelineBuilder::new()
        .engine_config(engine_config)
        .model(Box::new(model))
        .tokenizer(Box::new(tokenizer))
        // Constraints are enabled below by `set_guardrails`, once the guardrail
        // set exists — enabling them here without one is a configuration error.
        .build()?;
    println!("  Pipeline ready with tokenizer and model\n");

    // 5. Create safety guardrails
    println!("Step 5: Setting up safety constraints...");
    let mut guardrails = GuardrailSet::new();

    // Add global bounds: all values should be in [-2.0, 2.0]
    let upper_bound = ConstraintBuilder::new()
        .name("temp_upper")
        .less_eq(2.0)
        .weight(1.0)
        .build()
        .unwrap();

    let lower_bound = ConstraintBuilder::new()
        .name("temp_lower")
        .greater_eq(-2.0)
        .weight(1.0)
        .build()
        .unwrap();

    guardrails.add_global(Guardrail::new(upper_bound, false));
    guardrails.add_global(Guardrail::new(lower_bound, false));

    pipeline.set_guardrails(guardrails);
    println!("  Added temperature bounds: [-2.0, 2.0]\n");

    // 6. Simulate temperature time-series
    println!("=== Running Prediction ===\n");

    // Initial temperature reading (normalized)
    let mut current_temp = Array1::from_vec(vec![
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
    ]);

    println!("Initial temperature vector:");
    println!("  Mean: {:.3}", current_temp.mean().unwrap());
    println!(
        "  Range: [{:.3}, {:.3}]",
        current_temp.iter().cloned().fold(f32::INFINITY, f32::min),
        current_temp
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!();

    // Multi-step prediction with constraints
    println!("Predicting next 10 time steps...\n");

    for step in 1..=10 {
        // Forward pass through pipeline
        // Pipeline flow: Raw → Tokenize → Model → Constrain → Decode
        let prediction = pipeline.forward(&current_temp)?;

        let mean_val = prediction.mean().unwrap();
        let min_val = prediction.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = prediction.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("Step {}: ", step);
        println!("  Mean: {:.3}", mean_val);
        println!("  Range: [{:.3}, {:.3}]", min_val, max_val);
        println!(
            "  Constraints: {}",
            if pipeline.has_constraints() {
                "✓ Enforced"
            } else {
                "✗ Disabled"
            }
        );

        // Update current state
        current_temp = prediction;
    }

    // 7. Demonstrate rollout
    println!("\n=== Rollout Prediction (20 steps) ===\n");
    let initial = Array1::from_elem(16, 0.5);
    let rollout_outputs = pipeline.rollout(&initial, 20)?;

    println!("Generated {} predictions", rollout_outputs.len());
    for (i, output) in rollout_outputs.iter().enumerate().step_by(5) {
        println!("  Step {}: mean={:.3}", i + 1, output.mean().unwrap_or(0.0));
    }

    // 8. Performance metrics
    println!("\n=== Pipeline Statistics ===");
    println!("  Total steps executed: {}", pipeline.engine().step_count());
    println!("  Has tokenizer: {}", pipeline.has_tokenizer());
    println!("  Has constraints: {}", pipeline.has_constraints());
    println!("  Preprocessing hooks: {}", pipeline.num_preprocess_hooks());
    println!(
        "  Postprocessing hooks: {}",
        pipeline.num_postprocess_hooks()
    );

    println!("\n=== Example Complete ===");
    Ok(())
}
