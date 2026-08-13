//! Continuous batching example
//!
//! This example demonstrates high-throughput inference with continuous batching:
//! - Dynamic batch formation
//! - Priority-based scheduling
//! - Efficient resource utilization
//!
//! Run with: cargo run --example continuous_batching

use kizzasi_inference::{BatchConfig, BatchScheduler, EngineConfig, Priority};
use kizzasi_model::s4::{S4Config, S4D};
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Continuous Batching Example ===\n");

    // 1. Configure batching
    let batch_config = BatchConfig::new()
        .max_batch_size(8)
        .max_wait_ms(5)
        .min_batch_size(2)
        .with_priority();

    let engine_config = EngineConfig::new(3, 3);

    // 2. Create scheduler with a model — a scheduler without one cannot serve
    //    requests and reports `NotInitialized` on the first step.
    let model = S4D::new(
        S4Config::new()
            .input_dim(3)
            .hidden_dim(64)
            .state_dim(16)
            .num_layers(2)
            .diagonal(true),
    )?;
    let mut scheduler = BatchScheduler::with_model(batch_config, engine_config, Box::new(model))?;

    println!("Submitting requests...\n");

    // 3. Submit multiple requests
    let input1 = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    let input2 = Array1::from_vec(vec![0.4, 0.5, 0.6]);
    let input3 = Array1::from_vec(vec![0.7, 0.8, 0.9]);

    // Regular priority
    let id1 = scheduler.submit(input1.clone(), 3)?;
    println!("Submitted request {} (Normal priority)", id1);

    // High priority - will be processed first
    let id2 = scheduler.submit_with_priority(input2.clone(), 5, Priority::High)?;
    println!("Submitted request {} (High priority)", id2);

    // Low priority
    let id3 = scheduler.submit_with_priority(input3.clone(), 2, Priority::Low)?;
    println!("Submitted request {} (Low priority)", id3);

    // More regular requests
    for i in 0..5 {
        let input = Array1::from_vec(vec![i as f32, i as f32 + 0.1, i as f32 + 0.2]);
        let id = scheduler.submit(input, 4)?;
        println!("Submitted request {} (Normal priority)", id);
    }

    println!("\nScheduler stats: {:?}\n", scheduler.stats());

    // 4. Process all requests
    println!("Processing all requests...\n");
    let responses = scheduler.process_all()?;

    // 5. Display results
    println!("=== Results ===");
    for response in responses {
        println!(
            "Request {}: {} steps completed ({} outputs) in {} µs",
            response.request_id,
            response.steps_completed,
            response.outputs.len(),
            response.inference_time_us
        );
    }

    println!("\nFinal stats: {:?}", scheduler.stats());
    println!("\n=== Done ===");
    Ok(())
}
