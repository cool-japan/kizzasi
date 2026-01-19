//! Async streaming prediction example
//!
//! This example demonstrates using Kizzasi's async streaming API for
//! real-time signal processing with tokio.
//!
//! Run with:
//! ```bash
//! cargo run --example streaming --features async
//! ```

#[cfg(not(feature = "async"))]
fn main() {
    println!("This example requires the 'async' feature.");
    println!("Run with: cargo run --example streaming --features async");
}

#[cfg(feature = "async")]
#[tokio::main]
async fn main() -> kizzasi::KizzasiResult<()> {
    use futures::StreamExt;
    use kizzasi::prelude::*;
    use std::time::Duration;

    println!("=== Kizzasi Async Streaming Example ===\n");

    // Example 1: PredictionStream with channel-based input
    println!("1. Channel-based Prediction Stream");
    println!("   ================================\n");

    let predictor = KizzasiBuilder::new()
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(32)
        .state_dim(4)
        .num_layers(1)
        .build()?;

    let (mut stream, tx) = PredictionStream::new(predictor, 16);

    println!("  ✓ Created prediction stream with buffer size 16");

    // Spawn a task to send inputs
    let send_handle = tokio::spawn(async move {
        println!("  → Sending 10 inputs...");
        for i in 0..10 {
            let input = array![i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3];
            if tx.send(input).await.is_err() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        println!("  ✓ All inputs sent");
    });

    // Receive and process outputs
    println!("  ← Receiving predictions...");
    let mut count = 0;
    while let Some(result) = stream.next().await {
        let output = result?;
        if !(3..7).contains(&count) {
            println!("    Prediction {}: {:?}", count + 1, output);
        } else if count == 3 {
            println!("    ...");
        }
        count += 1;
    }
    println!("  ✓ Received {} predictions\n", count);

    send_handle.await.unwrap();

    // Example 2: AsyncPredictor with request-response pattern
    println!("2. Async Predictor (Request-Response)");
    println!("   ===================================\n");

    let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
    let mut async_pred = AsyncPredictor::spawn(predictor, 8);

    println!("  ✓ Spawned async predictor in background");

    // Send multiple predictions concurrently
    println!("  → Sending 5 predictions...");
    for i in 0..5 {
        let input = array![i as f32 * 0.1, i as f32 * 0.2];
        async_pred.send(input).await.unwrap();
    }

    // Receive results
    println!("  ← Receiving results...");
    for i in 0..5 {
        if let Some(result) = async_pred.recv().await {
            let output = result?;
            println!("    Result {}: {:?}", i + 1, output);
        }
    }
    println!();

    // Example 3: StreamProcessor with custom source
    println!("3. Stream Processor (Transform Pattern)");
    println!("   =====================================\n");

    let predictor = KizzasiBuilder::new()
        .input_dim(1)
        .output_dim(1)
        .hidden_dim(16)
        .state_dim(4)
        .num_layers(1)
        .build()?;

    // Create a simple signal generator stream
    let signal_stream = futures::stream::iter((0..20).map(|i| {
        let t = i as f32 * 0.1;
        array![(t * 2.0 * std::f32::consts::PI).sin()]
    }));

    let mut processor = StreamProcessor::new(predictor, signal_stream);

    println!("  ✓ Created stream processor");
    println!("  → Processing sine wave signal (20 samples)...\n");

    let mut count = 0;
    while let Some(result) = processor.next().await {
        let output = result?;
        if !(5..15).contains(&count) {
            println!(
                "    Sample {}: Input→{:.4}, Predicted→{:.4}",
                count + 1,
                (count as f32 * 0.1 * 2.0 * std::f32::consts::PI).sin(),
                output[0]
            );
        } else if count == 5 {
            println!("    ...");
        }
        count += 1;
    }
    println!("\n  ✓ Processed {} samples\n", count);

    // Example 4: Multiple concurrent predictions
    println!("4. Concurrent Predictions (Parallelism)");
    println!("   ====================================\n");

    let predictor = KizzasiBuilder::lightweight_preset(1, 1).build()?;
    let mut async_pred = AsyncPredictor::spawn(predictor, 32);

    println!("  ✓ Spawned async predictor");
    println!("  → Sending 20 inputs rapidly...\n");

    // Send many inputs without waiting
    for i in 0..20 {
        let input = array![i as f32 * 0.05];
        async_pred.send(input).await.unwrap();
    }

    // Collect all results
    let mut results = Vec::new();
    for _ in 0..20 {
        if let Some(result) = async_pred.recv().await {
            results.push(result?);
        }
    }

    println!("  ✓ Received {} results", results.len());
    println!("    First: {:?}", results.first().unwrap());
    println!("    Last:  {:?}", results.last().unwrap());
    println!();

    println!("=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  • PredictionStream: Channel-based async prediction");
    println!("  • AsyncPredictor: Background task with send/recv");
    println!("  • StreamProcessor: Transform pattern for streams");
    println!("  • All patterns are composable and non-blocking");
    println!("  • Ideal for real-time applications with async I/O");

    Ok(())
}
