//! Audio signal processing example
//!
//! This example demonstrates using Kizzasi for audio signal prediction,
//! useful for tasks like next-sample prediction, audio compression,
//! and generative audio.
//!
//! Run with:
//! ```bash
//! cargo run --example audio_processing
//! ```

use kizzasi::prelude::*;
use scirs2_core::ndarray::s;
use std::f32::consts::PI;

fn main() -> Result<()> {
    println!("=== Kizzasi Audio Processing Example ===\n");

    // Create predictor with audio preset
    let mut predictor = KizzasiBuilder::audio_preset().build()?;

    println!("✓ Audio predictor created");
    println!("  - Sample rate: 44.1kHz (conceptual)");
    println!("  - Channels: Mono (1 channel)");
    println!("  - Context window: {} samples", predictor.context_window());
    println!("  - Optimized for audio signals\n");

    // Generate a synthetic audio signal (sine wave)
    let sample_rate = 44100.0;
    let frequency = 440.0; // A4 note
    let duration = 0.1; // 100ms
    let num_samples = (sample_rate * duration) as usize;

    println!("Generating test signal:");
    println!("  - Frequency: {} Hz (A4 note)", frequency);
    println!("  - Duration: {} ms", duration * 1000.0);
    println!("  - Samples: {}\n", num_samples);

    let signal: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * PI * frequency * t).sin() * 0.5 // Amplitude 0.5
        })
        .collect();

    // Feed signal to predictor and collect predictions
    println!("Processing signal through predictor...");
    let mut predictions = Vec::new();

    for (i, &sample) in signal.iter().enumerate() {
        let input = array![sample];
        let output = predictor.step(&input)?;

        predictions.push(output[0]);

        if i % 1000 == 0 {
            println!("  Processed {} / {} samples", i, num_samples);
        }
    }

    println!("✓ Processing complete\n");

    // Analyze prediction quality
    println!("Analyzing predictions:");

    // Calculate mean squared error for last 1000 samples
    // (assuming the model has adapted to the signal)
    let start_idx = num_samples.saturating_sub(1000);
    let mut mse = 0.0;
    let mut count = 0;

    for i in start_idx..num_samples.min(predictions.len()) {
        if i > 0 && i < signal.len() {
            let error = signal[i] - predictions[i - 1];
            mse += error * error;
            count += 1;
        }
    }

    if count > 0 {
        mse /= count as f32;
        println!("  Mean Squared Error (last 1000): {:.6}", mse);
        println!("  RMSE: {:.6}", mse.sqrt());
    }

    // Show sample predictions
    println!("\nSample predictions (first 10):");
    for i in 0..10.min(predictions.len()) {
        if i < signal.len() {
            println!(
                "  Sample {}: Input={:.4}, Predicted={:.4}, Actual={:.4}",
                i,
                if i > 0 { signal[i - 1] } else { 0.0 },
                predictions[i],
                signal[i]
            );
        }
    }
    println!();

    // Multi-step prediction (generate continuation)
    println!("Multi-step prediction (generate 100 samples):");
    predictor.reset();

    // Prime with first few samples
    let prime_length = 100;
    for &sample in signal.iter().take(prime_length) {
        let _ = predictor.step(&array![sample])?;
    }

    // Generate continuation
    let last_sample = array![signal[prime_length - 1]];
    let continuation = predictor.predict_n(&last_sample, 100)?;

    println!("  ✓ Generated {} samples", continuation.len());
    println!("  First 5 generated: {:?}", continuation.slice(s![0..5, 0]));
    println!("  (This is speculative generation based on learned pattern)\n");

    // Demonstrate batch processing
    println!("Batch processing (10 samples):");
    predictor.reset();

    let batch_inputs: Vec<Array1<f32>> = signal.iter().take(10).map(|&s| array![s]).collect();

    let batch_outputs = predictor.predict_batch(&batch_inputs)?;
    println!("  ✓ Processed {} samples in batch", batch_outputs.len());
    println!();

    println!("=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  • Audio preset optimized for 44.1kHz signals");
    println!("  • Can learn temporal patterns in audio");
    println!("  • Useful for: compression, generation, denoising");
    println!("  • Large context window captures long-range dependencies");

    Ok(())
}
