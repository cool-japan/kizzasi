//! Basic Quantizer Examples
//!
//! This example demonstrates the usage of basic quantizers:
//! - LinearQuantizer for uniform quantization
//! - MuLawCodec for logarithmic audio quantization
//!
//! Run with:
//! ```bash
//! cargo run --example basic_quantizers
//! ```

use kizzasi_tokenizer::{LinearQuantizer, MuLawCodec, Quantizer, SignalTokenizer};
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Quantizer Examples ===\n");

    // Example 1: Linear Quantizer
    println!("1. Linear Quantizer (8-bit)");
    println!("----------------------------");

    // Create a linear quantizer with 8 bits (256 levels) in range [-1.0, 1.0]
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 8)?;
    println!("Created quantizer with {} levels", quantizer.num_levels());

    // Create a test signal
    let signal = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0]);
    println!("Original signal: {:?}", signal);

    // Encode the signal
    let encoded = quantizer.encode(&signal)?;
    println!("Encoded tokens: {:?}", encoded);

    // Decode back to continuous
    let decoded = quantizer.decode(&encoded)?;
    println!("Decoded signal: {:?}", decoded);

    // Compute reconstruction error
    let error: f32 = signal
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / signal.len() as f32;
    println!("Mean Squared Error: {:.6}\n", error);

    // Example 2: μ-law Codec
    println!("2. μ-law Codec (Logarithmic)");
    println!("----------------------------");

    // Create a μ-law codec with 8 bits
    let mulaw = MuLawCodec::new(8);
    println!("Created μ-law codec with μ={}", mulaw.mu());

    // Create a signal with varying amplitudes
    let signal = Array1::from_vec(vec![
        0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, -0.01, -0.05, -0.1, -0.2, -0.5, -0.8, -1.0,
    ]);
    println!("Original signal: {:?}", signal);

    // Encode with μ-law
    let encoded = mulaw.encode(&signal)?;
    println!("Encoded tokens: {:?}", encoded);

    // Decode
    let decoded = mulaw.decode(&encoded)?;
    println!("Decoded signal: {:?}", decoded);

    // Compute error for small vs large signals
    let small_signal_idx = 1; // 0.01
    let large_signal_idx = 7; // 1.0
    let small_error = (signal[small_signal_idx] - decoded[small_signal_idx]).abs();
    let large_error = (signal[large_signal_idx] - decoded[large_signal_idx]).abs();

    println!("Error for small signal (0.01): {:.6}", small_error);
    println!("Error for large signal (1.0): {:.6}", large_error);
    println!(
        "μ-law provides better resolution for small signals (relative error: {:.2}% vs {:.2}%)\n",
        (small_error / signal[small_signal_idx].abs()) * 100.0,
        (large_error / signal[large_signal_idx].abs()) * 100.0
    );

    // Example 3: Comparison of Linear vs μ-law
    println!("3. Linear vs μ-law Comparison");
    println!("------------------------------");

    // Generate a sine wave
    let n_samples = 100;
    let sine_wave: Array1<f32> = Array1::from_vec(
        (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / n_samples as f32).sin() * 0.5)
            .collect(),
    );

    // Quantize with both methods
    let linear_quantizer = LinearQuantizer::new(-1.0, 1.0, 8)?;
    let mulaw_codec = MuLawCodec::new(8);

    let linear_decoded = linear_quantizer.decode(&linear_quantizer.encode(&sine_wave)?)?;
    let mulaw_decoded = mulaw_codec.decode(&mulaw_codec.encode(&sine_wave)?)?;

    // Compute SNR for both
    let compute_snr = |original: &Array1<f32>, reconstructed: &Array1<f32>| -> f32 {
        let signal_power: f32 = original.iter().map(|x| x.powi(2)).sum();
        let noise_power: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        10.0 * (signal_power / noise_power).log10()
    };

    let linear_snr = compute_snr(&sine_wave, &linear_decoded);
    let mulaw_snr = compute_snr(&sine_wave, &mulaw_decoded);

    println!("Sine wave SNR with Linear quantizer: {:.2} dB", linear_snr);
    println!("Sine wave SNR with μ-law codec: {:.2} dB", mulaw_snr);

    // Example 4: Different bit depths
    println!("\n4. Effect of Bit Depth");
    println!("----------------------");

    for bits in [4, 6, 8, 10, 12] {
        let quantizer = LinearQuantizer::new(-1.0, 1.0, bits)?;
        let decoded = quantizer.decode(&quantizer.encode(&sine_wave)?)?;
        let snr = compute_snr(&sine_wave, &decoded);
        println!("{:2}-bit quantization: SNR = {:.2} dB", bits, snr);
    }

    Ok(())
}
