//! Audio Processing Pipeline Examples
//!
//! This example demonstrates real-world audio processing pipelines:
//! - Complete audio tokenization workflow
//! - Multi-stage processing with different tokenizers
//! - Streaming audio processing for long signals
//! - Quality/bitrate tradeoff analysis
//! - Rate-distortion optimization
//!
//! Run with:
//! ```bash
//! cargo run --example audio_pipeline --all-features
//! ```

use kizzasi_tokenizer::metrics::{CompressionMetrics, QualityMetrics, RateDistortionCurve};
use kizzasi_tokenizer::*;
use scirs2_core::ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Audio Processing Pipeline Examples ===\n");

    // Example 1: Basic Audio Tokenization Pipeline
    basic_audio_pipeline()?;

    // Example 2: Multi-Stage Processing
    multi_stage_pipeline()?;

    // Example 3: Streaming Audio Processing
    streaming_audio_pipeline()?;

    // Example 4: Quality vs Bitrate Tradeoff
    quality_bitrate_tradeoff()?;

    // Example 5: Rate-Distortion Optimization
    rate_distortion_optimization()?;

    Ok(())
}

/// Example 1: Basic end-to-end audio tokenization
fn basic_audio_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Audio Tokenization Pipeline");
    println!("=====================================\n");

    // Simulate 1 second of audio at 16kHz sample rate
    let sample_rate = 16000;
    let duration = 1.0;
    let num_samples = (sample_rate as f32 * duration) as usize;

    // Generate synthetic audio: mix of sine waves
    let mut signal = Array1::zeros(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        // 440 Hz (A4) + 880 Hz (A5) + 1320 Hz (E6)
        signal[i] = 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
            + 0.1 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin();
    }

    println!(
        "Input: {} samples ({:.1}s at {}Hz)",
        num_samples, duration, sample_rate
    );
    println!(
        "Signal range: [{:.3}, {:.3}]",
        signal.iter().cloned().fold(f32::INFINITY, f32::min),
        signal.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Stage 1: Quantization (8-bit μ-law compression)
    println!("\nStage 1: μ-law Quantization (8-bit)");
    let mulaw = MuLawCodec::new(8);

    let start = Instant::now();
    let quantized = mulaw.encode(&signal)?;
    let encode_time = start.elapsed();

    println!("  Encoded in {:?}", encode_time);
    println!(
        "  Original: {} samples × 32 bits = {} bytes",
        num_samples,
        num_samples * 4
    );
    println!(
        "  Compressed: {} tokens × 8 bits = {} bytes",
        quantized.len(),
        quantized.len()
    );

    let compression = CompressionMetrics::compute(num_samples, 32, quantized.len());
    println!("  Compression ratio: {:.2}x", compression.compression_ratio);
    println!("  Space savings: {:.1}%", compression.space_savings_percent);

    // Stage 2: Reconstruction
    println!("\nStage 2: Reconstruction");
    let start = Instant::now();
    let reconstructed = mulaw.decode(&quantized)?;
    let decode_time = start.elapsed();

    println!("  Decoded in {:?}", decode_time);

    // Stage 3: Quality Analysis
    println!("\nStage 3: Quality Analysis");
    let metrics = QualityMetrics::compute(&signal, &reconstructed)?;
    println!("  SNR: {:.2} dB", metrics.snr_db);
    println!("  MSE: {:.6}", metrics.mse);
    println!("  MAE: {:.6}", metrics.mae);

    println!("\n{}\n", "=".repeat(60));
    Ok(())
}

/// Example 2: Multi-stage processing with different tokenizers
fn multi_stage_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Multi-Stage Processing Pipeline");
    println!("===================================\n");

    // Generate test signal: chirp (frequency sweep)
    let num_samples = 1024;
    let mut signal = Array1::zeros(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / num_samples as f32;
        let freq = 100.0 + t * 1900.0; // Sweep from 100Hz to 2kHz
        signal[i] = (2.0 * std::f32::consts::PI * freq * t).sin();
    }

    println!("Input: {} sample chirp signal", num_samples);

    // Stage 1: Wavelet decomposition
    println!("\nStage 1: Wavelet Decomposition");
    let wavelet_config = WaveletConfig {
        levels: 3,
        family: WaveletFamily::Daubechies4,
        bits: 8,
    };
    let wavelet = WaveletTokenizer::new(wavelet_config)?;

    let wavelet_tokens = wavelet.encode(&signal)?;
    println!("  Wavelet tokens: {} values", wavelet_tokens.len());

    // Stage 2: Quantization of wavelet coefficients
    println!("\nStage 2: Dead-Zone Quantization");
    let deadzone = DeadZoneQuantizer::new(6, 0.05, -2.0, 2.0)?;

    let quantized = deadzone.encode(&wavelet_tokens)?;
    println!("  Quantized tokens: {}", quantized.len());

    // Count near-zero coefficients
    let near_zero = wavelet_tokens.iter().filter(|&&x| x.abs() < 0.05).count();
    println!(
        "  Near-zero coefficients: {} ({:.1}%)",
        near_zero,
        near_zero as f32 / wavelet_tokens.len() as f32 * 100.0
    );

    // Stage 3: Reconstruction
    println!("\nStage 3: Reconstruction");
    let dequantized = deadzone.decode(&quantized)?;
    let reconstructed = wavelet.decode(&dequantized)?;

    // Quality metrics
    let metrics = QualityMetrics::compute(&signal, &reconstructed)?;
    println!("\nQuality Metrics:");
    println!("  SNR: {:.2} dB", metrics.snr_db);
    println!("  Normalized MSE: {:.6}", metrics.nmse);

    println!("\n{}\n", "=".repeat(60));
    Ok(())
}

/// Example 3: Streaming audio processing for long signals
fn streaming_audio_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Streaming Audio Processing");
    println!("==============================\n");

    // Simulate 10 seconds of audio
    let sample_rate = 16000;
    let duration = 10.0;
    let total_samples = (sample_rate as f32 * duration) as usize;

    println!(
        "Processing {} samples ({:.1}s) in streaming mode",
        total_samples, duration
    );

    // Create base tokenizer
    let base_tokenizer = LinearQuantizer::new(-1.0, 1.0, 8)?;

    // Create streaming tokenizer with 1024-sample chunks and 128-sample overlap
    let chunk_size = 1024;
    let overlap = 128;
    let streaming = StreamingTokenizer::new(base_tokenizer, chunk_size, overlap)?;

    println!("  Chunk size: {} samples", chunk_size);
    println!("  Overlap: {} samples", overlap);
    println!(
        "  Expected chunks: ~{}",
        (total_samples + chunk_size - 1) / (chunk_size - overlap)
    );

    // Generate long signal
    let mut signal = Array1::zeros(total_samples);
    for i in 0..total_samples {
        let t = i as f32 / sample_rate as f32;
        signal[i] = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
    }

    // Process in streaming mode
    println!("\nProcessing...");
    let start = Instant::now();
    let encoded_chunks = streaming.encode_streaming(&signal)?;
    let encode_time = start.elapsed();

    let start = Instant::now();
    let decoded = streaming.decode_streaming(&encoded_chunks)?;
    let decode_time = start.elapsed();

    println!(
        "  Encode time: {:?} ({:.2} samples/ms)",
        encode_time,
        total_samples as f64 / encode_time.as_secs_f64() / 1000.0
    );
    println!(
        "  Decode time: {:?} ({:.2} samples/ms)",
        decode_time,
        total_samples as f64 / decode_time.as_secs_f64() / 1000.0
    );

    // Verify reconstruction (truncate to original length due to overlap-add padding)
    let decoded_truncated = if decoded.len() > total_samples {
        decoded
            .slice(scirs2_core::ndarray::s![..total_samples])
            .to_owned()
    } else {
        decoded
    };

    let metrics = QualityMetrics::compute(&signal, &decoded_truncated)?;
    println!("\nStreaming Quality:");
    println!("  SNR: {:.2} dB", metrics.snr_db);
    println!(
        "  Output length: {} samples (input: {}, truncated to match)",
        decoded_truncated.len(),
        total_samples
    );

    println!("\n{}\n", "=".repeat(60));
    Ok(())
}

/// Example 4: Quality vs Bitrate tradeoff analysis
fn quality_bitrate_tradeoff() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Quality vs Bitrate Tradeoff");
    println!("===============================\n");

    // Generate test signal
    let num_samples = 2048;
    let mut signal = Array1::zeros(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / num_samples as f32;
        signal[i] = 0.7 * (2.0 * std::f32::consts::PI * 440.0 * t * 20.0).sin();
    }

    println!("Testing different bit depths:\n");
    println!(
        "{:<10} {:<15} {:<15} {:<15}",
        "Bits", "SNR (dB)", "Compression", "Throughput"
    );
    println!("{}", "-".repeat(60));

    for bits in [2, 4, 6, 8, 10, 12, 14, 16] {
        let quantizer = LinearQuantizer::new(-1.0, 1.0, bits)?;

        let start = Instant::now();
        let encoded = quantizer.encode(&signal)?;
        let decoded = quantizer.decode(&encoded)?;
        let elapsed = start.elapsed();

        let metrics = QualityMetrics::compute(&signal, &decoded)?;
        let compression = CompressionMetrics::compute(num_samples, 32, encoded.len());
        let throughput = (num_samples as f64 / elapsed.as_secs_f64()) / 1_000_000.0;

        println!(
            "{:<10} {:<15.2} {:<15.2} {:<15.2} Msamples/s",
            bits, metrics.snr_db, compression.compression_ratio, throughput
        );
    }

    println!("\nObservations:");
    println!("  - SNR increases ~6 dB per bit");
    println!("  - Lower bit depths offer better compression");
    println!("  - Throughput remains roughly constant");

    println!("\n{}\n", "=".repeat(60));
    Ok(())
}

/// Example 5: Rate-distortion optimization
fn rate_distortion_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Rate-Distortion Optimization");
    println!("================================\n");

    // Generate test signal
    let num_samples = 1024;
    let mut signal = Array1::zeros(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / num_samples as f32;
        signal[i] = (2.0 * std::f32::consts::PI * 440.0 * t * 10.0).sin() * 0.8;
    }

    // Build RD curve by testing different configurations
    let mut rd_curve = RateDistortionCurve::new();

    println!("Building Rate-Distortion curve...\n");

    // Test different adaptive quantizer configurations
    for bits in [4, 6, 8, 10, 12] {
        let quantizer = AdaptiveQuantizer::new(bits, 32, 0.5, -1.0, 1.0)?;

        let encoded = quantizer.encode(&signal)?;
        let decoded = quantizer.decode(&encoded)?;

        let metrics = QualityMetrics::compute(&signal, &decoded)?;
        let rate = bits as f64;

        rd_curve.add_point(rate, metrics.mse, metrics.snr_db);

        println!(
            "  {} bits: SNR = {:.2} dB, MSE = {:.6}",
            bits, metrics.snr_db, metrics.mse
        );
    }

    // Find optimal operating points
    println!("\nOptimal Operating Points:");

    if let Some(point) = rd_curve.find_best_for_snr(30.0) {
        println!("  For SNR ≥ 30 dB:");
        println!("    Rate: {:.1} bits/sample", point.rate);
        println!("    Actual SNR: {:.2} dB", point.snr_db);
    }

    if let Some(point) = rd_curve.find_best_for_rate(8.0) {
        println!("  For rate ≤ 8 bits/sample:");
        println!("    SNR: {:.2} dB", point.snr_db);
        println!("    Distortion: {:.6}", point.distortion);
    }

    println!("\n{}\n", "=".repeat(60));
    Ok(())
}
