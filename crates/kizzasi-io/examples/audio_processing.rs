//! Audio Processing Example
//!
//! Demonstrates how to:
//! - Capture audio from a microphone
//! - Apply real-time filtering
//! - Compute spectrograms
//! - Save processed audio

use kizzasi_io::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Audio Processing Example");
    println!("========================\n");

    // List available audio devices
    println!("Available input devices:");
    let input_devices = AudioInput::list_devices()?;
    for (i, device) in input_devices.iter().enumerate() {
        println!("  {}: {}", i, device);
    }
    println!();

    // Configure audio input
    let audio_config = AudioConfig::new()
        .sample_rate(44100)
        .channels(1)
        .buffer_size(1024);

    println!("Audio Configuration:");
    println!("  Sample Rate: {} Hz", audio_config.sample_rate);
    println!("  Channels: {}", audio_config.channels);
    println!("  Buffer Size: {} samples\n", audio_config.buffer_size);

    // Create audio input (commented out - requires actual audio device)
    // let mut audio_input = AudioInput::new(audio_config)?;
    // audio_input.start()?;

    // Instead, use a signal generator for demonstration
    println!("Generating test signal (1000 Hz sine wave)...");
    let mut generator = SineGenerator::new(1000.0, 44100.0, 0.5);
    let signal = generator.generate(44100); // 1 second

    // Create signal processor
    let mut processor = SignalProcessor::new(signal.len());

    // 1. Apply lowpass filter
    println!("\n1. Applying lowpass filter (cutoff: 0.2)");
    let filtered = processor.apply_filter(
        &signal,
        Filter::LowPass {
            cutoff: 0.2,
            order: 4,
        },
    )?;
    println!("   Filtered signal length: {} samples", filtered.len());

    // 2. Compute FFT
    println!("\n2. Computing FFT...");
    let fft_result = processor.fft(&signal)?;
    let magnitude: Vec<f32> = fft_result.iter().map(|c| c.norm()).collect();
    let peak_bin = magnitude
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    let peak_freq = (peak_bin as f32 * 44100.0) / signal.len() as f32;
    println!("   Peak frequency: {:.2} Hz", peak_freq);

    // 3. Compute spectrogram
    println!("\n3. Computing spectrogram...");
    let spec = processor.spectrogram(&signal, 512, 128, WindowType::Hann)?;
    println!(
        "   Spectrogram dimensions: {} frames × {} bins",
        spec.num_frames, spec.num_bins
    );
    println!(
        "   Time resolution: {:.3} ms/frame",
        spec.hop_length as f32 / 44100.0 * 1000.0
    );

    // 4. Compute MFCC
    println!("\n4. Computing MFCC...");
    let mfcc = processor.mfcc(&signal, 13, 512, 128, 40)?; // 13 MFCCs, 512 FFT, 128 hop, 40 mel bands
    println!(
        "   MFCC coefficients: {} frames × {} dimensions",
        mfcc.len(),
        if mfcc.is_empty() { 0 } else { mfcc[0].len() }
    );

    // 5. Resample signal
    println!("\n5. Resampling to 48 kHz...");
    let mut resampler = StreamingResampler::new(44100.0, 48000.0);
    let resampled = resampler.process(&signal.to_vec());
    println!("   Original length: {} samples", signal.len());
    println!("   Resampled length: {} samples", resampled.len());

    // 6. Health monitoring
    println!("\n6. Monitoring signal health...");
    let mut monitor = HealthMonitor::new();
    monitor.record_samples(&signal.to_vec());

    let health = monitor.health();
    let quality = monitor.signal_quality();

    println!("   Samples processed: {}", health.samples_processed);
    println!("   Signal quality:");
    println!("     SNR: {:.2} dB", quality.snr_db);
    println!("     Crest factor: {:.2}", quality.crest_factor);
    println!("     DC offset: {:.4}", quality.dc_offset);

    println!("\nExample completed successfully!");
    Ok(())
}
