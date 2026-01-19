//! Real-Time Signal Filtering Example
//!
//! Demonstrates how to:
//! - Create real-time audio/signal streams
//! - Apply filters in real-time
//! - Use ring buffers for low-latency processing
//! - Monitor performance and latency

use kizzasi_io::*;
use std::time::Instant;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Real-Time Signal Filtering Example");
    println!("===================================\n");

    // Configuration
    let sample_rate = 44100.0;
    let buffer_size = 256; // Small buffer for low latency
    let channels = 1;

    println!("Stream Configuration:");
    println!("  Sample Rate: {} Hz", sample_rate);
    println!(
        "  Buffer Size: {} samples ({:.2} ms)",
        buffer_size,
        buffer_size as f32 / sample_rate * 1000.0
    );
    println!("  Channels: {}\n", channels);

    // Create signal generator
    let mut sine_gen = SineGenerator::new(440.0, sample_rate, 0.5);

    // Create ring buffer for low-latency processing
    let mut ring_buffer = SignalRingBuffer::new(1024);

    // Create filters
    let mut lowpass_filter = IirFilter::butterworth_lowpass(0.1)?;
    let mut highpass_filter = IirFilter::butterworth_highpass(0.05)?;

    println!("Filters configured:");
    println!("  Lowpass: Butterworth, cutoff=0.1");
    println!("  Highpass: Butterworth, cutoff=0.05\n");

    // Health monitoring
    let mut monitor = HealthMonitor::new();

    // Lock-free queue for zero-copy processing
    let signal_queue = SignalQueue::new(4096, buffer_size);

    println!("Processing signal in real-time...\n");

    // Simulate real-time processing loop
    for iteration in 0..20 {
        let start = Instant::now();

        // Generate input signal
        let input_signal = sine_gen.generate(buffer_size);

        // Push to ring buffer
        ring_buffer.push_slice(input_signal.as_slice().unwrap());

        // Push to signal queue (zero-copy)
        let _ = signal_queue.write_samples(input_signal.as_slice().unwrap());

        // Apply lowpass filter
        let filtered_low = lowpass_filter.process(&input_signal);

        // Apply highpass filter
        let _filtered_high = highpass_filter.process(&input_signal);

        // Record samples for monitoring
        monitor.record_samples(filtered_low.as_slice().unwrap());

        // Measure latency
        let latency = start.elapsed();
        monitor.record_latency(latency);

        // Display statistics every 5 iterations
        if iteration % 5 == 0 && iteration > 0 {
            let health = monitor.health();
            let quality = monitor.signal_quality();
            let latency_stats = monitor.latency_stats();

            println!("Iteration {}:", iteration);
            println!(
                "  Ring buffer: {} samples, RMS={:.3}",
                ring_buffer.len(),
                ring_buffer.rms()
            );
            println!(
                "  Signal quality: SNR={:.2} dB, Crest={:.2}",
                quality.snr_db, quality.crest_factor
            );
            println!(
                "  Latency: avg={:.2}µs, p99={:.2}µs",
                latency_stats.mean.as_micros(),
                latency_stats.p99.as_micros()
            );
            println!("  Health status: {:?}\n", health.status);
        }

        // Simulate real-time constraints
        sleep(Duration::from_millis(5)).await;
    }

    // Final statistics
    println!("\nFinal Statistics:");
    println!("================");

    let health = monitor.health();
    let quality = monitor.signal_quality();
    let latency_stats = monitor.latency_stats();

    println!("Processing:");
    println!("  Total samples: {}", health.samples_processed);
    println!("  Underruns: {}", health.underruns);
    println!("  Overruns: {}", health.overruns);

    println!("\nSignal Quality:");
    println!("  SNR: {:.2} dB", quality.snr_db);
    println!("  Crest factor: {:.2}", quality.crest_factor);
    println!("  DC offset: {:.4}", quality.dc_offset);
    println!("  Clipping ratio: {:.3}%", quality.clipping_ratio * 100.0);

    println!("\nLatency:");
    println!("  Min: {:.2} µs", latency_stats.min.as_micros());
    println!("  Avg: {:.2} µs", latency_stats.mean.as_micros());
    println!("  Max: {:.2} µs", latency_stats.max.as_micros());
    println!("  P50: {:.2} µs", latency_stats.p50.as_micros());
    println!("  P95: {:.2} µs", latency_stats.p95.as_micros());
    println!("  P99: {:.2} µs", latency_stats.p99.as_micros());

    println!("\nRing Buffer Statistics:");
    println!("  Length: {} samples", ring_buffer.len());
    println!("  Mean: {:.3}", ring_buffer.mean());
    println!("  RMS: {:.3}", ring_buffer.rms());
    println!("  Peak-to-peak: {:.3}", ring_buffer.peak_to_peak());

    // Example of zero-copy buffer usage
    println!("\nZero-Copy Buffer Example:");
    let data = vec![1.0f32; 1024];
    let shared_buffer = SharedSignalBuffer::new(data);

    let slice1 = shared_buffer.slice(0, 512);
    let slice2 = shared_buffer.slice(512, 1024);

    println!("  Shared buffer references: {}", shared_buffer.ref_count());
    println!("  Slice 1 length: {}", slice1.len());
    println!("  Slice 2 length: {}", slice2.len());

    drop(slice1);
    drop(slice2);
    println!("  References after drop: {}", shared_buffer.ref_count());

    println!("\nExample completed successfully!");
    Ok(())
}
