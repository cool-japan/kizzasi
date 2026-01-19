//! Multi-channel Beamforming Example
//!
//! Demonstrates spatial filtering and direction-of-arrival estimation:
//! - Delay-and-Sum beamforming
//! - MVDR (Minimum Variance Distortionless Response)
//! - Adaptive beamforming with LMS
//! - Direction of Arrival (DOA) estimation

use kizzasi_io::{AdaptiveBeamformer, DOAEstimator, DelayAndSum, MicrophoneArray, MVDR};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Multi-channel Beamforming Examples");
    println!("===================================\n");

    // Configuration
    let sample_rate = 16000.0; // Hz
    let num_mics = 4;
    let mic_spacing = 0.05f32; // meters
    let sound_speed = 343.0; // m/s

    println!("System Configuration:");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Number of microphones: {}", num_mics);
    println!("  Microphone spacing: {} m", mic_spacing);
    println!("  Sound speed: {} m/s\n", sound_speed);

    // Create microphone array (linear array)
    let mut mic_positions = Vec::new();
    for i in 0..num_mics {
        mic_positions.push([i as f64 * mic_spacing as f64, 0.0, 0.0]);
    }

    let mic_array = MicrophoneArray::linear(num_mics, mic_spacing, sample_rate as f32);

    println!("Microphone Array Geometry:");
    println!("  Type: Linear array");
    println!("  Positions:");
    for (i, pos) in mic_positions.iter().enumerate() {
        println!(
            "    Mic {}: [{:.3}, {:.3}, {:.3}] m",
            i + 1,
            pos[0],
            pos[1],
            pos[2]
        );
    }
    println!();

    // Simulate signals from different directions
    let duration = 1.0; // seconds
    let n = (sample_rate * duration) as usize;

    println!("Creating simulated acoustic scene...");

    // Source 1: Speech from 0 degrees (front)
    let source_angle_1 = 0.0_f64.to_radians();
    let source_freq_1 = 1000.0; // Hz

    // Source 2: Noise from 60 degrees
    let source_angle_2 = 60.0_f64.to_radians();
    let source_freq_2 = 500.0;

    println!(
        "  Source 1: {} Hz tone at {} degrees (target)",
        source_freq_1, 0.0
    );
    println!(
        "  Source 2: {} Hz tone at {} degrees (interference)",
        source_freq_2, 60.0
    );
    println!();

    // Generate microphone signals with proper delays
    let mut mic_signals: Vec<Vec<f64>> = vec![vec![0.0; n]; num_mics];

    for (mic_idx, mic_signal) in mic_signals.iter_mut().enumerate() {
        let mic_pos = mic_positions[mic_idx][0];

        for (i, sig) in mic_signal.iter_mut().enumerate() {
            let t = i as f64 / sample_rate;

            // Source 1 delay
            let delay_1 = mic_pos * source_angle_1.cos() / sound_speed;
            let t1 = t - delay_1;
            let signal_1 = if t1 >= 0.0 {
                (2.0 * PI * source_freq_1 * t1).sin()
            } else {
                0.0
            };

            // Source 2 delay
            let delay_2 = mic_pos * source_angle_2.cos() / sound_speed;
            let t2 = t - delay_2;
            let signal_2 = if t2 >= 0.0 {
                0.5 * (2.0 * PI * source_freq_2 * t2).sin()
            } else {
                0.0
            };

            // Add some noise
            let noise = ((i * mic_idx * 12345) % 10000) as f64 / 100000.0;

            *sig = signal_1 + signal_2 + noise;
        }
    }

    // Example 1: Delay-and-Sum Beamforming
    println!("=== Delay-and-Sum Beamforming ===");
    println!("Classic beamforming technique using aligned delays\n");

    let target_angle = 0.0_f64.to_radians(); // Point toward source 1
    let das = DelayAndSum::new(mic_array.clone());

    // Convert mic_signals to Array2
    use scirs2_core::ndarray::Array2;
    let mut signals_array = Array2::zeros((num_mics, n));
    for (i, sig) in mic_signals.iter().enumerate() {
        for (j, &val) in sig.iter().enumerate() {
            signals_array[[i, j]] = val as f32;
        }
    }

    let output_das = das.process(&signals_array, target_angle as f32, 0.0)?; // elevation = 0 (horizontal)

    println!("✓ Delay-and-Sum beamforming complete");
    println!("  Output length: {} samples", output_das.len());
    println!("  Target direction: {} degrees", 0.0);

    // Measure signal enhancement
    let input_power = compute_power(&mic_signals[0]);
    let output_power = compute_power_f32(output_das.as_slice().expect("Array conversion failed"));
    let gain_db = 10.0 * (output_power as f64 / input_power).log10();

    println!("  Array gain: {:.2} dB", gain_db);
    println!("\n  Characteristics:");
    println!("    + Simple and robust");
    println!("    + Works well with multiple sources");
    println!("    - Not optimal for noise suppression");
    println!();

    // Example 2: MVDR Beamforming
    println!("=== MVDR (Minimum Variance Distortionless Response) ===");
    println!("Optimal beamforming that minimizes output variance\n");

    let mvdr = MVDR::new(mic_array.clone(), Some(1e-3));

    let output_mvdr = mvdr.process(&signals_array, target_angle as f32, 0.0)?; // elevation = 0 (horizontal)

    println!("✓ MVDR beamforming complete");
    println!("  Output length: {} samples", output_mvdr.len());
    println!("  Regularization: 1e-3");

    let mvdr_power = compute_power_f32(output_mvdr.as_slice().expect("Array conversion failed"));
    let mvdr_gain_db = 10.0 * (mvdr_power as f64 / input_power).log10();

    println!("  Array gain: {:.2} dB", mvdr_gain_db);
    println!("\n  Characteristics:");
    println!("    + Optimal noise suppression");
    println!("    + Adaptive to interference");
    println!("    - Requires accurate steering vector");
    println!("    - Sensitive to mismatch");
    println!();

    // Example 3: Adaptive Beamforming
    println!("=== Adaptive Beamforming (LMS) ===");
    println!("Real-time adaptive beamforming using LMS algorithm\n");

    let desired_signal: Vec<f32> = mic_signals[0].iter().map(|&x| x as f32).collect();
    let mut adaptive = AdaptiveBeamformer::new(mic_array.clone(), 0.01);

    // Process each time sample
    let mut output_adaptive = Vec::new();
    use scirs2_core::ndarray::Array1;
    for t in 0..n {
        let input_sample = Array1::from_vec((0..num_mics).map(|m| signals_array[[m, t]]).collect());
        let (output, _error) = adaptive.adapt(&input_sample, desired_signal[t]);
        output_adaptive.push(output);
    }

    println!("✓ Adaptive beamforming complete");
    println!("  Learning rate: 0.01");
    println!("  Filter length: 1");

    let adaptive_power = compute_power_f32(&output_adaptive);
    let adaptive_gain_db = 10.0 * (adaptive_power as f64 / input_power).log10();

    println!("  Array gain: {:.2} dB", adaptive_gain_db);
    println!("\n  Characteristics:");
    println!("    + Adapts to changing environment");
    println!("    + No prior knowledge needed");
    println!("    - Convergence time required");
    println!("    - May not be optimal in non-stationary scenarios");
    println!();

    // Example 4: Direction of Arrival Estimation
    println!("=== Direction of Arrival (DOA) Estimation ===");
    println!("Estimating source directions using spatial filtering\n");

    let doa = DOAEstimator::new(mic_array);

    // Search over angular range (SRP-PHAT scans the entire 360 degree azimuth)
    let search_resolution = 36; // 10 degree resolution

    let (best_azimuth, max_power) = doa.estimate_srp(&signals_array, search_resolution)?;

    println!("✓ DOA estimation complete");
    println!("  Search resolution: {} angular samples", search_resolution);
    println!(
        "  Angular resolution: ~{:.1} degrees",
        360.0 / search_resolution as f64
    );
    println!("\n  Detected source direction:");
    println!(
        "    Azimuth: {:.1}° (power: {:.3})",
        best_azimuth.to_degrees(),
        max_power
    );

    println!("\n  DOA estimation methods:");
    println!("    - SRP (Steered Response Power)");
    println!("    - MUSIC (Multiple Signal Classification)");
    println!("    - ESPRIT (Estimation of Signal Parameters via Rotational Invariance)");
    println!();

    // Comparison and Summary
    println!("=== Summary and Recommendations ===");
    println!();
    println!("Beamforming Method Comparison:");
    println!();
    println!("Delay-and-Sum:");
    println!("  Best for: General purpose, multiple sources");
    println!("  Pros: Robust, simple, works well in various conditions");
    println!("  Cons: Not optimal for noise suppression");
    println!("  Typical gain: {:.1} dB", gain_db);
    println!();
    println!("MVDR:");
    println!("  Best for: Maximum noise/interference suppression");
    println!("  Pros: Optimal under certain conditions, adaptive");
    println!("  Cons: Requires covariance estimation, sensitive to errors");
    println!("  Typical gain: {:.1} dB", mvdr_gain_db);
    println!();
    println!("Adaptive (LMS):");
    println!("  Best for: Time-varying environments");
    println!("  Pros: Self-adjusting, minimal prior knowledge");
    println!("  Cons: Convergence time, computational cost");
    println!("  Typical gain: {:.1} dB", adaptive_gain_db);
    println!();
    println!("Applications:");
    println!("  - Hands-free communication systems");
    println!("  - Smart speakers and voice assistants");
    println!("  - Hearing aids");
    println!("  - Sonar and radar systems");
    println!("  - Astronomical radio telescopes");
    println!("  - Conference room audio");

    Ok(())
}

/// Compute signal power (RMS squared) for f64
fn compute_power(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }

    let sum: f64 = signal.iter().map(|&x| x * x).sum();
    sum / signal.len() as f64
}

/// Compute signal power (RMS squared) for f32
fn compute_power_f32(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }

    let sum: f32 = signal.iter().map(|&x| x * x).sum();
    sum / signal.len() as f32
}
