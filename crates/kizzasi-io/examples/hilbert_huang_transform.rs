//! Hilbert-Huang Transform (HHT) Example
//!
//! Demonstrates EMD decomposition and Hilbert spectral analysis
//! for non-stationary and non-linear signal analysis

use kizzasi_io::{EmdConfig, EmpiricalModeDecomposition, EnsembleEmd};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hilbert-Huang Transform Example");
    println!("================================\n");

    // Configuration
    let sample_rate = 1000.0; // Hz
    let duration = 2.0; // seconds
    let n = (sample_rate * duration) as usize;

    // Create a complex non-stationary signal:
    // - Low frequency component (5 Hz) with increasing amplitude
    // - High frequency component (50 Hz) with decreasing amplitude
    // - Chirp signal (frequency increases linearly)
    let mut signal = vec![0.0; n];

    for (i, sig) in signal.iter_mut().enumerate() {
        let t = i as f64 / sample_rate;

        // Low frequency with increasing amplitude
        let amp1 = 0.5 + 0.5 * t;
        let comp1 = amp1 * (2.0 * PI * 5.0 * t).sin();

        // High frequency with decreasing amplitude
        let amp2 = 1.0 - 0.5 * t;
        let comp2 = amp2 * (2.0 * PI * 50.0 * t).sin();

        // Chirp: frequency increases from 10 Hz to 30 Hz
        let freq_start = 10.0;
        let freq_end = 30.0;
        let freq = freq_start + (freq_end - freq_start) * t / duration;
        let phase = 2.0 * PI * freq * t;
        let chirp = 0.7 * phase.sin();

        *sig = comp1 + comp2 + chirp;
    }

    println!("Signal characteristics:");
    println!("  Duration: {:.1} seconds", duration);
    println!("  Sample rate: {:.0} Hz", sample_rate);
    println!("  Samples: {}", n);
    println!("  Components: Low-freq (5 Hz), High-freq (50 Hz), Chirp (10-30 Hz)\n");

    // EMD decomposition
    println!("Performing EMD decomposition...");
    let config = EmdConfig {
        max_imfs: 8,
        max_sifting_iterations: 100,
        sd_threshold: 0.3,
        min_extrema: 3,
        boundary_extension: true,
        extension_method: "mirror".to_string(),
    };

    let emd = EmpiricalModeDecomposition::new(sample_rate, config.clone());
    let result = emd.decompose(&signal)?;

    println!("EMD Results:");
    println!("  Number of IMFs extracted: {}", result.imfs.len());
    println!("  Residual length: {}\n", result.residual.len());

    // Analyze each IMF
    for (i, imf) in result.imfs.iter().enumerate() {
        println!("IMF #{}", i + 1);

        // Compute statistics
        let mean_freq: f64 = imf.frequency.iter().sum::<f64>() / imf.frequency.len() as f64;
        let mean_amp: f64 = imf.amplitude.iter().sum::<f64>() / imf.amplitude.len() as f64;

        let max_amp = imf.amplitude.iter().copied().fold(0.0_f64, f64::max);

        println!("  Mean instantaneous frequency: {:.2} Hz", mean_freq);
        println!("  Mean amplitude: {:.4}", mean_amp);
        println!("  Maximum amplitude: {:.4}", max_amp);

        // Show sample of instantaneous frequencies
        if imf.frequency.len() >= 5 {
            print!("  Frequency evolution (first 5 samples): ");
            for j in 0..5 {
                print!("{:.2} ", imf.frequency[j]);
            }
            println!("Hz");
        }
        println!();
    }

    // Residual analysis
    let mean_residual: f64 = result.residual.iter().sum::<f64>() / result.residual.len() as f64;
    let min_residual = result
        .residual
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_residual = result
        .residual
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("Residual (trend):");
    println!("  Mean value: {:.6}", mean_residual);
    println!("  Range: [{:.4}, {:.4}]\n", min_residual, max_residual);

    // Ensemble EMD (EEMD) for more robust decomposition
    println!("Performing EEMD (Ensemble EMD)...");
    let eemd = EnsembleEmd::new(
        sample_rate,
        config,
        10,   // ensemble size
        0.05, // noise amplitude (5% of signal std)
    );

    let eemd_result = eemd.decompose(&signal)?;

    println!("EEMD Results:");
    println!("  Number of IMFs extracted: {}", eemd_result.imfs.len());
    println!("  Ensemble averaging reduces mode mixing\n");

    // Verify reconstruction
    let mut reconstructed = vec![0.0; n];
    for imf in &eemd_result.imfs {
        for (i, val) in imf.data.iter().enumerate().take(n) {
            reconstructed[i] += val;
        }
    }
    for (i, val) in eemd_result.residual.iter().enumerate().take(n) {
        reconstructed[i] += val;
    }

    let reconstruction_error: f64 = signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(s, r)| (s - r).powi(2))
        .sum::<f64>()
        / n as f64;

    println!("Reconstruction quality:");
    println!("  Mean squared error: {:.2e}", reconstruction_error);
    println!("  (Lower is better, indicates successful decomposition)\n");

    // Hilbert Spectral Analysis
    println!("Hilbert Spectral Analysis:");
    println!("  Each IMF contains:");
    println!("    - Instantaneous amplitude (envelope)");
    println!("    - Instantaneous frequency");
    println!("    - Instantaneous phase");
    println!("  This provides time-frequency representation for non-stationary signals\n");

    println!("Applications of HHT:");
    println!("  - Biomedical signal analysis (EEG, ECG)");
    println!("  - Vibration analysis and fault detection");
    println!("  - Geophysical signal processing");
    println!("  - Speech and audio analysis");
    println!("  - Climate and meteorological data analysis");

    Ok(())
}
