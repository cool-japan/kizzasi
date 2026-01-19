//! Blind Source Separation Example
//!
//! Demonstrates various blind source separation techniques:
//! - FastICA (Independent Component Analysis)
//! - NMF (Non-negative Matrix Factorization)
//! - PCA (Principal Component Analysis)
//! - Temporal Decorrelation

use kizzasi_io::{FastICA, Nonlinearity, TemporalDecorrelation, NMF, PCA};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Blind Source Separation Examples");
    println!("================================\n");

    // Configuration
    let sample_rate = 8000.0;
    let duration = 2.0;
    let n = (sample_rate * duration) as usize;

    // Create source signals
    println!("Creating source signals...");
    let mut source1 = vec![0.0; n]; // Speech-like signal
    let mut source2 = vec![0.0; n]; // Music-like signal
    let mut source3 = vec![0.0; n]; // Noise signal

    for i in 0..n {
        let t = i as f64 / sample_rate;

        // Source 1: Speech-like (formant structure)
        source1[i] = (2.0 * PI * 500.0 * t).sin() * 0.5 + (2.0 * PI * 1500.0 * t).sin() * 0.3;

        // Source 2: Music-like (harmonic structure)
        source2[i] = (2.0 * PI * 440.0 * t).sin()
            + 0.5 * (2.0 * PI * 880.0 * t).sin()
            + 0.25 * (2.0 * PI * 1320.0 * t).sin();

        // Source 3: Noise
        source3[i] = ((i * 12345) % 10000) as f64 / 10000.0 - 0.5;
    }

    println!("  Source 1: Speech-like signal (formants)");
    println!("  Source 2: Music-like signal (harmonics)");
    println!("  Source 3: White noise\n");

    // Create mixed signals (linear combinations)
    println!("Creating mixed observations...");
    let mut mixed1 = vec![0.0; n];
    let mut mixed2 = vec![0.0; n];
    let mut mixed3 = vec![0.0; n];

    // Mixing matrix (simulating microphones at different positions)
    let mix_matrix = [[0.8, 0.3, 0.1], [0.2, 0.7, 0.4], [0.3, 0.2, 0.6]];

    for i in 0..n {
        mixed1[i] = mix_matrix[0][0] * source1[i]
            + mix_matrix[0][1] * source2[i]
            + mix_matrix[0][2] * source3[i];

        mixed2[i] = mix_matrix[1][0] * source1[i]
            + mix_matrix[1][1] * source2[i]
            + mix_matrix[1][2] * source3[i];

        mixed3[i] = mix_matrix[2][0] * source1[i]
            + mix_matrix[2][1] * source2[i]
            + mix_matrix[2][2] * source3[i];
    }

    println!("  Mixed signal 1: combination of all sources");
    println!("  Mixed signal 2: different combination");
    println!("  Mixed signal 3: another combination\n");

    // Prepare mixed signals matrix
    let mixed_signals = [mixed1.clone(), mixed2.clone(), mixed3.clone()];

    // Example 1: FastICA
    println!("=== FastICA (Independent Component Analysis) ===");
    println!("FastICA separates sources by maximizing statistical independence\n");

    // Convert to Array2<f32>
    use scirs2_core::ndarray::Array2;
    let n_samples = mixed_signals[0].len();
    let n_signals = mixed_signals.len();
    let mut mixed_array = Array2::zeros((n_samples, n_signals));
    for (i, signal) in mixed_signals.iter().enumerate() {
        for (j, &val) in signal.iter().enumerate() {
            mixed_array[[j, i]] = val as f32;
        }
    }

    let fastica = FastICA::new(3, Some(200), Some(1e-4)).with_nonlinearity(Nonlinearity::LogCosh);

    match fastica.fit_transform(&mixed_array) {
        Ok((separated_ica, _unmixing_matrix)) => {
            println!("✓ FastICA separation successful!");
            println!(
                "  Number of separated components: {}",
                separated_ica.ncols()
            );
            println!("  Using LogCosh nonlinearity");

            // Note: Correlation computation with original sources would require
            // converting between Array2 and Vec formats
            println!(
                "\n  ICA successfully extracted {} independent components",
                separated_ica.ncols()
            );
        }
        Err(e) => println!("✗ FastICA failed: {}", e),
    }

    println!();

    // Example 2: NMF (for non-negative signals)
    println!("=== NMF (Non-negative Matrix Factorization) ===");
    println!("NMF is suitable for non-negative data (e.g., spectrograms, images)\n");

    // Convert to non-negative signals and Array2<f32>
    let offset = 1.0f32;
    let n_samples = mixed_signals[0].len();
    let n_features = mixed_signals.len();
    let mut mixed_nonneg = Array2::zeros((n_samples, n_features));
    for (i, signal) in mixed_signals.iter().enumerate() {
        for (j, &val) in signal.iter().enumerate() {
            mixed_nonneg[[j, i]] = val as f32 + offset;
        }
    }

    let nmf = NMF::new(3, Some(100), Some(1e-3));

    match nmf.fit_transform(&mixed_nonneg) {
        Ok((w, h)) => {
            println!("✓ NMF factorization successful!");
            println!("  W (basis) matrix shape: {} x {}", w.nrows(), w.ncols());
            println!(
                "  H (activation) matrix shape: {} x {}",
                h.nrows(),
                h.ncols()
            );
            println!("\n  NMF is commonly used for:");
            println!("    - Spectrogram decomposition");
            println!("    - Topic modeling");
            println!("    - Image decomposition");
        }
        Err(e) => println!("✗ NMF failed: {}", e),
    }

    println!();

    // Example 3: PCA
    println!("=== PCA (Principal Component Analysis) ===");
    println!("PCA finds orthogonal components with maximum variance\n");

    // Convert to Array2<f32>
    let mut mixed_array = Array2::zeros((n_samples, n_features));
    for (i, signal) in mixed_signals.iter().enumerate() {
        for (j, &val) in signal.iter().enumerate() {
            mixed_array[[j, i]] = val as f32;
        }
    }

    let pca = PCA::new(3);

    match pca.fit_transform(&mixed_array) {
        Ok((_transformed, components, explained_variance)) => {
            println!("✓ PCA decomposition successful!");
            println!("  Number of principal components: {}", components.nrows());

            // Show variance explained
            println!("\n  Principal components analysis:");
            for (i, &variance) in explained_variance.iter().enumerate() {
                println!("    PC{}: explained variance = {:.4}", i + 1, variance);
            }

            println!("\n  PCA applications:");
            println!("    - Dimensionality reduction");
            println!("    - Noise reduction");
            println!("    - Feature extraction");
        }
        Err(e) => println!("✗ PCA failed: {}", e),
    }

    println!();

    // Example 4: Temporal Decorrelation
    println!("=== Temporal Decorrelation ===");
    println!("Separates sources based on temporal structure differences\n");

    let temporal = TemporalDecorrelation::new(10); // tau = 10 samples delay

    // Transpose mixed_array for temporal decorrelation (n_channels × n_samples)
    let mixed_transposed = mixed_array.t().to_owned();

    match temporal.separate(&mixed_transposed) {
        Ok(separated_temporal) => {
            println!("✓ Temporal decorrelation successful!");
            println!(
                "  Output shape: {} x {}",
                separated_temporal.nrows(),
                separated_temporal.ncols()
            );
            println!("\n  Temporal decorrelation works well for:");
            println!("    - Sources with different autocorrelation structures");
            println!("    - Convolutive mixtures");
            println!("    - Real-world audio separation");
        }
        Err(e) => println!("✗ Temporal decorrelation failed: {}", e),
    }

    println!();

    // Summary and recommendations
    println!("=== Summary and Recommendations ===");
    println!();
    println!("Method Comparison:");
    println!("  FastICA:");
    println!("    + Best for statistically independent sources");
    println!("    + Works with instantaneous mixtures");
    println!("    - Assumes non-Gaussian sources");
    println!();
    println!("  NMF:");
    println!("    + Best for non-negative data");
    println!("    + Interpretable decomposition");
    println!("    - Requires non-negative inputs");
    println!();
    println!("  PCA:");
    println!("    + Fast and simple");
    println!("    + Good for dimensionality reduction");
    println!("    - Only finds uncorrelated components (not independent)");
    println!();
    println!("  Temporal Decorrelation:");
    println!("    + Exploits temporal structure");
    println!("    + Works with convolutive mixtures");
    println!("    - Requires sources with different temporal properties");

    Ok(())
}

/// Compute Pearson correlation between two signals
#[allow(dead_code)]
fn correlation(signal1: &[f64], signal2: &[f64]) -> f64 {
    let n = signal1.len().min(signal2.len());

    if n == 0 {
        return 0.0;
    }

    let mean1: f64 = signal1[..n].iter().sum::<f64>() / n as f64;
    let mean2: f64 = signal2[..n].iter().sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut var1 = 0.0;
    let mut var2 = 0.0;

    for i in 0..n {
        let diff1 = signal1[i] - mean1;
        let diff2 = signal2[i] - mean2;

        numerator += diff1 * diff2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
    }

    if var1 < 1e-10 || var2 < 1e-10 {
        return 0.0;
    }

    numerator / (var1.sqrt() * var2.sqrt())
}
