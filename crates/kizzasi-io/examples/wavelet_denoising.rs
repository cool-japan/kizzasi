//! Wavelet Denoising Example
//!
//! Demonstrates how to:
//! - Perform wavelet decomposition
//! - Apply wavelet denoising
//! - Reconstruct signals
//! - Compare different wavelet families

use kizzasi_io::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Wavelet Denoising Example");
    println!("=========================\n");

    // Generate clean signal
    println!("Generating clean signal...");
    let sample_rate = 1000.0;
    let duration = 1.0;
    let n_samples = (sample_rate * duration) as usize;

    let mut sine_gen = SineGenerator::new(50.0, sample_rate, 0.8);
    let clean_signal = sine_gen.generate(n_samples);

    // Add noise
    println!("Adding noise to signal...");
    let mut noise_gen = WhiteNoiseGenerator::new(0.2);
    let noise = noise_gen.generate(n_samples);

    let noisy_signal: Vec<f32> = clean_signal
        .iter()
        .zip(noise.iter())
        .map(|(s, n)| s + n)
        .collect();

    println!("  Clean signal: {} samples", clean_signal.len());
    println!("  Noise added with amplitude: 0.2\n");

    // Test different wavelet families
    let wavelets = vec![
        ("Haar", WaveletType::Haar),
        ("Daubechies-4", WaveletType::Daubechies4),
        ("Daubechies-6", WaveletType::Daubechies6),
        ("Symlet-4", WaveletType::Symlet4),
        ("Coiflet-1", WaveletType::Coiflet1),
    ];

    for (name, wavelet_type) in wavelets {
        println!("\n{} Wavelet:", name);
        println!("{}", "=".repeat(name.len() + 9));

        let analyzer = WaveletAnalyzer::new(wavelet_type);

        // Single-level decomposition
        println!("\n1. Single-level DWT:");
        let dwt_result = analyzer.dwt(&noisy_signal);
        println!(
            "   Approximation coefficients: {}",
            dwt_result.approximation.len()
        );
        println!("   Detail coefficients: {}", dwt_result.detail.len());

        // Multi-level decomposition
        let levels = 3;
        println!("\n2. Multi-level DWT ({} levels):", levels);
        let multilevel = analyzer.dwt_multilevel(&noisy_signal, levels);
        println!("   Total levels: {}", multilevel.levels);
        println!(
            "   Approximation length: {}",
            multilevel.approximation.len()
        );
        for (i, detail) in multilevel.details.iter().enumerate() {
            println!("   Level {} detail: {}", i + 1, detail.len());
        }

        // Wavelet denoising
        println!("\n3. Wavelet Denoising:");
        let threshold = 0.1; // Soft threshold for noise removal
        let denoised = analyzer.denoise(&noisy_signal, levels, threshold);
        println!("   Denoised signal length: {}", denoised.len());

        // Calculate denoising effectiveness
        let original_energy: f32 = noisy_signal.iter().map(|x| x * x).sum();
        let denoised_energy: f32 = denoised.iter().map(|x| x * x).sum();
        let energy_reduction = (1.0 - denoised_energy / original_energy) * 100.0;

        println!("   Energy reduction: {:.2}%", energy_reduction);

        // Calculate MSE with respect to clean signal
        let mse: f32 = clean_signal
            .iter()
            .zip(denoised.iter())
            .map(|(c, d)| (c - d).powi(2))
            .sum::<f32>()
            / clean_signal.len() as f32;

        println!("   MSE vs clean signal: {:.6}", mse);
    }

    // Stationary Wavelet Transform
    println!("\n\nStationary Wavelet Transform (SWT):");
    println!("====================================");

    let analyzer = WaveletAnalyzer::new(WaveletType::Daubechies4);
    let levels = 2;

    println!("Computing SWT with {} levels...", levels);
    let swt_result = analyzer.swt(&noisy_signal, levels);

    println!("SWT Results:");
    for (i, (approx, detail)) in swt_result.iter().enumerate() {
        println!(
            "  Level {}: approx={}, detail={}",
            i + 1,
            approx.len(),
            detail.len()
        );
    }

    // Wavelet energy analysis
    println!("\n\nWavelet Energy Analysis:");
    println!("========================");

    let multilevel_for_energy = analyzer.dwt_multilevel(&noisy_signal, 3);
    let energy_result = analyzer.wavelet_energy(&multilevel_for_energy);
    println!("Energy distribution across levels:");
    for (i, &energy) in energy_result.iter().enumerate() {
        println!("  Level {}: {:.2}", i, energy);
    }

    let total_energy: f32 = energy_result.iter().sum();
    println!("Total energy: {:.2}", total_energy);

    // Signal reconstruction
    println!("\n\nSignal Reconstruction:");
    println!("======================");

    println!("Performing DWT...");
    let dwt = analyzer.dwt(&noisy_signal);

    println!("Performing IDWT (reconstruction)...");
    let reconstructed = analyzer.idwt(&dwt.approximation, &dwt.detail, noisy_signal.len());

    println!("  Original length: {}", noisy_signal.len());
    println!("  Reconstructed length: {}", reconstructed.len());

    // Check reconstruction error
    let max_len = noisy_signal.len().min(reconstructed.len());
    let reconstruction_error: f32 = noisy_signal[..max_len]
        .iter()
        .zip(reconstructed[..max_len].iter())
        .map(|(o, r)| (o - r).abs())
        .sum::<f32>()
        / max_len as f32;

    println!(
        "  Average reconstruction error: {:.6}",
        reconstruction_error
    );

    println!("\nExample completed successfully!");
    println!("\nKey Insights:");
    println!("- Different wavelet families have different characteristics");
    println!("- Higher decomposition levels capture different frequency components");
    println!("- Wavelet denoising preserves signal features while removing noise");
    println!("- SWT provides translation-invariant decomposition");

    Ok(())
}
