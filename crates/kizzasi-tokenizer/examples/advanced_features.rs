//! Advanced Features Example
//!
//! This example demonstrates the usage of advanced ML features for tokenizers:
//! - Token Dropout for regularization during training
//! - Jitter Injection for robustness testing
//! - Temporal Coherence Constraints for smoothing
//! - Hierarchical Tokenization for multi-resolution coding
//!
//! Run with:
//! ```bash
//! cargo run --example advanced_features --features vqvae
//! ```

#[cfg(feature = "vqvae")]
use kizzasi_tokenizer::{
    add_jitter, apply_temporal_coherence, apply_token_dropout, HierarchicalConfig,
    HierarchicalTokenizer, JitterConfig, TemporalCoherenceConfig, TemporalFilterType,
    TokenDropoutConfig,
};

#[cfg(feature = "vqvae")]
use scirs2_core::ndarray::Array1;

#[cfg(feature = "vqvae")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Features Example ===\n");

    // Create a test signal
    let n_samples = 256;
    let signal = Array1::from_vec(
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / n_samples as f32;
                (2.0 * std::f32::consts::PI * 10.0 * t).sin()
            })
            .collect(),
    );

    println!("Test signal: {} samples (10 Hz sine wave)\n", signal.len());

    // Example 1: Token Dropout
    println!("1. Token Dropout (Regularization)");
    println!("----------------------------------");

    let dropout_config = TokenDropoutConfig {
        dropout_rate: 0.2,     // Drop 20% of tokens
        fill_value: 0.0,       // Replace with zeros
        scale_remaining: true, // Scale remaining by 1/(1-p)
    };

    println!("Dropout rate: {}%", dropout_config.dropout_rate * 100.0);
    println!("Fill value: {}", dropout_config.fill_value);
    println!(
        "Inverted dropout scaling: {}",
        dropout_config.scale_remaining
    );

    // Apply dropout in training mode
    let tokens_with_dropout = apply_token_dropout(&signal, &dropout_config, true)?;

    let dropout_count = tokens_with_dropout
        .iter()
        .filter(|&&x| x == dropout_config.fill_value)
        .count();

    println!("\nOriginal signal mean: {:.4}", signal.mean().unwrap());
    println!(
        "After dropout mean: {:.4}",
        tokens_with_dropout.mean().unwrap()
    );
    println!(
        "Tokens dropped: {} (expected ~{})",
        dropout_count,
        (signal.len() as f32 * dropout_config.dropout_rate) as usize
    );
    println!("Dropout prevents overfitting during training\n");

    // Example 2: Jitter Injection
    println!("2. Jitter Injection (Robustness Testing)");
    println!("----------------------------------------");

    let jitter_config = JitterConfig {
        noise_std: 0.05,           // Standard deviation of Gaussian noise
        apply_at_inference: false, // Don't apply during inference
        target_snr_db: None,       // Use fixed std instead of SNR
    };

    println!("Noise std: {}", jitter_config.noise_std);

    let signal_with_jitter = add_jitter(&signal, &jitter_config, true)?;

    let noise: Array1<f32> = &signal_with_jitter - &signal;
    let noise_std = noise.std(0.0);
    println!("Actual noise std: {:.4}", noise_std);

    // SNR-based jitter
    let snr_jitter_config = JitterConfig::with_snr(20.0);

    println!("\nSNR-based jitter:");
    println!(
        "Target SNR: {} dB",
        snr_jitter_config.target_snr_db.unwrap()
    );

    let signal_with_snr_jitter = add_jitter(&signal, &snr_jitter_config, true)?;
    let snr = compute_snr(&signal, &signal_with_snr_jitter);
    println!("Actual SNR: {:.2} dB", snr);
    println!("Jitter helps test model robustness to noisy inputs\n");

    // Example 3: Temporal Coherence
    println!("3. Temporal Coherence Constraints");
    println!("----------------------------------");

    // Create a noisy signal
    let noisy_signal = add_jitter(&signal, &jitter_config, true)?;

    // EMA smoothing
    let ema_config = TemporalCoherenceConfig {
        filter_type: TemporalFilterType::ExponentialMovingAverage,
        smoothness: 0.9, // High smoothness
        window_size: 0,  // Not used for EMA
    };

    let ema_smoothed = apply_temporal_coherence(&noisy_signal, &ema_config)?;
    let ema_mse = compute_mse(&signal, &ema_smoothed);
    println!("EMA smoothing (α=0.9):");
    println!("  MSE vs original: {:.6}", ema_mse);

    // SMA smoothing
    let sma_config = TemporalCoherenceConfig {
        filter_type: TemporalFilterType::SimpleMovingAverage,
        smoothness: 0.0, // Not used for SMA
        window_size: 5,  // 5-sample window
    };

    let sma_smoothed = apply_temporal_coherence(&noisy_signal, &sma_config)?;
    let sma_mse = compute_mse(&signal, &sma_smoothed);
    println!("SMA smoothing (window=5):");
    println!("  MSE vs original: {:.6}", sma_mse);

    // Gaussian smoothing
    let gaussian_config = TemporalCoherenceConfig {
        filter_type: TemporalFilterType::GaussianWeighted,
        smoothness: 2.0, // Sigma for Gaussian kernel
        window_size: 11, // Window size (should be odd)
    };

    let gaussian_smoothed = apply_temporal_coherence(&noisy_signal, &gaussian_config)?;
    let gaussian_mse = compute_mse(&signal, &gaussian_smoothed);
    println!("Gaussian smoothing (σ=2.0, window=11):");
    println!("  MSE vs original: {:.6}", gaussian_mse);
    println!("\nTemporal coherence reduces high-frequency noise\n");

    // Example 4: Hierarchical Tokenization
    println!("4. Hierarchical Tokenization (Variable Bitrate)");
    println!("-----------------------------------------------");

    let embed_dim = 64;
    let hierarchical_config = HierarchicalConfig::exponential(
        256, // Base codebook size
        3,   // Number of levels
        0.5, // Decay factor (each level has 50% fewer codes)
    );

    let hierarchical_tokenizer =
        HierarchicalTokenizer::new(embed_dim, hierarchical_config.clone())?;

    println!("Hierarchy: {} levels", hierarchical_config.num_levels);
    println!("Codebook sizes: {:?}", hierarchical_config.codebook_sizes);
    println!(
        "Using residual coding: {}",
        hierarchical_config.use_residual
    );
    println!("Vector dimension: {}", embed_dim);

    // Create a signal of appropriate size
    let hierarchical_signal =
        Array1::from_vec((0..embed_dim).map(|i| (i as f32 * 0.1).sin()).collect());

    // Encode with all levels (highest quality)
    let encoded_3_levels = hierarchical_tokenizer.encode_with_levels(&hierarchical_signal, 3)?;
    let decoded_3_levels = hierarchical_tokenizer.decode_hierarchical(&encoded_3_levels)?;
    let mse_3_levels = compute_mse(&hierarchical_signal, &decoded_3_levels);

    println!("\n3 levels used:");
    println!("  Indices: {:?}", encoded_3_levels);
    println!("  MSE: {:.6}", mse_3_levels);

    // Encode with 2 levels (medium quality)
    let encoded_2_levels = hierarchical_tokenizer.encode_with_levels(&hierarchical_signal, 2)?;
    let decoded_2_levels = hierarchical_tokenizer.decode_hierarchical(&encoded_2_levels)?;
    let mse_2_levels = compute_mse(&hierarchical_signal, &decoded_2_levels);

    println!("\n2 levels used:");
    println!("  Indices: {:?}", encoded_2_levels);
    println!("  MSE: {:.6}", mse_2_levels);

    // Encode with 1 level (lowest quality)
    let encoded_1_level = hierarchical_tokenizer.encode_with_levels(&hierarchical_signal, 1)?;
    let decoded_1_level = hierarchical_tokenizer.decode_hierarchical(&encoded_1_level)?;
    let mse_1_level = compute_mse(&hierarchical_signal, &decoded_1_level);

    println!("\n1 level used:");
    println!("  Indices: {:?}", encoded_1_level);
    println!("  MSE: {:.6}", mse_1_level);

    println!("\nHierarchical tokenization enables variable bitrate coding");
    println!("More levels → Higher quality, Higher bitrate");
    println!("Fewer levels → Lower quality, Lower bitrate\n");

    // Example 5: Combined Usage
    println!("5. Combined Usage (Full Pipeline)");
    println!("----------------------------------");

    println!("Realistic ML training pipeline:");
    println!("1. Add jitter for data augmentation");
    println!("2. Apply token dropout for regularization");
    println!("3. Use temporal coherence for smoothing");

    // Create training data with augmentation
    let mut augmented_signal = signal.clone();

    // Step 1: Add jitter
    augmented_signal = add_jitter(&augmented_signal, &jitter_config, true)?;
    println!("\nAfter jitter: std = {:.4}", augmented_signal.std(0.0));

    // Step 2: Apply dropout
    augmented_signal = apply_token_dropout(&augmented_signal, &dropout_config, true)?;
    let dropout_pct = (augmented_signal.iter().filter(|&&x| x == 0.0).count() as f32
        / augmented_signal.len() as f32)
        * 100.0;
    println!("After dropout: {:.1}% zeros", dropout_pct);

    // Step 3: Smooth with temporal coherence
    augmented_signal = apply_temporal_coherence(&augmented_signal, &ema_config)?;
    println!("After smoothing: std = {:.4}", augmented_signal.std(0.0));

    let final_mse = compute_mse(&signal, &augmented_signal);
    println!("\nFinal MSE vs original: {:.6}", final_mse);
    println!("This pipeline improves model generalization and robustness");

    Ok(())
}

#[cfg(feature = "vqvae")]
fn compute_mse(signal: &Array1<f32>, reconstructed: &Array1<f32>) -> f32 {
    signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / signal.len() as f32
}

#[cfg(feature = "vqvae")]
fn compute_snr(signal: &Array1<f32>, noisy: &Array1<f32>) -> f32 {
    let signal_power: f32 = signal.iter().map(|x| x.powi(2)).sum();
    let noise_power: f32 = signal
        .iter()
        .zip(noisy.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    10.0 * (signal_power / noise_power).log10()
}

#[cfg(not(feature = "vqvae"))]
fn main() {
    println!("This example requires the 'vqvae' feature.");
    println!("Run with: cargo run --example advanced_features --features vqvae");
}
