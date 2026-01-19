//! VQ-VAE Tokenizer Example
//!
//! This example demonstrates the usage of Vector Quantized Variational AutoEncoder (VQ-VAE)
//! for learned discrete representations of continuous signals.
//!
//! Run with:
//! ```bash
//! cargo run --example vqvae_tokenizer --features vqvae
//! ```

#[cfg(feature = "vqvae")]
use kizzasi_tokenizer::{SignalTokenizer, VQConfig, VQVAETokenizer};

#[cfg(feature = "vqvae")]
use scirs2_core::ndarray::{s, Array1};

#[cfg(feature = "vqvae")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VQ-VAE Tokenizer Example ===\n");

    // Example 1: Basic VQ-VAE usage
    println!("1. Basic VQ-VAE Tokenization");
    println!("----------------------------");

    // Configuration for VQ-VAE
    let config = VQConfig {
        codebook_size: 256,    // Number of discrete codes
        embed_dim: 64,         // Dimension of each code vector
        commitment_beta: 0.25, // Weight for commitment loss
        ema_decay: 0.99,       // Exponential moving average decay for codebook updates
        epsilon: 1e-5,         // Small constant for numerical stability
        use_ema: true,         // Use EMA updates (recommended for training)
    };

    println!("Configuration:");
    println!("  Codebook size: {}", config.codebook_size);
    println!("  Embedding dimension: {}", config.embed_dim);
    println!("  Commitment beta: {}", config.commitment_beta);
    println!("  EMA decay: {}", config.ema_decay);

    // Create VQ-VAE tokenizer
    let input_dim = 128; // Dimension of input signal
    let tokenizer = VQVAETokenizer::new(input_dim, config.clone());

    // Generate a test signal
    let signal = Array1::from_vec(
        (0..input_dim)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 16.0).sin())
            .collect(),
    );
    println!("\nOriginal signal shape: {}", signal.len());
    println!("First 10 samples: {:?}", &signal.slice(s![..10]).to_vec());

    // Encode the signal
    let encoded = tokenizer.encode(&signal)?;
    println!("\nEncoded embedding shape: {}", encoded.len());
    println!(
        "First 10 embedding values: {:?}",
        &encoded.slice(s![..10]).to_vec()
    );

    // Decode back
    let decoded = tokenizer.decode(&encoded)?;
    println!("\nDecoded signal shape: {}", decoded.len());
    println!(
        "First 10 decoded samples: {:?}",
        &decoded.slice(s![..10]).to_vec()
    );

    // Compute reconstruction error
    let mse: f32 = signal
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / signal.len() as f32;
    println!("Mean Squared Error: {:.6}", mse);

    // Example 2: Codebook usage statistics
    println!("\n2. Codebook Usage Analysis");
    println!("--------------------------");

    // Create another tokenizer for this example
    let tokenizer2 = VQVAETokenizer::new(input_dim, config);
    let n_signals = 100;

    for i in 0..n_signals {
        let signal = Array1::from_vec(
            (0..input_dim)
                .map(|j| ((i as f32 * 0.1 + j as f32) * 0.01).sin())
                .collect(),
        );

        let _encoded = tokenizer2.encode(&signal)?;

        // Get the quantized indices (this is a simplified version)
        // In practice, you'd need to expose the quantizer's indices
        // For now, we'll just encode many signals
    }

    println!("Encoded {} signals through VQ-VAE", n_signals);
    println!("This helps the codebook learn diverse representations");

    // Example 3: Comparing different codebook sizes
    println!("\n3. Effect of Codebook Size");
    println!("--------------------------");

    for codebook_size in [64, 128, 256, 512, 1024] {
        let config = VQConfig {
            codebook_size,
            embed_dim: 64,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        };

        let tokenizer = VQVAETokenizer::new(input_dim, config);

        // Encode and decode
        let encoded = tokenizer.encode(&signal)?;
        let decoded = tokenizer.decode(&encoded)?;

        // Compute error
        let mse: f32 = signal
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / signal.len() as f32;

        println!(
            "Codebook size {:4}: MSE = {:.6}, Bits/sample = {:.2}",
            codebook_size,
            mse,
            (codebook_size as f32).log2()
        );
    }

    // Example 4: EMA vs Non-EMA updates
    println!("\n4. EMA vs Non-EMA Codebook Updates");
    println!("-----------------------------------");

    let ema_config = VQConfig {
        codebook_size: 256,
        embed_dim: 64,
        commitment_beta: 0.25,
        ema_decay: 0.99,
        epsilon: 1e-5,
        use_ema: true,
    };

    let non_ema_config = VQConfig {
        codebook_size: 256,
        embed_dim: 64,
        commitment_beta: 0.25,
        ema_decay: 0.0, // Not used when use_ema is false
        epsilon: 1e-5,
        use_ema: false,
    };

    let ema_tokenizer = VQVAETokenizer::new(input_dim, ema_config);
    let non_ema_tokenizer = VQVAETokenizer::new(input_dim, non_ema_config);

    println!("EMA updates provide smoother codebook adaptation");
    println!("Non-EMA updates allow faster adaptation to new data");

    // Encode with both
    let ema_encoded = ema_tokenizer.encode(&signal)?;
    let non_ema_encoded = non_ema_tokenizer.encode(&signal)?;

    println!("\nEMA encoded shape: {}", ema_encoded.len());
    println!("Non-EMA encoded shape: {}", non_ema_encoded.len());

    Ok(())
}

#[cfg(not(feature = "vqvae"))]
fn main() {
    println!("This example requires the 'vqvae' feature.");
    println!("Run with: cargo run --example vqvae_tokenizer --features vqvae");
}
