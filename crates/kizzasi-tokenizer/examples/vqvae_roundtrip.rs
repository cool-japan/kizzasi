//! VQ-VAE encode/decode roundtrip example.
//!
//! Demonstrates training a small VQ codebook on a synthetic 1D dataset via the
//! exponential-moving-average (EMA) update rule, encoding the dataset to
//! discrete indices, decoding back to an approximation, and reporting the
//! mean-squared error as a function of codebook size.
//!
//! The example exercises [`VectorQuantizer`]'s public API:
//!
//! - `VectorQuantizer::new(VQConfig)`         – construct
//! - `initialize_from_data(&[Array1<f32>])`   – k-means++ init
//! - `quantize_batch(&[Array1<f32>])`         – encode-and-snap
//! - `update_ema(&outputs, &indices)`         – EMA training step
//! - `get_codebook_entry(idx)`                – decode an index
//! - `usage_stats()`                          – inspect codebook utilisation
//!
//! Run with:
//! ```bash
//! cargo run --example vqvae_roundtrip -p kizzasi-tokenizer --features vqvae
//! ```

#[cfg(feature = "vqvae")]
use kizzasi_tokenizer::vqvae_core::vector_quantizer::{VQConfig, VectorQuantizer};
#[cfg(feature = "vqvae")]
use scirs2_core::ndarray::Array1;
#[cfg(feature = "vqvae")]
use scirs2_core::random::{rngs::StdRng, Random};

#[cfg(feature = "vqvae")]
fn synthesize_dataset(n_samples: usize, dim: usize, rng: &mut Random<StdRng>) -> Vec<Array1<f32>> {
    // Mixture of a handful of Gaussian-ish clusters so that quantisation
    // produces something more interesting than uniform noise.
    let num_clusters = 8usize;
    let mut centres: Vec<Array1<f32>> = Vec::with_capacity(num_clusters);
    for _ in 0..num_clusters {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        centres.push(Array1::from(v));
    }

    let mut data = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let centre = &centres[i % num_clusters];
        let noise: Vec<f32> = (0..dim).map(|_| rng.gen_range(-0.05..0.05)).collect();
        let mut sample = centre.clone();
        for (s, n) in sample.iter_mut().zip(noise.iter()) {
            *s += *n;
        }
        data.push(sample);
    }
    data
}

#[cfg(feature = "vqvae")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VQ-VAE Roundtrip Demo ===\n");

    let n_samples = 256usize;
    let dim = 8usize;

    let mut rng = Random::seed(42);
    let data = synthesize_dataset(n_samples, dim, &mut rng);

    println!(
        "Synthetic dataset: {} samples of dimension {} (8 latent clusters)\n",
        n_samples, dim
    );

    // Sweep codebook sizes to show the rate-distortion trade-off.
    for &k in &[4usize, 16, 64] {
        let config = VQConfig {
            codebook_size: k,
            embed_dim: dim,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        };

        let mut vq = VectorQuantizer::new(config);
        vq.initialize_from_data(&data)?;

        // A handful of EMA passes — enough to settle the codebook on this toy data.
        for _ in 0..50 {
            let (indices, _) = vq.quantize_batch(&data)?;
            vq.update_ema(&data, &indices)?;
        }

        // Encode + decode roundtrip via the trained codebook.
        let (indices, _) = vq.quantize_batch(&data)?;
        let mut recon: Vec<Array1<f32>> = Vec::with_capacity(indices.len());
        for &idx in &indices {
            recon.push(vq.get_codebook_entry(idx)?);
        }

        let mut sq_err = 0.0_f32;
        for (orig, rec) in data.iter().zip(recon.iter()) {
            for (a, b) in orig.iter().zip(rec.iter()) {
                sq_err += (a - b).powi(2);
            }
        }
        let mse = sq_err / (n_samples * dim) as f32;
        let (total_uses, used_codes, util) = vq.usage_stats();

        println!(
            "Codebook size {:>3}: MSE = {:.6} | used {:>3}/{:<3} codes ({:>5.1}% util) | total assignments {}",
            k,
            mse,
            used_codes,
            k,
            util * 100.0,
            total_uses
        );
    }

    println!("\nLarger codebooks reduce roundtrip MSE at the cost of more bits/code.");
    Ok(())
}

#[cfg(not(feature = "vqvae"))]
fn main() {
    println!("This example requires the 'vqvae' feature.");
    println!("Run with: cargo run --example vqvae_roundtrip -p kizzasi-tokenizer --features vqvae");
}
