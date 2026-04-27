//! Vector Quantized Variational AutoEncoder (VQ-VAE)
//!
//! VQ-VAE provides discrete representation learning through a learned codebook
//! of embedding vectors. It's widely used in neural audio codecs like SoundStream,
//! Encodec, and Jukebox.
//!
//! ## Algorithm
//!
//! 1. **Encoding**: Map input to continuous latent space
//! 2. **Quantization**: Replace each latent vector with nearest codebook entry
//! 3. **Decoding**: Map quantized latents back to signal space
//!
//! ## Training
//!
//! - **Codebook Loss**: Pulls codebook entries toward encoder outputs
//! - **Commitment Loss**: Encourages encoder to commit to codebook entries
//! - **Straight-Through Estimator**: Passes gradients through quantization
//!
//! ## References
//!
//! - van den Oord et al., "Neural Discrete Representation Learning" (2017)
//! - Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2" (2019)
//!
//! ## Module Layout
//!
//! The implementation is split across:
//! - [`vqvae_core`](super::vqvae_core): Core quantization primitives
//!   (VectorQuantizer, VQVAETokenizer, ResidualVQ, RVQVAETokenizer, ProductQuantizer)

pub use crate::vqvae_core::{
    BatchQuantizeResult, ProductQuantizer, ProductQuantizerConfig, RVQVAETokenizer, ResidualVQ,
    VQConfig, VQVAETokenizer, VectorQuantizer,
};
