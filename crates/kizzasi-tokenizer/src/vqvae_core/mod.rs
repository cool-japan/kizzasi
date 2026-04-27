//! VQ-VAE Core — VectorQuantizer, VQVAETokenizer, ResidualVQ, RVQVAETokenizer, ProductQuantizer
//!
//! This module contains the fundamental vector quantization building blocks:
//!
//! - [`VQConfig`] — configuration for vector quantization
//! - [`VectorQuantizer`] — single-stage VQ with learned codebook and EMA updates
//! - [`VQVAETokenizer`] — VQ-VAE encoder/decoder wrapper
//! - [`ResidualVQ`] — multi-stage residual vector quantization
//! - [`RVQVAETokenizer`] — RVQ-VAE encoder/decoder wrapper
//! - [`ProductQuantizerConfig`] / [`ProductQuantizer`] — product quantization

pub mod product_quantizer;
pub mod residual_vq;
pub mod vector_quantizer;
pub mod vqvae_tokenizer;

pub use product_quantizer::*;
pub use residual_vq::*;
pub use vector_quantizer::*;
pub use vqvae_tokenizer::*;

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that all public types from sub-modules are accessible through vqvae_core.
    #[test]
    fn test_vqvae_core_public_api_accessible() {
        // VQConfig + VectorQuantizer
        let config = VQConfig::default();
        let _vq = VectorQuantizer::new(config.clone());

        // VQVAETokenizer
        let _tok = VQVAETokenizer::new(16, config.clone());

        // ResidualVQ
        let _rvq = ResidualVQ::new(2, config.clone());

        // RVQVAETokenizer
        let _rvqvae = RVQVAETokenizer::new(16, 2, config);

        // ProductQuantizerConfig + ProductQuantizer + BatchQuantizeResult type alias
        let pq_config = ProductQuantizerConfig {
            num_subspaces: 2,
            codebook_size_per_subspace: 8,
            embed_dim: 16,
            ..Default::default()
        };
        let _pq = ProductQuantizer::new(pq_config).expect("ProductQuantizer creation failed");

        // BatchQuantizeResult is a type alias; assert the alias resolves correctly
        let _result: BatchQuantizeResult = (vec![], vec![]);
    }
}
