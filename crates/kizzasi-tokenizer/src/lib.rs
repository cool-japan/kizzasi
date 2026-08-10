//! # kizzasi-tokenizer
//!
//! Signal quantization and tokenization for Kizzasi AGSP.
//!
//! This crate provides methods for converting continuous signals into
//! representations suitable for autoregressive prediction:
//!
//! - **Continuous Embedding**: Direct float-to-latent projection (no discretization)
//! - **VQ-VAE**: Vector Quantized embeddings with learned codebook
//! - **μ-law**: Logarithmic quantization for audio signals
//! - **Linear Quantization**: Simple uniform quantization
//!
//! ## AGSP Philosophy
//!
//! Unlike LLMs that tokenize text into discrete vocabulary indices,
//! AGSP models can work with continuous signals directly. However,
//! discretization can still be useful for:
//!
//! - Reducing model complexity
//! - Enabling cross-modal transfer
//! - Improving training stability
//!
//! ## COOLJAPAN Ecosystem
//!
//! This crate follows KIZZASI_POLICY.md and uses `scirs2-core` for all
//! array and numerical operations.

pub mod advanced_features;
pub mod advanced_quant;
pub mod batch;
pub mod compat;
pub mod cross_modal;
// continuous and gpu_quant require candle-core which is not wasm32-compatible
#[cfg(not(target_arch = "wasm32"))]
mod continuous;
pub mod domain_specific;
pub mod enhanced_multiscale;
pub mod entropy;
mod error;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu_quant;
pub mod metrics;
mod mulaw;
pub mod multi_speaker;
mod multiscale;
#[cfg(feature = "vqvae")]
pub mod neural_codec;
// persistence and serde_utils depend on continuous module (candle-core, not wasm32-safe)
pub mod peaq;
pub mod perceptual;
#[cfg(not(target_arch = "wasm32"))]
pub mod persistence;
pub mod pretraining;
pub mod profiling;
mod quantizer;
#[cfg(not(target_arch = "wasm32"))]
pub mod serde_utils;
pub mod simd_quant;
pub mod specialized;
pub mod transformer;
pub mod types;
pub mod utils;

#[cfg(feature = "wasm")]
pub mod wasm_bindings;

#[cfg(feature = "vqvae")]
pub mod vqvae_core;

#[cfg(feature = "vqvae")]
pub mod vqvae;

#[cfg(not(target_arch = "wasm32"))]
pub use continuous::{
    ContinuousTokenizer, ReconstructionMetrics, TrainableContinuousTokenizer, TrainingConfig,
};
pub use error::{TokenizerError, TokenizerResult};
pub use mulaw::MuLawCodec;
pub use multiscale::{
    MultiScaleTokenizer, PoolMethod, PyramidTokenizer, ScaleLevel, UpsampleMethod,
};
pub use quantizer::{LinearQuantizer, Quantizer};

// Re-export advanced quantizers
pub use advanced_quant::{
    AdaptiveQuantizer, DeadZoneQuantizer, EntropyConstrainedQuantizer, NonUniformQuantizer,
};

#[cfg(feature = "vqvae")]
pub use vqvae::{
    ProductQuantizer, ProductQuantizerConfig, RVQVAETokenizer, ResidualVQ, VQConfig,
    VQVAETokenizer, VectorQuantizer,
};

// Re-export batch types
pub use batch::{BatchTokenizer, StreamingTokenizer};

// Re-export entropy coding
pub use entropy::{
    compression_ratio, compute_frequencies, ArithmeticDecoder, ArithmeticEncoder,
    BitrateController, HuffmanDecoder, HuffmanEncoder, RangeDecoder, RangeEncoder,
};

// Re-export persistence types (not available on wasm32)
#[cfg(not(target_arch = "wasm32"))]
pub use persistence::{load_config, save_config, ModelCheckpoint, ModelMetadata, ModelVersion};

// Re-export PEAQ evaluator
pub use self::peaq::{OdgGrade, PeaqConfig, PeaqEvaluator, PeaqMovs, PeaqResult};

// Re-export perceptual quantizer types
pub use self::perceptual::{
    absolute_threshold_db, frequency_to_bark, BarkBands, PerceptualQuantizer,
};

// Re-export specialized tokenizers
pub use specialized::{
    DCTConfig, DCTTokenizer, FourierConfig, FourierTokenizer, KMeansConfig, KMeansTokenizer,
    WaveletConfig, WaveletFamily, WaveletTokenizer,
};

// Re-export advanced features
pub use advanced_features::{
    add_batch_jitter, add_jitter, apply_batch_token_dropout, apply_temporal_coherence,
    apply_token_dropout, HierarchicalConfig, HierarchicalTokenizer, JitterConfig,
    TemporalCoherenceConfig, TemporalFilterType, TokenDropoutConfig,
};

// Re-export compatibility types
pub use compat::{AudioMetadata, DType, ModelConfig, OnnxConfig, PyTorchCompat, TensorInfo};

// Re-export neural codec types
#[cfg(feature = "vqvae")]
pub use neural_codec::{NeuralCodec, NeuralCodecConfig};

// Re-export multi-speaker tokenizer
pub use self::multi_speaker::{
    MultiSpeakerConfig, MultiSpeakerToken, MultiSpeakerTokenizer, SpeakerCodebook,
};

// Re-export domain-specific tokenizers
pub use domain_specific::{
    EnvironmentalTokenizer, EnvironmentalTokenizerConfig, MusicTokenizer, MusicTokenizerConfig,
    SpeechTokenizer, SpeechTokenizerConfig,
};

// Re-export transformer tokenizer
pub use transformer::{
    FeedForward, LayerNorm, MultiHeadAttention, PositionalEncoding, TransformerConfig,
    TransformerEncoderLayer, TransformerTokenizer,
};

// Re-export pre-training utilities
pub use pretraining::{
    ContrastiveConfig, ContrastiveLearning, MSMConfig, MaskedSignalModeling, TemporalPrediction,
    TemporalPredictionConfig,
};

// Re-export profiling utilities
pub use profiling::{
    AllocationEvent, EventType, MemoryProfiler, MemorySnapshot, ProfileScope, ScopeStats,
    TimelineAnalyzer,
};

// Re-export cross-modal types
pub use cross_modal::{
    CrossModalAligner, CrossModalSequence, CrossModalToken, CrossModalTokenizer, ModalityKind,
    ModalityTokenizerConfig,
};

// Re-export core types
pub use scirs2_core::ndarray::{Array1, Array2};

/// Trait for signal tokenization
pub trait SignalTokenizer {
    /// Encode a continuous signal into tokens/embeddings
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>>;

    /// Decode tokens/embeddings back to continuous signal
    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>>;

    /// Get the embedding dimension
    fn embed_dim(&self) -> usize;

    /// Get the vocabulary size (for discrete tokenizers, 0 for continuous)
    fn vocab_size(&self) -> usize;
}

/// Configuration for tokenizer selection
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TokenizerType {
    /// Direct continuous embedding (no discretization)
    Continuous { embed_dim: usize },
    /// μ-law companding (8-bit or 16-bit)
    MuLaw { bits: u8 },
    /// Linear uniform quantization
    Linear { bits: u8, min: f32, max: f32 },
    /// Vector quantized (VQ-VAE style)
    VectorQuantized {
        codebook_size: usize,
        embed_dim: usize,
    },
    /// Multi-scale hierarchical tokenization
    MultiScale {
        embed_dim_per_level: usize,
        num_levels: usize,
    },
    /// Pyramid tokenization with residual encoding
    Pyramid {
        embed_dim_per_level: usize,
        num_levels: usize,
    },
}

impl Default for TokenizerType {
    fn default() -> Self {
        TokenizerType::Continuous { embed_dim: 256 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_type_default() {
        let t = TokenizerType::default();
        match t {
            TokenizerType::Continuous { embed_dim } => assert_eq!(embed_dim, 256),
            _ => panic!("Expected Continuous tokenizer"),
        }
    }
}
