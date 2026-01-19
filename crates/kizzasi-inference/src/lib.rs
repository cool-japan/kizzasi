//! # kizzasi-inference
//!
//! Unified autoregressive inference engine for Kizzasi AGSP.
//!
//! This crate provides the core inference loop that combines:
//! - Signal tokenization (via kizzasi-tokenizer)
//! - Model forward pass (via kizzasi-model)
//! - Constraint enforcement (via kizzasi-logic)
//!
//! ## The Inference Pipeline
//!
//! ```text
//! Raw Signal → Tokenize → Model → Constrain → Decode → Output
//!     ↑                     ↓
//!     └──── Hidden State ───┘
//! ```
//!
//! ## Autoregressive Prediction
//!
//! As described in a.md, the AGSP predicts "the next value" based on
//! history, similar to how LLMs predict "the next token". This crate
//! implements:
//!
//! - Single-step prediction: `step(input) -> output`
//! - Multi-step rollout: `rollout(input, steps) -> [outputs]`
//! - Streaming inference: Real-time continuous prediction
//!
//! ## COOLJAPAN Ecosystem
//!
//! This crate follows KIZZASI_POLICY.md and coordinates between
//! all kizzasi-* crates.

mod batch;
mod checkpoint;
mod compression;
mod context;
mod engine;
mod ensemble;
mod error;
mod lora;
mod metrics;
mod multimodal;
mod pipeline;
mod pool;
mod precision;
mod registry;
mod sampling;
mod speculative;
pub mod temporal;

#[cfg(feature = "async")]
mod hotswap;

#[cfg(feature = "streaming")]
pub mod streaming;

#[cfg(any(feature = "websocket", feature = "mqtt", feature = "grpc"))]
pub mod adapters;

pub mod versioning;

pub use batch::{
    BatchConfig, BatchRequest, BatchResponse, BatchScheduler, Priority, SchedulerStats,
};
pub use checkpoint::{Checkpoint, CheckpointManager, CheckpointMetadata};
pub use compression::{CompressedState, CompressionMethod, StateCompressor};
pub use context::{ContextConfig, InferenceContext};
pub use engine::{EngineConfig, InferenceEngine, InferenceMode, ModelInfo};
pub use ensemble::{EnsembleBuilder, EnsembleConfig, EnsembleStrategy, ModelEnsemble};
pub use error::{InferenceError, InferenceResult};

#[cfg(feature = "async")]
pub use hotswap::{HotSwapManager, ModelInstance, SwapEvent, SwapStrategy};

pub use lora::{
    LoraAdapter, LoraAdapterBuilder, LoraAdapterLoader, LoraAdapterManager, LoraConfig,
};
pub use metrics::{InferenceMetrics, InferenceProfiler, MetricsSummary, ProfileBreakdown, Timer};
pub use multimodal::{
    FusionStrategy, ModalityConfig, ModalityPreprocessor, ModalityType, MultiModalPipeline,
    MultiModalPipelineBuilder,
};
pub use pipeline::{Pipeline, PipelineBuilder, PostprocessHook, PreprocessHook};
pub use pool::{BufferKey, PoolStats, PooledBuffer, TensorPool};
pub use precision::{
    ComputePrecision, PrecisionConfig, PrecisionConverter, PrecisionMode, PrecisionStats,
};
pub use registry::{ModelBuilder, ModelConfig, ModelRegistry};
pub use sampling::{
    AdaptiveRejectionSampler, Beam, BeamSearch, ConstrainedBeamSearch, ConstraintFn,
    CustomSamplingFn, FallbackStrategy, RejectionSampler, Sampler, SamplingConfig,
    SamplingStrategy,
};
pub use speculative::{SpeculativeConfig, SpeculativeDecoder};
pub use temporal::{LTLFormula, STLFormula, TemporalBound, TemporalConstraintEnforcer};
pub use versioning::{
    FallbackStrategy as VersionFallbackStrategy, HealthCheck, HealthStatus, ModelMetadata,
    ModelStats as VersionModelStats, ModelVersion, ModelVersionManager, VersioningConfig,
};

// Re-export core types
pub use kizzasi_core::SignalPredictor;
pub use kizzasi_model::{AutoregressiveModel, ModelType};
pub use kizzasi_tokenizer::SignalTokenizer;
pub use scirs2_core::ndarray::{Array1, Array2};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imports() {
        // Verify re-exports work
        let _: ModelType = ModelType::Mamba2;
    }
}
