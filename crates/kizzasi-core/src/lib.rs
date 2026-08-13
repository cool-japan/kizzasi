//! # kizzasi-core
//!
//! Core SSM (State Space Model) engine for Kizzasi AGSP.
//!
//! Implements linear-time State Space Models (Mamba/S4/RWKV) for efficient
//! processing of continuous signal streams with O(1) inference step complexity.
//!
//! ## COOLJAPAN Ecosystem
//!
//! This crate follows the KIZZASI_POLICY.md and uses `scirs2-core` for all
//! array and numerical operations.

// NOTE on `no_std`: this crate previously declared
// `#![cfg_attr(not(feature = "std"), no_std)]` here, advertised as "Core SSM
// works without the standard library" in the README. That claim did not
// hold: `candle-core`, `candle-nn`, `safetensors`, `serde_json` and `chrono`
// are unconditional (non-optional) std-only dependencies in Cargo.toml, and
// dozens of modules (simd.rs, parallel.rs, pool.rs, profiling.rs, metrics.rs,
// weights.rs, gpu_utils.rs, dataloader.rs, training_loop.rs, ...) use `std::`
// directly with no `#[cfg]` guard, so `cargo build --no-default-features`
// never actually linked. Making the claim true would mean feature-gating
// those std dependencies (removing weight loading, PyTorch checkpoint
// conversion, and most of the training stack behind `std`) and auditing
// every module for std usage -- out of proportion for this pass, and the
// crate is squarely std-oriented in practice. The `std` feature is kept
// (default-on) as an inert compatibility flag since downstream `Cargo.toml`s
// may already reference it; genuine `no_std` embedded support lives in the
// self-contained `embedded_alloc` and `fixed_point` modules, which do not
// depend on the rest of the crate and can be used standalone.

pub mod attention;
mod config;
pub mod conv;
pub mod dataloader;
pub mod device;
pub mod efficient_attention;
pub mod embedded_alloc;
mod embedding;
mod error;
pub mod fixed_point;
pub mod flash_attention;
pub mod gpu_utils;
pub mod h3;
pub mod kernel_fusion;
pub mod lora;
pub mod mamba2;
pub mod metrics;
pub mod nn;
pub mod numerics;
pub mod optimizations;
pub mod optimizer;
pub mod parallel;
pub mod pool;
pub mod profiling;
pub mod pruning;
pub mod pytorch_compat;
pub mod quantization;
pub mod retnet;
pub mod rwkv7;
pub mod s4d;
pub mod s5;
pub mod scan;
pub mod scheduler;
pub mod sequences;
pub mod simd;
pub mod simd_aarch64;
pub mod simd_avx512;
pub mod simd_neon;
pub use simd_aarch64::{
    add_f32, dot_product_f32, l2_norm_f32, normalize_f32, relu_f32, rms_norm_f32, scale_f32,
    softmax_f32, ssm_state_update_f32,
};
mod ssm;
pub mod ssm_backend;
mod state;
pub mod training;
pub mod training_checkpoint;
pub mod training_core;
pub mod training_loop;
pub mod weights;

pub use attention::{GatedLinearAttention, MultiHeadSSMAttention, MultiHeadSSMConfig};
pub use config::{KizzasiConfig, ModelType};
pub use conv::{CausalConv1d, DepthwiseCausalConv1d, DilatedCausalConv1d, DilatedStack, ShortConv};
pub use dataloader::{
    BatchIterator, DataLoaderConfig, TimeSeriesAugmentation, TimeSeriesDataLoader,
};
pub use device::{
    get_best_device, is_metal_available, list_devices, DeviceConfig, DeviceInfo, DeviceType,
};
pub use efficient_attention::{
    EfficientAttentionConfig, EfficientMultiHeadAttention, FusedAttentionKernel,
};
pub use embedded_alloc::{BumpAllocator, EmbeddedAllocator, FixedPool, StackAllocator, StackGuard};
pub use embedding::ContinuousEmbedding;
pub use error::{CoreError, CoreResult};
pub use flash_attention::{flash_attention_fused, FlashAttention, FlashAttentionConfig};
pub use gpu_utils::{GPUMemoryPool, MemoryStats, TensorPrefetch, TensorTransfer, TransferBatch};
pub use h3::{DiagonalSSM, H3Config, H3Layer, H3Model, ShiftSSM};
pub use kernel_fusion::{
    fused_ffn_gelu, fused_layernorm_gelu, fused_layernorm_silu, fused_linear_activation,
    fused_multihead_output, fused_qkv_projection, fused_quantize_dequantize, fused_softmax_attend,
    fused_ssm_step,
};
pub use lora::{LoRAAdapter, LoRAConfig, LoRALayer};
pub use mamba2::{Mamba2Config, Mamba2Layer, Mamba2Model};
pub use metrics::{MetricsLogger, MetricsSummary, TrainingMetrics};
pub use nn::{
    gelu, gelu_fast, layer_norm, leaky_relu, log_softmax, relu, rms_norm, sigmoid, silu, softmax,
    tanh, Activation, ActivationType, GatedLinearUnit, LayerNorm, NormType,
};
pub use optimizations::{
    acquire_workspace, ilp, prefetch, release_workspace, CacheAligned, DiscretizationCache,
    SSMWorkspace, WorkspaceGuard,
};
pub use optimizer::KizzasiAdamW;
pub use parallel::{BatchProcessor, ParallelConfig};
pub use pool::{ArrayPool, MultiArrayPool, PoolStats, PooledArray};
pub use profiling::{
    CounterStats, MemoryProfiler, PerfCounter, ProfilerMemoryStats, ProfilingSession, ScopeTimer,
    Timer,
};
pub use pruning::{
    GradientPruner, PruningConfig, PruningGranularity, PruningMask, PruningStrategy,
    StructuredPruner,
};
pub use pytorch_compat::{
    detect_checkpoint_architecture, PyTorchCheckpoint, PyTorchConverter, WeightMapping,
};
pub use quantization::{
    DynamicQuantizer, QuantizationParams, QuantizationScheme, QuantizationType, QuantizedTensor,
};
pub use retnet::{MultiScaleRetention, RetNetConfig, RetNetLayer, RetNetModel};
pub use rwkv7::{ChannelMixing, RWKV7Config, RWKV7Layer, RWKV7Model, TimeMixing};
pub use s4d::{S4DConfig, S4DLayer, S4DModel};
pub use s5::{S5Config, S5Layer, S5Model};
pub use scan::{
    parallel_scan, parallel_ssm_batch, parallel_ssm_scan, segmented_scan, AssociativeOp,
    SSMElement, SSMScanOp,
};
pub use scheduler::{
    ConstantScheduler, CosineScheduler, ExponentialScheduler, LRScheduler, LinearScheduler,
    OneCycleScheduler, PolynomialScheduler, StepScheduler,
};
pub use sequences::{
    apply_mask, masked_mean, masked_sum, pad_sequences, PackedSequence, PaddingStrategy,
    SequenceMask,
};
pub use ssm::{SelectiveSSM, StateSpaceModel};
pub use ssm_backend::{default_backend, CpuSsmBackend, SsmBackend};
pub use state::HiddenState;
pub use training::{
    CheckpointMetadata, ConstraintLoss, Loss, MixedPrecision, SchedulerType, TrainableSSM, Trainer,
    TrainingConfig,
};
pub use weights::{WeightFormat, WeightLoadConfig, WeightLoader, WeightPruner};

// Re-export scirs2-core types for convenience
pub use scirs2_core::ndarray::Array1;

/// Core trait for autoregressive signal prediction
pub trait SignalPredictor {
    /// Update hidden state and predict the next signal vector
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>>;

    /// Reset the hidden state to initial values
    fn reset(&mut self);

    /// Get the current context window size
    fn context_window(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = KizzasiConfig::new()
            .model_type(ModelType::Mamba2)
            .context_window(8192)
            .hidden_dim(256);

        assert_eq!(config.get_context_window(), 8192);
        assert_eq!(config.get_hidden_dim(), 256);
    }
}
