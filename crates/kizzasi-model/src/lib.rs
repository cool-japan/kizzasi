//! # kizzasi-model
//!
//! Model architectures for Kizzasi AGSP (Autoregressive General-Purpose Signal Predictor).
//!
//! This crate implements various State Space Model architectures optimized for
//! continuous signal prediction with O(1) inference step complexity:
//!
//! - **Mamba/Mamba2**: Selective State Space Models with input-dependent dynamics
//! - **RWKV**: Linear attention with time-mixing and channel-mixing
//! - **S4/S4D**: Structured State Space Models with diagonal state matrices
//! - **Transformer**: Standard attention for comparison (O(N) per step)
//!
//! ## COOLJAPAN Ecosystem
//!
//! This crate follows KIZZASI_POLICY.md and uses `scirs2-core` for all
//! array and numerical operations.
//!
//! ## Architecture Philosophy
//!
//! As described in the AGSP concept, these models treat all signals
//! (audio, video, sensors, actions) as equivalent tokenized sequences,
//! enabling cross-modal prediction and world model construction.

#[cfg(feature = "hf-hub")]
pub mod hf_hub;
#[cfg(feature = "hf-hub")]
pub use hf_hub::{load_from_hub, HfHubClient, HfHubConfig, HfModelInfo};

pub mod arch_search;
pub use arch_search::{
    search_best_arch, ArchCandidate, ArchSearchConfig, ArchSearchResult, ArchSearchSpace,
    EvolutionarySearcher, GridSearcher, RandomArchSearcher,
};

pub mod backprop;
pub mod backprop_ssm;
pub mod batch;
pub(crate) mod binary_io;
pub mod blas_ops;
pub mod cache_friendly;
pub mod checkpoint;
pub mod compression;
pub mod curriculum;
pub mod distributed;
pub mod dynamic_quantization;
pub mod early_exit;
mod error;
pub mod factory;
pub mod flash_linear_attn;
pub mod gguf;
pub(crate) mod gguf_dequant;
pub mod gradient_checkpoint;
pub mod huggingface;
pub mod huggingface_loader;
pub mod incremental_loader;
pub mod loader;
pub mod lora;
pub mod mixed_precision;
pub mod moe;
pub mod onnx_export;
pub mod parallel_multihead;
pub mod profiling;
pub mod prune;
pub mod pytorch_compat;
pub mod quantization;
pub mod quantize;
pub mod simd_ops;
pub mod speculative;
pub mod state_io;
pub mod training;
pub mod training_loop;
pub mod visualization;

#[cfg(feature = "mamba")]
pub mod mamba;

#[cfg(feature = "mamba")]
pub mod mamba2;

pub mod interpretability;

pub mod rwkv;

pub mod rwkv5;

pub mod rwkv7;

pub mod s4;

pub mod s5;

pub mod h3;

pub mod hybrid;

pub mod multimodal;

pub mod neural_ode;

pub mod spiking;

pub mod temporal_multiscale;

pub mod transformer;

pub mod tensorlogic_bridge;
pub use tensorlogic_bridge::{compile_constraints, constraint_from_tl_expr};

pub use error::{ModelError, ModelResult};

// Re-export backward pass / autograd types
pub use backprop::{
    layer_norm_backward, linear_backward, silu_backward, softmax_backward, GradAccumulator,
    GradientTape, SsmBackward, SsmGradients, Tensor,
};
pub use gguf::{GgufFile, GgufInspection, GgufMetaValue, GgufQuantType, GgufTensorInfo};
pub use incremental_loader::{
    GgufFileSource, IncrementalModelLoader, SafeTensorsSource, WeightSource,
};
pub use loader::{ModelLoader, TensorInfo, WeightLoader};
pub use lora::{LoraAdapter, LoraAdapterSummary, LoraConfig, LoraLinear, QLoraLinear};
pub use multimodal::{
    FusionStrategy, Modality, ModalityAligner, MultiModalConfig, MultiModalModel,
};
pub use neural_ode::{
    AugmentedNeuralOde, NeuralOdeConfig, NeuralOdeModel, OdeIntegrator, OdeSolver,
};
pub use spiking::{
    LifLayer, MembranePotential, ResetMode, SpikingConfig, SpikingNeuralNetwork, StdpConfig,
};
pub use temporal_multiscale::{MultiScaleConfig, MultiScaleModel, ScaleFusion};

// Re-export training loop types
pub use training_loop::{
    AdamOptimizer, ArrayDataProvider, ConstantScheduler, DataProvider, ExponentialScheduler,
    LrScheduler, Optimizer, SgdOptimizer, StepDecayScheduler, TrainingCallback, TrainingConfig,
    TrainingLoop, TrainingResult,
};

// Re-export distributed training types
pub use distributed::{
    average_gradients, partition_indices, run_parallel_workers, sgd_step, CommBackend,
    DataParallelModel, DistributedConfig, GradientBuffer, GradientStrategy, GradientSync,
    LocalGradientSync, SharedGradientStore, ThreadedGradientSync,
};

// Re-export curriculum learning types
pub use curriculum::{CurriculumDataProvider, CurriculumScheduler, CurriculumStrategy};

// Re-export gradient checkpointing types
pub use gradient_checkpoint::{ActivationCheckpointer, CheckpointConfig};

// Re-export speculative decoding types
pub use speculative::{SpeculativeDecoder, SpeculativeResult};

// Re-export adaptive early exit types
pub use early_exit::{AdaptiveComputation, EarlyExitConfig, ExitCriterion, ExitStats};

// Re-export BLAS operations for convenience
pub use blas_ops::{
    axpy, batch_matmul_vec, dot, matmul_mat, matmul_vec, norm_frobenius, norm_l2, transpose,
    BlasConfig,
};

// Re-export profiling utilities
pub use profiling::{
    BottleneckInfo, BottleneckSeverity, ComprehensiveComparison, ComprehensiveProfiler,
    ModelBottleneckAnalysis,
};

// Re-export core types
pub use interpretability::{
    ActivationStats, CompressionAnalysis, GatingAnalysis, InterpretabilityReport, LayerProbe,
    SensitivityAnalyzer, StateTrajectory,
};
// Re-export visualization types
pub use visualization::{
    matrix_to_csv, signal_to_svg_sparkline, ActivationHistogram, GatingPatternRecorder,
    PhasePortrait,
};

// Re-export new compression types
pub use compression::{CompressionReport, LowRankApprox, MagnitudePruner, StructuredPruner};

// Re-export full state I/O types
pub use state_io::{decode_f32_slice, encode_f32_slice, ModelSnapshot};

pub use kizzasi_core::{CoreResult, HiddenState, SignalPredictor};
pub use rwkv5::{Rwkv5Config, Rwkv5Model, Rwkv5State};
pub use rwkv7::{Rwkv7Config, Rwkv7Model, Rwkv7State, Rwkv7TimeMixing};
pub use scirs2_core::ndarray::{Array1, Array2};

/// Trait for model architectures that support autoregressive prediction
pub trait AutoregressiveModel: SignalPredictor + Send {
    /// Get the model's hidden dimension
    fn hidden_dim(&self) -> usize;

    /// Get the model's state dimension (for SSMs)
    fn state_dim(&self) -> usize;

    /// Get number of layers
    fn num_layers(&self) -> usize;

    /// Get model type identifier
    fn model_type(&self) -> ModelType;

    /// Get current hidden states for all layers
    fn get_states(&self) -> Vec<HiddenState>;

    /// Set hidden states for all layers
    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()>;

    /// Load weights from a JSON file (`HashMap<String, Vec<f32>>` format).
    ///
    /// Override this method in model implementations that support weight loading.
    /// The default implementation returns an error indicating the model does not
    /// support JSON weight loading.
    fn load_weights_json(&mut self, _path: &std::path::Path) -> ModelResult<()> {
        Err(ModelError::unsupported_operation(
            "load_weights_json",
            format!(
                "{} (model does not implement JSON weight loading)",
                std::any::type_name::<Self>()
            ),
        ))
    }

    /// Save weights to a JSON file (`HashMap<String, Vec<f32>>` format).
    ///
    /// Override this method in model implementations that support weight saving.
    /// The default implementation returns an error indicating the model does not
    /// support JSON weight saving.
    fn save_weights_json(&self, _path: &std::path::Path) -> ModelResult<()> {
        Err(ModelError::unsupported_operation(
            "save_weights_json",
            format!(
                "{} (model does not implement JSON weight saving)",
                std::any::type_name::<Self>()
            ),
        ))
    }
}

/// Enumeration of supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModelType {
    /// Mamba: Selective State Space Model
    Mamba,
    /// Mamba2: Enhanced selective SSM with SSD
    Mamba2,
    /// RWKV: Linear attention with time-mixing (v6)
    Rwkv,
    /// RWKV v5: Multi-head WKV with static time decay
    Rwkv5,
    /// S4: Structured State Space Model
    S4,
    /// S4D: S4 with diagonal state matrix
    S4D,
    /// Standard Transformer (for comparison)
    Transformer,
    /// Neural ODE: Continuous-time dynamics via ODE solver
    NeuralOde,
    /// Multi-modal fusion model
    MultiModal,
    /// Spiking Neural Network: biologically-inspired LIF neurons
    Snn,
    /// Multi-Scale Temporal Model: multiple temporal resolutions
    MultiScale,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Mamba => write!(f, "Mamba"),
            ModelType::Mamba2 => write!(f, "Mamba2"),
            ModelType::Rwkv => write!(f, "RWKV"),
            ModelType::Rwkv5 => write!(f, "RWKV5"),
            ModelType::S4 => write!(f, "S4"),
            ModelType::S4D => write!(f, "S4D"),
            ModelType::Transformer => write!(f, "Transformer"),
            ModelType::NeuralOde => write!(f, "NeuralODE"),
            ModelType::MultiModal => write!(f, "MultiModal"),
            ModelType::Snn => write!(f, "SNN"),
            ModelType::MultiScale => write!(f, "MultiScale"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_display() {
        assert_eq!(format!("{}", ModelType::Mamba2), "Mamba2");
        assert_eq!(format!("{}", ModelType::Rwkv), "RWKV");
    }
}
