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

pub mod batch;
pub mod blas_ops;
pub mod cache_friendly;
pub mod checkpoint;
pub mod compression;
pub mod dynamic_quantization;
mod error;
pub mod factory;
pub mod huggingface;
pub mod huggingface_loader;
pub mod loader;
pub mod mixed_precision;
pub mod moe;
pub mod parallel_multihead;
pub mod profiling;
pub mod pytorch_compat;
pub mod quantization;
pub mod simd_ops;
pub mod training;

#[cfg(feature = "mamba")]
pub mod mamba;

#[cfg(feature = "mamba")]
pub mod mamba2;

pub mod rwkv;

pub mod rwkv7;

pub mod s4;

pub mod s5;

pub mod h3;

pub mod hybrid;

pub mod transformer;

pub use error::{ModelError, ModelResult};
pub use loader::{ModelLoader, TensorInfo, WeightLoader};

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
pub use kizzasi_core::{CoreResult, HiddenState, SignalPredictor};
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
}

/// Enumeration of supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModelType {
    /// Mamba: Selective State Space Model
    Mamba,
    /// Mamba2: Enhanced selective SSM with SSD
    Mamba2,
    /// RWKV: Linear attention with time-mixing
    Rwkv,
    /// S4: Structured State Space Model
    S4,
    /// S4D: S4 with diagonal state matrix
    S4D,
    /// Standard Transformer (for comparison)
    Transformer,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Mamba => write!(f, "Mamba"),
            ModelType::Mamba2 => write!(f, "Mamba2"),
            ModelType::Rwkv => write!(f, "RWKV"),
            ModelType::S4 => write!(f, "S4"),
            ModelType::S4D => write!(f, "S4D"),
            ModelType::Transformer => write!(f, "Transformer"),
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
