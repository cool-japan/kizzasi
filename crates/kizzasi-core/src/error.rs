//! Error types for kizzasi-core

#[cfg(not(feature = "std"))]
use alloc::string::String;

use scirs2_core::ndarray::ShapeError;
use thiserror::Error;

/// Result type alias for core operations
pub type CoreResult<T> = Result<T, CoreError>;

/// Errors that can occur in the core SSM engine
#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Model not initialized")]
    NotInitialized,

    #[error("Weight loading failed: {0}")]
    WeightLoadError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("{0}")]
    Generic(String),

    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    #[error("Shape error: {0}")]
    ShapeError(#[from] ShapeError),
}
