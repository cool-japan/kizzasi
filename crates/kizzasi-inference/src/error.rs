//! Error types for kizzasi-inference

use thiserror::Error;

/// Result type alias for inference operations
pub type InferenceResult<T> = Result<T, InferenceError>;

/// Errors that can occur during inference
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Engine not initialized")]
    NotInitialized,

    #[error("Pipeline configuration error: {0}")]
    PipelineConfig(String),

    #[error("Tokenization failed: {0}")]
    TokenizationError(String),

    #[error("Model forward pass failed: {0}")]
    ForwardError(String),

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Constraint enforcement failed: {0}")]
    ConstraintError(String),

    #[error("Context overflow: max {max} steps, requested {requested}")]
    ContextOverflow { max: usize, requested: usize },

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Core error: {0}")]
    CoreError(#[from] kizzasi_core::CoreError),

    #[error("Model error: {0}")]
    ModelError(#[from] kizzasi_model::ModelError),

    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] kizzasi_tokenizer::TokenizerError),

    #[error("Logic error: {0}")]
    LogicError(#[from] kizzasi_logic::LogicError),

    #[error("Serialization/Deserialization error: {0}")]
    SerializationError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Initialization error: {0}")]
    InitializationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Lock error: {0}")]
    LockError(String),
}
