//! Error types for kizzasi-model
//!
//! Comprehensive error handling with contextual information for debugging.

use thiserror::Error;

/// Result type alias for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Errors that can occur in model operations
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Invalid model configuration: {message}")]
    InvalidConfig { message: String },

    #[error("Dimension mismatch in {context}: expected {expected}, got {got}")]
    DimensionMismatch {
        context: String,
        expected: usize,
        got: usize,
    },

    #[error("Model not initialized: {details}")]
    NotInitialized { details: String },

    #[error("Weight loading failed for tensor '{tensor_name}': {reason}")]
    WeightLoadError { tensor_name: String, reason: String },

    #[error("Tensor not found: '{name}' in model '{model}'")]
    TensorNotFound { name: String, model: String },

    #[error("Load error in {context}: {message}")]
    LoadError { context: String, message: String },

    #[error("Forward pass error at layer {layer_idx}: {message}")]
    ForwardError { layer_idx: usize, message: String },

    #[error("State count mismatch for {model}: expected {expected} layers, got {got}")]
    StateCountMismatch {
        model: String,
        expected: usize,
        got: usize,
    },

    #[error("Invalid batch size: expected {expected}, got {got}")]
    InvalidBatchSize { expected: usize, got: usize },

    #[error("Numerical instability detected in {operation}: {details}")]
    NumericalInstability { operation: String, details: String },

    #[error("Unsupported operation: {operation} for model type {model_type}")]
    UnsupportedOperation {
        operation: String,
        model_type: String,
    },

    #[error("Quantization error: {message}")]
    QuantizationError { message: String },

    #[error("Memory allocation failed: requested {bytes} bytes for {purpose}")]
    AllocationError { bytes: usize, purpose: String },

    #[error("Index out of bounds: index {index} exceeds limit {limit} in {context}")]
    IndexOutOfBounds {
        index: usize,
        limit: usize,
        context: String,
    },

    #[error("Core error: {0}")]
    CoreError(#[from] kizzasi_core::CoreError),

    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

impl ModelError {
    /// Create an invalid config error (backward compatible)
    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }

    /// Create a dimension mismatch error with context
    pub fn dimension_mismatch(context: impl Into<String>, expected: usize, got: usize) -> Self {
        Self::DimensionMismatch {
            context: context.into(),
            expected,
            got,
        }
    }

    /// Create a not initialized error (backward compatible)
    pub fn not_initialized(details: impl Into<String>) -> Self {
        Self::NotInitialized {
            details: details.into(),
        }
    }

    /// Create a load error (backward compatible)
    pub fn load_error(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self::LoadError {
            context: context.into(),
            message: message.into(),
        }
    }

    /// Create a simple load error (backward compatible)
    pub fn simple_load_error(message: impl Into<String>) -> Self {
        Self::LoadError {
            context: "general".into(),
            message: message.into(),
        }
    }

    /// Create a forward error with layer index
    pub fn forward_error(layer_idx: usize, message: impl Into<String>) -> Self {
        Self::ForwardError {
            layer_idx,
            message: message.into(),
        }
    }

    /// Create a weight loading error
    pub fn weight_load_error(tensor_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::WeightLoadError {
            tensor_name: tensor_name.into(),
            reason: reason.into(),
        }
    }

    /// Create a tensor not found error
    pub fn tensor_not_found(name: impl Into<String>, model: impl Into<String>) -> Self {
        Self::TensorNotFound {
            name: name.into(),
            model: model.into(),
        }
    }

    /// Create a state count mismatch error
    pub fn state_count_mismatch(model: impl Into<String>, expected: usize, got: usize) -> Self {
        Self::StateCountMismatch {
            model: model.into(),
            expected,
            got,
        }
    }

    /// Create a numerical instability error
    pub fn numerical_instability(operation: impl Into<String>, details: impl Into<String>) -> Self {
        Self::NumericalInstability {
            operation: operation.into(),
            details: details.into(),
        }
    }

    /// Create an unsupported operation error
    pub fn unsupported_operation(
        operation: impl Into<String>,
        model_type: impl Into<String>,
    ) -> Self {
        Self::UnsupportedOperation {
            operation: operation.into(),
            model_type: model_type.into(),
        }
    }

    /// Create a quantization error
    pub fn quantization_error(message: impl Into<String>) -> Self {
        Self::QuantizationError {
            message: message.into(),
        }
    }
}
