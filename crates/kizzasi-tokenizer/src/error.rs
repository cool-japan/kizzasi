//! Error types for kizzasi-tokenizer
//!
//! Provides comprehensive error types with detailed context for debugging
//! and user-friendly error messages.

use thiserror::Error;

/// Result type alias for tokenizer operations
pub type TokenizerResult<T> = Result<T, TokenizerError>;

/// Errors that can occur in tokenizer operations
#[derive(Error, Debug)]
pub enum TokenizerError {
    /// Configuration parameter is invalid
    ///
    /// This error includes detailed context about what parameter was invalid
    /// and why, to help users fix their configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Array dimensions don't match expected values
    ///
    /// Provides both the expected and actual dimensions to help diagnose
    /// shape mismatches in multi-dimensional arrays.
    #[error("Dimension mismatch in {context}: expected {expected}, got {got}")]
    DimensionMismatch {
        expected: usize,
        got: usize,
        context: String,
    },

    /// Signal encoding operation failed
    ///
    /// Wraps detailed information about why encoding failed, such as
    /// invalid input ranges, NaN values, or algorithmic failures.
    #[error("Encoding failed in {operation}: {reason}")]
    EncodingError { operation: String, reason: String },

    /// Signal decoding operation failed
    ///
    /// Includes context about which decoding step failed and why.
    #[error("Decoding failed in {operation}: {reason}")]
    DecodingError { operation: String, reason: String },

    /// Codebook has not been initialized before use
    ///
    /// This typically happens when trying to use a VQ-VAE tokenizer
    /// before training or loading a pretrained codebook.
    #[error("Codebook not initialized: {hint}")]
    CodebookNotInitialized { hint: String },

    /// Value is outside the valid range
    ///
    /// Provides the offending value and the valid range for debugging.
    #[error("Value out of range in {context}: {value} not in [{min}, {max}]")]
    ValueOutOfRange {
        value: f32,
        min: f32,
        max: f32,
        context: String,
    },

    /// Error from kizzasi-core crate (not available on wasm32)
    #[cfg(not(target_arch = "wasm32"))]
    #[error("Core error: {0}")]
    CoreError(#[from] kizzasi_core::CoreError),

    /// Internal error that should not happen under normal circumstances
    ///
    /// If you encounter this error, it likely indicates a bug in the library.
    /// Please report it with the full error message and context.
    #[error("Internal error (please report this): {0}")]
    InternalError(String),

    /// Invalid input data provided to a function
    ///
    /// This error provides context about what was invalid in the input,
    /// such as empty arrays, NaN values, or incorrect types.
    #[error("Invalid input in {operation}: {reason}")]
    InvalidInput { operation: String, reason: String },

    /// Serialization or deserialization failed
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// File I/O operation failed
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Training operation failed
    #[error("Training error at epoch {epoch}: {reason}")]
    TrainingError { epoch: usize, reason: String },

    /// Numerical computation resulted in invalid values (NaN, Inf)
    #[error("Numerical error in {operation}: {reason}")]
    NumericalError { operation: String, reason: String },

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {resource} - {details}")]
    ResourceLimitExceeded { resource: String, details: String },

    /// Feature not yet implemented
    #[error("Feature not implemented: {0}")]
    NotImplemented(String),
}

impl TokenizerError {
    /// Create a dimension mismatch error with context
    pub fn dim_mismatch(expected: usize, got: usize, context: impl Into<String>) -> Self {
        Self::DimensionMismatch {
            expected,
            got,
            context: context.into(),
        }
    }

    /// Create an encoding error with context
    pub fn encoding(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::EncodingError {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create a decoding error with context
    pub fn decoding(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::DecodingError {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create a value out of range error with context
    pub fn out_of_range(value: f32, min: f32, max: f32, context: impl Into<String>) -> Self {
        Self::ValueOutOfRange {
            value,
            min,
            max,
            context: context.into(),
        }
    }

    /// Create an invalid input error with context
    pub fn invalid_input(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create a numerical error with context
    pub fn numerical(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::NumericalError {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create a training error with context
    pub fn training(epoch: usize, reason: impl Into<String>) -> Self {
        Self::TrainingError {
            epoch,
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = TokenizerError::dim_mismatch(10, 20, "input validation");
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("20"));
        assert!(err.to_string().contains("input validation"));

        let err2 = TokenizerError::encoding("VQ-VAE", "codebook lookup failed");
        assert!(err2.to_string().contains("VQ-VAE"));
        assert!(err2.to_string().contains("codebook lookup failed"));
    }

    #[test]
    fn test_helper_constructors() {
        let err = TokenizerError::out_of_range(1.5, 0.0, 1.0, "quantization");
        assert!(matches!(err, TokenizerError::ValueOutOfRange { .. }));

        let err2 = TokenizerError::invalid_input("encode", "empty array");
        assert!(matches!(err2, TokenizerError::InvalidInput { .. }));
    }
}
