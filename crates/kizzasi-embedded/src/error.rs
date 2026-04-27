//! Error types for kizzasi-embedded

use core::fmt;

/// Errors that can occur during embedded SSM inference
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddedError {
    /// Input dimensions do not match model configuration
    DimensionMismatch { expected: usize, got: usize },
    /// Invalid configuration parameter
    InvalidConfig(&'static str),
    /// Numerical instability detected (e.g., division by zero, NaN)
    NumericalInstability,
    /// Output buffer is too small to hold the result
    BufferTooSmall { required: usize, available: usize },
}

impl fmt::Display for EmbeddedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbeddedError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            EmbeddedError::InvalidConfig(msg) => {
                write!(f, "invalid configuration: {msg}")
            }
            EmbeddedError::NumericalInstability => {
                write!(f, "numerical instability detected")
            }
            EmbeddedError::BufferTooSmall {
                required,
                available,
            } => {
                write!(
                    f,
                    "buffer too small: required {required}, available {available}"
                )
            }
        }
    }
}

/// Result type alias for embedded operations
pub type EmbeddedResult<T> = Result<T, EmbeddedError>;
