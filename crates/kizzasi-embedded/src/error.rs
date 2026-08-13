//! Error types for kizzasi-embedded

use core::fmt;

/// Errors that can occur during embedded SSM inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

// `core::error::Error` has been stable since Rust 1.81 (the workspace MSRV is
// 1.89) and is available in `no_std`, so this works on every supported build
// and lets `EmbeddedError` propagate with `?` into `Box<dyn Error>` /
// `anyhow::Error` chains without a hand-written wrapper.
impl core::error::Error for EmbeddedError {}

/// Result type alias for embedded operations
pub type EmbeddedResult<T> = Result<T, EmbeddedError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_error_is_std_error() {
        fn as_dyn(e: &EmbeddedError) -> &dyn core::error::Error {
            e
        }
        let err = EmbeddedError::DimensionMismatch {
            expected: 4,
            got: 3,
        };
        let dyn_err = as_dyn(&err);
        // Display must round-trip through the trait object.
        #[cfg(feature = "std")]
        {
            assert_eq!(
                std::string::ToString::to_string(dyn_err),
                "dimension mismatch: expected 4, got 3"
            );
        }
        assert!(dyn_err.source().is_none(), "leaf error has no source");
    }

    #[test]
    fn test_question_mark_into_boxed_error() {
        #[cfg(feature = "std")]
        {
            fn inner() -> EmbeddedResult<()> {
                Err(EmbeddedError::NumericalInstability)
            }
            fn outer() -> Result<(), std::boxed::Box<dyn core::error::Error>> {
                inner()?;
                Ok(())
            }
            assert!(outer().is_err(), "`?` must convert into Box<dyn Error>");
        }
    }

    #[test]
    fn test_error_is_eq() {
        assert_eq!(
            EmbeddedError::BufferTooSmall {
                required: 8,
                available: 4
            },
            EmbeddedError::BufferTooSmall {
                required: 8,
                available: 4
            }
        );
        assert_ne!(
            EmbeddedError::NumericalInstability,
            EmbeddedError::InvalidConfig("x")
        );
    }
}
