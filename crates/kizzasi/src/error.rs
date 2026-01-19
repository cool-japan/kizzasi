//! Error types for the Kizzasi facade crate

use thiserror::Error;

/// Result type alias for Kizzasi operations
pub type KizzasiResult<T> = Result<T, KizzasiError>;

/// Unified error type for Kizzasi with detailed context and recovery suggestions
#[derive(Error, Debug)]
pub enum KizzasiError {
    #[error("Core error: {0}")]
    Core(#[from] kizzasi_core::CoreError),

    #[cfg(feature = "logic")]
    #[error("Logic error: {0}")]
    Logic(#[from] kizzasi_logic::LogicError),

    #[cfg(feature = "io")]
    #[error("IO error: {0}")]
    Io(#[from] kizzasi_io::IoError),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
        context: String,
    },

    #[error("Invalid state: {reason}")]
    InvalidState {
        reason: String,
        recovery: Option<String>,
    },

    #[error("Model not ready: {reason}")]
    ModelNotReady { reason: String, suggestion: String },

    #[error("Resource exhausted: {resource}")]
    ResourceExhausted {
        resource: String,
        current: usize,
        limit: usize,
        suggestion: String,
    },
}

impl KizzasiError {
    /// Create a configuration error with a message
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create an inference error with a message
    pub fn inference(msg: impl Into<String>) -> Self {
        Self::Inference(msg.into())
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize, context: impl Into<String>) -> Self {
        Self::DimensionMismatch {
            expected,
            actual,
            context: context.into(),
        }
    }

    /// Create an invalid state error
    pub fn invalid_state(reason: impl Into<String>) -> Self {
        Self::InvalidState {
            reason: reason.into(),
            recovery: None,
        }
    }

    /// Create an invalid state error with recovery suggestion
    pub fn invalid_state_with_recovery(
        reason: impl Into<String>,
        recovery: impl Into<String>,
    ) -> Self {
        Self::InvalidState {
            reason: reason.into(),
            recovery: Some(recovery.into()),
        }
    }

    /// Create a model not ready error
    pub fn model_not_ready(reason: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self::ModelNotReady {
            reason: reason.into(),
            suggestion: suggestion.into(),
        }
    }

    /// Create a resource exhausted error
    pub fn resource_exhausted(
        resource: impl Into<String>,
        current: usize,
        limit: usize,
        suggestion: impl Into<String>,
    ) -> Self {
        Self::ResourceExhausted {
            resource: resource.into(),
            current,
            limit,
            suggestion: suggestion.into(),
        }
    }

    /// Get recovery suggestions for this error
    pub fn recovery_suggestion(&self) -> Option<String> {
        match self {
            Self::DimensionMismatch {
                expected,
                actual,
                context,
            } => Some(format!(
                "Ensure input dimension matches configuration. Expected {}, got {}. Context: {}",
                expected, actual, context
            )),
            Self::InvalidState {
                recovery: Some(r), ..
            } => Some(r.clone()),
            Self::ModelNotReady { suggestion, .. } => Some(suggestion.clone()),
            Self::ResourceExhausted { suggestion, .. } => Some(suggestion.clone()),
            Self::Config(msg) => {
                if msg.contains("input_dim") {
                    Some("Set input_dim > 0 in configuration".to_string())
                } else if msg.contains("output_dim") {
                    Some("Set output_dim > 0 in configuration".to_string())
                } else if msg.contains("hidden_dim") {
                    Some("Set hidden_dim > 0 in configuration".to_string())
                } else {
                    Some("Check configuration parameters for valid values".to_string())
                }
            }
            _ => None,
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::DimensionMismatch { .. }
                | Self::InvalidState { .. }
                | Self::ModelNotReady { .. }
                | Self::Config(_)
        )
    }

    /// Get the error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::Core(_) => ErrorCategory::Core,
            #[cfg(feature = "logic")]
            Self::Logic(_) => ErrorCategory::Logic,
            #[cfg(feature = "io")]
            Self::Io(_) => ErrorCategory::Io,
            Self::Config(_) => ErrorCategory::Configuration,
            Self::Inference(_) => ErrorCategory::Inference,
            Self::DimensionMismatch { .. } => ErrorCategory::Validation,
            Self::InvalidState { .. } => ErrorCategory::State,
            Self::ModelNotReady { .. } => ErrorCategory::Initialization,
            Self::ResourceExhausted { .. } => ErrorCategory::Resource,
        }
    }
}

/// Error category for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Core,
    Logic,
    Io,
    Configuration,
    Inference,
    Validation,
    State,
    Initialization,
    Resource,
}

impl ErrorCategory {
    /// Get a human-readable description of the category
    pub fn description(&self) -> &'static str {
        match self {
            Self::Core => "Core system error",
            Self::Logic => "Constraint logic error",
            Self::Io => "Input/output error",
            Self::Configuration => "Configuration error",
            Self::Inference => "Inference execution error",
            Self::Validation => "Input validation error",
            Self::State => "Invalid state error",
            Self::Initialization => "Initialization error",
            Self::Resource => "Resource limit error",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch_error() {
        let err = KizzasiError::dimension_mismatch(3, 5, "input array");
        assert!(matches!(err, KizzasiError::DimensionMismatch { .. }));
        assert_eq!(err.category(), ErrorCategory::Validation);
        assert!(err.recovery_suggestion().is_some());
    }

    #[test]
    fn test_invalid_state_with_recovery() {
        let err = KizzasiError::invalid_state_with_recovery(
            "Model not initialized",
            "Call build() on the builder first",
        );
        assert!(matches!(err, KizzasiError::InvalidState { .. }));
        assert!(err.recovery_suggestion().is_some());
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_model_not_ready() {
        let err = KizzasiError::model_not_ready(
            "Weights not loaded",
            "Call load_weights() or provide weights_path in config",
        );
        assert_eq!(err.category(), ErrorCategory::Initialization);
    }

    #[test]
    fn test_resource_exhausted() {
        let err = KizzasiError::resource_exhausted(
            "context buffer",
            10000,
            8192,
            "Reduce context window size or clear buffer",
        );
        assert!(matches!(err, KizzasiError::ResourceExhausted { .. }));
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(
            ErrorCategory::Configuration.description(),
            "Configuration error"
        );
        assert_eq!(
            ErrorCategory::Validation.description(),
            "Input validation error"
        );
    }
}
