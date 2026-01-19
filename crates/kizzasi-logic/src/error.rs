//! Error types for kizzasi-logic

use thiserror::Error;

/// Result type alias for logic operations
pub type LogicResult<T> = Result<T, LogicError>;

/// Errors that can occur in the logic module
#[derive(Error, Debug)]
pub enum LogicError {
    #[error("Invalid constraint: {0}")]
    InvalidConstraint(String),

    #[error("Constraint violation: {constraint} - value {value} violates {bound}")]
    ConstraintViolation {
        constraint: String,
        value: f32,
        bound: String,
    },

    #[error("Projection failed: {0}")]
    ProjectionFailed(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Infeasible constraint: {0}")]
    InfeasibleConstraint(String),

    #[error("Core error: {0}")]
    CoreError(#[from] kizzasi_core::CoreError),
}
