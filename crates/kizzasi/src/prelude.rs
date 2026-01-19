//! Prelude module for convenient imports
//!
//! ```rust,ignore
//! use kizzasi::prelude::*;
//! ```

// Main predictor and builder
pub use crate::{Kizzasi, KizzasiBuilder, SignalInput};

// Checkpointing
pub use crate::{CheckpointMetadata, PredictorCheckpoint};

// Error types
pub use crate::{ErrorCategory, KizzasiError, KizzasiResult};

// Core types
pub use kizzasi_core::{
    ContinuousEmbedding, CoreError, CoreResult, HiddenState, KizzasiConfig, ModelType,
    SelectiveSSM, SignalPredictor, StateSpaceModel,
};

// Array types from scirs2-core
pub use scirs2_core::ndarray::{array, Array1, Array2};

// Logic types when feature is enabled
#[cfg(feature = "logic")]
pub use kizzasi_logic::{
    BoundType, ComposedConstraint, ConstrainedInference, ConstrainedProjection, Constraint,
    ConstraintBuilder, Guardrail, GuardrailSet, LogicError, LogicResult, LogicalOperator,
};

// IO types when feature is enabled
#[cfg(feature = "io")]
pub use kizzasi_io::{
    Filter, IoError, IoResult, MemoryStream, SignalProcessor, SignalStream, StreamConfig,
};

// Async streaming types when feature is enabled
#[cfg(feature = "async")]
pub use crate::{AsyncPredictor, PredictionStream, StreamProcessor};

// Convenience alias for Result
pub type Result<T> = KizzasiResult<T>;
