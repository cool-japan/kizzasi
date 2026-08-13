//! Prelude module for convenient imports
//!
//! ```rust,ignore
//! use kizzasi::prelude::*;
//! ```

// Main predictor and builder
pub use crate::{Kizzasi, KizzasiBuilder, SignalInput};

// Checkpointing
pub use crate::{CheckpointMetadata, FullStateCheckpoint, PredictorCheckpoint};

// Extensibility, ensembles, tuning, telemetry and versioning — the prelude
// mirrors the crate root so a glob import does not silently omit half the API.
pub use crate::{AutoTuner, TuningConfig, TuningRecommendation};
pub use crate::{DeploymentStrategy, ModelRegistry, ModelVersion, SemanticVersion};
pub use crate::{EnsemblePredictor, EnsembleStats, VotingStrategy};
pub use crate::{LazyKizzasi, Plugin, PluginManager, PluginPhase};
pub use crate::{MetricEvent, MetricValue, MetricsCollector, MetricsSnapshot};
pub use crate::{OptimizationConfig, OptimizationStats, OptimizedPredictor};

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

// Async streaming, pooling and distributed types when the feature is enabled
#[cfg(feature = "async")]
pub use crate::{AsyncPredictor, PredictionStream, StreamProcessor};

#[cfg(feature = "async")]
pub use crate::{ConnectionPool, PoolConfig};

#[cfg(feature = "async")]
pub use crate::{DistributedConfig, DistributedPredictor, LoadBalancingStrategy};

/// Convenience alias for [`KizzasiResult`].
///
/// The error parameter defaults to [`KizzasiError`] but can still be named
/// explicitly. A one-parameter alias would shadow `std::result::Result` for
/// every glob importer, so any user function written as `Result<T, E>` would
/// fail with a wrong-number-of-generic-arguments error.
pub type Result<T, E = KizzasiError> = core::result::Result<T, E>;
