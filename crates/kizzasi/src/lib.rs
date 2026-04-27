//! # Kizzasi (兆候)
//!
//! **Autoregressive General-Purpose Signal Predictor (AGSP)**
//!
//! *"Predicting the flux of the world with the precision of logic."*
//!
//! Kizzasi is a Rust-native AGSP designed for continuous signal streams—audio,
//! sensor data, robotics control signals, and video frames. Unlike traditional
//! LLMs that operate on discrete text tokens, Kizzasi is built for the
//! continuous domain.
//!
//! ## Core Innovation: Neuro-Symbolic Architecture
//!
//! Kizzasi combines the learning capability of State Space Models (Mamba/RWKV)
//! with the strict reliability of TensorLogic. This ensures that predicted
//! signals not only follow statistical likelihoods but also adhere to defined
//! physical laws, safety constraints, and logical rules.
//!
//! ## COOLJAPAN Ecosystem
//!
//! This crate is part of the COOLJAPAN scientific computing ecosystem and
//! follows the KIZZASI_POLICY.md for dependency management, using:
//! - `scirs2-core` for array and numerical operations
//! - `tensorlogic` for constraint logic
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use kizzasi::prelude::*;
//!
//! fn main() -> Result<()> {
//!     // Initialize predictor with Mamba2 backend
//!     let config = KizzasiConfig::new()
//!         .model_type(ModelType::Mamba2)
//!         .context_window(8192);
//!
//!     let mut predictor = Kizzasi::new(config)?;
//!
//!     // Single step prediction
//!     let input = array![0.1, 0.2, 0.3];
//!     let output = predictor.step(&input)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Common Use Cases
//!
//! ### Audio Processing
//! ```rust,ignore
//! let mut predictor = KizzasiBuilder::audio_preset().build()?;
//! let sample = array![0.5]; // Single sample
//! let next = predictor.step(&sample)?;
//! ```
//!
//! ### Robotics Control
//! ```rust,ignore
//! let mut predictor = KizzasiBuilder::robotics_preset(6).build()?; // 6-DOF
//! let joint_angles = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
//! let predicted = predictor.step(&joint_angles)?;
//! ```
//!
//! ### Sensor Monitoring
//! ```rust,ignore
//! let mut predictor = KizzasiBuilder::sensor_preset(10).build()?;
//! let readings = array![/* 10 sensor values */];
//! let forecast = predictor.predict_n(&readings, 5)?; // 5 steps ahead
//! ```
//!
//! ### With Safety Constraints
//! ```rust,ignore
//! # #[cfg(feature = "logic")]
//! # {
//! use kizzasi::{ConstraintBuilder, Guardrail, GuardrailSet};
//!
//! let mut predictor = KizzasiBuilder::robotics_preset(3).build()?;
//!
//! // Add safety bounds
//! let mut guardrails = GuardrailSet::new();
//! let constraint = ConstraintBuilder::new()
//!     .name("joint_limits")
//!     .greater_eq(-3.14)
//!     .less_eq(3.14)
//!     .build()?;
//! guardrails.add_global(Guardrail::new(constraint, false));
//! predictor.set_guardrails(guardrails);
//! # }
//! ```
//!
//! ## Features
//!
//! - `std` (default): Standard library support
//! - `full` (default): Enable all features (io, logic, async, config-files, macros)
//! - `io`: Physical world connectors (MQTT, Audio)
//! - `logic`: TensorLogic constraint enforcement
//! - `async`: Async/streaming APIs, connection pooling with tokio
//! - `config-files`: TOML/YAML configuration file support
//! - `macros`: Derive macros for custom configurations
//!
//! ## Performance
//!
//! Kizzasi is optimized for real-time inference:
//! - Zero-copy operations where possible
//! - Efficient state management
//! - SIMD-optimized computations via scirs2-core
//! - Batch processing support
//!
//! Run benchmarks with: `cargo bench --bench predictor_benchmarks`
//!
//! ## Thread Safety
//!
//! `Kizzasi` predictors are `Send` but not `Sync`. For concurrent predictions:
//! - Use `fork()` to create independent predictors per thread
//! - Or wrap in `Arc<Mutex<Kizzasi>>` for shared access
//! - Async APIs are available with the `async` feature
//!
//! ## Advanced Features
//!
//! ### Model Versioning and A/B Testing
//! Manage multiple model versions with deployment strategies:
//! ```rust,ignore
//! use kizzasi::versioning::{ModelRegistry, DeploymentStrategy};
//!
//! let mut registry = ModelRegistry::new();
//! registry.deploy("2.0.0", DeploymentStrategy::Canary { traffic_percent: 10 })?;
//! let model = registry.select_for_request(request_id)?;
//! ```
//!
//! ### Telemetry and Metrics
//! Production-grade metrics collection and monitoring:
//! ```rust,ignore
//! use kizzasi::telemetry::{MetricsCollector, MetricEvent};
//!
//! let metrics = MetricsCollector::new("my_service");
//! metrics.record(MetricEvent::Prediction { latency_us: 1500, input_dim: 64, output_dim: 64 });
//! let stats = metrics.snapshot();
//! println!("P99 latency: {:.2}ms", stats.p99_latency_ms);
//! ```
//!
//! ### Connection Pooling
//! Efficient resource management for I/O operations (requires `async` feature):
//! ```rust,ignore
//! use kizzasi::pool::{ConnectionPool, PoolConfig};
//!
//! let config = PoolConfig::default().with_max_connections(10);
//! let pool = ConnectionPool::new(factory, config).await?;
//! let conn = pool.acquire().await?;
//! ```
//!
//! ## Examples
//!
//! See the `examples/` directory for comprehensive examples:
//! - `basic_prediction` - Core API usage
//! - `with_guardrails` - Constraint enforcement
//! - `audio_processing` - Audio signal prediction
//! - `robotics_control` - Robot control systems
//! - `streaming` - Async streaming APIs
//! - `anomaly_detection` - Real-time anomaly detection
//! - `model_checkpointing` - Save/load predictor configuration
//! - `custom_model` - Advanced configuration patterns
//! - `production_deployment` - Full production setup with versioning, metrics, and pooling
//! - `metrics_monitoring` - Comprehensive metrics and monitoring

pub mod autotuning;
mod checkpoint;
pub mod ensemble;
mod error;
mod lazy;
pub mod optimization;
pub mod plugin;
mod predictor;
pub mod telemetry;
pub mod versioning;

#[cfg(feature = "async")]
mod streaming;

#[cfg(feature = "async")]
pub mod pool;

#[cfg(feature = "async")]
pub mod distributed;

#[cfg(feature = "config-files")]
pub mod config;

pub mod prelude;

pub use autotuning::{
    AdaptiveTuner, AutoTuner, TuningConfig, TuningRecommendation, WorkloadProfile,
};
pub use checkpoint::{CheckpointMetadata, FullStateCheckpoint, PredictorCheckpoint};
pub use ensemble::{EnsemblePredictor, EnsembleStats, ModelStats, VotingStrategy};
pub use error::{ErrorCategory, KizzasiError, KizzasiResult};
pub use lazy::LazyKizzasi;
pub use optimization::{CacheStats, OptimizationConfig, OptimizationStats, OptimizedPredictor};
pub use plugin::{LoggingPlugin, Plugin, PluginContext, PluginManager, PluginPhase, StatsPlugin};
pub use predictor::{Kizzasi, KizzasiBuilder, SignalInput};
pub use telemetry::{
    Instrumented, MetricEvent, MetricValue, MetricsCollector, MetricsConfig, MetricsSnapshot,
};
pub use versioning::{
    DeploymentStrategy, ModelMetadata, ModelRegistry, ModelVersion, RegistryStats, SemanticVersion,
};

#[cfg(feature = "async")]
pub use streaming::{AsyncPredictor, PredictionStream, StreamProcessor};

#[cfg(feature = "async")]
pub use pool::{ConnectionFactory, ConnectionPool, PoolConfig, PoolStats};

#[cfg(feature = "async")]
pub use distributed::{
    distributed_predict, distributed_predict_with_config, DistributedConfig, DistributedPredictor,
    LoadBalancingStrategy, WorkerStats,
};

#[cfg(feature = "config-files")]
pub use config::{ConfigFile, ConfigFormat, ConfigLoader};

#[cfg(feature = "macros")]
pub use kizzasi_macros::{Instrumented, KizzasiConfig, Preset};

// Re-export core types
pub use kizzasi_core::{
    ContinuousEmbedding, CoreError, CoreResult, HiddenState, KizzasiConfig, ModelType,
    SelectiveSSM, SignalPredictor, StateSpaceModel,
};

// Re-export logic types when feature is enabled
#[cfg(feature = "logic")]
pub use kizzasi_logic::{
    BoundType, ComposedConstraint, ConstrainedInference, ConstrainedProjection, Constraint,
    ConstraintBuilder, Guardrail, GuardrailSet, LogicError, LogicResult, LogicalOperator,
};

// Re-export io types when feature is enabled
#[cfg(feature = "io")]
pub use kizzasi_io::{
    Filter, IoError, IoResult, MemoryStream, SignalProcessor, SignalStream, StreamConfig,
};

#[cfg(all(feature = "io", feature = "mqtt"))]
pub use kizzasi_io::{MqttClient, MqttConfig, MqttStream};

#[cfg(all(feature = "io", feature = "audio"))]
pub use kizzasi_io::{AudioConfig, AudioInput};

// Re-export scirs2-core array types
pub use scirs2_core::ndarray::{array, Array1, Array2};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kizzasi_creation() {
        let config = KizzasiConfig::new()
            .model_type(ModelType::Mamba2)
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let predictor = Kizzasi::new(config);
        assert!(predictor.is_ok());
    }
}
