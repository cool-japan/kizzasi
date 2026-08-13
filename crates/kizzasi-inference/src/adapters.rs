//! Network adapters for streaming inference
//!
//! This module provides network adapters for exposing inference engines over various protocols:
//! - WebSocket: Real-time bidirectional streaming
//! - MQTT: Publish/subscribe messaging for IoT scenarios
//! - gRPC: High-performance RPC with streaming support
//!
//! # Features
//!
//! - `websocket`: Enable WebSocket adapter
//! - `mqtt`: Enable MQTT adapter
//! - `grpc`: Enable gRPC adapter
//! - `network`: Enable all network adapters
//!
//! # Examples
//!
//! ## WebSocket Streaming
//!
//! ```rust,ignore
//! use kizzasi_inference::adapters::WebSocketAdapter;
//! use kizzasi_inference::streaming::StreamingEngine;
//!
//! let engine = StreamingEngine::new(config);
//! let adapter = WebSocketAdapter::new(engine);
//! adapter.serve("127.0.0.1:8080").await?;
//! ```
//!
//! ## MQTT Publishing
//!
//! ```rust,ignore
//! use kizzasi_inference::adapters::MqttAdapter;
//! use kizzasi_inference::streaming::StreamingEngine;
//!
//! let engine = StreamingEngine::new(config);
//! let adapter = MqttAdapter::new(engine, "mqtt://broker:1883");
//! adapter.subscribe("agsp/input").await?;
//! adapter.publish_to("agsp/output").await?;
//! ```

#[cfg(feature = "websocket")]
pub mod websocket;

#[cfg(feature = "mqtt")]
pub mod mqtt;

#[cfg(feature = "grpc")]
pub mod grpc;

#[cfg(feature = "rest")]
pub mod rest;

#[cfg(feature = "websocket")]
pub use websocket::WebSocketAdapter;

#[cfg(feature = "mqtt")]
pub use mqtt::MqttAdapter;

#[cfg(feature = "grpc")]
pub use grpc::GrpcAdapter;

#[cfg(feature = "rest")]
pub use rest::{RestAdapter, RestServer};

use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};

#[cfg(feature = "async")]
use crate::error::InferenceResult;

/// Message format for network communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMessage {
    /// Request ID for tracking
    pub request_id: String,

    /// Input signal data
    pub input: Vec<f32>,

    /// Optional configuration overrides
    #[serde(default)]
    pub config: MessageConfig,

    /// Timestamp (Unix epoch milliseconds)
    pub timestamp: i64,
}

/// Response message from inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Request ID matching the original request
    pub request_id: String,

    /// Generated output signal
    pub output: Vec<f32>,

    /// Inference latency in milliseconds
    pub latency_ms: f64,

    /// Number of tokens generated
    pub num_tokens: usize,

    /// Timestamp (Unix epoch milliseconds)
    pub timestamp: i64,
}

/// Configuration overrides in message
///
/// These are honoured by the adapters through
/// [`StreamingEngine::step_async_with`](crate::streaming::StreamingEngine::step_async_with):
/// an override the engine cannot apply is reported back to the client as an
/// error rather than silently discarded.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageConfig {
    /// Temperature for sampling
    pub temperature: Option<f32>,

    /// Top-k for sampling
    pub top_k: Option<usize>,

    /// Top-p for sampling
    pub top_p: Option<f32>,

    /// Maximum number of tokens to generate
    pub max_tokens: Option<usize>,
}

#[cfg(feature = "streaming")]
impl From<&MessageConfig> for crate::streaming::SamplingOverrides {
    fn from(config: &MessageConfig) -> Self {
        Self {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            max_tokens: config.max_tokens,
        }
    }
}

/// Flatten a multi-step rollout into a single response payload
///
/// Returns the concatenated samples and the number of generated steps, which is
/// what [`InferenceResponse::num_tokens`] reports.
#[cfg(feature = "streaming")]
pub fn flatten_outputs(outputs: &[Array1<f32>]) -> (Vec<f32>, usize) {
    let flat = outputs.iter().flat_map(|o| o.iter().copied()).collect();
    (flat, outputs.len())
}

impl InferenceMessage {
    /// Create a new inference message
    pub fn new(request_id: impl Into<String>, input: Vec<f32>) -> Self {
        Self {
            request_id: request_id.into(),
            input,
            config: MessageConfig::default(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Convert to Array1
    pub fn to_array(&self) -> Array1<f32> {
        Array1::from_vec(self.input.clone())
    }
}

impl InferenceResponse {
    /// Create a new inference response
    pub fn new(
        request_id: impl Into<String>,
        output: Vec<f32>,
        latency_ms: f64,
        num_tokens: usize,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            output,
            latency_ms,
            num_tokens,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Convert to Array1
    pub fn to_array(&self) -> Array1<f32> {
        Array1::from_vec(self.output.clone())
    }
}

/// Trait for network adapters
///
/// `start`/`stop` take `&self` (not `&mut self`) so `stop` can be called
/// through a shared `Arc<Adapter>` handle while `start` is in flight on
/// another task — e.g. `let adapter = Arc::new(adapter); let a2 =
/// adapter.clone(); tokio::spawn(async move { a2.start().await });
/// adapter.stop().await;`.
#[cfg(feature = "async")]
pub trait NetworkAdapter {
    /// Start the adapter and serve until [`NetworkAdapter::stop`] is called
    /// (or the underlying transport ends on its own).
    fn start(&self) -> impl std::future::Future<Output = InferenceResult<()>>;

    /// Signal a running [`NetworkAdapter::start`] call to shut down.
    ///
    /// Implementations must actually interrupt the running serve loop (e.g.
    /// via a `tokio::sync::watch` flag observed with `select!`/
    /// `wait_for`), not merely flip a flag nothing reads.
    fn stop(&self) -> impl std::future::Future<Output = InferenceResult<()>>;

    /// Check if the adapter is currently serving.
    ///
    /// Must be a plain, synchronous, panic-free read (e.g. backed by
    /// `watch::Sender::borrow` or an `AtomicBool`) — never one that blocks on
    /// the async runtime, which panics outside a multi-threaded Tokio
    /// context.
    fn is_running(&self) -> bool;
}
