//! Error types for the `kizzasi-webgpu` crate.
//!
//! All wgpu-sourced errors are captured as `String` to avoid gating
//! the error enum on the `webgpu` feature flag.

use thiserror::Error;

/// Errors that may arise from WebGPU backend operations.
#[derive(Debug, Error)]
pub enum WebGpuError {
    /// Adapter request failed (no suitable GPU found).
    #[error("failed to request adapter: {0}")]
    AdapterRequest(String),

    /// Device or queue creation failed.
    #[error("failed to request device: {0}")]
    DeviceRequest(String),

    /// Buffer byte size mismatch between expected and actual.
    #[error("buffer size mismatch: expected {expected}, got {got}")]
    BufferSizeMismatch { expected: u64, got: u64 },

    /// GPU buffer mapping failed.
    #[error("map buffer failed: {0}")]
    MapBuffer(String),

    /// The `webgpu` feature is not compiled in; no GPU operations are available.
    #[error("backend not available (compile with --features webgpu)")]
    BackendUnavailable,

    /// General-purpose error wrapper.
    #[error("{0}")]
    Other(String),
}
