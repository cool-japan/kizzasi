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

    /// A request exceeds a hard limit of the logical device.
    ///
    /// Returned *before* any dispatch, so an over-sized request fails with a
    /// recoverable error instead of a wgpu validation abort.
    #[error("device limit exceeded: {what} requires {required}, device allows {limit}")]
    DeviceLimitExceeded {
        /// Human-readable description of the resource that overflowed.
        what: String,
        /// The amount the operation asked for.
        required: u64,
        /// The maximum the device supports.
        limit: u64,
    },

    /// A [`crate::GpuBuffer`] carries metadata only and has no GPU allocation.
    #[error("buffer '{0}' has no GPU allocation (metadata-only buffer)")]
    NoGpuAllocation(String),

    /// General-purpose error wrapper.
    ///
    /// Also used for wgpu validation / out-of-memory errors captured by the
    /// device error scope that every GPU entry point installs.
    #[error("{0}")]
    Other(String),
}
