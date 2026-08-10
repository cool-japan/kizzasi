//! GPU buffer types and usage flags.

/// Usage classification for GPU buffers.
///
/// This is a thin abstraction over `wgpu::BufferUsages` that remains
/// available even when the `webgpu` feature is disabled, so that
/// downstream code can refer to buffer metadata without gating on the
/// feature flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBufferUsage {
    /// Storage buffer readable/writable by compute shaders.
    ///
    /// Backed by `STORAGE | COPY_SRC | COPY_DST` in wgpu.
    Storage,
    /// Staging buffer for CPU↔GPU transfers.
    ///
    /// Not directly shader-accessible; backed by `MAP_READ | COPY_DST` in wgpu.
    Staging,
}

/// An owned handle to a GPU buffer, plus associated metadata.
///
/// The inner `wgpu::Buffer` is only present when compiled with `--features webgpu`.
/// The metadata fields (`size_bytes`, `usage`, `label`) are always accessible.
pub struct GpuBuffer {
    /// The underlying wgpu buffer handle.
    #[cfg(feature = "webgpu")]
    pub(crate) inner: wgpu::Buffer,

    /// Size of the buffer in bytes.
    pub size_bytes: u64,

    /// How the buffer is intended to be used.
    pub usage: GpuBufferUsage,

    /// Human-readable label for debugging and GPU profiling.
    pub label: String,
}

impl GpuBuffer {
    /// Constructs a metadata-only `GpuBuffer` (no GPU allocation).
    ///
    /// This constructor is primarily useful in tests that run without a GPU.
    /// When the `webgpu` feature is enabled, use [`super::WebGpuBackend::upload_f32`]
    /// instead to create a GPU-backed buffer.
    #[cfg(not(feature = "webgpu"))]
    pub fn metadata_only(size_bytes: u64, usage: GpuBufferUsage, label: impl Into<String>) -> Self {
        Self {
            size_bytes,
            usage,
            label: label.into(),
        }
    }

    /// Constructs a `GpuBuffer` directly from a `wgpu::Buffer`.
    ///
    /// Intended for internal use inside `WebGpuBackend`.
    #[cfg(feature = "webgpu")]
    pub(crate) fn from_wgpu(
        inner: wgpu::Buffer,
        size_bytes: u64,
        usage: GpuBufferUsage,
        label: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            size_bytes,
            usage,
            label: label.into(),
        }
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("size_bytes", &self.size_bytes)
            .field("usage", &self.usage)
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}
