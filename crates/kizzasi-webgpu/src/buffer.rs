//! GPU buffer types and usage flags.

#[cfg(feature = "webgpu")]
use crate::error::WebGpuError;

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
/// The metadata fields (`size_bytes`, `usage`, `label`) are always accessible.
/// A buffer created by [`crate::WebGpuBackend::upload_f32`] or returned by one
/// of the `*_gpu_buf` kernels owns a live `wgpu::Buffer`; one created by
/// [`GpuBuffer::metadata_only`] does not, and passing it to a GPU operation
/// returns [`WebGpuError::NoGpuAllocation`].
pub struct GpuBuffer {
    /// The underlying wgpu buffer handle, when the buffer is GPU-resident.
    #[cfg(feature = "webgpu")]
    pub(crate) inner: Option<wgpu::Buffer>,

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
    /// Available in every feature configuration — enabling `webgpu` never
    /// removes this constructor, so Cargo's graph-wide feature unification
    /// cannot break a downstream build that uses it.
    ///
    /// The result is *not* GPU-resident: GPU operations reject it with
    /// [`WebGpuError::NoGpuAllocation`].
    /// To obtain a usable buffer call
    /// [`WebGpuBackend::upload_f32`](crate::WebGpuBackend::upload_f32).
    pub fn metadata_only(size_bytes: u64, usage: GpuBufferUsage, label: impl Into<String>) -> Self {
        Self {
            #[cfg(feature = "webgpu")]
            inner: None,
            size_bytes,
            usage,
            label: label.into(),
        }
    }

    /// Returns `true` when this buffer owns a live GPU allocation.
    pub fn is_gpu_resident(&self) -> bool {
        #[cfg(feature = "webgpu")]
        {
            self.inner.is_some()
        }

        #[cfg(not(feature = "webgpu"))]
        {
            false
        }
    }

    /// Number of `f32` values the buffer can hold.
    pub fn len_f32(&self) -> u64 {
        self.size_bytes / std::mem::size_of::<f32>() as u64
    }

    /// Constructs a `GpuBuffer` directly from a `wgpu::Buffer`.
    ///
    /// Intended for internal use inside `WebGpuBackend` and the kernels.
    #[cfg(feature = "webgpu")]
    pub(crate) fn from_wgpu(
        inner: wgpu::Buffer,
        size_bytes: u64,
        usage: GpuBufferUsage,
        label: impl Into<String>,
    ) -> Self {
        Self {
            inner: Some(inner),
            size_bytes,
            usage,
            label: label.into(),
        }
    }

    /// Borrow the underlying wgpu buffer, or fail if this is metadata-only.
    #[cfg(feature = "webgpu")]
    pub(crate) fn wgpu_buffer(&self) -> Result<&wgpu::Buffer, WebGpuError> {
        self.inner
            .as_ref()
            .ok_or_else(|| WebGpuError::NoGpuAllocation(self.label.clone()))
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("size_bytes", &self.size_bytes)
            .field("usage", &self.usage)
            .field("label", &self.label)
            .field("gpu_resident", &self.is_gpu_resident())
            .finish_non_exhaustive()
    }
}
