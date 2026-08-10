//! [`SsmBackend`] implementation using [`WebGpuBackend`].
//!
//! [`WebGpuSsmBackend`] wraps a [`WebGpuBackend`] and dispatches short
//! sequences (≤ [`crate::ssm_scan::MAX_SINGLE_PASS_LEN`]) to the GPU Blelloch
//! kernel.  Longer sequences fall back transparently to the CPU implementation
//! in `kizzasi-core`.

use kizzasi_core::ssm_backend::{CpuSsmBackend, SsmBackend};
use kizzasi_core::{CoreError, CoreResult};

use crate::ssm_scan::ssm_scan_gpu;
use crate::WebGpuBackend;

/// GPU-accelerated SSM backend wrapping [`WebGpuBackend`].
///
/// # Fallback behaviour
///
/// Sequences longer than [`crate::ssm_scan::MAX_SINGLE_PASS_LEN`] (256) are
/// transparently forwarded to [`CpuSsmBackend`].  This keeps the public API
/// uniform regardless of input length.
///
/// # Feature gate
///
/// The struct is always available as a type, but `ssm_scan` returns an error
/// for non-trivial inputs when compiled without `--features webgpu`.
pub struct WebGpuSsmBackend {
    backend: WebGpuBackend,
}

impl WebGpuSsmBackend {
    /// Wrap an existing [`WebGpuBackend`].
    pub fn new(backend: WebGpuBackend) -> Self {
        Self { backend }
    }
}

impl SsmBackend for WebGpuSsmBackend {
    fn ssm_scan(&self, elements: &[(f32, f32)]) -> CoreResult<Vec<(f32, f32)>> {
        if elements.len() > crate::ssm_scan::MAX_SINGLE_PASS_LEN {
            // Fall back to CPU for over-length sequences.
            return CpuSsmBackend.ssm_scan(elements);
        }
        ssm_scan_gpu(&self.backend, elements)
            .map_err(|e| CoreError::Generic(format!("GPU SSM scan failed: {e}")))
    }

    fn backend_name(&self) -> &str {
        "webgpu"
    }
}

impl std::fmt::Debug for WebGpuSsmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebGpuSsmBackend")
            .field("backend", &self.backend)
            .finish()
    }
}
