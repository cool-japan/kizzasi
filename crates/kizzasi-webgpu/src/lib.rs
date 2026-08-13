//! WebGPU acceleration backend for Kizzasi signal processing.
//!
//! This crate provides a `wgpu`-backed GPU compute layer for the Kizzasi
//! ecosystem.  By default it compiles to 100 % pure Rust with no GPU
//! dependencies. Activate the `webgpu` feature to unlock the actual GPU
//! operations:
//!
//! ```toml
//! [dependencies]
//! kizzasi-webgpu = { version = "0.2", features = ["webgpu"] }
//! ```
//!
//! # Architecture
//!
//! ```text
//! WebGpuBackend ──► wgpu::Device + wgpu::Queue + pipeline cache
//!      │
//!      ├── upload_f32()   →  GpuBuffer (STORAGE | COPY_SRC | COPY_DST)
//!      ├── download_f32() ←  GpuBuffer via staging (MAP_READ | COPY_DST)
//!      │
//!      ├── silu_gpu / rms_norm_gpu / matvec_gpu / ssm_scan_gpu   (&[f32] in, Vec<f32> out)
//!      └── *_gpu_buf variants                                    (GpuBuffer in and out)
//! ```
//!
//! The `*_gpu_buf` kernels keep intermediates resident on the device, so a
//! chain such as `matvec → silu → rms_norm` costs one upload and one download
//! instead of three round trips.
//!
//! # Error handling
//!
//! Every GPU entry point runs inside a wgpu error scope and validates its
//! request against the device limits first, so validation failures, allocation
//! failures and over-sized requests return [`WebGpuError`] instead of aborting
//! the process through wgpu's panicking default error handler.
//!
//! # Feature gate
//! Without `--features webgpu`, all entry points return
//! [`WebGpuError::BackendUnavailable`] so the crate is always usable as a
//! compile-time dependency in feature-gated codepaths.

pub mod backend;
pub mod buffer;
pub mod elementwise;
pub mod error;
pub mod matvec;
#[cfg(feature = "webgpu")]
pub(crate) mod pipeline;
pub mod ssm_backend;
pub mod ssm_scan;

pub use backend::{AdapterInfo, DeviceLimits, WebGpuBackend};
pub use buffer::{GpuBuffer, GpuBufferUsage};
pub use elementwise::{rms_norm_gpu, rms_norm_gpu_buf, silu_gpu, silu_gpu_buf, MAX_RMS_NORM_LEN};
pub use error::WebGpuError;
pub use matvec::{matvec_gpu, matvec_gpu_buf};
pub use ssm_backend::{WebGpuSsmBackend, DEFAULT_GPU_THRESHOLD};
pub use ssm_scan::{ssm_scan_gpu, ssm_scan_gpu_buf, MAX_SINGLE_PASS_LEN};

/// Convenience `Result` alias for this crate.
pub type WebGpuResult<T> = Result<T, WebGpuError>;

// ── Shared test helpers ───────────────────────────────────────────────────────

#[cfg(all(test, feature = "webgpu"))]
pub(crate) mod test_support {
    use crate::{WebGpuBackend, WebGpuError};

    /// Returns `true` when the environment demands a real GPU.
    ///
    /// Set `KIZZASI_REQUIRE_GPU=1` so a machine without an adapter fails the
    /// GPU tests instead of skipping them silently.
    fn require_gpu() -> bool {
        std::env::var("KIZZASI_REQUIRE_GPU")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    /// Obtain a backend, or `None` when this machine has no GPU adapter.
    pub(crate) async fn try_backend() -> Option<WebGpuBackend> {
        match WebGpuBackend::new().await {
            Ok(backend) => Some(backend),
            Err(WebGpuError::AdapterRequest(message)) => {
                assert!(
                    !require_gpu(),
                    "KIZZASI_REQUIRE_GPU is set but no GPU adapter is available: {message}"
                );
                eprintln!("no GPU adapter found — skipping GPU test");
                None
            }
            Err(err) => panic!("unexpected error creating WebGpuBackend: {err}"),
        }
    }

    /// Assert `got` matches `expected` within `tol`, scaled by magnitude.
    pub(crate) fn assert_slices_close(expected: &[f32], got: &[f32], tol: f32) {
        assert_eq!(expected.len(), got.len(), "length mismatch");
        for (index, (&want, &have)) in expected.iter().zip(got.iter()).enumerate() {
            assert_close(want, have, tol, index, "value");
        }
    }

    /// Assert two `(a, bu)` sequences match within `tol`.
    pub(crate) fn assert_pairs_close(expected: &[(f32, f32)], got: &[(f32, f32)], tol: f32) {
        assert_eq!(expected.len(), got.len(), "length mismatch");
        for (index, (&(want_a, want_bu), &(have_a, have_bu))) in
            expected.iter().zip(got.iter()).enumerate()
        {
            assert_close(want_a, have_a, tol, index, "a");
            assert_close(want_bu, have_bu, tol, index, "bu");
        }
    }

    fn assert_close(expected: f32, got: f32, tol: f32, index: usize, field: &str) {
        let scale = expected.abs().max(1.0);
        assert!(
            (expected - got).abs() <= tol * scale,
            "element {index} ({field}): expected {expected}, got {got} (tolerance {tol})"
        );
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that without the `webgpu` feature the backend reports itself
    /// as unavailable.  This test always compiles and always passes regardless
    /// of GPU availability.
    #[tokio::test]
    async fn test_backend_unavailable_without_feature() {
        let result = WebGpuBackend::new().await;

        #[cfg(not(feature = "webgpu"))]
        {
            assert!(
                matches!(result, Err(WebGpuError::BackendUnavailable)),
                "expected BackendUnavailable without webgpu feature, got: {result:?}"
            );
        }

        // With the feature enabled the constructor either succeeds (GPU present)
        // or fails with a different error (no adapter).  Either is acceptable here.
        #[cfg(feature = "webgpu")]
        {
            match result {
                Ok(_) | Err(WebGpuError::AdapterRequest(_)) => {}
                Err(e) => panic!("unexpected error with webgpu feature: {e}"),
            }
        }
    }

    /// `GpuBuffer::metadata_only` must exist in **every** feature
    /// configuration — enabling a feature may only add API, never remove it.
    #[test]
    fn test_gpu_buffer_metadata() {
        let buf = GpuBuffer::metadata_only(128, GpuBufferUsage::Storage, "test-buf");
        assert_eq!(buf.size_bytes, 128);
        assert_eq!(buf.usage, GpuBufferUsage::Storage);
        assert_eq!(buf.label, "test-buf");
        assert_eq!(buf.len_f32(), 32);
        assert!(
            !buf.is_gpu_resident(),
            "metadata-only buffers own no GPU allocation"
        );

        assert_ne!(GpuBufferUsage::Staging, GpuBufferUsage::Storage);
    }

    /// Verifies that `AdapterInfo` fields are accessible and cloneable.
    #[test]
    fn test_adapter_info_fields() {
        let info = AdapterInfo {
            name: "Test GPU".into(),
            backend: "Metal".into(),
            driver: "1.0".into(),
        };
        let cloned = info.clone();
        assert_eq!(info.name, cloned.name);
        assert_eq!(info.backend, cloned.backend);
        assert_eq!(info.driver, cloned.driver);
    }

    /// Error display messages compile and are non-empty.
    #[test]
    fn test_error_display() {
        let e = WebGpuError::BackendUnavailable;
        assert!(!e.to_string().is_empty());

        let e2 = WebGpuError::AdapterRequest("no GPU found".into());
        assert!(e2.to_string().contains("no GPU found"));

        let e3 = WebGpuError::BufferSizeMismatch {
            expected: 16,
            got: 13,
        };
        assert!(e3.to_string().contains("16"));
        assert!(e3.to_string().contains("13"));

        let e4 = WebGpuError::DeviceLimitExceeded {
            what: "storage binding".into(),
            required: 1024,
            limit: 512,
        };
        assert!(e4.to_string().contains("1024"));
        assert!(e4.to_string().contains("512"));

        let e5 = WebGpuError::NoGpuAllocation("ghost".into());
        assert!(e5.to_string().contains("ghost"));
    }
}

/// GPU round-trip integration tests — only compiled with `--features webgpu`.
#[cfg(all(test, feature = "webgpu"))]
mod integration_tests {
    use super::*;
    use crate::test_support::try_backend;

    /// Full upload → download round-trip test.
    ///
    /// Gracefully skips if no GPU adapter is available in the test environment.
    #[tokio::test]
    async fn test_upload_download_roundtrip() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let input: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, -1.0, f32::MAX, f32::MIN_POSITIVE];
        let buf = backend
            .upload_f32(&input, "roundtrip-test")
            .expect("upload_f32 failed");
        assert_eq!(buf.size_bytes, (input.len() * 4) as u64);
        assert_eq!(buf.usage, GpuBufferUsage::Storage);
        assert!(buf.is_gpu_resident());

        let output = backend.download_f32(&buf).expect("download_f32 failed");

        assert_eq!(input.len(), output.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bit-exact round-trip failed: {a} != {b}"
            );
        }
    }

    /// Verify that `submit_noop` succeeds when a GPU is available.
    #[tokio::test]
    async fn test_submit_noop() {
        let Some(backend) = try_backend().await else {
            return;
        };
        backend.submit_noop().expect("submit_noop failed");
    }

    /// Verify adapter info is non-empty when a GPU is available.
    #[tokio::test]
    async fn test_adapter_info_with_gpu() {
        let Some(backend) = try_backend().await else {
            return;
        };
        let info = backend.adapter_info();
        assert!(!info.name.is_empty(), "adapter name should not be empty");
        assert!(!info.backend.is_empty(), "backend name should not be empty");
    }
}
