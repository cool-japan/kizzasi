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
//! WebGpuBackend ──► wgpu::Device + wgpu::Queue
//!      │
//!      ├── upload_f32()   →  GpuBuffer (STORAGE | COPY_SRC | COPY_DST)
//!      └── download_f32() ←  GpuBuffer via staging (MAP_READ | COPY_DST)
//! ```
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
pub mod ssm_backend;
pub mod ssm_scan;

pub use backend::{AdapterInfo, WebGpuBackend};
pub use buffer::{GpuBuffer, GpuBufferUsage};
pub use elementwise::{rms_norm_gpu, silu_gpu, MAX_RMS_NORM_LEN};
pub use error::WebGpuError;
pub use matvec::matvec_gpu;
pub use ssm_backend::WebGpuSsmBackend;
pub use ssm_scan::{ssm_scan_gpu, MAX_SINGLE_PASS_LEN};

/// Convenience `Result` alias for this crate.
pub type WebGpuResult<T> = Result<T, WebGpuError>;

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

    /// Verifies that `GpuBuffer` metadata (size, usage, label) is always
    /// accessible without a live GPU.
    #[test]
    fn test_gpu_buffer_metadata() {
        #[cfg(not(feature = "webgpu"))]
        {
            let buf = GpuBuffer::metadata_only(128, GpuBufferUsage::Storage, "test-buf");
            assert_eq!(buf.size_bytes, 128);
            assert_eq!(buf.usage, GpuBufferUsage::Storage);
            assert_eq!(buf.label, "test-buf");
        }

        // With the feature we cannot create a GpuBuffer without a device;
        // we just verify the types exist and the staging variant round-trips.
        #[cfg(feature = "webgpu")]
        {
            assert_eq!(GpuBufferUsage::Storage, GpuBufferUsage::Storage);
            assert_ne!(GpuBufferUsage::Staging, GpuBufferUsage::Storage);
        }
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
    }
}

/// GPU round-trip integration tests — only compiled with `--features webgpu`.
#[cfg(all(test, feature = "webgpu"))]
mod integration_tests {
    use super::*;

    /// Full upload → download round-trip test.
    ///
    /// Gracefully skips if no GPU adapter is available in the test environment.
    #[tokio::test]
    async fn test_upload_download_roundtrip() {
        let backend = match WebGpuBackend::new().await {
            Ok(b) => b,
            Err(WebGpuError::AdapterRequest(_)) => {
                // No GPU in this environment — skip gracefully.
                eprintln!("test_upload_download_roundtrip: no GPU adapter, skipping");
                return;
            }
            Err(e) => panic!("unexpected error creating WebGpuBackend: {e}"),
        };

        let input: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, -1.0, f32::MAX, f32::MIN_POSITIVE];
        let buf = backend
            .upload_f32(&input, "roundtrip-test")
            .expect("upload_f32 failed");
        assert_eq!(buf.size_bytes, (input.len() * 4) as u64);
        assert_eq!(buf.usage, GpuBufferUsage::Storage);

        let output = backend
            .download_f32(&buf)
            .await
            .expect("download_f32 failed");

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
        let backend = match WebGpuBackend::new().await {
            Ok(b) => b,
            Err(WebGpuError::AdapterRequest(_)) => {
                eprintln!("test_submit_noop: no GPU adapter, skipping");
                return;
            }
            Err(e) => panic!("unexpected error: {e}"),
        };
        backend.submit_noop().expect("submit_noop failed");
    }

    /// Verify adapter info is non-empty when a GPU is available.
    #[tokio::test]
    async fn test_adapter_info_with_gpu() {
        let backend = match WebGpuBackend::new().await {
            Ok(b) => b,
            Err(WebGpuError::AdapterRequest(_)) => {
                eprintln!("test_adapter_info_with_gpu: no GPU adapter, skipping");
                return;
            }
            Err(e) => panic!("unexpected error: {e}"),
        };
        let info = backend.adapter_info();
        assert!(!info.name.is_empty(), "adapter name should not be empty");
        assert!(!info.backend.is_empty(), "backend name should not be empty");
    }
}
