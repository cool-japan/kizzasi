//! GPU-accelerated elementwise operations: SiLU activation and RMS Norm.
//!
//! # SiLU
//! Computes the Swish activation: `output[i] = input[i] / (1 + exp(-input[i]))`.
//! Any input length is supported, up to the device's storage-binding and
//! dispatch limits.
//!
//! # RMS Norm
//! Computes: `output[i] = (input[i] / rms(input)) * weight[i]`
//! where `rms(x) = sqrt(mean(x²) + eps)`.
//!
//! The RMS Norm uses a single-pass, single-work-group kernel with a parallel
//! tree reduction in shared memory. This limits it to at most
//! [`MAX_RMS_NORM_LEN`] elements; longer inputs return
//! [`WebGpuError::Other`] rather than a silently truncated result, so callers
//! can route them to a CPU path.
//!
//! # Feature gate
//! Without `--features webgpu`, all functions return
//! [`WebGpuError::BackendUnavailable`].

use crate::buffer::GpuBuffer;
use crate::{WebGpuBackend, WebGpuError};

#[cfg(feature = "webgpu")]
use crate::backend::{decode_f32, f32_slice_as_bytes, read_staging, F32_BYTES};
#[cfg(feature = "webgpu")]
use crate::buffer::GpuBufferUsage;
#[cfg(feature = "webgpu")]
use crate::pipeline::KernelKind;

/// Maximum input length supported by the GPU RMS Norm kernel.
///
/// The single-pass kernel uses one work-group of 256 threads.
pub const MAX_RMS_NORM_LEN: usize = 256;

/// Work-group size shared by both elementwise kernels.
#[cfg(feature = "webgpu")]
const WORKGROUP_SIZE: u32 = 256;

// ── Public API ────────────────────────────────────────────────────────────────

/// GPU SiLU activation: `output[i] = input[i] / (1 + exp(-input[i]))`.
///
/// # Errors
///
/// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
/// - [`WebGpuError::DeviceLimitExceeded`] if the input exceeds the device's
///   storage-binding or dispatch limits.
/// - [`WebGpuError::Other`] for wgpu pipeline or dispatch errors.
pub fn silu_gpu(backend: &WebGpuBackend, input: &[f32]) -> Result<Vec<f32>, WebGpuError> {
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = (backend, input);
        Err(WebGpuError::BackendUnavailable)
    }

    #[cfg(feature = "webgpu")]
    {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        silu_impl(backend, input)
    }
}

/// Device-resident variant of [`silu_gpu`].
///
/// Takes a GPU buffer and returns a new GPU buffer, so intermediates in a
/// `matvec → silu → rms_norm` chain never round-trip through host memory.
///
/// # Errors
///
/// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
/// - [`WebGpuError::NoGpuAllocation`] if `input` is metadata-only.
/// - [`WebGpuError::BufferSizeMismatch`] if `input` is empty or not a whole
///   number of `f32` values.
pub fn silu_gpu_buf(backend: &WebGpuBackend, input: &GpuBuffer) -> Result<GpuBuffer, WebGpuError> {
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = (backend, input);
        Err(WebGpuError::BackendUnavailable)
    }

    #[cfg(feature = "webgpu")]
    {
        silu_buf_impl(backend, input)
    }
}

/// GPU RMS Norm: `output[i] = (input[i] / rms(input)) * weight[i]`.
///
/// The single-pass GPU kernel supports at most [`MAX_RMS_NORM_LEN`] (256)
/// elements. For longer sequences use a CPU fallback.
///
/// # Errors
///
/// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
/// - [`WebGpuError::BufferSizeMismatch`] if `weight.len() != input.len()`.
/// - [`WebGpuError::Other`] if `input.len() > MAX_RMS_NORM_LEN` or for wgpu
///   pipeline / dispatch errors.
pub fn rms_norm_gpu(
    backend: &WebGpuBackend,
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, WebGpuError> {
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = (backend, input, weight, eps);
        Err(WebGpuError::BackendUnavailable)
    }

    #[cfg(feature = "webgpu")]
    {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        if weight.len() != input.len() {
            return Err(WebGpuError::BufferSizeMismatch {
                expected: input.len() as u64,
                got: weight.len() as u64,
            });
        }
        check_rms_norm_len(input.len())?;
        rms_norm_impl(backend, input, weight, eps)
    }
}

/// Device-resident variant of [`rms_norm_gpu`].
///
/// # Errors
///
/// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
/// - [`WebGpuError::NoGpuAllocation`] if either buffer is metadata-only.
/// - [`WebGpuError::BufferSizeMismatch`] if the buffers differ in size, are
///   empty, or are not a whole number of `f32` values.
/// - [`WebGpuError::Other`] if the element count exceeds [`MAX_RMS_NORM_LEN`].
pub fn rms_norm_gpu_buf(
    backend: &WebGpuBackend,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    eps: f32,
) -> Result<GpuBuffer, WebGpuError> {
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = (backend, input, weight, eps);
        Err(WebGpuError::BackendUnavailable)
    }

    #[cfg(feature = "webgpu")]
    {
        rms_norm_buf_impl(backend, input, weight, eps)
    }
}

// ── GPU implementation — only compiled with `--features webgpu` ──────────────

/// Reject inputs the single-work-group RMS kernel cannot reduce exactly.
#[cfg(feature = "webgpu")]
fn check_rms_norm_len(len: usize) -> Result<(), WebGpuError> {
    if len > MAX_RMS_NORM_LEN {
        return Err(WebGpuError::Other(format!(
            "rms_norm_gpu: input length {len} exceeds maximum {MAX_RMS_NORM_LEN} for \
             single-pass kernel; use a CPU fallback for larger inputs"
        )));
    }
    Ok(())
}

/// Number of `f32` values in `buf`, rejecting empty or misaligned buffers.
#[cfg(feature = "webgpu")]
fn buffer_element_count(buf: &GpuBuffer) -> Result<u32, WebGpuError> {
    let f32_bytes = F32_BYTES as u64;
    if buf.size_bytes == 0 || !buf.size_bytes.is_multiple_of(f32_bytes) {
        return Err(WebGpuError::BufferSizeMismatch {
            expected: buf.size_bytes.next_multiple_of(f32_bytes).max(f32_bytes),
            got: buf.size_bytes,
        });
    }
    u32::try_from(buf.size_bytes / f32_bytes).map_err(|_| WebGpuError::DeviceLimitExceeded {
        what: format!("buffer '{}' element count", buf.label),
        required: buf.size_bytes / f32_bytes,
        limit: u64::from(u32::MAX),
    })
}

/// Narrow a host element count to the `u32` the shader indexes with.
#[cfg(feature = "webgpu")]
fn element_count(len: usize, what: &str) -> Result<u32, WebGpuError> {
    u32::try_from(len).map_err(|_| WebGpuError::DeviceLimitExceeded {
        what: what.to_string(),
        required: len as u64,
        limit: u64::from(u32::MAX),
    })
}

/// Resources kept alive until the encoded command buffer is submitted.
#[cfg(feature = "webgpu")]
struct KernelResources {
    #[allow(dead_code, reason = "kept alive until the command buffer is submitted")]
    uniform: wgpu::Buffer,
    #[allow(dead_code, reason = "kept alive until the command buffer is submitted")]
    bind_group: wgpu::BindGroup,
}

/// Encode one SiLU dispatch over `n` elements.
#[cfg(feature = "webgpu")]
fn encode_silu(
    backend: &WebGpuBackend,
    encoder: &mut wgpu::CommandEncoder,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    n: u32,
) -> Result<KernelResources, WebGpuError> {
    let workgroups = n.div_ceil(WORKGROUP_SIZE);
    backend.check_workgroups("silu", workgroups)?;

    let (device, _queue) = backend.device_and_queue();
    let uniform = backend.create_uniform_buffer("silu-params", &n.to_ne_bytes());
    let pipeline = backend.pipeline(KernelKind::Silu)?;
    let bind_group = pipeline.bind_group(device, "silu-bg", &[&uniform, input, output]);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("silu-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    Ok(KernelResources {
        uniform,
        bind_group,
    })
}

/// Encode one RMS-norm dispatch over `n` elements (single work-group).
#[cfg(feature = "webgpu")]
fn encode_rms_norm(
    backend: &WebGpuBackend,
    encoder: &mut wgpu::CommandEncoder,
    input: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    output: &wgpu::Buffer,
    n: u32,
    eps: f32,
) -> Result<KernelResources, WebGpuError> {
    let mut params = [0u8; 8];
    params[0..4].copy_from_slice(&n.to_ne_bytes());
    params[4..8].copy_from_slice(&eps.to_ne_bytes());

    let (device, _queue) = backend.device_and_queue();
    let uniform = backend.create_uniform_buffer("rms-norm-params", &params);
    let pipeline = backend.pipeline(KernelKind::RmsNorm)?;
    let bind_group = pipeline.bind_group(device, "rms-norm-bg", &[&uniform, input, weight, output]);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rms-norm-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // The tree reduction requires the whole input in one work-group.
        cpass.dispatch_workgroups(1, 1, 1);
    }

    Ok(KernelResources {
        uniform,
        bind_group,
    })
}

#[cfg(feature = "webgpu")]
fn silu_impl(backend: &WebGpuBackend, input: &[f32]) -> Result<Vec<f32>, WebGpuError> {
    let n = element_count(input.len(), "silu input length")?;
    let bytes = f32_slice_as_bytes(input);
    let size_bytes = bytes.len() as u64;

    backend.run_scoped(|| {
        let input_buf = backend.upload_bytes("silu-input", bytes)?;
        let output_buf = backend.create_storage_buffer("silu-output", size_bytes)?;
        let staging_buf = backend.create_staging_buffer("silu-staging", size_bytes)?;

        let (device, queue) = backend.device_and_queue();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("silu-encoder"),
        });

        let _resources = encode_silu(backend, &mut encoder, &input_buf, &output_buf, n)?;

        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, size_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        read_staging(device, &staging_buf, size_bytes, decode_f32)
    })
}

#[cfg(feature = "webgpu")]
fn silu_buf_impl(backend: &WebGpuBackend, input: &GpuBuffer) -> Result<GpuBuffer, WebGpuError> {
    let source = input.wgpu_buffer()?;
    let n = buffer_element_count(input)?;
    let size_bytes = input.size_bytes;

    backend.run_scoped(|| {
        let output_buf = backend.create_storage_buffer("silu-output", size_bytes)?;

        let (device, queue) = backend.device_and_queue();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("silu-buf-encoder"),
        });

        let _resources = encode_silu(backend, &mut encoder, source, &output_buf, n)?;
        queue.submit(std::iter::once(encoder.finish()));

        Ok(GpuBuffer::from_wgpu(
            output_buf,
            size_bytes,
            GpuBufferUsage::Storage,
            "silu-output",
        ))
    })
}

#[cfg(feature = "webgpu")]
fn rms_norm_impl(
    backend: &WebGpuBackend,
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, WebGpuError> {
    let n = element_count(input.len(), "rms norm input length")?;
    let input_bytes = f32_slice_as_bytes(input);
    let weight_bytes = f32_slice_as_bytes(weight);
    let size_bytes = input_bytes.len() as u64;

    backend.run_scoped(|| {
        let input_buf = backend.upload_bytes("rms-norm-input", input_bytes)?;
        let weight_buf = backend.upload_bytes("rms-norm-weight", weight_bytes)?;
        let output_buf = backend.create_storage_buffer("rms-norm-output", size_bytes)?;
        let staging_buf = backend.create_staging_buffer("rms-norm-staging", size_bytes)?;

        let (device, queue) = backend.device_and_queue();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rms-norm-encoder"),
        });

        let _resources = encode_rms_norm(
            backend,
            &mut encoder,
            &input_buf,
            &weight_buf,
            &output_buf,
            n,
            eps,
        )?;

        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, size_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        read_staging(device, &staging_buf, size_bytes, decode_f32)
    })
}

#[cfg(feature = "webgpu")]
fn rms_norm_buf_impl(
    backend: &WebGpuBackend,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    eps: f32,
) -> Result<GpuBuffer, WebGpuError> {
    let input_source = input.wgpu_buffer()?;
    let weight_source = weight.wgpu_buffer()?;

    if input.size_bytes != weight.size_bytes {
        return Err(WebGpuError::BufferSizeMismatch {
            expected: input.size_bytes,
            got: weight.size_bytes,
        });
    }

    let n = buffer_element_count(input)?;
    check_rms_norm_len(n as usize)?;
    let size_bytes = input.size_bytes;

    backend.run_scoped(|| {
        let output_buf = backend.create_storage_buffer("rms-norm-output", size_bytes)?;

        let (device, queue) = backend.device_and_queue();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rms-norm-buf-encoder"),
        });

        let _resources = encode_rms_norm(
            backend,
            &mut encoder,
            input_source,
            weight_source,
            &output_buf,
            n,
            eps,
        )?;
        queue.submit(std::iter::once(encoder.finish()));

        Ok(GpuBuffer::from_wgpu(
            output_buf,
            size_bytes,
            GpuBufferUsage::Storage,
            "rms-norm-output",
        ))
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Without the `webgpu` feature, `silu_gpu` must return `BackendUnavailable`
    /// from the function itself.
    #[cfg(not(feature = "webgpu"))]
    #[test]
    fn test_silu_unavailable_without_feature() {
        let backend = crate::backend::unavailable_backend();
        let result = silu_gpu(&backend, &[1.0, 2.0, 3.0]);
        assert!(
            matches!(result, Err(WebGpuError::BackendUnavailable)),
            "expected BackendUnavailable, got: {result:?}"
        );
    }

    /// Without the `webgpu` feature, `rms_norm_gpu` must return `BackendUnavailable`.
    #[cfg(not(feature = "webgpu"))]
    #[test]
    fn test_rms_norm_unavailable_without_feature() {
        let backend = crate::backend::unavailable_backend();
        let result = rms_norm_gpu(&backend, &[1.0, 2.0], &[1.0, 1.0], 1e-6);
        assert!(
            matches!(result, Err(WebGpuError::BackendUnavailable)),
            "expected BackendUnavailable, got: {result:?}"
        );
    }

    /// The device-resident variants must be unavailable too.
    #[cfg(not(feature = "webgpu"))]
    #[test]
    fn test_buf_variants_unavailable_without_feature() {
        let backend = crate::backend::unavailable_backend();
        let buf = GpuBuffer::metadata_only(16, crate::GpuBufferUsage::Storage, "in");
        assert!(matches!(
            silu_gpu_buf(&backend, &buf),
            Err(WebGpuError::BackendUnavailable)
        ));
        assert!(matches!(
            rms_norm_gpu_buf(&backend, &buf, &buf, 1e-6),
            Err(WebGpuError::BackendUnavailable)
        ));
    }

    /// `rms_norm_gpu` must return `BufferSizeMismatch` when weight length differs.
    #[tokio::test]
    async fn test_rms_norm_weight_length_check() {
        let input = vec![1.0_f32; 4];
        let weight = vec![1.0_f32; 3]; // wrong length
        match WebGpuBackend::new().await {
            Ok(backend) => {
                let err = rms_norm_gpu(&backend, &input, &weight, 1e-6)
                    .expect_err("should fail on weight length mismatch");
                assert!(
                    matches!(
                        err,
                        WebGpuError::BufferSizeMismatch {
                            expected: 4,
                            got: 3
                        }
                    ),
                    "unexpected error: {err:?}"
                );
            }
            Err(_) => {
                // No GPU; the length check is a pure logic test.
            }
        }
    }

    /// `rms_norm_gpu` must return `Other` when input exceeds `MAX_RMS_NORM_LEN`.
    #[tokio::test]
    async fn test_rms_norm_length_limit() {
        let n = MAX_RMS_NORM_LEN + 1;
        let input = vec![1.0_f32; n];
        let weight = vec![1.0_f32; n];
        match WebGpuBackend::new().await {
            Ok(backend) => {
                let err = rms_norm_gpu(&backend, &input, &weight, 1e-6)
                    .expect_err("should fail when length exceeds limit");
                assert!(
                    matches!(err, WebGpuError::Other(_)),
                    "unexpected error: {err:?}"
                );
            }
            Err(_) => {
                // No GPU; length check is pure logic.
            }
        }
    }
}

#[cfg(all(test, feature = "webgpu"))]
mod gpu_tests {
    use super::*;
    use crate::test_support::{assert_slices_close, try_backend};

    fn silu_reference(input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
    }

    fn rms_norm_reference(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let rms = ((sum_sq / input.len() as f32) + eps).sqrt();
        input
            .iter()
            .zip(weight.iter())
            .map(|(&x, &w)| (x / rms) * w)
            .collect()
    }

    /// Known SiLU values: silu(0) ≈ 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689.
    #[tokio::test]
    async fn test_silu_known_values() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let input = [0.0_f32, 1.0, -1.0];
        let result = silu_gpu(&backend, &input).expect("GPU silu failed");

        assert_eq!(result.len(), 3);
        assert_slices_close(&silu_reference(&input), &result, 1e-5);
    }

    /// SiLU on 64 values: max absolute diff vs CPU reference must be < 1e-5.
    #[tokio::test]
    async fn test_silu_matches_cpu() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let input: Vec<f32> = (0..64)
            .map(|i| (i as f32 - 32.0) / 8.0) // range [-4, 3.875]
            .collect();

        let gpu = silu_gpu(&backend, &input).expect("GPU silu failed");
        assert_eq!(gpu.len(), 64);
        assert_slices_close(&silu_reference(&input), &gpu, 1e-5);
    }

    /// SiLU across the multi-work-group boundary (n = 255/256/257/1000).
    #[tokio::test]
    async fn test_silu_multi_workgroup_lengths() {
        let Some(backend) = try_backend().await else {
            return;
        };

        for n in [255_usize, 256, 257, 1000] {
            let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) / 32.0).collect();
            let gpu = silu_gpu(&backend, &input).expect("GPU silu failed");
            assert_eq!(gpu.len(), n, "length mismatch at n={n}");
            assert_slices_close(&silu_reference(&input), &gpu, 1e-5);
        }
    }

    /// Calling a kernel repeatedly must reuse the cached pipeline and keep
    /// returning identical results.
    #[tokio::test]
    async fn test_silu_repeated_calls_are_stable() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let input: Vec<f32> = (0..300).map(|i| (i as f32 - 150.0) / 40.0).collect();
        let first = silu_gpu(&backend, &input).expect("first silu failed");
        for _ in 0..3 {
            let again = silu_gpu(&backend, &input).expect("repeat silu failed");
            assert_eq!(first, again, "cached pipeline produced a different result");
        }
    }

    /// Uniform input [1,…,1] with uniform weights [1,…,1] → output ≈ [1,…,1].
    #[tokio::test]
    async fn test_rms_norm_uniform_input() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let n = 16usize;
        let input = vec![1.0_f32; n];
        let weight = vec![1.0_f32; n];

        let result = rms_norm_gpu(&backend, &input, &weight, 1e-6).expect("GPU rms_norm failed");

        assert_eq!(result.len(), n);
        assert_slices_close(&vec![1.0_f32; n], &result, 1e-4);
    }

    /// RMS Norm: compare GPU output to CPU reference for 32 and 256 values.
    #[tokio::test]
    async fn test_rms_norm_matches_cpu() {
        let Some(backend) = try_backend().await else {
            return;
        };

        for n in [32_usize, MAX_RMS_NORM_LEN] {
            let input: Vec<f32> = (0..n).map(|i| ((i * 3 + 1) % 7) as f32 / 3.0).collect();
            let weight: Vec<f32> = (0..n)
                .map(|i| 0.5 + ((i * 5 + 2) % 4) as f32 / 4.0)
                .collect();
            let eps = 1e-5_f32;

            let gpu = rms_norm_gpu(&backend, &input, &weight, eps).expect("GPU rms_norm failed");
            assert_eq!(gpu.len(), n);
            assert_slices_close(&rms_norm_reference(&input, &weight, eps), &gpu, 1e-4);
        }
    }

    /// One element past the kernel's capacity must be a typed error, never a
    /// truncated result.
    #[tokio::test]
    async fn test_rms_norm_boundary_error() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let n = MAX_RMS_NORM_LEN + 1;
        let input = vec![0.5_f32; n];
        let weight = vec![1.0_f32; n];
        let err = rms_norm_gpu(&backend, &input, &weight, 1e-6).expect_err("must reject n=257");
        assert!(matches!(err, WebGpuError::Other(_)), "got: {err:?}");
    }

    /// Device-resident SiLU must match the host-slice variant bit for bit.
    #[tokio::test]
    async fn test_silu_gpu_buf_matches_slice_variant() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let input: Vec<f32> = (0..500).map(|i| (i as f32 - 250.0) / 50.0).collect();
        let expected = silu_gpu(&backend, &input).expect("slice silu failed");

        let uploaded = backend.upload_f32(&input, "silu-buf-in").expect("upload");
        let resident = silu_gpu_buf(&backend, &uploaded).expect("resident silu failed");
        let downloaded = backend.download_f32(&resident).expect("download");

        assert_eq!(expected, downloaded);
    }

    /// Device-resident RMS norm must match the host-slice variant.
    #[tokio::test]
    async fn test_rms_norm_gpu_buf_matches_slice_variant() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let n = 128usize;
        let input: Vec<f32> = (0..n).map(|i| ((i * 3 + 1) % 7) as f32 / 3.0).collect();
        let weight: Vec<f32> = (0..n).map(|i| 0.5 + (i % 4) as f32 / 4.0).collect();
        let eps = 1e-5_f32;

        let expected = rms_norm_gpu(&backend, &input, &weight, eps).expect("slice rms failed");

        let input_buf = backend.upload_f32(&input, "rms-in").expect("upload input");
        let weight_buf = backend.upload_f32(&weight, "rms-w").expect("upload weight");
        let resident =
            rms_norm_gpu_buf(&backend, &input_buf, &weight_buf, eps).expect("resident rms failed");
        let downloaded = backend.download_f32(&resident).expect("download");

        assert_eq!(expected, downloaded);
    }

    /// A metadata-only buffer must be rejected, not dereferenced.
    #[tokio::test]
    async fn test_silu_gpu_buf_rejects_metadata_only() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let buf = GpuBuffer::metadata_only(16, GpuBufferUsage::Storage, "ghost");
        let err = silu_gpu_buf(&backend, &buf).expect_err("metadata-only must fail");
        assert!(
            matches!(err, WebGpuError::NoGpuAllocation(_)),
            "got: {err:?}"
        );
    }

    /// Chaining kernels on the device must equal chaining them on the host.
    #[tokio::test]
    async fn test_device_resident_chain() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let n = 64usize;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) / 16.0).collect();
        let weight = vec![1.0_f32; n];
        let eps = 1e-6_f32;

        let host_silu = silu_gpu(&backend, &input).expect("host silu");
        let host_chain = rms_norm_gpu(&backend, &host_silu, &weight, eps).expect("host rms");

        let input_buf = backend.upload_f32(&input, "chain-in").expect("upload");
        let weight_buf = backend.upload_f32(&weight, "chain-w").expect("upload");
        let silu_buf = silu_gpu_buf(&backend, &input_buf).expect("resident silu");
        let rms_buf =
            rms_norm_gpu_buf(&backend, &silu_buf, &weight_buf, eps).expect("resident rms");
        let device_chain = backend.download_f32(&rms_buf).expect("download");

        assert_eq!(host_chain, device_chain);
    }
}
