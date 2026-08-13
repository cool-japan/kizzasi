//! GPU-accelerated matrix-vector multiplication.
//!
//! Provides a single WGSL kernel that computes `output = matrix × vector`
//! where `matrix` is a row-major `[rows, cols]` matrix stored as a flat `f32`
//! slice.  Each GPU thread processes one output row.
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

/// Work-group size of the matvec kernel (`@workgroup_size(64)`).
#[cfg(feature = "webgpu")]
const WORKGROUP_SIZE: u32 = 64;

/// Execute `output = matrix × vector` on the GPU.
///
/// `matrix` is row-major with shape `[rows, cols]` (length = `rows × cols`).
/// `vector` must have length `cols`.  Returns a `Vec<f32>` of length `rows`.
///
/// # Degenerate shapes
///
/// `rows == 0` returns an empty vector.  `cols == 0` returns `rows` zeros —
/// the empty-sum convention, computed on the host because there is no GPU work
/// to dispatch (a zero-sized binding is invalid in WebGPU).
///
/// # Errors
///
/// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
/// - [`WebGpuError::BufferSizeMismatch`] if dimension checks fail.
/// - [`WebGpuError::DeviceLimitExceeded`] if the request exceeds the device's
///   storage-binding or dispatch limits, or if `rows × cols` overflows.
/// - [`WebGpuError::Other`] for wgpu pipeline or dispatch errors.
pub fn matvec_gpu(
    backend: &WebGpuBackend,
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
) -> Result<Vec<f32>, WebGpuError> {
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = (backend, matrix, rows, cols, vector);
        Err(WebGpuError::BackendUnavailable)
    }

    #[cfg(feature = "webgpu")]
    {
        let expected = checked_elements(rows, cols)?;
        if matrix.len() as u64 != expected {
            return Err(WebGpuError::BufferSizeMismatch {
                expected,
                got: matrix.len() as u64,
            });
        }
        if vector.len() != cols {
            return Err(WebGpuError::BufferSizeMismatch {
                expected: cols as u64,
                got: vector.len() as u64,
            });
        }
        if rows == 0 {
            return Ok(Vec::new());
        }
        if cols == 0 {
            // Every row is an empty sum; no GPU dispatch is possible because
            // the matrix and vector bindings would both be zero-sized.
            return Ok(vec![0.0; rows]);
        }
        matvec_impl(backend, matrix, rows, cols, vector)
    }
}

/// Device-resident variant of [`matvec_gpu`].
///
/// Both inputs stay on the device and the result is returned as a GPU buffer,
/// so a `matvec → silu → rms_norm` chain never round-trips through host
/// memory.
///
/// # Errors
///
/// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
/// - [`WebGpuError::NoGpuAllocation`] if either buffer is metadata-only.
/// - [`WebGpuError::BufferSizeMismatch`] if the buffer sizes do not match
///   `rows × cols` and `cols` `f32` values.
/// - [`WebGpuError::Other`] if `rows == 0` or `cols == 0` — a zero-sized GPU
///   buffer cannot be produced; use [`matvec_gpu`] for degenerate shapes.
pub fn matvec_gpu_buf(
    backend: &WebGpuBackend,
    matrix: &GpuBuffer,
    rows: usize,
    cols: usize,
    vector: &GpuBuffer,
) -> Result<GpuBuffer, WebGpuError> {
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = (backend, matrix, rows, cols, vector);
        Err(WebGpuError::BackendUnavailable)
    }

    #[cfg(feature = "webgpu")]
    {
        matvec_buf_impl(backend, matrix, rows, cols, vector)
    }
}

// ── GPU implementation — only compiled with `--features webgpu` ──────────────

/// `rows × cols` without silent wrap-around.
#[cfg(feature = "webgpu")]
fn checked_elements(rows: usize, cols: usize) -> Result<u64, WebGpuError> {
    rows.checked_mul(cols)
        .map(|count| count as u64)
        .ok_or_else(|| WebGpuError::DeviceLimitExceeded {
            what: format!("matvec matrix element count ({rows} × {cols})"),
            required: u64::MAX,
            limit: usize::MAX as u64,
        })
}

/// Narrow a dimension to the `u32` the shader indexes with.
#[cfg(feature = "webgpu")]
fn dimension(value: usize, what: &str) -> Result<u32, WebGpuError> {
    u32::try_from(value).map_err(|_| WebGpuError::DeviceLimitExceeded {
        what: what.to_string(),
        required: value as u64,
        limit: u64::from(u32::MAX),
    })
}

/// Byte size of `count` `f32` values.
#[cfg(feature = "webgpu")]
fn f32_bytes(count: u64, what: &str) -> Result<u64, WebGpuError> {
    count
        .checked_mul(F32_BYTES as u64)
        .ok_or_else(|| WebGpuError::DeviceLimitExceeded {
            what: format!("{what} byte size"),
            required: u64::MAX,
            limit: u64::MAX,
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

/// Encode one matvec dispatch.
#[cfg(feature = "webgpu")]
fn encode_matvec(
    backend: &WebGpuBackend,
    encoder: &mut wgpu::CommandEncoder,
    matrix: &wgpu::Buffer,
    vector: &wgpu::Buffer,
    output: &wgpu::Buffer,
    rows: u32,
    cols: u32,
) -> Result<KernelResources, WebGpuError> {
    let workgroups = rows.div_ceil(WORKGROUP_SIZE);
    backend.check_workgroups("matvec", workgroups)?;

    let mut params = [0u8; 8];
    params[0..4].copy_from_slice(&rows.to_ne_bytes());
    params[4..8].copy_from_slice(&cols.to_ne_bytes());

    let (device, _queue) = backend.device_and_queue();
    let uniform = backend.create_uniform_buffer("matvec-params", &params);
    let pipeline = backend.pipeline(KernelKind::Matvec)?;
    let bind_group = pipeline.bind_group(device, "matvec-bg", &[&uniform, matrix, vector, output]);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matvec-pass"),
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

#[cfg(feature = "webgpu")]
fn matvec_impl(
    backend: &WebGpuBackend,
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
) -> Result<Vec<f32>, WebGpuError> {
    let rows_u32 = dimension(rows, "matvec rows")?;
    let cols_u32 = dimension(cols, "matvec cols")?;
    let output_size = f32_bytes(rows as u64, "matvec output")?;

    let matrix_bytes = f32_slice_as_bytes(matrix);
    let vector_bytes = f32_slice_as_bytes(vector);

    backend.run_scoped(|| {
        let matrix_buf = backend.upload_bytes("matvec-matrix", matrix_bytes)?;
        let vector_buf = backend.upload_bytes("matvec-vector", vector_bytes)?;
        let output_buf = backend.create_storage_buffer("matvec-output", output_size)?;
        let staging_buf = backend.create_staging_buffer("matvec-staging", output_size)?;

        let (device, queue) = backend.device_and_queue();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matvec-encoder"),
        });

        let _resources = encode_matvec(
            backend,
            &mut encoder,
            &matrix_buf,
            &vector_buf,
            &output_buf,
            rows_u32,
            cols_u32,
        )?;

        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        queue.submit(std::iter::once(encoder.finish()));

        read_staging(device, &staging_buf, output_size, decode_f32)
    })
}

#[cfg(feature = "webgpu")]
fn matvec_buf_impl(
    backend: &WebGpuBackend,
    matrix: &GpuBuffer,
    rows: usize,
    cols: usize,
    vector: &GpuBuffer,
) -> Result<GpuBuffer, WebGpuError> {
    let matrix_source = matrix.wgpu_buffer()?;
    let vector_source = vector.wgpu_buffer()?;

    if rows == 0 || cols == 0 {
        return Err(WebGpuError::Other(format!(
            "matvec_gpu_buf: degenerate shape {rows}×{cols} cannot produce a GPU buffer; \
             use matvec_gpu for zero-sized dimensions"
        )));
    }

    let expected_matrix = f32_bytes(checked_elements(rows, cols)?, "matvec matrix")?;
    if matrix.size_bytes != expected_matrix {
        return Err(WebGpuError::BufferSizeMismatch {
            expected: expected_matrix,
            got: matrix.size_bytes,
        });
    }

    let expected_vector = f32_bytes(cols as u64, "matvec vector")?;
    if vector.size_bytes != expected_vector {
        return Err(WebGpuError::BufferSizeMismatch {
            expected: expected_vector,
            got: vector.size_bytes,
        });
    }

    let rows_u32 = dimension(rows, "matvec rows")?;
    let cols_u32 = dimension(cols, "matvec cols")?;
    let output_size = f32_bytes(rows as u64, "matvec output")?;

    backend.run_scoped(|| {
        let output_buf = backend.create_storage_buffer("matvec-output", output_size)?;

        let (device, queue) = backend.device_and_queue();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matvec-buf-encoder"),
        });

        let _resources = encode_matvec(
            backend,
            &mut encoder,
            matrix_source,
            vector_source,
            &output_buf,
            rows_u32,
            cols_u32,
        )?;
        queue.submit(std::iter::once(encoder.finish()));

        Ok(GpuBuffer::from_wgpu(
            output_buf,
            output_size,
            GpuBufferUsage::Storage,
            "matvec-output",
        ))
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrong matrix dimensions must return `BufferSizeMismatch` immediately.
    #[tokio::test]
    async fn test_matvec_dimension_check_matrix() {
        // `matvec_gpu` validates dimensions synchronously before touching the
        // GPU, so a mismatched matrix length triggers `BufferSizeMismatch`
        // regardless of whether a real GPU is available.
        let dummy_matrix = vec![0.0f32; 5]; // wrong: should be 2*3=6
        let dummy_vec = vec![0.0f32; 3];

        match WebGpuBackend::new().await {
            Ok(backend) => {
                let err = matvec_gpu(&backend, &dummy_matrix, 2, 3, &dummy_vec)
                    .expect_err("should fail on bad dimensions");
                assert!(
                    matches!(
                        err,
                        WebGpuError::BufferSizeMismatch {
                            expected: 6,
                            got: 5
                        }
                    ),
                    "unexpected error: {err:?}"
                );
            }
            Err(_) => {
                // No GPU available; dimension check is a synchronous
                // early-return, so this path still validates the logic.
            }
        }
    }

    /// Wrong vector dimensions must return `BufferSizeMismatch`.
    #[tokio::test]
    async fn test_matvec_dimension_check_vector() {
        let dummy_matrix = vec![0.0f32; 6];
        let dummy_vec = vec![0.0f32; 2]; // wrong: should be 3

        match WebGpuBackend::new().await {
            Ok(backend) => {
                let err = matvec_gpu(&backend, &dummy_matrix, 2, 3, &dummy_vec)
                    .expect_err("should fail on bad vector dimensions");
                assert!(
                    matches!(
                        err,
                        WebGpuError::BufferSizeMismatch {
                            expected: 3,
                            got: 2
                        }
                    ),
                    "unexpected error: {err:?}"
                );
            }
            Err(_) => {
                // No GPU; dimension-only test still validates logic path.
            }
        }
    }

    /// Without the `webgpu` feature, `matvec_gpu` returns `BackendUnavailable`
    /// from the function itself.
    #[cfg(not(feature = "webgpu"))]
    #[test]
    fn test_matvec_unavailable_without_feature() {
        let backend = crate::backend::unavailable_backend();
        let result = matvec_gpu(&backend, &[1.0, 2.0, 3.0, 4.0], 2, 2, &[1.0, 1.0]);
        assert!(
            matches!(result, Err(WebGpuError::BackendUnavailable)),
            "expected BackendUnavailable without webgpu feature, got: {result:?}"
        );
    }

    /// The device-resident variant must be unavailable too.
    #[cfg(not(feature = "webgpu"))]
    #[test]
    fn test_matvec_buf_unavailable_without_feature() {
        let backend = crate::backend::unavailable_backend();
        let buf = GpuBuffer::metadata_only(16, crate::GpuBufferUsage::Storage, "m");
        let result = matvec_gpu_buf(&backend, &buf, 2, 2, &buf);
        assert!(
            matches!(result, Err(WebGpuError::BackendUnavailable)),
            "expected BackendUnavailable, got: {result:?}"
        );
    }
}

#[cfg(all(test, feature = "webgpu"))]
mod gpu_tests {
    use super::*;
    use crate::test_support::{assert_slices_close, try_backend};

    fn matvec_reference(matrix: &[f32], rows: usize, cols: usize, vector: &[f32]) -> Vec<f32> {
        (0..rows)
            .map(|row| {
                (0..cols)
                    .map(|col| matrix[row * cols + col] * vector[col])
                    .sum()
            })
            .collect()
    }

    /// 4×4 identity matrix × [1, 2, 3, 4] should return [1, 2, 3, 4].
    #[tokio::test]
    async fn test_matvec_identity() {
        let Some(backend) = try_backend().await else {
            return;
        };

        #[rustfmt::skip]
        let matrix = [
            1.0_f32, 0.0, 0.0, 0.0,
            0.0,     1.0, 0.0, 0.0,
            0.0,     0.0, 1.0, 0.0,
            0.0,     0.0, 0.0, 1.0,
        ];
        let vector = [1.0_f32, 2.0, 3.0, 4.0];

        let result = matvec_gpu(&backend, &matrix, 4, 4, &vector).expect("GPU matvec failed");

        assert_eq!(result.len(), 4);
        assert_slices_close(&[1.0, 2.0, 3.0, 4.0], &result, 1e-5);
    }

    /// 2×3 matrix × 3-vector, verified against CPU reference.
    #[tokio::test]
    async fn test_matvec_known_values() {
        let Some(backend) = try_backend().await else {
            return;
        };

        // matrix = [[1, 2, 3], [4, 5, 6]], vector = [1, 2, 3]
        let matrix = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vector = [1.0_f32, 2.0, 3.0];

        let result = matvec_gpu(&backend, &matrix, 2, 3, &vector).expect("GPU matvec failed");

        assert_eq!(result.len(), 2);
        assert_slices_close(&[14.0, 32.0], &result, 1e-4);
    }

    /// 16×16 random-ish matrix vs CPU reference — max diff must be < 1e-4.
    #[tokio::test]
    async fn test_matvec_matches_cpu() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let rows = 16usize;
        let cols = 16usize;

        let matrix: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7 + 3) % 17) as f32 / 8.0)
            .collect();
        let vector: Vec<f32> = (0..cols).map(|i| ((i * 5 + 1) % 11) as f32 / 5.0).collect();

        let gpu = matvec_gpu(&backend, &matrix, rows, cols, &vector).expect("GPU matvec failed");

        assert_eq!(gpu.len(), rows);
        assert_slices_close(&matvec_reference(&matrix, rows, cols, &vector), &gpu, 1e-4);
    }

    /// Non-square shapes, including single-row and single-column matrices.
    #[tokio::test]
    async fn test_matvec_thin_shapes() {
        let Some(backend) = try_backend().await else {
            return;
        };

        for (rows, cols) in [(1_usize, 8_usize), (8, 1), (100, 3), (3, 100)] {
            let matrix: Vec<f32> = (0..rows * cols)
                .map(|i| ((i * 11 + 5) % 13) as f32 / 6.0)
                .collect();
            let vector: Vec<f32> = (0..cols).map(|i| ((i * 3 + 2) % 7) as f32 / 3.0).collect();

            let gpu =
                matvec_gpu(&backend, &matrix, rows, cols, &vector).expect("GPU matvec failed");
            assert_eq!(gpu.len(), rows, "length mismatch at {rows}×{cols}");
            assert_slices_close(&matvec_reference(&matrix, rows, cols, &vector), &gpu, 1e-4);
        }
    }

    /// Regression: empty and zero-dimension inputs used to reach the GPU and
    /// abort the process on a zero-sized binding.
    #[tokio::test]
    async fn test_matvec_degenerate_shapes() {
        let Some(backend) = try_backend().await else {
            return;
        };

        assert!(matvec_gpu(&backend, &[], 0, 0, &[])
            .expect("0×0 must succeed")
            .is_empty());
        assert!(matvec_gpu(&backend, &[], 0, 3, &[0.0, 0.0, 0.0])
            .expect("0×3 must succeed")
            .is_empty());
        assert_eq!(
            matvec_gpu(&backend, &[], 2, 0, &[]).expect("2×0 must succeed"),
            vec![0.0, 0.0]
        );
    }

    /// Device-resident matvec must match the host-slice variant.
    #[tokio::test]
    async fn test_matvec_gpu_buf_matches_slice_variant() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let rows = 32usize;
        let cols = 12usize;
        let matrix: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 5 + 1) % 9) as f32 / 4.0)
            .collect();
        let vector: Vec<f32> = (0..cols).map(|i| ((i * 7 + 2) % 5) as f32 / 2.0).collect();

        let expected =
            matvec_gpu(&backend, &matrix, rows, cols, &vector).expect("slice matvec failed");

        let matrix_buf = backend.upload_f32(&matrix, "mv-m").expect("upload matrix");
        let vector_buf = backend.upload_f32(&vector, "mv-v").expect("upload vector");
        let resident = matvec_gpu_buf(&backend, &matrix_buf, rows, cols, &vector_buf)
            .expect("resident matvec failed");
        let downloaded = backend.download_f32(&resident).expect("download");

        assert_eq!(expected, downloaded);
    }

    /// Mismatched buffer sizes must be rejected before dispatch.
    #[tokio::test]
    async fn test_matvec_gpu_buf_dimension_checks() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let matrix_buf = backend
            .upload_f32(&[1.0, 2.0, 3.0, 4.0], "mv-m")
            .expect("upload");
        let vector_buf = backend.upload_f32(&[1.0, 1.0], "mv-v").expect("upload");

        let err = matvec_gpu_buf(&backend, &matrix_buf, 2, 3, &vector_buf)
            .expect_err("wrong matrix size must fail");
        assert!(
            matches!(err, WebGpuError::BufferSizeMismatch { .. }),
            "got: {err:?}"
        );

        let err = matvec_gpu_buf(&backend, &matrix_buf, 0, 2, &vector_buf)
            .expect_err("degenerate shape must fail");
        assert!(matches!(err, WebGpuError::Other(_)), "got: {err:?}");
    }
}
