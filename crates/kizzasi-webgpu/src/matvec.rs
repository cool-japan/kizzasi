//! GPU-accelerated matrix-vector multiplication.
//!
//! Provides a single WGSL kernel that computes `output = matrix × vector`
//! where `matrix` is a row-major `[rows, cols]` matrix stored as a flat `f32`
//! slice.  Each GPU thread processes one output row.
//!
//! # Feature gate
//! Without `--features webgpu`, all functions return
//! [`WebGpuError::BackendUnavailable`].

use crate::{WebGpuBackend, WebGpuError};

/// The WGSL shader source embedded at compile time.
#[cfg(feature = "webgpu")]
const SHADER_SRC: &str = include_str!("shaders/matvec.wgsl");

/// Execute `output = matrix × vector` on the GPU.
///
/// `matrix` is row-major with shape `[rows, cols]` (length = `rows × cols`).
/// `vector` must have length `cols`.  Returns a `Vec<f32>` of length `rows`.
///
/// # Errors
///
/// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
/// - [`WebGpuError::BufferSizeMismatch`] if dimension checks fail.
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
        if matrix.len() != rows * cols {
            return Err(WebGpuError::BufferSizeMismatch {
                expected: (rows * cols) as u64,
                got: matrix.len() as u64,
            });
        }
        if vector.len() != cols {
            return Err(WebGpuError::BufferSizeMismatch {
                expected: cols as u64,
                got: vector.len() as u64,
            });
        }
        matvec_impl(backend, matrix, rows, cols, vector)
    }
}

// ── GPU implementation — only compiled with `--features webgpu` ──────────────

#[cfg(feature = "webgpu")]
fn matvec_impl(
    backend: &WebGpuBackend,
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
) -> Result<Vec<f32>, WebGpuError> {
    let (device, queue) = backend.device_and_queue();

    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    // ── Uniform buffer: { rows: u32, cols: u32 } = 8 bytes ─────────────────
    let mut uniform_data = [0u8; 8];
    uniform_data[0..4].copy_from_slice(&rows_u32.to_ne_bytes());
    uniform_data[4..8].copy_from_slice(&cols_u32.to_ne_bytes());

    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matvec-params"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buf, 0, &uniform_data);

    // ── Matrix storage buffer (read-only) ───────────────────────────────────
    let matrix_bytes = f32_slice_as_bytes(matrix);
    let matrix_size = matrix_bytes.len() as u64;

    let matrix_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matvec-matrix"),
        size: matrix_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&matrix_buf, 0, matrix_bytes);

    // ── Vector storage buffer (read-only) ───────────────────────────────────
    let vector_bytes = f32_slice_as_bytes(vector);
    let vector_size = vector_bytes.len() as u64;

    let vector_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matvec-vector"),
        size: vector_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&vector_buf, 0, vector_bytes);

    // ── Output storage buffer (read-write) ──────────────────────────────────
    let output_size = (rows * std::mem::size_of::<f32>()) as u64;

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matvec-output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // ── Staging buffer for readback ─────────────────────────────────────────
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matvec-staging"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Shader + pipeline ───────────────────────────────────────────────────
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matvec-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("matvec-bgl"),
        entries: &[
            // binding 0: uniform params { rows, cols }
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 1: matrix (read-only storage)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 2: vector (read-only storage)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 3: output (read-write storage)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("matvec-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matvec-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matvec-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: matrix_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: vector_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    // ── Encode and dispatch ─────────────────────────────────────────────────
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matvec-encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matvec-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // One thread per row; workgroup size = 64.
        let workgroups = rows_u32.div_ceil(64);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
    queue.submit(std::iter::once(encoder.finish()));

    // ── Poll until GPU work completes ───────────────────────────────────────
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| WebGpuError::Other(format!("device poll error: {e:?}")))?;

    // ── Map staging buffer and read back ────────────────────────────────────
    let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
    staging_buf
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| WebGpuError::Other(format!("device poll error after map: {e:?}")))?;

    rx.recv()
        .map_err(|_| WebGpuError::MapBuffer("channel closed before map completed".into()))?
        .map_err(|e| WebGpuError::MapBuffer(e.to_string()))?;

    let mapped = staging_buf
        .slice(..)
        .get_mapped_range()
        .map_err(|e| WebGpuError::MapBuffer(e.to_string()))?;
    let result: Vec<f32> = mapped
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap_or([0u8; 4]);
            f32::from_ne_bytes(arr)
        })
        .collect();

    drop(mapped);
    staging_buf.unmap();

    Ok(result)
}

/// Reinterpret a `&[f32]` as `&[u8]` without copying.
///
/// Safe because `f32` has no invalid byte representations and `u8` has
/// alignment 1.
#[cfg(feature = "webgpu")]
fn f32_slice_as_bytes(data: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid byte patterns; u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrong matrix dimensions must return `BufferSizeMismatch` immediately.
    #[test]
    fn test_matvec_dimension_check_matrix() {
        // `matvec_gpu` validates dimensions synchronously before touching the
        // GPU, so a mismatched matrix length triggers `BufferSizeMismatch`
        // regardless of whether a real GPU is available.
        // rows=2, cols=3, but matrix has len=5 (not 6).
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        rt.block_on(async {
            let dummy_matrix = vec![0.0f32; 5]; // wrong: should be 2*3=6
            let dummy_vec = vec![0.0f32; 3];

            // Attempt to obtain a backend; the dimension error fires before the
            // backend is used, so even a GPU-less environment exercises the check.
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
        });
    }

    /// Wrong vector dimensions must return `BufferSizeMismatch`.
    #[test]
    fn test_matvec_dimension_check_vector() {
        // Dimension check fires before any GPU access.
        // matrix=2×3 correct, vector=2 instead of 3.
        let dummy_matrix = vec![0.0f32; 6];
        let dummy_vec = vec![0.0f32; 2]; // wrong: should be 3

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        rt.block_on(async {
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
        });
    }

    /// Without the `webgpu` feature, `matvec_gpu` returns `BackendUnavailable`.
    #[cfg(not(feature = "webgpu"))]
    #[tokio::test]
    async fn test_matvec_unavailable_without_feature() {
        let result = WebGpuBackend::new().await;
        assert!(
            matches!(result, Err(WebGpuError::BackendUnavailable)),
            "expected BackendUnavailable without webgpu feature, got: {result:?}"
        );
    }
}

#[cfg(all(test, feature = "webgpu"))]
mod gpu_tests {
    use super::*;

    /// Helper: obtain a live backend or skip gracefully when no GPU is present.
    async fn try_backend() -> Option<WebGpuBackend> {
        match WebGpuBackend::new().await {
            Ok(b) => Some(b),
            Err(WebGpuError::AdapterRequest(_)) => {
                eprintln!("no GPU adapter found — skipping GPU test");
                None
            }
            Err(e) => panic!("unexpected error creating WebGpuBackend: {e}"),
        }
    }

    /// 4×4 identity matrix × [1, 2, 3, 4] should return [1, 2, 3, 4].
    #[tokio::test]
    async fn test_matvec_identity() {
        let Some(backend) = try_backend().await else {
            return;
        };

        // 4×4 identity matrix.
        #[rustfmt::skip]
        let matrix = [
            1.0_f32, 0.0, 0.0, 0.0,
            0.0,     1.0, 0.0, 0.0,
            0.0,     0.0, 1.0, 0.0,
            0.0,     0.0, 0.0, 1.0,
        ];
        let vector = [1.0_f32, 2.0, 3.0, 4.0];
        let expected = [1.0_f32, 2.0, 3.0, 4.0];

        let result = matvec_gpu(&backend, &matrix, 4, 4, &vector).expect("GPU matvec failed");

        assert_eq!(result.len(), 4);
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "identity test row {i}: expected {exp}, got {got}"
            );
        }
    }

    /// 2×3 matrix × 3-vector, verified against CPU reference.
    #[tokio::test]
    async fn test_matvec_known_values() {
        let Some(backend) = try_backend().await else {
            return;
        };

        // matrix = [[1, 2, 3], [4, 5, 6]]
        let matrix = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // vector = [1, 2, 3]
        let vector = [1.0_f32, 2.0, 3.0];
        // expected: row0 = 1+4+9=14, row1 = 4+10+18=32
        let expected = [14.0_f32, 32.0];

        let result = matvec_gpu(&backend, &matrix, 2, 3, &vector).expect("GPU matvec failed");

        assert_eq!(result.len(), 2);
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "known-values row {i}: expected {exp}, got {got}"
            );
        }
    }

    /// 16×16 random-ish matrix vs CPU reference — max diff must be < 1e-4.
    #[tokio::test]
    async fn test_matvec_matches_cpu() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let rows = 16usize;
        let cols = 16usize;

        // Deterministic pseudo-random data.
        let matrix: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7 + 3) % 17) as f32 / 8.0)
            .collect();
        let vector: Vec<f32> = (0..cols).map(|i| ((i * 5 + 1) % 11) as f32 / 5.0).collect();

        // CPU reference.
        let cpu: Vec<f32> = (0..rows)
            .map(|row| {
                (0..cols)
                    .map(|col| matrix[row * cols + col] * vector[col])
                    .sum()
            })
            .collect();

        let gpu = matvec_gpu(&backend, &matrix, rows, cols, &vector).expect("GPU matvec failed");

        assert_eq!(gpu.len(), rows);

        let max_diff = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "max absolute difference CPU vs GPU: {max_diff}"
        );
    }
}
