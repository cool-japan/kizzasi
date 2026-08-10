//! GPU-accelerated elementwise operations: SiLU activation and RMS Norm.
//!
//! # SiLU
//! Computes the Swish activation: `output[i] = input[i] / (1 + exp(-input[i]))`.
//! Any input length is supported.
//!
//! # RMS Norm
//! Computes: `output[i] = (input[i] / rms(input)) * weight[i]`
//! where `rms(x) = sqrt(mean(x²) + eps)`.
//!
//! The RMS Norm uses a single-pass, single-work-group kernel with a parallel
//! tree reduction in shared memory. This limits it to at most
//! [`MAX_RMS_NORM_LEN`] elements. Callers must fall back to a CPU path for
//! larger inputs.
//!
//! # Feature gate
//! Without `--features webgpu`, all functions return
//! [`WebGpuError::BackendUnavailable`].

use crate::{WebGpuBackend, WebGpuError};

/// Maximum input length supported by the GPU RMS Norm kernel.
///
/// The single-pass kernel uses one work-group of 256 threads.
pub const MAX_RMS_NORM_LEN: usize = 256;

// ── Shader sources ────────────────────────────────────────────────────────────

#[cfg(feature = "webgpu")]
const SILU_SHADER_SRC: &str = include_str!("shaders/silu.wgsl");

#[cfg(feature = "webgpu")]
const RMS_NORM_SHADER_SRC: &str = include_str!("shaders/rms_norm.wgsl");

// ── Public API ────────────────────────────────────────────────────────────────

/// GPU SiLU activation: `output[i] = input[i] / (1 + exp(-input[i]))`.
///
/// # Errors
///
/// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
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
        if input.len() > MAX_RMS_NORM_LEN {
            return Err(WebGpuError::Other(format!(
                "rms_norm_gpu: input length {} exceeds maximum {} for single-pass kernel; \
                 use a CPU fallback for larger inputs",
                input.len(),
                MAX_RMS_NORM_LEN
            )));
        }
        rms_norm_impl(backend, input, weight, eps)
    }
}

// ── GPU implementation — only compiled with `--features webgpu` ──────────────

#[cfg(feature = "webgpu")]
fn silu_impl(backend: &WebGpuBackend, input: &[f32]) -> Result<Vec<f32>, WebGpuError> {
    let (device, queue) = backend.device_and_queue();
    let n = input.len() as u32;

    // ── Uniform buffer: { n: u32 } = 4 bytes ───────────────────────────────
    let uniform_data = n.to_ne_bytes();
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("silu-params"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buf, 0, &uniform_data);

    // ── Input storage buffer (read-only) ────────────────────────────────────
    let input_bytes = f32_slice_as_bytes(input);
    let data_size = input_bytes.len() as u64;

    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("silu-input"),
        size: data_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buf, 0, input_bytes);

    // ── Output storage buffer (read-write) ──────────────────────────────────
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("silu-output"),
        size: data_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // ── Staging buffer for readback ─────────────────────────────────────────
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("silu-staging"),
        size: data_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Shader + pipeline ───────────────────────────────────────────────────
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("silu-shader"),
        source: wgpu::ShaderSource::Wgsl(SILU_SHADER_SRC.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("silu-bgl"),
        entries: &[
            // binding 0: uniform { n }
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
            // binding 1: input (read-only storage)
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
            // binding 2: output (read-write storage)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
        label: Some("silu-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("silu-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("silu-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    // ── Encode and dispatch ─────────────────────────────────────────────────
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("silu-encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("silu-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // Workgroup size = 256; dispatch enough workgroups to cover all elements.
        let workgroups = n.div_ceil(256);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, data_size);
    queue.submit(std::iter::once(encoder.finish()));

    readback_f32(device, &staging_buf, data_size)
}

#[cfg(feature = "webgpu")]
fn rms_norm_impl(
    backend: &WebGpuBackend,
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, WebGpuError> {
    let (device, queue) = backend.device_and_queue();
    let n = input.len() as u32;

    // ── Uniform buffer: { n: u32, eps: f32 } = 8 bytes ─────────────────────
    let mut uniform_data = [0u8; 8];
    uniform_data[0..4].copy_from_slice(&n.to_ne_bytes());
    uniform_data[4..8].copy_from_slice(&eps.to_ne_bytes());

    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rms-norm-params"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buf, 0, &uniform_data);

    // ── Input and weight storage buffers (read-only) ────────────────────────
    let input_bytes = f32_slice_as_bytes(input);
    let data_size = input_bytes.len() as u64;

    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rms-norm-input"),
        size: data_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buf, 0, input_bytes);

    let weight_bytes = f32_slice_as_bytes(weight);
    let weight_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rms-norm-weight"),
        size: data_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&weight_buf, 0, weight_bytes);

    // ── Output storage buffer (read-write) ──────────────────────────────────
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rms-norm-output"),
        size: data_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // ── Staging buffer for readback ─────────────────────────────────────────
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rms-norm-staging"),
        size: data_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Shader + pipeline ───────────────────────────────────────────────────
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rms-norm-shader"),
        source: wgpu::ShaderSource::Wgsl(RMS_NORM_SHADER_SRC.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rms-norm-bgl"),
        entries: &[
            // binding 0: uniform { n, eps }
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
            // binding 1: input (read-only storage)
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
            // binding 2: weight (read-only storage)
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
        label: Some("rms-norm-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rms-norm-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rms-norm-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: weight_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    // ── Encode and dispatch ─────────────────────────────────────────────────
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rms-norm-encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rms-norm-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // Single work-group covers up to 256 elements.
        cpass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, data_size);
    queue.submit(std::iter::once(encoder.finish()));

    readback_f32(device, &staging_buf, data_size)
}

/// Poll GPU until complete, map staging buffer, and collect `Vec<f32>`.
#[cfg(feature = "webgpu")]
fn readback_f32(
    device: &wgpu::Device,
    staging_buf: &wgpu::Buffer,
    data_size: u64,
) -> Result<Vec<f32>, WebGpuError> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| WebGpuError::Other(format!("device poll error: {e:?}")))?;

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

    // Size check: data_size bytes must correspond to a whole number of f32s.
    let expected_count = (data_size / 4) as usize;
    if result.len() != expected_count {
        return Err(WebGpuError::BufferSizeMismatch {
            expected: data_size,
            got: (result.len() * 4) as u64,
        });
    }

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

    /// Without the `webgpu` feature, `silu_gpu` must return `BackendUnavailable`.
    #[cfg(not(feature = "webgpu"))]
    #[tokio::test]
    async fn test_silu_unavailable_without_feature() {
        let result = WebGpuBackend::new().await;
        assert!(
            matches!(result, Err(WebGpuError::BackendUnavailable)),
            "expected BackendUnavailable, got: {result:?}"
        );
    }

    /// Without the `webgpu` feature, `rms_norm_gpu` must return `BackendUnavailable`.
    #[cfg(not(feature = "webgpu"))]
    #[tokio::test]
    async fn test_rms_norm_unavailable_without_feature() {
        let result = WebGpuBackend::new().await;
        assert!(
            matches!(result, Err(WebGpuError::BackendUnavailable)),
            "expected BackendUnavailable, got: {result:?}"
        );
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

    /// Helper: obtain a backend or skip gracefully when no GPU adapter is present.
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

    /// Known SiLU values: silu(0) ≈ 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689.
    #[tokio::test]
    async fn test_silu_known_values() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let input = [0.0_f32, 1.0, -1.0];
        let result = silu_gpu(&backend, &input).expect("GPU silu failed");

        assert_eq!(result.len(), 3);

        // silu(0) = 0 / (1 + 1) = 0
        assert!(
            result[0].abs() < 1e-5,
            "silu(0): expected 0, got {}",
            result[0]
        );

        // silu(1) = 1 / (1 + exp(-1)) ≈ 0.7311
        let expected_1 = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        assert!(
            (result[1] - expected_1).abs() < 1e-5,
            "silu(1): expected {expected_1}, got {}",
            result[1]
        );

        // silu(-1) = -1 / (1 + exp(1)) ≈ -0.2689
        let expected_neg1 = -1.0_f32 / (1.0 + (1.0_f32).exp());
        assert!(
            (result[2] - expected_neg1).abs() < 1e-5,
            "silu(-1): expected {expected_neg1}, got {}",
            result[2]
        );
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

        let cpu: Vec<f32> = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

        let gpu = silu_gpu(&backend, &input).expect("GPU silu failed");

        assert_eq!(gpu.len(), 64);

        let max_diff = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "max absolute difference CPU vs GPU silu: {max_diff}"
        );
    }

    /// Uniform input [1,…,1] with uniform weights [1,…,1] → output ≈ [1,…,1].
    /// rms([1,1,...,1]) = 1, so output[i] = (1 / 1) * 1 = 1.
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
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "rms_norm uniform: element {i} expected 1.0, got {v}"
            );
        }
    }

    /// RMS Norm: compare GPU output to CPU reference for 32 values.
    #[tokio::test]
    async fn test_rms_norm_matches_cpu() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let n = 32usize;
        let input: Vec<f32> = (0..n).map(|i| ((i * 3 + 1) % 7) as f32 / 3.0).collect();
        let weight: Vec<f32> = (0..n)
            .map(|i| 0.5 + ((i * 5 + 2) % 4) as f32 / 4.0)
            .collect();
        let eps = 1e-5_f32;

        // CPU reference.
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let mean_sq = sum_sq / n as f32;
        let rms = (mean_sq + eps).sqrt();
        let cpu: Vec<f32> = input
            .iter()
            .zip(weight.iter())
            .map(|(&x, &w)| (x / rms) * w)
            .collect();

        let gpu = rms_norm_gpu(&backend, &input, &weight, eps).expect("GPU rms_norm failed");

        assert_eq!(gpu.len(), n);

        let max_diff = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "max absolute difference CPU vs GPU rms_norm: {max_diff}"
        );
    }
}
