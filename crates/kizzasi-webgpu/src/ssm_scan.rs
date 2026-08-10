//! GPU-accelerated SSM prefix scan via WGSL Blelloch kernel.
//!
//! Provides a single-pass inclusive SSM associative scan for sequences up to
//! [`MAX_SINGLE_PASS_LEN`] elements.  Longer sequences fall back to the CPU
//! implementation in `kizzasi-core`.
//!
//! # Associative operator
//! `(a₁, bu₁) ⊗ (a₂, bu₂) = (a₂·a₁, a₂·bu₁ + bu₂)`
//!
//! Identity: `(1.0, 0.0)`.

use crate::error::WebGpuError;
use crate::WebGpuBackend;

/// Maximum sequence length that fits in one work-group (256 elements).
pub const MAX_SINGLE_PASS_LEN: usize = 256;

/// The WGSL shader source, embedded at compile time.
#[cfg(feature = "webgpu")]
const SHADER_SRC: &str = include_str!("shaders/ssm_scan.wgsl");

/// Execute an inclusive SSM associative scan on the GPU.
///
/// Sequences longer than [`MAX_SINGLE_PASS_LEN`] are not supported by this
/// single-pass kernel; callers should fall back to the CPU path.
///
/// # Errors
///
/// - [`WebGpuError::BackendUnavailable`] if compiled without `--features webgpu`.
/// - [`WebGpuError::Other`] for wgpu pipeline / dispatch errors.
pub fn ssm_scan_gpu(
    backend: &WebGpuBackend,
    elements: &[(f32, f32)],
) -> Result<Vec<(f32, f32)>, WebGpuError> {
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = (backend, elements);
        Err(WebGpuError::BackendUnavailable)
    }

    #[cfg(feature = "webgpu")]
    {
        ssm_scan_impl(backend, elements)
    }
}

#[cfg(feature = "webgpu")]
fn ssm_scan_impl(
    backend: &WebGpuBackend,
    elements: &[(f32, f32)],
) -> Result<Vec<(f32, f32)>, WebGpuError> {
    if elements.is_empty() {
        return Ok(Vec::new());
    }
    if elements.len() == 1 {
        return Ok(elements.to_vec());
    }

    let (device, queue) = backend.device_and_queue();
    let n = elements.len() as u32;

    // Flatten (a, bu) pairs to an f32 slice for upload.
    let flat: Vec<f32> = elements.iter().flat_map(|&(a, bu)| [a, bu]).collect();

    let flat_bytes: &[u8] = f32_slice_as_bytes(&flat);
    let input_size = flat_bytes.len() as u64;

    // --- Buffers -------------------------------------------------------

    // Uniform buffer: single u32 holding `n`.
    let params_data: [u8; 4] = n.to_ne_bytes();
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ssm-scan-params"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buf, 0, &params_data);

    // Input storage buffer.
    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ssm-scan-input"),
        size: input_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buf, 0, flat_bytes);

    // Output storage buffer.
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ssm-scan-output"),
        size: input_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Staging buffer for readback.
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ssm-scan-staging"),
        size: input_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Shader + pipeline -----------------------------------------------

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ssm-scan-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ssm-scan-bgl"),
        entries: &[
            // binding 0: uniform params
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
            // binding 1: input storage (read-only)
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
            // binding 2: output storage (read-write)
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
        label: Some("ssm-scan-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ssm-scan-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ssm-scan-bg"),
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

    // --- Encode and dispatch -------------------------------------------

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ssm-scan-encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ssm-scan-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // Single work-group covers up to 256 elements.
        let workgroups = (elements.len() as u32).div_ceil(256);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    // Copy output buffer → staging buffer.
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, input_size);

    queue.submit(std::iter::once(encoder.finish()));

    // Block until GPU work completes.
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| WebGpuError::Other(format!("device poll error: {e:?}")))?;

    // --- Map and read back --------------------------------------------

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
    let result_flat: Vec<f32> = mapped
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap_or([0u8; 4]);
            f32::from_ne_bytes(arr)
        })
        .collect();

    drop(mapped);
    staging_buf.unmap();

    // Re-pack interleaved f32 pairs back into (a, bu) tuples.
    let output: Vec<(f32, f32)> = result_flat
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect();

    Ok(output)
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
    #[cfg(not(feature = "webgpu"))]
    use super::*;

    /// Without the `webgpu` feature, `ssm_scan_gpu` must return
    /// `Err(WebGpuError::BackendUnavailable)`.
    #[cfg(not(feature = "webgpu"))]
    #[tokio::test]
    async fn test_ssm_scan_gpu_unavailable_without_feature() {
        // Without the `webgpu` feature the constructor returns
        // `BackendUnavailable` immediately; verify the error variant is correct.
        let result = WebGpuBackend::new().await;
        assert!(
            matches!(result, Err(WebGpuError::BackendUnavailable)),
            "expected BackendUnavailable, got: {result:?}",
        );
    }
}

#[cfg(all(test, feature = "webgpu"))]
mod gpu_tests {
    use super::*;
    use kizzasi_core::ssm_backend::{CpuSsmBackend, SsmBackend};

    /// Helper: obtain a backend or skip the test gracefully when no GPU is
    /// available in the current CI / sandbox environment.
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

    #[tokio::test]
    async fn test_ssm_scan_gpu_two_elements() {
        let Some(backend) = try_backend().await else {
            return;
        };
        // (0.9, 1.0) ⊗ (0.8, 2.0) = (0.8·0.9, 0.8·1.0 + 2.0) = (0.72, 2.8)
        let elements = [(0.9_f32, 1.0_f32), (0.8_f32, 2.0_f32)];
        let result = ssm_scan_gpu(&backend, &elements).expect("GPU scan failed");
        assert_eq!(result.len(), 2);
        assert!(
            (result[0].0 - 0.9).abs() < 1e-4,
            "result[0].0: expected 0.9, got {}",
            result[0].0
        );
        assert!(
            (result[0].1 - 1.0).abs() < 1e-4,
            "result[0].1: expected 1.0, got {}",
            result[0].1
        );
        assert!(
            (result[1].0 - 0.72).abs() < 1e-4,
            "result[1].0: expected 0.72, got {}",
            result[1].0
        );
        assert!(
            (result[1].1 - 2.8).abs() < 1e-4,
            "result[1].1: expected 2.8, got {}",
            result[1].1
        );
    }

    #[tokio::test]
    async fn test_ssm_scan_gpu_identity() {
        let Some(backend) = try_backend().await else {
            return;
        };
        // (1.0, 0.0) ⊗ (0.5, 3.0) = (0.5·1.0, 0.5·0.0 + 3.0) = (0.5, 3.0)
        let elements = [(1.0_f32, 0.0_f32), (0.5_f32, 3.0_f32)];
        let result = ssm_scan_gpu(&backend, &elements).expect("GPU scan failed");
        assert_eq!(result.len(), 2);
        assert!(
            (result[0].0 - 1.0).abs() < 1e-4,
            "result[0].0: expected 1.0, got {}",
            result[0].0
        );
        assert!(
            (result[0].1 - 0.0).abs() < 1e-4,
            "result[0].1: expected 0.0, got {}",
            result[0].1
        );
        assert!(
            (result[1].0 - 0.5).abs() < 1e-4,
            "result[1].0: expected 0.5, got {}",
            result[1].0
        );
        assert!(
            (result[1].1 - 3.0).abs() < 1e-4,
            "result[1].1: expected 3.0, got {}",
            result[1].1
        );
    }

    #[tokio::test]
    async fn test_ssm_scan_gpu_matches_cpu() {
        let Some(backend) = try_backend().await else {
            return;
        };

        // Generate 64 deterministic pseudo-random elements.
        let elements: Vec<(f32, f32)> = (0..64_usize)
            .map(|i| {
                // Simple LCG-like pattern: values in (0,1) range.
                let a = 0.5 + 0.4 * ((i * 13 + 7) % 10) as f32 / 10.0;
                let bu = ((i * 7 + 3) % 5) as f32 * 0.5;
                (a, bu)
            })
            .collect();

        let cpu_backend = CpuSsmBackend;
        let cpu_result = cpu_backend.ssm_scan(&elements).expect("CPU scan failed");
        let gpu_result = ssm_scan_gpu(&backend, &elements).expect("GPU scan failed");

        assert_eq!(cpu_result.len(), gpu_result.len());

        let max_diff = cpu_result
            .iter()
            .zip(gpu_result.iter())
            .map(|(&(ca, cbu), &(ga, gbu))| (ca - ga).abs().max((cbu - gbu).abs()))
            .fold(0.0_f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "max absolute difference between CPU and GPU scan: {max_diff}"
        );
    }
}
