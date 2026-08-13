//! GPU-accelerated SSM prefix scan via WGSL Blelloch kernels.
//!
//! Implements an inclusive SSM associative scan for sequences of **any**
//! length: each work-group scans a block of [`MAX_SINGLE_PASS_LEN`] elements,
//! the per-block aggregates are scanned by the same kernel (recursively, when
//! there is more than one block of aggregates), and a final pass folds the
//! resulting block prefixes back into the per-block results.
//!
//! The only hard limits are the device's own — storage binding size and
//! work-groups per dispatch — and exceeding either returns
//! [`WebGpuError::DeviceLimitExceeded`] before anything is dispatched.
//!
//! # Associative operator
//! `(a₁, bu₁) ⊗ (a₂, bu₂) = (a₂·a₁, a₂·bu₁ + bu₂)`
//!
//! Identity: `(1.0, 0.0)`.

use crate::buffer::GpuBuffer;
use crate::error::WebGpuError;
use crate::WebGpuBackend;

#[cfg(feature = "webgpu")]
use crate::backend::{decode_pairs, read_staging, F32_BYTES};
#[cfg(feature = "webgpu")]
use crate::buffer::GpuBufferUsage;
#[cfg(feature = "webgpu")]
use crate::pipeline::KernelKind;

/// Number of elements one work-group scans in a single block (256).
///
/// Sequences longer than this are **not** rejected: the multi-block driver
/// scans them exactly, using this value as the block size.
pub const MAX_SINGLE_PASS_LEN: usize = 256;

/// Block size as used by the dispatcher, matching `@workgroup_size(256)`.
#[cfg(feature = "webgpu")]
const BLOCK_SIZE: u32 = 256;

/// Bytes occupied by one `(a, bu)` element.
#[cfg(feature = "webgpu")]
const PAIR_BYTES: usize = F32_BYTES * 2;

/// Execute an inclusive SSM associative scan on the GPU.
///
/// Sequences of any length are supported; the kernel automatically switches to
/// the multi-block path beyond [`MAX_SINGLE_PASS_LEN`] elements.
///
/// # Errors
///
/// - [`WebGpuError::BackendUnavailable`] if compiled without `--features webgpu`.
/// - [`WebGpuError::DeviceLimitExceeded`] if the sequence exceeds the device's
///   storage-binding or dispatch limits.
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

/// Device-resident variant of [`ssm_scan_gpu`].
///
/// `input` holds interleaved `(a, bu)` pairs (`[a₀, bu₀, a₁, bu₁, …]`), and the
/// returned buffer holds the scan result in the same layout without ever
/// touching host memory — so `matvec → silu → scan` chains stay on the device.
///
/// # Errors
///
/// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
/// - [`WebGpuError::NoGpuAllocation`] if `input` is metadata-only.
/// - [`WebGpuError::BufferSizeMismatch`] if `input` is empty or its byte size
///   is not a multiple of `2 × size_of::<f32>()`.
/// - [`WebGpuError::DeviceLimitExceeded`] / [`WebGpuError::Other`] as above.
pub fn ssm_scan_gpu_buf(
    backend: &WebGpuBackend,
    input: &GpuBuffer,
) -> Result<GpuBuffer, WebGpuError> {
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = (backend, input);
        Err(WebGpuError::BackendUnavailable)
    }

    #[cfg(feature = "webgpu")]
    {
        ssm_scan_buf_impl(backend, input)
    }
}

// ── GPU implementation — only compiled with `--features webgpu` ──────────────

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

    let n = element_count(elements.len())?;
    let bytes = pairs_to_bytes(elements);
    let size_bytes = bytes.len() as u64;

    backend.run_scoped(|| {
        let input = backend.upload_bytes("ssm-scan-input", &bytes)?;
        let output = backend.create_storage_buffer("ssm-scan-output", size_bytes)?;
        let staging = backend.create_staging_buffer("ssm-scan-staging", size_bytes)?;

        let (device, queue) = backend.device_and_queue();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ssm-scan-encoder"),
        });

        // Held until after submission so no intermediate is freed early.
        let _scratch = encode_scan(backend, &mut encoder, &input, &output, n)?;

        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, size_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        read_staging(device, &staging, size_bytes, decode_pairs)
    })
}

#[cfg(feature = "webgpu")]
fn ssm_scan_buf_impl(backend: &WebGpuBackend, input: &GpuBuffer) -> Result<GpuBuffer, WebGpuError> {
    let source = input.wgpu_buffer()?;
    let size_bytes = input.size_bytes;

    if size_bytes == 0 || !size_bytes.is_multiple_of(PAIR_BYTES as u64) {
        return Err(WebGpuError::BufferSizeMismatch {
            expected: size_bytes
                .next_multiple_of(PAIR_BYTES as u64)
                .max(PAIR_BYTES as u64),
            got: size_bytes,
        });
    }

    let pairs = usize::try_from(size_bytes / PAIR_BYTES as u64)
        .map_err(|_| WebGpuError::Other("ssm_scan_gpu_buf: buffer too large".into()))?;
    let n = element_count(pairs)?;

    backend.run_scoped(|| {
        let output = backend.create_storage_buffer("ssm-scan-output", size_bytes)?;

        let (device, queue) = backend.device_and_queue();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ssm-scan-buf-encoder"),
        });

        let _scratch = encode_scan(backend, &mut encoder, source, &output, n)?;
        queue.submit(std::iter::once(encoder.finish()));

        Ok(GpuBuffer::from_wgpu(
            output,
            size_bytes,
            GpuBufferUsage::Storage,
            "ssm-scan-output",
        ))
    })
}

/// GPU resources that must outlive the command buffer they were encoded into.
#[cfg(feature = "webgpu")]
struct ScanScratch {
    #[allow(dead_code, reason = "kept alive until the command buffer is submitted")]
    buffers: Vec<wgpu::Buffer>,
    #[allow(dead_code, reason = "kept alive until the command buffer is submitted")]
    bind_groups: Vec<wgpu::BindGroup>,
}

/// Encode the full multi-block scan of `n` elements from `input` into `output`.
///
/// Pass structure (all in one command encoder, one compute pass per dispatch
/// so that cross-dispatch writes are visible to the following reads):
///
/// 1. level 0: block scan of the sequence → per-block aggregates
/// 2. level *k*: block scan of level *k−1*'s aggregates, until one block remains
/// 3. levels descending: apply each level's scanned aggregates as block prefixes
#[cfg(feature = "webgpu")]
fn encode_scan(
    backend: &WebGpuBackend,
    encoder: &mut wgpu::CommandEncoder,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    n: u32,
) -> Result<ScanScratch, WebGpuError> {
    let (device, _queue) = backend.device_and_queue();

    // Element count at each level: level 0 is the sequence, level k+1 is the
    // aggregate of level k's blocks.  The last level fits in a single block.
    let mut level_lens: Vec<u32> = Vec::new();
    let mut current = n;
    loop {
        level_lens.push(current);
        let blocks = current.div_ceil(BLOCK_SIZE);
        if blocks == 1 {
            break;
        }
        current = blocks;
    }

    let levels = level_lens.len();
    let mut buffers: Vec<wgpu::Buffer> = Vec::new();
    let mut bind_groups: Vec<wgpu::BindGroup> = Vec::new();

    // Outputs for levels 1.. (level 0 writes into the caller's `output`).
    let mut extra_outputs: Vec<wgpu::Buffer> = Vec::with_capacity(levels.saturating_sub(1));
    // Block aggregates for every level.
    let mut block_sums: Vec<wgpu::Buffer> = Vec::with_capacity(levels);

    for (level, &len) in level_lens.iter().enumerate() {
        let blocks = len.div_ceil(BLOCK_SIZE);
        backend.check_workgroups(&format!("ssm scan level {level}"), blocks)?;

        if level > 0 {
            extra_outputs.push(backend.create_storage_buffer(
                &format!("ssm-scan-level{level}-output"),
                pair_bytes(len)?,
            )?);
        }
        block_sums.push(
            backend.create_storage_buffer(
                &format!("ssm-scan-level{level}-sums"),
                pair_bytes(blocks)?,
            )?,
        );
    }

    // ── Upward pass: scan each level's blocks ───────────────────────────────
    for (level, &len) in level_lens.iter().enumerate() {
        let blocks = len.div_ceil(BLOCK_SIZE);
        let source = if level == 0 {
            input
        } else {
            level_buffer(&block_sums, level - 1, "block sums")?
        };
        let destination = level_output(output, &extra_outputs, level)?;
        let sums = level_buffer(&block_sums, level, "block sums")?;

        let params = backend
            .create_uniform_buffer(&format!("ssm-scan-level{level}-params"), &len.to_ne_bytes());
        let pipeline = backend.pipeline(KernelKind::SsmScanBlock)?;
        let bind_group = pipeline.bind_group(
            device,
            &format!("ssm-scan-block-bg-l{level}"),
            &[&params, source, destination, sums],
        );

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ssm-scan-block-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(blocks, 1, 1);
        }

        buffers.push(params);
        bind_groups.push(bind_group);
    }

    // ── Downward pass: fold each level's scanned aggregates back in ─────────
    for level in (0..levels.saturating_sub(1)).rev() {
        let len = *level_lens
            .get(level)
            .ok_or_else(|| WebGpuError::Other(format!("ssm scan: missing level {level}")))?;
        let blocks = len.div_ceil(BLOCK_SIZE);

        // Level `level + 1` holds the scanned aggregates of `level`'s blocks.
        let prefixes = level_output(output, &extra_outputs, level + 1)?;
        let data = level_output(output, &extra_outputs, level)?;

        let params = backend
            .create_uniform_buffer(&format!("ssm-scan-apply{level}-params"), &len.to_ne_bytes());
        let pipeline = backend.pipeline(KernelKind::SsmScanApply)?;
        let bind_group = pipeline.bind_group(
            device,
            &format!("ssm-scan-apply-bg-l{level}"),
            &[&params, prefixes, data],
        );

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ssm-scan-apply-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(blocks, 1, 1);
        }

        buffers.push(params);
        bind_groups.push(bind_group);
    }

    buffers.extend(extra_outputs);
    buffers.extend(block_sums);

    Ok(ScanScratch {
        buffers,
        bind_groups,
    })
}

/// Borrow the output buffer of `level` (level 0 is the caller's buffer).
#[cfg(feature = "webgpu")]
fn level_output<'a>(
    output: &'a wgpu::Buffer,
    extra: &'a [wgpu::Buffer],
    level: usize,
) -> Result<&'a wgpu::Buffer, WebGpuError> {
    if level == 0 {
        return Ok(output);
    }
    extra.get(level - 1).ok_or_else(|| {
        WebGpuError::Other(format!("ssm scan: missing output buffer for level {level}"))
    })
}

/// Borrow entry `level` of a per-level buffer list.
#[cfg(feature = "webgpu")]
fn level_buffer<'a>(
    buffers: &'a [wgpu::Buffer],
    level: usize,
    what: &str,
) -> Result<&'a wgpu::Buffer, WebGpuError> {
    buffers
        .get(level)
        .ok_or_else(|| WebGpuError::Other(format!("ssm scan: missing {what} for level {level}")))
}

/// Byte size of `count` interleaved `(a, bu)` pairs.
#[cfg(feature = "webgpu")]
fn pair_bytes(count: u32) -> Result<u64, WebGpuError> {
    u64::from(count)
        .checked_mul(PAIR_BYTES as u64)
        .ok_or_else(|| WebGpuError::Other(format!("ssm scan: size overflow for {count} elements")))
}

/// Narrow a host element count to the `u32` the shader indexes with.
#[cfg(feature = "webgpu")]
fn element_count(len: usize) -> Result<u32, WebGpuError> {
    u32::try_from(len).map_err(|_| WebGpuError::DeviceLimitExceeded {
        what: "ssm scan sequence length".into(),
        required: len as u64,
        limit: u64::from(u32::MAX),
    })
}

/// Flatten `(a, bu)` pairs into the interleaved byte layout the kernel reads.
#[cfg(feature = "webgpu")]
fn pairs_to_bytes(elements: &[(f32, f32)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(elements.len() * PAIR_BYTES);
    for &(a, bu) in elements {
        bytes.extend_from_slice(&a.to_ne_bytes());
        bytes.extend_from_slice(&bu.to_ne_bytes());
    }
    bytes
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, not(feature = "webgpu")))]
mod tests {
    use super::*;
    use crate::backend::unavailable_backend;

    /// Without the `webgpu` feature, `ssm_scan_gpu` must return
    /// `Err(WebGpuError::BackendUnavailable)` — exercised on the function
    /// itself, not on the constructor.
    #[test]
    fn test_ssm_scan_gpu_unavailable_without_feature() {
        let backend = unavailable_backend();
        let result = ssm_scan_gpu(&backend, &[(0.9, 1.0), (0.8, 2.0)]);
        assert!(
            matches!(result, Err(WebGpuError::BackendUnavailable)),
            "expected BackendUnavailable, got: {result:?}",
        );
    }

    /// The device-resident variant must be unavailable too.
    #[test]
    fn test_ssm_scan_gpu_buf_unavailable_without_feature() {
        let backend = unavailable_backend();
        let buf = GpuBuffer::metadata_only(16, crate::GpuBufferUsage::Storage, "scan-in");
        let result = ssm_scan_gpu_buf(&backend, &buf);
        assert!(
            matches!(result, Err(WebGpuError::BackendUnavailable)),
            "expected BackendUnavailable, got: {result:?}",
        );
    }
}

#[cfg(all(test, feature = "webgpu"))]
mod gpu_tests {
    use super::*;
    use crate::test_support::{assert_pairs_close, try_backend};
    use kizzasi_core::ssm_backend::{CpuSsmBackend, SsmBackend};

    /// Deterministic, numerically well-behaved scan input.
    fn sample_elements(n: usize) -> Vec<(f32, f32)> {
        (0..n)
            .map(|i| {
                let a = 0.5 + 0.4 * ((i * 13 + 7) % 10) as f32 / 10.0;
                let bu = ((i * 7 + 3) % 5) as f32 * 0.5;
                (a, bu)
            })
            .collect()
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
        assert_pairs_close(&[(0.9, 1.0), (0.72, 2.8)], &result, 1e-4);
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
        assert_pairs_close(&[(1.0, 0.0), (0.5, 3.0)], &result, 1e-4);
    }

    #[tokio::test]
    async fn test_ssm_scan_gpu_matches_cpu() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let elements = sample_elements(64);
        let cpu_result = CpuSsmBackend.ssm_scan(&elements).expect("CPU scan failed");
        let gpu_result = ssm_scan_gpu(&backend, &elements).expect("GPU scan failed");

        assert_eq!(cpu_result.len(), gpu_result.len());
        assert_pairs_close(&cpu_result, &gpu_result, 1e-4);
    }

    /// Regression: sequences that span more than one work-group used to be
    /// scanned as if the sequence restarted at every 256-element boundary.
    #[tokio::test]
    async fn test_ssm_scan_gpu_multi_block_boundaries() {
        let Some(backend) = try_backend().await else {
            return;
        };

        for n in [255_usize, 256, 257, 511, 512, 513, 1000] {
            let elements = sample_elements(n);
            let cpu_result = CpuSsmBackend.ssm_scan(&elements).expect("CPU scan failed");
            let gpu_result = ssm_scan_gpu(&backend, &elements).expect("GPU scan failed");

            assert_eq!(gpu_result.len(), n, "length mismatch at n={n}");
            assert_pairs_close(&cpu_result, &gpu_result, 1e-3);
        }
    }

    /// Three-level recursion: 66 000 elements → 258 aggregates → 2 aggregates.
    #[tokio::test]
    async fn test_ssm_scan_gpu_three_level_recursion() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let n = 66_000_usize;
        let elements = sample_elements(n);
        let cpu_result = CpuSsmBackend.ssm_scan(&elements).expect("CPU scan failed");
        let gpu_result = ssm_scan_gpu(&backend, &elements).expect("GPU scan failed");

        assert_eq!(gpu_result.len(), n);
        assert_pairs_close(&cpu_result, &gpu_result, 1e-3);
    }

    /// Empty and single-element inputs stay on the fast path.
    #[tokio::test]
    async fn test_ssm_scan_gpu_degenerate_lengths() {
        let Some(backend) = try_backend().await else {
            return;
        };

        assert!(ssm_scan_gpu(&backend, &[])
            .expect("empty scan failed")
            .is_empty());
        let single = ssm_scan_gpu(&backend, &[(0.3, 1.5)]).expect("single-element scan failed");
        assert_eq!(single, vec![(0.3, 1.5)]);
    }

    /// The device-resident variant must agree with the host-slice variant.
    #[tokio::test]
    async fn test_ssm_scan_gpu_buf_roundtrip() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let elements = sample_elements(700);
        let flat: Vec<f32> = elements.iter().flat_map(|&(a, bu)| [a, bu]).collect();

        let input = backend.upload_f32(&flat, "scan-buf-in").expect("upload");
        let output = ssm_scan_gpu_buf(&backend, &input).expect("resident scan failed");
        let downloaded = backend.download_f32(&output).expect("download");

        let expected = ssm_scan_gpu(&backend, &elements).expect("host scan failed");
        let got: Vec<(f32, f32)> = downloaded.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        assert_pairs_close(&expected, &got, 1e-5);
    }

    /// A buffer whose byte size is not a whole number of pairs is rejected.
    #[tokio::test]
    async fn test_ssm_scan_gpu_buf_rejects_odd_size() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let input = backend
            .upload_f32(&[1.0, 2.0, 3.0], "scan-buf-odd")
            .expect("upload");
        let err = ssm_scan_gpu_buf(&backend, &input).expect_err("odd size must fail");
        assert!(
            matches!(err, WebGpuError::BufferSizeMismatch { .. }),
            "got: {err:?}"
        );
    }
}
