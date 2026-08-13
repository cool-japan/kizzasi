//! Lazily-built, cached compute pipelines.
//!
//! WGSL parsing, shader validation and backend shader compilation dominate the
//! cost of a small dispatch.  Building the pipeline on every kernel call pays
//! that cost per invocation, which is exactly the wrong trade for the per-step
//! inference loop this backend serves.  [`PipelineCache`] therefore builds each
//! kernel's shader module, bind-group layout and compute pipeline once, on
//! first use, and hands out borrowed references afterwards.
//!
//! The cache is `Send + Sync` and lock-free on the hot path: each slot is a
//! [`OnceLock`], so a cache hit is a single atomic load.

use std::sync::OnceLock;

use crate::error::WebGpuError;

// ── Shader sources, embedded at compile time ─────────────────────────────────

const SSM_SCAN_BLOCK_SRC: &str = include_str!("shaders/ssm_scan_block.wgsl");
const SSM_SCAN_APPLY_SRC: &str = include_str!("shaders/ssm_scan_apply.wgsl");
const SILU_SRC: &str = include_str!("shaders/silu.wgsl");
const RMS_NORM_SRC: &str = include_str!("shaders/rms_norm.wgsl");
const MATVEC_SRC: &str = include_str!("shaders/matvec.wgsl");

// ── Kernel identification ────────────────────────────────────────────────────

/// The compute kernels this crate ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KernelKind {
    /// Per-block Blelloch SSM scan producing block aggregates.
    SsmScanBlock,
    /// Applies scanned block prefixes to the per-block scan results.
    SsmScanApply,
    /// SiLU (Swish) activation.
    Silu,
    /// Single-work-group RMS normalisation.
    RmsNorm,
    /// Row-major matrix-vector product.
    Matvec,
}

/// The kind of buffer binding a shader slot expects.
#[derive(Debug, Clone, Copy)]
enum BindingKind {
    /// Uniform buffer (kernel parameters).
    Uniform,
    /// Read-only storage buffer.
    ReadStorage,
    /// Read-write storage buffer.
    ReadWriteStorage,
}

impl BindingKind {
    fn binding_type(self) -> wgpu::BindingType {
        let ty = match self {
            BindingKind::Uniform => wgpu::BufferBindingType::Uniform,
            BindingKind::ReadStorage => wgpu::BufferBindingType::Storage { read_only: true },
            BindingKind::ReadWriteStorage => wgpu::BufferBindingType::Storage { read_only: false },
        };
        wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        }
    }
}

/// A fully built compute pipeline plus the layout its bind groups need.
pub(crate) struct CachedPipeline {
    /// Layout shared by every bind group created for this kernel.
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    /// The compiled compute pipeline.
    pub(crate) pipeline: wgpu::ComputePipeline,
}

impl CachedPipeline {
    /// Build a bind group binding `buffers` to slots `0..buffers.len()`.
    pub(crate) fn bind_group(
        &self,
        device: &wgpu::Device,
        label: &str,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .zip(0u32..)
            .map(|(buffer, binding)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bind_group_layout,
            entries: &entries,
        })
    }
}

/// Per-device cache of the crate's compute pipelines.
#[derive(Default)]
pub(crate) struct PipelineCache {
    ssm_scan_block: OnceLock<CachedPipeline>,
    ssm_scan_apply: OnceLock<CachedPipeline>,
    silu: OnceLock<CachedPipeline>,
    rms_norm: OnceLock<CachedPipeline>,
    matvec: OnceLock<CachedPipeline>,
}

impl PipelineCache {
    /// Return the pipeline for `kind`, building it on first use.
    ///
    /// # Errors
    ///
    /// Returns [`WebGpuError::Other`] if the cache slot cannot be read back
    /// after initialisation.  Shader compilation failures surface through the
    /// caller's device error scope.
    pub(crate) fn get(
        &self,
        device: &wgpu::Device,
        kind: KernelKind,
    ) -> Result<&CachedPipeline, WebGpuError> {
        let (cell, label, source, bindings): (_, _, _, &[BindingKind]) = match kind {
            KernelKind::SsmScanBlock => (
                &self.ssm_scan_block,
                "ssm-scan-block",
                SSM_SCAN_BLOCK_SRC,
                &[
                    BindingKind::Uniform,
                    BindingKind::ReadStorage,
                    BindingKind::ReadWriteStorage,
                    BindingKind::ReadWriteStorage,
                ],
            ),
            KernelKind::SsmScanApply => (
                &self.ssm_scan_apply,
                "ssm-scan-apply",
                SSM_SCAN_APPLY_SRC,
                &[
                    BindingKind::Uniform,
                    BindingKind::ReadStorage,
                    BindingKind::ReadWriteStorage,
                ],
            ),
            KernelKind::Silu => (
                &self.silu,
                "silu",
                SILU_SRC,
                &[
                    BindingKind::Uniform,
                    BindingKind::ReadStorage,
                    BindingKind::ReadWriteStorage,
                ],
            ),
            KernelKind::RmsNorm => (
                &self.rms_norm,
                "rms-norm",
                RMS_NORM_SRC,
                &[
                    BindingKind::Uniform,
                    BindingKind::ReadStorage,
                    BindingKind::ReadStorage,
                    BindingKind::ReadWriteStorage,
                ],
            ),
            KernelKind::Matvec => (
                &self.matvec,
                "matvec",
                MATVEC_SRC,
                &[
                    BindingKind::Uniform,
                    BindingKind::ReadStorage,
                    BindingKind::ReadStorage,
                    BindingKind::ReadWriteStorage,
                ],
            ),
        };

        if let Some(cached) = cell.get() {
            return Ok(cached);
        }

        // Two threads racing here both build; `set` keeps the first result and
        // the loser's pipeline is dropped.  Correctness is unaffected.
        let built = build_pipeline(device, label, source, bindings);
        let _ = cell.set(built);

        cell.get().ok_or_else(|| {
            WebGpuError::Other(format!(
                "pipeline cache slot '{label}' failed to initialise"
            ))
        })
    }
}

/// Compile `source` and build the compute pipeline for `bindings`.
fn build_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    bindings: &[BindingKind],
) -> CachedPipeline {
    let shader_label = format!("{label}-shader");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&shader_label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
        .iter()
        .zip(0u32..)
        .map(|(kind, binding)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: kind.binding_type(),
            count: None,
        })
        .collect();

    let bgl_label = format!("{label}-bgl");
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&bgl_label),
        entries: &entries,
    });

    let layout_label = format!("{label}-pipeline-layout");
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&layout_label),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline_label = format!("{label}-pipeline");
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&pipeline_label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    CachedPipeline {
        bind_group_layout,
        pipeline,
    }
}
