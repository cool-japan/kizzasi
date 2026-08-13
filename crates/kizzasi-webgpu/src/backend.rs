//! WebGPU compute backend implementation.
//!
//! The [`WebGpuBackend`] struct is only functional when compiled with
//! `--features webgpu`. Without it, every constructor returns
//! [`WebGpuError::BackendUnavailable`].
//!
//! # Error handling
//!
//! wgpu routes validation and out-of-memory failures to a *default error
//! handler that panics*, which would turn a recoverable GPU error into a
//! process abort.  This backend therefore
//!
//! 1. installs a non-panicking uncaptured-error handler on the device, and
//! 2. wraps every GPU entry point in a validation + out-of-memory error scope
//!    (`WebGpuBackend::run_scoped`), mapping anything captured to
//!    [`WebGpuError::Other`].
//!
//! Combined with the up-front device-limit checks, a bad request produces an
//! `Err` rather than killing the host process.

#[cfg(feature = "webgpu")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "webgpu")]
use std::time::Duration;

#[cfg(feature = "webgpu")]
use tracing::{debug, error, info};

use crate::buffer::GpuBuffer;
#[cfg(feature = "webgpu")]
use crate::buffer::GpuBufferUsage;
use crate::error::WebGpuError;
#[cfg(feature = "webgpu")]
use crate::pipeline::{CachedPipeline, KernelKind, PipelineCache};

/// Bytes per `f32`.
#[cfg(feature = "webgpu")]
pub(crate) const F32_BYTES: usize = std::mem::size_of::<f32>();

/// Upper bound on how long a blocking GPU wait may take before it is reported
/// as an error instead of hanging the caller forever.
#[cfg(feature = "webgpu")]
pub(crate) const GPU_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Summary information about the selected GPU adapter.
///
/// Unlike `wgpu::AdapterInfo`, this struct is always available regardless
/// of whether the `webgpu` feature is enabled.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    /// Human-readable GPU name (e.g. "Apple M2", "NVIDIA GeForce RTX 4080").
    pub name: String,
    /// Rendering backend in use (e.g. "Metal", "Vulkan", "Dx12").
    pub backend: String,
    /// Driver version string as reported by the OS.
    pub driver: String,
}

/// The device limits this crate's kernels validate their requests against.
///
/// Available without the `webgpu` feature so callers can size work items in
/// feature-gated code paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLimits {
    /// Largest byte size a single storage buffer binding may have.
    pub max_storage_buffer_binding_size: u64,
    /// Largest byte size a single buffer allocation may have.
    pub max_buffer_size: u64,
    /// Largest work-group count per dispatch dimension.
    pub max_compute_workgroups_per_dimension: u32,
}

/// WebGPU compute backend for signal processing acceleration.
///
/// # Feature gate
/// Only functional when compiled with `--features webgpu`. Without the
/// feature, [`WebGpuBackend::new`] returns [`WebGpuError::BackendUnavailable`].
///
/// # Example
/// ```no_run
/// # #[cfg(feature = "webgpu")]
/// # async fn example() -> Result<(), kizzasi_webgpu::WebGpuError> {
/// use kizzasi_webgpu::WebGpuBackend;
///
/// let backend = WebGpuBackend::new().await?;
/// let info = backend.adapter_info();
/// println!("GPU: {} ({})", info.name, info.backend);
///
/// let buf = backend.upload_f32(&[1.0_f32, 2.0, 3.0], "example")?;
/// let data = backend.download_f32(&buf)?;
/// assert_eq!(data, [1.0, 2.0, 3.0]);
/// # Ok(())
/// # }
/// ```
pub struct WebGpuBackend {
    /// The logical wgpu device handle.
    #[cfg(feature = "webgpu")]
    device: wgpu::Device,
    /// The command queue bound to the device.
    #[cfg(feature = "webgpu")]
    queue: wgpu::Queue,
    /// Limits the device was created with, cached for pre-dispatch validation.
    #[cfg(feature = "webgpu")]
    limits: DeviceLimits,
    /// Slot filled by the uncaptured-error handler installed on the device.
    #[cfg(feature = "webgpu")]
    uncaptured_error: Arc<Mutex<Option<String>>>,
    /// Lazily built compute pipelines, shared across all kernel invocations.
    #[cfg(feature = "webgpu")]
    pipelines: PipelineCache,
    /// Adapter metadata cached at construction time.
    adapter_info_cache: AdapterInfo,
}

impl WebGpuBackend {
    /// Initialize a WebGPU compute backend.
    ///
    /// Requests the high-performance adapter and creates a logical device with
    /// the adapter's own limits, so capable hardware is not pinned to wgpu's
    /// downlevel defaults.  Use
    /// [`with_limits`](WebGpuBackend::with_limits) to request something else.
    ///
    /// The instance uses the platform-native backend (Metal on macOS,
    /// Vulkan on Linux/Windows, DX12 on Windows).
    ///
    /// # Errors
    ///
    /// - [`WebGpuError::BackendUnavailable`] when compiled without `--features webgpu`.
    /// - [`WebGpuError::AdapterRequest`] when no suitable GPU adapter is found.
    /// - [`WebGpuError::DeviceRequest`] when device creation fails.
    pub async fn new() -> Result<Self, WebGpuError> {
        #[cfg(not(feature = "webgpu"))]
        {
            Err(WebGpuError::BackendUnavailable)
        }

        #[cfg(feature = "webgpu")]
        {
            Self::new_impl(None).await
        }
    }

    /// Initialize a backend with explicit device limits.
    ///
    /// Requesting limits the adapter cannot satisfy fails with
    /// [`WebGpuError::DeviceRequest`] rather than aborting.
    ///
    /// # Errors
    ///
    /// Same as [`WebGpuBackend::new`].
    #[cfg(feature = "webgpu")]
    pub async fn with_limits(limits: wgpu::Limits) -> Result<Self, WebGpuError> {
        Self::new_impl(Some(limits)).await
    }

    /// Returns adapter information (GPU name, backend, driver).
    pub fn adapter_info(&self) -> AdapterInfo {
        self.adapter_info_cache.clone()
    }

    /// Returns the limits the logical device was created with.
    #[cfg(feature = "webgpu")]
    pub fn device_limits(&self) -> DeviceLimits {
        self.limits
    }

    /// Upload a slice of `f32` values to a GPU storage buffer.
    ///
    /// The returned [`GpuBuffer`] has `STORAGE | COPY_SRC | COPY_DST` usage
    /// so it can be bound in compute shaders, fed to the `*_gpu_buf` kernels
    /// and copied back to staging buffers.
    ///
    /// # Errors
    ///
    /// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
    /// - [`WebGpuError::Other`] for an empty slice — GPU buffers cannot be
    ///   zero-sized — or for a captured wgpu error.
    /// - [`WebGpuError::DeviceLimitExceeded`] if the data exceeds the device's
    ///   buffer limits.
    pub fn upload_f32(&self, data: &[f32], label: &str) -> Result<GpuBuffer, WebGpuError> {
        #[cfg(not(feature = "webgpu"))]
        {
            let _ = (data, label);
            Err(WebGpuError::BackendUnavailable)
        }

        #[cfg(feature = "webgpu")]
        {
            self.run_scoped(|| self.upload_f32_impl(data, label))
        }
    }

    /// Download `f32` values from a GPU storage buffer back to the CPU.
    ///
    /// This call **blocks** the calling thread until the GPU mapping completes
    /// (bounded by an internal 30 s timeout).  It is deliberately not `async`:
    /// the underlying `wgpu` synchronisation is blocking, and pretending
    /// otherwise would stall an async executor's worker thread.  From an async
    /// context, call it inside `spawn_blocking`.
    ///
    /// # Errors
    ///
    /// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
    /// - [`WebGpuError::NoGpuAllocation`] for a metadata-only buffer.
    /// - [`WebGpuError::BufferSizeMismatch`] if the buffer byte count is not a
    ///   multiple of `size_of::<f32>()`.
    /// - [`WebGpuError::MapBuffer`] if the GPU mapping fails or times out.
    pub fn download_f32(&self, buf: &GpuBuffer) -> Result<Vec<f32>, WebGpuError> {
        #[cfg(not(feature = "webgpu"))]
        {
            let _ = buf;
            Err(WebGpuError::BackendUnavailable)
        }

        #[cfg(feature = "webgpu")]
        {
            self.run_scoped(|| self.download_f32_impl(buf))
        }
    }

    /// Submit an empty command buffer — useful for pipeline synchronisation in tests.
    ///
    /// # Errors
    ///
    /// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
    pub fn submit_noop(&self) -> Result<(), WebGpuError> {
        #[cfg(not(feature = "webgpu"))]
        {
            Err(WebGpuError::BackendUnavailable)
        }

        #[cfg(feature = "webgpu")]
        {
            self.run_scoped(|| self.submit_noop_impl())
        }
    }
}

// ── Private implementation — only compiled with the `webgpu` feature ─────────

#[cfg(feature = "webgpu")]
impl WebGpuBackend {
    async fn new_impl(requested_limits: Option<wgpu::Limits>) -> Result<Self, WebGpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        debug!("wgpu instance created, requesting adapter");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                // Limit bucketing exists to reduce GPU-fingerprinting surface when `wgpu` is
                // exposed to untrusted web content; kizzasi-webgpu is a native, trusted compute
                // backend, so bucketing is left off (matches wgpu's own internal default).
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| WebGpuError::AdapterRequest(e.to_string()))?;

        let raw_info = adapter.get_info();
        let adapter_info_cache = AdapterInfo {
            name: raw_info.name.clone(),
            backend: format!("{:?}", raw_info.backend),
            driver: raw_info.driver.clone(),
        };

        info!(
            gpu = %raw_info.name,
            backend = ?raw_info.backend,
            driver = %raw_info.driver,
            "WebGPU adapter selected",
        );

        // No kernel in this crate uses f16, so no optional device feature is
        // requested; asking for one that gates no behaviour only misleads
        // triage.
        let required_limits = requested_limits.unwrap_or_else(|| adapter.limits());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("kizzasi-webgpu device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            })
            .await
            .map_err(|e| WebGpuError::DeviceRequest(e.to_string()))?;

        // Replace wgpu's panicking default error handler.  Errors raised
        // outside an active error scope are recorded here and surfaced by the
        // next `run_scoped` call instead of aborting the process.
        let uncaptured_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&uncaptured_error);
        device.on_uncaptured_error(Arc::new(move |err: wgpu::Error| {
            error!(error = %err, "uncaptured wgpu error");
            let mut slot = match sink.lock() {
                Ok(slot) => slot,
                Err(poisoned) => poisoned.into_inner(),
            };
            if slot.is_none() {
                *slot = Some(err.to_string());
            }
        }));

        let device_limits = device.limits();
        let limits = DeviceLimits {
            max_storage_buffer_binding_size: device_limits.max_storage_buffer_binding_size,
            max_buffer_size: device_limits.max_buffer_size,
            max_compute_workgroups_per_dimension: device_limits
                .max_compute_workgroups_per_dimension,
        };

        debug!(?limits, "wgpu device created successfully");

        Ok(Self {
            device,
            queue,
            limits,
            uncaptured_error,
            pipelines: PipelineCache::default(),
            adapter_info_cache,
        })
    }

    /// Expose device and queue handles to crate-internal GPU code.
    ///
    /// This keeps the raw wgpu handles encapsulated while allowing kernel
    /// modules (e.g. `ssm_scan`) to drive compute pipeline execution.
    pub(crate) fn device_and_queue(&self) -> (&wgpu::Device, &wgpu::Queue) {
        (&self.device, &self.queue)
    }

    /// Return the cached pipeline for `kind`, building it on first use.
    pub(crate) fn pipeline(&self, kind: KernelKind) -> Result<&CachedPipeline, WebGpuError> {
        self.pipelines.get(&self.device, kind)
    }

    /// Take (and clear) the error recorded by the uncaptured-error handler.
    fn take_uncaptured_error(&self) -> Option<String> {
        let mut slot = match self.uncaptured_error.lock() {
            Ok(slot) => slot,
            Err(poisoned) => poisoned.into_inner(),
        };
        slot.take()
    }

    /// Run `op` inside validation + out-of-memory error scopes.
    ///
    /// Any wgpu error raised while `op` runs is captured and returned as
    /// [`WebGpuError::Other`] instead of reaching wgpu's panicking default
    /// handler.
    pub(crate) fn run_scoped<T, F>(&self, op: F) -> Result<T, WebGpuError>
    where
        F: FnOnce() -> Result<T, WebGpuError>,
    {
        // Drop anything left over from an earlier, unrelated failure so it is
        // not misattributed to this operation.
        let _ = self.take_uncaptured_error();

        let validation_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let oom_scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

        let outcome = op();

        // Scopes must be popped in reverse order of creation.  On native
        // backends both futures are already resolved; a guard dropped on an
        // early return pops its scope in `Drop`.
        let oom_error = resolve_now(oom_scope.pop())?;
        let validation_error = resolve_now(validation_scope.pop())?;

        let captured = oom_error
            .or(validation_error)
            .map(|err| err.to_string())
            .or_else(|| self.take_uncaptured_error());

        match (outcome, captured) {
            (Ok(value), None) => Ok(value),
            (Ok(_), Some(gpu_error)) => Err(WebGpuError::Other(format!("wgpu error: {gpu_error}"))),
            (Err(err), None) => Err(err),
            (Err(err), Some(gpu_error)) => Err(WebGpuError::Other(format!(
                "wgpu error: {gpu_error} (operation also reported: {err})"
            ))),
        }
    }

    /// Reject a storage binding larger than the device supports.
    pub(crate) fn check_binding_size(&self, what: &str, bytes: u64) -> Result<(), WebGpuError> {
        if bytes > self.limits.max_storage_buffer_binding_size {
            return Err(WebGpuError::DeviceLimitExceeded {
                what: format!("{what} storage binding (bytes)"),
                required: bytes,
                limit: self.limits.max_storage_buffer_binding_size,
            });
        }
        if bytes > self.limits.max_buffer_size {
            return Err(WebGpuError::DeviceLimitExceeded {
                what: format!("{what} buffer allocation (bytes)"),
                required: bytes,
                limit: self.limits.max_buffer_size,
            });
        }
        Ok(())
    }

    /// Reject a dispatch wider than the device supports.
    pub(crate) fn check_workgroups(&self, what: &str, count: u32) -> Result<(), WebGpuError> {
        if count > self.limits.max_compute_workgroups_per_dimension {
            return Err(WebGpuError::DeviceLimitExceeded {
                what: format!("{what} work-group count"),
                required: u64::from(count),
                limit: u64::from(self.limits.max_compute_workgroups_per_dimension),
            });
        }
        Ok(())
    }

    /// Allocate a `STORAGE | COPY_SRC | COPY_DST` buffer of `size_bytes`.
    pub(crate) fn create_storage_buffer(
        &self,
        label: &str,
        size_bytes: u64,
    ) -> Result<wgpu::Buffer, WebGpuError> {
        if size_bytes == 0 {
            return Err(WebGpuError::Other(format!(
                "{label}: cannot create a zero-sized GPU buffer"
            )));
        }
        self.check_binding_size(label, size_bytes)?;
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    /// Allocate a `MAP_READ | COPY_DST` staging buffer of `size_bytes`.
    pub(crate) fn create_staging_buffer(
        &self,
        label: &str,
        size_bytes: u64,
    ) -> Result<wgpu::Buffer, WebGpuError> {
        if size_bytes == 0 {
            return Err(WebGpuError::Other(format!(
                "{label}: cannot create a zero-sized staging buffer"
            )));
        }
        if size_bytes > self.limits.max_buffer_size {
            return Err(WebGpuError::DeviceLimitExceeded {
                what: format!("{label} staging allocation (bytes)"),
                required: size_bytes,
                limit: self.limits.max_buffer_size,
            });
        }
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    /// Allocate a `UNIFORM | COPY_DST` buffer and queue `data` into it.
    pub(crate) fn create_uniform_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, data);
        buffer
    }

    /// Allocate a storage buffer and queue `bytes` into it.
    pub(crate) fn upload_bytes(
        &self,
        label: &str,
        bytes: &[u8],
    ) -> Result<wgpu::Buffer, WebGpuError> {
        let buffer = self.create_storage_buffer(label, bytes.len() as u64)?;
        self.queue.write_buffer(&buffer, 0, bytes);
        Ok(buffer)
    }

    fn upload_f32_impl(&self, data: &[f32], label: &str) -> Result<GpuBuffer, WebGpuError> {
        if data.is_empty() {
            return Err(WebGpuError::Other(format!(
                "{label}: cannot upload an empty slice; GPU buffers cannot be zero-sized"
            )));
        }

        let size_bytes = std::mem::size_of_val(data) as u64;
        let buffer = self.upload_bytes(label, f32_slice_as_bytes(data))?;

        debug!(
            label,
            size_bytes, "f32 slice uploaded to GPU storage buffer"
        );

        Ok(GpuBuffer::from_wgpu(
            buffer,
            size_bytes,
            GpuBufferUsage::Storage,
            label,
        ))
    }

    fn download_f32_impl(&self, buf: &GpuBuffer) -> Result<Vec<f32>, WebGpuError> {
        let source = buf.wgpu_buffer()?;
        let size_bytes = buf.size_bytes;
        let f32_size = F32_BYTES as u64;

        if !size_bytes.is_multiple_of(f32_size) {
            return Err(WebGpuError::BufferSizeMismatch {
                expected: (size_bytes / f32_size) * f32_size,
                got: size_bytes,
            });
        }

        let staging = self.create_staging_buffer(&format!("{}-staging", buf.label), size_bytes)?;

        // Encode the copy from storage → staging.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kizzasi-webgpu download encoder"),
            });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let floats = read_staging(&self.device, &staging, size_bytes, decode_f32)?;

        debug!(
            label = %buf.label,
            count = floats.len(),
            "f32 data downloaded from GPU",
        );

        Ok(floats)
    }

    fn submit_noop_impl(&self) -> Result<(), WebGpuError> {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kizzasi-webgpu noop"),
            });
        self.queue.submit(std::iter::once(encoder.finish()));
        debug!("noop command submitted");
        Ok(())
    }
}

// ── GPU synchronisation helpers ──────────────────────────────────────────────

/// Resolve a future that native `wgpu` guarantees is already complete.
///
/// Used for `pop_error_scope`, whose future is `Ready` on every native
/// backend.  A `Pending` result is reported rather than silently discarded.
#[cfg(feature = "webgpu")]
fn resolve_now<F: std::future::Future>(future: F) -> Result<F::Output, WebGpuError> {
    let mut future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut cx) {
        std::task::Poll::Ready(value) => Ok(value),
        std::task::Poll::Pending => Err(WebGpuError::Other(
            "wgpu error scope did not resolve synchronously on this backend".into(),
        )),
    }
}

/// Block until queued GPU work has completed, with a bounded wait.
#[cfg(feature = "webgpu")]
pub(crate) fn poll_device(device: &wgpu::Device, stage: &str) -> Result<(), WebGpuError> {
    match device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(GPU_WAIT_TIMEOUT),
    }) {
        Ok(_) => Ok(()),
        Err(wgpu::PollError::Timeout) => Err(WebGpuError::Other(format!(
            "GPU did not complete {stage} within {GPU_WAIT_TIMEOUT:?}"
        ))),
        Err(err) => Err(WebGpuError::Other(format!(
            "device poll error during {stage}: {err}"
        ))),
    }
}

/// Map `staging`, hand the bytes to `decode`, and unmap on every exit path.
#[cfg(feature = "webgpu")]
pub(crate) fn read_staging<T, F>(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    size_bytes: u64,
    decode: F,
) -> Result<T, WebGpuError>
where
    F: FnOnce(&[u8]) -> Result<T, WebGpuError>,
{
    poll_device(device, "compute submission")?;

    let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

    poll_device(device, "buffer map")?;

    match rx.recv_timeout(GPU_WAIT_TIMEOUT) {
        Ok(Ok(())) => {}
        Ok(Err(err)) => return Err(WebGpuError::MapBuffer(err.to_string())),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            return Err(WebGpuError::MapBuffer(format!(
                "buffer map did not complete within {GPU_WAIT_TIMEOUT:?}"
            )))
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            return Err(WebGpuError::MapBuffer(
                "channel closed before map completed".into(),
            ))
        }
    }

    // From here the buffer is mapped: every exit path must unmap it.
    let decoded = match staging.slice(..).get_mapped_range() {
        Ok(mapped) => {
            let mapped_len = mapped.len() as u64;
            let outcome = if mapped_len == size_bytes {
                decode(&mapped)
            } else {
                Err(WebGpuError::BufferSizeMismatch {
                    expected: size_bytes,
                    got: mapped_len,
                })
            };
            drop(mapped);
            outcome
        }
        Err(err) => Err(WebGpuError::MapBuffer(err.to_string())),
    };

    staging.unmap();
    decoded
}

/// Decode a mapped range as a `Vec<f32>`.
#[cfg(feature = "webgpu")]
pub(crate) fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, WebGpuError> {
    if !bytes.len().is_multiple_of(F32_BYTES) {
        return Err(WebGpuError::BufferSizeMismatch {
            expected: (bytes.len() / F32_BYTES * F32_BYTES) as u64,
            got: bytes.len() as u64,
        });
    }

    let mut out = Vec::with_capacity(bytes.len() / F32_BYTES);
    for chunk in bytes.chunks_exact(F32_BYTES) {
        out.push(f32_from_bytes(chunk)?);
    }
    Ok(out)
}

/// Decode a mapped range as interleaved `(a, bu)` pairs.
#[cfg(feature = "webgpu")]
pub(crate) fn decode_pairs(bytes: &[u8]) -> Result<Vec<(f32, f32)>, WebGpuError> {
    let pair_bytes = F32_BYTES * 2;
    if !bytes.len().is_multiple_of(pair_bytes) {
        return Err(WebGpuError::BufferSizeMismatch {
            expected: (bytes.len() / pair_bytes * pair_bytes) as u64,
            got: bytes.len() as u64,
        });
    }

    let mut out = Vec::with_capacity(bytes.len() / pair_bytes);
    for chunk in bytes.chunks_exact(pair_bytes) {
        let (a_bytes, bu_bytes) = chunk.split_at(F32_BYTES);
        out.push((f32_from_bytes(a_bytes)?, f32_from_bytes(bu_bytes)?));
    }
    Ok(out)
}

/// Decode exactly four native-endian bytes as an `f32`.
#[cfg(feature = "webgpu")]
fn f32_from_bytes(bytes: &[u8]) -> Result<f32, WebGpuError> {
    let array = <[u8; 4]>::try_from(bytes).map_err(|_| WebGpuError::BufferSizeMismatch {
        expected: F32_BYTES as u64,
        got: bytes.len() as u64,
    })?;
    Ok(f32::from_ne_bytes(array))
}

// ── Byte-casting helper ───────────────────────────────────────────────────────

/// Reinterpret a `&[f32]` as `&[u8]` without copying.
///
/// This is safe because:
/// - `f32` has no invalid byte representations.
/// - The resulting slice has the same lifetime as the input.
/// - `u8` has alignment 1, so no alignment issues can arise.
///
/// We avoid pulling in the `bytemuck` crate for this single use.
#[cfg(feature = "webgpu")]
pub(crate) fn f32_slice_as_bytes(data: &[f32]) -> &[u8] {
    // SAFETY: see doc comment above.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

impl std::fmt::Debug for WebGpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebGpuBackend")
            .field("adapter", &self.adapter_info_cache)
            .finish_non_exhaustive()
    }
}

// ── Test-only helpers ─────────────────────────────────────────────────────────

/// Construct a backend shell for tests that run without the `webgpu` feature.
///
/// In that configuration the struct holds no wgpu handles at all, so the value
/// is enough to exercise the `BackendUnavailable` arms of the kernel entry
/// points — which are exactly the arms a default-feature test run compiles.
#[cfg(all(test, not(feature = "webgpu")))]
pub(crate) fn unavailable_backend() -> WebGpuBackend {
    WebGpuBackend {
        adapter_info_cache: AdapterInfo {
            name: "unavailable".into(),
            backend: "none".into(),
            driver: "none".into(),
        },
    }
}

#[cfg(all(test, feature = "webgpu"))]
mod gpu_tests {
    use super::*;

    async fn try_backend() -> Option<WebGpuBackend> {
        crate::test_support::try_backend().await
    }

    /// A wgpu validation error must surface as `Err`, not as a process abort.
    ///
    /// Without the error scope installed by `run_scoped`, wgpu's default
    /// handler panics the process on this request.
    #[tokio::test]
    async fn test_error_scope_captures_validation_error() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let (device, _queue) = backend.device_and_queue();
        let result = backend.run_scoped(|| {
            // Far beyond `max_buffer_size` on any device: wgpu raises a
            // validation error through the error sink.
            let _invalid = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("deliberately-oversized"),
                size: u64::MAX / 2,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            Ok(())
        });

        match result {
            Err(WebGpuError::Other(message)) => {
                assert!(
                    message.contains("wgpu error"),
                    "expected a captured wgpu error, got: {message}"
                );
            }
            other => panic!("expected captured validation error, got: {other:?}"),
        }
    }

    /// Device limits must be populated and used by the pre-dispatch checks.
    #[tokio::test]
    async fn test_device_limit_checks() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let limits = backend.device_limits();
        assert!(limits.max_storage_buffer_binding_size > 0);
        assert!(limits.max_compute_workgroups_per_dimension > 0);

        let err = backend
            .check_binding_size("test", limits.max_buffer_size.saturating_add(1))
            .expect_err("oversized binding must be rejected");
        assert!(matches!(err, WebGpuError::DeviceLimitExceeded { .. }));

        let err = backend
            .check_workgroups(
                "test",
                limits
                    .max_compute_workgroups_per_dimension
                    .saturating_add(1),
            )
            .expect_err("oversized dispatch must be rejected");
        assert!(matches!(err, WebGpuError::DeviceLimitExceeded { .. }));
    }

    /// `upload_f32` must reject an empty slice instead of creating a
    /// zero-sized buffer that fails validation at bind time.
    #[tokio::test]
    async fn test_upload_empty_slice_is_rejected() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let err = backend
            .upload_f32(&[], "empty")
            .expect_err("empty upload must fail");
        assert!(matches!(err, WebGpuError::Other(_)), "got: {err:?}");
    }

    /// Downloading from a metadata-only buffer must be a typed error.
    #[tokio::test]
    async fn test_download_metadata_only_buffer() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let buf = GpuBuffer::metadata_only(16, GpuBufferUsage::Storage, "metadata-only");
        let err = backend
            .download_f32(&buf)
            .expect_err("metadata-only buffer has no GPU allocation");
        assert!(
            matches!(err, WebGpuError::NoGpuAllocation(_)),
            "got: {err:?}"
        );
    }
}
