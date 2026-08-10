//! WebGPU compute backend implementation.
//!
//! The [`WebGpuBackend`] struct is only functional when compiled with
//! `--features webgpu`. Without it, every constructor returns
//! [`WebGpuError::BackendUnavailable`].

#[cfg(feature = "webgpu")]
use tracing::{debug, info};

use crate::buffer::GpuBuffer;
#[cfg(feature = "webgpu")]
use crate::buffer::GpuBufferUsage;
use crate::error::WebGpuError;

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
/// let data = backend.download_f32(&buf).await?;
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
    /// Adapter metadata cached at construction time.
    adapter_info_cache: AdapterInfo,
}

impl WebGpuBackend {
    /// Initialize a WebGPU compute backend.
    ///
    /// Requests the high-performance adapter and creates a logical device.
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
            Self::new_impl().await
        }
    }

    /// Returns adapter information (GPU name, backend, driver).
    pub fn adapter_info(&self) -> AdapterInfo {
        self.adapter_info_cache.clone()
    }

    /// Upload a slice of `f32` values to a GPU storage buffer.
    ///
    /// The returned [`GpuBuffer`] has `STORAGE | COPY_SRC | COPY_DST` usage
    /// so it can be bound in compute shaders and copied to staging buffers.
    ///
    /// # Errors
    ///
    /// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
    pub fn upload_f32(&self, data: &[f32], label: &str) -> Result<GpuBuffer, WebGpuError> {
        #[cfg(not(feature = "webgpu"))]
        {
            let _ = (data, label);
            Err(WebGpuError::BackendUnavailable)
        }

        #[cfg(feature = "webgpu")]
        {
            self.upload_f32_impl(data, label)
        }
    }

    /// Download `f32` values from a GPU storage buffer back to the CPU.
    ///
    /// The download is synchronous from the caller's perspective: this method
    /// blocks (via `device.poll`) until the GPU mapping completes.
    ///
    /// # Errors
    ///
    /// - [`WebGpuError::BackendUnavailable`] without `--features webgpu`.
    /// - [`WebGpuError::BufferSizeMismatch`] if the buffer byte count is not a
    ///   multiple of `size_of::<f32>()`.
    /// - [`WebGpuError::MapBuffer`] if the GPU mapping fails.
    pub async fn download_f32(&self, buf: &GpuBuffer) -> Result<Vec<f32>, WebGpuError> {
        #[cfg(not(feature = "webgpu"))]
        {
            let _ = buf;
            Err(WebGpuError::BackendUnavailable)
        }

        #[cfg(feature = "webgpu")]
        {
            self.download_f32_impl(buf).await
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
            self.submit_noop_impl()
        }
    }
}

// ── Private implementation — only compiled with the `webgpu` feature ─────────

#[cfg(feature = "webgpu")]
impl WebGpuBackend {
    async fn new_impl() -> Result<Self, WebGpuError> {
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

        // Request device. Enable SHADER_F16 if the adapter supports it.
        let features = if adapter.features().contains(wgpu::Features::SHADER_F16) {
            debug!("SHADER_F16 supported, enabling");
            wgpu::Features::SHADER_F16
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("kizzasi-webgpu device"),
                required_features: features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            })
            .await
            .map_err(|e| WebGpuError::DeviceRequest(e.to_string()))?;

        debug!("wgpu device created successfully");

        Ok(Self {
            device,
            queue,
            adapter_info_cache,
        })
    }

    fn upload_f32_impl(&self, data: &[f32], label: &str) -> Result<GpuBuffer, WebGpuError> {
        use wgpu::util::DeviceExt as _;

        let size_bytes = std::mem::size_of_val(data) as u64;

        // Reinterpret the f32 slice as raw bytes for the wgpu initialiser.
        // SAFETY: f32 has no padding or invalid byte patterns; the slice is valid for the lifetime
        // of `data`, and `u8` has no alignment requirements.
        let bytes: &[u8] = bytemuck_cast(data);

        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

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

    async fn download_f32_impl(&self, buf: &GpuBuffer) -> Result<Vec<f32>, WebGpuError> {
        let size_bytes = buf.size_bytes;
        let f32_size = std::mem::size_of::<f32>() as u64;

        if !size_bytes.is_multiple_of(f32_size) {
            return Err(WebGpuError::BufferSizeMismatch {
                expected: (size_bytes / f32_size) * f32_size,
                got: size_bytes,
            });
        }

        // Create a staging buffer for the MAP_READ download.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}-staging", buf.label)),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Encode the copy from storage → staging.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kizzasi-webgpu download encoder"),
            });
        encoder.copy_buffer_to_buffer(&buf.inner, 0, &staging, 0, size_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Block until the GPU is done.
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| WebGpuError::Other(format!("device poll error: {e:?}")))?;

        // Map the staging buffer and read back the data.
        let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                // The send can only fail if the receiver was dropped, which cannot happen
                // because we hold `rx` alive in this scope.
                let _ = tx.send(result);
            });

        // Poll again to drive the mapping callback.
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| WebGpuError::Other(format!("device poll error after map: {e:?}")))?;

        rx.recv()
            .map_err(|_| WebGpuError::MapBuffer("channel closed before map completed".into()))?
            .map_err(|e| WebGpuError::MapBuffer(e.to_string()))?;

        let mapped = staging
            .slice(..)
            .get_mapped_range()
            .map_err(|e| WebGpuError::MapBuffer(e.to_string()))?;
        let bytes: &[u8] = &mapped;
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_ne_bytes(arr)
            })
            .collect();

        drop(mapped);
        staging.unmap();

        debug!(
            label = %buf.label,
            count = floats.len(),
            "f32 data downloaded from GPU",
        );

        Ok(floats)
    }

    /// Expose device and queue handles to crate-internal GPU code.
    ///
    /// This keeps the raw wgpu handles encapsulated while allowing kernel
    /// modules (e.g. `ssm_scan`) to drive compute pipeline execution.
    pub(crate) fn device_and_queue(&self) -> (&wgpu::Device, &wgpu::Queue) {
        (&self.device, &self.queue)
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
fn bytemuck_cast(data: &[f32]) -> &[u8] {
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
