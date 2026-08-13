//! [`SsmBackend`] implementation using [`WebGpuBackend`].
//!
//! [`WebGpuSsmBackend`] is a **hybrid** backend: it runs the GPU Blelloch scan
//! for sequences at or above a configurable crossover length and the CPU scan
//! below it, because a short scan is dominated by dispatch and readback
//! overhead rather than arithmetic.  If the GPU path fails for any reason
//! (device loss, allocation failure, a captured validation error) it logs a
//! warning and completes the request on the CPU rather than failing the
//! caller's inference.

use std::sync::atomic::{AtomicU8, Ordering};

use kizzasi_core::ssm_backend::{CpuSsmBackend, SsmBackend};
use kizzasi_core::CoreResult;
use tracing::warn;

use crate::ssm_scan::ssm_scan_gpu;
use crate::WebGpuBackend;

/// Default sequence length at which [`WebGpuSsmBackend`] switches to the GPU.
///
/// Below this the fixed cost of buffer uploads, a dispatch and a blocking
/// readback exceeds the cost of the sequential CPU scan.  Tune it for the
/// target device with [`WebGpuSsmBackend::with_gpu_threshold`].
pub const DEFAULT_GPU_THRESHOLD: usize = 1024;

/// No scan has run yet.
const PATH_NONE: u8 = 0;
/// The most recent scan ran on the GPU.
const PATH_GPU: u8 = 1;
/// The most recent scan ran on the CPU because the input was short.
const PATH_CPU_SHORT: u8 = 2;
/// The most recent scan ran on the CPU after the GPU path failed.
const PATH_CPU_FALLBACK: u8 = 3;

/// GPU-accelerated SSM backend wrapping [`WebGpuBackend`].
///
/// # Dispatch policy
///
/// | Input length | Path |
/// |---|---|
/// | `< gpu_threshold` | [`CpuSsmBackend`] |
/// | `>= gpu_threshold` | GPU multi-block scan |
/// | GPU path returned `Err` | [`CpuSsmBackend`], with a `warn!` log |
///
/// The GPU kernel handles sequences of any length, so the threshold is purely
/// a performance choice — never a correctness one.
///
/// # Feature gate
///
/// The struct is always available as a type.  Without `--features webgpu` the
/// GPU path is unavailable, so every scan degrades to the CPU implementation
/// and [`backend_name`](SsmBackend::backend_name) reports that.
pub struct WebGpuSsmBackend {
    backend: WebGpuBackend,
    gpu_threshold: usize,
    last_path: AtomicU8,
}

impl WebGpuSsmBackend {
    /// Wrap an existing [`WebGpuBackend`] with the default crossover length.
    pub fn new(backend: WebGpuBackend) -> Self {
        Self {
            backend,
            gpu_threshold: DEFAULT_GPU_THRESHOLD,
            last_path: AtomicU8::new(PATH_NONE),
        }
    }

    /// Set the sequence length at or above which the GPU path is used.
    ///
    /// A threshold of `0` sends every non-trivial sequence to the GPU.
    pub fn with_gpu_threshold(mut self, threshold: usize) -> Self {
        self.gpu_threshold = threshold;
        self
    }

    /// The configured GPU crossover length.
    pub fn gpu_threshold(&self) -> usize {
        self.gpu_threshold
    }

    /// Borrow the wrapped GPU backend.
    pub fn backend(&self) -> &WebGpuBackend {
        &self.backend
    }

    fn record_path(&self, path: u8) {
        self.last_path.store(path, Ordering::Relaxed);
    }
}

impl SsmBackend for WebGpuSsmBackend {
    fn ssm_scan(&self, elements: &[(f32, f32)]) -> CoreResult<Vec<(f32, f32)>> {
        if elements.len() < self.gpu_threshold {
            // Too short to amortise dispatch + readback; the CPU scan wins.
            self.record_path(PATH_CPU_SHORT);
            return CpuSsmBackend.ssm_scan(elements);
        }

        match ssm_scan_gpu(&self.backend, elements) {
            Ok(result) => {
                self.record_path(PATH_GPU);
                Ok(result)
            }
            Err(err) => {
                warn!(
                    error = %err,
                    len = elements.len(),
                    "GPU SSM scan failed; completing on the CPU backend",
                );
                self.record_path(PATH_CPU_FALLBACK);
                CpuSsmBackend.ssm_scan(elements)
            }
        }
    }

    /// Reports the path the most recent [`ssm_scan`](SsmBackend::ssm_scan)
    /// actually took, so telemetry can distinguish GPU work from CPU work.
    fn backend_name(&self) -> &str {
        match self.last_path.load(Ordering::Relaxed) {
            PATH_GPU => "webgpu",
            PATH_CPU_SHORT => "webgpu-hybrid(cpu-short-input)",
            PATH_CPU_FALLBACK => "webgpu-hybrid(cpu-fallback)",
            _ => "webgpu-hybrid",
        }
    }
}

impl std::fmt::Debug for WebGpuSsmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebGpuSsmBackend")
            .field("backend", &self.backend)
            .field("gpu_threshold", &self.gpu_threshold)
            .field("last_path", &self.backend_name())
            .finish()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "webgpu"))]
mod gpu_tests {
    use super::*;
    use crate::test_support::{assert_pairs_close, try_backend};

    fn sample_elements(n: usize) -> Vec<(f32, f32)> {
        (0..n)
            .map(|i| {
                let a = 0.5 + 0.4 * ((i * 13 + 7) % 10) as f32 / 10.0;
                let bu = ((i * 7 + 3) % 5) as f32 * 0.5;
                (a, bu)
            })
            .collect()
    }

    /// Regression: long sequences used to be routed *away* from the GPU while
    /// short ones paid full pipeline setup.  The policy is now inverted, and
    /// both sides of the threshold must agree with the CPU reference.
    #[tokio::test]
    async fn test_threshold_routes_long_sequences_to_gpu() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let hybrid = WebGpuSsmBackend::new(backend).with_gpu_threshold(512);
        assert_eq!(hybrid.gpu_threshold(), 512);

        let long = sample_elements(2000);
        let gpu_result = hybrid.ssm_scan(&long).expect("long scan failed");
        assert_eq!(hybrid.backend_name(), "webgpu");
        assert_pairs_close(
            &CpuSsmBackend.ssm_scan(&long).expect("cpu scan failed"),
            &gpu_result,
            1e-3,
        );

        let short = sample_elements(16);
        let cpu_result = hybrid.ssm_scan(&short).expect("short scan failed");
        assert_eq!(hybrid.backend_name(), "webgpu-hybrid(cpu-short-input)");
        assert_pairs_close(
            &CpuSsmBackend.ssm_scan(&short).expect("cpu scan failed"),
            &cpu_result,
            1e-5,
        );
    }

    /// With a zero threshold every sequence goes to the GPU and must still
    /// match the CPU reference across the block boundary.
    #[tokio::test]
    async fn test_zero_threshold_uses_gpu_everywhere() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let hybrid = WebGpuSsmBackend::new(backend).with_gpu_threshold(0);

        for n in [2_usize, 257, 600] {
            let elements = sample_elements(n);
            let result = hybrid.ssm_scan(&elements).expect("scan failed");
            assert_eq!(result.len(), n);
            assert_pairs_close(
                &CpuSsmBackend.ssm_scan(&elements).expect("cpu scan failed"),
                &result,
                1e-3,
            );
        }
        assert_eq!(hybrid.backend_name(), "webgpu");
    }

    /// The default threshold must be reported before any scan has run.
    #[tokio::test]
    async fn test_default_threshold_and_initial_name() {
        let Some(backend) = try_backend().await else {
            return;
        };

        let hybrid = WebGpuSsmBackend::new(backend);
        assert_eq!(hybrid.gpu_threshold(), DEFAULT_GPU_THRESHOLD);
        assert_eq!(hybrid.backend_name(), "webgpu-hybrid");
    }
}
