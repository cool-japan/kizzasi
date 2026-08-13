//! Backend selection for SSM prefix scans.
//!
//! [`SsmBackend`] and [`CpuSsmBackend`] are defined in `kizzasi-core`; the GPU
//! implementation (`WebGpuSsmBackend`) lives in `kizzasi-webgpu`, which
//! *depends on* `kizzasi-core`. That direction is what makes this module
//! necessary: `kizzasi-core` cannot reach the GPU backend without a dependency
//! cycle, so the facade crate — which already sits above both — is where the
//! two are joined.
//!
//! ```text
//! kizzasi (this crate, `webgpu` feature)
//!    ├── kizzasi-core        — SsmBackend trait + CpuSsmBackend
//!    └── kizzasi-webgpu      — WebGpuSsmBackend (impl SsmBackend)
//! ```
//!
//! # What the `webgpu` feature actually changes
//!
//! Off (the default): [`select_ssm_backend`] always yields [`CpuSsmBackend`],
//! and `kizzasi-webgpu` is not compiled or linked at all.
//!
//! On: [`select_ssm_backend`] tries to initialise a wgpu adapter (Metal on
//! Apple, Vulkan on Linux, DX12 on Windows) and yields the hybrid
//! `WebGpuSsmBackend`, which runs long scans on the GPU and short ones on the
//! CPU. If no adapter is reachable it logs the reason and falls back to
//! [`CpuSsmBackend`] — the returned value always works.
//!
//! # Scope
//!
//! `SsmBackend` is the scalar diagonal-A scan interface: `(a_t, bu_t)` pairs
//! in, prefix states out. `kizzasi_core::parallel_ssm_scan` is a *different*,
//! multi-dimensional interface (`SSMElement` with a `state_dim`-wide `A`/`B`)
//! and is not routed through this trait, so enabling `webgpu` does not
//! silently change what the core scan functions do. Callers that want the GPU
//! path take a backend from here and call [`SsmBackend::ssm_scan`] on it.

pub use kizzasi_core::ssm_backend::{CpuSsmBackend, SsmBackend};

/// Whether this build contains the WebGPU backend at all.
///
/// `true` only when compiled with `--features webgpu`. This reports what was
/// *compiled in*, not whether a GPU adapter exists — for that, call
/// [`select_ssm_backend`] and read [`SsmBackend::backend_name`].
pub const fn webgpu_compiled_in() -> bool {
    cfg!(feature = "webgpu")
}

/// A CPU scan backend, without touching any GPU machinery.
///
/// Useful when a caller wants a deterministic reference implementation, or
/// cannot await [`select_ssm_backend`].
pub fn cpu_ssm_backend() -> Box<dyn SsmBackend> {
    Box::new(CpuSsmBackend)
}

/// Select the best SSM scan backend this build can actually reach.
///
/// Async because wgpu adapter/device acquisition is async; with the `webgpu`
/// feature off, the returned future is ready immediately.
///
/// Never fails: an unreachable or failing GPU degrades to [`CpuSsmBackend`]
/// with a `warn!` explaining why, so callers get a working backend either way.
pub async fn select_ssm_backend() -> Box<dyn SsmBackend> {
    #[cfg(feature = "webgpu")]
    {
        match kizzasi_webgpu::WebGpuBackend::new().await {
            Ok(backend) => {
                let hybrid = kizzasi_webgpu::WebGpuSsmBackend::new(backend);
                tracing::info!(
                    adapter = ?hybrid.backend().adapter_info(),
                    gpu_threshold = hybrid.gpu_threshold(),
                    "using WebGPU SSM scan backend",
                );
                Box::new(hybrid)
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "WebGPU backend unavailable; using the CPU SSM scan backend",
                );
                Box::new(CpuSsmBackend)
            }
        }
    }

    #[cfg(not(feature = "webgpu"))]
    {
        Box::new(CpuSsmBackend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference scan, written independently of `CpuSsmBackend` so the test
    /// checks the operator rather than restating the implementation.
    fn reference_scan(elements: &[(f32, f32)]) -> Vec<(f32, f32)> {
        let mut a_acc = 1.0f32;
        let mut bu_acc = 0.0f32;
        elements
            .iter()
            .map(|&(a, bu)| {
                a_acc *= a;
                bu_acc = a * bu_acc + bu;
                (a_acc, bu_acc)
            })
            .collect()
    }

    #[test]
    fn cpu_backend_matches_reference() {
        let elements: Vec<(f32, f32)> = (0..64)
            .map(|i| (0.9 + (i % 5) as f32 * 0.01, (i % 7) as f32))
            .collect();

        let backend = cpu_ssm_backend();
        let got = backend.ssm_scan(&elements).expect("cpu scan failed");
        let want = reference_scan(&elements);

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g.0 - w.0).abs() < 1e-4 && (g.1 - w.1).abs() < 1e-4,
                "element {i}: got {g:?}, want {w:?}"
            );
        }
    }

    #[test]
    fn webgpu_compiled_in_tracks_the_feature() {
        assert_eq!(webgpu_compiled_in(), cfg!(feature = "webgpu"));
    }

    /// The selector must return a usable backend in every build, including one
    /// with no GPU present — that is the whole point of the fallback.
    #[test]
    fn selected_backend_scans_correctly() {
        let elements: Vec<(f32, f32)> = (0..1024)
            .map(|i| (0.99, ((i % 11) as f32) * 0.5 - 2.0))
            .collect();

        // A tiny executor: `select_ssm_backend` is the only await, and the
        // CPU path is ready immediately, so this avoids pulling a runtime
        // into builds that do not enable the `async` feature.
        let backend = futures_lite_block_on(select_ssm_backend());
        let got = backend.ssm_scan(&elements).expect("scan failed");
        let want = reference_scan(&elements);

        assert!(!backend.backend_name().is_empty());
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g.0 - w.0).abs() < 1e-3 && (g.1 - w.1).abs() < 1e-3,
                "element {i}: got {g:?}, want {w:?}"
            );
        }
    }

    /// Minimal block-on for a future that only ever yields on GPU
    /// initialisation; parks the thread instead of spinning, and needs no
    /// async runtime dependency (and no `unsafe`: `Box::pin` does the
    /// pinning).
    fn futures_lite_block_on<F: std::future::Future>(fut: F) -> F::Output {
        use std::sync::Arc;
        use std::task::{Context, Poll, Wake, Waker};

        struct ThreadWaker(std::thread::Thread);
        impl Wake for ThreadWaker {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
            fn wake_by_ref(self: &Arc<Self>) {
                self.0.unpark();
            }
        }

        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut fut = Box::pin(fut);
        loop {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::park(),
            }
        }
    }
}
