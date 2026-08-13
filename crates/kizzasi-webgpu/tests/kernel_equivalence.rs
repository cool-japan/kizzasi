//! End-to-end equivalence tests: every WGSL kernel against a scalar CPU
//! reference, over randomised inputs and around every internal size boundary.
//!
//! These tests exercise the crate through its public API only.  They are
//! compiled solely with `--features webgpu` and skip gracefully when the host
//! has no GPU adapter — set `KIZZASI_REQUIRE_GPU=1` to turn a missing adapter
//! into a failure instead.

#![cfg(feature = "webgpu")]

use kizzasi_core::ssm_backend::{CpuSsmBackend, SsmBackend};
use kizzasi_webgpu::{
    matvec_gpu, matvec_gpu_buf, rms_norm_gpu, rms_norm_gpu_buf, silu_gpu, silu_gpu_buf,
    ssm_scan_gpu, ssm_scan_gpu_buf, WebGpuBackend, WebGpuError, WebGpuSsmBackend, MAX_RMS_NORM_LEN,
    MAX_SINGLE_PASS_LEN,
};

/// Deterministic linear congruential generator — no external dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Uniform value in `[0, 1)`.
    fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32)
    }

    /// Uniform value in `[low, high)`.
    fn next_range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_unit()
    }
}

fn require_gpu() -> bool {
    std::env::var("KIZZASI_REQUIRE_GPU")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

async fn try_backend() -> Option<WebGpuBackend> {
    match WebGpuBackend::new().await {
        Ok(backend) => Some(backend),
        Err(WebGpuError::AdapterRequest(message)) => {
            assert!(
                !require_gpu(),
                "KIZZASI_REQUIRE_GPU is set but no GPU adapter is available: {message}"
            );
            eprintln!("no GPU adapter found — skipping GPU equivalence test");
            None
        }
        Err(err) => panic!("unexpected error creating WebGpuBackend: {err}"),
    }
}

fn assert_close(expected: f32, got: f32, tol: f32, context: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (expected - got).abs() <= tol * scale,
        "{context}: expected {expected}, got {got} (tolerance {tol})"
    );
}

// ── SiLU ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn silu_matches_cpu_reference() {
    let Some(backend) = try_backend().await else {
        return;
    };

    let mut rng = Lcg::new(0x5eed_1234);
    for n in [1_usize, 7, 64, 255, 256, 257, 1000, 4096] {
        let input: Vec<f32> = (0..n).map(|_| rng.next_range(-8.0, 8.0)).collect();
        let expected: Vec<f32> = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        let got = silu_gpu(&backend, &input).expect("silu_gpu failed");

        assert_eq!(got.len(), n, "silu length mismatch at n={n}");
        for (i, (&want, &have)) in expected.iter().zip(got.iter()).enumerate() {
            assert_close(want, have, 1e-5, &format!("silu n={n} index {i}"));
        }
    }
}

// ── RMS Norm ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn rms_norm_matches_cpu_reference() {
    let Some(backend) = try_backend().await else {
        return;
    };

    let mut rng = Lcg::new(0xabcd_0001);
    let eps = 1e-5_f32;

    for n in [1_usize, 15, 16, 128, MAX_RMS_NORM_LEN] {
        let input: Vec<f32> = (0..n).map(|_| rng.next_range(-3.0, 3.0)).collect();
        let weight: Vec<f32> = (0..n).map(|_| rng.next_range(0.25, 2.0)).collect();

        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let rms = ((sum_sq / n as f32) + eps).sqrt();
        let expected: Vec<f32> = input
            .iter()
            .zip(weight.iter())
            .map(|(&x, &w)| (x / rms) * w)
            .collect();

        let got = rms_norm_gpu(&backend, &input, &weight, eps).expect("rms_norm_gpu failed");
        assert_eq!(got.len(), n, "rms_norm length mismatch at n={n}");
        for (i, (&want, &have)) in expected.iter().zip(got.iter()).enumerate() {
            assert_close(want, have, 1e-4, &format!("rms_norm n={n} index {i}"));
        }
    }
}

#[tokio::test]
async fn rms_norm_rejects_oversized_input() {
    let Some(backend) = try_backend().await else {
        return;
    };

    let n = MAX_RMS_NORM_LEN + 1;
    let err = rms_norm_gpu(&backend, &vec![1.0; n], &vec![1.0; n], 1e-6)
        .expect_err("input beyond the single-work-group kernel must be rejected");
    assert!(matches!(err, WebGpuError::Other(_)), "got: {err:?}");
}

// ── Matvec ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn matvec_matches_cpu_reference() {
    let Some(backend) = try_backend().await else {
        return;
    };

    let mut rng = Lcg::new(0x1357_9bdf);
    for (rows, cols) in [(1_usize, 1_usize), (1, 64), (64, 1), (63, 65), (128, 33)] {
        let matrix: Vec<f32> = (0..rows * cols)
            .map(|_| rng.next_range(-2.0, 2.0))
            .collect();
        let vector: Vec<f32> = (0..cols).map(|_| rng.next_range(-2.0, 2.0)).collect();

        let expected: Vec<f32> = (0..rows)
            .map(|row| {
                (0..cols)
                    .map(|col| matrix[row * cols + col] * vector[col])
                    .sum()
            })
            .collect();

        let got = matvec_gpu(&backend, &matrix, rows, cols, &vector).expect("matvec_gpu failed");
        assert_eq!(got.len(), rows, "matvec length mismatch at {rows}×{cols}");
        for (i, (&want, &have)) in expected.iter().zip(got.iter()).enumerate() {
            assert_close(want, have, 1e-4, &format!("matvec {rows}×{cols} row {i}"));
        }
    }
}

#[tokio::test]
async fn matvec_handles_degenerate_shapes() {
    let Some(backend) = try_backend().await else {
        return;
    };

    assert!(matvec_gpu(&backend, &[], 0, 0, &[])
        .expect("0×0 must not fail")
        .is_empty());
    assert_eq!(
        matvec_gpu(&backend, &[], 3, 0, &[]).expect("3×0 must not fail"),
        vec![0.0, 0.0, 0.0]
    );
}

// ── SSM scan ─────────────────────────────────────────────────────────────────

fn random_elements(rng: &mut Lcg, n: usize) -> Vec<(f32, f32)> {
    (0..n)
        .map(|_| (rng.next_range(0.4, 0.95), rng.next_range(-1.0, 1.0)))
        .collect()
}

#[tokio::test]
async fn ssm_scan_matches_cpu_reference_across_block_boundaries() {
    let Some(backend) = try_backend().await else {
        return;
    };

    let mut rng = Lcg::new(0x2468_ace0);
    let block = MAX_SINGLE_PASS_LEN;

    for n in [
        2_usize,
        block - 1,
        block,
        block + 1,
        2 * block,
        2 * block + 1,
        1000,
        block * block + 3,
    ] {
        let elements = random_elements(&mut rng, n);
        let expected = CpuSsmBackend
            .ssm_scan(&elements)
            .expect("CPU reference scan failed");
        let got = ssm_scan_gpu(&backend, &elements).expect("ssm_scan_gpu failed");

        assert_eq!(got.len(), n, "scan length mismatch at n={n}");
        for (i, (&(want_a, want_bu), &(have_a, have_bu))) in
            expected.iter().zip(got.iter()).enumerate()
        {
            assert_close(want_a, have_a, 1e-3, &format!("scan n={n} index {i} (a)"));
            assert_close(
                want_bu,
                have_bu,
                1e-3,
                &format!("scan n={n} index {i} (bu)"),
            );
        }
    }
}

#[tokio::test]
async fn ssm_hybrid_backend_agrees_with_cpu_on_both_sides_of_threshold() {
    let Some(backend) = try_backend().await else {
        return;
    };

    let mut rng = Lcg::new(0x0f0f_0f0f);
    let hybrid = WebGpuSsmBackend::new(backend).with_gpu_threshold(256);

    for n in [8_usize, 255, 256, 4096] {
        let elements = random_elements(&mut rng, n);
        let expected = CpuSsmBackend
            .ssm_scan(&elements)
            .expect("CPU reference scan failed");
        let got = hybrid.ssm_scan(&elements).expect("hybrid scan failed");

        assert_eq!(got.len(), n);
        for (i, (&(want_a, want_bu), &(have_a, have_bu))) in
            expected.iter().zip(got.iter()).enumerate()
        {
            assert_close(want_a, have_a, 1e-3, &format!("hybrid n={n} index {i} (a)"));
            assert_close(
                want_bu,
                have_bu,
                1e-3,
                &format!("hybrid n={n} index {i} (bu)"),
            );
        }

        let expected_path = if n >= 256 {
            "webgpu"
        } else {
            "webgpu-hybrid(cpu-short-input)"
        };
        assert_eq!(hybrid.backend_name(), expected_path, "path report at n={n}");
    }
}

// ── Device residency ─────────────────────────────────────────────────────────

#[tokio::test]
async fn device_resident_chain_matches_host_chain() {
    let Some(backend) = try_backend().await else {
        return;
    };

    let mut rng = Lcg::new(0xfeed_face);
    let rows = 128usize;
    let cols = 64usize;

    let matrix: Vec<f32> = (0..rows * cols)
        .map(|_| rng.next_range(-1.0, 1.0))
        .collect();
    let vector: Vec<f32> = (0..cols).map(|_| rng.next_range(-1.0, 1.0)).collect();
    let weight: Vec<f32> = (0..rows).map(|_| rng.next_range(0.5, 1.5)).collect();
    let eps = 1e-6_f32;

    // Host path: three full round trips.
    let host_matvec = matvec_gpu(&backend, &matrix, rows, cols, &vector).expect("host matvec");
    let host_silu = silu_gpu(&backend, &host_matvec).expect("host silu");

    // Device-resident path: one upload, one download.
    let matrix_buf = backend.upload_f32(&matrix, "chain-matrix").expect("upload");
    let vector_buf = backend.upload_f32(&vector, "chain-vector").expect("upload");
    let matvec_buf =
        matvec_gpu_buf(&backend, &matrix_buf, rows, cols, &vector_buf).expect("resident matvec");
    let silu_buf = silu_gpu_buf(&backend, &matvec_buf).expect("resident silu");
    let device_silu = backend.download_f32(&silu_buf).expect("download");

    assert_eq!(host_silu, device_silu);

    // RMS norm over the first MAX_RMS_NORM_LEN values of the chain.
    let head: Vec<f32> = host_silu.iter().copied().take(MAX_RMS_NORM_LEN).collect();
    let head_weight: Vec<f32> = weight.iter().copied().take(MAX_RMS_NORM_LEN).collect();
    let host_rms = rms_norm_gpu(&backend, &head, &head_weight, eps).expect("host rms");

    let head_buf = backend.upload_f32(&head, "chain-head").expect("upload");
    let head_weight_buf = backend
        .upload_f32(&head_weight, "chain-head-weight")
        .expect("upload");
    let rms_buf =
        rms_norm_gpu_buf(&backend, &head_buf, &head_weight_buf, eps).expect("resident rms");
    let device_rms = backend.download_f32(&rms_buf).expect("download");

    assert_eq!(host_rms, device_rms);
}

#[tokio::test]
async fn device_resident_scan_matches_host_scan() {
    let Some(backend) = try_backend().await else {
        return;
    };

    let mut rng = Lcg::new(0x00c0_ffee);
    let elements = random_elements(&mut rng, 1500);
    let flat: Vec<f32> = elements.iter().flat_map(|&(a, bu)| [a, bu]).collect();

    let host = ssm_scan_gpu(&backend, &elements).expect("host scan");

    let input = backend.upload_f32(&flat, "scan-in").expect("upload");
    let output = ssm_scan_gpu_buf(&backend, &input).expect("resident scan");
    let downloaded = backend.download_f32(&output).expect("download");

    assert_eq!(downloaded.len(), host.len() * 2);
    for (i, (&(want_a, want_bu), chunk)) in host.iter().zip(downloaded.chunks_exact(2)).enumerate()
    {
        assert_close(want_a, chunk[0], 1e-6, &format!("resident scan {i} (a)"));
        assert_close(want_bu, chunk[1], 1e-6, &format!("resident scan {i} (bu)"));
    }
}
