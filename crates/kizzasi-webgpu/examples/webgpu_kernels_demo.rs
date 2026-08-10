//! Demonstrates GPU-accelerated matvec, SiLU, and RMS Norm kernels.
//!
//! Run with:
//! ```text
//! cargo run --example webgpu_kernels_demo --features webgpu -p kizzasi-webgpu
//! ```

use kizzasi_webgpu::{WebGpuBackend, WebGpuError};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = WebGpuBackend::new()
        .await
        .map_err(|e| -> Box<dyn std::error::Error> {
            match e {
                WebGpuError::AdapterRequest(_) => {
                    "No GPU adapter found. Please run on a machine with a GPU.".into()
                }
                other => other.to_string().into(),
            }
        })?;

    let info = backend.adapter_info();
    println!("GPU: {} ({})", info.name, info.backend);
    println!("Driver: {}", info.driver);
    println!();

    // ── Matvec demo ───────────────────────────────────────────────────────────
    println!("=== Matrix-Vector Multiplication ===");
    println!("4×4 identity matrix × [1, 2, 3, 4] → should be [1, 2, 3, 4]");

    #[rustfmt::skip]
    let identity_4x4 = [
        1.0_f32, 0.0, 0.0, 0.0,
        0.0,     1.0, 0.0, 0.0,
        0.0,     0.0, 1.0, 0.0,
        0.0,     0.0, 0.0, 1.0,
    ];
    let vec4 = [1.0_f32, 2.0, 3.0, 4.0];

    let matvec_result = kizzasi_webgpu::matvec_gpu(&backend, &identity_4x4, 4, 4, &vec4)?;
    println!("Result: {:?}", matvec_result);

    let matvec_ok = matvec_result
        .iter()
        .zip(vec4.iter())
        .all(|(&got, &exp)| (got - exp).abs() < 1e-4);
    println!(
        "Correct: {}",
        if matvec_ok {
            "YES"
        } else {
            "NO (values differ)"
        }
    );
    println!();

    // ── Non-trivial matvec: 2×3 matrix ────────────────────────────────────────
    println!("2×3 matrix [[1,2,3],[4,5,6]] × [1,2,3] → should be [14, 32]");
    let mat23 = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vec3 = [1.0_f32, 2.0, 3.0];
    let matvec23 = kizzasi_webgpu::matvec_gpu(&backend, &mat23, 2, 3, &vec3)?;
    println!("Result: {:?}", matvec23);
    println!();

    // ── SiLU demo ─────────────────────────────────────────────────────────────
    println!("=== SiLU Activation: x / (1 + exp(-x)) ===");
    let silu_input = [-3.0_f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let silu_cpu: Vec<f32> = silu_input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    let silu_gpu = kizzasi_webgpu::silu_gpu(&backend, &silu_input)?;

    println!("Input:   {:?}", silu_input);
    println!("CPU ref: {:?}", fmt_vec(&silu_cpu));
    println!("GPU:     {:?}", fmt_vec(&silu_gpu));

    let silu_max_err = silu_cpu
        .iter()
        .zip(silu_gpu.iter())
        .map(|(&c, &g)| (c - g).abs())
        .fold(0.0_f32, f32::max);
    println!("Max error: {:.2e}", silu_max_err);
    println!(
        "Correct: {}",
        if silu_max_err < 1e-5 { "YES" } else { "NO" }
    );
    println!();

    // ── RMS Norm demo ─────────────────────────────────────────────────────────
    println!("=== RMS Norm: output[i] = (input[i] / rms(input)) * weight[i] ===");
    println!("Uniform input [1,1,1,1] with uniform weights [1,1,1,1] → should be [1,1,1,1]");

    let rms_input = vec![1.0_f32; 4];
    let rms_weight = vec![1.0_f32; 4];
    let eps = 1e-6_f32;

    let rms_gpu = kizzasi_webgpu::rms_norm_gpu(&backend, &rms_input, &rms_weight, eps)?;
    println!("Result: {:?}", rms_gpu);

    let rms_ok = rms_gpu.iter().all(|&v| (v - 1.0).abs() < 1e-4);
    println!("Correct: {}", if rms_ok { "YES" } else { "NO" });
    println!();

    // ── Larger RMS Norm with non-trivial values ────────────────────────────────
    println!("Non-trivial RMS Norm (8 elements):");
    let rms_input2: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let rms_weight2 = vec![1.0_f32; 8];

    // CPU reference.
    let sum_sq: f32 = rms_input2.iter().map(|&x| x * x).sum();
    let rms_val = ((sum_sq / 8.0) + eps).sqrt();
    let rms_cpu: Vec<f32> = rms_input2.iter().map(|&x| x / rms_val).collect();

    let rms_gpu2 = kizzasi_webgpu::rms_norm_gpu(&backend, &rms_input2, &rms_weight2, eps)?;

    println!("Input:   {:?}", rms_input2);
    println!("CPU ref: {:?}", fmt_vec(&rms_cpu));
    println!("GPU:     {:?}", fmt_vec(&rms_gpu2));

    let rms_max_err = rms_cpu
        .iter()
        .zip(rms_gpu2.iter())
        .map(|(&c, &g)| (c - g).abs())
        .fold(0.0_f32, f32::max);
    println!("Max error: {:.2e}", rms_max_err);
    println!("Correct: {}", if rms_max_err < 1e-4 { "YES" } else { "NO" });

    println!();
    println!("All kernel demos completed.");
    Ok(())
}

/// Format a float vector with 4 decimal places.
fn fmt_vec(v: &[f32]) -> Vec<String> {
    v.iter().map(|&x| format!("{:.4}", x)).collect()
}
