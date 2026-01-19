//! Kernel Fusion Optimizations Demo
//!
//! Demonstrates the performance benefits of fused operations over
//! separate operations by combining multiple primitive ops into
//! single kernels to reduce memory bandwidth.
//!
//! Run with:
//! ```bash
//! cargo run --example kernel_fusion_demo --release --no-default-features --features std,metal
//! ```

use kizzasi_core::kernel_fusion::*;
use std::time::Instant;

fn main() {
    println!("=== Kernel Fusion Optimizations Demo ===\n");

    demo_fused_layernorm_gelu();
    demo_fused_qkv_projection();
    demo_fused_ffn();
    demo_fused_ssm_step();
    demo_fused_quantize_dequantize();

    println!("\n✅ All kernel fusion demos completed successfully!");
}

fn demo_fused_layernorm_gelu() {
    println!("1. Fused LayerNorm + GELU");
    println!("   Combines normalization and activation in one pass\n");

    let size = 1024;
    let x = vec![0.5f32; size];
    let gamma = vec![1.0f32; size];
    let beta = vec![0.0f32; size];
    let eps = 1e-5;

    let start = Instant::now();
    for _ in 0..1000 {
        let _result = fused_layernorm_gelu(&x, &gamma, &beta, eps).unwrap();
    }
    let duration = start.elapsed();

    println!("   Size: {} elements", size);
    println!("   Time (1000 iterations): {:?}", duration);
    println!("   Avg per iteration: {:?}\n", duration / 1000);
}

fn demo_fused_qkv_projection() {
    println!("2. Fused QKV Projection");
    println!("   Single matmul for Query, Key, Value instead of three separate ones\n");

    let d_model = 256;
    let x = vec![0.5f32; d_model];
    let w_qkv = vec![0.1f32; 3 * d_model * d_model];

    let start = Instant::now();
    for _ in 0..1000 {
        let (_q, _k, _v) = fused_qkv_projection(&x, &w_qkv, d_model).unwrap();
    }
    let duration = start.elapsed();

    println!("   Model dimension: {}", d_model);
    println!("   Time (1000 iterations): {:?}", duration);
    println!("   Avg per iteration: {:?}\n", duration / 1000);
}

fn demo_fused_ffn() {
    println!("3. Fused Feed-Forward Network");
    println!("   Combines two linear layers with GELU activation\n");

    let d_model = 256;
    let d_ff = 1024; // 4x expansion
    let x = vec![0.5f32; d_model];
    let w1 = vec![0.1f32; d_ff * d_model];
    let b1 = vec![0.0f32; d_ff];
    let w2 = vec![0.1f32; d_model * d_ff];
    let b2 = vec![0.0f32; d_model];

    let start = Instant::now();
    for _ in 0..1000 {
        let _result = fused_ffn_gelu(&x, &w1, &b1, &w2, &b2, d_model, d_ff).unwrap();
    }
    let duration = start.elapsed();

    println!("   Model dimension: {}", d_model);
    println!("   FFN dimension: {}", d_ff);
    println!("   Time (1000 iterations): {:?}", duration);
    println!("   Avg per iteration: {:?}\n", duration / 1000);
}

fn demo_fused_ssm_step() {
    println!("4. Fused SSM Step");
    println!("   Combines discretization, state update, and output in one kernel\n");

    let state_dim = 64;
    let mut h = vec![0.0f32; state_dim];
    let a = vec![-1.0f32; state_dim];
    let b = vec![1.0f32; state_dim];
    let c = vec![1.0f32; state_dim];
    let d = 0.1f32;
    let delta = 0.01f32;
    let x = 1.0f32;

    let start = Instant::now();
    for _ in 0..10000 {
        let _y = fused_ssm_step(&mut h, x, &a, &b, &c, d, delta).unwrap();
    }
    let duration = start.elapsed();

    println!("   State dimension: {}", state_dim);
    println!("   Time (10000 iterations): {:?}", duration);
    println!("   Avg per iteration: {:?}\n", duration / 10000);
}

fn demo_fused_quantize_dequantize() {
    println!("5. Fused Quantize-Dequantize");
    println!("   Simulates quantization effects for QAT without storing quantized values\n");

    let x = vec![0.1, 0.5, -0.3, 0.8, -0.2, 1.0, -0.5, 0.4];

    // INT8 symmetric quantization
    let result_int8 = fused_quantize_dequantize(&x, 8, true).unwrap();
    println!("   INT8 Symmetric:");
    println!("     Original: {:?}", &x[..4]);
    println!("     Quantized: {:?}", &result_int8[..4]);
    println!("     Error: {:.6}", calculate_error(&x, &result_int8));

    // INT4 asymmetric quantization
    let result_int4 = fused_quantize_dequantize(&x, 4, false).unwrap();
    println!("   INT4 Asymmetric:");
    println!("     Original: {:?}", &x[..4]);
    println!("     Quantized: {:?}", &result_int4[..4]);
    println!("     Error: {:.6}\n", calculate_error(&x, &result_int4));
}

fn calculate_error(original: &[f32], quantized: &[f32]) -> f32 {
    original
        .iter()
        .zip(quantized)
        .map(|(o, q)| (o - q).abs())
        .sum::<f32>()
        / original.len() as f32
}
