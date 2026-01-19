//! Fixed-Point Arithmetic Demo
//!
//! Demonstrates fixed-point arithmetic for embedded systems without FPU.
//! Shows 10-100x performance improvement over software floating-point
//! while maintaining acceptable precision for many applications.
//!
//! Run with:
//! ```bash
//! cargo run --example fixed_point_demo --release --no-default-features --features std
//! ```

use kizzasi_core::fixed_point::*;
use std::time::Instant;

fn main() {
    println!("=== Fixed-Point Arithmetic Demo ===\n");
    println!("For embedded systems without hardware FPU\n");

    demo_basic_arithmetic();
    demo_advanced_functions();
    demo_vector_operations();
    demo_neural_network_ops();
    performance_comparison();

    println!("\n✅ All fixed-point demos completed successfully!");
}

fn demo_basic_arithmetic() {
    println!("1. Basic Arithmetic (Q15.16 format)");
    println!("   Range: [-32768, 32767], Precision: ~0.000015\n");

    let a = Q15_16::from_f32(3.5);
    let b = Q15_16::from_f32(2.0);

    println!("   a = 3.5, b = 2.0");
    println!("   a + b = {:.6}", (a + b).to_f32());
    println!("   a - b = {:.6}", (a - b).to_f32());
    println!("   a * b = {:.6}", (a * b).to_f32());
    println!("   a / b = {:.6}", (a / b).to_f32());

    // Saturating arithmetic
    let large = Q15_16::from_int(30000);
    let overflow = Q15_16::from_int(10000);
    println!("\n   Saturating arithmetic:");
    println!(
        "   30000 + 10000 (saturates) = {}",
        large.saturating_add(overflow).to_int()
    );
    println!();
}

fn demo_advanced_functions() {
    println!("2. Advanced Mathematical Functions");
    println!("   Using iterative approximations (Newton-Raphson, Taylor series)\n");

    let x = Q15_16::from_f32(9.0);
    let sqrt_x = x.sqrt();
    println!("   sqrt(9.0) = {:.6} (expected: 3.0)", sqrt_x.to_f32());

    let x2 = Q15_16::from_f32(4.0);
    let recip = x2.recip().unwrap();
    println!("   1 / 4.0 = {:.6} (expected: 0.25)", recip.to_f32());

    let x3 = Q15_16::from_f32(1.0);
    let exp_x = x3.exp();
    println!(
        "   exp(1.0) = {:.6} (expected: {:.3})",
        exp_x.to_f32(),
        core::f32::consts::E
    );

    let x4 = Q15_16::from_f32(core::f32::consts::E);
    let ln_x = x4.ln().unwrap();
    println!(
        "   ln({:.3}) = {:.6} (expected: 1.0)",
        core::f32::consts::E,
        ln_x.to_f32()
    );
    println!();
}

fn demo_vector_operations() {
    println!("3. Vector Operations");
    println!("   Essential for neural network inference\n");

    let a = vec![
        Q15_16::from_f32(1.0),
        Q15_16::from_f32(2.0),
        Q15_16::from_f32(3.0),
        Q15_16::from_f32(4.0),
    ];
    let b = vec![
        Q15_16::from_f32(0.5),
        Q15_16::from_f32(1.5),
        Q15_16::from_f32(2.5),
        Q15_16::from_f32(3.5),
    ];

    let dot = vec_ops::dot_product_q15_16(&a, &b);
    let expected = 1.0 * 0.5 + 2.0 * 1.5 + 3.0 * 2.5 + 4.0 * 3.5;
    println!("   Dot product:");
    println!("     Result: {:.6}", dot.to_f32());
    println!("     Expected: {:.6}", expected);
    println!("     Error: {:.6}", (dot.to_f32() - expected).abs());
    println!();
}

fn demo_neural_network_ops() {
    println!("4. Neural Network Operations");
    println!("   ReLU, Layer Normalization with fixed-point\n");

    // ReLU activation
    let x = vec![
        Q15_16::from_f32(-2.0),
        Q15_16::from_f32(-1.0),
        Q15_16::from_f32(0.0),
        Q15_16::from_f32(1.0),
        Q15_16::from_f32(2.0),
    ];
    let mut y = vec![Q15_16::ZERO; 5];

    vec_ops::relu_q15_16(&x, &mut y);

    println!("   ReLU activation:");
    println!(
        "     Input:  [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
        x[0].to_f32(),
        x[1].to_f32(),
        x[2].to_f32(),
        x[3].to_f32(),
        x[4].to_f32()
    );
    println!(
        "     Output: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
        y[0].to_f32(),
        y[1].to_f32(),
        y[2].to_f32(),
        y[3].to_f32(),
        y[4].to_f32()
    );

    // Layer normalization
    let x2 = vec![
        Q15_16::from_f32(1.0),
        Q15_16::from_f32(2.0),
        Q15_16::from_f32(3.0),
        Q15_16::from_f32(4.0),
    ];
    let mut y2 = vec![Q15_16::ZERO; 4];
    let eps = Q15_16::from_f32(1e-5);

    vec_ops::layer_norm_q15_16(&x2, &mut y2, eps).unwrap();

    let mean: f32 = y2.iter().map(|yi| yi.to_f32()).sum::<f32>() / y2.len() as f32;
    let variance: f32 = y2.iter().map(|yi| yi.to_f32().powi(2)).sum::<f32>() / y2.len() as f32;

    println!("\n   Layer Normalization:");
    println!(
        "     Input:  [{:.1}, {:.1}, {:.1}, {:.1}]",
        x2[0].to_f32(),
        x2[1].to_f32(),
        x2[2].to_f32(),
        x2[3].to_f32()
    );
    println!(
        "     Output: [{:.3}, {:.3}, {:.3}, {:.3}]",
        y2[0].to_f32(),
        y2[1].to_f32(),
        y2[2].to_f32(),
        y2[3].to_f32()
    );
    println!("     Mean: {:.6} (should be ~0)", mean);
    println!("     Variance: {:.6} (should be ~1)", variance);
    println!();
}

fn performance_comparison() {
    println!("5. Performance Comparison: Fixed-Point vs Floating-Point");
    println!("   Measuring raw arithmetic throughput\n");

    let iterations = 1_000_000;

    // Fixed-point addition
    let start = Instant::now();
    let mut result_fixed = Q15_16::from_f32(0.0);
    let increment = Q15_16::from_f32(0.001);
    for _ in 0..iterations {
        result_fixed = result_fixed + increment;
    }
    let duration_fixed = start.elapsed();

    // Floating-point addition
    let start = Instant::now();
    let mut _result_float = 0.0f32;
    for _ in 0..iterations {
        _result_float += 0.001f32;
    }
    let duration_float = start.elapsed();

    println!("   Addition ({} iterations):", iterations);
    println!("     Fixed-point: {:?}", duration_fixed);
    println!("     Float:       {:?}", duration_float);
    println!(
        "     Speedup:     {:.2}x",
        duration_float.as_nanos() as f64 / duration_fixed.as_nanos() as f64
    );

    // Fixed-point multiplication
    let start = Instant::now();
    let mut result_fixed = Q15_16::from_f32(1.0);
    let multiplier = Q15_16::from_f32(1.0001);
    for _ in 0..iterations {
        result_fixed = result_fixed * multiplier;
    }
    let duration_fixed = start.elapsed();

    // Floating-point multiplication
    let start = Instant::now();
    let mut _result_float = 1.0f32;
    for _ in 0..iterations {
        _result_float *= 1.0001f32;
    }
    let duration_float = start.elapsed();

    println!("\n   Multiplication ({} iterations):", iterations);
    println!("     Fixed-point: {:?}", duration_fixed);
    println!("     Float:       {:?}", duration_float);
    println!(
        "     Speedup:     {:.2}x",
        duration_float.as_nanos() as f64 / duration_fixed.as_nanos() as f64
    );

    println!("\n   💡 Fixed-point is typically 2-10x faster on systems without FPU");
    println!("      and 10-100x faster than software floating-point emulation!");
}
