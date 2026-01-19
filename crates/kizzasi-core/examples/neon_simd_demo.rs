//! ARM NEON SIMD Optimizations Demo
//!
//! Demonstrates SIMD optimizations for ARM/AArch64 platforms including
//! Apple Silicon (M1/M2/M3), Raspberry Pi, and ARM mobile devices.
//!
//! Features 128-bit SIMD with 4x f32 parallelism and automatic fallback
//! to scalar operations on non-ARM platforms.
//!
//! Run with:
//! ```bash
//! cargo run --example neon_simd_demo --release --no-default-features --features std
//! ```

use kizzasi_core::simd_neon::*;
use std::time::Instant;

fn main() {
    println!("=== ARM NEON SIMD Optimizations Demo ===\n");

    check_neon_availability();
    demo_dot_product();
    demo_vector_operations();
    demo_matrix_vector();
    demo_activations();
    demo_layer_norm();
    performance_comparison();

    println!("\n✅ All NEON SIMD demos completed successfully!");
}

fn check_neon_availability() {
    println!("1. Platform Detection\n");

    if is_neon_available() {
        println!("   ✅ ARM NEON SIMD is available on this platform!");
        println!("   Platform: ARM/AArch64 (Apple Silicon, Raspberry Pi, etc.)");
        println!("   Features: 128-bit SIMD, 4x f32 parallel, FMA support\n");
    } else {
        println!("   ⚠️  NEON not available - using scalar fallback");
        println!("   Platform: x86/x86_64 or non-NEON ARM");
        println!("   Operations will run correctly but without SIMD acceleration\n");
    }
}

fn demo_dot_product() {
    println!("2. Dot Product (4x parallelism with FMA)");
    println!("   Computes sum of element-wise products\n");

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    let result = neon_dot_product(&a, &b);
    let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

    println!("   Vector a: {:?}", &a[..4]);
    println!("   Vector b: {:?}", &b[..4]);
    println!("   Dot product: {:.2}", result);
    println!("   Expected: {:.2}", expected);
    println!("   Error: {:.6}\n", (result - expected).abs());
}

fn demo_vector_operations() {
    println!("3. Vector Operations (add, mul, fma)");
    println!("   Element-wise operations with NEON acceleration\n");

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];

    // Vector addition: c = a + b
    neon_vec_add(&a, &b, &mut c).unwrap();
    println!("   Addition (c = a + b):");
    println!("     a: {:?}", a);
    println!("     b: {:?}", b);
    println!("     c: {:?}", c);

    // Vector multiplication: c = a * b
    neon_vec_mul(&a, &b, &mut c).unwrap();
    println!("\n   Multiplication (c = a * b):");
    println!("     c: {:?}", c);

    // Fused multiply-add: c = c + a * b
    let mut c2 = vec![1.0; 4];
    neon_vec_fma(&a, &b, &mut c2).unwrap();
    println!("\n   Fused Multiply-Add (c = c + a * b, c initially [1,1,1,1]):");
    println!("     c: {:?}\n", c2);
}

fn demo_matrix_vector() {
    println!("4. Matrix-Vector Multiplication");
    println!("   y = A @ x using NEON dot products\n");

    let matrix = vec![
        1.0, 2.0, 3.0, // row 1
        4.0, 5.0, 6.0, // row 2
        7.0, 8.0, 9.0, // row 3
    ];
    let x = vec![1.0, 2.0, 3.0];
    let mut y = vec![0.0; 3];

    neon_matvec(&matrix, &x, &mut y, 3, 3).unwrap();

    println!("   Matrix A (3x3):");
    println!(
        "     [{:.1}, {:.1}, {:.1}]",
        matrix[0], matrix[1], matrix[2]
    );
    println!(
        "     [{:.1}, {:.1}, {:.1}]",
        matrix[3], matrix[4], matrix[5]
    );
    println!(
        "     [{:.1}, {:.1}, {:.1}]",
        matrix[6], matrix[7], matrix[8]
    );
    println!("   Vector x: {:?}", x);
    println!("   Result y: {:?}", y);
    println!("   Expected: [14.0, 32.0, 50.0]\n");
}

fn demo_activations() {
    println!("5. Activation Functions");
    println!("   ReLU with NEON SIMD\n");

    let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 1.5];
    let mut y = vec![0.0; 8];

    neon_relu(&x, &mut y).unwrap();

    println!("   ReLU activation:");
    println!("     Input:  {:?}", &x[..5]);
    println!("     Output: {:?}\n", &y[..5]);
}

fn demo_layer_norm() {
    println!("6. Layer Normalization");
    println!("   Normalizes to zero mean and unit variance\n");

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut y = vec![0.0; 8];
    let eps = 1e-5;

    neon_layer_norm(&x, &mut y, eps).unwrap();

    let mean: f32 = y.iter().sum::<f32>() / y.len() as f32;
    let variance: f32 = y.iter().map(|&yi| yi.powi(2)).sum::<f32>() / y.len() as f32;

    println!("   Input:  {:?}", &x[..5]);
    println!(
        "   Output: [{:.3}, {:.3}, {:.3}, {:.3}, ...]",
        y[0], y[1], y[2], y[3]
    );
    println!("   Mean: {:.6} (target: 0.0)", mean);
    println!("   Variance: {:.6} (target: 1.0)\n", variance);
}

fn performance_comparison() {
    println!("7. Performance Comparison");
    println!("   NEON vs Standard implementation\n");

    let sizes = [256, 512, 1024, 2048];

    for &size in &sizes {
        let a = vec![0.5f32; size];
        let b = vec![0.3f32; size];
        let iterations = 10000;

        // NEON version
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = neon_dot_product(&a, &b);
        }
        let duration_neon = start.elapsed();

        // Standard version
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = kizzasi_core::simd::dot_product(&a, &b);
        }
        let duration_std = start.elapsed();

        let speedup = duration_std.as_nanos() as f64 / duration_neon.as_nanos() as f64;

        println!("   Size: {} elements ({} iterations)", size, iterations);
        println!("     NEON:     {:?}", duration_neon);
        println!("     Standard: {:?}", duration_std);
        println!("     Speedup:  {:.2}x", speedup);
        println!();
    }

    if is_neon_available() {
        println!("   💡 On ARM platforms with NEON, expect 2-4x speedup!");
    } else {
        println!("   💡 On this platform (non-ARM), NEON falls back to scalar operations.");
    }
}
