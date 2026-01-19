//! Hot path profiling example using the profiling utilities
//!
//! This example profiles critical SSM operations to identify performance bottlenecks.

use kizzasi_core::profiling::ProfilingSession;
use kizzasi_core::time_block;
use kizzasi_core::{KizzasiConfig, SelectiveSSM, SignalPredictor};
use scirs2_core::ndarray::Array1;

fn main() {
    println!("=== Kizzasi Core Hot Path Profiling ===\n");

    // Create profiling session
    let mut session = ProfilingSession::new("HotPath Analysis");

    // Test configurations
    let configs = vec![
        ("Small (d=64)", 64, 8),
        ("Medium (d=256)", 256, 16),
        ("Large (d=512)", 512, 32),
    ];

    // Create counters for each operation
    let ssm_step_idx = session.add_counter("ssm_step");
    let embedding_idx = session.add_counter("embedding");
    let state_update_idx = session.add_counter("state_update");
    let matrix_ops_idx = session.add_counter("matrix_ops");

    for (name, hidden_dim, state_dim) in configs {
        println!("Configuration: {}", name);
        println!("{}", "=".repeat(50));

        // Create SSM
        let config = KizzasiConfig::new()
            .input_dim(8)
            .output_dim(8)
            .hidden_dim(hidden_dim)
            .state_dim(state_dim)
            .num_layers(4);

        let mut ssm = SelectiveSSM::new(config).expect("SSM creation should succeed");
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

        // Profile SSM step
        {
            let counter = session.counter(ssm_step_idx).expect("counter should exist");
            time_block!(counter, {
                for _ in 0..1000 {
                    let _ = ssm.step(&input).expect("step should succeed");
                }
            });
        }

        // Profile embedding
        {
            let counter = session
                .counter(embedding_idx)
                .expect("counter should exist");
            let embedding = ssm.embedding();
            time_block!(counter, {
                for _ in 0..1000 {
                    let _ = embedding.embed(&input).expect("embed should succeed");
                }
            });
        }

        // Profile state update
        {
            let counter = session
                .counter(state_update_idx)
                .expect("counter should exist");
            time_block!(counter, {
                for _ in 0..1000 {
                    ssm.reset();
                    let _ = ssm.step(&input).expect("step should succeed");
                }
            });
        }

        // Profile matrix operations
        {
            let counter = session
                .counter(matrix_ops_idx)
                .expect("counter should exist");
            let a_matrices = ssm.a_matrices();
            let b_matrices = ssm.b_matrices();
            time_block!(counter, {
                for _ in 0..1000 {
                    for (a, b) in a_matrices.iter().zip(b_matrices.iter()) {
                        let _ = a + b; // Matrix addition
                    }
                }
            });
        }

        println!();
    }

    // Print profiling report
    println!("\n=== Profiling Report ===\n");
    let report = session.report();
    println!("{}", report);

    println!("\n=== Optimization Recommendations ===\n");
    println!("1. SSM Step: Consider caching discretized matrices");
    println!("2. Embedding: Use SIMD for dot products");
    println!("3. State Update: Reduce allocations with in-place operations");
    println!("4. Matrix Ops: Use fused kernels for common operations");
}
