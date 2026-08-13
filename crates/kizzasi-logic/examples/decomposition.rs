//! Constraint decomposition example
//!
//! Demonstrates how to decompose large-scale constraint problems using:
//! - Consensus ADMM for distributed optimization
//! - Block coordinate descent for structured problems
//! - Dual decomposition for separable constraints
//! - Hierarchical decomposition for very large problems

use kizzasi_logic::{
    block_utils, ADMMConfig, Block, BlockCoordinateDescent, ConsensusADMM, DualDecomposition,
    HierarchicalDecomposition,
};
use scirs2_core::ndarray::{Array1, Array2};

fn main() {
    println!("=== Constraint Decomposition Example ===\n");

    // ============================================
    // 1. Consensus ADMM
    // ============================================
    println!("1. Consensus ADMM");
    println!("   Solves: min Σᵢ fᵢ(xᵢ) s.t. xᵢ = z ∀i\n");

    let config = ADMMConfig {
        rho: 1.0,
        max_iterations: 100,
        primal_tol: 1e-3,
        dual_tol: 1e-3,
        adaptive_rho: true,
        rho_update_factor: 2.0,
    };

    let num_blocks = 3;
    let dimension = 5;
    let mut admm = ConsensusADMM::new(num_blocks, dimension, config);

    // Initialize from a starting point
    let x0 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    admm.initialize(&x0).expect("matching dimension");

    println!("   Configuration:");
    println!("   - Number of blocks: {}", num_blocks);
    println!("   - Dimension: {}", dimension);
    println!("   - Initial point: {:?}", x0);

    // Simulate a few iterations with simple quadratic local objectives
    println!("\n   Simulating ADMM iterations:");
    for iter in 0..5 {
        // Local update: minimize ‖xᵢ - target‖² + (ρ/2)‖xᵢ - z + uᵢ‖²
        let (primal_res, dual_res) = admm.iterate(|block_id, _x_old, z_minus_u, rho| {
            // For demonstration: each block has a different target
            let target_val = (block_id + 1) as f32 * 2.0;
            let target = Array1::from_elem(dimension, target_val);

            // Closed-form solution for quadratic: x = (target + ρ*(z-u)) / (1+ρ)
            (&target + &(z_minus_u * rho)) / (1.0 + rho)
        });

        admm.update_rho(primal_res, dual_res);

        if iter % 10 == 0 {
            println!(
                "   Iter {}: primal_res = {:.4}, dual_res = {:.4}",
                iter + 1,
                primal_res,
                dual_res
            );
        }

        if admm.has_converged(primal_res, dual_res) {
            println!("   ✓ Converged at iteration {}!", iter + 1);
            break;
        }
    }

    println!("   Final consensus solution: {:?}\n", admm.solution());

    // ============================================
    // 2. Block Coordinate Descent
    // ============================================
    println!("2. Block Coordinate Descent");
    println!("   Optimizes over blocks of variables sequentially\n");

    // Create blocks for a 10-dimensional problem
    let blocks = vec![
        Block::new("block_1", vec![0, 1, 2]),
        Block::new("block_2", vec![3, 4, 5]),
        Block::new("block_3", vec![6, 7]),
        Block::new("block_4", vec![8, 9]),
    ];

    let mut bcd = BlockCoordinateDescent::new(blocks, 10)
        .unwrap()
        .with_max_iterations(100)
        .with_tolerance(1e-4);

    println!("   Number of blocks: {}", bcd.num_blocks());
    for i in 0..bcd.num_blocks() {
        if let Some(block) = bcd.block(i) {
            println!(
                "   - {}: {} variables {:?}",
                block.name,
                block.size(),
                block.indices
            );
        }
    }

    // Initialize
    let x0_bcd = Array1::from_vec(vec![1.0; 10]);
    bcd.initialize(&x0_bcd).expect("matching dimension");

    println!("\n   Running block updates:");
    for iter in 0..3 {
        let mut total_change = 0.0;
        for block_id in 0..bcd.num_blocks() {
            // Simple block update: project to zeros (for demonstration)
            let change = bcd
                .update_block(block_id, |_full_x, indices| {
                    Array1::from_vec(vec![0.0; indices.len()])
                })
                .expect("block update returns one value per index");
            total_change += change;
        }
        println!("   Iter {}: total change = {:.4}", iter + 1, total_change);
    }

    println!("   Final solution: {:?}\n", bcd.solution());

    // ============================================
    // 3. Block Utilities
    // ============================================
    println!("3. Block Utilities");
    println!("   Helper functions for creating block structures\n");

    // Uniform blocks
    let uniform = block_utils::uniform_blocks(15, 4);
    println!(
        "   Uniform blocks (dim=15, size=4): {} blocks",
        uniform.len()
    );
    for (i, block) in uniform.iter().enumerate() {
        println!("   - Block {}: {:?}", i, block.indices);
    }

    // Overlapping blocks
    println!("\n   Overlapping blocks (dim=12, size=5, overlap=2):");
    let overlapping = block_utils::overlapping_blocks(12, 5, 2).expect("valid overlap");
    println!("   Number of blocks: {}", overlapping.len());
    for (i, block) in overlapping.iter().enumerate() {
        println!("   - Block {}: {:?}", i, block.indices);
    }

    // ============================================
    // 4. Dual Decomposition
    // ============================================
    println!("\n4. Dual Decomposition");
    println!("   Exploits separability in constraint structure\n");

    // Coupling matrix: 2 coupling constraints, 3 subproblems
    let coupling = Array2::from_shape_vec((2, 3), vec![1.0, 0.5, 0.0, 0.0, 0.5, 1.0]).unwrap();

    let mut dual_decomp = DualDecomposition::new(3, coupling)
        .with_step_size(0.05)
        .with_max_iterations(100);

    println!("   Number of subproblems: 3");
    println!("   Number of coupling constraints: 2");
    println!("   Step size: 0.05");

    // Simulate dual updates
    println!("\n   Simulating dual ascent:");
    for iter in 0..3 {
        // Dummy constraint violations for demonstration
        let violations =
            Array1::from_vec(vec![0.5 - (iter as f32) * 0.1, 0.3 - (iter as f32) * 0.05]);
        dual_decomp
            .update_duals(&violations)
            .expect("matching violation vector");

        println!(
            "   Iter {}: dual vars = {:?}",
            iter + 1,
            dual_decomp.dual_variables()
        );
    }

    // ============================================
    // 5. Hierarchical Decomposition
    // ============================================
    println!("\n5. Hierarchical Decomposition");
    println!("   Multi-level decomposition for very large problems\n");

    // Create a two-level hierarchy
    let fine_blocks = block_utils::uniform_blocks(20, 2);
    let coarse_blocks = block_utils::uniform_blocks(20, 5);

    match HierarchicalDecomposition::from_blocks(coarse_blocks, fine_blocks) {
        Ok(hier) => {
            println!("   Created hierarchical decomposition");
            println!("   Number of levels: {}", hier.num_levels());

            for i in 0..hier.num_levels() {
                if let Some(level) = hier.level(i) {
                    println!("   Level {}: {} subproblems", i, level.subproblems.len());
                }
            }
        }
        Err(e) => println!("   Error: {}", e),
    }

    println!("\n=== Summary ===");
    println!("Decomposition methods enable:");
    println!("• Parallel constraint solving");
    println!("• Scalability to high dimensions");
    println!("• Exploiting problem structure");
    println!("• Distributed optimization");
    println!("\nEssential for large-scale constrained ML systems!");
}
