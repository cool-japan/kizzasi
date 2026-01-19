//! Training integration example
//!
//! Demonstrates how to integrate constraints into neural network training.

use kizzasi_logic::{ConstraintBuilder, DifferentiableProjection, PenaltyFunction};

fn main() {
    println!("=== Training Integration Example ===\n");

    // ============================================
    // 1. Basic Constraint Setup
    // ============================================
    println!("1. Basic Constraint Setup for Training");

    let velocity_constraint = ConstraintBuilder::new()
        .name("max_velocity")
        .less_than(10.0)
        .build()
        .unwrap();

    println!("   Constraint: {}", velocity_constraint.name());
    println!("   Type: Upper bound (< 10.0)");
    println!("   Purpose: Ensure predictions stay within safe limits\n");

    // ============================================
    // 2. Penalty Functions
    // ============================================
    println!("2. Penalty Functions");
    println!("   Different penalties for soft constraint enforcement:\n");

    let violation = 2.0f32;
    let weight = 1.0f32;

    let l1_penalty = PenaltyFunction::L1.compute(violation, weight);
    let l2_penalty = PenaltyFunction::L2.compute(violation, weight);
    let huber_penalty = PenaltyFunction::Huber { delta: 1.0 }.compute(violation, weight);

    println!("   Violation: {}", violation);
    println!("   Weight: {}", weight);
    println!("   L1 penalty: {:.4}", l1_penalty);
    println!("   L2 penalty: {:.4}", l2_penalty);
    println!("   Huber penalty: {:.4}\n", huber_penalty);

    // ============================================
    // 3. Differentiable Projection
    // ============================================
    println!("3. Differentiable Projection (Soft Projection)");
    println!("   Smooth projections that preserve gradients:\n");

    let diff_proj = DifferentiableProjection::new(1.0);

    println!("   Temperature: {}", diff_proj.temperature());
    println!("   Soft box projections (lower=0, upper=10):\n");

    for x in &[-5.0f32, 0.0, 5.0, 10.0, 15.0] {
        let projected = diff_proj.soft_project_box(*x, 0.0, 10.0);
        println!("   x={:>5.1} -> projected={:>5.2}", x, projected);
    }

    println!("\n   (Smoother than hard projection, gradient-friendly)\n");

    // ============================================
    // 4. Temperature Effect
    // ============================================
    println!("4. Temperature Effect on Soft Projection");
    println!("   Lower temperature -> closer to hard projection:\n");

    let x_test = 15.0f32;
    for &temp in &[5.0f32, 2.0, 1.0, 0.5, 0.1] {
        let proj = DifferentiableProjection::new(temp);
        let result = proj.soft_project_box(x_test, 0.0, 10.0);
        println!("   Temperature {:.1}: x={} -> {:.2}", temp, x_test, result);
    }

    println!("\n=== Training Loop Pattern ===");
    println!("```rust");
    println!("// Setup");
    println!("let constraints = vec![max_velocity, max_acceleration];");
    println!("let penalty = PenaltyFunction::L2;");
    println!("let loss_fn = ConstraintAwareLoss::new(constraints, penalty, weight);");
    println!();
    println!("// Training loop");
    println!("for epoch in 1..num_epochs {{");
    println!("    let prediction = model.forward(batch);");
    println!("    ");
    println!("    // Compute task loss (e.g., MSE)");
    println!("    let task_loss = mse(prediction, target);");
    println!("    ");
    println!("    // Add constraint penalty");
    println!("    let total_loss = loss_fn.compute_loss(&prediction, task_loss);");
    println!("    ");
    println!("    // Backprop and update");
    println!("    total_loss.backward();");
    println!("    optimizer.step();");
    println!("}}");
    println!("```");

    println!("\n=== Key Concepts ===");
    println!("1. Soft constraints via penalty functions");
    println!("   • L1: Linear penalty, sparse solutions");
    println!("   • L2: Quadratic penalty, smooth optimization");
    println!("   • Huber: Robust to outliers");
    println!();
    println!("2. Differentiable projections");
    println!("   • Smooth approximations preserve gradients");
    println!("   • Temperature controls trade-off");
    println!();
    println!("3. Constraint-aware loss");
    println!("   • Combines task objective + constraint violations");
    println!("   • Weighted to balance performance vs safety");
    println!();
    println!("4. Annealing strategies");
    println!("   • Start with soft constraints (high temperature)");
    println!("   • Gradually tighten (decrease temperature)");
    println!("   • Increase penalty weights over training");

    println!("\n=== Summary ===");
    println!("Constraint-aware training ensures ML models:");
    println!("• Respect physical/logical constraints");
    println!("• Stay within safe operating regions");
    println!("• Produce physically plausible predictions");
    println!("\nEssential for physics-informed and safety-critical ML!");
}
