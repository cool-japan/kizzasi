//! Guardrails example with constraint enforcement
//!
//! This example demonstrates using TensorLogic constraints to enforce
//! safety bounds and logical rules on predictions.
//!
//! Run with:
//! ```bash
//! cargo run --example with_guardrails --features logic
//! ```

use kizzasi::prelude::*;

#[cfg(not(feature = "logic"))]
fn main() {
    println!("This example requires the 'logic' feature.");
    println!("Run with: cargo run --example with_guardrails --features logic");
}

#[cfg(feature = "logic")]
fn main() -> Result<()> {
    println!("=== Kizzasi Guardrails Example ===\n");

    // Create a predictor for a 3-DOF robot arm
    // Each output represents joint angles in radians
    let mut predictor = KizzasiBuilder::robotics_preset(3).build()?;

    println!("✓ Created robotics predictor (3-DOF)");
    println!("  - Input/Output: Joint angles [rad]");
    println!("  - No constraints yet\n");

    // Test without guardrails
    let input = array![0.5, 1.0, -0.5];
    println!("Prediction without guardrails:");
    println!("  Input:  {:?}", input);

    let output = predictor.step(&input)?;
    println!("  Output: {:?}", output);
    println!("  (may exceed safe ranges)\n");

    // Create guardrails with safety bounds
    println!("Setting up guardrails:");
    println!("  - All joints: [-π, π] range\n");

    let mut guardrails = GuardrailSet::new();

    // Create a bound constraint for joint limits (-π to π)
    let joint_constraint = ConstraintBuilder::new()
        .name("joint_limits")
        .greater_eq(-std::f32::consts::PI)
        .less_eq(std::f32::consts::PI)
        .build()?;

    // Add as global guardrail (applies to all dimensions)
    guardrails.add_global(Guardrail::new(joint_constraint, false));

    predictor.set_guardrails(guardrails);
    println!("✓ Guardrails installed");

    // Test with guardrails
    predictor.reset();
    println!("\nPrediction with guardrails:");
    println!("  Input:  {:?}", input);

    let output = predictor.step(&input)?;
    println!("  Output: {:?}", output);
    println!("  ✓ Constrained to safe ranges\n");

    // Validate predictions
    let test_values = [
        array![0.0, 0.0, 0.0],
        array![2.0, 0.5, 1.5],
        array![-2.0, -0.5, -1.5],
        array![4.0, 2.0, 4.0], // This should be invalid
    ];

    println!("Validating test values:");
    for (i, val) in test_values.iter().enumerate() {
        let is_valid = predictor.validate(val);
        let loss = predictor.violation_loss(val);
        println!(
            "  Value {}: {:?} - Valid: {}, Loss: {:.4}",
            i + 1,
            val,
            is_valid,
            loss
        );
    }
    println!();

    // Multi-step prediction with constraints
    println!("Multi-step constrained prediction:");
    predictor.reset();

    let predictions = predictor.predict_n(&input, 5)?;
    for (i, row) in predictions.outer_iter().enumerate() {
        let is_valid = predictor.validate(&row.to_owned());
        println!("  Step {}: {:?} (Valid: {})", i + 1, row, is_valid);
    }
    println!();

    // Demonstrate guardrail checking
    println!("Checking guardrail status:");
    println!("  Has guardrails: {}", predictor.has_guardrails());

    predictor.clear_guardrails();
    println!("  After clearing: {}", predictor.has_guardrails());
    println!();

    println!("=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  • Guardrails enforce safety constraints on predictions");
    println!("  • Violations are automatically corrected via projection");
    println!("  • Useful for robotics, control systems, and safety-critical apps");

    Ok(())
}
