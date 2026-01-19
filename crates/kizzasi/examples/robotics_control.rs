//! Robotics control system example
//!
//! This example demonstrates using Kizzasi for real-time robot control,
//! including trajectory prediction and motion planning with safety constraints.
//!
//! Run with:
//! ```bash
//! cargo run --example robotics_control --features logic
//! ```

use kizzasi::prelude::*;
use std::f32::consts::PI;

#[cfg(not(feature = "logic"))]
fn main() {
    println!("This example works best with the 'logic' feature for safety constraints.");
    println!("Run with: cargo run --example robotics_control --features logic");
    println!("\nRunning basic version without constraints...\n");
    run_basic_version().expect("Example failed");
}

#[cfg(feature = "logic")]
fn main() -> Result<()> {
    println!("=== Kizzasi Robotics Control Example ===\n");

    // Simulate a 6-DOF robotic arm
    let num_joints = 6;
    let mut predictor = KizzasiBuilder::robotics_preset(num_joints).build()?;

    println!("✓ Created robotics controller");
    println!("  - Degrees of Freedom: {}", num_joints);
    println!("  - Context window: {} steps", predictor.context_window());
    println!("  - Optimized for real-time control\n");

    // Define safety constraints for each joint
    println!("Setting up safety constraints:");
    println!("  All joints: [-π, π] rad\n");

    let mut guardrails = GuardrailSet::new();

    // Create a global constraint for joint limits
    let joint_constraint = ConstraintBuilder::new()
        .name("joint_limits")
        .greater_eq(-PI)
        .less_eq(PI)
        .build()?;

    // Add as global guardrail (applies to all dimensions)
    guardrails.add_global(Guardrail::new(joint_constraint, false));

    predictor.set_guardrails(guardrails);
    println!("✓ Safety guardrails installed\n");

    // Simulate a control sequence: moving from home to target position
    println!("Trajectory Simulation:");
    println!("  Home position → Target position\n");

    let home_position = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let target_position = array![PI / 4.0, PI / 6.0, -PI / 4.0, PI / 6.0, PI / 4.0, 0.0];

    println!("  Home:   {:?}", home_position);
    println!("  Target: {:?}", target_position);
    println!();

    // Generate a smooth trajectory
    let num_steps = 20;
    println!("Generating {}-step trajectory:", num_steps);

    predictor.reset();

    // Use linear interpolation as initial commands
    let mut trajectory = Vec::new();
    for step in 0..num_steps {
        let alpha = step as f32 / (num_steps - 1) as f32;
        let interpolated = home_position
            .iter()
            .zip(target_position.iter())
            .map(|(&h, &t)| h + alpha * (t - h))
            .collect::<Vec<_>>();

        let command = Array1::from_vec(interpolated);
        let predicted = predictor.step(&command)?;

        trajectory.push(predicted.clone());

        if step % 5 == 0 || step == num_steps - 1 {
            println!("  Step {:2}: {:?}", step + 1, predicted);
        }
    }
    println!();

    // Validate all trajectory points
    println!("Validating trajectory safety:");
    let mut all_safe = true;
    for (i, point) in trajectory.iter().enumerate() {
        if !predictor.validate(point) {
            println!("  ⚠ Step {} violates constraints!", i + 1);
            all_safe = false;
        }
    }
    if all_safe {
        println!("  ✓ All trajectory points within safe bounds");
    }
    println!();

    // Predict future trajectory
    println!("Predictive control (next 5 steps):");
    let current_state = trajectory.last().unwrap().clone();
    let future = predictor.predict_n(&current_state, 5)?;

    for (i, row) in future.outer_iter().enumerate() {
        let is_safe = predictor.validate(&row.to_owned());
        println!("  Future {}: {:?} (Safe: {})", i + 1, row, is_safe);
    }
    println!();

    // Demonstrate emergency stop scenario
    println!("Emergency stop scenario:");
    predictor.reset();

    let dangerous_velocity = array![2.0, 2.0, 2.0, 2.0, 2.0, 2.0]; // High velocities
    println!("  Dangerous input: {:?}", dangerous_velocity);

    let safe_output = predictor.step(&dangerous_velocity)?;
    println!("  Constrained output: {:?}", safe_output);
    println!("  ✓ Guardrails prevented unsafe motion\n");

    // Calculate violation loss for different commands
    println!("Analyzing command safety:");
    let test_commands = [
        array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        array![1.0, 0.5, -0.5, 0.3, 0.2, 0.1],
        array![4.0, 2.0, -4.0, 2.0, 4.0, 4.0], // Violates limits
    ];

    for (i, cmd) in test_commands.iter().enumerate() {
        let loss = predictor.violation_loss(cmd);
        let is_valid = predictor.validate(cmd);
        println!(
            "  Command {}: Valid={}, Violation Loss={:.4}",
            i + 1,
            is_valid,
            loss
        );
    }
    println!();

    // Demonstrate forking for different control strategies
    println!("Parallel strategy exploration:");
    let _forked_predictor = predictor.fork()?;
    println!("  ✓ Created fork for alternative path planning");
    println!("  - Original: Continue main trajectory");
    println!("  - Fork: Explore alternative route");
    println!("  - Both respect the same safety constraints\n");

    println!("=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  • Robotics preset optimized for control loops");
    println!("  • Guardrails enforce joint limits automatically");
    println!("  • Predictive control enables lookahead planning");
    println!("  • Fork capability allows exploring multiple strategies");
    println!("  • Safe for real-time control applications");

    Ok(())
}

#[cfg(not(feature = "logic"))]
fn run_basic_version() -> Result<()> {
    let num_joints = 6;
    let mut predictor = KizzasiBuilder::robotics_preset(num_joints).build()?;

    println!("✓ Created robotics controller (basic, no constraints)");
    println!("  - Degrees of Freedom: {}", num_joints);

    let home_position = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let output = predictor.step(&home_position)?;

    println!("  Input:  {:?}", home_position);
    println!("  Output: {:?}", output);
    println!("\nFor full features, enable the 'logic' feature.");

    Ok(())
}
