//! Basic constraint usage examples
//!
//! This example demonstrates the fundamental constraint types
//! and how to use them for signal validation and projection.

use kizzasi_logic::*;

fn main() {
    println!("=== kizzasi-logic: Constraint Basics ===\n");

    // Example 1: Simple scalar constraints
    example_scalar_constraints();

    // Example 2: Linear constraints
    example_linear_constraints();

    // Example 3: Composed constraints
    example_composed_constraints();

    // Example 4: Temporal constraints
    example_temporal_constraints();

    // Example 5: Geometric constraints
    example_geometric_constraints();
}

fn example_scalar_constraints() {
    println!("--- Scalar Constraints ---");

    // Create a simple bound constraint: x <= 10
    let max_velocity = ConstraintBuilder::new()
        .name("max_velocity")
        .less_than(10.0)
        .build()
        .expect("Failed to build constraint");

    // Check values
    let safe_velocity = 8.0;
    let unsafe_velocity = 12.0;

    println!("  max_velocity constraint: x < 10");
    println!(
        "  Check {}: {}",
        safe_velocity,
        max_velocity.check(safe_velocity)
    );
    println!(
        "  Check {}: {}",
        unsafe_velocity,
        max_velocity.check(unsafe_velocity)
    );

    // Project unsafe value onto valid region
    let projected = max_velocity.project(unsafe_velocity);
    println!("  Projected {} -> {}\n", unsafe_velocity, projected);

    // Range constraint
    let temperature = ConstraintBuilder::new()
        .name("temperature_range")
        .in_range(-10.0, 40.0)
        .build()
        .expect("Failed to build constraint");

    println!("  temperature_range constraint: -10 <= x <= 40");
    println!("  Check -5.0: {}", temperature.check(-5.0));
    println!("  Check 50.0: {}", temperature.check(50.0));
    println!("  Projected 50.0 -> {}\n", temperature.project(50.0));
}

fn example_linear_constraints() {
    println!("--- Linear Constraints ---");

    // Linear constraint: x + y <= 5
    let sum_constraint = LinearConstraint::less_eq(vec![1.0, 1.0], 5.0);

    let point1 = vec![2.0, 2.0]; // Satisfies
    let point2 = vec![3.0, 4.0]; // Violates (sum = 7)

    println!("  Constraint: x + y <= 5");
    println!("  Check {:?}: {}", point1, sum_constraint.check(&point1));
    println!("  Check {:?}: {}", point2, sum_constraint.check(&point2));

    let violation = sum_constraint.violation(&point2);
    println!("  Violation at {:?}: {}", point2, violation);

    let projected = sum_constraint.project(&point2);
    println!("  Projected {:?} -> {:?}\n", point2, projected);

    // Multiple linear constraints
    let constraints = vec![
        LinearConstraint::less_eq(vec![1.0, 0.0], 3.0), // x <= 3
        LinearConstraint::less_eq(vec![0.0, 1.0], 4.0), // y <= 4
        LinearConstraint::greater_eq(vec![1.0, 1.0], 2.0), // x + y >= 2
    ];
    let constraint_set = LinearConstraintSet::new(constraints);

    let test_point = vec![5.0, 5.0];
    println!("  Constraint set: x <= 3, y <= 4, x+y >= 2");
    println!(
        "  All satisfied at {:?}: {}",
        test_point,
        constraint_set.check_all(&test_point)
    );
    println!(
        "  Total violation: {}\n",
        constraint_set.total_violation(&test_point)
    );
}

fn example_composed_constraints() {
    println!("--- Composed Constraints ---");

    // Create individual constraints
    let min_speed = ConstraintBuilder::new()
        .name("min_speed")
        .greater_than(5.0)
        .build()
        .expect("Failed to build");

    let max_speed = ConstraintBuilder::new()
        .name("max_speed")
        .less_than(100.0)
        .build()
        .expect("Failed to build");

    // Compose with AND: speed > 5 AND speed < 100
    let speed_range =
        ComposedConstraint::single(min_speed).and(ComposedConstraint::single(max_speed));

    println!("  Composed: (speed > 5) AND (speed < 100)");
    println!("  Check 50.0: {}", speed_range.check(50.0));
    println!("  Check 3.0: {}", speed_range.check(3.0));
    println!("  Check 150.0: {}\n", speed_range.check(150.0));
}

fn example_temporal_constraints() {
    println!("--- Temporal Constraints ---");

    // Rate of change constraint: |dx/dt| <= 2.0
    let acceleration_limit = TemporalConstraintBuilder::new()
        .name("max_acceleration")
        .max_rate(2.0)
        .dt(0.1)
        .build()
        .expect("Failed to build temporal constraint");

    println!("  Temporal constraint: |dx/dt| <= 2.0 with dt=0.1");

    let prev_velocity = 10.0;
    let curr_velocity_ok = 10.15; // Change = 0.15, rate = 1.5 ✓
    let curr_velocity_bad = 10.5; // Change = 0.5, rate = 5.0 ✗

    println!(
        "  From {} to {}: {}",
        prev_velocity,
        curr_velocity_ok,
        acceleration_limit.check(prev_velocity, curr_velocity_ok)
    );
    println!(
        "  From {} to {}: {}",
        prev_velocity,
        curr_velocity_bad,
        acceleration_limit.check(prev_velocity, curr_velocity_bad)
    );

    // Project to satisfy rate limit
    let projected = acceleration_limit.project(prev_velocity, curr_velocity_bad);
    println!("  Projected {} -> {}\n", curr_velocity_bad, projected);

    // Temporal checker with multiple constraints
    let constraints = vec![acceleration_limit];
    let mut checker = TemporalChecker::new(constraints);

    println!("  Temporal checker tracking history:");
    let values = vec![10.0, 10.15, 10.28, 10.40];
    for &val in &values {
        let results = checker.check(&[val]);
        println!("    Value {}: {:?}", val, results);
    }
    println!();
}

fn example_geometric_constraints() {
    println!("--- Geometric Set Constraints ---");

    // Box constraint: [0, 10] x [0, 5]
    let safe_region =
        GeometricSet::box_constraint(vec![0.0, 0.0], vec![10.0, 5.0]).expect("valid box set");

    println!("  Box: [0,10] x [0,5]");
    println!("  Contains [5, 2]: {}", safe_region.contains(&[5.0, 2.0]));
    println!("  Contains [12, 3]: {}", safe_region.contains(&[12.0, 3.0]));

    let outside_point = vec![12.0, -1.0];
    let projected = safe_region.project(&outside_point);
    println!("  Projected {:?} -> {:?}\n", outside_point, projected);

    // Ball constraint: ||x - center|| <= radius
    let ball = GeometricSet::ball(vec![0.0, 0.0], 5.0).expect("valid ball set");

    println!("  Ball: center=[0,0], radius=5");
    println!("  Contains [3, 0]: {}", ball.contains(&[3.0, 0.0]));
    println!("  Contains [10, 0]: {}", ball.contains(&[10.0, 0.0]));

    let far_point = vec![10.0, 0.0];
    let projected_ball = ball.project(&far_point);
    println!("  Projected {:?} -> {:?}\n", far_point, projected_ball);

    // L-infinity ball: max|xi - ci| <= r
    let linf_ball = GeometricSet::l_inf_ball(vec![0.0, 0.0], 3.0).expect("valid L-inf ball");

    println!("  L-inf Ball: center=[0,0], radius=3");
    println!("  Contains [2, 2]: {}", linf_ball.contains(&[2.0, 2.0]));
    println!("  Contains [5, 1]: {}", linf_ball.contains(&[5.0, 1.0]));
}
