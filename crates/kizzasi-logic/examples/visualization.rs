//! Visualization and debugging example
//!
//! Demonstrates how to use visualization tools for:
//! - Analyzing constraint violations
//! - Debugging constraint behavior
//! - Generating inspection reports
//! - Time series analysis

use kizzasi_logic::{
    ConstraintBuilder, ConstraintInspector, ConstraintReport, ConstraintTimeSeries, ViolationStats,
};

fn main() {
    println!("=== Constraint Visualization and Debugging ===\n");

    // ============================================
    // 1. Violation Statistics
    // ============================================
    println!("1. Violation Statistics");
    println!("   Analyze how often constraints are violated\n");

    let temp_constraint = ConstraintBuilder::new()
        .name("temperature")
        .in_range(15.0, 25.0)
        .build()
        .unwrap();

    // Simulate temperature readings
    let temperatures = vec![
        18.0, 22.0, 24.0, 26.0, 28.0, // Day 1: heating up, violations
        25.0, 23.0, 21.0, 19.0, 17.0, // Day 2: cooling down
        16.0, 18.0, 20.0, 22.0, 24.0, // Day 3: within range
        14.0, 12.0, 10.0, 15.0, 18.0, // Day 4: too cold
    ];

    let stats = ViolationStats::from_values(&temp_constraint, &temperatures);
    println!("{}", stats);

    // ============================================
    // 2. Time Series Analysis
    // ============================================
    println!("\n2. Time Series Analysis");
    println!("   Track constraint violations over time\n");

    let velocity_constraint = ConstraintBuilder::new()
        .name("max_velocity")
        .less_than(100.0)
        .build()
        .unwrap();

    let mut velocity_series = ConstraintTimeSeries::new("velocity_monitor");

    // Simulate velocity readings (m/s)
    let velocities = vec![
        50.0, 75.0, 95.0, 110.0, 120.0, // Accelerating, exceeding limit
        115.0, 105.0, 98.0, 85.0, 70.0, // Decelerating back to safe
        65.0, 60.0, 55.0, 50.0, 45.0, // Stable, safe speed
    ];

    for (t, &v) in velocities.iter().enumerate() {
        velocity_series.add_sample(&velocity_constraint, v, t);
    }

    let v_stats = velocity_series.stats();
    println!("Velocity constraint statistics:");
    println!("  Samples: {}", v_stats.num_samples);
    println!(
        "  Violations: {} ({:.1}%)",
        v_stats.num_violations,
        v_stats.violation_rate() * 100.0
    );
    println!(
        "  Max violation: {:.2} m/s over limit",
        v_stats.max_violation
    );

    // Find violation periods
    let periods = velocity_series.violation_periods();
    println!("\n  Violation periods:");
    for (start, end, max_viol) in periods {
        println!(
            "    Time {}-{}: max violation = {:.2} m/s",
            start, end, max_viol
        );
    }

    // ============================================
    // 3. Constraint Report
    // ============================================
    println!("\n3. Constraint Report");
    println!("   Compare multiple constraints\n");

    let constraints = vec![
        (
            "Temperature",
            ConstraintBuilder::new()
                .name("temp")
                .in_range(15.0, 25.0)
                .build()
                .unwrap(),
            vec![20.0, 22.0, 28.0, 30.0, 24.0, 18.0],
        ),
        (
            "Humidity",
            ConstraintBuilder::new()
                .name("humidity")
                .in_range(30.0, 70.0)
                .build()
                .unwrap(),
            vec![45.0, 50.0, 55.0, 75.0, 80.0, 60.0],
        ),
        (
            "Pressure",
            ConstraintBuilder::new()
                .name("pressure")
                .in_range(980.0, 1020.0)
                .build()
                .unwrap(),
            vec![1000.0, 1005.0, 1010.0, 1015.0, 1020.0, 1025.0],
        ),
    ];

    let mut report = ConstraintReport::new();

    for (name, constraint, values) in &constraints {
        let stats = ViolationStats::from_values(constraint, values);
        report.add_constraint(*name, stats);
    }

    print!("{}", report);

    // Find most problematic constraint
    if let Some((name, stats)) = report.most_violated() {
        println!("Most violated constraint: {}", name);
        println!("  Violation rate: {:.1}%", stats.violation_rate() * 100.0);
    }

    if let Some((name, stats)) = report.largest_violations() {
        println!("\nLargest violations: {}", name);
        println!("  Max violation: {:.2}", stats.max_violation);
    }

    // ============================================
    // 4. Constraint Inspector
    // ============================================
    println!("\n4. Constraint Inspector");
    println!("   Interactive debugging tool\n");

    let power_constraint = ConstraintBuilder::new()
        .name("max_power")
        .less_than(1000.0)
        .build()
        .unwrap();

    let mut inspector = ConstraintInspector::new();

    // Add power consumption samples (Watts)
    let power_readings = vec![
        750.0, 820.0, 950.0, 1100.0, 1250.0, 1050.0, 900.0, 800.0, 750.0, 700.0,
    ];

    for &reading in &power_readings {
        inspector.add_sample(reading);
    }

    let inspection = inspector.inspect(&power_constraint, "max_power");
    inspection.print_summary();
    println!();
    inspection.print_violations(5);

    // ============================================
    // 5. Multi-Constraint Inspection
    // ============================================
    println!("\n5. Multi-Constraint Inspection");
    println!("   Inspect same data against multiple constraints\n");

    let signal_data = vec![-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1.0];

    let constraints_to_check = vec![
        (
            "signal_min",
            ConstraintBuilder::new()
                .name("min")
                .greater_than(0.0)
                .build()
                .unwrap(),
        ),
        (
            "signal_max",
            ConstraintBuilder::new()
                .name("max")
                .less_than(2.0)
                .build()
                .unwrap(),
        ),
        (
            "signal_range",
            ConstraintBuilder::new()
                .name("range")
                .in_range(0.5, 1.5)
                .build()
                .unwrap(),
        ),
    ];

    let mut signal_inspector = ConstraintInspector::new();
    for &val in &signal_data {
        signal_inspector.add_sample(val);
    }

    println!(
        "Inspecting {} signal samples:\n",
        signal_inspector.sample_count()
    );

    for (name, constraint) in &constraints_to_check {
        let result = signal_inspector.inspect(constraint, name);
        println!("Constraint: {}", name);
        println!(
            "  Satisfied: {}/{} ({:.1}%)",
            result.satisfied().len(),
            result.total_samples(),
            (result.satisfied().len() as f32 / result.total_samples() as f32) * 100.0
        );
        println!(
            "  Violated: {}/{} ({:.1}%)",
            result.violated().len(),
            result.total_samples(),
            result.violation_rate() * 100.0
        );
        println!();
    }

    println!("=== Summary ===");
    println!("Visualization tools help you:");
    println!("• Understand constraint behavior");
    println!("• Debug violations");
    println!("• Compare multiple constraints");
    println!("• Analyze time series data");
    println!("\nEssential for developing robust constrained ML systems!");
}
