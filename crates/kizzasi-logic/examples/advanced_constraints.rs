//! Advanced constraints example
//!
//! Demonstrates usage of chance constraints, robust constraints, and risk-aware constraints.

use kizzasi_logic::{
    CVaRConstraint, ChanceConstraint, DistributionallyRobustConstraint, RobustConstraint,
};

fn main() {
    println!("=== Advanced Constraints Example ===\n");

    // ============================================
    // 1. Chance Constraints
    // ============================================
    println!("1. Chance Constraints");
    println!("   Probabilistic constraints: Pr[g(x) <= 0] >= 1 - ε\n");

    let chance = ChanceConstraint::gaussian(
        "temperature_limit",
        0.95, // 95% confidence
        25.0, // mean temperature
        3.0,  // std dev
    );

    println!("   Constraint: {}", chance.name());
    println!("   Confidence: {}", chance.confidence());
    println!("   Tightened bound: {:.2}°C", chance.get_tightened_bound());
    println!("   (Mean + 1.96*σ for 95% confidence)\n");

    // ============================================
    // 2. Robust Constraints
    // ============================================
    println!("2. Robust Constraints");
    println!("   Worst-case constraints: g(x, ξ) <= 0 ∀ξ ∈ Ξ\n");

    let robust = RobustConstraint::box_uncertain(
        "power_limit",
        vec![-5.0, -3.0], // min deviations
        vec![5.0, 3.0],   // max deviations
    );

    let x = [10.0, 20.0];
    let worst_case = robust.worst_case_scenario(&x);
    println!("   Constraint: {}", robust.name());
    println!("   Nominal values: {:?}", x);
    println!("   Worst-case scenario: {:?}\n", worst_case);

    // Ellipsoidal uncertainty
    let _robust_ellip = RobustConstraint::ellipsoidal_uncertain(
        "response_time",
        vec![100.0, 50.0], // nominal response times (ms)
        10.0,              // uncertainty radius
    );

    println!("   Ellipsoidal uncertainty:");
    println!("   Nominal: [100ms, 50ms]");
    println!("   Uncertainty radius: 10ms\n");

    // ============================================
    // 3. CVaR Constraints (Risk-Aware)
    // ============================================
    println!("3. CVaR Constraints (Conditional Value at Risk)");
    println!("   Risk measure: CVaR_α(loss) <= threshold\n");

    let cvar = CVaRConstraint::new(
        "portfolio_risk",
        0.05,  // α = 5% (worst 5% of cases)
        100.0, // threshold
        1000,  // number of scenarios
    );

    // Sample losses from different scenarios
    let scenario_losses: Vec<f32> = (0..100)
        .map(|i| {
            // Simulated losses: most are small, few are large
            if i < 95 {
                (i as f32) * 0.5
            } else {
                (i as f32) * 2.0
            }
        })
        .collect();

    let cvar_value = cvar.compute_cvar(&scenario_losses);
    println!("   Constraint: {}", cvar.name());
    println!("   α (risk level): {}", cvar.alpha());
    println!("   Threshold: {}", cvar.threshold());
    println!("   Computed CVaR: {:.2}", cvar_value);
    println!(
        "   Satisfied: {}",
        if cvar.check(&scenario_losses) {
            "Yes"
        } else {
            "No"
        }
    );
    println!(
        "   (CVaR is average of worst {}% scenarios)\n",
        cvar.alpha() * 100.0
    );

    // ============================================
    // 4. Distributionally Robust Constraints
    // ============================================
    println!("4. Distributionally Robust Constraints");
    println!("   Worst-case over probability distributions: sup_{{P ∈ P}} E_P[loss]\n");

    let drc = DistributionallyRobustConstraint::wasserstein(
        "energy_consumption",
        50.0, // threshold
        100,  // number of samples
        0.5,  // Wasserstein radius
    );

    // Sample consumption data
    let consumption_data = vec![30.0, 35.0, 40.0, 45.0, 50.0, 55.0];

    let wc_expectation = drc.worst_case_expectation(&consumption_data);
    println!("   Constraint: {}", drc.name());
    println!("   Ambiguity set: Wasserstein ball");
    println!("   Threshold: {}", drc.threshold());
    println!("   Sample mean: {:.2}", {
        let sum: f32 = consumption_data.iter().sum();
        sum / consumption_data.len() as f32
    });
    println!("   Worst-case expectation: {:.2}", wc_expectation);
    println!(
        "   Satisfied: {}",
        if drc.check(&consumption_data) {
            "Yes"
        } else {
            "No"
        }
    );
    println!("\n   (Accounts for distributional uncertainty)\n");

    // ============================================
    // 5. Moment-Based Distributionally Robust
    // ============================================
    println!("5. Moment-Based Distributionally Robust");

    let drc_moment = DistributionallyRobustConstraint::moment_based(
        "manufacturing_defects",
        10.0,           // threshold
        vec![5.0, 3.0], // mean defect rates
        1.0,            // covariance radius
    );

    let defect_data = vec![4.0, 5.0, 6.0, 5.0, 7.0, 4.0];

    println!("   Constraint: {}", drc_moment.name());
    println!("   Ambiguity set: Moment-based");
    println!("   Known mean bounds: [5.0, 3.0]");
    println!("   Covariance radius: 1.0");
    println!(
        "   Worst-case exp: {:.2}",
        drc_moment.worst_case_expectation(&defect_data)
    );
    println!(
        "   Satisfied: {}",
        if drc_moment.check(&defect_data) {
            "Yes"
        } else {
            "No"
        }
    );

    println!("\n=== Summary ===");
    println!("Advanced constraints enable handling:");
    println!("• Probabilistic uncertainty (chance constraints)");
    println!("• Worst-case scenarios (robust constraints)");
    println!("• Tail risk (CVaR constraints)");
    println!("• Distributional ambiguity (DRO constraints)");
    println!("\nThese are essential for safety-critical ML systems!");
}
