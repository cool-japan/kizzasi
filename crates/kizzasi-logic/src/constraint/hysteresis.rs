use serde::{Deserialize, Serialize};

// ============================================================================
// Hysteresis Constraints
// ============================================================================

/// Hysteresis constraint for preventing rapid state transitions
///
/// Implements hysteresis (schmitt trigger) behavior to avoid chattering
/// when a signal is near a threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HysteresisConstraint {
    name: String,
    /// Lower threshold
    low_threshold: f32,
    /// Upper threshold
    high_threshold: f32,
    /// Current state (true = high, false = low)
    state: bool,
    /// Weight for loss computation
    weight: f32,
}

impl HysteresisConstraint {
    /// Create a new hysteresis constraint
    ///
    /// - When state is low: transition to high when value > high_threshold
    /// - When state is high: transition to low when value < low_threshold
    pub fn new(name: impl Into<String>, low_threshold: f32, high_threshold: f32) -> Self {
        assert!(
            low_threshold < high_threshold,
            "Low threshold must be less than high threshold"
        );
        Self {
            name: name.into(),
            low_threshold,
            high_threshold,
            state: false,
            weight: 1.0,
        }
    }

    /// Set initial state
    pub fn with_initial_state(mut self, state: bool) -> Self {
        self.state = state;
        self
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Update state based on new value and check constraint
    pub fn update_and_check(&mut self, value: f32) -> bool {
        let new_state = if self.state {
            // Currently high: check if should transition to low
            value >= self.low_threshold
        } else {
            // Currently low: check if should transition to high
            value > self.high_threshold
        };

        self.state = new_state;
        true // Hysteresis always "satisfies" by design
    }

    /// Get current state
    pub fn state(&self) -> bool {
        self.state
    }

    /// Check if value would cause a state transition
    pub fn would_transition(&self, value: f32) -> bool {
        if self.state {
            value < self.low_threshold
        } else {
            value > self.high_threshold
        }
    }

    /// Get the constraint name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get weight
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Reset to initial state
    pub fn reset(&mut self, initial_state: bool) {
        self.state = initial_state;
    }
}

/// Checker for multiple hysteresis constraints
#[derive(Debug, Clone)]
pub struct HysteresisChecker {
    constraints: Vec<HysteresisConstraint>,
}

impl HysteresisChecker {
    /// Create a new hysteresis checker
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Add a hysteresis constraint
    pub fn add(&mut self, constraint: HysteresisConstraint) {
        self.constraints.push(constraint);
    }

    /// Update all constraints with new value
    pub fn update(&mut self, value: f32) -> Vec<(String, bool)> {
        self.constraints
            .iter_mut()
            .map(|c| {
                let state = c.update_and_check(value);
                (c.name().to_string(), state)
            })
            .collect()
    }

    /// Get all current states
    pub fn states(&self) -> Vec<(String, bool)> {
        self.constraints
            .iter()
            .map(|c| (c.name().to_string(), c.state()))
            .collect()
    }

    /// Reset all constraints
    pub fn reset(&mut self) {
        for c in &mut self.constraints {
            c.reset(false);
        }
    }
}

impl Default for HysteresisChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        AffineEquality, ComposedConstraint, ConstraintBuilder, ConstraintSet, LinearConstraint,
        LinearConstraintSet, PenaltyFunction, QuadraticConstraint, QuadraticConstraintSet,
        SlidingWindowChecker, SlidingWindowConstraint, SlidingWindowFn, SoftHardConstraint,
        TemporalChecker, TemporalConstraintBuilder,
    };

    #[test]
    fn test_less_than_constraint() {
        let c = ConstraintBuilder::new()
            .name("max_vel")
            .less_than(10.0)
            .build()
            .unwrap();

        assert!(c.check(5.0));
        assert!(!c.check(15.0));
        assert_eq!(c.violation(5.0), 0.0);
        assert_eq!(c.violation(15.0), 5.0);
        // Regression (finding 124): the old assertion here was
        // `10.0 - f32::EPSILON`, which is itself exactly `10.0` in f32
        // arithmetic once the bound's own ULP (≈9.5e-7 at magnitude 10)
        // exceeds `f32::EPSILON` (≈1.19e-7) — the old `project` had the
        // identical bug, so both sides collapsed to `10.0` and the
        // assertion passed for the wrong reason, on a value `check` itself
        // rejects (`10.0 < 10.0` is `false`). `project` now returns the
        // true adjacent representable value, which `check` accepts.
        let projected = c.project(15.0);
        assert_eq!(projected, 10.0_f32.next_down());
        assert!(c.check(projected), "projected value must satisfy check()");
    }

    #[test]
    fn test_range_constraint() {
        let c = ConstraintBuilder::new()
            .name("temp_range")
            .in_range(-10.0, 100.0)
            .build()
            .unwrap();

        assert!(c.check(50.0));
        assert!(!c.check(-20.0));
        assert!(!c.check(150.0));
        assert_eq!(c.project(-20.0), -10.0);
        assert_eq!(c.project(150.0), 100.0);
    }

    #[test]
    fn test_composed_and() {
        let c1 = ConstraintBuilder::new()
            .name("lower_bound")
            .greater_eq(0.0)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("upper_bound")
            .less_eq(10.0)
            .build()
            .unwrap();

        let composed = ComposedConstraint::single(c1).and(ComposedConstraint::single(c2));

        assert!(composed.check(5.0)); // In range
        assert!(!composed.check(-1.0)); // Below lower
        assert!(!composed.check(11.0)); // Above upper
    }

    #[test]
    fn test_composed_or() {
        let c1 = ConstraintBuilder::new()
            .name("negative")
            .less_than(0.0)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("large")
            .greater_than(100.0)
            .build()
            .unwrap();

        let composed = ComposedConstraint::single(c1).or(ComposedConstraint::single(c2));

        assert!(composed.check(-5.0)); // Negative
        assert!(composed.check(150.0)); // Large
        assert!(!composed.check(50.0)); // Neither
    }

    #[test]
    fn test_composed_not() {
        let c = ConstraintBuilder::new()
            .name("positive")
            .greater_eq(0.0)
            .build()
            .unwrap();

        let composed = ComposedConstraint::single(c).negate();

        assert!(composed.check(-1.0)); // Not positive
        assert!(!composed.check(1.0)); // Positive (fails negate)
    }

    #[test]
    fn test_composed_implies() {
        // If x > 50, then x < 100
        let premise = ConstraintBuilder::new()
            .name("large")
            .greater_than(50.0)
            .build()
            .unwrap();
        let conclusion = ConstraintBuilder::new()
            .name("bounded")
            .less_than(100.0)
            .build()
            .unwrap();

        let composed =
            ComposedConstraint::single(premise).implies(ComposedConstraint::single(conclusion));

        assert!(composed.check(30.0)); // Premise false, ok
        assert!(composed.check(75.0)); // Premise true, conclusion true
        assert!(!composed.check(150.0)); // Premise true, conclusion false
    }

    #[test]
    fn test_composed_projection_and() {
        let c1 = ConstraintBuilder::new()
            .name("lower")
            .greater_eq(0.0)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("upper")
            .less_eq(10.0)
            .build()
            .unwrap();

        let composed = ComposedConstraint::single(c1).and(ComposedConstraint::single(c2));

        assert_eq!(composed.project(-5.0), 0.0); // Projects to lower bound
        assert_eq!(composed.project(15.0), 10.0); // Projects to upper bound
        assert_eq!(composed.project(5.0), 5.0); // Already valid
    }

    #[test]
    fn test_composed_projection_or() {
        let c1 = ConstraintBuilder::new()
            .name("small")
            .less_eq(0.0)
            .build()
            .unwrap();
        let c2 = ConstraintBuilder::new()
            .name("large")
            .greater_eq(10.0)
            .build()
            .unwrap();

        let composed = ComposedConstraint::single(c1).or(ComposedConstraint::single(c2));

        // Value 5 is invalid; closer to 10 than to 0
        assert_eq!(composed.project(6.0), 10.0);
        // Value 3 is closer to 0
        assert_eq!(composed.project(3.0), 0.0);
    }

    #[test]
    fn test_check_all_dimensions() {
        let c = ConstraintBuilder::new()
            .name("bounded")
            .in_range(-1.0, 1.0)
            .build()
            .unwrap();

        let composed = ComposedConstraint::single(c);

        assert!(composed.check_all(&[0.0, 0.5, -0.5]));
        assert!(!composed.check_all(&[0.0, 2.0, 0.0]));
    }

    #[test]
    fn test_temporal_max_rate() {
        let c = TemporalConstraintBuilder::new()
            .name("max_velocity")
            .max_rate(10.0) // max 10 units/s
            .dt(0.1) // 100ms time step
            .build()
            .unwrap();

        // Rate = (5 - 0) / 0.1 = 50 > 10, violation
        assert!(!c.check(0.0, 5.0));
        // Rate = (0.5 - 0) / 0.1 = 5 <= 10, satisfied
        assert!(c.check(0.0, 0.5));
        // Rate = (0 - 0.5) / 0.1 = -5, |rate| = 5 <= 10, satisfied
        assert!(c.check(0.5, 0.0));
    }

    #[test]
    fn test_temporal_rate_range() {
        let c = TemporalConstraintBuilder::new()
            .name("rate_bounded")
            .rate_range(-5.0, 10.0)
            .dt(1.0)
            .build()
            .unwrap();

        // Rate = 3, in range
        assert!(c.check(0.0, 3.0));
        // Rate = -3, in range
        assert!(c.check(3.0, 0.0));
        // Rate = 15, above max
        assert!(!c.check(0.0, 15.0));
        // Rate = -10, below min
        assert!(!c.check(10.0, 0.0));
    }

    #[test]
    fn test_temporal_monotonic() {
        let inc = TemporalConstraintBuilder::new()
            .name("monotonic_inc")
            .monotonic_increasing()
            .dt(1.0)
            .build()
            .unwrap();

        let dec = TemporalConstraintBuilder::new()
            .name("monotonic_dec")
            .monotonic_decreasing()
            .dt(1.0)
            .build()
            .unwrap();

        assert!(inc.check(0.0, 1.0)); // increasing ok
        assert!(!inc.check(1.0, 0.0)); // decreasing fails
        assert!(inc.check(0.0, 0.0)); // same value ok

        assert!(!dec.check(0.0, 1.0)); // increasing fails
        assert!(dec.check(1.0, 0.0)); // decreasing ok
        assert!(dec.check(0.0, 0.0)); // same value ok
    }

    #[test]
    fn test_temporal_projection() {
        let c = TemporalConstraintBuilder::new()
            .name("max_rate")
            .max_rate(10.0)
            .dt(0.1)
            .build()
            .unwrap();

        // Rate = (5 - 0) / 0.1 = 50, exceeds max
        // Projected: 0 + sign(50) * 10 * 0.1 = 1.0
        let projected = c.project(0.0, 5.0);
        assert!((projected - 1.0).abs() < 1e-5);

        // Rate = 5, within limit
        let projected = c.project(0.0, 0.5);
        assert!((projected - 0.5).abs() < 1e-5);

        // Negative rate exceeding limit
        let projected = c.project(5.0, 0.0);
        // Rate = -50, exceeds max
        // Projected: 5 + sign(-50) * 10 * 0.1 = 4.0
        assert!((projected - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_checker() {
        let c = TemporalConstraintBuilder::new()
            .name("velocity_limit")
            .max_rate(10.0)
            .dt(1.0)
            .build()
            .unwrap();

        let mut checker = TemporalChecker::new(vec![c]);

        // First check initializes state
        let result = checker.check(&[0.0]);
        assert!(result[0].1); // Always true on first check

        // Second check: rate = 5, within limit
        let result = checker.check(&[5.0]);
        assert!(result[0].1);

        // Third check: rate = 50, exceeds limit
        let result = checker.check(&[55.0]);
        assert!(!result[0].1);
    }

    #[test]
    fn test_temporal_checker_project() {
        let c = TemporalConstraintBuilder::new()
            .name("rate_limit")
            .max_rate(10.0)
            .dt(1.0)
            .build()
            .unwrap();

        let mut checker = TemporalChecker::new(vec![c]);

        // Initialize with first value
        let _ = checker.project(&[0.0]);

        // Project a value that exceeds rate
        let projected = checker.project(&[50.0]);
        // Should be limited to 0 + 10 * 1 = 10
        assert!((projected[0] - 10.0).abs() < 1e-5);

        // Next projection from 10
        let projected = checker.project(&[5.0]);
        // Rate = -5, within limit, no change
        assert!((projected[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_violation() {
        let c = TemporalConstraintBuilder::new()
            .name("rate_limit")
            .max_rate(10.0)
            .dt(1.0)
            .build()
            .unwrap();

        // Rate = 15, exceeds by 5
        assert!((c.violation(0.0, 15.0) - 5.0).abs() < 1e-5);
        // Rate = 5, no violation
        assert_eq!(c.violation(0.0, 5.0), 0.0);
    }

    #[test]
    fn test_temporal_builder_validation() {
        // Missing name
        let result = TemporalConstraintBuilder::new()
            .max_rate(10.0)
            .dt(1.0)
            .build();
        assert!(result.is_err());

        // Missing rate type
        let result = TemporalConstraintBuilder::new()
            .name("test")
            .dt(1.0)
            .build();
        assert!(result.is_err());

        // Missing dt
        let result = TemporalConstraintBuilder::new()
            .name("test")
            .max_rate(10.0)
            .build();
        assert!(result.is_err());

        // Invalid dt (zero)
        let result = TemporalConstraintBuilder::new()
            .name("test")
            .max_rate(10.0)
            .dt(0.0)
            .build();
        assert!(result.is_err());

        // Invalid dt (negative)
        let result = TemporalConstraintBuilder::new()
            .name("test")
            .max_rate(10.0)
            .dt(-1.0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_temporal_multi_dimension() {
        let c = TemporalConstraintBuilder::new()
            .name("dim1_rate")
            .dimension(1)
            .max_rate(10.0)
            .dt(1.0)
            .build()
            .unwrap();

        let mut checker = TemporalChecker::new(vec![c]);

        // Initialize
        let _ = checker.check(&[0.0, 0.0, 0.0]);

        // Dim 1 rate = 5, within limit
        let result = checker.check(&[100.0, 5.0, 100.0]);
        assert!(result[0].1); // Only checks dimension 1

        // Dim 1 rate = 50, exceeds limit
        let result = checker.check(&[0.0, 55.0, 0.0]);
        assert!(!result[0].1);
    }

    // ========================================================================
    // Linear Constraint Tests
    // ========================================================================

    #[test]
    fn test_linear_equality_constraint() {
        // 2*x0 + 3*x1 = 7
        let c = LinearConstraint::equality(
            vec![2.0, 3.0],
            7.0,
            0.01, // tolerance
        );

        assert!(c.check(&[2.0, 1.0])); // 2*2 + 3*1 = 7
        assert!(c.check(&[0.5, 2.0])); // 2*0.5 + 3*2 = 7
        assert!(!c.check(&[1.0, 1.0])); // 2*1 + 3*1 = 5 != 7
    }

    #[test]
    fn test_linear_inequality_constraint() {
        // x0 + x1 <= 10
        let c = LinearConstraint::less_eq(vec![1.0, 1.0], 10.0);

        assert!(c.check(&[3.0, 5.0])); // 3+5=8 <= 10
        assert!(c.check(&[5.0, 5.0])); // 5+5=10 <= 10
        assert!(!c.check(&[6.0, 5.0])); // 6+5=11 > 10
    }

    #[test]
    fn test_linear_constraint_projection() {
        // x0 + x1 <= 5
        let c = LinearConstraint::less_eq(vec![1.0, 1.0], 5.0);

        // Already satisfies
        let proj = c.project(&[2.0, 2.0]);
        assert!((proj[0] - 2.0).abs() < 0.01);
        assert!((proj[1] - 2.0).abs() < 0.01);

        // Needs projection: [4, 4] -> sum=8, need to reduce by 3
        let proj = c.project(&[4.0, 4.0]);
        let sum: f32 = proj.iter().sum();
        assert!((sum - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_linear_constraint_set() {
        let constraints = vec![
            LinearConstraint::less_eq(vec![1.0, 0.0], 5.0), // x0 <= 5
            LinearConstraint::less_eq(vec![0.0, 1.0], 5.0), // x1 <= 5
            LinearConstraint::greater_eq(vec![1.0, 0.0], 0.0), // x0 >= 0
            LinearConstraint::greater_eq(vec![0.0, 1.0], 0.0), // x1 >= 0
        ];

        let set = LinearConstraintSet::new(constraints);

        assert!(set.check_all(&[2.0, 3.0])); // in box [0,5]x[0,5]
        assert!(!set.check_all(&[6.0, 3.0])); // x0 > 5
        assert!(!set.check_all(&[-1.0, 3.0])); // x0 < 0
    }

    #[test]
    fn test_affine_equality() {
        // Ax = b where A is 2x3, b is 2-vector
        // x0 + x1 + x2 = 6
        // x0 - x1 = 0 (x0 = x1)
        let a = vec![vec![1.0, 1.0, 1.0], vec![1.0, -1.0, 0.0]];
        let b = vec![6.0, 0.0];

        let c = AffineEquality::new(a, b, 0.01);

        assert!(c.check(&[2.0, 2.0, 2.0])); // 2+2+2=6, 2-2=0
        assert!(!c.check(&[1.0, 2.0, 3.0])); // 1+2+3=6, but 1-2=-1 != 0
    }

    // ========================================================================
    // Quadratic Constraint Tests
    // ========================================================================

    #[test]
    fn test_quadratic_ball_constraint() {
        // Ball centered at origin with radius 2: ||x||² <= 4
        let c = QuadraticConstraint::ball(vec![0.0, 0.0], 2.0);

        assert!(c.check(&[0.0, 0.0])); // origin
        assert!(c.check(&[1.0, 1.0])); // inside (1+1=2 <= 4)
        assert!(c.check(&[2.0, 0.0])); // on boundary (4 <= 4)
        assert!(!c.check(&[2.0, 2.0])); // outside (4+4=8 > 4)
    }

    #[test]
    fn test_quadratic_ball_with_center() {
        // Ball centered at (1, 1) with radius 1
        let c = QuadraticConstraint::ball(vec![1.0, 1.0], 1.0);

        assert!(c.check(&[1.0, 1.0])); // center
        assert!(c.check(&[1.5, 1.0])); // inside
        assert!(!c.check(&[3.0, 3.0])); // outside
    }

    #[test]
    fn test_quadratic_constraint_violation() {
        // ||x||² <= 1
        let c = QuadraticConstraint::ball(vec![0.0, 0.0], 1.0);

        assert_eq!(c.violation(&[0.5, 0.0]), 0.0); // inside
        assert!((c.violation(&[1.0, 1.0]) - 1.0).abs() < 0.01); // outside: 2 - 1 = 1
    }

    #[test]
    fn test_quadratic_constraint_gradient() {
        // For a ball at origin: Q = I, c = 0
        // f(x) = x'Ix = ||x||², gradient = 2x
        // Create constraint directly to avoid ball() linear terms
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let c = QuadraticConstraint::less_eq(q, vec![0.0, 0.0], 1.0);
        let grad = c.gradient(&[1.0, 2.0]);
        // gradient should be 2*[1, 2] = [2, 4]
        assert!((grad[0] - 2.0).abs() < 0.01);
        assert!((grad[1] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_quadratic_constraint_projection() {
        // ||x||² <= 1
        let c = QuadraticConstraint::ball(vec![0.0, 0.0], 1.0);

        // Point inside - no change
        let proj = c.project(&[0.5, 0.0], 100, 0.1);
        assert!((proj[0] - 0.5).abs() < 0.1);

        // Point outside - should project to boundary
        let proj = c.project(&[2.0, 0.0], 100, 0.1);
        let norm: f32 = proj.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.2); // Should be near unit circle
    }

    #[test]
    fn test_quadratic_ellipsoid() {
        // Ellipsoid: 4x² + y² <= 1 (stretched in y direction)
        let a = vec![vec![4.0, 0.0], vec![0.0, 1.0]];
        let c = QuadraticConstraint::ellipsoid(a, vec![0.0, 0.0]);

        assert!(c.check(&[0.0, 0.0])); // center
        assert!(c.check(&[0.25, 0.5])); // 4*0.0625 + 0.25 = 0.5 <= 1
        assert!(!c.check(&[0.5, 0.5])); // 4*0.25 + 0.25 = 1.25 > 1
    }

    #[test]
    fn test_quadratic_constraint_set() {
        // Two overlapping balls
        let c1 = QuadraticConstraint::ball(vec![0.0, 0.0], 2.0);
        let c2 = QuadraticConstraint::ball(vec![1.0, 0.0], 2.0);

        let set = QuadraticConstraintSet::new(vec![c1, c2]);

        assert!(set.check_all(&[0.5, 0.0])); // in both
        assert!(!set.check_all(&[-2.0, 0.0])); // in first only
        assert!(!set.check_all(&[3.0, 0.0])); // in second only
    }

    // ========================================================================
    // Sliding Window Constraint Tests
    // ========================================================================

    #[test]
    fn test_sliding_window_mean() {
        let mut c = SlidingWindowConstraint::new(
            "mean_check",
            3,
            SlidingWindowFn::MeanInRange { lo: 0.0, hi: 10.0 },
        );

        // Fill buffer
        let _ = c.push_and_check(1.0);
        let _ = c.push_and_check(2.0);
        let (sat, _) = c.push_and_check(3.0); // mean = 2.0, in range
        assert!(sat);

        // Add high value
        let (sat, _) = c.push_and_check(30.0); // [2, 3, 30], mean = 11.67 > 10
        assert!(!sat);
    }

    #[test]
    fn test_sliding_window_variance() {
        let mut c = SlidingWindowConstraint::new(
            "var_check",
            3,
            SlidingWindowFn::MaxVariance { max_var: 1.0 },
        );

        let _ = c.push_and_check(1.0);
        let _ = c.push_and_check(1.0);
        let (sat, _) = c.push_and_check(1.0); // var = 0
        assert!(sat);

        let (sat, _) = c.push_and_check(10.0); // [1, 1, 10], high variance
        assert!(!sat);
    }

    #[test]
    fn test_sliding_window_max_range() {
        let mut c = SlidingWindowConstraint::new(
            "range_check",
            4,
            SlidingWindowFn::MaxRange { max_range: 5.0 },
        );

        for val in [1.0, 2.0, 3.0, 4.0] {
            let _ = c.push_and_check(val);
        }
        assert!(c.is_ready());
        let (sat, _) = c.check_window(); // range = 3, ok
        assert!(sat);

        let _ = c.push_and_check(10.0); // [2, 3, 4, 10], range = 8 > 5
        let (sat, _) = c.check_window();
        assert!(!sat);
    }

    #[test]
    fn test_sliding_window_bounded_variation() {
        let mut c = SlidingWindowConstraint::new(
            "bv_check",
            4,
            SlidingWindowFn::BoundedVariation { max_variation: 3.0 },
        );

        for val in [0.0, 1.0, 2.0, 3.0] {
            let _ = c.push_and_check(val);
        }
        // |1-0| + |2-1| + |3-2| = 3, ok
        let (sat, _) = c.check_window();
        assert!(sat);

        let _ = c.push_and_check(10.0); // [1, 2, 3, 10], variation = 1+1+7 = 9 > 3
        let (sat, _) = c.check_window();
        assert!(!sat);
    }

    #[test]
    fn test_sliding_window_trend() {
        let mut c = SlidingWindowConstraint::new(
            "trend_check",
            4,
            SlidingWindowFn::TrendInRange {
                min_slope: -0.5,
                max_slope: 2.0,
            },
        );

        // Gentle upward trend
        for val in [0.0, 1.0, 2.0, 3.0] {
            let _ = c.push_and_check(val);
        }
        let (sat, _) = c.check_window(); // slope = 1.0, in range
        assert!(sat);

        // Steep upward trend
        let _ = c.push_and_check(20.0); // [1, 2, 3, 20]
        let (sat, _) = c.check_window();
        assert!(!sat); // slope > 2.0
    }

    #[test]
    fn test_sliding_window_checker() {
        let mut checker = SlidingWindowChecker::new();
        checker.add(SlidingWindowConstraint::new(
            "mean",
            3,
            SlidingWindowFn::MeanInRange { lo: 0.0, hi: 10.0 },
        ));
        checker.add(SlidingWindowConstraint::new(
            "range",
            3,
            SlidingWindowFn::MaxRange { max_range: 5.0 },
        ));

        // Fill buffers
        for val in [1.0, 2.0, 3.0] {
            let _ = checker.push_and_check(val);
        }

        assert!(checker.all_satisfied());

        // Add violating value
        let _ = checker.push_and_check(20.0);
        assert!(!checker.all_satisfied());
    }

    #[test]
    fn test_sliding_window_all_in_range() {
        let mut c = SlidingWindowConstraint::new(
            "all_check",
            3,
            SlidingWindowFn::AllInRange { lo: 0.0, hi: 5.0 },
        );

        for val in [1.0, 2.0, 3.0] {
            let _ = c.push_and_check(val);
        }
        let (sat, _) = c.check_window();
        assert!(sat);

        let _ = c.push_and_check(10.0); // [2, 3, 10], 10 out of range
        let (sat, viol) = c.check_window();
        assert!(!sat);
        assert!((viol - 5.0).abs() < 0.01); // 10 - 5 = 5 violation
    }

    #[test]
    fn test_sliding_window_any_in_range() {
        let mut c = SlidingWindowConstraint::new(
            "any_check",
            3,
            SlidingWindowFn::AnyInRange { lo: 0.0, hi: 5.0 },
        );

        for val in [10.0, 20.0, 30.0] {
            let _ = c.push_and_check(val);
        }
        let (sat, _) = c.check_window(); // none in range
        assert!(!sat);

        let _ = c.push_and_check(3.0); // [20, 30, 3], 3 is in range
        let (sat, _) = c.check_window();
        assert!(sat);
    }

    // ========================================================================
    // Soft vs Hard Constraint Tests
    // ========================================================================

    #[test]
    fn test_penalty_function_l1() {
        let penalty = PenaltyFunction::L1;
        assert_eq!(penalty.compute(0.0, 1.0), 0.0); // no violation
        assert_eq!(penalty.compute(2.0, 1.0), 2.0); // L1 = weight * |viol|
        assert_eq!(penalty.compute(3.0, 2.0), 6.0); // 2 * 3
    }

    #[test]
    fn test_penalty_function_l2() {
        let penalty = PenaltyFunction::L2;
        assert_eq!(penalty.compute(0.0, 1.0), 0.0);
        assert_eq!(penalty.compute(2.0, 1.0), 4.0); // L2 = weight * viol²
        assert_eq!(penalty.compute(3.0, 2.0), 18.0); // 2 * 9
    }

    #[test]
    fn test_penalty_function_huber() {
        let penalty = PenaltyFunction::Huber { delta: 1.0 };

        // Small violation: L2 region
        assert!((penalty.compute(0.5, 1.0) - 0.125).abs() < 0.01); // 0.5 * 0.5²

        // Large violation: L1 region
        assert!((penalty.compute(2.0, 1.0) - 1.5).abs() < 0.01); // delta * viol - 0.5 * delta²
    }

    #[test]
    fn test_soft_hard_constraint_modes() {
        // Hard constraint: x <= 5
        let c_hard = SoftHardConstraint::hard(LinearConstraint::less_eq(vec![1.0], 5.0));
        assert!(c_hard.is_hard());
        assert!(!c_hard.is_soft());

        // Soft constraint
        let c_soft = SoftHardConstraint::soft(LinearConstraint::less_eq(vec![1.0], 5.0));
        assert!(c_soft.is_soft());
        assert!(!c_soft.is_hard());
    }

    #[test]
    fn test_soft_constraint_loss() {
        // x <= 5 as soft constraint with L2 penalty
        let c = SoftHardConstraint::soft(LinearConstraint::less_eq(vec![1.0], 5.0))
            .with_penalty(PenaltyFunction::L2)
            .with_weight(2.0);

        // Within constraint: no loss
        assert_eq!(c.loss(&[3.0]), 0.0);

        // Violating constraint: L2 loss
        // violation = 7 - 5 = 2, loss = 2.0 * 2² = 8
        assert!((c.loss(&[7.0]) - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_hard_constraint_loss() {
        // x <= 5 as hard constraint
        let c = SoftHardConstraint::hard(LinearConstraint::less_eq(vec![1.0], 5.0));

        // Within constraint: no loss
        assert_eq!(c.loss(&[3.0]), 0.0);

        // Violating constraint: infinite loss
        assert_eq!(c.loss(&[7.0]), f32::MAX);
    }

    #[test]
    fn test_constraint_set_mixed() {
        let mut set: ConstraintSet<LinearConstraint> = ConstraintSet::new();

        // Hard constraint: x >= 0
        set.add_hard(LinearConstraint::greater_eq(vec![1.0], 0.0));

        // Soft constraint: x <= 10 with L2 penalty
        set.add_soft(
            LinearConstraint::less_eq(vec![1.0], 10.0),
            PenaltyFunction::L2,
            1.0,
        );

        // Point satisfying both
        assert!(set.all_satisfied(&[5.0]));
        assert!(set.all_hard_satisfied(&[5.0]));
        assert_eq!(set.soft_loss(&[5.0]), 0.0);

        // Point violating soft constraint only
        assert!(!set.all_satisfied(&[15.0]));
        assert!(set.all_hard_satisfied(&[15.0]));
        assert!((set.soft_loss(&[15.0]) - 25.0).abs() < 0.01); // (15-10)² = 25

        // Point violating hard constraint
        assert!(!set.all_satisfied(&[-1.0]));
        assert!(!set.all_hard_satisfied(&[-1.0]));
        assert_eq!(set.total_loss(&[-1.0]), f32::MAX);
    }

    #[test]
    fn test_constraint_priority() {
        let mut set: ConstraintSet<LinearConstraint> = ConstraintSet::new();

        set.add(
            SoftHardConstraint::soft(LinearConstraint::less_eq(vec![1.0], 5.0)).with_priority(1),
        );
        set.add(
            SoftHardConstraint::soft(LinearConstraint::less_eq(vec![1.0], 10.0)).with_priority(3),
        );
        set.add(
            SoftHardConstraint::soft(LinearConstraint::less_eq(vec![1.0], 8.0)).with_priority(2),
        );

        let sorted = set.by_priority();
        assert_eq!(sorted[0].priority(), 3);
        assert_eq!(sorted[1].priority(), 2);
        assert_eq!(sorted[2].priority(), 1);
    }

    #[test]
    fn test_penalty_gradient() {
        // L2 gradient: 2 * weight * violation
        let penalty = PenaltyFunction::L2;
        assert_eq!(penalty.gradient(0.0, 1.0), 0.0);
        assert!((penalty.gradient(2.0, 1.0) - 4.0).abs() < 0.01); // 2 * 1 * 2

        // L1 gradient: weight (constant)
        let penalty = PenaltyFunction::L1;
        assert_eq!(penalty.gradient(0.0, 1.0), 0.0);
        assert_eq!(penalty.gradient(5.0, 2.0), 2.0);
    }

    #[test]
    fn test_soft_constraint_with_quadratic() {
        // Ball constraint as soft
        let ball = QuadraticConstraint::ball(vec![0.0, 0.0], 1.0);
        let c = SoftHardConstraint::soft(ball)
            .with_penalty(PenaltyFunction::L2)
            .with_weight(1.0);

        // Inside ball
        assert!(c.check(&[0.5, 0.0]));
        assert_eq!(c.loss(&[0.5, 0.0]), 0.0);

        // Outside ball: ||[2,0]||² = 4 > 1, violation = 3
        assert!(!c.check(&[2.0, 0.0]));
        assert!((c.loss(&[2.0, 0.0]) - 9.0).abs() < 0.1); // L2: 3² = 9
    }
}
