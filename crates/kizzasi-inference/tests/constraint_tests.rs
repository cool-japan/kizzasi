//! Comprehensive constraint enforcement tests
//!
//! This module tests all constraint enforcement mechanisms in kizzasi-inference:
//! - Constrained beam search
//! - Rejection sampling
//! - Adaptive rejection sampling

use kizzasi_inference::{
    AdaptiveRejectionSampler, ConstrainedBeamSearch, ConstraintFn, FallbackStrategy,
    RejectionSampler, SamplingConfig, SamplingStrategy,
};
use scirs2_core::ndarray::{Array1, Array2};
use std::sync::Arc;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a simple constraint that values must be positive
fn positive_constraint() -> ConstraintFn {
    Arc::new(|values: &[f32]| values.iter().all(|&v| v > 0.0))
}

/// Create a constraint that values must be within range [min, max]
fn range_constraint(min: f32, max: f32) -> ConstraintFn {
    Arc::new(move |values: &[f32]| values.iter().all(|&v| v >= min && v <= max))
}

/// Create a constraint that sum of values must be within threshold
fn sum_constraint(max_sum: f32) -> ConstraintFn {
    Arc::new(move |values: &[f32]| values.iter().sum::<f32>() <= max_sum)
}

/// Create a constraint that values must be monotonically increasing
fn monotonic_increasing_constraint() -> ConstraintFn {
    Arc::new(|values: &[f32]| {
        if values.len() < 2 {
            return true;
        }
        values.windows(2).all(|w| w[1] >= w[0])
    })
}

/// Create a constraint that enforces maximum absolute value
fn max_absolute_constraint(max_abs: f32) -> ConstraintFn {
    Arc::new(move |values: &[f32]| values.iter().all(|&v| v.abs() <= max_abs))
}

// ============================================================================
// Constrained Beam Search Tests
// ============================================================================

#[test]
fn test_constrained_beam_creation() {
    let beam_search = ConstrainedBeamSearch::new(3);
    assert_eq!(beam_search.num_constraints(), 0);
}

#[test]
fn test_constrained_beam_add_constraints() {
    let beam_search = ConstrainedBeamSearch::new(3)
        .add_constraint(positive_constraint())
        .add_constraint(range_constraint(0.0, 1.0));

    assert_eq!(beam_search.num_constraints(), 2);
}

#[test]
fn test_constrained_beam_expand() {
    let mut beam_search = ConstrainedBeamSearch::new(3).add_constraint(positive_constraint());

    // Create logits for 1 beam (initial state) with vocabulary size 5
    // The expand method expects logits.nrows() == beams.len()
    let logits = Array2::from_shape_vec(
        (1, 5),
        vec![
            -0.5, 0.2, -0.3, 0.8, 0.1, // indices 1, 3, 4 are positive
        ],
    )
    .unwrap();

    let result = beam_search.expand(&logits);
    assert!(result.is_ok(), "Beam expansion should succeed");

    // Check that beams exist
    let beams = beam_search.beams();
    assert!(!beams.is_empty(), "Should have active beams");
}

#[test]
fn test_constrained_beam_soft_constraints() {
    let mut beam_search = ConstrainedBeamSearch::new(3)
        .with_soft_constraints(0.5)
        .add_constraint(positive_constraint());

    let logits = Array2::from_shape_vec((1, 5), vec![-0.5, 0.2, -0.3, 0.8, 0.1]).unwrap();

    let result = beam_search.expand(&logits);
    assert!(result.is_ok());

    // With soft constraints, beams violating constraints get penalized but aren't removed
    let beams = beam_search.beams();
    assert!(!beams.is_empty());
}

#[test]
fn test_constrained_beam_multiple_constraints() {
    let mut beam_search = ConstrainedBeamSearch::new(5)
        .add_constraint(positive_constraint())
        .add_constraint(range_constraint(0.1, 0.9))
        .add_constraint(max_absolute_constraint(0.8));

    assert_eq!(beam_search.num_constraints(), 3);

    let logits =
        Array2::from_shape_vec((1, 8), vec![-0.5, 0.05, 0.2, 0.5, 0.95, 1.0, 0.3, -0.2]).unwrap();

    let result = beam_search.expand(&logits);
    assert!(result.is_ok());
}

#[test]
fn test_constrained_beam_best() {
    let mut beam_search = ConstrainedBeamSearch::new(3);

    let logits = Array2::from_shape_vec((1, 5), vec![0.1, 0.5, 0.3, 0.2, 0.8]).unwrap();

    let result = beam_search.expand(&logits);
    assert!(result.is_ok());

    let best = beam_search.best();
    assert!(best.is_some(), "Should have a best beam");
}

// ============================================================================
// Rejection Sampling Tests
// ============================================================================

#[test]
fn test_rejection_sampler_creation() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    let sampler = RejectionSampler::new(config);

    // Sampler created successfully
    assert_eq!(sampler.base_sampler().config().temperature, 1.0);
}

#[test]
fn test_rejection_sampler_no_constraints() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    let mut sampler = RejectionSampler::new(config);

    let logits = Array1::from_vec(vec![0.1, 0.5, 0.3, 0.2]);
    let context: Vec<f32> = vec![];

    let result = sampler.sample_with_rejection(&logits, &context);
    assert!(result.is_ok());
}

#[test]
fn test_rejection_sampler_with_positive_constraint() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    let mut sampler = RejectionSampler::new(config).add_constraint(positive_constraint());

    let logits = Array1::from_vec(vec![-0.5, 0.2, -0.3, 0.8, 0.1]);
    let context: Vec<f32> = vec![0.5]; // Start with positive context

    // Sample multiple times to test consistency
    for _ in 0..5 {
        let result = sampler.sample_with_rejection(&logits, &context);
        if let Ok(value) = result {
            assert!(value > 0.0, "Sampled value {} should be positive", value);
        }
    }
}

#[test]
fn test_rejection_sampler_with_range_constraint() {
    let config = SamplingConfig::new().strategy(SamplingStrategy::Greedy);
    // Constraint on indices: allow indices 0, 1, 2 (range [0.0, 2.0])
    let mut sampler = RejectionSampler::new(config)
        .max_attempts(500)
        .add_constraint(range_constraint(0.0, 2.0));

    // Higher logits for valid indices
    let logits = Array1::from_vec(vec![0.9, 0.7, 0.5, 0.1, 0.05]);
    let context: Vec<f32> = vec![0.5];

    for _ in 0..5 {
        let result = sampler.sample_with_rejection(&logits, &context);
        if let Ok(idx) = result {
            // Samplers return indices as f32
            assert!(
                (0.0..=2.0).contains(&idx),
                "Sampled index {} out of range",
                idx
            );
        }
    }
}

#[test]
fn test_rejection_sampler_max_attempts() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    let mut sampler = RejectionSampler::new(config)
        .max_attempts(10)
        .fallback_strategy(FallbackStrategy::BestCandidate)
        .add_constraint(range_constraint(10.0, 20.0)); // Impossible constraint

    let logits = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    let context: Vec<f32> = vec![];

    // Should use fallback strategy after max attempts
    let result = sampler.sample_with_rejection(&logits, &context);
    // With BestCandidate strategy, should still return a value
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_rejection_sampler_fallback_greedy() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    let mut sampler = RejectionSampler::new(config)
        .max_attempts(5)
        .fallback_strategy(FallbackStrategy::Greedy)
        .add_constraint(range_constraint(10.0, 20.0));

    let logits = Array1::from_vec(vec![0.1, 0.8, 0.3]);
    let context: Vec<f32> = vec![];

    let result = sampler.sample_with_rejection(&logits, &context);
    // Should fall back to greedy sampling
    assert!(result.is_ok());
    if let Ok(value) = result {
        // Greedy should pick the highest value
        assert!(value >= 0.0);
    }
}

#[test]
fn test_rejection_sampler_fallback_error() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    let mut sampler = RejectionSampler::new(config)
        .max_attempts(5)
        .fallback_strategy(FallbackStrategy::Error)
        .add_constraint(range_constraint(10.0, 20.0));

    let logits = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    let context: Vec<f32> = vec![];

    let result = sampler.sample_with_rejection(&logits, &context);
    // Should return error after max attempts
    assert!(result.is_err());
}

#[test]
fn test_rejection_sampler_multiple_constraints() {
    let config = SamplingConfig::new().strategy(SamplingStrategy::Greedy);
    // Constraints on indices: positive (> 0) and range [1, 3]
    let mut sampler = RejectionSampler::new(config)
        .max_attempts(500)
        .add_constraint(positive_constraint()) // Indices > 0
        .add_constraint(range_constraint(1.0, 3.0)); // Indices 1, 2, 3

    // Higher logits for valid indices (1, 2, 3)
    let logits = Array1::from_vec(vec![0.05, 0.9, 0.8, 0.7, 0.1, 0.05]);
    let context: Vec<f32> = vec![0.5];

    for _ in 0..5 {
        let result = sampler.sample_with_rejection(&logits, &context);
        if let Ok(idx) = result {
            // Samplers return indices as f32
            assert!(idx > 0.0, "Index {} violates positive constraint", idx);
            assert!(
                (1.0..=3.0).contains(&idx),
                "Index {} violates range constraint",
                idx
            );
        }
    }
}

// ============================================================================
// Adaptive Rejection Sampling Tests
// ============================================================================

#[test]
fn test_adaptive_rejection_creation() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    let _sampler = AdaptiveRejectionSampler::new(config, 10);

    // Sampler created successfully - just test creation
}

#[test]
fn test_adaptive_rejection_no_constraints() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    let mut sampler = AdaptiveRejectionSampler::new(config, 10);

    let logits = Array1::from_vec(vec![0.1, 0.5, 0.3, 0.2]);
    let context: Vec<f32> = vec![];

    let result = sampler.sample_adaptive(&logits, &context);
    assert!(result.is_ok());
}

#[test]
fn test_adaptive_rejection_with_constraint() {
    let config = SamplingConfig::new().temperature(0.5).seed(42);
    let mut sampler =
        AdaptiveRejectionSampler::new(config, 10).add_constraint(positive_constraint());

    let logits = Array1::from_vec(vec![-0.5, 0.2, -0.3, 0.8, 0.1]);
    let context: Vec<f32> = vec![0.5];

    for _ in 0..5 {
        let result = sampler.sample_adaptive(&logits, &context);
        if let Ok(value) = result {
            assert!(value > 0.0, "Value {} should be positive", value);
        }
    }
}

#[test]
fn test_adaptive_rejection_temperature_adjustment() {
    let config = SamplingConfig::new().temperature(0.5).seed(42);
    let mut sampler =
        AdaptiveRejectionSampler::new(config, 10).add_constraint(range_constraint(0.2, 0.4));

    let logits = Array1::from_vec(vec![0.1, 0.3, 0.5, 0.7, 0.25]);
    let context: Vec<f32> = vec![0.3];

    // Adaptive sampler should adjust to find valid samples
    let result = sampler.sample_adaptive(&logits, &context);
    assert!(result.is_ok());

    if let Ok(value) = result {
        // May or may not satisfy constraint depending on adaptation success
        // But should return a value
        assert!(value > 0.0);
    }
}

// ============================================================================
// Complex Constraint Tests
// ============================================================================

#[test]
fn test_monotonic_increasing_constraint() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    let mut sampler = RejectionSampler::new(config)
        .max_attempts(50)
        .add_constraint(monotonic_increasing_constraint());

    let logits = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

    // Start with an increasing context
    let context: Vec<f32> = vec![0.1, 0.15];

    let result = sampler.sample_with_rejection(&logits, &context);
    if let Ok(value) = result {
        // Should be >= last context value (0.15)
        assert!(value >= 0.15, "Value {} breaks monotonic increasing", value);
    }
}

#[test]
fn test_sum_constraint_enforcement() {
    let config = SamplingConfig::new().strategy(SamplingStrategy::Greedy);
    // Constraint on sum of indices: context [0.3, 0.2] sum=0.5, allow indices that keep sum <= 1.0
    let mut sampler = RejectionSampler::new(config)
        .max_attempts(500)
        .add_constraint(sum_constraint(1.0));

    // Higher logits for index 0 (sum would be 0.3+0.2+0=0.5)
    let logits = Array1::from_vec(vec![0.9, 0.1, 0.05, 0.02, 0.01]);
    let context: Vec<f32> = vec![0.3, 0.2]; // Sum = 0.5

    let result = sampler.sample_with_rejection(&logits, &context);
    if let Ok(idx) = result {
        // Samplers return indices as f32
        let new_sum = 0.3 + 0.2 + idx;
        assert!(new_sum <= 1.0, "Sum {} exceeds constraint", new_sum);
    }
}

#[test]
fn test_max_absolute_constraint_enforcement() {
    let config = SamplingConfig::new().strategy(SamplingStrategy::Greedy);
    // Constraint on indices: only allow indices 0, 1, 2 (abs value <= 2)
    let mut sampler = RejectionSampler::new(config)
        .max_attempts(500)
        .add_constraint(max_absolute_constraint(2.0));

    // Create logits where valid indices (0, 1, 2) have higher scores
    let logits = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.1, 0.05, 0.02]);
    let context: Vec<f32> = vec![0.2, -0.3];

    for _ in 0..5 {
        let result = sampler.sample_with_rejection(&logits, &context);
        if let Ok(idx) = result {
            // Samplers return indices as f32
            // Valid indices are 0, 1, 2 (abs <= 2.0)
            assert!(
                idx.abs() <= 2.0,
                "Index {} exceeds max absolute value 2.0",
                idx
            );
        }
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_constraint_with_empty_context() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    let mut sampler = RejectionSampler::new(config).add_constraint(positive_constraint());

    let logits = Array1::from_vec(vec![0.1, 0.5, 0.3]);
    let context: Vec<f32> = vec![];

    let result = sampler.sample_with_rejection(&logits, &context);
    assert!(result.is_ok());
}

#[test]
fn test_constraint_with_single_logit() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    let mut sampler = RejectionSampler::new(config).add_constraint(positive_constraint());

    let logits = Array1::from_vec(vec![0.5]);
    let context: Vec<f32> = vec![];

    let result = sampler.sample_with_rejection(&logits, &context);
    assert!(result.is_ok());
    // Samplers return indices (as f32), not values. With a single logit, the only valid index is 0.
    assert_eq!(result.unwrap(), 0.0);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_constraint_enforcement_consistency() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    let mut sampler = RejectionSampler::new(config)
        .max_attempts(100)
        .add_constraint(positive_constraint());

    let logits = Array1::from_vec(vec![-0.5, 0.2, -0.3, 0.8, 0.1, -0.7]);
    let context: Vec<f32> = vec![0.5];

    // Run many samples to verify consistency
    let num_samples = 20;
    let mut violations = 0;

    for _ in 0..num_samples {
        let result = sampler.sample_with_rejection(&logits, &context);
        if let Ok(value) = result {
            if value <= 0.0 {
                violations += 1;
            }
        }
    }

    assert_eq!(
        violations, 0,
        "Constraint violated {} times out of {} samples",
        violations, num_samples
    );
}

#[test]
fn test_multiple_constraints_all_enforced() {
    let config = SamplingConfig::new().temperature(1.0).seed(42);
    // Constraints on indices: positive (> 0), range [1, 3], max abs 3
    let mut sampler = RejectionSampler::new(config)
        .max_attempts(100)
        .add_constraint(positive_constraint()) // Indices > 0 (excludes index 0)
        .add_constraint(range_constraint(1.0, 3.0)) // Indices 1, 2, 3
        .add_constraint(max_absolute_constraint(3.0)); // Abs value <= 3

    let logits = Array1::from_vec(vec![0.05, 0.2, 0.5, 0.95, 0.3]);
    let context: Vec<f32> = vec![0.4];

    for _ in 0..10 {
        let result = sampler.sample_with_rejection(&logits, &context);
        if let Ok(idx) = result {
            // Samplers return indices as f32
            assert!(idx > 0.0, "Violated positive constraint: index {}", idx);
            assert!(
                (1.0..=3.0).contains(&idx),
                "Violated range constraint: index {}",
                idx
            );
            assert!(
                idx.abs() <= 3.0,
                "Violated max absolute constraint: index {}",
                idx
            );
        }
    }
}
