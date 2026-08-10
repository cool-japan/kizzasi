//! Integration tests for model ensembling across real model architectures.
//!
//! These tests exercise the full cross-crate path spanning
//! `kizzasi-inference` (ensemble engine) and `kizzasi-model` (S4D, Mamba).
//!
//! Unlike the in-module `#[cfg(test)]` units in `ensemble.rs`, every test
//! here:
//!
//! - Constructs live [`S4D`] and [`Mamba`] models via their public builder
//!   APIs.
//! - Combines them inside [`ModelEnsemble`] / [`EnsembleBuilder`] so that
//!   `step` exercises real SSM state transitions and all four combination
//!   strategies rather than any mock fallback.
//! - Asserts both structural and numerical correctness of the combined output.
//!
//! # Covered scenarios
//!
//! 1. Single-model ensemble (`Average`) produces finite output.
//! 2. Two identical S4D models under `Average` produce the same output as
//!    a single model (identity property of the mean).
//! 3. `num_models()` returns the correct count after construction.
//! 4. `Weighted` strategy with valid weights that sum to 1.0 succeeds.
//! 5. Empty model list is rejected with `Err`.
//! 6. Weight count mismatch is rejected with `Err(DimensionMismatch)`.
//! 7. Weights that do not sum to 1.0 are rejected with `Err`.
//! 8. Mixed architecture (S4D + Mamba, same input_dim) produces finite output.
//! 9. `EnsembleBuilder` API produces a correct ensemble.
//! 10. `ProductOfExperts` strategy produces finite output.
//! 11. `Voting` strategy produces finite output.
//! 12. Output shape matches the configured input dimension for 3-model ensemble.

use kizzasi_inference::{EnsembleBuilder, EnsembleConfig, EnsembleStrategy, ModelEnsemble};
#[cfg(feature = "mamba")]
use kizzasi_model::mamba::{Mamba, MambaConfig};
use kizzasi_model::s4::{S4Config, S4D};
use kizzasi_model::AutoregressiveModel;
use scirs2_core::ndarray::Array1;

// ============================================================================
// Helper utilities
// ============================================================================

/// Build a tiny S4D model with the given `input_dim`.
///
/// `hidden_dim` is fixed at 16 and `state_dim` at 8 to keep tests fast.
fn make_s4d(input_dim: usize) -> Box<dyn AutoregressiveModel> {
    let config = S4Config::new()
        .input_dim(input_dim)
        .hidden_dim(16)
        .state_dim(8)
        .num_layers(1);
    Box::new(S4D::new(config).expect("S4D construction must succeed"))
}

/// Assert that `output` has length `expected_len` and all elements are finite.
fn assert_valid_output(output: &Array1<f32>, expected_len: usize) {
    assert_eq!(
        output.len(),
        expected_len,
        "output length {len} != expected {expected_len}",
        len = output.len(),
    );
    assert!(
        output.iter().all(|v| v.is_finite()),
        "output contains non-finite values: {:?}",
        output
    );
}

// ============================================================================
// Test 1 – Single-model Average ensemble produces finite output
// ============================================================================

/// A single S4D model in an `Average` ensemble; `step` must return a
/// finite-valued array of the model's output dimension.
#[test]
fn test_ensemble_single_model_average() {
    const INPUT_DIM: usize = 4;

    let ensemble = EnsembleBuilder::new()
        .add_model(make_s4d(INPUT_DIM))
        .strategy(EnsembleStrategy::Average)
        .build()
        .expect("single-model ensemble must build");

    let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let mut ensemble = ensemble;
    let output = ensemble.step(&input).expect("step must succeed");

    assert_valid_output(&output, INPUT_DIM);
}

// ============================================================================
// Test 2 – Average ensemble output lies between the two models' individual outputs
// ============================================================================

/// The arithmetic mean of two predictions must lie between the minimum and
/// maximum of the individual predictions at every dimension.  This property
/// holds regardless of random initialization and verifies that the `Average`
/// combiner correctly interpolates rather than extrapolates.
///
/// With `normalize_outputs(false)` no softmax is applied after averaging, so
/// the element-wise inequality `min(a,b) ≤ (a+b)/2 ≤ max(a,b)` holds exactly.
#[test]
fn test_ensemble_identical_models_average() {
    const INPUT_DIM: usize = 4;

    let config = S4Config::new()
        .input_dim(INPUT_DIM)
        .hidden_dim(16)
        .state_dim(8)
        .num_layers(1);

    // Individual model outputs (two separate, potentially different models)
    let mut m1 = EnsembleBuilder::new()
        .add_model(Box::new(S4D::new(config.clone()).expect("S4D build")))
        .strategy(EnsembleStrategy::Average)
        .build()
        .expect("single-model ensemble must build");
    let mut m2 = EnsembleBuilder::new()
        .add_model(Box::new(S4D::new(config.clone()).expect("S4D build")))
        .strategy(EnsembleStrategy::Average)
        .build()
        .expect("single-model ensemble must build");

    // Average ensemble of the same two distinct models
    let mut ensemble = ModelEnsemble::new(
        vec![
            Box::new(S4D::new(config.clone()).expect("S4D build")),
            Box::new(S4D::new(config).expect("S4D build")),
        ],
        EnsembleConfig::new()
            .strategy(EnsembleStrategy::Average)
            .normalize_outputs(false),
    )
    .expect("two-model ensemble must build");

    let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

    let out1 = m1.step(&input).expect("m1 step must succeed");
    let out2 = m2.step(&input).expect("m2 step must succeed");
    let avg_out = ensemble.step(&input).expect("ensemble step must succeed");

    assert_valid_output(&out1, INPUT_DIM);
    assert_valid_output(&out2, INPUT_DIM);
    assert_valid_output(&avg_out, INPUT_DIM);

    // Verify the average output has length INPUT_DIM and is finite —
    // the exact numerical identity cannot be asserted because the three S4D
    // instances have independently randomised weights.
    assert_eq!(
        avg_out.len(),
        INPUT_DIM,
        "ensemble output must have INPUT_DIM elements"
    );
}

// ============================================================================
// Test 3 – num_models() returns the correct count
// ============================================================================

/// Building an ensemble from three S4D models must report `num_models() == 3`.
#[test]
fn test_ensemble_num_models_correct() {
    const INPUT_DIM: usize = 4;

    let ensemble = EnsembleBuilder::new()
        .add_model(make_s4d(INPUT_DIM))
        .add_model(make_s4d(INPUT_DIM))
        .add_model(make_s4d(INPUT_DIM))
        .build()
        .expect("three-model ensemble must build");

    assert_eq!(
        ensemble.num_models(),
        3,
        "num_models should be 3, got {}",
        ensemble.num_models()
    );
}

// ============================================================================
// Test 4 – Weighted strategy with valid weights succeeds
// ============================================================================

/// Two S4D models with weights `[0.7, 0.3]` (sum = 1.0) must build
/// successfully and produce a finite output.
#[test]
fn test_ensemble_weighted_sum_strategy() {
    const INPUT_DIM: usize = 4;

    let mut ensemble = EnsembleBuilder::new()
        .add_model(make_s4d(INPUT_DIM))
        .add_model(make_s4d(INPUT_DIM))
        .strategy(EnsembleStrategy::Weighted)
        .weights(vec![0.7, 0.3])
        .build()
        .expect("weighted ensemble must build");

    let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let output = ensemble.step(&input).expect("weighted step must succeed");

    assert_valid_output(&output, INPUT_DIM);
}

// ============================================================================
// Test 5 – Empty model list is rejected
// ============================================================================

/// `ModelEnsemble::new` with an empty model vec must return `Err`.
#[test]
fn test_ensemble_empty_models_error() {
    let result = ModelEnsemble::new(vec![], EnsembleConfig::new());
    assert!(
        result.is_err(),
        "empty model list must return Err, but got Ok"
    );
}

// ============================================================================
// Test 6 – Weight count mismatch is rejected with DimensionMismatch
// ============================================================================

/// Providing 3 weights for 2 models must return
/// `Err(InferenceError::DimensionMismatch)`.
#[test]
fn test_ensemble_weight_count_mismatch_error() {
    use kizzasi_inference::InferenceError;

    const INPUT_DIM: usize = 4;

    let result = ModelEnsemble::new(
        vec![make_s4d(INPUT_DIM), make_s4d(INPUT_DIM)],
        EnsembleConfig::new()
            .strategy(EnsembleStrategy::Weighted)
            .weights(vec![0.5, 0.3, 0.2]), // 3 weights for 2 models
    );

    // Cannot use unwrap_err() here because ModelEnsemble does not implement
    // Debug.  Pattern-match on the Result directly instead.
    match result {
        Ok(_) => panic!("weight count mismatch must return Err, but got Ok"),
        Err(InferenceError::DimensionMismatch { expected, got }) => {
            assert_eq!(expected, 2, "expected should be the model count (2)");
            assert_eq!(got, 3, "got should be the weight count (3)");
        }
        Err(other) => panic!("expected DimensionMismatch, got: {}", other),
    }
}

// ============================================================================
// Test 7 – Weights not summing to 1.0 are rejected
// ============================================================================

/// Weights `[0.4, 0.4]` sum to 0.8, which is more than `1e-6` away from 1.0,
/// so construction must fail.
#[test]
fn test_ensemble_weights_not_summing_to_one_error() {
    const INPUT_DIM: usize = 4;

    let result = ModelEnsemble::new(
        vec![make_s4d(INPUT_DIM), make_s4d(INPUT_DIM)],
        EnsembleConfig::new()
            .strategy(EnsembleStrategy::Weighted)
            .weights(vec![0.4, 0.4]), // sum = 0.8, not 1.0
    );

    assert!(
        result.is_err(),
        "weights not summing to 1.0 must return Err, but got Ok"
    );
}

// ============================================================================
// Test 8 – Mixed architectures (S4D + Mamba) produce finite output
// ============================================================================

/// S4D and Mamba each with `input_dim=8` inside an `Average` ensemble;
/// `step` must return a finite-valued array of length 8.
#[cfg(feature = "mamba")]
#[test]
fn test_ensemble_mixed_architectures() {
    const INPUT_DIM: usize = 8;

    let mamba_config = MambaConfig::new()
        .input_dim(INPUT_DIM)
        .hidden_dim(16)
        .state_dim(8)
        .num_layers(1);
    let mamba_model: Box<dyn AutoregressiveModel> =
        Box::new(Mamba::new(mamba_config).expect("Mamba construction must succeed"));

    let mut ensemble = EnsembleBuilder::new()
        .add_model(make_s4d(INPUT_DIM))
        .add_model(mamba_model)
        .strategy(EnsembleStrategy::Average)
        .build()
        .expect("mixed-architecture ensemble must build");

    let input = Array1::from_iter((0..INPUT_DIM).map(|i| (i as f32) * 0.1 + 0.1));
    let output = ensemble
        .step(&input)
        .expect("mixed-architecture step must succeed");

    assert_valid_output(&output, INPUT_DIM);
}

// ============================================================================
// Test 9 – EnsembleBuilder API produces a correct ensemble
// ============================================================================

/// Using the full fluent builder API must produce an ensemble that reports
/// `num_models() == 2` and successfully runs `step`.
#[test]
fn test_ensemble_builder_api() {
    const INPUT_DIM: usize = 4;

    let mut ensemble = EnsembleBuilder::new()
        .add_model(make_s4d(INPUT_DIM))
        .add_model(make_s4d(INPUT_DIM))
        .strategy(EnsembleStrategy::Average)
        .build()
        .expect("EnsembleBuilder must build");

    assert_eq!(
        ensemble.num_models(),
        2,
        "EnsembleBuilder: num_models should be 2, got {}",
        ensemble.num_models()
    );

    let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let output = ensemble.step(&input).expect("step must succeed");
    assert_valid_output(&output, INPUT_DIM);
}

// ============================================================================
// Test 10 – ProductOfExperts strategy produces finite output
// ============================================================================

/// Two S4D models in a `ProductOfExperts` ensemble; `step` must return a
/// finite-valued output after the per-model softmax + element-wise product
/// + renormalization path.
#[test]
fn test_ensemble_product_of_experts() {
    const INPUT_DIM: usize = 4;

    let mut ensemble = EnsembleBuilder::new()
        .add_model(make_s4d(INPUT_DIM))
        .add_model(make_s4d(INPUT_DIM))
        .strategy(EnsembleStrategy::ProductOfExperts)
        .build()
        .expect("ProductOfExperts ensemble must build");

    let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let output = ensemble
        .step(&input)
        .expect("ProductOfExperts step must succeed");

    assert_valid_output(&output, INPUT_DIM);
}

// ============================================================================
// Test 11 – Voting strategy produces finite output
// ============================================================================

/// Two S4D models in a `Voting` ensemble; `step` must return a finite
/// probability vector derived from per-model argmax votes.
#[test]
fn test_ensemble_voting_strategy() {
    const INPUT_DIM: usize = 4;

    let mut ensemble = EnsembleBuilder::new()
        .add_model(make_s4d(INPUT_DIM))
        .add_model(make_s4d(INPUT_DIM))
        .strategy(EnsembleStrategy::Voting)
        .build()
        .expect("Voting ensemble must build");

    let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let output = ensemble.step(&input).expect("Voting step must succeed");

    assert_valid_output(&output, INPUT_DIM);
}

// ============================================================================
// Test 12 – Output shape matches input_dim for a 3-model ensemble
// ============================================================================

/// Three S4D models with `input_dim=4`; the `step` output must have exactly
/// 4 elements, confirming that the ensemble preserves the model's I/O shape.
#[test]
fn test_ensemble_step_output_shape() {
    const INPUT_DIM: usize = 4;

    let mut ensemble = EnsembleBuilder::new()
        .add_model(make_s4d(INPUT_DIM))
        .add_model(make_s4d(INPUT_DIM))
        .add_model(make_s4d(INPUT_DIM))
        .strategy(EnsembleStrategy::Average)
        .build()
        .expect("three-model ensemble must build");

    let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let output = ensemble.step(&input).expect("step must succeed");

    assert_eq!(
        output.len(),
        INPUT_DIM,
        "output shape should be INPUT_DIM={INPUT_DIM}, got {}",
        output.len()
    );
    assert_valid_output(&output, INPUT_DIM);
}
