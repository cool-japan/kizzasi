//! Cross-crate integration tests for Neural ODE model architectures.
//!
//! These tests exercise the full public API of `NeuralOdeModel`,
//! `AugmentedNeuralOde`, and `OdeIntegrator` — including their roles as
//! `SignalPredictor` and `AutoregressiveModel` implementations.
//!
//! Unlike the in-module `#[cfg(test)]` units in `neural_ode.rs`, every test
//! here:
//!
//! - Imports only through the public crate API (no `crate::` or `super::`).
//! - Exercises real integration paths rather than hand-crafted internal state.
//! - Validates both structural and numerical properties at the trait boundary.
//!
//! # Covered scenarios
//!
//! ## OdeIntegrator
//!
//! 1. 2-D harmonic oscillator returns to initial state after one full period.
//! 2. RK4 error < Euler error on scalar exponential decay.
//! 3. RK4 convergence order: halving dt reduces error by at least 4×.
//! 4. Adaptive RK45 achieves high accuracy on scalar exponential decay.
//!
//! ## NeuralOdeModel
//!
//! 5. Single `step()` produces a finite output of the correct length.
//! 6. Ten consecutive steps remain finite and advance `current_time()`.
//! 7. `get_states` / `set_states` round-trip: restored state reproduces the
//!    same output as the uninterrupted run.
//! 8. `reset()` restores the model to its initial state (output matches a
//!    freshly constructed model).
//! 9. `model_type()`, `hidden_dim()`, and `context_window()` return the
//!    expected values for the `small` preset.
//!
//! ## AugmentedNeuralOde
//!
//! 10. `effective_dim()`, `original_dim()`, and `augment_dim()` are
//!     consistent for a 4-augment model.
//! 11. `AugmentedNeuralOde::new(_, 0)` returns `Err`.
//! 12. Single `step()` on a 2-input, 3-augment model produces a finite
//!     output of the correct length.
//! 13. Ten consecutive steps remain finite and advance `current_time()`.

use kizzasi_core::SignalPredictor;
use kizzasi_model::neural_ode::OdeDynamics;
use kizzasi_model::{
    AugmentedNeuralOde, AutoregressiveModel, ModelType, NeuralOdeConfig, NeuralOdeModel,
    OdeIntegrator, OdeSolver,
};
use scirs2_core::ndarray::{array, Array1};

// ============================================================================
// Helper utilities
// ============================================================================

/// Construct a small NeuralOdeModel (input_dim=1, hidden=64, layers=2, RK4).
fn make_small_model() -> NeuralOdeModel {
    NeuralOdeModel::small().expect("small model creation must succeed")
}

/// Assert that `output` has length `expected_len` and all values are finite and
/// at least one value is non-trivially non-zero.
fn assert_finite_output(output: &Array1<f32>, expected_len: usize) {
    assert_eq!(
        output.len(),
        expected_len,
        "output length {} does not match expected {}",
        output.len(),
        expected_len,
    );
    assert!(
        output.iter().all(|v| v.is_finite()),
        "output must be finite, got: {:?}",
        output
    );
    assert!(
        output.iter().any(|v| v.abs() > 0.0),
        "output should be non-trivially non-zero, got: {:?}",
        output
    );
}

// ============================================================================
// Test 1 – 2-D harmonic oscillator returns to origin after one full period
// ============================================================================

/// The harmonic oscillator dx/dt = [[0,1],[-1,0]] x has the analytic solution
/// x(T) = x(0) when T = 2π.  RK4 with dt = 0.01 should recover x(0) to within
/// 1% relative error, confirming both solver correctness and closedness under
/// one full revolution.
#[test]
fn test_ode_integrator_2d_harmonic_oscillator_accuracy() {
    let dt = 0.01_f32;
    let total_time = 2.0 * std::f32::consts::PI;
    let steps = (total_time / dt).round() as usize;

    // One integration step per `integrate` call so that we accumulate exactly
    // `steps` Euler-equivalent sub-steps at RK4 quality.
    let integrator = OdeIntegrator::new(OdeSolver::Rk4, dt, 1);

    let dynamics = |x: &Array1<f32>, _t: f32| -> kizzasi_model::ModelResult<Array1<f32>> {
        Ok(array![x[1], -x[0]])
    };

    let mut state = array![1.0_f32, 0.0];
    for _ in 0..steps {
        state = integrator
            .integrate(&state, 0.0, &dynamics)
            .expect("RK4 harmonic oscillator step must succeed");
    }

    // After one full period x should return to [1, 0].
    assert!(
        (state[0] - 1.0).abs() < 0.01,
        "harmonic oscillator x[0] should return to 1.0 after one period, got {}",
        state[0]
    );
    assert!(
        state[1].abs() < 0.01,
        "harmonic oscillator x[1] should return to 0.0 after one period, got {}",
        state[1]
    );
}

// ============================================================================
// Test 2 – RK4 error < Euler error on scalar exponential decay
// ============================================================================

/// For dx/dt = -x starting at x₀ = 1, the analytic solution at t = 1 is
/// e⁻¹ ≈ 0.3679.  Using dt = 0.01 and 100 steps, RK4 should be markedly more
/// accurate than Euler and achieve < 1e-5 absolute error.
#[test]
fn test_ode_rk4_vs_euler_accuracy_comparison() {
    let analytic = std::f32::consts::E.powf(-1.0);

    let run_decay = |solver: OdeSolver, dt: f32| -> f32 {
        let integrator = OdeIntegrator::new(solver, dt, 1);
        let dynamics = |x: &Array1<f32>, _: f32| -> kizzasi_model::ModelResult<Array1<f32>> {
            Ok(x.mapv(|v| -v))
        };
        let mut s = array![1.0_f32];
        let steps = (1.0_f32 / dt).round() as usize;
        for _ in 0..steps {
            s = integrator
                .integrate(&s, 0.0, &dynamics)
                .expect("integration step must succeed");
        }
        (s[0] - analytic).abs()
    };

    let euler_err = run_decay(OdeSolver::Euler, 0.01);
    let rk4_err = run_decay(OdeSolver::Rk4, 0.01);

    assert!(
        rk4_err < euler_err,
        "RK4 error {rk4_err} should be strictly less than Euler error {euler_err}"
    );
    assert!(
        rk4_err < 1e-5,
        "RK4 should be highly accurate on scalar decay, got error {rk4_err}"
    );
}

// ============================================================================
// Test 3 – RK4 convergence order: halving dt reduces error by at least 4×
// ============================================================================

/// RK4 is 4th-order accurate: halving dt reduces the global error by ~16×.
/// We test the weaker bound of ≥ 4× reduction per halving (allowing for
/// floating-point accumulation) on dx/dt = -x integrated to t = 1.0.
///
/// We use t = 1.0 (not 0.1) so that the absolute error stays well above f32's
/// machine epsilon (~1.19e-7) at the finest grid, avoiding spurious monotonicity
/// failures caused by the error floor being reached before the coarsest level.
#[test]
fn test_ode_rk4_convergence_order() {
    let analytic = std::f32::consts::E.powf(-1.0);

    let err_at_dt = |dt: f32| -> f32 {
        let integrator = OdeIntegrator::new(OdeSolver::Rk4, dt, 1);
        let dynamics = |x: &Array1<f32>, _: f32| -> kizzasi_model::ModelResult<Array1<f32>> {
            Ok(x.mapv(|v| -v))
        };
        let mut s = array![1.0_f32];
        let steps = (1.0_f32 / dt).round() as usize;
        for _ in 0..steps {
            s = integrator
                .integrate(&s, 0.0, &dynamics)
                .expect("integration step must succeed");
        }
        (s[0] - analytic).abs()
    };

    // Three grids: dt = 0.1, 0.05, 0.025 — errors should decrease monotonically.
    let err_coarse = err_at_dt(0.1);
    let err_fine = err_at_dt(0.05);
    let err_finest = err_at_dt(0.025);

    // Require at least a 4× improvement from coarse → fine (4th-order lower bound).
    assert!(
        err_fine < err_coarse,
        "halving dt should reduce RK4 error: coarse={err_coarse} fine={err_fine}"
    );
    assert!(
        err_fine < err_coarse / 4.0,
        "RK4 error ratio should show ≥ 4th-order tendency: coarse={err_coarse} fine={err_fine}"
    );

    // The finest grid must not be worse than the fine grid, unless both have
    // already hit the f32 precision floor (< 10× machine epsilon ≈ 1.2e-6).
    let f32_floor = 1.2e-6_f32;
    if err_fine > f32_floor {
        assert!(
            err_finest <= err_fine,
            "quartering dt should not increase RK4 error above f32 floor: fine={err_fine} finest={err_finest}"
        );
    }
}

// ============================================================================
// Test 4 – Adaptive RK45 achieves high accuracy on scalar decay
// ============================================================================

/// `AdaptiveRk45 { tol: 1e-5 }` on dx/dt = -x integrated over 10 steps of
/// dt = 0.1 (total t = 1) should achieve < 0.01 absolute error against the
/// analytic e⁻¹.
#[test]
fn test_ode_adaptive_rk45_accuracy() {
    let integrator = OdeIntegrator::new(OdeSolver::AdaptiveRk45 { tol: 1e-5 }, 0.1, 10);

    let dynamics =
        |x: &Array1<f32>, _: f32| -> kizzasi_model::ModelResult<Array1<f32>> { Ok(x.mapv(|v| -v)) };

    let result = integrator
        .integrate(&array![1.0_f32], 0.0, &dynamics)
        .expect("adaptive RK45 integration must succeed");

    let analytic = std::f32::consts::E.powf(-1.0);
    let error = (result[0] - analytic).abs();

    assert!(
        error < 0.01,
        "adaptive RK45 error should be < 0.01, got {error} (result={}, analytic={analytic})",
        result[0]
    );
}

// ============================================================================
// Test 5 – Single step produces finite output of correct length
// ============================================================================

/// A single `step(&[0.5])` on a freshly constructed small model must return a
/// 1-element finite, non-trivially non-zero array.
#[test]
fn test_neural_ode_model_step_finite() {
    let mut model = make_small_model();
    let input = Array1::from_vec(vec![0.5_f32]);

    let output = model.step(&input).expect("step must succeed");

    assert_finite_output(&output, 1);
}

// ============================================================================
// Test 6 – Ten consecutive steps stay finite and advance current_time()
// ============================================================================

/// The model must remain numerically stable across 10 consecutive steps and
/// the internal ODE clock must advance from 0 to a strictly positive value.
#[test]
fn test_neural_ode_model_multi_step() {
    let mut model = make_small_model();
    let input = Array1::from_vec(vec![0.3_f32]);

    for step_idx in 0..10 {
        let out = model.step(&input).expect("step must succeed");
        assert_eq!(out.len(), 1, "output length mismatch at step {step_idx}");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite output at step {step_idx}: {:?}",
            out
        );
    }

    assert!(
        model.current_time() > 0.0,
        "current_time() must advance after 10 steps, got {}",
        model.current_time()
    );
}

// ============================================================================
// Test 7 – get_states / set_states round-trip fidelity
// ============================================================================

/// Run 3 steps to accumulate hidden state, snapshot, then diverge by running
/// 2 additional steps.  Restore the snapshot, replay the same 3rd-position
/// step sequence, and verify that the output matches the original run to within
/// 1e-3 absolute error.
#[test]
fn test_neural_ode_model_state_roundtrip() {
    let mut model = make_small_model();
    let input = Array1::from_vec(vec![0.5_f32]);

    // === Accumulate 3 steps to build non-trivial hidden state ===
    for _ in 0..3 {
        model.step(&input).expect("warm-up step must succeed");
    }

    // Snapshot hidden state and ODE time.
    let snapshot = model.get_states();
    let snapshot_time = model.current_time();

    // Diverge: run 2 more steps beyond the snapshot.
    model.step(&input).expect("diverge step 1 must succeed");
    model.step(&input).expect("diverge step 2 must succeed");

    // Step once more from the diverged position (this is output_a).
    let output_a = model.step(&input).expect("step after diverge must succeed");

    // Restore the 3-step snapshot.
    model.set_states(snapshot).expect("set_states must succeed");
    model.set_time(snapshot_time);

    // Replay the exact same diverge steps from the snapshot position.
    model
        .step(&input)
        .expect("replay diverge step 1 must succeed");
    model
        .step(&input)
        .expect("replay diverge step 2 must succeed");

    // This step should reproduce output_a.
    let output_b = model.step(&input).expect("step after restore must succeed");

    let diff: f32 = output_a
        .iter()
        .zip(output_b.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff < 1e-3,
        "restored state must produce the same output: diff={diff} output_a={output_a:?} output_b={output_b:?}"
    );
}

// ============================================================================
// Test 8 – reset() restores the model to its initial state
// ============================================================================

/// After running some steps and calling `reset()`, the next step must match
/// the output of a freshly constructed model given the same input.
#[test]
fn test_neural_ode_model_reset() {
    let mut model = make_small_model();
    let input = Array1::from_vec(vec![0.5_f32]);

    // Run one step to change state.
    let _out_before = model.step(&input).expect("pre-reset step must succeed");

    model.reset();

    let out_after_reset = model.step(&input).expect("step after reset must succeed");

    // A fresh model should produce the exact same output.
    let mut fresh = make_small_model();
    let out_fresh = fresh.step(&input).expect("fresh model step must succeed");

    let diff: f32 = out_after_reset
        .iter()
        .zip(out_fresh.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff < 1e-5,
        "reset model must behave identically to a fresh model: diff={diff}"
    );
}

// ============================================================================
// Test 9 – model_type(), hidden_dim(), context_window() for small preset
// ============================================================================

/// The `small()` preset must report `ModelType::NeuralOde`, `hidden_dim == 64`,
/// and a positive `context_window`.
#[test]
fn test_neural_ode_model_type_and_dims() {
    let model = make_small_model();

    assert_eq!(
        model.model_type(),
        ModelType::NeuralOde,
        "model_type must be ModelType::NeuralOde"
    );
    assert_eq!(
        model.hidden_dim(),
        64,
        "small() preset hidden_dim should be 64"
    );
    assert!(
        model.context_window() > 0,
        "context_window must be positive, got {}",
        model.context_window()
    );
}

// ============================================================================
// Test 10 – AugmentedNeuralOde dimension accessors are consistent
// ============================================================================

/// `AugmentedNeuralOde::new(small(1), 4)` must report original_dim = 64 (the
/// hidden_dim of the small preset), augment_dim = 4, and
/// effective_dim = original_dim + augment_dim = 68.
#[test]
fn test_augmented_ode_dimensions() {
    let config = NeuralOdeConfig::small(1);
    let original_hidden = config.hidden_dim; // 64

    let aug = AugmentedNeuralOde::new(config, 4).expect("augmented ODE creation must succeed");

    assert_eq!(
        aug.original_dim(),
        original_hidden,
        "original_dim mismatch: expected {original_hidden}, got {}",
        aug.original_dim()
    );
    assert_eq!(
        aug.augment_dim(),
        4,
        "augment_dim mismatch: expected 4, got {}",
        aug.augment_dim()
    );
    assert_eq!(
        aug.effective_dim(),
        original_hidden + 4,
        "effective_dim must equal original_dim + augment_dim: expected {}, got {}",
        original_hidden + 4,
        aug.effective_dim()
    );
}

// ============================================================================
// Test 11 – AugmentedNeuralOde::new with augment_dim=0 returns Err
// ============================================================================

/// The augmented model requires at least one extra dimension; zero must be
/// rejected with `Err` rather than producing a degenerate model.
#[test]
fn test_augmented_ode_zero_augment_fails() {
    let config = NeuralOdeConfig::small(1);
    let result = AugmentedNeuralOde::new(config, 0);

    assert!(
        result.is_err(),
        "AugmentedNeuralOde::new with augment_dim=0 must return Err"
    );
}

// ============================================================================
// Test 12 – Single step on 2-input, 3-augment model produces finite output
// ============================================================================

/// The output dimension of the augmented model's `step` is determined by
/// `input_dim`, not by the augmented hidden space.  A 2-input, 3-augment model
/// must return a 2-element finite array.
#[test]
fn test_augmented_ode_step_finite() {
    let config = NeuralOdeConfig::small(2); // input_dim = 2

    let mut aug = AugmentedNeuralOde::new(config, 3).expect("augmented ODE creation must succeed");

    let input = Array1::from_vec(vec![0.5_f32, -0.3]);
    let output = aug.step(&input).expect("augmented step must succeed");

    assert_eq!(
        output.len(),
        2,
        "output should match input_dim=2, got {}",
        output.len()
    );
    assert!(
        output.iter().all(|v| v.is_finite()),
        "augmented step output must be finite, got: {:?}",
        output
    );
}

// ============================================================================
// Test 13 – Ten consecutive augmented steps stay finite and advance time
// ============================================================================

/// Validates that the augmented model is numerically stable for at least 10
/// steps with a 1-input, 2-augment configuration.
#[test]
fn test_augmented_ode_multi_step() {
    let config = NeuralOdeConfig::small(1);
    let mut aug = AugmentedNeuralOde::new(config, 2).expect("augmented ODE creation must succeed");

    let input = Array1::from_vec(vec![0.5_f32]);

    for step_idx in 0..10 {
        let out = aug.step(&input).expect("augmented step must succeed");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "augmented model output must be finite at step {step_idx}: {:?}",
            out
        );
    }

    assert!(
        aug.current_time() > 0.0,
        "augmented model current_time() must advance after 10 steps, got {}",
        aug.current_time()
    );
}

// ============================================================================
// Test 14 – OdeDynamics forward is accessible via neural_ode module path
// ============================================================================

/// Confirms that `OdeDynamics` is accessible via `kizzasi_model::neural_ode`
/// (not re-exported at crate root) and that a forward pass on a zero state
/// at t=0 returns a finite hidden vector.
#[test]
fn test_ode_dynamics_accessible_via_module_path() {
    let dynamics = OdeDynamics::new(8, 2).expect("OdeDynamics creation must succeed");
    let x = Array1::<f32>::zeros(8);

    let result = dynamics
        .forward(&x, 0.0)
        .expect("OdeDynamics::forward must succeed");

    assert_eq!(
        result.len(),
        8,
        "dynamics output length should equal hidden_dim"
    );
    assert!(
        result.iter().all(|v| v.is_finite()),
        "OdeDynamics::forward must return finite values, got: {:?}",
        result
    );
}
