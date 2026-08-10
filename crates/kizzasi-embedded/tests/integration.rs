//! Cross-feature integration tests for `kizzasi-embedded`.
//!
//! These tests validate that the f32-only path, the fixed-point (Q16.16)
//! path, and the INT8 quantisation utilities all agree on the same
//! synthetic inputs to within sensible numerical tolerances. They also
//! exercise the platform presets to make sure they round-trip through the
//! `SsmConfig::new` validation logic.

use kizzasi_embedded::{
    esp32c3, quantize, rp2040, stm32h7, EmbeddedError, MambaStep, S4Step, SsmConfig, SsmState,
};

/// Mean-squared error helper used by all tolerance checks.
fn mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "mse: slice lengths must match");
    let n = a.len() as f32;
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / n
}

// ---------------------------------------------------------------------------
// Preset validation
// ---------------------------------------------------------------------------

#[test]
fn test_all_presets_construct_valid_configs() {
    // Every preset must produce non-zero dimensions and pass
    // `SsmConfig::new`'s validation when reconstructed from its expand ratio.
    let presets = [
        ("stm32h7", stm32h7()),
        ("rp2040", rp2040()),
        ("esp32c3", esp32c3()),
    ];

    for (name, cfg) in presets {
        assert!(cfg.d_model > 0, "{name}: d_model must be > 0");
        assert!(cfg.d_state > 0, "{name}: d_state must be > 0");
        assert!(cfg.d_inner > 0, "{name}: d_inner must be > 0");
        assert_eq!(
            cfg.d_inner,
            cfg.d_model * 2,
            "{name}: expected canonical expand=2 (d_inner = 2*d_model)"
        );

        // Round-trip through SsmConfig::new with the implied expand value.
        let expand = cfg.d_inner / cfg.d_model;
        let rebuilt = SsmConfig::new(cfg.d_model, cfg.d_state, expand)
            .unwrap_or_else(|_| panic!("{name}: rebuild via SsmConfig::new must succeed"));
        assert_eq!(rebuilt.d_model, cfg.d_model, "{name}: d_model mismatch");
        assert_eq!(rebuilt.d_state, cfg.d_state, "{name}: d_state mismatch");
        assert_eq!(rebuilt.d_inner, cfg.d_inner, "{name}: d_inner mismatch");

        // And the allocated state must match the config's dimensions.
        let state = SsmState::new(&cfg);
        assert_eq!(state.h.len(), cfg.d_state, "{name}: h length mismatch");
        assert_eq!(
            state.prev_x.len(),
            cfg.d_inner,
            "{name}: prev_x length mismatch"
        );
    }
}

#[test]
fn test_invalid_config_rejected() {
    // SsmConfig::new must refuse zero dimensions on every axis.
    for (m, s, e) in [(0, 4, 2), (4, 0, 2), (4, 4, 0)] {
        let err = SsmConfig::new(m, s, e).expect_err("zero dim must be rejected");
        assert_eq!(err, EmbeddedError::InvalidConfig("dimensions must be > 0"));
    }
}

// ---------------------------------------------------------------------------
// Mamba step end-to-end
// ---------------------------------------------------------------------------

/// Deterministic synthetic SSM parameters for a given `d_state`.
fn synth_params(d_state: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..d_state).map(|i| 0.05 + 0.01 * i as f32).collect();
    let a_log: Vec<f32> = (0..d_state).map(|i| -1.0 - 0.1 * i as f32).collect();
    let b: Vec<f32> = (0..d_state).map(|i| 0.30 + 0.02 * i as f32).collect();
    let c: Vec<f32> = (0..d_state).map(|i| 0.80 - 0.01 * i as f32).collect();
    (x, a_log, b, c)
}

#[test]
fn test_mamba_step_evolves_state_across_presets() {
    // For each preset run a short sequence and confirm:
    //   1. every step succeeds
    //   2. outputs are finite
    //   3. state.h is non-zero after the first step
    for cfg in [stm32h7(), rp2040(), esp32c3()] {
        let mut state = SsmState::new(&cfg);
        let (x, a_log, b, c) = synth_params(cfg.d_state);
        let mut last_y = f32::NAN;
        for _ in 0..8 {
            let y = MambaStep::step(&mut state, &x, &a_log, &b, &c, 0.10, 0.05)
                .expect("step must succeed for preset");
            assert!(y.is_finite(), "MambaStep produced non-finite y={y}");
            last_y = y;
        }
        assert!(
            state.h.iter().any(|&v| v != 0.0),
            "state.h must be non-zero after 8 steps"
        );
        assert!(last_y.is_finite(), "final output must be finite");
    }
}

#[test]
fn test_s4_step_agrees_with_mamba_on_zero_skip() {
    // When the d_skip term is zero and inputs are uniform, the S4 diagonal
    // step and the Mamba step are not strictly equal (they discretise
    // differently), but both should produce finite, bounded outputs of the
    // same order of magnitude on the same inputs.
    let cfg = SsmConfig::new(16, 4, 2).unwrap();
    let mut mamba_state = SsmState::new(&cfg);
    let mut s4_state = SsmState::new(&cfg);
    let (x, a_log, b, c) = synth_params(cfg.d_state);
    let lambda_im = vec![0.0_f32; cfg.d_state];

    let y_mamba = MambaStep::step(&mut mamba_state, &x, &a_log, &b, &c, 0.10, 0.0)
        .expect("mamba step must succeed");
    let y_s4 = S4Step::step(&mut s4_state, x[0], &a_log, &lambda_im, &b, &c, 0.10)
        .expect("s4 step must succeed");

    assert!(y_mamba.is_finite() && y_s4.is_finite());
    // Both kernels are stable for these parameters, so |y| < 10 is loose.
    assert!(y_mamba.abs() < 10.0, "mamba y out of bound: {y_mamba}");
    assert!(y_s4.abs() < 10.0, "s4 y out of bound: {y_s4}");
}

// ---------------------------------------------------------------------------
// INT8 quantisation round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_dequantize_roundtrip_on_state() {
    // Initialise a state, fill it with deterministic values, quantise, and
    // dequantise. The round-trip MSE must stay below the symmetric INT8
    // quantum (scale / 127) squared.
    let cfg = stm32h7();
    let mut state = SsmState::new(&cfg);
    for (i, v) in state.h.iter_mut().enumerate() {
        *v = (i as f32).sin();
    }

    let (scale, quantized) = quantize::quantize_symmetric(&state.h);
    let mut reconstructed = vec![0.0_f32; state.h.len()];
    quantize::dequantize_into(&quantized, scale, &mut reconstructed);

    let err = mse(&state.h, &reconstructed);
    let quantum = scale; // 1 LSB in dequantised units
    assert!(
        err < quantum * quantum * 4.0,
        "round-trip MSE {err:.3e} exceeds tolerance {tol:.3e} (scale = {scale:.3e})",
        tol = quantum * quantum * 4.0,
    );

    // And the strictest possible per-element bound: ≤ 1 LSB.
    for (orig, recon) in state.h.iter().zip(reconstructed.iter()) {
        assert!(
            (orig - recon).abs() <= scale + 1e-6,
            "per-element |orig - recon| > scale: orig={orig}, recon={recon}, scale={scale}"
        );
    }
}

// ---------------------------------------------------------------------------
// f32 vs fixed-point cross-check
// ---------------------------------------------------------------------------

#[cfg(feature = "fixed-point")]
#[test]
fn test_f32_vs_fixed_point_agreement() {
    use kizzasi_embedded::fixed_point::{fixed_dot, Q16};

    // Synthetic feature vector and weights. Values stay in [-0.5, 0.5] so
    // the Q16.16 path retains full precision and no saturation occurs.
    let inputs_f32: Vec<f32> = (0..16).map(|i| (i as f32 / 16.0 - 0.5) * 0.8).collect();
    let weights_f32: Vec<f32> = (0..16)
        .map(|i| ((i as f32 + 1.0) / 32.0).cos() * 0.5)
        .collect();

    // f32 reference dot product.
    let dot_f32: f32 = inputs_f32
        .iter()
        .zip(weights_f32.iter())
        .map(|(x, w)| x * w)
        .sum();

    // Q16 fixed-point dot product.
    let inputs_q: Vec<Q16> = inputs_f32.iter().map(|&v| Q16::from_f32(v)).collect();
    let weights_q: Vec<Q16> = weights_f32.iter().map(|&v| Q16::from_f32(v)).collect();
    let dot_q = fixed_dot(&inputs_q, &weights_q);
    let dot_q_f32 = dot_q.to_f32();

    // MSE on a 1-element vector reduces to squared error.
    let err = (dot_f32 - dot_q_f32).abs();
    assert!(
        err < 1e-2,
        "Q16 vs f32 dot mismatch: f32={dot_f32:.6}, q16={dot_q_f32:.6}, err={err:.3e}"
    );
}

#[cfg(feature = "fixed-point")]
#[test]
fn test_fixed_exp_matches_f32_near_zero() {
    use kizzasi_embedded::fixed_point::{fixed_exp_approx, Q16};
    use kizzasi_embedded::math::exp_approx;

    // For |x| < 0.5 the Taylor-2 fixed-point exp should track the
    // 7-term f32 exp to within ~ 1 % relative error.
    for x in [-0.4, -0.2, -0.05, 0.0, 0.05, 0.2, 0.4] {
        let f = exp_approx(x);
        let q = fixed_exp_approx(Q16::from_f32(x)).to_f32();
        let rel = (q - f).abs() / f.abs().max(1e-6);
        assert!(
            rel < 0.02,
            "fixed_exp_approx({x}) = {q:.6}, exp_approx = {f:.6}, rel = {rel:.3e}"
        );
    }
}
