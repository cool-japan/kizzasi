//! Cross-feature integration tests for `kizzasi-embedded`.
//!
//! These tests validate that the f32-only path, the fixed-point (Q16.16)
//! path, and the INT8 quantisation utilities all agree on the same
//! synthetic inputs to within sensible numerical tolerances. They also
//! exercise the platform presets to make sure they round-trip through the
//! `SsmConfig::new` validation logic.

use kizzasi_embedded::{
    esp32c3, quantize, rp2040, stm32h7, DepthwiseConv1d, EmbeddedError, MambaStep, S4State, S4Step,
    SsmConfig, SsmState,
};

/// Mean-squared error helper used by all tolerance checks.
fn mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "mse: slice lengths must match");
    let n = a.len() as f32;
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / n
}

/// HiPPO checkpoint convention: `a_log[n] = ln(n + 1)`, i.e. non-negative.
/// The effective state matrix is `A = -exp(a_log)`, matching
/// `kizzasi-model::mamba` and `kizzasi-core::mamba2`.
fn hippo_a_log(d_state: usize) -> Vec<f32> {
    (0..d_state).map(|n| ((n + 1) as f32).ln()).collect()
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
        assert!(cfg.validate().is_ok(), "{name}: validate() must pass");
        assert_eq!(
            cfg.d_inner,
            cfg.d_model * 2,
            "{name}: expected canonical expand=2 (d_inner = 2*d_model)"
        );

        // Round-trip through SsmConfig::new with the implied expand value.
        let expand = cfg.d_inner / cfg.d_model;
        let rebuilt = SsmConfig::new(cfg.d_model, cfg.d_state, expand)
            .unwrap_or_else(|_| panic!("{name}: rebuild via SsmConfig::new must succeed"));
        assert_eq!(rebuilt, cfg, "{name}: rebuilt config must match the preset");

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
    // And an overflowing d_model * expand.
    assert_eq!(
        SsmConfig::new(usize::MAX, 4, 2),
        Err(EmbeddedError::InvalidConfig("d_model * expand overflows"))
    );
}

// ---------------------------------------------------------------------------
// Mamba step end-to-end
// ---------------------------------------------------------------------------

/// Deterministic synthetic SSM parameters for a given `d_state`.
fn synth_params(d_state: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..d_state).map(|i| 0.05 + 0.01 * i as f32).collect();
    let a_log = hippo_a_log(d_state);
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
fn test_mamba_stays_bounded_over_a_long_sequence_with_checkpoint_weights() {
    // Regression test for the `a_log` convention. Real checkpoints store
    // log(-A) with non-negative values (HiPPO: ln(1..=d_state)). Under the
    // old `A_bar = exp(delta_sp * a_log)` those produced A_bar >= 1 and the
    // state grew without bound over a sequence, with no error raised.
    for cfg in [stm32h7(), rp2040(), esp32c3()] {
        let mut state = SsmState::new(&cfg);
        let (x, a_log, b, c) = synth_params(cfg.d_state);
        let mut peak = 0.0_f32;
        for step in 0..2_000 {
            let y = MambaStep::step(&mut state, &x, &a_log, &b, &c, 0.10, 0.0)
                .expect("step must succeed");
            assert!(y.is_finite(), "step {step}: output diverged to {y}");
            peak = peak.max(y.abs());
        }
        assert!(
            peak < 10.0,
            "output must stay bounded over 2000 steps, peak = {peak}"
        );
        for (i, &h) in state.h.iter().enumerate() {
            assert!(
                h.is_finite() && h.abs() < 10.0,
                "state.h[{i}] = {h} must stay bounded"
            );
        }
    }
}

#[test]
fn test_mamba_slice_and_state_forms_agree() {
    let cfg = rp2040();
    let (x, a_log, b, c) = synth_params(cfg.d_state);
    let mut state = SsmState::new(&cfg);
    let mut h = vec![0.0_f32; cfg.d_state];

    for _ in 0..4 {
        let y_state = MambaStep::step(&mut state, &x, &a_log, &b, &c, 0.1, 0.05)
            .expect("state form must succeed");
        let y_slice = MambaStep::step_slice(&mut h, &x, &a_log, &b, &c, 0.1, 0.05)
            .expect("slice form must succeed");
        assert_eq!(
            y_state, y_slice,
            "the allocator-free slice form must be bit-identical"
        );
    }
    assert_eq!(state.h, h);
}

#[test]
fn test_depthwise_conv_then_mamba_pipeline() {
    // `prev_x` is no longer dead memory: it is the convolution history of the
    // token-shift stage that feeds the selective scan.
    let cfg = SsmConfig::new(4, 4, 2).expect("valid config"); // d_inner = 8
    let mut state = SsmState::new(&cfg);
    let w_curr = vec![0.8_f32; cfg.d_inner];
    let w_prev = vec![0.2_f32; cfg.d_inner];
    let mut conv_out = vec![0.0_f32; cfg.d_inner];

    let (_, a_log, b, c) = synth_params(cfg.d_state);
    let mut outputs = Vec::new();
    for t in 0..6 {
        let frame: Vec<f32> = (0..cfg.d_inner)
            .map(|i| 0.1 * (t as f32 + 1.0) + 0.01 * i as f32)
            .collect();
        DepthwiseConv1d::step(&mut state, &frame, &w_curr, &w_prev, &[], &mut conv_out)
            .expect("conv step must succeed");
        // Feed the first d_state channels of the convolved frame into the SSM.
        let ssm_in = &conv_out[..cfg.d_state];
        let y = MambaStep::step(&mut state, ssm_in, &a_log, &b, &c, 0.1, 0.0)
            .expect("ssm step must succeed");
        outputs.push(y);
    }
    assert!(outputs.iter().all(|v| v.is_finite()));
    assert!(
        state.prev_x.iter().any(|&v| v != 0.0),
        "prev_x must carry the convolution history"
    );
}

#[test]
fn test_s4_step_uses_the_imaginary_pole() {
    // `lambda_im` used to be validated and discarded, so no oscillatory mode
    // was reachable. It must now change the trajectory.
    let cfg = SsmConfig::new(16, 4, 2).expect("valid config");
    let mut real_only = S4State::new(&cfg);
    let mut oscillating = S4State::new(&cfg);
    let lambda_re = vec![-0.2_f32; cfg.d_state];
    let zero_im = vec![0.0_f32; cfg.d_state];
    let osc_im = vec![3.0_f32; cfg.d_state];
    let b = vec![1.0_f32; cfg.d_state];
    let c = vec![0.5_f32; cfg.d_state];

    let mut any_difference = false;
    let mut went_negative = false;
    for t in 0..16 {
        let x = if t == 0 { 1.0 } else { 0.0 };
        let y_real = S4Step::step(&mut real_only, x, &lambda_re, &zero_im, &b, &c, 0.3)
            .expect("s4 step must succeed");
        let y_osc = S4Step::step(&mut oscillating, x, &lambda_re, &osc_im, &b, &c, 0.3)
            .expect("s4 step must succeed");
        assert!(y_real.is_finite() && y_osc.is_finite());
        if (y_real - y_osc).abs() > 1e-3 {
            any_difference = true;
        }
        if y_osc < 0.0 {
            went_negative = true;
        }
    }
    assert!(any_difference, "lambda_im must influence the output");
    assert!(
        went_negative,
        "a complex pole must produce ringing, which a real-only decay cannot"
    );
}

#[test]
fn test_s4_and_mamba_are_both_bounded() {
    // Both kernels are stable for these parameters and produce finite,
    // same-order outputs (they discretise differently, so they are not equal).
    let cfg = SsmConfig::new(16, 4, 2).expect("valid config");
    let mut mamba_state = SsmState::new(&cfg);
    let mut s4_state = S4State::new(&cfg);
    let (x, a_log, b, c) = synth_params(cfg.d_state);
    let lambda_re: Vec<f32> = a_log.iter().map(|v| -v.exp()).collect();
    let lambda_im = vec![0.5_f32; cfg.d_state];

    let y_mamba = MambaStep::step(&mut mamba_state, &x, &a_log, &b, &c, 0.10, 0.0)
        .expect("mamba step must succeed");
    let y_s4 = S4Step::step(&mut s4_state, x[0], &lambda_re, &lambda_im, &b, &c, 0.10)
        .expect("s4 step must succeed");

    assert!(y_mamba.is_finite() && y_s4.is_finite());
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
    quantize::dequantize_into(&quantized, scale, &mut reconstructed)
        .expect("equal lengths must succeed");

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

#[test]
fn test_dequantize_into_reports_a_short_buffer() {
    // Release builds used to silently write a partial result.
    let quantized = [3_i8; 12];
    let mut output = [0.0_f32; 4];
    assert_eq!(
        quantize::dequantize_into(&quantized, 0.25, &mut output),
        Err(EmbeddedError::BufferTooSmall {
            required: 12,
            available: 4
        })
    );
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
    let dot_q_f32 = fixed_dot(&inputs_q, &weights_q)
        .expect("equal lengths")
        .to_f32();

    let err = (dot_f32 - dot_q_f32).abs();
    assert!(
        err < 1e-3,
        "Q16 vs f32 dot mismatch: f32={dot_f32:.6}, q16={dot_q_f32:.6}, err={err:.3e}"
    );
}

#[cfg(feature = "fixed-point")]
#[test]
fn test_fixed_exp_matches_f32_near_zero() {
    use kizzasi_embedded::fixed_point::{fixed_exp_approx, Q16};
    use kizzasi_embedded::math::exp_approx;

    // The documented contract is 0.1 % relative, so the test asserts 0.1 %.
    // The old test asserted 2 % against a documented 1 % claim, which is
    // exactly why the 3 %-error-at-x=-0.5 defect survived.
    for x in [
        -3.0_f32, -1.0, -0.5, -0.4, -0.2, 0.0, 0.2, 0.4, 0.5, 1.0, 3.0,
    ] {
        let f = exp_approx(x);
        let q = fixed_exp_approx(Q16::from_f32(x)).to_f32();
        let rel = (q - f).abs() / f.abs().max(1e-6);
        assert!(
            rel < 1e-3,
            "fixed_exp_approx({x}) = {q:.6}, exp_approx = {f:.6}, rel = {rel:.3e}"
        );
    }
    // And it must never exceed 1 for a negative argument — the old parabola
    // returned 2.5 at x = -3, i.e. a growing SSM decay factor.
    for i in 0..=1000 {
        let x = -(i as f32) / 100.0;
        assert!(
            fixed_exp_approx(Q16::from_f32(x)).to_f32() <= 1.0 + 1e-4,
            "fixed_exp_approx({x}) must not exceed 1"
        );
    }
}

#[cfg(feature = "fixed-point")]
#[test]
fn test_q16_ssm_tracks_f32_ssm() {
    // The claim the README makes for `--features fixed-point`: the SSM
    // recurrence itself runs off the FPU and still tracks the f32 kernel.
    use kizzasi_embedded::fixed_point::Q16;
    use kizzasi_embedded::MambaStepQ16;

    let cfg = rp2040();
    let (x, a_log, b, c) = synth_params(cfg.d_state);
    let to_q = |v: &[f32]| -> Vec<Q16> { v.iter().map(|&f| Q16::from_f32(f)).collect() };
    let (x_q, a_log_q, b_q, c_q) = (to_q(&x), to_q(&a_log), to_q(&b), to_q(&c));

    let mut state_f32 = SsmState::new(&cfg);
    let mut h_q = vec![Q16::ZERO; cfg.d_state];

    let delta = 0.1_f32;
    let d_skip = 0.05_f32;
    let mut worst = 0.0_f32;
    for step in 0..256 {
        let y_f = MambaStep::step(&mut state_f32, &x, &a_log, &b, &c, delta, d_skip)
            .expect("f32 step must succeed");
        let y_q = MambaStepQ16::step_slice(
            &mut h_q,
            &x_q,
            &a_log_q,
            &b_q,
            &c_q,
            Q16::from_f32(delta),
            Q16::from_f32(d_skip),
        )
        .expect("Q16 step must succeed");
        let err = (y_f - y_q.to_f32()).abs();
        assert!(
            err < 2e-3,
            "step {step}: f32 = {y_f:.6}, Q16 = {:.6}, err = {err:.3e}",
            y_q.to_f32()
        );
        worst = worst.max(err);
    }
    for (i, (&hf, &hq)) in state_f32.h.iter().zip(h_q.iter()).enumerate() {
        assert!(
            (hf - hq.to_f32()).abs() < 2e-3,
            "h[{i}]: f32 = {hf:.6}, Q16 = {:.6}",
            hq.to_f32()
        );
    }
}

#[cfg(feature = "fixed-point")]
#[test]
fn test_q16_ssm_is_stable_over_a_long_sequence() {
    use kizzasi_embedded::fixed_point::Q16;
    use kizzasi_embedded::{MambaStepQ16, Q16SsmState};

    let cfg = esp32c3();
    let (x, a_log, b, c) = synth_params(cfg.d_state);
    let to_q = |v: &[f32]| -> Vec<Q16> { v.iter().map(|&f| Q16::from_f32(f)).collect() };
    let (x_q, a_log_q, b_q, c_q) = (to_q(&x), to_q(&a_log), to_q(&b), to_q(&c));

    let mut state = Q16SsmState::new(&cfg);
    for step in 0..5_000 {
        let y = MambaStepQ16::step(
            &mut state,
            &x_q,
            &a_log_q,
            &b_q,
            &c_q,
            Q16::from_f32(0.1),
            Q16::ZERO,
        )
        .expect("Q16 step must succeed");
        assert!(
            y.to_f32().abs() < 10.0,
            "step {step}: Q16 output {} left the stable band",
            y.to_f32()
        );
    }
    for (i, &h) in state.h.iter().enumerate() {
        assert!(
            h != Q16::MAX && h != Q16::MIN,
            "h[{i}] saturated: the Q16 recurrence is not contracting"
        );
    }
}
