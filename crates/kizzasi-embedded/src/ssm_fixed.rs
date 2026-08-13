//! Q16.16 fixed-point SSM inference — the whole recurrence off the FPU.
//!
//! This module is the fixed-point twin of [`crate::ssm`]. Every arithmetic
//! operation in [`MambaStepQ16::step_slice`] — softplus, exp, the ZOH
//! discretisation, the state update and the output projection — is integer
//! `i32`/`i64` work on [`Q16`], so a Cortex-M0/M0+ or any other FPU-less core
//! runs the selective-SSM recurrence without touching a single float and
//! without linking a soft-float library into the hot loop.
//!
//! # Precision contract
//!
//! * All arithmetic **saturates**; nothing wraps (see [`crate::fixed_point`]).
//! * The state, the parameters and the output are all Q16.16, so every
//!   quantity is on a `1.5e-5` grid and is bounded by `±32768`.
//! * Against the `f32` kernel in [`crate::ssm`], with parameters and inputs in
//!   the range a real deployment uses (`|x| <= 1`, `|B|,|C| <= 2`,
//!   `a_log` in `[0, 3]`), the per-step output agrees to within `2e-3`
//!   absolute — verified by
//!   `tests/integration.rs::test_q16_ssm_tracks_f32_ssm`.
//! * `S4Step` has no Q16 twin: it needs `sin`/`cos`, and the crate does not
//!   provide fixed-point trigonometry. Use the `f32` kernel for S4D.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "fixed-point")] {
//! use kizzasi_embedded::fixed_point::Q16;
//! use kizzasi_embedded::ssm_fixed::MambaStepQ16;
//!
//! // Heap-free: the state lives in a plain array.
//! let mut h = [Q16::ZERO; 4];
//! let x = [Q16::from_f32(0.25); 4];
//! // Checkpoint convention: A = -exp(a_log), so a_log is non-negative.
//! let a_log = [Q16::ZERO, Q16::from_f32(0.693), Q16::from_f32(1.099), Q16::from_f32(1.386)];
//! let b = [Q16::from_f32(0.5); 4];
//! let c = [Q16::from_f32(1.0); 4];
//!
//! let y = MambaStepQ16::step_slice(
//!     &mut h, &x, &a_log, &b, &c, Q16::from_f32(0.1), Q16::ZERO,
//! ).expect("dimensions match");
//! assert!(y.to_f32().abs() < 1.0);
//! # }
//! ```

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{vec, vec::Vec};

use crate::error::{EmbeddedError, EmbeddedResult};
use crate::fixed_point::{fixed_exp_approx, fixed_softplus, Q16};
#[cfg(feature = "alloc")]
use crate::ssm::SsmConfig;

/// `20.0` in Q16.16 — the clamp applied to `delta * A` before exponentiation,
/// mirroring the host implementation in `kizzasi-model`.
const DELTA_A_CLAMP_RAW: i32 = 20 << 16;

/// `0.001` in Q16.16 — below this `delta` the exact ZOH formula is worse than
/// its first-order Taylor limit.
const DELTA_TAYLOR_EPS_RAW: i32 = 66;

/// Assert that a parameter slice matches the state dimension.
#[inline]
fn check_len(expected: usize, got: usize) -> EmbeddedResult<()> {
    if expected == got {
        Ok(())
    } else {
        Err(EmbeddedError::DimensionMismatch { expected, got })
    }
}

/// Heap-allocated Q16.16 SSM state (needs the `alloc` feature).
///
/// On an MCU without an allocator, skip this type entirely and hand
/// [`MambaStepQ16::step_slice`] a `[Q16; D_STATE]` array.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q16SsmState {
    /// Recurrent state vector: shape `[d_state]`
    pub h: Vec<Q16>,
}

#[cfg(feature = "alloc")]
impl Q16SsmState {
    /// Allocate a new zeroed Q16.16 state for the given config.
    #[must_use]
    pub fn new(config: &SsmConfig) -> Self {
        Self {
            h: vec![Q16::ZERO; config.d_state],
        }
    }

    /// Zero the state, ready for a fresh sequence.
    pub fn reset(&mut self) {
        self.h.iter_mut().for_each(|v| *v = Q16::ZERO);
    }
}

/// One Mamba selective-SSM recurrence step, entirely in Q16.16.
///
/// The scheme is identical to [`crate::ssm::MambaStep`], including the
/// `A = -exp(a_log)` checkpoint convention and the exact ZOH `B_bar`:
///
/// ```text
/// delta_sp  = softplus(delta)
/// A[i]      = -exp(a_log[i])
/// A_bar[i]  = exp(clamp(delta_sp * A[i], -20, 20))
/// B_bar[i]  = (A_bar[i] - 1) / A[i] * B[i]
/// h_t[i]    = A_bar[i] * h_{t-1}[i] + B_bar[i] * x[i]
/// y         = Σ_i C[i] * h_t[i]  +  d_skip * Σ_j x[j]
/// ```
pub struct MambaStepQ16;

impl MambaStepQ16 {
    /// Execute one Q16.16 recurrence step over a borrowed state slice.
    ///
    /// Needs no allocator and performs no floating-point arithmetic.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddedError::DimensionMismatch`] when any parameter slice
    /// has a length other than `h.len()`.
    pub fn step_slice(
        h: &mut [Q16],
        x: &[Q16],
        a_log: &[Q16],
        b: &[Q16],
        c: &[Q16],
        delta: Q16,
        d_skip: Q16,
    ) -> EmbeddedResult<Q16> {
        let ds = h.len();
        check_len(ds, x.len())?;
        check_len(ds, a_log.len())?;
        check_len(ds, b.len())?;
        check_len(ds, c.len())?;

        let delta_sp = fixed_softplus(delta);

        // Skip connection: d_skip * sum(x), saturating throughout.
        let mut sum_x = Q16::ZERO;
        for &xi in x.iter() {
            sum_x = sum_x.saturating_add(xi);
        }
        let mut y = d_skip * sum_x;

        for ((((hi, &xi), &ali), &bi), &ci) in h
            .iter_mut()
            .zip(x.iter())
            .zip(a_log.iter())
            .zip(b.iter())
            .zip(c.iter())
        {
            // A = -exp(a_log): strictly negative (or zero after underflow).
            let a = fixed_exp_approx(ali).saturating_neg();
            let arg = Q16::from_raw(
                (delta_sp * a)
                    .to_raw()
                    .clamp(-DELTA_A_CLAMP_RAW, DELTA_A_CLAMP_RAW),
            );
            let a_bar = fixed_exp_approx(arg);

            // B_bar: exact ZOH, falling back to the first-order Taylor limit
            // where the division is ill-conditioned (tiny delta, or an A that
            // has underflowed to zero on the Q16.16 grid).
            let b_bar = if delta_sp.to_raw().abs() < DELTA_TAYLOR_EPS_RAW || a.to_raw() == 0 {
                delta_sp * bi
            } else {
                match a_bar.saturating_sub(Q16::ONE).checked_div(a) {
                    Ok(scale) => scale * bi,
                    // Unreachable: `a` is non-zero on this branch.
                    Err(_) => delta_sp * bi,
                }
            };

            *hi = (a_bar * *hi).saturating_add(b_bar * xi);
            y = y.saturating_add(ci * *hi);
        }

        Ok(y)
    }

    /// Execute one Q16.16 recurrence step against a [`Q16SsmState`].
    ///
    /// # Errors
    ///
    /// See [`step_slice`](Self::step_slice).
    #[cfg(feature = "alloc")]
    pub fn step(
        state: &mut Q16SsmState,
        x: &[Q16],
        a_log: &[Q16],
        b: &[Q16],
        c: &[Q16],
        delta: Q16,
        d_skip: Q16,
    ) -> EmbeddedResult<Q16> {
        Self::step_slice(&mut state.h, x, a_log, b, c, delta, d_skip)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssm::MambaStep;

    fn to_q(values: &[f32]) -> Vec<Q16> {
        values.iter().map(|&v| Q16::from_f32(v)).collect()
    }

    #[test]
    fn test_q16_step_runs_without_state_struct() {
        let mut h = [Q16::ZERO; 4];
        let x = [Q16::from_f32(0.25); 4];
        let a_log = [Q16::ZERO; 4];
        let b = [Q16::from_f32(0.5); 4];
        let c = [Q16::ONE; 4];
        let y = MambaStepQ16::step_slice(&mut h, &x, &a_log, &b, &c, Q16::from_f32(0.1), Q16::ZERO)
            .expect("dimensions match");
        assert!(y.to_f32().abs() > 0.0, "output must be non-trivial");
        assert!(h.iter().any(|&v| v != Q16::ZERO), "state must evolve");
    }

    #[test]
    fn test_q16_step_dimension_mismatch() {
        let mut h = [Q16::ZERO; 4];
        assert_eq!(
            MambaStepQ16::step_slice(
                &mut h,
                &[Q16::ZERO; 3],
                &[Q16::ZERO; 4],
                &[Q16::ZERO; 4],
                &[Q16::ZERO; 4],
                Q16::ONE,
                Q16::ZERO
            ),
            Err(EmbeddedError::DimensionMismatch {
                expected: 4,
                got: 3
            })
        );
    }

    #[test]
    fn test_q16_step_tracks_f32_step() {
        // The core cross-check: run both kernels on identical parameters and
        // compare per-step outputs.
        let d_state = 8;
        let a_log_f: Vec<f32> = (0..d_state).map(|n| ((n + 1) as f32).ln()).collect();
        let x_f: Vec<f32> = (0..d_state).map(|i| 0.05 + 0.05 * i as f32).collect();
        let b_f: Vec<f32> = (0..d_state).map(|i| 0.3 + 0.02 * i as f32).collect();
        let c_f: Vec<f32> = (0..d_state).map(|i| 0.8 - 0.01 * i as f32).collect();
        let delta = 0.1_f32;
        let d_skip = 0.05_f32;

        let (a_log_q, x_q) = (to_q(&a_log_f), to_q(&x_f));
        let (b_q, c_q) = (to_q(&b_f), to_q(&c_f));

        let mut h_f = vec![0.0_f32; d_state];
        let mut h_q = vec![Q16::ZERO; d_state];

        let mut worst = 0.0_f32;
        for step in 0..64 {
            let y_f = MambaStep::step_slice(&mut h_f, &x_f, &a_log_f, &b_f, &c_f, delta, d_skip)
                .expect("f32 step");
            let y_q = MambaStepQ16::step_slice(
                &mut h_q,
                &x_q,
                &a_log_q,
                &b_q,
                &c_q,
                Q16::from_f32(delta),
                Q16::from_f32(d_skip),
            )
            .expect("q16 step");
            let err = (y_f - y_q.to_f32()).abs();
            assert!(
                err < 2e-3,
                "step {step}: f32 y = {y_f}, Q16 y = {}, err = {err:.3e}",
                y_q.to_f32()
            );
            worst = worst.max(err);
        }
        // The state vectors must also agree, not just the scalar output.
        for (i, (&hf, &hq)) in h_f.iter().zip(h_q.iter()).enumerate() {
            assert!(
                (hf - hq.to_f32()).abs() < 2e-3,
                "h[{i}]: f32 = {hf}, Q16 = {}",
                hq.to_f32()
            );
        }
        assert!(worst > 0.0, "the two paths should not be bit-identical");
    }

    #[test]
    fn test_q16_step_is_stable_over_a_long_sequence() {
        // The fixed-point path must inherit the f32 path's contraction: with
        // checkpoint-convention a_log the state cannot run away.
        let d_state = 8;
        let a_log = to_q(
            &(0..d_state)
                .map(|n| ((n + 1) as f32).ln())
                .collect::<Vec<_>>(),
        );
        let x = vec![Q16::from_f32(0.5); d_state];
        let b = vec![Q16::ONE; d_state];
        let c = vec![Q16::ONE; d_state];
        let mut h = vec![Q16::ZERO; d_state];

        for _ in 0..5_000 {
            let y =
                MambaStepQ16::step_slice(&mut h, &x, &a_log, &b, &c, Q16::from_f32(0.1), Q16::ZERO)
                    .expect("step must succeed");
            assert!(
                y.to_f32().abs() < 100.0,
                "Q16 output must stay bounded, got {}",
                y.to_f32()
            );
        }
        for (i, &hi) in h.iter().enumerate() {
            assert!(
                hi != Q16::MAX && hi != Q16::MIN,
                "h[{i}] saturated, the recurrence is not contracting"
            );
        }
    }

    #[test]
    fn test_q16_state_struct_matches_slice_form() {
        let cfg = SsmConfig::new(8, 4, 2).expect("valid config");
        let mut state = Q16SsmState::new(&cfg);
        assert_eq!(state.h.len(), 4);

        let x = vec![Q16::from_f32(0.3); 4];
        let a_log = vec![Q16::from_f32(0.5); 4];
        let b = vec![Q16::from_f32(0.7); 4];
        let c = vec![Q16::ONE; 4];

        let mut h = vec![Q16::ZERO; 4];
        let y_slice = MambaStepQ16::step_slice(&mut h, &x, &a_log, &b, &c, Q16::ONE, Q16::ZERO)
            .expect("slice form");
        let y_state = MambaStepQ16::step(&mut state, &x, &a_log, &b, &c, Q16::ONE, Q16::ZERO)
            .expect("state form");
        assert_eq!(y_slice, y_state, "both entry points must agree exactly");
        assert_eq!(state.h, h);

        state.reset();
        assert!(state.h.iter().all(|&v| v == Q16::ZERO));
    }

    #[test]
    fn test_q16_step_saturates_rather_than_wrapping() {
        // Deliberately absurd parameters: the output must clamp, never flip
        // sign, and never produce a bogus small value.
        let mut h = [Q16::MAX; 2];
        let x = [Q16::MAX; 2];
        let a_log = [Q16::MIN; 2]; // A -> 0 => no decay
        let b = [Q16::MAX; 2];
        let c = [Q16::MAX; 2];
        let y = MambaStepQ16::step_slice(&mut h, &x, &a_log, &b, &c, Q16::MAX, Q16::MAX)
            .expect("dimensions match");
        assert_eq!(y, Q16::MAX, "the output must saturate high, not wrap");
        for &hi in h.iter() {
            assert!(!hi.is_negative(), "state must not flip sign on overflow");
        }
    }
}
