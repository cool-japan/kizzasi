//! SSM inference steps for embedded targets.
//!
//! Every kernel in this module has two entry points:
//!
//! * a **slice** form (`*_slice`) that borrows the recurrent state as
//!   `&mut [f32]`. It needs no allocator at all, so it works on bare-metal
//!   targets built with `--no-default-features --features libm`, where the
//!   caller owns the state as a plain `[f32; N]` array in `.bss`.
//! * a **state** form that takes the heap-allocated [`SsmState`] /
//!   [`S4State`] convenience wrappers. These require the `alloc` feature.
//!
//! Neither form allocates at inference time.

// Pull `Vec` and the `vec!` macro from the `alloc` crate when building
// without `std`. The `std` feature implies `alloc`, but when `std` is on we
// rely on libstd's prelude to already provide both.
#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{vec, vec::Vec};

use crate::error::{EmbeddedError, EmbeddedResult};
use crate::math::{cos_approx, exp_approx, sin_approx, softplus};

/// Clamp applied to `delta * A` before exponentiation, mirroring the host
/// implementation in `kizzasi-model` (`mamba.rs`).
const DELTA_A_CLAMP: f32 = 20.0;

/// Below this `|delta|` the exact ZOH formula for `B_bar` is numerically
/// worse than its first-order Taylor limit `delta * B`.
const DELTA_TAYLOR_EPS: f32 = 1e-3;

/// Below this `|A|` the exact ZOH formula divides by (almost) zero; the
/// L'Hôpital limit of `(e^{delta*a} - 1) / a` as `a -> 0` is `delta`.
const A_TAYLOR_EPS: f32 = 1e-8;

/// Assert that every parameter slice matches the state dimension.
#[inline]
fn check_len(expected: usize, got: usize) -> EmbeddedResult<()> {
    if expected == got {
        Ok(())
    } else {
        Err(EmbeddedError::DimensionMismatch { expected, got })
    }
}

/// Configuration for a single SSM layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmConfig {
    /// Model dimension (input/output size)
    pub d_model: usize,
    /// SSM state dimension (recurrent state size)
    pub d_state: usize,
    /// Inner / expanded dimension (typically d_model * expand)
    pub d_inner: usize,
}

impl SsmConfig {
    /// Construct a new config, validating that all dimensions are non-zero.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddedError::InvalidConfig`] when any dimension is zero,
    /// or when `d_model * expand` overflows `usize` — a real hazard on the
    /// 32-bit MCU targets this crate exists for, where the product wraps to a
    /// small bogus `d_inner` in release builds.
    pub fn new(d_model: usize, d_state: usize, expand: usize) -> EmbeddedResult<Self> {
        if d_model == 0 || d_state == 0 || expand == 0 {
            return Err(EmbeddedError::InvalidConfig("dimensions must be > 0"));
        }
        let d_inner = d_model
            .checked_mul(expand)
            .ok_or(EmbeddedError::InvalidConfig("d_model * expand overflows"))?;
        Ok(Self {
            d_model,
            d_state,
            d_inner,
        })
    }

    /// Re-check the invariants of a config that was built by struct literal.
    ///
    /// [`SsmConfig`]'s fields are public and the platform presets construct
    /// the struct directly, so [`new`](Self::new)'s validation can be
    /// bypassed. Call this before allocating state from a config that did not
    /// come from `new`.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddedError::InvalidConfig`] when any dimension is zero.
    pub fn validate(&self) -> EmbeddedResult<()> {
        if self.d_model == 0 || self.d_state == 0 || self.d_inner == 0 {
            return Err(EmbeddedError::InvalidConfig("dimensions must be > 0"));
        }
        Ok(())
    }

    /// Tiny Mamba config suitable for very constrained embedded targets.
    #[must_use]
    pub fn mamba_tiny() -> Self {
        Self {
            d_model: 64,
            d_state: 8,
            d_inner: 128,
        }
    }

    /// Small Mamba config for mid-range embedded targets.
    #[must_use]
    pub fn mamba_small() -> Self {
        Self {
            d_model: 128,
            d_state: 16,
            d_inner: 256,
        }
    }
}

/// Mamba SSM hidden state (heap-allocated via `alloc`).
///
/// Holds the recurrent state vector `h` (length `d_state`) and the causal
/// convolution history `prev_x` (length `d_inner`, consumed by
/// [`DepthwiseConv1d`]). Both are zeroed on construction and can be reset via
/// [`reset`](Self::reset).
///
/// This type needs an allocator. On MCUs without one, keep the two buffers as
/// plain arrays and call the `*_slice` kernels directly.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct SsmState {
    /// Recurrent state vector: shape `[d_state]`
    pub h: Vec<f32>,
    /// Previous convolution input: shape `[d_inner]`
    pub prev_x: Vec<f32>,
}

#[cfg(feature = "alloc")]
impl SsmState {
    /// Allocate a new zeroed state for the given config.
    #[must_use]
    pub fn new(config: &SsmConfig) -> Self {
        Self {
            h: vec![0.0_f32; config.d_state],
            prev_x: vec![0.0_f32; config.d_inner],
        }
    }

    /// Zero out all state vectors, ready for a fresh sequence.
    pub fn reset(&mut self) {
        self.h.iter_mut().for_each(|v| *v = 0.0);
        self.prev_x.iter_mut().for_each(|v| *v = 0.0);
    }
}

/// Complex-diagonal S4 hidden state (heap-allocated via `alloc`).
///
/// [`S4Step`] carries a genuinely complex state, so the real and imaginary
/// parts are stored as two parallel `[d_state]` buffers.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct S4State {
    /// Real part of the recurrent state: shape `[d_state]`
    pub h_re: Vec<f32>,
    /// Imaginary part of the recurrent state: shape `[d_state]`
    pub h_im: Vec<f32>,
}

#[cfg(feature = "alloc")]
impl S4State {
    /// Allocate a new zeroed complex state for the given config.
    #[must_use]
    pub fn new(config: &SsmConfig) -> Self {
        Self {
            h_re: vec![0.0_f32; config.d_state],
            h_im: vec![0.0_f32; config.d_state],
        }
    }

    /// Zero out both halves of the state, ready for a fresh sequence.
    pub fn reset(&mut self) {
        self.h_re.iter_mut().for_each(|v| *v = 0.0);
        self.h_im.iter_mut().for_each(|v| *v = 0.0);
    }
}

/// One Mamba SSM recurrence step.
///
/// Implements the selective SSM scan for a single time step using
/// exact ZOH (zero-order hold) discretisation:
///
/// ```text
/// delta_sp  = softplus(delta)
/// A[i]      = -exp(a_log[i])                       // strictly negative
/// A_bar[i]  = exp(clamp(delta_sp * A[i], -20, 20)) // in (0, 1)
/// B_bar[i]  = (A_bar[i] - 1) / A[i] * B[i]         // -> delta_sp * B[i] as A -> 0
/// h_t[i]    = A_bar[i] * h_{t-1}[i] + B_bar[i] * x[i]
/// y         = Σ_i C[i] * h_t[i]  +  d_skip * Σ_j x[j]
/// ```
///
/// # Parameter convention
///
/// `a_log` is **log-space**: the effective state matrix is `A = -exp(a_log)`,
/// which is exactly the convention used by `kizzasi-model`
/// (`mixer.A_log` / `ssm.log_a`, HiPPO-initialised to `ln(1..=d_state)`) and
/// `kizzasi-core` (`mamba2.rs`). Checkpoint values are therefore **positive**;
/// feeding raw negative values, as an earlier revision of this crate's
/// examples did, now produces a *slower* decay rather than a divergence.
///
/// # Arguments
/// * `x`      — input vector of length `d_state`
/// * `a_log`  — log(-A) diagonal, length `d_state`
/// * `b`      — B matrix diagonal, length `d_state`
/// * `c`      — C output-projection vector, length `d_state`
/// * `delta`  — raw step size (softplus applied internally)
/// * `d_skip` — skip-connection scalar
pub struct MambaStep;

impl MambaStep {
    /// Execute one recurrence step over a borrowed state slice.
    ///
    /// `h` is the recurrent state of length `d_state`; it is updated in place.
    /// This form needs no allocator.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddedError::DimensionMismatch`] when any parameter slice
    /// has a length other than `h.len()`.
    pub fn step_slice(
        h: &mut [f32],
        x: &[f32],
        a_log: &[f32],
        b: &[f32],
        c: &[f32],
        delta: f32,
        d_skip: f32,
    ) -> EmbeddedResult<f32> {
        let ds = h.len();
        check_len(ds, x.len())?;
        check_len(ds, a_log.len())?;
        check_len(ds, b.len())?;
        check_len(ds, c.len())?;

        // Discretise step size via softplus (always strictly positive).
        let delta_sp = softplus(delta);

        // Skip connection: d_skip * sum(x)
        let mut y = d_skip * x.iter().sum::<f32>();

        // SSM recurrence with exact ZOH discretisation.
        for ((((hi, &xi), &ali), &bi), &ci) in h
            .iter_mut()
            .zip(x.iter())
            .zip(a_log.iter())
            .zip(b.iter())
            .zip(c.iter())
        {
            // A = -exp(a_log): strictly negative, so A_bar is in (0, 1) and
            // the recurrence is unconditionally contracting.
            let a = -exp_approx(ali);
            let a_bar = exp_approx((delta_sp * a).clamp(-DELTA_A_CLAMP, DELTA_A_CLAMP));
            let b_bar = if delta_sp.abs() < DELTA_TAYLOR_EPS || a.abs() < A_TAYLOR_EPS {
                // First-order Taylor limit; also the L'Hôpital limit as A -> 0.
                delta_sp * bi
            } else {
                (a_bar - 1.0) / a * bi
            };
            *hi = a_bar * *hi + b_bar * xi;
            y += ci * *hi;
        }

        Ok(y)
    }

    /// Execute one recurrence step against a heap-allocated [`SsmState`].
    ///
    /// Convenience wrapper around [`step_slice`](Self::step_slice).
    ///
    /// # Errors
    ///
    /// See [`step_slice`](Self::step_slice).
    #[cfg(feature = "alloc")]
    pub fn step(
        state: &mut SsmState,
        x: &[f32],
        a_log: &[f32],
        b: &[f32],
        c: &[f32],
        delta: f32,
        d_skip: f32,
    ) -> EmbeddedResult<f32> {
        Self::step_slice(&mut state.h, x, a_log, b, c, delta, d_skip)
    }
}

/// Causal depthwise 1-D convolution with kernel size 2 over `d_inner`
/// channels — the token-shift half of a Mamba block.
///
/// ```text
/// out[i]    = w_curr[i] * x[i] + w_prev[i] * prev_x[i] + bias[i]
/// prev_x[i] = x[i]
/// ```
///
/// This is what the `prev_x` buffer in [`SsmState`] is for: it carries the
/// single previous frame that a causal kernel-2 convolution needs, so the
/// whole block stays O(1) in time and memory per step. Mamba applies
/// [`silu`](crate::math::silu) to `out` before the selective scan; that is
/// left to the caller so the same kernel can serve other architectures.
pub struct DepthwiseConv1d;

impl DepthwiseConv1d {
    /// Run one convolution step over borrowed buffers. Needs no allocator.
    ///
    /// `bias` may be empty, in which case every channel bias is `0.0`;
    /// otherwise it must have `d_inner` entries.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddedError::DimensionMismatch`] when any slice length
    /// differs from `prev_x.len()` (`bias` may additionally be empty).
    pub fn step_slice(
        prev_x: &mut [f32],
        x: &[f32],
        w_curr: &[f32],
        w_prev: &[f32],
        bias: &[f32],
        out: &mut [f32],
    ) -> EmbeddedResult<()> {
        let di = prev_x.len();
        check_len(di, x.len())?;
        check_len(di, w_curr.len())?;
        check_len(di, w_prev.len())?;
        check_len(di, out.len())?;
        if !bias.is_empty() {
            check_len(di, bias.len())?;
        }

        for (i, ((((p, &xi), &w0), &w1), o)) in prev_x
            .iter_mut()
            .zip(x.iter())
            .zip(w_curr.iter())
            .zip(w_prev.iter())
            .zip(out.iter_mut())
            .enumerate()
        {
            let bi = bias.get(i).copied().unwrap_or(0.0);
            *o = w0 * xi + w1 * *p + bi;
            *p = xi;
        }
        Ok(())
    }

    /// Run one convolution step against a heap-allocated [`SsmState`].
    ///
    /// # Errors
    ///
    /// See [`step_slice`](Self::step_slice).
    #[cfg(feature = "alloc")]
    pub fn step(
        state: &mut SsmState,
        x: &[f32],
        w_curr: &[f32],
        w_prev: &[f32],
        bias: &[f32],
        out: &mut [f32],
    ) -> EmbeddedResult<()> {
        Self::step_slice(&mut state.prev_x, x, w_curr, w_prev, bias, out)
    }
}

/// Diagonal S4 (S4D) SSM step with genuine complex poles.
///
/// Each index `i` carries one pole `λ[i] = lambda_re[i] + j·lambda_im[i]`, so
/// the state is complex and is kept as two parallel real buffers:
///
/// ```text
/// decay[i] = exp(lambda_re[i] * dt)
/// θ[i]     = lambda_im[i] * dt
/// h_re'[i] = decay[i] * (cos θ[i] * h_re[i] - sin θ[i] * h_im[i]) + B[i] * x
/// h_im'[i] = decay[i] * (sin θ[i] * h_re[i] + cos θ[i] * h_im[i])
/// y        = Σ_i C[i] * h_re'[i]                 // = Re(C·h), C real
/// ```
///
/// `B` and `C` are real, so the input is injected into the real part only and
/// the output is `Re(C·h)`.
///
/// # Output convention
///
/// The output takes `Re(C·h)`, **not** the `2·Re(C·h)` of the conjugate-pair
/// S4D formulation. This is deliberate: with `lambda_im = 0` the kernel then
/// reduces exactly to the real diagonal SSM in the host implementation
/// (`kizzasi-model/src/s4.rs`, `forward_step`: `y = Σ C[i] * h[i]`), so the
/// same weights give the same magnitude on host and device. If your weights
/// assume the conjugate-pair convention, fold the factor of two into `C`.
///
/// Setting `lambda_im = 0` collapses the kernel to a purely real decay, which
/// is what the previous revision computed — but that revision *validated*
/// `lambda_im` and then discarded it, so oscillatory modes (the entire point
/// of the complex poles in S4D) were silently unreachable.
pub struct S4Step;

impl S4Step {
    /// Execute one S4D step over borrowed state slices. Needs no allocator.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddedError::DimensionMismatch`] when any slice length
    /// differs from `h_re.len()`.
    #[allow(clippy::too_many_arguments)]
    pub fn step_slice(
        h_re: &mut [f32],
        h_im: &mut [f32],
        x: f32,
        lambda_re: &[f32],
        lambda_im: &[f32],
        b: &[f32],
        c: &[f32],
        dt: f32,
    ) -> EmbeddedResult<f32> {
        let ds = h_re.len();
        check_len(ds, h_im.len())?;
        check_len(ds, lambda_re.len())?;
        check_len(ds, lambda_im.len())?;
        check_len(ds, b.len())?;
        check_len(ds, c.len())?;

        let mut y = 0.0_f32;
        for (((((hr, hi), &lr), &li), &bi), &ci) in h_re
            .iter_mut()
            .zip(h_im.iter_mut())
            .zip(lambda_re.iter())
            .zip(lambda_im.iter())
            .zip(b.iter())
            .zip(c.iter())
        {
            let decay = exp_approx(lr * dt);
            let theta = li * dt;
            let cos_t = cos_approx(theta);
            let sin_t = sin_approx(theta);
            let next_re = decay * (cos_t * *hr - sin_t * *hi) + bi * x;
            let next_im = decay * (sin_t * *hr + cos_t * *hi);
            *hr = next_re;
            *hi = next_im;
            // Re(C·h) with real C — matches the host `s4.rs` output scale.
            y += ci * next_re;
        }
        Ok(y)
    }

    /// Execute one S4D step against a heap-allocated [`S4State`].
    ///
    /// # Errors
    ///
    /// See [`step_slice`](Self::step_slice).
    #[cfg(feature = "alloc")]
    pub fn step(
        state: &mut S4State,
        x: f32,
        lambda_re: &[f32],
        lambda_im: &[f32],
        b: &[f32],
        c: &[f32],
        dt: f32,
    ) -> EmbeddedResult<f32> {
        Self::step_slice(
            &mut state.h_re,
            &mut state.h_im,
            x,
            lambda_re,
            lambda_im,
            b,
            c,
            dt,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HiPPO-style checkpoint values: `a_log[n] = ln(n + 1)`, all >= 0.
    fn hippo_a_log(d_state: usize) -> Vec<f32> {
        (0..d_state).map(|n| ((n + 1) as f32).ln()).collect()
    }

    #[test]
    fn test_ssm_config_new_valid() {
        let cfg = SsmConfig::new(128, 16, 2).expect("SsmConfig::new(128, 16, 2) should succeed");
        assert_eq!(cfg.d_model, 128);
        assert_eq!(cfg.d_state, 16);
        assert_eq!(cfg.d_inner, 256);
    }

    #[test]
    fn test_ssm_config_new_zero_dim() {
        let result = SsmConfig::new(0, 16, 2);
        assert_eq!(
            result,
            Err(EmbeddedError::InvalidConfig("dimensions must be > 0")),
            "SsmConfig::new(0, 16, 2) should return Err"
        );
    }

    #[test]
    fn test_ssm_config_new_overflow_is_rejected() {
        // `usize::MAX * 2` used to wrap to a small bogus `d_inner` in release
        // and abort on overflow in debug.
        assert_eq!(
            SsmConfig::new(usize::MAX, 1, 2),
            Err(EmbeddedError::InvalidConfig("d_model * expand overflows"))
        );
        assert_eq!(
            SsmConfig::new(usize::MAX / 2 + 1, 4, 2),
            Err(EmbeddedError::InvalidConfig("d_model * expand overflows"))
        );
        // A product that exactly fits must still succeed.
        assert!(SsmConfig::new(usize::MAX, 1, 1).is_ok());
    }

    #[test]
    fn test_ssm_config_validate() {
        assert!(SsmConfig::mamba_tiny().validate().is_ok());
        let bogus = SsmConfig {
            d_model: 4,
            d_state: 0,
            d_inner: 8,
        };
        assert_eq!(
            bogus.validate(),
            Err(EmbeddedError::InvalidConfig("dimensions must be > 0")),
            "struct-literal configs must be re-checkable"
        );
    }

    #[test]
    fn test_ssm_state_new_shape() {
        let cfg = SsmConfig::new(64, 8, 2).expect("valid config");
        let state = SsmState::new(&cfg);
        assert_eq!(state.h.len(), cfg.d_state, "h must have length d_state");
        assert_eq!(
            state.prev_x.len(),
            cfg.d_inner,
            "prev_x must have length d_inner"
        );
    }

    #[test]
    fn test_ssm_state_reset() {
        let cfg = SsmConfig::new(64, 4, 2).expect("valid config");
        let mut state = SsmState::new(&cfg);
        state.h.iter_mut().for_each(|v| *v = 1.5);
        state.prev_x.iter_mut().for_each(|v| *v = -2.0);
        state.reset();
        assert!(
            state.h.iter().all(|&v| v == 0.0),
            "h must be zero after reset"
        );
        assert!(
            state.prev_x.iter().all(|&v| v == 0.0),
            "prev_x must be zero after reset"
        );
    }

    #[test]
    fn test_mamba_step_basic() {
        let cfg = SsmConfig::new(8, 4, 2).expect("valid config");
        let mut state = SsmState::new(&cfg);
        let ds = cfg.d_state;
        let x = vec![0.1_f32; ds];
        let a_log = hippo_a_log(ds);
        let b = vec![0.5_f32; ds];
        let c = vec![1.0_f32; ds];

        let y = MambaStep::step(&mut state, &x, &a_log, &b, &c, 0.1, 0.0)
            .expect("MambaStep::step should succeed");
        assert!(y.is_finite(), "output y must be finite, got {y}");
        assert!(
            state.h.iter().any(|&v| v != 0.0),
            "state h should be non-zero after a step"
        );
    }

    #[test]
    fn test_mamba_step_dimension_mismatch() {
        let cfg = SsmConfig::new(8, 4, 2).expect("valid config");
        let mut state = SsmState::new(&cfg);
        // x has wrong length (3 instead of 4)
        let x = vec![0.1_f32; 3];
        let a_log = hippo_a_log(4);
        let b = vec![0.5_f32; 4];
        let c = vec![1.0_f32; 4];
        assert_eq!(
            MambaStep::step(&mut state, &x, &a_log, &b, &c, 0.1, 0.0),
            Err(EmbeddedError::DimensionMismatch {
                expected: 4,
                got: 3
            })
        );
    }

    #[test]
    fn test_mamba_step_is_stable_for_checkpoint_convention_a_log() {
        // Regression test for the `a_log` sign convention. Under the old code
        // (`a_bar = exp(delta_sp * a_log)`) a real checkpoint — HiPPO
        // `a_log[n] = ln(n+1) >= 0` — produced `A_bar >= 1`, i.e. a growing
        // recurrence that diverged silently over a sequence. With
        // `A = -exp(a_log)` the recurrence is unconditionally contracting.
        let d_state = 16;
        let mut h = vec![0.0_f32; d_state];
        let a_log = hippo_a_log(d_state);
        let x = vec![0.5_f32; d_state];
        let b = vec![1.0_f32; d_state];
        let c = vec![1.0_f32; d_state];

        let mut peak = 0.0_f32;
        for _ in 0..1000 {
            let y = MambaStep::step_slice(&mut h, &x, &a_log, &b, &c, 0.1, 0.0)
                .expect("step must succeed");
            assert!(y.is_finite(), "output diverged to {y}");
            peak = peak.max(y.abs());
        }
        for &hi in h.iter() {
            assert!(
                hi.is_finite() && hi.abs() < 100.0,
                "state must stay bounded over 1000 steps, got {hi}"
            );
        }
        assert!(
            peak < 100.0,
            "output must stay bounded over 1000 steps, peak = {peak}"
        );
    }

    #[test]
    fn test_mamba_step_matches_host_zoh_reference() {
        // Bit-for-bit convention check against the `kizzasi-model` host
        // implementation (mamba.rs:443-472), recomputed here in f64.
        let d_state = 4;
        let a_log = hippo_a_log(d_state);
        let x: Vec<f32> = (0..d_state).map(|i| 0.1 + 0.05 * i as f32).collect();
        let b: Vec<f32> = (0..d_state).map(|i| 0.3 + 0.02 * i as f32).collect();
        let c: Vec<f32> = (0..d_state).map(|i| 0.8 - 0.01 * i as f32).collect();
        let delta = 0.25_f32;

        let mut h = vec![0.0_f32; d_state];
        let y = MambaStep::step_slice(&mut h, &x, &a_log, &b, &c, delta, 0.0)
            .expect("step must succeed");

        // Reference: delta_sp = softplus(delta); A = -exp(a_log);
        // A_bar = exp(delta_sp*A); B_bar = (A_bar - 1)/A * B.
        let delta_sp = (1.0_f64 + (delta as f64).exp()).ln();
        let mut want = 0.0_f64;
        for i in 0..d_state {
            let a = -(a_log[i] as f64).exp();
            let a_bar = (delta_sp * a).exp();
            let b_bar = (a_bar - 1.0) / a * b[i] as f64;
            let hi = b_bar * x[i] as f64; // h starts at zero
            want += c[i] as f64 * hi;
            assert!(
                ((h[i] as f64) - hi).abs() < 1e-5,
                "h[{i}] = {}, reference {hi}",
                h[i]
            );
        }
        assert!(
            ((y as f64) - want).abs() < 1e-5,
            "y = {y}, ZOH reference {want}"
        );
    }

    #[test]
    fn test_mamba_step_zoh_differs_from_euler() {
        // The old code used the first-order Euler B_bar = delta_sp * B, which
        // is ~5 % off from exact ZOH at delta = 0.1, A = -1.
        let a_log = vec![0.0_f32]; // A = -exp(0) = -1
        let x = vec![1.0_f32];
        let b = vec![1.0_f32];
        let c = vec![1.0_f32];
        let mut h = vec![0.0_f32; 1];
        // softplus(delta) chosen so delta_sp ~= 0.1
        let delta = -2.2521687_f32; // softplus(-2.2521687) ~= 0.1
        let y = MambaStep::step_slice(&mut h, &x, &a_log, &b, &c, delta, 0.0)
            .expect("step must succeed");
        let delta_sp = softplus(delta) as f64;
        let euler = delta_sp;
        let zoh = ((-delta_sp).exp() - 1.0) / -1.0;
        assert!(
            ((y as f64) - zoh).abs() < 1e-5,
            "y = {y} must match exact ZOH {zoh}, not Euler {euler}"
        );
        assert!(
            (zoh - euler).abs() > 1e-3,
            "the two schemes must be measurably different at this delta"
        );
    }

    #[test]
    fn test_mamba_step_slice_needs_no_state_struct() {
        // The allocator-free entry point: no `Vec` anywhere.
        let mut h = [0.0_f32; 4];
        let a_log: [f32; 4] = core::array::from_fn(|n| crate::math::ln_approx((n + 1) as f32));
        let x = [0.1_f32, 0.2, 0.3, 0.4];
        let b = [0.5_f32; 4];
        let c = [1.0_f32; 4];
        let y = MambaStep::step_slice(&mut h, &x, &a_log, &b, &c, 0.1, 0.05)
            .expect("slice form must succeed");
        assert!(y.is_finite());
        assert!(h.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_depthwise_conv_uses_prev_x() {
        let cfg = SsmConfig::new(2, 2, 2).expect("valid config"); // d_inner = 4
        let mut state = SsmState::new(&cfg);
        let w_curr = [1.0_f32; 4];
        let w_prev = [0.5_f32; 4];
        let mut out = [0.0_f32; 4];

        let x1 = [1.0_f32, 2.0, 3.0, 4.0];
        DepthwiseConv1d::step(&mut state, &x1, &w_curr, &w_prev, &[], &mut out)
            .expect("conv step must succeed");
        // First frame: prev_x is zero, so out == w_curr * x.
        assert_eq!(out, x1, "first frame must be w_curr * x");
        assert_eq!(
            state.prev_x.as_slice(),
            &x1,
            "prev_x must retain the frame it just consumed"
        );

        let x2 = [10.0_f32, 20.0, 30.0, 40.0];
        DepthwiseConv1d::step(&mut state, &x2, &w_curr, &w_prev, &[], &mut out)
            .expect("conv step must succeed");
        for i in 0..4 {
            let want = x2[i] + 0.5 * x1[i];
            assert!(
                (out[i] - want).abs() < 1e-6,
                "out[{i}] = {}, expected {want}",
                out[i]
            );
        }
    }

    #[test]
    fn test_depthwise_conv_bias_and_mismatch() {
        let mut prev_x = [0.0_f32; 3];
        let x = [1.0_f32; 3];
        let w_curr = [1.0_f32; 3];
        let w_prev = [0.0_f32; 3];
        let bias = [0.25_f32, -0.5, 1.0];
        let mut out = [0.0_f32; 3];
        DepthwiseConv1d::step_slice(&mut prev_x, &x, &w_curr, &w_prev, &bias, &mut out)
            .expect("bias path must succeed");
        assert_eq!(out, [1.25, 0.5, 2.0]);

        let short = [0.0_f32; 2];
        assert_eq!(
            DepthwiseConv1d::step_slice(&mut prev_x, &short, &w_curr, &w_prev, &[], &mut out),
            Err(EmbeddedError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
        assert_eq!(
            DepthwiseConv1d::step_slice(&mut prev_x, &x, &w_curr, &w_prev, &short, &mut out),
            Err(EmbeddedError::DimensionMismatch {
                expected: 3,
                got: 2
            }),
            "a non-empty bias must have the full length"
        );
    }

    #[test]
    fn test_s4_step_basic() {
        let cfg = SsmConfig::new(8, 4, 2).expect("valid config");
        let mut state = S4State::new(&cfg);
        let ds = cfg.d_state;
        let lambda_re = vec![-0.5_f32; ds];
        let lambda_im = vec![0.1_f32; ds];
        let b = vec![1.0_f32; ds];
        let c = vec![1.0_f32; ds];
        let y = S4Step::step(&mut state, 1.0, &lambda_re, &lambda_im, &b, &c, 0.01)
            .expect("S4Step::step should succeed");
        assert!(y.is_finite(), "S4 output must be finite, got {y}");
    }

    #[test]
    fn test_s4_lambda_im_changes_the_output() {
        // Regression test: `lambda_im` used to be validated and then thrown
        // away, so no oscillatory mode was representable.
        let d_state = 4;
        let lambda_re = vec![-0.1_f32; d_state];
        let b = vec![1.0_f32; d_state];
        let c = vec![1.0_f32; d_state];
        let dt = 0.5_f32;

        let mut zero_im = (vec![0.0_f32; d_state], vec![0.0_f32; d_state]);
        let mut with_im = (vec![0.0_f32; d_state], vec![0.0_f32; d_state]);
        let li_zero = vec![0.0_f32; d_state];
        let li_osc = vec![2.0_f32; d_state];

        let mut differ = false;
        let mut rang = false;
        for t in 0..8 {
            let x = if t == 0 { 1.0 } else { 0.0 };
            let y0 = S4Step::step_slice(
                &mut zero_im.0,
                &mut zero_im.1,
                x,
                &lambda_re,
                &li_zero,
                &b,
                &c,
                dt,
            )
            .expect("step must succeed");
            let y1 = S4Step::step_slice(
                &mut with_im.0,
                &mut with_im.1,
                x,
                &lambda_re,
                &li_osc,
                &b,
                &c,
                dt,
            )
            .expect("step must succeed");
            if (y0 - y1).abs() > 1e-3 {
                differ = true;
            }
            // A real-decay-only kernel can never go negative after a single
            // positive impulse; a complex pole rings.
            if with_im.0.iter().any(|&v| v < 0.0) || y1 < 0.0 {
                rang = true;
            }
            assert!(
                y0 >= 0.0,
                "the real-only kernel must not ring, got y = {y0} at step {t}"
            );
        }
        assert!(
            differ,
            "lambda_im must influence the output; it is not a discarded parameter"
        );
        assert!(
            rang,
            "an oscillatory pole must drive the state negative at some point"
        );
    }

    #[test]
    fn test_s4_impulse_response_matches_complex_reference() {
        // Compare against an explicit complex-arithmetic reference in f64.
        let lambda_re = [-0.3_f32];
        let lambda_im = [1.7_f32];
        let b = [1.0_f32];
        let c = [0.75_f32];
        let dt = 0.2_f32;
        let mut h_re = [0.0_f32];
        let mut h_im = [0.0_f32];

        let (mut ref_re, mut ref_im) = (0.0_f64, 0.0_f64);
        let decay = ((lambda_re[0] * dt) as f64).exp();
        let theta = (lambda_im[0] * dt) as f64;

        for t in 0..12 {
            let x = if t == 0 { 1.0_f32 } else { 0.0 };
            let y = S4Step::step_slice(&mut h_re, &mut h_im, x, &lambda_re, &lambda_im, &b, &c, dt)
                .expect("step must succeed");

            let nr = decay * (theta.cos() * ref_re - theta.sin() * ref_im) + x as f64;
            let ni = decay * (theta.sin() * ref_re + theta.cos() * ref_im);
            ref_re = nr;
            ref_im = ni;
            let want = c[0] as f64 * ref_re;
            assert!(
                ((y as f64) - want).abs() < 1e-4,
                "step {t}: y = {y}, complex reference {want}"
            );
        }
    }

    #[test]
    fn test_s4_matches_host_real_diagonal_convention_when_lambda_im_is_zero() {
        // With no imaginary part the kernel must reproduce the host
        // `kizzasi-model/src/s4.rs::forward_step` exactly:
        //   h[i] = exp(dt * lambda_re[i]) * h[i] + B[i] * x
        //   y    = Σ C[i] * h[i]              (no conjugate-pair factor of 2)
        let d_state = 4;
        let lambda_re: Vec<f32> = (0..d_state).map(|n| -((n + 1) as f32)).collect();
        let lambda_im = vec![0.0_f32; d_state];
        let b: Vec<f32> = (0..d_state).map(|i| 0.3 + 0.1 * i as f32).collect();
        let c: Vec<f32> = (0..d_state).map(|i| 0.9 - 0.1 * i as f32).collect();
        let dt = 0.2_f32;

        let mut h_re = vec![0.0_f32; d_state];
        let mut h_im = vec![0.0_f32; d_state];
        let mut reference = vec![0.0_f64; d_state];

        for t in 0..10 {
            let x = 0.5 + 0.1 * t as f32;
            let y = S4Step::step_slice(&mut h_re, &mut h_im, x, &lambda_re, &lambda_im, &b, &c, dt)
                .expect("step must succeed");

            let mut want = 0.0_f64;
            for i in 0..d_state {
                let decay = ((lambda_re[i] * dt) as f64).exp();
                reference[i] = decay * reference[i] + b[i] as f64 * x as f64;
                want += c[i] as f64 * reference[i];
            }
            assert!(
                ((y as f64) - want).abs() < 1e-5,
                "step {t}: y = {y}, host-convention reference {want}"
            );
            assert!(
                h_im.iter().all(|&v| v == 0.0),
                "a purely real pole must leave the imaginary state at zero"
            );
        }
    }

    #[test]
    fn test_s4_step_dimension_mismatch() {
        let mut h_re = [0.0_f32; 4];
        let mut h_im = [0.0_f32; 3];
        assert_eq!(
            S4Step::step_slice(
                &mut h_re, &mut h_im, 1.0, &[0.0; 4], &[0.0; 4], &[0.0; 4], &[0.0; 4], 0.1
            ),
            Err(EmbeddedError::DimensionMismatch {
                expected: 4,
                got: 3
            })
        );
    }

    #[test]
    fn test_s4_state_reset() {
        let cfg = SsmConfig::new(8, 4, 2).expect("valid config");
        let mut state = S4State::new(&cfg);
        state.h_re.iter_mut().for_each(|v| *v = 3.0);
        state.h_im.iter_mut().for_each(|v| *v = -3.0);
        state.reset();
        assert!(state.h_re.iter().all(|&v| v == 0.0));
        assert!(state.h_im.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_mamba_tiny_config() {
        let cfg = SsmConfig::mamba_tiny();
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.d_state, 8);
        assert_eq!(cfg.d_inner, 128);
        let state = SsmState::new(&cfg);
        assert_eq!(state.h.len(), 8);
        assert_eq!(state.prev_x.len(), 128);
    }

    #[test]
    fn test_mamba_small_config() {
        let cfg = SsmConfig::mamba_small();
        assert_eq!(cfg.d_model, 128);
        assert_eq!(cfg.d_state, 16);
        assert_eq!(cfg.d_inner, 256);
    }
}
