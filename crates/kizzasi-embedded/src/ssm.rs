//! SSM inference steps for embedded targets.
//!
//! All operations use heap-allocated state (via `alloc::vec::Vec`) for the
//! recurrent hidden state, while the step functions themselves require no
//! additional heap allocation at inference time.

// Pull `Vec` and the `vec!` macro from the `alloc` crate when building
// without `std`. The `std` feature implies `alloc`, but when `std` is on we
// rely on libstd's prelude to already provide both.
#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{vec, vec::Vec};

use crate::error::{EmbeddedError, EmbeddedResult};
use crate::math::{exp_approx, softplus};

/// Configuration for a single SSM layer.
#[derive(Debug, Clone)]
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
    pub fn new(d_model: usize, d_state: usize, expand: usize) -> EmbeddedResult<Self> {
        if d_model == 0 || d_state == 0 || expand == 0 {
            return Err(EmbeddedError::InvalidConfig("dimensions must be > 0"));
        }
        Ok(Self {
            d_model,
            d_state,
            d_inner: d_model * expand,
        })
    }

    /// Tiny Mamba config suitable for very constrained embedded targets.
    pub fn mamba_tiny() -> Self {
        Self {
            d_model: 64,
            d_state: 8,
            d_inner: 128,
        }
    }

    /// Small Mamba config for mid-range embedded targets.
    pub fn mamba_small() -> Self {
        Self {
            d_model: 128,
            d_state: 16,
            d_inner: 256,
        }
    }
}

/// SSM hidden state (heap-allocated via `alloc`).
///
/// Holds the recurrent state vector `h` and the previous input projection
/// `prev_x`. Both are zeroed on construction and can be reset via `reset`.
#[derive(Debug, Clone)]
pub struct SsmState {
    /// Recurrent state vector: shape `[d_state]`
    pub h: Vec<f32>,
    /// Previous input projection: shape `[d_inner]`
    pub prev_x: Vec<f32>,
}

impl SsmState {
    /// Allocate a new zeroed state for the given config.
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

/// One Mamba SSM recurrence step.
///
/// Implements the selective SSM scan for a single time step using
/// ZOH (zero-order hold) discretisation:
///
/// ```text
/// delta_sp  = softplus(delta)
/// A_bar[i]  = exp(delta_sp * a_log[i])
/// B_bar[i]  = delta_sp * B[i]
/// h_t[i]    = A_bar[i] * h_{t-1}[i] + B_bar[i] * x[i]
/// y         = Σ_i C[i] * h_t[i]  +  d_skip * Σ_j x[j]
/// ```
///
/// # Arguments
/// * `state`  — mutable SSM hidden state (updated in place)
/// * `x`      — input vector of length `d_state`
/// * `a_log`  — log(-A) diagonal, length `d_state`
/// * `b`      — B matrix diagonal, length `d_state`
/// * `c`      — C output-projection vector, length `d_state`
/// * `delta`  — raw step size (softplus applied internally)
/// * `d_skip` — skip-connection scalar
pub struct MambaStep;

impl MambaStep {
    /// Execute one recurrence step, returning the scalar output y.
    pub fn step(
        state: &mut SsmState,
        x: &[f32],
        a_log: &[f32],
        b: &[f32],
        c: &[f32],
        delta: f32,
        d_skip: f32,
    ) -> EmbeddedResult<f32> {
        let ds = state.h.len();
        if x.len() != ds {
            return Err(EmbeddedError::DimensionMismatch {
                expected: ds,
                got: x.len(),
            });
        }
        if a_log.len() != ds {
            return Err(EmbeddedError::DimensionMismatch {
                expected: ds,
                got: a_log.len(),
            });
        }
        if b.len() != ds {
            return Err(EmbeddedError::DimensionMismatch {
                expected: ds,
                got: b.len(),
            });
        }
        if c.len() != ds {
            return Err(EmbeddedError::DimensionMismatch {
                expected: ds,
                got: c.len(),
            });
        }

        // Discretise step size via softplus
        let delta_sp = softplus(delta);

        // Skip connection: d_skip * sum(x)
        let mut y = d_skip * x.iter().sum::<f32>();

        // SSM recurrence with ZOH discretisation
        for i in 0..ds {
            let a_bar = exp_approx(delta_sp * a_log[i]);
            let b_bar = delta_sp * b[i];
            state.h[i] = a_bar * state.h[i] + b_bar * x[i];
            y += c[i] * state.h[i];
        }

        Ok(y)
    }
}

/// Simplified S4 diagonal SSM step.
///
/// Implements a complex-diagonal SSM recurrence for a single scalar input,
/// using real-part-only approximation suitable for embedded targets without
/// complex number support:
///
/// ```text
/// decay[i]  = exp(lambda_re[i] * dt)
/// h_t[i]   = decay[i] * h_{t-1}[i] + B[i] * x
/// y         = Σ_i C[i] * h_t[i]
/// ```
pub struct S4Step;

impl S4Step {
    /// Execute one S4 diagonal recurrence step, returning the scalar output.
    pub fn step(
        state: &mut SsmState,
        x: f32,
        lambda_re: &[f32],
        lambda_im: &[f32],
        b: &[f32],
        c: &[f32],
        dt: f32,
    ) -> EmbeddedResult<f32> {
        let ds = state.h.len();
        if lambda_re.len() != ds {
            return Err(EmbeddedError::DimensionMismatch {
                expected: ds,
                got: lambda_re.len(),
            });
        }
        if lambda_im.len() != ds {
            return Err(EmbeddedError::DimensionMismatch {
                expected: ds,
                got: lambda_im.len(),
            });
        }
        if b.len() != ds {
            return Err(EmbeddedError::DimensionMismatch {
                expected: ds,
                got: b.len(),
            });
        }
        if c.len() != ds {
            return Err(EmbeddedError::DimensionMismatch {
                expected: ds,
                got: c.len(),
            });
        }

        let mut y = 0.0_f32;
        for i in 0..ds {
            // Simplified: real-part decay only (imaginary part ignored for embedded)
            let decay = exp_approx(lambda_re[i] * dt);
            state.h[i] = decay * state.h[i] + b[i] * x;
            y += c[i] * state.h[i];
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssm_config_new_valid() {
        let cfg = SsmConfig::new(128, 16, 2);
        assert!(cfg.is_ok(), "SsmConfig::new(128, 16, 2) should succeed");
        let cfg = cfg.unwrap();
        assert_eq!(cfg.d_model, 128);
        assert_eq!(cfg.d_state, 16);
        assert_eq!(cfg.d_inner, 256);
    }

    #[test]
    fn test_ssm_config_new_zero_dim() {
        let result = SsmConfig::new(0, 16, 2);
        assert!(
            result.is_err(),
            "SsmConfig::new(0, 16, 2) should return Err"
        );
        assert_eq!(
            result.unwrap_err(),
            EmbeddedError::InvalidConfig("dimensions must be > 0")
        );
    }

    #[test]
    fn test_ssm_state_new_shape() {
        let cfg = SsmConfig::new(64, 8, 2).unwrap();
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
        let cfg = SsmConfig::new(64, 4, 2).unwrap();
        let mut state = SsmState::new(&cfg);
        // Perturb state
        state.h.iter_mut().for_each(|v| *v = 1.5);
        state.prev_x.iter_mut().for_each(|v| *v = -2.0);
        // Reset
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
        let cfg = SsmConfig::new(8, 4, 2).unwrap();
        let mut state = SsmState::new(&cfg);
        let ds = cfg.d_state;
        let x = vec![0.1_f32; ds];
        let a_log = vec![-1.0_f32; ds];
        let b = vec![0.5_f32; ds];
        let c = vec![1.0_f32; ds];
        let delta = 0.1_f32;
        let d_skip = 0.0_f32;

        let result = MambaStep::step(&mut state, &x, &a_log, &b, &c, delta, d_skip);
        assert!(result.is_ok(), "MambaStep::step should succeed");
        let y = result.unwrap();
        assert!(y.is_finite(), "output y must be finite, got {y}");
        // State h must have been updated from zeros
        assert!(
            state.h.iter().any(|&v| v != 0.0),
            "state h should be non-zero after a step"
        );
    }

    #[test]
    fn test_mamba_step_dimension_mismatch() {
        let cfg = SsmConfig::new(8, 4, 2).unwrap();
        let mut state = SsmState::new(&cfg);
        // x has wrong length (3 instead of 4)
        let x = vec![0.1_f32; 3];
        let a_log = vec![-1.0_f32; 4];
        let b = vec![0.5_f32; 4];
        let c = vec![1.0_f32; 4];
        let result = MambaStep::step(&mut state, &x, &a_log, &b, &c, 0.1, 0.0);
        assert!(
            matches!(
                result,
                Err(EmbeddedError::DimensionMismatch {
                    expected: 4,
                    got: 3
                })
            ),
            "expected DimensionMismatch error, got {result:?}"
        );
    }

    #[test]
    fn test_s4_step_basic() {
        let cfg = SsmConfig::new(8, 4, 2).unwrap();
        let mut state = SsmState::new(&cfg);
        let ds = cfg.d_state;
        let lambda_re = vec![-0.5_f32; ds];
        let lambda_im = vec![0.1_f32; ds];
        let b = vec![1.0_f32; ds];
        let c = vec![1.0_f32; ds];
        let result = S4Step::step(&mut state, 1.0, &lambda_re, &lambda_im, &b, &c, 0.01);
        assert!(result.is_ok(), "S4Step::step should succeed");
        let y = result.unwrap();
        assert!(y.is_finite(), "S4 output must be finite, got {y}");
    }

    #[test]
    fn test_mamba_tiny_config() {
        let cfg = SsmConfig::mamba_tiny();
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.d_state, 8);
        assert_eq!(cfg.d_inner, 128);
        // Verify state allocation works
        let state = SsmState::new(&cfg);
        assert_eq!(state.h.len(), 8);
        assert_eq!(state.prev_x.len(), 128);
    }
}
