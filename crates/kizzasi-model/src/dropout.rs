//! Inverted dropout used by the sequence-model layers.
//!
//! Dropout is a *training-time* regularizer. Every model in this crate starts in
//! inference mode, where [`apply_dropout`] is a no-op — `SignalPredictor::step`
//! stays deterministic, which is what streaming callers expect. Switch a model
//! into training mode with its `set_training` method (e.g.
//! [`crate::mamba::Mamba::set_training`]) to make the configured
//! `MambaConfig::dropout` / `RwkvConfig::dropout` / … rate take effect.
//!
//! The *inverted* formulation scales the surviving activations by `1/(1-p)` at
//! training time, so no compensating rescale is needed at inference time.

use scirs2_core::ndarray::Array1;
use scirs2_core::random::{rng, RngExt};

/// Apply inverted dropout to `x` in place.
///
/// Each element is zeroed with probability `p`; the survivors are multiplied by
/// `1 / (1 - p)` so the expected value of every activation is unchanged.
///
/// The call returns immediately — leaving `x` untouched — when `training` is
/// `false`, or when `p` is not a finite value strictly inside `(0, 1)`. A rate
/// of exactly `1.0` would zero the whole activation and is therefore treated as
/// "disabled" rather than silently destroying the signal.
pub fn apply_dropout(x: &mut Array1<f32>, p: f32, training: bool) {
    if !training || !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return;
    }
    let scale = 1.0 / (1.0 - p);
    let mut prng = rng();
    for v in x.iter_mut() {
        if prng.random::<f32>() < p {
            *v = 0.0;
        } else {
            *v *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_mode_is_a_no_op() {
        let mut x = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        apply_dropout(&mut x, 0.5, false);
        assert_eq!(x.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn zero_rate_is_a_no_op_even_while_training() {
        let mut x = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        apply_dropout(&mut x, 0.0, true);
        assert_eq!(x.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn non_finite_rate_is_rejected() {
        let mut x = Array1::from_vec(vec![1.0f32, 2.0]);
        apply_dropout(&mut x, f32::NAN, true);
        assert_eq!(x.to_vec(), vec![1.0, 2.0]);
        apply_dropout(&mut x, 1.0, true);
        assert_eq!(x.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn training_mode_zeroes_some_and_rescales_the_rest() {
        let mut x = Array1::from_elem(4096, 1.0f32);
        apply_dropout(&mut x, 0.5, true);

        let zeros = x.iter().filter(|v| **v == 0.0).count();
        assert!(
            (1000..3100).contains(&zeros),
            "expected roughly half the units dropped, got {zeros}"
        );
        for v in x.iter() {
            assert!(
                *v == 0.0 || (*v - 2.0).abs() < 1e-6,
                "survivors must be scaled by 1/(1-p) = 2, got {v}"
            );
        }
        // Inverted dropout preserves the expected activation.
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        assert!((mean - 1.0).abs() < 0.15, "mean drifted to {mean}");
    }
}
