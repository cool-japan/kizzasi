//! End-to-end integration tests covering the full Kizzasi pipeline.
//!
//! Each test exercises a different cross-crate slice:
//!
//! 1. signal generation -> model inference (kizzasi-io style sine wave -> kizzasi-model)
//! 2. constrained inference with output range guardrails
//!    (kizzasi-logic Constraint + manual projection over [-1, 1])
//! 3. streaming with FilterTransformer (kizzasi-inference::streaming)
//!
//! These tests deliberately stay within the public APIs of each crate so that
//! a change to a `pub(crate)` internal cannot accidentally break the contract
//! validated here.

use kizzasi::prelude::*;

/// Synthesize a sine wave with the given frequency, sample rate, and sample count.
///
/// Defined locally so the tests stay device-free and do not depend on any
/// real audio source. The output uses `scirs2_core::ndarray::Array1<f32>`
/// per KIZZASI_POLICY.
fn synthesize_sine(freq_hz: f32, sample_rate: f32, n_samples: usize) -> Array1<f32> {
    Array1::from_iter((0..n_samples).map(|i| {
        let t = i as f32 / sample_rate;
        (2.0 * std::f32::consts::PI * freq_hz * t).sin()
    }))
}

#[test]
fn test_signal_pipeline_audio_to_inference() {
    use kizzasi_model::mamba::{Mamba, MambaConfig};

    // Synthesize 256 samples of a 440 Hz sine wave at 8 kHz.
    let signal = synthesize_sine(440.0, 8000.0, 256);
    assert_eq!(signal.len(), 256);

    // Build a tiny scalar Mamba (input_dim = output_dim = 1).
    let config = MambaConfig::new()
        .input_dim(1)
        .hidden_dim(16)
        .state_dim(8)
        .num_layers(1);
    let mut mamba = Mamba::new(config).expect("Mamba construction must succeed");

    // Step through the sine wave sample-by-sample.
    let mut outputs = Vec::with_capacity(signal.len());
    for &x in signal.iter() {
        let input = Array1::from_vec(vec![x]);
        let y = mamba.step(&input).expect("mamba step must succeed");
        assert_eq!(y.len(), 1, "output dim should be 1");
        assert!(
            y.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
        outputs.push(y[0]);
    }

    // Sanity check: the model should react to a non-zero input, so the
    // outputs should not be identically zero.
    let mean: f32 = outputs.iter().copied().sum::<f32>() / outputs.len() as f32;
    let variance: f32 = outputs
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f32>()
        / outputs.len() as f32;
    assert!(
        variance > 1e-8,
        "outputs should have nonzero variance (variance = {variance})"
    );
}

#[test]
fn test_constrained_inference_pipeline() {
    use kizzasi_model::mamba::{Mamba, MambaConfig};

    // Build the same tiny scalar Mamba used in test #1.
    let config = MambaConfig::new()
        .input_dim(1)
        .hidden_dim(16)
        .state_dim(8)
        .num_layers(1);
    let mut mamba = Mamba::new(config).expect("Mamba construction must succeed");

    // Construct a [-1, 1] range constraint via the public kizzasi-logic API
    // re-exported through the `kizzasi::prelude` (requires the `logic`
    // feature, which is on by default via `full`).
    let constraint: Constraint = ConstraintBuilder::new()
        .name("amplitude_bound")
        .in_range(-1.0, 1.0)
        .build()
        .expect("constraint must build with valid bounds");

    // Verify the constraint's polarity is sane: in-range values pass,
    // out-of-range values fail. `Constraint::check` operates on a scalar.
    assert!(constraint.check(0.5), "0.5 must satisfy [-1, 1]");
    assert!(constraint.check(-1.0), "-1.0 must satisfy [-1, 1]");
    assert!(constraint.check(1.0), "1.0 must satisfy [-1, 1]");
    assert!(!constraint.check(1.5), "1.5 must violate [-1, 1]");
    assert!(!constraint.check(-2.0), "-2.0 must violate [-1, 1]");

    // Drive the model with 50 sine samples and apply the constraint as a
    // post-step projection. This proves that a logic-crate constraint can
    // be composed with a model-crate step in a real pipeline without any
    // private API access.
    let signal = synthesize_sine(440.0, 8000.0, 50);
    for &x in signal.iter() {
        let input = Array1::from_vec(vec![x]);
        let y = mamba.step(&input).expect("mamba step must succeed");
        assert!(y.iter().all(|v| v.is_finite()), "outputs must be finite");

        // Apply the constraint's projection on each output dimension.
        let projected: Array1<f32> = y.iter().map(|v| constraint.project(*v)).collect();
        assert!(
            projected.iter().all(|v| (-1.0..=1.0).contains(v)),
            "projected outputs must all be in [-1, 1]"
        );
        assert!(
            projected.iter().all(|v| constraint.check(*v)),
            "projected outputs must all satisfy the constraint"
        );
    }
}

#[tokio::test]
async fn test_streaming_with_filter() {
    use futures::stream::{self, Stream, StreamExt};
    use kizzasi_inference::streaming::{FilterTransformer, StreamTransformer};
    use std::pin::Pin;

    // Build a synthetic stream where every other sample is essentially zero
    // and every other sample is exactly 1.0. The filter should drop the
    // near-zero half.
    let signal: Vec<f32> = (0..100)
        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
        .collect();
    let input_count = signal.len();
    let expected_kept = signal.iter().filter(|&&x| x.abs() >= 1e-3).count();
    assert_eq!(
        expected_kept, 50,
        "test invariant: 50 of 100 samples are above the threshold"
    );

    let input_stream: Pin<Box<dyn Stream<Item = f32> + Send>> = Box::pin(stream::iter(signal));

    let filter = FilterTransformer::new(|x: &f32| x.abs() >= 1e-3);
    let filtered = filter.transform(input_stream);

    let collected: Vec<f32> = filtered.collect().await;

    assert!(
        collected.len() < input_count,
        "filter must drop at least one near-zero sample"
    );
    assert!(
        collected.iter().all(|x| x.abs() >= 1e-3),
        "every surviving sample must clear the threshold"
    );
    assert_eq!(
        collected.len(),
        expected_kept,
        "filter must keep exactly the above-threshold samples"
    );
}
