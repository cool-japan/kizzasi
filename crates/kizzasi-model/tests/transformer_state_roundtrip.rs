//! Transformer KV-cache state round-trip fidelity tests.
//!
//! These tests verify that `get_states` / `set_states` faithfully preserves
//! both the key *and* value caches so that subsequent `step()` calls are
//! numerically identical regardless of whether the KV-cache was saved and
//! restored or left in place.

use kizzasi_core::SignalPredictor;
use kizzasi_model::transformer::{Transformer, TransformerConfig};
use kizzasi_model::AutoregressiveModel;
use scirs2_core::ndarray::Array1;

/// Construct a small Transformer suitable for fast unit tests.
fn small_transformer() -> Transformer {
    Transformer::new(
        TransformerConfig::new()
            .hidden_dim(8)
            .num_heads(2)
            .num_layers(2)
            .max_seq_len(16),
    )
    .expect("failed to build small Transformer")
}

/// Step a model through a slice of scalar inputs.
fn step_many(model: &mut Transformer, inputs: &[f32]) {
    for &v in inputs {
        let input = Array1::from_vec(vec![v]);
        let _ = model.step(&input).expect("step failed");
    }
}

// ---------------------------------------------------------------------------
// Test 1: empty-cache round-trip
// ---------------------------------------------------------------------------

/// A freshly-created Transformer has an empty KV-cache.
/// `get_states` + `set_states` on a fresh model must leave it in a state where
/// the next step produces a finite output.
#[test]
fn test_transformer_empty_cache_roundtrip() {
    let mut model = small_transformer();

    let snapshot = model.get_states();
    assert_eq!(snapshot.len(), 2, "expected one HiddenState per layer");

    model
        .set_states(snapshot)
        .expect("set_states on empty cache must succeed");

    let input = Array1::from_vec(vec![0.5_f32]);
    let output = model
        .step(&input)
        .expect("step after empty-cache roundtrip failed");

    assert!(
        output.iter().all(|&x| x.is_finite()),
        "output after empty-cache roundtrip must be finite, got {:?}",
        output
    );
}

// ---------------------------------------------------------------------------
// Test 2: KV-cache round-trip fidelity
// ---------------------------------------------------------------------------

/// Uses a **single** model:
/// 1. Run 4 steps to build up KV-cache state, then take a snapshot.
/// 2. Immediately step with the probe input → record as `expected_output`.
/// 3. Advance 2 more steps (diverge).
/// 4. Restore the 4-step snapshot.
/// 5. Step with the same probe input → record as `restored_output`.
///
/// `expected_output` and `restored_output` must match to within 1e-5.
/// This confirms that both K *and* V caches are faithfully preserved so the
/// attention computation is identical after restore.
#[test]
fn test_transformer_kv_cache_roundtrip_fidelity() {
    let inputs: Vec<f32> = (0..8).map(|i| 0.05_f32 + i as f32 * 0.07_f32).collect();
    let probe = Array1::from_vec(vec![0.42_f32]);

    let mut model = small_transformer();

    // Build up 4 steps of KV-cache history.
    step_many(&mut model, &inputs[..4]);

    // Snapshot at the 4-step mark.
    let snapshot = model.get_states();

    // Baseline: probe step immediately after the snapshot.
    let expected_output = model.step(&probe).expect("baseline probe step failed");

    // Deviate: two more steps push the KV-cache beyond the snapshot position.
    step_many(&mut model, &inputs[5..7]);

    // Restore the 4-step snapshot and replay the probe.
    model
        .set_states(snapshot)
        .expect("set_states after divergence failed");

    let restored_output = model
        .step(&probe)
        .expect("step after KV-cache restore failed");

    assert_eq!(
        expected_output.len(),
        restored_output.len(),
        "output length mismatch after restore"
    );

    for i in 0..expected_output.len() {
        let diff = (expected_output[i] - restored_output[i]).abs();
        assert!(
            diff < 1e-5_f32,
            "KV-cache round-trip fidelity failure at index {i}: \
             expected {exp} restored {got} diff {diff}",
            exp = expected_output[i],
            got = restored_output[i],
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: wrong state count returns Err
// ---------------------------------------------------------------------------

/// Passing a `states` vector with the wrong number of elements must return `Err`
/// without panicking.
#[test]
fn test_transformer_state_count_mismatch_err() {
    let mut model = small_transformer(); // 2 layers

    // Build a state vec with only 1 element for a 2-layer model.
    let mut one_layer_states = model.get_states();
    one_layer_states.truncate(1);

    let result = model.set_states(one_layer_states);
    assert!(
        result.is_err(),
        "set_states with wrong state count must return Err"
    );

    // Also check the opposite direction: too many states.
    let mut three_layer_states = model.get_states();
    let extra = three_layer_states[0].clone();
    three_layer_states.push(extra);

    let result = model.set_states(three_layer_states);
    assert!(
        result.is_err(),
        "set_states with excess states must return Err"
    );
}

// ---------------------------------------------------------------------------
// Test 4: immediate round-trip preserves output
// ---------------------------------------------------------------------------

/// Run 3 steps to build non-trivial KV-cache state, then:
/// 1. Step once and record the output (baseline).
/// 2. Restore the 3-step snapshot and step again with the same input.
///
/// The two outputs must be bit-for-bit identical (within f32 rounding).
#[test]
fn test_transformer_state_round_trip_values_preserved() {
    let inputs: Vec<f32> = vec![0.1, 0.3, 0.5];
    let probe = Array1::from_vec(vec![0.7_f32]);

    let mut model = small_transformer();
    step_many(&mut model, &inputs);

    // Snapshot BEFORE the probe step.
    let snapshot = model.get_states();

    // Baseline: step with probe.
    let baseline = model.step(&probe).expect("baseline step failed");

    // Restore to the pre-probe snapshot and step again.
    model
        .set_states(snapshot)
        .expect("set_states after immediate snapshot failed");
    let after_restore = model.step(&probe).expect("post-restore step failed");

    assert_eq!(baseline.len(), after_restore.len());
    for i in 0..baseline.len() {
        let diff = (baseline[i] - after_restore[i]).abs();
        assert!(
            diff < 1e-5_f32,
            "value not preserved at index {i}: baseline={b} after_restore={a} diff={diff}",
            b = baseline[i],
            a = after_restore[i],
        );
    }
}
