//! Integration tests for speculative decoding across real model architectures.
//!
//! These tests exercise the full cross-crate path spanning
//! `kizzasi-inference` (speculative decoder) and `kizzasi-model` (S4D, Mamba).
//!
//! Unlike the in-module `#[cfg(test)]` units in `speculative.rs`, every test
//! here:
//!
//! - Constructs live [`S4D`] (and optionally [`Mamba`]) models via their
//!   public builder APIs.
//! - Feeds models into [`SpeculativeDecoder`] so that `generate` exercises
//!   real SSM state transitions and draft/verify token selection rather than
//!   any mock fallback path.
//! - Asserts both structural and numerical correctness of the generated
//!   sequence.
//!
//! # Covered scenarios
//!
//! 1. Token count equals `max_tokens` after generation.
//! 2. All generated token values are finite (no NaN / Inf).
//! 3. Acceptance rate is in `[0.0, 1.0]` after generation.
//! 4. Identical main/draft models with greedy verification yield acceptance
//!    rate of ~1.0.
//! 5. `reset_stats` zeroes the acceptance-rate numerator/denominator.
//! 6. Different `num_draft_tokens` values both produce correct output lengths.
//! 7. The full config builder chain produces the expected field values.
//! 8. `max_tokens = 0` returns an empty sequence.
//! 9. `max_tokens = 1` returns exactly one finite token.
//! 10. Mixed architecture (S4D main + Mamba draft) still generates the
//!     correct number of tokens.

use kizzasi_inference::{SpeculativeConfig, SpeculativeDecoder};
#[cfg(feature = "mamba")]
use kizzasi_model::mamba::{Mamba, MambaConfig};
use kizzasi_model::s4::{S4Config, S4D};
use kizzasi_model::AutoregressiveModel;
use scirs2_core::ndarray::Array1;

// ============================================================================
// Helper utilities
// ============================================================================

/// Build a tiny S4D model whose input dimension is `input_dim`.
///
/// `input_dim` must be 1 for speculative decoding, because the decoder feeds
/// each sampled scalar token (a length-1 array) as the next model input.
fn make_s4d(input_dim: usize) -> Box<dyn AutoregressiveModel> {
    let config = S4Config::new()
        .input_dim(input_dim)
        .hidden_dim(16)
        .state_dim(8)
        .num_layers(1);
    Box::new(S4D::new(config).expect("S4D construction must succeed"))
}

/// Build the standard scalar-input S4D used across most speculative tests.
fn make_scalar_s4d() -> Box<dyn AutoregressiveModel> {
    make_s4d(1)
}

/// Assert that a generated token array has the expected shape and all
/// elements are finite.
fn assert_token_valid(token: &Array1<f32>, expected_len: usize) {
    assert_eq!(
        token.len(),
        expected_len,
        "token length {len} != expected {expected_len}",
        len = token.len(),
    );
    assert!(
        token.iter().all(|v| v.is_finite()),
        "token contains non-finite values: {:?}",
        token
    );
}

// ============================================================================
// Test 1 – Generated sequence has exactly `max_tokens` elements
// ============================================================================

/// Two S4D(input_dim=1) models in a speculative decoder; `generate` with
/// `max_tokens=8` must return a `Vec` of exactly 8 tokens.
#[test]
fn test_speculative_generates_correct_token_count() {
    let main_model = make_scalar_s4d();
    let draft_model = make_scalar_s4d();
    let config = SpeculativeConfig::new().num_draft_tokens(4);

    let mut dec = SpeculativeDecoder::new(main_model, draft_model, config);

    let input = Array1::from_vec(vec![0.5]);
    let output = dec.generate(&input, 8).expect("generate must succeed");

    assert_eq!(
        output.len(),
        8,
        "sequence length {len} != 8",
        len = output.len()
    );
}

// ============================================================================
// Test 2 – All generated token values are finite
// ============================================================================

/// Each element in the generated sequence must be a finite scalar; NaN or
/// Inf indicates a numerical blow-up in the SSM forward pass.
#[test]
fn test_speculative_output_values_finite() {
    let main_model = make_scalar_s4d();
    let draft_model = make_scalar_s4d();
    let config = SpeculativeConfig::new().num_draft_tokens(4);

    let mut dec = SpeculativeDecoder::new(main_model, draft_model, config);

    let input = Array1::from_vec(vec![0.5]);
    let output = dec.generate(&input, 8).expect("generate must succeed");

    for (i, token) in output.iter().enumerate() {
        assert_token_valid(token, 1);
        assert!(
            token[0].is_finite(),
            "token[{i}][0] = {} is not finite",
            token[0]
        );
    }
}

// ============================================================================
// Test 3 – Acceptance rate is in [0.0, 1.0] after generation
// ============================================================================

/// After at least one generate call the acceptance rate must lie inside
/// `[0.0, 1.0]`; values outside this interval indicate a statistics bug.
#[test]
fn test_speculative_acceptance_rate_valid_range() {
    let main_model = make_scalar_s4d();
    let draft_model = make_scalar_s4d();
    let config = SpeculativeConfig::new().num_draft_tokens(4);

    let mut dec = SpeculativeDecoder::new(main_model, draft_model, config);

    let input = Array1::from_vec(vec![0.5]);
    dec.generate(&input, 8).expect("generate must succeed");

    let rate = dec.acceptance_rate();
    assert!(
        (0.0..=1.0).contains(&rate),
        "acceptance_rate = {rate} is outside [0.0, 1.0]"
    );
}

// ============================================================================
// Test 4 – Identical models + greedy verification → acceptance rate ≈ 1.0
// ============================================================================

/// When main and draft models are constructed with the exact same
/// `S4Config` their initial weights are identical (deterministic
/// initialization).  With `greedy_verification=true` the draft and main
/// predictions therefore always agree, so the acceptance rate should be
/// 1.0 (tolerance 0.01).
#[test]
fn test_speculative_acceptance_rate_identical_models() {
    let shared_config = S4Config::new()
        .input_dim(1)
        .hidden_dim(16)
        .state_dim(8)
        .num_layers(1);

    let main_model: Box<dyn AutoregressiveModel> =
        Box::new(S4D::new(shared_config.clone()).expect("S4D construction must succeed"));
    let draft_model: Box<dyn AutoregressiveModel> =
        Box::new(S4D::new(shared_config).expect("S4D construction must succeed"));

    let config = SpeculativeConfig::new()
        .num_draft_tokens(4)
        .greedy_verification(true);

    let mut dec = SpeculativeDecoder::new(main_model, draft_model, config);

    let input = Array1::from_vec(vec![0.5]);
    dec.generate(&input, 8).expect("generate must succeed");

    let rate = dec.acceptance_rate();
    assert!(
        rate >= 0.99,
        "identical models + greedy: expected acceptance_rate ≈ 1.0, got {rate}"
    );
}

// ============================================================================
// Test 5 – reset_stats clears acceptance rate back to 0.0
// ============================================================================

/// After generating tokens the stats accumulator is non-empty; calling
/// `reset_stats` must zero both the numerator and denominator so that
/// `acceptance_rate()` returns `0.0` (the value for zero total tokens).
#[test]
fn test_speculative_reset_stats() {
    let main_model = make_scalar_s4d();
    let draft_model = make_scalar_s4d();
    let config = SpeculativeConfig::new().num_draft_tokens(4);

    let mut dec = SpeculativeDecoder::new(main_model, draft_model, config);

    let input = Array1::from_vec(vec![0.5]);
    dec.generate(&input, 6).expect("generate must succeed");

    dec.reset_stats();

    assert_eq!(
        dec.acceptance_rate(),
        0.0,
        "acceptance_rate after reset_stats must be 0.0"
    );
}

// ============================================================================
// Test 6 – Different num_draft_tokens values both produce correct lengths
// ============================================================================

/// Whether the draft model proposes 1 or 4 candidates per round, the
/// total generated sequence length must equal `max_tokens` in both cases.
#[test]
fn test_speculative_different_draft_token_counts() {
    let max_tokens = 8_usize;
    let input = Array1::from_vec(vec![0.5]);

    for &num_draft in &[1_usize, 4_usize] {
        let main_model = make_scalar_s4d();
        let draft_model = make_scalar_s4d();
        let config = SpeculativeConfig::new().num_draft_tokens(num_draft);

        let mut dec = SpeculativeDecoder::new(main_model, draft_model, config);
        let output = dec
            .generate(&input, max_tokens)
            .unwrap_or_else(|e| panic!("generate with num_draft_tokens={num_draft} failed: {e}"));

        assert_eq!(
            output.len(),
            max_tokens,
            "num_draft_tokens={num_draft}: expected {max_tokens} tokens, got {}",
            output.len()
        );
    }
}

// ============================================================================
// Test 7 – Full config builder chain produces correct field values
// ============================================================================

/// Every builder method must survive the chain and set the corresponding
/// public field on the stored config to the exact value passed in.
#[test]
fn test_speculative_config_builder_chain() {
    let main_model = make_scalar_s4d();
    let draft_model = make_scalar_s4d();

    let built_config = SpeculativeConfig::new()
        .num_draft_tokens(3)
        .draft_temperature(0.8)
        .main_temperature(1.0)
        .greedy_verification(false);

    let dec = SpeculativeDecoder::new(main_model, draft_model, built_config);

    let cfg = dec.config();
    assert_eq!(cfg.num_draft_tokens, 3, "num_draft_tokens should be 3");
    assert!(
        (cfg.draft_temperature - 0.8).abs() < 1e-6,
        "draft_temperature should be 0.8, got {}",
        cfg.draft_temperature
    );
    assert!(
        (cfg.main_temperature - 1.0).abs() < 1e-6,
        "main_temperature should be 1.0, got {}",
        cfg.main_temperature
    );
    assert!(
        !cfg.greedy_verification,
        "greedy_verification should be false"
    );
}

// ============================================================================
// Test 8 – max_tokens = 0 returns an empty sequence
// ============================================================================

/// Requesting zero tokens must produce an empty `Vec`, not an error.
#[test]
fn test_speculative_empty_generation() {
    let main_model = make_scalar_s4d();
    let draft_model = make_scalar_s4d();
    let config = SpeculativeConfig::new().num_draft_tokens(4);

    let mut dec = SpeculativeDecoder::new(main_model, draft_model, config);

    let input = Array1::from_vec(vec![0.5]);
    let output = dec.generate(&input, 0).expect("generate(0) must succeed");

    assert!(
        output.is_empty(),
        "max_tokens=0 must produce empty vec, got {} tokens",
        output.len()
    );
}

// ============================================================================
// Test 9 – max_tokens = 1 returns exactly one finite token
// ============================================================================

/// Generating a single token must return a `Vec` of length 1 whose sole
/// element is a scalar array `[v]` with `v.is_finite()`.
#[test]
fn test_speculative_single_token_generation() {
    let main_model = make_scalar_s4d();
    let draft_model = make_scalar_s4d();
    let config = SpeculativeConfig::new().num_draft_tokens(4);

    let mut dec = SpeculativeDecoder::new(main_model, draft_model, config);

    let input = Array1::from_vec(vec![0.5]);
    let output = dec.generate(&input, 1).expect("generate(1) must succeed");

    assert_eq!(
        output.len(),
        1,
        "max_tokens=1 must produce exactly 1 token, got {}",
        output.len()
    );
    assert_token_valid(&output[0], 1);
}

// ============================================================================
// Test 10 – Mixed architecture: S4D main + Mamba draft
// ============================================================================

/// Using an S4D model as main and a Mamba model as draft (both with
/// `input_dim=1`) exercises the cross-architecture speculative decoding
/// path.  The decoder must still generate the requested number of tokens
/// and all values must be finite.
#[cfg(feature = "mamba")]
#[test]
fn test_speculative_with_mamba_draft() {
    let main_model: Box<dyn AutoregressiveModel> = make_scalar_s4d();

    let draft_config = MambaConfig::new()
        .input_dim(1)
        .hidden_dim(16)
        .state_dim(8)
        .num_layers(1);
    let draft_model: Box<dyn AutoregressiveModel> =
        Box::new(Mamba::new(draft_config).expect("Mamba construction must succeed"));

    let config = SpeculativeConfig::new()
        .num_draft_tokens(4)
        .greedy_verification(false);

    let mut dec = SpeculativeDecoder::new(main_model, draft_model, config);

    let input = Array1::from_vec(vec![0.5]);
    let output = dec
        .generate(&input, 8)
        .expect("mixed-architecture generate must succeed");

    assert_eq!(
        output.len(),
        8,
        "mixed-architecture: expected 8 tokens, got {}",
        output.len()
    );

    for (i, token) in output.iter().enumerate() {
        assert_token_valid(token, 1);
        assert!(
            token[0].is_finite(),
            "mixed-arch token[{i}][0] = {} is not finite",
            token[0]
        );
    }
}
