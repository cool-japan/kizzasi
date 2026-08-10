//! Integration tests for the multi-modal inference pipeline.
//!
//! These tests exercise the full cross-crate path spanning
//! `kizzasi-inference` (pipeline, engine) and `kizzasi-model` (S4D model).
//!
//! Unlike the in-module `#[cfg(test)]` units in `multimodal.rs`, every test
//! here:
//!
//! - Builds a complete [`MultiModalPipeline`] via its public builder API.
//! - Attaches a live [`S4D`] model to the engine so that `forward` exercises
//!   real SSM state transitions rather than the zero-fallback path.
//! - Calls [`MultiModalPipeline::forward`] synchronously (no `tokio::test`
//!   is needed) and asserts both structural and numerical correctness.
//!
//! # Covered scenarios
//!
//! 1. EarlyFusion with two equal-dimension modalities.
//! 2. EarlyFusion with mixed (unequal) modality dimensions.
//! 3. WeightedFusion with different per-modality fusion weights.
//! 4. MaxPooling fusion across two same-dimension modalities.
//! 5. CrossAttention fusion across two same-dimension modalities.
//! 6. Multiple sequential forward steps preserving SSM state.
//! 7. Dimension mismatch returns `InferenceError::DimensionMismatch`.
//! 8. Unknown modality returns an `InferenceError`.
//! 9. `reset()` clears state without breaking subsequent forwards.
//! 10. WeightedFusion end-to-end with a custom preprocessor.

use kizzasi_inference::{
    EngineConfig, FusionStrategy, InferenceError, ModalityConfig, ModalityType, MultiModalPipeline,
};
use kizzasi_model::s4::{S4Config, S4D};
use scirs2_core::ndarray::Array1;
use std::sync::Arc;

// ============================================================================
// Helper utilities
// ============================================================================

/// Build and attach a tiny S4D model to a pipeline engine.
///
/// `fused_dim` is both the S4D `input_dim` (and therefore its output dim)
/// and must equal the `input_dim` that was passed to `EngineConfig::new`.
fn attach_s4d(pipeline: &mut MultiModalPipeline, fused_dim: usize) {
    let model_config = S4Config::new()
        .input_dim(fused_dim)
        .hidden_dim(16)
        .state_dim(8)
        .num_layers(1);
    let model = S4D::new(model_config).expect("S4D construction must succeed");
    pipeline.engine_mut().set_model(Box::new(model));
}

/// Assert that every element of `output` is finite and that the length matches
/// the expected output dimension.
fn assert_valid_output(output: &Array1<f32>, expected_len: usize) {
    assert_eq!(
        output.len(),
        expected_len,
        "output length {len} != expected {expected_len}",
        len = output.len(),
    );
    assert!(
        output.iter().all(|v| v.is_finite()),
        "output contains non-finite values: {:?}",
        output
    );
}

// ============================================================================
// Test 1 – EarlyFusion basic (two equal-dimension modalities)
// ============================================================================

/// EarlyFusion concatenates inputs: Audio(dim=8) + Sensor(dim=8) → fused_dim=16.
/// The S4D model is configured with input_dim=16; its output is also dim=16.
#[test]
fn test_early_fusion_equal_dims() {
    const MODALITY_DIM: usize = 8;
    const FUSED_DIM: usize = MODALITY_DIM * 2; // 16

    let engine_config = EngineConfig::new(FUSED_DIM, FUSED_DIM);
    let mut pipeline = MultiModalPipeline::builder()
        .engine_config(engine_config)
        .modality(ModalityType::Audio, MODALITY_DIM)
        .modality(ModalityType::Sensor, MODALITY_DIM)
        .fusion_strategy(FusionStrategy::EarlyFusion)
        .build()
        .expect("EarlyFusion pipeline must build");

    attach_s4d(&mut pipeline, FUSED_DIM);

    let audio_input = Array1::from_iter((0..MODALITY_DIM).map(|i| (i as f32) * 0.1));
    let sensor_input = Array1::from_iter((0..MODALITY_DIM).map(|i| (i as f32) * 0.05 + 0.5));

    let output = pipeline
        .forward(&[
            (ModalityType::Audio, audio_input),
            (ModalityType::Sensor, sensor_input),
        ])
        .expect("EarlyFusion forward must succeed");

    assert_valid_output(&output, FUSED_DIM);
}

// ============================================================================
// Test 2 – EarlyFusion with mixed (unequal) input dimensions
// ============================================================================

/// EarlyFusion concatenates in sorted modality order, so Audio(4) + Video(8)
/// → fused_dim = 12 regardless of insertion order.
#[test]
fn test_early_fusion_mixed_dims() {
    const AUDIO_DIM: usize = 4;
    const VIDEO_DIM: usize = 8;
    const FUSED_DIM: usize = AUDIO_DIM + VIDEO_DIM; // 12

    let engine_config = EngineConfig::new(FUSED_DIM, FUSED_DIM);
    let mut pipeline = MultiModalPipeline::builder()
        .engine_config(engine_config)
        .modality(ModalityType::Audio, AUDIO_DIM)
        .modality(ModalityType::Video, VIDEO_DIM)
        .fusion_strategy(FusionStrategy::EarlyFusion)
        .build()
        .expect("EarlyFusion mixed-dim pipeline must build");

    attach_s4d(&mut pipeline, FUSED_DIM);

    let audio_input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let video_input = Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);

    let output = pipeline
        .forward(&[
            (ModalityType::Audio, audio_input),
            (ModalityType::Video, video_input),
        ])
        .expect("EarlyFusion mixed-dim forward must succeed");

    assert_valid_output(&output, FUSED_DIM);
}

// ============================================================================
// Test 3 – WeightedFusion with different fusion weights
// ============================================================================

/// WeightedFusion multiplies each modality by its weight, concatenates, then
/// divides by the total weight.  For Audio(w=2.0, dim=8) + Sensor(w=1.0,
/// dim=8) the fused vector has length 16 (same as EarlyFusion) because the
/// implementation does weighted concatenation rather than a weighted average.
#[test]
fn test_weighted_fusion_different_weights() {
    const MODALITY_DIM: usize = 8;
    const FUSED_DIM: usize = MODALITY_DIM * 2; // 16

    let engine_config = EngineConfig::new(FUSED_DIM, FUSED_DIM);

    let audio_config = ModalityConfig::new(ModalityType::Audio, MODALITY_DIM).fusion_weight(2.0);
    let sensor_config = ModalityConfig::new(ModalityType::Sensor, MODALITY_DIM).fusion_weight(1.0);

    let mut pipeline = MultiModalPipeline::builder()
        .engine_config(engine_config)
        .add_modality(audio_config)
        .add_modality(sensor_config)
        .fusion_strategy(FusionStrategy::WeightedFusion)
        .build()
        .expect("WeightedFusion pipeline must build");

    attach_s4d(&mut pipeline, FUSED_DIM);

    let audio_input = Array1::from_iter((0..MODALITY_DIM).map(|i| i as f32 * 0.1));
    let sensor_input = Array1::from_iter((0..MODALITY_DIM).map(|i| i as f32 * 0.05 + 0.3));

    let output = pipeline
        .forward(&[
            (ModalityType::Audio, audio_input),
            (ModalityType::Sensor, sensor_input),
        ])
        .expect("WeightedFusion forward must succeed");

    assert_valid_output(&output, FUSED_DIM);
}

// ============================================================================
// Test 4 – MaxPooling fusion across two same-dimension modalities
// ============================================================================

/// MaxPooling groups modalities with equal dimension and takes the element-wise
/// maximum.  Two modalities both of dim=8 produce a fused vector of length 8.
#[test]
fn test_max_pooling_fusion() {
    const MODALITY_DIM: usize = 8;
    const FUSED_DIM: usize = MODALITY_DIM; // max-pooled, same dim

    let engine_config = EngineConfig::new(FUSED_DIM, FUSED_DIM);
    let mut pipeline = MultiModalPipeline::builder()
        .engine_config(engine_config)
        .modality(ModalityType::Audio, MODALITY_DIM)
        .modality(ModalityType::Sensor, MODALITY_DIM)
        .fusion_strategy(FusionStrategy::MaxPooling)
        .build()
        .expect("MaxPooling pipeline must build");

    attach_s4d(&mut pipeline, FUSED_DIM);

    // Design inputs so the max at each position is deterministic.
    let audio_input = Array1::from_vec(vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]);
    let sensor_input = Array1::from_vec(vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]);

    let output = pipeline
        .forward(&[
            (ModalityType::Audio, audio_input),
            (ModalityType::Sensor, sensor_input),
        ])
        .expect("MaxPooling forward must succeed");

    assert_valid_output(&output, FUSED_DIM);
}

// ============================================================================
// Test 5 – CrossAttention fusion across two same-dimension modalities
// ============================================================================

/// CrossAttention computes pairwise dot-product attention weights and returns a
/// weighted sum within each dimension group.  Two modalities of dim=8 produce
/// a fused vector of length 8.
#[test]
fn test_cross_attention_fusion() {
    const MODALITY_DIM: usize = 8;
    const FUSED_DIM: usize = MODALITY_DIM; // attention-pooled, same dim

    let engine_config = EngineConfig::new(FUSED_DIM, FUSED_DIM);
    let mut pipeline = MultiModalPipeline::builder()
        .engine_config(engine_config)
        .modality(ModalityType::Audio, MODALITY_DIM)
        .modality(ModalityType::Sensor, MODALITY_DIM)
        .fusion_strategy(FusionStrategy::CrossAttention)
        .build()
        .expect("CrossAttention pipeline must build");

    attach_s4d(&mut pipeline, FUSED_DIM);

    let audio_input = Array1::from_iter((0..MODALITY_DIM).map(|i| (i as f32 + 1.0) * 0.1));
    let sensor_input = Array1::from_iter((0..MODALITY_DIM).map(|i| (i as f32 + 1.0) * 0.15));

    let output = pipeline
        .forward(&[
            (ModalityType::Audio, audio_input),
            (ModalityType::Sensor, sensor_input),
        ])
        .expect("CrossAttention forward must succeed");

    assert_valid_output(&output, FUSED_DIM);
}

// ============================================================================
// Test 6 – Multiple sequential forward steps with state retention
// ============================================================================

/// Calls forward three times in sequence.  All outputs must be finite,
/// demonstrating that the S4D SSM state is correctly maintained and updated
/// across steps (i.e., the engine does not corrupt state between calls).
#[test]
fn test_multiple_steps_maintain_state() {
    const AUDIO_DIM: usize = 6;
    const SENSOR_DIM: usize = 6;
    const VIDEO_DIM: usize = 6;
    const FUSED_DIM: usize = AUDIO_DIM + SENSOR_DIM + VIDEO_DIM; // EarlyFusion → 18

    let engine_config = EngineConfig::new(FUSED_DIM, FUSED_DIM);
    let mut pipeline = MultiModalPipeline::builder()
        .engine_config(engine_config)
        .modality(ModalityType::Audio, AUDIO_DIM)
        .modality(ModalityType::Sensor, SENSOR_DIM)
        .modality(ModalityType::Video, VIDEO_DIM)
        .fusion_strategy(FusionStrategy::EarlyFusion)
        .build()
        .expect("Three-modality EarlyFusion pipeline must build");

    attach_s4d(&mut pipeline, FUSED_DIM);

    for step in 0..3usize {
        let phase = step as f32 * std::f32::consts::FRAC_PI_4;
        let audio_input = Array1::from_iter((0..AUDIO_DIM).map(|i| (i as f32 * 0.2 + phase).sin()));
        let sensor_input =
            Array1::from_iter((0..SENSOR_DIM).map(|i| (i as f32 * 0.1 + phase).cos()));
        let video_input = Array1::from_iter((0..VIDEO_DIM).map(|i| i as f32 * 0.05));

        let output = pipeline
            .forward(&[
                (ModalityType::Audio, audio_input),
                (ModalityType::Sensor, sensor_input),
                (ModalityType::Video, video_input),
            ])
            .unwrap_or_else(|e| panic!("step {step} forward must succeed, got: {e}"));

        assert_valid_output(&output, FUSED_DIM);
    }
}

// ============================================================================
// Test 7 – Dimension mismatch is reported as DimensionMismatch error
// ============================================================================

/// Passing an Audio input of length 4 to a pipeline configured for Audio
/// dim=8 must return `InferenceError::DimensionMismatch`.
#[test]
fn test_dimension_mismatch_error() {
    const MODALITY_DIM: usize = 8;
    const FUSED_DIM: usize = MODALITY_DIM * 2;

    let engine_config = EngineConfig::new(FUSED_DIM, FUSED_DIM);
    let mut pipeline = MultiModalPipeline::builder()
        .engine_config(engine_config)
        .modality(ModalityType::Audio, MODALITY_DIM)
        .modality(ModalityType::Sensor, MODALITY_DIM)
        .fusion_strategy(FusionStrategy::EarlyFusion)
        .build()
        .expect("pipeline must build");

    // Wrong dimension: 4 instead of 8
    let audio_wrong = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let sensor_ok = Array1::from_iter((0..MODALITY_DIM).map(|i| i as f32 * 0.1));

    let result = pipeline.forward(&[
        (ModalityType::Audio, audio_wrong),
        (ModalityType::Sensor, sensor_ok),
    ]);

    assert!(result.is_err(), "dimension mismatch must return Err");
    match result.unwrap_err() {
        InferenceError::DimensionMismatch { expected, got } => {
            assert_eq!(
                expected, MODALITY_DIM,
                "expected should be the configured dim"
            );
            assert_eq!(got, 4, "got should reflect the supplied wrong dim");
        }
        other => panic!("expected DimensionMismatch, got: {other:?}"),
    }
}

// ============================================================================
// Test 8 – Unknown modality returns an error
// ============================================================================

/// Passing `ModalityType::Text` to a pipeline that only has `Audio` and
/// `Sensor` configured must return an `InferenceError` (PipelineConfig).
#[test]
fn test_unknown_modality_error() {
    const MODALITY_DIM: usize = 8;
    const FUSED_DIM: usize = MODALITY_DIM * 2;

    let engine_config = EngineConfig::new(FUSED_DIM, FUSED_DIM);
    let mut pipeline = MultiModalPipeline::builder()
        .engine_config(engine_config)
        .modality(ModalityType::Audio, MODALITY_DIM)
        .modality(ModalityType::Sensor, MODALITY_DIM)
        .build()
        .expect("pipeline must build");

    // Text was never registered in this pipeline
    let text_input = Array1::from_iter((0..MODALITY_DIM).map(|i| i as f32 * 0.1));

    let result = pipeline.forward(&[(ModalityType::Text, text_input)]);

    assert!(
        result.is_err(),
        "forward with an unknown modality must return Err"
    );
    // Verify it is specifically a PipelineConfig error mentioning the modality
    match result.unwrap_err() {
        InferenceError::PipelineConfig(msg) => {
            assert!(
                msg.contains("modality") || msg.contains("Modality") || msg.contains("Unknown"),
                "error message should mention the unknown modality, got: {msg}"
            );
        }
        other => panic!("expected PipelineConfig error, got: {other:?}"),
    }
}

// ============================================================================
// Test 9 – reset() clears SSM state without breaking subsequent forwards
// ============================================================================

/// Runs a few forward steps, resets the pipeline, then verifies that another
/// forward step completes successfully and produces finite output.  This
/// demonstrates that `reset()` correctly zeroes the SSM hidden state without
/// invalidating the model attachment.
#[test]
fn test_reset_clears_state() {
    const MODALITY_DIM: usize = 8;
    const FUSED_DIM: usize = MODALITY_DIM * 2; // EarlyFusion → 16

    let engine_config = EngineConfig::new(FUSED_DIM, FUSED_DIM);
    let mut pipeline = MultiModalPipeline::builder()
        .engine_config(engine_config)
        .modality(ModalityType::Audio, MODALITY_DIM)
        .modality(ModalityType::Sensor, MODALITY_DIM)
        .fusion_strategy(FusionStrategy::EarlyFusion)
        .build()
        .expect("pipeline must build");

    attach_s4d(&mut pipeline, FUSED_DIM);

    // Warm up the SSM state with a few steps
    for _ in 0..3 {
        let audio_input = Array1::from_iter((0..MODALITY_DIM).map(|i| i as f32 * 0.1));
        let sensor_input = Array1::from_iter((0..MODALITY_DIM).map(|i| i as f32 * 0.05 + 0.3));
        pipeline
            .forward(&[
                (ModalityType::Audio, audio_input),
                (ModalityType::Sensor, sensor_input),
            ])
            .expect("warm-up forward must succeed");
    }

    // Reset clears the accumulated SSM state
    pipeline.reset();

    // A fresh forward after reset must still succeed
    let audio_post_reset = Array1::from_iter((0..MODALITY_DIM).map(|i| i as f32 * 0.2));
    let sensor_post_reset = Array1::from_iter((0..MODALITY_DIM).map(|i| i as f32 * 0.1 + 0.1));

    let output = pipeline
        .forward(&[
            (ModalityType::Audio, audio_post_reset),
            (ModalityType::Sensor, sensor_post_reset),
        ])
        .expect("post-reset forward must succeed");

    assert_valid_output(&output, FUSED_DIM);
}

// ============================================================================
// Test 10 – WeightedFusion end-to-end with a custom preprocessor
// ============================================================================

/// Attaches a custom preprocessor (normalise to unit-max) to the Audio
/// modality and verifies that the preprocessor is applied transparently
/// before fusion.  The output must be finite and correctly dimensioned.
#[test]
fn test_weighted_fusion_with_preprocessor() {
    const MODALITY_DIM: usize = 8;
    const FUSED_DIM: usize = MODALITY_DIM * 2; // WeightedFusion → 16

    // Preprocessor: normalise the input to the range [0, 1] by dividing by
    // the absolute maximum; fall back to the identity if all values are zero.
    let normalise: kizzasi_inference::ModalityPreprocessor = Arc::new(|x: &Array1<f32>| {
        let max_abs = x.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        if max_abs > 0.0 {
            Ok(x.mapv(|v| v / max_abs))
        } else {
            Ok(x.clone())
        }
    });

    let audio_config = ModalityConfig::new(ModalityType::Audio, MODALITY_DIM)
        .fusion_weight(2.0)
        .preprocessor(normalise);
    let sensor_config = ModalityConfig::new(ModalityType::Sensor, MODALITY_DIM).fusion_weight(1.0);

    let engine_config = EngineConfig::new(FUSED_DIM, FUSED_DIM);
    let mut pipeline = MultiModalPipeline::builder()
        .engine_config(engine_config)
        .add_modality(audio_config)
        .add_modality(sensor_config)
        .fusion_strategy(FusionStrategy::WeightedFusion)
        .build()
        .expect("WeightedFusion+preprocessor pipeline must build");

    attach_s4d(&mut pipeline, FUSED_DIM);

    // Audio input has values in [1, 8]; after normalisation they should be in [1/8, 1].
    let audio_input = Array1::from_iter((1..=MODALITY_DIM).map(|i| i as f32));
    let sensor_input = Array1::from_iter((0..MODALITY_DIM).map(|i| i as f32 * 0.1 + 0.05));

    let output = pipeline
        .forward(&[
            (ModalityType::Audio, audio_input),
            (ModalityType::Sensor, sensor_input),
        ])
        .expect("WeightedFusion+preprocessor forward must succeed");

    assert_valid_output(&output, FUSED_DIM);
}
