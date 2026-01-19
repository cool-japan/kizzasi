//! Multi-modal inference with constraint enforcement
//!
//! This example demonstrates:
//! - Multi-modal pipeline with audio, video, and sensor inputs
//! - Constraint enforcement using kizzasi-logic
//! - Different fusion strategies
//! - Real-world AGSP scenario

use kizzasi_inference::{
    EngineConfig, FusionStrategy, InferenceResult, ModalityConfig, ModalityPreprocessor,
    ModalityType, MultiModalPipeline, SamplingConfig, SamplingStrategy,
};
use scirs2_core::ndarray::Array1;
use std::sync::Arc;

fn main() -> InferenceResult<()> {
    println!("=== Multi-Modal Inference Example ===\n");

    // 1. Create modality-specific preprocessors
    let audio_preprocessor: ModalityPreprocessor = Arc::new(|input| {
        // Normalize audio to [-1, 1]
        let max_val = input.iter().cloned().fold(0.0f32, f32::max);
        if max_val > 0.0 {
            Ok(input.mapv(|x| x / max_val))
        } else {
            Ok(input.clone())
        }
    });

    let video_preprocessor: ModalityPreprocessor = Arc::new(|input| {
        // Apply simple smoothing filter
        let mut smoothed = input.clone();
        for i in 1..input.len() - 1 {
            smoothed[i] = (input[i - 1] + input[i] + input[i + 1]) / 3.0;
        }
        Ok(smoothed)
    });

    let sensor_preprocessor: ModalityPreprocessor = Arc::new(|input| {
        // Remove outliers (clamp to [-3, 3])
        Ok(input.mapv(|x| x.clamp(-3.0, 3.0)))
    });

    // 2. Configure modalities with different weights
    let audio_config = ModalityConfig::new(ModalityType::Audio, 16)
        .preprocessor(audio_preprocessor)
        .fusion_weight(2.0); // Audio is most important

    let video_config = ModalityConfig::new(ModalityType::Video, 16)
        .preprocessor(video_preprocessor)
        .fusion_weight(1.5);

    let sensor_config = ModalityConfig::new(ModalityType::Sensor, 16)
        .preprocessor(sensor_preprocessor)
        .fusion_weight(1.0);

    // 3. Create engine with sampling
    let engine_config = EngineConfig::new(48, 32)
        .sampling(
            SamplingConfig::new()
                .strategy(SamplingStrategy::TopP)
                .top_p(0.9)
                .temperature(0.8),
        )
        .use_embeddings(true);

    // 4. Test different fusion strategies
    let fusion_strategies = vec![
        FusionStrategy::EarlyFusion,
        FusionStrategy::WeightedFusion,
        FusionStrategy::CrossAttention,
        FusionStrategy::MaxPooling,
    ];

    for strategy in fusion_strategies {
        println!("Testing fusion strategy: {:?}", strategy);

        let mut pipeline = MultiModalPipeline::builder()
            .engine_config(engine_config.clone())
            .add_modality(audio_config.clone())
            .add_modality(video_config.clone())
            .add_modality(sensor_config.clone())
            .fusion_strategy(strategy)
            .build()?;

        // Simulate multi-modal inputs
        let audio_input = Array1::from_vec(vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
        ]);

        let video_input = Array1::from_vec(vec![
            0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        ]);

        let sensor_input = Array1::from_vec(vec![
            0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
        ]);

        // Forward pass
        let output = pipeline.forward(&[
            (ModalityType::Audio, audio_input),
            (ModalityType::Video, video_input),
            (ModalityType::Sensor, sensor_input),
        ])?;

        println!("  Output shape: {}", output.len());
        println!(
            "  Output range: [{:.3}, {:.3}]",
            output.iter().cloned().fold(f32::INFINITY, f32::min),
            output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        println!();
    }

    // 5. Multi-step rollout with different modalities
    println!("=== Multi-Step Rollout ===\n");

    let mut pipeline = MultiModalPipeline::builder()
        .engine_config(engine_config)
        .add_modality(audio_config)
        .add_modality(video_config)
        .add_modality(sensor_config)
        .fusion_strategy(FusionStrategy::CrossAttention)
        .build()?;

    let audio = Array1::from_elem(16, 0.3);
    let video = Array1::from_elem(16, 0.6);
    let sensor = Array1::from_elem(16, 0.9);

    println!("Running 5-step rollout...");
    for step in 0..5 {
        let output = pipeline.forward(&[
            (ModalityType::Audio, audio.clone()),
            (ModalityType::Video, video.clone()),
            (ModalityType::Sensor, sensor.clone()),
        ])?;

        println!("Step {}: ", step + 1);
        println!(
            "  Output range: [{:.3}, {:.3}]",
            output.iter().cloned().fold(f32::INFINITY, f32::min),
            output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
