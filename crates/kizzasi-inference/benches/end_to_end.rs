//! End-to-end benchmarks for kizzasi-inference
//!
//! Comprehensive benchmarks covering:
//! - Single-step inference
//! - Multi-step rollout
//! - Batch processing
//! - Streaming inference
//! - Multi-modal pipelines
//! - Ensemble models
//! - Speculative decoding

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_inference::{
    BatchConfig, BatchScheduler, EngineConfig, EnsembleBuilder, EnsembleStrategy, FusionStrategy,
    InferenceEngine, ModalityType, MultiModalPipeline, PipelineBuilder, SamplingConfig,
    SamplingStrategy, SpeculativeConfig, SpeculativeDecoder,
};
use kizzasi_model::rwkv::{Rwkv, RwkvConfig};
use kizzasi_model::s4::{S4Config, S4D};
use kizzasi_model::transformer::{Transformer, TransformerConfig};
use scirs2_core::ndarray::Array1;
use std::hint::black_box;

/// Benchmark single-step inference with different models
fn bench_single_step_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_step_inference");

    // S4D model
    let s4_config = S4Config::new()
        .input_dim(64)
        .hidden_dim(256)
        .state_dim(64)
        .num_layers(4)
        .diagonal(true);
    let s4_model = S4D::new(s4_config).unwrap();
    let mut s4_engine = InferenceEngine::with_model(EngineConfig::new(64, 64), Box::new(s4_model));

    // RWKV model
    let rwkv_config = RwkvConfig::new()
        .input_dim(64)
        .hidden_dim(256)
        .intermediate_dim(512)
        .num_layers(4);
    let rwkv_model = Rwkv::new(rwkv_config).unwrap();
    let mut rwkv_engine =
        InferenceEngine::with_model(EngineConfig::new(64, 64), Box::new(rwkv_model));

    // Transformer model
    let transformer_config = TransformerConfig::new()
        .input_dim(64)
        .hidden_dim(256)
        .num_heads(8)
        .num_layers(4)
        .max_seq_len(128);
    let transformer_model = Transformer::new(transformer_config).unwrap();
    let mut transformer_engine =
        InferenceEngine::with_model(EngineConfig::new(64, 64), Box::new(transformer_model));

    let input = Array1::from_elem(64, 0.5);

    group.throughput(Throughput::Elements(1));

    group.bench_function("s4d", |b| {
        b.iter(|| s4_engine.step(black_box(&input)).unwrap())
    });

    group.bench_function("rwkv", |b| {
        b.iter(|| rwkv_engine.step(black_box(&input)).unwrap())
    });

    group.bench_function("transformer", |b| {
        b.iter(|| transformer_engine.step(black_box(&input)).unwrap())
    });

    group.finish();
}

/// Benchmark multi-step rollout with different sequence lengths
fn bench_rollout(c: &mut Criterion) {
    let mut group = c.benchmark_group("rollout");

    let s4_config = S4Config::new()
        .input_dim(32)
        .hidden_dim(128)
        .state_dim(32)
        .num_layers(2)
        .diagonal(true);

    let input = Array1::from_elem(32, 0.5);

    for steps in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*steps as u64));
        group.bench_with_input(BenchmarkId::from_parameter(steps), steps, |b, &steps| {
            b.iter(|| {
                let s4_model = S4D::new(s4_config.clone()).unwrap();
                let mut engine =
                    InferenceEngine::with_model(EngineConfig::new(32, 32), Box::new(s4_model));
                let mut current = input.clone();
                for _ in 0..steps {
                    current = engine.step(black_box(&current)).unwrap();
                }
                current
            })
        });
    }

    group.finish();
}

/// Benchmark batch processing with different batch sizes
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");

    let batch_config = BatchConfig::new().max_batch_size(64);
    let engine_config = EngineConfig::new(32, 32);

    for batch_size in [1, 4, 8, 16, 32].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let mut scheduler =
                        BatchScheduler::new(batch_config.clone(), engine_config.clone()).unwrap();

                    // Submit requests
                    for _ in 0..batch_size {
                        let input = Array1::from_elem(32, 0.5);
                        scheduler.submit(black_box(input), 10);
                    }

                    // Process batch
                    scheduler.process_all().unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark different sampling strategies
fn bench_sampling_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_strategies");

    let s4_config = S4Config::new()
        .input_dim(32)
        .hidden_dim(128)
        .state_dim(32)
        .num_layers(2)
        .diagonal(true);

    let input = Array1::from_elem(32, 0.5);

    // Greedy
    let greedy_config = EngineConfig::new(32, 32)
        .sampling(SamplingConfig::new().strategy(SamplingStrategy::Greedy));
    let mut greedy_engine = InferenceEngine::with_model(
        greedy_config,
        Box::new(S4D::new(s4_config.clone()).unwrap()),
    );

    // Top-k
    let topk_config = EngineConfig::new(32, 32).sampling(
        SamplingConfig::new()
            .strategy(SamplingStrategy::TopK)
            .top_k(5),
    );
    let mut topk_engine =
        InferenceEngine::with_model(topk_config, Box::new(S4D::new(s4_config.clone()).unwrap()));

    // Top-p
    let topp_config = EngineConfig::new(32, 32).sampling(
        SamplingConfig::new()
            .strategy(SamplingStrategy::TopP)
            .top_p(0.9),
    );
    let mut topp_engine =
        InferenceEngine::with_model(topp_config, Box::new(S4D::new(s4_config).unwrap()));

    group.bench_function("greedy", |b| {
        b.iter(|| greedy_engine.step(black_box(&input)).unwrap())
    });

    group.bench_function("top_k", |b| {
        b.iter(|| topk_engine.step(black_box(&input)).unwrap())
    });

    group.bench_function("top_p", |b| {
        b.iter(|| topp_engine.step(black_box(&input)).unwrap())
    });

    group.finish();
}

/// Benchmark multi-modal pipelines with different fusion strategies
fn bench_multimodal_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("multimodal_fusion");

    let audio = Array1::from_elem(16, 0.3);
    let video = Array1::from_elem(16, 0.6);
    let sensor = Array1::from_elem(16, 0.9);

    for strategy in [
        FusionStrategy::EarlyFusion,
        FusionStrategy::LateFusion,
        FusionStrategy::WeightedFusion,
        FusionStrategy::MaxPooling,
        FusionStrategy::CrossAttention,
        FusionStrategy::Hierarchical,
    ]
    .iter()
    {
        let mut pipeline = MultiModalPipeline::builder()
            .engine_config(EngineConfig::new(48, 32))
            .modality(ModalityType::Audio, 16)
            .modality(ModalityType::Video, 16)
            .modality(ModalityType::Sensor, 16)
            .fusion_strategy(*strategy)
            .build()
            .unwrap();

        group.bench_function(format!("{:?}", strategy), |b| {
            b.iter(|| {
                pipeline
                    .forward(black_box(&[
                        (ModalityType::Audio, audio.clone()),
                        (ModalityType::Video, video.clone()),
                        (ModalityType::Sensor, sensor.clone()),
                    ]))
                    .unwrap()
            })
        });
    }

    group.finish();
}

/// Benchmark ensemble models with different strategies
fn bench_ensemble(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble");

    // Create multiple models
    let s4_config = S4Config::new()
        .input_dim(32)
        .hidden_dim(128)
        .state_dim(32)
        .num_layers(2)
        .diagonal(true);

    let input = Array1::from_elem(32, 0.5);

    // Create separate model sets since Box<dyn AutoregressiveModel> doesn't implement Clone
    let models_avg: Vec<Box<dyn kizzasi_model::AutoregressiveModel>> = vec![
        Box::new(S4D::new(s4_config.clone()).unwrap()),
        Box::new(S4D::new(s4_config.clone()).unwrap()),
        Box::new(S4D::new(s4_config.clone()).unwrap()),
    ];

    let models_weighted: Vec<Box<dyn kizzasi_model::AutoregressiveModel>> = vec![
        Box::new(S4D::new(s4_config.clone()).unwrap()),
        Box::new(S4D::new(s4_config.clone()).unwrap()),
        Box::new(S4D::new(s4_config).unwrap()),
    ];

    // Averaging ensemble
    let mut avg_ensemble = EnsembleBuilder::new()
        .strategy(EnsembleStrategy::Average)
        .add_models(models_avg)
        .build()
        .unwrap();

    // Weighted ensemble
    let mut weighted_ensemble = EnsembleBuilder::new()
        .strategy(EnsembleStrategy::Weighted)
        .add_models(models_weighted)
        .weights(vec![0.5, 0.3, 0.2])
        .build()
        .unwrap();

    group.bench_function("averaging", |b| {
        b.iter(|| avg_ensemble.step(black_box(&input)).unwrap())
    });

    group.bench_function("weighted", |b| {
        b.iter(|| weighted_ensemble.step(black_box(&input)).unwrap())
    });

    group.finish();
}

/// Benchmark speculative decoding
fn bench_speculative_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("speculative_decoding");

    // Draft model (smaller/faster)
    let draft_config = S4Config::new()
        .input_dim(32)
        .hidden_dim(64)
        .state_dim(16)
        .num_layers(1)
        .diagonal(true);
    let draft_model = S4D::new(draft_config).unwrap();

    // Target model (larger/slower)
    let target_config = S4Config::new()
        .input_dim(32)
        .hidden_dim(256)
        .state_dim(64)
        .num_layers(4)
        .diagonal(true);
    let target_model = S4D::new(target_config).unwrap();

    let spec_config = SpeculativeConfig::new().num_draft_tokens(4);

    let mut decoder =
        SpeculativeDecoder::new(Box::new(target_model), Box::new(draft_model), spec_config);

    let input = Array1::from_elem(32, 0.5);

    group.bench_function("speculative_decode", |b| {
        b.iter(|| decoder.generate(black_box(&input), 10).unwrap())
    });

    group.finish();
}

/// Benchmark pipeline with full stack
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    let s4_config = S4Config::new()
        .input_dim(32)
        .hidden_dim(128)
        .state_dim(32)
        .num_layers(2)
        .diagonal(true);
    let s4_model = S4D::new(s4_config).unwrap();

    let engine_config = EngineConfig::new(32, 32)
        .sampling(
            SamplingConfig::new()
                .strategy(SamplingStrategy::TopK)
                .top_k(5),
        )
        .use_embeddings(true);

    let mut pipeline = PipelineBuilder::new()
        .engine_config(engine_config)
        .model(Box::new(s4_model))
        .with_constraints()
        .build()
        .unwrap();

    let input = Array1::from_elem(32, 0.5);

    group.bench_function("single_step", |b| {
        b.iter(|| pipeline.forward(black_box(&input)).unwrap())
    });

    group.bench_function("rollout_20", |b| {
        b.iter(|| pipeline.rollout(black_box(&input), 20).unwrap())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_step_inference,
    bench_rollout,
    bench_batch_processing,
    bench_sampling_strategies,
    bench_multimodal_fusion,
    bench_ensemble,
    bench_speculative_decoding,
    bench_full_pipeline,
);
criterion_main!(benches);
