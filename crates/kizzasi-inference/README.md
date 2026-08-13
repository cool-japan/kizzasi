# kizzasi-inference

**Status:** Stable — 263/263 tests passing, 506 public API items, 0 unimplemented stubs

Unified autoregressive inference engine for Kizzasi AGSP.

## Overview

Production-grade inference pipeline with sampling strategies, batching, streaming, and constraint enforcement. Supports all Kizzasi model architectures.

## Features

- **Sampling Strategies**: Greedy, temperature, top-k, top-p, beam search, plus constrained and rejection sampling
- **Batching**: Dynamic/continuous batching with priority scheduling
- **Streaming**: Async streaming with backpressure handling
- **Constraints**: Integration with kizzasi-logic for safety guardrails, plus temporal logic (LTL/STL) constraints
- **Multi-Modal**: Audio, video, sensor, and text fusion pipelines
- **Ensembling**: Multi-model ensembles (averaging, weighted, voting, product-of-experts)
- **Speculative Decoding**: Draft-model-assisted generation for faster autoregressive inference
- **LoRA Adapters**: Load and apply LoRA adapters at inference time
- **Mixed Precision**: FP16/BF16 inference with FP32 accumulation
- **Checkpointing**: Save/load inference state and configuration
- **Network Adapters**: WebSocket, MQTT, gRPC, and REST for real-time inference
- **Hot-Swapping**: Runtime model switching without interruption
- **Model Versioning**: Semantic versioning with health checks and automatic fallback

## Quick Start

```rust
use kizzasi_inference::{
    EngineConfig, ModelBuilder, ModelRegistry, PipelineBuilder, SamplingConfig, SamplingStrategy,
};
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Register and instantiate a model.
    //    Transformer (like S4/S4D, RWKV/RWKV5, Mamba/Mamba2, NeuralODE) has
    //    no output-projection layer, so output_dim must equal input_dim;
    //    MultiModal/Snn/MultiScale are the architectures that support
    //    projecting to a different output_dim.
    let mut registry = ModelRegistry::new();
    let model_config = ModelBuilder::transformer().dims(1, 128, 1).layers(3).build();
    registry.register("transformer", model_config);
    let model = registry.create_model("transformer")?;

    // 2. Configure sampling and the engine
    let sampling = SamplingConfig::new()
        .strategy(SamplingStrategy::TopK)
        .top_k(5)
        .temperature(0.8);
    let engine_config = EngineConfig::new(1, 1).sampling(sampling).use_embeddings(true);

    // 3. Build the pipeline
    let mut pipeline = PipelineBuilder::new()
        .engine_config(engine_config)
        .model(model)
        .with_constraints() // enable constraint enforcement hooks
        .build()?;

    // 4. Single-step prediction
    let input = Array1::from_vec(vec![0.5]);
    let output = pipeline.forward(&input)?;
    println!("Output: {output:?}");

    // 5. Multi-step rollout
    pipeline.reset();
    let initial = Array1::from_vec(vec![0.3]);
    let predictions = pipeline.rollout(&initial, 7)?;
    println!("Generated {} predictions", predictions.len());

    Ok(())
}
```

Streaming inference (requires the `streaming` feature):

```rust
use futures::stream::StreamExt;
use kizzasi_inference::streaming::{StreamConfig, StreamingEngine};
use scirs2_core::ndarray::Array1;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = StreamConfig::default();
    let engine = StreamingEngine::new(config)?;

    let input_stream = futures::stream::iter(vec![Array1::from_vec(vec![0.5])]);
    let mut predictions = engine.predict_stream(input_stream);
    while let Some(prediction) = predictions.next().await {
        let output = prediction?;
        println!("{output:?}");
    }

    Ok(())
}
```

## Performance

- Single-step latency: <100μs (Mamba2)
- Throughput: 320K predictions/sec (16 workers)
- 263 comprehensive tests, all passing

## Documentation

- [API Documentation](https://docs.rs/kizzasi-inference)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
