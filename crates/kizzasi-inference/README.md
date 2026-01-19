# kizzasi-inference

Unified autoregressive inference engine for Kizzasi AGSP.

## Overview

Production-grade inference pipeline with sampling strategies, batching, streaming, and constraint enforcement. Supports all Kizzasi model architectures.

## Features

- **Sampling Strategies**: Greedy, temperature, top-k, top-p, beam search
- **Batching**: Dynamic batching with continuous processing
- **Streaming**: Async streaming with backpressure handling
- **Constraints**: Integration with kizzasi-logic for safety guardrails
- **Multi-Modal**: Audio, video, sensor fusion pipelines
- **Checkpointing**: Save/load inference state and configuration
- **Network Adapters**: WebSocket, MQTT, gRPC for real-time inference
- **Hot-Swapping**: Runtime model switching without interruption

## Quick Start

```rust
use kizzasi_inference::{InferenceEngine, Pipeline, GreedySampler};

// Create inference pipeline
let pipeline = Pipeline::builder()
    .model(my_model)
    .tokenizer(my_tokenizer)
    .sampler(GreedySampler::new())
    .build()?;

// Single-step prediction
let input = Array1::zeros(32);
let output = pipeline.predict(&input)?;

// Multi-step rollout
let predictions = pipeline.predict_n(&input, 100)?;

// Streaming inference
let mut stream = pipeline.stream(input_stream).await?;
while let Some(prediction) = stream.next().await {
    // Process prediction
}
```

## Performance

- Single-step latency: <100μs (Mamba2)
- Throughput: 320K predictions/sec (16 workers)
- 177 comprehensive tests, all passing

## Documentation

- [API Documentation](https://docs.rs/kizzasi-inference)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
