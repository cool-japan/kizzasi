# Performance Tuning Guide for kizzasi-inference

This guide provides recommendations for optimizing inference performance in various scenarios.

## Table of Contents

1. [Memory Efficiency](#memory-efficiency)
2. [Throughput Optimization](#throughput-optimization)
3. [Latency Optimization](#latency-optimization)
4. [Model-Specific Tuning](#model-specific-tuning)
5. [Hardware Considerations](#hardware-considerations)

---

## Memory Efficiency

### Inference Modes

Choose the appropriate `InferenceMode` based on your memory constraints:

```rust
use kizzasi_inference::{EngineConfig, InferenceMode};

// For maximum memory efficiency (streaming/edge devices)
let config = EngineConfig::new(input_dim, output_dim)
    .inference_mode(InferenceMode::Streaming);

// For low-memory environments
let config = EngineConfig::new(input_dim, output_dim)
    .inference_mode(InferenceMode::LowMemory);

// For quantized inference (reduces state size)
let config = EngineConfig::new(input_dim, output_dim)
    .inference_mode(InferenceMode::Quantized);
```

**Performance Characteristics:**
- `Standard`: Full precision, no optimization (baseline)
- `LowMemory`: ~30-50% memory reduction, <5% accuracy loss
- `Streaming`: ~50-70% memory reduction, optimized for real-time
- `Quantized`: ~60-75% memory reduction, <10% accuracy loss

### State Compression

For long sequences or when checkpointing states:

```rust
use kizzasi_inference::{StateCompressor, CompressionMethod};

// Choose compression method based on sparsity
let compressor = StateCompressor::new(CompressionMethod::QuantizedSparse)
    .with_sparsity_threshold(1e-4);

// Compress states before saving
let compressed = compressor.compress(&hidden_state)?;

// Compression ratios (typical):
// - Quantize8Bit: 4x reduction
// - Quantize4Bit: 8x reduction
// - Sparse: 5-20x (depending on sparsity)
// - QuantizedSparse: 10-40x (best for sparse data)
```

### Context Management

Limit history length for memory-constrained scenarios:

```rust
let config = EngineConfig::new(input_dim, output_dim)
    .max_history_length(128)  // Limit to last 128 steps
    .state_prune_threshold(1e-5);  // Aggressively prune small values
```

---

## Throughput Optimization

### Batch Processing

Use batch inference for maximum throughput:

```rust
use kizzasi_inference::{BatchScheduler, BatchConfig, Priority};

let batch_config = BatchConfig {
    max_batch_size: 32,  // Tune based on model size and memory
    max_wait_ms: 50,      // Balance latency vs throughput
    enable_reordering: true,
};

let mut scheduler = BatchScheduler::new(batch_config);

// Submit requests
scheduler.submit(request, Priority::Normal)?;

// Process batches
let results = scheduler.process_batch(&mut engine)?;
```

**Tuning Guidelines:**
- **Small models** (< 100M params): batch_size = 32-64
- **Medium models** (100M-1B params): batch_size = 16-32
- **Large models** (> 1B params): batch_size = 4-16
- **max_wait_ms**: 10-50ms for interactive, 100-500ms for batch processing

### Continuous Batching

For serving workloads with variable request rates:

```rust
use kizzasi_inference::streaming::StreamConfig;

let config = StreamConfig::new()
    .batch_size(16)
    .max_latency_ms(50)
    .adaptive_batching(true);  // Dynamically adjust batch size
```

### Speculative Decoding

For 2-3x speedup with compatible draft/main model pairs:

```rust
use kizzasi_inference::{SpeculativeDecoder, SpeculativeConfig};

let config = SpeculativeConfig::new()
    .num_draft_tokens(4)      // More tokens = higher speedup (if accepted)
    .draft_temperature(1.0)
    .greedy_verification(true);

let mut decoder = SpeculativeDecoder::new(
    main_model,
    draft_model,  // Should be 3-10x faster than main model
    config
);

// Monitor acceptance rate (aim for >60%)
println!("Acceptance rate: {:.1}%", decoder.acceptance_rate() * 100.0);
```

**Draft Model Selection:**
- Draft should be 3-10x faster than main model
- Similar architecture helps (e.g., RWKV-small → RWKV-large)
- Aim for >60% acceptance rate for net speedup

---

## Latency Optimization

### Streaming Mode

For real-time applications with strict latency requirements:

```rust
#[cfg(feature = "streaming")]
{
    use kizzasi_inference::streaming::StreamingEngine;

    let config = StreamConfig::new()
        .batch_size(1)        // Disable batching for lowest latency
        .buffer_size(64)      // Small buffer
        .max_latency_ms(10);  // Strict latency bound

    let engine = StreamingEngine::new(config)?;
}
```

### Sampling Strategy

Choose sampling strategies based on latency requirements:

```rust
use kizzasi_inference::{SamplingConfig, SamplingStrategy};

// Fastest: Greedy (no sampling overhead)
let config = SamplingConfig::new()
    .strategy(SamplingStrategy::Greedy);

// Fast: Temperature (minimal overhead)
let config = SamplingConfig::new()
    .strategy(SamplingStrategy::Temperature)
    .temperature(0.8);

// Slower: Top-K (sorting overhead)
let config = SamplingConfig::new()
    .top_k(40);

// Slowest: Beam Search (multiple hypotheses)
let config = SamplingConfig::new()
    .beam_search(4);
```

**Latency Impact:**
- Greedy: baseline (fastest)
- Temperature: +5-10% latency
- Top-K: +10-20% latency
- Top-P: +15-25% latency
- Beam Search: +100-300% latency (scales with beam width)

---

## Model-Specific Tuning

### RWKV Models

```rust
use kizzasi_model::rwkv::RwkvConfig;

let config = RwkvConfig::new()
    .hidden_dim(512)      // Smaller = faster
    .num_layers(12)       // Fewer layers = faster
    .num_heads(8);        // Balance quality vs speed

// RWKV is very efficient for long sequences
// - O(1) state size regardless of sequence length
// - Ideal for streaming applications
```

### S4/S4D Models

```rust
use kizzasi_model::s4::{S4Config, S4D};

let config = S4Config::new()
    .hidden_dim(256)
    .state_dim(64)        // Larger state = better quality, more memory
    .diagonal(true);      // S4D: faster than full S4

// S4D is ~2-3x faster than full S4
// State size: O(state_dim) vs O(state_dim^2)
```

### Transformer Models

```rust
use kizzasi_model::transformer::TransformerConfig;

let config = TransformerConfig::new()
    .num_heads(8)
    .head_dim(64)
    .use_kv_cache(true);  // Essential for autoregressive inference

// KV-cache reduces inference from O(n^2) to O(n)
// Memory trade-off: stores past keys/values
```

---

## Hardware Considerations

### CPU Optimization

```rust
// Enable all available CPU features
// Set environment variable before running:
// RUSTFLAGS="-C target-cpu=native"

// Use scirs2-core's optimized BLAS operations
// Will automatically use AVX2/AVX512 if available
```

**CPU-Specific Tips:**
- **AVX2** (Intel/AMD): ~2x faster than baseline
- **AVX512** (newer Intel): ~3-4x faster
- **NEON** (ARM/Apple Silicon): ~2-3x faster
- Set thread count: `RAYON_NUM_THREADS=8`

### Memory Hierarchy

```rust
// Optimize for L1/L2 cache
let config = EngineConfig::new(input_dim, output_dim)
    .context(ContextConfig::new()
        .max_context(512));  // Keep working set in cache

// Typical cache sizes:
// L1: 32-64 KB per core
// L2: 256-512 KB per core
// L3: 8-32 MB shared
```

### Batch Size Tuning

```rust
// Find optimal batch size empirically:
fn find_optimal_batch_size(engine: &mut InferenceEngine) -> usize {
    let sizes = vec![1, 2, 4, 8, 16, 32, 64];
    let mut best_throughput = 0.0;
    let mut best_size = 1;

    for &size in &sizes {
        let start = std::time::Instant::now();
        // Run 100 batches
        for _ in 0..100 {
            let inputs = vec![Array1::zeros(input_dim); size];
            engine.step_batch(&inputs)?;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let throughput = (100.0 * size as f64) / elapsed;

        if throughput > best_throughput {
            best_throughput = throughput;
            best_size = size;
        }
    }

    best_size
}
```

---

## Benchmarking

Use the built-in profiler to identify bottlenecks:

```rust
use kizzasi_inference::{InferenceProfiler, Timer};

let mut profiler = InferenceProfiler::new();

{
    let _timer = Timer::new(&mut profiler, "model_forward");
    engine.step(&input)?;
}

let summary = profiler.summary();
println!("Model forward: {:.2}ms", summary.avg_time_ms("model_forward"));
println!("Total time: {:.2}ms", summary.total_time_ms);
```

---

## Performance Checklist

Before deploying to production:

- [ ] Choose appropriate `InferenceMode` for memory constraints
- [ ] Enable batching if throughput > latency priority
- [ ] Tune `max_batch_size` and `max_wait_ms` empirically
- [ ] Enable KV-cache for Transformer models
- [ ] Consider speculative decoding for large models
- [ ] Profile with `InferenceProfiler` to find bottlenecks
- [ ] Set `RUSTFLAGS="-C target-cpu=native"` for CPU optimization
- [ ] Monitor memory usage and adjust `max_context` accordingly
- [ ] Use `Quantized` mode for edge deployment
- [ ] Test under load to verify latency requirements

---

## Common Performance Issues

### Issue: High Memory Usage

**Solutions:**
1. Enable `InferenceMode::LowMemory` or `Streaming`
2. Reduce `max_context` length
3. Use state compression for checkpoints
4. Prune unused model layers (if applicable)

### Issue: Low Throughput

**Solutions:**
1. Increase batch size
2. Enable continuous batching
3. Use speculative decoding
4. Reduce sampling complexity (use Greedy)

### Issue: High Latency

**Solutions:**
1. Disable batching (batch_size = 1)
2. Use Greedy sampling
3. Reduce model size (fewer layers/hidden_dim)
4. Enable streaming mode

### Issue: OOM (Out of Memory)

**Solutions:**
1. Use `InferenceMode::Quantized`
2. Reduce batch size
3. Limit max_context length
4. Enable state pruning (higher threshold)
5. Use state compression

---

## Advanced Techniques

### Model Ensembling

Trade-off latency for accuracy:

```rust
use kizzasi_inference::{EnsembleBuilder, EnsembleStrategy};

let ensemble = EnsembleBuilder::new()
    .add_model(model1)
    .add_model(model2)
    .strategy(EnsembleStrategy::Weighted)
    .weights(vec![0.7, 0.3])  // Favor faster model
    .build()?;

// Expect 2x latency, 10-20% accuracy improvement
```

### Custom Sampling

Implement domain-specific sampling for better quality/speed trade-off:

```rust
use kizzasi_inference::CustomSamplingFn;
use std::sync::Arc;

let custom_fn: CustomSamplingFn = Arc::new(|logits, _temp| {
    // Your custom logic here
    // E.g., combine multiple strategies, use domain knowledge, etc.
    Ok(0.0)
});

let sampler = Sampler::with_custom_fn(config, custom_fn);
```

---

## Monitoring in Production

```rust
use kizzasi_inference::InferenceMetrics;

let mut metrics = InferenceMetrics::new();

// Track key metrics
metrics.record_latency(latency_ms);
metrics.record_throughput(samples_per_sec);
metrics.record_memory_usage(memory_mb);

// Export to monitoring system (Prometheus, etc.)
```

**Key Metrics to Monitor:**
- P50, P95, P99 latency
- Throughput (samples/sec)
- Memory usage (RSS, heap)
- Batch size distribution
- Cache hit rate (for KV-cache)
- Acceptance rate (for speculative decoding)

---

## Conclusion

Performance tuning is an iterative process. Always:
1. **Measure** first with profiling
2. **Identify** bottlenecks
3. **Optimize** the critical path
4. **Verify** improvements with benchmarks
5. **Monitor** in production

For questions or optimization help, see the [examples](../examples/) directory.
