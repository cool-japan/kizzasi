# Streaming Best Practices for kizzasi-inference

This guide provides best practices for building robust, efficient streaming inference pipelines using `kizzasi-inference`.

## Table of Contents

1. [Async/Await Patterns](#asyncawait-patterns)
2. [Backpressure Handling](#backpressure-handling)
3. [Stream Transformer Composition](#stream-transformer-composition)
4. [Error Handling](#error-handling)
5. [Configuration Tuning](#configuration-tuning)
6. [Batching Strategies](#batching-strategies)
7. [Real-World Integration](#real-world-integration)
8. [Monitoring and Debugging](#monitoring-and-debugging)
9. [Common Patterns](#common-patterns)
10. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

---

## Async/Await Patterns

### Basic Streaming Setup

```rust
use kizzasi_inference::streaming::{StreamingEngine, StreamConfig};
use kizzasi_inference::engine::EngineConfig;
use futures::stream::{self, StreamExt};
use scirs2_core::ndarray::Array1;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure the streaming engine
    let config = StreamConfig::new()
        .buffer_size(1024)
        .batch_size(1)
        .max_latency_ms(50);

    let engine = StreamingEngine::new(config)?;

    // Create an input stream (e.g., from sensor data)
    let input_stream = stream::iter(vec![
        Array1::from_vec(vec![0.1, 0.2, 0.3]),
        Array1::from_vec(vec![0.4, 0.5, 0.6]),
        // ... more inputs
    ]);

    // Process stream asynchronously
    let mut predictions = engine.predict_stream(input_stream);

    while let Some(result) = predictions.next().await {
        match result {
            Ok(prediction) => println!("Prediction: {:?}", prediction),
            Err(e) => eprintln!("Error: {:?}", e),
        }
    }

    Ok(())
}
```

### Single-Step Async Inference

For one-off predictions in async contexts:

```rust
async fn process_single_sample(
    engine: &StreamingEngine,
    input: Array1<f32>,
) -> Result<Array1<f32>, Box<dyn std::error::Error>> {
    let prediction = engine.step_async(input).await?;
    Ok(prediction)
}
```

### Autoregressive Rollouts

For multi-step prediction:

```rust
async fn forecast_future(
    engine: &StreamingEngine,
    initial_state: Array1<f32>,
    forecast_steps: usize,
) -> Result<Vec<Array1<f32>>, Box<dyn std::error::Error>> {
    // Generate N future predictions autoregressively
    let predictions = engine.rollout_async(initial_state, forecast_steps).await?;
    Ok(predictions)
}
```

**Best Practice**: Use `rollout_async` when you need multiple future steps, as it's more efficient than chaining individual `step_async` calls.

---

## Backpressure Handling

### Understanding Buffer Sizes

The `buffer_size` parameter controls how many items can be queued between producer and consumer:

```rust
let config = StreamConfig::new()
    .buffer_size(1024);  // Can queue up to 1024 samples
```

**Tuning Guidelines:**
- **Too small** (< 64): May cause blocking and reduced throughput
- **Too large** (> 4096): Wastes memory and increases latency
- **Recommended**: Start with 1024, adjust based on metrics

### Bounded Channels and Flow Control

The streaming engine uses bounded channels (`tokio::sync::mpsc`) for natural backpressure:

```rust
// Producer will block when buffer is full
// This prevents memory exhaustion

use kizzasi_inference::streaming::CallbackStreamHandle;

async fn producer(handle: CallbackStreamHandle<Array1<f32>>) {
    for sample in sensor_data() {
        // Will block if consumer is slow
        handle.push(sample).await.unwrap();
    }
}
```

### Non-Blocking Push

For systems that can't afford to block:

```rust
async fn non_blocking_producer(handle: CallbackStreamHandle<Array1<f32>>) {
    for sample in sensor_data() {
        match handle.try_push(sample) {
            Ok(_) => {},
            Err(e) => {
                // Handle buffer full case
                eprintln!("Buffer full, dropping sample: {:?}", e);
                metrics.record_drop();
            }
        }
    }
}
```

**Best Practice**: Use blocking `push()` for critical data, `try_push()` for non-critical or high-frequency data where drops are acceptable.

### Latency-Aware Processing

Balance throughput and latency with `max_latency_ms`:

```rust
let config = StreamConfig::new()
    .batch_size(32)
    .max_latency_ms(50);  // Force processing if batch not full after 50ms
```

This ensures samples aren't delayed indefinitely waiting for a full batch.

---

## Stream Transformer Composition

Stream transformers allow composable processing pipelines.

### Map Transformer

Transform each item individually:

```rust
use kizzasi_inference::streaming::{MapTransformer, StreamTransformer};

// Normalize inputs
let normalizer = MapTransformer::new(|x: Array1<f32>| {
    let mean = x.mean().unwrap_or(0.0);
    let std = x.std(0.0).unwrap_or(1.0);
    (x - mean) / std.max(1e-6)
});

let normalized_stream = normalizer.transform(Box::pin(input_stream));
```

### Buffer Transformer

Accumulate items into fixed-size batches:

```rust
use kizzasi_inference::streaming::BufferTransformer;

// Collect samples into batches of 16
let batcher = BufferTransformer::new(16);
let batched_stream = batcher.transform(Box::pin(input_stream));

// Process batches
while let Some(batch) = batched_stream.next().await {
    println!("Processing batch of {} samples", batch.len());
}
```

### Debounce Transformer

Only emit items after a period of inactivity:

```rust
use kizzasi_inference::streaming::DebounceTransformer;
use tokio::time::Duration;

// Only process if no new input for 100ms
let debouncer = DebounceTransformer::new(Duration::from_millis(100));
let debounced_stream = debouncer.transform(Box::pin(input_stream));
```

**Use Case**: User input processing, where you want to wait for the user to finish typing.

### Throttle Transformer

Limit the rate of items:

```rust
use kizzasi_inference::streaming::ThrottleTransformer;

// Process at most 10 items per second
let throttler = ThrottleTransformer::new(Duration::from_millis(100));
let throttled_stream = throttler.transform(Box::pin(input_stream));
```

**Use Case**: Rate-limiting API calls or controlling inference load.

### Chaining Transformers

Compose multiple transformers for complex pipelines:

```rust
// 1. Normalize inputs
let normalizer = MapTransformer::new(|x: Array1<f32>| normalize(x));

// 2. Debounce to reduce load
let debouncer = DebounceTransformer::new(Duration::from_millis(50));

// 3. Buffer into batches
let batcher = BufferTransformer::new(8);

// Chain transformers
let processed = normalizer.transform(Box::pin(input_stream));
let debounced = debouncer.transform(processed);
let batched = batcher.transform(debounced);

// Now process batches
while let Some(batch) = batched.next().await {
    // Process batch of normalized, debounced inputs
}
```

**Best Practice**: Keep transformer chains short (2-4 transformers) to maintain readability and debuggability.

---

## Error Handling

### Stream-Level Error Handling

```rust
let mut predictions = engine.predict_stream(input_stream);

while let Some(result) = predictions.next().await {
    match result {
        Ok(prediction) => {
            // Process successful prediction
            process_prediction(prediction);
        }
        Err(e) => {
            // Log error and continue processing
            eprintln!("Inference error: {:?}", e);
            metrics.record_error();

            // Optionally reset engine state if corruption suspected
            if is_state_corruption(&e) {
                engine.reset_async().await?;
            }
        }
    }
}
```

### Graceful Degradation

Implement fallback strategies for robustness:

```rust
use std::time::Duration;
use tokio::time::timeout;

async fn predict_with_fallback(
    engine: &StreamingEngine,
    input: Array1<f32>,
) -> Array1<f32> {
    // Try with timeout
    match timeout(Duration::from_millis(100), engine.step_async(input.clone())).await {
        Ok(Ok(prediction)) => prediction,
        Ok(Err(e)) => {
            eprintln!("Inference failed: {:?}, using fallback", e);
            fallback_prediction(&input)
        }
        Err(_) => {
            eprintln!("Inference timeout, using fallback");
            fallback_prediction(&input)
        }
    }
}

fn fallback_prediction(input: &Array1<f32>) -> Array1<f32> {
    // Simple heuristic or cached prediction
    input.clone() // Echo input as simplest fallback
}
```

### State Reset After Errors

```rust
let mut consecutive_errors = 0;

while let Some(result) = predictions.next().await {
    match result {
        Ok(prediction) => {
            consecutive_errors = 0;
            // Process prediction
        }
        Err(e) => {
            consecutive_errors += 1;
            eprintln!("Error: {:?}", e);

            // Reset state after multiple failures
            if consecutive_errors >= 3 {
                eprintln!("Multiple failures, resetting engine state");
                engine.reset_async().await?;
                consecutive_errors = 0;
            }
        }
    }
}
```

**Best Practice**: Always have a fallback strategy. Never let a single inference error crash your entire pipeline.

---

## Configuration Tuning

### Latency-Optimized Configuration

For real-time systems with strict latency requirements:

```rust
let config = StreamConfig::new()
    .buffer_size(64)          // Small buffer
    .batch_size(1)            // No batching
    .max_latency_ms(10)       // Strict latency bound
    .adaptive_batching(false);

let mut engine_config = EngineConfig::new(input_dim, output_dim);
engine_config.inference_mode = InferenceMode::Streaming;

let stream_config = config.engine(engine_config);
```

**Expected Performance:**
- Latency: 5-15ms per sample
- Throughput: 50-200 samples/sec (CPU-dependent)

### Throughput-Optimized Configuration

For batch processing with relaxed latency:

```rust
let config = StreamConfig::new()
    .buffer_size(2048)        // Large buffer
    .batch_size(32)           // Batch processing
    .max_latency_ms(500)      // Relaxed latency
    .adaptive_batching(true); // Dynamically adjust batch size
```

**Expected Performance:**
- Latency: 100-1000ms per sample
- Throughput: 500-2000 samples/sec (CPU-dependent)

### Balanced Configuration

For general-purpose applications:

```rust
let config = StreamConfig::new()
    .buffer_size(1024)
    .batch_size(8)
    .max_latency_ms(50)
    .adaptive_batching(true);
```

**Expected Performance:**
- Latency: 20-100ms per sample
- Throughput: 200-800 samples/sec

### Adaptive Batching

Enable adaptive batching for variable load:

```rust
let config = StreamConfig::new()
    .batch_size(16)           // Target batch size
    .max_latency_ms(50)
    .adaptive_batching(true); // Adjust based on load
```

**How it Works:**
- Under high load: batches approach `batch_size`
- Under low load: smaller batches to reduce latency
- Automatically balances throughput vs latency

---

## Batching Strategies

### When to Use Batching

**Use batching when:**
- Throughput is more important than latency
- Model has high per-sample overhead
- Input rate is high and bursty
- You can tolerate 50-500ms latency

**Don't batch when:**
- Latency requirement < 20ms
- Input rate is very low (< 10 samples/sec)
- Each sample requires different model parameters

### Batch Size Selection

```rust
// For small models (< 100M params)
let config = StreamConfig::new().batch_size(32);

// For medium models (100M-1B params)
let config = StreamConfig::new().batch_size(16);

// For large models (> 1B params)
let config = StreamConfig::new().batch_size(4);
```

### Dynamic Batch Sizing

Implement custom batching logic:

```rust
use tokio::time::{interval, Duration};

async fn dynamic_batching(
    input_stream: impl Stream<Item = Array1<f32>>,
    engine: StreamingEngine,
) {
    let mut input = Box::pin(input_stream);
    let mut batch = Vec::new();
    let mut timer = interval(Duration::from_millis(50));

    // Start with small batch size
    let mut current_batch_size = 4;
    let max_batch_size = 32;

    loop {
        tokio::select! {
            Some(item) = input.next() => {
                batch.push(item);

                if batch.len() >= current_batch_size {
                    process_batch(&engine, &batch).await;

                    // Increase batch size if we're keeping up
                    current_batch_size = (current_batch_size * 2).min(max_batch_size);
                    batch.clear();
                }
            }
            _ = timer.tick() => {
                if !batch.is_empty() {
                    process_batch(&engine, &batch).await;

                    // Decrease batch size if load is low
                    current_batch_size = (current_batch_size / 2).max(1);
                    batch.clear();
                }
            }
            else => break,
        }
    }
}
```

---

## Real-World Integration

### Event-Driven Systems

Use `CallbackStream` for callback-based sources:

```rust
use kizzasi_inference::streaming::{CallbackStream, CallbackStreamHandle};

// Create callback stream
let (handle, stream) = CallbackStream::new(1024);

// Register callback with external system
register_sensor_callback(handle.clone());

// Process stream
let mut predictions = engine.predict_stream(stream);
while let Some(result) = predictions.next().await {
    // Handle predictions
}

// In sensor callback:
fn on_sensor_data(handle: CallbackStreamHandle<Array1<f32>>, data: SensorData) {
    let input = Array1::from_vec(data.values);

    // Push to stream (async)
    tokio::spawn(async move {
        let _ = handle.push(input).await;
    });
}
```

### Message Queue Integration

```rust
use tokio::sync::mpsc;

async fn kafka_to_stream(
    kafka_consumer: KafkaConsumer,
    engine: StreamingEngine,
) -> Result<(), Box<dyn std::error::Error>> {
    let (tx, rx) = mpsc::channel(1024);

    // Spawn Kafka consumer task
    tokio::spawn(async move {
        while let Ok(message) = kafka_consumer.poll(Duration::from_millis(100)) {
            if let Some(payload) = message.payload() {
                let input = parse_input(payload);
                let _ = tx.send(input).await;
            }
        }
    });

    // Convert channel to stream
    let input_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

    // Process predictions
    let mut predictions = engine.predict_stream(input_stream);
    while let Some(result) = predictions.next().await {
        // Publish predictions back to Kafka
        publish_prediction(result?).await?;
    }

    Ok(())
}
```

### WebSocket Streaming (Conceptual)

```rust
// Note: WebSocket integration is future work, but here's the pattern

async fn websocket_inference(
    ws_stream: WebSocketStream,
    engine: StreamingEngine,
) -> Result<(), Box<dyn std::error::Error>> {
    let (mut write, mut read) = ws_stream.split();

    // Create bidirectional stream
    let input_stream = read.filter_map(|msg| async move {
        match msg {
            Ok(Message::Binary(data)) => Some(parse_array1(&data)),
            _ => None,
        }
    });

    // Process and send back predictions
    let mut predictions = engine.predict_stream(input_stream);

    while let Some(result) = predictions.next().await {
        match result {
            Ok(prediction) => {
                let data = serialize_array1(&prediction);
                write.send(Message::Binary(data)).await?;
            }
            Err(e) => {
                let error_msg = format!("Error: {:?}", e);
                write.send(Message::Text(error_msg)).await?;
            }
        }
    }

    Ok(())
}
```

### File-Based Streaming

Process large files line-by-line:

```rust
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::fs::File;

async fn process_csv_file(
    path: &str,
    engine: StreamingEngine,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let input_stream = async_stream::stream! {
        while let Ok(Some(line)) = lines.next_line().await {
            if let Ok(values) = parse_csv_line(&line) {
                yield Array1::from_vec(values);
            }
        }
    };

    let mut predictions = engine.predict_stream(Box::pin(input_stream));
    let mut output_file = File::create("predictions.csv").await?;

    while let Some(result) = predictions.next().await {
        let prediction = result?;
        write_csv_line(&mut output_file, &prediction).await?;
    }

    Ok(())
}
```

---

## Monitoring and Debugging

### Using StreamMetrics

Track streaming performance:

```rust
use kizzasi_inference::streaming::StreamMetrics;
use std::time::Instant;

let mut metrics = StreamMetrics::new();

while let Some(result) = predictions.next().await {
    let start = Instant::now();
    let prediction = result?;
    let latency_us = start.elapsed().as_micros() as u64;

    // Update metrics
    metrics.update(1, latency_us);

    // Log metrics periodically
    if metrics.samples_processed % 1000 == 0 {
        println!("Stream Metrics:");
        println!("  Samples: {}", metrics.samples_processed);
        println!("  Batches: {}", metrics.batches_processed);
        println!("  Avg batch size: {:.1}", metrics.avg_batch_size);
        println!("  Avg latency: {:.1}µs", metrics.avg_latency_us);
        println!("  Peak latency: {}µs", metrics.peak_latency_us);
    }
}
```

### Custom Metrics Integration

```rust
struct StreamMonitor {
    metrics: StreamMetrics,
    dropped_samples: usize,
    error_count: usize,
}

impl StreamMonitor {
    fn new() -> Self {
        Self {
            metrics: StreamMetrics::new(),
            dropped_samples: 0,
            error_count: 0,
        }
    }

    fn record_success(&mut self, latency_us: u64) {
        self.metrics.update(1, latency_us);
    }

    fn record_drop(&mut self) {
        self.dropped_samples += 1;
    }

    fn record_error(&mut self) {
        self.error_count += 1;
    }

    fn report(&self) {
        let throughput = self.metrics.samples_processed as f64
            / (self.metrics.avg_latency_us / 1_000_000.0);

        println!("=== Stream Monitor Report ===");
        println!("Throughput: {:.1} samples/sec", throughput);
        println!("Success rate: {:.2}%",
            100.0 * self.metrics.samples_processed as f64
                / (self.metrics.samples_processed + self.error_count) as f64);
        println!("Drop rate: {:.2}%",
            100.0 * self.dropped_samples as f64
                / (self.metrics.samples_processed + self.dropped_samples) as f64);
    }
}
```

### Debug Logging

```rust
use tracing::{info, warn, error, debug};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let engine = StreamingEngine::new(config)?;
    let mut predictions = engine.predict_stream(input_stream);

    while let Some(result) = predictions.next().await {
        match result {
            Ok(prediction) => {
                debug!("Prediction: {:?}", prediction);
            }
            Err(e) => {
                error!("Inference error: {:?}", e);
            }
        }
    }

    Ok(())
}
```

### Performance Profiling

```rust
use std::time::Instant;

struct LatencyTracker {
    samples: Vec<u64>,
}

impl LatencyTracker {
    fn new() -> Self {
        Self { samples: Vec::new() }
    }

    fn record(&mut self, latency_us: u64) {
        self.samples.push(latency_us);
    }

    fn percentiles(&self) -> (u64, u64, u64) {
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();

        let p50 = sorted[sorted.len() / 2];
        let p95 = sorted[sorted.len() * 95 / 100];
        let p99 = sorted[sorted.len() * 99 / 100];

        (p50, p95, p99)
    }
}

// Usage
let mut tracker = LatencyTracker::new();

while let Some(result) = predictions.next().await {
    let start = Instant::now();
    let prediction = result?;
    let latency_us = start.elapsed().as_micros() as u64;

    tracker.record(latency_us);
}

let (p50, p95, p99) = tracker.percentiles();
println!("Latency - P50: {}µs, P95: {}µs, P99: {}µs", p50, p95, p99);
```

---

## Common Patterns

### Pattern 1: Continuous Monitoring

```rust
async fn continuous_monitoring_pipeline(
    sensor_stream: impl Stream<Item = SensorReading>,
    engine: StreamingEngine,
    alert_threshold: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Transform sensor readings to model inputs
    let input_stream = sensor_stream.map(|reading| reading.to_array1());

    // Process stream
    let mut predictions = engine.predict_stream(input_stream);

    while let Some(result) = predictions.next().await {
        let prediction = result?;

        // Check for anomalies
        if is_anomaly(&prediction, alert_threshold) {
            send_alert(prediction).await?;
        }
    }

    Ok(())
}
```

### Pattern 2: Request-Response with Timeout

```rust
async fn request_response_inference(
    engine: &StreamingEngine,
    input: Array1<f32>,
    timeout_ms: u64,
) -> Result<Array1<f32>, Box<dyn std::error::Error>> {
    match tokio::time::timeout(
        Duration::from_millis(timeout_ms),
        engine.step_async(input),
    ).await {
        Ok(Ok(prediction)) => Ok(prediction),
        Ok(Err(e)) => Err(e.into()),
        Err(_) => Err("Timeout".into()),
    }
}
```

### Pattern 3: Fan-Out Processing

Process one input with multiple engines:

```rust
async fn fan_out_inference(
    input: Array1<f32>,
    engines: Vec<StreamingEngine>,
) -> Vec<Array1<f32>> {
    let futures = engines
        .iter()
        .map(|engine| engine.step_async(input.clone()));

    let results = futures::future::join_all(futures).await;

    results
        .into_iter()
        .filter_map(|r| r.ok())
        .collect()
}
```

### Pattern 4: Multi-Stage Pipeline

```rust
async fn multi_stage_pipeline(
    input_stream: impl Stream<Item = Array1<f32>>,
    stage1_engine: StreamingEngine,
    stage2_engine: StreamingEngine,
) -> impl Stream<Item = InferenceResult<Array1<f32>>> {
    // Stage 1: Initial processing
    let stage1_stream = stage1_engine.predict_stream(input_stream);

    // Stage 2: Refine predictions
    let stage2_stream = stage1_stream.then(move |result| {
        let engine = stage2_engine.clone();
        async move {
            match result {
                Ok(intermediate) => engine.step_async(intermediate).await,
                Err(e) => Err(e),
            }
        }
    });

    stage2_stream
}
```

### Pattern 5: State Checkpointing

Periodically save state for fault tolerance:

```rust
use kizzasi_inference::checkpoint::CheckpointManager;

async fn inference_with_checkpointing(
    mut predictions: impl Stream<Item = InferenceResult<Array1<f32>>>,
    checkpoint_mgr: &mut CheckpointManager,
    checkpoint_interval: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut count = 0;

    while let Some(result) = predictions.next().await {
        let prediction = result?;

        // Process prediction
        handle_prediction(prediction);

        count += 1;

        // Checkpoint periodically
        if count % checkpoint_interval == 0 {
            checkpoint_mgr.save("streaming_checkpoint.bin").await?;
            println!("Checkpointed at sample {}", count);
        }
    }

    Ok(())
}
```

---

## Anti-Patterns to Avoid

### ❌ Anti-Pattern 1: Blocking in Async Context

**Bad:**
```rust
// DON'T: Blocking operation in async context
async fn bad_pattern(engine: &StreamingEngine) {
    let prediction = engine.step_async(input).await.unwrap();

    // This blocks the entire async runtime!
    std::thread::sleep(Duration::from_secs(1));
}
```

**Good:**
```rust
// DO: Use async sleep
async fn good_pattern(engine: &StreamingEngine) {
    let prediction = engine.step_async(input).await.unwrap();

    tokio::time::sleep(Duration::from_secs(1)).await;
}
```

### ❌ Anti-Pattern 2: Unbounded Memory Growth

**Bad:**
```rust
// DON'T: Collect entire stream into memory
let all_predictions: Vec<_> = predictions.collect().await;
```

**Good:**
```rust
// DO: Process incrementally
while let Some(prediction) = predictions.next().await {
    process(prediction);
}
```

### ❌ Anti-Pattern 3: Ignoring Errors

**Bad:**
```rust
// DON'T: Silently ignore errors
while let Some(Ok(prediction)) = predictions.next().await {
    process(prediction);
}
// Errors are silently dropped!
```

**Good:**
```rust
// DO: Handle errors explicitly
while let Some(result) = predictions.next().await {
    match result {
        Ok(prediction) => process(prediction),
        Err(e) => {
            eprintln!("Error: {:?}", e);
            metrics.record_error();
        }
    }
}
```

### ❌ Anti-Pattern 4: Over-Batching

**Bad:**
```rust
// DON'T: Use huge batch sizes for low latency
let config = StreamConfig::new()
    .batch_size(1024)  // Way too large!
    .max_latency_ms(10);  // Can't meet this requirement
```

**Good:**
```rust
// DO: Match batch size to latency requirements
let config = StreamConfig::new()
    .batch_size(4)
    .max_latency_ms(10);
```

### ❌ Anti-Pattern 5: Forgetting Backpressure

**Bad:**
```rust
// DON'T: Spawn unbounded tasks
for sample in samples {
    tokio::spawn(async move {
        engine.step_async(sample).await;
    });
}
// Can spawn thousands of tasks!
```

**Good:**
```rust
// DO: Use semaphore for bounded concurrency
let semaphore = Arc::new(Semaphore::new(16));

for sample in samples {
    let permit = semaphore.clone().acquire_owned().await;
    let engine = engine.clone();

    tokio::spawn(async move {
        let _permit = permit;
        engine.step_async(sample).await;
    });
}
```

### ❌ Anti-Pattern 6: Not Resetting State

**Bad:**
```rust
// DON'T: Reuse engine across independent sequences without reset
for sequence in sequences {
    for sample in sequence {
        engine.step_async(sample).await?;
    }
    // State from previous sequence pollutes next one!
}
```

**Good:**
```rust
// DO: Reset state between independent sequences
for sequence in sequences {
    engine.reset_async().await?;

    for sample in sequence {
        engine.step_async(sample).await?;
    }
}
```

---

## Performance Checklist

Before deploying streaming inference to production:

- [ ] Tune buffer size based on expected throughput
- [ ] Set appropriate batch size for latency requirements
- [ ] Configure max_latency_ms to balance throughput/latency
- [ ] Implement proper error handling with fallbacks
- [ ] Add monitoring and metrics collection
- [ ] Test backpressure behavior under load
- [ ] Validate memory usage under sustained load
- [ ] Implement state reset logic for independent sequences
- [ ] Add health checks and alerting
- [ ] Profile and optimize hot paths
- [ ] Document expected performance characteristics
- [ ] Test failure recovery mechanisms

---

## Conclusion

Streaming inference requires careful attention to:
1. **Async patterns**: Proper use of async/await without blocking
2. **Backpressure**: Bounded channels and flow control
3. **Error handling**: Graceful degradation and recovery
4. **Monitoring**: Metrics and observability
5. **Configuration**: Tuning for your specific requirements

Start with the balanced configuration, measure performance, then optimize based on your specific constraints.

For more performance optimization techniques, see [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md).
