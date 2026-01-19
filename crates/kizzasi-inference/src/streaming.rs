//! Streaming inference with async support
//!
//! This module provides async/await support for real-time signal prediction,
//! enabling:
//! - **Non-blocking inference**: Process streams without blocking
//! - **Backpressure handling**: Manage flow control for real-time systems
//! - **Batch processing**: Accumulate multiple samples for efficiency
//! - **Stream adapters**: Connect to various async sources (MQTT, WebSocket, etc.)
//!
//! # Example
//!
//! ```ignore
//! use kizzasi_inference::streaming::{StreamingEngine, StreamConfig};
//! use futures::stream::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = StreamConfig::default();
//! let engine = StreamingEngine::new(config)?;
//!
//! // Assume input_stream is a futures::stream::Stream
//! let input_stream = futures::stream::iter(vec![]);
//!
//! // Process stream asynchronously
//! let mut predictions = engine.predict_stream(input_stream);
//! while let Some(prediction) = predictions.next().await {
//!     // Handle prediction
//! }
//! # Ok(())
//! # }
//! ```

use crate::engine::{EngineConfig, InferenceEngine};
use crate::error::InferenceResult;
use futures::stream::{Stream, StreamExt};
use kizzasi_model::AutoregressiveModel;
use scirs2_core::ndarray::Array1;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::{mpsc, Mutex};
use tokio::time::{Duration, Instant};

/// Configuration for streaming inference
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamConfig {
    /// Engine configuration
    pub engine: EngineConfig,
    /// Buffer size for input samples
    pub buffer_size: usize,
    /// Batch size for processing (1 = no batching)
    pub batch_size: usize,
    /// Maximum latency before forcing a batch (milliseconds)
    pub max_latency_ms: u64,
    /// Enable adaptive batching based on load
    pub adaptive_batching: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            engine: EngineConfig::default(),
            buffer_size: 1024,
            batch_size: 1,
            max_latency_ms: 100,
            adaptive_batching: false,
        }
    }
}

impl StreamConfig {
    /// Create a new stream configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set engine configuration
    pub fn engine(mut self, config: EngineConfig) -> Self {
        self.engine = config;
        self
    }

    /// Set buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set maximum latency
    pub fn max_latency_ms(mut self, ms: u64) -> Self {
        self.max_latency_ms = ms;
        self
    }

    /// Enable adaptive batching
    pub fn adaptive_batching(mut self, enable: bool) -> Self {
        self.adaptive_batching = enable;
        self
    }
}

/// Streaming inference engine with async support
pub struct StreamingEngine {
    config: StreamConfig,
    engine: Arc<Mutex<InferenceEngine>>,
}

impl StreamingEngine {
    /// Create a new streaming engine
    pub fn new(config: StreamConfig) -> InferenceResult<Self> {
        let engine = InferenceEngine::new(config.engine.clone());
        Ok(Self {
            config,
            engine: Arc::new(Mutex::new(engine)),
        })
    }

    /// Process a single sample asynchronously
    pub async fn step_async(&self, input: Array1<f32>) -> InferenceResult<Array1<f32>> {
        let mut engine = self.engine.lock().await;
        engine.step(&input)
    }

    /// Set the model for this streaming engine
    /// This will recreate the engine with the model to ensure proper configuration
    pub async fn set_model(&mut self, model: Box<dyn AutoregressiveModel>) {
        let config = {
            let engine = self.engine.lock().await;
            engine.config().clone()
        };

        // Create new engine with model (which auto-configures context)
        let new_engine = InferenceEngine::with_model(config, model);

        // Replace the engine
        *self.engine.lock().await = new_engine;
    }

    /// Process a stream of inputs and produce a stream of predictions
    ///
    /// This is the main entry point for streaming inference. It consumes
    /// an input stream and produces an output stream with predictions.
    pub fn predict_stream<S>(
        &self,
        input_stream: S,
    ) -> Pin<Box<dyn Stream<Item = InferenceResult<Array1<f32>>> + Send>>
    where
        S: Stream<Item = Array1<f32>> + Send + 'static,
    {
        let engine = self.engine.clone();
        let config = self.config.clone();

        if config.batch_size == 1 {
            // No batching - process each sample individually
            Box::pin(input_stream.then(move |input| {
                let engine = engine.clone();
                async move {
                    let mut engine = engine.lock().await;
                    engine.step(&input)
                }
            }))
        } else {
            // Batched processing with latency bounds
            Box::pin(Self::batched_stream_impl(input_stream, config, engine))
        }
    }

    /// Create a batched stream processor
    fn batched_stream_impl<S>(
        input_stream: S,
        config: StreamConfig,
        engine: Arc<Mutex<InferenceEngine>>,
    ) -> impl Stream<Item = InferenceResult<Array1<f32>>> + Send
    where
        S: Stream<Item = Array1<f32>> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel(config.buffer_size);

        // Spawn batching task
        tokio::spawn(async move {
            let mut input_stream = Box::pin(input_stream);
            let mut batch = Vec::with_capacity(config.batch_size);
            let mut last_process = Instant::now();
            let max_latency = Duration::from_millis(config.max_latency_ms);

            loop {
                tokio::select! {
                    // Receive new input
                    Some(input) = input_stream.next() => {
                        batch.push(input);

                        // Process batch if full or latency exceeded
                        if batch.len() >= config.batch_size
                            || last_process.elapsed() >= max_latency
                        {
                            Self::process_batch(&engine, &mut batch, &tx).await;
                            last_process = Instant::now();
                        }
                    }
                    // Force batch processing on timeout
                    _ = tokio::time::sleep_until(last_process + max_latency) => {
                        if !batch.is_empty() {
                            Self::process_batch(&engine, &mut batch, &tx).await;
                            last_process = Instant::now();
                        }
                    }
                    else => break,
                }
            }

            // Process remaining batch
            if !batch.is_empty() {
                Self::process_batch(&engine, &mut batch, &tx).await;
            }
        });

        tokio_stream::wrappers::ReceiverStream::new(rx)
    }

    /// Process a batch of inputs
    async fn process_batch(
        engine: &Arc<Mutex<InferenceEngine>>,
        batch: &mut Vec<Array1<f32>>,
        tx: &mpsc::Sender<InferenceResult<Array1<f32>>>,
    ) {
        let mut engine = engine.lock().await;

        for input in batch.drain(..) {
            let result = engine.step(&input);
            if tx.send(result).await.is_err() {
                // Receiver dropped, stop processing
                break;
            }
        }
    }

    /// Process a rollout of N steps asynchronously
    ///
    /// Given an initial input, predict N future steps autoregressively.
    pub async fn rollout_async(
        &self,
        initial: Array1<f32>,
        steps: usize,
    ) -> InferenceResult<Vec<Array1<f32>>> {
        let mut engine = self.engine.lock().await;
        let mut predictions = Vec::with_capacity(steps);
        let mut current = initial;

        for _ in 0..steps {
            let pred = engine.step(&current)?;
            predictions.push(pred.clone());
            current = pred;
        }

        Ok(predictions)
    }

    /// Reset the engine state asynchronously
    pub async fn reset_async(&self) -> InferenceResult<()> {
        let mut engine = self.engine.lock().await;
        engine.reset();
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }
}

/// A stream adapter that wraps a callback-based source
///
/// This is useful for integrating with event-driven systems
/// that don't naturally fit the Stream trait.
pub struct CallbackStream<T> {
    receiver: mpsc::Receiver<T>,
}

impl<T> CallbackStream<T> {
    /// Create a new callback stream with given buffer size
    pub fn new(buffer_size: usize) -> (CallbackStreamHandle<T>, Self) {
        let (tx, rx) = mpsc::channel(buffer_size);
        let handle = CallbackStreamHandle { sender: tx };
        let stream = CallbackStream { receiver: rx };
        (handle, stream)
    }
}

impl<T> Stream for CallbackStream<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

/// Handle for pushing data into a callback stream
pub struct CallbackStreamHandle<T> {
    sender: mpsc::Sender<T>,
}

impl<T> CallbackStreamHandle<T> {
    /// Push a new item into the stream
    pub async fn push(&self, item: T) -> Result<(), mpsc::error::SendError<T>> {
        self.sender.send(item).await
    }

    /// Try to push without blocking
    pub fn try_push(&self, item: T) -> Result<(), mpsc::error::TrySendError<T>> {
        self.sender.try_send(item)
    }
}

impl<T> Clone for CallbackStreamHandle<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
        }
    }
}

/// Metrics for streaming performance
#[derive(Debug, Clone, Default)]
pub struct StreamMetrics {
    /// Total samples processed
    pub samples_processed: usize,
    /// Total batches processed
    pub batches_processed: usize,
    /// Average batch size
    pub avg_batch_size: f32,
    /// Average latency per sample (microseconds)
    pub avg_latency_us: f32,
    /// Peak latency (microseconds)
    pub peak_latency_us: u64,
}

impl StreamMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update metrics with new batch
    pub fn update(&mut self, batch_size: usize, latency_us: u64) {
        self.samples_processed += batch_size;
        self.batches_processed += 1;
        self.avg_batch_size = self.samples_processed as f32 / self.batches_processed as f32;

        // Update running average latency
        let total_latency = self.avg_latency_us * (self.samples_processed - batch_size) as f32;
        self.avg_latency_us = (total_latency + latency_us as f32) / self.samples_processed as f32;

        self.peak_latency_us = self.peak_latency_us.max(latency_us);
    }
}

// ============================================================================
// Stream Transformers
// ============================================================================

/// Stream transformer trait for composable stream operations
pub trait StreamTransformer<I, O>: Send {
    /// Transform input stream to output stream
    fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = I> + Send>>,
    ) -> Pin<Box<dyn Stream<Item = O> + Send>>;
}

// Note: FilterTransformer temporarily disabled due to lifetime complexity
// TODO: Implement a simpler version or use external crate

/// Map transformer: transforms each item using a function
pub struct MapTransformer<I, O, F>
where
    F: Fn(I) -> O + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    mapper: Arc<F>,
    _phantom: std::marker::PhantomData<(I, O)>,
}

impl<I, O, F> MapTransformer<I, O, F>
where
    F: Fn(I) -> O + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    /// Create a new map transformer
    pub fn new(mapper: F) -> Self {
        Self {
            mapper: Arc::new(mapper),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<I, O, F> StreamTransformer<I, O> for MapTransformer<I, O, F>
where
    F: Fn(I) -> O + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = I> + Send>>,
    ) -> Pin<Box<dyn Stream<Item = O> + Send>> {
        let mapper = self.mapper.clone();
        Box::pin(input.map(move |item| mapper(item)))
    }
}

/// Buffer transformer: accumulates items into fixed-size chunks
pub struct BufferTransformer {
    buffer_size: usize,
}

impl BufferTransformer {
    /// Create a new buffer transformer
    pub fn new(buffer_size: usize) -> Self {
        Self { buffer_size }
    }
}

impl<T: Clone + Send + 'static> StreamTransformer<T, Vec<T>> for BufferTransformer {
    fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = T> + Send>>,
    ) -> Pin<Box<dyn Stream<Item = Vec<T>> + Send>> {
        let buffer_size = self.buffer_size;
        Box::pin(async_stream::stream! {
            let mut buffered = input;
            let mut buffer = Vec::with_capacity(buffer_size);

            while let Some(item) = buffered.next().await {
                buffer.push(item);
                if buffer.len() >= buffer_size {
                    yield buffer.clone();
                    buffer.clear();
                }
            }

            // Yield remaining items
            if !buffer.is_empty() {
                yield buffer;
            }
        })
    }
}

/// Debounce transformer: only emits items after a period of inactivity
pub struct DebounceTransformer {
    duration: Duration,
}

impl DebounceTransformer {
    /// Create a new debounce transformer
    pub fn new(duration: Duration) -> Self {
        Self { duration }
    }
}

impl<T: Clone + Send + 'static> StreamTransformer<T, T> for DebounceTransformer {
    fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = T> + Send>>,
    ) -> Pin<Box<dyn Stream<Item = T> + Send>> {
        let duration = self.duration;
        Box::pin(async_stream::stream! {
            let mut buffered = input;
            let mut last_item: Option<T> = None;
            let mut timer = tokio::time::interval(duration);

            loop {
                tokio::select! {
                    Some(item) = buffered.next() => {
                        last_item = Some(item);
                        timer.reset();
                    }
                    _ = timer.tick() => {
                        if let Some(item) = last_item.take() {
                            yield item;
                        }
                    }
                    else => break,
                }
            }
        })
    }
}

/// Throttle transformer: limits the rate of items
pub struct ThrottleTransformer {
    rate: Duration,
}

impl ThrottleTransformer {
    /// Create a new throttle transformer with given rate limit
    pub fn new(rate: Duration) -> Self {
        Self { rate }
    }
}

impl<T: Send + 'static> StreamTransformer<T, T> for ThrottleTransformer {
    fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = T> + Send>>,
    ) -> Pin<Box<dyn Stream<Item = T> + Send>> {
        let rate = self.rate;
        Box::pin(async_stream::stream! {
            let mut buffered = input;
            let mut interval = tokio::time::interval(rate);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            while let Some(item) = buffered.next().await {
                interval.tick().await;
                yield item;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;

    #[tokio::test]
    async fn test_streaming_engine_creation() {
        let config = StreamConfig::new();
        let engine = StreamingEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_step_async() {
        use kizzasi_model::s4::{S4Config, S4D};

        let model_config = S4Config::new()
            .input_dim(1)
            .hidden_dim(64)
            .state_dim(16)
            .num_layers(2)
            .diagonal(true);
        let model = S4D::new(model_config).unwrap();

        let mut config = StreamConfig::new();
        config.engine = EngineConfig::new(1, 10);
        let mut engine = StreamingEngine::new(config).unwrap();
        engine.set_model(Box::new(model)).await;

        let input = Array1::from_vec(vec![0.5]);
        let result = engine.step_async(input).await;
        if let Err(e) = &result {
            eprintln!("Error in test_step_async: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_rollout_async() {
        use kizzasi_model::rwkv::{Rwkv, RwkvConfig};

        let model_config = RwkvConfig::new()
            .input_dim(1)
            .hidden_dim(64)
            .intermediate_dim(256)
            .num_layers(2);
        let model = Rwkv::new(model_config).unwrap();

        let mut config = StreamConfig::new();
        config.engine = EngineConfig::new(1, 10);
        let mut engine = StreamingEngine::new(config).unwrap();
        engine.set_model(Box::new(model)).await;

        let initial = Array1::from_vec(vec![0.5]);
        let result = engine.rollout_async(initial, 10).await;
        if let Err(e) = &result {
            eprintln!("Error in test_rollout_async: {:?}", e);
        }
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 10);
    }

    #[tokio::test]
    async fn test_callback_stream() {
        let (handle, mut stream) = CallbackStream::new(10);

        tokio::spawn(async move {
            for i in 0..5 {
                let _ = handle.push(i).await;
            }
        });

        let mut count = 0;
        while stream.next().await.is_some() {
            count += 1;
            if count >= 5 {
                break;
            }
        }
        assert_eq!(count, 5);
    }

    #[tokio::test]
    async fn test_stream_metrics() {
        let mut metrics = StreamMetrics::new();
        metrics.update(10, 1000);
        metrics.update(5, 2000);

        assert_eq!(metrics.samples_processed, 15);
        assert_eq!(metrics.batches_processed, 2);
        assert!((metrics.avg_batch_size - 7.5).abs() < 0.001);
    }
}
