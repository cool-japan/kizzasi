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
use crate::error::{InferenceError, InferenceResult};
use crate::sampling::{Sampler, SamplingStrategy};
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
    ///
    /// When enabled the effective batch size starts at 1 and is doubled (capped at
    /// `batch_size`) each time a batch fills before its latency deadline, and
    /// halved (floored at 1) each time the deadline fires with a partially filled
    /// batch. A quiet stream therefore keeps single-sample latency while a busy one
    /// converges on `batch_size`. When disabled the batch size is fixed at
    /// `batch_size`.
    pub adaptive_batching: bool,
}

/// Per-request overrides for a single streaming inference call
///
/// Adapters translate their protocol's optional sampling fields into this type.
/// Overrides the engine cannot honour are reported as errors by
/// [`StreamingEngine::step_async_with`] rather than being dropped silently.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingOverrides {
    /// Sampling temperature
    pub temperature: Option<f32>,
    /// Top-k truncation
    pub top_k: Option<usize>,
    /// Top-p (nucleus) truncation
    pub top_p: Option<f32>,
    /// Number of autoregressive steps to generate (defaults to 1)
    pub max_tokens: Option<usize>,
}

impl SamplingOverrides {
    /// Create an empty override set
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether any sampling parameter (as opposed to `max_tokens`) is set
    pub fn has_sampling_overrides(&self) -> bool {
        self.temperature.is_some() || self.top_k.is_some() || self.top_p.is_some()
    }
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

    /// Process one request with per-request sampling overrides
    ///
    /// Returns `overrides.max_tokens` autoregressive predictions (one by default).
    /// The engine's own sampling configuration is restored before returning, so
    /// concurrent requests do not inherit each other's overrides.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidConfiguration`] when an override cannot be
    /// honoured, instead of accepting and ignoring it:
    ///
    /// - `max_tokens == 0`;
    /// - a non-positive `temperature`, a `top_k` of zero, or a `top_p` outside
    ///   `(0, 1]`;
    /// - any sampling override on an engine with `use_embeddings == false`, whose
    ///   continuous predictions carry no categorical distribution to sample from.
    ///
    /// `top_k` and `top_p` may both be set: [`Sampler`] composes them (top-k
    /// mask, then top-p nucleus mask within the surviving candidates) rather
    /// than treating them as mutually exclusive.
    pub async fn step_async_with(
        &self,
        input: Array1<f32>,
        overrides: &SamplingOverrides,
    ) -> InferenceResult<Vec<Array1<f32>>> {
        let steps = overrides.max_tokens.unwrap_or(1);
        if steps == 0 {
            return Err(InferenceError::InvalidConfiguration(
                "max_tokens must be at least 1".to_string(),
            ));
        }
        if let Some(temperature) = overrides.temperature {
            // NaN is rejected too: it is neither `> 0.0` nor `<= 0.0`.
            if !(temperature.is_finite() && temperature > 0.0) {
                return Err(InferenceError::InvalidConfiguration(format!(
                    "temperature must be a positive finite number, got {}",
                    temperature
                )));
            }
        }
        if overrides.top_k == Some(0) {
            return Err(InferenceError::InvalidConfiguration(
                "top_k must be at least 1".to_string(),
            ));
        }
        if let Some(top_p) = overrides.top_p {
            if !(top_p.is_finite() && top_p > 0.0 && top_p <= 1.0) {
                return Err(InferenceError::InvalidConfiguration(format!(
                    "top_p must lie in (0, 1], got {}",
                    top_p
                )));
            }
        }

        let mut engine = self.engine.lock().await;

        let has_sampling_overrides = overrides.has_sampling_overrides();
        if has_sampling_overrides && !engine.config().use_embeddings {
            return Err(InferenceError::InvalidConfiguration(
                "temperature/top_k/top_p overrides require an engine configured with \
                 use_embeddings = true; continuous predictions have no categorical \
                 distribution to sample from"
                    .to_string(),
            ));
        }

        let previous = engine.sampler().config().clone();
        if has_sampling_overrides {
            let mut sampling = previous.clone();
            if let Some(temperature) = overrides.temperature {
                sampling.temperature = temperature;
                sampling.strategy = SamplingStrategy::Temperature;
            }
            if let Some(top_k) = overrides.top_k {
                sampling.strategy = SamplingStrategy::TopK;
                sampling.top_k = Some(top_k);
            }
            if let Some(top_p) = overrides.top_p {
                sampling.strategy = SamplingStrategy::TopP;
                sampling.top_p = Some(top_p);
            }
            *engine.sampler_mut() = Sampler::new(sampling);
        }

        let mut outputs = Vec::with_capacity(steps);
        let mut current = input;
        let mut result = Ok(());
        for _ in 0..steps {
            match engine.step(&current) {
                Ok(output) => {
                    current = output.clone();
                    outputs.push(output);
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }

        if has_sampling_overrides {
            *engine.sampler_mut() = Sampler::new(previous);
        }

        result.map(|()| outputs)
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
    ///
    /// The task ends as soon as the source stream is exhausted, which drops `tx`
    /// and terminates the returned stream. The latency deadline is advanced every
    /// time the timer fires — including when the batch is empty — so an idle
    /// stream parks on the timer instead of spinning on an already-expired
    /// deadline.
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

            // Adaptive batching starts conservatively and grows with load; a fixed
            // policy simply pins the target at the configured batch size.
            let max_batch = config.batch_size.max(1);
            let mut target_batch = if config.adaptive_batching {
                1
            } else {
                max_batch
            };

            loop {
                tokio::select! {
                    // Receive new input
                    maybe_input = input_stream.next() => {
                        let Some(input) = maybe_input else {
                            // Source exhausted: drain and finish.
                            break;
                        };
                        batch.push(input);

                        // Process batch if full or latency exceeded
                        if batch.len() >= target_batch || last_process.elapsed() >= max_latency {
                            let filled = batch.len() >= target_batch;
                            Self::process_batch(&engine, &mut batch, &tx).await;
                            last_process = Instant::now();

                            if config.adaptive_batching && filled {
                                // Demand is keeping up with the current target: grow.
                                target_batch = target_batch.saturating_mul(2).min(max_batch);
                            }
                        }
                    }
                    // Force batch processing on timeout
                    _ = tokio::time::sleep_until(last_process + max_latency) => {
                        let partial = !batch.is_empty() && batch.len() < target_batch;
                        if !batch.is_empty() {
                            Self::process_batch(&engine, &mut batch, &tx).await;
                        }
                        // Always advance the deadline, otherwise it stays in the
                        // past and the timer completes immediately forever.
                        last_process = Instant::now();

                        if config.adaptive_batching && partial {
                            // The deadline is being missed: shrink towards latency.
                            target_batch = (target_batch / 2).max(1);
                        }
                    }
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
    ///
    /// The accumulated samples are evaluated one at a time against the shared
    /// engine: batching here is an emission/latency policy, not a fused forward
    /// pass, because the model interface consumes one signal vector per call.
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

    /// Get information about the attached model, if any.
    pub async fn model_info(&self) -> Option<crate::engine::ModelInfo> {
        self.engine.lock().await.model_info()
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

/// Filter transformer: yields only items where the predicate returns true.
///
/// The predicate takes `&I` and runs synchronously inside the stream's filter
/// step, so there is no borrow held across an `.await` point — the future
/// returned to `StreamExt::filter` is already-ready (`futures::future::ready`).
/// This sidesteps the async lifetime issues that a naive predicate signature
/// (`async fn(&I) -> bool`) would otherwise introduce.
pub struct FilterTransformer<I, F>
where
    F: Fn(&I) -> bool + Send + Sync + 'static,
    I: Send + 'static,
{
    predicate: Arc<F>,
    _phantom: std::marker::PhantomData<I>,
}

impl<I, F> FilterTransformer<I, F>
where
    F: Fn(&I) -> bool + Send + Sync + 'static,
    I: Send + 'static,
{
    /// Create a new filter transformer with the given predicate.
    pub fn new(predicate: F) -> Self {
        Self {
            predicate: Arc::new(predicate),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<I, F> StreamTransformer<I, I> for FilterTransformer<I, F>
where
    F: Fn(&I) -> bool + Send + Sync + 'static,
    I: Send + 'static,
{
    fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = I> + Send>>,
    ) -> Pin<Box<dyn Stream<Item = I> + Send>> {
        let predicate = self.predicate.clone();
        Box::pin(input.filter(move |item| {
            let keep = (predicate)(item);
            futures::future::ready(keep)
        }))
    }
}

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
                    maybe_item = buffered.next() => {
                        match maybe_item {
                            Some(item) => {
                                last_item = Some(item);
                                timer.reset();
                            }
                            None => {
                                // Source exhausted: emit whatever is pending and
                                // terminate. Matching only `Some(..)` here used to
                                // disable this branch and leave the timer ticking
                                // forever, so the output stream never ended.
                                if let Some(item) = last_item.take() {
                                    yield item;
                                }
                                break;
                            }
                        }
                    }
                    _ = timer.tick() => {
                        if let Some(item) = last_item.take() {
                            yield item;
                        }
                    }
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

    async fn engine_with_model(config: StreamConfig) -> StreamingEngine {
        use kizzasi_model::s4::{S4Config, S4D};

        let model = S4D::new(
            S4Config::new()
                .input_dim(1)
                .hidden_dim(32)
                .state_dim(8)
                .num_layers(1)
                .diagonal(true),
        )
        .expect("S4D construction must succeed");

        let mut engine = StreamingEngine::new(config).expect("engine construction must succeed");
        engine.set_model(Box::new(model)).await;
        engine
    }

    /// Regression: the batched path never observed the end of its input stream, so
    /// the sender was never dropped and the output stream never terminated (while
    /// the task spun on an expired deadline, burning a core).
    #[tokio::test]
    async fn test_batched_predict_stream_terminates() {
        let mut config = StreamConfig::new();
        config.engine = EngineConfig::new(1, 1);
        config.batch_size = 4;
        config.max_latency_ms = 20;
        let engine = engine_with_model(config).await;

        let inputs: Vec<Array1<f32>> = (0..10)
            .map(|i| Array1::from_vec(vec![i as f32 * 0.1]))
            .collect();
        let stream = engine.predict_stream(futures::stream::iter(inputs));

        let collected = tokio::time::timeout(Duration::from_secs(10), stream.collect::<Vec<_>>())
            .await
            .expect("batched predict_stream must terminate");

        assert_eq!(collected.len(), 10);
        assert!(collected.iter().all(|r| r.is_ok()));
    }

    /// The same must hold with adaptive batching enabled.
    #[tokio::test]
    async fn test_adaptive_batched_predict_stream_terminates() {
        let mut config = StreamConfig::new();
        config.engine = EngineConfig::new(1, 1);
        config.batch_size = 8;
        config.max_latency_ms = 20;
        config.adaptive_batching = true;
        let engine = engine_with_model(config).await;

        let inputs: Vec<Array1<f32>> = (0..12)
            .map(|i| Array1::from_vec(vec![i as f32 * 0.1]))
            .collect();
        let stream = engine.predict_stream(futures::stream::iter(inputs));

        let collected = tokio::time::timeout(Duration::from_secs(10), stream.collect::<Vec<_>>())
            .await
            .expect("adaptive predict_stream must terminate");

        assert_eq!(collected.len(), 12);
    }

    /// Regression: `DebounceTransformer` never observed the end of its input.
    #[tokio::test]
    async fn test_debounce_transformer_terminates() {
        let transformer = DebounceTransformer::new(Duration::from_millis(5));
        let input: Pin<Box<dyn Stream<Item = i32> + Send>> =
            Box::pin(futures::stream::iter(vec![1, 2, 3]));
        let output = transformer.transform(input);

        let collected: Vec<i32> =
            tokio::time::timeout(Duration::from_secs(10), output.collect::<Vec<i32>>())
                .await
                .expect("debounce stream must terminate");

        assert!(!collected.is_empty());
        assert!(collected.iter().all(|v| (1..=3).contains(v)));
    }

    /// Regression: every network adapter accepted per-request sampling overrides
    /// and threw them away.
    #[tokio::test]
    async fn test_step_async_with_generates_max_tokens() {
        let mut config = StreamConfig::new();
        config.engine = EngineConfig::new(1, 1);
        let engine = engine_with_model(config).await;

        let overrides = SamplingOverrides {
            max_tokens: Some(4),
            ..SamplingOverrides::new()
        };
        let outputs = engine
            .step_async_with(Array1::from_vec(vec![0.5]), &overrides)
            .await
            .expect("rollout must succeed");
        assert_eq!(outputs.len(), 4);

        // No overrides at all still means a single step.
        let single = engine
            .step_async_with(Array1::from_vec(vec![0.5]), &SamplingOverrides::new())
            .await
            .expect("single step must succeed");
        assert_eq!(single.len(), 1);
    }

    #[tokio::test]
    async fn test_step_async_with_rejects_unsupported_overrides() {
        let mut config = StreamConfig::new();
        config.engine = EngineConfig::new(1, 1);
        let engine = engine_with_model(config).await;

        let input = Array1::from_vec(vec![0.5]);

        // Sampling overrides on a continuous engine are rejected, not ignored.
        let temperature = SamplingOverrides {
            temperature: Some(0.2),
            ..SamplingOverrides::new()
        };
        assert!(matches!(
            engine.step_async_with(input.clone(), &temperature).await,
            Err(InferenceError::InvalidConfiguration(_))
        ));

        let zero_tokens = SamplingOverrides {
            max_tokens: Some(0),
            ..SamplingOverrides::new()
        };
        assert!(matches!(
            engine.step_async_with(input, &zero_tokens).await,
            Err(InferenceError::InvalidConfiguration(_))
        ));
    }

    /// Regression: `top_k` and `top_p` overrides used to be rejected together
    /// as "mutually exclusive", even though `Sampler` composes them (top-k
    /// mask, then top-p nucleus mask). Both set together must now be
    /// accepted.
    #[tokio::test]
    async fn test_step_async_with_composes_top_k_and_top_p() {
        let mut config = StreamConfig::new();
        config.engine = EngineConfig::new(1, 1).use_embeddings(true);
        let engine = engine_with_model(config).await;

        let both = SamplingOverrides {
            top_k: Some(4),
            top_p: Some(0.9),
            max_tokens: Some(1),
            ..SamplingOverrides::new()
        };
        let result = engine
            .step_async_with(Array1::from_vec(vec![0.5]), &both)
            .await;
        assert!(
            result.is_ok(),
            "top_k and top_p together must be accepted (composed), got {:?}",
            result.err()
        );
    }

    /// On a discrete engine the overrides reach the sampler, and the engine's own
    /// configuration is restored afterwards.
    #[tokio::test]
    async fn test_step_async_with_applies_and_restores_sampling() {
        use crate::sampling::SamplingConfig;

        let mut config = StreamConfig::new();
        config.engine = EngineConfig::new(1, 1)
            .use_embeddings(true)
            .sampling(SamplingConfig::new().seed(42));
        let engine = engine_with_model(config).await;

        let overrides = SamplingOverrides {
            temperature: Some(0.25),
            ..SamplingOverrides::new()
        };
        let outputs = engine
            .step_async_with(Array1::from_vec(vec![0.5]), &overrides)
            .await
            .expect("override must be accepted");
        assert_eq!(outputs.len(), 1);

        let restored = engine.engine.lock().await;
        let sampling = restored.sampler().config();
        assert!((sampling.temperature - 1.0).abs() < 1e-6);
        assert_eq!(sampling.strategy, SamplingStrategy::Greedy);
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

    #[tokio::test]
    async fn test_filter_transformer_basic() {
        let transformer = FilterTransformer::new(|x: &i32| *x % 2 == 0);
        let input: Pin<Box<dyn Stream<Item = i32> + Send>> =
            Box::pin(futures::stream::iter(vec![1, 2, 3, 4, 5, 6]));
        let output = transformer.transform(input);
        let collected: Vec<i32> = output.collect().await;
        assert_eq!(collected, vec![2, 4, 6]);
    }

    #[tokio::test]
    async fn test_filter_transformer_passes_all() {
        let transformer = FilterTransformer::new(|_: &i32| true);
        let data = vec![10, 20, 30, 40, 50];
        let expected = data.clone();
        let input: Pin<Box<dyn Stream<Item = i32> + Send>> = Box::pin(futures::stream::iter(data));
        let output = transformer.transform(input);
        let collected: Vec<i32> = output.collect().await;
        assert_eq!(collected, expected);
    }

    #[tokio::test]
    async fn test_filter_transformer_passes_none() {
        let transformer = FilterTransformer::new(|_: &i32| false);
        let input: Pin<Box<dyn Stream<Item = i32> + Send>> =
            Box::pin(futures::stream::iter(vec![1, 2, 3, 4, 5]));
        let output = transformer.transform(input);
        let collected: Vec<i32> = output.collect().await;
        assert!(collected.is_empty());
    }

    #[tokio::test]
    async fn test_filter_transformer_preserves_order() {
        let transformer = FilterTransformer::new(|x: &i32| *x > 3);
        let input: Pin<Box<dyn Stream<Item = i32> + Send>> =
            Box::pin(futures::stream::iter(vec![3, 1, 4, 1, 5, 9, 2, 6]));
        let output = transformer.transform(input);
        let collected: Vec<i32> = output.collect().await;
        assert_eq!(collected, vec![4, 5, 9, 6]);
    }
}
