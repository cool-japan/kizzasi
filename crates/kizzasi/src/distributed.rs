//! Distributed prediction framework for Kizzasi
//!
//! Enables scaling predictions across multiple worker processes or machines.
//!
//! # Features
//!
//! - **Multi-worker prediction**: Distribute workload across multiple predictors
//! - **Load balancing**: Round-robin and least-loaded strategies
//! - **Fault tolerance**: workers whose task has died are marked unhealthy and
//!   skipped by every load-balancing strategy; requests are retried on a
//!   different worker up to `max_retries`. Dead workers are not respawned.
//! - **Backpressure**: worker queues are bounded by `max_pending_per_worker`;
//!   a saturated pool reports `ResourceExhausted` instead of growing without
//!   limit
//! - **Both local and remote**: Support for multi-threaded and networked workers
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi::distributed::{DistributedPredictor, WorkerConfig};
//! use kizzasi::prelude::*;
//!
//! let config = KizzasiConfig::new()
//!     .input_dim(3)
//!     .output_dim(3)
//!     .hidden_dim(64);
//!
//! // Create distributed predictor with 4 workers
//! let mut dist_predictor = DistributedPredictor::new(config, 4).await?;
//!
//! // Predictions are automatically load-balanced across workers
//! let input = array![0.1, 0.2, 0.3];
//! let output = dist_predictor.predict(&input).await?;
//! ```

use crate::error::{KizzasiError, KizzasiResult};
use crate::predictor::Kizzasi;
use kizzasi_core::KizzasiConfig;
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{rng, RngExt};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

#[cfg(feature = "logic")]
use kizzasi_logic::GuardrailSet;

/// Load balancing strategy for distributed prediction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Simple round-robin distribution
    RoundRobin,
    /// Send to least loaded worker
    LeastLoaded,
    /// Random worker selection
    Random,
}

/// Configuration for distributed predictor
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Number of worker instances
    pub num_workers: usize,
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Maximum pending requests per worker
    pub max_pending_per_worker: usize,
    /// Enable automatic retry on worker failure
    pub auto_retry: bool,
    /// Maximum retry attempts
    pub max_retries: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            strategy: LoadBalancingStrategy::RoundRobin,
            max_pending_per_worker: 100,
            auto_retry: true,
            max_retries: 3,
        }
    }
}

impl DistributedConfig {
    /// Create a new distributed configuration
    pub fn new(num_workers: usize) -> Self {
        Self {
            num_workers,
            ..Default::default()
        }
    }

    /// Set load balancing strategy
    pub fn strategy(mut self, strategy: LoadBalancingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set maximum pending requests per worker
    pub fn max_pending_per_worker(mut self, max: usize) -> Self {
        self.max_pending_per_worker = max;
        self
    }

    /// Enable/disable automatic retry
    pub fn auto_retry(mut self, enabled: bool) -> Self {
        self.auto_retry = enabled;
        self
    }

    /// Set maximum retry attempts
    pub fn max_retries(mut self, max: usize) -> Self {
        self.max_retries = max;
        self
    }
}

/// Request message for workers
#[derive(Debug, Clone)]
struct PredictionRequest {
    input: Array1<f32>,
    response_tx: mpsc::UnboundedSender<KizzasiResult<Array1<f32>>>,
}

/// Control and work messages accepted by a worker task.
#[derive(Debug, Clone)]
enum WorkerMsg {
    /// Run one prediction.
    Predict(Box<PredictionRequest>),
    /// Replace the worker's guardrail set (or clear it with `None`).
    #[cfg(feature = "logic")]
    SetGuardrails(Box<Option<GuardrailSet>>),
}

/// Worker handle for tracking worker state
struct WorkerHandle {
    tx: mpsc::Sender<WorkerMsg>,
    pending_count: Arc<RwLock<usize>>,
    is_healthy: Arc<RwLock<bool>>,
}

impl WorkerHandle {
    fn new(tx: mpsc::Sender<WorkerMsg>) -> Self {
        Self {
            tx,
            pending_count: Arc::new(RwLock::new(0)),
            is_healthy: Arc::new(RwLock::new(true)),
        }
    }

    async fn is_healthy(&self) -> bool {
        *self.is_healthy.read().await
    }

    async fn mark_unhealthy(&self) {
        *self.is_healthy.write().await = false;
    }

    async fn pending_count(&self) -> usize {
        *self.pending_count.read().await
    }

    /// Enqueue a prediction without blocking.
    ///
    /// The channel is bounded by `max_pending_per_worker`, so a saturated
    /// worker is reported rather than growing an unbounded backlog.
    async fn send_request(&self, req: PredictionRequest) -> Result<(), SendFailure> {
        // Reserve the slot before enqueuing so `pending_count` never
        // undercounts a request that is already in flight.
        {
            let mut count = self.pending_count.write().await;
            *count += 1;
        }
        match self.tx.try_send(WorkerMsg::Predict(Box::new(req))) {
            Ok(()) => Ok(()),
            Err(err) => {
                self.decrement_pending().await;
                Err(match err {
                    mpsc::error::TrySendError::Full(_) => SendFailure::Saturated,
                    mpsc::error::TrySendError::Closed(_) => SendFailure::Closed,
                })
            }
        }
    }

    /// Send a control message, waiting for capacity if the worker is busy.
    async fn send_control(&self, msg: WorkerMsg) -> Result<(), SendFailure> {
        self.tx.send(msg).await.map_err(|_| SendFailure::Closed)
    }

    async fn decrement_pending(&self) {
        let mut count = self.pending_count.write().await;
        if *count > 0 {
            *count -= 1;
        }
    }
}

/// Why a worker refused a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SendFailure {
    /// The worker already holds `max_pending_per_worker` requests.
    Saturated,
    /// The worker task is gone.
    Closed,
}

/// Distributed predictor that manages multiple worker instances
///
/// This coordinator distributes prediction workload across multiple Kizzasi
/// predictor instances, providing parallel inference capabilities.
pub struct DistributedPredictor {
    config: DistributedConfig,
    #[allow(dead_code)]
    model_config: KizzasiConfig,
    workers: Vec<WorkerHandle>,
    current_worker: Arc<RwLock<usize>>, // For round-robin
    #[cfg(feature = "logic")]
    guardrails: Option<GuardrailSet>,
}

impl DistributedPredictor {
    /// Create a new distributed predictor
    ///
    /// # Arguments
    ///
    /// * `model_config` - Configuration for the underlying model
    /// * `num_workers` - Number of worker instances to create
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = KizzasiConfig::new().input_dim(3).output_dim(3);
    /// let predictor = DistributedPredictor::new(config, 4).await?;
    /// ```
    pub async fn new(model_config: KizzasiConfig, num_workers: usize) -> KizzasiResult<Self> {
        let dist_config = DistributedConfig::new(num_workers);
        Self::with_config(model_config, dist_config).await
    }

    /// Create a distributed predictor with custom configuration
    pub async fn with_config(
        model_config: KizzasiConfig,
        dist_config: DistributedConfig,
    ) -> KizzasiResult<Self> {
        #[cfg(feature = "logic")]
        {
            Self::build(model_config, dist_config, None).await
        }
        #[cfg(not(feature = "logic"))]
        {
            Self::build(model_config, dist_config).await
        }
    }

    /// Create a distributed predictor whose workers all enforce `guardrails`.
    ///
    /// Guardrails supplied here are installed in every worker *before* it
    /// serves its first request, so no prediction can escape unconstrained.
    #[cfg(feature = "logic")]
    pub async fn with_guardrails(
        model_config: KizzasiConfig,
        dist_config: DistributedConfig,
        guardrails: GuardrailSet,
    ) -> KizzasiResult<Self> {
        Self::build(model_config, dist_config, Some(guardrails)).await
    }

    #[cfg(feature = "logic")]
    async fn build(
        model_config: KizzasiConfig,
        dist_config: DistributedConfig,
        guardrails: Option<GuardrailSet>,
    ) -> KizzasiResult<Self> {
        let workers = Self::spawn_workers(&model_config, &dist_config, guardrails.clone());

        Ok(Self {
            config: dist_config,
            model_config,
            workers,
            current_worker: Arc::new(RwLock::new(0)),
            guardrails,
        })
    }

    #[cfg(not(feature = "logic"))]
    async fn build(
        model_config: KizzasiConfig,
        dist_config: DistributedConfig,
    ) -> KizzasiResult<Self> {
        let workers = Self::spawn_workers(&model_config, &dist_config);

        Ok(Self {
            config: dist_config,
            model_config,
            workers,
            current_worker: Arc::new(RwLock::new(0)),
        })
    }

    fn spawn_workers(
        model_config: &KizzasiConfig,
        dist_config: &DistributedConfig,
        #[cfg(feature = "logic")] guardrails: Option<GuardrailSet>,
    ) -> Vec<WorkerHandle> {
        let mut workers = Vec::with_capacity(dist_config.num_workers);

        // Bounded so a producer faster than the workers is told to back off
        // instead of growing an unbounded queue until the process OOMs.
        let capacity = dist_config.max_pending_per_worker.max(1);

        for _ in 0..dist_config.num_workers {
            let (tx, rx) = mpsc::channel(capacity);
            let worker_handle = WorkerHandle::new(tx);

            let worker_model_config = model_config.clone();
            let is_healthy = worker_handle.is_healthy.clone();
            #[cfg(feature = "logic")]
            let worker_guardrails = guardrails.clone();

            tokio::spawn(async move {
                Self::worker_task(
                    worker_model_config,
                    rx,
                    is_healthy,
                    #[cfg(feature = "logic")]
                    worker_guardrails,
                )
                .await;
            });

            workers.push(worker_handle);
        }

        workers
    }

    /// Worker task that processes prediction requests
    async fn worker_task(
        config: KizzasiConfig,
        mut rx: mpsc::Receiver<WorkerMsg>,
        is_healthy: Arc<RwLock<bool>>,
        #[cfg(feature = "logic")] guardrails: Option<GuardrailSet>,
    ) {
        // Create predictor for this worker
        let mut predictor = match Kizzasi::new(config) {
            Ok(p) => p,
            Err(e) => {
                *is_healthy.write().await = false;
                tracing::error!("Failed to create predictor for worker: {:?}", e);
                return;
            }
        };

        // Install guardrails before serving anything, so no prediction can
        // escape unconstrained.
        #[cfg(feature = "logic")]
        if let Some(set) = guardrails {
            predictor.set_guardrails(set);
        }

        while let Some(msg) = rx.recv().await {
            match msg {
                WorkerMsg::Predict(req) => {
                    let result = predictor.step(&req.input);
                    let _ = req.response_tx.send(result);
                    // `predict` owns the pending-count decrement. Decrementing
                    // here as well saturated the counter at 0 and made the
                    // LeastLoaded strategy always return worker 0.
                }
                #[cfg(feature = "logic")]
                WorkerMsg::SetGuardrails(set) => match *set {
                    Some(guardrails) => predictor.set_guardrails(guardrails),
                    None => predictor.clear_guardrails(),
                },
            }
        }
    }

    /// Set guardrails for all workers.
    ///
    /// The set is broadcast to every worker, so every subsequent prediction is
    /// constrained no matter which worker serves it. Returns an error if a
    /// worker could not be reached; the workers that did accept the update
    /// keep it, and unreachable workers are marked unhealthy.
    ///
    /// This is `async` and fallible because it performs real work: the
    /// previous signature stored the set on the coordinator and never told any
    /// worker, so predictions came back unconstrained.
    #[cfg(feature = "logic")]
    pub async fn set_guardrails(&mut self, guardrails: GuardrailSet) -> KizzasiResult<()> {
        self.broadcast_guardrails(Some(guardrails)).await
    }

    /// Clear guardrails on every worker.
    #[cfg(feature = "logic")]
    pub async fn clear_guardrails(&mut self) -> KizzasiResult<()> {
        self.broadcast_guardrails(None).await
    }

    #[cfg(feature = "logic")]
    async fn broadcast_guardrails(
        &mut self,
        guardrails: Option<GuardrailSet>,
    ) -> KizzasiResult<()> {
        self.guardrails = guardrails.clone();

        let mut failures = 0usize;
        for worker in &self.workers {
            let msg = WorkerMsg::SetGuardrails(Box::new(guardrails.clone()));
            if worker.send_control(msg).await.is_err() {
                worker.mark_unhealthy().await;
                failures += 1;
            }
        }

        if failures > 0 {
            return Err(KizzasiError::inference(format!(
                "failed to apply guardrails to {failures} of {} workers",
                self.workers.len()
            )));
        }
        Ok(())
    }

    /// The guardrail set currently installed on the workers, if any.
    #[cfg(feature = "logic")]
    pub fn guardrails(&self) -> Option<&GuardrailSet> {
        self.guardrails.as_ref()
    }

    /// Indices of the workers that are still healthy.
    ///
    /// Every strategy routes only to these; sending to a worker whose task has
    /// died would otherwise be retried forever under RoundRobin.
    async fn healthy_indices(&self) -> Vec<usize> {
        let mut healthy = Vec::with_capacity(self.workers.len());
        for (idx, worker) in self.workers.iter().enumerate() {
            if worker.is_healthy().await {
                healthy.push(idx);
            }
        }
        healthy
    }

    /// Select a worker based on the load balancing strategy
    async fn select_worker(&self) -> KizzasiResult<usize> {
        let healthy = self.healthy_indices().await;
        if healthy.is_empty() {
            return Err(KizzasiError::inference("No healthy workers available"));
        }

        match self.config.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let mut current = self.current_worker.write().await;
                let position = *current % healthy.len();
                *current = current.wrapping_add(1);
                healthy
                    .get(position)
                    .copied()
                    .ok_or_else(|| KizzasiError::inference("No healthy workers available"))
            }
            LoadBalancingStrategy::LeastLoaded => {
                let mut min_load = usize::MAX;
                let mut min_idx = None;

                for &idx in &healthy {
                    let Some(worker) = self.workers.get(idx) else {
                        continue;
                    };
                    let pending = worker.pending_count().await;
                    if pending < min_load {
                        min_load = pending;
                        min_idx = Some(idx);
                    }
                }

                min_idx.ok_or_else(|| KizzasiError::inference("No healthy workers available"))
            }
            LoadBalancingStrategy::Random => {
                // A coarse system clock returns the same nanosecond value for
                // consecutive calls, which used to pin every "random" pick to
                // the same worker; use the process RNG instead.
                let mut generator = rng();
                let position = generator.random_range(0..healthy.len());
                healthy
                    .get(position)
                    .copied()
                    .ok_or_else(|| KizzasiError::inference("No healthy workers available"))
            }
        }
    }

    /// Perform a distributed prediction
    ///
    /// Automatically selects an appropriate worker and sends the request.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let input = array![0.1, 0.2, 0.3];
    /// let output = predictor.predict(&input).await?;
    /// ```
    pub async fn predict(&self, input: &Array1<f32>) -> KizzasiResult<Array1<f32>> {
        let mut attempts = 0;
        let max_attempts = if self.config.auto_retry {
            self.config.max_retries
        } else {
            1
        };

        loop {
            attempts += 1;

            // Select worker
            let worker_idx = self.select_worker().await?;
            let worker = self
                .workers
                .get(worker_idx)
                .ok_or_else(|| KizzasiError::inference("selected worker no longer exists"))?;

            // Create response channel
            let (response_tx, mut response_rx) = mpsc::unbounded_channel();

            // Send request
            let req = PredictionRequest {
                input: input.clone(),
                response_tx,
            };

            match worker.send_request(req).await {
                Ok(()) => {}
                Err(SendFailure::Closed) => {
                    // The worker task is gone; stop routing to it.
                    worker.mark_unhealthy().await;
                    if attempts >= max_attempts {
                        return Err(KizzasiError::inference("Failed to send request to worker"));
                    }
                    continue;
                }
                Err(SendFailure::Saturated) => {
                    if attempts >= max_attempts {
                        return Err(KizzasiError::resource_exhausted(
                            "distributed worker queue",
                            self.config.max_pending_per_worker,
                            self.config.max_pending_per_worker,
                            "raise max_pending_per_worker, add workers, or slow the producer",
                        ));
                    }
                    continue;
                }
            }

            // Wait for response. Exactly one decrement per accepted request:
            // the worker task no longer decrements, so the counter tracks the
            // real in-flight depth that LeastLoaded relies on.
            let response = response_rx.recv().await;
            worker.decrement_pending().await;

            match response {
                Some(Ok(output)) => {
                    return Ok(output);
                }
                Some(Err(e)) => {
                    if attempts >= max_attempts {
                        return Err(e);
                    }
                }
                None => {
                    // The worker dropped the response channel without
                    // answering: treat it as dead rather than retrying it.
                    worker.mark_unhealthy().await;
                    if attempts >= max_attempts {
                        return Err(KizzasiError::inference(
                            "Worker channel closed unexpectedly",
                        ));
                    }
                }
            }
        }
    }

    /// Batch predict with parallel processing
    ///
    /// Distributes batch items across workers for parallel processing.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let inputs = vec![
    ///     array![0.1, 0.2, 0.3],
    ///     array![0.4, 0.5, 0.6],
    ///     array![0.7, 0.8, 0.9],
    /// ];
    /// let outputs = predictor.predict_batch(&inputs).await?;
    /// ```
    pub async fn predict_batch(&self, inputs: &[Array1<f32>]) -> KizzasiResult<Vec<Array1<f32>>> {
        let mut tasks = Vec::with_capacity(inputs.len());

        for input in inputs {
            let input_clone = input.clone();
            let self_ref = self;
            tasks.push(async move { self_ref.predict(&input_clone).await });
        }

        let results = futures::future::join_all(tasks).await;

        results.into_iter().collect()
    }

    /// Get statistics about worker health and load
    pub async fn worker_stats(&self) -> Vec<WorkerStats> {
        let mut stats = Vec::with_capacity(self.workers.len());

        for (idx, worker) in self.workers.iter().enumerate() {
            stats.push(WorkerStats {
                worker_id: idx,
                is_healthy: worker.is_healthy().await,
                pending_requests: worker.pending_count().await,
            });
        }

        stats
    }

    /// Get the number of active workers
    pub async fn num_active_workers(&self) -> usize {
        let mut count = 0;
        for worker in &self.workers {
            if worker.is_healthy().await {
                count += 1;
            }
        }
        count
    }
}

/// Statistics for a worker instance
#[derive(Debug, Clone)]
pub struct WorkerStats {
    /// Worker ID
    pub worker_id: usize,
    /// Whether the worker is healthy
    pub is_healthy: bool,
    /// Number of pending requests
    pub pending_requests: usize,
}

/// Convenience function for distributed prediction with default settings.
///
/// Creates a distributed predictor with the specified number of workers,
/// processes all inputs in parallel, and returns the results.
///
/// # Arguments
///
/// * `model_config` - Configuration for the underlying model
/// * `inputs` - Batch of input signals to predict
/// * `num_workers` - Number of worker instances
///
/// # Example
///
/// ```rust,ignore
/// use kizzasi::distributed::distributed_predict;
/// use kizzasi::KizzasiConfig;
/// use scirs2_core::ndarray::array;
///
/// let config = KizzasiConfig::new().input_dim(3).output_dim(3).hidden_dim(64);
/// let inputs = vec![array![0.1, 0.2, 0.3], array![0.4, 0.5, 0.6]];
/// let outputs = distributed_predict(&config, &inputs, 4).await?;
/// ```
pub async fn distributed_predict(
    model_config: &KizzasiConfig,
    inputs: &[Array1<f32>],
    num_workers: usize,
) -> KizzasiResult<Vec<Array1<f32>>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let predictor = DistributedPredictor::new(model_config.clone(), num_workers).await?;
    predictor.predict_batch(inputs).await
}

/// Convenience function for distributed prediction with custom configuration.
///
/// Like [`distributed_predict`] but accepts a [`DistributedConfig`] for
/// fine-grained control over load balancing, retry behavior, etc.
///
/// # Example
///
/// ```rust,ignore
/// use kizzasi::distributed::{distributed_predict_with_config, DistributedConfig, LoadBalancingStrategy};
/// use kizzasi::KizzasiConfig;
/// use scirs2_core::ndarray::array;
///
/// let config = KizzasiConfig::new().input_dim(3).output_dim(3).hidden_dim(64);
/// let dist_config = DistributedConfig::new(4).strategy(LoadBalancingStrategy::LeastLoaded);
/// let inputs = vec![array![0.1, 0.2, 0.3]];
/// let outputs = distributed_predict_with_config(&config, &inputs, dist_config).await?;
/// ```
pub async fn distributed_predict_with_config(
    model_config: &KizzasiConfig,
    inputs: &[Array1<f32>],
    dist_config: DistributedConfig,
) -> KizzasiResult<Vec<Array1<f32>>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let predictor = DistributedPredictor::with_config(model_config.clone(), dist_config).await?;
    predictor.predict_batch(inputs).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[tokio::test]
    async fn test_distributed_predictor_creation() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64);

        let predictor = DistributedPredictor::new(config, 2).await.unwrap();
        assert_eq!(predictor.workers.len(), 2);
    }

    #[tokio::test]
    async fn test_distributed_single_prediction() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64);

        let predictor = DistributedPredictor::new(config, 2).await.unwrap();

        let input = array![0.1, 0.2, 0.3];
        let output = predictor.predict(&input).await.unwrap();

        assert_eq!(output.len(), 3);
    }

    #[tokio::test]
    async fn test_distributed_batch_prediction() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let predictor = DistributedPredictor::new(config, 3).await.unwrap();

        let inputs = vec![
            array![0.1, 0.2],
            array![0.3, 0.4],
            array![0.5, 0.6],
            array![0.7, 0.8],
        ];

        let outputs = predictor.predict_batch(&inputs).await.unwrap();

        assert_eq!(outputs.len(), 4);
        for output in outputs {
            assert_eq!(output.len(), 2);
        }
    }

    #[tokio::test]
    async fn test_worker_stats() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let predictor = DistributedPredictor::new(config, 2).await.unwrap();

        let stats = predictor.worker_stats().await;
        assert_eq!(stats.len(), 2);

        for stat in stats {
            assert!(stat.is_healthy);
            assert_eq!(stat.pending_requests, 0);
        }
    }

    #[tokio::test]
    async fn test_load_balancing_strategies() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        // Test round-robin
        let dist_config = DistributedConfig::new(3).strategy(LoadBalancingStrategy::RoundRobin);
        let predictor = DistributedPredictor::with_config(config.clone(), dist_config)
            .await
            .unwrap();

        let input = array![0.1, 0.2];
        let _ = predictor.predict(&input).await.unwrap();
        let _ = predictor.predict(&input).await.unwrap();

        // Test least loaded
        let dist_config = DistributedConfig::new(3).strategy(LoadBalancingStrategy::LeastLoaded);
        let predictor = DistributedPredictor::with_config(config.clone(), dist_config)
            .await
            .unwrap();

        let _ = predictor.predict(&input).await.unwrap();

        // Test random
        let dist_config = DistributedConfig::new(3).strategy(LoadBalancingStrategy::Random);
        let predictor = DistributedPredictor::with_config(config, dist_config)
            .await
            .unwrap();

        let _ = predictor.predict(&input).await.unwrap();
    }

    #[tokio::test]
    async fn test_concurrent_predictions() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let predictor = Arc::new(DistributedPredictor::new(config, 4).await.unwrap());

        let mut tasks = vec![];
        for i in 0..10 {
            let predictor_clone = predictor.clone();
            tasks.push(tokio::spawn(async move {
                let input = array![i as f32 * 0.1, (i + 1) as f32 * 0.1];
                predictor_clone.predict(&input).await
            }));
        }

        let results = futures::future::join_all(tasks).await;
        assert_eq!(results.len(), 10);

        for result in results {
            assert!(result.unwrap().is_ok());
        }
    }

    #[tokio::test]
    async fn test_distributed_predict_convenience() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let inputs = vec![array![0.1, 0.2], array![0.3, 0.4], array![0.5, 0.6]];

        let outputs = distributed_predict(&config, &inputs, 2).await.unwrap();
        assert_eq!(outputs.len(), 3);
        for output in &outputs {
            assert_eq!(output.len(), 2);
        }
    }

    #[tokio::test]
    async fn test_distributed_predict_with_config() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let dist_config = DistributedConfig::new(2).strategy(LoadBalancingStrategy::LeastLoaded);

        let inputs = vec![array![0.1, 0.2]];

        let outputs = distributed_predict_with_config(&config, &inputs, dist_config)
            .await
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 2);
    }

    #[cfg(feature = "logic")]
    fn bounded_guardrails(lower: f32, upper: f32) -> GuardrailSet {
        use kizzasi_logic::{ConstraintBuilder, Guardrail};

        let mut guardrails = GuardrailSet::new();
        // `in_range`, not chained greater_eq/less_eq: ConstraintBuilder keeps
        // only the last bound set, so chaining would silently drop one side.
        let constraint = ConstraintBuilder::new()
            .name("bounds")
            .in_range(lower, upper)
            .build()
            .unwrap();
        guardrails.add_global(Guardrail::new(constraint, false));
        guardrails
    }

    #[cfg(feature = "logic")]
    #[tokio::test]
    async fn test_guardrails_reach_workers_at_construction() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let predictor = DistributedPredictor::with_guardrails(
            config,
            DistributedConfig::new(2),
            bounded_guardrails(-0.01, 0.01),
        )
        .await
        .unwrap();

        for _ in 0..8 {
            let output = predictor.predict(&array![5.0, -5.0]).await.unwrap();
            for &value in output.iter() {
                assert!(
                    (-0.011..=0.011).contains(&value),
                    "worker returned unconstrained value {value}"
                );
            }
        }
    }

    #[cfg(feature = "logic")]
    #[tokio::test]
    async fn test_set_guardrails_propagates_to_workers() {
        // Regression: `set_guardrails` used to store the set on the
        // coordinator and never tell any worker, so every prediction came back
        // unconstrained with no error.
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let mut predictor = DistributedPredictor::new(config, 2).await.unwrap();
        predictor
            .set_guardrails(bounded_guardrails(-0.01, 0.01))
            .await
            .unwrap();

        for _ in 0..8 {
            let output = predictor.predict(&array![5.0, -5.0]).await.unwrap();
            for &value in output.iter() {
                assert!(
                    (-0.011..=0.011).contains(&value),
                    "worker ignored the broadcast guardrails ({value})"
                );
            }
        }

        assert!(predictor.guardrails().is_some());
    }

    #[tokio::test]
    async fn test_pending_count_tracks_in_flight_requests() {
        // Regression: the pending counter was decremented twice per request,
        // saturating at 0, so LeastLoaded always returned worker 0.
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let dist_config = DistributedConfig::new(3).strategy(LoadBalancingStrategy::LeastLoaded);
        let predictor = Arc::new(
            DistributedPredictor::with_config(config, dist_config)
                .await
                .unwrap(),
        );

        // LeastLoaded selects on `pending_count`, so the counter is the root
        // cause: with a double decrement it saturated at 0 and every worker
        // tied, making the strategy always return worker 0. Assert the
        // accounting directly rather than racing the scheduler.
        let worker = &predictor.workers[0];
        let (response_tx, _response_rx) = mpsc::unbounded_channel();
        worker
            .send_request(PredictionRequest {
                input: array![0.1, 0.2],
                response_tx,
            })
            .await
            .unwrap();

        // The pending count must reflect the in-flight request instead of
        // being decremented twice back to zero.
        let mut observed = 0;
        for _ in 0..100 {
            observed = worker.pending_count().await;
            if observed >= 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(observed, 1, "pending count must track in-flight requests");
    }

    #[tokio::test]
    async fn test_saturated_worker_reports_resource_exhaustion() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let dist_config = DistributedConfig::new(1)
            .max_pending_per_worker(1)
            .auto_retry(false);
        let predictor = DistributedPredictor::with_config(config, dist_config)
            .await
            .unwrap();

        // Fill the single worker slot without draining it.
        let worker = &predictor.workers[0];
        for _ in 0..64 {
            let (response_tx, response_rx) = mpsc::unbounded_channel();
            std::mem::forget(response_rx);
            if worker
                .send_request(PredictionRequest {
                    input: array![0.1, 0.2],
                    response_tx,
                })
                .await
                .is_err()
            {
                // Saturation reached: the channel is bounded, as documented.
                return;
            }
        }

        panic!("worker channel accepted unbounded requests despite max_pending_per_worker = 1");
    }

    #[tokio::test]
    async fn test_distributed_predict_empty_batch() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let inputs: Vec<Array1<f32>> = vec![];
        let outputs = distributed_predict(&config, &inputs, 2).await.unwrap();
        assert!(outputs.is_empty());
    }
}
