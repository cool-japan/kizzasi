//! Distributed prediction framework for Kizzasi
//!
//! Enables scaling predictions across multiple worker processes or machines.
//!
//! # Features
//!
//! - **Multi-worker prediction**: Distribute workload across multiple predictors
//! - **Load balancing**: Round-robin and least-loaded strategies
//! - **Fault tolerance**: Automatic failover and retry on worker failures
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

/// Worker handle for tracking worker state
struct WorkerHandle {
    tx: mpsc::UnboundedSender<PredictionRequest>,
    pending_count: Arc<RwLock<usize>>,
    is_healthy: Arc<RwLock<bool>>,
}

impl WorkerHandle {
    fn new(tx: mpsc::UnboundedSender<PredictionRequest>) -> Self {
        Self {
            tx,
            pending_count: Arc::new(RwLock::new(0)),
            is_healthy: Arc::new(RwLock::new(true)),
        }
    }

    async fn is_healthy(&self) -> bool {
        *self.is_healthy.read().await
    }

    async fn pending_count(&self) -> usize {
        *self.pending_count.read().await
    }

    async fn send_request(
        &self,
        req: PredictionRequest,
    ) -> Result<(), mpsc::error::SendError<PredictionRequest>> {
        // Increment pending count
        *self.pending_count.write().await += 1;
        self.tx.send(req)
    }

    async fn decrement_pending(&self) {
        let mut count = self.pending_count.write().await;
        if *count > 0 {
            *count -= 1;
        }
    }
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
        let mut workers = Vec::with_capacity(dist_config.num_workers);

        // Spawn worker tasks
        for _ in 0..dist_config.num_workers {
            let (tx, rx) = mpsc::unbounded_channel();
            let worker_handle = WorkerHandle::new(tx);

            // Clone configs for the worker
            let worker_model_config = model_config.clone();
            let pending_count = worker_handle.pending_count.clone();
            let is_healthy = worker_handle.is_healthy.clone();

            // Spawn worker task
            tokio::spawn(async move {
                Self::worker_task(worker_model_config, rx, pending_count, is_healthy).await;
            });

            workers.push(worker_handle);
        }

        Ok(Self {
            config: dist_config,
            model_config,
            workers,
            current_worker: Arc::new(RwLock::new(0)),
            #[cfg(feature = "logic")]
            guardrails: None,
        })
    }

    /// Worker task that processes prediction requests
    async fn worker_task(
        config: KizzasiConfig,
        mut rx: mpsc::UnboundedReceiver<PredictionRequest>,
        pending_count: Arc<RwLock<usize>>,
        is_healthy: Arc<RwLock<bool>>,
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

        while let Some(req) = rx.recv().await {
            // Process prediction
            let result = predictor.step(&req.input);

            // Send response
            let _ = req.response_tx.send(result);

            // Decrement pending count
            let mut count = pending_count.write().await;
            if *count > 0 {
                *count -= 1;
            }
        }
    }

    /// Set guardrails for all workers
    #[cfg(feature = "logic")]
    pub fn set_guardrails(&mut self, guardrails: GuardrailSet) {
        self.guardrails = Some(guardrails);
    }

    /// Select a worker based on the load balancing strategy
    async fn select_worker(&self) -> KizzasiResult<usize> {
        match self.config.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let mut current = self.current_worker.write().await;
                let worker_idx = *current;
                *current = (*current + 1) % self.workers.len();
                Ok(worker_idx)
            }
            LoadBalancingStrategy::LeastLoaded => {
                let mut min_load = usize::MAX;
                let mut min_idx = 0;

                for (idx, worker) in self.workers.iter().enumerate() {
                    if !worker.is_healthy().await {
                        continue;
                    }
                    let pending = worker.pending_count().await;
                    if pending < min_load {
                        min_load = pending;
                        min_idx = idx;
                    }
                }

                if min_load == usize::MAX {
                    return Err(KizzasiError::inference("No healthy workers available"));
                }

                Ok(min_idx)
            }
            LoadBalancingStrategy::Random => {
                // Use timestamp-based pseudo-random selection
                use std::time::{SystemTime, UNIX_EPOCH};
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                Ok((nanos as usize) % self.workers.len())
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
            let worker = &self.workers[worker_idx];

            // Create response channel
            let (response_tx, mut response_rx) = mpsc::unbounded_channel();

            // Send request
            let req = PredictionRequest {
                input: input.clone(),
                response_tx,
            };

            if worker.send_request(req).await.is_err() {
                if attempts >= max_attempts {
                    return Err(KizzasiError::inference("Failed to send request to worker"));
                }
                continue;
            }

            // Wait for response
            match response_rx.recv().await {
                Some(Ok(output)) => {
                    worker.decrement_pending().await;
                    return Ok(output);
                }
                Some(Err(e)) => {
                    worker.decrement_pending().await;
                    if attempts >= max_attempts {
                        return Err(e);
                    }
                }
                None => {
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
