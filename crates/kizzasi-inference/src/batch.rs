//! Continuous batching for multi-request inference
//!
//! This module implements continuous batching (also known as iteration-level
//! scheduling): requests join and leave the in-flight set between steps rather than
//! waiting for a whole batch to finish, so a short request is never held hostage by
//! a long one sharing its batch.
//!
//! # Key Features
//!
//! - **Dynamic batch formation**: Requests are grouped on-the-fly
//! - **Early exit**: Completed sequences leave the batch immediately
//! - **Variable-length sequences**: Different requests can have different lengths
//! - **Priority scheduling**: High-priority requests can skip the queue
//! - **Per-request state**: Each request carries its own hidden state, swapped into
//!   the shared engine for the duration of its step
//!
//! # What this does not do
//!
//! Within one scheduler step the active requests are evaluated **sequentially**
//! against a single engine. [`kizzasi_model::AutoregressiveModel`] consumes one
//! signal vector per call, so there is no fused batched forward pass here and no
//! amortisation of weight reads across requests: `max_batch_size` bounds
//! concurrency and memory, not arithmetic intensity. A backend that gains a batched
//! forward entry point can be plugged in behind [`BatchScheduler::step`] without
//! changing this module's scheduling contract.
//!
//! # References
//!
//! - Orca paper: <https://www.usenix.org/system/files/osdi22-yu.pdf>
//! - vLLM: <https://arxiv.org/abs/2309.06180>

use crate::engine::{EngineConfig, InferenceEngine};
use crate::error::{InferenceError, InferenceResult};
use kizzasi_core::HiddenState;
use kizzasi_model::AutoregressiveModel;
use scirs2_core::ndarray::Array1;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for continuous batching
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum waiting time before forming a batch (milliseconds)
    pub max_wait_ms: u64,
    /// Minimum batch size before processing (0 = process immediately)
    pub min_batch_size: usize,
    /// Enable priority-based scheduling
    pub enable_priority: bool,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait_ms: 10,
            min_batch_size: 1,
            enable_priority: false,
            max_seq_len: 2048,
        }
    }
}

impl BatchConfig {
    /// Create a new batch configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum batch size
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Set maximum wait time
    pub fn max_wait_ms(mut self, ms: u64) -> Self {
        self.max_wait_ms = ms;
        self
    }

    /// Set minimum batch size
    pub fn min_batch_size(mut self, size: usize) -> Self {
        self.min_batch_size = size;
        self
    }

    /// Enable priority scheduling
    pub fn with_priority(mut self) -> Self {
        self.enable_priority = true;
        self
    }

    /// Set the maximum generated sequence length
    ///
    /// Requests asking for more steps than this are rejected by
    /// [`BatchScheduler::submit`].
    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }
}

/// Priority level for inference requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// A single inference request in the batch
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Unique request ID
    pub id: u64,
    /// Input data
    pub input: Array1<f32>,
    /// Maximum number of steps to generate
    pub max_steps: usize,
    /// Priority level
    pub priority: Priority,
    /// Timestamp when request was received
    pub received_at: Instant,
    /// Current step number
    pub current_step: usize,
    /// Per-request hidden state
    ///
    /// Every request carries its own autoregressive state, which the scheduler
    /// swaps into the shared engine around each step. `None` until the first step,
    /// where it is initialised from the engine's model architecture.
    pub states: Option<Vec<HiddenState>>,
    /// Outputs generated so far, one per completed step
    pub outputs: Vec<Array1<f32>>,
}

impl BatchRequest {
    /// Create a new batch request
    pub fn new(id: u64, input: Array1<f32>, max_steps: usize) -> Self {
        Self {
            id,
            input,
            max_steps,
            priority: Priority::Normal,
            received_at: Instant::now(),
            current_step: 0,
            states: None,
            outputs: Vec::with_capacity(max_steps),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Check if request is complete
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.max_steps
    }

    /// Get waiting time in milliseconds
    pub fn wait_time_ms(&self) -> u64 {
        self.received_at.elapsed().as_millis() as u64
    }
}

/// Response from a batch inference request
#[derive(Debug, Clone)]
pub struct BatchResponse {
    /// Request ID
    pub request_id: u64,
    /// Generated outputs (one per step)
    pub outputs: Vec<Array1<f32>>,
    /// Number of steps completed
    pub steps_completed: usize,
    /// Whether the request is complete
    pub is_complete: bool,
    /// Total inference time in microseconds
    pub inference_time_us: u64,
}

/// Record of a request that has finished, retained for scheduler statistics
///
/// The generated outputs are handed to the caller in the [`BatchResponse`] rather
/// than duplicated here, so bookkeeping costs a fixed number of bytes per request.
#[derive(Debug, Clone)]
pub struct CompletedRequest {
    /// Request ID
    pub request_id: u64,
    /// Number of steps generated
    pub steps_completed: usize,
    /// Duration of the scheduler step that completed the request (microseconds)
    pub inference_time_us: u64,
}

/// Continuous batching scheduler
pub struct BatchScheduler {
    config: BatchConfig,
    engine: InferenceEngine,
    /// Queue of pending requests
    pending: VecDeque<BatchRequest>,
    /// Currently processing requests
    active: Vec<BatchRequest>,
    /// Records of requests that have finished
    completed: Vec<CompletedRequest>,
    /// Next request ID
    next_id: u64,
    /// Last batch formation time
    last_batch_time: Instant,
}

impl BatchScheduler {
    /// Create a new batch scheduler with no model attached
    ///
    /// The scheduler cannot run inference until a model is installed with
    /// [`BatchScheduler::set_model`] — [`BatchScheduler::step`] returns
    /// [`InferenceError::NotInitialized`] until then. Prefer
    /// [`BatchScheduler::with_model`], which installs one up front.
    pub fn new(config: BatchConfig, engine_config: EngineConfig) -> InferenceResult<Self> {
        let engine = InferenceEngine::new(engine_config);

        Ok(Self {
            config,
            engine,
            pending: VecDeque::new(),
            active: Vec::new(),
            completed: Vec::new(),
            next_id: 0,
            last_batch_time: Instant::now(),
        })
    }

    /// Create a new batch scheduler backed by `model`
    pub fn with_model(
        config: BatchConfig,
        engine_config: EngineConfig,
        model: Box<dyn AutoregressiveModel>,
    ) -> InferenceResult<Self> {
        let engine = InferenceEngine::with_model(engine_config, model);

        Ok(Self {
            config,
            engine,
            pending: VecDeque::new(),
            active: Vec::new(),
            completed: Vec::new(),
            next_id: 0,
            last_batch_time: Instant::now(),
        })
    }

    /// Install (or replace) the model used to serve batched requests
    ///
    /// In-flight per-request states are dropped, since they belong to the previous
    /// model's architecture.
    pub fn set_model(&mut self, model: Box<dyn AutoregressiveModel>) {
        self.engine.set_model(model);
        for request in &mut self.active {
            request.states = None;
        }
        for request in &mut self.pending {
            request.states = None;
        }
    }

    /// Check whether a model is installed
    pub fn has_model(&self) -> bool {
        self.engine.has_model()
    }

    /// Get the underlying inference engine
    pub fn engine(&self) -> &InferenceEngine {
        &self.engine
    }

    /// Get mutable access to the underlying inference engine
    pub fn engine_mut(&mut self) -> &mut InferenceEngine {
        &mut self.engine
    }

    /// Submit a new inference request
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidConfiguration`] when `max_steps` is zero or
    /// exceeds [`BatchConfig::max_seq_len`].
    pub fn submit(&mut self, input: Array1<f32>, max_steps: usize) -> InferenceResult<u64> {
        self.check_max_steps(max_steps)?;

        let id = self.next_id;
        self.next_id += 1;

        let request = BatchRequest::new(id, input, max_steps);
        self.pending.push_back(request);

        Ok(id)
    }

    /// Submit a request with priority
    ///
    /// # Errors
    ///
    /// See [`BatchScheduler::submit`].
    pub fn submit_with_priority(
        &mut self,
        input: Array1<f32>,
        max_steps: usize,
        priority: Priority,
    ) -> InferenceResult<u64> {
        self.check_max_steps(max_steps)?;

        let id = self.next_id;
        self.next_id += 1;

        let request = BatchRequest::new(id, input, max_steps).with_priority(priority);

        // Insert based on priority if enabled
        if self.config.enable_priority {
            let insert_pos = self
                .pending
                .iter()
                .position(|r| r.priority < priority)
                .unwrap_or(self.pending.len());
            self.pending.insert(insert_pos, request);
        } else {
            self.pending.push_back(request);
        }

        Ok(id)
    }

    /// Enforce the configured sequence-length limit on an incoming request
    fn check_max_steps(&self, max_steps: usize) -> InferenceResult<()> {
        if max_steps == 0 {
            return Err(InferenceError::InvalidConfiguration(
                "max_steps must be at least 1".to_string(),
            ));
        }
        if max_steps > self.config.max_seq_len {
            return Err(InferenceError::InvalidConfiguration(format!(
                "max_steps {} exceeds the configured max_seq_len {}",
                max_steps, self.config.max_seq_len
            )));
        }
        Ok(())
    }

    /// Check if it's time to form a new batch
    fn should_form_batch(&self) -> bool {
        if self.pending.is_empty() {
            return false;
        }

        // Check if we have enough requests
        if self.pending.len() >= self.config.min_batch_size {
            return true;
        }

        // Check if we've waited long enough
        let wait_time = self.last_batch_time.elapsed();
        wait_time >= Duration::from_millis(self.config.max_wait_ms)
    }

    /// Form a new batch from pending requests
    fn form_batch(&mut self) {
        let batch_size = self
            .config
            .max_batch_size
            .min(self.pending.len())
            .min(self.config.max_batch_size.saturating_sub(self.active.len()));

        for _ in 0..batch_size {
            if let Some(request) = self.pending.pop_front() {
                self.active.push(request);
            }
        }

        self.last_batch_time = Instant::now();
    }

    /// Process one step for all active requests
    ///
    /// Each active request is advanced by exactly one autoregressive step against
    /// its own hidden state, which is swapped into the shared engine for the
    /// duration of the step and swapped back out afterwards — requests never
    /// observe each other's history.
    ///
    /// Requests that reach `max_steps` leave the batch immediately and are
    /// returned (and recorded in [`BatchScheduler::stats`]) with the full list of
    /// their generated outputs.
    pub fn step(&mut self) -> InferenceResult<Vec<BatchResponse>> {
        // Form a batch when the policy says so, and unconditionally when nothing is
        // active but work is queued — otherwise the scheduler would spin without
        // making progress until `max_wait_ms` expires.
        if self.should_form_batch() || (self.active.is_empty() && !self.pending.is_empty()) {
            self.form_batch();
        }

        if self.active.is_empty() {
            return Ok(Vec::new());
        }

        if !self.engine.has_model() {
            return Err(InferenceError::NotInitialized);
        }

        let start = Instant::now();
        let mut responses = Vec::new();

        // Process each active request. Completed requests are removed with
        // `swap_remove`, so draining a batch of n costs O(n) rather than O(n^2);
        // the active set is unordered, scheduling order lives in `pending`.
        let mut i = 0;
        while i < self.active.len() {
            let request_states = match self.active[i].states.take() {
                Some(states) => states,
                None => self.engine.fresh_states(),
            };

            // Give the engine this request's state for the duration of the step.
            let engine_states = self.engine.swap_states(request_states)?;
            let step_result = self.engine.step(&self.active[i].input);
            // Take the request's updated state back out, restoring the engine's own.
            let updated_states = self.engine.swap_states(engine_states)?;

            let output = step_result?;

            let request = &mut self.active[i];
            request.states = Some(updated_states);
            request.current_step += 1;
            request.input = output.clone(); // Use output as next input
            request.outputs.push(output);

            // Check if complete
            if request.is_complete() {
                let completed_request = self.active.swap_remove(i);
                let inference_time = start.elapsed().as_micros() as u64;

                self.completed.push(CompletedRequest {
                    request_id: completed_request.id,
                    steps_completed: completed_request.current_step,
                    inference_time_us: inference_time,
                });
                responses.push(BatchResponse {
                    request_id: completed_request.id,
                    steps_completed: completed_request.current_step,
                    outputs: completed_request.outputs,
                    is_complete: true,
                    inference_time_us: inference_time,
                });
            } else {
                i += 1;
            }
        }

        Ok(responses)
    }

    /// Process all active and pending requests until completion
    pub fn process_all(&mut self) -> InferenceResult<Vec<BatchResponse>> {
        let mut all_responses = Vec::new();

        while !self.pending.is_empty() || !self.active.is_empty() {
            let responses = self.step()?;
            all_responses.extend(responses);
        }

        Ok(all_responses)
    }

    /// Records of all requests completed since the last [`BatchScheduler::reset`]
    pub fn completed(&self) -> &[CompletedRequest] {
        &self.completed
    }

    /// Get statistics about the scheduler
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            pending_requests: self.pending.len(),
            active_requests: self.active.len(),
            completed_requests: self.completed.len(),
            total_submitted: self.next_id,
        }
    }

    /// Reset the scheduler
    pub fn reset(&mut self) {
        self.pending.clear();
        self.active.clear();
        self.completed.clear();
        self.engine.reset();
    }
}

/// Statistics about the batch scheduler
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub pending_requests: usize,
    pub active_requests: usize,
    pub completed_requests: usize,
    pub total_submitted: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config() {
        let config = BatchConfig::new()
            .max_batch_size(16)
            .max_wait_ms(5)
            .min_batch_size(4)
            .with_priority();

        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.max_wait_ms, 5);
        assert_eq!(config.min_batch_size, 4);
        assert!(config.enable_priority);
    }

    #[test]
    fn test_batch_request() {
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let request = BatchRequest::new(1, input, 10);

        assert_eq!(request.id, 1);
        assert_eq!(request.max_steps, 10);
        assert_eq!(request.current_step, 0);
        assert!(!request.is_complete());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_scheduler_creation() {
        let batch_config = BatchConfig::new();
        let engine_config = EngineConfig::new(3, 3);

        let scheduler = BatchScheduler::new(batch_config, engine_config);
        assert!(scheduler.is_ok());
    }

    #[test]
    fn test_scheduler_submit() {
        let batch_config = BatchConfig::new();
        let engine_config = EngineConfig::new(3, 3);
        let mut scheduler = BatchScheduler::new(batch_config, engine_config)
            .expect("scheduler construction must succeed");

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let id = scheduler.submit(input, 5).expect("submit must succeed");

        assert_eq!(id, 0);
        assert_eq!(scheduler.stats().pending_requests, 1);
    }

    #[test]
    fn test_scheduler_priority() {
        let batch_config = BatchConfig::new().with_priority();
        let engine_config = EngineConfig::new(3, 3);
        let mut scheduler = BatchScheduler::new(batch_config, engine_config)
            .expect("scheduler construction must succeed");

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Submit with different priorities
        let _id1 = scheduler
            .submit_with_priority(input.clone(), 5, Priority::Low)
            .expect("submit must succeed");
        let _id2 = scheduler
            .submit_with_priority(input.clone(), 5, Priority::High)
            .expect("submit must succeed");
        let _id3 = scheduler
            .submit_with_priority(input.clone(), 5, Priority::Normal)
            .expect("submit must succeed");

        // High priority should be first
        assert_eq!(scheduler.pending[0].priority, Priority::High);
        assert_eq!(scheduler.stats().pending_requests, 3);
    }

    #[test]
    fn test_scheduler_stats() {
        let batch_config = BatchConfig::new();
        let engine_config = EngineConfig::new(3, 3);
        let mut scheduler = BatchScheduler::new(batch_config, engine_config)
            .expect("scheduler construction must succeed");

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        scheduler
            .submit(input.clone(), 5)
            .expect("submit must succeed");
        scheduler
            .submit(input.clone(), 5)
            .expect("submit must succeed");

        let stats = scheduler.stats();
        assert_eq!(stats.pending_requests, 2);
        assert_eq!(stats.total_submitted, 2);
    }

    use crate::testutil::CountingModel;

    /// Build a small deterministic model for scheduler tests.
    fn test_model() -> Box<dyn AutoregressiveModel> {
        Box::new(CountingModel::new())
    }

    /// Regression: `BatchScheduler` had no way to attach a model at all, so every
    /// call to `step`/`process_all` failed with `NotInitialized`.
    #[test]
    fn test_scheduler_process_all_with_model() {
        let batch_config = BatchConfig::new().max_batch_size(4).min_batch_size(1);
        let engine_config = EngineConfig::new(1, 1);
        let mut scheduler = BatchScheduler::with_model(batch_config, engine_config, test_model())
            .expect("scheduler construction must succeed");

        assert!(scheduler.has_model());

        scheduler
            .submit(Array1::from_vec(vec![0.5]), 3)
            .expect("submit must succeed");
        scheduler
            .submit(Array1::from_vec(vec![0.25]), 5)
            .expect("submit must succeed");

        let mut responses = scheduler.process_all().expect("process_all must succeed");
        responses.sort_by_key(|r| r.request_id);

        assert_eq!(responses.len(), 2);
        // Every generated step is returned, not just the last one.
        assert_eq!(responses[0].steps_completed, 3);
        assert_eq!(responses[0].outputs.len(), 3);
        assert_eq!(responses[1].steps_completed, 5);
        assert_eq!(responses[1].outputs.len(), 5);

        // `completed_requests` used to be permanently zero.
        let stats = scheduler.stats();
        assert_eq!(stats.completed_requests, 2);
        assert_eq!(stats.pending_requests, 0);
        assert_eq!(stats.active_requests, 0);
        assert_eq!(scheduler.completed().len(), 2);
    }

    /// Regression: all active requests shared the engine's single hidden state, so
    /// a request's output depended on which other requests were in flight.
    #[test]
    fn test_requests_do_not_contaminate_each_other() {
        let batch_config = BatchConfig::new().max_batch_size(4).min_batch_size(1);
        let mut scheduler = BatchScheduler::with_model(
            batch_config,
            EngineConfig::new(1, 1),
            Box::new(CountingModel::new()),
        )
        .expect("scheduler construction must succeed");

        let input = Array1::from_vec(vec![0.5]);
        scheduler
            .submit(input.clone(), 4)
            .expect("submit must succeed");
        scheduler
            .submit(input.clone(), 4)
            .expect("submit must succeed");

        let mut responses = scheduler.process_all().expect("process_all must succeed");
        responses.sort_by_key(|r| r.request_id);
        assert_eq!(responses.len(), 2);

        // Identical inputs sharing a batch must produce identical outputs.
        assert_eq!(
            responses[0].outputs, responses[1].outputs,
            "batched requests must be independent"
        );

        // And they must match a lone request run on its own scheduler.
        let mut solo = BatchScheduler::with_model(
            BatchConfig::new().max_batch_size(4).min_batch_size(1),
            EngineConfig::new(1, 1),
            Box::new(CountingModel::new()),
        )
        .expect("scheduler construction must succeed");
        solo.submit(input, 4).expect("submit must succeed");
        let solo_responses = solo.process_all().expect("process_all must succeed");
        assert_eq!(solo_responses.len(), 1);
        assert_eq!(solo_responses[0].outputs, responses[0].outputs);
    }

    #[test]
    fn test_scheduler_without_model_reports_not_initialized() {
        let mut scheduler = BatchScheduler::new(BatchConfig::new(), EngineConfig::new(1, 1))
            .expect("scheduler construction must succeed");
        assert!(!scheduler.has_model());

        scheduler
            .submit(Array1::from_vec(vec![0.5]), 2)
            .expect("submit must succeed");

        assert!(matches!(
            scheduler.step(),
            Err(InferenceError::NotInitialized)
        ));
    }

    /// Regression: `BatchConfig::max_seq_len` was declared, defaulted and never read.
    #[test]
    fn test_max_seq_len_is_enforced() {
        let mut scheduler =
            BatchScheduler::new(BatchConfig::new().max_seq_len(4), EngineConfig::new(1, 1))
                .expect("scheduler construction must succeed");

        let input = Array1::from_vec(vec![0.5]);
        assert!(scheduler.submit(input.clone(), 4).is_ok());
        assert!(matches!(
            scheduler.submit(input.clone(), 5),
            Err(InferenceError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            scheduler.submit_with_priority(input.clone(), 9, Priority::High),
            Err(InferenceError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            scheduler.submit(input, 0),
            Err(InferenceError::InvalidConfiguration(_))
        ));
        assert_eq!(scheduler.stats().pending_requests, 1);
    }
}
