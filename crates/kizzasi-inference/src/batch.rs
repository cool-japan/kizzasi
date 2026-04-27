//! Continuous batching for high-throughput inference
//!
//! This module implements continuous batching (also known as iteration-level scheduling),
//! which allows dynamic batching of inference requests to maximize GPU/CPU utilization
//! while maintaining low latency for individual requests.
//!
//! # Key Features
//!
//! - **Dynamic batch formation**: Requests are grouped on-the-fly
//! - **Early exit**: Completed sequences leave the batch immediately
//! - **Variable-length sequences**: Different requests can have different lengths
//! - **Priority scheduling**: High-priority requests can skip the queue
//!
//! # References
//!
//! - Orca paper: <https://www.usenix.org/system/files/osdi22-yu.pdf>
//! - vLLM: <https://arxiv.org/abs/2309.06180>

use crate::engine::{EngineConfig, InferenceEngine};
use crate::error::InferenceResult;
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

/// Continuous batching scheduler
pub struct BatchScheduler {
    config: BatchConfig,
    engine: InferenceEngine,
    /// Queue of pending requests
    pending: VecDeque<BatchRequest>,
    /// Currently processing requests
    active: Vec<BatchRequest>,
    /// Completed responses
    completed: Vec<BatchResponse>,
    /// Next request ID
    next_id: u64,
    /// Last batch formation time
    last_batch_time: Instant,
}

impl BatchScheduler {
    /// Create a new batch scheduler
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

    /// Submit a new inference request
    pub fn submit(&mut self, input: Array1<f32>, max_steps: usize) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let request = BatchRequest::new(id, input, max_steps);
        self.pending.push_back(request);

        id
    }

    /// Submit a request with priority
    pub fn submit_with_priority(
        &mut self,
        input: Array1<f32>,
        max_steps: usize,
        priority: Priority,
    ) -> u64 {
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

        id
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
            .min(self.config.max_batch_size - self.active.len());

        for _ in 0..batch_size {
            if let Some(request) = self.pending.pop_front() {
                self.active.push(request);
            }
        }

        self.last_batch_time = Instant::now();
    }

    /// Process one step for all active requests
    pub fn step(&mut self) -> InferenceResult<Vec<BatchResponse>> {
        // Form batch if needed
        if self.should_form_batch() {
            self.form_batch();
        }

        if self.active.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();
        let mut responses = Vec::new();

        // Process each active request
        let mut i = 0;
        while i < self.active.len() {
            let request = &mut self.active[i];

            // Run one inference step
            let output = self.engine.step(&request.input)?;

            request.current_step += 1;
            request.input = output.clone(); // Use output as next input

            // Check if complete
            if request.is_complete() {
                let completed_request = self.active.remove(i);
                let inference_time = start.elapsed().as_micros() as u64;

                responses.push(BatchResponse {
                    request_id: completed_request.id,
                    outputs: vec![output],
                    steps_completed: completed_request.current_step,
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
        let mut scheduler = BatchScheduler::new(batch_config, engine_config).unwrap();

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let id = scheduler.submit(input, 5);

        assert_eq!(id, 0);
        assert_eq!(scheduler.stats().pending_requests, 1);
    }

    #[test]
    fn test_scheduler_priority() {
        let batch_config = BatchConfig::new().with_priority();
        let engine_config = EngineConfig::new(3, 3);
        let mut scheduler = BatchScheduler::new(batch_config, engine_config).unwrap();

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Submit with different priorities
        let _id1 = scheduler.submit_with_priority(input.clone(), 5, Priority::Low);
        let _id2 = scheduler.submit_with_priority(input.clone(), 5, Priority::High);
        let _id3 = scheduler.submit_with_priority(input.clone(), 5, Priority::Normal);

        // High priority should be first
        assert_eq!(scheduler.pending[0].priority, Priority::High);
        assert_eq!(scheduler.stats().pending_requests, 3);
    }

    #[test]
    fn test_scheduler_stats() {
        let batch_config = BatchConfig::new();
        let engine_config = EngineConfig::new(3, 3);
        let mut scheduler = BatchScheduler::new(batch_config, engine_config).unwrap();

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        scheduler.submit(input.clone(), 5);
        scheduler.submit(input.clone(), 5);

        let stats = scheduler.stats();
        assert_eq!(stats.pending_requests, 2);
        assert_eq!(stats.total_submitted, 2);
    }
}
