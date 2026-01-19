//! Batch processing utilities for efficient stream operations
//!
//! This module provides tools for batching stream data and processing
//! multiple samples efficiently.

use crate::error::{IoError, IoResult};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Batch configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size (number of samples)
    pub max_size: usize,
    /// Maximum time to wait before flushing batch
    pub max_wait_time: Duration,
    /// Minimum batch size before processing
    pub min_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_size: 1024,
            max_wait_time: Duration::from_millis(100),
            min_size: 1,
        }
    }
}

impl BatchConfig {
    /// Create configuration for low latency
    pub fn low_latency() -> Self {
        Self {
            max_size: 256,
            max_wait_time: Duration::from_millis(10),
            min_size: 1,
        }
    }

    /// Create configuration for high throughput
    pub fn high_throughput() -> Self {
        Self {
            max_size: 4096,
            max_wait_time: Duration::from_millis(500),
            min_size: 512,
        }
    }
}

/// Batch accumulator for collecting samples
pub struct BatchAccumulator {
    config: BatchConfig,
    buffer: VecDeque<f32>,
    last_flush: Instant,
}

impl BatchAccumulator {
    /// Create new batch accumulator
    pub fn new(config: BatchConfig) -> Self {
        let capacity = config.max_size;
        Self {
            config,
            buffer: VecDeque::with_capacity(capacity),
            last_flush: Instant::now(),
        }
    }

    /// Push sample to batch
    pub fn push(&mut self, sample: f32) {
        self.buffer.push_back(sample);
    }

    /// Push multiple samples
    pub fn push_slice(&mut self, samples: &[f32]) {
        self.buffer.extend(samples.iter());
    }

    /// Check if batch should be flushed
    pub fn should_flush(&self) -> bool {
        // Flush if max size reached
        if self.buffer.len() >= self.config.max_size {
            return true;
        }

        // Flush if timeout reached and min size met
        if self.buffer.len() >= self.config.min_size
            && self.last_flush.elapsed() >= self.config.max_wait_time
        {
            return true;
        }

        false
    }

    /// Flush accumulated samples
    pub fn flush(&mut self) -> Vec<f32> {
        let samples: Vec<f32> = self.buffer.drain(..).collect();
        self.last_flush = Instant::now();
        samples
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear buffer without flushing
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.last_flush = Instant::now();
    }
}

/// Batch processor for applying operations to batches
pub struct BatchProcessor<F>
where
    F: FnMut(&[f32]) -> IoResult<Vec<f32>>,
{
    accumulator: BatchAccumulator,
    processor: F,
}

impl<F> BatchProcessor<F>
where
    F: FnMut(&[f32]) -> IoResult<Vec<f32>>,
{
    /// Create new batch processor
    pub fn new(config: BatchConfig, processor: F) -> Self {
        Self {
            accumulator: BatchAccumulator::new(config),
            processor,
        }
    }

    /// Process input sample
    pub fn process(&mut self, sample: f32) -> IoResult<Option<Vec<f32>>> {
        self.accumulator.push(sample);

        if self.accumulator.should_flush() {
            let batch = self.accumulator.flush();
            let result = (self.processor)(&batch)?;
            return Ok(Some(result));
        }

        Ok(None)
    }

    /// Process multiple samples
    pub fn process_batch(&mut self, samples: &[f32]) -> IoResult<Vec<Vec<f32>>> {
        let mut results = Vec::new();

        for &sample in samples {
            if let Some(result) = self.process(sample)? {
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Force flush pending samples
    pub fn flush(&mut self) -> IoResult<Option<Vec<f32>>> {
        if self.accumulator.is_empty() {
            return Ok(None);
        }

        let batch = self.accumulator.flush();
        let result = (self.processor)(&batch)?;
        Ok(Some(result))
    }

    /// Get number of pending samples
    pub fn pending(&self) -> usize {
        self.accumulator.len()
    }
}

/// Parallel batch processor using multiple threads
pub struct ParallelBatchProcessor<F>
where
    F: Fn(&[f32]) -> IoResult<Vec<f32>> + Send + Sync + Clone + 'static,
{
    processor: F,
    num_threads: usize,
}

impl<F> ParallelBatchProcessor<F>
where
    F: Fn(&[f32]) -> IoResult<Vec<f32>> + Send + Sync + Clone + 'static,
{
    /// Create new parallel batch processor
    pub fn new(_config: BatchConfig, processor: F, num_threads: usize) -> Self {
        Self {
            processor,
            num_threads: num_threads.max(1),
        }
    }

    /// Process batch in parallel
    pub async fn process_parallel(&mut self, samples: &[f32]) -> IoResult<Vec<f32>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let chunk_size = samples.len().div_ceil(self.num_threads);
        let chunks: Vec<&[f32]> = samples.chunks(chunk_size).collect();

        let mut tasks = Vec::new();
        for chunk in chunks {
            let chunk_vec = chunk.to_vec();
            let processor = self.processor.clone();

            let task = tokio::spawn(async move { processor(&chunk_vec) });

            tasks.push(task);
        }

        // Collect results
        let mut results = Vec::new();
        for task in tasks {
            let result = task
                .await
                .map_err(|e| IoError::SignalError(format!("Task failed: {}", e)))??;
            results.extend(result);
        }

        Ok(results)
    }
}

/// Windowed batch processor for sliding window operations
pub struct WindowedBatchProcessor {
    window_size: usize,
    overlap: usize,
    buffer: VecDeque<f32>,
}

impl WindowedBatchProcessor {
    /// Create new windowed batch processor
    pub fn new(window_size: usize, overlap: usize) -> IoResult<Self> {
        if overlap >= window_size {
            return Err(IoError::InvalidConfig(
                "Overlap must be less than window size".to_string(),
            ));
        }

        Ok(Self {
            window_size,
            overlap,
            buffer: VecDeque::with_capacity(window_size),
        })
    }

    /// Create with 50% overlap
    pub fn with_half_overlap(window_size: usize) -> IoResult<Self> {
        Self::new(window_size, window_size / 2)
    }

    /// Push sample and get window if ready
    pub fn push(&mut self, sample: f32) -> Option<Vec<f32>> {
        self.buffer.push_back(sample);

        if self.buffer.len() >= self.window_size {
            let window: Vec<f32> = self.buffer.iter().copied().collect();

            // Remove samples for next window (based on overlap)
            let to_remove = self.window_size - self.overlap;
            for _ in 0..to_remove {
                self.buffer.pop_front();
            }

            return Some(window);
        }

        None
    }

    /// Push multiple samples and get all ready windows
    pub fn push_batch(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut windows = Vec::new();

        for &sample in samples {
            if let Some(window) = self.push(sample) {
                windows.push(window);
            }
        }

        windows
    }

    /// Get current buffer size
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Statistics accumulator for batch processing
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    pub total_batches: u64,
    pub total_samples: u64,
    pub avg_batch_size: f32,
    pub min_batch_size: usize,
    pub max_batch_size: usize,
}

impl BatchStats {
    /// Create new batch statistics
    pub fn new() -> Self {
        Self {
            total_batches: 0,
            total_samples: 0,
            avg_batch_size: 0.0,
            min_batch_size: usize::MAX,
            max_batch_size: 0,
        }
    }

    /// Update with new batch
    pub fn update(&mut self, batch_size: usize) {
        self.total_batches += 1;
        self.total_samples += batch_size as u64;
        self.min_batch_size = self.min_batch_size.min(batch_size);
        self.max_batch_size = self.max_batch_size.max(batch_size);
        self.avg_batch_size = self.total_samples as f32 / self.total_batches as f32;
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_accumulator() {
        let config = BatchConfig {
            max_size: 5,
            max_wait_time: Duration::from_secs(1),
            min_size: 1,
        };

        let mut acc = BatchAccumulator::new(config);

        // Add samples
        for i in 0..4 {
            acc.push(i as f32);
            assert!(!acc.should_flush());
        }

        // Should flush at max size
        acc.push(4.0);
        assert!(acc.should_flush());

        let batch = acc.flush();
        assert_eq!(batch, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(acc.is_empty());
    }

    #[test]
    fn test_batch_processor() {
        let config = BatchConfig {
            max_size: 3,
            max_wait_time: Duration::from_secs(1),
            min_size: 1,
        };

        let mut processor = BatchProcessor::new(config, |batch| {
            // Sum all values in batch
            Ok(vec![batch.iter().sum()])
        });

        // Process samples
        assert!(processor.process(1.0).unwrap().is_none());
        assert!(processor.process(2.0).unwrap().is_none());

        // Should trigger processing
        let result = processor.process(3.0).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![6.0]); // 1+2+3
    }

    #[test]
    fn test_windowed_batch_processor() {
        let mut processor = WindowedBatchProcessor::new(3, 1).unwrap();

        // Not enough samples yet
        assert!(processor.push(1.0).is_none());
        assert!(processor.push(2.0).is_none());

        // Should return first window
        let window = processor.push(3.0).unwrap();
        assert_eq!(window, vec![1.0, 2.0, 3.0]);

        // Next window with overlap=1
        assert!(processor.push(4.0).is_none());
        let window = processor.push(5.0).unwrap();
        assert_eq!(window, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_batch_stats() {
        let mut stats = BatchStats::new();

        stats.update(10);
        stats.update(20);
        stats.update(30);

        assert_eq!(stats.total_batches, 3);
        assert_eq!(stats.total_samples, 60);
        assert_eq!(stats.min_batch_size, 10);
        assert_eq!(stats.max_batch_size, 30);
        assert_eq!(stats.avg_batch_size, 20.0);
    }

    #[tokio::test]
    async fn test_parallel_batch_processor() {
        let config = BatchConfig::default();
        let processor =
            |batch: &[f32]| -> IoResult<Vec<f32>> { Ok(batch.iter().map(|x| x * 2.0).collect()) };

        let mut parallel = ParallelBatchProcessor::new(config, processor, 4);

        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = parallel.process_parallel(&samples).await.unwrap();

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }
}
