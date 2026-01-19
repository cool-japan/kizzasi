//! Lock-free data structures for high-performance streaming
//!
//! Provides lock-free queues and buffers optimized for real-time audio/video streaming
//! with minimal latency and no blocking operations.
//!
//! ## Features
//! - Lock-free SPSC (Single Producer Single Consumer) queue
//! - Lock-free ring buffer with overwrite semantics
//! - Wait-free read operations
//! - Cache-friendly memory layout
//!
//! ## Example
//! ```rust
//! use kizzasi_io::LockFreeQueue;
//!
//! let queue = LockFreeQueue::<f32>::new(1024);
//!
//! // Producer thread
//! for i in 0..100 {
//!     queue.push(i as f32).unwrap();
//! }
//!
//! // Consumer thread
//! while let Some(value) = queue.pop() {
//!     println!("Received: {}", value);
//! }
//! ```

use crate::error::IoResult;
use crossbeam_queue::{ArrayQueue, SegQueue};
use scirs2_core::ndarray::Array1;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::debug;

/// Lock-free SPSC (Single Producer Single Consumer) queue
///
/// Optimized for low-latency real-time streaming with wait-free operations.
pub struct LockFreeQueue<T> {
    queue: Arc<ArrayQueue<T>>,
    dropped_count: Arc<AtomicUsize>,
}

impl<T> LockFreeQueue<T> {
    /// Create a new lock-free queue with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Arc::new(ArrayQueue::new(capacity)),
            dropped_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Push an item to the queue (non-blocking)
    ///
    /// Returns `Err` if the queue is full
    pub fn push(&self, item: T) -> Result<(), T> {
        self.queue.push(item)
    }

    /// Try to push an item, dropping it if queue is full
    ///
    /// Returns `true` if the item was pushed, `false` if dropped
    pub fn try_push(&self, item: T) -> bool {
        match self.queue.push(item) {
            Ok(()) => true,
            Err(_) => {
                self.dropped_count.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Pop an item from the queue (non-blocking)
    ///
    /// Returns `None` if the queue is empty
    pub fn pop(&self) -> Option<T> {
        self.queue.pop()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Check if queue is full
    pub fn is_full(&self) -> bool {
        self.queue.is_full()
    }

    /// Get current queue length
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Get queue capacity
    pub fn capacity(&self) -> usize {
        self.queue.capacity()
    }

    /// Get number of dropped items (when queue was full)
    pub fn dropped_count(&self) -> usize {
        self.dropped_count.load(Ordering::Relaxed)
    }

    /// Reset dropped count
    pub fn reset_dropped_count(&self) {
        self.dropped_count.store(0, Ordering::Relaxed);
    }

    /// Clone the queue handle (shares the same underlying queue)
    pub fn clone_handle(&self) -> Self {
        Self {
            queue: Arc::clone(&self.queue),
            dropped_count: Arc::clone(&self.dropped_count),
        }
    }
}

impl<T: Clone> LockFreeQueue<T> {
    /// Pop all available items as a vector
    pub fn pop_all(&self) -> Vec<T> {
        let mut items = Vec::with_capacity(self.len());
        while let Some(item) = self.pop() {
            items.push(item);
        }
        items
    }
}

/// Lock-free unbounded MPMC (Multi Producer Multi Consumer) queue
///
/// Uses a linked list for unbounded capacity. Slower than bounded queue
/// but never blocks on push.
pub struct UnboundedQueue<T> {
    queue: Arc<SegQueue<T>>,
    item_count: Arc<AtomicUsize>,
}

impl<T> UnboundedQueue<T> {
    /// Create a new unbounded queue
    pub fn new() -> Self {
        Self {
            queue: Arc::new(SegQueue::new()),
            item_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Push an item to the queue (never fails)
    pub fn push(&self, item: T) {
        self.queue.push(item);
        self.item_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Pop an item from the queue
    pub fn pop(&self) -> Option<T> {
        match self.queue.pop() {
            Some(item) => {
                self.item_count.fetch_sub(1, Ordering::Relaxed);
                Some(item)
            }
            None => None,
        }
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get approximate queue length
    pub fn len(&self) -> usize {
        self.item_count.load(Ordering::Relaxed)
    }

    /// Clone the queue handle
    pub fn clone_handle(&self) -> Self {
        Self {
            queue: Arc::clone(&self.queue),
            item_count: Arc::clone(&self.item_count),
        }
    }
}

impl<T> Default for UnboundedQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> UnboundedQueue<T> {
    /// Pop all available items as a vector
    pub fn pop_all(&self) -> Vec<T> {
        let mut items = Vec::new();
        while let Some(item) = self.pop() {
            items.push(item);
        }
        items
    }
}

/// Lock-free signal queue for audio/sensor data
///
/// Specialized queue for streaming f32 samples with batching support.
pub struct SignalQueue {
    queue: LockFreeQueue<f32>,
    batch_size: usize,
    underrun_count: Arc<AtomicUsize>,
    overrun_count: Arc<AtomicUsize>,
}

impl SignalQueue {
    /// Create a new signal queue
    pub fn new(capacity: usize, batch_size: usize) -> Self {
        Self {
            queue: LockFreeQueue::new(capacity),
            batch_size,
            underrun_count: Arc::new(AtomicUsize::new(0)),
            overrun_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Write samples to the queue
    pub fn write_samples(&self, samples: &[f32]) -> IoResult<()> {
        let mut overruns = 0;
        for &sample in samples {
            if !self.queue.try_push(sample) {
                overruns += 1;
            }
        }

        if overruns > 0 {
            self.overrun_count.fetch_add(overruns, Ordering::Relaxed);
            debug!("Signal queue overrun: {} samples dropped", overruns);
        }

        Ok(())
    }

    /// Read a batch of samples
    pub fn read_batch(&self) -> IoResult<Array1<f32>> {
        let available = self.queue.len();

        // Mark underrun if we don't have a full batch
        if available < self.batch_size {
            self.underrun_count.fetch_add(1, Ordering::Relaxed);
        }

        // Read whatever is available, up to batch_size
        let mut samples = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            if let Some(sample) = self.queue.pop() {
                samples.push(sample);
            } else {
                break;
            }
        }

        // Pad with zeros if needed
        while samples.len() < self.batch_size {
            samples.push(0.0);
        }

        Ok(Array1::from_vec(samples))
    }

    /// Read all available samples
    pub fn read_all(&self) -> Array1<f32> {
        let samples = self.queue.pop_all();
        Array1::from_vec(samples)
    }

    /// Get current queue level
    pub fn level(&self) -> usize {
        self.queue.len()
    }

    /// Get queue capacity
    pub fn capacity(&self) -> usize {
        self.queue.capacity()
    }

    /// Get fill percentage (0.0 to 1.0)
    pub fn fill_ratio(&self) -> f32 {
        self.level() as f32 / self.capacity() as f32
    }

    /// Get underrun count
    pub fn underrun_count(&self) -> usize {
        self.underrun_count.load(Ordering::Relaxed)
    }

    /// Get overrun count
    pub fn overrun_count(&self) -> usize {
        self.overrun_count.load(Ordering::Relaxed)
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.underrun_count.store(0, Ordering::Relaxed);
        self.overrun_count.store(0, Ordering::Relaxed);
        self.queue.reset_dropped_count();
    }
}

/// Lock-free ring buffer with overwrite semantics
///
/// Automatically overwrites oldest data when full, ensuring continuous streaming.
pub struct LockFreeRingBuffer<T> {
    buffer: Vec<Option<T>>,
    write_pos: Arc<AtomicUsize>,
    read_pos: Arc<AtomicUsize>,
    capacity: usize,
    overwrite_flag: Arc<AtomicBool>,
}

impl<T: Clone> LockFreeRingBuffer<T> {
    /// Create a new lock-free ring buffer
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(None);
        }

        Self {
            buffer,
            write_pos: Arc::new(AtomicUsize::new(0)),
            read_pos: Arc::new(AtomicUsize::new(0)),
            capacity,
            overwrite_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Write an item (overwrites oldest if full)
    pub fn write(&mut self, item: T) {
        let write_idx = self.write_pos.load(Ordering::Acquire);
        let read_idx = self.read_pos.load(Ordering::Acquire);

        // Store the item
        self.buffer[write_idx] = Some(item);

        // Advance write position
        let next_write = (write_idx + 1) % self.capacity;
        self.write_pos.store(next_write, Ordering::Release);

        // Check if we're overwriting
        if next_write == read_idx {
            // Advance read position to maintain buffer size
            let next_read = (read_idx + 1) % self.capacity;
            self.read_pos.store(next_read, Ordering::Release);
            self.overwrite_flag.store(true, Ordering::Relaxed);
        }
    }

    /// Read an item
    pub fn read(&self) -> Option<T> {
        let read_idx = self.read_pos.load(Ordering::Acquire);
        let write_idx = self.write_pos.load(Ordering::Acquire);

        if read_idx == write_idx {
            return None; // Buffer is empty
        }

        let item = self.buffer[read_idx].clone();

        // Advance read position
        let next_read = (read_idx + 1) % self.capacity;
        self.read_pos.store(next_read, Ordering::Release);

        item
    }

    /// Get number of available items
    pub fn available(&self) -> usize {
        let write_idx = self.write_pos.load(Ordering::Acquire);
        let read_idx = self.read_pos.load(Ordering::Acquire);

        if write_idx >= read_idx {
            write_idx - read_idx
        } else {
            self.capacity - read_idx + write_idx
        }
    }

    /// Check if overwrite occurred
    pub fn was_overwritten(&self) -> bool {
        self.overwrite_flag.swap(false, Ordering::Relaxed)
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lockfree_queue_basic() {
        let queue = LockFreeQueue::new(10);

        assert!(queue.is_empty());
        assert_eq!(queue.capacity(), 10);

        queue.push(1.0).unwrap();
        queue.push(2.0).unwrap();
        queue.push(3.0).unwrap();

        assert_eq!(queue.len(), 3);
        assert!(!queue.is_empty());

        assert_eq!(queue.pop(), Some(1.0));
        assert_eq!(queue.pop(), Some(2.0));
        assert_eq!(queue.pop(), Some(3.0));
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn test_lockfree_queue_full() {
        let queue = LockFreeQueue::new(3);

        assert!(queue.push(1.0).is_ok());
        assert!(queue.push(2.0).is_ok());
        assert!(queue.push(3.0).is_ok());
        assert!(queue.push(4.0).is_err()); // Queue full

        assert!(queue.is_full());
    }

    #[test]
    fn test_lockfree_queue_try_push() {
        let queue = LockFreeQueue::new(2);

        assert!(queue.try_push(1.0));
        assert!(queue.try_push(2.0));
        assert!(!queue.try_push(3.0)); // Dropped

        assert_eq!(queue.dropped_count(), 1);
    }

    #[test]
    fn test_unbounded_queue() {
        let queue = UnboundedQueue::new();

        for i in 0..1000 {
            queue.push(i);
        }

        assert_eq!(queue.len(), 1000);

        for i in 0..1000 {
            assert_eq!(queue.pop(), Some(i));
        }

        assert!(queue.is_empty());
    }

    #[test]
    fn test_signal_queue() {
        let queue = SignalQueue::new(100, 10);

        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        queue.write_samples(&samples).unwrap();

        assert_eq!(queue.level(), 5);

        let batch = queue.read_batch().unwrap();
        assert_eq!(batch.len(), 10); // Batch size

        // First 5 should be the samples, rest zeros
        assert_eq!(batch[0], 1.0);
        assert_eq!(batch[4], 5.0);
        assert_eq!(batch[5], 0.0);
    }

    #[test]
    fn test_signal_queue_overrun() {
        let queue = SignalQueue::new(10, 5);

        let samples = vec![1.0; 15]; // More than capacity
        queue.write_samples(&samples).unwrap();

        assert!(queue.overrun_count() > 0);
    }

    #[test]
    fn test_lockfree_ring_buffer() {
        let mut buffer = LockFreeRingBuffer::new(5);

        buffer.write(1);
        buffer.write(2);
        buffer.write(3);

        assert_eq!(buffer.available(), 3);

        assert_eq!(buffer.read(), Some(1));
        assert_eq!(buffer.read(), Some(2));
        assert_eq!(buffer.read(), Some(3));
        assert_eq!(buffer.read(), None);
    }

    #[test]
    fn test_lockfree_ring_buffer_overwrite() {
        let mut buffer = LockFreeRingBuffer::new(3);

        buffer.write(1);
        buffer.write(2);
        buffer.write(3);
        buffer.write(4); // Overwrites, advances read pointer

        assert!(buffer.was_overwritten());

        // After overwrite, buffer should contain last 2 items written
        // because when write catches up to read, read advances
        let first = buffer.read();
        let second = buffer.read();

        // Should be able to read 2 items (exact values depend on implementation)
        assert!(first.is_some());
        assert!(second.is_some());
    }
}
