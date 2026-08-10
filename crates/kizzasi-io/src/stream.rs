//! Signal stream abstractions

use crate::error::IoResult;
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for signal streams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// Number of channels
    pub channels: usize,
    /// Buffer size
    pub buffer_size: usize,
    /// Read timeout
    pub timeout: Option<Duration>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            timeout: Some(Duration::from_secs(5)),
        }
    }
}

impl StreamConfig {
    /// Create a new stream configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the sample rate
    pub fn sample_rate(mut self, rate: f32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Set the number of channels
    pub fn channels(mut self, n: usize) -> Self {
        self.channels = n;
        self
    }

    /// Set the buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set the timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

/// Trait for signal streams
///
/// Note: Not all stream implementations are Send (e.g., AudioInput with cpal::Stream).
/// Use `SendableSignalStream` when Send is required.
pub trait SignalStream {
    /// Read the next signal buffer
    fn read(&mut self) -> IoResult<Array1<f32>>;

    /// Check if stream is still active
    fn is_active(&self) -> bool;

    /// Get stream configuration
    fn config(&self) -> &StreamConfig;

    /// Close the stream
    fn close(&mut self) -> IoResult<()>;
}

/// Async trait for signal streams
///
/// Provides async I/O for signal streams, enabling non-blocking reads
/// and better integration with async runtimes like Tokio.
#[async_trait::async_trait]
pub trait AsyncSignalStream: Send {
    /// Read the next signal buffer asynchronously
    async fn read(&mut self) -> IoResult<Array1<f32>>;

    /// Check if stream is still active
    fn is_active(&self) -> bool;

    /// Get stream configuration
    fn config(&self) -> &StreamConfig;

    /// Close the stream asynchronously
    async fn close(&mut self) -> IoResult<()>;
}

/// In-memory signal stream for testing
#[derive(Debug)]
pub struct MemoryStream {
    config: StreamConfig,
    data: Vec<f32>,
    position: usize,
    active: bool,
}

impl MemoryStream {
    /// Create a new memory stream from data
    pub fn new(data: Vec<f32>, config: StreamConfig) -> Self {
        Self {
            config,
            data,
            position: 0,
            active: true,
        }
    }

    /// Create from an array
    pub fn from_array(data: Array1<f32>, config: StreamConfig) -> Self {
        Self::new(data.to_vec(), config)
    }
}

impl SignalStream for MemoryStream {
    fn read(&mut self) -> IoResult<Array1<f32>> {
        if !self.active || self.position >= self.data.len() {
            self.active = false;
            return Ok(Array1::zeros(self.config.buffer_size));
        }

        let end = (self.position + self.config.buffer_size).min(self.data.len());
        let mut buffer = vec![0.0; self.config.buffer_size];

        for (i, val) in self.data[self.position..end].iter().enumerate() {
            buffer[i] = *val;
        }

        self.position = end;
        Ok(Array1::from_vec(buffer))
    }

    fn is_active(&self) -> bool {
        self.active && self.position < self.data.len()
    }

    fn config(&self) -> &StreamConfig {
        &self.config
    }

    fn close(&mut self) -> IoResult<()> {
        self.active = false;
        Ok(())
    }
}

/// Ring buffer for real-time signal processing
///
/// A lock-free single-producer single-consumer ring buffer optimized for
/// real-time audio and sensor data. Provides O(1) push/pop operations
/// with minimal allocation.
#[derive(Debug)]
pub struct RingBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    read_pos: usize,
    write_pos: usize,
    len: usize,
}

impl<T: Clone + Default> RingBuffer<T> {
    /// Create a new ring buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            data: vec![T::default(); capacity],
            capacity,
            read_pos: 0,
            write_pos: 0,
            len: 0,
        }
    }

    /// Push an element, overwriting oldest if full
    pub fn push(&mut self, value: T) {
        self.data[self.write_pos] = value;
        self.write_pos = (self.write_pos + 1) % self.capacity;

        if self.len < self.capacity {
            self.len += 1;
        } else {
            // Buffer was full, advance read position
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }
    }

    /// Push multiple elements
    pub fn push_slice(&mut self, values: &[T]) {
        for val in values {
            self.push(val.clone());
        }
    }

    /// Pop the oldest element
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        let value = self.data[self.read_pos].clone();
        self.read_pos = (self.read_pos + 1) % self.capacity;
        self.len -= 1;
        Some(value)
    }

    /// Peek at the oldest element without removing
    pub fn peek(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            Some(&self.data[self.read_pos])
        }
    }

    /// Peek at the newest element
    pub fn peek_back(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            let idx = if self.write_pos == 0 {
                self.capacity - 1
            } else {
                self.write_pos - 1
            };
            Some(&self.data[idx])
        }
    }

    /// Get element at index (0 = oldest)
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        let idx = (self.read_pos + index) % self.capacity;
        Some(&self.data[idx])
    }

    /// Current number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is the buffer empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Is the buffer full?
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
        self.len = 0;
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Available space for writing
    pub fn available(&self) -> usize {
        self.capacity - self.len
    }

    /// Read into a slice, returns number read
    pub fn read_into(&mut self, buffer: &mut [T]) -> usize {
        let count = buffer.len().min(self.len);
        for (i, slot) in buffer.iter_mut().enumerate().take(count) {
            if let Some(val) = self.pop() {
                *slot = val;
            } else {
                return i;
            }
        }
        count
    }

    /// Copy contents to a Vec without modifying buffer
    pub fn to_vec(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let idx = (self.read_pos + i) % self.capacity;
            result.push(self.data[idx].clone());
        }
        result
    }
}

/// Ring buffer iterator
pub struct RingBufferIter<'a, T> {
    buffer: &'a RingBuffer<T>,
    index: usize,
}

impl<T: Clone + Default> RingBuffer<T> {
    /// Iterate over elements (oldest to newest)
    pub fn iter(&self) -> RingBufferIter<'_, T> {
        RingBufferIter {
            buffer: self,
            index: 0,
        }
    }
}

impl<'a, T: Clone + Default> Iterator for RingBufferIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buffer.len() {
            None
        } else {
            let result = self.buffer.get(self.index);
            self.index += 1;
            result
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.buffer.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Clone + Default> ExactSizeIterator for RingBufferIter<'a, T> {}

/// Float ring buffer with signal processing operations
#[derive(Debug)]
pub struct SignalRingBuffer {
    buffer: RingBuffer<f32>,
}

impl SignalRingBuffer {
    /// Create a new signal ring buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: RingBuffer::new(capacity),
        }
    }

    /// Push a sample
    pub fn push(&mut self, sample: f32) {
        self.buffer.push(sample);
    }

    /// Push multiple samples
    pub fn push_slice(&mut self, samples: &[f32]) {
        self.buffer.push_slice(samples);
    }

    /// Pop oldest sample
    pub fn pop(&mut self) -> Option<f32> {
        self.buffer.pop()
    }

    /// Compute mean of buffered samples
    pub fn mean(&self) -> f32 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.buffer.iter().sum();
        sum / self.buffer.len() as f32
    }

    /// Compute variance
    pub fn variance(&self) -> f32 {
        if self.buffer.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let sum_sq: f32 = self.buffer.iter().map(|x| (x - mean).powi(2)).sum();
        sum_sq / (self.buffer.len() - 1) as f32
    }

    /// Compute standard deviation
    pub fn std(&self) -> f32 {
        self.variance().sqrt()
    }

    /// Get min value
    pub fn min(&self) -> Option<f32> {
        self.buffer.iter().cloned().reduce(f32::min)
    }

    /// Get max value
    pub fn max(&self) -> Option<f32> {
        self.buffer.iter().cloned().reduce(f32::max)
    }

    /// Compute RMS (root mean square)
    pub fn rms(&self) -> f32 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = self.buffer.iter().map(|x| x * x).sum();
        (sum_sq / self.buffer.len() as f32).sqrt()
    }

    /// Get peak-to-peak amplitude
    pub fn peak_to_peak(&self) -> f32 {
        match (self.min(), self.max()) {
            (Some(min), Some(max)) => max - min,
            _ => 0.0,
        }
    }

    /// Compute zero-crossing rate
    pub fn zero_crossing_rate(&self) -> f32 {
        if self.buffer.len() < 2 {
            return 0.0;
        }
        let mut crossings = 0usize;
        let mut prev = *self.buffer.peek().unwrap_or(&0.0);
        for sample in self.buffer.iter().skip(1) {
            if (prev >= 0.0 && *sample < 0.0) || (prev < 0.0 && *sample >= 0.0) {
                crossings += 1;
            }
            prev = *sample;
        }
        crossings as f32 / (self.buffer.len() - 1) as f32
    }

    /// Get current length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Is full?
    pub fn is_full(&self) -> bool {
        self.buffer.is_full()
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Convert to array
    pub fn to_array(&self) -> Array1<f32> {
        Array1::from_vec(self.buffer.to_vec())
    }

    /// Iterate over samples
    pub fn iter(&self) -> RingBufferIter<'_, f32> {
        self.buffer.iter()
    }
}

// ============================================================================
// Async Stream Implementations
// ============================================================================

/// Async in-memory signal stream for testing
#[derive(Debug)]
pub struct AsyncMemoryStream {
    config: StreamConfig,
    data: Vec<f32>,
    position: usize,
    active: bool,
}

impl AsyncMemoryStream {
    /// Create a new async memory stream from data
    pub fn new(data: Vec<f32>, config: StreamConfig) -> Self {
        Self {
            config,
            data,
            position: 0,
            active: true,
        }
    }

    /// Create from an array
    pub fn from_array(data: Array1<f32>, config: StreamConfig) -> Self {
        Self::new(data.to_vec(), config)
    }
}

#[async_trait::async_trait]
impl AsyncSignalStream for AsyncMemoryStream {
    async fn read(&mut self) -> IoResult<Array1<f32>> {
        // Simulate async I/O delay
        tokio::time::sleep(tokio::time::Duration::from_micros(10)).await;

        if !self.active || self.position >= self.data.len() {
            self.active = false;
            return Ok(Array1::zeros(self.config.buffer_size));
        }

        let end = (self.position + self.config.buffer_size).min(self.data.len());
        let mut buffer = vec![0.0; self.config.buffer_size];

        for (i, val) in self.data[self.position..end].iter().enumerate() {
            buffer[i] = *val;
        }

        self.position = end;
        Ok(Array1::from_vec(buffer))
    }

    fn is_active(&self) -> bool {
        self.active && self.position < self.data.len()
    }

    fn config(&self) -> &StreamConfig {
        &self.config
    }

    async fn close(&mut self) -> IoResult<()> {
        self.active = false;
        Ok(())
    }
}

/// Async channel-based stream adapter
///
/// Wraps a tokio channel receiver to provide async stream interface
pub struct ChannelStream {
    config: StreamConfig,
    receiver: tokio::sync::mpsc::Receiver<Vec<f32>>,
    active: bool,
}

impl ChannelStream {
    /// Create a new channel stream
    pub fn new(config: StreamConfig, receiver: tokio::sync::mpsc::Receiver<Vec<f32>>) -> Self {
        Self {
            config,
            receiver,
            active: true,
        }
    }
}

#[async_trait::async_trait]
impl AsyncSignalStream for ChannelStream {
    async fn read(&mut self) -> IoResult<Array1<f32>> {
        if !self.active {
            return Ok(Array1::zeros(self.config.buffer_size));
        }

        match self.receiver.recv().await {
            Some(data) => {
                let mut buffer = vec![0.0; self.config.buffer_size];
                let copy_len = data.len().min(self.config.buffer_size);
                buffer[..copy_len].copy_from_slice(&data[..copy_len]);
                Ok(Array1::from_vec(buffer))
            }
            None => {
                self.active = false;
                Ok(Array1::zeros(self.config.buffer_size))
            }
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn config(&self) -> &StreamConfig {
        &self.config
    }

    async fn close(&mut self) -> IoResult<()> {
        self.active = false;
        self.receiver.close();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stream() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let config = StreamConfig::new().buffer_size(4);
        let mut stream = MemoryStream::new(data, config);

        assert!(stream.is_active());

        let buf1 = stream.read().expect("MemoryStream::read should succeed");
        assert_eq!(buf1[0], 1.0);
        assert_eq!(buf1[3], 4.0);

        let buf2 = stream.read().expect("MemoryStream::read should succeed");
        assert_eq!(buf2[0], 5.0);

        assert!(!stream.is_active());
    }

    #[test]
    fn test_ring_buffer_basic() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(4);
        assert!(buf.is_empty());
        assert_eq!(buf.capacity(), 4);

        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.len(), 3);
        assert!(!buf.is_full());

        assert_eq!(buf.pop(), Some(1));
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_ring_buffer_overwrite() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert!(buf.is_full());

        // Push overwrites oldest
        buf.push(4);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.pop(), Some(2)); // 1 was overwritten
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), Some(4));
    }

    #[test]
    fn test_ring_buffer_peek() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(4);
        buf.push(10);
        buf.push(20);
        buf.push(30);

        assert_eq!(buf.peek(), Some(&10));
        assert_eq!(buf.peek_back(), Some(&30));
        assert_eq!(buf.get(1), Some(&20));
    }

    #[test]
    fn test_ring_buffer_iter() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(4);
        buf.push(1);
        buf.push(2);
        buf.push(3);

        let collected: Vec<_> = buf.iter().cloned().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn test_signal_ring_buffer_stats() {
        let mut buf = SignalRingBuffer::new(5);
        buf.push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        assert!((buf.mean() - 3.0).abs() < 0.01);
        assert!(buf.min() == Some(1.0));
        assert!(buf.max() == Some(5.0));
        assert!((buf.peak_to_peak() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_signal_ring_buffer_rms() {
        let mut buf = SignalRingBuffer::new(4);
        buf.push_slice(&[1.0, 1.0, 1.0, 1.0]);
        assert!((buf.rms() - 1.0).abs() < 0.01);

        buf.clear();
        buf.push_slice(&[3.0, 4.0]); // sqrt((9+16)/2) = sqrt(12.5) ≈ 3.54
        assert!((buf.rms() - 3.536).abs() < 0.01);
    }

    #[test]
    fn test_signal_ring_buffer_zero_crossing() {
        let mut buf = SignalRingBuffer::new(10);
        // Sine-like: positive, negative, positive
        buf.push_slice(&[1.0, 0.5, -0.5, -1.0, -0.5, 0.5, 1.0]);
        let zcr = buf.zero_crossing_rate();
        // 2 crossings in 6 transitions = 0.333
        assert!((zcr - 0.333).abs() < 0.01);
    }
}
