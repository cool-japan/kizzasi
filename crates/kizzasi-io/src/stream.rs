//! Signal stream abstractions
//!
//! # Read contract
//!
//! Every [`SignalStream`] / [`AsyncSignalStream`] implementation in this crate
//! obeys exactly these three outcomes, so a consumer can always distinguish
//! "the sensor is silent", "the buffer is starved" and "the stream ended":
//!
//! 1. `Ok(buffer)` -- `buffer` contains **only samples that were actually
//!    received**. It is never zero-padded, and
//!    `1 <= buffer.len() <= config().buffer_size`. A live source only ever
//!    returns a full `buffer_size` block; a finite source (a file, an
//!    in-memory buffer, or a live source that has been closed) returns its
//!    remaining samples as one final short block.
//! 2. `Err(IoError::BufferEmpty)` -- a full block is not available yet and
//!    more samples may still arrive. **Nothing was consumed**: the partial
//!    data stays buffered, so retrying later is lossless.
//! 3. `Err(IoError::EndOfStream)` -- the stream is permanently exhausted (or
//!    was closed and fully drained). `is_active()` is `false` from then on.
//!
//! Any other `Err` is a genuine transport/protocol failure.
//!
//! Several implementations previously returned
//! `Ok(Array1::zeros(buffer_size))` for cases 2 and 3, which presented
//! fabricated silence as sensor data and silently corrupted every downstream
//! RMS / SNR / spectrum computation. No implementation does that any more.

use crate::error::{IoError, IoResult};
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
    /// Read the next signal buffer.
    ///
    /// Returns only real samples (never zero-padded), `Err(IoError::BufferEmpty)`
    /// when nothing is available yet, and `Err(IoError::EndOfStream)` once the
    /// stream is permanently exhausted. See the module-level read contract.
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
    /// Read the next signal buffer asynchronously.
    ///
    /// Returns only real samples (never zero-padded), `Err(IoError::BufferEmpty)`
    /// when nothing is available yet, and `Err(IoError::EndOfStream)` once the
    /// stream is permanently exhausted. See the module-level read contract.
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
            return Err(IoError::EndOfStream);
        }

        let end = (self.position + self.config.buffer_size).min(self.data.len());
        // The tail of the source is returned as a SHORT buffer rather than
        // being zero-padded up to `buffer_size`: padding would present
        // fabricated silence as data, and dropping it would lose samples.
        let buffer = self.data[self.position..end].to_vec();

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
            return Err(IoError::EndOfStream);
        }

        let end = (self.position + self.config.buffer_size).min(self.data.len());
        // Short (never zero-padded) tail read -- see the module read contract.
        let buffer = self.data[self.position..end].to_vec();

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
/// Wraps a tokio channel receiver to provide async stream interface.
///
/// Producer chunks are re-blocked to `config.buffer_size`: samples that do not
/// fit in the current block are carried over to the next `read()` instead of
/// being discarded (a previous version truncated every chunk to `buffer_size`
/// and zero-padded short ones).
pub struct ChannelStream {
    config: StreamConfig,
    receiver: tokio::sync::mpsc::Receiver<Vec<f32>>,
    pending: std::collections::VecDeque<f32>,
    active: bool,
}

impl ChannelStream {
    /// Create a new channel stream
    pub fn new(config: StreamConfig, receiver: tokio::sync::mpsc::Receiver<Vec<f32>>) -> Self {
        Self {
            config,
            receiver,
            pending: std::collections::VecDeque::new(),
            active: true,
        }
    }

    /// Number of samples carried over from previous chunks
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

#[async_trait::async_trait]
impl AsyncSignalStream for ChannelStream {
    async fn read(&mut self) -> IoResult<Array1<f32>> {
        let block = self.config.buffer_size.max(1);

        if self.active {
            // Accumulate producer chunks until a full block is available.
            while self.pending.len() < block {
                match self.receiver.recv().await {
                    Some(data) => self.pending.extend(data),
                    None => {
                        self.active = false;
                        break;
                    }
                }
            }
        }

        if self.pending.is_empty() {
            return Err(if self.active {
                IoError::BufferEmpty
            } else {
                IoError::EndOfStream
            });
        }

        // Either a full block, or (once the channel closed) the final short
        // block of genuinely received samples -- never zero-padded.
        let take = block.min(self.pending.len());
        let data: Vec<f32> = self.pending.drain(..take).collect();
        Ok(Array1::from_vec(data))
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

    // === Read contract: no fabricated zero-fill (id=40) ===

    #[test]
    fn test_memory_stream_reports_end_of_stream_instead_of_zeros() {
        // Regression: a previous version returned
        // Ok(Array1::zeros(buffer_size)) forever once exhausted, which is
        // indistinguishable from a genuinely silent source.
        let config = StreamConfig::new().buffer_size(4);
        let mut stream = MemoryStream::new(vec![1.0, 2.0, 3.0, 4.0], config);

        let first = stream.read().expect("first read should succeed");
        assert_eq!(first.len(), 4);

        for _ in 0..3 {
            assert!(
                matches!(stream.read(), Err(IoError::EndOfStream)),
                "an exhausted MemoryStream must report EndOfStream, never zeros"
            );
        }
        assert!(!stream.is_active());
    }

    #[test]
    fn test_memory_stream_returns_short_tail_without_padding() {
        // 6 samples with buffer_size 4 -> one full read and one 2-sample
        // read. The tail must NOT be padded out to 4 with fake zeros, and no
        // sample may be dropped.
        let config = StreamConfig::new().buffer_size(4);
        let mut stream = MemoryStream::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], config);

        let first = stream.read().expect("first read should succeed");
        assert_eq!(first.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        let tail = stream.read().expect("tail read should succeed");
        assert_eq!(
            tail.to_vec(),
            vec![5.0, 6.0],
            "the tail must be returned short, not zero-padded"
        );

        assert!(matches!(stream.read(), Err(IoError::EndOfStream)));
    }

    #[test]
    fn test_memory_stream_empty_source_is_end_of_stream() {
        let config = StreamConfig::new().buffer_size(8);
        let mut stream = MemoryStream::new(Vec::new(), config);
        assert!(matches!(stream.read(), Err(IoError::EndOfStream)));
    }

    #[tokio::test]
    async fn test_async_memory_stream_reports_end_of_stream() {
        let config = StreamConfig::new().buffer_size(4);
        let mut stream = AsyncMemoryStream::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], config);

        let first = stream.read().await.expect("first read should succeed");
        assert_eq!(first.len(), 4);
        let tail = stream.read().await.expect("tail read should succeed");
        assert_eq!(tail.to_vec(), vec![5.0]);
        assert!(matches!(stream.read().await, Err(IoError::EndOfStream)));
    }

    #[tokio::test]
    async fn test_channel_stream_reblocks_and_reports_end_of_stream() {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let config = StreamConfig::new().buffer_size(4);
        let mut stream = ChannelStream::new(config, rx);

        // Producer chunks that do not line up with buffer_size: a previous
        // version truncated the 6-sample chunk to 4 and dropped the rest.
        tx.send(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .await
            .expect("send should succeed");
        tx.send(vec![7.0]).await.expect("send should succeed");
        drop(tx);

        let first = stream.read().await.expect("first block");
        assert_eq!(first.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        let second = stream.read().await.expect("carried-over block");
        assert_eq!(
            second.to_vec(),
            vec![5.0, 6.0, 7.0],
            "carried-over samples must survive and must not be zero-padded"
        );

        assert!(matches!(stream.read().await, Err(IoError::EndOfStream)));
        assert!(!stream.is_active());
        assert!(matches!(stream.read().await, Err(IoError::EndOfStream)));
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
