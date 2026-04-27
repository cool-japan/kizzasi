//! Zero-copy buffer implementations
//!
//! Provides efficient buffer types that minimize memory allocations
//! and copies for high-performance streaming applications.

use bytes::{Bytes, BytesMut};
use scirs2_core::ndarray::Array1;
use std::sync::Arc;

/// Zero-copy signal buffer using Arc for shared ownership
///
/// Allows multiple readers to access the same buffer without copying.
#[derive(Debug, Clone)]
pub struct SharedSignalBuffer {
    /// Shared buffer data
    data: Arc<Vec<f32>>,
    /// Start offset
    offset: usize,
    /// Length
    len: usize,
}

impl SharedSignalBuffer {
    /// Create a new shared buffer from a vector
    pub fn new(data: Vec<f32>) -> Self {
        let len = data.len();
        Self {
            data: Arc::new(data),
            offset: 0,
            len,
        }
    }

    /// Create from an Arc
    pub fn from_arc(data: Arc<Vec<f32>>) -> Self {
        let len = data.len();
        Self {
            data,
            offset: 0,
            len,
        }
    }

    /// Create a slice view of this buffer (zero-copy)
    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= self.len);
        Self {
            data: Arc::clone(&self.data),
            offset: self.offset + start,
            len: end - start,
        }
    }

    /// Get a reference to the data slice
    pub fn as_slice(&self) -> &[f32] {
        &self.data[self.offset..self.offset + self.len]
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Convert to Array1 (copies data)
    pub fn to_array(&self) -> Array1<f32> {
        Array1::from_vec(self.as_slice().to_vec())
    }

    /// Get the number of strong references
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }
}

/// Zero-copy byte buffer for raw data streams
#[derive(Debug, Clone)]
pub struct ZeroCopyBuffer {
    /// Underlying bytes
    data: Bytes,
}

impl ZeroCopyBuffer {
    /// Create a new buffer from bytes
    pub fn new(data: Bytes) -> Self {
        Self { data }
    }

    /// Create from a vector (takes ownership)
    pub fn from_vec(vec: Vec<u8>) -> Self {
        Self {
            data: Bytes::from(vec),
        }
    }

    /// Create from a slice (copies)
    pub fn from_slice(slice: &[u8]) -> Self {
        Self {
            data: Bytes::copy_from_slice(slice),
        }
    }

    /// Get a slice of this buffer (zero-copy)
    pub fn slice(&self, range: impl std::ops::RangeBounds<usize>) -> Self {
        Self {
            data: self.data.slice(range),
        }
    }

    /// Get the underlying bytes
    pub fn as_bytes(&self) -> &Bytes {
        &self.data
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Convert to f32 samples (assumes IEEE 754 encoding)
    pub fn to_f32_samples(&self) -> Vec<f32> {
        let byte_slice = self.data.as_ref();
        let sample_count = byte_slice.len() / 4;
        let mut samples = Vec::with_capacity(sample_count);

        for i in 0..sample_count {
            let offset = i * 4;
            let bytes = [
                byte_slice[offset],
                byte_slice[offset + 1],
                byte_slice[offset + 2],
                byte_slice[offset + 3],
            ];
            samples.push(f32::from_le_bytes(bytes));
        }

        samples
    }

    /// Convert to `Array1<f32>`
    pub fn to_array(&self) -> Array1<f32> {
        Array1::from_vec(self.to_f32_samples())
    }
}

/// Mutable zero-copy buffer builder
pub struct ZeroCopyBufferMut {
    /// Mutable bytes
    data: BytesMut,
}

impl ZeroCopyBufferMut {
    /// Create a new mutable buffer with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: BytesMut::with_capacity(capacity),
        }
    }

    /// Append bytes
    pub fn extend_from_slice(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }

    /// Append f32 sample
    pub fn push_f32(&mut self, sample: f32) {
        self.data.extend_from_slice(&sample.to_le_bytes());
    }

    /// Append multiple f32 samples
    pub fn extend_f32(&mut self, samples: &[f32]) {
        for &sample in samples {
            self.push_f32(sample);
        }
    }

    /// Freeze into immutable buffer
    pub fn freeze(self) -> ZeroCopyBuffer {
        ZeroCopyBuffer {
            data: self.data.freeze(),
        }
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }
}

/// Pool of reusable buffers to reduce allocation overhead
pub struct BufferPool {
    /// Pool of available buffers
    pool: crossbeam_queue::SegQueue<BytesMut>,
    /// Default buffer capacity
    capacity: usize,
}

impl BufferPool {
    /// Create a new buffer pool with default capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            pool: crossbeam_queue::SegQueue::new(),
            capacity,
        }
    }

    /// Acquire a buffer from the pool or create a new one
    pub fn acquire(&self) -> BytesMut {
        self.pool
            .pop()
            .unwrap_or_else(|| BytesMut::with_capacity(self.capacity))
    }

    /// Return a buffer to the pool
    pub fn release(&self, mut buffer: BytesMut) {
        buffer.clear();
        if buffer.capacity() >= self.capacity {
            self.pool.push(buffer);
        }
        // Drop buffers that are too small
    }

    /// Get pool size (approximate)
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Check if pool is empty
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_signal_buffer() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let buffer = SharedSignalBuffer::new(data);

        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.ref_count(), 1);

        let slice = buffer.slice(1, 4);
        assert_eq!(slice.len(), 3);
        assert_eq!(slice.as_slice(), &[2.0, 3.0, 4.0]);
        assert_eq!(buffer.ref_count(), 2);
    }

    #[test]
    fn test_zerocopy_buffer() {
        let mut builder = ZeroCopyBufferMut::with_capacity(16);
        builder.push_f32(1.0);
        builder.push_f32(2.0);
        builder.push_f32(3.0);

        let buffer = builder.freeze();
        assert_eq!(buffer.len(), 12); // 3 floats * 4 bytes

        let samples = buffer.to_f32_samples();
        assert_eq!(samples, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_zerocopy_buffer_slice() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let buffer = ZeroCopyBuffer::from_vec(data);

        let slice = buffer.slice(2..6);
        assert_eq!(slice.len(), 4);
        assert_eq!(slice.as_bytes().as_ref(), &[3, 4, 5, 6]);
    }

    #[test]
    fn test_buffer_pool() {
        let pool = BufferPool::new(1024);
        assert!(pool.is_empty());

        let buffer1 = pool.acquire();
        assert_eq!(buffer1.capacity(), 1024);

        pool.release(buffer1);
        assert_eq!(pool.len(), 1);

        let buffer2 = pool.acquire();
        assert_eq!(buffer2.capacity(), 1024);
    }

    #[test]
    fn test_shared_buffer_to_array() {
        let data = vec![1.0, 2.0, 3.0];
        let buffer = SharedSignalBuffer::new(data);
        let array = buffer.to_array();
        assert_eq!(array.len(), 3);
        assert_eq!(array[0], 1.0);
    }
}
