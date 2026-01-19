//! Memory pool for efficient tensor allocation and reuse.
//!
//! This module provides a buffer pool to reduce allocations during inference
//! by reusing pre-allocated tensor buffers. This is especially important for
//! streaming scenarios where state tensors are frequently created and destroyed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{InferenceError, InferenceResult};

/// A key identifying a buffer shape and type configuration.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct BufferKey {
    /// Total number of elements in the buffer
    pub size: usize,
    /// Element type identifier (e.g., "f32", "f64")
    pub dtype: String,
    /// Optional semantic tag for specialized pools
    pub tag: Option<String>,
}

impl BufferKey {
    /// Create a new buffer key for f32 tensors.
    pub fn f32(size: usize) -> Self {
        Self {
            size,
            dtype: "f32".to_string(),
            tag: None,
        }
    }

    /// Create a new buffer key for f64 tensors.
    pub fn f64(size: usize) -> Self {
        Self {
            size,
            dtype: "f64".to_string(),
            tag: None,
        }
    }

    /// Add a semantic tag to this key.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }
}

/// A pooled buffer that returns itself to the pool when dropped.
pub struct PooledBuffer<T> {
    data: Vec<T>,
    key: BufferKey,
    pool: Arc<Mutex<TensorPoolInner>>,
}

impl<T> PooledBuffer<T> {
    /// Get a reference to the underlying data.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Get a mutable reference to the underlying data.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get the size of the buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Consume the buffer and return the underlying Vec.
    /// This prevents the buffer from being returned to the pool.
    pub fn into_vec(mut self) -> Vec<T> {
        // Take data and leave empty vec to prevent return to pool
        std::mem::take(&mut self.data)
    }
}

impl<T> Drop for PooledBuffer<T> {
    fn drop(&mut self) {
        // Return buffer to pool only if it's not empty
        if !self.data.is_empty() {
            if let Ok(mut pool) = self.pool.lock() {
                pool.return_raw_buffer(self.key.clone(), std::mem::take(&mut self.data));
            }
        }
    }
}

impl<T> std::ops::Deref for PooledBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> std::ops::DerefMut for PooledBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// A thread-safe memory pool for tensor buffers.
///
/// The pool maintains separate storage for different buffer configurations
/// and automatically grows as needed. Buffers are returned to the pool when dropped.
#[derive(Clone)]
pub struct TensorPool {
    inner: Arc<Mutex<TensorPoolInner>>,
}

struct TensorPoolInner {
    /// Storage for f32 buffers
    f32_buffers: HashMap<BufferKey, Vec<Vec<f32>>>,
    /// Storage for f64 buffers
    f64_buffers: HashMap<BufferKey, Vec<Vec<f64>>>,
    /// Maximum number of buffers to keep per key
    max_buffers_per_key: usize,
    /// Statistics
    stats: PoolStats,
}

#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of buffer allocations
    pub total_allocations: usize,
    /// Number of buffer reuses from pool
    pub total_reuses: usize,
    /// Number of buffers returned to pool
    pub total_returns: usize,
    /// Number of buffers discarded (pool full)
    pub total_discards: usize,
}

impl TensorPool {
    /// Create a new tensor pool with default capacity (16 buffers per key).
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    /// Create a new tensor pool with specified maximum buffers per key.
    pub fn with_capacity(max_buffers_per_key: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TensorPoolInner {
                f32_buffers: HashMap::new(),
                f64_buffers: HashMap::new(),
                max_buffers_per_key,
                stats: PoolStats::default(),
            })),
        }
    }

    /// Acquire a pooled f32 buffer.
    pub fn acquire_f32(&self, key: BufferKey) -> InferenceResult<PooledBuffer<f32>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| InferenceError::LockError(format!("Failed to acquire lock: {}", e)))?;
        let data = inner.f32_buffers.get_mut(&key).and_then(|pool| pool.pop());

        let data = if let Some(mut buf) = data {
            inner.stats.total_reuses += 1;
            // Clear the buffer for reuse
            buf.clear();
            buf.resize(key.size, 0.0);
            buf
        } else {
            inner.stats.total_allocations += 1;
            vec![0.0; key.size]
        };

        drop(inner); // Release lock before returning

        Ok(PooledBuffer {
            data,
            key,
            pool: self.inner.clone(),
        })
    }

    /// Acquire a pooled f64 buffer.
    pub fn acquire_f64(&self, key: BufferKey) -> InferenceResult<PooledBuffer<f64>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| InferenceError::LockError(format!("Failed to acquire lock: {}", e)))?;
        let data = inner.f64_buffers.get_mut(&key).and_then(|pool| pool.pop());

        let data = if let Some(mut buf) = data {
            inner.stats.total_reuses += 1;
            // Clear the buffer for reuse
            buf.clear();
            buf.resize(key.size, 0.0);
            buf
        } else {
            inner.stats.total_allocations += 1;
            vec![0.0; key.size]
        };

        drop(inner); // Release lock before returning

        Ok(PooledBuffer {
            data,
            key,
            pool: self.inner.clone(),
        })
    }

    /// Clear all pooled buffers.
    pub fn clear(&self) -> InferenceResult<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| InferenceError::LockError(format!("Failed to acquire lock: {}", e)))?;
        inner.f32_buffers.clear();
        inner.f64_buffers.clear();
        Ok(())
    }

    /// Get pool statistics.
    pub fn stats(&self) -> InferenceResult<PoolStats> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| InferenceError::LockError(format!("Failed to acquire lock: {}", e)))?;
        Ok(inner.stats.clone())
    }

    /// Get the current number of pooled buffers.
    pub fn pooled_count(&self) -> InferenceResult<usize> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| InferenceError::LockError(format!("Failed to acquire lock: {}", e)))?;
        Ok(inner.f32_buffers.values().map(|v| v.len()).sum::<usize>()
            + inner.f64_buffers.values().map(|v| v.len()).sum::<usize>())
    }
}

impl TensorPoolInner {
    fn return_raw_buffer<T>(&mut self, key: BufferKey, buffer: Vec<T>) {
        self.stats.total_returns += 1;

        match key.dtype.as_str() {
            "f32" => {
                let pool = self.f32_buffers.entry(key).or_default();
                if pool.len() < self.max_buffers_per_key {
                    // Safe because we know T is f32 for dtype "f32"
                    let buffer: Vec<f32> = unsafe { std::mem::transmute(buffer) };
                    pool.push(buffer);
                } else {
                    self.stats.total_discards += 1;
                }
            }
            "f64" => {
                let pool = self.f64_buffers.entry(key).or_default();
                if pool.len() < self.max_buffers_per_key {
                    // Safe because we know T is f64 for dtype "f64"
                    let buffer: Vec<f64> = unsafe { std::mem::transmute(buffer) };
                    pool.push(buffer);
                } else {
                    self.stats.total_discards += 1;
                }
            }
            _ => {
                // Unknown dtype, discard
                self.stats.total_discards += 1;
            }
        }
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_basic() {
        let pool = TensorPool::new();
        let key = BufferKey::f32(1024);

        // First acquisition should allocate
        let buf1 = pool
            .acquire_f32(key.clone())
            .expect("Failed to acquire buffer");
        assert_eq!(buf1.len(), 1024);
        let stats1 = pool.stats().expect("Failed to get stats");
        assert_eq!(stats1.total_allocations, 1);
        assert_eq!(stats1.total_reuses, 0);

        // Drop and reacquire should reuse
        drop(buf1);
        let buf2 = pool
            .acquire_f32(key.clone())
            .expect("Failed to acquire buffer");
        let stats2 = pool.stats().expect("Failed to get stats");
        assert_eq!(stats2.total_allocations, 1);
        assert_eq!(stats2.total_reuses, 1);
        assert_eq!(stats2.total_returns, 1);

        drop(buf2);
    }

    #[test]
    fn test_buffer_pool_multiple_keys() {
        let pool = TensorPool::new();
        let key1 = BufferKey::f32(512);
        let key2 = BufferKey::f32(1024);
        let key3 = BufferKey::f64(512);

        let buf1 = pool
            .acquire_f32(key1.clone())
            .expect("Failed to acquire buffer");
        let buf2 = pool
            .acquire_f32(key2.clone())
            .expect("Failed to acquire buffer");
        let buf3 = pool
            .acquire_f64(key3.clone())
            .expect("Failed to acquire buffer");

        assert_eq!(buf1.len(), 512);
        assert_eq!(buf2.len(), 1024);
        assert_eq!(buf3.len(), 512);

        drop(buf1);
        drop(buf2);
        drop(buf3);

        let stats = pool.stats().expect("Failed to get stats");
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.total_returns, 3);
    }

    #[test]
    fn test_buffer_pool_capacity_limit() {
        let pool = TensorPool::with_capacity(2);
        let key = BufferKey::f32(100);

        // Create 3 buffers simultaneously (before dropping any)
        // This forces 3 allocations since pool is empty
        let buf1 = pool
            .acquire_f32(key.clone())
            .expect("Failed to acquire buffer");
        let buf2 = pool
            .acquire_f32(key.clone())
            .expect("Failed to acquire buffer");
        let buf3 = pool
            .acquire_f32(key.clone())
            .expect("Failed to acquire buffer");

        // Now drop all 3 - they will try to return to pool
        // But pool capacity is 2, so one should be discarded
        drop(buf1);
        drop(buf2);
        drop(buf3);

        let stats = pool.stats().expect("Failed to get stats");
        // All 3 were new allocations (pool was empty)
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.total_reuses, 0);
        // All 3 returns attempted, but 1 discarded due to capacity
        assert_eq!(stats.total_returns, 3);
        assert_eq!(stats.total_discards, 1);
        // Pool should only have 2 buffers
        assert_eq!(pool.pooled_count().expect("Failed to get count"), 2);
    }

    #[test]
    fn test_buffer_pool_tagged_keys() {
        let pool = TensorPool::new();
        let key1 = BufferKey::f32(1024).with_tag("state");
        let key2 = BufferKey::f32(1024).with_tag("output");
        let key3 = BufferKey::f32(1024); // No tag

        let buf1 = pool
            .acquire_f32(key1.clone())
            .expect("Failed to acquire buffer");
        let buf2 = pool
            .acquire_f32(key2.clone())
            .expect("Failed to acquire buffer");
        let buf3 = pool
            .acquire_f32(key3.clone())
            .expect("Failed to acquire buffer");

        assert_eq!(buf1.len(), 1024);
        assert_eq!(buf2.len(), 1024);
        assert_eq!(buf3.len(), 1024);

        drop(buf1);
        drop(buf2);
        drop(buf3);

        // All should be separate pools
        let stats = pool.stats().expect("Failed to get stats");
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(pool.pooled_count().expect("Failed to get count"), 3);
    }

    #[test]
    fn test_buffer_clear() {
        let pool = TensorPool::new();
        let key = BufferKey::f32(100);

        let mut buf = pool
            .acquire_f32(key.clone())
            .expect("Failed to acquire buffer");
        buf[0] = 42.0;
        drop(buf);

        // After reacquisition, buffer should be cleared
        let buf2 = pool.acquire_f32(key).expect("Failed to acquire buffer");
        assert_eq!(buf2[0], 0.0);
    }

    #[test]
    fn test_pooled_buffer_into_vec() {
        let pool = TensorPool::new();
        let key = BufferKey::f32(100);

        let mut buf = pool
            .acquire_f32(key.clone())
            .expect("Failed to acquire buffer");
        buf[0] = 42.0;

        let vec = buf.into_vec();
        assert_eq!(vec[0], 42.0);
        assert_eq!(vec.len(), 100);

        // Buffer should not have been returned to pool
        let stats = pool.stats().expect("Failed to get stats");
        assert_eq!(stats.total_returns, 0);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(TensorPool::new());
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let pool = pool.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let key = BufferKey::f32(1024).with_tag(format!("thread_{}", i));
                        let buf = pool.acquire_f32(key).expect("Failed to acquire buffer");
                        assert_eq!(buf.len(), 1024);
                        drop(buf);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let stats = pool.stats().expect("Failed to get stats");
        assert!(stats.total_allocations > 0);
        assert!(stats.total_reuses > 0);
    }
}
