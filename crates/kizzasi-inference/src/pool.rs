//! Memory pool for efficient tensor allocation and reuse.
//!
//! This module provides a buffer pool to reduce allocations during inference
//! by reusing pre-allocated tensor buffers. This is especially important for
//! streaming scenarios where state tensors are frequently created and destroyed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{InferenceError, InferenceResult};

/// A key identifying a buffer shape and type configuration.
///
/// `dtype` must agree with the element type of the acquiring method
/// ([`TensorPool::acquire_f32`] requires `"f32"`, [`TensorPool::acquire_f64`]
/// requires `"f64"`); a mismatch is rejected with
/// [`InferenceError::InvalidConfiguration`]. Prefer the [`BufferKey::f32`] /
/// [`BufferKey::f64`] constructors, which always agree.
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
    /// Monomorphic return hook installed by the typed `acquire_*` constructor.
    ///
    /// The element type is fixed when the buffer is created, so the buffer is
    /// stored in the correctly typed pool without ever reinterpreting its memory.
    return_fn: fn(&mut TensorPoolInner, BufferKey, Vec<T>),
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
                (self.return_fn)(&mut pool, self.key.clone(), std::mem::take(&mut self.data));
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

    /// Reject a key whose `dtype` disagrees with the element type being acquired.
    fn check_dtype(key: &BufferKey, expected: &str) -> InferenceResult<()> {
        if key.dtype != expected {
            return Err(InferenceError::InvalidConfiguration(format!(
                "BufferKey dtype '{}' does not match the requested element type '{}'; \
                 use BufferKey::{}(size)",
                key.dtype, expected, expected
            )));
        }
        Ok(())
    }

    /// Acquire a pooled f32 buffer.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidConfiguration`] if `key.dtype` is not
    /// `"f32"`, and [`InferenceError::LockError`] if the pool mutex is poisoned.
    pub fn acquire_f32(&self, key: BufferKey) -> InferenceResult<PooledBuffer<f32>> {
        Self::check_dtype(&key, "f32")?;

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
            return_fn: TensorPoolInner::return_f32,
        })
    }

    /// Acquire a pooled f64 buffer.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidConfiguration`] if `key.dtype` is not
    /// `"f64"`, and [`InferenceError::LockError`] if the pool mutex is poisoned.
    pub fn acquire_f64(&self, key: BufferKey) -> InferenceResult<PooledBuffer<f64>> {
        Self::check_dtype(&key, "f64")?;

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
            return_fn: TensorPoolInner::return_f64,
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

    /// Acquire an `Array1<f32>` of `size` zeroed elements, reusing a
    /// previously-[`release_array1_f32`](TensorPool::release_array1_f32)d
    /// buffer's backing allocation when one of the right size is available.
    ///
    /// Unlike [`TensorPool::acquire_f32`], the returned value is a plain
    /// `Array1<f32>` rather than a [`PooledBuffer`]: `ndarray` has no drop
    /// hook to return storage to a pool automatically, so callers give the
    /// buffer back explicitly via `release_array1_f32` once done with it.
    pub fn acquire_array1_f32(
        &self,
        size: usize,
    ) -> InferenceResult<scirs2_core::ndarray::Array1<f32>> {
        let buffer = self.acquire_f32(BufferKey::f32(size))?;
        Ok(scirs2_core::ndarray::Array1::from_vec(buffer.into_vec()))
    }

    /// Return an `Array1<f32>` previously obtained from
    /// [`TensorPool::acquire_array1_f32`] so its backing allocation can be
    /// reused by a future `acquire_*` call.
    pub fn release_array1_f32(
        &self,
        array: scirs2_core::ndarray::Array1<f32>,
    ) -> InferenceResult<()> {
        let key = BufferKey::f32(array.len());
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| InferenceError::LockError(format!("Failed to acquire lock: {}", e)))?;
        inner.return_f32(key, array.into_raw_vec_and_offset().0);
        Ok(())
    }
}

impl TensorPoolInner {
    /// Return an `f32` buffer to its pool.
    ///
    /// Installed as `PooledBuffer::<f32>::return_fn` by [`TensorPool::acquire_f32`],
    /// so the element type is statically known here and the buffer is stored
    /// as-is — no `transmute`, and therefore no way for a mismatched `dtype`
    /// string to make a `Vec<f64>` be deallocated with an `f32` layout.
    fn return_f32(&mut self, key: BufferKey, buffer: Vec<f32>) {
        self.stats.total_returns += 1;
        let max_buffers = self.max_buffers_per_key;

        let stored = {
            let pool = self.f32_buffers.entry(key).or_default();
            if pool.len() < max_buffers {
                pool.push(buffer);
                true
            } else {
                false
            }
        };

        if !stored {
            self.stats.total_discards += 1;
        }
    }

    /// Return an `f64` buffer to its pool. See [`TensorPoolInner::return_f32`].
    fn return_f64(&mut self, key: BufferKey, buffer: Vec<f64>) {
        self.stats.total_returns += 1;
        let max_buffers = self.max_buffers_per_key;

        let stored = {
            let pool = self.f64_buffers.entry(key).or_default();
            if pool.len() < max_buffers {
                pool.push(buffer);
                true
            } else {
                false
            }
        };

        if !stored {
            self.stats.total_discards += 1;
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

    /// Regression: `acquire_f64(BufferKey::f32(n))` used to hand back a
    /// `PooledBuffer<f64>` whose key claimed `"f32"`; on drop the `Vec<f64>` was
    /// `transmute`d into a `Vec<f32>` and pushed into the f32 pool, where reusing
    /// it deallocated a 128-byte/align-8 allocation with a 64-byte/align-4 layout.
    #[test]
    fn test_mismatched_dtype_key_is_rejected() {
        let pool = TensorPool::new();

        let wrong_f64 = pool.acquire_f64(BufferKey::f32(16));
        assert!(
            matches!(wrong_f64, Err(InferenceError::InvalidConfiguration(_))),
            "acquiring f64 with an f32 key must be rejected"
        );

        let wrong_f32 = pool.acquire_f32(BufferKey::f64(16));
        assert!(
            matches!(wrong_f32, Err(InferenceError::InvalidConfiguration(_))),
            "acquiring f32 with an f64 key must be rejected"
        );

        let custom = BufferKey {
            size: 16,
            dtype: "i8".to_string(),
            tag: None,
        };
        assert!(matches!(
            pool.acquire_f32(custom),
            Err(InferenceError::InvalidConfiguration(_))
        ));

        // Nothing was pooled, so no cross-type buffer can be handed out later.
        assert_eq!(pool.pooled_count().expect("Failed to get count"), 0);
    }

    /// Buffers must round-trip through the pool with their values intact and
    /// their element type unchanged.
    #[test]
    fn test_f64_buffers_round_trip_without_reinterpretation() {
        let pool = TensorPool::new();
        let key = BufferKey::f64(8);

        {
            let mut buf = pool
                .acquire_f64(key.clone())
                .expect("Failed to acquire buffer");
            for (i, slot) in buf.data_mut().iter_mut().enumerate() {
                *slot = i as f64;
            }
        }

        let reused = pool.acquire_f64(key).expect("Failed to acquire buffer");
        assert_eq!(reused.len(), 8);
        // `acquire_*` clears the buffer, so reuse never leaks previous contents.
        assert!(reused.data().iter().all(|&x| x == 0.0));

        let stats = pool.stats().expect("Failed to get stats");
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_reuses, 1);
    }

    #[test]
    fn test_array1_f32_round_trips_through_pool() {
        let pool = TensorPool::new();

        let mut a = pool.acquire_array1_f32(4).expect("acquire must succeed");
        assert_eq!(a.len(), 4);
        assert!(a.iter().all(|&x| x == 0.0));
        a[0] = 42.0;

        pool.release_array1_f32(a).expect("release must succeed");

        let stats_before = pool.stats().expect("stats");
        assert_eq!(stats_before.total_allocations, 1);
        assert_eq!(stats_before.total_returns, 1);

        // Reacquiring the same size must reuse the released allocation
        // (a real hit, not another fresh allocation), and the previous
        // contents must not leak through.
        let b = pool.acquire_array1_f32(4).expect("acquire must succeed");
        let stats_after = pool.stats().expect("stats");
        assert_eq!(
            stats_after.total_allocations, 1,
            "must reuse, not reallocate"
        );
        assert_eq!(stats_after.total_reuses, 1);
        assert!(b.iter().all(|&x| x == 0.0), "reused buffer must be cleared");
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
