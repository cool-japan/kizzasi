//! Memory pooling for allocation reuse
//!
//! Provides object pools for frequently allocated data structures to reduce
//! allocation overhead in real-time signal processing applications.

use scirs2_core::ndarray::Array1;
use std::cell::RefCell;
use std::collections::VecDeque;

/// A pool for reusing Array1 allocations
#[derive(Debug)]
pub struct ArrayPool {
    /// Size of arrays in this pool
    array_size: usize,
    /// Maximum pool capacity
    max_capacity: usize,
    /// Pool of available arrays
    pool: RefCell<VecDeque<Array1<f32>>>,
    /// Statistics
    stats: RefCell<PoolStats>,
}

/// Pool usage statistics
#[derive(Debug, Default, Clone, Copy)]
pub struct PoolStats {
    /// Number of allocations from pool
    pub hits: u64,
    /// Number of allocations that missed (required new allocation)
    pub misses: u64,
    /// Number of returns to pool
    pub returns: u64,
    /// Number of returns dropped (pool was full)
    pub drops: u64,
}

impl PoolStats {
    /// Get hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64 * 100.0
        }
    }
}

impl ArrayPool {
    /// Create a new array pool
    pub fn new(array_size: usize, max_capacity: usize) -> Self {
        Self {
            array_size,
            max_capacity,
            pool: RefCell::new(VecDeque::with_capacity(max_capacity)),
            stats: RefCell::new(PoolStats::default()),
        }
    }

    /// Get array size
    pub fn array_size(&self) -> usize {
        self.array_size
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.pool.borrow().len()
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        *self.stats.borrow()
    }

    /// Acquire an array from the pool, or create a new one
    pub fn acquire(&self) -> Array1<f32> {
        let mut pool = self.pool.borrow_mut();
        let mut stats = self.stats.borrow_mut();

        if let Some(array) = pool.pop_front() {
            stats.hits += 1;
            array
        } else {
            stats.misses += 1;
            Array1::zeros(self.array_size)
        }
    }

    /// Acquire an array and fill it with a value
    pub fn acquire_filled(&self, value: f32) -> Array1<f32> {
        let mut array = self.acquire();
        array.fill(value);
        array
    }

    /// Acquire an array and fill it with zeros
    pub fn acquire_zeros(&self) -> Array1<f32> {
        let mut array = self.acquire();
        array.fill(0.0);
        array
    }

    /// Return an array to the pool
    pub fn release(&self, array: Array1<f32>) {
        // Only accept arrays of the correct size
        if array.len() != self.array_size {
            return;
        }

        let mut pool = self.pool.borrow_mut();
        let mut stats = self.stats.borrow_mut();

        if pool.len() < self.max_capacity {
            pool.push_back(array);
            stats.returns += 1;
        } else {
            stats.drops += 1;
            // Array is dropped here
        }
    }

    /// Clear the pool
    pub fn clear(&self) {
        self.pool.borrow_mut().clear();
    }

    /// Pre-allocate arrays to fill the pool
    pub fn warm(&self) {
        let mut pool = self.pool.borrow_mut();
        while pool.len() < self.max_capacity {
            pool.push_back(Array1::zeros(self.array_size));
        }
    }
}

/// A scoped handle that automatically returns an array to the pool
pub struct PooledArray<'a> {
    array: Option<Array1<f32>>,
    pool: &'a ArrayPool,
}

impl<'a> PooledArray<'a> {
    /// Create a new pooled array handle
    pub fn new(pool: &'a ArrayPool) -> Self {
        Self {
            array: Some(pool.acquire()),
            pool,
        }
    }

    /// Create a new pooled array filled with zeros
    pub fn zeros(pool: &'a ArrayPool) -> Self {
        Self {
            array: Some(pool.acquire_zeros()),
            pool,
        }
    }

    /// Get a reference to the underlying array
    pub fn as_array(&self) -> &Array1<f32> {
        self.array.as_ref().unwrap()
    }

    /// Get a mutable reference to the underlying array
    pub fn as_array_mut(&mut self) -> &mut Array1<f32> {
        self.array.as_mut().unwrap()
    }

    /// Take ownership of the array, preventing automatic return
    pub fn take(mut self) -> Array1<f32> {
        self.array.take().unwrap()
    }
}

impl Drop for PooledArray<'_> {
    fn drop(&mut self) {
        if let Some(array) = self.array.take() {
            self.pool.release(array);
        }
    }
}

impl std::ops::Deref for PooledArray<'_> {
    type Target = Array1<f32>;

    fn deref(&self) -> &Self::Target {
        self.as_array()
    }
}

impl std::ops::DerefMut for PooledArray<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_array_mut()
    }
}

/// Multi-size memory pool for arrays of different sizes
#[derive(Debug)]
pub struct MultiArrayPool {
    /// Pools for different sizes
    pools: Vec<ArrayPool>,
    /// Size classes (sorted ascending)
    sizes: Vec<usize>,
}

impl MultiArrayPool {
    /// Create a new multi-size pool with default size classes
    pub fn new() -> Self {
        // Common sizes for signal processing: powers of 2
        Self::with_sizes(&[32, 64, 128, 256, 512, 1024, 2048, 4096], 8)
    }

    /// Create a pool with custom size classes
    pub fn with_sizes(sizes: &[usize], capacity_per_size: usize) -> Self {
        let mut sorted_sizes: Vec<usize> = sizes.to_vec();
        sorted_sizes.sort_unstable();

        let pools = sorted_sizes
            .iter()
            .map(|&size| ArrayPool::new(size, capacity_per_size))
            .collect();

        Self {
            pools,
            sizes: sorted_sizes,
        }
    }

    /// Acquire an array of at least the requested size
    pub fn acquire(&self, min_size: usize) -> Array1<f32> {
        // Find the smallest pool that can satisfy the request
        if let Some(idx) = self.sizes.iter().position(|&s| s >= min_size) {
            self.pools[idx].acquire()
        } else {
            // Size is larger than any pool, allocate directly
            Array1::zeros(min_size)
        }
    }

    /// Acquire an array and fill with zeros
    pub fn acquire_zeros(&self, min_size: usize) -> Array1<f32> {
        let mut arr = self.acquire(min_size);
        arr.fill(0.0);
        arr
    }

    /// Release an array back to the appropriate pool
    pub fn release(&self, array: Array1<f32>) {
        let size = array.len();
        // Find exact match
        if let Some(idx) = self.sizes.iter().position(|&s| s == size) {
            self.pools[idx].release(array);
        }
        // If no exact match, array is dropped
    }

    /// Get aggregate statistics
    pub fn stats(&self) -> PoolStats {
        let mut total = PoolStats::default();
        for pool in &self.pools {
            let s = pool.stats();
            total.hits += s.hits;
            total.misses += s.misses;
            total.returns += s.returns;
            total.drops += s.drops;
        }
        total
    }

    /// Warm up all pools
    pub fn warm(&self) {
        for pool in &self.pools {
            pool.warm();
        }
    }

    /// Clear all pools
    pub fn clear(&self) {
        for pool in &self.pools {
            pool.clear();
        }
    }
}

impl Default for MultiArrayPool {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local memory pool for high-performance allocation
thread_local! {
    static LOCAL_POOL: MultiArrayPool = MultiArrayPool::new();
}

/// Acquire an array from the thread-local pool
pub fn tl_acquire(min_size: usize) -> Array1<f32> {
    LOCAL_POOL.with(|pool| pool.acquire(min_size))
}

/// Acquire a zeroed array from the thread-local pool
pub fn tl_acquire_zeros(min_size: usize) -> Array1<f32> {
    LOCAL_POOL.with(|pool| pool.acquire_zeros(min_size))
}

/// Release an array to the thread-local pool
pub fn tl_release(array: Array1<f32>) {
    LOCAL_POOL.with(|pool| pool.release(array));
}

/// Get statistics from the thread-local pool
pub fn tl_stats() -> PoolStats {
    LOCAL_POOL.with(|pool| pool.stats())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_pool_basic() {
        let pool = ArrayPool::new(64, 4);

        // Acquire from empty pool (miss)
        let arr1 = pool.acquire();
        assert_eq!(arr1.len(), 64);

        // Return to pool
        pool.release(arr1);
        assert_eq!(pool.size(), 1);

        // Acquire again (hit)
        let arr2 = pool.acquire();
        assert_eq!(arr2.len(), 64);
        assert_eq!(pool.size(), 0);

        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_array_pool_capacity() {
        let pool = ArrayPool::new(32, 2);

        // Fill pool
        let a1 = pool.acquire();
        let a2 = pool.acquire();
        let a3 = pool.acquire();

        pool.release(a1);
        pool.release(a2);
        pool.release(a3); // Should be dropped

        let stats = pool.stats();
        assert_eq!(stats.returns, 2);
        assert_eq!(stats.drops, 1);
    }

    #[test]
    fn test_pooled_array_scope() {
        let pool = ArrayPool::new(32, 4);

        {
            let mut arr = PooledArray::zeros(&pool);
            arr[0] = 1.0;
            assert_eq!(arr.len(), 32);
        } // arr is returned to pool here

        assert_eq!(pool.size(), 1);
    }

    #[test]
    fn test_pooled_array_take() {
        let pool = ArrayPool::new(32, 4);

        let owned = {
            let arr = PooledArray::zeros(&pool);
            arr.take() // Take ownership
        };

        assert_eq!(owned.len(), 32);
        assert_eq!(pool.size(), 0); // Nothing returned
    }

    #[test]
    fn test_multi_pool() {
        let pool = MultiArrayPool::with_sizes(&[32, 64, 128], 4);

        // Request 50 elements, should get 64
        let arr = pool.acquire(50);
        assert_eq!(arr.len(), 64);

        pool.release(arr);
        assert_eq!(pool.stats().returns, 1);
    }

    #[test]
    fn test_pool_warm() {
        let pool = ArrayPool::new(64, 4);
        pool.warm();
        assert_eq!(pool.size(), 4);

        // All acquires should be hits now
        for _ in 0..4 {
            let _ = pool.acquire();
        }
        let stats = pool.stats();
        assert_eq!(stats.hits, 4);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_hit_rate() {
        let stats = PoolStats {
            hits: 80,
            misses: 20,
            returns: 0,
            drops: 0,
        };
        assert!((stats.hit_rate() - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_thread_local_pool() {
        let arr = tl_acquire_zeros(100);
        assert!(arr.len() >= 100);

        tl_release(arr);

        let stats = tl_stats();
        assert!(stats.misses >= 1);
    }
}
