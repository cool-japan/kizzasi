//! # Cache-Friendly Memory Layouts
//!
//! Optimized memory layouts for state management in SSM models.
//! Reduces cache misses and improves performance through:
//! - Memory alignment to cache line boundaries
//! - Contiguous memory allocation
//! - Structure of Arrays (SoA) layout for SIMD efficiency
//! - Memory pooling and reuse
//!
//! ## Performance Benefits
//! - **Reduced Cache Misses**: Aligned and contiguous memory
//! - **Better SIMD Utilization**: SoA layout enables vectorization
//! - **Lower Allocation Overhead**: Memory pooling reduces allocations
//! - **Improved Prefetching**: Sequential access patterns

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::Array1;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::{self, NonNull};
use tracing::{debug, trace};

/// Cache line size in bytes (typical for x86-64 and ARM)
pub const CACHE_LINE_SIZE: usize = 64;

/// Alignment for SIMD operations (32 bytes for AVX2, 64 for AVX-512)
pub const SIMD_ALIGNMENT: usize = 64;

/// Configuration for cache-friendly memory allocation
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// Alignment for memory allocations (must be power of 2)
    pub alignment: usize,
    /// Enable memory prefetching hints
    pub enable_prefetch: bool,
    /// Use memory pooling for repeated allocations
    pub use_pooling: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            alignment: SIMD_ALIGNMENT,
            enable_prefetch: true,
            use_pooling: true,
        }
    }
}

/// Aligned memory buffer for cache-friendly storage
pub struct AlignedBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    layout: Layout,
}

impl<T> std::fmt::Debug for AlignedBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("alignment", &self.layout.align())
            .finish()
    }
}

impl<T> AlignedBuffer<T> {
    /// Create a new aligned buffer with specified capacity and alignment
    pub fn new(capacity: usize, alignment: usize) -> ModelResult<Self> {
        if capacity == 0 {
            return Err(ModelError::invalid_config("Buffer capacity must be > 0"));
        }

        if !alignment.is_power_of_two() {
            return Err(ModelError::invalid_config(format!(
                "Alignment must be power of 2, got {}",
                alignment
            )));
        }

        let size = capacity * std::mem::size_of::<T>();
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| ModelError::invalid_config(format!("Invalid layout: {}", e)))?;

        let ptr = unsafe {
            let raw_ptr = alloc(layout);
            if raw_ptr.is_null() {
                return Err(ModelError::AllocationError {
                    bytes: size,
                    purpose: "aligned buffer".into(),
                });
            }
            NonNull::new_unchecked(raw_ptr as *mut T)
        };

        debug!(
            "Allocated aligned buffer: {} bytes, alignment {}",
            size, alignment
        );

        Ok(Self {
            ptr,
            len: 0,
            capacity,
            layout,
        })
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get a slice of the buffer
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable slice of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Push a value to the buffer
    pub fn push(&mut self, value: T) -> ModelResult<()> {
        if self.len >= self.capacity {
            return Err(ModelError::invalid_config("Buffer capacity exceeded"));
        }

        unsafe {
            ptr::write(self.ptr.as_ptr().add(self.len), value);
        }
        self.len += 1;
        Ok(())
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

unsafe impl<T: Send> Send for AlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

/// Structure of Arrays (SoA) layout for multi-layer states
///
/// Instead of storing states as Array of Structures:
/// ```text
/// [Layer0{h, c}, Layer1{h, c}, Layer2{h, c}]
/// ```
///
/// We use Structure of Arrays:
/// ```text
/// {h: [Layer0, Layer1, Layer2], c: [Layer0, Layer1, Layer2]}
/// ```
///
/// This enables better SIMD vectorization and cache utilization.
#[derive(Debug)]
pub struct SoAStateStorage {
    /// Hidden states for all layers (contiguous)
    hidden_states: AlignedBuffer<f32>,
    /// Cell states for all layers (contiguous) - for LSTM-like models
    cell_states: Option<AlignedBuffer<f32>>,
    /// Number of layers
    num_layers: usize,
    /// State dimension per layer
    state_dim: usize,
    /// Configuration
    config: CacheConfig,
}

impl SoAStateStorage {
    /// Create a new SoA state storage
    pub fn new(
        num_layers: usize,
        state_dim: usize,
        use_cell_states: bool,
        config: CacheConfig,
    ) -> ModelResult<Self> {
        let total_elements = num_layers * state_dim;

        let mut hidden_states = AlignedBuffer::new(total_elements, config.alignment)?;
        // Initialize all elements to zero
        for _ in 0..total_elements {
            hidden_states.push(0.0)?;
        }

        let cell_states = if use_cell_states {
            let mut buf = AlignedBuffer::new(total_elements, config.alignment)?;
            for _ in 0..total_elements {
                buf.push(0.0)?;
            }
            Some(buf)
        } else {
            None
        };

        debug!(
            "Created SoA state storage: {} layers × {} dims = {} elements",
            num_layers, state_dim, total_elements
        );

        Ok(Self {
            hidden_states,
            cell_states,
            num_layers,
            state_dim,
            config,
        })
    }

    /// Get hidden state for a specific layer
    pub fn get_hidden(&self, layer_idx: usize) -> ModelResult<Array1<f32>> {
        if layer_idx >= self.num_layers {
            return Err(ModelError::IndexOutOfBounds {
                index: layer_idx,
                limit: self.num_layers,
                context: "layer index".into(),
            });
        }

        let start = layer_idx * self.state_dim;
        let end = start + self.state_dim;
        let slice = &self.hidden_states.as_slice()[start..end];

        Ok(Array1::from_vec(slice.to_vec()))
    }

    /// Set hidden state for a specific layer
    pub fn set_hidden(&mut self, layer_idx: usize, state: &Array1<f32>) -> ModelResult<()> {
        if layer_idx >= self.num_layers {
            return Err(ModelError::IndexOutOfBounds {
                index: layer_idx,
                limit: self.num_layers,
                context: "layer index".into(),
            });
        }

        if state.len() != self.state_dim {
            return Err(ModelError::dimension_mismatch(
                "state dimension",
                self.state_dim,
                state.len(),
            ));
        }

        let start = layer_idx * self.state_dim;
        let slice = &mut self.hidden_states.as_mut_slice()[start..start + self.state_dim];

        // Handle non-contiguous arrays by copying element-wise
        if let Some(state_slice) = state.as_slice() {
            slice.copy_from_slice(state_slice);
        } else {
            for (i, &val) in state.iter().enumerate() {
                slice[i] = val;
            }
        }

        Ok(())
    }

    /// Get all hidden states as a contiguous slice (SIMD-friendly)
    pub fn get_all_hidden(&self) -> &[f32] {
        self.hidden_states.as_slice()
    }

    /// Get all hidden states as mutable slice (SIMD-friendly)
    pub fn get_all_hidden_mut(&mut self) -> &mut [f32] {
        self.hidden_states.as_mut_slice()
    }

    /// Reset all states to zero
    pub fn reset(&mut self) {
        let hidden_slice = self.hidden_states.as_mut_slice();
        hidden_slice.fill(0.0);

        if let Some(ref mut cell_states) = self.cell_states {
            let cell_slice = cell_states.as_mut_slice();
            cell_slice.fill(0.0);
        }

        trace!("Reset all states to zero");
    }

    /// Prefetch state data for next layer (hint to CPU)
    #[inline]
    pub fn prefetch_layer(&self, layer_idx: usize) {
        if !self.config.enable_prefetch || layer_idx >= self.num_layers {
            return;
        }

        let start = layer_idx * self.state_dim;
        let ptr = unsafe { self.hidden_states.as_slice().as_ptr().add(start) };

        // Use compiler intrinsic for prefetching
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            unsafe {
                _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                // ARM equivalent: PRFM PLDL1KEEP
                core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr);
            }
        }
    }
}

/// Memory pool for reusable state buffers
pub struct StatePool {
    /// Pool of available buffers
    pool: Vec<AlignedBuffer<f32>>,
    /// Configuration
    config: CacheConfig,
    /// Statistics
    allocations: usize,
    reuses: usize,
}

impl StatePool {
    /// Create a new state pool
    pub fn new(config: CacheConfig) -> Self {
        Self {
            pool: Vec::new(),
            config,
            allocations: 0,
            reuses: 0,
        }
    }

    /// Acquire a buffer from the pool or allocate a new one
    pub fn acquire(&mut self, capacity: usize) -> ModelResult<AlignedBuffer<f32>> {
        // Try to find a suitable buffer in the pool
        if let Some(pos) = self.pool.iter().position(|buf| buf.capacity() >= capacity) {
            self.reuses += 1;
            let mut buffer = self.pool.swap_remove(pos);
            buffer.clear();
            trace!(
                "Reused buffer from pool (reuse rate: {:.1}%)",
                self.reuse_rate() * 100.0
            );
            Ok(buffer)
        } else {
            self.allocations += 1;
            let buffer = AlignedBuffer::new(capacity, self.config.alignment)?;
            debug!(
                "Allocated new buffer ({} total allocations)",
                self.allocations
            );
            Ok(buffer)
        }
    }

    /// Return a buffer to the pool
    pub fn release(&mut self, buffer: AlignedBuffer<f32>) {
        if self.config.use_pooling {
            self.pool.push(buffer);
        }
        // Otherwise, buffer is dropped
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            allocations: self.allocations,
            reuses: self.reuses,
            pool_size: self.pool.len(),
            reuse_rate: self.reuse_rate(),
        }
    }

    /// Calculate reuse rate
    fn reuse_rate(&self) -> f32 {
        let total = self.allocations + self.reuses;
        if total == 0 {
            0.0
        } else {
            self.reuses as f32 / total as f32
        }
    }
}

/// Pool statistics
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    pub allocations: usize,
    pub reuses: usize,
    pub pool_size: usize,
    pub reuse_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_creation() {
        let buffer: AlignedBuffer<f32> = AlignedBuffer::new(100, 64).expect("Failed to allocate");
        assert_eq!(buffer.capacity(), 100);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_aligned_buffer_push() {
        let mut buffer: AlignedBuffer<f32> =
            AlignedBuffer::new(10, 64).expect("Failed to allocate");

        for i in 0..10 {
            buffer.push(i as f32).expect("Push failed");
        }

        assert_eq!(buffer.len(), 10);
        assert!(!buffer.is_empty());

        let slice = buffer.as_slice();
        for (i, &value) in slice.iter().enumerate() {
            assert!((value - i as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn test_aligned_buffer_clear() {
        let mut buffer: AlignedBuffer<f32> =
            AlignedBuffer::new(10, 64).expect("Failed to allocate");

        for i in 0..5 {
            buffer.push(i as f32).expect("Push failed");
        }

        assert_eq!(buffer.len(), 5);
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_soa_state_storage() {
        let config = CacheConfig::default();
        let storage =
            SoAStateStorage::new(4, 128, false, config).expect("Failed to create storage");

        assert_eq!(storage.num_layers, 4);
        assert_eq!(storage.state_dim, 128);
    }

    #[test]
    fn test_soa_get_set_hidden() {
        let config = CacheConfig::default();
        let mut storage =
            SoAStateStorage::new(3, 64, false, config).expect("Failed to create storage");

        let state = Array1::from_vec(vec![1.0; 64]);
        storage.set_hidden(1, &state).expect("Set hidden failed");

        let retrieved = storage.get_hidden(1).expect("Get hidden failed");
        assert_eq!(retrieved.len(), 64);
        assert!((retrieved[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_soa_reset() {
        let config = CacheConfig::default();
        let mut storage =
            SoAStateStorage::new(2, 32, false, config).expect("Failed to create storage");

        let state = Array1::from_vec(vec![5.0; 32]);
        storage.set_hidden(0, &state).expect("Set hidden failed");
        storage.set_hidden(1, &state).expect("Set hidden failed");

        storage.reset();

        let retrieved0 = storage.get_hidden(0).expect("Get hidden failed");
        let retrieved1 = storage.get_hidden(1).expect("Get hidden failed");

        assert!(retrieved0.iter().all(|&x| x.abs() < 1e-6));
        assert!(retrieved1.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_state_pool() {
        let config = CacheConfig::default();
        let mut pool = StatePool::new(config);

        let buf1 = pool.acquire(100).expect("Acquire failed");
        assert_eq!(buf1.capacity(), 100);

        pool.release(buf1);

        let buf2 = pool.acquire(50).expect("Acquire failed");
        assert_eq!(buf2.capacity(), 100); // Reused larger buffer

        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.reuses, 1);
        assert!((stats.reuse_rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = CacheConfig::default();
        let mut storage =
            SoAStateStorage::new(2, 64, false, config).expect("Failed to create storage");

        let wrong_state = Array1::from_vec(vec![1.0; 32]); // Wrong size
        let result = storage.set_hidden(0, &wrong_state);

        assert!(result.is_err());
    }

    #[test]
    fn test_index_out_of_bounds() {
        let config = CacheConfig::default();
        let storage = SoAStateStorage::new(2, 64, false, config).expect("Failed to create storage");

        let result = storage.get_hidden(5); // Out of bounds
        assert!(result.is_err());
    }

    #[test]
    fn test_alignment_validation() {
        let result = AlignedBuffer::<f32>::new(100, 63); // Not power of 2
        assert!(result.is_err());
    }
}
