//! Performance optimizations for kizzasi-core
//!
//! This module provides optimizations based on profiling results:
//! 1. Allocation reduction through object pooling
//! 2. Cache-friendly data layouts
//! 3. Instruction-level parallelism
//! 4. Prefetching strategies

use scirs2_core::ndarray::{Array1, Array2};
use std::cell::RefCell;

/// Cache for discretized SSM matrices to avoid recomputation
#[derive(Debug)]
pub struct DiscretizationCache {
    /// Cached A_bar matrices (one per layer)
    a_bar_cache: Vec<Array2<f32>>,
    /// Cached B_bar matrices (one per layer)
    b_bar_cache: Vec<Array2<f32>>,
    /// Delta value used for discretization
    cached_delta: f32,
    /// Whether cache is valid
    valid: bool,
}

impl DiscretizationCache {
    /// Create a new discretization cache
    pub fn new(num_layers: usize, hidden_dim: usize, state_dim: usize) -> Self {
        let a_bar_cache = (0..num_layers)
            .map(|_| Array2::zeros((hidden_dim, state_dim)))
            .collect();
        let b_bar_cache = (0..num_layers)
            .map(|_| Array2::zeros((hidden_dim, state_dim)))
            .collect();

        Self {
            a_bar_cache,
            b_bar_cache,
            cached_delta: 0.0,
            valid: false,
        }
    }

    /// Update the cache with new discretized matrices
    pub fn update(&mut self, layer_idx: usize, delta: f32, a_bar: Array2<f32>, b_bar: Array2<f32>) {
        if layer_idx < self.a_bar_cache.len() {
            self.a_bar_cache[layer_idx] = a_bar;
            self.b_bar_cache[layer_idx] = b_bar;
            self.cached_delta = delta;
            self.valid = true;
        }
    }

    /// Get cached discretized matrices if valid
    pub fn get(&self, layer_idx: usize, delta: f32) -> Option<(&Array2<f32>, &Array2<f32>)> {
        if self.valid
            && (delta - self.cached_delta).abs() < 1e-6
            && layer_idx < self.a_bar_cache.len()
        {
            Some((&self.a_bar_cache[layer_idx], &self.b_bar_cache[layer_idx]))
        } else {
            None
        }
    }

    /// Invalidate the cache
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Check if cache is valid for given delta
    pub fn is_valid(&self, delta: f32) -> bool {
        self.valid && (delta - self.cached_delta).abs() < 1e-6
    }
}

/// Preallocated workspace for SSM computations to reduce allocations
#[derive(Debug)]
pub struct SSMWorkspace {
    /// Temporary storage for intermediate results
    temp_hidden: Array1<f32>,
    /// Temporary storage for state updates
    temp_state: Array2<f32>,
    /// Temporary storage for layer outputs
    temp_output: Array1<f32>,
}

impl SSMWorkspace {
    /// Create a new workspace
    pub fn new(hidden_dim: usize, state_dim: usize) -> Self {
        Self {
            temp_hidden: Array1::zeros(hidden_dim),
            temp_state: Array2::zeros((hidden_dim, state_dim)),
            temp_output: Array1::zeros(hidden_dim),
        }
    }

    /// Get temporary hidden vector (mutable)
    pub fn temp_hidden_mut(&mut self) -> &mut Array1<f32> {
        &mut self.temp_hidden
    }

    /// Get temporary state matrix (mutable)
    pub fn temp_state_mut(&mut self) -> &mut Array2<f32> {
        &mut self.temp_state
    }

    /// Get temporary output vector (mutable)
    pub fn temp_output_mut(&mut self) -> &mut Array1<f32> {
        &mut self.temp_output
    }

    /// Reset all temporary storage to zeros
    pub fn clear(&mut self) {
        self.temp_hidden.fill(0.0);
        self.temp_state.fill(0.0);
        self.temp_output.fill(0.0);
    }
}

// Thread-local workspace pool to avoid allocations
thread_local! {
    static WORKSPACE_POOL: RefCell<Vec<SSMWorkspace>> = const { RefCell::new(Vec::new()) };
}

/// Acquire a workspace from the pool
pub fn acquire_workspace(hidden_dim: usize, state_dim: usize) -> SSMWorkspace {
    WORKSPACE_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        pool.pop()
            .unwrap_or_else(|| SSMWorkspace::new(hidden_dim, state_dim))
    })
}

/// Return a workspace to the pool
pub fn release_workspace(mut workspace: SSMWorkspace) {
    workspace.clear();
    WORKSPACE_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < 16 {
            // Limit pool size
            pool.push(workspace);
        }
    });
}

/// RAII guard for automatic workspace return
pub struct WorkspaceGuard {
    workspace: Option<SSMWorkspace>,
}

impl WorkspaceGuard {
    /// Create a new workspace guard
    pub fn new(hidden_dim: usize, state_dim: usize) -> Self {
        Self {
            workspace: Some(acquire_workspace(hidden_dim, state_dim)),
        }
    }

    /// Get reference to the workspace
    pub fn get(&self) -> &SSMWorkspace {
        self.workspace.as_ref().expect("workspace should exist")
    }

    /// Get mutable reference to the workspace
    pub fn get_mut(&mut self) -> &mut SSMWorkspace {
        self.workspace.as_mut().expect("workspace should exist")
    }
}

impl Drop for WorkspaceGuard {
    fn drop(&mut self) {
        if let Some(workspace) = self.workspace.take() {
            release_workspace(workspace);
        }
    }
}

/// Prefetch hint for cache optimization
#[inline(always)]
pub fn prefetch<T>(_ptr: *const T) {
    // Prefetch is a hint and can be platform-specific
    // For now, this is a no-op that will be optimized by the compiler
    // In release builds with target-specific features, this can be expanded
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        core::arch::x86_64::_mm_prefetch::<3>(_ptr as *const i8);
    }

    // ARM prefetch intrinsics are unstable, so we skip them for now
    // The compiler's auto-vectorization will handle prefetching on ARM
}

/// Cache-aligned buffer for better memory performance
#[repr(align(64))]
pub struct CacheAligned<T> {
    data: T,
}

impl<T> CacheAligned<T> {
    /// Create a new cache-aligned value
    pub fn new(data: T) -> Self {
        Self { data }
    }

    /// Get a reference to the inner data
    pub fn get(&self) -> &T {
        &self.data
    }

    /// Get a mutable reference to the inner data
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Consume and return the inner data
    pub fn into_inner(self) -> T {
        self.data
    }
}

/// Instruction-level parallelism optimizations
pub mod ilp {
    use scirs2_core::ndarray::{Array1, ArrayView1};

    /// Dot product with manual loop unrolling for ILP
    #[inline]
    pub fn dot_unrolled(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
        let len = a.len().min(b.len());
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let chunks = len / 4;
        let remainder = len % 4;

        // Process 4 elements at a time for ILP
        for i in 0..chunks {
            let idx = i * 4;
            sum0 += a[idx] * b[idx];
            sum1 += a[idx + 1] * b[idx + 1];
            sum2 += a[idx + 2] * b[idx + 2];
            sum3 += a[idx + 3] * b[idx + 3];
        }

        // Process remainder
        let mut sum_remainder = 0.0f32;
        for i in (chunks * 4)..(chunks * 4 + remainder) {
            sum_remainder += a[i] * b[i];
        }

        sum0 + sum1 + sum2 + sum3 + sum_remainder
    }

    /// Vector addition with loop unrolling
    #[inline]
    pub fn add_unrolled(a: &Array1<f32>, b: &Array1<f32>, out: &mut Array1<f32>) {
        let len = a.len().min(b.len()).min(out.len());
        let chunks = len / 4;
        let remainder = len % 4;

        for i in 0..chunks {
            let idx = i * 4;
            out[idx] = a[idx] + b[idx];
            out[idx + 1] = a[idx + 1] + b[idx + 1];
            out[idx + 2] = a[idx + 2] + b[idx + 2];
            out[idx + 3] = a[idx + 3] + b[idx + 3];
        }

        for i in (chunks * 4)..(chunks * 4 + remainder) {
            out[i] = a[i] + b[i];
        }
    }

    /// Fused multiply-add with loop unrolling
    #[inline]
    pub fn fma_unrolled(a: &Array1<f32>, b: &Array1<f32>, c: &Array1<f32>, out: &mut Array1<f32>) {
        let len = a.len().min(b.len()).min(c.len()).min(out.len());
        let chunks = len / 4;
        let remainder = len % 4;

        for i in 0..chunks {
            let idx = i * 4;
            out[idx] = a[idx].mul_add(b[idx], c[idx]);
            out[idx + 1] = a[idx + 1].mul_add(b[idx + 1], c[idx + 1]);
            out[idx + 2] = a[idx + 2].mul_add(b[idx + 2], c[idx + 2]);
            out[idx + 3] = a[idx + 3].mul_add(b[idx + 3], c[idx + 3]);
        }

        for i in (chunks * 4)..(chunks * 4 + remainder) {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discretization_cache() {
        let mut cache = DiscretizationCache::new(2, 64, 8);
        assert!(!cache.is_valid(0.1));

        let a_bar = Array2::ones((64, 8));
        let b_bar = Array2::ones((64, 8));

        cache.update(0, 0.1, a_bar.clone(), b_bar.clone());
        assert!(cache.is_valid(0.1));

        let (cached_a, cached_b) = cache.get(0, 0.1).expect("cache should hit");
        assert_eq!(cached_a.shape(), &[64, 8]);
        assert_eq!(cached_b.shape(), &[64, 8]);

        cache.invalidate();
        assert!(!cache.is_valid(0.1));
    }

    #[test]
    fn test_workspace() {
        let mut workspace = SSMWorkspace::new(64, 8);
        workspace.temp_hidden_mut().fill(1.0);
        assert_eq!(workspace.temp_hidden_mut().len(), 64);

        workspace.clear();
        assert_eq!(workspace.temp_hidden_mut().sum(), 0.0);
    }

    #[test]
    fn test_workspace_pool() {
        let workspace1 = acquire_workspace(64, 8);
        assert_eq!(workspace1.temp_hidden.len(), 64);

        release_workspace(workspace1);

        let workspace2 = acquire_workspace(64, 8);
        assert_eq!(workspace2.temp_hidden.len(), 64);
    }

    #[test]
    fn test_workspace_guard() {
        let mut guard = WorkspaceGuard::new(64, 8);
        guard.get_mut().temp_hidden_mut().fill(1.0);
        assert_eq!(guard.get().temp_hidden.len(), 64);
    }

    #[test]
    fn test_cache_aligned() {
        let aligned = CacheAligned::new(vec![1.0f32, 2.0, 3.0]);
        assert_eq!(aligned.get().len(), 3);

        let mut aligned = CacheAligned::new(42);
        *aligned.get_mut() = 100;
        assert_eq!(*aligned.get(), 100);
    }

    #[test]
    fn test_ilp_dot_unrolled() {
        use scirs2_core::ndarray::arr1;

        let a = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(&[2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = ilp::dot_unrolled(a.view(), b.view());
        let expected: f32 = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0 + 5.0 * 6.0;
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_ilp_add_unrolled() {
        use scirs2_core::ndarray::arr1;

        let a = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(&[2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut out = Array1::zeros(5);

        ilp::add_unrolled(&a, &b, &mut out);
        assert_eq!(out[0], 3.0);
        assert_eq!(out[4], 11.0);
    }

    #[test]
    fn test_ilp_fma_unrolled() {
        use scirs2_core::ndarray::arr1;

        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let b = arr1(&[2.0, 3.0, 4.0, 5.0]);
        let c = arr1(&[1.0, 1.0, 1.0, 1.0]);
        let mut out = Array1::zeros(4);

        ilp::fma_unrolled(&a, &b, &c, &mut out);
        assert_eq!(out[0], 1.0 * 2.0 + 1.0);
        assert_eq!(out[3], 4.0 * 5.0 + 1.0);
    }
}
