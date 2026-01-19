//! Parallel computation utilities for multi-layer SSM processing
//!
//! Provides parallel execution strategies for:
//! - Batch processing of multiple inputs
//! - Parallel layer computation where data dependencies allow
//! - Multi-threaded matrix operations
//!
//! Uses scirs2-core parallel abstractions (NOT rayon directly per KIZZASI_POLICY.md).
//! Parallel features are enabled via scirs2-core's parallel feature.

use scirs2_core::ndarray::{Array1, Array2};

/// Batch processor for parallel input processing
///
/// When scirs2-core parallel features are available, uses multi-threaded processing.
/// Falls back to sequential processing otherwise.
#[derive(Debug)]
pub struct BatchProcessor {
    /// Number of worker threads (0 = auto-detect)
    num_threads: usize,
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchProcessor {
    /// Create a new batch processor with automatic thread count
    pub fn new() -> Self {
        Self { num_threads: 0 }
    }

    /// Create a batch processor with specific thread count
    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }

    /// Get the number of threads
    pub fn num_threads(&self) -> usize {
        if self.num_threads == 0 {
            num_cpus_hint()
        } else {
            self.num_threads
        }
    }

    /// Process a batch of inputs
    ///
    /// Uses parallel processing via scirs2-core when available.
    pub fn process_batch<F, T, R>(&self, inputs: &[T], f: F) -> Vec<R>
    where
        F: Fn(&T) -> R,
    {
        // Currently using sequential processing
        // Will use scirs2_core::parallel when API is stabilized
        inputs.iter().map(f).collect()
    }

    /// Process multiple layers
    ///
    /// For independent layer computations (e.g., different attention heads).
    /// Sequential layer dependencies still require sequential processing.
    pub fn process_layers_parallel<F, R>(&self, num_layers: usize, f: F) -> Vec<R>
    where
        F: Fn(usize) -> R,
    {
        // Currently using sequential processing
        // Will use scirs2_core::parallel when API is stabilized
        (0..num_layers).map(f).collect()
    }
}

/// Matrix-vector multiplication for batched operations
///
/// Uses parallel processing via scirs2-core when available.
pub fn parallel_matvec_batch(
    matrices: &[Array2<f32>],
    vectors: &[Array1<f32>],
) -> Vec<Array1<f32>> {
    // Currently using sequential processing
    // Will use scirs2_core::parallel when API is stabilized
    matrices
        .iter()
        .zip(vectors.iter())
        .map(|(m, v)| m.dot(v))
        .collect()
}

/// Element-wise operations on arrays
///
/// Uses parallel processing via scirs2-core when available.
pub fn parallel_map<F>(data: &mut [f32], f: F)
where
    F: Fn(f32) -> f32,
{
    // Currently using sequential processing
    // Will use scirs2_core::parallel when API is stabilized
    data.iter_mut().for_each(|x| *x = f(*x));
}

/// Reduction (sum)
///
/// Uses parallel processing via scirs2-core when available.
pub fn parallel_sum(data: &[f32]) -> f32 {
    // Currently using sequential processing
    // Will use scirs2_core::parallel when API is stabilized
    data.iter().sum()
}

/// Dot product for large vectors
///
/// Uses SIMD-optimized version, and will use parallel processing via
/// scirs2-core for very large vectors when API is stabilized.
pub fn parallel_dot(a: &[f32], b: &[f32]) -> f32 {
    // Use SIMD version (already optimized)
    crate::simd::dot_product(a, b)
}

/// Hint for number of CPUs
fn num_cpus_hint() -> usize {
    // Will use scirs2_core::parallel::num_threads() when API is stabilized
    // For now, use a reasonable default
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

/// Configuration for parallel execution
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Enable parallel batch processing
    pub parallel_batch: bool,
    /// Enable parallel layer computation (for independent heads)
    pub parallel_heads: bool,
    /// Minimum batch size to trigger parallel processing
    pub min_batch_size: usize,
    /// Minimum vector size for parallel operations
    pub min_vector_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            parallel_batch: true,
            parallel_heads: true,
            min_batch_size: 4,
            min_vector_size: 4096,
        }
    }
}

impl ParallelConfig {
    /// Create configuration optimized for throughput
    pub fn throughput() -> Self {
        Self {
            parallel_batch: true,
            parallel_heads: true,
            min_batch_size: 2,
            min_vector_size: 2048,
        }
    }

    /// Create configuration optimized for latency (less parallelism)
    pub fn latency() -> Self {
        Self {
            parallel_batch: false,
            parallel_heads: false,
            min_batch_size: 16,
            min_vector_size: 8192,
        }
    }

    /// Should use parallel batch processing for this batch size?
    pub fn should_parallel_batch(&self, batch_size: usize) -> bool {
        self.parallel_batch && batch_size >= self.min_batch_size
    }

    /// Should use parallel heads for this number of heads?
    pub fn should_parallel_heads(&self, num_heads: usize) -> bool {
        self.parallel_heads && num_heads >= 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_processor() {
        let processor = BatchProcessor::new();
        let inputs = vec![1, 2, 3, 4, 5];
        let results = processor.process_batch(&inputs, |&x| x * 2);
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::default();
        assert!(config.should_parallel_batch(4));
        assert!(!config.should_parallel_batch(2));
    }

    #[test]
    fn test_parallel_dot() {
        let a: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let b: Vec<f32> = vec![1.0; 100];
        let result = parallel_dot(&a, &b);
        let expected: f32 = (0..100).map(|x| x as f32).sum();
        assert!((result - expected).abs() < 1e-3);
    }

    #[test]
    fn test_parallel_sum() {
        let data: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let result = parallel_sum(&data);
        let expected: f32 = (0..100).map(|x| x as f32).sum();
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_parallel_matvec_batch() {
        let m1 = Array2::eye(3);
        let m2 = Array2::eye(3);
        let v1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let results = parallel_matvec_batch(&[m1, m2], &[v1.clone(), v2.clone()]);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], v1);
        assert_eq!(results[1], v2);
    }
}
