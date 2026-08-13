//! Parallel computation utilities for multi-layer SSM processing
//!
//! Provides parallel execution strategies for:
//! - Batch processing of multiple inputs
//! - Parallel layer computation where data dependencies allow
//! - Multi-threaded matrix operations
//!
//! Uses scirs2-core parallel abstractions (NOT rayon directly per KIZZASI_POLICY.md),
//! via `scirs2_core::parallel_ops` -- the same API `scan.rs`'s
//! `parallel_ssm_batch` and `segmented_scan` already use successfully. Every
//! function below dispatches to a real multi-threaded iterator once the
//! input crosses `ParallelConfig`'s size threshold; below the threshold
//! (or when the config disables parallelism), the sequential path is used
//! since thread dispatch overhead would dominate on small inputs.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::parallel_ops::{
    IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

/// Default size threshold (element/item count) above which the free
/// functions in this module (`parallel_matvec_batch`, `parallel_map`,
/// `parallel_sum`) dispatch to `scirs2_core::parallel_ops` instead of a
/// plain sequential iterator. Mirrors `ParallelConfig::default()`'s
/// `min_batch_size`/`min_vector_size` order of magnitude for callers that
/// don't have a `ParallelConfig` handy.
const DEFAULT_PARALLEL_THRESHOLD: usize = 256;

/// Batch processor for parallel input processing
///
/// Uses `scirs2_core::parallel_ops` for multi-threaded processing once the
/// input size crosses `num_threads()`'s implied parallelism (see
/// [`Self::process_batch`]/[`Self::process_layers_parallel`]); falls back to
/// sequential iteration for small inputs, where thread dispatch overhead
/// would dominate.
#[derive(Debug)]
pub struct BatchProcessor {
    /// Number of worker threads (0 = auto-detect). Scales the size
    /// threshold below which `process_batch`/`process_layers_parallel` stay
    /// sequential: more available threads lowers the bar for when
    /// parallelizing pays off relative to dispatch overhead.
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

    /// Minimum input length at which `process_batch`/`process_layers_parallel`
    /// switch from sequential to `scirs2_core::parallel_ops`. Scales
    /// inversely with `num_threads()`: with more threads available, a
    /// smaller batch already has enough work per thread to be worth
    /// dispatching.
    fn parallel_threshold(&self) -> usize {
        let threads = self.num_threads().max(1);
        (DEFAULT_PARALLEL_THRESHOLD / threads).max(8)
    }

    /// Process a batch of inputs.
    ///
    /// Dispatches to `scirs2_core::parallel_ops::into_par_iter` when
    /// `inputs.len()` is at least `Self::parallel_threshold`; otherwise
    /// iterates sequentially, since thread dispatch overhead would dominate
    /// a small batch.
    pub fn process_batch<F, T, R>(&self, inputs: &[T], f: F) -> Vec<R>
    where
        F: Fn(&T) -> R + Send + Sync,
        T: Sync,
        R: Send,
    {
        if inputs.len() >= self.parallel_threshold() {
            inputs.into_par_iter().map(&f).collect()
        } else {
            inputs.iter().map(f).collect()
        }
    }

    /// Process multiple layers.
    ///
    /// For independent layer computations (e.g., different attention
    /// heads). Sequential layer dependencies still require sequential
    /// processing -- callers with data dependencies between layers must not
    /// use this. Dispatches to `scirs2_core::parallel_ops` when
    /// `num_layers` is at least `Self::parallel_threshold`.
    pub fn process_layers_parallel<F, R>(&self, num_layers: usize, f: F) -> Vec<R>
    where
        F: Fn(usize) -> R + Send + Sync,
        R: Send,
    {
        if num_layers >= self.parallel_threshold() {
            (0..num_layers).into_par_iter().map(f).collect()
        } else {
            (0..num_layers).map(f).collect()
        }
    }
}

/// Matrix-vector multiplication for batched operations.
///
/// Dispatches to `scirs2_core::parallel_ops` when the batch is at least
/// `DEFAULT_PARALLEL_THRESHOLD` items; otherwise iterates sequentially.
/// Truncates to `min(matrices.len(), vectors.len())`, matching the
/// well-defined-prefix convention `simd.rs`'s kernels use for mismatched
/// input lengths.
pub fn parallel_matvec_batch(
    matrices: &[Array2<f32>],
    vectors: &[Array1<f32>],
) -> Vec<Array1<f32>> {
    let n = matrices.len().min(vectors.len());
    if n >= DEFAULT_PARALLEL_THRESHOLD {
        (0..n)
            .into_par_iter()
            .map(|i| matrices[i].dot(&vectors[i]))
            .collect()
    } else {
        matrices[..n]
            .iter()
            .zip(vectors[..n].iter())
            .map(|(m, v)| m.dot(v))
            .collect()
    }
}

/// Element-wise operations on arrays, in place.
///
/// Dispatches to `scirs2_core::parallel_ops::par_iter_mut` when
/// `data.len()` is at least `DEFAULT_PARALLEL_THRESHOLD`; otherwise
/// iterates sequentially.
pub fn parallel_map<F>(data: &mut [f32], f: F)
where
    F: Fn(f32) -> f32 + Send + Sync,
{
    if data.len() >= DEFAULT_PARALLEL_THRESHOLD {
        data.par_iter_mut().for_each(|x| *x = f(*x));
    } else {
        data.iter_mut().for_each(|x| *x = f(*x));
    }
}

/// Reduction (sum).
///
/// Dispatches to `scirs2_core::parallel_ops::into_par_iter` when
/// `data.len()` is at least `DEFAULT_PARALLEL_THRESHOLD`; otherwise sums
/// sequentially.
pub fn parallel_sum(data: &[f32]) -> f32 {
    if data.len() >= DEFAULT_PARALLEL_THRESHOLD {
        data.into_par_iter().sum()
    } else {
        data.iter().sum()
    }
}

/// Dot product for large vectors
///
/// Delegates to [`crate::simd::dot_product`], which dispatches at runtime to
/// the AVX-512/AVX2 or NEON kernel for the current target (portable unrolled
/// scalar elsewhere). Parallel chunking via scirs2-core will be layered on top
/// for very large vectors when that API is stabilized.
pub fn parallel_dot(a: &[f32], b: &[f32]) -> f32 {
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

    // ------------------------------------------------------------------
    // Regression tests: every function below used to be a plain sequential
    // iterator regardless of input size, contradicting its own doc comment
    // ("Uses parallel processing via scirs2-core when available"). These
    // exercise inputs at/above `DEFAULT_PARALLEL_THRESHOLD` (or an
    // equivalent `BatchProcessor` threshold) so the
    // `scirs2_core::parallel_ops` dispatch branch actually runs, and check
    // results still match a plain sequential reference.
    // ------------------------------------------------------------------

    #[test]
    fn test_process_batch_matches_sequential_reference_above_threshold() {
        let processor = BatchProcessor::with_threads(4); // threshold = 256/4 = 64
        let inputs: Vec<i64> = (0..1000).collect();
        assert!(inputs.len() >= processor.parallel_threshold());

        let parallel_results = processor.process_batch(&inputs, |&x| x * x - 3);
        let sequential_results: Vec<i64> = inputs.iter().map(|&x| x * x - 3).collect();

        assert_eq!(parallel_results, sequential_results);
    }

    #[test]
    fn test_process_layers_parallel_matches_sequential_reference_above_threshold() {
        let processor = BatchProcessor::with_threads(2); // threshold = 256/2 = 128
        let num_layers = 500;
        assert!(num_layers >= processor.parallel_threshold());

        let parallel_results = processor.process_layers_parallel(num_layers, |i| i * 2 + 1);
        let sequential_results: Vec<usize> = (0..num_layers).map(|i| i * 2 + 1).collect();

        assert_eq!(parallel_results, sequential_results);
    }

    #[test]
    fn test_parallel_matvec_batch_above_threshold_matches_sequential() {
        let n = DEFAULT_PARALLEL_THRESHOLD + 50;
        let matrices: Vec<Array2<f32>> = (0..n)
            .map(|i| Array2::from_shape_fn((3, 3), |(r, c)| (i + r + c) as f32 * 0.01))
            .collect();
        let vectors: Vec<Array1<f32>> = (0..n)
            .map(|i| Array1::from_shape_fn(3, |c| (i * c + 1) as f32 * 0.1))
            .collect();

        let parallel_results = parallel_matvec_batch(&matrices, &vectors);
        let sequential_results: Vec<Array1<f32>> = matrices
            .iter()
            .zip(vectors.iter())
            .map(|(m, v)| m.dot(v))
            .collect();

        assert_eq!(parallel_results.len(), sequential_results.len());
        for (p, s) in parallel_results.iter().zip(sequential_results.iter()) {
            for i in 0..3 {
                assert!((p[i] - s[i]).abs() < 1e-5, "mismatch: {p:?} vs {s:?}");
            }
        }
    }

    #[test]
    fn test_parallel_map_above_threshold_matches_sequential() {
        let n = DEFAULT_PARALLEL_THRESHOLD + 17;
        let mut parallel_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 10.0).collect();
        let sequential_data = parallel_data.clone();

        parallel_map(&mut parallel_data, |x| x * x + 1.0);
        let sequential_result: Vec<f32> = sequential_data.iter().map(|&x| x * x + 1.0).collect();

        assert_eq!(parallel_data.len(), sequential_result.len());
        for (p, s) in parallel_data.iter().zip(sequential_result.iter()) {
            assert!((p - s).abs() < 1e-4, "mismatch: {p} vs {s}");
        }
    }

    #[test]
    fn test_parallel_sum_above_threshold_matches_sequential() {
        let n = DEFAULT_PARALLEL_THRESHOLD + 123;
        // Integer-valued f32s so the sum is exact regardless of the
        // reduction's evaluation order, letting us assert bit-exact
        // equality between the parallel and sequential sums.
        let data: Vec<f32> = (0..n).map(|i| (i % 97) as f32).collect();

        let parallel_result = parallel_sum(&data);
        let sequential_result: f32 = data.iter().sum();

        assert_eq!(parallel_result.to_bits(), sequential_result.to_bits());
    }

    #[test]
    fn test_batch_processor_num_threads_influences_parallel_threshold() {
        // Regression: `num_threads` was stored and reported back but never
        // influenced any actual behaviour. `parallel_threshold` is the
        // (new) place it matters: more configured threads means a smaller
        // batch already crosses into the parallel path.
        let single_threaded = BatchProcessor::with_threads(1);
        let many_threaded = BatchProcessor::with_threads(64);
        assert!(single_threaded.parallel_threshold() > many_threaded.parallel_threshold());
    }
}
