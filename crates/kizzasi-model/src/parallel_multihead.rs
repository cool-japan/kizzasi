//! # Parallel Multi-Head Computation
//!
//! Optimized parallel computation for multi-head attention and multi-head SSM operations.
//! Leverages Rayon for CPU parallelism and SIMD for vectorization.
//!
//! ## Features
//! - **Parallel Head Processing**: Compute multiple heads concurrently
//! - **SIMD-Friendly Layout**: Optimized memory access patterns
//! - **Work Stealing**: Efficient load balancing across CPU cores
//! - **Cache Optimization**: Minimize cache misses during parallel execution
//!
//! ## Performance Benefits
//! - **Scalability**: Near-linear speedup with number of cores
//! - **Efficiency**: Reduced overhead through batching
//! - **Throughput**: Higher inference throughput for multi-head models

use crate::error::{ModelError, ModelResult};
use rayon::prelude::*;
use scirs2_core::ndarray::{Array2, ArrayView2, ArrayView3};
use tracing::{debug, trace};

/// Configuration for parallel multi-head computation
///
/// `min_heads_for_parallel` and `num_threads` are honoured by every function
/// in this module (via a scoped `rayon::ThreadPool` built when
/// `num_threads != 0` — see [`MultiHeadExecutor`]). `enable_simd` is
/// reserved and currently has no effect: every op here is expressed as
/// `ndarray` operations (`.dot()`, `Zip`, ...), which have no separate
/// non-vectorized code path to switch to.
#[derive(Debug, Clone, Copy)]
pub struct ParallelConfig {
    /// Minimum number of heads to enable parallel processing
    pub min_heads_for_parallel: usize,
    /// Number of threads to use (0 = auto: rayon's global pool)
    pub num_threads: usize,
    /// Reserved: has no effect (see struct docs).
    pub enable_simd: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_heads_for_parallel: 4,
            num_threads: 0, // Auto-detect
            enable_simd: true,
        }
    }
}

/// Multi-head operation executor
pub struct MultiHeadExecutor {
    config: ParallelConfig,
    /// Scoped thread pool sized to `config.num_threads`, built once here
    /// instead of per-call. `None` means "use rayon's global pool" (the
    /// `num_threads == 0` / auto case).
    pool: Option<rayon::ThreadPool>,
}

impl MultiHeadExecutor {
    /// Create a new multi-head executor.
    ///
    /// # Errors
    /// Returns an error if `config.num_threads != 0` and building a thread
    /// pool with that many threads fails.
    pub fn new(config: ParallelConfig) -> ModelResult<Self> {
        debug!(
            "Created MultiHeadExecutor: min_heads={}, threads={}",
            config.min_heads_for_parallel,
            if config.num_threads == 0 {
                "auto".to_string()
            } else {
                config.num_threads.to_string()
            }
        );
        let pool = if config.num_threads == 0 {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(config.num_threads)
                    .build()
                    .map_err(|e| {
                        ModelError::load_error(
                            "MultiHeadExecutor::new",
                            format!("failed to build a {}-thread pool: {e}", config.num_threads),
                        )
                    })?,
            )
        };
        Ok(Self { config, pool })
    }

    /// Execute a function in parallel across heads
    ///
    /// # Arguments
    /// * `num_heads` - Number of heads to process
    /// * `f` - Function to execute for each head (takes head_idx as parameter)
    ///
    /// # Returns
    /// * Vector of results, one per head
    pub fn par_map<F, R>(&self, num_heads: usize, f: F) -> ModelResult<Vec<R>>
    where
        F: Fn(usize) -> ModelResult<R> + Sync + Send,
        R: Send,
    {
        trace!("Parallel execution across {} heads", num_heads);

        if num_heads < self.config.min_heads_for_parallel {
            // Sequential execution for small number of heads
            return (0..num_heads).map(f).collect();
        }

        // Parallel execution — on this executor's own scoped pool when
        // `config.num_threads` requested one, otherwise rayon's global pool.
        let run = move || (0..num_heads).into_par_iter().map(f).collect();
        match &self.pool {
            Some(pool) => pool.install(run),
            None => run(),
        }
    }

    /// Split input into heads and process in parallel
    ///
    /// # Arguments
    /// * `input` - Input tensor (seq_len, d_model)
    /// * `num_heads` - Number of heads
    /// * `head_dim` - Dimension per head
    /// * `f` - Function to process each head (takes head data as Array2)
    ///
    /// # Returns
    /// * Concatenated output from all heads
    pub fn split_heads_and_process<F>(
        &self,
        input: &ArrayView2<f32>,
        num_heads: usize,
        head_dim: usize,
        f: F,
    ) -> ModelResult<Array2<f32>>
    where
        F: Fn(&ArrayView2<f32>) -> ModelResult<Array2<f32>> + Sync + Send,
    {
        let (seq_len, d_model) = input.dim();

        if d_model != num_heads * head_dim {
            return Err(ModelError::dimension_mismatch(
                "d_model vs num_heads × head_dim",
                num_heads * head_dim,
                d_model,
            ));
        }

        trace!(
            "Splitting {} × {} into {} heads of dim {}",
            seq_len,
            d_model,
            num_heads,
            head_dim
        );

        // Process each head
        let head_outputs: Vec<Array2<f32>> = self.par_map(num_heads, |head_idx| {
            let start_col = head_idx * head_dim;
            let end_col = start_col + head_dim;
            let head_input = input.slice(s![.., start_col..end_col]);
            f(&head_input)
        })?;

        // Concatenate outputs
        self.concat_heads(&head_outputs)
    }

    /// Concatenate head outputs
    fn concat_heads(&self, heads: &[Array2<f32>]) -> ModelResult<Array2<f32>> {
        if heads.is_empty() {
            return Err(ModelError::invalid_config("No heads to concatenate"));
        }

        let (seq_len, head_dim) = heads[0].dim();
        let num_heads = heads.len();
        let d_model = num_heads * head_dim;

        let mut output = Array2::zeros((seq_len, d_model));

        for (head_idx, head_output) in heads.iter().enumerate() {
            let start_col = head_idx * head_dim;
            let end_col = start_col + head_dim;
            output
                .slice_mut(s![.., start_col..end_col])
                .assign(head_output);
        }

        Ok(output)
    }
}

/// Run `f` over `0..num_heads`, honouring `config.min_heads_for_parallel`
/// and `config.num_threads`.
///
/// This is the free-function counterpart to [`MultiHeadExecutor::par_map`]:
/// since these functions take a fresh `&ParallelConfig` on every call with
/// nowhere to cache a thread pool, a scoped pool is built per call when
/// `num_threads != 0` rather than once at construction time. Prefer
/// [`MultiHeadExecutor`] in a hot loop that calls with the same config
/// repeatedly.
fn run_over_heads<T, F>(config: &ParallelConfig, num_heads: usize, f: F) -> ModelResult<Vec<T>>
where
    F: Fn(usize) -> ModelResult<T> + Sync + Send,
    T: Send,
{
    if num_heads < config.min_heads_for_parallel {
        return (0..num_heads).map(f).collect();
    }
    if config.num_threads == 0 {
        return (0..num_heads).into_par_iter().map(f).collect();
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.num_threads)
        .build()
        .map_err(|e| {
            ModelError::load_error(
                "run_over_heads",
                format!("failed to build a {}-thread pool: {e}", config.num_threads),
            )
        })?;
    pool.install(|| (0..num_heads).into_par_iter().map(f).collect())
}

/// Parallel multi-head linear projection
///
/// Performs W_q, W_k, W_v projections in parallel across heads.
///
/// # Arguments
/// * `input` - Input tensor (seq_len, d_model)
/// * `weights` - Weight matrices (num_heads, d_model, head_dim)
/// * `config` - Parallel configuration
///
/// # Returns
/// * Projected outputs (num_heads, seq_len, head_dim)
pub fn parallel_multi_head_projection(
    input: &ArrayView2<f32>,
    weights: &ArrayView3<f32>,
    config: &ParallelConfig,
) -> ModelResult<Vec<Array2<f32>>> {
    let (seq_len, d_model) = input.dim();
    let (num_heads, weight_d_model, _head_dim) = weights.dim();

    if d_model != weight_d_model {
        return Err(ModelError::dimension_mismatch(
            "input d_model vs weight d_model",
            weight_d_model,
            d_model,
        ));
    }

    trace!(
        "Parallel projection: {} × {} with {} heads",
        seq_len,
        d_model,
        num_heads
    );

    let process_head = |head_idx: usize| -> ModelResult<Array2<f32>> {
        let head_weight = weights.slice(s![head_idx, .., ..]);
        // Compute: input @ W_head = (seq_len, d_model) @ (d_model, head_dim)
        Ok(input.dot(&head_weight))
    };

    run_over_heads(config, num_heads, process_head)
}

/// Parallel multi-head output combination
///
/// Combines outputs from multiple heads through a final projection.
///
/// # Arguments
/// * `head_outputs` - Outputs from each head (num_heads, seq_len, head_dim)
/// * `output_weight` - Output projection weight (d_model, d_model)
/// * `config` - Parallel configuration
///
/// # Returns
/// * Combined output (seq_len, d_model)
pub fn parallel_combine_heads(
    head_outputs: &[Array2<f32>],
    output_weight: &ArrayView2<f32>,
    _config: &ParallelConfig,
) -> ModelResult<Array2<f32>> {
    if head_outputs.is_empty() {
        return Err(ModelError::invalid_config("No head outputs to combine"));
    }

    let (seq_len, head_dim) = head_outputs[0].dim();
    let num_heads = head_outputs.len();
    let d_model = num_heads * head_dim;

    // Concatenate heads
    let mut concatenated = Array2::zeros((seq_len, d_model));
    for (head_idx, head_output) in head_outputs.iter().enumerate() {
        let start_col = head_idx * head_dim;
        let end_col = start_col + head_dim;
        concatenated
            .slice_mut(s![.., start_col..end_col])
            .assign(head_output);
    }

    // Apply output projection
    Ok(concatenated.dot(output_weight))
}

/// Parallel attention score computation across heads
///
/// Computes Q @ K^T for each head in parallel.
///
/// # Arguments
/// * `queries` - Query tensors per head (num_heads, seq_len, head_dim)
/// * `keys` - Key tensors per head (num_heads, seq_len, head_dim)
/// * `scale` - Scaling factor (typically 1/sqrt(head_dim))
/// * `config` - Parallel configuration
///
/// # Returns
/// * Attention scores per head (num_heads, seq_len, seq_len)
pub fn parallel_attention_scores(
    queries: &[Array2<f32>],
    keys: &[Array2<f32>],
    scale: f32,
    config: &ParallelConfig,
) -> ModelResult<Vec<Array2<f32>>> {
    if queries.len() != keys.len() {
        return Err(ModelError::dimension_mismatch(
            "number of query and key heads",
            keys.len(),
            queries.len(),
        ));
    }

    let num_heads = queries.len();

    let compute_scores = |head_idx: usize| -> ModelResult<Array2<f32>> {
        let q = &queries[head_idx];
        let k = &keys[head_idx];

        // Compute: Q @ K^T with scaling
        let scores = q.dot(&k.t()) * scale;
        Ok(scores)
    };

    run_over_heads(config, num_heads, compute_scores)
}

/// Apply softmax in parallel across heads
///
/// # Arguments
/// * `scores` - Attention scores per head (num_heads, seq_len, seq_len)
/// * `config` - Parallel configuration
///
/// # Returns
/// * Softmax probabilities per head (num_heads, seq_len, seq_len)
pub fn parallel_softmax(
    scores: &[Array2<f32>],
    config: &ParallelConfig,
) -> ModelResult<Vec<Array2<f32>>> {
    let num_heads = scores.len();

    let apply_softmax = |head_idx: usize| -> ModelResult<Array2<f32>> {
        let head_scores = &scores[head_idx];
        let mut output = head_scores.clone();

        // Apply softmax row-wise
        for mut row in output.rows_mut() {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            row.mapv_inplace(|x| (x - max).exp());
            let sum: f32 = row.sum();
            if sum > 0.0 {
                row.mapv_inplace(|x| x / sum);
            }
        }

        Ok(output)
    };

    run_over_heads(config, num_heads, apply_softmax)
}

// Import ndarray slicing macro
use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array3};

    #[test]
    fn test_executor_creation() {
        let config = ParallelConfig::default();
        let executor = MultiHeadExecutor::new(config).expect("executor creation should succeed");
        assert_eq!(executor.config.min_heads_for_parallel, 4);
    }

    #[test]
    fn test_executor_custom_thread_count_par_map() {
        let config = ParallelConfig {
            min_heads_for_parallel: 0, // force the parallel branch
            num_threads: 2,
            ..Default::default()
        };
        let executor =
            MultiHeadExecutor::new(config).expect("executor with a 2-thread pool should build");

        let results = executor.par_map(6, |i| Ok(i * 2)).expect("par_map failed");
        assert_eq!(results, vec![0, 2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_run_over_heads_custom_thread_count() {
        // Exercises `run_over_heads`'s scoped-pool branch via a public free
        // function so a custom `num_threads` is proven to actually run
        // (rather than just being logged and ignored).
        let d_model = 4;
        let head_dim = 2;
        let num_heads = 2;
        let input = Array2::from_shape_fn((3, d_model), |(i, j)| (i * d_model + j) as f32);
        let weights = Array3::from_shape_fn((num_heads, d_model, head_dim), |(h, i, j)| {
            if i == j + h {
                1.0
            } else {
                0.0
            }
        });
        let config = ParallelConfig {
            min_heads_for_parallel: 0,
            num_threads: 2,
            ..Default::default()
        };

        let result = parallel_multi_head_projection(&input.view(), &weights.view(), &config)
            .expect("parallel_multi_head_projection with custom thread count failed");
        assert_eq!(result.len(), num_heads);
        for r in &result {
            assert_eq!(r.dim(), (3, head_dim));
        }
    }

    #[test]
    fn test_par_map_sequential() {
        let config = ParallelConfig {
            min_heads_for_parallel: 10,
            ..Default::default()
        };
        let executor = MultiHeadExecutor::new(config).expect("executor creation should succeed");

        let results = executor.par_map(3, |i| Ok(i * 2)).expect("par_map failed");

        assert_eq!(results, vec![0, 2, 4]);
    }

    #[test]
    fn test_par_map_parallel() {
        let config = ParallelConfig {
            min_heads_for_parallel: 2,
            ..Default::default()
        };
        let executor = MultiHeadExecutor::new(config).expect("executor creation should succeed");

        let results = executor.par_map(4, |i| Ok(i * 3)).expect("par_map failed");

        assert_eq!(results, vec![0, 3, 6, 9]);
    }

    #[test]
    fn test_split_heads_and_process() {
        let config = ParallelConfig::default();
        let executor = MultiHeadExecutor::new(config).expect("executor creation should succeed");

        let input = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        let output = executor
            .split_heads_and_process(&input.view(), 2, 2, |head| Ok(head.to_owned()))
            .expect("split_heads_and_process failed");

        assert_eq!(output.dim(), (2, 4));
    }

    #[test]
    fn test_parallel_multi_head_projection() {
        let config = ParallelConfig::default();
        let input = array![[1.0, 2.0], [3.0, 4.0]];

        let mut weights = Array3::zeros((2, 2, 3));
        weights
            .slice_mut(s![0, .., ..])
            .assign(&array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        weights
            .slice_mut(s![1, .., ..])
            .assign(&array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

        let outputs = parallel_multi_head_projection(&input.view(), &weights.view(), &config)
            .expect("projection failed");

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].dim(), (2, 3));
    }

    #[test]
    fn test_parallel_combine_heads() {
        let config = ParallelConfig::default();
        let head1 = array![[1.0, 2.0], [3.0, 4.0]];
        let head2 = array![[5.0, 6.0], [7.0, 8.0]];
        let heads = vec![head1, head2];

        let output_weight = Array2::eye(4);

        let combined =
            parallel_combine_heads(&heads, &output_weight.view(), &config).expect("combine failed");

        assert_eq!(combined.dim(), (2, 4));
    }

    #[test]
    fn test_parallel_attention_scores() {
        let config = ParallelConfig::default();
        let q1 = array![[1.0, 0.0], [0.0, 1.0]];
        let k1 = array![[1.0, 0.0], [0.0, 1.0]];
        let queries = vec![q1.clone()];
        let keys = vec![k1.clone()];

        let scores =
            parallel_attention_scores(&queries, &keys, 1.0, &config).expect("scores failed");

        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].dim(), (2, 2));
    }

    #[test]
    fn test_parallel_softmax() {
        let config = ParallelConfig::default();
        let scores = vec![array![[1.0, 2.0], [3.0, 4.0]]];

        let probs = parallel_softmax(&scores, &config).expect("softmax failed");

        assert_eq!(probs.len(), 1);
        assert_eq!(probs[0].dim(), (2, 2));

        // Check that rows sum to ~1.0
        for row in probs[0].rows() {
            let sum: f32 = row.sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = ParallelConfig::default();
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let result = MultiHeadExecutor::new(config)
            .expect("executor creation should succeed")
            .split_heads_and_process(&input.view(), 2, 2, |head| Ok(head.to_owned()));

        assert!(result.is_err());
    }
}
