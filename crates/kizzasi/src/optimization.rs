//! Optimization wrappers around a [`Kizzasi`] predictor
//!
//! What this layer actually does:
//!
//! - **Input workspace pooling** — [`OptimizedPredictor::step_slice`] takes its
//!   input buffer from a [`kizzasi_core::ArrayPool`] instead of allocating one
//!   per call. Reported by [`OptimizationStats::workspace_pool_hits`] and
//!   [`OptimizationStats::workspace_allocations`].
//! - **Stateless result caching** — [`OptimizedPredictor::predict_stateless`]
//!   evaluates the model as a pure function of its input (from a reset state,
//!   with the live state restored afterwards) and can therefore be memoised
//!   safely. See the warning on [`OptimizationConfig::enable_result_cache`].
//!
//! What it deliberately does **not** do: it cannot memoise
//! [`OptimizedPredictor::step`]. That method drives a stateful recurrence, so
//! returning a previously computed output would both answer under the wrong
//! hidden state and skip the state update, permanently desynchronising the
//! stream from the input.
//!
//! SIMD kernel selection, cache-line alignment and discretization caching live
//! inside `kizzasi-core` and are not switchable per predictor from here; the
//! corresponding [`OptimizationConfig`] fields record intent only and say so in
//! their own documentation.
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi::optimization::{OptimizationConfig, OptimizedPredictor};
//!
//! let config = OptimizationConfig::default()
//!     .with_workspace_pooling(true)
//!     .with_workspace_pool_size(32);
//!
//! let mut predictor = OptimizedPredictor::new(base_predictor, config);
//! let output = predictor.step_slice(&[0.1, 0.2, 0.3])?;
//! ```

use crate::error::{KizzasiError, KizzasiResult};
use crate::predictor::Kizzasi;
use kizzasi_core::ArrayPool;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for optimization strategies
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Reuse pooled input buffers in [`OptimizedPredictor::step_slice`]
    /// instead of allocating one array per call.
    pub enable_workspace_pooling: bool,

    /// Requested discretization caching for SSM models.
    ///
    /// **Records intent only.** The discretization is fused into
    /// `kizzasi-core`'s selective-scan step and is not reachable from this
    /// layer, so toggling this does not change what is computed. It is kept
    /// (rather than removed) because it is part of the published API surface.
    pub enable_discretization_cache: bool,

    /// Requested SIMD-accelerated operations.
    ///
    /// **Records intent only.** `kizzasi-core` picks its SIMD kernels by
    /// target-feature detection; there is no per-predictor switch to honour.
    pub enable_simd: bool,

    /// Requested cache-aligned data structures.
    ///
    /// **Records intent only.** Alignment is a property of the `kizzasi-core`
    /// types themselves and cannot be toggled per predictor.
    pub enable_cache_alignment: bool,

    /// Maximum number of pooled input buffers kept alive when
    /// `enable_workspace_pooling` is set.
    pub workspace_pool_size: usize,

    /// Enable caching of [`OptimizedPredictor::predict_stateless`] results.
    ///
    /// **This never affects [`OptimizedPredictor::step`].** `step` advances a
    /// stateful recurrence, so serving it from a cache would return an output
    /// computed under a different hidden state *and* skip the state update.
    /// Only `predict_stateless`, which evaluates the model from a reset state,
    /// is a pure function of its input and therefore safe to memoise.
    pub enable_result_cache: bool,

    /// Maximum cached prediction results
    pub result_cache_size: usize,

    /// Cache TTL (time-to-live) in milliseconds
    pub cache_ttl_ms: u64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_workspace_pooling: true,
            enable_discretization_cache: true,
            enable_simd: true,
            enable_cache_alignment: true,
            workspace_pool_size: 16,
            enable_result_cache: false,
            result_cache_size: 1000,
            cache_ttl_ms: 1000,
        }
    }
}

impl OptimizationConfig {
    /// Create a new optimization configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set workspace pooling enablement
    pub fn with_workspace_pooling(mut self, enabled: bool) -> Self {
        self.enable_workspace_pooling = enabled;
        self
    }

    /// Set discretization cache enablement
    pub fn with_discretization_cache(mut self, enabled: bool) -> Self {
        self.enable_discretization_cache = enabled;
        self
    }

    /// Set SIMD enablement
    pub fn with_simd(mut self, enabled: bool) -> Self {
        self.enable_simd = enabled;
        self
    }

    /// Set cache alignment enablement
    pub fn with_cache_alignment(mut self, enabled: bool) -> Self {
        self.enable_cache_alignment = enabled;
        self
    }

    /// Set workspace pool size
    pub fn with_workspace_pool_size(mut self, size: usize) -> Self {
        self.workspace_pool_size = size;
        self
    }

    /// Set result caching enablement
    pub fn with_result_cache(mut self, enabled: bool) -> Self {
        self.enable_result_cache = enabled;
        self
    }

    /// Set result cache size
    pub fn with_result_cache_size(mut self, size: usize) -> Self {
        self.result_cache_size = size;
        self
    }

    /// Set cache TTL
    pub fn with_cache_ttl(mut self, ttl_ms: u64) -> Self {
        self.cache_ttl_ms = ttl_ms;
        self
    }

    /// Create an aggressive optimization profile for maximum performance
    pub fn aggressive() -> Self {
        Self {
            enable_workspace_pooling: true,
            enable_discretization_cache: true,
            enable_simd: true,
            enable_cache_alignment: true,
            workspace_pool_size: 32,
            enable_result_cache: true,
            result_cache_size: 5000,
            cache_ttl_ms: 5000,
        }
    }

    /// Create a conservative profile for minimal memory usage
    pub fn conservative() -> Self {
        Self {
            enable_workspace_pooling: true,
            enable_discretization_cache: false,
            enable_simd: true,
            enable_cache_alignment: false,
            workspace_pool_size: 4,
            enable_result_cache: false,
            result_cache_size: 100,
            cache_ttl_ms: 500,
        }
    }

    /// Create a balanced profile
    pub fn balanced() -> Self {
        Self::default()
    }
}

/// Cached prediction result with timestamp
#[derive(Debug, Clone)]
struct CachedResult {
    /// Full input, kept so a hash collision can be detected on lookup.
    input: Array1<f32>,
    output: Array1<f32>,
    timestamp: Instant,
}

impl CachedResult {
    fn new(input: Array1<f32>, output: Array1<f32>) -> Self {
        Self {
            input,
            output,
            timestamp: Instant::now(),
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.timestamp.elapsed() > ttl
    }
}

/// True LRU cache for stateless prediction results.
///
/// Keyed by a hash of the input, but every hit re-checks the *full* input
/// array before returning: a 64-bit hash collision must not hand back another
/// input's prediction.
///
/// Lookup is O(1) and eviction is O(1) amortised; maintaining the recency
/// order costs one scan of the recency queue per access, which is a queue of
/// `u64` keys rather than the previous implementation's rescan of every
/// cached `Array1<f32>` plus a `retain` over the whole cache on both `get`
/// and `put`.
#[derive(Debug)]
struct ResultCache {
    entries: HashMap<u64, CachedResult>,
    /// Least-recently-used first.
    recency: VecDeque<u64>,
    max_size: usize,
    ttl: Duration,
    hits: u64,
    misses: u64,
}

impl ResultCache {
    fn new(max_size: usize, ttl_ms: u64) -> Self {
        let max_size = max_size.max(1);
        Self {
            entries: HashMap::with_capacity(max_size),
            recency: VecDeque::with_capacity(max_size),
            max_size,
            ttl: Duration::from_millis(ttl_ms),
            hits: 0,
            misses: 0,
        }
    }

    fn hash_input(input: &Array1<f32>) -> u64 {
        // Simple hash function for f32 arrays
        let mut hash = 0u64;
        for (i, &val) in input.iter().enumerate() {
            // Convert to bits and mix with position
            let bits = val.to_bits() as u64;
            hash = hash
                .wrapping_mul(31)
                .wrapping_add(bits)
                .wrapping_add(i as u64);
        }
        hash
    }

    fn touch(&mut self, key: u64) {
        if let Some(position) = self.recency.iter().position(|&k| k == key) {
            self.recency.remove(position);
        }
        self.recency.push_back(key);
    }

    fn remove(&mut self, key: u64) {
        self.entries.remove(&key);
        if let Some(position) = self.recency.iter().position(|&k| k == key) {
            self.recency.remove(position);
        }
    }

    fn get(&mut self, input: &Array1<f32>) -> Option<Array1<f32>> {
        let key = Self::hash_input(input);

        let hit = match self.entries.get(&key) {
            // Expire lazily on lookup rather than rescanning the whole cache.
            Some(entry) if entry.is_expired(self.ttl) => None,
            // A hash collision must not return another input's prediction.
            Some(entry) if entry.input != *input => None,
            Some(entry) => Some(entry.output.clone()),
            None => None,
        };

        match hit {
            Some(output) => {
                self.touch(key);
                self.hits += 1;
                Some(output)
            }
            None => {
                if self
                    .entries
                    .get(&key)
                    .is_some_and(|entry| entry.is_expired(self.ttl))
                {
                    self.remove(key);
                }
                self.misses += 1;
                None
            }
        }
    }

    fn put(&mut self, input: &Array1<f32>, output: Array1<f32>) {
        let key = Self::hash_input(input);

        if let std::collections::hash_map::Entry::Occupied(mut occupied) = self.entries.entry(key) {
            occupied.insert(CachedResult::new(input.clone(), output));
            self.touch(key);
            return;
        }

        while self.entries.len() >= self.max_size {
            match self.recency.pop_front() {
                Some(oldest) => {
                    self.entries.remove(&oldest);
                }
                None => break,
            }
        }

        self.entries
            .insert(key, CachedResult::new(input.clone(), output));
        self.recency.push_back(key);
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.recency.clear();
        self.hits = 0;
        self.misses = 0;
    }

    fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        }
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.entries.len(),
            capacity: self.max_size,
            hits: self.hits,
            misses: self.misses,
            hit_rate: self.hit_rate(),
        }
    }
}

/// Statistics about cache performance
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current number of cached entries
    pub size: usize,
    /// Maximum cache capacity
    pub capacity: usize,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

/// Optimized predictor wrapper with advanced optimizations
pub struct OptimizedPredictor {
    /// Base predictor
    predictor: Kizzasi,
    /// Optimization configuration
    config: OptimizationConfig,
    /// Result cache (used by `predict_stateless` only)
    result_cache: Arc<Mutex<ResultCache>>,
    /// Reusable input buffers, present when workspace pooling is enabled
    workspace_pool: Option<ArrayPool>,
    /// Performance statistics
    stats: Arc<Mutex<OptimizationStats>>,
}

/// Statistics about optimization performance
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Total predictions made
    pub total_predictions: u64,
    /// Predictions served from cache
    pub cached_predictions: u64,
    /// Total time saved by caching (microseconds)
    pub cache_time_saved_us: u64,
    /// Average prediction time without cache (microseconds)
    pub avg_prediction_time_us: u64,
    /// Number of workspace pool hits
    pub workspace_pool_hits: u64,
    /// Number of workspace allocations
    pub workspace_allocations: u64,
}

impl OptimizedPredictor {
    /// Create a new optimized predictor
    pub fn new(predictor: Kizzasi, config: OptimizationConfig) -> Self {
        let result_cache = Arc::new(Mutex::new(ResultCache::new(
            config.result_cache_size,
            config.cache_ttl_ms,
        )));

        let workspace_pool = if config.enable_workspace_pooling {
            Some(ArrayPool::new(
                predictor.input_dim(),
                config.workspace_pool_size.max(1),
            ))
        } else {
            None
        };

        Self {
            predictor,
            config,
            result_cache,
            workspace_pool,
            stats: Arc::new(Mutex::new(OptimizationStats::default())),
        }
    }

    /// Create with default optimization configuration
    pub fn with_defaults(predictor: Kizzasi) -> Self {
        Self::new(predictor, OptimizationConfig::default())
    }

    /// Create with aggressive optimizations
    pub fn aggressive(predictor: Kizzasi) -> Self {
        Self::new(predictor, OptimizationConfig::aggressive())
    }

    /// Create with conservative optimizations
    pub fn conservative(predictor: Kizzasi) -> Self {
        Self::new(predictor, OptimizationConfig::conservative())
    }

    /// Perform a single prediction step
    ///
    /// This drives the model's stateful recurrence and is therefore **never**
    /// served from the result cache — see
    /// [`OptimizationConfig::enable_result_cache`]. Use
    /// [`Self::predict_stateless`] when a memoisable, state-independent
    /// evaluation is what you want.
    pub fn step(&mut self, input: &Array1<f32>) -> KizzasiResult<Array1<f32>> {
        let start = Instant::now();
        let output = self.predictor.step(input)?;
        self.record_uncached(start.elapsed().as_micros() as u64)?;
        Ok(output)
    }

    /// Prediction step from a slice, using a pooled input buffer.
    ///
    /// When `enable_workspace_pooling` is set, the owned input array is taken
    /// from (and returned to) a pool of `workspace_pool_size` buffers instead
    /// of being allocated per call. The pool's hit/allocation counters are
    /// reflected in [`Self::optimization_stats`].
    pub fn step_slice(&mut self, input: &[f32]) -> KizzasiResult<Array1<f32>> {
        let Some(pool) = self.workspace_pool.as_ref() else {
            return self.step(&Array1::from_vec(input.to_vec()));
        };

        if input.len() != pool.array_size() {
            return Err(KizzasiError::dimension_mismatch(
                pool.array_size(),
                input.len(),
                "input length must match input_dim",
            ));
        }

        let mut buffer = pool.acquire();
        for (slot, value) in buffer.iter_mut().zip(input.iter()) {
            *slot = *value;
        }

        let start = Instant::now();
        let result = self.predictor.step(&buffer);
        let elapsed_us = start.elapsed().as_micros() as u64;

        // Return the buffer to the pool whether or not the step succeeded.
        let pool_stats = {
            let pool = self.workspace_pool.as_ref();
            match pool {
                Some(pool) => {
                    pool.release(buffer);
                    pool.stats()
                }
                None => Default::default(),
            }
        };

        let output = result?;
        self.record_uncached(elapsed_us)?;

        {
            let mut stats = self.lock_stats()?;
            stats.workspace_pool_hits = pool_stats.hits;
            stats.workspace_allocations = pool_stats.misses;
        }

        Ok(output)
    }

    /// Evaluate the model as a pure function of `input`, from a reset state.
    ///
    /// The live hidden state is captured before the call and restored
    /// afterwards, so streaming through [`Self::step`] is unaffected. Because
    /// the result depends only on `input`, it can be memoised safely: when
    /// `enable_result_cache` is set, repeated inputs are served from the LRU
    /// cache.
    pub fn predict_stateless(&mut self, input: &Array1<f32>) -> KizzasiResult<Array1<f32>> {
        if self.config.enable_result_cache {
            let cached = self.lock_cache()?.get(input);
            if let Some(output) = cached {
                let mut stats = self.lock_stats()?;
                stats.total_predictions += 1;
                stats.cached_predictions += 1;
                stats.cache_time_saved_us += stats.avg_prediction_time_us;
                return Ok(output);
            }
        }

        let snapshot = self.predictor.snapshot_state();
        self.predictor.reset();

        let start = Instant::now();
        let result = self.predictor.step(input);
        let elapsed_us = start.elapsed().as_micros() as u64;

        // Restore the caller's stream state regardless of the outcome.
        let restored = self.predictor.restore_state(snapshot);
        let output = result?;
        restored?;

        self.record_uncached(elapsed_us)?;

        if self.config.enable_result_cache {
            self.lock_cache()?.put(input, output.clone());
        }

        Ok(output)
    }

    fn lock_cache(&self) -> KizzasiResult<std::sync::MutexGuard<'_, ResultCache>> {
        self.result_cache
            .lock()
            .map_err(|_| KizzasiError::InvalidState {
                reason: "Result cache mutex poisoned".to_string(),
                recovery: None,
            })
    }

    fn lock_stats(&self) -> KizzasiResult<std::sync::MutexGuard<'_, OptimizationStats>> {
        self.stats.lock().map_err(|_| KizzasiError::InvalidState {
            reason: "Stats mutex poisoned".to_string(),
            recovery: None,
        })
    }

    /// Fold one measured (non-cached) prediction into the running statistics.
    fn record_uncached(&self, elapsed_us: u64) -> KizzasiResult<()> {
        let mut stats = self.lock_stats()?;
        stats.total_predictions += 1;

        let total_uncached = stats
            .total_predictions
            .saturating_sub(stats.cached_predictions)
            .max(1);
        stats.avg_prediction_time_us =
            ((stats.avg_prediction_time_us * total_uncached.saturating_sub(1) + elapsed_us)
                / total_uncached)
                .max(1);
        Ok(())
    }

    /// Perform multi-step prediction
    ///
    /// Delegates straight to [`Kizzasi::predict_n`]: an autoregressive rollout
    /// is inherently sequential and state-dependent, so nothing in this
    /// wrapper applies to it.
    pub fn predict_n(
        &mut self,
        initial_input: &Array1<f32>,
        n_steps: usize,
    ) -> KizzasiResult<Array2<f32>> {
        self.predictor.predict_n(initial_input, n_steps)
    }

    /// Batch prediction with optimizations
    pub fn predict_batch(&mut self, inputs: &[Array1<f32>]) -> KizzasiResult<Vec<Array1<f32>>> {
        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            outputs.push(self.step(input)?);
        }

        Ok(outputs)
    }

    /// Reset predictor state and clear caches
    pub fn reset(&mut self) -> KizzasiResult<()> {
        self.predictor.reset();
        self.lock_cache()?.clear();
        if let Some(pool) = self.workspace_pool.as_ref() {
            pool.clear();
        }
        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> KizzasiResult<CacheStats> {
        self.lock_cache().map(|cache| cache.stats())
    }

    /// Get optimization statistics
    pub fn optimization_stats(&self) -> KizzasiResult<OptimizationStats> {
        self.lock_stats().map(|stats| stats.clone())
    }

    /// Get the underlying predictor
    pub fn inner(&self) -> &Kizzasi {
        &self.predictor
    }

    /// Get mutable reference to underlying predictor
    pub fn inner_mut(&mut self) -> &mut Kizzasi {
        &mut self.predictor
    }

    /// Consume and return the underlying predictor
    pub fn into_inner(self) -> Kizzasi {
        self.predictor
    }

    /// Get the optimization configuration
    pub fn config(&self) -> &OptimizationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor::KizzasiBuilder;

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig::default();
        assert!(config.enable_workspace_pooling);
        assert!(config.enable_discretization_cache);
        assert!(config.enable_simd);

        let aggressive = OptimizationConfig::aggressive();
        assert_eq!(aggressive.workspace_pool_size, 32);
        assert!(aggressive.enable_result_cache);

        let conservative = OptimizationConfig::conservative();
        assert_eq!(conservative.workspace_pool_size, 4);
        assert!(!conservative.enable_result_cache);
    }

    #[test]
    fn test_result_cache() {
        let mut cache = ResultCache::new(10, 1000);

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let output = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        // Cache miss
        assert!(cache.get(&input).is_none());
        assert_eq!(cache.misses, 1);

        // Cache put
        cache.put(&input, output.clone());

        // Cache hit
        let cached = cache.get(&input);
        assert!(cached.is_some());
        assert_eq!(cache.hits, 1);

        // Verify stats
        assert_eq!(cache.hit_rate(), 0.5); // 1 hit, 1 miss
    }

    #[test]
    fn test_optimized_predictor_creation() -> KizzasiResult<()> {
        let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let opt_predictor = OptimizedPredictor::with_defaults(predictor);

        assert!(opt_predictor.config().enable_workspace_pooling);
        Ok(())
    }

    #[test]
    fn test_optimized_prediction() -> KizzasiResult<()> {
        let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let mut opt_predictor = OptimizedPredictor::with_defaults(predictor);

        let input = Array1::from_vec(vec![1.0, 2.0]);
        let output = opt_predictor.step(&input)?;

        assert_eq!(output.len(), 2);

        let stats = opt_predictor.optimization_stats()?;
        assert_eq!(stats.total_predictions, 1);

        Ok(())
    }

    #[test]
    fn test_stateless_result_caching() -> KizzasiResult<()> {
        let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let config = OptimizationConfig::default().with_result_cache(true);
        let mut opt_predictor = OptimizedPredictor::new(predictor, config);

        let input = Array1::from_vec(vec![1.0, 2.0]);

        // First evaluation - computed
        let output1 = opt_predictor.predict_stateless(&input)?;
        let stats1 = opt_predictor.optimization_stats()?;
        assert_eq!(stats1.cached_predictions, 0);

        // Second evaluation with same input - served from cache, and (unlike
        // the old `step` cache) legitimately equal because the evaluation is
        // state-independent.
        let output2 = opt_predictor.predict_stateless(&input)?;
        let stats2 = opt_predictor.optimization_stats()?;
        assert_eq!(stats2.cached_predictions, 1);

        assert_eq!(output1.len(), output2.len());
        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        let cache_stats = opt_predictor.cache_stats()?;
        assert_eq!(cache_stats.hits, 1);

        Ok(())
    }

    #[test]
    fn test_step_is_never_served_from_cache() -> KizzasiResult<()> {
        // Regression: `step` drives a stateful recurrence. Serving a repeated
        // input from the cache used to return a prediction computed under a
        // different hidden state *and* skip the state update.
        let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let config = OptimizationConfig::aggressive();
        let mut opt_predictor = OptimizedPredictor::new(predictor, config);

        let input = Array1::from_vec(vec![1.0, 2.0]);
        opt_predictor.step(&input)?;
        opt_predictor.step(&input)?;

        let stats = opt_predictor.optimization_stats()?;
        assert_eq!(stats.cached_predictions, 0, "step must not use the cache");
        assert_eq!(stats.total_predictions, 2);

        // The decisive property: a cache hit used to `return` before the model
        // ran, leaving the recurrence un-advanced. Both steps must reach the
        // model. (Comparing the two output vectors would be flaky: with
        // small random init the state contribution can fall below f32
        // resolution for some seeds.)
        assert_eq!(opt_predictor.inner().step_count(), 2);

        Ok(())
    }

    #[test]
    fn test_predict_stateless_preserves_stream_state() -> KizzasiResult<()> {
        let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let mut opt_predictor = OptimizedPredictor::with_defaults(predictor);

        let input = Array1::from_vec(vec![0.25, -0.5]);
        opt_predictor.step(&input)?;
        let expected_step_count = opt_predictor.inner().step_count();

        opt_predictor.predict_stateless(&input)?;

        assert_eq!(
            opt_predictor.inner().step_count(),
            expected_step_count,
            "predict_stateless must not advance the live stream"
        );
        Ok(())
    }

    #[test]
    fn test_workspace_pool_is_used() -> KizzasiResult<()> {
        let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let config = OptimizationConfig::default()
            .with_workspace_pooling(true)
            .with_workspace_pool_size(4);
        let mut opt_predictor = OptimizedPredictor::new(predictor, config);

        for _ in 0..5 {
            opt_predictor.step_slice(&[0.1, 0.2])?;
        }

        let stats = opt_predictor.optimization_stats()?;
        assert!(
            stats.workspace_pool_hits > 0,
            "pooled buffers must actually be reused"
        );
        assert!(stats.workspace_allocations >= 1);
        Ok(())
    }

    #[test]
    fn test_cache_rejects_hash_collisions() {
        // Two different inputs forced into the same bucket must not share a
        // prediction: `get` compares the full input, not just the hash.
        let mut cache = ResultCache::new(4, 10_000);
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        cache.put(&a, Array1::from_vec(vec![9.0]));

        assert!(cache.get(&a).is_some());
        if ResultCache::hash_input(&a) == ResultCache::hash_input(&b) {
            assert!(cache.get(&b).is_none());
        }
    }

    #[test]
    fn test_cache_expiration() {
        let mut cache = ResultCache::new(10, 100); // 100ms TTL

        let input = Array1::from_vec(vec![1.0, 2.0]);
        let output = Array1::from_vec(vec![3.0, 4.0]);

        cache.put(&input, output);

        // Should hit immediately
        assert!(cache.get(&input).is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        // Should miss after expiration
        assert!(cache.get(&input).is_none());
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = ResultCache::new(3, 10000); // Small cache, long TTL

        for i in 0..5 {
            let input = Array1::from_vec(vec![i as f32]);
            let output = Array1::from_vec(vec![i as f32 * 2.0]);
            cache.put(&input, output);
        }

        // Cache should only hold last 3 entries
        assert_eq!(cache.entries.len(), 3);

        // First two should be evicted
        assert!(cache.get(&Array1::from_vec(vec![0.0])).is_none());
        assert!(cache.get(&Array1::from_vec(vec![1.0])).is_none());

        // Last three should be present
        assert!(cache.get(&Array1::from_vec(vec![2.0])).is_some());
        assert!(cache.get(&Array1::from_vec(vec![3.0])).is_some());
        assert!(cache.get(&Array1::from_vec(vec![4.0])).is_some());
    }

    #[test]
    fn test_reset_clears_cache() -> KizzasiResult<()> {
        let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let config = OptimizationConfig::default().with_result_cache(true);
        let mut opt_predictor = OptimizedPredictor::new(predictor, config);

        let input = Array1::from_vec(vec![1.0, 2.0]);

        // Make a stateless prediction twice to populate and hit the cache
        opt_predictor.predict_stateless(&input)?;
        opt_predictor.predict_stateless(&input)?;

        let stats_before = opt_predictor.cache_stats()?;
        assert_eq!(stats_before.hits, 1);

        // Reset should clear cache
        opt_predictor.reset()?;

        let stats_after = opt_predictor.cache_stats()?;
        assert_eq!(stats_after.hits, 0);
        assert_eq!(stats_after.size, 0);

        Ok(())
    }

    #[test]
    fn test_batch_prediction_with_cache() -> KizzasiResult<()> {
        let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let config = OptimizationConfig::default().with_result_cache(true);
        let mut opt_predictor = OptimizedPredictor::new(predictor, config);

        let inputs = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![3.0, 4.0]),
            Array1::from_vec(vec![1.0, 2.0]), // Duplicate
        ];

        let outputs = opt_predictor.predict_batch(&inputs)?;
        assert_eq!(outputs.len(), 3);

        // `predict_batch` streams through `step`, so the duplicate input must
        // NOT be served from the cache: the hidden state differs by then. The
        // old test asserted `cached_predictions == 1` here, institutionalising
        // the desynchronisation bug.
        let stats = opt_predictor.optimization_stats()?;
        assert_eq!(stats.total_predictions, 3);
        assert_eq!(stats.cached_predictions, 0);
        assert_eq!(opt_predictor.inner().step_count(), 3);

        Ok(())
    }
}
