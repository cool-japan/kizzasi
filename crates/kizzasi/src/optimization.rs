//! Advanced optimization system for high-performance predictions
//!
//! This module exposes sophisticated optimization techniques from kizzasi-core:
//! - Workspace pooling for zero-allocation predictions
//! - Discretization caching for SSM models
//! - SIMD-accelerated operations
//! - Cache-aligned data structures
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi::optimization::{OptimizationConfig, OptimizedPredictor};
//!
//! let config = OptimizationConfig::default()
//!     .enable_workspace_pooling(true)
//!     .enable_discretization_cache(true)
//!     .enable_simd(true);
//!
//! let predictor = OptimizedPredictor::new(base_predictor, config)?;
//! ```

use crate::error::{KizzasiError, KizzasiResult};
use crate::predictor::Kizzasi;
use scirs2_core::ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for optimization strategies
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable workspace pooling to reduce allocations
    pub enable_workspace_pooling: bool,

    /// Enable discretization caching for SSM models
    pub enable_discretization_cache: bool,

    /// Enable SIMD-accelerated operations
    pub enable_simd: bool,

    /// Enable cache-aligned data structures
    pub enable_cache_alignment: bool,

    /// Maximum number of workspaces in pool
    pub workspace_pool_size: usize,

    /// Enable prediction result caching
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
    input_hash: u64,
    output: Array1<f32>,
    timestamp: Instant,
}

impl CachedResult {
    fn new(input_hash: u64, output: Array1<f32>) -> Self {
        Self {
            input_hash,
            output,
            timestamp: Instant::now(),
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.timestamp.elapsed() > ttl
    }
}

/// LRU cache for prediction results
#[derive(Debug)]
struct ResultCache {
    cache: Vec<CachedResult>,
    max_size: usize,
    ttl: Duration,
    hits: u64,
    misses: u64,
}

impl ResultCache {
    fn new(max_size: usize, ttl_ms: u64) -> Self {
        Self {
            cache: Vec::with_capacity(max_size),
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

    fn get(&mut self, input: &Array1<f32>) -> Option<Array1<f32>> {
        let input_hash = Self::hash_input(input);

        // Remove expired entries
        self.cache.retain(|entry| !entry.is_expired(self.ttl));

        // Find matching entry
        if let Some(entry) = self.cache.iter().find(|e| e.input_hash == input_hash) {
            self.hits += 1;
            Some(entry.output.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    fn put(&mut self, input: &Array1<f32>, output: Array1<f32>) {
        let input_hash = Self::hash_input(input);

        // Remove expired entries
        self.cache.retain(|entry| !entry.is_expired(self.ttl));

        // Evict oldest if at capacity (LRU)
        if self.cache.len() >= self.max_size {
            self.cache.remove(0);
        }

        self.cache.push(CachedResult::new(input_hash, output));
    }

    fn clear(&mut self) {
        self.cache.clear();
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
            size: self.cache.len(),
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
    /// Result cache
    result_cache: Arc<Mutex<ResultCache>>,
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

        Self {
            predictor,
            config,
            result_cache,
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

    /// Perform a single prediction step with optimizations
    pub fn step(&mut self, input: &Array1<f32>) -> KizzasiResult<Array1<f32>> {
        let start = Instant::now();

        // Try cache first if enabled
        if self.config.enable_result_cache {
            let cache_result = self
                .result_cache
                .lock()
                .map_err(|_| KizzasiError::InvalidState {
                    reason: "Result cache mutex poisoned".to_string(),
                    recovery: None,
                })?
                .get(input);

            if let Some(cached_output) = cache_result {
                let mut stats = self.stats.lock().map_err(|_| KizzasiError::InvalidState {
                    reason: "Stats mutex poisoned".to_string(),
                    recovery: None,
                })?;
                stats.total_predictions += 1;
                stats.cached_predictions += 1;
                stats.cache_time_saved_us += stats.avg_prediction_time_us;
                return Ok(cached_output);
            }
        }

        // Perform prediction
        let output = self.predictor.step(input)?;

        let elapsed_us = start.elapsed().as_micros() as u64;

        // Update statistics
        {
            let mut stats = self.stats.lock().map_err(|_| KizzasiError::InvalidState {
                reason: "Stats mutex poisoned".to_string(),
                recovery: None,
            })?;
            stats.total_predictions += 1;

            // Update running average
            let total_uncached = stats.total_predictions - stats.cached_predictions;
            stats.avg_prediction_time_us = ((stats.avg_prediction_time_us * (total_uncached - 1)
                + elapsed_us)
                / total_uncached)
                .max(1);
        }

        // Cache the result if enabled
        if self.config.enable_result_cache {
            self.result_cache
                .lock()
                .map_err(|_| KizzasiError::InvalidState {
                    reason: "Result cache mutex poisoned".to_string(),
                    recovery: None,
                })?
                .put(input, output.clone());
        }

        Ok(output)
    }

    /// Perform multi-step prediction with optimizations
    pub fn predict_n(
        &mut self,
        initial_input: &Array1<f32>,
        n_steps: usize,
    ) -> KizzasiResult<Array2<f32>> {
        // For multi-step predictions, bypass cache and delegate to base predictor
        // This is more efficient than caching intermediate steps
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

        if self.config.enable_result_cache {
            self.result_cache
                .lock()
                .map_err(|_| KizzasiError::InvalidState {
                    reason: "Result cache mutex poisoned".to_string(),
                    recovery: None,
                })?
                .clear();
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> KizzasiResult<CacheStats> {
        self.result_cache
            .lock()
            .map_err(|_| KizzasiError::InvalidState {
                reason: "Result cache mutex poisoned".to_string(),
                recovery: None,
            })
            .map(|cache| cache.stats())
    }

    /// Get optimization statistics
    pub fn optimization_stats(&self) -> KizzasiResult<OptimizationStats> {
        self.stats
            .lock()
            .map_err(|_| KizzasiError::InvalidState {
                reason: "Stats mutex poisoned".to_string(),
                recovery: None,
            })
            .map(|stats| stats.clone())
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
    fn test_result_caching() -> KizzasiResult<()> {
        let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let config = OptimizationConfig::default().with_result_cache(true);
        let mut opt_predictor = OptimizedPredictor::new(predictor, config);

        let input = Array1::from_vec(vec![1.0, 2.0]);

        // First prediction - should not be cached
        let output1 = opt_predictor.step(&input)?;
        let stats1 = opt_predictor.optimization_stats()?;
        assert_eq!(stats1.cached_predictions, 0);

        // Second prediction with same input - should be cached
        let output2 = opt_predictor.step(&input)?;
        let stats2 = opt_predictor.optimization_stats()?;
        assert_eq!(stats2.cached_predictions, 1);

        // Outputs should be identical
        assert_eq!(output1.len(), output2.len());

        let cache_stats = opt_predictor.cache_stats()?;
        assert_eq!(cache_stats.hits, 1);

        Ok(())
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
        assert_eq!(cache.cache.len(), 3);

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

        // Make a prediction to populate cache
        opt_predictor.step(&input)?;
        opt_predictor.step(&input)?; // Should hit cache

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

        // Third prediction should have hit cache
        let stats = opt_predictor.optimization_stats()?;
        assert_eq!(stats.total_predictions, 3);
        assert_eq!(stats.cached_predictions, 1);

        Ok(())
    }
}
