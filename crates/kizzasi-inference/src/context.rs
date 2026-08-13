//! Inference context management
//!
//! Manages the sliding window of past inputs and hidden states
//! for autoregressive prediction.

use crate::error::{InferenceError, InferenceResult};
use crate::pool::TensorPool;
use kizzasi_core::HiddenState;
use scirs2_core::ndarray::Array1;
use std::collections::VecDeque;

/// Configuration for inference context
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextConfig {
    /// Maximum context window size
    pub max_context: usize,
    /// Whether to store full history or just hidden states
    pub store_history: bool,
    /// Number of model layers (for state management)
    pub num_layers: usize,
    /// Hidden state dimension
    pub hidden_dim: usize,
    /// State dimension (for SSMs)
    pub state_dim: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context: 8192,
            store_history: false,
            num_layers: 4,
            hidden_dim: 256,
            state_dim: 16,
        }
    }
}

impl ContextConfig {
    /// Create a new context configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum context size
    pub fn max_context(mut self, size: usize) -> Self {
        self.max_context = size;
        self
    }

    /// Enable/disable history storage
    pub fn store_history(mut self, store: bool) -> Self {
        self.store_history = store;
        self
    }

    /// Set number of layers
    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }
}

/// Manages inference context including history and hidden states
pub struct InferenceContext {
    config: ContextConfig,
    /// History of past inputs (if store_history is true)
    history: VecDeque<Array1<f32>>,
    /// Hidden states for each layer
    states: Vec<HiddenState>,
    /// Number of steps processed
    step_count: usize,
    /// Optional memory pool for efficient allocation
    pool: Option<TensorPool>,
}

impl InferenceContext {
    /// Create a new inference context
    pub fn new(config: ContextConfig) -> Self {
        let states = (0..config.num_layers)
            .map(|_| HiddenState::new(config.hidden_dim, config.state_dim))
            .collect();

        Self {
            config,
            history: VecDeque::new(),
            states,
            step_count: 0,
            pool: None,
        }
    }

    /// Create a new inference context with memory pooling enabled
    pub fn with_pool(config: ContextConfig, pool: TensorPool) -> Self {
        let states = (0..config.num_layers)
            .map(|_| HiddenState::new(config.hidden_dim, config.state_dim))
            .collect();

        Self {
            config,
            history: VecDeque::new(),
            states,
            step_count: 0,
            pool: Some(pool),
        }
    }

    /// Get reference to the memory pool if available
    pub fn pool(&self) -> Option<&TensorPool> {
        self.pool.as_ref()
    }

    /// Enable memory pooling with specified pool
    pub fn enable_pooling(&mut self, pool: TensorPool) {
        self.pool = Some(pool);
    }

    /// Disable memory pooling
    pub fn disable_pooling(&mut self) {
        self.pool = None;
    }

    /// Reset the context to initial state
    pub fn reset(&mut self) {
        self.history.clear();
        for state in &mut self.states {
            state.reset();
        }
        self.step_count = 0;
    }

    /// Add an owned input to the history
    ///
    /// When `store_history` is disabled the array is dropped immediately and only
    /// the step counter advances. Callers holding a borrow should use
    /// [`InferenceContext::push_ref`] so that no copy is made on the default
    /// configuration (`store_history == false`).
    pub fn push(&mut self, input: Array1<f32>) {
        if self.config.store_history {
            self.evict_oldest_if_full();
            self.history.push_back(input);
        }
        self.step_count += 1;
    }

    /// Add a borrowed input to the history
    ///
    /// The input is cloned only when `store_history` is enabled, so on the default
    /// configuration this costs a single counter increment instead of a full copy
    /// of the input vector. When a [`TensorPool`] is attached (see
    /// [`InferenceContext::with_pool`]/[`InferenceContext::enable_pooling`]),
    /// the clone's backing storage is served from the pool instead of a
    /// fresh allocation whenever a same-sized buffer is available — the pool
    /// is kept supplied by `InferenceContext::evict_oldest_if_full`
    /// returning evicted entries to it.
    pub fn push_ref(&mut self, input: &Array1<f32>) {
        if self.config.store_history {
            self.evict_oldest_if_full();
            let stored = match &self.pool {
                Some(pool) => match pool.acquire_array1_f32(input.len()) {
                    Ok(mut buf) => {
                        buf.assign(input);
                        buf
                    }
                    Err(_) => input.clone(),
                },
                None => input.clone(),
            };
            self.history.push_back(stored);
        }
        self.step_count += 1;
    }

    /// Evict the oldest history entry if the buffer is at `max_context`
    /// capacity, returning its backing storage to the pool when one is
    /// attached.
    fn evict_oldest_if_full(&mut self) {
        if self.history.len() < self.config.max_context {
            return;
        }
        if let Some(old) = self.history.pop_front() {
            if let Some(pool) = &self.pool {
                let _ = pool.release_array1_f32(old);
            }
        }
    }

    /// Get the current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get the history length
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Get recent history as slice
    pub fn recent_history(&self, n: usize) -> Vec<&Array1<f32>> {
        self.history.iter().rev().take(n).collect()
    }

    /// Get hidden states
    pub fn states(&self) -> &[HiddenState] {
        &self.states
    }

    /// Get mutable hidden states
    pub fn states_mut(&mut self) -> &mut [HiddenState] {
        &mut self.states
    }

    /// Move the hidden states out of the context, leaving it empty
    ///
    /// This avoids the deep copy that `states().to_vec()` performs when the states
    /// are handed to a model. The caller **must** put a state vector back with
    /// [`InferenceContext::restore_states`] before the context is used again,
    /// including on error paths.
    pub fn take_states(&mut self) -> Vec<HiddenState> {
        std::mem::take(&mut self.states)
    }

    /// Put hidden states back into the context
    ///
    /// Counterpart of [`InferenceContext::take_states`].
    pub fn restore_states(&mut self, states: Vec<HiddenState>) {
        self.states = states;
    }

    /// Update hidden state for a layer
    pub fn update_state(&mut self, layer: usize, state: HiddenState) -> InferenceResult<()> {
        if layer >= self.states.len() {
            return Err(InferenceError::DimensionMismatch {
                expected: self.states.len(),
                got: layer + 1,
            });
        }
        self.states[layer] = state;
        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &ContextConfig {
        &self.config
    }

    /// Check if context is at capacity
    pub fn is_full(&self) -> bool {
        self.step_count >= self.config.max_context
    }

    /// Get the full history
    pub fn history(&self) -> &VecDeque<Array1<f32>> {
        &self.history
    }

    /// Trim history to specified length (for memory efficiency)
    ///
    /// Entries evicted this way are returned to the attached [`TensorPool`],
    /// same as capacity-triggered eviction in [`InferenceContext::push_ref`].
    pub fn trim_history(&mut self, max_len: usize) {
        while self.history.len() > max_len {
            if let Some(old) = self.history.pop_front() {
                if let Some(pool) = &self.pool {
                    let _ = pool.release_array1_f32(old);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let config = ContextConfig::new().max_context(100).num_layers(2);
        let ctx = InferenceContext::new(config);

        assert_eq!(ctx.step_count(), 0);
        assert_eq!(ctx.states().len(), 2);
    }

    #[test]
    fn test_context_push() {
        let config = ContextConfig::new().store_history(true).max_context(5);
        let mut ctx = InferenceContext::new(config);

        for i in 0..10 {
            ctx.push(Array1::from_vec(vec![i as f32]));
        }

        assert_eq!(ctx.step_count(), 10);
        assert_eq!(ctx.history_len(), 5); // Max context
    }

    /// Regression: `InferenceContext::pool` was written and read back by the
    /// getter, but no allocation ever went through it — a configured
    /// `TensorPool` had zero measurable effect. History churn under
    /// `store_history` (the per-step clone `push_ref` performs) must now
    /// actually reuse buffers from the pool once eviction starts, and
    /// correctness (the stored values) must be unaffected by the reuse.
    #[test]
    fn test_pooled_history_reuses_buffers() {
        let config = ContextConfig::new().store_history(true).max_context(2);
        let pool = crate::pool::TensorPool::new();
        let mut ctx = InferenceContext::with_pool(config, pool.clone());

        for i in 0..5 {
            ctx.push_ref(&Array1::from_vec(vec![i as f32, i as f32]));
        }

        let stats = pool.stats().expect("stats must be readable");
        assert!(
            stats.total_reuses > 0,
            "history churn past max_context should reuse buffers from the pool, got stats: {:?}",
            stats
        );
        assert_eq!(ctx.history_len(), 2);

        // Correctness survives buffer reuse: the two most recent pushes are
        // still exactly [3,3] then [4,4].
        let recent = ctx.recent_history(2);
        assert_eq!(recent[0].as_slice().unwrap(), &[4.0, 4.0]);
        assert_eq!(recent[1].as_slice().unwrap(), &[3.0, 3.0]);
    }

    /// Without a pool attached (the default), `push_ref` must behave exactly
    /// as before: a plain clone, no pool interaction.
    #[test]
    fn test_push_ref_without_pool_is_unaffected() {
        let config = ContextConfig::new().store_history(true).max_context(2);
        let mut ctx = InferenceContext::new(config);

        for i in 0..3 {
            ctx.push_ref(&Array1::from_vec(vec![i as f32]));
        }

        assert_eq!(ctx.history_len(), 2);
        let recent = ctx.recent_history(1);
        assert_eq!(recent[0].as_slice().unwrap(), &[2.0]);
    }

    #[test]
    fn test_context_reset() {
        let config = ContextConfig::new().store_history(true);
        let mut ctx = InferenceContext::new(config);

        ctx.push(Array1::from_vec(vec![1.0]));
        ctx.push(Array1::from_vec(vec![2.0]));

        assert_eq!(ctx.step_count(), 2);

        ctx.reset();

        assert_eq!(ctx.step_count(), 0);
        assert_eq!(ctx.history_len(), 0);
    }
}
