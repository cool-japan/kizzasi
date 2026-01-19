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

    /// Add an input to the history
    pub fn push(&mut self, input: Array1<f32>) {
        if self.config.store_history {
            if self.history.len() >= self.config.max_context {
                self.history.pop_front();
            }
            self.history.push_back(input);
        }
        self.step_count += 1;
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
    pub fn trim_history(&mut self, max_len: usize) {
        while self.history.len() > max_len {
            self.history.pop_front();
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
