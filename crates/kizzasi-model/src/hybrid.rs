//! Hybrid Mamba+Attention Model
//!
//! This module implements a hybrid architecture combining Mamba's selective SSM
//! with multi-head attention, achieving both efficiency and expressiveness.
//!
//! # Architecture Strategy
//!
//! The hybrid model alternates between Mamba layers (for local/efficient processing)
//! and Attention layers (for global context), getting the best of both:
//!
//! - **Mamba layers**: O(1) per-step inference, selective state dynamics
//! - **Attention layers**: Global context, explicit long-range dependencies
//!
//! # Layer Configuration
//!
//! ```text
//! Input → [Mamba] → [Attention] → [Mamba] → [Attention] → ... → Output
//! ```
//!
//! Or interleaved:
//! ```text
//! Input → [Mamba] → [Mamba] → [Attention] → [Mamba] → [Mamba] → [Attention] → ...
//! ```
//!
//! # Use Cases
//!
//! - **Long sequences**: Attention provides global context while Mamba handles local patterns
//! - **Few-shot learning**: Attention for in-context learning, Mamba for parameter efficiency
//! - **Multimodal**: Different modalities can use different layer types
//!
//! # References
//!
//! - Combines ideas from Mamba and Transformer architectures
//! - Inspired by hybrid models like Jamba (AI21 Labs)

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{silu, softmax, CoreResult, HiddenState, SignalPredictor};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, RngExt};
use std::collections::VecDeque;

#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

/// Layer type in hybrid model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Mamba selective SSM layer
    Mamba,
    /// Multi-head attention layer
    Attention,
}

/// Configuration for hybrid Mamba+Attention model
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// State dimension for Mamba layers
    pub state_dim: usize,
    /// Total number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Maximum sequence length for attention
    pub max_seq_len: usize,
    /// Layer pattern (e.g., [Mamba, Mamba, Attention, ...])
    pub layer_pattern: Vec<LayerType>,
}

impl HybridConfig {
    /// Create a new hybrid config with alternating layers
    pub fn alternating(
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Self {
        let layer_pattern = (0..num_layers)
            .map(|i| {
                if i % 2 == 0 {
                    LayerType::Mamba
                } else {
                    LayerType::Attention
                }
            })
            .collect();

        Self {
            input_dim,
            hidden_dim,
            state_dim: 64,
            num_layers,
            num_heads,
            max_seq_len: 2048,
            layer_pattern,
        }
    }

    /// Create a config with mostly Mamba, occasional attention
    pub fn mamba_heavy(
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Self {
        let layer_pattern = (0..num_layers)
            .map(|i| {
                // Attention every 4 layers
                if i % 4 == 3 {
                    LayerType::Attention
                } else {
                    LayerType::Mamba
                }
            })
            .collect();

        Self {
            input_dim,
            hidden_dim,
            state_dim: 64,
            num_layers,
            num_heads,
            max_seq_len: 2048,
            layer_pattern,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> ModelResult<()> {
        if self.hidden_dim == 0 {
            return Err(ModelError::invalid_config("hidden_dim must be > 0"));
        }
        if self.state_dim == 0 {
            return Err(ModelError::invalid_config("state_dim must be > 0"));
        }
        if self.num_layers == 0 {
            return Err(ModelError::invalid_config("num_layers must be > 0"));
        }
        if self.num_heads == 0 {
            return Err(ModelError::invalid_config("num_heads must be > 0"));
        }
        if !self.hidden_dim.is_multiple_of(self.num_heads) {
            return Err(ModelError::invalid_config(
                "hidden_dim must be divisible by num_heads",
            ));
        }
        if self.layer_pattern.len() != self.num_layers {
            return Err(ModelError::invalid_config(
                "layer_pattern length must equal num_layers",
            ));
        }
        Ok(())
    }
}

/// Simplified Mamba layer for hybrid model
#[allow(dead_code)]
struct MambaBlock {
    hidden_dim: usize,
    state_dim: usize,
    /// Projection matrices
    proj_in: Array2<f32>,
    proj_out: Array2<f32>,
    /// SSM parameters (simplified)
    a_log: Array1<f32>,
    b_matrix: Array2<f32>,
    c_matrix: Array2<f32>,
    /// Current state
    state: Array1<f32>,
}

impl MambaBlock {
    fn new(hidden_dim: usize, state_dim: usize) -> Self {
        let mut rng = rng();

        let scale = (2.0 / (hidden_dim + hidden_dim) as f32).sqrt();
        let proj_in = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let proj_out = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Initialize SSM parameters
        let a_log = Array1::from_shape_fn(state_dim, |i| -((i + 1) as f32).ln());

        let scale = (1.0 / state_dim as f32).sqrt();
        let b_matrix = Array2::from_shape_fn((state_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let c_matrix = Array2::from_shape_fn((hidden_dim, state_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let state = Array1::zeros(state_dim);

        Self {
            hidden_dim,
            state_dim,
            proj_in,
            proj_out,
            a_log,
            b_matrix,
            c_matrix,
            state,
        }
    }

    fn forward(&mut self, x: &Array1<f32>) -> Array1<f32> {
        // Input projection
        let projected = x.dot(&self.proj_in);

        // SSM dynamics with selective mechanism
        let a_bar = self.a_log.mapv(|a| (0.001 * a.exp()).exp());
        self.state = &self.state * &a_bar + self.b_matrix.dot(&projected) * 0.001;

        // Output
        let ssm_out = self.c_matrix.dot(&self.state);

        // Gate with SiLU
        let gated = silu(&projected) * &ssm_out;

        // Output projection
        gated.dot(&self.proj_out)
    }

    fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

/// Simplified attention layer for hybrid model
#[allow(dead_code)]
struct AttentionBlock {
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
    /// Query, Key, Value projections
    q_proj: Array2<f32>,
    k_proj: Array2<f32>,
    v_proj: Array2<f32>,
    /// Output projection
    o_proj: Array2<f32>,
    /// KV cache
    k_cache: VecDeque<Array1<f32>>,
    v_cache: VecDeque<Array1<f32>>,
    max_cache_len: usize,
}

impl AttentionBlock {
    fn new(hidden_dim: usize, num_heads: usize, max_seq_len: usize) -> Self {
        let mut rng = rng();
        let head_dim = hidden_dim / num_heads;

        let scale = (2.0 / (hidden_dim + hidden_dim) as f32).sqrt();
        let q_proj = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let k_proj = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let v_proj = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let o_proj = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        Self {
            hidden_dim,
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            k_cache: VecDeque::new(),
            v_cache: VecDeque::new(),
            max_cache_len: max_seq_len,
        }
    }

    fn forward(&mut self, x: &Array1<f32>) -> Array1<f32> {
        // Compute Q, K, V
        let q = x.dot(&self.q_proj);
        let k = x.dot(&self.k_proj);
        let v = x.dot(&self.v_proj);

        // Add to cache
        self.k_cache.push_back(k.clone());
        self.v_cache.push_back(v.clone());

        // Trim cache
        while self.k_cache.len() > self.max_cache_len {
            self.k_cache.pop_front();
            self.v_cache.pop_front();
        }

        // Compute attention (simplified single-head version)
        let cache_len = self.k_cache.len();
        let mut attention_out = Array1::zeros(self.hidden_dim);

        if cache_len > 0 {
            // Compute attention scores
            let mut scores = Vec::with_capacity(cache_len);
            for k_cached in &self.k_cache {
                let score = q.dot(k_cached) / (self.head_dim as f32).sqrt();
                scores.push(score);
            }

            // Softmax
            let scores_array = Array1::from_vec(scores);
            let attn_weights = softmax(&scores_array);

            // Weighted sum of values
            for (weight, v_cached) in attn_weights.iter().zip(self.v_cache.iter()) {
                attention_out = attention_out + v_cached * *weight;
            }
        }

        // Output projection
        attention_out.dot(&self.o_proj)
    }

    fn reset(&mut self) {
        self.k_cache.clear();
        self.v_cache.clear();
    }
}

/// Enum for hybrid layer
enum HybridLayer {
    Mamba(MambaBlock),
    Attention(AttentionBlock),
}

impl HybridLayer {
    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        match self {
            HybridLayer::Mamba(mamba) => Ok(mamba.forward(x)),
            HybridLayer::Attention(attn) => Ok(attn.forward(x)),
        }
    }

    fn reset(&mut self) {
        match self {
            HybridLayer::Mamba(mamba) => mamba.reset(),
            HybridLayer::Attention(attn) => attn.reset(),
        }
    }
}

/// Hybrid Mamba+Attention model
pub struct HybridModel {
    config: HybridConfig,
    layers: Vec<HybridLayer>,
    /// Input/output projections
    input_proj: Array2<f32>,
    output_proj: Array2<f32>,
}

impl HybridModel {
    /// Create a new hybrid model
    #[instrument(skip(config), fields(num_layers = config.num_layers))]
    pub fn new(config: HybridConfig) -> ModelResult<Self> {
        debug!("Creating new Hybrid Mamba+Attention model");
        config.validate()?;

        let mut rng = rng();

        // Input projection
        let scale = (2.0 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let input_proj = Array2::from_shape_fn((config.input_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Output projection
        let scale = (2.0 / (config.hidden_dim + config.input_dim) as f32).sqrt();
        let output_proj = Array2::from_shape_fn((config.hidden_dim, config.input_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Create layers based on pattern
        let mut layers = Vec::with_capacity(config.num_layers);
        for (i, &layer_type) in config.layer_pattern.iter().enumerate() {
            trace!("Initializing hybrid layer {} as {:?}", i, layer_type);
            let layer = match layer_type {
                LayerType::Mamba => {
                    HybridLayer::Mamba(MambaBlock::new(config.hidden_dim, config.state_dim))
                }
                LayerType::Attention => HybridLayer::Attention(AttentionBlock::new(
                    config.hidden_dim,
                    config.num_heads,
                    config.max_seq_len,
                )),
            };
            layers.push(layer);
        }

        debug!(
            "Hybrid model created successfully with {} layers",
            layers.len()
        );
        Ok(Self {
            config,
            layers,
            input_proj,
            output_proj,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    /// Count layers of each type
    pub fn layer_counts(&self) -> (usize, usize) {
        let mamba_count = self
            .config
            .layer_pattern
            .iter()
            .filter(|&&t| t == LayerType::Mamba)
            .count();
        let attention_count = self.config.num_layers - mamba_count;
        (mamba_count, attention_count)
    }
}

impl SignalPredictor for HybridModel {
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Project input
        let mut hidden = input.dot(&self.input_proj);

        // Pass through hybrid layers
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden)?;
        }

        // Project output
        let output = hidden.dot(&self.output_proj);
        Ok(output)
    }

    #[instrument(skip(self))]
    fn reset(&mut self) {
        debug!("Resetting Hybrid model state");
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    fn context_window(&self) -> usize {
        // Context window is determined by attention layers
        self.config.max_seq_len
    }
}

impl AutoregressiveModel for HybridModel {
    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn state_dim(&self) -> usize {
        self.config.state_dim
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn model_type(&self) -> ModelType {
        ModelType::Mamba // Hybrid, but Mamba-based
    }

    fn get_states(&self) -> Vec<HiddenState> {
        // Simplified state extraction
        (0..self.config.num_layers)
            .map(|_| HiddenState::new(self.config.hidden_dim, self.config.state_dim))
            .collect()
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != self.config.num_layers {
            return Err(ModelError::state_count_mismatch(
                "Hybrid",
                self.config.num_layers,
                states.len(),
            ));
        }
        // State setting would require more complex handling of different layer types
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_creation_alternating() {
        let config = HybridConfig::alternating(32, 64, 4, 4);
        let model = HybridModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_hybrid_creation_mamba_heavy() {
        let config = HybridConfig::mamba_heavy(32, 64, 8, 4);
        let model = HybridModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_hybrid_forward() {
        let config = HybridConfig::alternating(32, 64, 4, 4);
        let mut model = HybridModel::new(config).expect("Failed to create HybridModel");

        let input = Array1::from_vec(vec![1.0; 32]);
        let output = model.step(&input);
        assert!(output.is_ok());
        assert_eq!(output.expect("Failed to get output").len(), 32);
    }

    #[test]
    fn test_hybrid_layer_counts() {
        let config = HybridConfig::alternating(32, 64, 6, 4);
        let model = HybridModel::new(config).expect("Failed to create HybridModel");
        let (mamba, attn) = model.layer_counts();
        assert_eq!(mamba, 3);
        assert_eq!(attn, 3);
    }

    #[test]
    fn test_hybrid_mamba_heavy_counts() {
        let config = HybridConfig::mamba_heavy(32, 64, 8, 4);
        let model = HybridModel::new(config).expect("Failed to create HybridModel");
        let (mamba, attn) = model.layer_counts();
        assert_eq!(mamba, 6);
        assert_eq!(attn, 2);
    }

    #[test]
    fn test_hybrid_reset() {
        let config = HybridConfig::alternating(32, 64, 4, 4);
        let mut model = HybridModel::new(config).expect("Failed to create HybridModel");

        let input = Array1::from_vec(vec![0.5; 32]);
        let _ = model.step(&input).expect("Failed to step model");

        model.reset();

        let output = model.step(&input).expect("Failed to get output");
        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_invalid_config() {
        let mut config = HybridConfig::alternating(32, 64, 4, 4);
        config.layer_pattern.push(LayerType::Mamba); // Mismatch with num_layers
        assert!(config.validate().is_err());
    }
}
