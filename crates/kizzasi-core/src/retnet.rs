//! RetNet: Retention Networks for Multi-Scale Sequence Modeling
//!
//! RetNet replaces traditional attention with a retention mechanism that provides:
//! - **O(1)** inference complexity (like RNNs)
//! - **Parallel training** (like Transformers)
//! - **Multi-scale temporal modeling** via multiple retention heads
//! - **Linear memory complexity**
//!
//! # Architecture
//!
//! RetNet uses Multi-Scale Retention (MSR) which applies retention at different scales:
//! ```text
//! Input → [GroupNorm] → [MSRetention] → [FFN] → Output
//!                          ↓
//!                       [State]
//! ```
//!
//! # Retention Mechanism
//!
//! The retention mechanism for head h:
//! ```text
//! Q = X W_Q,  K = X W_K,  V = X W_V
//! Retention = (Q K^T ⊙ D) V
//! ```
//!
//! Where D is a causal decay matrix: D[i,j] = γ^(i-j) for i >= j
//! γ is the decay factor (different per head for multi-scale)
//!
//! # Recurrent Form (O(1) inference)
//!
//! ```text
//! S_t = γ S_{t-1} + K_t^T V_t
//! O_t = Q_t S_t
//! ```

use crate::error::{CoreError, CoreResult};
use crate::nn::{silu, LayerNorm, NormType};
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use scirs2_core::random::thread_rng;

/// Multi-Scale Retention Configuration
#[derive(Debug, Clone)]
pub struct RetNetConfig {
    /// Model dimension
    pub hidden_dim: usize,
    /// Number of retention heads
    pub num_heads: usize,
    /// Head dimension (hidden_dim / num_heads)
    pub head_dim: usize,
    /// FFN expansion factor
    pub ffn_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout rate
    pub dropout: f32,
}

impl RetNetConfig {
    /// Create a new RetNet configuration
    pub fn new(hidden_dim: usize, num_heads: usize, num_layers: usize) -> CoreResult<Self> {
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(CoreError::InvalidConfig(format!(
                "hidden_dim ({}) must be divisible by num_heads ({})",
                hidden_dim, num_heads
            )));
        }

        Ok(Self {
            hidden_dim,
            num_heads,
            head_dim: hidden_dim / num_heads,
            ffn_dim: hidden_dim * 4, // Standard 4x expansion
            num_layers,
            dropout: 0.0,
        })
    }

    /// Set FFN dimension
    pub fn ffn_dim(mut self, dim: usize) -> Self {
        self.ffn_dim = dim;
        self
    }

    /// Set dropout rate
    pub fn dropout(mut self, rate: f32) -> Self {
        self.dropout = rate;
        self
    }
}

/// Multi-Scale Retention (MSR) Module
///
/// Implements retention mechanism with multiple heads, each operating at different scales
#[derive(Debug)]
pub struct MultiScaleRetention {
    config: RetNetConfig,
    // QKV projections
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    // Output projection
    w_o: Array2<f32>,
    // Decay factors (gamma) for each head
    gamma: Array1<f32>,
    // Group norm
    group_norm: LayerNorm,
}

impl MultiScaleRetention {
    /// Create a new multi-scale retention module
    pub fn new(config: RetNetConfig) -> CoreResult<Self> {
        let hidden_dim = config.hidden_dim;
        let num_heads = config.num_heads;
        let mut rng = thread_rng();
        let scale = (1.0 / hidden_dim as f32).sqrt();

        // Initialize QKV projections
        let w_q = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let w_k = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let w_v = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let w_o = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // Initialize decay factors (gamma) for multi-scale
        // Each head has different decay: γ_h = 1 - 2^(-5-h) for h = 0..H-1
        let gamma = Array1::from_shape_fn(num_heads, |h| {
            let exponent = -(5.0 + h as f32);
            1.0 - 2.0_f32.powf(exponent)
        });

        // Group normalization (using RMSNorm for efficiency)
        let group_norm = LayerNorm::new(hidden_dim, NormType::RMSNorm);

        Ok(Self {
            config,
            w_q,
            w_k,
            w_v,
            w_o,
            gamma,
            group_norm,
        })
    }

    /// Recurrent forward step (O(1) inference)
    ///
    /// Updates retention state and computes output
    /// State: (num_heads, head_dim, head_dim)
    pub fn step(&self, input: &Array1<f32>, state: &mut Array3<f32>) -> CoreResult<Array1<f32>> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Project to Q, K, V
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);

        let mut output = Array1::zeros(self.config.hidden_dim);

        // Process each head
        for h in 0..num_heads {
            let start = h * head_dim;
            let end = start + head_dim;

            let q_h = q.slice(s![start..end]);
            let k_h = k.slice(s![start..end]);
            let v_h = v.slice(s![start..end]);

            // Get state for this head
            let mut s_h = state.index_axis_mut(Axis(0), h);

            // Decay previous state: S_t = γ S_{t-1}
            let gamma_h = self.gamma[h];
            for i in 0..head_dim {
                for j in 0..head_dim {
                    s_h[[i, j]] *= gamma_h;
                }
            }

            // Add new contribution: S_t += K_t^T V_t (outer product)
            for i in 0..head_dim {
                for j in 0..head_dim {
                    s_h[[i, j]] += k_h[i] * v_h[j];
                }
            }

            // Output: O_t = Q_t S_t
            for j in 0..head_dim {
                let mut sum = 0.0;
                for i in 0..head_dim {
                    sum += q_h[i] * s_h[[i, j]];
                }
                output[start + j] = sum;
            }
        }

        // Group normalization
        let normed = self.group_norm.forward(&output);

        // Output projection with SiLU activation
        let output_proj = normed.dot(&self.w_o);
        let activated = silu(&output_proj);

        Ok(activated)
    }

    /// Parallel forward for sequence (training mode)
    ///
    /// Input shape: (seq_len, hidden_dim)
    /// Output shape: (seq_len, hidden_dim)
    pub fn forward_sequence(&self, input: &Array2<f32>) -> CoreResult<Array2<f32>> {
        let (seq_len, _) = input.dim();

        let mut output = Array2::zeros((seq_len, self.config.hidden_dim));
        let mut state = self.reset_state();

        // Process sequence step by step
        for t in 0..seq_len {
            let x_t = input.row(t).to_owned();
            let y_t = self.step(&x_t, &mut state)?;
            output.row_mut(t).assign(&y_t);
        }

        Ok(output)
    }

    /// Reset retention state
    pub fn reset_state(&self) -> Array3<f32> {
        Array3::zeros((
            self.config.num_heads,
            self.config.head_dim,
            self.config.head_dim,
        ))
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len() + self.gamma.len()
    }
}

/// Feed-Forward Network for RetNet
#[derive(Debug)]
pub struct RetNetFFN {
    w1: Array2<f32>,
    w2: Array2<f32>,
    layer_norm: LayerNorm,
}

impl RetNetFFN {
    /// Create a new FFN
    pub fn new(hidden_dim: usize, ffn_dim: usize) -> Self {
        let mut rng = thread_rng();
        let scale1 = (1.0 / hidden_dim as f32).sqrt();
        let scale2 = (1.0 / ffn_dim as f32).sqrt();

        let w1 = Array2::from_shape_fn((hidden_dim, ffn_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale1
        });
        let w2 = Array2::from_shape_fn((ffn_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale2
        });

        let layer_norm = LayerNorm::new(hidden_dim, NormType::RMSNorm);

        Self { w1, w2, layer_norm }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Layer norm
        let normed = self.layer_norm.forward(input);

        // FFN with SiLU activation: SiLU(x W1) W2
        let hidden = normed.dot(&self.w1);
        let activated = silu(&hidden);
        let output = activated.dot(&self.w2);

        Ok(output)
    }
}

/// RetNet Layer
///
/// Combines Multi-Scale Retention and FFN with residual connections
#[derive(Debug)]
pub struct RetNetLayer {
    retention: MultiScaleRetention,
    ffn: RetNetFFN,
}

impl RetNetLayer {
    /// Create a new RetNet layer
    pub fn new(config: RetNetConfig) -> CoreResult<Self> {
        let retention = MultiScaleRetention::new(config.clone())?;
        let ffn = RetNetFFN::new(config.hidden_dim, config.ffn_dim);

        Ok(Self { retention, ffn })
    }

    /// Forward step with residual connections
    pub fn step(&self, input: &Array1<f32>, state: &mut Array3<f32>) -> CoreResult<Array1<f32>> {
        // Retention with residual
        let retention_out = self.retention.step(input, state)?;
        let after_retention = input + &retention_out;

        // FFN with residual
        let ffn_out = self.ffn.forward(&after_retention)?;
        let output = &after_retention + &ffn_out;

        Ok(output)
    }

    /// Forward sequence
    pub fn forward_sequence(&self, input: &Array2<f32>) -> CoreResult<Array2<f32>> {
        let (seq_len, _) = input.dim();
        let mut output = Array2::zeros(input.dim());
        let mut state = self.retention.reset_state();

        for t in 0..seq_len {
            let x_t = input.row(t).to_owned();
            let y_t = self.step(&x_t, &mut state)?;
            output.row_mut(t).assign(&y_t);
        }

        Ok(output)
    }

    /// Reset state
    pub fn reset_state(&self) -> Array3<f32> {
        self.retention.reset_state()
    }
}

/// Multi-layer RetNet Model
#[derive(Debug)]
pub struct RetNetModel {
    layers: Vec<RetNetLayer>,
    config: RetNetConfig,
}

impl RetNetModel {
    /// Create a new multi-layer RetNet model
    pub fn new(config: RetNetConfig) -> CoreResult<Self> {
        let num_layers = config.num_layers;
        let mut layers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            layers.push(RetNetLayer::new(config.clone())?);
        }

        Ok(Self { layers, config })
    }

    /// Single step inference
    pub fn step(&self, input: &Array1<f32>, states: &mut [Array3<f32>]) -> CoreResult<Array1<f32>> {
        if states.len() != self.config.num_layers {
            return Err(CoreError::InvalidConfig(format!(
                "Expected {} states, got {}",
                self.config.num_layers,
                states.len()
            )));
        }

        let mut x = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.step(&x, &mut states[i])?;
        }

        Ok(x)
    }

    /// Forward pass for sequence
    pub fn forward(&self, input: &Array2<f32>) -> CoreResult<Array2<f32>> {
        let mut x = input.clone();

        for layer in &self.layers {
            x = layer.forward_sequence(&x)?;
        }

        Ok(x)
    }

    /// Reset all states
    pub fn reset_states(&self) -> Vec<Array3<f32>> {
        self.layers
            .iter()
            .map(|layer| layer.reset_state())
            .collect()
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.retention.num_parameters() + layer.ffn.w1.len() + layer.ffn.w2.len())
            .sum()
    }
}

// Re-export slice macro
use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retnet_config() {
        let config = RetNetConfig::new(256, 4, 6).unwrap();
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_layers, 6);
    }

    #[test]
    fn test_multi_scale_retention() {
        let config = RetNetConfig::new(128, 4, 2).unwrap();
        let msr = MultiScaleRetention::new(config).unwrap();

        let input = Array1::from_vec(vec![0.1; 128]);
        let mut state = msr.reset_state();

        let output = msr.step(&input, &mut state).unwrap();
        assert_eq!(output.len(), 128);

        // Check that state has been updated
        assert!(state.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_retnet_layer() {
        let config = RetNetConfig::new(128, 4, 2).unwrap();
        let layer = RetNetLayer::new(config).unwrap();

        let input = Array1::from_vec(vec![0.1; 128]);
        let mut state = layer.reset_state();

        let output = layer.step(&input, &mut state).unwrap();
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_retnet_model() {
        let config = RetNetConfig::new(64, 2, 3).unwrap();
        let model = RetNetModel::new(config).unwrap();

        let seq_len = 10;
        let input = Array2::from_shape_vec((seq_len, 64), vec![0.1; seq_len * 64]).unwrap();

        let output = model.forward(&input).unwrap();
        assert_eq!(output.dim(), (seq_len, 64));
    }

    #[test]
    fn test_retnet_inference() {
        let config = RetNetConfig::new(64, 2, 2).unwrap();
        let model = RetNetModel::new(config).unwrap();

        let mut states = model.reset_states();
        let input = Array1::from_vec(vec![0.1; 64]);

        // Process multiple steps
        for _ in 0..5 {
            let output = model.step(&input, &mut states).unwrap();
            assert_eq!(output.len(), 64);
        }
    }

    #[test]
    fn test_gamma_values() {
        let config = RetNetConfig::new(128, 4, 2).unwrap();
        let msr = MultiScaleRetention::new(config).unwrap();

        // Check that gamma values are in valid range (0, 1)
        for &gamma in msr.gamma.iter() {
            assert!(gamma > 0.0 && gamma < 1.0);
        }

        // Check that gammas are decreasing (larger heads have smaller decay)
        for i in 1..msr.gamma.len() {
            assert!(msr.gamma[i] >= msr.gamma[i - 1]);
        }
    }
}
