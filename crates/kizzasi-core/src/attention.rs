//! Multi-head SSM Attention mechanisms
//!
//! Implements multi-head attention patterns optimized for State Space Models.
//! Includes both standard multi-head attention and specialized SSM variants.

use crate::error::{CoreError, CoreResult};
use crate::numerics::{safe_exp, softmax_stable};
use crate::simd;
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use scirs2_core::random::thread_rng;

/// Multi-head SSM Attention configuration
#[derive(Debug, Clone)]
pub struct MultiHeadSSMConfig {
    /// Model dimension (d_model)
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension (d_model / num_heads)
    pub head_dim: usize,
    /// State dimension per head
    pub state_dim: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Use causal masking
    pub causal: bool,
}

impl MultiHeadSSMConfig {
    /// Create a new configuration
    pub fn new(hidden_dim: usize, num_heads: usize, state_dim: usize) -> CoreResult<Self> {
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
            state_dim,
            dropout: 0.0,
            causal: true,
        })
    }

    /// Set dropout rate
    pub fn dropout(mut self, rate: f32) -> Self {
        self.dropout = rate;
        self
    }

    /// Set causal masking
    pub fn causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }
}

/// Multi-head SSM Attention layer
///
/// Implements multi-head attention specifically optimized for SSMs:
/// - Supports both standard attention and SSM-specific variants
/// - SIMD-optimized matrix operations
/// - Memory-efficient causal masking
/// - Compatible with linear-time SSM inference
#[derive(Debug)]
pub struct MultiHeadSSMAttention {
    config: MultiHeadSSMConfig,
    // Query, Key, Value projections
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    // Output projection
    w_o: Array2<f32>,
    // Optional bias terms
    b_q: Option<Array1<f32>>,
    b_k: Option<Array1<f32>>,
    b_v: Option<Array1<f32>>,
    b_o: Option<Array1<f32>>,
}

impl MultiHeadSSMAttention {
    /// Create a new multi-head SSM attention layer
    pub fn new(config: MultiHeadSSMConfig, use_bias: bool) -> CoreResult<Self> {
        let hidden_dim = config.hidden_dim;
        let mut rng = thread_rng();
        let scale = (1.0 / hidden_dim as f32).sqrt();

        // Initialize projection matrices with Xavier/Glorot initialization
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

        // Optional bias terms
        let (b_q, b_k, b_v, b_o) = if use_bias {
            (
                Some(Array1::zeros(hidden_dim)),
                Some(Array1::zeros(hidden_dim)),
                Some(Array1::zeros(hidden_dim)),
                Some(Array1::zeros(hidden_dim)),
            )
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            config,
            w_q,
            w_k,
            w_v,
            w_o,
            b_q,
            b_k,
            b_v,
            b_o,
        })
    }

    /// Forward pass for a single query vector (inference mode)
    ///
    /// This is optimized for O(1) per-step inference in SSMs.
    pub fn forward_step(
        &self,
        query: &Array1<f32>,
        key_cache: &Array2<f32>,
        value_cache: &Array2<f32>,
    ) -> CoreResult<Array1<f32>> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let seq_len = key_cache.nrows();

        // Project query
        let q = self.project_qkv(&self.w_q, &self.b_q, query);

        // Reshape to multi-head: (hidden_dim,) -> (num_heads, head_dim)
        let q_heads = self.reshape_to_heads(&q)?;

        // Compute attention scores for each head
        let mut attn_output = Array1::zeros(self.config.hidden_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();

        for h in 0..num_heads {
            let q_h = q_heads.slice(s![h, ..]);

            // Compute attention scores: scores[i] = q · k[i]
            let mut scores = Array1::zeros(seq_len);
            for i in 0..seq_len {
                let k_i = key_cache.slice(s![i, h * head_dim..(h + 1) * head_dim]);
                scores[i] = simd::dot_view(q_h, k_i) * scale;
            }

            // Apply causal mask if needed
            if self.config.causal {
                // For inference, all cached keys are valid (from past)
                // No masking needed as we only attend to past
            }

            // Softmax over scores
            let attn_weights = softmax_stable(&scores);

            // Weighted sum of values
            let mut context = Array1::zeros(head_dim);
            for i in 0..seq_len {
                let v_i = value_cache.slice(s![i, h * head_dim..(h + 1) * head_dim]);
                let weight = attn_weights[i];
                for j in 0..head_dim {
                    context[j] += weight * v_i[j];
                }
            }

            // Copy to output
            let start = h * head_dim;
            let end = start + head_dim;
            attn_output.slice_mut(s![start..end]).assign(&context);
        }

        // Output projection
        let output = if let Some(ref bias) = self.b_o {
            attn_output.dot(&self.w_o) + bias
        } else {
            attn_output.dot(&self.w_o)
        };

        Ok(output)
    }

    /// Forward pass for a batch of sequences (training mode)
    ///
    /// Input shape: (batch_size, seq_len, hidden_dim)
    /// Output shape: (batch_size, seq_len, hidden_dim)
    pub fn forward_batch(
        &self,
        input: &Array3<f32>,
        mask: Option<&Array2<bool>>,
    ) -> CoreResult<Array3<f32>> {
        let (batch_size, seq_len, _hidden_dim) = input.dim();
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        let mut output = Array3::zeros((batch_size, seq_len, self.config.hidden_dim));

        // Process each batch item
        for b in 0..batch_size {
            let input_batch = input.index_axis(Axis(0), b);

            // Project Q, K, V for all positions
            let mut q_all = Array2::zeros((seq_len, self.config.hidden_dim));
            let mut k_all = Array2::zeros((seq_len, self.config.hidden_dim));
            let mut v_all = Array2::zeros((seq_len, self.config.hidden_dim));

            for t in 0..seq_len {
                let x_t = input_batch.index_axis(Axis(0), t).to_owned();
                q_all
                    .index_axis_mut(Axis(0), t)
                    .assign(&self.project_qkv(&self.w_q, &self.b_q, &x_t));
                k_all
                    .index_axis_mut(Axis(0), t)
                    .assign(&self.project_qkv(&self.w_k, &self.b_k, &x_t));
                v_all
                    .index_axis_mut(Axis(0), t)
                    .assign(&self.project_qkv(&self.w_v, &self.b_v, &x_t));
            }

            // Compute attention for each position
            for t in 0..seq_len {
                let q_t = q_all.index_axis(Axis(0), t).to_owned();
                let q_heads = self.reshape_to_heads(&q_t)?;

                let mut attn_output = Array1::zeros(self.config.hidden_dim);
                let scale = 1.0 / (head_dim as f32).sqrt();

                for h in 0..num_heads {
                    let q_h = q_heads.slice(s![h, ..]);

                    // Compute attention scores
                    let attend_len = if self.config.causal { t + 1 } else { seq_len };
                    let mut scores = Array1::zeros(attend_len);

                    for i in 0..attend_len {
                        let k_i = k_all.slice(s![i, h * head_dim..(h + 1) * head_dim]);
                        scores[i] = simd::dot_view(q_h, k_i) * scale;
                    }

                    // Apply mask if provided
                    if let Some(mask_data) = mask {
                        for i in 0..attend_len {
                            if !mask_data[[b, i]] {
                                scores[i] = f32::NEG_INFINITY;
                            }
                        }
                    }

                    // Softmax
                    let attn_weights = softmax_stable(&scores);

                    // Weighted sum of values
                    let mut context = Array1::zeros(head_dim);
                    for i in 0..attend_len {
                        let v_i = v_all.slice(s![i, h * head_dim..(h + 1) * head_dim]);
                        let weight = attn_weights[i];
                        for j in 0..head_dim {
                            context[j] += weight * v_i[j];
                        }
                    }

                    // Copy to output
                    let start = h * head_dim;
                    let end = start + head_dim;
                    attn_output.slice_mut(s![start..end]).assign(&context);
                }

                // Output projection
                let out_t = if let Some(ref bias) = self.b_o {
                    attn_output.dot(&self.w_o) + bias
                } else {
                    attn_output.dot(&self.w_o)
                };

                output
                    .index_axis_mut(Axis(0), b)
                    .index_axis_mut(Axis(0), t)
                    .assign(&out_t);
            }
        }

        Ok(output)
    }

    /// Project input through QKV matrix
    fn project_qkv(
        &self,
        weight: &Array2<f32>,
        bias: &Option<Array1<f32>>,
        input: &Array1<f32>,
    ) -> Array1<f32> {
        if let Some(ref b) = bias {
            input.dot(weight) + b
        } else {
            input.dot(weight)
        }
    }

    /// Reshape flat vector to multi-head format
    /// Input: (hidden_dim,) -> Output: (num_heads, head_dim)
    fn reshape_to_heads(&self, x: &Array1<f32>) -> CoreResult<Array2<f32>> {
        if x.len() != self.config.hidden_dim {
            return Err(CoreError::DimensionMismatch {
                expected: self.config.hidden_dim,
                got: x.len(),
            });
        }

        let mut result = Array2::zeros((self.config.num_heads, self.config.head_dim));
        for h in 0..self.config.num_heads {
            let start = h * self.config.head_dim;
            let end = start + self.config.head_dim;
            result.row_mut(h).assign(&x.slice(s![start..end]));
        }

        Ok(result)
    }

    /// Get configuration
    pub fn config(&self) -> &MultiHeadSSMConfig {
        &self.config
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len();
        let bias_params = if self.b_q.is_some() {
            4 * self.config.hidden_dim
        } else {
            0
        };
        weight_params + bias_params
    }
}

/// Gated Linear Attention (Griffin-style)
///
/// Implements efficient gated linear attention for SSMs:
/// - Linear complexity in sequence length
/// - Gating mechanism for selective attention
/// - Compatible with SSM recurrence
#[derive(Debug)]
pub struct GatedLinearAttention {
    hidden_dim: usize,
    // Gate projection
    w_gate: Array2<f32>,
    // Query/Key projections
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    // Output projection
    w_o: Array2<f32>,
}

impl GatedLinearAttention {
    /// Create a new gated linear attention layer
    pub fn new(hidden_dim: usize) -> CoreResult<Self> {
        let mut rng = thread_rng();
        let scale = (1.0 / hidden_dim as f32).sqrt();

        let w_gate = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let w_q = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let w_k = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let w_o = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        Ok(Self {
            hidden_dim,
            w_gate,
            w_q,
            w_k,
            w_o,
        })
    }

    /// Forward step with linear attention
    ///
    /// Uses linear attention: O(d^2) per step instead of O(n*d)
    pub fn forward_step(
        &self,
        input: &Array1<f32>,
        kv_state: &mut Array2<f32>,
    ) -> CoreResult<Array1<f32>> {
        // Project to query, key, gate
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let g = input.dot(&self.w_gate);

        // Apply gating (sigmoid)
        let gate = g.mapv(|x| 1.0 / (1.0 + safe_exp(-x)));

        // Update KV state: kv_state += k ⊗ (g * v)
        let gated_value = &gate * input;
        for i in 0..self.hidden_dim {
            for j in 0..self.hidden_dim {
                kv_state[[i, j]] += k[i] * gated_value[j];
            }
        }

        // Attention output: q^T * kv_state
        let mut attn_out = Array1::zeros(self.hidden_dim);
        for j in 0..self.hidden_dim {
            let mut sum = 0.0;
            for i in 0..self.hidden_dim {
                sum += q[i] * kv_state[[i, j]];
            }
            attn_out[j] = sum;
        }

        // Output projection
        let output = attn_out.dot(&self.w_o);
        Ok(output)
    }

    /// Reset KV state
    pub fn reset_state(&self) -> Array2<f32> {
        Array2::zeros((self.hidden_dim, self.hidden_dim))
    }
}

// Re-export slice macro if needed
use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multihead_ssm_config() {
        let config = MultiHeadSSMConfig::new(512, 8, 64).unwrap();
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
    }

    #[test]
    fn test_multihead_ssm_attention() {
        let config = MultiHeadSSMConfig::new(64, 4, 16).unwrap();
        let attn = MultiHeadSSMAttention::new(config, false).unwrap();

        let query = Array1::from_vec(vec![0.1; 64]);
        let key_cache = Array2::from_shape_vec((10, 64), vec![0.1; 640]).unwrap();
        let value_cache = Array2::from_shape_vec((10, 64), vec![0.2; 640]).unwrap();

        let output = attn.forward_step(&query, &key_cache, &value_cache).unwrap();
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_gated_linear_attention() {
        let gla = GatedLinearAttention::new(64).unwrap();
        let input = Array1::from_vec(vec![0.1; 64]);
        let mut kv_state = gla.reset_state();

        let output = gla.forward_step(&input, &mut kv_state).unwrap();
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_multihead_batch_forward() {
        let config = MultiHeadSSMConfig::new(64, 4, 16).unwrap();
        let attn = MultiHeadSSMAttention::new(config, false).unwrap();

        let batch_size = 2;
        let seq_len = 5;
        let input = Array3::from_shape_vec(
            (batch_size, seq_len, 64),
            vec![0.1; batch_size * seq_len * 64],
        )
        .unwrap();

        let output = attn.forward_batch(&input, None).unwrap();
        assert_eq!(output.dim(), (batch_size, seq_len, 64));
    }
}
