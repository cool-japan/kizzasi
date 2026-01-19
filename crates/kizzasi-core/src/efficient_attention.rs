//! Memory-efficient attention implementations
//!
//! Provides optimized attention mechanisms that reduce memory usage
//! for long sequences through chunked processing and kernel fusion.

use crate::error::{CoreError, CoreResult};
use crate::numerics;
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::random::thread_rng;

/// Configuration for memory-efficient attention
#[derive(Debug, Clone)]
pub struct EfficientAttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Maximum sequence length to process in one chunk
    pub chunk_size: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Dropout rate (for training)
    pub dropout: f32,
}

impl Default for EfficientAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            chunk_size: 512,
            causal: true,
            dropout: 0.0,
        }
    }
}

/// Memory-efficient multi-head attention using chunked processing
///
/// Reduces memory usage from O(n²) to O(chunk_size²) by processing
/// the sequence in chunks, similar to FlashAttention.
pub struct EfficientMultiHeadAttention {
    config: EfficientAttentionConfig,
    // Projection matrices
    wq: Array2<f32>,
    wk: Array2<f32>,
    wv: Array2<f32>,
    wo: Array2<f32>,
}

impl EfficientMultiHeadAttention {
    /// Create a new efficient multi-head attention layer
    pub fn new(config: EfficientAttentionConfig, hidden_dim: usize) -> CoreResult<Self> {
        if !hidden_dim.is_multiple_of(config.num_heads) {
            return Err(CoreError::InvalidConfig(format!(
                "Hidden dim {} must be divisible by num_heads {}",
                hidden_dim, config.num_heads
            )));
        }

        let head_dim = hidden_dim / config.num_heads;

        // Initialize projection matrices using Xavier/Glorot initialization
        let scale = (2.0 / (hidden_dim + head_dim) as f32).sqrt();

        let wq = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (thread_rng().random::<f32>() - 0.5) * 2.0 * scale
        });
        let wk = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (thread_rng().random::<f32>() - 0.5) * 2.0 * scale
        });
        let wv = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (thread_rng().random::<f32>() - 0.5) * 2.0 * scale
        });
        let wo = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (thread_rng().random::<f32>() - 0.5) * 2.0 * scale
        });

        Ok(Self {
            config,
            wq,
            wk,
            wv,
            wo,
        })
    }

    /// Forward pass with chunked processing for memory efficiency
    ///
    /// # Arguments
    /// * `x` - Input of shape [seq_len, hidden_dim]
    ///
    /// # Returns
    /// Output of shape [seq_len, hidden_dim]
    pub fn forward(&self, x: &Array2<f32>) -> CoreResult<Array2<f32>> {
        let (seq_len, hidden_dim) = x.dim();
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;

        // Project to Q, K, V
        let q = x.dot(&self.wq);
        let k = x.dot(&self.wk);
        let v = x.dot(&self.wv);

        // Reshape to [seq_len, num_heads, head_dim]
        let q_heads = self.reshape_to_heads(&q, seq_len, num_heads, head_dim)?;
        let k_heads = self.reshape_to_heads(&k, seq_len, num_heads, head_dim)?;
        let v_heads = self.reshape_to_heads(&v, seq_len, num_heads, head_dim)?;

        // Process attention in chunks to save memory
        let output_heads = self.chunked_attention(&q_heads, &k_heads, &v_heads)?;

        // Reshape back to [seq_len, hidden_dim]
        let output = self.reshape_from_heads(&output_heads, seq_len, hidden_dim)?;

        // Output projection
        let result = output.dot(&self.wo);

        Ok(result)
    }

    /// Reshape tensor to separate heads
    fn reshape_to_heads(
        &self,
        x: &Array2<f32>,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> CoreResult<Array3<f32>> {
        let mut result = Array3::zeros((seq_len, num_heads, head_dim));

        for i in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    result[[i, h, d]] = x[[i, h * head_dim + d]];
                }
            }
        }

        Ok(result)
    }

    /// Reshape from heads back to flat
    fn reshape_from_heads(
        &self,
        x: &Array3<f32>,
        seq_len: usize,
        hidden_dim: usize,
    ) -> CoreResult<Array2<f32>> {
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let mut result = Array2::zeros((seq_len, hidden_dim));

        for i in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    result[[i, h * head_dim + d]] = x[[i, h, d]];
                }
            }
        }

        Ok(result)
    }

    /// Chunked attention computation to reduce memory usage
    fn chunked_attention(
        &self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        v: &Array3<f32>,
    ) -> CoreResult<Array3<f32>> {
        let (seq_len, num_heads, head_dim) = q.dim();
        let chunk_size = self.config.chunk_size.min(seq_len);
        let scale = (head_dim as f32).sqrt();

        let mut output = Array3::zeros((seq_len, num_heads, head_dim));

        // Process in chunks to reduce memory footprint
        let num_chunks = seq_len.div_ceil(chunk_size);

        for chunk_idx in 0..num_chunks {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(seq_len);
            let chunk_len = chunk_end - chunk_start;

            // Extract chunk
            let q_chunk = q.slice(scirs2_core::ndarray::s![chunk_start..chunk_end, .., ..]);

            // Compute attention scores for this chunk
            for i in 0..chunk_len {
                let q_pos = chunk_start + i;

                for h in 0..num_heads {
                    // Compute attention scores against all keys
                    let mut scores = Array1::zeros(seq_len);
                    let mut max_score = f32::NEG_INFINITY;

                    // Determine the range of keys to attend to (causal masking)
                    let k_end = if self.config.causal {
                        q_pos + 1
                    } else {
                        seq_len
                    };

                    // Compute scores
                    for j in 0..k_end {
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_chunk[[i, h, d]] * k[[j, h, d]];
                        }
                        score /= scale;
                        scores[j] = score;
                        max_score = max_score.max(score);
                    }

                    // Apply softmax with numerical stability
                    let mut sum = 0.0f32;
                    for j in 0..k_end {
                        scores[j] = numerics::safe_exp(scores[j] - max_score);
                        sum += scores[j];
                    }

                    if sum > 0.0 {
                        for j in 0..k_end {
                            scores[j] /= sum;
                        }
                    }

                    // Compute weighted sum of values
                    for d in 0..head_dim {
                        let mut weighted_sum = 0.0f32;
                        for j in 0..k_end {
                            weighted_sum += scores[j] * v[[j, h, d]];
                        }
                        output[[q_pos, h, d]] = weighted_sum;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Get configuration
    pub fn config(&self) -> &EfficientAttentionConfig {
        &self.config
    }
}

/// Fused attention kernel that combines QK^T matmul and softmax
///
/// This reduces memory traffic by fusing operations together.
pub struct FusedAttentionKernel;

impl FusedAttentionKernel {
    /// Compute attention with fused operations
    ///
    /// Computes softmax(QK^T / sqrt(d)) V in a fused manner to reduce memory usage.
    pub fn forward(
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
        causal: bool,
    ) -> CoreResult<Array2<f32>> {
        let (seq_len_q, dim) = q.dim();
        let (seq_len_k, _) = k.dim();
        let scale = (dim as f32).sqrt();

        let mut output = Array2::zeros((seq_len_q, dim));

        // Process each query position
        for i in 0..seq_len_q {
            let q_vec = q.row(i);

            // Compute scores and apply softmax in one pass
            let k_end = if causal { i + 1 } else { seq_len_k };
            let mut scores = Array1::zeros(k_end);
            let mut max_score = f32::NEG_INFINITY;

            // Compute scores
            for j in 0..k_end {
                let k_vec = k.row(j);
                let score = q_vec.dot(&k_vec) / scale;
                scores[j] = score;
                max_score = max_score.max(score);
            }

            // Softmax with numerical stability
            let mut sum = 0.0f32;
            for j in 0..k_end {
                scores[j] = numerics::safe_exp(scores[j] - max_score);
                sum += scores[j];
            }

            if sum > 0.0 {
                for j in 0..k_end {
                    scores[j] /= sum;
                }
            }

            // Compute weighted sum of values
            for j in 0..k_end {
                let v_vec = v.row(j);
                let weight = scores[j];
                for d in 0..dim {
                    output[[i, d]] += weight * v_vec[d];
                }
            }
        }

        Ok(output)
    }

    /// Sequential version of fused attention (parallel version removed due to rayon dependency)
    ///
    /// For now, this is identical to `forward` but kept for API compatibility.
    pub fn forward_parallel(
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
        causal: bool,
    ) -> CoreResult<Array2<f32>> {
        // Use sequential version for now
        // TODO: Add true parallel implementation when rayon support is available in scirs2
        Self::forward(q, k, v, causal)
    }

    #[allow(dead_code)]
    fn forward_parallel_internal(
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
        causal: bool,
    ) -> CoreResult<Array2<f32>> {
        let (seq_len_q, dim) = q.dim();
        let (seq_len_k, _) = k.dim();
        let scale = (dim as f32).sqrt();

        // Process positions sequentially for now
        let rows: Vec<Array1<f32>> = (0..seq_len_q)
            .map(|i| {
                let q_vec = q.row(i);
                let k_end = if causal { i + 1 } else { seq_len_k };

                // Compute scores
                let mut scores = Array1::zeros(k_end);
                let mut max_score = f32::NEG_INFINITY;

                for j in 0..k_end {
                    let k_vec = k.row(j);
                    let score = q_vec.dot(&k_vec) / scale;
                    scores[j] = score;
                    max_score = max_score.max(score);
                }

                // Softmax
                let mut sum = 0.0f32;
                for j in 0..k_end {
                    scores[j] = numerics::safe_exp(scores[j] - max_score);
                    sum += scores[j];
                }

                if sum > 0.0 {
                    for j in 0..k_end {
                        scores[j] /= sum;
                    }
                }

                // Weighted sum
                let mut output_row = Array1::zeros(dim);
                for j in 0..k_end {
                    let v_vec = v.row(j);
                    let weight = scores[j];
                    for d in 0..dim {
                        output_row[d] += weight * v_vec[d];
                    }
                }

                output_row
            })
            .collect();

        // Stack rows into output matrix
        let mut output = Array2::zeros((seq_len_q, dim));
        for (i, row) in rows.iter().enumerate() {
            output.row_mut(i).assign(row);
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efficient_attention_creation() {
        let config = EfficientAttentionConfig::default();
        let hidden_dim = 512;

        let attn = EfficientMultiHeadAttention::new(config, hidden_dim);
        assert!(attn.is_ok());
    }

    #[test]
    fn test_efficient_attention_forward() {
        let config = EfficientAttentionConfig {
            num_heads: 4,
            head_dim: 16,
            chunk_size: 64,
            causal: true,
            dropout: 0.0,
        };
        let hidden_dim = 64; // 4 heads * 16 dim/head

        let attn = EfficientMultiHeadAttention::new(config, hidden_dim).unwrap();

        let seq_len = 32;
        let x = Array2::from_shape_fn((seq_len, hidden_dim), |_| thread_rng().random::<f32>());

        let output = attn.forward(&x);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.dim(), (seq_len, hidden_dim));
    }

    #[test]
    fn test_fused_attention_basic() {
        let seq_len = 8;
        let dim = 16;

        let q = Array2::from_shape_fn((seq_len, dim), |_| {
            scirs2_core::random::thread_rng().random::<f32>()
        });
        let k = Array2::from_shape_fn((seq_len, dim), |_| {
            scirs2_core::random::thread_rng().random::<f32>()
        });
        let v = Array2::from_shape_fn((seq_len, dim), |_| {
            scirs2_core::random::thread_rng().random::<f32>()
        });

        let output = FusedAttentionKernel::forward(&q, &k, &v, true);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.dim(), (seq_len, dim));
    }

    #[test]
    fn test_fused_attention_causal_vs_non_causal() {
        let seq_len = 4;
        let dim = 8;

        // Use varying values so the difference is visible
        let q = Array2::from_shape_fn((seq_len, dim), |(i, j)| i as f32 + j as f32 * 0.1);
        let k = Array2::from_shape_fn((seq_len, dim), |(i, j)| i as f32 * 0.5 + j as f32 * 0.2);
        let v = Array2::from_shape_fn((seq_len, dim), |(i, j)| (i + j) as f32 * 0.3);

        let causal_output = FusedAttentionKernel::forward(&q, &k, &v, true).unwrap();
        let non_causal_output = FusedAttentionKernel::forward(&q, &k, &v, false).unwrap();

        // Causal and non-causal should give different results
        let diff = (&causal_output - &non_causal_output)
            .mapv(|x| x.abs())
            .sum();

        assert!(
            diff > 0.01,
            "Causal and non-causal outputs should differ, got diff={}",
            diff
        );
    }

    #[test]
    fn test_fused_attention_parallel() {
        let seq_len = 16;
        let dim = 32;

        let q = Array2::from_shape_fn((seq_len, dim), |_| {
            scirs2_core::random::thread_rng().random::<f32>()
        });
        let k = Array2::from_shape_fn((seq_len, dim), |_| {
            scirs2_core::random::thread_rng().random::<f32>()
        });
        let v = Array2::from_shape_fn((seq_len, dim), |_| {
            scirs2_core::random::thread_rng().random::<f32>()
        });

        let output_seq = FusedAttentionKernel::forward(&q, &k, &v, false).unwrap();
        let output_par = FusedAttentionKernel::forward_parallel(&q, &k, &v, false).unwrap();

        // Sequential and parallel should give same results
        let diff = (&output_seq - &output_par).mapv(|x| x.abs()).sum();

        assert!(diff < 1e-3, "Sequential and parallel outputs should match");
    }

    #[test]
    fn test_chunked_attention_dimensions() {
        let config = EfficientAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            chunk_size: 16,
            causal: false,
            dropout: 0.0,
        };

        let hidden_dim = 16;
        let seq_len = 64;

        let attn = EfficientMultiHeadAttention::new(config, hidden_dim).unwrap();

        // Use fixed seed data for reproducibility
        let x = Array2::from_shape_fn((seq_len, hidden_dim), |(i, j)| {
            ((i * 7 + j * 3) % 100) as f32 / 100.0
        });

        let output = attn.forward(&x).unwrap();

        // Check output dimensions are correct
        assert_eq!(output.dim(), (seq_len, hidden_dim));

        // Check that output values are reasonable (not NaN or Inf)
        for val in output.iter() {
            assert!(val.is_finite(), "Output contains non-finite values");
        }
    }
}
