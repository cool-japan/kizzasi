//! Fused Kernel Optimizations
//!
//! This module provides fused operations that combine multiple primitive operations
//! into single kernels to reduce memory bandwidth and improve cache utilization.
//!
//! # Performance Benefits
//!
//! - Reduced memory traffic (fewer loads/stores)
//! - Improved cache locality
//! - Better instruction-level parallelism
//! - Lower kernel launch overhead (for GPU)
//!
//! # Available Fusions
//!
//! - LayerNorm + Activation (GELU, SiLU, etc.)
//! - Attention QKV projection (single matmul)
//! - FFN (Linear + Activation + Linear)
//! - SSM step (Discretize + Update + Output)
//! - Quantize + Dequantize (for mixed precision)

use crate::error::{CoreError, CoreResult};
use crate::numerics::safe_exp;
use crate::simd::dot_product;

/// Numerically stable softmax for slices
///
/// Computes softmax with max subtraction for numerical stability.
fn stable_softmax_slice(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_x: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_x.iter().sum();
    exp_x.iter().map(|&v| v / sum).collect()
}

/// Fused LayerNorm + GELU activation
///
/// Combines layer normalization and GELU activation into a single pass.
/// This is common in transformer architectures.
///
/// # Arguments
///
/// * `x` - Input vector
/// * `gamma` - Scale parameter
/// * `beta` - Shift parameter
/// * `eps` - Epsilon for numerical stability
///
/// # Returns
///
/// Normalized and activated output
pub fn fused_layernorm_gelu(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> CoreResult<Vec<f32>> {
    let n = x.len();
    if gamma.len() != n || beta.len() != n {
        return Err(CoreError::DimensionMismatch {
            expected: n,
            got: gamma.len(),
        });
    }

    // Single pass: compute mean and variance
    let sum = x.iter().sum::<f32>();
    let mean = sum / n as f32;

    let var_sum = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>();
    let variance = var_sum / n as f32;
    let std_inv = 1.0 / (variance + eps).sqrt();

    // Fused normalization + GELU
    let output: Vec<f32> = x
        .iter()
        .zip(gamma)
        .zip(beta)
        .map(|((&xi, &g), &b)| {
            // Normalize
            let normalized = (xi - mean) * std_inv;
            let scaled = normalized * g + b;

            // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            let x3 = scaled * scaled * scaled;
            let inner: f32 = 0.797_884_6 * (scaled + 0.044715 * x3);
            0.5 * scaled * (1.0 + inner.tanh())
        })
        .collect();

    Ok(output)
}

/// Fused LayerNorm + SiLU (Swish) activation
///
/// Combines layer normalization and SiLU activation.
///
/// # Arguments
///
/// * `x` - Input vector
/// * `gamma` - Scale parameter
/// * `beta` - Shift parameter
/// * `eps` - Epsilon for numerical stability
pub fn fused_layernorm_silu(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> CoreResult<Vec<f32>> {
    let n = x.len();
    if gamma.len() != n || beta.len() != n {
        return Err(CoreError::DimensionMismatch {
            expected: n,
            got: gamma.len(),
        });
    }

    // Single pass: compute mean and variance
    let sum = x.iter().sum::<f32>();
    let mean = sum / n as f32;

    let var_sum = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>();
    let variance = var_sum / n as f32;
    let std_inv = 1.0 / (variance + eps).sqrt();

    // Fused normalization + SiLU
    let output: Vec<f32> = x
        .iter()
        .zip(gamma)
        .zip(beta)
        .map(|((&xi, &g), &b)| {
            // Normalize
            let normalized = (xi - mean) * std_inv;
            let scaled = normalized * g + b;

            // SiLU: x * sigmoid(x)
            scaled / (1.0 + (-scaled).exp())
        })
        .collect();

    Ok(output)
}

/// Fused QKV projection for attention
///
/// Combines Query, Key, Value projections into a single matrix multiplication.
/// This reduces three separate matmuls to one, saving memory bandwidth.
///
/// # Arguments
///
/// * `x` - Input vector (d_model,)
/// * `w_qkv` - Combined QKV weight matrix (3*d_model, d_model)
/// * `d_model` - Model dimension
///
/// # Returns
///
/// Tuple of (Q, K, V) vectors, each of size d_model
pub fn fused_qkv_projection(
    x: &[f32],
    w_qkv: &[f32],
    d_model: usize,
) -> CoreResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if x.len() != d_model {
        return Err(CoreError::DimensionMismatch {
            expected: d_model,
            got: x.len(),
        });
    }

    if w_qkv.len() != 3 * d_model * d_model {
        return Err(CoreError::DimensionMismatch {
            expected: 3 * d_model * d_model,
            got: w_qkv.len(),
        });
    }

    // Single matmul: [3*d_model, d_model] @ [d_model] = [3*d_model]
    let mut qkv = vec![0.0; 3 * d_model];

    for i in 0..3 * d_model {
        let row = &w_qkv[i * d_model..(i + 1) * d_model];
        qkv[i] = dot_product(row, x);
    }

    // Split into Q, K, V
    let q = qkv[0..d_model].to_vec();
    let k = qkv[d_model..2 * d_model].to_vec();
    let v = qkv[2 * d_model..3 * d_model].to_vec();

    Ok((q, k, v))
}

/// Fused FFN (Feed-Forward Network)
///
/// Combines two linear layers with activation in between:
/// FFN(x) = W2 * activation(W1 * x + b1) + b2
///
/// # Arguments
///
/// * `x` - Input vector
/// * `w1` - First layer weights
/// * `b1` - First layer bias
/// * `w2` - Second layer weights
/// * `b2` - Second layer bias
/// * `d_model` - Model dimension
/// * `d_ff` - FFN hidden dimension
pub fn fused_ffn_gelu(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    d_model: usize,
    d_ff: usize,
) -> CoreResult<Vec<f32>> {
    if x.len() != d_model {
        return Err(CoreError::DimensionMismatch {
            expected: d_model,
            got: x.len(),
        });
    }

    // First linear + GELU activation
    let mut hidden = vec![0.0; d_ff];
    for i in 0..d_ff {
        let row = &w1[i * d_model..(i + 1) * d_model];
        let linear = dot_product(row, x) + b1[i];

        // GELU activation
        let x3 = linear * linear * linear;
        let inner: f32 = 0.797_884_6 * (linear + 0.044715 * x3);
        hidden[i] = 0.5 * linear * (1.0 + inner.tanh());
    }

    // Second linear
    let mut output = vec![0.0; d_model];
    for i in 0..d_model {
        let row = &w2[i * d_ff..(i + 1) * d_ff];
        output[i] = dot_product(row, &hidden) + b2[i];
    }

    Ok(output)
}

/// Fused SSM step
///
/// Combines discretization, state update, and output computation:
/// 1. Discretize: Ā = exp(Δ * A), B̄ = Δ * B
/// 2. Update: h' = Ā * h + B̄ * x
/// 3. Output: y = C * h' + D * x
///
/// # Arguments
///
/// * `h` - Current hidden state
/// * `x` - Input value
/// * `a` - State transition parameter (diagonal)
/// * `b` - Input mixing parameter
/// * `c` - Output projection parameter
/// * `d` - Skip connection parameter
/// * `delta` - Time step
pub fn fused_ssm_step(
    h: &mut [f32],
    x: f32,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: f32,
    delta: f32,
) -> CoreResult<f32> {
    let n = h.len();
    if a.len() != n || b.len() != n || c.len() != n {
        return Err(CoreError::DimensionMismatch {
            expected: n,
            got: a.len(),
        });
    }

    // Fused discretization + state update
    let mut y = 0.0;
    for i in 0..n {
        // Discretize and update in one step
        let a_discrete = safe_exp(delta * a[i]);
        let b_discrete = delta * b[i];

        // Update state
        h[i] = a_discrete * h[i] + b_discrete * x;

        // Accumulate output
        y += c[i] * h[i];
    }

    // Add skip connection
    y += d * x;

    Ok(y)
}

/// Fused quantize-dequantize operation
///
/// Simulates quantization effects without actually storing quantized values.
/// Useful for QAT (Quantization-Aware Training).
///
/// # Arguments
///
/// * `x` - Input values
/// * `bits` - Number of bits (4 or 8)
/// * `symmetric` - Use symmetric quantization
pub fn fused_quantize_dequantize(x: &[f32], bits: u8, symmetric: bool) -> CoreResult<Vec<f32>> {
    if bits != 4 && bits != 8 {
        return Err(CoreError::Generic(
            "Only 4-bit and 8-bit quantization supported".to_string(),
        ));
    }

    let q_max = (1 << bits) - 1;
    let q_max_f = q_max as f32;

    if symmetric {
        // Symmetric: [-max, max] -> [0, q_max]
        let abs_max = x.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
        let scale = abs_max / (q_max_f / 2.0);

        if scale == 0.0 {
            return Ok(vec![0.0; x.len()]);
        }

        Ok(x.iter()
            .map(|&v| {
                let q = (v / scale).round().clamp(-(q_max_f / 2.0), q_max_f / 2.0);
                q * scale
            })
            .collect())
    } else {
        // Asymmetric: [min, max] -> [0, q_max]
        let min = x.iter().copied().fold(f32::INFINITY, f32::min);
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max - min) / q_max_f;

        if scale == 0.0 {
            return Ok(vec![min; x.len()]);
        }

        Ok(x.iter()
            .map(|&v| {
                let q = ((v - min) / scale).round().clamp(0.0, q_max_f);
                q * scale + min
            })
            .collect())
    }
}

/// Fused matrix-vector multiplication with bias and activation
///
/// Combines y = activation(W @ x + b) into a single fused operation.
///
/// # Arguments
///
/// * `x` - Input vector
/// * `w` - Weight matrix (row-major)
/// * `b` - Bias vector
/// * `rows` - Number of output dimensions
/// * `cols` - Number of input dimensions
/// * `activation` - Activation function (0=none, 1=relu, 2=gelu, 3=silu)
pub fn fused_linear_activation(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    rows: usize,
    cols: usize,
    activation: u8,
) -> CoreResult<Vec<f32>> {
    if x.len() != cols || w.len() != rows * cols || b.len() != rows {
        return Err(CoreError::DimensionMismatch {
            expected: cols,
            got: x.len(),
        });
    }

    let mut output = vec![0.0; rows];

    for i in 0..rows {
        let row = &w[i * cols..(i + 1) * cols];
        let mut y = dot_product(row, x) + b[i];

        // Apply activation
        y = match activation {
            0 => y,          // None
            1 => y.max(0.0), // ReLU
            2 => {
                // GELU
                let x3 = y * y * y;
                let inner: f32 = 0.797_884_6 * (y + 0.044715 * x3);
                0.5 * y * (1.0 + inner.tanh())
            }
            3 => y / (1.0 + (-y).exp()), // SiLU
            _ => {
                return Err(CoreError::Generic(
                    "Unknown activation function".to_string(),
                ))
            }
        };

        output[i] = y;
    }

    Ok(output)
}

/// Fused softmax with attention weights
///
/// Combines softmax computation with weighted sum in a single pass.
/// This is the core of attention mechanism.
///
/// # Arguments
///
/// * `scores` - Attention scores (pre-softmax)
/// * `values` - Value vectors to attend to
/// * `d_k` - Dimension of each value vector
pub fn fused_softmax_attend(scores: &[f32], values: &[f32], d_k: usize) -> CoreResult<Vec<f32>> {
    let seq_len = scores.len();
    if values.len() != seq_len * d_k {
        return Err(CoreError::DimensionMismatch {
            expected: seq_len * d_k,
            got: values.len(),
        });
    }

    // Compute softmax weights
    let weights = stable_softmax_slice(scores);

    // Weighted sum of values
    let mut output = vec![0.0; d_k];
    for (i, &w) in weights.iter().enumerate() {
        let value = &values[i * d_k..(i + 1) * d_k];
        for j in 0..d_k {
            output[j] += w * value[j];
        }
    }

    Ok(output)
}

/// Fused multi-head attention projection
///
/// Combines multi-head concatenation and output projection:
/// O = W_o @ concat(head_1, ..., head_h)
///
/// # Arguments
///
/// * `heads` - All attention heads concatenated
/// * `w_o` - Output projection matrix
/// * `num_heads` - Number of attention heads
/// * `d_k` - Dimension per head
pub fn fused_multihead_output(
    heads: &[f32],
    w_o: &[f32],
    num_heads: usize,
    d_k: usize,
) -> CoreResult<Vec<f32>> {
    let d_model = num_heads * d_k;

    if heads.len() != d_model || w_o.len() != d_model * d_model {
        return Err(CoreError::DimensionMismatch {
            expected: d_model,
            got: heads.len(),
        });
    }

    // Single matmul instead of concat + matmul
    let mut output = vec![0.0; d_model];
    for i in 0..d_model {
        let row = &w_o[i * d_model..(i + 1) * d_model];
        output[i] = dot_product(row, heads);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_layernorm_gelu() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];

        let result = fused_layernorm_gelu(&x, &gamma, &beta, 1e-5).unwrap();

        assert_eq!(result.len(), 4);
        // Verify all values are finite
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_fused_layernorm_silu() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];

        let result = fused_layernorm_silu(&x, &gamma, &beta, 1e-5).unwrap();

        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_fused_qkv_projection() {
        let d_model = 4;
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w_qkv = vec![1.0; 3 * d_model * d_model];

        let (q, k, v) = fused_qkv_projection(&x, &w_qkv, d_model).unwrap();

        assert_eq!(q.len(), d_model);
        assert_eq!(k.len(), d_model);
        assert_eq!(v.len(), d_model);
        assert!(q.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_fused_ffn_gelu() {
        let d_model = 4;
        let d_ff = 8;
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w1 = vec![0.1; d_ff * d_model];
        let b1 = vec![0.0; d_ff];
        let w2 = vec![0.1; d_model * d_ff];
        let b2 = vec![0.0; d_model];

        let result = fused_ffn_gelu(&x, &w1, &b1, &w2, &b2, d_model, d_ff).unwrap();

        assert_eq!(result.len(), d_model);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_fused_ssm_step() {
        let mut h = vec![0.0, 0.0, 0.0, 0.0];
        let a = vec![-1.0, -1.0, -1.0, -1.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];
        let d = 0.1;
        let delta = 0.01;

        let y = fused_ssm_step(&mut h, 1.0, &a, &b, &c, d, delta).unwrap();

        assert!(y.is_finite());
        assert!(h.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_fused_quantize_dequantize_symmetric() {
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let result = fused_quantize_dequantize(&x, 8, true).unwrap();

        assert_eq!(result.len(), x.len());
        // Values should be close to original
        for (orig, quant) in x.iter().zip(result.iter()) {
            assert!((orig - quant).abs() < 0.1);
        }
    }

    #[test]
    fn test_fused_quantize_dequantize_asymmetric() {
        let x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let result = fused_quantize_dequantize(&x, 8, false).unwrap();

        assert_eq!(result.len(), x.len());
        for (orig, quant) in x.iter().zip(result.iter()) {
            assert!((orig - quant).abs() < 0.1);
        }
    }

    #[test]
    fn test_fused_linear_activation_none() {
        let x = vec![1.0, 2.0];
        let w = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let b = vec![0.0, 0.0];

        let result = fused_linear_activation(&x, &w, &b, 2, 2, 0).unwrap();

        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_fused_linear_activation_relu() {
        let x = vec![1.0, -1.0];
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![0.0, 0.0];

        let result = fused_linear_activation(&x, &w, &b, 2, 2, 1).unwrap();

        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.0); // ReLU clamps negative
    }

    #[test]
    fn test_fused_softmax_attend() {
        let scores = vec![1.0, 2.0, 3.0];
        let values = vec![
            1.0, 0.0, // value 1
            0.0, 1.0, // value 2
            0.5, 0.5, // value 3
        ];
        let d_k = 2;

        let result = fused_softmax_attend(&scores, &values, d_k).unwrap();

        assert_eq!(result.len(), d_k);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_fused_multihead_output() {
        let num_heads = 2;
        let d_k = 2;
        let d_model = num_heads * d_k;

        let heads = vec![1.0, 2.0, 3.0, 4.0];
        let w_o = vec![1.0; d_model * d_model];

        let result = fused_multihead_output(&heads, &w_o, num_heads, d_k).unwrap();

        assert_eq!(result.len(), d_model);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_fused_operations_dimension_check() {
        let x = vec![1.0, 2.0];
        let gamma = vec![1.0]; // Wrong size
        let beta = vec![0.0];

        assert!(fused_layernorm_gelu(&x, &gamma, &beta, 1e-5).is_err());
    }
}
