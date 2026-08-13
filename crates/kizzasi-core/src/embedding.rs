//! Continuous signal embeddings
//!
//! Unlike discrete token embeddings in LLMs, this module provides
//! direct mapping of continuous float values to latent space.

use crate::error::{CoreError, CoreResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;
use serde::{Deserialize, Serialize};

/// Continuous embedding layer for signal values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousEmbedding {
    /// Linear projection weights
    weights: Array2<f32>,
    /// Bias terms
    bias: Array1<f32>,
    /// Input dimension
    input_dim: usize,
    /// Output (embedding) dimension
    embed_dim: usize,
}

impl ContinuousEmbedding {
    /// Create a new continuous embedding layer
    pub fn new(input_dim: usize, embed_dim: usize) -> Self {
        let mut rng = thread_rng();
        // Xavier initialization
        let scale = (2.0 / (input_dim + embed_dim) as f32).sqrt();
        let weights = Array2::from_shape_fn((input_dim, embed_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });
        let bias = Array1::zeros(embed_dim);

        Self {
            weights,
            bias,
            input_dim,
            embed_dim,
        }
    }

    /// Embed a continuous signal vector
    pub fn embed(&self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        if input.len() != self.input_dim {
            return Err(CoreError::DimensionMismatch {
                expected: self.input_dim,
                got: input.len(),
            });
        }

        // Linear projection: output = input @ weights + bias
        let output = input.dot(&self.weights) + &self.bias;
        Ok(output)
    }

    /// Get the embedding dimension
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get the input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Apply layer normalization
    ///
    /// Computes the variance via a single allocation-free iterator sum
    /// (rather than `input.mapv(...).sum()`, which allocated and immediately
    /// dropped a whole extra `hidden_dim`-length array just to reduce it),
    /// and normalizes with one reciprocal multiply per element instead of a
    /// division. Called `num_layers + 1` times per recurrence step from
    /// `SelectiveSSM::recurrence_step`, so this is on a genuinely hot path.
    pub fn layer_norm(input: &Array1<f32>, eps: f32) -> Array1<f32> {
        let n = input.len() as f32;
        let mean = if n > 0.0 { input.sum() / n } else { 0.0 };
        let var = if n > 0.0 {
            input
                .iter()
                .map(|&x| {
                    let d = x - mean;
                    d * d
                })
                .sum::<f32>()
                / n
        } else {
            1.0
        };
        let inv_std = 1.0 / (var + eps).sqrt();
        input.mapv(|x| (x - mean) * inv_std)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_embedding() {
        let embed = ContinuousEmbedding::new(3, 64);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let output = embed.embed(&input).expect("embedding should succeed");
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_layer_norm() {
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normed = ContinuousEmbedding::layer_norm(&input, 1e-5);
        // Check that mean is approximately 0
        let n = normed.len() as f32;
        let mean = if n > 0.0 { normed.sum() / n } else { 0.0 };
        assert!(mean.abs() < 1e-5);
    }
}
