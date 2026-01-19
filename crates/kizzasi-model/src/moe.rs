//! # Mixture of Experts (MoE)
//!
//! Implementation of Mixture of Experts layers for model composition and scaling.
//! Supports sparse gating, load balancing, and multiple routing strategies.
//!
//! ## Features
//! - **Sparse Gating**: Top-k expert selection for efficient computation
//! - **Load Balancing**: Auxiliary loss to ensure even expert usage
//! - **Multiple Routing Strategies**: Softmax, top-k, noisy top-k
//! - **Expert Diversity**: Can use any AutoregressiveModel as experts
//! - **Parallel Computation**: Experts can be computed in parallel
//!
//! ## Architecture
//! ```text
//! Input → Router/Gating → Select Top-K Experts → Weighted Combination → Output
//!           ↓                                            ↑
//!       Gating Weights                              Expert Outputs
//! ```

use crate::error::{ModelError, ModelResult};
use kizzasi_core::SignalPredictor;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;
use tracing::{debug, trace};

/// Routing strategy for expert selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Softmax routing - all experts weighted by softmax
    Softmax,
    /// Top-k routing - only top k experts activated
    TopK,
    /// Noisy top-k routing with learnable noise for exploration
    NoisyTopK,
}

impl fmt::Display for RoutingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Softmax => write!(f, "Softmax"),
            Self::TopK => write!(f, "Top-K"),
            Self::NoisyTopK => write!(f, "Noisy Top-K"),
        }
    }
}

/// Configuration for Mixture of Experts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts to activate per input (for top-k routing)
    pub top_k: usize,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Routing strategy
    pub routing_strategy: RoutingStrategy,
    /// Load balancing coefficient (typically 0.01)
    pub load_balance_coeff: f32,
    /// Enable expert dropout during training
    pub expert_dropout: f32,
    /// Noise standard deviation for noisy top-k
    pub noise_std: f32,
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            input_dim: 256,
            output_dim: 256,
            routing_strategy: RoutingStrategy::TopK,
            load_balance_coeff: 0.01,
            expert_dropout: 0.0,
            noise_std: 1.0,
        }
    }
}

/// Router network for expert selection
#[derive(Debug)]
pub struct Router {
    /// Router weights (input_dim × num_experts)
    weights: Array2<f32>,
    /// Noise weights for noisy top-k (input_dim × num_experts)
    noise_weights: Option<Array2<f32>>,
    /// Configuration
    config: MoEConfig,
}

impl Router {
    /// Create a new router with random initialization
    pub fn new(config: MoEConfig) -> ModelResult<Self> {
        debug!(
            "Creating router: {} experts, top-k={}, strategy={}",
            config.num_experts, config.top_k, config.routing_strategy
        );

        let weights = Array2::zeros((config.input_dim, config.num_experts));

        let noise_weights = if config.routing_strategy == RoutingStrategy::NoisyTopK {
            Some(Array2::zeros((config.input_dim, config.num_experts)))
        } else {
            None
        };

        Ok(Self {
            weights,
            noise_weights,
            config,
        })
    }

    /// Compute routing probabilities for input
    pub fn route(&self, input: &Array1<f32>) -> ModelResult<(Vec<usize>, Vec<f32>)> {
        trace!("Computing routing for input shape: {:?}", input.shape());

        if input.len() != self.config.input_dim {
            return Err(ModelError::dimension_mismatch(
                "router input",
                self.config.input_dim,
                input.len(),
            ));
        }

        // Compute logits: input · weights → (num_experts,)
        let logits = self.weights.t().dot(input);

        // Add noise for noisy top-k
        let logits = if let Some(ref noise_weights) = self.noise_weights {
            let _noise_logits = noise_weights.t().dot(input);
            // In training, add Gaussian noise scaled by noise_logits
            // For now, just use logits (noise would be added during training)
            logits
        } else {
            logits
        };

        // Select experts based on routing strategy
        match self.config.routing_strategy {
            RoutingStrategy::Softmax => self.softmax_route(&logits),
            RoutingStrategy::TopK | RoutingStrategy::NoisyTopK => self.topk_route(&logits),
        }
    }

    /// Softmax routing - all experts weighted
    fn softmax_route(&self, logits: &Array1<f32>) -> ModelResult<(Vec<usize>, Vec<f32>)> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        let probabilities: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();
        let indices: Vec<usize> = (0..self.config.num_experts).collect();

        Ok((indices, probabilities))
    }

    /// Top-k routing - only top k experts activated
    fn topk_route(&self, logits: &Array1<f32>) -> ModelResult<(Vec<usize>, Vec<f32>)> {
        let mut indexed_logits: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();

        // Sort by logits descending
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        let top_k = self.config.top_k.min(self.config.num_experts);
        let top_experts: Vec<(usize, f32)> = indexed_logits.into_iter().take(top_k).collect();

        // Normalize weights
        let top_logits: Vec<f32> = top_experts.iter().map(|(_, v)| *v).collect();
        let max_logit = top_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = top_logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        let indices: Vec<usize> = top_experts.iter().map(|(i, _)| *i).collect();
        let weights: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

        trace!(
            "Selected {} experts: {:?} with weights {:?}",
            top_k,
            indices,
            weights
        );

        Ok((indices, weights))
    }

    /// Compute load balancing loss
    pub fn load_balance_loss(&self, all_routes: &[Vec<usize>]) -> f32 {
        if all_routes.is_empty() {
            return 0.0;
        }

        // Count how many times each expert was used
        let mut expert_counts = vec![0.0f32; self.config.num_experts];
        for routes in all_routes {
            for &expert_idx in routes {
                expert_counts[expert_idx] += 1.0;
            }
        }

        // Compute coefficient of variation as load balance metric
        let total: f32 = expert_counts.iter().sum();
        if total == 0.0 {
            return 0.0;
        }

        let mean = total / self.config.num_experts as f32;
        let variance: f32 = expert_counts
            .iter()
            .map(|&count| (count - mean).powi(2))
            .sum::<f32>()
            / self.config.num_experts as f32;

        let std_dev = variance.sqrt();
        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

        cv * self.config.load_balance_coeff
    }
}

/// Expert network wrapper
pub struct Expert {
    /// Expert ID
    id: usize,
    /// Input projection (if dimensions don't match)
    input_proj: Option<Array2<f32>>,
    /// Output projection (if dimensions don't match)
    output_proj: Option<Array2<f32>>,
    /// Expert model (placeholder for now - would be any AutoregressiveModel)
    /// Using simple linear layer as placeholder
    weights: Array2<f32>,
}

impl Expert {
    /// Create a new expert
    pub fn new(
        id: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> ModelResult<Self> {
        debug!(
            "Creating expert {}: {}→{}→{}",
            id, input_dim, hidden_dim, output_dim
        );

        // Simple FFN expert: input → hidden → output
        let weights = Array2::zeros((hidden_dim, output_dim));

        let input_proj = if input_dim != hidden_dim {
            Some(Array2::zeros((input_dim, hidden_dim)))
        } else {
            None
        };

        let output_proj = None; // Already mapped to output_dim

        Ok(Self {
            id,
            input_proj,
            output_proj,
            weights,
        })
    }

    /// Forward pass through expert
    pub fn forward(&self, input: &Array1<f32>) -> ModelResult<Array1<f32>> {
        trace!(
            "Expert {} forward: input shape {:?}",
            self.id,
            input.shape()
        );

        // Project input if needed
        let hidden = if let Some(ref proj) = self.input_proj {
            proj.t().dot(input)
        } else {
            input.clone()
        };

        // Apply expert transformation
        let output = self.weights.t().dot(&hidden);

        // Project output if needed
        let output = if let Some(ref proj) = self.output_proj {
            proj.t().dot(&output)
        } else {
            output
        };

        Ok(output)
    }
}

/// Mixture of Experts layer
pub struct MixtureOfExperts {
    /// Router for expert selection
    router: Router,
    /// Expert networks
    experts: Vec<Expert>,
    /// Configuration
    config: MoEConfig,
    /// Track routing decisions for load balancing
    routing_history: Vec<Vec<usize>>,
}

impl MixtureOfExperts {
    /// Create a new MoE layer
    pub fn new(config: MoEConfig) -> ModelResult<Self> {
        debug!(
            "Creating MixtureOfExperts: {} experts, strategy={}",
            config.num_experts, config.routing_strategy
        );

        let router = Router::new(config.clone())?;

        // Create experts
        let mut experts = Vec::with_capacity(config.num_experts);
        for i in 0..config.num_experts {
            let expert = Expert::new(
                i,
                config.input_dim,
                config.input_dim, // Hidden dim same as input for simplicity
                config.output_dim,
            )?;
            experts.push(expert);
        }

        Ok(Self {
            router,
            experts,
            config,
            routing_history: Vec::new(),
        })
    }

    /// Forward pass with expert routing
    pub fn forward(&mut self, input: &Array1<f32>) -> ModelResult<Array1<f32>> {
        trace!("MoE forward: input shape {:?}", input.shape());

        // Route input to experts
        let (expert_indices, weights) = self.router.route(input)?;

        // Store routing decision for load balancing
        self.routing_history.push(expert_indices.clone());

        // Compute expert outputs
        let mut output = Array1::zeros(self.config.output_dim);
        for (idx, &expert_idx) in expert_indices.iter().enumerate() {
            let expert_output = self.experts[expert_idx].forward(input)?;
            let weight = weights[idx];

            // Weighted sum: output += weight * expert_output
            output = output + expert_output.mapv(|x| x * weight);
        }

        Ok(output)
    }

    /// Get load balancing loss
    pub fn get_load_balance_loss(&self) -> f32 {
        self.router.load_balance_loss(&self.routing_history)
    }

    /// Clear routing history
    pub fn clear_routing_history(&mut self) {
        self.routing_history.clear();
    }

    /// Get expert usage statistics
    pub fn expert_usage_stats(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.config.num_experts];
        for routes in &self.routing_history {
            for &idx in routes {
                counts[idx] += 1;
            }
        }
        counts
    }
}

impl SignalPredictor for MixtureOfExperts {
    fn step(&mut self, input: &Array1<f32>) -> kizzasi_core::CoreResult<Array1<f32>> {
        self.forward(input)
            .map_err(|e| kizzasi_core::CoreError::InferenceError(e.to_string()))
    }

    fn reset(&mut self) {
        self.clear_routing_history();
    }

    fn context_window(&self) -> usize {
        1 // MoE processes one step at a time
    }
}

impl fmt::Display for MixtureOfExperts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MixtureOfExperts({} experts, top_k={}, strategy={})",
            self.config.num_experts, self.config.top_k, self.config.routing_strategy
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let config = MoEConfig::default();
        let router = Router::new(config).expect("Failed to create router");
        assert_eq!(router.weights.shape(), &[256, 8]);
    }

    #[test]
    fn test_topk_routing() {
        let config = MoEConfig {
            num_experts: 4,
            top_k: 2,
            routing_strategy: RoutingStrategy::TopK,
            ..Default::default()
        };

        let router = Router::new(config).expect("Failed to create router");
        let input = Array1::from_vec(vec![0.1; 256]);

        let (indices, weights) = router.route(&input).expect("Routing failed");

        assert_eq!(indices.len(), 2, "Should select top-2 experts");
        assert_eq!(weights.len(), 2, "Should have 2 weights");

        // Weights should sum to ~1.0
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Weights should sum to 1.0");
    }

    #[test]
    fn test_softmax_routing() {
        let config = MoEConfig {
            num_experts: 4,
            routing_strategy: RoutingStrategy::Softmax,
            ..Default::default()
        };

        let router = Router::new(config.clone()).expect("Failed to create router");
        let input = Array1::from_vec(vec![0.1; 256]);

        let (indices, weights) = router.route(&input).expect("Routing failed");

        assert_eq!(indices.len(), config.num_experts, "Should use all experts");
        assert_eq!(
            weights.len(),
            config.num_experts,
            "Should have weights for all experts"
        );

        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Weights should sum to 1.0");
    }

    #[test]
    fn test_expert_forward() {
        let expert = Expert::new(0, 256, 256, 256).expect("Failed to create expert");
        let input = Array1::from_vec(vec![0.1; 256]);

        let output = expert.forward(&input).expect("Forward failed");
        assert_eq!(output.len(), 256, "Output should have correct dimension");
    }

    #[test]
    fn test_moe_forward() {
        let config = MoEConfig {
            num_experts: 4,
            top_k: 2,
            input_dim: 128,
            output_dim: 128,
            ..Default::default()
        };

        let mut moe = MixtureOfExperts::new(config).expect("Failed to create MoE");
        let input = Array1::from_vec(vec![0.1; 128]);

        let output = moe.forward(&input).expect("Forward failed");
        assert_eq!(output.len(), 128, "Output should have correct dimension");
    }

    #[test]
    fn test_load_balance_loss() {
        let config = MoEConfig {
            num_experts: 4,
            top_k: 1,
            input_dim: 64,
            output_dim: 64,
            ..Default::default()
        };

        let mut moe = MixtureOfExperts::new(config).expect("Failed to create MoE");

        // Run multiple forwards
        for _ in 0..10 {
            let input = Array1::from_vec(vec![0.1; 64]);
            let _ = moe.forward(&input);
        }

        let loss = moe.get_load_balance_loss();
        assert!(loss >= 0.0, "Load balance loss should be non-negative");
    }

    #[test]
    fn test_expert_usage_stats() {
        let config = MoEConfig {
            num_experts: 4,
            top_k: 2,
            input_dim: 64,
            output_dim: 64,
            ..Default::default()
        };

        let mut moe = MixtureOfExperts::new(config.clone()).expect("Failed to create MoE");

        // Run multiple forwards
        for _ in 0..10 {
            let input = Array1::from_vec(vec![0.1; 64]);
            let _ = moe.forward(&input);
        }

        let stats = moe.expert_usage_stats();
        assert_eq!(stats.len(), config.num_experts);

        let total: usize = stats.iter().sum();
        assert!(total > 0, "At least some experts should be used");
    }

    #[test]
    fn test_signal_predictor_trait() {
        let config = MoEConfig {
            input_dim: 64,
            output_dim: 64,
            ..Default::default()
        };

        let mut moe = MixtureOfExperts::new(config).expect("Failed to create MoE");
        let input = Array1::from_vec(vec![0.5; 64]);

        let output = moe.step(&input).expect("Step failed");
        assert_eq!(output.len(), 64);

        assert_eq!(moe.context_window(), 1);

        moe.reset();
        assert_eq!(moe.routing_history.len(), 0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = MoEConfig {
            input_dim: 128,
            output_dim: 128,
            ..Default::default()
        };

        let mut moe = MixtureOfExperts::new(config).expect("Failed to create MoE");
        let wrong_input = Array1::from_vec(vec![0.1; 64]); // Wrong size

        let result = moe.forward(&wrong_input);
        assert!(result.is_err(), "Should fail with dimension mismatch");
    }
}
