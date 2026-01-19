//! # Structured Pruning
//!
//! Advanced pruning algorithms for model compression with structure preservation.
//!
//! ## Features
//!
//! - **Magnitude Pruning**: Remove weights below threshold
//! - **Structured Pruning**: Remove entire channels, filters, or heads
//! - **L1/L2 Norm Pruning**: Importance scoring based on norms
//! - **Gradient-based Pruning**: Use gradient information for importance
//! - **Progressive Pruning**: Gradual pruning with retraining
//! - **Group Lasso**: Structured sparsity via regularization
//!
//! ## References
//!
//! - "Learning both Weights and Connections for Efficient Neural Networks" (Han et al., 2015)
//! - "Pruning Filters for Efficient ConvNets" (Li et al., 2017)
//! - "The Lottery Ticket Hypothesis" (Frankle & Carbin, 2019)

use crate::{CoreError, CoreResult};
use scirs2_core::ndarray::Array2;
#[allow(unused_imports)]
use scirs2_core::ndarray::Axis; // Used in tests
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pruning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PruningStrategy {
    /// Unstructured magnitude-based pruning
    Magnitude,
    /// L1 norm-based structured pruning
    L1Norm,
    /// L2 norm-based structured pruning
    L2Norm,
    /// Gradient magnitude-based pruning
    Gradient,
    /// Random pruning (baseline)
    Random,
}

/// Pruning granularity for structured pruning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PruningGranularity {
    /// Unstructured (individual weights)
    Unstructured,
    /// Channel-wise (entire input/output channels)
    Channel,
    /// Filter-wise (entire filters in convolutions)
    Filter,
    /// Head-wise (entire attention heads)
    Head,
    /// Block-wise (entire transformer/SSM blocks)
    Block,
}

/// Pruning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Pruning strategy
    pub strategy: PruningStrategy,
    /// Pruning granularity
    pub granularity: PruningGranularity,
    /// Target sparsity ratio (0.0 to 1.0)
    pub target_sparsity: f32,
    /// Whether to use global threshold (vs per-layer)
    pub global_threshold: bool,
    /// Number of pruning iterations (for progressive pruning)
    pub num_iterations: usize,
    /// Whether to keep pruned weights for recovery
    pub keep_pruned_weights: bool,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            strategy: PruningStrategy::Magnitude,
            granularity: PruningGranularity::Unstructured,
            target_sparsity: 0.5,
            global_threshold: false,
            num_iterations: 1,
            keep_pruned_weights: false,
        }
    }
}

impl PruningConfig {
    /// Create new pruning configuration
    pub fn new(strategy: PruningStrategy, target_sparsity: f32) -> Self {
        Self {
            strategy,
            target_sparsity,
            ..Default::default()
        }
    }

    /// Set granularity
    pub fn with_granularity(mut self, granularity: PruningGranularity) -> Self {
        self.granularity = granularity;
        self
    }

    /// Enable global thresholding
    pub fn with_global_threshold(mut self) -> Self {
        self.global_threshold = true;
        self
    }

    /// Set number of iterations for progressive pruning
    pub fn with_iterations(mut self, num_iterations: usize) -> Self {
        self.num_iterations = num_iterations;
        self
    }

    /// Keep pruned weights
    pub fn with_keep_weights(mut self) -> Self {
        self.keep_pruned_weights = true;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> CoreResult<()> {
        if self.target_sparsity < 0.0 || self.target_sparsity >= 1.0 {
            return Err(CoreError::InvalidConfig(
                "target_sparsity must be in [0, 1)".into(),
            ));
        }
        if self.num_iterations == 0 {
            return Err(CoreError::InvalidConfig(
                "num_iterations must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Pruning mask for a weight tensor
#[derive(Debug, Clone)]
pub struct PruningMask {
    /// Binary mask (1 = keep, 0 = prune)
    pub mask: Array2<f32>,
    /// Pruned weights (if keep_pruned_weights is enabled)
    pub pruned_weights: Option<Array2<f32>>,
    /// Sparsity ratio achieved
    pub sparsity: f32,
}

impl PruningMask {
    /// Create a new pruning mask
    pub fn new(mask: Array2<f32>) -> Self {
        let total = mask.len();
        let zeros = mask.iter().filter(|&&x| x == 0.0).count();
        let sparsity = zeros as f32 / total as f32;

        Self {
            mask,
            pruned_weights: None,
            sparsity,
        }
    }

    /// Apply mask to weights
    pub fn apply(&self, weights: &Array2<f32>) -> Array2<f32> {
        weights * &self.mask
    }

    /// Count remaining (non-zero) parameters
    pub fn num_parameters(&self) -> usize {
        self.mask.iter().filter(|&&x| x != 0.0).count()
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        1.0 / (1.0 - self.sparsity).max(1e-6)
    }
}

/// Structured pruner for neural network layers
pub struct StructuredPruner {
    config: PruningConfig,
    masks: HashMap<String, PruningMask>,
}

impl StructuredPruner {
    /// Create a new structured pruner
    pub fn new(config: PruningConfig) -> CoreResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            masks: HashMap::new(),
        })
    }

    /// Prune a 2D weight matrix
    pub fn prune(&mut self, name: &str, weights: &Array2<f32>) -> CoreResult<PruningMask> {
        let mask = match self.config.granularity {
            PruningGranularity::Unstructured => self.prune_unstructured(weights)?,
            PruningGranularity::Channel => self.prune_channels(weights)?,
            PruningGranularity::Filter => self.prune_filters(weights)?,
            _ => {
                return Err(CoreError::InvalidConfig(format!(
                    "Granularity {:?} not yet implemented for 2D tensors",
                    self.config.granularity
                )))
            }
        };

        self.masks.insert(name.to_string(), mask.clone());
        Ok(mask)
    }

    /// Unstructured pruning (individual weights)
    fn prune_unstructured(&self, weights: &Array2<f32>) -> CoreResult<PruningMask> {
        let importance = self.compute_importance(weights)?;
        let threshold = self.compute_threshold(&importance)?;

        let mask = importance.mapv(|v| if v.abs() >= threshold { 1.0 } else { 0.0 });
        Ok(PruningMask::new(mask))
    }

    /// Channel-wise pruning (prune entire output channels)
    fn prune_channels(&self, weights: &Array2<f32>) -> CoreResult<PruningMask> {
        let (out_channels, _in_features) = weights.dim();

        // Compute importance score for each output channel
        let mut channel_importance = Vec::with_capacity(out_channels);
        for channel_idx in 0..out_channels {
            let channel = weights.row(channel_idx);
            let importance = match self.config.strategy {
                PruningStrategy::L1Norm => channel.iter().map(|x| x.abs()).sum::<f32>(),
                PruningStrategy::L2Norm => channel.iter().map(|x| x.powi(2)).sum::<f32>().sqrt(),
                PruningStrategy::Magnitude => {
                    channel.iter().map(|x| x.abs()).sum::<f32>() / channel.len() as f32
                }
                _ => channel.iter().map(|x| x.abs()).sum::<f32>(),
            };
            channel_importance.push((channel_idx, importance));
        }

        // Sort by importance (ascending)
        channel_importance.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Determine how many channels to prune
        let num_to_prune = (out_channels as f32 * self.config.target_sparsity) as usize;

        // Create mask
        let mut mask = Array2::ones(weights.dim());
        for &(channel_idx, _) in channel_importance.iter().take(num_to_prune) {
            mask.row_mut(channel_idx).fill(0.0);
        }

        Ok(PruningMask::new(mask))
    }

    /// Filter-wise pruning (similar to channel pruning for conv layers)
    fn prune_filters(&self, weights: &Array2<f32>) -> CoreResult<PruningMask> {
        // For 2D weights, treat as channels
        self.prune_channels(weights)
    }

    /// Compute importance scores for weights
    fn compute_importance(&self, weights: &Array2<f32>) -> CoreResult<Array2<f32>> {
        let importance = match self.config.strategy {
            PruningStrategy::Magnitude => weights.mapv(|x| x.abs()),
            PruningStrategy::L1Norm => weights.mapv(|x| x.abs()),
            PruningStrategy::L2Norm => weights.mapv(|x| x.powi(2)),
            PruningStrategy::Random => {
                // Random importance for baseline
                use scirs2_core::random::thread_rng;
                let mut rng = thread_rng();
                Array2::from_shape_fn(weights.dim(), |_| rng.random::<f32>())
            }
            PruningStrategy::Gradient => {
                // Would need gradient information; use magnitude as fallback
                weights.mapv(|x| x.abs())
            }
        };

        Ok(importance)
    }

    /// Compute threshold for pruning
    fn compute_threshold(&self, importance: &Array2<f32>) -> CoreResult<f32> {
        // Flatten and sort importance values
        let mut values: Vec<f32> = importance.iter().copied().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find threshold at target sparsity percentile
        let threshold_idx = (values.len() as f32 * self.config.target_sparsity) as usize;
        let threshold = values.get(threshold_idx).copied().unwrap_or(0.0);

        Ok(threshold)
    }

    /// Progressive pruning over multiple iterations
    pub fn prune_progressive(
        &mut self,
        name: &str,
        weights: &Array2<f32>,
    ) -> CoreResult<Vec<PruningMask>> {
        let mut masks = Vec::with_capacity(self.config.num_iterations);
        let sparsity_per_iter = self.config.target_sparsity / self.config.num_iterations as f32;

        let mut current_weights = weights.clone();
        for iter in 0..self.config.num_iterations {
            // Adjust target sparsity for this iteration
            let iter_config = PruningConfig {
                target_sparsity: sparsity_per_iter,
                ..self.config.clone()
            };

            let mut iter_pruner = StructuredPruner::new(iter_config)?;
            let mask = iter_pruner.prune(&format!("{}_{}", name, iter), &current_weights)?;

            // Apply mask for next iteration
            current_weights = mask.apply(&current_weights);
            masks.push(mask);
        }

        // Store final mask
        if let Some(final_mask) = masks.last() {
            self.masks.insert(name.to_string(), final_mask.clone());
        }

        Ok(masks)
    }

    /// Get mask for a layer
    pub fn get_mask(&self, name: &str) -> Option<&PruningMask> {
        self.masks.get(name)
    }

    /// Get all masks
    pub fn masks(&self) -> &HashMap<String, PruningMask> {
        &self.masks
    }

    /// Compute global sparsity across all pruned layers
    pub fn global_sparsity(&self) -> f32 {
        if self.masks.is_empty() {
            return 0.0;
        }

        let total_params: usize = self.masks.values().map(|m| m.mask.len()).sum();
        let pruned_params: usize = self
            .masks
            .values()
            .map(|m| m.mask.iter().filter(|&&x| x == 0.0).count())
            .sum();

        pruned_params as f32 / total_params as f32
    }

    /// Get compression ratio across all layers
    pub fn global_compression_ratio(&self) -> f32 {
        let sparsity = self.global_sparsity();
        1.0 / (1.0 - sparsity).max(1e-6)
    }
}

/// Gradient-based pruning (requires gradient information)
pub struct GradientPruner {
    pruner: StructuredPruner,
    /// Gradient accumulation for importance scoring
    gradient_accumulator: HashMap<String, Array2<f32>>,
}

impl GradientPruner {
    /// Create a new gradient-based pruner
    pub fn new(config: PruningConfig) -> CoreResult<Self> {
        Ok(Self {
            pruner: StructuredPruner::new(config)?,
            gradient_accumulator: HashMap::new(),
        })
    }

    /// Accumulate gradient for a layer
    pub fn accumulate_gradient(&mut self, name: &str, gradient: &Array2<f32>) {
        let acc = self
            .gradient_accumulator
            .entry(name.to_string())
            .or_insert_with(|| Array2::zeros(gradient.dim()));
        *acc = &*acc + gradient;
    }

    /// Prune using accumulated gradients
    pub fn prune_with_gradients(
        &mut self,
        name: &str,
        weights: &Array2<f32>,
    ) -> CoreResult<PruningMask> {
        // Get accumulated gradients
        let gradients = self
            .gradient_accumulator
            .get(name)
            .ok_or_else(|| CoreError::InvalidConfig("No gradients accumulated".into()))?;

        // Compute importance as |weight * gradient|
        let importance = weights * gradients;
        let importance = importance.mapv(|x| x.abs());

        // Use importance to create mask
        let threshold = self.compute_gradient_threshold(&importance)?;
        let mask = importance.mapv(|v| if v >= threshold { 1.0 } else { 0.0 });

        let pruning_mask = PruningMask::new(mask);
        self.pruner
            .masks
            .insert(name.to_string(), pruning_mask.clone());

        Ok(pruning_mask)
    }

    /// Compute threshold from gradient-based importance
    fn compute_gradient_threshold(&self, importance: &Array2<f32>) -> CoreResult<f32> {
        let mut values: Vec<f32> = importance.iter().copied().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_idx = (values.len() as f32 * self.pruner.config.target_sparsity) as usize;
        Ok(values.get(threshold_idx).copied().unwrap_or(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruning_config() {
        let config = PruningConfig::new(PruningStrategy::Magnitude, 0.5);
        assert_eq!(config.strategy, PruningStrategy::Magnitude);
        assert_eq!(config.target_sparsity, 0.5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_pruning_config_validation() {
        let mut config = PruningConfig::new(PruningStrategy::Magnitude, 1.5);
        assert!(config.validate().is_err());

        config.target_sparsity = -0.1;
        assert!(config.validate().is_err());

        config.target_sparsity = 0.5;
        config.num_iterations = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_unstructured_pruning() {
        let config = PruningConfig::new(PruningStrategy::Magnitude, 0.5);
        let mut pruner = StructuredPruner::new(config).unwrap();

        // Create weights with uniform distribution for predictable pruning
        let weights = Array2::from_shape_fn((10, 10), |(i, j)| ((i * 10 + j) as f32) * 0.01);

        let mask = pruner.prune("layer1", &weights).unwrap();
        // Should be close to 50% sparsity, allow some tolerance
        assert!(
            mask.sparsity >= 0.45 && mask.sparsity <= 0.55,
            "Expected sparsity ~0.5, got {}",
            mask.sparsity
        );
    }

    #[test]
    fn test_channel_pruning() {
        let config = PruningConfig::new(PruningStrategy::L2Norm, 0.5)
            .with_granularity(PruningGranularity::Channel);
        let mut pruner = StructuredPruner::new(config).unwrap();

        let weights = Array2::from_shape_fn((8, 16), |(i, _j)| {
            if i < 4 {
                1.0
            } else {
                0.1
            } // First 4 channels more important
        });

        let mask = pruner.prune("layer1", &weights).unwrap();

        // Check that entire channels are pruned
        for row in mask.mask.axis_iter(Axis(0)) {
            let sum: f32 = row.sum();
            assert!(sum == 0.0 || sum == row.len() as f32);
        }
    }

    #[test]
    fn test_pruning_mask_apply() {
        let mask_data = Array2::from_shape_fn((4, 4), |(i, j)| if i == j { 1.0 } else { 0.0 });
        let mask = PruningMask::new(mask_data);

        let weights = Array2::ones((4, 4));
        let pruned = mask.apply(&weights);

        // Should only keep diagonal elements
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_eq!(pruned[[i, j]], 1.0);
                } else {
                    assert_eq!(pruned[[i, j]], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_progressive_pruning() {
        let config = PruningConfig::new(PruningStrategy::Magnitude, 0.6).with_iterations(3);
        let mut pruner = StructuredPruner::new(config).unwrap();

        let weights = Array2::from_shape_fn((8, 8), |(i, j)| (i as f32 + j as f32) * 0.1);

        let masks = pruner.prune_progressive("layer1", &weights).unwrap();
        assert_eq!(masks.len(), 3);

        // Sparsity should increase with iterations
        for i in 1..masks.len() {
            assert!(masks[i].sparsity >= masks[i - 1].sparsity);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let mask = PruningMask::new(Array2::from_shape_fn((10, 10), |(i, j)| {
            if i + j < 5 {
                1.0
            } else {
                0.0
            }
        }));

        let ratio = mask.compression_ratio();
        assert!(ratio > 1.0); // Should have compression
        assert!(ratio < 10.0); // But not too extreme
    }

    #[test]
    fn test_global_sparsity() {
        let config = PruningConfig::new(PruningStrategy::Magnitude, 0.5);
        let mut pruner = StructuredPruner::new(config).unwrap();

        let weights1 = Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f32);
        let weights2 = Array2::from_shape_fn((4, 4), |(i, j)| (i * j) as f32);

        pruner.prune("layer1", &weights1).unwrap();
        pruner.prune("layer2", &weights2).unwrap();

        let global_sparsity = pruner.global_sparsity();
        assert!((0.4..=0.6).contains(&global_sparsity));
    }

    #[test]
    fn test_gradient_pruner_accumulation() {
        let config = PruningConfig::new(PruningStrategy::Gradient, 0.5);
        let mut pruner = GradientPruner::new(config).unwrap();

        let gradient1 = Array2::ones((4, 4));
        let gradient2 = Array2::ones((4, 4)) * 2.0;

        pruner.accumulate_gradient("layer1", &gradient1);
        pruner.accumulate_gradient("layer1", &gradient2);

        let accumulated = &pruner.gradient_accumulator["layer1"];
        assert_eq!(accumulated[[0, 0]], 3.0);
    }

    #[test]
    fn test_random_pruning() {
        let config = PruningConfig::new(PruningStrategy::Random, 0.5);
        let mut pruner = StructuredPruner::new(config).unwrap();

        let weights = Array2::ones((10, 10));
        let mask = pruner.prune("layer1", &weights).unwrap();

        // Should achieve approximately 50% sparsity
        assert!(mask.sparsity >= 0.4 && mask.sparsity <= 0.6);
    }
}
