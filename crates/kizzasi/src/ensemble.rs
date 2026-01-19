//! Multi-model ensemble predictions with voting strategies
//!
//! This module provides sophisticated ensemble prediction capabilities:
//! - Multiple models running in parallel
//! - Weighted voting and averaging
//! - Confidence-based model selection
//! - Dynamic model weighting based on performance
//! - Fallback and redundancy strategies
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi::ensemble::{EnsemblePredictor, VotingStrategy};
//!
//! let ensemble = EnsemblePredictor::new(VotingStrategy::WeightedAverage);
//! ensemble.add_model(predictor1, 1.0)?;
//! ensemble.add_model(predictor2, 0.8)?;
//!
//! let result = ensemble.predict(&input)?;
//! ```

use crate::error::{KizzasiError, KizzasiResult};
use crate::predictor::Kizzasi;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

/// Strategy for combining predictions from multiple models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VotingStrategy {
    /// Simple average of all predictions
    Average,

    /// Weighted average using model weights
    WeightedAverage,

    /// Median of all predictions
    Median,

    /// Select prediction from model with highest weight
    Weighted,

    /// Select based on confidence scores
    ConfidenceBased,

    /// Majority voting (for classification tasks)
    MajorityVote,
}

/// Statistics for ensemble prediction
#[derive(Debug, Clone)]
pub struct EnsembleStats {
    /// Number of models in ensemble
    pub num_models: usize,

    /// Total predictions made
    pub total_predictions: u64,

    /// Average prediction variance across models
    pub avg_variance: f64,

    /// Model-specific statistics
    pub model_stats: HashMap<String, ModelStats>,
}

/// Per-model statistics
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Model name/ID
    pub model_id: String,

    /// Number of times this model was selected
    pub selection_count: u64,

    /// Average confidence score
    pub avg_confidence: f64,

    /// Current weight
    pub weight: f64,
}

/// Configuration for a model in the ensemble
struct EnsembleModel {
    predictor: Kizzasi,
    weight: f64,
    model_id: String,
    selection_count: u64,
    avg_confidence: f64,
}

/// Multi-model ensemble predictor
pub struct EnsemblePredictor {
    models: Vec<EnsembleModel>,
    strategy: VotingStrategy,
    total_predictions: u64,
    enable_dynamic_weighting: bool,
}

impl EnsemblePredictor {
    /// Create a new ensemble with the specified voting strategy
    pub fn new(strategy: VotingStrategy) -> Self {
        Self {
            models: Vec::new(),
            strategy,
            total_predictions: 0,
            enable_dynamic_weighting: false,
        }
    }

    /// Create an ensemble using average voting
    pub fn with_average() -> Self {
        Self::new(VotingStrategy::Average)
    }

    /// Create an ensemble using weighted average
    pub fn with_weighted_average() -> Self {
        Self::new(VotingStrategy::WeightedAverage)
    }

    /// Create an ensemble using median voting
    pub fn with_median() -> Self {
        Self::new(VotingStrategy::Median)
    }

    /// Enable dynamic weight adjustment based on performance
    pub fn enable_dynamic_weighting(mut self) -> Self {
        self.enable_dynamic_weighting = true;
        self
    }

    /// Add a model to the ensemble
    pub fn add_model(&mut self, predictor: Kizzasi, weight: f64) -> KizzasiResult<()> {
        self.add_model_with_id(predictor, weight, format!("model_{}", self.models.len()))
    }

    /// Add a model with a specific ID
    pub fn add_model_with_id(
        &mut self,
        predictor: Kizzasi,
        weight: f64,
        model_id: String,
    ) -> KizzasiResult<()> {
        if weight < 0.0 {
            return Err(KizzasiError::InvalidState {
                reason: format!("Model weight must be non-negative, got {}", weight),
                recovery: None,
            });
        }

        // Validate compatibility with existing models
        if let Some(first_model) = self.models.first() {
            if predictor.input_dim() != first_model.predictor.input_dim() {
                return Err(KizzasiError::DimensionMismatch {
                    expected: first_model.predictor.input_dim(),
                    actual: predictor.input_dim(),
                    context: "Input dimensions must match across all ensemble models".to_string(),
                });
            }
            if predictor.output_dim() != first_model.predictor.output_dim() {
                return Err(KizzasiError::DimensionMismatch {
                    expected: first_model.predictor.output_dim(),
                    actual: predictor.output_dim(),
                    context: "Output dimensions must match across all ensemble models".to_string(),
                });
            }
        }

        self.models.push(EnsembleModel {
            predictor,
            weight,
            model_id,
            selection_count: 0,
            avg_confidence: 1.0,
        });

        Ok(())
    }

    /// Remove a model from the ensemble
    pub fn remove_model(&mut self, model_id: &str) -> KizzasiResult<Kizzasi> {
        let index = self
            .models
            .iter()
            .position(|m| m.model_id == model_id)
            .ok_or_else(|| KizzasiError::InvalidState {
                reason: format!("Model '{}' not found in ensemble", model_id),
                recovery: None,
            })?;

        Ok(self.models.remove(index).predictor)
    }

    /// Perform ensemble prediction
    pub fn predict(&mut self, input: &Array1<f32>) -> KizzasiResult<Array1<f32>> {
        if self.models.is_empty() {
            return Err(KizzasiError::InvalidState {
                reason: "Ensemble has no models".to_string(),
                recovery: None,
            });
        }

        // Collect predictions from all models
        let mut predictions = Vec::with_capacity(self.models.len());
        let mut weights = Vec::with_capacity(self.models.len());

        for model in &mut self.models {
            let prediction = model.predictor.step(input)?;
            predictions.push(prediction);
            weights.push(model.weight);
        }

        // Combine predictions based on strategy
        let result = match self.strategy {
            VotingStrategy::Average => self.average_predictions(&predictions),
            VotingStrategy::WeightedAverage => self.weighted_average(&predictions, &weights),
            VotingStrategy::Median => self.median_predictions(&predictions),
            VotingStrategy::Weighted => {
                // Select the prediction from the highest-weight model
                let max_idx = weights
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                self.models[max_idx].selection_count += 1;
                predictions[max_idx].clone()
            }
            VotingStrategy::ConfidenceBased => {
                // Select based on confidence (use weight as proxy for confidence)
                let confidences: Vec<f64> = self
                    .models
                    .iter()
                    .map(|m| m.avg_confidence * m.weight)
                    .collect();

                let max_idx = confidences
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                self.models[max_idx].selection_count += 1;
                predictions[max_idx].clone()
            }
            VotingStrategy::MajorityVote => {
                // For continuous outputs, use weighted average as approximation
                self.weighted_average(&predictions, &weights)
            }
        };

        self.total_predictions += 1;

        Ok(result)
    }

    /// Perform batch ensemble prediction
    pub fn predict_batch(&mut self, inputs: &[Array1<f32>]) -> KizzasiResult<Vec<Array1<f32>>> {
        inputs.iter().map(|input| self.predict(input)).collect()
    }

    /// Reset all models in the ensemble
    pub fn reset_all(&mut self) {
        for model in &mut self.models {
            model.predictor.reset();
        }
    }

    /// Get ensemble statistics
    pub fn stats(&self) -> EnsembleStats {
        let mut model_stats = HashMap::new();

        for model in &self.models {
            model_stats.insert(
                model.model_id.clone(),
                ModelStats {
                    model_id: model.model_id.clone(),
                    selection_count: model.selection_count,
                    avg_confidence: model.avg_confidence,
                    weight: model.weight,
                },
            );
        }

        EnsembleStats {
            num_models: self.models.len(),
            total_predictions: self.total_predictions,
            avg_variance: 0.0, // TODO: Calculate actual variance
            model_stats,
        }
    }

    /// Update model weights dynamically based on performance
    pub fn update_weights(&mut self, model_id: &str, new_weight: f64) -> KizzasiResult<()> {
        let model = self
            .models
            .iter_mut()
            .find(|m| m.model_id == model_id)
            .ok_or_else(|| KizzasiError::InvalidState {
                reason: format!("Model '{}' not found", model_id),
                recovery: None,
            })?;

        if new_weight < 0.0 {
            return Err(KizzasiError::InvalidState {
                reason: format!("Weight must be non-negative, got {}", new_weight),
                recovery: None,
            });
        }

        model.weight = new_weight;
        Ok(())
    }

    /// Get the number of models in the ensemble
    pub fn num_models(&self) -> usize {
        self.models.len()
    }

    /// Get the voting strategy
    pub fn strategy(&self) -> VotingStrategy {
        self.strategy
    }

    /// Set the voting strategy
    pub fn set_strategy(&mut self, strategy: VotingStrategy) {
        self.strategy = strategy;
    }

    // Helper methods for combining predictions

    fn average_predictions(&self, predictions: &[Array1<f32>]) -> Array1<f32> {
        let dim = predictions[0].len();
        let mut result = Array1::zeros(dim);

        for prediction in predictions {
            result += prediction;
        }

        result / predictions.len() as f32
    }

    fn weighted_average(&self, predictions: &[Array1<f32>], weights: &[f64]) -> Array1<f32> {
        let dim = predictions[0].len();
        let mut result = Array1::zeros(dim);
        let weight_sum: f64 = weights.iter().sum();

        if weight_sum == 0.0 {
            return self.average_predictions(predictions);
        }

        for (prediction, &weight) in predictions.iter().zip(weights.iter()) {
            result += &(prediction * (weight as f32 / weight_sum as f32));
        }

        result
    }

    fn median_predictions(&self, predictions: &[Array1<f32>]) -> Array1<f32> {
        let dim = predictions[0].len();
        let mut result = Array1::zeros(dim);

        // For each dimension, calculate median across all predictions
        for i in 0..dim {
            let mut values: Vec<f32> = predictions.iter().map(|p| p[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            result[i] = if values.len().is_multiple_of(2) {
                (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
            } else {
                values[values.len() / 2]
            };
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor::KizzasiBuilder;

    #[test]
    fn test_ensemble_creation() {
        let ensemble = EnsemblePredictor::new(VotingStrategy::Average);
        assert_eq!(ensemble.num_models(), 0);
        assert_eq!(ensemble.strategy(), VotingStrategy::Average);
    }

    #[test]
    fn test_add_model() -> KizzasiResult<()> {
        let mut ensemble = EnsemblePredictor::with_average();

        let predictor1 = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let predictor2 = KizzasiBuilder::lightweight_preset(2, 2).build()?;

        ensemble.add_model(predictor1, 1.0)?;
        ensemble.add_model(predictor2, 0.8)?;

        assert_eq!(ensemble.num_models(), 2);

        Ok(())
    }

    #[test]
    fn test_dimension_mismatch() -> KizzasiResult<()> {
        let mut ensemble = EnsemblePredictor::with_average();

        let predictor1 = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let predictor2 = KizzasiBuilder::lightweight_preset(3, 3).build()?;

        ensemble.add_model(predictor1, 1.0)?;

        // Should fail due to dimension mismatch
        let result = ensemble.add_model(predictor2, 0.8);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_average_prediction() -> KizzasiResult<()> {
        let mut ensemble = EnsemblePredictor::with_average();

        let predictor1 = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let predictor2 = KizzasiBuilder::lightweight_preset(2, 2).build()?;

        ensemble.add_model(predictor1, 1.0)?;
        ensemble.add_model(predictor2, 1.0)?;

        let input = Array1::from_vec(vec![1.0, 2.0]);
        let output = ensemble.predict(&input)?;

        assert_eq!(output.len(), 2);

        let stats = ensemble.stats();
        assert_eq!(stats.total_predictions, 1);

        Ok(())
    }

    #[test]
    fn test_weighted_average() -> KizzasiResult<()> {
        let mut ensemble = EnsemblePredictor::with_weighted_average();

        let predictor1 = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let predictor2 = KizzasiBuilder::lightweight_preset(2, 2).build()?;

        ensemble.add_model(predictor1, 0.8)?;
        ensemble.add_model(predictor2, 0.2)?;

        let input = Array1::from_vec(vec![1.0, 2.0]);
        let output = ensemble.predict(&input)?;

        assert_eq!(output.len(), 2);

        Ok(())
    }

    #[test]
    fn test_median_prediction() -> KizzasiResult<()> {
        let mut ensemble = EnsemblePredictor::with_median();

        for _ in 0..3 {
            let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
            ensemble.add_model(predictor, 1.0)?;
        }

        let input = Array1::from_vec(vec![1.0, 2.0]);
        let output = ensemble.predict(&input)?;

        assert_eq!(output.len(), 2);

        Ok(())
    }

    #[test]
    fn test_remove_model() -> KizzasiResult<()> {
        let mut ensemble = EnsemblePredictor::with_average();

        let predictor1 = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let predictor2 = KizzasiBuilder::lightweight_preset(2, 2).build()?;

        ensemble.add_model_with_id(predictor1, 1.0, "model_a".to_string())?;
        ensemble.add_model_with_id(predictor2, 1.0, "model_b".to_string())?;

        assert_eq!(ensemble.num_models(), 2);

        let _ = ensemble.remove_model("model_a")?;
        assert_eq!(ensemble.num_models(), 1);

        Ok(())
    }

    #[test]
    fn test_reset_all() -> KizzasiResult<()> {
        let mut ensemble = EnsemblePredictor::with_average();

        let predictor1 = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let predictor2 = KizzasiBuilder::lightweight_preset(2, 2).build()?;

        ensemble.add_model(predictor1, 1.0)?;
        ensemble.add_model(predictor2, 1.0)?;

        ensemble.reset_all();

        Ok(())
    }

    #[test]
    fn test_batch_prediction() -> KizzasiResult<()> {
        let mut ensemble = EnsemblePredictor::with_average();

        let predictor1 = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        let predictor2 = KizzasiBuilder::lightweight_preset(2, 2).build()?;

        ensemble.add_model(predictor1, 1.0)?;
        ensemble.add_model(predictor2, 1.0)?;

        let inputs = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![3.0, 4.0]),
        ];

        let outputs = ensemble.predict_batch(&inputs)?;
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].len(), 2);

        Ok(())
    }

    #[test]
    fn test_update_weights() -> KizzasiResult<()> {
        let mut ensemble = EnsemblePredictor::with_weighted_average();

        let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        ensemble.add_model_with_id(predictor, 1.0, "model_x".to_string())?;

        ensemble.update_weights("model_x", 0.5)?;

        let stats = ensemble.stats();
        assert_eq!(
            stats.model_stats.get("model_x").map(|s| s.weight),
            Some(0.5)
        );

        Ok(())
    }

    #[test]
    fn test_negative_weight_rejected() {
        let mut ensemble = EnsemblePredictor::with_average();
        let predictor = KizzasiBuilder::lightweight_preset(2, 2)
            .build()
            .expect("Failed to build");

        let result = ensemble.add_model(predictor, -0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_ensemble_prediction() {
        let mut ensemble = EnsemblePredictor::with_average();
        let input = Array1::from_vec(vec![1.0, 2.0]);

        let result = ensemble.predict(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_switching() -> KizzasiResult<()> {
        let mut ensemble = EnsemblePredictor::with_average();

        let predictor = KizzasiBuilder::lightweight_preset(2, 2).build()?;
        ensemble.add_model(predictor, 1.0)?;

        assert_eq!(ensemble.strategy(), VotingStrategy::Average);

        ensemble.set_strategy(VotingStrategy::Median);
        assert_eq!(ensemble.strategy(), VotingStrategy::Median);

        Ok(())
    }
}
