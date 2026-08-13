//! Model ensembling for robust inference
//!
//! This module provides ensembling strategies to combine predictions from
//! multiple models, improving accuracy and robustness.
//!
//! ## Ensemble Methods
//!
//! 1. **Averaging**: Average predictions from all models
//! 2. **Weighted**: Weighted average based on model confidence
//! 3. **Voting**: Majority voting for discrete outputs
//! 4. **Stacking**: Use a meta-model to combine predictions

use crate::error::{InferenceError, InferenceResult};
use crate::sampling::{Sampler, SamplingConfig};
use kizzasi_model::AutoregressiveModel;
use scirs2_core::ndarray::Array1;

/// Ensemble combination strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleStrategy {
    /// Simple averaging of all model outputs
    Average,
    /// Weighted average (weights must be provided)
    Weighted,
    /// Maximum voting (selects most common prediction)
    Voting,
    /// Product of experts (multiply probabilities)
    ProductOfExperts,
}

/// Configuration for model ensemble
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Ensemble strategy
    pub strategy: EnsembleStrategy,
    /// Model weights (for weighted averaging)
    pub weights: Option<Vec<f32>>,
    /// Whether to normalize (softmax) the combined output *after*
    /// `Average`/`Weighted` combine the component predictions.
    ///
    /// Off by default: an AGSP predicting continuous next-signal values
    /// wants the averaged *signal*, not a probability simplex, so turning
    /// this on is an explicit opt-in rather than a silent default. `Voting`
    /// and `ProductOfExperts` are unaffected — they always produce a
    /// probability-like output by construction.
    pub normalize_outputs: bool,
    /// Softmax temperature applied wherever this ensemble produces a
    /// probability distribution — `normalize_outputs == true` for
    /// `Average`/`Weighted`, and always for `ProductOfExperts` (which
    /// softmaxes each model's prediction before multiplying). Values `<= 0.0`
    /// or non-finite are treated as `1.0` (no scaling).
    pub temperature: f32,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            strategy: EnsembleStrategy::Average,
            weights: None,
            normalize_outputs: false,
            temperature: 1.0,
        }
    }
}

impl EnsembleConfig {
    /// Create a new ensemble configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set ensemble strategy
    pub fn strategy(mut self, strategy: EnsembleStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set model weights for weighted averaging
    pub fn weights(mut self, weights: Vec<f32>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Enable/disable output normalization
    pub fn normalize_outputs(mut self, normalize: bool) -> Self {
        self.normalize_outputs = normalize;
        self
    }

    /// Set temperature for final sampling
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }
}

/// Model ensemble for combining multiple models
pub struct ModelEnsemble {
    /// Component models
    models: Vec<Box<dyn AutoregressiveModel>>,
    /// Configuration
    config: EnsembleConfig,
    /// Sampler for final output
    sampler: Sampler,
}

impl ModelEnsemble {
    /// Create a new model ensemble
    pub fn new(
        models: Vec<Box<dyn AutoregressiveModel>>,
        config: EnsembleConfig,
    ) -> InferenceResult<Self> {
        if models.is_empty() {
            return Err(InferenceError::ForwardError(
                "Ensemble must contain at least one model".to_string(),
            ));
        }

        // Validate weights if provided
        if let Some(ref weights) = config.weights {
            if weights.len() != models.len() {
                return Err(InferenceError::DimensionMismatch {
                    expected: models.len(),
                    got: weights.len(),
                });
            }

            // Check weights are positive and sum to 1
            let sum: f32 = weights.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(InferenceError::ForwardError(format!(
                    "Ensemble weights must sum to 1.0, got {}",
                    sum
                )));
            }
        }

        let sampler_config = SamplingConfig::new().temperature(config.temperature);
        let sampler = Sampler::new(sampler_config);

        Ok(Self {
            models,
            config,
            sampler,
        })
    }

    /// Get number of models in ensemble
    pub fn num_models(&self) -> usize {
        self.models.len()
    }

    /// Perform ensemble inference step
    pub fn step(&mut self, input: &Array1<f32>) -> InferenceResult<Array1<f32>> {
        // Collect predictions from all models
        let mut predictions = Vec::with_capacity(self.models.len());

        for model in &mut self.models {
            let pred = model
                .step(input)
                .map_err(|e| InferenceError::ForwardError(e.to_string()))?;
            predictions.push(pred);
        }

        // Combine predictions based on strategy
        self.combine_predictions(&predictions)
    }

    /// Combine predictions from multiple models
    fn combine_predictions(&mut self, predictions: &[Array1<f32>]) -> InferenceResult<Array1<f32>> {
        if predictions.is_empty() {
            return Err(InferenceError::ForwardError(
                "No predictions to combine".to_string(),
            ));
        }

        let output_dim = predictions[0].len();

        // Verify all predictions have same dimension
        for pred in predictions {
            if pred.len() != output_dim {
                return Err(InferenceError::DimensionMismatch {
                    expected: output_dim,
                    got: pred.len(),
                });
            }
        }

        match self.config.strategy {
            EnsembleStrategy::Average => self.combine_average(predictions, output_dim),
            EnsembleStrategy::Weighted => self.combine_weighted(predictions, output_dim),
            EnsembleStrategy::Voting => self.combine_voting(predictions),
            EnsembleStrategy::ProductOfExperts => {
                self.combine_product_of_experts(predictions, output_dim)
            }
        }
    }

    /// Average ensemble
    fn combine_average(
        &self,
        predictions: &[Array1<f32>],
        output_dim: usize,
    ) -> InferenceResult<Array1<f32>> {
        let mut combined = Array1::zeros(output_dim);
        let n = predictions.len() as f32;

        for pred in predictions {
            combined += pred;
        }

        combined /= n;

        if self.config.normalize_outputs {
            combined = self.normalize(&combined);
        }

        Ok(combined)
    }

    /// Weighted average ensemble
    fn combine_weighted(
        &self,
        predictions: &[Array1<f32>],
        output_dim: usize,
    ) -> InferenceResult<Array1<f32>> {
        let weights = self.config.weights.as_ref().ok_or_else(|| {
            InferenceError::ForwardError("Weights not provided for weighted ensemble".to_string())
        })?;

        let mut combined = Array1::zeros(output_dim);

        for (pred, &weight) in predictions.iter().zip(weights.iter()) {
            combined += &(pred * weight);
        }

        if self.config.normalize_outputs {
            combined = self.normalize(&combined);
        }

        Ok(combined)
    }

    /// Voting ensemble (for discrete outputs)
    fn combine_voting(&mut self, predictions: &[Array1<f32>]) -> InferenceResult<Array1<f32>> {
        // For each model, get the argmax
        let votes: Vec<usize> = predictions
            .iter()
            .map(|pred| {
                pred.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();

        // Count votes
        let output_dim = predictions[0].len();
        let mut vote_counts = vec![0usize; output_dim];
        for &vote in &votes {
            if vote < output_dim {
                vote_counts[vote] += 1;
            }
        }

        // Convert counts to probabilities
        let total_votes = votes.len() as f32;
        let combined = Array1::from_vec(
            vote_counts
                .iter()
                .map(|&count| count as f32 / total_votes)
                .collect(),
        );

        Ok(combined)
    }

    /// Product of experts ensemble
    fn combine_product_of_experts(
        &self,
        predictions: &[Array1<f32>],
        output_dim: usize,
    ) -> InferenceResult<Array1<f32>> {
        let mut combined = Array1::ones(output_dim);

        // Multiply all predictions (after softmax)
        for pred in predictions {
            let normalized = self.softmax(pred);
            combined *= &normalized;
        }

        // Normalize the product
        let sum: f32 = combined.sum();
        if sum > 0.0 {
            combined /= sum;
        }

        Ok(combined)
    }

    /// Normalize output to probabilities
    fn normalize(&self, output: &Array1<f32>) -> Array1<f32> {
        self.softmax(output)
    }

    /// Apply temperature-scaled softmax: `softmax(x / temperature)`.
    ///
    /// A `temperature < 1.0` sharpens the distribution towards the largest
    /// element, `> 1.0` smooths it towards uniform. `EnsembleConfig::temperature`
    /// previously built a `Sampler` that nothing ever invoked, making it
    /// silently inert; applying it here is the real, minimal effect it can
    /// have without changing the shape of `step`'s output (which sampling
    /// via `self.sampler` would, by collapsing it to a single index).
    fn softmax(&self, x: &Array1<f32>) -> Array1<f32> {
        let temperature = self.config.temperature;
        let scaled = if temperature.is_finite() && temperature > 0.0 && temperature != 1.0 {
            x.mapv(|v| v / temperature)
        } else {
            x.clone()
        };

        let max_x = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_x = scaled.mapv(|v| (v - max_x).exp());
        let sum_exp: f32 = exp_x.sum();

        if sum_exp > 0.0 {
            exp_x / sum_exp
        } else {
            Array1::from_elem(x.len(), 1.0 / x.len() as f32)
        }
    }

    /// Get ensemble configuration
    pub fn config(&self) -> &EnsembleConfig {
        &self.config
    }

    /// Get mutable access to sampler
    pub fn sampler_mut(&mut self) -> &mut Sampler {
        &mut self.sampler
    }
}

/// Builder for creating model ensembles
pub struct EnsembleBuilder {
    models: Vec<Box<dyn AutoregressiveModel>>,
    config: EnsembleConfig,
}

impl EnsembleBuilder {
    /// Create a new ensemble builder
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            config: EnsembleConfig::default(),
        }
    }

    /// Add a model to the ensemble
    pub fn add_model(mut self, model: Box<dyn AutoregressiveModel>) -> Self {
        self.models.push(model);
        self
    }

    /// Add multiple models
    pub fn add_models(mut self, models: Vec<Box<dyn AutoregressiveModel>>) -> Self {
        self.models.extend(models);
        self
    }

    /// Set ensemble strategy
    pub fn strategy(mut self, strategy: EnsembleStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set model weights
    pub fn weights(mut self, weights: Vec<f32>) -> Self {
        self.config.weights = Some(weights);
        self
    }

    /// Enable/disable output normalization (softmax after combining)
    pub fn normalize_outputs(mut self, normalize: bool) -> Self {
        self.config.normalize_outputs = normalize;
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp;
        self
    }

    /// Build the ensemble
    pub fn build(self) -> InferenceResult<ModelEnsemble> {
        ModelEnsemble::new(self.models, self.config)
    }
}

impl Default for EnsembleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kizzasi_model::s4::{S4Config, S4D};

    #[test]
    fn test_ensemble_creation() {
        let model1 = create_test_model();
        let model2 = create_test_model();

        let ensemble = EnsembleBuilder::new()
            .add_model(Box::new(model1))
            .add_model(Box::new(model2))
            .build();

        assert!(ensemble.is_ok());
        let ensemble = ensemble.unwrap();
        assert_eq!(ensemble.num_models(), 2);
    }

    #[test]
    fn test_ensemble_average() {
        let model1 = create_test_model();
        let model2 = create_test_model();

        let mut ensemble = EnsembleBuilder::new()
            .add_model(Box::new(model1))
            .add_model(Box::new(model2))
            .strategy(EnsembleStrategy::Average)
            .build()
            .unwrap();

        let input = Array1::from_vec(vec![0.5]);
        let output = ensemble.step(&input);

        assert!(output.is_ok());
    }

    #[test]
    fn test_ensemble_weighted() {
        let model1 = create_test_model();
        let model2 = create_test_model();

        let mut ensemble = EnsembleBuilder::new()
            .add_model(Box::new(model1))
            .add_model(Box::new(model2))
            .strategy(EnsembleStrategy::Weighted)
            .weights(vec![0.7, 0.3])
            .build()
            .unwrap();

        let input = Array1::from_vec(vec![0.5]);
        let output = ensemble.step(&input);

        assert!(output.is_ok());
    }

    #[test]
    fn test_ensemble_voting() {
        let model1 = create_test_model();
        let model2 = create_test_model();
        let model3 = create_test_model();

        let mut ensemble = EnsembleBuilder::new()
            .add_model(Box::new(model1))
            .add_model(Box::new(model2))
            .add_model(Box::new(model3))
            .strategy(EnsembleStrategy::Voting)
            .build()
            .unwrap();

        let input = Array1::from_vec(vec![0.5]);
        let output = ensemble.step(&input);

        assert!(output.is_ok());
    }

    #[test]
    fn test_invalid_weights() {
        let model1 = create_test_model();
        let model2 = create_test_model();

        let result = EnsembleBuilder::new()
            .add_model(Box::new(model1))
            .add_model(Box::new(model2))
            .strategy(EnsembleStrategy::Weighted)
            .weights(vec![0.5, 0.6]) // Sum > 1.0
            .build();

        assert!(result.is_err());
    }

    fn create_test_model() -> S4D {
        let config = S4Config::new()
            .input_dim(1)
            .hidden_dim(64)
            .state_dim(16)
            .num_layers(2)
            .diagonal(true);

        S4D::new(config).unwrap()
    }

    /// A model whose `step` always returns a fixed, non-uniform, multi-element
    /// vector — deterministic (unlike S4D's randomly-initialised weights) and
    /// multi-dimensional (unlike `CountingModel`, whose single-element output
    /// can't exercise softmax sharpening/smoothing).
    #[derive(Debug, Clone)]
    struct FixedVectorModel {
        output: Vec<f32>,
    }

    impl FixedVectorModel {
        fn new(output: Vec<f32>) -> Self {
            Self { output }
        }
    }

    impl kizzasi_core::SignalPredictor for FixedVectorModel {
        fn step(&mut self, _input: &Array1<f32>) -> kizzasi_core::CoreResult<Array1<f32>> {
            Ok(Array1::from_vec(self.output.clone()))
        }
        fn reset(&mut self) {}
        fn context_window(&self) -> usize {
            usize::MAX
        }
    }

    impl AutoregressiveModel for FixedVectorModel {
        fn hidden_dim(&self) -> usize {
            self.output.len()
        }
        fn state_dim(&self) -> usize {
            1
        }
        fn num_layers(&self) -> usize {
            1
        }
        fn model_type(&self) -> kizzasi_model::ModelType {
            kizzasi_model::ModelType::S4D
        }
        fn get_states(&self) -> Vec<kizzasi_core::HiddenState> {
            vec![]
        }
        fn set_states(
            &mut self,
            _states: Vec<kizzasi_core::HiddenState>,
        ) -> kizzasi_model::ModelResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_normalize_outputs_defaults_to_false() {
        assert!(!EnsembleConfig::default().normalize_outputs);
    }

    /// Regression: `normalize_outputs` used to default to `true`, and
    /// combination happened *after* averaging despite being documented as
    /// happening before — turning a continuous averaged signal into a
    /// probability simplex by default. With normalization off (the new
    /// default), `Average` over two identical predictions must return that
    /// prediction unchanged.
    #[test]
    fn test_average_without_normalization_returns_raw_prediction() {
        use crate::testutil::CountingModel;
        use kizzasi_core::SignalPredictor;

        let mut solo = CountingModel::new();
        let input = Array1::from_vec(vec![0.5]);
        let expected = solo.step(&input).expect("solo step must succeed");

        let mut ensemble = ModelEnsemble::new(
            vec![
                Box::new(CountingModel::new()),
                Box::new(CountingModel::new()),
            ],
            EnsembleConfig::new(), // normalize_outputs defaults to false
        )
        .expect("ensemble must build");

        let combined = ensemble.step(&input).expect("ensemble step must succeed");
        assert_eq!(
            combined, expected,
            "averaging two identical CountingModel predictions with normalization off must \
             return the raw prediction, not a softmaxed one"
        );
    }

    /// Regression: `EnsembleConfig::temperature` built a `Sampler` that
    /// nothing ever invoked, so it had no effect on `step`'s output.
    #[test]
    fn test_temperature_changes_normalized_distribution_sharpness() {
        let input = Array1::from_vec(vec![0.0]);

        let mut sharp = ModelEnsemble::new(
            vec![
                Box::new(FixedVectorModel::new(vec![1.0, 2.0, 3.0])),
                Box::new(FixedVectorModel::new(vec![1.0, 2.0, 3.0])),
            ],
            EnsembleConfig::new()
                .normalize_outputs(true)
                .temperature(0.1),
        )
        .unwrap();
        let sharp_out = sharp.step(&input).unwrap();

        let mut smooth = ModelEnsemble::new(
            vec![
                Box::new(FixedVectorModel::new(vec![1.0, 2.0, 3.0])),
                Box::new(FixedVectorModel::new(vec![1.0, 2.0, 3.0])),
            ],
            EnsembleConfig::new()
                .normalize_outputs(true)
                .temperature(5.0),
        )
        .unwrap();
        let smooth_out = smooth.step(&input).unwrap();

        // Averaging two identical [1,2,3] predictions gives [1,2,3] either
        // way pre-softmax; temperature must still change how sharply softmax
        // spreads it across the three elements.
        assert!(
            (sharp_out[2] - smooth_out[2]).abs() > 0.05,
            "temperature must change the normalized distribution's sharpness: sharp={:?} smooth={:?}",
            sharp_out.to_vec(),
            smooth_out.to_vec()
        );
        assert!(
            sharp_out[2] > 0.9,
            "low temperature must sharpen toward the max element, got {:?}",
            sharp_out.to_vec()
        );
    }

    #[test]
    fn test_ensemble_builder_normalize_outputs() {
        let model1 = create_test_model();
        let model2 = create_test_model();

        let ensemble = EnsembleBuilder::new()
            .add_model(Box::new(model1))
            .add_model(Box::new(model2))
            .normalize_outputs(true)
            .build()
            .expect("ensemble must build");

        assert!(ensemble.config().normalize_outputs);
    }
}
