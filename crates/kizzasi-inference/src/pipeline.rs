//! Inference pipeline construction
//!
//! Provides a builder pattern for constructing complete inference pipelines
//! with tokenization, model, and optional constraints.
//!
//! # Pipeline Hooks
//!
//! The pipeline supports preprocessing and postprocessing hooks that allow
//! custom transformations at different stages:
//!
//! ```text
//! Input → [Preprocess] → Tokenize → Model → Constrain → [Postprocess] → Output
//! ```

use crate::engine::{EngineConfig, InferenceEngine};
use crate::error::{InferenceError, InferenceResult};
use kizzasi_logic::{ConstrainedInference, GuardrailSet};
use kizzasi_model::AutoregressiveModel;
use kizzasi_tokenizer::SignalTokenizer;
use scirs2_core::ndarray::Array1;
use std::sync::Arc;

/// Preprocessing hook that transforms input before model forward pass
pub type PreprocessHook = Arc<dyn Fn(&Array1<f32>) -> InferenceResult<Array1<f32>> + Send + Sync>;

/// Postprocessing hook that transforms output after model forward pass
pub type PostprocessHook = Arc<dyn Fn(&Array1<f32>) -> InferenceResult<Array1<f32>> + Send + Sync>;

/// A complete inference pipeline
pub struct Pipeline {
    /// The inference engine
    engine: InferenceEngine,
    /// Optional tokenizer for preprocessing
    tokenizer: Option<Box<dyn SignalTokenizer>>,
    /// Whether to apply tokenization
    use_tokenizer: bool,
    /// Whether constraints are enabled
    constraints_enabled: bool,
    /// Guardrail set for constraint enforcement
    guardrails: Option<GuardrailSet>,
    /// Preprocessing hooks (applied before tokenization)
    preprocess_hooks: Vec<PreprocessHook>,
    /// Postprocessing hooks (applied after detokenization)
    postprocess_hooks: Vec<PostprocessHook>,
}

impl Pipeline {
    /// Create a single prediction step through the pipeline
    ///
    /// The complete flow is:
    /// 1. Apply preprocessing hooks
    /// 2. Optional tokenization of input signal
    /// 3. Model forward pass
    /// 4. Optional constraint enforcement
    /// 5. Optional detokenization
    /// 6. Apply postprocessing hooks
    pub fn forward(&mut self, input: &Array1<f32>) -> InferenceResult<Array1<f32>> {
        // Step 1: Preprocessing hooks
        let mut preprocessed = input.clone();
        for hook in &self.preprocess_hooks {
            preprocessed = hook(&preprocessed)?;
        }

        // Step 2: Tokenization (if enabled)
        let tokenized = if self.use_tokenizer {
            if let Some(tokenizer) = &self.tokenizer {
                tokenizer
                    .encode(&preprocessed)
                    .map_err(|e| InferenceError::TokenizationError(e.to_string()))?
            } else {
                return Err(InferenceError::TokenizationError(
                    "Tokenizer enabled but not provided".to_string(),
                ));
            }
        } else {
            preprocessed
        };

        // Step 3: Model forward pass
        let output = self.engine.step(&tokenized)?;

        // Step 4: Constraint enforcement (if enabled)
        let constrained = if self.constraints_enabled {
            self.apply_constraints(&output)?
        } else {
            output
        };

        // Step 5: Detokenization (if enabled)
        let decoded = if self.use_tokenizer {
            if let Some(tokenizer) = &self.tokenizer {
                tokenizer
                    .decode(&constrained)
                    .map_err(|e| InferenceError::TokenizationError(e.to_string()))?
            } else {
                return Err(InferenceError::TokenizationError(
                    "Tokenizer enabled but not provided".to_string(),
                ));
            }
        } else {
            constrained
        };

        // Step 6: Postprocessing hooks
        let mut postprocessed = decoded;
        for hook in &self.postprocess_hooks {
            postprocessed = hook(&postprocessed)?;
        }

        Ok(postprocessed)
    }

    /// Apply constraints to the output
    ///
    /// If guardrails are configured, enforces constraints by projecting
    /// the output onto the constraint-satisfying manifold
    fn apply_constraints(&self, output: &Array1<f32>) -> InferenceResult<Array1<f32>> {
        if let Some(ref guardrails) = self.guardrails {
            // Use guardrails to constrain the output
            guardrails
                .constrain(output)
                .map_err(|e| InferenceError::ConstraintError(e.to_string()))
        } else {
            // No guardrails configured, return unchanged
            Ok(output.clone())
        }
    }

    /// Set the guardrails for constraint enforcement
    pub fn set_guardrails(&mut self, guardrails: GuardrailSet) {
        self.guardrails = Some(guardrails);
        self.constraints_enabled = true;
    }

    /// Remove guardrails
    pub fn clear_guardrails(&mut self) {
        self.guardrails = None;
        self.constraints_enabled = false;
    }

    /// Get a reference to the guardrails
    pub fn guardrails(&self) -> Option<&GuardrailSet> {
        self.guardrails.as_ref()
    }

    /// Perform multi-step prediction through the pipeline
    pub fn rollout(
        &mut self,
        initial: &Array1<f32>,
        steps: usize,
    ) -> InferenceResult<Vec<Array1<f32>>> {
        let mut outputs = Vec::with_capacity(steps);
        let mut current = initial.clone();

        for _ in 0..steps {
            let output = self.forward(&current)?;
            outputs.push(output.clone());
            current = output;
        }

        Ok(outputs)
    }

    /// Reset the pipeline state
    pub fn reset(&mut self) {
        self.engine.reset();
    }

    /// Get the underlying engine
    pub fn engine(&self) -> &InferenceEngine {
        &self.engine
    }

    /// Get mutable access to the engine
    pub fn engine_mut(&mut self) -> &mut InferenceEngine {
        &mut self.engine
    }

    /// Check if constraints are enabled
    pub fn has_constraints(&self) -> bool {
        self.constraints_enabled
    }

    /// Check if tokenizer is enabled
    pub fn has_tokenizer(&self) -> bool {
        self.use_tokenizer && self.tokenizer.is_some()
    }

    /// Add a preprocessing hook
    pub fn add_preprocess_hook(&mut self, hook: PreprocessHook) {
        self.preprocess_hooks.push(hook);
    }

    /// Add a postprocessing hook
    pub fn add_postprocess_hook(&mut self, hook: PostprocessHook) {
        self.postprocess_hooks.push(hook);
    }

    /// Get number of preprocessing hooks
    pub fn num_preprocess_hooks(&self) -> usize {
        self.preprocess_hooks.len()
    }

    /// Get number of postprocessing hooks
    pub fn num_postprocess_hooks(&self) -> usize {
        self.postprocess_hooks.len()
    }

    /// Clear all preprocessing hooks
    pub fn clear_preprocess_hooks(&mut self) {
        self.preprocess_hooks.clear();
    }

    /// Clear all postprocessing hooks
    pub fn clear_postprocess_hooks(&mut self) {
        self.postprocess_hooks.clear();
    }
}

/// Builder for constructing inference pipelines
pub struct PipelineBuilder {
    engine_config: Option<EngineConfig>,
    model: Option<Box<dyn AutoregressiveModel>>,
    tokenizer: Option<Box<dyn SignalTokenizer>>,
    use_tokenizer: bool,
    constraints_enabled: bool,
    guardrails: Option<GuardrailSet>,
    preprocess_hooks: Vec<PreprocessHook>,
    postprocess_hooks: Vec<PostprocessHook>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            engine_config: None,
            model: None,
            tokenizer: None,
            use_tokenizer: false,
            constraints_enabled: false,
            guardrails: None,
            preprocess_hooks: Vec::new(),
            postprocess_hooks: Vec::new(),
        }
    }

    /// Set the engine configuration
    pub fn engine_config(mut self, config: EngineConfig) -> Self {
        self.engine_config = Some(config);
        self
    }

    /// Set the model
    pub fn model(mut self, model: Box<dyn AutoregressiveModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the tokenizer
    pub fn tokenizer(mut self, tokenizer: Box<dyn SignalTokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self.use_tokenizer = true;
        self
    }

    /// Enable/disable tokenizer usage
    pub fn use_tokenizer(mut self, use_tok: bool) -> Self {
        self.use_tokenizer = use_tok;
        self
    }

    /// Enable constraint enforcement
    pub fn with_constraints(mut self) -> Self {
        self.constraints_enabled = true;
        self
    }

    /// Set guardrails for constraint enforcement
    pub fn guardrails(mut self, guardrails: GuardrailSet) -> Self {
        self.guardrails = Some(guardrails);
        self.constraints_enabled = true;
        self
    }

    /// Add a preprocessing hook
    pub fn add_preprocess_hook(mut self, hook: PreprocessHook) -> Self {
        self.preprocess_hooks.push(hook);
        self
    }

    /// Add a postprocessing hook
    pub fn add_postprocess_hook(mut self, hook: PostprocessHook) -> Self {
        self.postprocess_hooks.push(hook);
        self
    }

    /// Build the pipeline
    pub fn build(self) -> InferenceResult<Pipeline> {
        let engine_config = self
            .engine_config
            .ok_or_else(|| InferenceError::PipelineConfig("engine_config not set".into()))?;

        let engine = if let Some(model) = self.model {
            InferenceEngine::with_model(engine_config, model)
        } else {
            InferenceEngine::new(engine_config)
        };

        Ok(Pipeline {
            engine,
            tokenizer: self.tokenizer,
            use_tokenizer: self.use_tokenizer,
            constraints_enabled: self.constraints_enabled,
            guardrails: self.guardrails,
            preprocess_hooks: self.preprocess_hooks,
            postprocess_hooks: self.postprocess_hooks,
        })
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplingConfig;

    #[test]
    fn test_pipeline_builder_basic() {
        let engine_config = EngineConfig::new(3, 3);
        let pipeline = PipelineBuilder::new()
            .engine_config(engine_config)
            .with_constraints()
            .build();

        assert!(pipeline.is_ok());
        let p = pipeline.unwrap();
        assert!(p.has_constraints());
        assert!(!p.has_tokenizer());
    }

    #[test]
    fn test_pipeline_missing_config() {
        let result = PipelineBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_with_model() {
        use kizzasi_model::s4::{S4Config, S4D};

        let model_config = S4Config::new()
            .input_dim(1)
            .hidden_dim(64)
            .state_dim(16)
            .num_layers(2)
            .diagonal(true);
        let model = S4D::new(model_config).unwrap();

        let engine_config = EngineConfig::new(1, 10);
        let mut pipeline = PipelineBuilder::new()
            .engine_config(engine_config)
            .model(Box::new(model))
            .build()
            .unwrap();

        let input = Array1::from_vec(vec![0.5]);
        let output = pipeline.forward(&input);

        assert!(output.is_ok());
    }

    #[test]
    fn test_pipeline_rollout() {
        use kizzasi_model::rwkv::{Rwkv, RwkvConfig};

        let model_config = RwkvConfig::new()
            .input_dim(1)
            .hidden_dim(64)
            .intermediate_dim(256)
            .num_layers(2);
        let model = Rwkv::new(model_config).unwrap();

        let engine_config = EngineConfig::new(1, 10);
        let mut pipeline = PipelineBuilder::new()
            .engine_config(engine_config)
            .model(Box::new(model))
            .build()
            .unwrap();

        let initial = Array1::from_vec(vec![0.5]);
        let outputs = pipeline.rollout(&initial, 5);

        assert!(outputs.is_ok());
        assert_eq!(outputs.unwrap().len(), 5);
    }

    #[test]
    fn test_pipeline_reset() {
        let engine_config = EngineConfig::new(1, 1);
        let mut pipeline = PipelineBuilder::new()
            .engine_config(engine_config)
            .build()
            .unwrap();

        pipeline.reset();
        assert_eq!(pipeline.engine().step_count(), 0);
    }

    #[test]
    fn test_pipeline_with_sampling() {
        use crate::sampling::SamplingStrategy;
        use kizzasi_model::s4::{S4Config, S4D};

        let model_config = S4Config::new()
            .input_dim(1)
            .hidden_dim(64)
            .state_dim(16)
            .num_layers(2)
            .diagonal(true);
        let model = S4D::new(model_config).unwrap();

        let sampling = SamplingConfig::new()
            .strategy(SamplingStrategy::TopK)
            .top_k(5);

        let engine_config = EngineConfig::new(1, 10)
            .sampling(sampling)
            .use_embeddings(true);

        let mut pipeline = PipelineBuilder::new()
            .engine_config(engine_config)
            .model(Box::new(model))
            .build()
            .unwrap();

        let input = Array1::from_vec(vec![0.5]);
        let output = pipeline.forward(&input);

        assert!(output.is_ok());
    }
}
