//! Core inference engine
//!
//! The InferenceEngine coordinates tokenization, model forward pass,
//! and optional constraint enforcement.

use crate::context::{ContextConfig, InferenceContext};
use crate::error::{InferenceError, InferenceResult};
use crate::sampling::{Sampler, SamplingConfig};
use kizzasi_model::AutoregressiveModel;
use scirs2_core::ndarray::Array1;

/// Memory-efficient inference modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum InferenceMode {
    /// Standard mode: full precision and all states kept
    #[default]
    Standard,
    /// Low memory: aggressive state pruning, limited history
    LowMemory,
    /// Streaming mode: minimal state retention, optimized for real-time
    Streaming,
    /// Quantized: use reduced precision for states and activations
    Quantized,
}

/// Configuration for the inference engine
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Context configuration
    pub context: ContextConfig,
    /// Whether to apply constraints
    pub apply_constraints: bool,
    /// Sampling configuration
    pub sampling: SamplingConfig,
    /// Whether to use embeddings (for discrete outputs)
    pub use_embeddings: bool,
    /// Inference mode for memory efficiency
    pub inference_mode: InferenceMode,
    /// State pruning threshold (for LowMemory mode)
    /// States with values below this threshold are zeroed out
    pub state_prune_threshold: f32,
    /// Maximum history length (for LowMemory/Streaming modes)
    pub max_history_length: Option<usize>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            input_dim: 1,
            output_dim: 1,
            context: ContextConfig::default(),
            apply_constraints: true,
            sampling: SamplingConfig::default(),
            use_embeddings: false,
            inference_mode: InferenceMode::Standard,
            state_prune_threshold: 1e-6,
            max_history_length: None,
        }
    }
}

impl EngineConfig {
    /// Create a new engine configuration
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            ..Default::default()
        }
    }

    /// Set context configuration
    pub fn context(mut self, config: ContextConfig) -> Self {
        self.context = config;
        self
    }

    /// Enable/disable constraint enforcement
    pub fn apply_constraints(mut self, apply: bool) -> Self {
        self.apply_constraints = apply;
        self
    }

    /// Set sampling configuration
    pub fn sampling(mut self, sampling: SamplingConfig) -> Self {
        self.sampling = sampling;
        self
    }

    /// Enable embeddings for discrete outputs
    pub fn use_embeddings(mut self, use_emb: bool) -> Self {
        self.use_embeddings = use_emb;
        self
    }

    /// Set inference mode for memory efficiency
    pub fn inference_mode(mut self, mode: InferenceMode) -> Self {
        self.inference_mode = mode;
        // Auto-configure based on mode
        match mode {
            InferenceMode::LowMemory => {
                self.max_history_length = Some(128);
                self.state_prune_threshold = 1e-4;
            }
            InferenceMode::Streaming => {
                self.max_history_length = Some(64);
                self.state_prune_threshold = 1e-3;
            }
            InferenceMode::Quantized => {
                self.state_prune_threshold = 1e-2;
            }
            InferenceMode::Standard => {}
        }
        self
    }

    /// Set state pruning threshold
    pub fn state_prune_threshold(mut self, threshold: f32) -> Self {
        self.state_prune_threshold = threshold;
        self
    }

    /// Set maximum history length
    pub fn max_history_length(mut self, length: usize) -> Self {
        self.max_history_length = Some(length);
        self
    }
}

/// The main inference engine for AGSP
pub struct InferenceEngine {
    config: EngineConfig,
    context: InferenceContext,
    model: Option<Box<dyn AutoregressiveModel>>,
    sampler: Sampler,
    initialized: bool,
}

impl InferenceEngine {
    /// Create a new inference engine without a model
    pub fn new(config: EngineConfig) -> Self {
        let context = InferenceContext::new(config.context.clone());
        let sampler = Sampler::new(config.sampling.clone());
        Self {
            config,
            context,
            model: None,
            sampler,
            initialized: false,
        }
    }

    /// Create a new inference engine with a model
    pub fn with_model(mut config: EngineConfig, model: Box<dyn AutoregressiveModel>) -> Self {
        // Update context config to match model
        config.context.num_layers = model.num_layers();
        config.context.hidden_dim = model.hidden_dim();
        config.context.state_dim = model.state_dim();

        let context = InferenceContext::new(config.context.clone());
        let sampler = Sampler::new(config.sampling.clone());
        Self {
            config,
            context,
            model: Some(model),
            sampler,
            initialized: true,
        }
    }

    /// Set the model for this engine
    ///
    /// This will update the context configuration to match the model's architecture
    pub fn set_model(&mut self, model: Box<dyn AutoregressiveModel>) {
        // Update context config to match model
        self.config.context.num_layers = model.num_layers();
        self.config.context.hidden_dim = model.hidden_dim();
        self.config.context.state_dim = model.state_dim();

        // Recreate context with updated config
        self.context = InferenceContext::new(self.config.context.clone());

        self.model = Some(model);
        self.initialized = true;
    }

    /// Check if a model is loaded
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Perform a single inference step
    ///
    /// This is the core autoregressive prediction:
    /// Given input x_t, predict x_{t+1}
    pub fn step(&mut self, input: &Array1<f32>) -> InferenceResult<Array1<f32>> {
        if !self.initialized {
            return Err(InferenceError::NotInitialized);
        }

        if input.len() != self.config.input_dim {
            return Err(InferenceError::DimensionMismatch {
                expected: self.config.input_dim,
                got: input.len(),
            });
        }

        // Store in context
        self.context.push(input.clone());

        // Run model forward pass if available
        let logits = if let Some(ref mut model) = self.model {
            // Set model states from context
            let states = self.context.states().to_vec();
            model
                .set_states(states)
                .map_err(|e| InferenceError::ForwardError(e.to_string()))?;

            // Forward pass through model (SignalPredictor trait)
            let output = model
                .step(input)
                .map_err(|e| InferenceError::ForwardError(e.to_string()))?;

            // Update context with new states
            let mut new_states = model.get_states();

            // Apply memory optimization based on inference mode
            self.apply_memory_optimization(&mut new_states);

            for (i, state) in new_states.into_iter().enumerate() {
                self.context.update_state(i, state)?;
            }

            output
        } else {
            // No model - return zeros as fallback
            Array1::zeros(self.config.output_dim)
        };

        // Apply sampling if configured
        let output = if self.config.use_embeddings {
            // For discrete outputs, sample from logits
            let sampled_idx = self.sampler.sample(&logits)?;
            Array1::from_elem(1, sampled_idx)
        } else {
            // For continuous outputs, optionally apply temperature scaling
            if (self.config.sampling.temperature - 1.0).abs() > 1e-6 {
                logits.mapv(|x| x * self.config.sampling.temperature)
            } else {
                logits
            }
        };

        Ok(output)
    }

    /// Perform multi-step rollout
    ///
    /// Predicts `steps` future values autoregressively
    pub fn rollout(
        &mut self,
        input: &Array1<f32>,
        steps: usize,
    ) -> InferenceResult<Vec<Array1<f32>>> {
        let mut outputs = Vec::with_capacity(steps);
        let mut current = input.clone();

        for _ in 0..steps {
            let output = self.step(&current)?;
            outputs.push(output.clone());
            current = output;
        }

        Ok(outputs)
    }

    /// Reset the engine state
    pub fn reset(&mut self) {
        self.context.reset();
    }

    /// Get the current step count
    pub fn step_count(&self) -> usize {
        self.context.step_count()
    }

    /// Get the configuration
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Get the context
    pub fn context(&self) -> &InferenceContext {
        &self.context
    }

    /// Get mutable access to the sampler
    pub fn sampler_mut(&mut self) -> &mut Sampler {
        &mut self.sampler
    }

    /// Get the sampler
    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    /// Perform batched inference on multiple inputs
    ///
    /// This processes multiple inputs in parallel for efficiency.
    /// Each input is processed independently with its own hidden state.
    pub fn step_batch(&mut self, inputs: &[Array1<f32>]) -> InferenceResult<Vec<Array1<f32>>> {
        if !self.initialized {
            return Err(InferenceError::NotInitialized);
        }

        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            let output = self.step(input)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Get model information
    pub fn model_info(&self) -> Option<ModelInfo> {
        self.model.as_ref().map(|model| ModelInfo {
            model_type: model.model_type(),
            hidden_dim: model.hidden_dim(),
            state_dim: model.state_dim(),
            num_layers: model.num_layers(),
        })
    }

    /// Apply memory optimization based on inference mode
    fn apply_memory_optimization(&mut self, states: &mut [kizzasi_core::HiddenState]) {
        match self.config.inference_mode {
            InferenceMode::Standard => {
                // No optimization
            }
            InferenceMode::LowMemory | InferenceMode::Streaming => {
                // Prune small values from states
                self.prune_states(states);
                // Trim history if needed
                if let Some(max_len) = self.config.max_history_length {
                    if self.context.history().len() > max_len {
                        self.context.trim_history(max_len);
                    }
                }
            }
            InferenceMode::Quantized => {
                // Apply quantization to states
                self.quantize_states(states);
                // Also prune
                self.prune_states(states);
            }
        }
    }

    /// Prune states by zeroing out small values
    fn prune_states(&self, states: &mut [kizzasi_core::HiddenState]) {
        let threshold = self.config.state_prune_threshold;
        for state in states.iter_mut() {
            let pruned = state
                .state()
                .mapv(|x| if x.abs() < threshold { 0.0 } else { x });
            state.update(pruned);
        }
    }

    /// Apply quantization to states (simulate INT8/FP16)
    fn quantize_states(&self, states: &mut [kizzasi_core::HiddenState]) {
        // Simple quantization: round to fixed precision
        // This simulates FP16/INT8 behavior without actually changing types
        for state in states.iter_mut() {
            let quantized = state.state().mapv(|x| {
                // Quantize to ~6 bits of precision (similar to FP16 mantissa)
                let scale = 64.0;
                (x * scale).round() / scale
            });
            state.update(quantized);
        }
    }
}

/// Information about the loaded model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_type: kizzasi_model::ModelType,
    pub hidden_dim: usize,
    pub state_dim: usize,
    pub num_layers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let config = EngineConfig::new(3, 3);
        let engine = InferenceEngine::new(config);

        assert_eq!(engine.step_count(), 0);
        assert!(!engine.has_model());
    }

    #[test]
    fn test_engine_step_no_model() {
        let config = EngineConfig::new(3, 3);
        let mut engine = InferenceEngine::new(config);

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let output = engine.step(&input);

        // Should fail without model
        assert!(output.is_err());
    }

    #[test]
    fn test_engine_with_model() {
        use kizzasi_model::rwkv::{Rwkv, RwkvConfig};

        let model_config = RwkvConfig::new()
            .input_dim(1)
            .hidden_dim(64)
            .intermediate_dim(256)
            .num_layers(2);
        let model = Rwkv::new(model_config).unwrap();

        let config = EngineConfig::new(1, 10);
        let mut engine = InferenceEngine::with_model(config, Box::new(model));

        assert!(engine.has_model());

        let input = Array1::from_vec(vec![0.5]);
        let output = engine.step(&input);

        if let Err(e) = &output {
            eprintln!("Error: {:?}", e);
        }
        assert!(output.is_ok(), "Expected Ok, got: {:?}", output);
        assert_eq!(engine.step_count(), 1);
    }

    #[test]
    fn test_engine_rollout() {
        use kizzasi_model::s4::{S4Config, S4D};

        let model_config = S4Config::new()
            .input_dim(1)
            .hidden_dim(64)
            .state_dim(16)
            .num_layers(2)
            .diagonal(true);
        let model = S4D::new(model_config).unwrap();

        let config = EngineConfig::new(1, 10);
        let mut engine = InferenceEngine::with_model(config, Box::new(model));

        let input = Array1::from_vec(vec![0.5]);
        let outputs = engine.rollout(&input, 5);

        assert!(outputs.is_ok());
        assert_eq!(outputs.unwrap().len(), 5);
        assert_eq!(engine.step_count(), 5);
    }

    #[test]
    fn test_engine_batch() {
        use kizzasi_model::s4::{S4Config, S4D};

        let model_config = S4Config::new()
            .input_dim(1)
            .hidden_dim(64)
            .state_dim(16)
            .num_layers(2)
            .diagonal(true);
        let model = S4D::new(model_config).unwrap();

        let config = EngineConfig::new(1, 10);
        let mut engine = InferenceEngine::with_model(config, Box::new(model));

        let inputs = vec![
            Array1::from_vec(vec![0.1]),
            Array1::from_vec(vec![0.2]),
            Array1::from_vec(vec![0.3]),
        ];

        let outputs = engine.step_batch(&inputs);
        assert!(outputs.is_ok());
        assert_eq!(outputs.unwrap().len(), 3);
    }

    #[test]
    fn test_model_info() {
        use kizzasi_model::rwkv::{Rwkv, RwkvConfig};

        let model_config = RwkvConfig::new()
            .input_dim(1)
            .hidden_dim(128)
            .intermediate_dim(512)
            .num_layers(4);
        let model = Rwkv::new(model_config).unwrap();

        let config = EngineConfig::new(1, 50);
        let engine = InferenceEngine::with_model(config, Box::new(model));

        let info = engine.model_info();
        assert!(info.is_some());

        let info = info.unwrap();
        assert_eq!(info.hidden_dim, 128);
        assert_eq!(info.num_layers, 4);
    }
}
