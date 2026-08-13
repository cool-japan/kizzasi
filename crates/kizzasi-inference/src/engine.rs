//! Core inference engine
//!
//! The InferenceEngine coordinates tokenization, model forward pass,
//! and optional constraint enforcement.

use crate::context::{ContextConfig, InferenceContext};
use crate::error::{InferenceError, InferenceResult};
use crate::lora::LoraAdapterManager;
use crate::pool::TensorPool;
use crate::precision::{PrecisionConfig, PrecisionConverter};
use crate::sampling::{Sampler, SamplingConfig};
use kizzasi_core::HiddenState;
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
    /// Whether a pipeline built from this configuration must enforce constraints
    ///
    /// Read by [`crate::pipeline::PipelineBuilder::build`]. Enforcement needs
    /// something to enforce, so building a pipeline with this set to `true` and no
    /// [`kizzasi_logic::GuardrailSet`] is a configuration error rather than a
    /// silent pass-through. Defaults to `false` because no guardrails are
    /// configured by default.
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
    /// Numeric precision applied to the model output of every step
    ///
    /// Defaults to [`PrecisionMode::FP32`](crate::precision::PrecisionMode::FP32),
    /// which leaves the output untouched.
    #[serde(default)]
    pub precision: PrecisionConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            input_dim: 1,
            output_dim: 1,
            context: ContextConfig::default(),
            apply_constraints: false,
            sampling: SamplingConfig::default(),
            use_embeddings: false,
            inference_mode: InferenceMode::Standard,
            state_prune_threshold: 1e-6,
            max_history_length: None,
            precision: PrecisionConfig::default(),
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

    /// Enable/disable constraint enforcement for pipelines built from this config
    ///
    /// See [`EngineConfig::apply_constraints`] — enabling this requires a
    /// [`kizzasi_logic::GuardrailSet`] to be supplied to the pipeline builder.
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

    /// Set the numeric precision applied to model outputs
    pub fn precision(mut self, precision: PrecisionConfig) -> Self {
        self.precision = precision;
        self
    }

    /// Set inference mode for memory efficiency
    ///
    /// Each mode carries defaults for `max_history_length` and
    /// `state_prune_threshold`. Those defaults are applied **only** to fields that
    /// are still at their struct default, so
    /// `EngineConfig::new(1, 1).state_prune_threshold(1e-8).inference_mode(mode)`
    /// and the reverse call order produce the same configuration — the builder is
    /// order-independent and never discards an explicitly configured value.
    pub fn inference_mode(mut self, mode: InferenceMode) -> Self {
        self.inference_mode = mode;

        let defaults = Self::default();
        // Mode-derived defaults: (max_history_length, state_prune_threshold)
        let (mode_history, mode_threshold) = match mode {
            InferenceMode::LowMemory => (Some(128), Some(1e-4)),
            InferenceMode::Streaming => (Some(64), Some(1e-3)),
            InferenceMode::Quantized => (None, Some(1e-2)),
            InferenceMode::Standard => (None, None),
        };

        if let Some(history) = mode_history {
            if self.max_history_length == defaults.max_history_length {
                self.max_history_length = Some(history);
            }
        }
        if let Some(threshold) = mode_threshold {
            if (self.state_prune_threshold - defaults.state_prune_threshold).abs() <= f32::EPSILON {
                self.state_prune_threshold = threshold;
            }
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

/// Reconcile one user-supplied context dimension against the attached model
///
/// The three architectural context dimensions are derived from the model. When the
/// caller left the field at its `ContextConfig` default it is filled in silently;
/// when the caller configured a *different* value the disagreement is reported —
/// as an error in `strict` mode, as a `tracing` warning otherwise. It is never
/// discarded silently.
fn reconcile_context_dim(
    field: &'static str,
    configured: &mut usize,
    default: usize,
    model_value: usize,
    strict: bool,
) -> InferenceResult<()> {
    if *configured == model_value {
        return Ok(());
    }

    if *configured != default {
        if strict {
            return Err(InferenceError::InvalidConfiguration(format!(
                "ContextConfig::{} is {} but the attached model reports {}; \
                 these dimensions are derived from the model architecture",
                field, configured, model_value
            )));
        }
        tracing::warn!(
            field,
            configured = *configured,
            model = model_value,
            "ContextConfig dimension disagrees with the attached model; adopting the model value"
        );
    }

    *configured = model_value;
    Ok(())
}

/// Align a context configuration with the architecture of `model`
fn align_context_with_model(
    context: &mut ContextConfig,
    model: &dyn AutoregressiveModel,
    strict: bool,
) -> InferenceResult<()> {
    let defaults = ContextConfig::default();
    reconcile_context_dim(
        "num_layers",
        &mut context.num_layers,
        defaults.num_layers,
        model.num_layers(),
        strict,
    )?;
    reconcile_context_dim(
        "hidden_dim",
        &mut context.hidden_dim,
        defaults.hidden_dim,
        model.hidden_dim(),
        strict,
    )?;
    reconcile_context_dim(
        "state_dim",
        &mut context.state_dim,
        defaults.state_dim,
        model.state_dim(),
        strict,
    )?;
    Ok(())
}

/// Seed a context's hidden states from the model's own (zeroed) initial states
///
/// [`ContextConfig`] describes a state as `hidden_dim x state_dim`, but the
/// [`AutoregressiveModel`] contract only promises that `get_states` and
/// `set_states` round-trip; a model is free to lay its recurrent state out
/// differently (RWKV packs several per-head vectors into a single matrix, so its
/// state has `3 * heads * head_dim + 2 * hidden_dim` rows). Taking the shape from
/// the model keeps that round-trip valid for every architecture instead of
/// assuming one and having `set_states` reject the result.
fn seed_states_from_model(context: &mut InferenceContext, model: &dyn AutoregressiveModel) {
    let mut states = model.get_states();
    if states.is_empty() {
        return;
    }
    for state in &mut states {
        state.reset();
    }
    context.restore_states(states);
}

/// The main inference engine for AGSP
pub struct InferenceEngine {
    config: EngineConfig,
    context: InferenceContext,
    model: Option<Box<dyn AutoregressiveModel>>,
    sampler: Sampler,
    precision: PrecisionConverter,
    lora: Option<LoraAdapterManager>,
    initialized: bool,
}

impl InferenceEngine {
    /// Create a new inference engine without a model
    pub fn new(config: EngineConfig) -> Self {
        let context = InferenceContext::new(config.context.clone());
        let sampler = Sampler::new(config.sampling.clone());
        let precision = PrecisionConverter::new(config.precision.clone());
        Self {
            config,
            context,
            model: None,
            sampler,
            precision,
            lora: None,
            initialized: false,
        }
    }

    /// Create a new inference engine with a model
    ///
    /// `config.context.num_layers`, `hidden_dim` and `state_dim` are derived from
    /// the model architecture. If the caller configured different values the model
    /// wins and a `tracing` warning is emitted; use
    /// [`InferenceEngine::try_with_model`] to turn that disagreement into an error.
    pub fn with_model(mut config: EngineConfig, model: Box<dyn AutoregressiveModel>) -> Self {
        // Cannot fail in non-strict mode.
        let _ = align_context_with_model(&mut config.context, model.as_ref(), false);
        Self::assemble_with_model(config, model)
    }

    /// Create a new inference engine with a model, validating the context config
    ///
    /// Returns [`InferenceError::InvalidConfiguration`] when the caller configured
    /// a context dimension that disagrees with the model architecture.
    pub fn try_with_model(
        mut config: EngineConfig,
        model: Box<dyn AutoregressiveModel>,
    ) -> InferenceResult<Self> {
        align_context_with_model(&mut config.context, model.as_ref(), true)?;
        Ok(Self::assemble_with_model(config, model))
    }

    fn assemble_with_model(config: EngineConfig, model: Box<dyn AutoregressiveModel>) -> Self {
        let mut context = InferenceContext::new(config.context.clone());
        seed_states_from_model(&mut context, model.as_ref());
        let sampler = Sampler::new(config.sampling.clone());
        let precision = PrecisionConverter::new(config.precision.clone());
        Self {
            config,
            context,
            model: Some(model),
            sampler,
            precision,
            lora: None,
            initialized: true,
        }
    }

    /// Set the model for this engine
    ///
    /// This updates the context configuration to match the model's architecture and
    /// resets the accumulated context. Configured context dimensions that disagree
    /// with the model are reported through `tracing::warn!`; use
    /// [`InferenceEngine::try_set_model`] to turn them into an error instead.
    pub fn set_model(&mut self, model: Box<dyn AutoregressiveModel>) {
        let mut context = self.config.context.clone();
        // Cannot fail in non-strict mode.
        let _ = align_context_with_model(&mut context, model.as_ref(), false);
        self.install_model(context, model);
    }

    /// Set the model for this engine, validating the context configuration
    ///
    /// On error the engine is left completely untouched.
    pub fn try_set_model(&mut self, model: Box<dyn AutoregressiveModel>) -> InferenceResult<()> {
        let mut context = self.config.context.clone();
        align_context_with_model(&mut context, model.as_ref(), true)?;
        self.install_model(context, model);
        Ok(())
    }

    fn install_model(&mut self, context: ContextConfig, model: Box<dyn AutoregressiveModel>) {
        self.config.context = context;
        // Recreate context with updated config
        self.context = InferenceContext::new(self.config.context.clone());
        seed_states_from_model(&mut self.context, model.as_ref());
        self.model = Some(model);
        self.initialized = true;
    }

    /// Check if a model is loaded
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Attach a LoRA adapter manager
    ///
    /// The active adapter (if any) is applied to the model output of every
    /// [`InferenceEngine::step`], before sampling.
    pub fn set_lora_manager(&mut self, manager: LoraAdapterManager) {
        self.lora = Some(manager);
    }

    /// Remove the attached LoRA adapter manager, returning it
    pub fn take_lora_manager(&mut self) -> Option<LoraAdapterManager> {
        self.lora.take()
    }

    /// Get the attached LoRA adapter manager
    pub fn lora_manager(&self) -> Option<&LoraAdapterManager> {
        self.lora.as_ref()
    }

    /// Get mutable access to the attached LoRA adapter manager
    pub fn lora_manager_mut(&mut self) -> Option<&mut LoraAdapterManager> {
        self.lora.as_mut()
    }

    /// Clone the engine's current hidden states
    pub fn clone_states(&self) -> Vec<HiddenState> {
        self.context.states().to_vec()
    }

    /// Swap the engine's hidden states, returning the previous ones
    ///
    /// This is the primitive that lets several logical requests share a single
    /// engine without contaminating each other's autoregressive state: swap the
    /// request's states in, call [`InferenceEngine::step`], then swap them back out.
    /// No copy is made.
    pub fn swap_states(&mut self, states: Vec<HiddenState>) -> InferenceResult<Vec<HiddenState>> {
        if states.len() != self.config.context.num_layers {
            return Err(InferenceError::DimensionMismatch {
                expected: self.config.context.num_layers,
                got: states.len(),
            });
        }
        let previous = self.context.take_states();
        self.context.restore_states(states);
        Ok(previous)
    }

    /// Create a fresh, zeroed hidden-state vector matching this engine's model
    ///
    /// The shape is taken from the attached model, which is the only authority on
    /// its own state layout; without a model it falls back to the context
    /// configuration's `hidden_dim x state_dim`.
    pub fn fresh_states(&self) -> Vec<HiddenState> {
        if let Some(ref model) = self.model {
            let mut states = model.get_states();
            if !states.is_empty() {
                for state in &mut states {
                    state.reset();
                }
                return states;
            }
        }

        (0..self.config.context.num_layers)
            .map(|_| {
                HiddenState::new(
                    self.config.context.hidden_dim,
                    self.config.context.state_dim,
                )
            })
            .collect()
    }

    /// Perform a single inference step
    ///
    /// This is the core autoregressive prediction:
    /// Given input x_t, predict x_{t+1}
    pub fn step(&mut self, input: &Array1<f32>) -> InferenceResult<Array1<f32>> {
        let _span = tracing::debug_span!("inference_step", input_len = input.len()).entered();
        if !self.initialized {
            return Err(InferenceError::NotInitialized);
        }

        if input.len() != self.config.input_dim {
            return Err(InferenceError::DimensionMismatch {
                expected: self.config.input_dim,
                got: input.len(),
            });
        }

        // Store in context (clones only when history storage is enabled)
        self.context.push_ref(input);

        // Run model forward pass. `initialized == true` implies a model is present,
        // but the pattern match keeps that invariant local and panic-free.
        let expected_layers = self.config.context.num_layers;
        let logits = if let Some(ref mut model) = self.model {
            // Hand the hidden states to the model by move instead of deep-copying
            // them. The context is left empty until the states are put back below,
            // which happens on every path including the error paths.
            let states = self.context.take_states();
            if let Err(e) = model.set_states(states) {
                self.context.restore_states(model.get_states());
                return Err(InferenceError::ForwardError(e.to_string()));
            }

            // Forward pass through model (SignalPredictor trait)
            let output = match model.step(input) {
                Ok(output) => output,
                Err(e) => {
                    self.context.restore_states(model.get_states());
                    return Err(InferenceError::ForwardError(e.to_string()));
                }
            };

            // Update context with new states
            let mut new_states = model.get_states();
            if new_states.len() != expected_layers {
                let got = new_states.len();
                self.context.restore_states(new_states);
                return Err(InferenceError::DimensionMismatch {
                    expected: expected_layers,
                    got,
                });
            }

            // Apply memory optimization based on inference mode
            self.apply_memory_optimization(&mut new_states);
            self.context.restore_states(new_states);

            output
        } else {
            return Err(InferenceError::NotInitialized);
        };

        // Apply the configured numeric precision to the activations
        let logits = self.precision.apply_1d(logits);

        // Apply the active LoRA adapter, if one is attached and activated
        let logits = match self.lora {
            Some(ref manager) if manager.active_adapter().is_some() => manager.apply(&logits)?,
            _ => logits,
        };

        // Apply sampling if configured
        let output = if self.config.use_embeddings {
            // For discrete outputs, sample from logits. `SamplingConfig::temperature`
            // is applied by the sampler (dividing the logits before the softmax).
            let sampled_idx = self.sampler.sample(&logits)?;
            Array1::from_elem(1, sampled_idx)
        } else {
            // Continuous predictions are returned unmodified. Temperature is only
            // meaningful for logits that feed a softmax; multiplying a continuous
            // prediction by it would be an undocumented output gain applied in the
            // opposite direction to the sampler's division, and it compounds across
            // `rollout` steps.
            logits
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

    /// Attach a [`TensorPool`] to this engine's context, so per-step history
    /// buffer churn (when [`ContextConfig::store_history`] is enabled) reuses
    /// allocations from `pool` instead of allocating fresh each time. This is
    /// the entry point that makes `TensorPool` reachable from an
    /// `InferenceEngine`; [`InferenceEngine::context`] only exposes a shared
    /// reference, so pooling could not otherwise be enabled once a context
    /// exists.
    pub fn enable_pooling(&mut self, pool: TensorPool) {
        self.context.enable_pooling(pool);
    }

    /// Disable memory pooling for this engine's context.
    pub fn disable_pooling(&mut self) {
        self.context.disable_pooling();
    }

    /// Get mutable access to the sampler
    pub fn sampler_mut(&mut self) -> &mut Sampler {
        &mut self.sampler
    }

    /// Get the sampler
    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    /// Perform inference on multiple independent inputs
    ///
    /// Every input is processed from the engine's *current* hidden state, so the
    /// result for input `i` never depends on the inputs before it. The engine's
    /// hidden state is restored to its pre-call value when the batch finishes,
    /// because a set of independent requests has no single successor state.
    ///
    /// Inputs are evaluated sequentially: the model interface takes one signal
    /// vector at a time, so this is a correctness/isolation helper rather than a
    /// throughput optimisation. Use [`crate::batch::BatchScheduler`] for
    /// multi-request scheduling.
    pub fn step_batch(&mut self, inputs: &[Array1<f32>]) -> InferenceResult<Vec<Array1<f32>>> {
        if !self.initialized {
            return Err(InferenceError::NotInitialized);
        }

        let entry_states = self.clone_states();
        let mut outputs = Vec::with_capacity(inputs.len());
        let mut failure = None;

        for input in inputs {
            // Restore the entry state so this input is isolated from the previous one.
            self.swap_states(entry_states.clone())?;
            match self.step(input) {
                Ok(output) => outputs.push(output),
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }

        // Independent inputs have no single successor state: restore the entry state.
        self.swap_states(entry_states)?;

        match failure {
            Some(e) => Err(e),
            None => Ok(outputs),
        }
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

    use crate::testutil::CountingModel;

    fn s4d_model(input_dim: usize) -> kizzasi_model::s4::S4D {
        use kizzasi_model::s4::{S4Config, S4D};

        S4D::new(
            S4Config::new()
                .input_dim(input_dim)
                .hidden_dim(64)
                .state_dim(16)
                .num_layers(2)
                .diagonal(true),
        )
        .expect("S4D construction must succeed")
    }

    /// Regression: on the continuous path `temperature` was applied as an output
    /// gain (multiplication), the inverse of the sampler's division, and it
    /// compounded across `rollout` steps.
    #[test]
    fn test_temperature_does_not_scale_continuous_output() {
        let baseline =
            InferenceEngine::with_model(EngineConfig::new(1, 1), Box::new(CountingModel::new()))
                .rollout(&Array1::from_vec(vec![0.5]), 4)
                .expect("rollout must succeed");

        for temperature in [0.5_f32, 2.0] {
            let config =
                EngineConfig::new(1, 1).sampling(SamplingConfig::new().temperature(temperature));
            let scaled = InferenceEngine::with_model(config, Box::new(CountingModel::new()))
                .rollout(&Array1::from_vec(vec![0.5]), 4)
                .expect("rollout must succeed");
            assert_eq!(
                scaled, baseline,
                "temperature {} must not rescale a continuous prediction",
                temperature
            );
        }
    }

    /// Regression: `step_batch` documented per-input isolation but ran a plain
    /// sequential loop, so input `i` observed the states left by input `i - 1`.
    #[test]
    fn test_step_batch_inputs_are_independent() {
        let inputs = vec![
            Array1::from_vec(vec![0.1]),
            Array1::from_vec(vec![0.2]),
            Array1::from_vec(vec![0.3]),
        ];

        let mut engine =
            InferenceEngine::with_model(EngineConfig::new(1, 1), Box::new(CountingModel::new()));
        let batched = engine.step_batch(&inputs).expect("step_batch must succeed");

        for (input, batched_output) in inputs.iter().zip(batched.iter()) {
            let mut fresh = InferenceEngine::with_model(
                EngineConfig::new(1, 1),
                Box::new(CountingModel::new()),
            );
            let solo = fresh.step(input).expect("step must succeed");
            assert_eq!(&solo, batched_output);
        }
    }

    /// The engine's own state is unchanged by a batch of independent inputs.
    #[test]
    fn test_step_batch_restores_entry_state() {
        let mut engine =
            InferenceEngine::with_model(EngineConfig::new(1, 1), Box::new(CountingModel::new()));
        engine
            .step(&Array1::from_vec(vec![0.4]))
            .expect("step must succeed");
        let before = engine.clone_states();

        engine
            .step_batch(&[Array1::from_vec(vec![0.1]), Array1::from_vec(vec![0.2])])
            .expect("step_batch must succeed");

        let after = engine.clone_states();
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(after.iter()) {
            assert_eq!(a.state(), b.state());
        }
    }

    /// Regression: builder call order changed the resulting configuration because
    /// `inference_mode` overwrote explicitly configured fields.
    #[test]
    fn test_inference_mode_is_order_independent() {
        let threshold_first = EngineConfig::new(1, 1)
            .state_prune_threshold(1e-8)
            .max_history_length(7)
            .inference_mode(InferenceMode::LowMemory);
        let mode_first = EngineConfig::new(1, 1)
            .inference_mode(InferenceMode::LowMemory)
            .state_prune_threshold(1e-8)
            .max_history_length(7);

        assert_eq!(
            threshold_first.state_prune_threshold,
            mode_first.state_prune_threshold
        );
        assert_eq!(
            threshold_first.max_history_length,
            mode_first.max_history_length
        );
        assert!((threshold_first.state_prune_threshold - 1e-8).abs() < 1e-12);
        assert_eq!(threshold_first.max_history_length, Some(7));

        // Untouched fields still receive the mode defaults.
        let defaults_only = EngineConfig::new(1, 1).inference_mode(InferenceMode::LowMemory);
        assert!((defaults_only.state_prune_threshold - 1e-4).abs() < 1e-9);
        assert_eq!(defaults_only.max_history_length, Some(128));
    }

    /// Regression: user-configured context dimensions were overwritten without a
    /// word whenever a model was attached.
    #[test]
    fn test_try_with_model_rejects_conflicting_context_dims() {
        let config = EngineConfig::new(1, 10).context(ContextConfig::new().num_layers(9));
        let result = InferenceEngine::try_with_model(config, Box::new(s4d_model(1)));
        assert!(matches!(
            result,
            Err(InferenceError::InvalidConfiguration(_))
        ));

        // Defaults are filled in from the model without complaint.
        let ok = InferenceEngine::try_with_model(EngineConfig::new(1, 10), Box::new(s4d_model(1)))
            .expect("default context config must be accepted");
        assert_eq!(ok.config().context.num_layers, 2);
    }

    #[test]
    fn test_try_set_model_leaves_engine_untouched_on_error() {
        let config = EngineConfig::new(1, 10).context(ContextConfig::new().num_layers(9));
        let mut engine = InferenceEngine::new(config);

        let result = engine.try_set_model(Box::new(s4d_model(1)));
        assert!(matches!(
            result,
            Err(InferenceError::InvalidConfiguration(_))
        ));
        assert!(!engine.has_model());
        assert_eq!(engine.config().context.num_layers, 9);
    }

    /// A failed forward pass must leave the hidden states in place rather than
    /// stranding the context with an empty state vector.
    #[test]
    fn test_failed_step_preserves_states() {
        let mut engine = InferenceEngine::with_model(
            EngineConfig::new(1, 1),
            Box::new(CountingModel::failing_for(1)),
        );
        let layers = engine.clone_states().len();

        assert!(engine.step(&Array1::from_vec(vec![0.5])).is_err());
        assert_eq!(engine.clone_states().len(), layers);

        // A dimension mismatch is rejected before the model is touched.
        assert!(engine.step(&Array1::from_vec(vec![0.1, 0.2])).is_err());
        assert_eq!(engine.clone_states().len(), layers);

        // The engine remains usable afterwards.
        let output = engine
            .step(&Array1::from_vec(vec![0.5]))
            .expect("engine must remain usable after a failed step");
        assert_eq!(output.len(), 1);
    }

    /// Regression: the LoRA module existed but was not reachable from inference,
    /// so activating an adapter had no effect on any prediction.
    #[test]
    fn test_lora_adapter_changes_predictions() {
        use crate::lora::{LoraAdapter, LoraAdapterManager, LoraConfig};
        use scirs2_core::ndarray::Array2;

        let mut engine =
            InferenceEngine::with_model(EngineConfig::new(1, 1), Box::new(CountingModel::new()));
        let baseline = engine
            .step(&Array1::from_vec(vec![0.5]))
            .expect("step must succeed");
        engine.reset();

        let dim = baseline.len();
        let adapter = LoraAdapter::new(
            Array2::from_elem((2, dim), 0.1),
            Array2::from_elem((dim, 2), 0.1),
            1.0,
            "boost",
        )
        .expect("adapter construction must succeed");
        let adapter_for_reference = adapter.clone();

        // `LoraAdapterManager::apply` (which the engine calls internally)
        // composes the delta with the base model output via
        // `apply_with_base`; `LoraAdapter::apply` alone now returns only the
        // bare scaled delta (see the `lora` module's regression coverage),
        // so the reference computation here must match `apply_with_base`,
        // not a bare `apply`.
        let expected = adapter_for_reference
            .apply_with_base(&baseline, &baseline)
            .expect("adapter application must succeed");

        let mut manager = LoraAdapterManager::new(LoraConfig::new());
        manager.register_adapter(adapter);
        engine.set_lora_manager(manager);

        // Registered but not activated: predictions are unchanged.
        let inactive = engine
            .step(&Array1::from_vec(vec![0.5]))
            .expect("step must succeed");
        assert_eq!(inactive, baseline);
        engine.reset();

        engine
            .lora_manager_mut()
            .expect("manager attached")
            .activate("boost")
            .expect("activation must succeed");
        let adapted = engine
            .step(&Array1::from_vec(vec![0.5]))
            .expect("step must succeed");
        assert_eq!(
            adapted, expected,
            "the active adapter must be applied to the model output"
        );
    }

    /// Regression: `TensorPool` was fully wired inside `InferenceContext`
    /// but completely unreachable from `InferenceEngine` — `context()` only
    /// exposed a shared reference, so a caller had no way to enable pooling
    /// once an engine existed. `InferenceEngine::enable_pooling` must make
    /// pooling reachable end-to-end through the engine's own `step()` path.
    #[test]
    fn test_engine_enable_pooling_reaches_context_history() {
        use crate::pool::TensorPool;

        let context = ContextConfig::new().store_history(true).max_context(2);
        let config = EngineConfig::new(1, 1).context(context);
        let mut engine = InferenceEngine::with_model(config, Box::new(CountingModel::new()));

        let pool = TensorPool::new();
        engine.enable_pooling(pool.clone());

        for i in 0..5 {
            engine
                .step(&Array1::from_vec(vec![i as f32]))
                .expect("step must succeed");
        }

        let stats = pool.stats().expect("stats must be readable");
        assert!(
            stats.total_reuses > 0,
            "pooling enabled via the engine must actually be exercised by step()'s \
             history churn, got stats: {:?}",
            stats
        );
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
