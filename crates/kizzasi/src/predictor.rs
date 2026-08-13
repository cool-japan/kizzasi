//! Main Kizzasi predictor implementation

use crate::backend::{Backend, StateSnapshot};
use crate::error::{KizzasiError, KizzasiResult};
use crate::plugin::{Plugin, PluginManager};
use kizzasi_core::{KizzasiConfig, ModelType, SelectiveSSM, StateSpaceModel};
use scirs2_core::ndarray::{Array1, Array2};
use std::path::Path;

#[cfg(feature = "logic")]
use kizzasi_logic::{ConstrainedInference, GuardrailSet};

// ============================================================================
// From Trait Implementations for Common Signal Types
// ============================================================================

/// Implements From trait for converting common signal types to `Array1<f32>`
/// This makes it easier to work with Kizzasi without manually creating arrays.
impl From<f32> for SignalInput {
    fn from(value: f32) -> Self {
        SignalInput(Array1::from_vec(vec![value]))
    }
}

impl From<Vec<f32>> for SignalInput {
    fn from(value: Vec<f32>) -> Self {
        SignalInput(Array1::from_vec(value))
    }
}

impl From<&[f32]> for SignalInput {
    fn from(value: &[f32]) -> Self {
        SignalInput(Array1::from_vec(value.to_vec()))
    }
}

impl<const N: usize> From<[f32; N]> for SignalInput {
    fn from(value: [f32; N]) -> Self {
        SignalInput(Array1::from_vec(value.to_vec()))
    }
}

impl From<Array1<f32>> for SignalInput {
    fn from(value: Array1<f32>) -> Self {
        SignalInput(value)
    }
}

/// Wrapper type for signal inputs with ergonomic conversions
#[derive(Debug, Clone)]
pub struct SignalInput(pub Array1<f32>);

impl SignalInput {
    /// Get a reference to the inner array
    pub fn as_array(&self) -> &Array1<f32> {
        &self.0
    }

    /// Consume and return the inner array
    pub fn into_array(self) -> Array1<f32> {
        self.0
    }
}

impl AsRef<Array1<f32>> for SignalInput {
    fn as_ref(&self) -> &Array1<f32> {
        &self.0
    }
}

/// Builder for constructing Kizzasi predictors with a fluent API
///
/// # Example
///
/// ```rust,ignore
/// let predictor = KizzasiBuilder::new()
///     .model_type(ModelType::Mamba2)
///     .input_dim(3)
///     .output_dim(3)
///     .hidden_dim(256)
///     .build()?;
/// ```
#[derive(Default)]
pub struct KizzasiBuilder {
    config: KizzasiConfig,
    #[cfg(feature = "logic")]
    guardrails: Option<GuardrailSet>,
}

impl KizzasiBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model type (Mamba, Mamba2, S4, RWKV)
    pub fn model_type(mut self, model_type: ModelType) -> Self {
        self.config = self.config.model_type(model_type);
        self
    }

    /// Set the context window size
    pub fn context_window(mut self, size: usize) -> Self {
        self.config = self.config.context_window(size);
        self
    }

    /// Set the hidden dimension
    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.config = self.config.hidden_dim(dim);
        self
    }

    /// Set the state dimension
    pub fn state_dim(mut self, dim: usize) -> Self {
        self.config = self.config.state_dim(dim);
        self
    }

    /// Set the number of layers
    pub fn num_layers(mut self, n: usize) -> Self {
        self.config = self.config.num_layers(n);
        self
    }

    /// Set the input dimension
    pub fn input_dim(mut self, dim: usize) -> Self {
        self.config = self.config.input_dim(dim);
        self
    }

    /// Set the output dimension
    pub fn output_dim(mut self, dim: usize) -> Self {
        self.config = self.config.output_dim(dim);
        self
    }

    /// Set the rank of the Δ (time-step) projection used by the selective scan
    pub fn dt_rank(mut self, rank: usize) -> Self {
        self.config = self.config.dt_rank(rank);
        self
    }

    /// Set the inner-dimension expansion factor (see
    /// [`KizzasiConfig::expansion_factor`]).
    ///
    /// Only [`ModelType::Mamba`] has a gated expansion branch; building any
    /// other model type with this set is rejected with a configuration error
    /// rather than silently ignoring it.
    pub fn expansion_factor(mut self, factor: usize) -> Self {
        self.config = self.config.expansion_factor(factor);
        self
    }

    /// Set the number of heads for multi-head architectures (see
    /// [`KizzasiConfig::num_heads`]).
    ///
    /// Only [`ModelType::Rwkv`] is multi-head; building any other model type
    /// with this set is rejected with a configuration error.
    pub fn num_heads(mut self, heads: usize) -> Self {
        self.config = self.config.num_heads(heads);
        self
    }

    /// Set the per-head dimension for multi-head architectures (see
    /// [`KizzasiConfig::head_dim`]).
    ///
    /// Must equal `hidden_dim / num_heads`. Only [`ModelType::Rwkv`] is
    /// multi-head; building any other model type with this set is rejected
    /// with a configuration error.
    pub fn head_dim(mut self, dim: usize) -> Self {
        self.config = self.config.head_dim(dim);
        self
    }

    /// Load weights from a file path.
    ///
    /// The file is read when [`Self::build`] constructs the predictor. It must
    /// be a weight file written by [`Kizzasi::save_weights`], or — for the
    /// [`ModelType::Mamba`] / [`ModelType::S4`] / [`ModelType::Rwkv`]
    /// backends — a `.safetensors` checkpoint using the `kizzasi-model` tensor
    /// naming convention. A missing file, a mismatched shape, or a checkpoint
    /// that matches no parameter of the model is a hard error: the builder
    /// never falls back to random initialisation once a path is given.
    ///
    /// Omitting `weights_path` yields a randomly-initialised model.
    pub fn weights_path(mut self, path: &str) -> Self {
        self.config = self.config.load_weights(path);
        self
    }

    /// Set guardrails for constraint enforcement
    #[cfg(feature = "logic")]
    pub fn guardrails(mut self, guardrails: GuardrailSet) -> Self {
        self.guardrails = Some(guardrails);
        self
    }

    /// Build the Kizzasi predictor
    pub fn build(self) -> KizzasiResult<Kizzasi> {
        // Validate configuration
        if self.config.get_input_dim() == 0 {
            return Err(KizzasiError::Config("input_dim must be > 0".into()));
        }
        if self.config.get_output_dim() == 0 {
            return Err(KizzasiError::Config("output_dim must be > 0".into()));
        }
        if self.config.get_hidden_dim() == 0 {
            return Err(KizzasiError::Config("hidden_dim must be > 0".into()));
        }

        let mut predictor = Kizzasi::new(self.config)?;

        #[cfg(feature = "logic")]
        if let Some(guardrails) = self.guardrails {
            predictor.set_guardrails(guardrails);
        }

        Ok(predictor)
    }
}

/// Preset configurations for common use cases
impl KizzasiBuilder {
    /// Audio processing preset (44.1kHz, mono)
    pub fn audio_preset() -> Self {
        Self::new()
            .model_type(ModelType::Mamba2)
            .input_dim(1)
            .output_dim(1)
            .hidden_dim(256)
            .state_dim(16)
            .num_layers(4)
            .context_window(8192)
    }

    /// Robotics control preset (multi-axis)
    pub fn robotics_preset(axes: usize) -> Self {
        Self::new()
            .model_type(ModelType::Mamba2)
            .input_dim(axes)
            .output_dim(axes)
            .hidden_dim(128)
            .state_dim(8)
            .num_layers(3)
            .context_window(1024)
    }

    /// Sensor fusion preset (multi-sensor input)
    pub fn sensor_preset(num_sensors: usize) -> Self {
        Self::new()
            .model_type(ModelType::Mamba2)
            .input_dim(num_sensors)
            .output_dim(num_sensors)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2)
            .context_window(2048)
    }

    /// Lightweight preset for embedded systems
    ///
    /// Uses [`ModelType::Mamba2`] — the plain selective-scan engine — because
    /// it is the smallest of the available architectures (no gated expansion
    /// branch, no causal convolution) and the only one that supports
    /// [`Kizzasi::fork`] and full-state checkpoints. Select
    /// [`ModelType::Mamba`] explicitly if you want the gated Mamba block.
    pub fn lightweight_preset(input_dim: usize, output_dim: usize) -> Self {
        Self::new()
            .model_type(ModelType::Mamba2)
            .input_dim(input_dim)
            .output_dim(output_dim)
            .hidden_dim(32)
            .state_dim(4)
            .num_layers(1)
            .context_window(512)
    }

    /// Video frame prediction preset (for spatial-temporal sequences)
    ///
    /// # Arguments
    /// * `frame_features` - Number of features per frame (e.g., latent dimension after encoding)
    ///
    /// Optimized for video frame prediction with large context windows
    /// to capture temporal dependencies across frames.
    pub fn video_preset(frame_features: usize) -> Self {
        Self::new()
            .model_type(ModelType::Mamba2)
            .input_dim(frame_features)
            .output_dim(frame_features)
            .hidden_dim(512)
            .state_dim(32)
            .num_layers(6)
            .context_window(16384) // Large context for long-range dependencies
    }

    /// Real-time control preset (optimized for low-latency control loops)
    ///
    /// # Arguments
    /// * `state_dim_arg` - Dimension of the control state vector
    /// * `action_dim` - Dimension of the action/control output
    ///
    /// Optimized for real-time control with minimal latency.
    /// Uses smaller model for faster inference.
    pub fn control_preset(state_dim_arg: usize, action_dim: usize) -> Self {
        Self::new()
            .model_type(ModelType::Mamba2)
            .input_dim(state_dim_arg)
            .output_dim(action_dim)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2)
            .context_window(256) // Smaller context for lower latency
    }

    /// Custom preset builder with recommended defaults
    ///
    /// Starts with sensible defaults that can be customized via builder pattern.
    /// This is useful as a starting point for creating custom configurations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let predictor = KizzasiBuilder::custom_preset()
    ///     .input_dim(10)
    ///     .output_dim(10)
    ///     .hidden_dim(128)
    ///     .build()?;
    /// ```
    pub fn custom_preset() -> Self {
        Self::new()
            .model_type(ModelType::Mamba2)
            .hidden_dim(128)
            .state_dim(16)
            .num_layers(3)
            .context_window(4096)
    }
}

/// The main Kizzasi AGSP predictor
///
/// Combines State Space Model prediction with optional constraint enforcement.
///
/// # Architecture selection
///
/// `config.get_model_type()` decides which engine runs; see
/// [`crate::backend`] for the full mapping. [`ModelType::Mamba2`] uses
/// `kizzasi-core`'s [`SelectiveSSM`]; [`ModelType::Mamba`], [`ModelType::S4`]
/// and [`ModelType::Rwkv`] use the corresponding `kizzasi-model`
/// architectures. Different model types genuinely compute different functions.
pub struct Kizzasi {
    /// The model engine selected by `config.get_model_type()`
    backend: Backend,
    /// Configuration
    config: KizzasiConfig,
    /// Optional guardrails for constraint enforcement
    #[cfg(feature = "logic")]
    guardrails: Option<GuardrailSet>,
    /// Plugin manager for extensibility
    plugins: PluginManager,
}

impl Kizzasi {
    /// Create a new Kizzasi predictor from configuration
    ///
    /// The architecture named by `config.get_model_type()` is constructed. If
    /// `config.get_weights_path()` is set, the weights are loaded before this
    /// function returns; a missing file, a shape mismatch, or a checkpoint
    /// that matches nothing in the model is returned as an error instead of
    /// silently leaving the model randomly initialised.
    pub fn new(config: KizzasiConfig) -> KizzasiResult<Self> {
        let mut backend = Backend::from_config(&config)?;

        if let Some(path) = config.get_weights_path() {
            backend.load_weights(Path::new(path), &config)?;
        }

        Ok(Self {
            backend,
            config,
            #[cfg(feature = "logic")]
            guardrails: None,
            plugins: PluginManager::new(),
        })
    }

    /// Create a Kizzasi predictor from an existing SSM
    ///
    /// This is primarily used for restoring from full state checkpoints. The
    /// resulting predictor always uses the [`SelectiveSSM`] engine, whatever
    /// `model_type` the embedded configuration names.
    pub fn from_ssm(ssm: SelectiveSSM) -> KizzasiResult<Self> {
        let config = ssm.config().clone();

        Ok(Self {
            backend: Backend::from_selective(ssm),
            config,
            #[cfg(feature = "logic")]
            guardrails: None,
            plugins: PluginManager::new(),
        })
    }

    /// Get a reference to the underlying selective SSM.
    ///
    /// Returns `None` when the predictor runs one of the `kizzasi-model`
    /// architectures ([`ModelType::Mamba`], [`ModelType::S4`],
    /// [`ModelType::Rwkv`]) rather than [`ModelType::Mamba2`]'s
    /// [`SelectiveSSM`].
    pub fn ssm(&self) -> Option<&SelectiveSSM> {
        self.backend.selective()
    }

    /// Get a mutable reference to the underlying selective SSM.
    ///
    /// Returns `None` for the `kizzasi-model` architectures; see [`Self::ssm`].
    pub fn ssm_mut(&mut self) -> Option<&mut SelectiveSSM> {
        self.backend.selective_mut()
    }

    /// Number of prediction steps taken since construction or the last reset
    pub fn step_count(&self) -> usize {
        self.backend.step_count()
    }

    /// Name of the engine actually running behind this predictor
    pub fn engine_name(&self) -> &'static str {
        self.backend.engine_name()
    }

    /// Persist every trainable parameter to `path`.
    ///
    /// The file written here is exactly what
    /// [`KizzasiBuilder::weights_path`] reads back, so
    /// `save_weights` → `weights_path` round-trips a trained model.
    ///
    /// Hidden state is *not* included — use
    /// [`Kizzasi::save_full_checkpoint`] for that.
    pub fn save_weights<P: AsRef<Path>>(&self, path: P) -> KizzasiResult<()> {
        self.backend.save_weights(path.as_ref(), &self.config)
    }

    /// Load parameters from a weight file into this predictor in place.
    ///
    /// Accepts the same inputs as [`KizzasiBuilder::weights_path`]. The hidden
    /// state is left untouched; call [`Self::reset`] afterwards to start a
    /// fresh sequence.
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> KizzasiResult<()> {
        self.backend.load_weights(path.as_ref(), &self.config)
    }

    /// Capture the recurrent state so it can be restored later.
    ///
    /// Used by profiling paths that must not leave synthetic samples in a live
    /// predictor's hidden state. Coverage matches [`Self::reset`]: complete for
    /// every backend except [`ModelType::S4`], whose causal-convolution
    /// history `kizzasi-model` does not expose.
    pub(crate) fn snapshot_state(&self) -> StateSnapshot {
        self.backend.snapshot_state()
    }

    /// Restore a state captured by [`Self::snapshot_state`].
    pub(crate) fn restore_state(&mut self, snapshot: StateSnapshot) -> KizzasiResult<()> {
        self.backend.restore_state(snapshot)
    }

    /// Add a plugin to the predictor
    pub fn add_plugin(&mut self, plugin: Box<dyn Plugin>) {
        self.plugins.add_plugin(plugin);
    }

    /// Remove a plugin by name
    pub fn remove_plugin(&mut self, name: &str) -> Option<Box<dyn Plugin>> {
        self.plugins.remove_plugin(name)
    }

    /// Get a reference to the plugin manager
    pub fn plugins(&self) -> &PluginManager {
        &self.plugins
    }

    /// Get a mutable reference to the plugin manager
    pub fn plugins_mut(&mut self) -> &mut PluginManager {
        &mut self.plugins
    }

    /// Set guardrails for constraint enforcement
    #[cfg(feature = "logic")]
    pub fn set_guardrails(&mut self, guardrails: GuardrailSet) {
        self.guardrails = Some(guardrails);
    }

    /// Clear guardrails
    #[cfg(feature = "logic")]
    pub fn clear_guardrails(&mut self) {
        self.guardrails = None;
    }

    /// Perform a single prediction step
    ///
    /// If any stage fails, every registered plugin's
    /// [`Plugin::on_error`](crate::plugin::Plugin::on_error) hook is invoked
    /// before the original error is returned.
    pub fn step(&mut self, input: &Array1<f32>) -> KizzasiResult<Array1<f32>> {
        let input_dim = self.config.get_input_dim();
        let output_dim = self.config.get_output_dim();

        match self.step_inner(input, input_dim, output_dim) {
            Ok(output) => Ok(output),
            Err(error) => {
                // A failure inside an error hook must never mask the original
                // error, so it is logged rather than propagated.
                if let Err(secondary) = self.plugins.execute_on_error(&error, input_dim, output_dim)
                {
                    tracing::warn!("plugin on_error hook failed: {}", secondary);
                }
                Err(error)
            }
        }
    }

    /// The fallible body of [`Self::step`], split out so a failure at any
    /// stage can be routed through the plugin error hook exactly once.
    fn step_inner(
        &mut self,
        input: &Array1<f32>,
        input_dim: usize,
        output_dim: usize,
    ) -> KizzasiResult<Array1<f32>> {
        // Execute pre-process plugins
        self.plugins
            .execute_pre_process(input, input_dim, output_dim)?;

        // Get raw prediction from the selected backend. The input is only
        // cloned when a plugin can actually rewrite it — on the common
        // no-plugin path the borrow is handed straight to the model.
        let mut prediction = if self.plugins.is_empty() {
            self.backend.step(input)?
        } else {
            let transformed = self
                .plugins
                .transform_input(input.clone(), input_dim, output_dim)?;
            self.backend.step(&transformed)?
        };

        // Apply guardrails if configured
        #[cfg(feature = "logic")]
        if let Some(ref guardrails) = self.guardrails {
            prediction = guardrails.constrain(&prediction)?;
        }

        // Transform output through plugins
        let transformed_output = self
            .plugins
            .transform_output(prediction, input_dim, output_dim)?;

        // Execute post-process plugins
        self.plugins
            .execute_post_process(input, &transformed_output, input_dim, output_dim)?;

        Ok(transformed_output)
    }

    /// Prediction step from a slice
    ///
    /// # Allocation behaviour
    ///
    /// This is an ergonomic wrapper, **not** an allocation-free path: the
    /// model's step signature takes an owned `Array1<f32>`, so the slice is
    /// copied into one owned array per call, and the prediction itself is
    /// returned as a freshly allocated array. Use [`Self::step_inplace`] to
    /// avoid allocating a `Vec` for the output on the caller's side.
    ///
    /// # Arguments
    ///
    /// * `input` - Input signal as a slice
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let data = vec![0.1, 0.2, 0.3];
    /// let output = predictor.step_slice(&data)?;
    /// ```
    pub fn step_slice(&mut self, input: &[f32]) -> KizzasiResult<Array1<f32>> {
        let input_array = Array1::from_vec(input.to_vec());
        self.step(&input_array)
    }

    /// Prediction that writes the result into a caller-provided buffer
    ///
    /// The output buffer must have exactly `output_dim` elements. This avoids
    /// the caller having to allocate a `Vec` per step; it does **not** make
    /// the call allocation-free — see [`Self::step_slice`] for what the input
    /// and the model step still allocate.
    ///
    /// # Arguments
    ///
    /// * `input` - Input signal as a slice
    /// * `output` - Pre-allocated output buffer
    ///
    /// # Returns
    ///
    /// Returns an error if the output buffer size doesn't match output_dim.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let input = vec![0.1, 0.2, 0.3];
    /// let mut output = vec![0.0; 3]; // Pre-allocate
    /// predictor.step_inplace(&input, &mut output)?;
    /// ```
    pub fn step_inplace(&mut self, input: &[f32], output: &mut [f32]) -> KizzasiResult<()> {
        let output_dim = self.config.get_output_dim();
        if output.len() != output_dim {
            return Err(KizzasiError::DimensionMismatch {
                expected: output_dim,
                actual: output.len(),
                context: "Output buffer size must match output_dim".into(),
            });
        }

        let result = self.step_slice(input)?;
        match result.as_slice() {
            Some(slice) => output.copy_from_slice(slice),
            // Non-contiguous storage cannot be memcpy'd; copy element-wise
            // rather than failing a call the caller cannot fix.
            None => {
                for (dst, src) in output.iter_mut().zip(result.iter()) {
                    *dst = *src;
                }
            }
        }
        Ok(())
    }

    /// Validate that autoregressive feedback is well-defined for this model.
    ///
    /// [`Self::predict_n`], [`Self::predict_until`] and
    /// [`Self::predict_n_inplace`] feed each prediction back in as the next
    /// input, which is only meaningful when `output_dim == input_dim`.
    fn check_autoregressive_dims(&self) -> KizzasiResult<()> {
        let input_dim = self.config.get_input_dim();
        let output_dim = self.config.get_output_dim();
        if input_dim == output_dim {
            return Ok(());
        }
        Err(KizzasiError::dimension_mismatch(
            input_dim,
            output_dim,
            "autoregressive prediction feeds each output back in as the next input, \
             so it requires output_dim == input_dim; use repeated step() calls with \
             externally supplied inputs for asymmetric models",
        ))
    }

    /// Predict multiple steps with pre-allocated output buffer
    ///
    /// More efficient than `predict_n` when you can pre-allocate the output.
    ///
    /// # Arguments
    ///
    /// * `input` - Input signal
    /// * `n_steps` - Number of steps to predict
    /// * `output` - Pre-allocated buffer of shape (n_steps, output_dim)
    ///
    /// # Returns
    ///
    /// Returns an error if buffer dimensions don't match, or if
    /// `output_dim != input_dim` (see [`Self::predict_n`]).
    pub fn predict_n_inplace(
        &mut self,
        input: &Array1<f32>,
        n_steps: usize,
        output: &mut Array2<f32>,
    ) -> KizzasiResult<()> {
        self.check_autoregressive_dims()?;
        let output_dim = self.config.get_output_dim();
        if output.shape() != [n_steps, output_dim] {
            return Err(KizzasiError::DimensionMismatch {
                expected: n_steps * output_dim,
                actual: output.len(),
                context: format!(
                    "Output buffer must be ({}, {}), got {:?}",
                    n_steps,
                    output_dim,
                    output.shape()
                ),
            });
        }

        let mut current_input = input.clone();
        for i in 0..n_steps {
            let step_output = self.step(&current_input)?;
            for (j, &val) in step_output.iter().enumerate() {
                output[[i, j]] = val;
            }
            current_input = step_output;
        }

        Ok(())
    }

    /// Reset the recurrent state
    ///
    /// # Per-backend completeness
    ///
    /// [`ModelType::Mamba2`] (`SelectiveSSM`), [`ModelType::Mamba`] and
    /// [`ModelType::Rwkv`] clear their entire recurrent state. For
    /// [`ModelType::S4`], `kizzasi-model`'s `S4DLayer::reset` clears the SSM
    /// state but not the layer's 3-tap causal-convolution history, so up to
    /// two previous frames still influence the next two steps. That is an
    /// upstream limitation of `kizzasi-model`, recorded here rather than
    /// papered over.
    pub fn reset(&mut self) {
        self.backend.reset();

        // Notify plugins
        let input_dim = self.config.get_input_dim();
        let output_dim = self.config.get_output_dim();
        if let Err(e) = self.plugins.execute_on_reset(input_dim, output_dim) {
            tracing::warn!("plugin on_reset hook failed: {}", e);
        }
    }

    /// Get the context window size
    pub fn context_window(&self) -> usize {
        self.backend.context_window()
    }

    /// Get the configuration
    pub fn config(&self) -> &KizzasiConfig {
        &self.config
    }

    /// Check if guardrails are set
    #[cfg(feature = "logic")]
    pub fn has_guardrails(&self) -> bool {
        self.guardrails.is_some()
    }

    /// Get a reference to the guardrails
    #[cfg(feature = "logic")]
    pub fn guardrails(&self) -> Option<&GuardrailSet> {
        self.guardrails.as_ref()
    }

    /// Validate a prediction against guardrails without modifying it
    #[cfg(feature = "logic")]
    pub fn validate(&self, prediction: &Array1<f32>) -> bool {
        if let Some(ref guardrails) = self.guardrails {
            guardrails.validate(prediction)
        } else {
            true
        }
    }

    /// Compute violation loss for training
    #[cfg(feature = "logic")]
    pub fn violation_loss(&self, prediction: &Array1<f32>) -> f32 {
        if let Some(ref guardrails) = self.guardrails {
            guardrails.violation_loss(prediction)
        } else {
            0.0
        }
    }

    /// Predict multiple steps ahead
    ///
    /// Returns an array of shape (n_steps, output_dim) containing predictions.
    /// Each step uses the previous prediction as input (autoregressive).
    ///
    /// # Errors
    ///
    /// Requires `output_dim == input_dim`, because each prediction is fed
    /// straight back in as the next input. An asymmetric model (for example
    /// `input_dim(12).output_dim(6)`) returns
    /// [`KizzasiError::DimensionMismatch`] instead of failing on the second
    /// iteration.
    pub fn predict_n(&mut self, input: &Array1<f32>, n_steps: usize) -> KizzasiResult<Array2<f32>> {
        self.check_autoregressive_dims()?;
        let output_dim = self.config.get_output_dim();
        let mut predictions = Array2::zeros((n_steps, output_dim));
        let mut current_input = input.clone();

        for i in 0..n_steps {
            let output = self.step(&current_input)?;
            for (j, &val) in output.iter().enumerate() {
                predictions[[i, j]] = val;
            }
            // Use output as next input (autoregressive)
            current_input = output;
        }

        Ok(predictions)
    }

    /// Predict until a condition is met
    ///
    /// Continues prediction until the predicate returns true or max_steps is reached.
    /// Returns all predictions up to that point.
    ///
    /// # Errors
    ///
    /// Requires `output_dim == input_dim`; see [`Self::predict_n`].
    pub fn predict_until<F>(
        &mut self,
        input: &Array1<f32>,
        max_steps: usize,
        predicate: F,
    ) -> KizzasiResult<Vec<Array1<f32>>>
    where
        F: Fn(&Array1<f32>, usize) -> bool,
    {
        self.check_autoregressive_dims()?;
        let mut predictions = Vec::with_capacity(max_steps);
        let mut current_input = input.clone();

        for step in 0..max_steps {
            let output = self.step(&current_input)?;
            predictions.push(output.clone());

            if predicate(&output, step) {
                break;
            }

            current_input = output;
        }

        Ok(predictions)
    }

    /// Predict over a batch of inputs (non-autoregressive)
    ///
    /// Each input is processed independently. Hidden state is maintained
    /// across the batch for temporal consistency.
    pub fn predict_batch(&mut self, inputs: &[Array1<f32>]) -> KizzasiResult<Vec<Array1<f32>>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.step(input)?);
        }
        Ok(outputs)
    }

    /// Get a copy of the predictor with the same weights and a fresh state
    ///
    /// The returned predictor carries **the same trained parameters** as
    /// `self`, so it computes the same function; only the recurrent state is
    /// reset. That makes it usable for running parallel predictions from the
    /// same starting point, and for per-thread sharding of one model.
    /// Guardrails are carried over; plugins are not (they are not required to
    /// be cloneable).
    ///
    /// # Errors
    ///
    /// Only the [`SelectiveSSM`] engine ([`ModelType::Mamba2`]) can be copied.
    /// The `kizzasi-model` architectures expose no in-memory weight export, so
    /// forking them would silently produce a differently-initialised model;
    /// this returns [`KizzasiError::InvalidState`] instead. Persist those with
    /// [`Self::save_weights`] and rebuild via
    /// [`KizzasiBuilder::weights_path`].
    pub fn fork(&self) -> KizzasiResult<Self> {
        let mut backend = self.backend.try_clone()?;
        backend.reset();

        Ok(Self {
            backend,
            config: self.config.clone(),
            #[cfg(feature = "logic")]
            guardrails: self.guardrails.clone(),
            plugins: PluginManager::new(),
        })
    }

    /// Hot-swap the underlying model with a new configuration
    ///
    /// This allows changing the model type, architecture, or parameters at runtime
    /// without creating a new predictor instance. The input/output dimensions must
    /// match for state compatibility.
    ///
    /// # Arguments
    ///
    /// * `new_config` - New configuration for the model
    /// * `preserve_guardrails` - Whether to keep current guardrails
    ///
    /// # Returns
    ///
    /// Returns an error if the new configuration is incompatible (dimension mismatch)
    /// or if model initialization fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut predictor = KizzasiBuilder::audio_preset().build()?;
    ///
    /// // Switch from Mamba2 to RWKV while keeping same dimensions
    /// let new_config = KizzasiConfig::new()
    ///     .model_type(ModelType::RWKV)
    ///     .input_dim(1)
    ///     .output_dim(1)
    ///     .hidden_dim(256)
    ///     .state_dim(16)
    ///     .num_layers(4);
    ///
    /// predictor.hot_swap(new_config, true)?;
    /// ```
    pub fn hot_swap(
        &mut self,
        new_config: KizzasiConfig,
        preserve_guardrails: bool,
    ) -> KizzasiResult<()> {
        // Validate dimension compatibility
        if new_config.get_input_dim() != self.config.get_input_dim() {
            return Err(KizzasiError::DimensionMismatch {
                expected: self.config.get_input_dim(),
                actual: new_config.get_input_dim(),
                context: "input_dim must match for hot-swap compatibility".into(),
            });
        }

        if new_config.get_output_dim() != self.config.get_output_dim() {
            return Err(KizzasiError::DimensionMismatch {
                expected: self.config.get_output_dim(),
                actual: new_config.get_output_dim(),
                context: "output_dim must match for hot-swap compatibility".into(),
            });
        }

        // Build the architecture named by the new configuration
        let mut new_backend =
            Backend::from_config(&new_config).map_err(|e| KizzasiError::ModelNotReady {
                reason: format!("Failed to initialize new model: {}", e),
                suggestion: "Check that the new configuration is valid and compatible".into(),
            })?;

        if let Some(path) = new_config.get_weights_path() {
            new_backend.load_weights(Path::new(path), &new_config)?;
        }

        // Store old guardrails if needed
        #[cfg(feature = "logic")]
        let old_guardrails = if preserve_guardrails {
            self.guardrails.clone()
        } else {
            None
        };

        // Swap the model
        self.backend = new_backend;
        self.config = new_config;

        // Restore guardrails if requested
        #[cfg(feature = "logic")]
        if preserve_guardrails {
            self.guardrails = old_guardrails;
        } else {
            self.guardrails = None;
        }

        Ok(())
    }

    /// Get the current model type
    pub fn model_type(&self) -> ModelType {
        self.config.get_model_type()
    }

    /// Get the input dimension
    pub fn input_dim(&self) -> usize {
        self.config.get_input_dim()
    }

    /// Get the output dimension
    pub fn output_dim(&self) -> usize {
        self.config.get_output_dim()
    }

    /// Get the hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.config.get_hidden_dim()
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.config.get_num_layers()
    }

    /// Get the state dimension
    pub fn state_dim(&self) -> usize {
        self.config.get_state_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kizzasi_core::ModelType;

    /// Plugin that fails on every step so the error hook can be observed.
    struct FailingPlugin {
        errors_seen: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Plugin for FailingPlugin {
        fn name(&self) -> &str {
            "failing"
        }

        fn transform_input(
            &mut self,
            _input: Array1<f32>,
            _ctx: &crate::plugin::PluginContext,
        ) -> KizzasiResult<Array1<f32>> {
            Err(KizzasiError::inference("plugin refused the input"))
        }

        fn on_error(
            &mut self,
            _error: &KizzasiError,
            _ctx: &crate::plugin::PluginContext,
        ) -> KizzasiResult<()> {
            self.errors_seen
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    #[test]
    fn test_kizzasi_step() {
        let config = KizzasiConfig::new()
            .model_type(ModelType::Mamba2)
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let mut predictor = Kizzasi::new(config).unwrap();
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let output = predictor.step(&input).unwrap();

        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_plugin_on_error_hook_is_invoked() {
        // Regression: `Kizzasi::step` propagated every failure with `?`, so
        // `PluginManager::execute_on_error` had zero call sites and the
        // documented `Plugin::on_error` hook never fired.
        let errors_seen = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut predictor = Kizzasi::new(
            KizzasiConfig::new()
                .input_dim(2)
                .output_dim(2)
                .hidden_dim(16),
        )
        .unwrap();
        predictor.add_plugin(Box::new(FailingPlugin {
            errors_seen: errors_seen.clone(),
        }));

        let result = predictor.step(&Array1::from_vec(vec![0.1, 0.2]));
        assert!(result.is_err());
        assert_eq!(errors_seen.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn test_kizzasi_reset() {
        let config = KizzasiConfig::new().input_dim(3).output_dim(3);

        let mut predictor = Kizzasi::new(config).unwrap();

        // Make some predictions
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let _ = predictor.step(&input);

        // Reset
        predictor.reset();

        // Should still work
        let output = predictor.step(&input).unwrap();
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_predict_n() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(32)
            .state_dim(4)
            .num_layers(1);

        let mut predictor = Kizzasi::new(config).unwrap();
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let predictions = predictor.predict_n(&input, 5).unwrap();
        assert_eq!(predictions.shape(), &[5, 3]);
    }

    #[test]
    fn test_predict_until() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(32)
            .state_dim(4)
            .num_layers(1);

        let mut predictor = Kizzasi::new(config).unwrap();
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Stop after 3 steps
        let predictions = predictor
            .predict_until(&input, 10, |_, step| step >= 2)
            .unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_predict_batch() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(1);

        let mut predictor = Kizzasi::new(config).unwrap();
        let inputs = vec![
            Array1::from_vec(vec![0.1, 0.2]),
            Array1::from_vec(vec![0.3, 0.4]),
            Array1::from_vec(vec![0.5, 0.6]),
        ];

        let outputs = predictor.predict_batch(&inputs).unwrap();
        assert_eq!(outputs.len(), 3);
        for output in &outputs {
            assert_eq!(output.len(), 2);
        }
    }

    #[test]
    fn test_fork() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(1);

        let predictor = Kizzasi::new(config).unwrap();
        let forked = predictor.fork().unwrap();

        assert_eq!(forked.context_window(), predictor.context_window());
    }

    #[test]
    fn test_fork_copies_weights_not_reinitialises_them() {
        // Regression: fork() used to call Kizzasi::new, re-drawing every
        // weight matrix, so the "copy" computed a completely different
        // function from its parent.
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(1);

        let mut parent = Kizzasi::new(config).unwrap();
        let input = Array1::from_vec(vec![0.3, -0.7]);

        // Drive the parent so the fork's state reset is observable.
        parent.step(&input).unwrap();
        parent.step(&input).unwrap();

        let mut forked = parent.fork().unwrap();
        parent.reset();

        let expected = parent.step(&input).unwrap();
        let actual = forked.step(&input).unwrap();

        assert_eq!(expected.len(), actual.len());
        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "fork must share the parent's weights (got {b}, expected {a})"
            );
        }

        // The fork starts from a fresh recurrent state.
        assert_eq!(forked.step_count(), 1);
    }

    #[test]
    fn test_fork_is_independent_of_parent() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(1);

        let parent = Kizzasi::new(config).unwrap();
        let mut forked = parent.fork().unwrap();
        let mut parent = parent;

        let input = Array1::from_vec(vec![0.1, 0.2]);
        for _ in 0..4 {
            parent.step(&input).unwrap();
        }

        // The fork's own state must be untouched by the parent's progress.
        assert_eq!(forked.step_count(), 0);
        forked.step(&input).unwrap();
        assert_eq!(forked.step_count(), 1);
    }

    #[test]
    fn test_model_types_dispatch_to_different_engines() {
        // Regression: every ModelType used to build the same SelectiveSSM.
        let build = |model_type: ModelType| {
            Kizzasi::new(
                KizzasiConfig::new()
                    .model_type(model_type)
                    .input_dim(3)
                    .output_dim(3)
                    .hidden_dim(32)
                    .state_dim(8)
                    .num_layers(2),
            )
            .unwrap()
        };

        assert!(build(ModelType::Mamba2).ssm().is_some());
        for model_type in [ModelType::Mamba, ModelType::S4, ModelType::Rwkv] {
            let predictor = build(model_type);
            assert!(
                predictor.ssm().is_none(),
                "{model_type:?} must not silently run SelectiveSSM"
            );
            assert_eq!(predictor.model_type(), model_type);
        }
    }

    #[test]
    fn test_weights_round_trip_through_weights_path() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(32)
            .state_dim(8)
            .num_layers(2);

        let mut trained = Kizzasi::new(config.clone()).unwrap();
        let input = Array1::from_vec(vec![0.2, -0.1, 0.4]);
        let expected = trained.step(&input).unwrap();

        let path = std::env::temp_dir().join("kizzasi_predictor_weight_roundtrip.json");
        trained.save_weights(&path).unwrap();

        let loaded_config = config.load_weights(&path.to_string_lossy());
        let mut restored = Kizzasi::new(loaded_config).unwrap();
        restored.reset();

        let mut fresh_parent = trained.fork().unwrap();
        let reference = fresh_parent.step(&input).unwrap();
        let actual = restored.step(&input).unwrap();

        for (a, b) in reference.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-6, "weights_path must restore weights");
        }
        assert_eq!(expected.len(), actual.len());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_weights_path_missing_file_is_an_error() {
        // Regression: weights_path used to be stored and never read, so a
        // user pointing at a trained checkpoint silently got a random model.
        let missing = std::env::temp_dir().join("kizzasi_definitely_missing_weights.json");
        let _ = std::fs::remove_file(&missing);

        let result = KizzasiBuilder::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16)
            .weights_path(&missing.to_string_lossy())
            .build();

        assert!(
            matches!(result, Err(KizzasiError::ModelNotReady { .. })),
            "a missing weights file must be a hard error, not random init"
        );
    }

    #[test]
    fn test_weights_file_shape_mismatch_is_an_error() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(32)
            .state_dim(8)
            .num_layers(2);
        let predictor = Kizzasi::new(config).unwrap();

        let path = std::env::temp_dir().join("kizzasi_predictor_weight_mismatch.json");
        predictor.save_weights(&path).unwrap();

        let wrong = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64) // differs from the saved file
            .state_dim(8)
            .num_layers(2)
            .load_weights(&path.to_string_lossy());

        assert!(Kizzasi::new(wrong).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_predict_n_rejects_asymmetric_dims() {
        // input_dim != output_dim cannot feed its own output back in.
        let mut predictor = KizzasiBuilder::control_preset(12, 6).build().unwrap();
        let input = Array1::from_vec(vec![0.0; 12]);

        assert!(matches!(
            predictor.predict_n(&input, 3),
            Err(KizzasiError::DimensionMismatch { .. })
        ));
        assert!(predictor.predict_until(&input, 3, |_, _| false).is_err());

        let mut buffer = Array2::zeros((3, 6));
        assert!(predictor.predict_n_inplace(&input, 3, &mut buffer).is_err());

        // Single steps still work for asymmetric models.
        assert_eq!(predictor.step(&input).unwrap().len(), 6);
    }

    #[test]
    fn test_unsupported_architecture_option_is_rejected() {
        // expansion_factor has no meaning for the selective-scan engine; it
        // must be refused rather than accepted and dropped.
        let result = KizzasiBuilder::new()
            .model_type(ModelType::Mamba2)
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16)
            .expansion_factor(4)
            .build();
        assert!(matches!(result, Err(KizzasiError::Config(_))));
    }

    #[test]
    fn test_kizzasi_builder() {
        let predictor = KizzasiBuilder::new()
            .model_type(ModelType::Mamba2)
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2)
            .build()
            .unwrap();

        assert_eq!(predictor.context_window(), 8192);
    }

    #[test]
    fn test_audio_preset() {
        let predictor = KizzasiBuilder::audio_preset().build().unwrap();
        assert_eq!(predictor.config().get_input_dim(), 1);
        assert_eq!(predictor.config().get_hidden_dim(), 256);
    }

    #[test]
    fn test_robotics_preset() {
        let predictor = KizzasiBuilder::robotics_preset(6).build().unwrap();
        assert_eq!(predictor.config().get_input_dim(), 6);
        assert_eq!(predictor.config().get_output_dim(), 6);
    }

    #[test]
    fn test_builder_validation() {
        // Test that zero input_dim fails
        let result = KizzasiBuilder::new().input_dim(0).output_dim(1).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_video_preset() {
        let predictor = KizzasiBuilder::video_preset(256).build().unwrap();
        assert_eq!(predictor.config().get_input_dim(), 256);
        assert_eq!(predictor.config().get_output_dim(), 256);
        assert_eq!(predictor.config().get_hidden_dim(), 512);
    }

    #[test]
    fn test_control_preset() {
        let predictor = KizzasiBuilder::control_preset(8, 4).build().unwrap();
        assert_eq!(predictor.config().get_input_dim(), 8);
        assert_eq!(predictor.config().get_output_dim(), 4);
        assert_eq!(predictor.context_window(), 256);
    }

    #[test]
    fn test_custom_preset() {
        let predictor = KizzasiBuilder::custom_preset()
            .input_dim(10)
            .output_dim(10)
            .build()
            .unwrap();
        assert_eq!(predictor.config().get_input_dim(), 10);
        assert_eq!(predictor.config().get_hidden_dim(), 128);
    }

    #[test]
    fn test_signal_input_from_f32() {
        let input: SignalInput = 0.5f32.into();
        assert_eq!(input.as_array().len(), 1);
        assert_eq!(input.as_array()[0], 0.5);
    }

    #[test]
    fn test_signal_input_from_vec() {
        let input: SignalInput = vec![0.1, 0.2, 0.3].into();
        assert_eq!(input.as_array().len(), 3);
        assert_eq!(input.as_array()[0], 0.1);
    }

    #[test]
    fn test_signal_input_from_slice() {
        let data: &[f32] = &[0.1f32, 0.2, 0.3];
        let input: SignalInput = data.into();
        assert_eq!(input.as_array().len(), 3);
    }

    #[test]
    fn test_signal_input_from_array() {
        let input: SignalInput = [0.1f32, 0.2, 0.3].into();
        assert_eq!(input.as_array().len(), 3);
        assert_eq!(input.as_array()[2], 0.3);
    }

    #[test]
    fn test_signal_input_into_array() {
        let input: SignalInput = vec![0.1, 0.2].into();
        let array = input.into_array();
        assert_eq!(array.len(), 2);
    }

    #[test]
    fn test_hot_swap_success() {
        let config = KizzasiConfig::new()
            .model_type(ModelType::Mamba2)
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let mut predictor = Kizzasi::new(config).unwrap();

        // Make a prediction with original model
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let _ = predictor.step(&input).unwrap();

        // Hot-swap to different model type with same dimensions
        let new_config = KizzasiConfig::new()
            .model_type(ModelType::Mamba) // Different type
            .input_dim(3) // Same dimensions
            .output_dim(3)
            .hidden_dim(128) // Different hidden dim
            .state_dim(16) // Different state dim
            .num_layers(4); // Different layers

        let result = predictor.hot_swap(new_config, false);
        assert!(result.is_ok());

        // Should still work after hot-swap
        let output = predictor.step(&input).unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(predictor.model_type(), ModelType::Mamba);
        assert_eq!(predictor.hidden_dim(), 128);

        // The engine really changed: ModelType::Mamba is served by
        // kizzasi-model, not by the SelectiveSSM the predictor started with.
        // Asserting only the stored enum (as this test used to) passed even
        // when hot_swap rebuilt the identical architecture.
        assert!(predictor.ssm().is_none());
    }

    #[test]
    fn test_hot_swap_dimension_mismatch_input() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64);

        let mut predictor = Kizzasi::new(config).unwrap();

        // Try to swap with different input dimension
        let new_config = KizzasiConfig::new()
            .input_dim(5) // Different!
            .output_dim(3)
            .hidden_dim(64);

        let result = predictor.hot_swap(new_config, false);
        assert!(result.is_err());

        if let Err(KizzasiError::DimensionMismatch {
            expected, actual, ..
        }) = result
        {
            assert_eq!(expected, 3);
            assert_eq!(actual, 5);
        } else {
            panic!("Expected DimensionMismatch error");
        }
    }

    #[test]
    fn test_hot_swap_dimension_mismatch_output() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64);

        let mut predictor = Kizzasi::new(config).unwrap();

        // Try to swap with different output dimension
        let new_config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(5) // Different!
            .hidden_dim(64);

        let result = predictor.hot_swap(new_config, false);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "logic")]
    fn test_hot_swap_preserve_guardrails() {
        use kizzasi_logic::{ConstraintBuilder, Guardrail, GuardrailSet};

        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64);

        let mut predictor = Kizzasi::new(config).unwrap();

        // Add guardrails
        let mut guardrails = GuardrailSet::new();
        let constraint = ConstraintBuilder::new()
            .name("test_constraint")
            .greater_eq(-1.0)
            .less_eq(1.0)
            .build()
            .unwrap();
        guardrails.add_global(Guardrail::new(constraint, false));
        predictor.set_guardrails(guardrails);

        assert!(predictor.has_guardrails());

        // Hot-swap with preserve_guardrails = true
        let new_config = KizzasiConfig::new()
            .model_type(ModelType::Mamba)
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(128);

        predictor.hot_swap(new_config, true).unwrap();

        // Guardrails should still be there
        assert!(predictor.has_guardrails());
    }

    #[test]
    #[cfg(feature = "logic")]
    fn test_hot_swap_discard_guardrails() {
        use kizzasi_logic::{ConstraintBuilder, Guardrail, GuardrailSet};

        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64);

        let mut predictor = Kizzasi::new(config).unwrap();

        // Add guardrails
        let mut guardrails = GuardrailSet::new();
        let constraint = ConstraintBuilder::new()
            .name("test_constraint")
            .greater_eq(-1.0)
            .less_eq(1.0)
            .build()
            .unwrap();
        guardrails.add_global(Guardrail::new(constraint, false));
        predictor.set_guardrails(guardrails);

        assert!(predictor.has_guardrails());

        // Hot-swap with preserve_guardrails = false
        let new_config = KizzasiConfig::new()
            .model_type(ModelType::Mamba)
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(128);

        predictor.hot_swap(new_config, false).unwrap();

        // Guardrails should be gone
        assert!(!predictor.has_guardrails());
    }

    #[test]
    fn test_accessor_methods() {
        let config = KizzasiConfig::new()
            .model_type(ModelType::Mamba2)
            .input_dim(5)
            .output_dim(7)
            .hidden_dim(128)
            .state_dim(16)
            .num_layers(4);

        let predictor = Kizzasi::new(config).unwrap();

        assert_eq!(predictor.model_type(), ModelType::Mamba2);
        assert_eq!(predictor.input_dim(), 5);
        assert_eq!(predictor.output_dim(), 7);
        assert_eq!(predictor.hidden_dim(), 128);
        assert_eq!(predictor.state_dim(), 16);
        assert_eq!(predictor.num_layers(), 4);
    }

    #[test]
    fn test_step_slice() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(32)
            .state_dim(4)
            .num_layers(1);

        let mut predictor = Kizzasi::new(config).unwrap();

        // Test with slice
        let input_data = vec![0.1, 0.2, 0.3];
        let output = predictor.step_slice(&input_data).unwrap();
        assert_eq!(output.len(), 3);

        // Test with array slice
        let input_array = [0.4f32, 0.5, 0.6];
        let output2 = predictor.step_slice(&input_array).unwrap();
        assert_eq!(output2.len(), 3);
    }

    #[test]
    fn test_step_inplace() {
        let config = KizzasiConfig::new()
            .input_dim(4)
            .output_dim(4)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(1);

        let mut predictor = Kizzasi::new(config).unwrap();

        let input = vec![0.1, 0.2, 0.3, 0.4];
        let mut output = vec![0.0; 4];

        predictor.step_inplace(&input, &mut output).unwrap();

        // Output should be filled
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_step_inplace_wrong_size() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(16);

        let mut predictor = Kizzasi::new(config).unwrap();

        let input = vec![0.1, 0.2, 0.3];
        let mut output = vec![0.0; 5]; // Wrong size!

        let result = predictor.step_inplace(&input, &mut output);
        assert!(result.is_err());

        if let Err(KizzasiError::DimensionMismatch {
            expected, actual, ..
        }) = result
        {
            assert_eq!(expected, 3);
            assert_eq!(actual, 5);
        }
    }

    #[test]
    fn test_predict_n_inplace() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(1);

        let mut predictor = Kizzasi::new(config).unwrap();

        let input = Array1::from_vec(vec![0.1, 0.2]);
        let n_steps = 5;
        let mut output = Array2::zeros((n_steps, 2));

        predictor
            .predict_n_inplace(&input, n_steps, &mut output)
            .unwrap();

        assert_eq!(output.shape(), &[5, 2]);

        // All outputs should be finite
        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_predict_n_inplace_wrong_shape() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16);

        let mut predictor = Kizzasi::new(config).unwrap();

        let input = Array1::from_vec(vec![0.1, 0.2]);
        let mut output = Array2::zeros((5, 3)); // Wrong shape!

        let result = predictor.predict_n_inplace(&input, 5, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_copy_equivalence() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(32)
            .state_dim(4)
            .num_layers(1);

        let mut predictor1 = Kizzasi::new(config.clone()).unwrap();
        let mut predictor2 = Kizzasi::new(config).unwrap();

        let input_array = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let input_slice = vec![0.1, 0.2, 0.3];

        // Both methods should produce same results
        let output1 = predictor1.step(&input_array).unwrap();
        let output2 = predictor2.step_slice(&input_slice).unwrap();

        assert_eq!(output1.len(), output2.len());
    }
}
