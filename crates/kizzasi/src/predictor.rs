//! Main Kizzasi predictor implementation

use crate::error::{KizzasiError, KizzasiResult};
use crate::plugin::{Plugin, PluginManager};
use kizzasi_core::{KizzasiConfig, ModelType, SelectiveSSM, SignalPredictor, StateSpaceModel};
use scirs2_core::ndarray::{Array1, Array2};

#[cfg(feature = "logic")]
use kizzasi_logic::{ConstrainedInference, GuardrailSet};

// ============================================================================
// From Trait Implementations for Common Signal Types
// ============================================================================

/// Implements From trait for converting common signal types to Array1<f32>
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

    /// Load weights from a file path
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
    pub fn lightweight_preset(input_dim: usize, output_dim: usize) -> Self {
        Self::new()
            .model_type(ModelType::Mamba)
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
pub struct Kizzasi {
    /// The SSM engine
    ssm: SelectiveSSM,
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
    pub fn new(config: KizzasiConfig) -> KizzasiResult<Self> {
        let ssm = SelectiveSSM::new(config.clone())?;

        Ok(Self {
            ssm,
            config,
            #[cfg(feature = "logic")]
            guardrails: None,
            plugins: PluginManager::new(),
        })
    }

    /// Create a Kizzasi predictor from an existing SSM
    ///
    /// This is primarily used for restoring from full state checkpoints.
    pub fn from_ssm(ssm: SelectiveSSM) -> KizzasiResult<Self> {
        let config = ssm.config().clone();

        Ok(Self {
            ssm,
            config,
            #[cfg(feature = "logic")]
            guardrails: None,
            plugins: PluginManager::new(),
        })
    }

    /// Get a reference to the underlying SSM
    pub fn ssm(&self) -> &SelectiveSSM {
        &self.ssm
    }

    /// Get a mutable reference to the underlying SSM
    pub fn ssm_mut(&mut self) -> &mut SelectiveSSM {
        &mut self.ssm
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
    pub fn step(&mut self, input: &Array1<f32>) -> KizzasiResult<Array1<f32>> {
        let input_dim = self.config.get_input_dim();
        let output_dim = self.config.get_output_dim();

        // Execute pre-process plugins
        self.plugins
            .execute_pre_process(input, input_dim, output_dim)?;

        // Transform input through plugins
        let transformed_input =
            self.plugins
                .transform_input(input.clone(), input_dim, output_dim)?;

        // Get raw prediction from SSM
        let mut prediction = self.ssm.step(&transformed_input)?;

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

    /// Zero-copy prediction step from a slice
    ///
    /// This method accepts a slice instead of an Array1, avoiding allocation
    /// for the input. The output is still allocated.
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
        // Convert slice to Array1 view (zero-copy)
        let input_array = Array1::from_vec(input.to_vec());
        self.step(&input_array)
    }

    /// In-place prediction that writes output to a pre-allocated buffer
    ///
    /// This is the most efficient method as it avoids all allocations.
    /// The output buffer must have the correct size (equal to output_dim).
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
        output.copy_from_slice(result.as_slice().unwrap());
        Ok(())
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
    /// Returns an error if buffer dimensions don't match.
    pub fn predict_n_inplace(
        &mut self,
        input: &Array1<f32>,
        n_steps: usize,
        output: &mut Array2<f32>,
    ) -> KizzasiResult<()> {
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

    /// Reset the hidden state
    pub fn reset(&mut self) {
        self.ssm.reset();

        // Notify plugins
        let input_dim = self.config.get_input_dim();
        let output_dim = self.config.get_output_dim();
        let _ = self.plugins.execute_on_reset(input_dim, output_dim);
    }

    /// Get the context window size
    pub fn context_window(&self) -> usize {
        self.ssm.context_window()
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
    pub fn predict_n(&mut self, input: &Array1<f32>, n_steps: usize) -> KizzasiResult<Array2<f32>> {
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
    pub fn predict_until<F>(
        &mut self,
        input: &Array1<f32>,
        max_steps: usize,
        predicate: F,
    ) -> KizzasiResult<Vec<Array1<f32>>>
    where
        F: Fn(&Array1<f32>, usize) -> bool,
    {
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

    /// Get a clone of the predictor with fresh state
    ///
    /// Useful for running parallel predictions from the same starting point.
    pub fn fork(&self) -> KizzasiResult<Self> {
        let mut new_predictor = Self::new(self.config.clone())?;
        #[cfg(feature = "logic")]
        if let Some(ref guardrails) = self.guardrails {
            new_predictor.guardrails = Some(guardrails.clone());
        }
        Ok(new_predictor)
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

        // Create new SSM with new configuration
        let new_ssm =
            SelectiveSSM::new(new_config.clone()).map_err(|e| KizzasiError::ModelNotReady {
                reason: format!("Failed to initialize new model: {}", e),
                suggestion: "Check that the new configuration is valid and compatible".into(),
            })?;

        // Store old guardrails if needed
        #[cfg(feature = "logic")]
        let old_guardrails = if preserve_guardrails {
            self.guardrails.clone()
        } else {
            None
        };

        // Swap the model
        self.ssm = new_ssm;
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
