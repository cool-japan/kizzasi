//! Lazy initialization support for deferred resource allocation

use crate::error::{KizzasiError, KizzasiResult};
use crate::predictor::Kizzasi;
use kizzasi_core::KizzasiConfig;
use scirs2_core::ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};

#[cfg(feature = "logic")]
use kizzasi_logic::GuardrailSet;

/// Lazy-initialized Kizzasi predictor
///
/// This wrapper defers model initialization until the first prediction is made,
/// reducing startup time and memory usage for applications that may not use
/// all configured predictors.
///
/// # Thread Safety
///
/// `LazyKizzasi` is `Send + Sync` and uses internal synchronization to ensure
/// thread-safe initialization. The first thread to call a prediction method
/// will perform initialization while other threads wait.
///
/// # Example
///
/// ```rust,ignore
/// use kizzasi::prelude::*;
///
/// // Configuration is stored, but model not yet created
/// let lazy_predictor = LazyKizzasi::new(
///     KizzasiConfig::new()
///         .model_type(ModelType::Mamba2)
///         .input_dim(3)
///         .output_dim(3)
///         .hidden_dim(128)
/// );
///
/// // Model is created on first prediction
/// let input = array![0.1, 0.2, 0.3];
/// let output = lazy_predictor.step(&input)?; // Initialization happens here
/// ```
pub struct LazyKizzasi {
    config: KizzasiConfig,
    #[cfg(feature = "logic")]
    guardrails: Option<GuardrailSet>,
    predictor: Arc<Mutex<Option<Kizzasi>>>,
    /// The last initialization error, replayed verbatim on retry.
    ///
    /// Initialization used to run inside a `std::sync::Once`, which marks
    /// itself completed even when the closure fails. The first attempt then
    /// surfaced the real error and every later attempt silently returned `Ok`
    /// with no predictor, so callers got a generic "initialization failed"
    /// with advice ("try again") that could never work.
    init_error: Arc<Mutex<Option<String>>>,
}

impl LazyKizzasi {
    /// Create a new lazy predictor with the given configuration
    ///
    /// The actual model initialization is deferred until the first prediction.
    pub fn new(config: KizzasiConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "logic")]
            guardrails: None,
            predictor: Arc::new(Mutex::new(None)),
            init_error: Arc::new(Mutex::new(None)),
        }
    }

    /// Set guardrails (before initialization)
    #[cfg(feature = "logic")]
    pub fn with_guardrails(mut self, guardrails: GuardrailSet) -> Self {
        self.guardrails = Some(guardrails);
        self
    }

    /// Check if the predictor has been initialized
    pub fn is_initialized(&self) -> bool {
        self.predictor
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Force initialization without making a prediction
    ///
    /// This can be useful for pre-warming the model or handling initialization
    /// errors separately from prediction logic.
    ///
    /// Initialization is performed at most once *successfully*. A failure is
    /// reported with its real cause and leaves the predictor retryable, so a
    /// second call after fixing the environment can still succeed — and a
    /// second call that fails again reports the same real cause rather than a
    /// generic placeholder.
    pub fn initialize(&self) -> KizzasiResult<()> {
        // Hold the predictor lock across the whole attempt so two threads
        // cannot both construct a model.
        let mut guard = self.predictor.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_some() {
            return Ok(());
        }

        let mut predictor = match Kizzasi::new(self.config.clone()) {
            Ok(predictor) => predictor,
            Err(e) => {
                *self.init_error.lock().unwrap_or_else(|p| p.into_inner()) = Some(e.to_string());
                return Err(e);
            }
        };

        #[cfg(feature = "logic")]
        if let Some(ref guardrails) = self.guardrails {
            predictor.set_guardrails(guardrails.clone());
        }

        *guard = Some(predictor);
        *self.init_error.lock().unwrap_or_else(|p| p.into_inner()) = None;
        Ok(())
    }

    /// The most recent initialization failure, if any.
    pub fn last_init_error(&self) -> Option<String> {
        self.init_error
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Get a mutable reference to the initialized predictor
    ///
    /// Initializes the predictor if not already done.
    fn get_predictor_mut<F, R>(&self, f: F) -> KizzasiResult<R>
    where
        F: FnOnce(&mut Kizzasi) -> KizzasiResult<R>,
    {
        self.initialize()?;

        let mut guard = self.predictor.lock().unwrap_or_else(|e| e.into_inner());
        let predictor = guard.as_mut().ok_or_else(|| {
            let reason = self
                .last_init_error()
                .unwrap_or_else(|| "predictor was not initialized".to_string());
            KizzasiError::InvalidState {
                reason,
                recovery: Some("Fix the configuration and call initialize() again".into()),
            }
        })?;

        f(predictor)
    }

    /// Perform a single prediction step
    pub fn step(&self, input: &Array1<f32>) -> KizzasiResult<Array1<f32>> {
        self.get_predictor_mut(|p| p.step(input))
    }

    /// Predict multiple steps ahead
    pub fn predict_n(&self, input: &Array1<f32>, n_steps: usize) -> KizzasiResult<Array2<f32>> {
        self.get_predictor_mut(|p| p.predict_n(input, n_steps))
    }

    /// Predict until a condition is met
    pub fn predict_until<F>(
        &self,
        input: &Array1<f32>,
        max_steps: usize,
        predicate: F,
    ) -> KizzasiResult<Vec<Array1<f32>>>
    where
        F: Fn(&Array1<f32>, usize) -> bool,
    {
        self.get_predictor_mut(|p| p.predict_until(input, max_steps, predicate))
    }

    /// Predict over a batch of inputs
    pub fn predict_batch(&self, inputs: &[Array1<f32>]) -> KizzasiResult<Vec<Array1<f32>>> {
        self.get_predictor_mut(|p| p.predict_batch(inputs))
    }

    /// Reset the hidden state
    pub fn reset(&self) -> KizzasiResult<()> {
        self.get_predictor_mut(|p| {
            p.reset();
            Ok(())
        })
    }

    /// Get the context window size
    pub fn context_window(&self) -> KizzasiResult<usize> {
        self.get_predictor_mut(|p| Ok(p.context_window()))
    }

    /// Get the configuration (without requiring initialization)
    pub fn config(&self) -> &KizzasiConfig {
        &self.config
    }

    /// Convert to a fully initialized Kizzasi predictor
    ///
    /// This consumes the LazyKizzasi and returns the underlying predictor,
    /// initializing it if needed.
    pub fn into_initialized(self) -> KizzasiResult<Kizzasi> {
        self.initialize()?;

        let predictor = Arc::try_unwrap(self.predictor)
            .map_err(|_| KizzasiError::InvalidState {
                reason: "Cannot unwrap predictor with multiple references".into(),
                recovery: Some("Ensure no other threads are using this predictor".into()),
            })?
            .into_inner()
            // A poisoned mutex still holds a valid predictor: a panic
            // elsewhere must not turn this into an unrecoverable error.
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .ok_or_else(|| KizzasiError::InvalidState {
                reason: "Predictor not initialized".into(),
                recovery: Some("Call initialize() first".into()),
            })?;

        Ok(predictor)
    }
}

// Send + Sync are derived by the compiler from the field types. The manual
// `unsafe impl`s that used to live here were unnecessary (every field is
// already Send + Sync) and actively harmful: being unconditional, they would
// have kept compiling if a future field stopped being thread-safe, silently
// making the type unsound.

impl Clone for LazyKizzasi {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            #[cfg(feature = "logic")]
            guardrails: self.guardrails.clone(),
            predictor: Arc::new(Mutex::new(None)),
            init_error: Arc::new(Mutex::new(None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_failed_initialization_is_retryable_and_keeps_the_real_error() {
        // Regression: `Once::call_once` marks itself completed even when the
        // closure fails, so the second attempt used to return Ok(()) with no
        // predictor and then a generic "initialization failed" message.
        let missing = std::env::temp_dir().join("kizzasi_lazy_missing_weights.json");
        let _ = std::fs::remove_file(&missing);
        let invalid = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(16)
            .load_weights(&missing.to_string_lossy());
        let lazy = LazyKizzasi::new(invalid);

        let first = lazy.initialize().unwrap_err().to_string();
        let second = lazy.initialize().unwrap_err().to_string();
        assert_eq!(first, second, "the real cause must be reported every time");
        assert!(!lazy.is_initialized());

        // Prediction surfaces the same real cause, not a placeholder.
        let err = lazy
            .step(&Array1::from_vec(vec![0.1, 0.2, 0.3]))
            .unwrap_err()
            .to_string();
        assert_eq!(err, first);
    }
    use kizzasi_core::ModelType;

    #[test]
    fn test_lazy_initialization() {
        let config = KizzasiConfig::new()
            .model_type(ModelType::Mamba2)
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let lazy = LazyKizzasi::new(config);

        // Should not be initialized yet
        assert!(!lazy.is_initialized());

        // First prediction triggers initialization
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let output = lazy.step(&input).unwrap();
        assert_eq!(output.len(), 3);

        // Now should be initialized
        assert!(lazy.is_initialized());

        // Subsequent predictions should work
        let output2 = lazy.step(&input).unwrap();
        assert_eq!(output2.len(), 3);
    }

    #[test]
    fn test_explicit_initialization() {
        let config = KizzasiConfig::new()
            .model_type(ModelType::Mamba2)
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let lazy = LazyKizzasi::new(config);
        assert!(!lazy.is_initialized());

        // Explicitly initialize
        lazy.initialize().unwrap();
        assert!(lazy.is_initialized());

        // Predictions should work
        let input = Array1::from_vec(vec![0.5, 0.6]);
        let output = lazy.step(&input).unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_predict_n() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(32)
            .state_dim(4)
            .num_layers(1);

        let lazy = LazyKizzasi::new(config);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let predictions = lazy.predict_n(&input, 5).unwrap();
        assert_eq!(predictions.shape(), &[5, 3]);
        assert!(lazy.is_initialized());
    }

    #[test]
    fn test_reset() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16);

        let lazy = LazyKizzasi::new(config);
        let input = Array1::from_vec(vec![0.1, 0.2]);

        // Make prediction
        let _ = lazy.step(&input).unwrap();

        // Reset should work
        lazy.reset().unwrap();

        // Should still be initialized
        assert!(lazy.is_initialized());

        // Predictions should still work
        let output = lazy.step(&input).unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_config_access_without_init() {
        let config = KizzasiConfig::new()
            .model_type(ModelType::Mamba2)
            .input_dim(5)
            .output_dim(5)
            .hidden_dim(128);

        let lazy = LazyKizzasi::new(config);

        // Can access config without initialization
        assert_eq!(lazy.config().get_input_dim(), 5);
        assert_eq!(lazy.config().get_hidden_dim(), 128);
        assert!(!lazy.is_initialized());
    }

    #[test]
    fn test_into_initialized() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64);

        let lazy = LazyKizzasi::new(config);
        let mut predictor = lazy.into_initialized().unwrap();

        // Should be able to use the predictor
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let output = predictor.step(&input);
        assert!(output.is_ok());
    }

    #[test]
    fn test_clone() {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32);

        let lazy1 = LazyKizzasi::new(config);

        // Initialize the first one
        let input = Array1::from_vec(vec![0.1, 0.2]);
        let _ = lazy1.step(&input).unwrap();
        assert!(lazy1.is_initialized());

        // Clone should create a new uninitialized instance
        let lazy2 = lazy1.clone();
        assert!(!lazy2.is_initialized());

        // But it should work when used
        let output = lazy2.step(&input).unwrap();
        assert_eq!(output.len(), 2);
        assert!(lazy2.is_initialized());
    }

    #[test]
    #[cfg(feature = "logic")]
    fn test_with_guardrails() {
        use kizzasi_logic::{ConstraintBuilder, Guardrail, GuardrailSet};

        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64);

        let mut guardrails = GuardrailSet::new();
        let constraint = ConstraintBuilder::new()
            .name("test_bounds")
            .greater_eq(-2.0)
            .less_eq(2.0)
            .build()
            .unwrap();
        guardrails.add_global(Guardrail::new(constraint, false));

        let lazy = LazyKizzasi::new(config).with_guardrails(guardrails);

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let output = lazy.step(&input).unwrap();
        assert_eq!(output.len(), 3);

        // Output should be constrained
        for &val in output.iter() {
            assert!((-2.0..=2.0).contains(&val));
        }
    }
}
