//! Hot-swapping model management
//!
//! Provides capability to swap models at runtime without stopping the inference service.
//! This is useful for A/B testing, gradual rollouts, and zero-downtime model updates.

#![allow(clippy::arc_with_non_send_sync)]
//!
//! # Features
//!
//! - Load new models in the background
//! - Atomic model switching with no dropped requests
//! - Graceful draining of in-flight requests
//! - Rollback support if new model fails
//! - Health checks before activation
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_inference::hotswap::{HotSwapManager, SwapStrategy};
//!
//! let manager = HotSwapManager::new(current_model);
//!
//! // Load new model in background
//! manager.prepare_swap("v2.0.0", new_model_path).await?;
//!
//! // Switch to new model atomically
//! manager.activate("v2.0.0", SwapStrategy::Immediate).await?;
//!
//! // Rollback if issues detected
//! manager.rollback().await?;
//! ```

use crate::error::{InferenceError, InferenceResult};
use crate::registry::ModelConfig;
use kizzasi_model::AutoregressiveModel;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Strategy for swapping models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapStrategy {
    /// Switch immediately (may interrupt in-flight requests)
    Immediate,

    /// Wait for all in-flight requests to complete before switching
    Graceful,

    /// Gradually shift traffic to new model (percentage-based)
    Gradual { percentage: u8 },
}

/// Model instance with metadata
#[derive(Clone)]
pub struct ModelInstance {
    /// Model identifier
    pub id: String,

    /// Model version
    pub version: String,

    /// The actual model (wrapped in RwLock for mutable access)
    pub model: Arc<RwLock<Box<dyn AutoregressiveModel>>>,

    /// Model configuration
    pub config: ModelConfig,

    /// When this model was loaded
    pub loaded_at: chrono::DateTime<chrono::Utc>,

    /// Health status
    pub healthy: bool,

    /// Number of active requests using this model
    pub active_requests: Arc<RwLock<usize>>,
}

impl ModelInstance {
    /// Create a new model instance
    pub fn new(
        id: impl Into<String>,
        version: impl Into<String>,
        model: Box<dyn AutoregressiveModel>,
        config: ModelConfig,
    ) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
            model: Arc::new(RwLock::new(model)),
            config,
            loaded_at: chrono::Utc::now(),
            healthy: true,
            active_requests: Arc::new(RwLock::new(0)),
        }
    }

    /// Increment active request count
    pub async fn acquire(&self) {
        *self.active_requests.write().await += 1;
    }

    /// Decrement active request count
    pub async fn release(&self) {
        let mut count = self.active_requests.write().await;
        if *count > 0 {
            *count -= 1;
        }
    }

    /// Get active request count
    pub async fn request_count(&self) -> usize {
        *self.active_requests.read().await
    }

    /// Run health check on model
    pub async fn health_check(&mut self) -> bool {
        // Basic health check: try a forward pass with dummy data. The probe
        // must match the model's *input* dimension, not its hidden dimension
        // (those differ for every normal configuration).
        let test_input = Array1::zeros(self.config.input_dim);

        let mut model = self.model.write().await;
        match model.step(&test_input) {
            Ok(_) => {
                self.healthy = true;
                true
            }
            Err(e) => {
                error!("Model health check failed: {}", e);
                self.healthy = false;
                false
            }
        }
    }
}

/// Hot-swap model manager
pub struct HotSwapManager {
    /// Currently active model
    active_model: Arc<RwLock<ModelInstance>>,

    /// Staged models (prepared but not yet active)
    staged_models: Arc<RwLock<HashMap<String, ModelInstance>>>,

    /// Previous model (for rollback)
    previous_model: Arc<RwLock<Option<ModelInstance>>>,

    /// Traffic split percentages (model_id -> percentage)
    traffic_split: Arc<RwLock<HashMap<String, u8>>>,

    /// Swap history
    swap_history: Arc<RwLock<Vec<SwapEvent>>>,
}

/// Swap event for auditing
#[derive(Debug, Clone)]
pub struct SwapEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub from_version: String,
    pub to_version: String,
    pub strategy: SwapStrategy,
    pub success: bool,
    pub error: Option<String>,
}

impl HotSwapManager {
    /// Create a new hot-swap manager with an initial model
    pub fn new(initial_model: ModelInstance) -> Self {
        Self {
            active_model: Arc::new(RwLock::new(initial_model)),
            staged_models: Arc::new(RwLock::new(HashMap::new())),
            previous_model: Arc::new(RwLock::new(None)),
            traffic_split: Arc::new(RwLock::new(HashMap::new())),
            swap_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get the currently active model
    pub async fn active_model(&self) -> ModelInstance {
        self.active_model.read().await.clone()
    }

    /// Prepare a new model for swapping (load in background)
    ///
    /// # Arguments
    ///
    /// * `model_id` - Unique identifier for the model
    /// * `version` - Version string for the model
    /// * `model` - The model instance to swap in
    /// * `config` - Model configuration
    pub async fn prepare_swap(
        &self,
        model_id: impl Into<String>,
        version: impl Into<String>,
        model: Box<dyn AutoregressiveModel>,
        config: ModelConfig,
    ) -> InferenceResult<()> {
        let model_id = model_id.into();
        let version = version.into();

        info!("Preparing model swap: {} (version: {})", model_id, version);

        // Create instance
        let mut instance = ModelInstance::new(&model_id, &version, model, config);

        // Run health check
        if !instance.health_check().await {
            return Err(InferenceError::InitializationError(
                "New model failed health check".to_string(),
            ));
        }

        // Stage the model
        self.staged_models.write().await.insert(model_id, instance);

        Ok(())
    }

    /// Activate a staged model
    pub async fn activate(&self, model_id: &str, strategy: SwapStrategy) -> InferenceResult<()> {
        let new_model = {
            let mut staged = self.staged_models.write().await;
            staged.remove(model_id).ok_or_else(|| {
                InferenceError::NotFound(format!("Staged model not found: {}", model_id))
            })?
            // `staged` (the write guard) is dropped here, at the end of this
            // block — required before calling into `swap_gradual`, which
            // re-acquires `staged_models` itself (to keep the candidate
            // reachable during a partial split); holding this guard across
            // that call would deadlock against itself.
        };

        let old_model = self.active_model.read().await.clone();

        info!(
            "Activating model swap: {} -> {} ({:?})",
            old_model.version, new_model.version, strategy
        );

        match strategy {
            SwapStrategy::Immediate => {
                self.swap_immediate(old_model.clone(), new_model.clone())
                    .await?;
            }
            SwapStrategy::Graceful => {
                self.swap_graceful(old_model.clone(), new_model.clone())
                    .await?;
            }
            SwapStrategy::Gradual { percentage } => {
                self.swap_gradual(old_model.clone(), new_model.clone(), percentage)
                    .await?;
            }
        }

        // Record swap event
        let event = SwapEvent {
            timestamp: chrono::Utc::now(),
            from_version: old_model.version.clone(),
            to_version: new_model.version.clone(),
            strategy,
            success: true,
            error: None,
        };
        self.swap_history.write().await.push(event);

        Ok(())
    }

    /// Immediate swap (atomic replacement)
    async fn swap_immediate(
        &self,
        old_model: ModelInstance,
        new_model: ModelInstance,
    ) -> InferenceResult<()> {
        // Store previous for rollback
        *self.previous_model.write().await = Some(old_model);

        // Atomic swap
        *self.active_model.write().await = new_model;

        info!("Model swapped immediately");
        Ok(())
    }

    /// Graceful swap (wait for in-flight requests)
    async fn swap_graceful(
        &self,
        old_model: ModelInstance,
        new_model: ModelInstance,
    ) -> InferenceResult<()> {
        // Wait for all active requests to complete
        let timeout = std::time::Duration::from_secs(60);
        let start = std::time::Instant::now();

        loop {
            let count = old_model.request_count().await;
            if count == 0 {
                break;
            }

            if start.elapsed() > timeout {
                warn!("Graceful swap timeout, proceeding anyway");
                break;
            }

            debug!("Waiting for {} in-flight requests to complete", count);
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        // Perform swap
        *self.previous_model.write().await = Some(old_model);
        *self.active_model.write().await = new_model;

        info!("Model swapped gracefully");
        Ok(())
    }

    /// Gradual swap (traffic shifting)
    ///
    /// Below 100% this does not touch `active_model`: the candidate is kept
    /// reachable in `staged_models` and [`HotSwapManager::select_model`] is
    /// the entry point that actually honours `traffic_split` on a
    /// per-request basis. At 100% the swap completes immediately, exactly
    /// like [`HotSwapManager::swap_immediate`].
    async fn swap_gradual(
        &self,
        old_model: ModelInstance,
        new_model: ModelInstance,
        percentage: u8,
    ) -> InferenceResult<()> {
        if percentage > 100 {
            return Err(InferenceError::InvalidConfiguration(
                "Percentage must be <= 100".to_string(),
            ));
        }

        if percentage == 100 {
            *self.previous_model.write().await = Some(old_model);
            *self.active_model.write().await = new_model;
            self.traffic_split.write().await.clear();
            info!("Gradual swap configured: 100% to new model (swap completed)");
            return Ok(());
        }

        // Keep the candidate reachable via `staged_models` for the duration
        // of the split — `activate` already removed it from there when it
        // was pulled out for this call.
        self.staged_models
            .write()
            .await
            .insert(new_model.id.clone(), new_model.clone());

        let mut split = self.traffic_split.write().await;
        split.insert(new_model.id.clone(), percentage);
        split.insert(old_model.id.clone(), 100 - percentage);
        drop(split);

        info!("Gradual swap configured: {}% to new model", percentage);
        Ok(())
    }

    /// Select a model instance for a single request, honouring any
    /// in-progress gradual/canary traffic split.
    ///
    /// With no split configured — the common case, and always the case
    /// immediately after an `Immediate` or `Graceful` swap, or once a
    /// `Gradual` swap reaches 100% — this simply returns the active model.
    /// During a `SwapStrategy::Gradual { percentage }` rollout it performs a
    /// weighted random draw over `traffic_split`: with probability
    /// `percentage / 100` it returns the staged candidate, and with the
    /// remaining probability it returns the current active model. This is
    /// the entry point request-handling code should call instead of
    /// [`HotSwapManager::active_model`] so that a configured split actually
    /// shifts traffic rather than being recorded and ignored.
    pub async fn select_model(&self) -> ModelInstance {
        let split = self.traffic_split.read().await;
        if split.is_empty() {
            return self.active_model.read().await.clone();
        }

        let active = self.active_model.read().await.clone();
        let staged = self.staged_models.read().await;

        let total: u32 = split.values().map(|&p| p as u32).sum();
        if total == 0 {
            return active;
        }

        // Weighted draw via a Send-safe RNG (no thread-local kept alive
        // across an `.await` point).
        let draw: f32 = {
            use scirs2_core::random::RngExt;
            crate::sampling::make_sampler_rng(None).random::<f32>()
        };
        let threshold = draw * total as f32;

        let mut cumulative = 0.0_f32;
        for (id, &percentage) in split.iter() {
            cumulative += percentage as f32;
            if threshold < cumulative {
                if *id == active.id {
                    return active;
                }
                if let Some(candidate) = staged.get(id) {
                    return candidate.clone();
                }
                // Stale split entry that no longer resolves to a live
                // instance: fall back to the active model rather than
                // erroring a live request over bookkeeping drift.
                return active;
            }
        }

        active
    }

    /// Rollback to previous model
    pub async fn rollback(&self) -> InferenceResult<()> {
        let previous = self.previous_model.write().await.take();

        match previous {
            Some(prev_model) => {
                let current = self.active_model.read().await.clone();

                info!(
                    "Rolling back: {} -> {}",
                    current.version, prev_model.version
                );

                *self.active_model.write().await = prev_model.clone();

                // Record rollback event
                let event = SwapEvent {
                    timestamp: chrono::Utc::now(),
                    from_version: current.version,
                    to_version: prev_model.version,
                    strategy: SwapStrategy::Immediate,
                    success: true,
                    error: Some("Rollback".to_string()),
                };
                self.swap_history.write().await.push(event);

                Ok(())
            }
            None => Err(InferenceError::NotFound(
                "No previous model to rollback to".to_string(),
            )),
        }
    }

    /// Get swap history
    pub async fn history(&self) -> Vec<SwapEvent> {
        self.swap_history.read().await.clone()
    }

    /// List staged models
    pub async fn staged_models(&self) -> Vec<String> {
        self.staged_models.read().await.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ModelRegistry;
    use kizzasi_model::ModelType;

    #[test]
    fn test_swap_strategy() {
        let immediate = SwapStrategy::Immediate;
        let graceful = SwapStrategy::Graceful;
        let gradual = SwapStrategy::Gradual { percentage: 50 };

        assert_eq!(immediate, SwapStrategy::Immediate);
        assert_eq!(graceful, SwapStrategy::Graceful);
        assert_eq!(gradual, SwapStrategy::Gradual { percentage: 50 });
    }

    #[tokio::test]
    async fn test_model_instance_request_tracking() {
        let config = ModelConfig::new(ModelType::S4D);

        let mut registry = ModelRegistry::new();
        registry.register("test_model", config.clone());
        let model = registry.create_model("test_model").unwrap();

        let instance = ModelInstance::new("test", "1.0.0", model, config);

        assert_eq!(instance.request_count().await, 0);

        instance.acquire().await;
        assert_eq!(instance.request_count().await, 1);

        instance.acquire().await;
        assert_eq!(instance.request_count().await, 2);

        instance.release().await;
        assert_eq!(instance.request_count().await, 1);

        instance.release().await;
        assert_eq!(instance.request_count().await, 0);
    }

    #[tokio::test]
    async fn test_hotswap_manager_creation() {
        let config = ModelConfig::new(ModelType::S4D);

        let mut registry = ModelRegistry::new();
        registry.register("initial_model", config.clone());
        let model = registry.create_model("initial_model").unwrap();

        let instance = ModelInstance::new("initial", "1.0.0", model, config);
        let manager = HotSwapManager::new(instance);

        let active = manager.active_model().await;
        assert_eq!(active.version, "1.0.0");
    }

    #[tokio::test]
    async fn test_rollback_without_previous() {
        let config = ModelConfig::new(ModelType::S4D);

        let mut registry = ModelRegistry::new();
        registry.register("rollback_test", config.clone());
        let model = registry.create_model("rollback_test").unwrap();

        let instance = ModelInstance::new("initial", "1.0.0", model, config);
        let manager = HotSwapManager::new(instance);

        let result = manager.rollback().await;
        assert!(result.is_err());
    }

    /// Regression: `health_check` used to probe with a `hidden_dim`-sized
    /// input, so any model with `input_dim != hidden_dim` (the normal case)
    /// failed its own health check.
    #[tokio::test]
    async fn test_health_check_uses_input_dim_not_hidden_dim() {
        let config = ModelConfig::new(ModelType::S4D)
            .input_dim(1)
            .hidden_dim(64)
            .state_dim(16);

        let mut registry = ModelRegistry::new();
        registry.register("hc_test", config.clone());
        let model = registry.create_model("hc_test").unwrap();

        let mut instance = ModelInstance::new("hc", "1.0.0", model, config);
        assert!(
            instance.health_check().await,
            "a correctly loaded model with input_dim != hidden_dim must be healthy"
        );
        assert!(instance.healthy);
    }

    /// Regression: `SwapStrategy::Gradual` used to write `traffic_split` and
    /// never read it — 100% of traffic stayed on the old model regardless of
    /// the configured percentage, while reporting `success: true`.
    #[tokio::test]
    async fn test_gradual_swap_partial_split_reaches_both_versions() {
        let config = ModelConfig::new(ModelType::S4D);
        let mut registry = ModelRegistry::new();

        registry.register("initial", config.clone());
        let old = registry.create_model("initial").unwrap();
        let old_instance = ModelInstance::new("old-id", "1.0.0", old, config.clone());
        let manager = HotSwapManager::new(old_instance);

        registry.register("new", config.clone());
        let new_model = registry.create_model("new").unwrap();
        manager
            .prepare_swap("new-id", "2.0.0", new_model, config)
            .await
            .expect("prepare_swap must succeed");

        // Before any swap, select_model must agree with active_model.
        assert_eq!(manager.select_model().await.version, "1.0.0");

        manager
            .activate("new-id", SwapStrategy::Gradual { percentage: 50 })
            .await
            .expect("gradual activation must succeed");

        // Below 100%, the active model must NOT have switched yet.
        assert_eq!(manager.active_model().await.version, "1.0.0");

        // Over many draws, a 50/50 split must route to both versions.
        let mut saw_old = false;
        let mut saw_new = false;
        for _ in 0..200 {
            match manager.select_model().await.version.as_str() {
                "1.0.0" => saw_old = true,
                "2.0.0" => saw_new = true,
                other => panic!("unexpected version routed: {other}"),
            }
            if saw_old && saw_new {
                break;
            }
        }
        assert!(
            saw_old,
            "50% split must route to the old model at least once in 200 draws"
        );
        assert!(
            saw_new,
            "50% split must route to the new (staged) model at least once in 200 draws"
        );
    }

    /// A `Gradual { percentage: 100 }` swap must complete immediately, like
    /// `Immediate`, and leave no residual traffic split.
    #[tokio::test]
    async fn test_gradual_swap_full_percentage_completes_immediately() {
        let config = ModelConfig::new(ModelType::S4D);
        let mut registry = ModelRegistry::new();

        registry.register("initial", config.clone());
        let old = registry.create_model("initial").unwrap();
        let old_instance = ModelInstance::new("old-id", "1.0.0", old, config.clone());
        let manager = HotSwapManager::new(old_instance);

        registry.register("new", config.clone());
        let new_model = registry.create_model("new").unwrap();
        manager
            .prepare_swap("new-id", "2.0.0", new_model, config)
            .await
            .expect("prepare_swap must succeed");

        manager
            .activate("new-id", SwapStrategy::Gradual { percentage: 100 })
            .await
            .expect("100% gradual swap must complete immediately");

        assert_eq!(manager.active_model().await.version, "2.0.0");
        assert_eq!(manager.select_model().await.version, "2.0.0");
    }
}
