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
        // Basic health check: try a forward pass with dummy data
        let test_input = Array1::zeros(self.config.hidden_dim);

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
        let mut staged = self.staged_models.write().await;
        let new_model = staged.remove(model_id).ok_or_else(|| {
            InferenceError::NotFound(format!("Staged model not found: {}", model_id))
        })?;

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

        // Set traffic split
        let mut split = self.traffic_split.write().await;
        split.insert(new_model.id.clone(), percentage);
        split.insert(old_model.id.clone(), 100 - percentage);

        // For gradual swap, we keep both models in staged
        // and route traffic based on percentage
        // This is a simplified implementation - in production you'd have
        // more sophisticated routing logic

        info!("Gradual swap configured: {}% to new model", percentage);

        // If 100%, complete the swap
        if percentage == 100 {
            *self.previous_model.write().await = Some(old_model);
            *self.active_model.write().await = new_model;
            split.clear();
        }

        Ok(())
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
}
