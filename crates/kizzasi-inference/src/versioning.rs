//! Model versioning and fallback management
//!
//! This module provides robust model versioning, health checking, and automatic
//! fallback mechanisms for production inference systems.

use crate::error::{InferenceError, InferenceResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Semantic version for models
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Major version (breaking changes)
    pub major: u32,
    /// Minor version (new features)
    pub minor: u32,
    /// Patch version (bug fixes)
    pub patch: u32,
}

impl ModelVersion {
    /// Create a new model version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse version from string (e.g., "1.2.3")
    pub fn parse(s: &str) -> InferenceResult<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(InferenceError::ForwardError(format!(
                "Invalid version format: {}",
                s
            )));
        }

        let major = parts[0].parse().map_err(|_| {
            InferenceError::ForwardError(format!("Invalid major version: {}", parts[0]))
        })?;
        let minor = parts[1].parse().map_err(|_| {
            InferenceError::ForwardError(format!("Invalid minor version: {}", parts[1]))
        })?;
        let patch = parts[2].parse().map_err(|_| {
            InferenceError::ForwardError(format!("Invalid patch version: {}", parts[2]))
        })?;

        Ok(Self::new(major, minor, patch))
    }

    /// Check if this version is compatible with another (same major version)
    pub fn is_compatible_with(&self, other: &ModelVersion) -> bool {
        self.major == other.major
    }

    // Note: to_string() is provided by Display trait implementation
}

impl std::fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Health status of a model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Model is healthy and operational
    Healthy,
    /// Model is degraded but operational
    Degraded,
    /// Model is unhealthy and should not be used
    Unhealthy,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Health status
    pub status: HealthStatus,
    /// Timestamp of the check
    pub timestamp: Instant,
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Number of requests processed
    pub request_count: usize,
    /// Additional details
    pub details: String,
}

impl HealthCheck {
    /// Create a new health check result
    pub fn new(status: HealthStatus) -> Self {
        Self {
            status,
            timestamp: Instant::now(),
            avg_latency_ms: 0.0,
            error_rate: 0.0,
            request_count: 0,
            details: String::new(),
        }
    }

    /// Check if the model is usable
    pub fn is_usable(&self) -> bool {
        matches!(self.status, HealthStatus::Healthy | HealthStatus::Degraded)
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model identifier
    pub id: String,
    /// Model version
    pub version: ModelVersion,
    /// Model architecture type
    pub architecture: String,
    /// Creation timestamp
    pub created_at: String,
    /// Model checksum (for integrity verification)
    pub checksum: Option<String>,
    /// Additional metadata
    pub extra: HashMap<String, String>,
}

impl ModelMetadata {
    /// Create new metadata
    pub fn new(
        id: impl Into<String>,
        version: ModelVersion,
        architecture: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            version,
            architecture: architecture.into(),
            created_at: chrono::Utc::now().to_rfc3339(),
            checksum: None,
            extra: HashMap::new(),
        }
    }

    /// Set checksum
    pub fn with_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.checksum = Some(checksum.into());
        self
    }

    /// Add extra metadata field
    pub fn add_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// Statistics for a versioned model
#[derive(Debug, Clone, Default)]
pub struct ModelStats {
    /// Total requests processed
    pub total_requests: usize,
    /// Total errors encountered
    pub total_errors: usize,
    /// Sum of latencies (for computing average)
    pub total_latency_ms: f64,
    /// Last health check
    pub last_health_check: Option<HealthCheck>,
}

impl ModelStats {
    /// Record a successful request
    pub fn record_success(&mut self, latency_ms: f64) {
        self.total_requests += 1;
        self.total_latency_ms += latency_ms;
    }

    /// Record an error
    pub fn record_error(&mut self, latency_ms: f64) {
        self.total_requests += 1;
        self.total_errors += 1;
        self.total_latency_ms += latency_ms;
    }

    /// Get average latency
    pub fn avg_latency_ms(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_latency_ms / self.total_requests as f64
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_errors as f64 / self.total_requests as f64
        }
    }

    /// Generate health check from current stats
    pub fn to_health_check(&self) -> HealthCheck {
        let error_rate = self.error_rate();
        let avg_latency = self.avg_latency_ms();

        let status = if error_rate > 0.5 {
            HealthStatus::Unhealthy
        } else if error_rate > 0.1 || avg_latency > 1000.0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        HealthCheck {
            status,
            timestamp: Instant::now(),
            avg_latency_ms: avg_latency,
            error_rate,
            request_count: self.total_requests,
            details: format!(
                "Requests: {}, Errors: {}, Avg Latency: {:.2}ms",
                self.total_requests, self.total_errors, avg_latency
            ),
        }
    }
}

/// Fallback strategy when primary model fails
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FallbackStrategy {
    /// Use the next available version
    NextVersion,
    /// Use the previous stable version
    PreviousStable,
    /// Use a specific version
    SpecificVersion,
    /// Return an error (no fallback)
    NoFallback,
}

/// Configuration for model versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningConfig {
    /// Fallback strategy
    pub fallback_strategy: FallbackStrategy,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Maximum error rate before marking unhealthy
    pub max_error_rate: f64,
    /// Maximum latency before marking degraded (ms)
    pub max_latency_ms: f64,
    /// Automatic recovery enabled
    pub auto_recovery: bool,
}

impl Default for VersioningConfig {
    fn default() -> Self {
        Self {
            fallback_strategy: FallbackStrategy::PreviousStable,
            health_check_interval: Duration::from_secs(60),
            max_error_rate: 0.1,
            max_latency_ms: 1000.0,
            auto_recovery: true,
        }
    }
}

/// Versioned model entry
struct VersionedModelEntry {
    metadata: ModelMetadata,
    stats: ModelStats,
    is_stable: bool,
}

/// Model version manager
pub struct ModelVersionManager {
    /// Map of model ID to versions
    models: Arc<RwLock<HashMap<String, Vec<VersionedModelEntry>>>>,
    /// Currently active versions per model ID
    active_versions: Arc<RwLock<HashMap<String, ModelVersion>>>,
    /// Configuration
    config: VersioningConfig,
}

impl ModelVersionManager {
    /// Create a new version manager
    pub fn new(config: VersioningConfig) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            active_versions: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Register a model version
    pub fn register_version(
        &self,
        metadata: ModelMetadata,
        is_stable: bool,
    ) -> InferenceResult<()> {
        let mut models = self.models.write().map_err(|e| {
            InferenceError::LockError(format!("Failed to acquire write lock: {}", e))
        })?;
        let entries = models.entry(metadata.id.clone()).or_default();

        // Check if version already exists
        if entries
            .iter()
            .any(|e| e.metadata.version == metadata.version)
        {
            return Err(InferenceError::ForwardError(format!(
                "Version {} already registered for model {}",
                metadata.version, metadata.id
            )));
        }

        entries.push(VersionedModelEntry {
            metadata,
            stats: ModelStats::default(),
            is_stable,
        });

        // Sort by version (descending)
        entries.sort_by(|a, b| b.metadata.version.cmp(&a.metadata.version));

        Ok(())
    }

    /// Get the active version for a model
    pub fn get_active_version(&self, model_id: &str) -> Option<ModelVersion> {
        let active = self.active_versions.read().ok()?;
        active.get(model_id).cloned()
    }

    /// Set the active version for a model
    pub fn set_active_version(&self, model_id: &str, version: ModelVersion) -> InferenceResult<()> {
        // Verify version exists
        let models = self.models.read().map_err(|e| {
            InferenceError::LockError(format!("Failed to acquire read lock: {}", e))
        })?;
        let entries = models
            .get(model_id)
            .ok_or_else(|| InferenceError::ForwardError(format!("Model {} not found", model_id)))?;

        if !entries.iter().any(|e| e.metadata.version == version) {
            return Err(InferenceError::ForwardError(format!(
                "Version {} not found for model {}",
                version, model_id
            )));
        }

        let mut active = self.active_versions.write().map_err(|e| {
            InferenceError::LockError(format!("Failed to acquire write lock: {}", e))
        })?;
        active.insert(model_id.to_string(), version);

        Ok(())
    }

    /// Record a request for a model version
    pub fn record_request(
        &self,
        model_id: &str,
        version: &ModelVersion,
        latency_ms: f64,
        is_error: bool,
    ) -> InferenceResult<()> {
        let mut models = self.models.write().map_err(|e| {
            InferenceError::LockError(format!("Failed to acquire write lock: {}", e))
        })?;
        let entries = models
            .get_mut(model_id)
            .ok_or_else(|| InferenceError::ForwardError(format!("Model {} not found", model_id)))?;

        let entry = entries
            .iter_mut()
            .find(|e| &e.metadata.version == version)
            .ok_or_else(|| {
                InferenceError::ForwardError(format!(
                    "Version {} not found for model {}",
                    version, model_id
                ))
            })?;

        if is_error {
            entry.stats.record_error(latency_ms);
        } else {
            entry.stats.record_success(latency_ms);
        }

        Ok(())
    }

    /// Perform health check on a model version
    pub fn health_check(
        &self,
        model_id: &str,
        version: &ModelVersion,
    ) -> InferenceResult<HealthCheck> {
        let mut models = self.models.write().map_err(|e| {
            InferenceError::LockError(format!("Failed to acquire write lock: {}", e))
        })?;
        let entries = models
            .get_mut(model_id)
            .ok_or_else(|| InferenceError::ForwardError(format!("Model {} not found", model_id)))?;

        let entry = entries
            .iter_mut()
            .find(|e| &e.metadata.version == version)
            .ok_or_else(|| {
                InferenceError::ForwardError(format!(
                    "Version {} not found for model {}",
                    version, model_id
                ))
            })?;

        let health_check = entry.stats.to_health_check();
        entry.stats.last_health_check = Some(health_check.clone());

        Ok(health_check)
    }

    /// Get fallback version for a model
    pub fn get_fallback_version(
        &self,
        model_id: &str,
        current_version: &ModelVersion,
    ) -> Option<ModelVersion> {
        let models = self.models.read().ok()?;
        let entries = models.get(model_id)?;

        match self.config.fallback_strategy {
            FallbackStrategy::NextVersion => {
                // Find next lower version
                entries
                    .iter()
                    .filter(|e| e.metadata.version < *current_version)
                    .map(|e| e.metadata.version.clone())
                    .next()
            }
            FallbackStrategy::PreviousStable => {
                // Find most recent stable version that's not current
                entries
                    .iter()
                    .filter(|e| e.is_stable && e.metadata.version != *current_version)
                    .map(|e| e.metadata.version.clone())
                    .next()
            }
            FallbackStrategy::NoFallback => None,
            FallbackStrategy::SpecificVersion => {
                // Would need to be configured separately
                None
            }
        }
    }

    /// List all versions for a model
    pub fn list_versions(&self, model_id: &str) -> Vec<ModelVersion> {
        let models = match self.models.read() {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        models
            .get(model_id)
            .map(|entries| entries.iter().map(|e| e.metadata.version.clone()).collect())
            .unwrap_or_default()
    }

    /// Get metadata for a specific version
    pub fn get_metadata(&self, model_id: &str, version: &ModelVersion) -> Option<ModelMetadata> {
        let models = self.models.read().ok()?;
        models.get(model_id).and_then(|entries| {
            entries
                .iter()
                .find(|e| &e.metadata.version == version)
                .map(|e| e.metadata.clone())
        })
    }

    /// Get statistics for a specific version
    pub fn get_stats(&self, model_id: &str, version: &ModelVersion) -> Option<ModelStats> {
        let models = self.models.read().ok()?;
        models.get(model_id).and_then(|entries| {
            entries
                .iter()
                .find(|e| &e.metadata.version == version)
                .map(|e| e.stats.clone())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_version_creation() {
        let version = ModelVersion::new(1, 2, 3);
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
    }

    #[test]
    fn test_model_version_parse() {
        let version = ModelVersion::parse("1.2.3").unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
    }

    #[test]
    fn test_model_version_parse_invalid() {
        assert!(ModelVersion::parse("1.2").is_err());
        assert!(ModelVersion::parse("1.2.x").is_err());
    }

    #[test]
    fn test_model_version_compatibility() {
        let v1 = ModelVersion::new(1, 2, 3);
        let v2 = ModelVersion::new(1, 3, 0);
        let v3 = ModelVersion::new(2, 0, 0);

        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));
    }

    #[test]
    fn test_model_version_ordering() {
        let v1 = ModelVersion::new(1, 0, 0);
        let v2 = ModelVersion::new(1, 1, 0);
        let v3 = ModelVersion::new(2, 0, 0);

        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1 < v3);
    }

    #[test]
    fn test_health_status() {
        let check = HealthCheck::new(HealthStatus::Healthy);
        assert!(check.is_usable());

        let check = HealthCheck::new(HealthStatus::Degraded);
        assert!(check.is_usable());

        let check = HealthCheck::new(HealthStatus::Unhealthy);
        assert!(!check.is_usable());
    }

    #[test]
    fn test_model_stats() {
        let mut stats = ModelStats::default();

        stats.record_success(100.0);
        stats.record_success(200.0);
        stats.record_error(300.0);

        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.avg_latency_ms(), 200.0);
        assert_eq!(stats.error_rate(), 1.0 / 3.0);
    }

    #[test]
    fn test_model_stats_health_check() {
        let mut stats = ModelStats::default();

        // Low error rate, low latency -> Healthy
        stats.record_success(50.0);
        stats.record_success(60.0);
        let check = stats.to_health_check();
        assert_eq!(check.status, HealthStatus::Healthy);

        // High error rate -> Unhealthy
        stats.record_error(100.0);
        stats.record_error(100.0);
        stats.record_error(100.0);
        let check = stats.to_health_check();
        assert_eq!(check.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_version_manager_register() {
        let config = VersioningConfig::default();
        let manager = ModelVersionManager::new(config);

        let metadata = ModelMetadata::new("test-model", ModelVersion::new(1, 0, 0), "transformer");

        manager.register_version(metadata, true).unwrap();

        let versions = manager.list_versions("test-model");
        assert_eq!(versions.len(), 1);
    }

    #[test]
    fn test_version_manager_active_version() {
        let config = VersioningConfig::default();
        let manager = ModelVersionManager::new(config);

        let v1 = ModelVersion::new(1, 0, 0);
        let metadata = ModelMetadata::new("test-model", v1.clone(), "transformer");
        manager.register_version(metadata, true).unwrap();

        manager
            .set_active_version("test-model", v1.clone())
            .unwrap();

        let active = manager.get_active_version("test-model");
        assert_eq!(active, Some(v1));
    }

    #[test]
    fn test_version_manager_record_request() {
        let config = VersioningConfig::default();
        let manager = ModelVersionManager::new(config);

        let v1 = ModelVersion::new(1, 0, 0);
        let metadata = ModelMetadata::new("test-model", v1.clone(), "transformer");
        manager.register_version(metadata, true).unwrap();

        manager
            .record_request("test-model", &v1, 100.0, false)
            .unwrap();
        manager
            .record_request("test-model", &v1, 200.0, true)
            .unwrap();

        let stats = manager.get_stats("test-model", &v1).unwrap();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.total_errors, 1);
    }

    #[test]
    fn test_version_manager_fallback() {
        let config = VersioningConfig {
            fallback_strategy: FallbackStrategy::PreviousStable,
            ..Default::default()
        };
        let manager = ModelVersionManager::new(config);

        let v1 = ModelVersion::new(1, 0, 0);
        let v2 = ModelVersion::new(1, 1, 0);

        let metadata1 = ModelMetadata::new("test-model", v1.clone(), "transformer");
        let metadata2 = ModelMetadata::new("test-model", v2.clone(), "transformer");

        manager.register_version(metadata1, true).unwrap(); // Stable
        manager.register_version(metadata2, false).unwrap(); // Not stable

        let fallback = manager.get_fallback_version("test-model", &v2);
        assert_eq!(fallback, Some(v1));
    }

    #[test]
    fn test_version_manager_health_check() {
        let config = VersioningConfig::default();
        let manager = ModelVersionManager::new(config);

        let v1 = ModelVersion::new(1, 0, 0);
        let metadata = ModelMetadata::new("test-model", v1.clone(), "transformer");
        manager.register_version(metadata, true).unwrap();

        // Record some successful requests
        manager
            .record_request("test-model", &v1, 50.0, false)
            .unwrap();
        manager
            .record_request("test-model", &v1, 60.0, false)
            .unwrap();

        let health = manager.health_check("test-model", &v1).unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
    }
}
