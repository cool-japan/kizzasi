//! Model versioning and lifecycle management.
//!
//! This module provides comprehensive model versioning capabilities including:
//! - Semantic versioning (major.minor.patch)
//! - Model registry for tracking multiple versions
//! - Version comparison and selection
//! - Deployment strategies (canary, blue-green, rolling)
//! - A/B testing support with traffic splitting
//! - Model metadata and changelog tracking
//!
//! # Example
//!
//! ```rust,no_run
//! use kizzasi::versioning::{ModelVersion, ModelRegistry, DeploymentStrategy};
//! use kizzasi::KizzasiConfig;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut registry = ModelRegistry::new();
//!
//! // Register models
//! let config_v1 = KizzasiConfig::new().context_window(4096);
//! let v1 = ModelVersion::new("1.0.0", config_v1, "Initial release")?;
//! registry.register(v1)?;
//!
//! let config_v2 = KizzasiConfig::new().context_window(8192);
//! let v2 = ModelVersion::new("2.0.0", config_v2, "Doubled context window")?;
//! registry.register(v2)?;
//!
//! // Get the latest version
//! let latest = registry.get_latest()?;
//! println!("Latest version: {}", latest.version());
//!
//! // Deploy with canary strategy
//! registry.deploy("2.0.0", DeploymentStrategy::Canary { traffic_percent: 10 })?;
//! # Ok(())
//! # }
//! ```

use crate::error::{KizzasiError, KizzasiResult};
use crate::{Kizzasi, KizzasiConfig};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

/// Semantic version number (major.minor.patch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl SemanticVersion {
    /// Create a new semantic version.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse from a string like "1.2.3".
    pub fn parse(s: &str) -> KizzasiResult<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(KizzasiError::invalid_state(format!(
                "Invalid semantic version format: {}. Expected 'major.minor.patch'",
                s
            )));
        }

        let major = parts[0]
            .parse()
            .map_err(|_| KizzasiError::invalid_state("Invalid major version"))?;
        let minor = parts[1]
            .parse()
            .map_err(|_| KizzasiError::invalid_state("Invalid minor version"))?;
        let patch = parts[2]
            .parse()
            .map_err(|_| KizzasiError::invalid_state("Invalid patch version"))?;

        Ok(Self {
            major,
            minor,
            patch,
        })
    }

    /// Check if this version is compatible with another (same major version).
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major
    }
}

impl fmt::Display for SemanticVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for SemanticVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SemanticVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => match self.minor.cmp(&other.minor) {
                Ordering::Equal => self.patch.cmp(&other.patch),
                other => other,
            },
            other => other,
        }
    }
}

/// Deployment strategy for rolling out new model versions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    /// Replace the old version completely.
    Immediate,

    /// Gradually shift traffic to the new version (percentage: 0-100).
    Canary { traffic_percent: u8 },

    /// Run both versions, switch when ready.
    BlueGreen,

    /// Gradually replace instances over time.
    Rolling { batch_size: usize },
}

/// SplitMix64 finalizer: decorrelates sequential request ids so a modulo
/// bucket does not band consecutive requests into the same arm.
fn mix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl DeploymentStrategy {
    /// Check if a request should use the new version based on strategy.
    ///
    /// For [`Self::Rolling`] this answers for a rollout that has not been
    /// advanced yet; use [`ModelRegistry::select_for_request`], which knows
    /// how far the rollout has progressed.
    pub fn should_use_new_version(&self, request_id: u64) -> bool {
        self.should_use_new_version_at(request_id, 0.0)
    }

    /// Traffic-splitting decision, given how far a rolling deployment has
    /// progressed (`rolled_fraction` in `0.0..=1.0`).
    pub fn should_use_new_version_at(&self, request_id: u64, rolled_fraction: f64) -> bool {
        match self {
            DeploymentStrategy::Immediate => true,
            DeploymentStrategy::Canary { traffic_percent } => {
                // Hash first: the raw request id is usually sequential, which
                // makes `id % 100` allocate long runs to the same arm.
                let percent = (*traffic_percent).min(100) as u64;
                (mix64(request_id) % 100) < percent
            }
            DeploymentStrategy::BlueGreen => false, // Manual switch
            DeploymentStrategy::Rolling { .. } => {
                let fraction = rolled_fraction.clamp(0.0, 1.0);
                ((mix64(request_id) % 10_000) as f64) < fraction * 10_000.0
            }
        }
    }
}

/// Metadata for a model version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Human-readable description of this version.
    pub description: String,

    /// Changelog or release notes.
    pub changelog: Vec<String>,

    /// Author of this version.
    pub author: Option<String>,

    /// Creation timestamp (ISO 8601).
    pub created_at: String,

    /// Deployment strategy.
    pub deployment: DeploymentStrategy,

    /// Custom tags for classification.
    pub tags: Vec<String>,

    /// Metrics or performance data.
    pub metrics: HashMap<String, f64>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            description: String::new(),
            changelog: Vec::new(),
            author: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            deployment: DeploymentStrategy::Immediate,
            tags: Vec::new(),
            metrics: HashMap::new(),
        }
    }
}

/// A versioned model with configuration and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    version: SemanticVersion,
    config: KizzasiConfig,
    metadata: ModelMetadata,
}

impl ModelVersion {
    /// Create a new model version.
    pub fn new(version: &str, config: KizzasiConfig, description: &str) -> KizzasiResult<Self> {
        let version = SemanticVersion::parse(version)?;
        let metadata = ModelMetadata {
            description: description.to_string(),
            ..Default::default()
        };

        Ok(Self {
            version,
            config,
            metadata,
        })
    }

    /// Create with full metadata.
    pub fn with_metadata(
        version: &str,
        config: KizzasiConfig,
        metadata: ModelMetadata,
    ) -> KizzasiResult<Self> {
        let version = SemanticVersion::parse(version)?;
        Ok(Self {
            version,
            config,
            metadata,
        })
    }

    /// Get the semantic version.
    pub fn version(&self) -> &SemanticVersion {
        &self.version
    }

    /// Get the configuration.
    pub fn config(&self) -> &KizzasiConfig {
        &self.config
    }

    /// Get the metadata.
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Get mutable metadata.
    pub fn metadata_mut(&mut self) -> &mut ModelMetadata {
        &mut self.metadata
    }

    /// Create a predictor from this version.
    pub fn create_predictor(&self) -> KizzasiResult<Kizzasi> {
        Kizzasi::new(self.config.clone())
    }

    /// Check compatibility with another version.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.version.is_compatible_with(&other.version)
    }
}

/// Registry for managing multiple model versions.
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    versions: HashMap<SemanticVersion, ModelVersion>,
    active_version: Option<SemanticVersion>,
    canary_version: Option<SemanticVersion>,
    /// Instances already migrated by a `Rolling` deployment.
    rolled_instances: usize,
    /// Total instances a `Rolling` deployment must migrate.
    total_instances: usize,
}

/// Default instance count assumed by a rolling deployment when the caller has
/// not called [`ModelRegistry::set_rollout_instances`].
pub const DEFAULT_ROLLOUT_INSTANCES: usize = 10;

impl ModelRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new model version.
    pub fn register(&mut self, model: ModelVersion) -> KizzasiResult<()> {
        let version = *model.version();

        if self.versions.contains_key(&version) {
            return Err(KizzasiError::invalid_state(format!(
                "Version {} already exists",
                version
            )));
        }

        self.versions.insert(version, model);

        // Set as active if it's the first version
        if self.active_version.is_none() {
            self.active_version = Some(version);
        }

        Ok(())
    }

    /// Get a specific version.
    pub fn get(&self, version: &str) -> KizzasiResult<&ModelVersion> {
        let version = SemanticVersion::parse(version)?;
        self.versions
            .get(&version)
            .ok_or_else(|| KizzasiError::invalid_state(format!("Version {} not found", version)))
    }

    /// Get a mutable reference to a specific version.
    pub fn get_mut(&mut self, version: &str) -> KizzasiResult<&mut ModelVersion> {
        let version = SemanticVersion::parse(version)?;
        self.versions
            .get_mut(&version)
            .ok_or_else(|| KizzasiError::invalid_state(format!("Version {} not found", version)))
    }

    /// Get the latest version.
    pub fn get_latest(&self) -> KizzasiResult<&ModelVersion> {
        self.versions
            .keys()
            .max()
            .and_then(|v| self.versions.get(v))
            .ok_or_else(|| KizzasiError::invalid_state("No versions registered"))
    }

    /// Get the currently active version.
    pub fn get_active(&self) -> KizzasiResult<&ModelVersion> {
        let version = self
            .active_version
            .ok_or_else(|| KizzasiError::invalid_state("No active version"))?;
        self.versions
            .get(&version)
            .ok_or_else(|| KizzasiError::invalid_state("Active version not found"))
    }

    /// Get the canary version if deployed.
    pub fn get_canary(&self) -> Option<&ModelVersion> {
        self.canary_version.and_then(|v| self.versions.get(&v))
    }

    /// List all versions in ascending order.
    pub fn list_versions(&self) -> Vec<&ModelVersion> {
        let mut versions: Vec<_> = self.versions.values().collect();
        versions.sort_by_key(|v| v.version());
        versions
    }

    /// Deploy a specific version with a strategy.
    pub fn deploy(&mut self, version: &str, strategy: DeploymentStrategy) -> KizzasiResult<()> {
        let version = SemanticVersion::parse(version)?;

        if !self.versions.contains_key(&version) {
            return Err(KizzasiError::invalid_state(format!(
                "Version {} not found",
                version
            )));
        }

        match strategy {
            DeploymentStrategy::Immediate => {
                self.active_version = Some(version);
                self.canary_version = None;
                self.rolled_instances = 0;
                self.total_instances = 0;
            }
            DeploymentStrategy::Canary { traffic_percent } => {
                if traffic_percent > 100 {
                    return Err(KizzasiError::config(format!(
                        "Canary traffic_percent must be 0..=100, got {traffic_percent}"
                    )));
                }
                self.canary_version = Some(version);
                // Keep current active version
            }
            DeploymentStrategy::BlueGreen => {
                self.canary_version = Some(version);
                // Switch is manual via promote_canary()
            }
            DeploymentStrategy::Rolling { batch_size } => {
                if batch_size == 0 {
                    return Err(KizzasiError::config("Rolling batch_size must be > 0"));
                }
                // A staged rollout starts with *no* traffic on the new
                // version and advances in batches via `advance_rollout`.
                // Treating it as an immediate cutover (as this arm used to)
                // gave a caller who explicitly asked for a gradual rollout a
                // 100% instant switch, recorded in the metadata as "Rolling".
                self.canary_version = Some(version);
                self.rolled_instances = 0;
                self.total_instances = self.total_instances.max(DEFAULT_ROLLOUT_INSTANCES);
            }
        }

        // Update deployment strategy in metadata
        if let Some(model) = self.versions.get_mut(&version) {
            model.metadata.deployment = strategy;
        }

        Ok(())
    }

    /// Promote canary to active (for BlueGreen deployments).
    pub fn promote_canary(&mut self) -> KizzasiResult<()> {
        let canary = self
            .canary_version
            .ok_or_else(|| KizzasiError::invalid_state("No canary version deployed"))?;

        self.active_version = Some(canary);
        self.canary_version = None;

        Ok(())
    }

    /// Rollback to a previous version.
    pub fn rollback(&mut self, version: &str) -> KizzasiResult<()> {
        let version = SemanticVersion::parse(version)?;

        if !self.versions.contains_key(&version) {
            return Err(KizzasiError::invalid_state(format!(
                "Version {} not found",
                version
            )));
        }

        self.active_version = Some(version);
        self.canary_version = None;

        Ok(())
    }

    /// Select a version for a given request (respects deployment strategy).
    ///
    /// Under [`DeploymentStrategy::Rolling`] the share of requests routed to
    /// the new version equals the share of instances already migrated; see
    /// [`Self::advance_rollout`].
    pub fn select_for_request(&self, request_id: u64) -> KizzasiResult<&ModelVersion> {
        if let Some(canary_version) = self.canary_version {
            if let Some(canary) = self.versions.get(&canary_version) {
                if canary
                    .metadata
                    .deployment
                    .should_use_new_version_at(request_id, self.rolled_fraction())
                {
                    return Ok(canary);
                }
            }
        }

        self.get_active()
    }

    /// Fraction of instances already migrated by a rolling deployment.
    pub fn rolled_fraction(&self) -> f64 {
        if self.total_instances == 0 {
            return 0.0;
        }
        self.rolled_instances as f64 / self.total_instances as f64
    }

    /// Set how many instances a rolling deployment has to migrate.
    ///
    /// Defaults to [`DEFAULT_ROLLOUT_INSTANCES`]; set it to the real fleet
    /// size before deploying so each batch corresponds to real instances.
    pub fn set_rollout_instances(&mut self, total_instances: usize) -> KizzasiResult<()> {
        if total_instances == 0 {
            return Err(KizzasiError::config("total_instances must be > 0"));
        }
        self.total_instances = total_instances;
        self.rolled_instances = self.rolled_instances.min(total_instances);
        Ok(())
    }

    /// Advance a rolling deployment by one batch.
    ///
    /// Returns the fraction of instances migrated so far. When every instance
    /// has been migrated the new version becomes active and the rollout ends.
    pub fn advance_rollout(&mut self) -> KizzasiResult<f64> {
        let canary = self
            .canary_version
            .ok_or_else(|| KizzasiError::invalid_state("No rolling deployment in progress"))?;

        let batch_size = match self.versions.get(&canary).map(|m| m.metadata.deployment) {
            Some(DeploymentStrategy::Rolling { batch_size }) => batch_size,
            _ => {
                return Err(KizzasiError::invalid_state(
                    "The pending deployment is not a Rolling deployment",
                ))
            }
        };

        if self.total_instances == 0 {
            self.total_instances = DEFAULT_ROLLOUT_INSTANCES;
        }

        self.rolled_instances = self
            .rolled_instances
            .saturating_add(batch_size)
            .min(self.total_instances);

        if self.rolled_instances >= self.total_instances {
            self.active_version = Some(canary);
            self.canary_version = None;
            self.rolled_instances = 0;
            self.total_instances = 0;
            return Ok(1.0);
        }

        Ok(self.rolled_fraction())
    }

    /// Remove a version from the registry.
    pub fn remove(&mut self, version: &str) -> KizzasiResult<ModelVersion> {
        let version = SemanticVersion::parse(version)?;

        // Don't allow removing active or canary versions
        if Some(version) == self.active_version {
            return Err(KizzasiError::invalid_state("Cannot remove active version"));
        }
        if Some(version) == self.canary_version {
            return Err(KizzasiError::invalid_state("Cannot remove canary version"));
        }

        self.versions
            .remove(&version)
            .ok_or_else(|| KizzasiError::invalid_state(format!("Version {} not found", version)))
    }

    /// Get statistics about the registry.
    pub fn stats(&self) -> RegistryStats {
        RegistryStats {
            total_versions: self.versions.len(),
            active_version: self.active_version.map(|v| v.to_string()),
            canary_version: self.canary_version.map(|v| v.to_string()),
            latest_version: self.versions.keys().max().map(|v| v.to_string()),
        }
    }
}

/// Statistics about the model registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStats {
    pub total_versions: usize,
    pub active_version: Option<String>,
    pub canary_version: Option<String>,
    pub latest_version: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_with_two_versions() -> ModelRegistry {
        let mut registry = ModelRegistry::new();
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16);
        registry
            .register(ModelVersion::new("1.0.0", config.clone(), "initial").unwrap())
            .unwrap();
        registry
            .register(ModelVersion::new("2.0.0", config, "next").unwrap())
            .unwrap();
        registry
            .deploy("1.0.0", DeploymentStrategy::Immediate)
            .unwrap();
        registry
    }

    #[test]
    fn test_rolling_deployment_is_gradual_not_immediate() {
        // Regression: the Rolling arm was `// Simplified: treat as immediate`
        // and did a 100% instant cutover, while `should_use_new_version`
        // returned false for Rolling so the router refused to route to it.
        let mut registry = registry_with_two_versions();
        registry.set_rollout_instances(10).unwrap();
        registry
            .deploy("2.0.0", DeploymentStrategy::Rolling { batch_size: 5 })
            .unwrap();

        // No instant cutover: 1.0.0 is still the active version.
        assert_eq!(
            registry.get_active().unwrap().version().to_string(),
            "1.0.0"
        );
        assert_eq!(registry.rolled_fraction(), 0.0);

        let new_version_share = |registry: &ModelRegistry| {
            let hits = (0..2000)
                .filter(|id| {
                    registry
                        .select_for_request(*id)
                        .map(|m| m.version().to_string() == "2.0.0")
                        .unwrap_or(false)
                })
                .count();
            hits as f64 / 2000.0
        };

        assert_eq!(new_version_share(&registry), 0.0);

        // First batch: half the fleet migrated, roughly half the traffic.
        let fraction = registry.advance_rollout().unwrap();
        assert!((fraction - 0.5).abs() < 1e-9);
        let share = new_version_share(&registry);
        assert!(
            (0.4..0.6).contains(&share),
            "rolling traffic share was {share}"
        );

        // Final batch completes the rollout.
        let fraction = registry.advance_rollout().unwrap();
        assert!((fraction - 1.0).abs() < 1e-9);
        assert_eq!(
            registry.get_active().unwrap().version().to_string(),
            "2.0.0"
        );
        assert!(registry.get_canary().is_none());
    }

    #[test]
    fn test_canary_percent_over_100_is_rejected() {
        let mut registry = registry_with_two_versions();
        assert!(registry
            .deploy(
                "2.0.0",
                DeploymentStrategy::Canary {
                    traffic_percent: 150
                }
            )
            .is_err());
    }

    #[test]
    fn test_rolling_batch_size_zero_is_rejected() {
        let mut registry = registry_with_two_versions();
        assert!(registry
            .deploy("2.0.0", DeploymentStrategy::Rolling { batch_size: 0 })
            .is_err());
    }

    #[test]
    fn test_canary_split_is_hashed_not_banded() {
        // Sequential request ids used to map straight onto `id % 100`, so a
        // 10% canary took ids 0..9 of every hundred - a deterministic band
        // rather than a sample.
        let strategy = DeploymentStrategy::Canary {
            traffic_percent: 10,
        };
        let hits = (0..10_000)
            .filter(|id| strategy.should_use_new_version(*id))
            .count();
        let share = hits as f64 / 10_000.0;
        assert!((0.08..0.12).contains(&share), "canary share was {share}");

        // The first hundred sequential ids must not all land in one arm.
        let head: Vec<bool> = (0..100)
            .map(|id| strategy.should_use_new_version(id))
            .collect();
        assert!(head.iter().any(|hit| *hit));
        assert!(head.iter().any(|hit| !*hit));
    }

    #[test]
    fn test_semantic_version() {
        let v1 = SemanticVersion::new(1, 0, 0);
        let v2 = SemanticVersion::new(1, 1, 0);
        let v3 = SemanticVersion::new(2, 0, 0);

        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));
    }

    #[test]
    fn test_version_parsing() {
        let v = SemanticVersion::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert_eq!(v.to_string(), "1.2.3");

        assert!(SemanticVersion::parse("invalid").is_err());
        assert!(SemanticVersion::parse("1.2").is_err());
    }

    #[test]
    fn test_model_registry() {
        let mut registry = ModelRegistry::new();

        let config1 = KizzasiConfig::new().context_window(4096);
        let v1 = ModelVersion::new("1.0.0", config1, "Initial").unwrap();
        registry.register(v1).unwrap();

        let config2 = KizzasiConfig::new().context_window(8192);
        let v2 = ModelVersion::new("2.0.0", config2, "Upgrade").unwrap();
        registry.register(v2).unwrap();

        assert_eq!(registry.list_versions().len(), 2);
        assert_eq!(
            registry.get_latest().unwrap().version().to_string(),
            "2.0.0"
        );
        assert_eq!(
            registry.get_active().unwrap().version().to_string(),
            "1.0.0"
        );
    }

    #[test]
    fn test_deployment_strategies() {
        let mut registry = ModelRegistry::new();

        let config1 = KizzasiConfig::new();
        let v1 = ModelVersion::new("1.0.0", config1, "v1").unwrap();
        registry.register(v1).unwrap();

        let config2 = KizzasiConfig::new();
        let v2 = ModelVersion::new("2.0.0", config2, "v2").unwrap();
        registry.register(v2).unwrap();

        // Deploy with canary
        registry
            .deploy(
                "2.0.0",
                DeploymentStrategy::Canary {
                    traffic_percent: 10,
                },
            )
            .unwrap();

        assert_eq!(
            registry.get_active().unwrap().version().to_string(),
            "1.0.0"
        );
        assert_eq!(
            registry.get_canary().unwrap().version().to_string(),
            "2.0.0"
        );

        // Promote canary
        registry.promote_canary().unwrap();
        assert_eq!(
            registry.get_active().unwrap().version().to_string(),
            "2.0.0"
        );
        assert!(registry.get_canary().is_none());
    }

    #[test]
    fn test_traffic_splitting() {
        let canary = DeploymentStrategy::Canary {
            traffic_percent: 25,
        };

        // Test that roughly 25% of requests go to new version
        let mut new_version_count = 0;
        for i in 0..1000 {
            if canary.should_use_new_version(i) {
                new_version_count += 1;
            }
        }

        // Allow some variance (20-30%)
        assert!((200..=300).contains(&new_version_count));
    }

    #[test]
    fn test_rollback() {
        let mut registry = ModelRegistry::new();

        let config1 = KizzasiConfig::new();
        let v1 = ModelVersion::new("1.0.0", config1, "v1").unwrap();
        registry.register(v1).unwrap();

        let config2 = KizzasiConfig::new();
        let v2 = ModelVersion::new("2.0.0", config2, "v2").unwrap();
        registry.register(v2).unwrap();

        registry
            .deploy("2.0.0", DeploymentStrategy::Immediate)
            .unwrap();
        assert_eq!(
            registry.get_active().unwrap().version().to_string(),
            "2.0.0"
        );

        registry.rollback("1.0.0").unwrap();
        assert_eq!(
            registry.get_active().unwrap().version().to_string(),
            "1.0.0"
        );
    }

    #[test]
    fn test_version_removal() {
        let mut registry = ModelRegistry::new();

        let config1 = KizzasiConfig::new();
        let v1 = ModelVersion::new("1.0.0", config1, "v1").unwrap();
        registry.register(v1).unwrap();

        let config2 = KizzasiConfig::new();
        let v2 = ModelVersion::new("2.0.0", config2, "v2").unwrap();
        registry.register(v2).unwrap();

        // Can't remove active version
        assert!(registry.remove("1.0.0").is_err());

        // Can remove non-active version
        assert!(registry.remove("2.0.0").is_ok());
        assert_eq!(registry.list_versions().len(), 1);
    }
}
