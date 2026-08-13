//! Configuration file loading support for TOML and YAML

use crate::error::{KizzasiError, KizzasiResult};
use kizzasi_core::{KizzasiConfig, ModelType};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Configuration file format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    /// TOML format (.toml)
    Toml,
    /// YAML format (.yaml, .yml)
    Yaml,
}

impl ConfigFormat {
    /// Detect format from file extension
    pub fn from_path(path: impl AsRef<Path>) -> Option<Self> {
        let path = path.as_ref();
        let ext = path.extension()?.to_str()?;
        match ext {
            "toml" => Some(Self::Toml),
            "yaml" | "yml" => Some(Self::Yaml),
            _ => None,
        }
    }
}

/// Serializable configuration structure
///
/// This mirrors `KizzasiConfig` but with all fields as Option for flexible loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigFile {
    /// Model type (Mamba, Mamba2, S4, Rwkv)
    #[serde(default)]
    pub model_type: Option<String>,

    /// Input dimension
    #[serde(default)]
    pub input_dim: Option<usize>,

    /// Output dimension
    #[serde(default)]
    pub output_dim: Option<usize>,

    /// Hidden dimension
    #[serde(default)]
    pub hidden_dim: Option<usize>,

    /// State dimension
    #[serde(default)]
    pub state_dim: Option<usize>,

    /// Number of layers
    #[serde(default)]
    pub num_layers: Option<usize>,

    /// Context window size
    #[serde(default)]
    pub context_window: Option<usize>,

    /// Path to weights file
    #[serde(default)]
    pub weights_path: Option<String>,

    /// Inner-dimension expansion factor.
    ///
    /// Consumed by `model_type = "mamba"` (the only architecture here with a
    /// gated expansion branch). Setting it for another model type is a
    /// configuration error at build time, not a silently dropped value.
    #[serde(default)]
    pub expansion_factor: Option<usize>,

    /// Head dimension for multi-head models.
    ///
    /// Consumed by `model_type = "rwkv"`; must equal `hidden_dim / num_heads`.
    /// Setting it for another model type is a configuration error at build
    /// time, not a silently dropped value.
    #[serde(default)]
    pub head_dim: Option<usize>,

    /// Number of heads for multi-head models.
    ///
    /// Consumed by `model_type = "rwkv"`; must divide `hidden_dim`. Setting it
    /// for another model type is a configuration error at build time, not a
    /// silently dropped value.
    #[serde(default)]
    pub num_heads: Option<usize>,
}

impl ConfigFile {
    /// Load configuration from a file
    ///
    /// Automatically detects format from file extension (.toml, .yaml, .yml)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ConfigFile::load("config.toml")?;
    /// ```
    pub fn load(path: impl AsRef<Path>) -> KizzasiResult<Self> {
        let path = path.as_ref();
        let format = ConfigFormat::from_path(path)
            .ok_or_else(|| KizzasiError::Config(format!("Unknown file extension: {:?}", path)))?;

        let content = fs::read_to_string(path)
            .map_err(|e| KizzasiError::Config(format!("Failed to read config file: {}", e)))?;

        Self::parse(&content, format)
    }

    /// Parse configuration from string
    ///
    /// # Arguments
    ///
    /// * `content` - Configuration content as string
    /// * `format` - Format of the configuration (TOML or YAML)
    pub fn parse(content: &str, format: ConfigFormat) -> KizzasiResult<Self> {
        match format {
            #[cfg(feature = "config-files")]
            ConfigFormat::Toml => toml::from_str(content)
                .map_err(|e| KizzasiError::Config(format!("Failed to parse TOML: {}", e))),
            #[cfg(feature = "config-files")]
            ConfigFormat::Yaml => serde_yaml::from_str(content)
                .map_err(|e| KizzasiError::Config(format!("Failed to parse YAML: {}", e))),
            #[cfg(not(feature = "config-files"))]
            _ => Err(KizzasiError::Config(
                "config-files feature not enabled".into(),
            )),
        }
    }

    /// Save configuration to a file
    ///
    /// Automatically selects format from file extension
    pub fn save(&self, path: impl AsRef<Path>) -> KizzasiResult<()> {
        let path = path.as_ref();
        let format = ConfigFormat::from_path(path)
            .ok_or_else(|| KizzasiError::Config(format!("Unknown file extension: {:?}", path)))?;

        let content = self.to_string(format)?;
        fs::write(path, content)
            .map_err(|e| KizzasiError::Config(format!("Failed to write config file: {}", e)))?;

        Ok(())
    }

    /// Convert to string in the specified format
    pub fn to_string(&self, format: ConfigFormat) -> KizzasiResult<String> {
        match format {
            #[cfg(feature = "config-files")]
            ConfigFormat::Toml => toml::to_string_pretty(self)
                .map_err(|e| KizzasiError::Config(format!("Failed to serialize TOML: {}", e))),
            #[cfg(feature = "config-files")]
            ConfigFormat::Yaml => serde_yaml::to_string(self)
                .map_err(|e| KizzasiError::Config(format!("Failed to serialize YAML: {}", e))),
            #[cfg(not(feature = "config-files"))]
            _ => Err(KizzasiError::Config(
                "config-files feature not enabled".into(),
            )),
        }
    }

    /// Convert to KizzasiConfig, filling in defaults where needed
    pub fn to_kizzasi_config(&self) -> KizzasiResult<KizzasiConfig> {
        let mut config = KizzasiConfig::new();

        // Model type
        if let Some(ref model_type_str) = self.model_type {
            let model_type = match model_type_str.to_lowercase().as_str() {
                "mamba" => ModelType::Mamba,
                "mamba2" => ModelType::Mamba2,
                "s4" => ModelType::S4,
                "rwkv" => ModelType::Rwkv,
                _ => {
                    return Err(KizzasiError::Config(format!(
                        "Unknown model type: {}. Valid types: mamba, mamba2, s4, rwkv",
                        model_type_str
                    )))
                }
            };
            config = config.model_type(model_type);
        }

        // Dimensions
        if let Some(dim) = self.input_dim {
            config = config.input_dim(dim);
        }
        if let Some(dim) = self.output_dim {
            config = config.output_dim(dim);
        }
        if let Some(dim) = self.hidden_dim {
            config = config.hidden_dim(dim);
        }
        if let Some(dim) = self.state_dim {
            config = config.state_dim(dim);
        }

        // Architecture parameters
        if let Some(n) = self.num_layers {
            config = config.num_layers(n);
        }
        if let Some(size) = self.context_window {
            config = config.context_window(size);
        }

        // Architecture options consumed by the model backends. They are
        // forwarded to KizzasiConfig rather than dropped; building a model
        // whose architecture cannot honour one of them is rejected with an
        // explicit error at `Kizzasi::new`.
        if let Some(factor) = self.expansion_factor {
            config = config.expansion_factor(factor);
        }
        if let Some(heads) = self.num_heads {
            config = config.num_heads(heads);
        }
        if let Some(dim) = self.head_dim {
            config = config.head_dim(dim);
        }

        // Optional parameters
        if let Some(ref path) = self.weights_path {
            config = config.load_weights(path);
        }

        Ok(config)
    }

    /// Create ConfigFile from KizzasiConfig
    pub fn from_kizzasi_config(config: &KizzasiConfig) -> Self {
        Self {
            model_type: Some(format!("{:?}", config.get_model_type())),
            input_dim: Some(config.get_input_dim()),
            output_dim: Some(config.get_output_dim()),
            hidden_dim: Some(config.get_hidden_dim()),
            state_dim: Some(config.get_state_dim()),
            num_layers: Some(config.get_num_layers()),
            context_window: Some(config.get_context_window()),
            weights_path: config.get_weights_path().map(str::to_string),
            expansion_factor: config.get_expansion_factor(),
            head_dim: config.get_head_dim(),
            num_heads: config.get_num_heads(),
        }
    }
}

/// Extension trait for KizzasiConfig to support file loading
pub trait ConfigLoader {
    /// Load configuration from a file
    fn from_file(path: impl AsRef<Path>) -> KizzasiResult<Self>
    where
        Self: Sized;

    /// Save configuration to a file
    fn to_file(&self, path: impl AsRef<Path>) -> KizzasiResult<()>;
}

impl ConfigLoader for KizzasiConfig {
    fn from_file(path: impl AsRef<Path>) -> KizzasiResult<Self> {
        let config_file = ConfigFile::load(path)?;
        config_file.to_kizzasi_config()
    }

    fn to_file(&self, path: impl AsRef<Path>) -> KizzasiResult<()> {
        let config_file = ConfigFile::from_kizzasi_config(self);
        config_file.save(path)
    }
}

#[cfg(all(test, feature = "config-files"))]
mod tests {
    use super::*;

    #[test]
    fn test_config_format_detection() {
        assert_eq!(
            ConfigFormat::from_path("config.toml"),
            Some(ConfigFormat::Toml)
        );
        assert_eq!(
            ConfigFormat::from_path("config.yaml"),
            Some(ConfigFormat::Yaml)
        );
        assert_eq!(
            ConfigFormat::from_path("config.yml"),
            Some(ConfigFormat::Yaml)
        );
        assert_eq!(ConfigFormat::from_path("config.json"), None);
    }

    #[test]
    fn test_toml_parsing() {
        let toml_content = r#"
            model_type = "Mamba2"
            input_dim = 3
            output_dim = 3
            hidden_dim = 128
            state_dim = 16
            num_layers = 4
            context_window = 8192
        "#;

        let config = ConfigFile::parse(toml_content, ConfigFormat::Toml).unwrap();
        assert_eq!(config.model_type.as_deref(), Some("Mamba2"));
        assert_eq!(config.input_dim, Some(3));
        assert_eq!(config.hidden_dim, Some(128));
    }

    #[test]
    fn test_yaml_parsing() {
        let yaml_content = r#"
model_type: Mamba2
input_dim: 3
output_dim: 3
hidden_dim: 128
state_dim: 16
num_layers: 4
context_window: 8192
        "#;

        let config = ConfigFile::parse(yaml_content, ConfigFormat::Yaml).unwrap();
        assert_eq!(config.model_type.as_deref(), Some("Mamba2"));
        assert_eq!(config.input_dim, Some(3));
        assert_eq!(config.hidden_dim, Some(128));
    }

    #[test]
    fn test_config_conversion() {
        let config_file = ConfigFile {
            model_type: Some("Mamba2".into()),
            input_dim: Some(5),
            output_dim: Some(5),
            hidden_dim: Some(256),
            state_dim: Some(16),
            num_layers: Some(3),
            context_window: Some(4096),
            weights_path: None,
            expansion_factor: None,
            head_dim: None,
            num_heads: None,
        };

        let kizzasi_config = config_file.to_kizzasi_config().unwrap();
        assert_eq!(kizzasi_config.get_input_dim(), 5);
        assert_eq!(kizzasi_config.get_output_dim(), 5);
        assert_eq!(kizzasi_config.get_hidden_dim(), 256);
        assert_eq!(kizzasi_config.get_model_type(), ModelType::Mamba2);
    }

    #[test]
    fn test_roundtrip_toml() {
        let original = KizzasiConfig::new()
            .model_type(ModelType::Mamba2)
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(128)
            .state_dim(16)
            .num_layers(4);

        let config_file = ConfigFile::from_kizzasi_config(&original);
        let toml_str = config_file.to_string(ConfigFormat::Toml).unwrap();

        let parsed = ConfigFile::parse(&toml_str, ConfigFormat::Toml).unwrap();
        let reconstructed = parsed.to_kizzasi_config().unwrap();

        assert_eq!(reconstructed.get_input_dim(), original.get_input_dim());
        assert_eq!(reconstructed.get_hidden_dim(), original.get_hidden_dim());
        assert_eq!(reconstructed.get_model_type(), original.get_model_type());
    }

    #[test]
    fn test_roundtrip_preserves_weights_path() {
        // Regression: `from_kizzasi_config` hard-coded `weights_path: None`
        // with the (false) comment "Not exposed in KizzasiConfig getters", so
        // load -> save -> load silently dropped the reference to the weights.
        let original = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .load_weights("model.safetensors");

        let toml_str = ConfigFile::from_kizzasi_config(&original)
            .to_string(ConfigFormat::Toml)
            .unwrap();
        let reconstructed = ConfigFile::parse(&toml_str, ConfigFormat::Toml)
            .unwrap()
            .to_kizzasi_config()
            .unwrap();

        assert_eq!(
            reconstructed.get_weights_path(),
            Some("model.safetensors"),
            "weights_path must survive a config round-trip"
        );
    }

    #[test]
    fn test_roundtrip_preserves_architecture_options() {
        // Regression: expansion_factor / head_dim / num_heads were parsed and
        // then silently discarded by `to_kizzasi_config`.
        let original = KizzasiConfig::new()
            .model_type(ModelType::Rwkv)
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(32)
            .num_heads(4)
            .head_dim(8);

        let toml_str = ConfigFile::from_kizzasi_config(&original)
            .to_string(ConfigFormat::Toml)
            .unwrap();
        let reconstructed = ConfigFile::parse(&toml_str, ConfigFormat::Toml)
            .unwrap()
            .to_kizzasi_config()
            .unwrap();

        assert_eq!(reconstructed.get_num_heads(), Some(4));
        assert_eq!(reconstructed.get_head_dim(), Some(8));

        // And the values actually reach the model.
        assert!(crate::Kizzasi::new(reconstructed).is_ok());
    }

    #[test]
    fn test_architecture_option_for_wrong_model_type_is_rejected() {
        let config_file = ConfigFile {
            model_type: Some("mamba2".to_string()),
            input_dim: Some(2),
            output_dim: Some(2),
            hidden_dim: Some(32),
            state_dim: Some(8),
            num_layers: Some(1),
            context_window: Some(64),
            weights_path: None,
            expansion_factor: Some(4),
            head_dim: None,
            num_heads: None,
        };

        let config = config_file.to_kizzasi_config().unwrap();
        // Accepted by the file parser, refused when the model is built —
        // never accepted-and-dropped.
        assert!(crate::Kizzasi::new(config).is_err());
    }

    #[test]
    fn test_invalid_model_type() {
        let config_file = ConfigFile {
            model_type: Some("InvalidModel".into()),
            input_dim: Some(3),
            output_dim: Some(3),
            hidden_dim: Some(64),
            state_dim: Some(8),
            num_layers: Some(2),
            context_window: Some(1024),
            weights_path: None,
            expansion_factor: None,
            head_dim: None,
            num_heads: None,
        };

        let result = config_file.to_kizzasi_config();
        assert!(result.is_err());
    }

    #[test]
    fn test_file_io() {
        use std::env;

        let config = KizzasiConfig::new()
            .model_type(ModelType::Mamba2)
            .input_dim(5)
            .output_dim(5)
            .hidden_dim(256);

        // Test TOML
        let toml_path = env::temp_dir().join("test_config.toml");
        config.to_file(&toml_path).unwrap();
        let loaded_toml = KizzasiConfig::from_file(&toml_path).unwrap();
        assert_eq!(loaded_toml.get_input_dim(), 5);
        assert_eq!(loaded_toml.get_hidden_dim(), 256);

        // Test YAML
        let yaml_path = env::temp_dir().join("test_config.yaml");
        config.to_file(&yaml_path).unwrap();
        let loaded_yaml = KizzasiConfig::from_file(&yaml_path).unwrap();
        assert_eq!(loaded_yaml.get_input_dim(), 5);
        assert_eq!(loaded_yaml.get_hidden_dim(), 256);

        // Cleanup
        let _ = fs::remove_file(toml_path);
        let _ = fs::remove_file(yaml_path);
    }
}
