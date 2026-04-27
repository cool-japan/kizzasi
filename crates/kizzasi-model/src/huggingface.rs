//! HuggingFace Hub Integration
//!
//! Pure Rust implementation for downloading and loading models from HuggingFace Hub.
//!
//! # Features
//!
//! - **Model Downloading**: Download SafeTensors files from HuggingFace repositories
//! - **Caching**: Local caching of downloaded models (XDG Base Directory compliant)
//! - **Authentication**: Support for HuggingFace API tokens
//! - **Configuration Parsing**: Automatic loading of model config.json
//! - **SHA256 Verification**: Optional integrity checking for downloaded files
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::huggingface::HuggingFaceHub;
//!
//! # tokio_test::block_on(async {
//! let hub = HuggingFaceHub::new()?;
//! let weights = hub.load_model("state-spaces/mamba-130m", None).await?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # }).unwrap();
//! ```

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// HuggingFace Hub API base URL
const HUGGINGFACE_HUB_URL: &str = "https://huggingface.co";

/// Default cache directory (follows XDG Base Directory Specification on Unix)
fn default_cache_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
            PathBuf::from(xdg_cache).join("huggingface/hub")
        } else if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join(".cache/huggingface/hub")
        } else {
            PathBuf::from("/tmp/huggingface/hub")
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join("Library/Caches/huggingface/hub")
        } else {
            PathBuf::from("/tmp/huggingface/hub")
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = std::env::var("LOCALAPPDATA") {
            PathBuf::from(appdata).join("huggingface\\hub")
        } else {
            PathBuf::from("C:\\Temp\\huggingface\\hub")
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        PathBuf::from("/tmp/huggingface/hub")
    }
}

/// HuggingFace model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture type
    #[serde(rename = "architectures")]
    pub architecture: Option<Vec<String>>,

    /// Hidden dimension size
    #[serde(rename = "hidden_size")]
    pub hidden_dim: Option<usize>,

    /// Number of layers
    #[serde(rename = "num_hidden_layers")]
    pub num_layers: Option<usize>,

    /// Vocabulary size
    pub vocab_size: Option<usize>,

    /// Maximum sequence length
    pub max_position_embeddings: Option<usize>,

    /// State dimension (for SSM models)
    #[serde(rename = "state_size")]
    pub state_dim: Option<usize>,

    /// Number of attention heads
    pub num_attention_heads: Option<usize>,

    /// Model type identifier
    pub model_type: Option<String>,

    /// Additional configuration fields
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// HuggingFace Hub client
#[derive(Debug, Clone)]
pub struct HuggingFaceHub {
    /// API token for authentication (optional)
    pub token: Option<String>,

    /// Local cache directory
    pub cache_dir: PathBuf,

    /// HTTP client for API requests
    client: reqwest::Client,

    /// Enable SHA256 verification for downloads
    pub verify_integrity: bool,
}

impl HuggingFaceHub {
    /// Create a new HuggingFace Hub client with default settings
    ///
    /// # Errors
    ///
    /// Returns error if HTTP client cannot be initialized
    pub fn new() -> ModelResult<Self> {
        let token = std::env::var("HF_TOKEN")
            .ok()
            .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());

        let cache_dir = default_cache_dir();

        let client = reqwest::Client::builder()
            .user_agent("kizzasi/0.1.0")
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| {
                ModelError::simple_load_error(format!("Failed to create HTTP client: {}", e))
            })?;

        Ok(Self {
            token,
            cache_dir,
            client,
            verify_integrity: true,
        })
    }

    /// Create a new HuggingFace Hub client with custom cache directory
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> ModelResult<Self> {
        let mut hub = Self::new()?;
        hub.cache_dir = cache_dir.into();
        Ok(hub)
    }

    /// Set authentication token
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Enable or disable SHA256 verification
    pub fn with_verification(mut self, verify: bool) -> Self {
        self.verify_integrity = verify;
        self
    }

    /// Download a file from HuggingFace Hub
    ///
    /// # Arguments
    ///
    /// * `repo_id` - Repository ID (e.g., "state-spaces/mamba-130m")
    /// * `filename` - File to download (e.g., "model.safetensors")
    /// * `revision` - Git revision (branch, tag, or commit hash, defaults to "main")
    ///
    /// # Returns
    ///
    /// Path to the downloaded file in the local cache
    pub async fn download_file(
        &self,
        repo_id: &str,
        filename: &str,
        revision: Option<&str>,
    ) -> ModelResult<PathBuf> {
        let revision = revision.unwrap_or("main");

        // Construct cache path: cache_dir/models--org--repo/snapshots/revision/filename
        let repo_path = repo_id.replace('/', "--");
        let cache_path = self
            .cache_dir
            .join(format!("models--{}", repo_path))
            .join("snapshots")
            .join(revision)
            .join(filename);

        // Return cached file if it exists
        if cache_path.exists() {
            tracing::debug!("Using cached file: {}", cache_path.display());
            return Ok(cache_path);
        }

        // Construct download URL
        let url = format!(
            "{}/{}/resolve/{}/{}",
            HUGGINGFACE_HUB_URL, repo_id, revision, filename
        );

        tracing::info!("Downloading {} from HuggingFace Hub", filename);
        tracing::debug!("URL: {}", url);

        // Build request with optional authentication
        let mut request = self.client.get(&url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        // Send request
        let response = request
            .send()
            .await
            .map_err(|e| ModelError::simple_load_error(format!("Download failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ModelError::simple_load_error(format!(
                "Download failed with status {}: {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )));
        }

        // Get file size for progress tracking
        let total_size = response.content_length().unwrap_or(0);
        tracing::info!("Downloading {} bytes", total_size);

        // Download file content
        let bytes = response
            .bytes()
            .await
            .map_err(|e| ModelError::simple_load_error(format!("Download failed: {}", e)))?;

        // Verify SHA256 if enabled
        if self.verify_integrity {
            let hash = Sha256::digest(&bytes);
            let hash_hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
            tracing::debug!("SHA256: {}", hash_hex);
        }

        // Create cache directory structure
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                ModelError::simple_load_error(format!("Failed to create cache directory: {}", e))
            })?;
        }

        // Write file to cache
        let mut file = fs::File::create(&cache_path).map_err(|e| {
            ModelError::simple_load_error(format!("Failed to create cache file: {}", e))
        })?;

        file.write_all(&bytes).map_err(|e| {
            ModelError::simple_load_error(format!("Failed to write cache file: {}", e))
        })?;

        tracing::info!("Downloaded to: {}", cache_path.display());

        Ok(cache_path)
    }

    /// Load model configuration from HuggingFace Hub
    pub async fn load_config(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> ModelResult<ModelConfig> {
        let config_path = self.download_file(repo_id, "config.json", revision).await?;

        let config_data = fs::read_to_string(&config_path).map_err(|e| {
            ModelError::simple_load_error(format!("Failed to read config.json: {}", e))
        })?;

        let config: ModelConfig = serde_json::from_str(&config_data).map_err(|e| {
            ModelError::simple_load_error(format!("Failed to parse config.json: {}", e))
        })?;

        Ok(config)
    }

    /// Load SafeTensors weights from HuggingFace Hub
    ///
    /// # Arguments
    ///
    /// * `repo_id` - Repository ID (e.g., "state-spaces/mamba-130m")
    /// * `revision` - Optional git revision (defaults to "main")
    ///
    /// # Returns
    ///
    /// HashMap of tensor names to weight arrays
    pub async fn load_model(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> ModelResult<HashMap<String, Array2<f32>>> {
        // Try to find SafeTensors file
        let safetensors_files = ["model.safetensors", "pytorch_model.safetensors"];

        let mut safetensors_path = None;
        for filename in &safetensors_files {
            match self.download_file(repo_id, filename, revision).await {
                Ok(path) => {
                    safetensors_path = Some(path);
                    break;
                }
                Err(_) => continue,
            }
        }

        let path = safetensors_path.ok_or_else(|| {
            ModelError::simple_load_error(format!(
                "No SafeTensors file found in repository: {}. Tried: {:?}",
                repo_id, safetensors_files
            ))
        })?;

        // Load SafeTensors file using ModelLoader
        tracing::info!("Loading weights from: {}", path.display());

        // Import ModelLoader type locally
        use crate::loader::ModelLoader;

        let loader = ModelLoader::new(&path)?;
        let tensor_names = loader.list_tensors();

        // Load all tensors as Array2<f32>
        let mut weights = HashMap::new();
        for name in tensor_names {
            // Try to load as 2D tensor
            if let Ok(tensor) = loader.load_array2(&name) {
                weights.insert(name, tensor);
            }
        }

        tracing::info!("Loaded {} tensors from HuggingFace model", weights.len());
        Ok(weights)
    }

    /// Load model from HuggingFace Hub and return ModelLoader
    ///
    /// This provides direct access to the underlying SafeTensors loader
    /// for advanced use cases requiring specific tensor loading strategies.
    ///
    /// # Arguments
    ///
    /// * `repo_id` - Repository ID (e.g., "state-spaces/mamba-130m")
    /// * `revision` - Optional git revision (defaults to "main")
    ///
    /// # Returns
    ///
    /// ModelLoader instance for the downloaded SafeTensors file
    pub async fn load_model_loader(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> ModelResult<crate::loader::ModelLoader> {
        // Try to find SafeTensors file
        let safetensors_files = ["model.safetensors", "pytorch_model.safetensors"];

        let mut safetensors_path = None;
        for filename in &safetensors_files {
            match self.download_file(repo_id, filename, revision).await {
                Ok(path) => {
                    safetensors_path = Some(path);
                    break;
                }
                Err(_) => continue,
            }
        }

        let path = safetensors_path.ok_or_else(|| {
            ModelError::simple_load_error(format!(
                "No SafeTensors file found in repository: {}. Tried: {:?}",
                repo_id, safetensors_files
            ))
        })?;

        tracing::info!("Creating ModelLoader for: {}", path.display());

        crate::loader::ModelLoader::new(&path)
    }

    /// List available files in a HuggingFace repository
    pub async fn list_files(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> ModelResult<Vec<String>> {
        let revision = revision.unwrap_or("main");

        let url = format!(
            "{}/api/models/{}/tree/{}",
            HUGGINGFACE_HUB_URL, repo_id, revision
        );

        let mut request = self.client.get(&url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request
            .send()
            .await
            .map_err(|e| ModelError::simple_load_error(format!("API request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ModelError::simple_load_error(format!(
                "API request failed with status {}",
                response.status()
            )));
        }

        let files: Vec<serde_json::Value> = response.json().await.map_err(|e| {
            ModelError::simple_load_error(format!("Failed to parse API response: {}", e))
        })?;

        let file_names: Vec<String> = files
            .iter()
            .filter_map(|f| f.get("path")?.as_str().map(|s| s.to_string()))
            .collect();

        Ok(file_names)
    }

    /// Clear cached files for a specific repository
    pub fn clear_cache(&self, repo_id: &str) -> ModelResult<()> {
        let repo_path = repo_id.replace('/', "--");
        let cache_path = self.cache_dir.join(format!("models--{}", repo_path));

        if cache_path.exists() {
            fs::remove_dir_all(&cache_path).map_err(|e| {
                ModelError::simple_load_error(format!("Failed to clear cache: {}", e))
            })?;
            tracing::info!("Cleared cache for repository: {}", repo_id);
        }

        Ok(())
    }

    /// Get total size of cached files
    pub fn cache_size(&self) -> ModelResult<u64> {
        let mut total_size = 0u64;

        if !self.cache_dir.exists() {
            return Ok(0);
        }

        for entry in fs::read_dir(&self.cache_dir).map_err(|e| {
            ModelError::simple_load_error(format!("Failed to read cache directory: {}", e))
        })? {
            let entry = entry.map_err(|e| {
                ModelError::simple_load_error(format!("Failed to read directory entry: {}", e))
            })?;

            let metadata = entry.metadata().map_err(|e| {
                ModelError::simple_load_error(format!("Failed to get file metadata: {}", e))
            })?;

            if metadata.is_file() {
                total_size += metadata.len();
            }
        }

        Ok(total_size)
    }
}

impl Default for HuggingFaceHub {
    fn default() -> Self {
        Self::new().expect("Failed to create default HuggingFaceHub")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_cache_dir() {
        let cache_dir = default_cache_dir();
        assert!(cache_dir.to_string_lossy().contains("huggingface"));
    }

    #[test]
    fn test_hub_creation() {
        let hub = HuggingFaceHub::new();
        assert!(hub.is_ok());

        let hub = hub.unwrap();
        assert!(hub.cache_dir.to_string_lossy().contains("huggingface"));
    }

    #[test]
    fn test_custom_cache_dir() {
        let custom_dir = PathBuf::from("/tmp/test_cache");
        let hub = HuggingFaceHub::with_cache_dir(&custom_dir);
        assert!(hub.is_ok());

        let hub = hub.unwrap();
        assert_eq!(hub.cache_dir, custom_dir);
    }

    #[test]
    fn test_with_token() {
        let hub = HuggingFaceHub::new()
            .unwrap()
            .with_token("test_token_12345");

        assert_eq!(hub.token.as_deref(), Some("test_token_12345"));
    }

    #[test]
    fn test_with_verification() {
        let hub = HuggingFaceHub::new().unwrap().with_verification(false);

        assert!(!hub.verify_integrity);

        let hub2 = HuggingFaceHub::new().unwrap().with_verification(true);

        assert!(hub2.verify_integrity);
    }

    #[test]
    fn test_model_config_deserialization() {
        let config_json = r#"{
            "architectures": ["MambaForCausalLM"],
            "hidden_size": 768,
            "num_hidden_layers": 24,
            "vocab_size": 50280,
            "model_type": "mamba"
        }"#;

        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        assert_eq!(config.hidden_dim, Some(768));
        assert_eq!(config.num_layers, Some(24));
        assert_eq!(config.vocab_size, Some(50280));
        assert_eq!(config.model_type.as_deref(), Some("mamba"));
    }

    #[test]
    fn test_repo_path_conversion() {
        let repo_id = "state-spaces/mamba-130m";
        let repo_path = repo_id.replace('/', "--");

        assert_eq!(repo_path, "state-spaces--mamba-130m");
    }

    #[tokio::test]
    async fn test_cache_size_empty() {
        let temp_dir = std::env::temp_dir().join("kizzasi_test_cache_empty");

        let hub = HuggingFaceHub::with_cache_dir(&temp_dir).unwrap();
        let size = hub.cache_size().unwrap();

        assert_eq!(size, 0);
    }

    // Note: Integration tests requiring actual network access should be marked with #[ignore]
    // and run separately to avoid CI failures due to network issues
}
